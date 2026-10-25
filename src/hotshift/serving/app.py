from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from hotshift.config import load_config
from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import CPUCacheEngine
from hotshift.policies.factory import create_policy
from hotshift.recommend.scoring import CatalogScorer
from hotshift.serving.scheduler import (
    BatchScheduler,
    QueueFullError,
    SchedulerUnavailableError,
)
from hotshift.serving.schemas import (
    PolicyRequest,
    PoolRequest,
    RecommendRequest,
    ScenarioStartRequest,
)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config_path = Path(os.getenv("HOTSHIFT_CONFIG", "configs/demo.yaml"))
        config = load_config(config_path)
        dataset_path = Path(os.getenv("HOTSHIFT_DATA_DIR", str(config.output_dir)))
        store = EmbeddingStore(dataset_path)
        backend = os.getenv("HOTSHIFT_BACKEND", "cpu")
        if backend == "cuda":
            from hotshift.engine.cuda import CUDACacheEngine, build_cuda_extension

            build_cuda_extension()
            engine = CUDACacheEngine(
                store,
                config.cache_capacity,
                create_policy(config.policy),
                int(os.getenv("HOTSHIFT_DEVICE", "0")),
            )
        elif backend == "cpu":
            engine = CPUCacheEngine(store, config.cache_capacity, create_policy(config.policy))
        else:
            raise RuntimeError("HOTSHIFT_BACKEND must be cpu or cuda")
        scheduler = BatchScheduler(
            engine,
            queue_capacity=config.queue_capacity,
            max_batch_requests=config.max_batch_requests,
            max_batch_ids=config.max_batch_ids,
            max_batch_delay_ms=config.max_batch_delay_ms,
        )
        await scheduler.start()
        catalog: CatalogScorer | None = None
        try:
            catalog = CatalogScorer(dataset_path)
        except FileNotFoundError:
            catalog = None
        app.state.config = config
        app.state.store = store
        app.state.engine = engine
        app.state.scheduler = scheduler
        app.state.catalog = catalog
        app.state.scenario_task = None
        app.state.scenario = None
        app.state.runs = {}
        try:
            yield
        finally:
            scenario_task = app.state.scenario_task
            if scenario_task is not None:
                scenario_task.cancel()
                try:
                    await scenario_task
                except asyncio.CancelledError:
                    pass
            await scheduler.close()

    app = FastAPI(title="HotShift API", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(request: Request, error: QueueFullError):
        return JSONResponse(status_code=429, content={"detail": str(error)}, headers={"Retry-After": "1"})

    @app.exception_handler(SchedulerUnavailableError)
    async def unavailable_handler(request: Request, error: SchedulerUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.exception_handler(asyncio.TimeoutError)
    async def timeout_handler(request: Request, error: asyncio.TimeoutError):
        return JSONResponse(status_code=504, content={"detail": "request timed out"})

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready(request: Request):
        scheduler: BatchScheduler = request.app.state.scheduler
        engine: CPUCacheEngine = request.app.state.engine
        if not scheduler.accepting or not engine.healthy:
            raise HTTPException(status_code=503, detail="engine is not ready")
        return {"status": "ready", "backend": engine.backend}

    @app.get("/api/v1/status")
    async def status(request: Request):
        result = request.app.state.engine.status()
        result["scenario_id"] = request.app.state.scenario
        return result

    @app.post("/api/v1/embeddings/pool")
    async def pool(body: PoolRequest, request: Request):
        engine: CPUCacheEngine = request.app.state.engine
        if len(body.history_ids) > request.app.state.config.max_history_ids:
            raise HTTPException(status_code=422, detail="history exceeds the configured ID limit")
        if any(item < 0 or item >= engine.store.rows for item in body.history_ids):
            raise HTTPException(status_code=422, detail="history contains an ID outside the dataset")
        scheduler: BatchScheduler = request.app.state.scheduler
        result, queue_ms, total_ms = await scheduler.submit(body.history_ids, body.timeout_ms)
        return {
            "request_id": str(uuid.uuid4()),
            "backend": engine.backend,
            "dataset_version": engine.store.manifest["dataset_version"],
            "state_generation": engine.generation,
            "policy": engine.policy.name,
            "vector": result,
            "timing_ms": {"queue": queue_ms, "server_total": total_ms},
        }

    @app.post("/api/v1/recommend")
    async def recommend(body: RecommendRequest, request: Request):
        engine: CPUCacheEngine = request.app.state.engine
        if len(body.history_ids) > request.app.state.config.max_history_ids:
            raise HTTPException(status_code=422, detail="history exceeds the configured ID limit")
        if any(item < 0 or item >= engine.store.rows for item in body.history_ids):
            raise HTTPException(status_code=422, detail="history contains an ID outside the dataset")
        scorer: CatalogScorer | None = request.app.state.catalog
        if scorer is None:
            raise HTTPException(status_code=503, detail="synthetic catalog is not generated")
        scheduler: BatchScheduler = request.app.state.scheduler
        vector, queue_ms, total_ms = await scheduler.submit(body.history_ids, body.timeout_ms)
        items = scorer.recommend(
            np.asarray(vector, dtype=np.float32),
            body.history_ids,
            body.top_k,
            body.exclude_seen,
        )
        return {
            "request_id": str(uuid.uuid4()),
            "backend": engine.backend,
            "dataset_version": engine.store.manifest["dataset_version"],
            "state_generation": engine.generation,
            "policy": engine.policy.name,
            "items": items,
            "timing_ms": {"queue": queue_ms, "server_total": total_ms},
        }

    @app.get("/api/v1/metrics")
    async def metrics(request: Request):
        return request.app.state.scheduler.metrics.snapshot()

    @app.get("/api/v1/catalog")
    async def catalog(request: Request, offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=100)):
        scorer: CatalogScorer | None = request.app.state.catalog
        if scorer is None:
            raise HTTPException(status_code=503, detail="synthetic catalog is not generated")
        page = scorer.items[offset : offset + limit]
        next_offset = offset + len(page) if offset + len(page) < len(scorer.items) else None
        return {"items": page, "next_offset": next_offset}

    @app.post("/api/v1/cache/reset")
    async def reset_cache(request: Request):
        if request.app.state.scenario_task is not None:
            raise HTTPException(status_code=409, detail="stop the active scenario before resetting")
        scheduler: BatchScheduler = request.app.state.scheduler
        await scheduler.exclusive(request.app.state.engine.reset)
        return {"state_generation": request.app.state.engine.generation}

    @app.put("/api/v1/policy")
    async def set_policy(body: PolicyRequest, request: Request):
        if request.app.state.scenario_task is not None:
            raise HTTPException(status_code=409, detail="stop the active scenario before switching policy")
        scheduler: BatchScheduler = request.app.state.scheduler
        engine: CPUCacheEngine = request.app.state.engine

        def update() -> None:
            engine.reset(create_policy(body.policy))

        await scheduler.exclusive(update)
        return {"policy": engine.policy.name, "state_generation": engine.generation}

    async def run_scenario(request: Request, run_id: str, settings: ScenarioStartRequest) -> None:
        config = request.app.state.config
        rng = random.Random(settings.seed)
        started = time.monotonic()
        count = 0
        errors = 0
        next_dispatch = started
        hotset = max(1, min(config.rows, request.app.state.engine.capacity * 2))
        outcome = "complete"
        try:
            while time.monotonic() - started < settings.duration_seconds:
                elapsed_fraction = (time.monotonic() - started) / settings.duration_seconds
                if settings.scenario == "uniform":
                    history = [rng.randrange(config.rows) for _ in range(settings.history_length)]
                else:
                    offset = (
                        config.rows // 2
                        if settings.scenario == "viral_release" and elapsed_fraction >= 0.5
                        else 0
                    )
                    if settings.scenario == "brief_burst" and 0.4 <= elapsed_fraction < 0.5:
                        offset = config.rows // 2
                    if settings.scenario == "gradual":
                        offset = int(elapsed_fraction * config.rows / 2)
                    history = [
                        (offset + rng.randrange(hotset)) % config.rows
                        for _ in range(settings.history_length)
                    ]
                try:
                    await request.app.state.scheduler.submit(history, 2000)
                except (TimeoutError, QueueFullError, SchedulerUnavailableError, RuntimeError, ValueError):
                    errors += 1
                count += 1
                next_dispatch += 1 / settings.request_rate
                await asyncio.sleep(max(0.0, next_dispatch - time.monotonic()))
        except asyncio.CancelledError:
            outcome = "stopped"
            raise
        finally:
            request.app.state.runs[run_id] = {
                "run_id": run_id,
                "scenario": settings.scenario,
                "requests": count,
                "errors": errors,
                "status": outcome,
            }
            request.app.state.scenario = None
            request.app.state.scenario_task = None

    @app.post("/api/v1/scenarios/start")
    async def start_scenario(body: ScenarioStartRequest, request: Request):
        if request.app.state.scenario_task is not None:
            raise HTTPException(status_code=409, detail="a scenario is already active")
        run_id = uuid.uuid4().hex
        request.app.state.scenario = run_id
        request.app.state.runs[run_id] = {"run_id": run_id, "scenario": body.scenario, "status": "running"}
        request.app.state.scenario_task = asyncio.create_task(run_scenario(request, run_id, body))
        return {"run_id": run_id, "status": "running", "seed": body.seed}

    @app.post("/api/v1/scenarios/stop")
    async def stop_scenario(request: Request):
        task = request.app.state.scenario_task
        if task is None:
            return {"status": "stopped"}
        run_id = request.app.state.scenario
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return request.app.state.runs.get(run_id, {"status": "stopped"})

    @app.get("/api/v1/runs")
    async def list_runs(request: Request):
        root = Path(os.getenv("HOTSHIFT_ARTIFACT_DIR", "artifacts")) / "runs"
        runs = list(request.app.state.runs.values())
        if root.is_dir():
            for summary_path in root.glob("*/summary.json"):
                run_dir = summary_path.parent.resolve()
                if run_dir.parent != root.resolve():
                    continue
                try:
                    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                runs.append({"run_id": manifest.get("run_id"), "status": "complete", "backend": manifest.get("backend"), "dataset_version": manifest.get("dataset_version"), "summary_count": len(summary), "timestamp_utc": manifest.get("timestamp_utc")})
        return {"runs": runs}

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(run_id: str, request: Request):
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        run = request.app.state.runs.get(run_id)
        if run is not None:
            return run
        root = Path(os.getenv("HOTSHIFT_ARTIFACT_DIR", "artifacts")) / "runs"
        directory = (root / run_id).resolve()
        if directory.parent != root.resolve():
            raise HTTPException(status_code=404, detail="run not found")
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="run not found") from None
        return {"manifest": manifest, "summary": summary}

    return app


app = create_app()

