from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from hotshift.benchmark.common import INCOMPLETE_MARKER, is_complete_run, write_json
from hotshift.config import HotShiftConfig, load_config
from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import CPUCacheEngine, limits_from_config
from hotshift.policies.base import CachePolicy
from hotshift.policies.factory import create_policy
from hotshift.policies.static import StaticPolicy
from hotshift.recommend.scoring import CatalogScorer
from hotshift.serving.scheduler import (
    BatchScheduler,
    EngineBusyError,
    QueueFullError,
    SchedulerUnavailableError,
)
from hotshift.serving.schemas import (
    PolicyRequest,
    PoolRequest,
    RecommendRequest,
    ScenarioStartRequest,
)
from hotshift.traffic.generate import TrafficSampler

logger = logging.getLogger("hotshift.api")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
LIVE_SCENARIOS = {
    "stable": "stable",
    "viral_release": "abrupt",
    "gradual": "gradual",
    "brief_burst": "brief_burst",
    "uniform": "uniform",
}
PHASE_LABELS = {
    "abrupt": ("original popularity", "viral release"),
    "brief_burst": ("original popularity", "brief burst", "after burst"),
    "gradual": ("original popularity", "drifting", "shifted"),
}
STATIC_WARMUP_REQUESTS = 2000
SCENARIO_MAX_IN_FLIGHT = 64

EngineFactory = Callable[[HotShiftConfig, EmbeddingStore, CachePolicy], Any]


def _artifact_root() -> Path:
    return Path(os.getenv("HOTSHIFT_ARTIFACT_DIR", "artifacts"))


def serving_policy(name: str, config: HotShiftConfig, rows: int, dimensions: int) -> CachePolicy:
    """Policies for the live service. Static preloads the most frequent IDs of a separate,
    seeded warmup prefix drawn from the stable live distribution, never from future traffic."""
    if name != "static":
        return create_policy(name, row_bytes=dimensions * 4)
    sampler = TrafficSampler(rows, "stable", config.seed)
    rng = np.random.default_rng([config.seed, 7])
    frequency: Counter[int] = Counter()
    for index in range(STATIC_WARMUP_REQUESTS):
        ids, _, _ = sampler.draw(rng, 32, 0.0, index)
        frequency.update(ids)
    ranked = sorted(frequency.items(), key=lambda pair: (-pair[1], pair[0]))
    return StaticPolicy(item for item, _ in ranked[: config.cache_capacity])


def default_engine_factory(config: HotShiftConfig, store: EmbeddingStore, policy: CachePolicy):
    backend = os.getenv("HOTSHIFT_BACKEND", "cpu")
    limits = limits_from_config(config)
    if backend == "cuda":
        # GPU mode fails clearly when CUDA is unavailable; it never falls back to CPU.
        from hotshift.engine.cuda import CUDACacheEngine, build_cuda_extension

        build_cuda_extension()
        return CUDACacheEngine(store, config.cache_capacity, policy, int(os.getenv("HOTSHIFT_DEVICE", "0")), limits=limits)
    if backend == "cpu":
        return CPUCacheEngine(store, config.cache_capacity, policy, limits)
    raise RuntimeError("HOTSHIFT_BACKEND must be cpu or cuda")


def create_app(engine_factory: EngineFactory | None = None) -> FastAPI:
    factory = engine_factory or default_engine_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config_path = Path(os.getenv("HOTSHIFT_CONFIG", "configs/demo.yaml"))
        config = load_config(config_path)
        dataset_path = Path(os.getenv("HOTSHIFT_DATA_DIR", str(config.output_dir)))
        store = EmbeddingStore(dataset_path)
        policy = serving_policy(config.policy, config, store.rows, store.dimensions)
        engine = factory(config, store, policy)
        scheduler = BatchScheduler(
            engine,
            queue_capacity=config.queue_capacity,
            max_batch_requests=config.max_batch_requests,
            max_batch_ids=config.max_batch_ids,
            max_batch_delay_ms=config.max_batch_delay_ms,
        )
        await scheduler.start()
        try:
            catalog: CatalogScorer | None = CatalogScorer(dataset_path)
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
        logger.info("ready backend=%s policy=%s rows=%d", engine.backend, engine.policy.name, store.rows)
        try:
            yield
        finally:
            scenario_task = app.state.scenario_task
            if scenario_task is not None:
                app.state.scenario["shutdown"] = True
                scenario_task.cancel()
                try:
                    await scenario_task
                except asyncio.CancelledError:
                    pass
            await scheduler.close(timeout_seconds=float(os.getenv("HOTSHIFT_SHUTDOWN_TIMEOUT", "5")))

    app = FastAPI(title="HotShift API", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(request: Request, error: QueueFullError):
        return JSONResponse(status_code=429, content={"detail": str(error)}, headers={"Retry-After": "1"})

    @app.exception_handler(EngineBusyError)
    async def busy_handler(request: Request, error: EngineBusyError):
        return JSONResponse(status_code=503, content={"detail": str(error), "busy": True}, headers={"Retry-After": "1"})

    @app.exception_handler(SchedulerUnavailableError)
    async def unavailable_handler(request: Request, error: SchedulerUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.exception_handler(asyncio.TimeoutError)
    async def timeout_handler(request: Request, error: asyncio.TimeoutError):
        return JSONResponse(status_code=504, content={"detail": "request timed out"})

    @app.exception_handler(ValueError)
    async def value_handler(request: Request, error: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready(request: Request):
        scheduler: BatchScheduler = request.app.state.scheduler
        engine = request.app.state.engine
        if scheduler.closed or not scheduler.accepting or not engine.healthy:
            reason = "engine is unhealthy" if not engine.healthy else "engine is not accepting requests"
            raise HTTPException(status_code=503, detail=reason)
        return {"status": "ready", "backend": engine.backend}

    @app.get("/api/v1/status")
    async def status(request: Request):
        result = request.app.state.engine.status()
        scenario = request.app.state.scenario
        result["scenario_id"] = scenario["run_id"] if scenario else None
        result["scenario"] = _public_scenario(scenario)
        result["busy"] = request.app.state.scheduler.exclusive_active
        result["limits"] = {
            "max_history_ids": request.app.state.config.max_history_ids,
            "max_batch_requests": request.app.state.config.max_batch_requests,
            "max_batch_ids": request.app.state.config.max_batch_ids,
            "queue_capacity": request.app.state.config.queue_capacity,
            "max_batch_delay_ms": request.app.state.config.max_batch_delay_ms,
        }
        return result

    def validate_history(request: Request, history: list[int]) -> None:
        if len(history) > request.app.state.config.max_history_ids:
            raise HTTPException(status_code=422, detail="history exceeds the configured ID limit")
        rows = request.app.state.store.rows
        if any(item < 0 or item >= rows for item in history):
            raise HTTPException(status_code=422, detail="history contains an ID outside the dataset")

    def envelope(request: Request, request_id: str, outcome) -> dict[str, Any]:
        engine = request.app.state.engine
        return {
            "request_id": request_id,
            "backend": engine.backend,
            "dataset_version": request.app.state.store.manifest["dataset_version"],
            "state_generation": outcome.generation,
            "policy": engine.policy.name,
            "batch_size": outcome.batch_size,
        }

    @app.post("/api/v1/embeddings/pool")
    async def pool(body: PoolRequest, request: Request):
        validate_history(request, body.history_ids)
        request_id = uuid.uuid4().hex
        outcome = await request.app.state.scheduler.submit(body.history_ids, body.timeout_ms, request_id)
        return {
            **envelope(request, request_id, outcome),
            "vector": outcome.vector,
            "timing_ms": {"queue": outcome.queue_ms, "batch_engine": outcome.batch_engine_ms, "server_total": outcome.server_ms},
        }

    @app.post("/api/v1/recommend")
    async def recommend(body: RecommendRequest, request: Request):
        validate_history(request, body.history_ids)
        scorer: CatalogScorer | None = request.app.state.catalog
        if scorer is None:
            raise HTTPException(status_code=503, detail="synthetic catalog is not generated")
        request_id = uuid.uuid4().hex
        outcome = await request.app.state.scheduler.submit(body.history_ids, body.timeout_ms, request_id)
        scoring_started = time.perf_counter()
        items, ranking = scorer.recommend(
            np.asarray(outcome.vector, dtype=np.float32),
            body.history_ids,
            body.top_k,
            body.exclude_seen,
        )
        scoring_ms = (time.perf_counter() - scoring_started) * 1000
        return {
            **envelope(request, request_id, outcome),
            "ranking": ranking,
            "items": items,
            "returned": len(items),
            "timing_ms": {
                "queue": outcome.queue_ms,
                "batch_engine": outcome.batch_engine_ms,
                "scoring": scoring_ms,
                "server_total": outcome.server_ms + scoring_ms,
            },
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
        return {"items": page, "next_offset": next_offset, "total": len(scorer.items)}

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
        engine = request.app.state.engine
        store = request.app.state.store
        policy = serving_policy(body.policy, request.app.state.config, store.rows, store.dimensions)

        def update() -> None:
            engine.reset(policy)

        await scheduler.exclusive(update)
        return {"policy": engine.policy.name, "state_generation": engine.generation}

    async def run_scenario(app: FastAPI, state: dict[str, Any], settings: ScenarioStartRequest) -> None:
        scheduler: BatchScheduler = app.state.scheduler
        trace_scenario = LIVE_SCENARIOS[settings.scenario]
        sampler = TrafficSampler(app.state.store.rows, trace_scenario, settings.seed)
        rng = np.random.default_rng([settings.seed, 0])
        labels = PHASE_LABELS.get(trace_scenario, ("steady",))
        started = time.monotonic()
        in_flight: set[asyncio.Task] = set()

        async def send(history: list[int]) -> None:
            try:
                await scheduler.submit(history, 2000)
                state["completed"] += 1
            except (TimeoutError, QueueFullError, SchedulerUnavailableError, ValueError) as error:
                state["errors"] += 1
                state["error_causes"][type(error).__name__] = state["error_causes"].get(type(error).__name__, 0) + 1

        try:
            index = 0
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= settings.duration_seconds:
                    break
                fraction = elapsed / settings.duration_seconds
                history, phase_id, _ = sampler.draw(rng, settings.history_length, fraction, index)
                if phase_id != state["phase_id"]:
                    state["phase_changes"].append({"phase_id": phase_id, "at_seconds": elapsed, "label": labels[min(phase_id, len(labels) - 1)]})
                state.update({"phase_id": phase_id, "phase": labels[min(phase_id, len(labels) - 1)], "progress": fraction})
                state["offered"] += 1
                if len(in_flight) >= SCENARIO_MAX_IN_FLIGHT:
                    state["dropped"] += 1
                else:
                    task = asyncio.create_task(send(history))
                    in_flight.add(task)
                    task.add_done_callback(in_flight.discard)
                index += 1
                next_dispatch = started + index / settings.request_rate
                await asyncio.sleep(max(0.0, next_dispatch - time.monotonic()))
            state["status"] = "draining"
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
            state.update({"status": "complete", "progress": 1.0})
        except asyncio.CancelledError:
            state["status"] = "interrupted" if state.get("shutdown") else "stopped"
            for task in list(in_flight):
                task.cancel()
            raise
        finally:
            state["finished_at"] = time.time()
            state["metrics"] = scheduler.metrics.snapshot()["window"]
            public = _public_scenario(state)
            app.state.runs[state["run_id"]] = public
            _persist_live_run(public)
            app.state.scenario = None
            app.state.scenario_task = None

    @app.post("/api/v1/scenarios/start")
    async def start_scenario(body: ScenarioStartRequest, request: Request):
        if request.app.state.scenario_task is not None:
            raise HTTPException(status_code=409, detail="a scenario is already active")
        if not request.app.state.engine.healthy:
            raise HTTPException(status_code=503, detail="engine is unhealthy")
        run_id = "live-" + uuid.uuid4().hex[:12]
        state: dict[str, Any] = {
            "run_id": run_id,
            "kind": "live_scenario",
            "scenario": body.scenario,
            "trace_scenario": LIVE_SCENARIOS[body.scenario],
            "seed": body.seed,
            "duration_seconds": body.duration_seconds,
            "request_rate": body.request_rate,
            "history_length": body.history_length,
            "status": "running",
            "progress": 0.0,
            "phase_id": None,
            "phase": None,
            "phase_changes": [],
            "offered": 0,
            "completed": 0,
            "errors": 0,
            "dropped": 0,
            "error_causes": {},
            "policy": request.app.state.engine.policy.name,
            "backend": request.app.state.engine.backend,
            "initial_cache_occupancy": len(request.app.state.engine.resident),
            "started_at": time.time(),
        }
        request.app.state.scenario = state
        request.app.state.scenario_task = asyncio.create_task(run_scenario(request.app, state, body))
        return {"run_id": run_id, "status": "running", "seed": body.seed}

    @app.post("/api/v1/scenarios/stop")
    async def stop_scenario(request: Request):
        task = request.app.state.scenario_task
        if task is None:
            return {"status": "stopped"}
        run_id = request.app.state.scenario["run_id"]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return request.app.state.runs.get(run_id, {"status": "stopped"})

    @app.get("/api/v1/runs")
    async def list_runs(request: Request):
        root = (_artifact_root() / "runs").resolve()
        runs: list[dict[str, Any]] = list(request.app.state.runs.values())
        if root.is_dir():
            for run_dir in sorted(root.iterdir()):
                if run_dir.name.startswith(".") or RUN_ID_PATTERN.fullmatch(run_dir.name) is None:
                    continue
                if (run_dir / INCOMPLETE_MARKER).exists():
                    runs.append({"run_id": run_dir.name, "status": "incomplete"})
                    continue
                if not is_complete_run(run_dir) or not (run_dir / "summary.json").is_file():
                    continue
                try:
                    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                runs.append({
                    "run_id": manifest.get("run_id", run_dir.name),
                    "status": "complete",
                    "kind": manifest.get("kind", "replay_matrix" if int(manifest.get("schema_version", 1)) >= 2 else "legacy_v1"),
                    "backend": manifest.get("backend"),
                    "dataset_version": manifest.get("dataset_version"),
                    "dataset_sha256": manifest.get("dataset_sha256"),
                    "rows": manifest.get("rows"),
                    "dimensions": manifest.get("dimensions"),
                    "cache_budget_bytes": manifest.get("cache_budget_bytes"),
                    "gpu": (manifest.get("environment") or {}).get("gpu"),
                    "smoke": manifest.get("smoke"),
                    "summary_count": len(summary) if isinstance(summary, list) else 1,
                    "timestamp_utc": manifest.get("timestamp_utc"),
                })
        return {"runs": runs}

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(run_id: str, request: Request):
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        run = request.app.state.runs.get(run_id)
        if run is not None:
            return run
        root = (_artifact_root() / "runs").resolve()
        directory = (root / run_id).resolve()
        if directory.parent != root or not is_complete_run(directory):
            raise HTTPException(status_code=404, detail="run not found")
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            aggregate_path = directory / "aggregate.json"
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.is_file() else None
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="run not found") from None
        return {"manifest": manifest, "summary": summary, "aggregate": aggregate}

    return app


def _public_scenario(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {key: value for key, value in state.items() if key != "shutdown"}


def _persist_live_run(state: dict[str, Any]) -> None:
    """Live scenario summaries are illustrative; formal comparisons use offline replay."""
    try:
        directory = _artifact_root() / "live"
        directory.mkdir(parents=True, exist_ok=True)
        payload = {**state, "complete": state.get("status") == "complete"}
        write_json(directory / f"{state['run_id']}.json", payload)
    except (OSError, ValueError):
        logger.exception("could not persist live scenario summary")


app = create_app()
