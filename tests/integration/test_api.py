import asyncio
import json
import threading
import time

import httpx
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from hotshift.benchmark.common import INCOMPLETE_MARKER
from hotshift.benchmark.loadtest import replay_open_loop, summarize_requests
from hotshift.data.catalog import generate_catalog
from hotshift.data.generate import generate_embeddings
from hotshift.engine.base import CPUCacheEngine, limits_from_config
from hotshift.serving.app import create_app
from hotshift.traffic.generate import generate_trace
from hotshift.traffic.replay import load_trace


def _setup(tmp_path, monkeypatch, rows=128, dims=8, queue_capacity=32, capacity=16, catalog=True):
    data_dir = tmp_path / "data"
    generate_embeddings(rows, dims, 42, data_dir)
    if catalog:
        generate_catalog(rows, dims, 42, data_dir, candidate_count=min(32, rows))
    config = {
        "profile": "test",
        "data": {"rows": rows, "dimensions": dims, "seed": 42, "output_dir": str(data_dir)},
        "cache": {"capacity": capacity, "policy": "lru"},
        "limits": {
            "max_history_ids": 256,
            "max_batch_requests": 64,
            "max_batch_ids": 16384,
            "queue_capacity": queue_capacity,
            "max_batch_delay_ms": 2,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("HOTSHIFT_CONFIG", str(config_path))
    monkeypatch.setenv("HOTSHIFT_BACKEND", "cpu")
    monkeypatch.setenv("HOTSHIFT_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("HOTSHIFT_DATA_DIR", raising=False)
    return config_path


class SlowEngine(CPUCacheEngine):
    """Real CPU engine with an injectable delay and failure switch."""

    delay = 0.0
    fail = False
    gate: threading.Event | None = None

    def pool(self, histories):
        if self.gate is not None:
            self.gate.wait(5)
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("simulated device fault")
        return super().pool(histories)


def slow_factory(engines):
    def factory(config, store, policy):
        engine = SlowEngine(store, config.cache_capacity, policy, limits_from_config(config))
        engines.append(engine)
        return engine

    return factory


def test_cpu_api_pool_recommend_and_policy_controls(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").json()["backend"] == "cpu"
        status = client.get("/api/v1/status").json()
        assert status["backend"] == "cpu" and status["device"] == "CPU" and status["memory"]["device_bytes"] is None
        pooled = client.post("/api/v1/embeddings/pool", json={"history_ids": [1, 1]})
        assert pooled.status_code == 200
        body = pooled.json()
        assert len(body["vector"]) == 8 and body["backend"] == "cpu"
        assert set(body["timing_ms"]) == {"queue", "batch_engine", "server_total"}
        recommended = client.post("/api/v1/recommend", json={"history_ids": [1], "top_k": 3})
        assert recommended.status_code == 200
        assert len(recommended.json()["items"]) == 3 and recommended.json()["ranking"] == "similarity"
        assert "scoring" in recommended.json()["timing_ms"]
        assert client.put("/api/v1/policy", json={"policy": "adaptive"}).json()["policy"] == "adaptive"
        assert client.put("/api/v1/policy", json={"policy": "static"}).json()["policy"] == "static"
        assert client.post("/api/v1/cache/reset").status_code == 200
        catalog = client.get("/api/v1/catalog", params={"limit": 5}).json()
        assert len(catalog["items"]) == 5 and catalog["next_offset"] == 5


@pytest.mark.parametrize(
    "payload",
    [
        {"history_ids": [-1]},
        {"history_ids": [128]},
        {"history_ids": [0] * 257},
        {"history_ids": "1,2"},
        {"history_ids": [1], "timeout_ms": 1},
        {"history_ids": [1], "unexpected": True},
    ],
)
def test_malformed_payloads_return_422(tmp_path, monkeypatch, payload):
    _setup(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        assert client.post("/api/v1/embeddings/pool", json=payload).status_code == 422
        assert client.put("/api/v1/policy", json={"policy": "fifo"}).status_code == 422


def test_empty_history_pool_is_zero_and_recommend_uses_popularity_fallback(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        assert client.post("/api/v1/embeddings/pool", json={"history_ids": []}).json()["vector"] == [0.0] * 8
        body = client.post("/api/v1/recommend", json={"history_ids": [], "top_k": 4}).json()
        assert body["ranking"] == "popularity_fallback"
        assert [item["item_id"] for item in body["items"]] == [0, 1, 2, 3]
        many = client.post("/api/v1/recommend", json={"history_ids": list(range(30)), "top_k": 10}).json()
        assert many["returned"] == 2  # 32 candidates, 30 seen and excluded


def test_cache_state_never_changes_results(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, capacity=4)
    history = [3, 9, 9, 40, 77]
    with TestClient(create_app()) as client:
        first = client.post("/api/v1/recommend", json={"history_ids": history, "top_k": 8}).json()
        cold_vector = client.post("/api/v1/embeddings/pool", json={"history_ids": history}).json()["vector"]
        for policy in ("no_cache", "static", "lru", "decayed_lfu", "adaptive"):
            client.put("/api/v1/policy", json={"policy": policy})
            for _ in range(3):  # cold, then warmed, then after evictions by other traffic
                client.post("/api/v1/embeddings/pool", json={"history_ids": list(range(50, 60))})
                again = client.post("/api/v1/recommend", json={"history_ids": history, "top_k": 8}).json()
                assert [item["item_id"] for item in again["items"]] == [item["item_id"] for item in first["items"]]
            vector = client.post("/api/v1/embeddings/pool", json={"history_ids": history}).json()["vector"]
            np.testing.assert_allclose(vector, cold_vector, atol=1e-6)


def test_queue_full_returns_429_with_retry_hint(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, queue_capacity=1)
    engines: list[SlowEngine] = []
    app = create_app(slow_factory(engines))

    async def exercise():
        async with app.router.lifespan_context(app):
            engines[0].gate = threading.Event()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                first = asyncio.create_task(client.post("/api/v1/embeddings/pool", json={"history_ids": [1]}))
                await asyncio.sleep(0.05)
                second = asyncio.create_task(client.post("/api/v1/embeddings/pool", json={"history_ids": [2]}))
                await asyncio.sleep(0.05)
                rejected = await client.post("/api/v1/embeddings/pool", json={"history_ids": [3]})
                engines[0].gate.set()
                assert rejected.status_code == 429 and rejected.headers["retry-after"] == "1"
                assert (await first).status_code == 200 and (await second).status_code == 200

    asyncio.run(exercise())


def test_queued_timeout_returns_504(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    engines: list[SlowEngine] = []
    app = create_app(slow_factory(engines))

    async def exercise():
        async with app.router.lifespan_context(app):
            engines[0].delay = 0.3
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                blocker = asyncio.create_task(client.post("/api/v1/embeddings/pool", json={"history_ids": [1]}))
                await asyncio.sleep(0.02)
                late = await client.post("/api/v1/embeddings/pool", json={"history_ids": [2], "timeout_ms": 50})
                assert late.status_code == 504
                assert (await blocker).status_code == 200

    asyncio.run(exercise())


def test_engine_failure_makes_service_unready_without_backend_fallback(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    engines: list[SlowEngine] = []
    with TestClient(create_app(slow_factory(engines))) as client:
        engines[0].fail = True
        assert client.post("/api/v1/embeddings/pool", json={"history_ids": [1]}).status_code == 503
        ready = client.get("/health/ready")
        assert ready.status_code == 503 and "unhealthy" in ready.json()["detail"]
        assert client.get("/api/v1/status").json()["backend"] == "cpu"
        engines[0].fail = False
        assert client.post("/api/v1/embeddings/pool", json={"history_ids": [1]}).status_code == 503
        assert client.post("/api/v1/scenarios/start", json={"scenario": "stable"}).status_code == 503


def test_reset_and_policy_switch_during_queued_work_drain_first(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    engines: list[SlowEngine] = []
    app = create_app(slow_factory(engines))

    async def exercise():
        async with app.router.lifespan_context(app):
            engines[0].delay = 0.05
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                work = [asyncio.create_task(client.post("/api/v1/embeddings/pool", json={"history_ids": [index]})) for index in range(5)]
                await asyncio.sleep(0.01)
                switched = await client.put("/api/v1/policy", json={"policy": "adaptive"})
                assert switched.status_code == 200
                assert [response.status_code for response in await asyncio.gather(*work)] == [200] * 5
                busy_during = await asyncio.gather(
                    client.post("/api/v1/cache/reset"),
                    client.post("/api/v1/cache/reset"),
                )
                codes = sorted(response.status_code for response in busy_during)
                assert codes in ([200, 200], [200, 503])

    asyncio.run(exercise())


def test_startup_fails_clearly_for_bad_configuration(tmp_path, monkeypatch):
    config_path = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("HOTSHIFT_BACKEND", "tpu")
    with pytest.raises(RuntimeError, match="HOTSHIFT_BACKEND"), TestClient(create_app()):
        pass
    monkeypatch.setenv("HOTSHIFT_BACKEND", "cpu")
    raw = yaml.safe_load(config_path.read_text())
    raw["cache"]["surprise"] = 1
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown keys"), TestClient(create_app()):
        pass


def test_cuda_backend_never_falls_back_to_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present")
    _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("HOTSHIFT_BACKEND", "cuda")
    with pytest.raises(RuntimeError, match="CUDA"), TestClient(create_app()):
        pass


def test_missing_dataset_and_corrupt_manifest_fail_startup(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "embedding_sha256": "f" * 64}))
    with pytest.raises(ValueError, match="checksum"), TestClient(create_app()):
        pass
    monkeypatch.setenv("HOTSHIFT_DATA_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError), TestClient(create_app()):
        pass


def test_scenario_lifecycle_locks_controls_and_reports_phase(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, rows=1000)
    with TestClient(create_app()) as client:
        started = client.post("/api/v1/scenarios/start", json={"scenario": "viral_release", "duration_seconds": 1.0, "request_rate": 50, "seed": 3})
        assert started.status_code == 200
        assert client.post("/api/v1/scenarios/start", json={"scenario": "stable"}).status_code == 409
        assert client.put("/api/v1/policy", json={"policy": "lru"}).status_code == 409
        assert client.post("/api/v1/cache/reset").status_code == 409
        time.sleep(0.6)
        scenario = client.get("/api/v1/status").json()["scenario"]
        assert scenario["seed"] == 3 and scenario["trace_scenario"] == "abrupt"
        assert scenario["phase"] in {"original popularity", "viral release"}
        stopped = client.post("/api/v1/scenarios/stop").json()
        assert stopped["status"] in {"stopped", "complete"} and stopped["offered"] > 0
        assert client.get("/api/v1/status").json()["scenario"] is None
        assert client.put("/api/v1/policy", json={"policy": "lru"}).status_code == 200
        completed = client.post("/api/v1/scenarios/start", json={"scenario": "stable", "duration_seconds": 1.0, "request_rate": 20})
        run_id = completed.json()["run_id"]
        time.sleep(1.4)
        result = client.get(f"/api/v1/runs/{run_id}").json()
        assert result["status"] == "complete" and result["completed"] > 0
    saved = json.loads((tmp_path / "artifacts" / "live" / f"{run_id}.json").read_text())
    assert saved["complete"] is True


def test_artifact_runs_can_be_listed_and_loaded_safely(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, rows=16, capacity=4)
    runs = tmp_path / "artifacts" / "runs"
    run_dir = runs / "run-with-dash"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"run_id":"run-with-dash","backend":"cpu"}', encoding="utf-8")
    (run_dir / "summary.json").write_text('[{"policy":"lru"}]', encoding="utf-8")
    partial = runs / "partial-run"
    partial.mkdir()
    (partial / "manifest.json").write_text('{"run_id":"partial-run"}')
    (partial / "summary.json").write_text("[]")
    (partial / INCOMPLETE_MARKER).write_text("interrupted")
    (runs / ".hidden-temp").mkdir()
    (tmp_path / "secret.json").write_text('{"secret": true}')
    with TestClient(create_app()) as client:
        listed = {run["run_id"]: run["status"] for run in client.get("/api/v1/runs").json()["runs"]}
        assert listed["run-with-dash"] == "complete"
        assert listed["partial-run"] == "incomplete"
        assert ".hidden-temp" not in listed
        loaded = client.get("/api/v1/runs/run-with-dash")
        assert loaded.status_code == 200 and loaded.json()["summary"][0]["policy"] == "lru"
        assert client.get("/api/v1/runs/partial-run").status_code == 404
        for attempt in ("%2E%2E", "..%2F..%2Fsecret", "run.with.dots", "a" * 65):
            assert client.get(f"/api/v1/runs/{attempt}").status_code == 404


def test_open_loop_replay_records_overload_rejections(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, rows=1000, queue_capacity=2)
    generate_trace(tmp_path / "overload.jsonl", 1000, 120, 8, 1, "overload", request_rate=400.0)
    events = load_trace(tmp_path / "overload.jsonl")
    engines: list[SlowEngine] = []
    app = create_app(slow_factory(engines))

    async def exercise():
        async with app.router.lifespan_context(app):
            engines[0].delay = 0.01
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                started = time.perf_counter()
                rows = await replay_open_loop(client, events, rate_scale=1.0, max_in_flight=64)
                return rows, time.perf_counter() - started

    rows, wall = asyncio.run(exercise())
    summary = summarize_requests(rows, wall)
    assert summary["offered_requests"] == 120
    assert summary["successful_requests"] + summary["rejected_requests"] + summary["expired_requests"] + summary["dropped_dispatches"] + summary["failed_requests"] == 120
    assert summary["rejected_requests"] > 0, "overload phase must exceed queue capacity"
    assert summary["by_phase"]["1"]["offered"] == 60
    json.dumps(summary, allow_nan=False)
