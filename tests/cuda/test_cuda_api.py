import asyncio

import pytest
import yaml
from fastapi.testclient import TestClient

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.cuda


def _config(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    from hotshift.data.catalog import generate_catalog
    from hotshift.data.generate import generate_embeddings
    from hotshift.engine.cuda import build_cuda_extension

    build_cuda_extension()
    data_dir = tmp_path / "data"
    generate_embeddings(512, 8, 27, data_dir)
    generate_catalog(512, 8, 27, data_dir, candidate_count=16)
    config = {
        "profile": "cuda-test",
        "data": {"rows": 512, "dimensions": 8, "seed": 27, "output_dir": str(data_dir)},
        "cache": {"capacity": 32, "policy": "lru"},
        "limits": {
            "max_history_ids": 32,
            "max_batch_requests": 8,
            "max_batch_ids": 256,
            "queue_capacity": 256,
            "max_batch_delay_ms": 2,
        },
    }
    config_path = tmp_path / "cuda.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("HOTSHIFT_CONFIG", str(config_path))
    monkeypatch.setenv("HOTSHIFT_BACKEND", "cuda")
    monkeypatch.setenv("HOTSHIFT_ARTIFACT_DIR", str(tmp_path / "artifacts"))


def test_cuda_api_pool_and_recommendation(tmp_path, monkeypatch):
    from hotshift.serving.app import create_app

    _config(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").json()["backend"] == "cuda"
        status = client.get("/api/v1/status").json()
        assert status["backend"] == "cuda" and status["memory"]["device_tensor_bytes"] > 0
        pooled = client.post("/api/v1/embeddings/pool", json={"history_ids": [1, 1, 2]})
        assert pooled.status_code == 200
        assert pooled.json()["backend"] == "cuda"
        assert len(pooled.json()["vector"]) == 8
        recommended = client.post("/api/v1/recommend", json={"history_ids": [1, 2], "top_k": 3})
        assert recommended.status_code == 200
        assert len(recommended.json()["items"]) == 3
        for policy in ("adaptive", "static", "no_cache"):
            assert client.put("/api/v1/policy", json={"policy": policy}).status_code == 200
            again = client.post("/api/v1/embeddings/pool", json={"history_ids": [1, 1, 2]}).json()["vector"]
            assert again == pytest.approx(pooled.json()["vector"], abs=1e-5)


def test_cuda_concurrent_requests_batch_correctly(tmp_path, monkeypatch):
    import httpx
    import numpy as np

    from hotshift.data.store import EmbeddingStore
    from hotshift.engine.cpu import pool_mean_reference
    from hotshift.serving.app import create_app

    _config(tmp_path, monkeypatch)
    app = create_app()
    histories = [[(index * 7 + offset) % 512 for offset in range(index % 32)] for index in range(200)]

    async def exercise():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                responses = await asyncio.gather(*(client.post("/api/v1/embeddings/pool", json={"history_ids": history}) for history in histories))
                metrics = (await client.get("/api/v1/metrics")).json()
        return responses, metrics

    responses, metrics = asyncio.run(exercise())
    store = EmbeddingStore(tmp_path / "data")
    for history, response in zip(histories, responses):
        assert response.status_code == 200
        expected = pool_mean_reference(store, np.asarray(history, dtype=np.int64), [0, len(history)])[0]
        np.testing.assert_allclose(response.json()["vector"], expected, atol=1e-5, rtol=1e-4)
    assert metrics["counts"]["completed_requests"] == 200
    assert metrics["counts"]["batches"] < 200, "concurrent requests should share microbatches"
    assert max(response.json()["batch_size"] for response in responses) > 1
