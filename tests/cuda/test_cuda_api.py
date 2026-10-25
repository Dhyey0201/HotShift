import pytest
import yaml
from fastapi.testclient import TestClient

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.cuda


def test_cuda_api_pool_and_recommendation(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")

    from hotshift.data.catalog import generate_catalog
    from hotshift.data.generate import generate_embeddings
    from hotshift.engine.cuda import build_cuda_extension
    from hotshift.serving.app import create_app

    build_cuda_extension()
    data_dir = tmp_path / "data"
    generate_embeddings(32, 8, 27, data_dir)
    generate_catalog(32, 8, 27, data_dir, candidate_count=16)
    config = {
        "profile": "cuda-test",
        "data": {"rows": 32, "dimensions": 8, "seed": 27, "output_dir": str(data_dir)},
        "cache": {"capacity": 8, "policy": "lru"},
        "limits": {
            "max_history_ids": 32,
            "max_batch_requests": 8,
            "max_batch_ids": 256,
            "queue_capacity": 16,
            "max_batch_delay_ms": 2,
        },
    }
    config_path = tmp_path / "cuda.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("HOTSHIFT_CONFIG", str(config_path))
    monkeypatch.setenv("HOTSHIFT_BACKEND", "cuda")
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").json()["backend"] == "cuda"
        pooled = client.post("/api/v1/embeddings/pool", json={"history_ids": [1, 1, 2]})
        assert pooled.status_code == 200
        assert pooled.json()["backend"] == "cuda"
        assert len(pooled.json()["vector"]) == 8
        recommended = client.post("/api/v1/recommend", json={"history_ids": [1, 2], "top_k": 3})
        assert recommended.status_code == 200
        assert len(recommended.json()["items"]) == 3
