import yaml
from fastapi.testclient import TestClient

from hotshift.data.catalog import generate_catalog
from hotshift.data.generate import generate_embeddings
from hotshift.serving.app import create_app


def test_cpu_api_pool_recommend_and_policy_controls(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    generate_embeddings(128, 8, 42, data_dir)
    generate_catalog(128, 8, 42, data_dir, candidate_count=32)
    config = {
        "profile": "test",
        "data": {"rows": 128, "dimensions": 8, "seed": 42, "output_dir": str(data_dir)},
        "cache": {"capacity": 16, "policy": "lru"},
        "limits": {
            "max_history_ids": 256,
            "max_batch_requests": 64,
            "max_batch_ids": 16384,
            "queue_capacity": 32,
            "max_batch_delay_ms": 2,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("HOTSHIFT_CONFIG", str(config_path))
    monkeypatch.setenv("HOTSHIFT_BACKEND", "cpu")
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        pooled = client.post("/api/v1/embeddings/pool", json={"history_ids": [1, 1]})
        assert pooled.status_code == 200
        assert len(pooled.json()["vector"]) == 8
        recommended = client.post("/api/v1/recommend", json={"history_ids": [1], "top_k": 3})
        assert recommended.status_code == 200
        assert len(recommended.json()["items"]) == 3
        assert client.put("/api/v1/policy", json={"policy": "adaptive"}).status_code == 200
        assert client.post("/api/v1/cache/reset").status_code == 200
        assert client.post("/api/v1/embeddings/pool", json={"history_ids": [-1]}).status_code == 422


def test_artifact_runs_can_be_listed_and_loaded_safely(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    generate_embeddings(16, 4, 42, data_dir)
    generate_catalog(16, 4, 42, data_dir, candidate_count=8)
    config = {
        "profile": "test",
        "data": {"rows": 16, "dimensions": 4, "seed": 42, "output_dir": str(data_dir)},
        "cache": {"capacity": 4, "policy": "lru"},
        "limits": {"max_history_ids": 16, "max_batch_requests": 4, "max_batch_ids": 64, "queue_capacity": 8, "max_batch_delay_ms": 2},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "artifacts" / "runs" / "run-with-dash"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"run_id":"run-with-dash","backend":"cpu"}', encoding="utf-8")
    (run_dir / "summary.json").write_text('[{"policy":"lru"}]', encoding="utf-8")
    monkeypatch.setenv("HOTSHIFT_CONFIG", str(config_path))
    monkeypatch.setenv("HOTSHIFT_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    with TestClient(create_app()) as client:
        assert any(run["run_id"] == "run-with-dash" for run in client.get("/api/v1/runs").json()["runs"])
        loaded = client.get("/api/v1/runs/run-with-dash")
        assert loaded.status_code == 200
        assert loaded.json()["summary"][0]["policy"] == "lru"
        assert client.get("/api/v1/runs/%2E%2E").status_code == 404

