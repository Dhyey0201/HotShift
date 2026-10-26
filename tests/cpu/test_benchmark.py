import gzip
import json

import pytest
import yaml

from hotshift.benchmark.common import INCOMPLETE_MARKER, RunDirectory, is_complete_run, sha256_file
from hotshift.benchmark.report import write_report
from hotshift.benchmark.runner import run_matrix
from hotshift.data.generate import generate_embeddings


def _matrix(tmp_path, **benchmark):
    data_dir = tmp_path / "data"
    generate_embeddings(2000, 8, 42, data_dir)
    config = {
        "profile": "test",
        "data": {"rows": 2000, "dimensions": 8, "seed": 42, "output_dir": str(data_dir)},
        "cache": {"capacity": 200, "policy": "adaptive"},
        "limits": {"max_history_ids": 256, "max_batch_requests": 64, "max_batch_ids": 16384, "queue_capacity": 64, "max_batch_delay_ms": 2},
        "benchmark": benchmark,
    }
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_cpu_matrix_writes_complete_verified_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTSHIFT_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    matrix = _matrix(
        tmp_path,
        scenarios=["stable", "abrupt"],
        policies=["no_cache", "static", "lru", "decayed_lfu", "adaptive"],
        baselines=[],
        microbatch_sizes=[1, 16],
        requests=300,
        history_length=8,
        repeats=2,
        cache_conditions=["warm", "cold"],
        verify_every=5,
    )
    run_dir = run_matrix(matrix)
    assert is_complete_run(run_dir)
    assert not (run_dir / INCOMPLETE_MARKER).exists()
    checksums = json.loads((run_dir / "checksums.json").read_text())
    for name, digest in checksums.items():
        assert sha256_file(run_dir / name) == digest
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    aggregate = json.loads((run_dir / "aggregate.json").read_text())
    assert manifest["backend"] == "cpu" and "git_dirty" in manifest
    # 2 scenarios x 2 seeds x 2 batch sizes x (5 warm + 4 cold, static skipped when cold)
    assert len(summary) == 2 * 2 * 2 * 9
    assert all(row["output_check"]["passed"] for row in summary)
    assert {row["trace_sha256"] for row in summary if row["scenario"] == "stable" and row["seed"] == 42} == {manifest["trace_sha256"]["stable-seed42"]}
    abrupt = [row for row in summary if row["scenario"] == "abrupt" and row["condition"] == "warm"]
    assert all(row["recovery"]["status"] in {"recovered", "not_recovered", "not_applicable"} for row in abrupt)
    assert all(row["recovery"]["target_hit_rate"] is not None for row in abrupt if row["policy"] != "no_cache")
    assert all(group["repeats"] == 2 for group in aggregate)
    with gzip.open(run_dir / "batch_samples.csv.gz", "rt") as stream:
        assert stream.readline().startswith("scenario,seed,condition")
    assert (run_dir / "windows.csv").read_text().count("\n") > 10
    report = write_report(run_dir.parent, tmp_path / "report" / "REPORT.md")
    text = report.read_text()
    assert "Unfavorable and low-benefit cases" in text
    assert "CPU simulator timings are not GPU timings" in text


def test_interrupted_run_is_marked_incomplete(tmp_path):
    with pytest.raises(KeyboardInterrupt), RunDirectory(tmp_path, "run-1") as run:
        (run.path / "summary.json").write_text("[]")
        raise KeyboardInterrupt
    assert (tmp_path / "run-1" / INCOMPLETE_MARKER).read_text().startswith("run did not complete")
    assert not is_complete_run(tmp_path / "run-1")


def test_cuda_benchmark_refuses_without_device(tmp_path):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present")
    with pytest.raises(RuntimeError, match="CUDA"):
        run_matrix(_matrix(tmp_path, requests=10), backend="cuda")
