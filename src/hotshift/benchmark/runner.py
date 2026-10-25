from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import yaml

from hotshift.config import load_config
from hotshift.data.generate import _sha256
from hotshift.data.store import EmbeddingStore
from hotshift.traffic.generate import generate_trace
from hotshift.traffic.replay import replay_trace


def run_matrix(matrix_path: str | Path, smoke: bool = False) -> Path:
    matrix_path = Path(matrix_path)
    config = load_config(matrix_path)
    raw = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    benchmark = raw.get("benchmark", {})
    scenarios = ["stable"] if smoke else benchmark.get("scenarios", ["stable"])
    policies = ["lru", "adaptive"] if smoke else benchmark.get(
        "policies", ["no_cache", "static", "lru", "decayed_lfu", "adaptive"]
    )
    batch_sizes = [16] if smoke else benchmark.get("microbatch_sizes", [1, 16, 64])
    request_count = 500 if smoke else int(benchmark.get("requests", 100000))
    trace_history_length = 8 if smoke else int(benchmark.get("history_length", 32))
    store = EmbeddingStore(config.output_dir)
    output_root = Path(os.getenv("HOTSHIFT_ARTIFACT_DIR", "artifacts")) / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=output_root))
    traces_dir = temporary / "traces"
    traces_dir.mkdir()
    summaries: list[dict[str, Any]] = []
    event_lines: list[dict[str, Any]] = []
    raw_batch_samples: list[dict[str, Any]] = []
    trace_checksums: dict[str, str] = {}
    try:
        for scenario in scenarios:
            trace_path = traces_dir / f"{scenario}.jsonl"
            trace_manifest = generate_trace(
                output_path=trace_path,
                rows=store.rows,
                requests=request_count,
                history_length=trace_history_length,
                seed=42,
                scenario=scenario,
            )
            trace_checksums[scenario] = str(trace_manifest["trace_sha256"])
            event_lines.append({"event": "trace_generated", "scenario": scenario, **trace_manifest})
            for batch_size in batch_sizes:
                for policy_name in policies:
                    summary = replay_trace(
                        trace_path=trace_path,
                        store=store,
                        capacity=config.cache_capacity,
                        policy_name=policy_name,
                        batch_size=int(batch_size),
                    )
                    summary.update({"run_id": run_id, "scenario": scenario, "batch_size": batch_size})
                    raw_batch_samples.extend(
                        {
                            "scenario": scenario,
                            "policy": policy_name,
                            "batch_size": batch_size,
                            "sample_index": index,
                            "engine_ms": sample,
                        }
                        for index, sample in enumerate(summary["batch_engine_samples_ms"])
                    )
                    summary.pop("batch_engine_samples_ms")
                    event_lines.append(
                        {
                            "event": "policy_replay_complete",
                            "scenario": scenario,
                            "policy": policy_name,
                            "batch_size": batch_size,
                            "measured_requests": summary["measured_requests"],
                        }
                    )
                    summaries.append(summary)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "backend": "cpu",
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": _version_or_none("torch"),
            },
            "policy_mode": "offline deterministic replay",
            "dataset_version": store.manifest["dataset_version"],
            "dataset_sha256": store.manifest["embedding_sha256"],
            "trace_sha256": trace_checksums,
            "rows": store.rows,
            "dimensions": store.dimensions,
            "cache_capacity": config.cache_capacity,
            "cache_budget_bytes": config.cache_bytes,
            "configuration": raw,
            "git_commit": _git_commit(),
            "smoke": smoke,
            "limitations": [
                "CPU reference execution; no CUDA kernel timing or GPU performance claims",
                "CPU policy simulator hit rates do not imply latency improvements",
            ],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (temporary / "summary.json").write_text(
            json.dumps(summaries, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        with (temporary / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = [
                "window_index",
                "window_scope",
                "scenario",
                "policy",
                "batch_size",
                "occurrence_hit_rate",
                "unique_hit_rate",
                "request_throughput",
                "host_miss_bytes",
            ]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for result in summaries:
                row = {name: result.get(name) for name in columns}
                row["window_index"] = 0
                row["window_scope"] = "entire_measured_evaluation_trace"
                writer.writerow(row)
        with (temporary / "batch_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = ["scenario", "policy", "batch_size", "sample_index", "engine_ms"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(raw_batch_samples)
        with (temporary / "events.jsonl").open("w", encoding="utf-8") as stream:
            for event in event_lines:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        checksums = {
            path.relative_to(temporary).as_posix(): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "checksums.json"
        }
        (temporary / "checksums.json").write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        destination = output_root / run_id
        temporary.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _version_or_none(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None

