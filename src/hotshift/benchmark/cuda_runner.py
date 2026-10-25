from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import yaml

from hotshift.config import load_config
from hotshift.data.generate import _sha256
from hotshift.data.store import EmbeddingStore
from hotshift.engine.cuda import CUDACacheEngine, build_cuda_extension
from hotshift.policies.factory import create_policy
from hotshift.policies.static import StaticPolicy
from hotshift.traffic.generate import generate_trace


def run_cuda_matrix(matrix_path: str | Path, smoke: bool = False) -> Path:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install a CUDA-enabled PyTorch build before running the CUDA benchmark") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requires an available CUDA device")
    build_cuda_extension()
    matrix_path = Path(matrix_path)
    config = load_config(matrix_path)
    raw = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    settings = raw.get("benchmark", {})
    scenarios = (["stable", "abrupt"] if smoke else settings.get(
        "scenarios", ["stable", "abrupt", "brief_burst", "uniform"]
    ))
    policies = (["no_cache", "static", "lru", "adaptive"] if smoke else settings.get(
        "policies", ["no_cache", "static", "lru", "decayed_lfu", "adaptive"]
    ))
    batch_sizes = [16] if smoke else settings.get("microbatch_sizes", [1, 16, 64])
    requests = 256 if smoke else int(settings.get("requests", 100000))
    history_length = 8 if smoke else int(settings.get("history_length", 32))
    repeats = 1 if smoke else int(settings.get("repeats", 3))
    seeds = [config.seed + index for index in range(repeats)]
    store = EmbeddingStore(config.output_dir)

    root = Path(os.getenv("HOTSHIFT_ARTIFACT_DIR", "artifacts")) / "runs"
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-cuda-" + uuid4().hex[:8]
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=root))
    traces_dir = temporary / "traces"
    traces_dir.mkdir()
    summaries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    trace_hashes: dict[str, str] = {}

    try:
        for scenario in scenarios:
            for seed in seeds:
                trace_path = traces_dir / f"{scenario}-seed{seed}.jsonl"
                trace_manifest = generate_trace(
                    trace_path,
                    store.rows,
                    requests=requests,
                    history_length=history_length,
                    seed=seed,
                    scenario=scenario,
                )
                trace_hashes[f"{scenario}-seed{seed}"] = str(trace_manifest["trace_sha256"])
                events.append({"event": "trace_generated", **trace_manifest})
                for batch_size in batch_sizes:
                    policy_order = list(policies)
                    rotation = seeds.index(seed) % len(policy_order)
                    policy_order = policy_order[rotation:] + policy_order[:rotation]
                    for policy_name in policy_order:
                        policy, warm_count, warm_items = _policy_for_trace(
                            policy_name, trace_path, config.cache_capacity
                        )
                        capacity = 0 if policy_name == "no_cache" else config.cache_capacity
                        torch.cuda.empty_cache()
                        engine = CUDACacheEngine(store, capacity, policy)
                        torch.cuda.reset_peak_memory_stats(engine.device)
                        result, batch_samples = _replay(
                            trace_path, engine, int(batch_size), skip_warmup=policy_name == "static"
                        )
                        result.update({
                            "backend": "cuda",
                            "run_id": run_id,
                            "scenario": scenario,
                            "seed": seed,
                            "policy": policy_name,
                            "batch_size": int(batch_size),
                            "capacity": capacity,
                            "warmup_requests": warm_count,
                            "static_warm_items": len(warm_items) if policy_name == "static" else None,
                            "device_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(engine.device)),
                            "device_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(engine.device)),
                        })
                        summaries.append(result)
                        for sample_index, sample in enumerate(batch_samples):
                            samples.append({
                                "scenario": scenario,
                                "seed": seed,
                                "policy": policy_name,
                                "batch_size": int(batch_size),
                                "sample_index": sample_index,
                                **sample,
                            })
                        events.append({
                            "event": "cuda_replay_complete",
                            "scenario": scenario,
                            "seed": seed,
                            "policy": policy_name,
                            "batch_size": int(batch_size),
                            "measured_requests": result["measured_requests"],
                        })
                        del engine
                        torch.cuda.synchronize()

        gpu_name = torch.cuda.get_device_name(0)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "backend": "cuda",
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": gpu_name,
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
                "driver": _nvidia_driver(),
                "nvcc": _nvcc_version(),
            },
            "policy_mode": "serialized CUDA cache execution with deterministic offline replay",
            "dataset_version": store.manifest["dataset_version"],
            "dataset_sha256": store.manifest["embedding_sha256"],
            "trace_sha256": trace_hashes,
            "rows": store.rows,
            "dimensions": store.dimensions,
            "configured_cache_capacity": config.cache_capacity,
            "cache_budget_bytes": config.cache_bytes,
            "configuration": raw,
            "git_commit": _git_commit(),
            "smoke": smoke,
            "timing_notes": [
                "batch_wall_ms includes host planning, transfers, synchronization, kernels, and policy maintenance",
                "cuda_stream_ms is the CUDA event interval on the current stream; it includes device-side idle gaps between event records",
                "request_throughput divides measured requests by summed measured batch wall time and excludes trace file parsing",
                "this serialized implementation does not overlap host planning and GPU work",
            ],
            "limitations": [
                "measurements apply only to the recorded GPU, driver, toolkit, data, and workload",
                "no speedup is claimed without repeated matched-baseline analysis",
            ],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (temporary / "summary.json").write_text(
            json.dumps(summaries, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        with (temporary / "batch_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = ["scenario", "seed", "policy", "batch_size", "sample_index", "batch_wall_ms", "cuda_stream_ms"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(samples)
        with (temporary / "events.jsonl").open("w", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        checksums = {
            path.relative_to(temporary).as_posix(): _sha256(path)
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "checksums.json"
        }
        (temporary / "checksums.json").write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        destination = root / run_id
        temporary.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _policy_for_trace(name: str, trace_path: Path, capacity: int):
    warm_frequency: Counter[int] = Counter()
    warm_requests = 0
    with trace_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if not event.get("warmup"):
                break
            warm_requests += 1
            warm_frequency.update(int(item) for item in event["history_ids"])
    warm_items = [item for item, _ in sorted(warm_frequency.items(), key=lambda pair: (-pair[1], pair[0]))[:capacity]]
    policy = StaticPolicy(warm_items) if name == "static" else create_policy(name)
    return policy, warm_requests, warm_items


def _replay(trace_path: Path, engine: CUDACacheEngine, batch_size: int, skip_warmup: bool):
    batch: list[dict[str, Any]] = []
    batch_warmup: bool | None = None
    wall_samples: list[float] = []
    stream_samples: list[float] = []
    occurrence_hits = occurrences = unique_hits = unique_ids = miss_bytes = measured_requests = 0

    def flush() -> None:
        nonlocal occurrence_hits, occurrences, unique_hits, unique_ids, miss_bytes
        nonlocal measured_requests
        if not batch:
            return
        warmup = bool(batch_warmup)
        if warmup and skip_warmup:
            batch.clear()
            return
        start_event = engine.torch.cuda.Event(enable_timing=True)
        end_event = engine.torch.cuda.Event(enable_timing=True)
        start_event.record()
        started = perf_counter()
        with engine.torch.cuda.nvtx.range("hotshift.cache_batch"):
            result = engine.pool([event["history_ids"] for event in batch])
        end_event.record()
        end_event.synchronize()
        wall_ms = (perf_counter() - started) * 1000
        if not warmup:
            measured_requests += len(batch)
            occurrence_hits += result.occurrence_hits
            occurrences += result.occurrences
            unique_hits += result.unique_hits
            unique_ids += result.unique_ids
            miss_bytes += result.host_miss_bytes
            wall_samples.append(wall_ms)
            stream_samples.append(float(start_event.elapsed_time(end_event)))
        batch.clear()

    with trace_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            event_warmup = bool(event.get("warmup"))
            if batch and (event_warmup != batch_warmup or len(batch) >= batch_size):
                flush()
            if not batch:
                batch_warmup = event_warmup
            batch.append(event)
        flush()

    elapsed = sum(wall_samples) / 1000.0
    return ({
        "offered_requests": measured_requests,
        "successful_requests": measured_requests,
        "failed_requests": 0,
        "expired_requests": 0,
        "cancelled_requests": 0,
        "measured_requests": measured_requests,
        "occurrences": occurrences,
        "occurrence_hits": occurrence_hits,
        "occurrence_hit_rate": occurrence_hits / occurrences if occurrences else None,
        "unique_ids": unique_ids,
        "unique_hits": unique_hits,
        "unique_hit_rate": unique_hits / unique_ids if unique_ids else None,
        "host_miss_bytes": miss_bytes,
        "wall_seconds": elapsed,
        "throughput_scope": "sum of measured batch wall times; excludes trace file reading and parsing",
        "request_throughput": measured_requests / elapsed if elapsed else None,
        "useful_lookup_ids_per_second": occurrences / elapsed if elapsed else None,
        "batch_wall_ms_p50": median(wall_samples) if wall_samples else None,
        "batch_wall_ms_p95": float(np.percentile(wall_samples, 95)) if wall_samples else None,
        "batch_wall_ms_p99": float(np.percentile(wall_samples, 99)) if wall_samples else None,
        "cuda_stream_ms_p50": median(stream_samples) if stream_samples else None,
        "cuda_stream_ms_p95": float(np.percentile(stream_samples, 95)) if stream_samples else None,
        "cuda_stream_ms_p99": float(np.percentile(stream_samples, 99)) if stream_samples else None,
        "batch_sample_count": len(wall_samples),
        "cache_occupancy": len(engine.resident),
    }, [
        {"batch_wall_ms": wall, "cuda_stream_ms": device}
        for wall, device in zip(wall_samples, stream_samples)
    ])


def _nvidia_driver() -> str | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _nvcc_version() -> str | None:
    executable = shutil.which("nvcc")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, check=True, timeout=5
        )
        return result.stdout.strip().splitlines()[-1]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
