"""Controlled fixed-batch replay matrix shared by the CPU and CUDA backends.

For each scenario and seed, one immutable trace is generated and replayed against every
policy and baseline with identical batch partitions, cache budget, and limits. Policy order
rotates per seed. Every engine first runs a short context/allocator warmup and is then reset
to its declared initial condition: ``warm`` replays the trace's warmup prefix (excluded from
metrics) and ``cold`` measures from an empty cache.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import yaml

from hotshift.benchmark.common import (
    SCHEMA_VERSION,
    RunDirectory,
    environment,
    git_state,
    new_run_id,
    write_csv,
    write_json,
)
from hotshift.config import BASELINE_NAMES, HotShiftConfig, load_config
from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import CPUCacheEngine, limits_from_config
from hotshift.policies.factory import create_policy
from hotshift.policies.static import StaticPolicy
from hotshift.traffic.generate import generate_trace
from hotshift.traffic.replay import (
    WINDOW_COLUMNS,
    OutputCheck,
    load_trace,
    phase_boundary,
    recovery,
    replay_engine,
    summarize,
    warm_items_from_trace,
    windows,
)

CUDA_VARIANTS: dict[str, dict[str, bool]] = {
    "optimized": {},
    "pageable_staging": {"pinned_staging": False},
    "host_admission": {"admit_from_miss_buffer": False},
}
SAMPLE_COLUMNS = [
    "scenario", "seed", "condition", "cache_fraction", "policy", "variant", "batch_size",
    "sample_index", "start_event", "phase_id", "warmup", "requests", "wall_ms", "device_ms",
]


class CUDAEventTimer:
    """CUDA event interval on the current stream around one engine call."""

    def __init__(self, torch) -> None:
        self.torch = torch
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self.begin.record()

    def stop(self) -> float:
        self.end.record()
        self.end.synchronize()
        return float(self.begin.elapsed_time(self.end))


def _settings(raw: dict[str, Any], smoke: bool, backend: str) -> dict[str, Any]:
    benchmark = raw.get("benchmark", {})
    default_baselines = ["torch_cpu"] if backend == "cpu" else list(BASELINE_NAMES)
    settings = {
        "scenarios": benchmark.get("scenarios", ["stable", "abrupt", "brief_burst", "uniform"]),
        "policies": benchmark.get("policies", ["no_cache", "static", "lru", "decayed_lfu", "adaptive"]),
        "baselines": benchmark.get("baselines", default_baselines),
        "microbatch_sizes": [int(size) for size in benchmark.get("microbatch_sizes", [1, 16, 64])],
        "requests": int(benchmark.get("requests", 100000)),
        "history_length": int(benchmark.get("history_length", 32)),
        "repeats": int(benchmark.get("repeats", 3)),
        "cache_fractions": [float(value) for value in benchmark.get("cache_fractions", [])],
        "cache_conditions": benchmark.get("cache_conditions", ["warm"]),
        "engine_variants": benchmark.get("engine_variants", ["optimized"]),
        "window_requests": benchmark.get("window_requests"),
        "verify_every": int(benchmark.get("verify_every", 50)),
    }
    if smoke:
        settings.update({
            "scenarios": ["stable", "abrupt"],
            "policies": ["no_cache", "static", "lru", "adaptive"],
            "microbatch_sizes": [16],
            "requests": 512,
            "history_length": 8,
            "repeats": 1,
            "engine_variants": ["optimized"],
            "cache_conditions": ["warm"],
            "verify_every": 4,
        })
    if backend == "cpu":
        settings["engine_variants"] = ["optimized"]
        settings["baselines"] = [name for name in settings["baselines"] if name == "torch_cpu" and _has_torch()]
    if settings["window_requests"] is None:
        settings["window_requests"] = max(1, settings["requests"] // 100)
    return settings


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _make_engine(backend: str, name: str, store: EmbeddingStore, capacity: int, warm_items: list[int], variant: str, config: HotShiftConfig):
    limits = limits_from_config(config)
    if name in BASELINE_NAMES:
        from hotshift.engine.baselines import TorchBaselineEngine

        return TorchBaselineEngine(store, name, limits=limits)
    policy = StaticPolicy(warm_items) if name == "static" else create_policy(name)
    budget = 0 if name == "no_cache" else capacity
    if backend == "cpu":
        return CPUCacheEngine(store, budget, policy, limits)
    from hotshift.engine.cuda import CUDACacheEngine

    return CUDACacheEngine(store, budget, policy, limits=limits, time_kernels=True, **CUDA_VARIANTS[variant])


def _warm_engine(engine) -> None:
    """Exercise context, allocator, and kernels, then restore the declared initial state."""
    rows = engine.store.rows
    sample = [[item % rows for item in range(index, index + 4)] for index in range(4)]
    for _ in range(3):
        engine.pool(sample)
    engine.reset()


def _initial_condition(name: str, condition: str) -> str:
    if name in BASELINE_NAMES:
        return "no_cache_tier"
    if name == "static":
        return "preloaded_from_trace_warmup_prefix"
    if name == "no_cache":
        return "empty_zero_capacity"
    return "empty_then_trace_warmup_prefix_replayed" if condition == "warm" else "empty_cold"


def run_matrix(matrix_path: str | Path, smoke: bool = False, backend: str = "cpu") -> Path:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")
    torch = None
    if backend == "cuda":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("install a CUDA-enabled PyTorch build before running the CUDA benchmark") from error
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requires an available CUDA device")
        from hotshift.engine.cuda import build_cuda_extension

        build_cuda_extension()
    matrix_path = Path(matrix_path)
    config = load_config(matrix_path)
    raw = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    settings = _settings(raw, smoke, backend)
    store = EmbeddingStore(config.output_dir)
    fractions = settings["cache_fractions"] or [config.cache_capacity / store.rows]
    window_requests = int(settings["window_requests"])
    seeds = [config.seed + index for index in range(settings["repeats"])]
    root = Path(os.getenv("HOTSHIFT_ARTIFACT_DIR", "artifacts")) / "runs"
    run_id = new_run_id("cuda" if backend == "cuda" else "cpu")

    summaries: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    events_log: list[dict[str, Any]] = []
    trace_hashes: dict[str, str] = {}
    reference_targets: dict[tuple, float | None] = {}
    warnings: list[str] = []
    engine_memory: dict[str, Any] = {}
    timer = CUDAEventTimer(torch) if torch is not None else None

    with RunDirectory(root, run_id) as run:
        traces_dir = run.path / "traces"
        traces_dir.mkdir()

        def replay_one(events, name, capacity, warm_items, variant, batch_size, condition):
            if torch is not None:
                torch.cuda.empty_cache()
            engine = _make_engine(backend, name, store, capacity, warm_items, variant, config)
            try:
                _warm_engine(engine)
                if torch is not None:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                check = OutputCheck()
                records = replay_engine(
                    events, engine, batch_size, window_requests,
                    device_timer=timer, verify_every=settings["verify_every"], output_check=check,
                )
                memory = engine.memory()
                if torch is not None:
                    memory["allocator_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                    memory["allocator_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                return records, check, memory, len(engine.resident)
            finally:
                engine.close()
                del engine

        for scenario in settings["scenarios"]:
            for seed_index, seed in enumerate(seeds):
                trace_path = traces_dir / f"{scenario}-seed{seed}.jsonl"
                trace_manifest = generate_trace(
                    trace_path, store.rows, requests=settings["requests"],
                    history_length=settings["history_length"], seed=seed, scenario=scenario,
                )
                trace_hashes[f"{scenario}-seed{seed}"] = str(trace_manifest["trace_sha256"])
                events_log.append({"event": "trace_generated", **trace_manifest})
                events = load_trace(trace_path)
                boundary = phase_boundary(events) if scenario == "abrupt" else None
                reference_events = None
                if boundary is not None:
                    # Warmed post-shift baseline: same budget and post-shift distribution.
                    reference_path = traces_dir / f"post_shift-seed{seed}.jsonl"
                    reference_manifest = generate_trace(
                        reference_path, store.rows, requests=max(2, settings["requests"] // 2),
                        history_length=settings["history_length"], seed=seed,
                        scenario="post_shift", warmup_fraction=0.5,
                    )
                    trace_hashes[f"post_shift-seed{seed}"] = str(reference_manifest["trace_sha256"])
                    events_log.append({"event": "trace_generated", **reference_manifest})
                    reference_events = load_trace(reference_path)
                for fraction in fractions:
                    capacity = max(1, min(store.rows, round(store.rows * fraction)))
                    warm_items, warm_requests = warm_items_from_trace(events, capacity)
                    for batch_size in settings["microbatch_sizes"]:
                        for condition in settings["cache_conditions"]:
                            names = list(settings["policies"])
                            if fraction == fractions[0] and condition == settings["cache_conditions"][0]:
                                names += list(settings["baselines"])
                            rotation = seed_index % len(names)
                            names = names[rotation:] + names[:rotation]
                            for name in names:
                                if condition == "cold" and name == "static":
                                    continue
                                variants = ["optimized"] if name in BASELINE_NAMES or name == "no_cache" else settings["engine_variants"]
                                for variant in variants:
                                    records, check, memory, occupancy = replay_one(
                                        events, name, capacity, warm_items, variant, batch_size, condition
                                    )
                                    summary = summarize(records, measure_warmup=condition == "cold")
                                    if name in BASELINE_NAMES and name == "torch_cpu":
                                        summary["occurrence_hit_rate"] = None
                                        summary["unique_hit_rate"] = None
                                        summary["null_reasons"]["occurrence_hit_rate"] = "CPU baseline has no GPU tier"
                                    target = None
                                    if reference_events is not None and condition == "warm" and name not in BASELINE_NAMES:
                                        key = (seed, fraction, batch_size, name, variant)
                                        if key not in reference_targets:
                                            ref_warm, _ = warm_items_from_trace(reference_events, capacity)
                                            ref_records, _, _, _ = replay_one(
                                                reference_events, name, capacity, ref_warm, variant, batch_size, "warm"
                                            )
                                            reference_targets[key] = summarize(ref_records)["occurrence_hit_rate"]
                                        target = reference_targets[key]
                                    rows = windows(records, window_requests)
                                    summary.update({
                                        "backend": backend if name not in BASELINE_NAMES else ("cuda" if name == "torch_gpu_resident" else "cpu"),
                                        "engine": "baseline" if name in BASELINE_NAMES else ("cuda_cache" if backend == "cuda" else "cpu_cache_simulator"),
                                        "run_id": run_id,
                                        "scenario": scenario,
                                        "seed": seed,
                                        "policy": name,
                                        "variant": variant,
                                        "batch_size": batch_size,
                                        "condition": condition,
                                        "initial_cache_condition": _initial_condition(name, condition),
                                        "cache_fraction": fraction,
                                        "capacity": 0 if name in ("no_cache", *BASELINE_NAMES) else capacity,
                                        "configured_capacity": capacity,
                                        "cache_budget_bytes": capacity * store.dimensions * 4,
                                        "warmup_rule": f"first {warm_requests} trace requests replayed, excluded from metrics" if condition == "warm" else "none; all requests measured",
                                        "static_warm_items": len(warm_items) if name == "static" else None,
                                        "final_cache_occupancy": occupancy,
                                        "trace_sha256": trace_hashes[f"{scenario}-seed{seed}"],
                                        "output_check": check.as_dict(),
                                        "memory": memory,
                                        "recovery": recovery(records, rows, boundary, target) if boundary is not None else None,
                                        "warnings": [] if check.failures == 0 else [f"{check.failures} verified batches exceeded tolerance"],
                                    })
                                    if name == "torch_gpu_resident":
                                        summary["warnings"].append("higher-memory reference: full table resident on the device; not an equal-budget competitor")
                                    if check.failures:
                                        warnings.append(f"{scenario}/{seed}/{name}/{variant}/batch{batch_size}: output mismatch")
                                    engine_memory[name] = memory
                                    summaries.append(summary)
                                    labels = {
                                        "scenario": scenario, "seed": seed, "condition": condition,
                                        "cache_fraction": fraction, "policy": name, "variant": variant,
                                        "batch_size": batch_size,
                                    }
                                    window_rows.extend({**labels, **row} for row in rows)
                                    samples.extend(
                                        {
                                            **labels, "sample_index": index, "start_event": record.start_event,
                                            "phase_id": record.phase_id, "warmup": record.warmup,
                                            "requests": record.requests, "wall_ms": record.wall_ms,
                                            "device_ms": record.device_ms,
                                        }
                                        for index, record in enumerate(records)
                                    )
                                    events_log.append({
                                        "event": "replay_complete", **labels,
                                        "measured_requests": summary["measured_requests"],
                                        "output_check_passed": check.as_dict()["passed"],
                                    })

        aggregate = aggregate_summaries(summaries)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "backend": backend,
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            **git_state(),
            "environment": environment(include_cuda=backend == "cuda"),
            "policy_mode": "serialized CUDA cache execution with deterministic offline replay" if backend == "cuda" else "CPU cache simulator with deterministic offline replay",
            "dataset_version": store.manifest["dataset_version"],
            "dataset_sha256": store.manifest["embedding_sha256"],
            "trace_sha256": trace_hashes,
            "rows": store.rows,
            "dimensions": store.dimensions,
            "configured_cache_capacity": config.cache_capacity,
            "cache_budget_bytes": config.cache_bytes,
            "cache_fractions": fractions,
            "window_requests": window_requests,
            "selected_matrix": settings,
            "configuration": raw,
            "smoke": smoke,
            "sample_counts": {"summaries": len(summaries), "windows": len(window_rows), "batch_samples": len(samples)},
            "memory_by_engine": engine_memory,
            "warnings": warnings,
            "timing_notes": [
                "batch_wall_ms spans one engine call: host planning, transfers, synchronization, kernels, policy update, and maintenance",
                "device_ms is a CUDA event interval on the current stream around the engine call; it includes idle gaps while the host plans",
                "request_throughput divides measured requests by summed batch wall time and excludes trace parsing",
                "this serialized implementation does not overlap host planning with GPU work",
                "per-request latency in offline replay equals its batch wall time; batch_samples.csv.gz is the documented requests.csv equivalent",
            ],
            "limitations": [
                "measurements apply only to the recorded hardware, driver, toolkit, data, and workload",
                "no speedup is claimed without repeated matched-baseline analysis",
                "CPU backend runs a cache simulator; its hit rates are policy semantics, not GPU performance",
            ],
        }
        write_json(run.path / "manifest.json", manifest)
        write_json(run.path / "summary.json", summaries)
        write_json(run.path / "aggregate.json", aggregate)
        write_csv(run.path / "windows.csv", window_rows, ["scenario", "seed", "condition", "cache_fraction", "policy", "variant", "batch_size", *WINDOW_COLUMNS])
        write_csv(run.path / "batch_samples.csv.gz", samples, SAMPLE_COLUMNS)
        with (run.path / "events.jsonl").open("w", encoding="utf-8") as stream:
            import json

            for event in events_log:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    return run.destination


AGGREGATE_FIELDS = (
    "request_throughput", "useful_lookup_ids_per_second", "occurrence_hit_rate", "unique_hit_rate",
    "batch_wall_ms_p50", "batch_wall_ms_p99", "device_ms_p50", "host_miss_bytes", "h2d_bytes",
    "policy_overhead_ms",
)


def aggregate_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median and range across repeated seeds; individual runs stay in summary.json."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for summary in summaries:
        key = (summary["scenario"], summary["policy"], summary["variant"], summary["batch_size"], summary["condition"], summary["cache_fraction"])
        groups.setdefault(key, []).append(summary)
    rows = []
    for (scenario, policy, variant, batch_size, condition, fraction), group in groups.items():
        row: dict[str, Any] = {
            "scenario": scenario, "policy": policy, "variant": variant, "batch_size": batch_size,
            "condition": condition, "cache_fraction": fraction, "backend": group[0]["backend"],
            "repeats": len(group), "seeds": [item["seed"] for item in group],
            "output_check_passed": all(item["output_check"]["passed"] for item in group),
        }
        for name in AGGREGATE_FIELDS:
            values = [item.get(name) for item in group if item.get(name) is not None]
            row[f"{name}_median"] = float(median(values)) if values else None
            row[f"{name}_min"] = float(np.min(values)) if values else None
            row[f"{name}_max"] = float(np.max(values)) if values else None
        outcomes = [item["recovery"]["status"] for item in group if item.get("recovery")]
        recovered = [item["recovery"]["recovery_requests"] for item in group if item.get("recovery") and item["recovery"]["status"] == "recovered"]
        row["recovery_statuses"] = outcomes or None
        row["recovery_requests_median"] = float(median(recovered)) if recovered and len(recovered) == len(outcomes) else None
        rows.append(row)
    return rows
