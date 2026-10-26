"""Operator-level CUDA microbenchmarks and a CUPTI-trace profiler capture.

The kernel benchmark compares three ways to produce pooled means from cache and miss rows:

* ``fused_pool``: the owned ``hotshift::pool_mean`` kernel reading through descriptors;
* ``gather_then_bag``: owned ``hotshift::gather_rows`` materializing unique rows, then
  PyTorch ``embedding_bag`` over them (the intermediate [U,D] buffer is included);
* ``resident_bag``: PyTorch ``embedding_bag`` directly over a resident table (no misses,
  no descriptors), a library reference for the pooling arithmetic alone.

Timings are CUDA event intervals over repeated launches after warmup; they exclude host
planning and transfers and are labeled as device operation spans.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from statistics import median
from typing import Any

from hotshift.benchmark.common import (
    SCHEMA_VERSION,
    RunDirectory,
    environment,
    git_state,
    new_run_id,
    write_json,
)

DIMENSIONS = (32, 128, 256)
BAG_LENGTHS = (1, 8, 32, 128, 256)


def _require_cuda():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("kernel benchmarks require a CUDA-enabled PyTorch build") from error
    if not torch.cuda.is_available():
        raise RuntimeError("kernel benchmarks require an available CUDA device")
    from hotshift.engine.cuda import build_cuda_extension

    build_cuda_extension()
    return torch


def _time(torch, operation, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        operation()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        begin.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0)
    return float(median(samples))


def _kernel_case(torch, table, generator, batch: int, bag_length: int, miss_fraction: float, repeats: int, warmup: int) -> dict[str, Any]:
    device = table.device
    cache_rows, dimensions = table.shape
    tokens = batch * bag_length
    ids = torch.randint(0, cache_rows, (tokens,), generator=generator)
    unique, inverse = torch.unique(ids, return_inverse=True)
    unique_count = unique.numel()
    miss_count = int(unique_count * miss_fraction)
    # The first miss_count unique IDs come from a separate miss buffer, the rest from cache.
    source = torch.empty(unique_count, dtype=torch.int32)
    source[:miss_count] = -(torch.arange(miss_count, dtype=torch.int32) + 1)
    source[miss_count:] = unique[miss_count:].to(torch.int32)
    miss_rows = table[unique[:miss_count].to(device)].contiguous()
    source_d = source.to(device)
    inverse_d = inverse.to(device)
    offsets_d = (torch.arange(batch + 1, dtype=torch.int64) * bag_length).to(device)
    ids_d = ids.to(device)

    def fused():
        return torch.ops.hotshift.pool_mean(table, miss_rows, source_d, inverse_d, offsets_d)

    def gather_then_bag():
        rows = torch.ops.hotshift.gather_rows(table, miss_rows, source_d)
        return torch.nn.functional.embedding_bag(inverse_d, rows, offsets_d, mode="mean", include_last_offset=True)

    def resident_bag():
        return torch.nn.functional.embedding_bag(ids_d, table, offsets_d, mode="mean", include_last_offset=True)

    expected = resident_bag()
    row = {
        "dimensions": int(dimensions),
        "bag_length": bag_length,
        "batch": batch,
        "tokens": tokens,
        "unique_ids": unique_count,
        "miss_rows": miss_count,
        "fused_pool_us": _time(torch, fused, repeats, warmup),
        "gather_then_bag_us": _time(torch, gather_then_bag, repeats, warmup),
        "resident_bag_us": _time(torch, resident_bag, repeats, warmup),
        "fused_max_abs_error": float((fused() - expected).abs().max().item()),
        "gather_max_abs_error": float((gather_then_bag() - expected).abs().max().item()),
        "intermediate_bytes_gather": unique_count * int(dimensions) * 4,
        "useful_row_bytes": tokens * int(dimensions) * 4,
    }
    row["fused_vs_gather_ratio"] = row["gather_then_bag_us"] / row["fused_pool_us"] if row["fused_pool_us"] else None
    return row


def run_kernel_benchmark(
    batch: int = 64,
    cache_rows: int = 65536,
    miss_fraction: float = 0.25,
    repeats: int = 50,
    warmup: int = 10,
    seed: int = 42,
) -> Path:
    torch = _require_cuda()
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    results: list[dict[str, Any]] = []
    for dimensions in DIMENSIONS:
        table = torch.randn((cache_rows, dimensions), generator=generator).to(device)
        for bag_length in BAG_LENGTHS:
            results.append(_kernel_case(torch, table, generator, batch, bag_length, miss_fraction, repeats, warmup))
        del table
        torch.cuda.empty_cache()
    root = Path(os.getenv("HOTSHIFT_ARTIFACT_DIR", "artifacts")) / "runs"
    run_id = new_run_id("kernels")
    with RunDirectory(root, run_id) as run:
        write_json(run.path / "manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "kind": "kernel_microbenchmark",
            "backend": "cuda",
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            **git_state(),
            "environment": environment(include_cuda=True),
            "parameters": {"batch": batch, "cache_rows": cache_rows, "miss_fraction": miss_fraction, "repeats": repeats, "warmup": warmup, "seed": seed},
            "timing_scope": "median CUDA event interval per call after warmup; device operation span only",
            "notes": [
                "useful_row_bytes counts requested row payload, not measured DRAM traffic",
                "resident_bag is a library reference without descriptors or misses",
            ],
        })
        write_json(run.path / "summary.json", results)
    return run.destination


def run_torch_profile(matrix_path: str | Path, requests: int = 2048, batch_size: int = 16) -> Path:
    """Replay a short trace under torch.profiler (CUPTI activity tracing, no counter access)."""
    _require_cuda()
    from torch.profiler import ProfilerActivity, profile

    from hotshift.config import load_config
    from hotshift.data.store import EmbeddingStore
    from hotshift.engine.base import limits_from_config
    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.factory import create_policy
    from hotshift.traffic.generate import generate_trace
    from hotshift.traffic.replay import load_trace, replay_engine, summarize

    config = load_config(matrix_path)
    store = EmbeddingStore(config.output_dir)
    root = Path(os.getenv("HOTSHIFT_ARTIFACT_DIR", "artifacts")) / "profiles"
    run_id = new_run_id("torchprof")
    with RunDirectory(root, run_id) as run:
        trace_path = run.path / "abrupt.jsonl"
        generate_trace(trace_path, store.rows, requests=requests, history_length=32, seed=config.seed, scenario="abrupt")
        events = load_trace(trace_path)
        engine = CUDACacheEngine(store, config.cache_capacity, create_policy("lru"), limits=limits_from_config(config))
        engine.record_functions = True
        engine.pool([[0, 1, 2, 3]])
        engine.reset()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as profiler:
            records = replay_engine(events, engine, batch_size, max(1, requests // 20))
        profiler.export_chrome_trace(str(run.path / "trace.json"))
        table = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
        (run.path / "key_averages.txt").write_text(table + "\n", encoding="utf-8")
        stage_rows = []
        for event in profiler.key_averages():
            if event.key.startswith("hotshift."):
                stage_rows.append({
                    "name": event.key,
                    "count": int(event.count),
                    "cpu_time_total_us": float(event.cpu_time_total),
                    "device_time_total_us": float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))),
                })
        summary = summarize(records)
        write_json(run.path / "summary.json", {"stages": stage_rows, "replay": summary})
        write_json(run.path / "manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "kind": "torch_profiler_capture",
            "backend": "cuda",
            **git_state(),
            "environment": environment(include_cuda=True),
            "requests": requests,
            "batch_size": batch_size,
            "policy": "lru",
            "notes": [
                "torch.profiler uses CUPTI activity tracing, which does not require GPU performance-counter permission",
                "profiler-instrumented timings are diagnostic and are not used as headline benchmark numbers",
            ],
        })
        engine.close()
    return run.destination
