from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import CPUCacheEngine
from hotshift.policies.factory import create_policy
from hotshift.policies.static import StaticPolicy


def replay_trace(
    trace_path: str | Path,
    store: EmbeddingStore,
    capacity: int,
    policy_name: str,
    batch_size: int = 16,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    initial_policy = create_policy(policy_name)
    warm_frequency: Counter[int] = Counter()
    if policy_name == "static":
        with Path(trace_path).open("r", encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if not event.get("warmup"):
                    break
                warm_frequency.update(int(item) for item in event["history_ids"])
        warm_items = [item for item, _ in sorted(warm_frequency.items(), key=lambda pair: (-pair[1], pair[0]))[:capacity]]
        initial_policy = StaticPolicy(warm_items)
    engine = CPUCacheEngine(store, capacity, initial_policy)
    if policy_name == "static":
        engine.resident = set(initial_policy.warm_items)
    latencies: list[float] = []
    hits = occurrences = unique_hits = unique_ids = host_miss_bytes = requests = 0
    measured_batches = 0
    warmup = 0
    batch: list[dict[str, Any]] = []
    batch_warmup: bool | None = None
    measured_started: float | None = None
    warmup_engine_ms = 0.0

    def flush_batch() -> None:
        nonlocal hits, occurrences, unique_hits, unique_ids, host_miss_bytes, requests
        nonlocal measured_batches, warmup, measured_started, warmup_engine_ms
        if not batch:
            return
        if not batch_warmup and measured_started is None:
            measured_started = perf_counter()
        result = engine.pool([event["history_ids"] for event in batch])
        requests += len(batch)
        if batch_warmup:
            warmup += len(batch)
            warmup_engine_ms += result.engine_ms
        else:
            hits += result.occurrence_hits
            occurrences += result.occurrences
            unique_hits += result.unique_hits
            unique_ids += result.unique_ids
            host_miss_bytes += result.host_miss_bytes
            latencies.append(result.engine_ms)
            measured_batches += 1
        batch.clear()

    with Path(trace_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            event_warmup = bool(event.get("warmup"))
            if batch and (event_warmup != batch_warmup or len(batch) >= batch_size):
                flush_batch()
            if not batch:
                batch_warmup = event_warmup
            batch.append(event)
        flush_batch()
    elapsed = perf_counter() - measured_started if measured_started is not None else 0.0
    policy = engine.policy
    flush = getattr(policy, "flush", None)
    if flush is not None:
        flush()
    return {
        "backend": "cpu",
        "policy": policy_name,
        "offered_requests": requests,
        "successful_requests": max(requests - warmup, 0),
        "failed_requests": 0,
        "expired_requests": 0,
        "cancelled_requests": 0,
        "measured_requests": max(requests - warmup, 0),
        "warmup_requests": warmup,
        "warmup_engine_ms": warmup_engine_ms,
        "occurrences": occurrences,
        "occurrence_hits": hits,
        "occurrence_hit_rate": hits / occurrences if occurrences else None,
        "unique_ids": unique_ids,
        "unique_hits": unique_hits,
        "unique_hit_rate": unique_hits / unique_ids if unique_ids else None,
        "host_miss_bytes": host_miss_bytes,
        "wall_seconds": elapsed,
        "request_throughput": max(requests - warmup, 0) / elapsed if elapsed > 0 else None,
        "useful_lookup_ids_per_second": occurrences / elapsed if elapsed > 0 else None,
        "batch_engine_ms_p50": median(latencies) if latencies else None,
        "batch_engine_samples_ms": latencies,
        "measured_batch_count": measured_batches,
        "cache_occupancy": len(engine.resident),
        "gpu_metrics": None,
        "limitation": "CPU replay latency does not estimate CUDA or GPU service performance",
    }

