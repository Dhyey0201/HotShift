"""Fixed-batch offline replay of an immutable trace against any engine.

Batch partitions depend only on the trace and the batch size: a batch ends when it is full
or when the warmup flag, the phase, or the trace window changes. Every engine therefore sees
identical batches. Policies observe only executed batches, in trace order.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from hotshift.benchmark.common import median_or_none, percentile, ratio
from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import CPUCacheEngine
from hotshift.engine.cpu import pool_mean_reference
from hotshift.policies.factory import create_policy
from hotshift.policies.static import StaticPolicy

ATOL = 1e-5
RTOL = 1e-4


@dataclass
class BatchRecord:
    start_event: int
    end_event: int
    phase_id: int
    warmup: bool
    requests: int
    scheduled_start_ns: int
    scheduled_end_ns: int
    occurrences: int
    occurrence_hits: int
    unique_ids: int
    unique_hits: int
    host_miss_bytes: int
    h2d_bytes: int | None
    d2h_bytes: int | None
    admissions: int
    evictions: int
    wall_ms: float
    device_ms: float | None
    stage_ms: dict[str, float] = field(default_factory=dict)
    policy_mode: str | None = None
    shift_score: float | None = None
    shift_transitions: int = 0


def load_trace(trace_path: str | Path) -> list[dict[str, Any]]:
    with Path(trace_path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def warm_items_from_trace(events: list[dict[str, Any]], capacity: int) -> tuple[list[int], int]:
    """Most frequent IDs of the warmup prefix only; the evaluation suffix is never consulted."""
    frequency: Counter[int] = Counter()
    warm_requests = 0
    for event in events:
        if not event.get("warmup"):
            break
        warm_requests += 1
        frequency.update(int(item) for item in event["history_ids"])
    ranked = sorted(frequency.items(), key=lambda pair: (-pair[1], pair[0]))
    return [item for item, _ in ranked[:capacity]], warm_requests


def phase_boundary(events: list[dict[str, Any]]) -> int | None:
    """First event index of the first phase change after the start."""
    first = events[0]["phase_id"] if events else None
    for event in events:
        if event["phase_id"] != first:
            return int(event["event_index"])
    return None


@dataclass
class OutputCheck:
    verified_batches: int = 0
    max_abs_error: float = 0.0
    failures: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "verified_batches": self.verified_batches,
            "max_abs_error": self.max_abs_error,
            "failures": self.failures,
            "atol": ATOL,
            "rtol": RTOL,
            "passed": self.failures == 0 and self.verified_batches > 0,
        }


def replay_engine(
    events: list[dict[str, Any]],
    engine,
    batch_size: int,
    window_requests: int,
    device_timer=None,
    verify_every: int = 0,
    output_check: OutputCheck | None = None,
) -> list[BatchRecord]:
    if batch_size <= 0 or window_requests <= 0:
        raise ValueError("batch_size and window_requests must be positive")
    records: list[BatchRecord] = []
    batch: list[dict[str, Any]] = []
    policy = engine.policy
    batch_index = 0

    def transitions() -> int:
        return sum(1 for item in getattr(policy, "mode_transitions", ()) if item.get("mode") == "shift")

    def flush() -> None:
        nonlocal batch_index
        if not batch:
            return
        histories = [event["history_ids"] for event in batch]
        before = transitions()
        if device_timer is not None:
            device_timer.start()
        started = perf_counter()
        result = engine.pool(histories)
        wall_ms = (perf_counter() - started) * 1000
        device_ms = device_timer.stop() if device_timer is not None else None
        if output_check is not None and verify_every and (batch_index % verify_every == 0):
            flat = [item for history in histories for item in history]
            offsets = np.zeros(len(histories) + 1, dtype=np.int64)
            np.cumsum([len(history) for history in histories], out=offsets[1:])
            expected = pool_mean_reference(engine.store, np.asarray(flat, dtype=np.int64), offsets)
            error = float(np.max(np.abs(result.outputs - expected))) if expected.size else 0.0
            output_check.verified_batches += 1
            output_check.max_abs_error = max(output_check.max_abs_error, error)
            if not np.allclose(result.outputs, expected, atol=ATOL, rtol=RTOL):
                output_check.failures += 1
        state = policy.state()
        first, last = batch[0], batch[-1]
        records.append(BatchRecord(
            start_event=int(first["event_index"]),
            end_event=int(last["event_index"]) + 1,
            phase_id=int(first["phase_id"]),
            warmup=bool(first.get("warmup")),
            requests=len(batch),
            scheduled_start_ns=int(first["scheduled_offset_ns"]),
            scheduled_end_ns=int(last["scheduled_offset_ns"]),
            occurrences=result.occurrences,
            occurrence_hits=result.occurrence_hits,
            unique_ids=result.unique_ids,
            unique_hits=result.unique_hits,
            host_miss_bytes=result.host_miss_bytes,
            h2d_bytes=result.h2d_bytes,
            d2h_bytes=result.d2h_bytes,
            admissions=result.admissions,
            evictions=result.evictions,
            wall_ms=wall_ms,
            device_ms=device_ms,
            stage_ms=dict(result.stage_ms),
            policy_mode=state.get("mode"),  # type: ignore[arg-type]
            shift_score=state.get("shift_score"),  # type: ignore[arg-type]
            shift_transitions=transitions() - before,
        ))
        batch_index += 1
        batch.clear()

    for event in events:
        if batch:
            head = batch[0]
            if (
                len(batch) >= batch_size
                or bool(event.get("warmup")) != bool(head.get("warmup"))
                or event["phase_id"] != head["phase_id"]
                or int(event["event_index"]) % window_requests == 0
            ):
                flush()
        batch.append(event)
    flush()
    flush_policy = getattr(policy, "flush", None)
    if flush_policy is not None:
        flush_policy()
    return records


def summarize(records: list[BatchRecord], measure_warmup: bool = False) -> dict[str, Any]:
    measured = [record for record in records if measure_warmup or not record.warmup]
    warm = [record for record in records if record.warmup and not measure_warmup]
    wall = [record.wall_ms for record in measured]
    device = [record.device_ms for record in measured if record.device_ms is not None]
    elapsed = sum(wall) / 1000.0
    requests = sum(record.requests for record in measured)
    occurrences = sum(record.occurrences for record in measured)
    hits = sum(record.occurrence_hits for record in measured)
    unique_ids = sum(record.unique_ids for record in measured)
    unique_hits = sum(record.unique_hits for record in measured)
    h2d = [record.h2d_bytes for record in measured if record.h2d_bytes is not None]
    d2h = [record.d2h_bytes for record in measured if record.d2h_bytes is not None]
    stage_names = sorted({name for record in measured for name in record.stage_ms})
    null_reasons: dict[str, str] = {}
    if not device:
        null_reasons["device_ms"] = "no device timer for this backend"
    if not h2d:
        null_reasons["h2d_bytes"] = "backend performs no explicit host-to-device copies"
    summary: dict[str, Any] = {
        "offered_requests": requests,
        "successful_requests": requests,
        "rejected_requests": 0,
        "expired_requests": 0,
        "cancelled_requests": 0,
        "failed_requests": 0,
        "measured_requests": requests,
        "warmup_requests": sum(record.requests for record in warm),
        "warmup_wall_ms": sum(record.wall_ms for record in warm),
        "occurrences": occurrences,
        "occurrence_hits": hits,
        "occurrence_hit_rate": ratio(hits, occurrences),
        "unique_ids": unique_ids,
        "unique_hits": unique_hits,
        "unique_hit_rate": ratio(unique_hits, unique_ids),
        "host_miss_bytes": sum(record.host_miss_bytes for record in measured),
        "h2d_bytes": sum(h2d) if h2d else None,
        "d2h_bytes": sum(d2h) if d2h else None,
        "admissions": sum(record.admissions for record in measured),
        "evictions": sum(record.evictions for record in measured),
        "wall_seconds": elapsed,
        "throughput_scope": "measured requests divided by summed batch wall time; excludes trace parsing and inter-batch gaps",
        "request_throughput": ratio(requests, elapsed),
        "useful_lookup_ids_per_second": ratio(occurrences, elapsed),
        "batch_wall_ms_p50": percentile(wall, 0.50),
        "batch_wall_ms_p95": percentile(wall, 0.95),
        "batch_wall_ms_p99": percentile(wall, 0.99),
        "device_ms_p50": percentile(device, 0.50),
        "device_ms_p95": percentile(device, 0.95),
        "device_ms_p99": percentile(device, 0.99),
        "batch_sample_count": len(wall),
        "stage_ms_p50": {name: median_or_none(record.stage_ms.get(name) for record in measured) for name in stage_names},
        "stage_ms_total": {name: sum(record.stage_ms.get(name, 0.0) for record in measured) for name in stage_names},
        "null_reasons": null_reasons,
    }
    policy_total = sum(
        record.stage_ms.get("policy_ms", 0.0) + record.stage_ms.get("maintenance_ms", 0.0) for record in measured
    )
    summary["policy_overhead_ms"] = policy_total
    return summary


WINDOW_COLUMNS = [
    "window_index", "window_scope", "start_event", "end_event", "scheduled_start_ns", "scheduled_end_ns",
    "phase_id", "warmup", "requests", "occurrences", "unique_ids", "occurrence_hits", "unique_hits",
    "occurrence_hit_rate", "unique_hit_rate", "host_miss_bytes", "h2d_bytes", "d2h_bytes",
    "admissions", "evictions", "shift_score", "policy_mode", "shift_transitions", "policy_ms",
    "maintenance_ms", "batch_wall_ms_sum", "batch_wall_ms_p99", "latency_sample_count",
]


def windows(records: list[BatchRecord], window_requests: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[BatchRecord]] = {}
    for record in records:
        grouped.setdefault(record.start_event // window_requests, []).append(record)
    rows = []
    for index in sorted(grouped):
        group = grouped[index]
        occurrences = sum(record.occurrences for record in group)
        unique_ids = sum(record.unique_ids for record in group)
        h2d = [record.h2d_bytes for record in group if record.h2d_bytes is not None]
        d2h = [record.d2h_bytes for record in group if record.d2h_bytes is not None]
        rows.append({
            "window_index": index,
            "window_scope": f"trace_requests:{window_requests}",
            "start_event": group[0].start_event,
            "end_event": group[-1].end_event,
            "scheduled_start_ns": group[0].scheduled_start_ns,
            "scheduled_end_ns": group[-1].scheduled_end_ns,
            "phase_id": group[0].phase_id,
            "warmup": group[0].warmup,
            "requests": sum(record.requests for record in group),
            "occurrences": occurrences,
            "unique_ids": unique_ids,
            "occurrence_hits": sum(record.occurrence_hits for record in group),
            "unique_hits": sum(record.unique_hits for record in group),
            "occurrence_hit_rate": ratio(sum(record.occurrence_hits for record in group), occurrences),
            "unique_hit_rate": ratio(sum(record.unique_hits for record in group), unique_ids),
            "host_miss_bytes": sum(record.host_miss_bytes for record in group),
            "h2d_bytes": sum(h2d) if h2d else None,
            "d2h_bytes": sum(d2h) if d2h else None,
            "admissions": sum(record.admissions for record in group),
            "evictions": sum(record.evictions for record in group),
            "shift_score": group[-1].shift_score,
            "policy_mode": group[-1].policy_mode,
            "shift_transitions": sum(record.shift_transitions for record in group),
            "policy_ms": sum(record.stage_ms.get("policy_ms", 0.0) for record in group),
            "maintenance_ms": sum(record.stage_ms.get("maintenance_ms", 0.0) for record in group),
            "batch_wall_ms_sum": sum(record.wall_ms for record in group),
            "batch_wall_ms_p99": percentile([record.wall_ms for record in group], 0.99),
            "latency_sample_count": len(group),
        })
    return rows


def recovery(
    records: list[BatchRecord],
    window_rows: list[dict[str, Any]],
    boundary_event: int | None,
    target_hit_rate: float | None,
    threshold: float = 0.9,
    consecutive: int = 3,
) -> dict[str, Any]:
    """Recovery after a known phase boundary, against a warmed post-shift reference target."""
    base: dict[str, Any] = {
        "boundary_event": boundary_event,
        "target_hit_rate": target_hit_rate,
        "threshold_fraction": threshold,
        "consecutive_windows": consecutive,
        "window_scope": window_rows[0]["window_scope"] if window_rows else None,
    }
    detection = next(
        (record for record in records if boundary_event is not None and record.start_event >= boundary_event and record.shift_transitions),
        None,
    )
    base["detection_requests"] = detection.start_event - boundary_event if detection and boundary_event is not None else None
    if boundary_event is None or target_hit_rate is None or target_hit_rate <= 0:
        return {**base, "status": "not_applicable", "reason": "no phase boundary or no positive reference hit rate"}
    after = [row for row in window_rows if row["start_event"] >= boundary_event and row["occurrence_hit_rate"] is not None]
    goal = threshold * target_hit_rate
    for position in range(len(after) - consecutive + 1):
        if all(row["occurrence_hit_rate"] >= goal for row in after[position : position + consecutive]):
            reached = after[position]
            processing_ms = sum(
                record.wall_ms for record in records if boundary_event <= record.start_event < reached["end_event"]
            )
            boundary_ns = next(record.scheduled_start_ns for record in records if record.start_event >= boundary_event)
            return {
                **base,
                "status": "recovered",
                "recovery_requests": reached["end_event"] - boundary_event,
                "recovery_processing_ms": processing_ms,
                "recovery_scheduled_seconds": (reached["scheduled_end_ns"] - boundary_ns) / 1e9,
            }
    return {**base, "status": "not_recovered"}


def replay_trace(
    trace_path: str | Path,
    store: EmbeddingStore,
    capacity: int,
    policy_name: str,
    batch_size: int = 16,
    window_requests: int = 1000,
) -> dict[str, Any]:
    """CPU replay of one policy; kept as a small convenience API for tests and notebooks."""
    events = load_trace(trace_path)
    if policy_name == "static":
        warm_items, _ = warm_items_from_trace(events, capacity)
        policy = StaticPolicy(warm_items)
    else:
        policy = create_policy(policy_name)
    engine = CPUCacheEngine(store, 0 if policy_name == "no_cache" else capacity, policy)
    records = replay_engine(events, engine, batch_size, window_requests)
    summary = summarize(records)
    summary.update({"backend": "cpu", "policy": policy_name, "cache_occupancy": len(engine.resident)})
    return summary
