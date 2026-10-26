from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _BatchSample:
    at: float
    requests: int
    occurrences: int
    hits: int
    unique_ids: int
    unique_hits: int
    host_miss_bytes: int
    h2d_bytes: int | None
    admissions: int
    evictions: int
    engine_ms: float


class MetricsCollector:
    """Counters plus bounded recent samples. Snapshots never touch engine or device state."""

    def __init__(self, sample_capacity: int = 4096, window_seconds: float = 5.0) -> None:
        self._lock = Lock()
        self.window_seconds = window_seconds
        self._batches: deque[_BatchSample] = deque(maxlen=sample_capacity)
        self._queue_delays: deque[tuple[float, float]] = deque(maxlen=sample_capacity)
        self._counts: Counter[str] = Counter()
        self._occurrences = 0
        self._hits = 0
        self._unique_ids = 0
        self._unique_hits = 0
        self._host_miss_bytes = 0
        self._h2d_bytes = 0
        self._queue_occupancy = 0
        self._queue_capacity = 0
        self._started_at = monotonic()

    def record_batch(self, result: Any, queue_delays: list[float], request_count: int) -> None:
        now = monotonic()
        with self._lock:
            self._counts["completed_requests"] += request_count
            self._counts["batches"] += 1
            self._counts["admissions"] += result.admissions
            self._counts["evictions"] += result.evictions
            self._occurrences += result.occurrences
            self._hits += result.occurrence_hits
            self._unique_ids += result.unique_ids
            self._unique_hits += result.unique_hits
            self._host_miss_bytes += result.host_miss_bytes
            self._h2d_bytes += result.h2d_bytes or 0
            self._batches.append(_BatchSample(
                now, request_count, result.occurrences, result.occurrence_hits, result.unique_ids,
                result.unique_hits, result.host_miss_bytes, result.h2d_bytes, result.admissions,
                result.evictions, float(result.engine_ms),
            ))
            self._queue_delays.extend((now, delay) for delay in queue_delays)

    def record_error(self, cause: str) -> None:
        with self._lock:
            self._counts[f"errors_{cause}"] += 1

    def set_queue_occupancy(self, occupancy: int, capacity: int = 0) -> None:
        with self._lock:
            self._queue_occupancy = occupancy
            self._queue_capacity = capacity or self._queue_capacity

    @staticmethod
    def _percentiles(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"p50": None, "p95": None, "p99": None, "sample_count": 0}
        array = np.asarray(values, dtype=np.float64)
        p50, p95, p99 = np.percentile(array, [50, 95, 99])
        return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "sample_count": int(array.size)}

    def snapshot(self) -> dict[str, Any]:
        now = monotonic()
        with self._lock:
            elapsed = max(now - self._started_at, 1e-9)
            cutoff = now - self.window_seconds
            recent = [sample for sample in self._batches if sample.at >= cutoff]
            window_span = min(self.window_seconds, elapsed)
            occurrences = sum(sample.occurrences for sample in recent)
            unique_ids = sum(sample.unique_ids for sample in recent)
            h2d = [sample.h2d_bytes for sample in recent if sample.h2d_bytes is not None]
            window = {
                "seconds": self.window_seconds,
                "batches": len(recent),
                "requests": sum(sample.requests for sample in recent),
                "occurrences": occurrences,
                "occurrence_hit_rate": sum(sample.hits for sample in recent) / occurrences if occurrences else None,
                "unique_hit_rate": sum(sample.unique_hits for sample in recent) / unique_ids if unique_ids else None,
                "useful_lookup_ids_per_second": occurrences / window_span,
                "request_throughput_per_second": sum(sample.requests for sample in recent) / window_span,
                "host_miss_bytes_per_second": sum(sample.host_miss_bytes for sample in recent) / window_span,
                "h2d_bytes_per_second": sum(h2d) / window_span if h2d else None,
                "admissions": sum(sample.admissions for sample in recent),
                "evictions": sum(sample.evictions for sample in recent),
                "batch_engine_ms": self._percentiles([sample.engine_ms for sample in recent]),
                "queue_delay_ms": self._percentiles([delay for at, delay in self._queue_delays if at >= cutoff]),
            }
            all_latencies = [sample.engine_ms for sample in self._batches]
            return {
                "counts": dict(self._counts),
                "occurrence_hit_rate": self._hits / self._occurrences if self._occurrences else None,
                "unique_hit_rate": self._unique_hits / self._unique_ids if self._unique_ids else None,
                "occurrences": self._occurrences,
                "unique_ids": self._unique_ids,
                "host_miss_bytes": self._host_miss_bytes,
                "h2d_bytes": self._h2d_bytes,
                "useful_lookup_ids_per_second": self._occurrences / elapsed,
                "request_throughput_per_second": self._counts["completed_requests"] / elapsed,
                "host_miss_bytes_per_second": self._host_miss_bytes / elapsed,
                "queue_occupancy": self._queue_occupancy,
                "queue_capacity": self._queue_capacity,
                "batch_engine_ms": self._percentiles(all_latencies),
                "queue_delay_ms": self._percentiles([delay for _, delay in self._queue_delays]),
                "window": window,
                "uptime_seconds": elapsed,
                "units": {
                    "latency": "milliseconds",
                    "bytes": "bytes",
                    "hit_rate": "fraction",
                    "rates": "per second",
                },
                "scope": {
                    "cumulative": "since service start",
                    "window": f"batches completed in the last {self.window_seconds:g} seconds",
                    "percentiles": f"exact over at most {self._batches.maxlen} most recent batch samples",
                    "batch_engine_ms": "shared batch duration, not a per-request GPU cost",
                },
                "sample_method": "bounded_recent_samples",
            }
