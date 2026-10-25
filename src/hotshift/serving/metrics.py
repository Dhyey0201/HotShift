from __future__ import annotations

from collections import Counter, deque
from threading import Lock
from time import monotonic
from typing import Any


class MetricsCollector:
    def __init__(self, sample_capacity: int = 4096) -> None:
        self._lock = Lock()
        self._latencies = deque(maxlen=sample_capacity)
        self._queue_delays = deque(maxlen=sample_capacity)
        self._counts: Counter[str] = Counter()
        self._occurrences = 0
        self._hits = 0
        self._unique_ids = 0
        self._unique_hits = 0
        self._host_miss_bytes = 0
        self._queue_occupancy = 0
        self._last_policy_ms = 0.0
        self._started_at = monotonic()

    def record_batch(self, result: Any, queue_delays: list[float], request_count: int) -> None:
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
            self._latencies.append(float(result.engine_ms))
            self._queue_delays.extend(queue_delays)

    def record_error(self, cause: str) -> None:
        with self._lock:
            self._counts[f"errors_{cause}"] += 1

    def set_queue_occupancy(self, occupancy: int) -> None:
        with self._lock:
            self._queue_occupancy = occupancy

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(int((len(ordered) - 1) * fraction), len(ordered) - 1)
        return ordered[index]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = list(self._latencies)
            queue_delays = list(self._queue_delays)
            return {
                "counts": dict(self._counts),
                "occurrence_hit_rate": self._hits / self._occurrences if self._occurrences else None,
                "unique_hit_rate": self._unique_hits / self._unique_ids if self._unique_ids else None,
                "occurrences": self._occurrences,
                "unique_ids": self._unique_ids,
                "host_miss_bytes": self._host_miss_bytes,
                "useful_lookup_ids_per_second": self._occurrences / max(monotonic() - self._started_at, 1e-9),
                "request_throughput_per_second": self._counts["completed_requests"] / max(monotonic() - self._started_at, 1e-9),
                "host_miss_bytes_per_second": self._host_miss_bytes / max(monotonic() - self._started_at, 1e-9),
                "queue_occupancy": self._queue_occupancy,
                "batch_engine_ms": {
                    "p50": self._percentile(latencies, 0.50),
                    "p95": self._percentile(latencies, 0.95),
                    "p99": self._percentile(latencies, 0.99),
                    "sample_count": len(latencies),
                },
                "queue_delay_ms": {
                    "p50": self._percentile(queue_delays, 0.50),
                    "p95": self._percentile(queue_delays, 0.95),
                    "p99": self._percentile(queue_delays, 0.99),
                    "sample_count": len(queue_delays),
                },
                "units": {"latency": "milliseconds", "bytes": "bytes", "hit_rate": "fraction"},
                "sample_method": "bounded_recent_samples",
            }

