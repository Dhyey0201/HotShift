from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from hotshift.data.store import EmbeddingStore
from hotshift.engine.cpu import pool_mean_reference
from hotshift.policies.base import CachePolicy
from hotshift.policies.static import StaticPolicy


@dataclass(frozen=True)
class BatchResult:
    outputs: np.ndarray
    occurrence_hits: int
    occurrences: int
    unique_hits: int
    unique_ids: int
    host_miss_bytes: int
    engine_ms: float
    admissions: int
    evictions: int


class CPUCacheEngine:
    """CPU execution model for cache semantics; it makes no GPU speed claims."""

    backend = "cpu"

    def __init__(self, store: EmbeddingStore, capacity: int, policy: CachePolicy) -> None:
        if not 0 <= capacity <= store.rows:
            raise ValueError("cache capacity must be between zero and the number of rows")
        self.store = store
        self.capacity = capacity
        self.policy = policy
        self.resident: set[int] = set()
        if isinstance(policy, StaticPolicy):
            self.resident = set(policy.warm_items[:capacity])
        self.generation = 0
        self.healthy = True

    def pool(self, histories: Sequence[Sequence[int]]) -> BatchResult:
        started = perf_counter()
        flat: list[int] = []
        offsets = [0]
        for history in histories:
            values = [int(item) for item in history]
            if len(values) > 256:
                raise ValueError("history exceeds the 256-ID request limit")
            if any(item < 0 or item >= self.store.rows for item in values):
                raise ValueError("history contains an ID outside the embedding table")
            flat.extend(values)
            offsets.append(len(flat))
        if len(histories) > 64 or len(flat) > 16384:
            raise ValueError("batch exceeds request or flattened-ID capacity")
        unique_order = list(dict.fromkeys(flat))
        unique = set(unique_order)
        occurrence_hits = sum(item in self.resident for item in flat)
        unique_hits = len(unique & self.resident)
        output = pool_mean_reference(self.store, flat, offsets)
        self.policy.observe(flat)
        decision = self.policy.choose(
            self.resident,
            (item for item in unique_order if item not in self.resident),
            self.capacity,
        )
        self.resident = set(decision.resident)
        elapsed_ms = (perf_counter() - started) * 1000
        return BatchResult(
            outputs=output,
            occurrence_hits=occurrence_hits,
            occurrences=len(flat),
            unique_hits=unique_hits,
            unique_ids=len(unique),
            host_miss_bytes=(len(unique) - unique_hits) * self.store.dimensions * 4,
            engine_ms=elapsed_ms,
            admissions=len(decision.admissions),
            evictions=len(decision.evictions),
        )

    def reset(self, policy: CachePolicy | None = None) -> None:
        self.resident.clear()
        self.policy.reset()
        if policy is not None:
            self.policy = policy
        if isinstance(self.policy, StaticPolicy):
            self.resident = set(self.policy.warm_items[: self.capacity])
        self.generation += 1

    def status(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "device": "CPU",
            "healthy": self.healthy,
            "dataset_version": self.store.manifest["dataset_version"],
            "rows": self.store.rows,
            "dimensions": self.store.dimensions,
            "cache_capacity": self.capacity,
            "cache_occupancy": len(self.resident),
            "policy": self.policy.name,
            "state_generation": self.generation,
        }

