from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from hotshift.data.store import EmbeddingStore
from hotshift.engine.cpu import pool_mean_reference
from hotshift.policies.base import CachePolicy
from hotshift.policies.static import StaticPolicy


@dataclass(frozen=True)
class EngineLimits:
    max_history_ids: int = 256
    max_batch_requests: int = 64
    max_batch_ids: int = 16384


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
    # Explicit host-to-device and device-to-host copy payloads; None when no device exists.
    h2d_bytes: int | None = None
    d2h_bytes: int | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)


def flatten_histories(
    histories: Sequence[Sequence[int]], rows: int, limits: EngineLimits
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a batch and return flattened int64 IDs and offsets [B+1]."""
    if len(histories) > limits.max_batch_requests:
        raise ValueError("batch exceeds the request capacity")
    lengths = np.fromiter((len(history) for history in histories), dtype=np.int64, count=len(histories))
    if lengths.size and int(lengths.max()) > limits.max_history_ids:
        raise ValueError(f"history exceeds the {limits.max_history_ids}-ID request limit")
    offsets = np.zeros(len(histories) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if int(offsets[-1]) > limits.max_batch_ids:
        raise ValueError("batch exceeds the flattened-ID capacity")
    if offsets[-1]:
        flat = np.concatenate([np.asarray(history, dtype=np.int64) for history in histories if len(history)])
    else:
        flat = np.zeros(0, dtype=np.int64)
    if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= rows):
        raise ValueError("history contains an ID outside the embedding table")
    return flat, offsets


class CPUCacheEngine:
    """CPU execution model for cache semantics; it makes no GPU speed claims."""

    backend = "cpu"

    def __init__(
        self,
        store: EmbeddingStore,
        capacity: int,
        policy: CachePolicy,
        limits: EngineLimits | None = None,
    ) -> None:
        if not 0 <= capacity <= store.rows:
            raise ValueError("cache capacity must be between zero and the number of rows")
        self.store = store
        self.capacity = capacity
        self.policy = policy
        self.limits = limits or EngineLimits()
        self.resident: set[int] = set()
        self.policy.configure(store.dimensions * 4)
        self._preload_static()
        self.generation = 0
        self.healthy = True

    def _preload_static(self) -> None:
        if isinstance(self.policy, StaticPolicy):
            self.resident = set(self.policy.warm_items[: self.capacity])

    def pool(self, histories: Sequence[Sequence[int]]) -> BatchResult:
        started = perf_counter()
        flat, offsets = flatten_histories(histories, self.store.rows, self.limits)
        unique = np.unique(flat)
        resident = self.resident
        hit_unique = np.fromiter((int(item) in resident for item in unique.tolist()), dtype=bool, count=unique.size)
        hit_lookup = dict(zip(unique.tolist(), hit_unique.tolist()))
        occurrence_hits = sum(hit_lookup[item] for item in flat.tolist())
        unique_hits = int(np.count_nonzero(hit_unique))
        planned = perf_counter()
        output = pool_mean_reference(self.store, flat, offsets)
        pooled = perf_counter()
        self.policy.observe(flat)
        decision = self.policy.choose(resident, unique[~hit_unique].tolist(), self.capacity)
        decision.apply(resident)
        if len(resident) > self.capacity:
            self.healthy = False
            raise RuntimeError("policy decision exceeded cache capacity")
        finished = perf_counter()
        return BatchResult(
            outputs=output,
            occurrence_hits=occurrence_hits,
            occurrences=int(flat.size),
            unique_hits=unique_hits,
            unique_ids=int(unique.size),
            host_miss_bytes=int(unique.size - unique_hits) * self.store.dimensions * 4,
            engine_ms=(finished - started) * 1000,
            admissions=len(decision.admissions),
            evictions=len(decision.evictions),
            stage_ms={
                "plan_ms": (planned - started) * 1000,
                "pool_ms": (pooled - planned) * 1000,
                "policy_ms": (finished - pooled) * 1000,
            },
        )

    def reset(self, policy: CachePolicy | None = None) -> None:
        self.resident.clear()
        self.policy.reset()
        if policy is not None:
            self.policy = policy
        self.policy.configure(self.store.dimensions * 4)
        self._preload_static()
        self.generation += 1

    def memory(self) -> dict[str, object]:
        return {
            "host_table_bytes": self.store.rows * self.store.dimensions * 4,
            "device_bytes": None,
            "device_bytes_reason": "CPU backend allocates no device memory",
        }

    def close(self) -> None:
        return None

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
            "policy_state": self.policy.state(),
            "state_generation": self.generation,
            "memory": self.memory(),
        }


def limits_from_config(config) -> EngineLimits:
    return EngineLimits(
        max_history_ids=config.max_history_ids,
        max_batch_requests=config.max_batch_requests,
        max_batch_ids=config.max_batch_ids,
    )
