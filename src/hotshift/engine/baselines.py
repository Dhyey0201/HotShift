"""PyTorch library baselines that share the engine batch contract.

``torch_cpu`` pools from the full host table with ``embedding_bag(mode="mean")``. It is the
optimized CPU library reference, not the NumPy loop oracle. ``torch_gpu_resident`` copies
the whole table to the device and pools there. It is a higher-memory reference, not an
equal-budget cache competitor. Neither baseline has a cache, so neither reports hit rates.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np

from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import BatchResult, EngineLimits, flatten_histories
from hotshift.policies.base import CachePolicy

BASELINE_NAMES = ("torch_cpu", "torch_gpu_resident")


class _BaselinePolicy(CachePolicy):
    def __init__(self, name: str) -> None:
        self.name = name

    def observe(self, occurrences) -> None:
        return None

    def reset(self) -> None:
        return None

    def state(self) -> dict[str, object]:
        return {"mode": "baseline", "shift_score": None}


class TorchBaselineEngine:
    def __init__(
        self,
        store: EmbeddingStore,
        name: str,
        device: int = 0,
        limits: EngineLimits | None = None,
    ) -> None:
        import torch

        if name not in BASELINE_NAMES:
            raise ValueError(f"baseline must be one of {', '.join(BASELINE_NAMES)}")
        self.torch = torch
        self.store = store
        self.name = name
        self.limits = limits or EngineLimits()
        self.policy = _BaselinePolicy(name)
        self.capacity = 0
        self.resident: set[int] = set()
        self.generation = 0
        self.healthy = True
        # Copy out of the read-only memory map once so PyTorch owns a writable host table.
        host_table = torch.from_numpy(np.array(store.table, dtype=np.float32, copy=True))
        if name == "torch_gpu_resident":
            if not torch.cuda.is_available():
                raise RuntimeError("torch_gpu_resident baseline requires a CUDA device")
            self.device = torch.device(f"cuda:{device}")
            self.table = host_table.to(self.device)
            del host_table
            torch.cuda.synchronize(self.device)
            self.backend = "cuda"
        else:
            self.device = torch.device("cpu")
            self.table = host_table
            self.backend = "cpu"

    def pool(self, histories: Sequence[Sequence[int]]) -> BatchResult:
        torch = self.torch
        started = perf_counter()
        flat, offsets = flatten_histories(histories, self.store.rows, self.limits)
        ids = torch.from_numpy(flat)
        bag_offsets = torch.from_numpy(offsets)
        h2d = d2h = None
        if self.device.type == "cuda":
            ids = ids.to(self.device)
            bag_offsets = bag_offsets.to(self.device)
            h2d = int(flat.nbytes + offsets.nbytes)
        planned = perf_counter()
        # include_last_offset uses the [B+1] offset contract; empty bags pool to zeros.
        output = torch.nn.functional.embedding_bag(
            ids, self.table, bag_offsets, mode="mean", include_last_offset=True
        )
        if self.device.type == "cuda":
            outputs = output.cpu().numpy()
            d2h = int(outputs.nbytes)
        else:
            outputs = output.numpy()
        finished = perf_counter()
        resident_hits = int(flat.size) if self.name == "torch_gpu_resident" else 0
        return BatchResult(
            outputs=outputs,
            occurrence_hits=resident_hits,
            occurrences=int(flat.size),
            unique_hits=int(np.unique(flat).size) if resident_hits else 0,
            unique_ids=int(np.unique(flat).size),
            host_miss_bytes=0,
            engine_ms=(finished - started) * 1000,
            admissions=0,
            evictions=0,
            h2d_bytes=h2d,
            d2h_bytes=d2h,
            stage_ms={"plan_ms": (planned - started) * 1000, "pool_ms": (finished - planned) * 1000},
        )

    def reset(self, policy: CachePolicy | None = None) -> None:
        self.generation += 1

    def memory(self) -> dict[str, object]:
        table_bytes = self.store.rows * self.store.dimensions * 4
        result: dict[str, object] = {"host_table_bytes": table_bytes}
        if self.device.type == "cuda":
            result.update({
                "device_table_bytes": table_bytes,
                "device_tensor_bytes": table_bytes,
                "allocator_allocated_bytes": int(self.torch.cuda.memory_allocated(self.device)),
                "allocator_reserved_bytes": int(self.torch.cuda.memory_reserved(self.device)),
            })
        return result

    def close(self) -> None:
        del self.table
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def status(self) -> dict[str, object]:
        return {"backend": self.backend, "baseline": self.name, "healthy": self.healthy}
