from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np

from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import BatchResult
from hotshift.policies.base import CachePolicy

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [
    str(ROOT / "csrc/ops.cpp"),
    str(ROOT / "csrc/probe.cu"),
    str(ROOT / "csrc/gather.cu"),
    str(ROOT / "csrc/pool.cu"),
    str(ROOT / "csrc/maintenance.cu"),
]


def build_cuda_extension(verbose: bool = False):
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as error:
        raise RuntimeError("install a CUDA-enabled PyTorch build before building the extension") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but PyTorch reports no available CUDA device")
    if not Path("/usr/local/cuda/bin/nvcc").exists() and not shutil.which("nvcc"):
        raise RuntimeError("CUDA backend requested but nvcc was not found")
    return load(
        name="hotshift_cuda_ops",
        sources=SOURCES,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        with_cuda=True,
        verbose=verbose,
    )


class CUDACacheEngine:
    """Single-owner GPU cache; batches and serialized maintenance share one stream."""

    backend = "cuda"

    def __init__(self, store: EmbeddingStore, capacity: int, policy: CachePolicy, device: int = 0) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but CUDA is unavailable")
        if not 0 <= capacity <= store.rows:
            raise ValueError("cache capacity must be between zero and the number of rows")
        self.torch = torch
        self.store = store
        self.capacity = capacity
        self.policy = policy
        self.device = torch.device(f"cuda:{device}")
        self.item_to_slot = torch.full((store.rows,), -1, dtype=torch.int32, device=self.device)
        self.cache_rows = torch.empty((capacity, store.dimensions), dtype=torch.float32, device=self.device)
        self.ids_buffer = torch.empty(16384, dtype=torch.int64, device=self.device)
        self.source_buffer = torch.empty(16384, dtype=torch.int32, device=self.device)
        self.inverse_buffer = torch.empty(16384, dtype=torch.int64, device=self.device)
        self.offsets_buffer = torch.empty(65, dtype=torch.int64, device=self.device)
        self.miss_rows_buffer = torch.empty((16384, store.dimensions), dtype=torch.float32, device=self.device)
        self.slot_to_item = [-1] * capacity
        self.resident: set[int] = set()
        self.generation = 0
        self.healthy = True
        torch.cuda.synchronize(self.device)

    def pool(self, histories: Sequence[Sequence[int]]) -> BatchResult:
        torch = self.torch
        started = perf_counter()
        flat = [int(item) for history in histories for item in history]
        if len(histories) > 64 or sum(map(len, histories)) > 16384:
            raise ValueError("batch exceeds request or flattened-ID capacity")
        if any(len(history) > 256 for history in histories):
            raise ValueError("history exceeds the 256-ID request limit")
        if any(item < 0 or item >= self.store.rows for item in flat):
            raise ValueError("history contains an ID outside the embedding table")
        ids_np = np.asarray(flat, dtype=np.int64)
        unique_np, inverse_np = np.unique(ids_np, return_inverse=True)
        offsets_np = np.zeros(len(histories) + 1, dtype=np.int64)
        np.cumsum([len(history) for history in histories], out=offsets_np[1:])
        with torch.cuda.device(self.device):
            unique_count = len(unique_np)
            ids = self.ids_buffer[:unique_count]
            ids.copy_(torch.as_tensor(unique_np, dtype=torch.int64, device=self.device))
            slots = torch.ops.hotshift.probe_slots(ids, self.item_to_slot)
            slot_values = slots.cpu().numpy()
            occurrence_hits = int(np.count_nonzero(slot_values[inverse_np] >= 0))
            unique_hits = int(np.count_nonzero(slot_values >= 0))
            miss_positions = np.flatnonzero(slot_values < 0)
            miss_ids = unique_np[miss_positions]
            miss_np = np.ascontiguousarray(self.store.table[miss_ids], dtype=np.float32)
            if len(miss_np):
                miss_rows = self.miss_rows_buffer[: len(miss_np)]
                miss_rows.copy_(torch.as_tensor(miss_np, dtype=torch.float32, device=self.device))
            else:
                miss_rows = self.miss_rows_buffer[:0]
            source_np = slot_values.astype(np.int32, copy=True)
            source_np[miss_positions] = -(np.arange(len(miss_positions), dtype=np.int32) + 1)
            source_index = self.source_buffer[:unique_count]
            source_index.copy_(torch.as_tensor(source_np, dtype=torch.int32, device=self.device))
            inverse = self.inverse_buffer[: len(inverse_np)]
            inverse.copy_(torch.as_tensor(inverse_np, dtype=torch.int64, device=self.device))
            offsets = self.offsets_buffer[: len(offsets_np)]
            offsets.copy_(torch.as_tensor(offsets_np, dtype=torch.int64, device=self.device))
            output = torch.ops.hotshift.pool_mean(
                self.cache_rows, miss_rows, source_index, inverse, offsets
            )
            output_np = output.cpu().numpy().copy()

            self.policy.observe(flat)
            decision = self.policy.choose(
                self.resident,
                (int(item) for item in unique_np if int(item) not in self.resident),
                self.capacity,
            )
            admissions = list(decision.admissions)
            evictions = list(decision.evictions)
            free_slots = [index for index, item in enumerate(self.slot_to_item) if item < 0]
            slots_to_update: list[int] = []
            old_keys: list[int] = []
            for item in evictions:
                slot = self.slot_to_item.index(item)
                slots_to_update.append(slot)
                old_keys.append(item)
            slots_to_update.extend(free_slots[: max(0, len(admissions) - len(evictions))])
            old_keys.extend([-1] * max(0, len(admissions) - len(evictions)))
            admissions = admissions[: len(slots_to_update)]
            if admissions:
                slot_tensor = torch.as_tensor(slots_to_update, dtype=torch.int64, device=self.device)
                new_rows_np = np.ascontiguousarray(self.store.table[np.asarray(admissions, dtype=np.int64)])
                self.cache_rows[slot_tensor] = torch.as_tensor(new_rows_np, device=self.device)
                self.policy_mapping_update(old_keys, admissions, slots_to_update)
                for slot, old_item, new_item in zip(slots_to_update, old_keys, admissions):
                    if old_item >= 0:
                        self.slot_to_item[slot] = -1
                    self.slot_to_item[slot] = new_item
            torch.cuda.synchronize(self.device)
            self.resident = {item for item in self.slot_to_item if item >= 0}
        return BatchResult(
            outputs=output_np,
            occurrence_hits=occurrence_hits,
            occurrences=len(flat),
            unique_hits=unique_hits,
            unique_ids=len(unique_np),
            host_miss_bytes=len(miss_np) * self.store.dimensions * 4,
            engine_ms=(perf_counter() - started) * 1000,
            admissions=len(admissions),
            evictions=len(evictions[: len(admissions)]),
        )

    def policy_mapping_update(self, old_keys: list[int], new_keys: list[int], slots: list[int]) -> None:
        torch = self.torch
        old_tensor = torch.tensor(old_keys, dtype=torch.int64, device=self.device)
        new_tensor = torch.tensor(new_keys, dtype=torch.int64, device=self.device)
        slot_tensor = torch.tensor(slots, dtype=torch.int32, device=self.device)
        torch.ops.hotshift.update_mapping_(self.item_to_slot, old_tensor, new_tensor, slot_tensor)

    def reset(self, policy: CachePolicy | None = None) -> None:
        self.torch.cuda.synchronize(self.device)
        self.item_to_slot.fill_(-1)
        self.slot_to_item = [-1] * self.capacity
        self.resident.clear()
        self.policy.reset()
        if policy is not None:
            self.policy = policy
        self.generation += 1

    def status(self) -> dict[str, object]:
        torch = self.torch
        return {
            "backend": self.backend,
            "device": torch.cuda.get_device_name(self.device),
            "healthy": self.healthy,
            "dataset_version": self.store.manifest["dataset_version"],
            "rows": self.store.rows,
            "dimensions": self.store.dimensions,
            "cache_capacity": self.capacity,
            "cache_occupancy": len(self.resident),
            "policy": self.policy.name,
            "state_generation": self.generation,
        }

