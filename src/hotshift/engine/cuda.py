from __future__ import annotations

import shutil
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import BatchResult, EngineLimits, flatten_histories
from hotshift.engine.buffers import CUDABuffers
from hotshift.policies.base import CachePolicy, PolicyDecision
from hotshift.policies.static import StaticPolicy

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [
    str(ROOT / "csrc/ops.cpp"),
    str(ROOT / "csrc/probe.cu"),
    str(ROOT / "csrc/gather.cu"),
    str(ROOT / "csrc/pool.cu"),
    str(ROOT / "csrc/maintenance.cu"),
]
_LOADED = False


def cxx_standard() -> str:
    """C++17 by default; C++20 when the installed PyTorch headers require it (2.12+ builds)."""
    import torch

    header = Path(torch.__file__).parent / "include/torch/csrc/api/include/torch/all.h"
    try:
        requires_cxx20 = "C++20 or later" in header.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        requires_cxx20 = False
    return "c++20" if requires_cxx20 else "c++17"


def build_cuda_extension(verbose: bool = False, require_device: bool = True) -> None:
    """Compile (or reuse the cached build of) the owned operators and load them once."""
    global _LOADED
    if _LOADED:
        return
    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as error:
        raise RuntimeError("install a CUDA-enabled PyTorch build before building the extension") from error
    if require_device and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but PyTorch reports no available CUDA device")
    if not Path("/usr/local/cuda/bin/nvcc").exists() and not shutil.which("nvcc"):
        raise RuntimeError("CUDA backend requested but nvcc was not found")
    standard = cxx_standard()
    load(
        name="hotshift_cuda_ops",
        sources=SOURCES,
        extra_cflags=["-O3", f"-std={standard}"],
        extra_cuda_cflags=["-O3", f"-std={standard}"],
        with_cuda=True,
        # Operators are registered by TORCH_LIBRARY; there is no CPython PyInit module.
        is_python_module=False,
        verbose=verbose,
    )
    _LOADED = True


class CUDACacheEngine:
    """Single-owner GPU cache with serialized maintenance on the current CUDA stream.

    Batch order: deduplicate on the host, probe the device map with a custom kernel, wait
    for the probe readback (an intentional, measured host boundary), pack misses into
    pinned staging, copy descriptors and misses, run the fused pooling kernel, copy the
    output back, then let the policy choose and apply one maintenance transaction before the
    next batch may probe. The CPU residency mirror is published only after that transaction
    completes on the device.

    ``pinned_staging`` and ``admit_from_miss_buffer`` select the optimized paths; turning
    them off keeps the readable baseline (pageable staging, admitted rows re-copied from the
    host table) available for profiling comparisons.
    """

    backend = "cuda"

    def __init__(
        self,
        store: EmbeddingStore,
        capacity: int,
        policy: CachePolicy,
        device: int = 0,
        limits: EngineLimits | None = None,
        pinned_staging: bool = True,
        admit_from_miss_buffer: bool = True,
        time_kernels: bool = False,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but CUDA is unavailable")
        if not 0 <= capacity <= store.rows:
            raise ValueError("cache capacity must be between zero and the number of rows")
        if capacity >= 2**31:
            raise ValueError("cache capacity must fit int32 slot indices")
        build_cuda_extension()
        self.torch = torch
        self.store = store
        self.capacity = capacity
        self.policy = policy
        self.limits = limits or EngineLimits()
        self.pinned_staging = pinned_staging
        self.admit_from_miss_buffer = admit_from_miss_buffer
        self.time_kernels = time_kernels
        self.record_functions = False
        self.device = torch.device(f"cuda:{device}")
        dimensions = store.dimensions
        with torch.cuda.device(self.device):
            self.item_to_slot = torch.full((store.rows,), -1, dtype=torch.int32, device=self.device)
            self.cache_rows = torch.empty((capacity, dimensions), dtype=torch.float32, device=self.device)
            self.buffers = CUDABuffers.allocate(
                torch,
                self.device,
                dimensions,
                self.limits.max_batch_requests,
                self.limits.max_batch_ids,
                pinned=pinned_staging,
            )
            self.probe_done = torch.cuda.Event()
            self.output_done = torch.cuda.Event()
            self.maintenance_done = torch.cuda.Event()
            self._timers = {
                name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for name in ("probe", "pool", "maintenance")
            }
        self._timed: set[str] = set()
        self._clear_mirror()
        self.generation = 0
        self.healthy = True
        self.policy.configure(dimensions * 4)
        self._preload_static()
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ helpers
    def _clear_mirror(self) -> None:
        self.slot_of: dict[int, int] = {}
        self.slot_to_item = np.full(self.capacity, -1, dtype=np.int64)
        # Stack of free slots; popping from the end hands out the lowest slot first.
        self.free_slots = list(range(self.capacity - 1, -1, -1))
        self.resident: set[int] = set()

    @contextmanager
    def _range(self, name: str):
        with self.torch.cuda.nvtx.range(name):
            if self.record_functions:
                with self.torch.profiler.record_function(name):
                    yield
            else:
                yield

    def _h2d(self, buffer, count: int):
        target = buffer.device[:count]
        if count:
            target.copy_(buffer.host[:count], non_blocking=True)
        return target

    def _timer_start(self, name: str) -> None:
        if self.time_kernels:
            self._timers[name][0].record()

    def _timer_stop(self, name: str) -> None:
        if self.time_kernels:
            self._timers[name][1].record()
            self._timed.add(name)

    def _sync(self, event) -> None:
        event.record(self.torch.cuda.current_stream(self.device))
        event.synchronize()

    # ------------------------------------------------------------- maintenance
    def _transaction(
        self,
        invalidate_items: np.ndarray,
        invalidate_slots: np.ndarray,
        publish_items: np.ndarray,
        publish_slots: np.ndarray,
        source_rows,
        source_positions: np.ndarray,
    ) -> int:
        """Invalidate old mappings, copy rows into reserved slots, then publish new mappings.

        Returns explicit host-to-device bytes. The call ends by synchronizing on
        ``maintenance_done``; host staging is never rewritten while a copy reads it.
        """
        torch = self.torch
        buffers = self.buffers
        capacity = buffers.maintenance_slots.view.size
        invalidations, publications = invalidate_items.size, publish_items.size
        if invalidations + publications > capacity:
            copied = 0
            for start in range(0, invalidations, capacity):
                end = start + capacity
                copied += self._transaction(
                    invalidate_items[start:end], invalidate_slots[start:end],
                    publish_items[:0], publish_slots[:0], source_rows, source_positions[:0],
                )
            for start in range(0, publications, capacity):
                end = start + capacity
                copied += self._transaction(
                    invalidate_items[:0], invalidate_slots[:0],
                    publish_items[start:end], publish_slots[start:end],
                    source_rows, source_positions[start:end],
                )
            return copied
        total = invalidations + publications
        buffers.maintenance_old.view[:invalidations] = invalidate_items
        buffers.maintenance_old.view[invalidations:total] = -1
        buffers.maintenance_new.view[:invalidations] = -1
        buffers.maintenance_new.view[invalidations:total] = publish_items
        buffers.maintenance_slots.view[:invalidations] = invalidate_slots
        buffers.maintenance_slots.view[invalidations:total] = publish_slots
        buffers.maintenance_sources.view[:publications] = source_positions
        old_keys = self._h2d(buffers.maintenance_old, total)
        new_keys = self._h2d(buffers.maintenance_new, total)
        slots = self._h2d(buffers.maintenance_slots, total)
        sources = self._h2d(buffers.maintenance_sources, publications)
        self._timer_start("maintenance")
        if invalidations:
            torch.ops.hotshift.update_mapping_(
                self.item_to_slot, old_keys[:invalidations], new_keys[:invalidations], slots[:invalidations]
            )
        if publications:
            torch.ops.hotshift.scatter_rows_(self.cache_rows, source_rows, sources, slots[invalidations:total])
            torch.ops.hotshift.update_mapping_(
                self.item_to_slot,
                old_keys[invalidations:total],
                new_keys[invalidations:total],
                slots[invalidations:total],
            )
        self._timer_stop("maintenance")
        self._sync(self.maintenance_done)
        return total * (8 + 8 + 4) + publications * 8

    def _admission_sources(self, admissions: np.ndarray, miss_ids: np.ndarray):
        """Return device source rows and positions for admitted items, plus extra H2D bytes."""
        if not admissions.size:
            return self.buffers.miss_rows.device, np.zeros(0, dtype=np.int64), 0
        if self.admit_from_miss_buffer:
            # Admitted rows are this batch's misses, already staged on the device.
            positions = np.searchsorted(miss_ids, admissions)
            clipped = np.minimum(positions, max(miss_ids.size - 1, 0))
            if not miss_ids.size or np.any(miss_ids[clipped] != admissions):
                raise RuntimeError("policy admitted an item that was not staged as a miss")
            return self.buffers.miss_rows.device, positions.astype(np.int64), 0
        staged = self.buffers.miss_rows
        np.take(self.store.table, admissions, axis=0, out=staged.view[: admissions.size], mode="clip")
        self._h2d(staged, admissions.size)
        return staged.device, np.arange(admissions.size, dtype=np.int64), admissions.size * self.store.dimensions * 4

    def _apply(self, decision: PolicyDecision, source_rows, positions: np.ndarray) -> int:
        admissions = np.asarray(decision.admissions, dtype=np.int64)
        evictions = np.asarray(decision.evictions, dtype=np.int64)
        if not admissions.size and not evictions.size:
            return 0
        if np.unique(admissions).size != admissions.size or np.unique(evictions).size != evictions.size:
            raise RuntimeError("policy produced duplicate admissions or evictions")
        try:
            freed = np.asarray([self.slot_of[int(item)] for item in evictions.tolist()], dtype=np.int32)
        except KeyError as error:
            raise RuntimeError(f"policy evicted non-resident item {error.args[0]}") from None
        if any(int(item) in self.slot_of for item in admissions.tolist()):
            raise RuntimeError("policy admitted an item that is already resident")
        from_free = max(0, admissions.size - freed.size)
        if from_free > len(self.free_slots):
            raise RuntimeError("policy admissions exceed the cache capacity")
        taken = self.free_slots[len(self.free_slots) - from_free:][::-1] if from_free else []
        new_slots = np.concatenate((freed[: admissions.size], np.asarray(taken, dtype=np.int32)))
        h2d = self._transaction(evictions, freed, admissions, new_slots, source_rows, positions)
        # The device transaction completed; publish the CPU mirror.
        for item, slot in zip(evictions.tolist(), freed.tolist()):
            del self.slot_of[item]
            self.slot_to_item[slot] = -1
            self.resident.discard(item)
        if from_free:
            del self.free_slots[len(self.free_slots) - from_free:]
        self.free_slots.extend(freed[admissions.size:].tolist()[::-1])
        for item, slot in zip(admissions.tolist(), new_slots.tolist()):
            self.slot_of[item] = slot
            self.slot_to_item[slot] = item
            self.resident.add(item)
        return h2d

    def _preload_static(self) -> None:
        if not isinstance(self.policy, StaticPolicy):
            return
        items = np.asarray(self.policy.warm_items[: self.capacity], dtype=np.int64)
        staged = self.buffers.miss_rows
        chunk = staged.view.shape[0]
        with self.torch.cuda.device(self.device):
            for start in range(0, items.size, chunk):
                part = items[start : start + chunk]
                np.take(self.store.table, part, axis=0, out=staged.view[: part.size], mode="clip")
                self._h2d(staged, part.size)
                self._apply(
                    PolicyDecision(tuple(part.tolist()), ()),
                    staged.device,
                    np.arange(part.size, dtype=np.int64),
                )

    # ------------------------------------------------------------------- batch
    def pool(self, histories: Sequence[Sequence[int]]) -> BatchResult:
        if not self.healthy:
            raise RuntimeError("CUDA engine is unhealthy; restart the worker")
        started = perf_counter()
        flat, offsets = flatten_histories(histories, self.store.rows, self.limits)
        try:
            return self._pool(flat, offsets, started)
        except Exception:
            # A device-side failure leaves cache state unknown; never keep serving from it.
            self.healthy = False
            raise

    def _pool(self, flat: np.ndarray, offsets: np.ndarray, started: float) -> BatchResult:
        torch = self.torch
        buffers = self.buffers
        dimensions = self.store.dimensions
        stages: dict[str, float] = {}
        self._timed.clear()
        h2d = d2h = 0
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self.maintenance_done)
            with self._range("hotshift.deduplicate"):
                unique, inverse = np.unique(flat, return_inverse=True)
            unique_count, token_count, bags = unique.size, flat.size, offsets.size - 1
            mark = perf_counter()
            stages["deduplicate_ms"] = (mark - started) * 1000

            with self._range("hotshift.probe"):
                if unique_count:
                    buffers.ids.view[:unique_count] = unique
                    ids = self._h2d(buffers.ids, unique_count)
                    self._timer_start("probe")
                    slots = torch.ops.hotshift.probe_slots(ids, self.item_to_slot)
                    self._timer_stop("probe")
                    buffers.slots.host[:unique_count].copy_(slots, non_blocking=True)
                    h2d += unique_count * 8
                    d2h += unique_count * 4
            with self._range("hotshift.probe_readback"):
                # Miss planning needs the probe results: an intentional host synchronization.
                self._sync(self.probe_done)
                slot_values = buffers.slots.view[:unique_count].copy()
            now = perf_counter()
            stages["probe_ms"], mark = (now - mark) * 1000, now

            hit_unique = slot_values >= 0
            unique_hits = int(np.count_nonzero(hit_unique))
            occurrence_hits = int(np.count_nonzero(hit_unique[inverse])) if token_count else 0
            miss_positions = np.flatnonzero(~hit_unique)
            miss_ids = unique[miss_positions]
            miss_count = miss_ids.size
            with self._range("hotshift.pack_misses"):
                if miss_count:
                    np.take(self.store.table, miss_ids, axis=0, out=buffers.miss_rows.view[:miss_count], mode="clip")
                source = buffers.source.view[:unique_count]
                source[:] = slot_values
                source[miss_positions] = -(np.arange(miss_count, dtype=np.int32) + 1)
                buffers.inverse.view[:token_count] = inverse
                buffers.offsets.view[: bags + 1] = offsets
            now = perf_counter()
            stages["pack_ms"], mark = (now - mark) * 1000, now

            with self._range("hotshift.transfer_misses"):
                miss_rows = self._h2d(buffers.miss_rows, miss_count)
                source_index = self._h2d(buffers.source, unique_count)
                unique_inverse = self._h2d(buffers.inverse, token_count)
                device_offsets = self._h2d(buffers.offsets, bags + 1)
                h2d += miss_count * dimensions * 4 + unique_count * 4 + token_count * 8 + (bags + 1) * 8
            with self._range("hotshift.pool"):
                self._timer_start("pool")
                output = torch.ops.hotshift.pool_mean(
                    self.cache_rows, miss_rows, source_index, unique_inverse, device_offsets
                )
                self._timer_stop("pool")
            with self._range("hotshift.output_copy"):
                if bags:
                    buffers.output.host[:bags].copy_(output, non_blocking=True)
                    d2h += bags * dimensions * 4
                self._sync(self.output_done)
                outputs = buffers.output.view[:bags].copy()
            now = perf_counter()
            stages["transfer_pool_output_ms"], mark = (now - mark) * 1000, now

            with self._range("hotshift.policy_update"):
                self.policy.observe(flat)
                decision = self.policy.choose(self.resident, miss_ids.tolist(), self.capacity)
            now = perf_counter()
            stages["policy_ms"], mark = (now - mark) * 1000, now
            with self._range("hotshift.maintenance"):
                admissions = np.asarray(decision.admissions, dtype=np.int64)
                source_rows, positions, extra = self._admission_sources(admissions, miss_ids)
                h2d += extra + self._apply(decision, source_rows, positions)
                if not decision.admissions and not decision.evictions:
                    self.maintenance_done.record(stream)
            now = perf_counter()
            stages["maintenance_ms"] = (now - mark) * 1000
            for name in self._timed:
                begin, end = self._timers[name]
                stages[f"{name}_gpu_ms"] = float(begin.elapsed_time(end))
        return BatchResult(
            outputs=outputs,
            occurrence_hits=occurrence_hits,
            occurrences=int(token_count),
            unique_hits=unique_hits,
            unique_ids=int(unique_count),
            host_miss_bytes=int(miss_count) * dimensions * 4,
            engine_ms=(perf_counter() - started) * 1000,
            admissions=len(decision.admissions),
            evictions=len(decision.evictions),
            h2d_bytes=h2d,
            d2h_bytes=d2h,
            stage_ms=stages,
        )

    # --------------------------------------------------------------- lifecycle
    def check_invariants(self) -> None:
        """Assert the device map, slot mirror, and cached rows agree. Synchronizes; tests only."""
        torch = self.torch
        torch.cuda.synchronize(self.device)
        mapping = self.item_to_slot.cpu().numpy()
        resident_items = np.flatnonzero(mapping >= 0)
        if set(resident_items.tolist()) != self.resident or len(self.slot_of) != len(self.resident):
            raise AssertionError("device map and CPU residency mirror disagree")
        slots = mapping[resident_items]
        if np.unique(slots).size != slots.size or (slots.size and (slots.min() < 0 or slots.max() >= self.capacity)):
            raise AssertionError("slots are duplicated or out of range")
        if not np.array_equal(self.slot_to_item[slots], resident_items):
            raise AssertionError("slot_to_item does not invert item_to_slot")
        if len(self.free_slots) + len(self.resident) != self.capacity:
            raise AssertionError("free slot accounting is inconsistent")
        if resident_items.size:
            rows = self.cache_rows[torch.as_tensor(slots, dtype=torch.int64, device=self.device)].cpu().numpy()
            if not np.array_equal(rows, self.store.table[resident_items]):
                raise AssertionError("cached rows differ from the immutable table")

    def reset(self, policy: CachePolicy | None = None) -> None:
        torch = self.torch
        torch.cuda.synchronize(self.device)
        self.item_to_slot.fill_(-1)
        self._clear_mirror()
        self.policy.reset()
        if policy is not None:
            self.policy = policy
        self.policy.configure(self.store.dimensions * 4)
        self._preload_static()
        torch.cuda.synchronize(self.device)
        self.generation += 1

    def memory(self) -> dict[str, object]:
        torch = self.torch
        dimensions = self.store.dimensions
        cache_bytes = self.capacity * dimensions * 4
        map_bytes = self.store.rows * 4
        staging = self.buffers.device_bytes()
        return {
            "host_table_bytes": self.store.rows * dimensions * 4,
            "host_staging_bytes": self.buffers.host_bytes(),
            "host_staging_pinned": self.buffers.pinned,
            "host_slot_mirror_bytes": int(self.slot_to_item.nbytes),
            "device_cache_rows_bytes": cache_bytes,
            "device_item_to_slot_bytes": map_bytes,
            "device_staging_bytes": staging,
            "device_tensor_bytes": cache_bytes + map_bytes + staging,
            "allocator_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "allocator_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
        }

    def close(self) -> None:
        torch = self.torch
        torch.cuda.synchronize(self.device)
        del self.cache_rows, self.item_to_slot, self.buffers
        torch.cuda.empty_cache()

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
            "policy_state": self.policy.state(),
            "state_generation": self.generation,
            "memory": self.memory(),
        }
