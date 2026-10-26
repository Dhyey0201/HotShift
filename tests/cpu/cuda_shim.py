"""A CPU stand-in for torch.cuda and the owned hotshift operators.

It lets CPU CI execute the *host-side* logic of ``CUDACacheEngine``: slot allocation,
descriptor construction, maintenance transactions, and the CPU mirror. The operator
replacements are reference PyTorch implementations, so this validates bookkeeping only;
it says nothing about the CUDA kernels, streams, events, or device memory safety.
"""

from __future__ import annotations

import types
from contextlib import nullcontext
from types import SimpleNamespace

import torch as real_torch


class _Event:
    def __init__(self, enable_timing: bool = False) -> None:
        self.recorded = False

    def record(self, stream=None) -> None:
        self.recorded = True

    def synchronize(self) -> None:
        return None

    def wait(self, stream=None) -> None:
        return None

    def elapsed_time(self, other) -> float:
        return 0.0


class _Stream:
    def wait_event(self, event) -> None:
        return None


def _probe(ids, item_to_slot):
    valid = (ids >= 0) & (ids < item_to_slot.numel())
    slots = item_to_slot[ids.clamp(0, max(item_to_slot.numel() - 1, 0))] if item_to_slot.numel() else ids.to(real_torch.int32)
    return real_torch.where(valid, slots, real_torch.full_like(slots, -1)).to(real_torch.int32)


def _unique_rows(cache_rows, miss_rows, source_index):
    rows = real_torch.empty((source_index.numel(), cache_rows.shape[1]), dtype=real_torch.float32)
    for index, source in enumerate(source_index.tolist()):
        rows[index] = cache_rows[source] if source >= 0 else miss_rows[-source - 1]
    return rows


def _pool(cache_rows, miss_rows, source_index, unique_inverse, offsets):
    rows = _unique_rows(cache_rows, miss_rows, source_index)
    if offsets.numel() == 1:
        return real_torch.zeros((0, cache_rows.shape[1]), dtype=real_torch.float32)
    if unique_inverse.numel() == 0:
        return real_torch.zeros((offsets.numel() - 1, cache_rows.shape[1]), dtype=real_torch.float32)
    return real_torch.nn.functional.embedding_bag(unique_inverse, rows, offsets, mode="mean", include_last_offset=True)


def _update_mapping(item_to_slot, old_keys, new_keys, slots):
    assert old_keys.numel() == new_keys.numel() == slots.numel()
    for old, new, slot in zip(old_keys.tolist(), new_keys.tolist(), slots.tolist()):
        if 0 <= old < item_to_slot.numel():
            item_to_slot[old] = -1
        if 0 <= new < item_to_slot.numel():
            item_to_slot[new] = slot


def _scatter_rows(cache_rows, source_rows, source_positions, slots):
    assert source_positions.numel() == slots.numel()
    for position, slot in zip(source_positions.tolist(), slots.tolist()):
        assert 0 <= slot < cache_rows.shape[0] and 0 <= position < source_rows.shape[0]
        cache_rows[slot] = source_rows[position]


def _strip(kwargs):
    kwargs.pop("device", None)
    kwargs.pop("pin_memory", None)
    return kwargs


def make_torch_shim() -> types.ModuleType:
    shim = types.ModuleType("torch")
    shim.__dict__.update({name: getattr(real_torch, name) for name in ("int32", "int64", "float32", "nn", "profiler", "from_numpy")})
    shim.device = lambda spec: real_torch.device("cpu")
    shim.empty = lambda *args, **kwargs: real_torch.empty(*args, **_strip(kwargs))
    shim.full = lambda *args, **kwargs: real_torch.full(*args, **_strip(kwargs))
    shim.tensor = lambda *args, **kwargs: real_torch.tensor(*args, **_strip(kwargs))
    shim.as_tensor = lambda *args, **kwargs: real_torch.as_tensor(*args, **_strip(kwargs))
    shim.cuda = SimpleNamespace(
        is_available=lambda: True,
        device=lambda device: nullcontext(),
        current_stream=lambda device=None: _Stream(),
        synchronize=lambda device=None: None,
        Event=_Event,
        nvtx=SimpleNamespace(range=lambda name: nullcontext()),
        memory_allocated=lambda device=None: 0,
        memory_reserved=lambda device=None: 0,
        empty_cache=lambda: None,
        get_device_name=lambda device=None: "CPU shim (not a GPU)",
    )
    shim.ops = SimpleNamespace(hotshift=SimpleNamespace(
        probe_slots=_probe, pool_mean=_pool, update_mapping_=_update_mapping, scatter_rows_=_scatter_rows,
    ))
    return shim
