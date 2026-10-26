"""Preallocated host and device buffers for the CUDA engine.

Every per-batch transfer slices these buffers and passes active sizes explicitly, so no
request performs ``cudaMalloc`` or ``pin_memory``. Host staging is reused only after the
engine has synchronized on the event that ends the transfer reading it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HostDeviceBuffer:
    """A device tensor paired with a same-shaped host staging tensor and its NumPy view."""

    device: Any
    host: Any
    view: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.device.numel() * self.device.element_size())


def _pair(torch, shape, dtype, device, pinned: bool) -> HostDeviceBuffer:
    host = torch.empty(shape, dtype=dtype, pin_memory=pinned)
    return HostDeviceBuffer(torch.empty(shape, dtype=dtype, device=device), host, host.numpy())


@dataclass
class CUDABuffers:
    ids: HostDeviceBuffer
    slots: HostDeviceBuffer
    source: HostDeviceBuffer
    inverse: HostDeviceBuffer
    offsets: HostDeviceBuffer
    miss_rows: HostDeviceBuffer
    output: HostDeviceBuffer
    maintenance_slots: HostDeviceBuffer
    maintenance_sources: HostDeviceBuffer
    maintenance_old: HostDeviceBuffer
    maintenance_new: HostDeviceBuffer
    pinned: bool

    @classmethod
    def allocate(
        cls,
        torch,
        device,
        dimensions: int,
        max_batch_requests: int,
        max_batch_ids: int,
        pinned: bool = True,
    ) -> CUDABuffers:
        return cls(
            ids=_pair(torch, (max_batch_ids,), torch.int64, device, pinned),
            slots=_pair(torch, (max_batch_ids,), torch.int32, device, pinned),
            source=_pair(torch, (max_batch_ids,), torch.int32, device, pinned),
            inverse=_pair(torch, (max_batch_ids,), torch.int64, device, pinned),
            offsets=_pair(torch, (max_batch_requests + 1,), torch.int64, device, pinned),
            # Worst case: every flattened ID in a microbatch is a unique miss.
            miss_rows=_pair(torch, (max_batch_ids, dimensions), torch.float32, device, pinned),
            output=_pair(torch, (max_batch_requests, dimensions), torch.float32, device, pinned),
            maintenance_slots=_pair(torch, (max_batch_ids,), torch.int32, device, pinned),
            maintenance_sources=_pair(torch, (max_batch_ids,), torch.int64, device, pinned),
            maintenance_old=_pair(torch, (max_batch_ids,), torch.int64, device, pinned),
            maintenance_new=_pair(torch, (max_batch_ids,), torch.int64, device, pinned),
            pinned=pinned,
        )

    def device_bytes(self) -> int:
        return sum(buffer.nbytes for buffer in self._all())

    def host_bytes(self) -> int:
        return sum(int(buffer.view.nbytes) for buffer in self._all())

    def _all(self) -> list[HostDeviceBuffer]:
        return [
            self.ids, self.slots, self.source, self.inverse, self.offsets, self.miss_rows,
            self.output, self.maintenance_slots, self.maintenance_sources,
            self.maintenance_old, self.maintenance_new,
        ]
