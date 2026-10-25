from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from hotshift.data.store import EmbeddingStore


def pool_mean_reference(
    table: np.ndarray | EmbeddingStore,
    ids: Sequence[int] | np.ndarray,
    offsets: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Return per-bag FP32 means; duplicates count and empty bags return zero."""
    rows = table.table if isinstance(table, EmbeddingStore) else np.asarray(table)
    if rows.ndim != 2 or rows.dtype != np.float32 or not rows.flags.c_contiguous:
        raise ValueError("table must be a contiguous row-major FP32 matrix")
    id_array = np.asarray(ids)
    offset_array = np.asarray(offsets)
    if id_array.ndim != 1 or id_array.dtype.kind not in "iu":
        raise ValueError("ids must be a one-dimensional integer array")
    if offset_array.ndim != 1 or offset_array.dtype.kind not in "iu" or offset_array.size == 0:
        raise ValueError("offsets must be a nonempty one-dimensional integer array")
    if int(offset_array[0]) != 0 or int(offset_array[-1]) != id_array.size:
        raise ValueError("offsets must start at zero and end at len(ids)")
    if np.any(offset_array[1:] < offset_array[:-1]):
        raise ValueError("offsets must be nondecreasing")
    if id_array.size and (np.any(id_array < 0) or np.any(id_array >= rows.shape[0])):
        raise ValueError("ids must be within the embedding table")

    result = np.zeros((offset_array.size - 1, rows.shape[1]), dtype=np.float32)
    for bag_index, (start, end) in enumerate(pairwise(offset_array)):
        start_index, end_index = int(start), int(end)
        if end_index > start_index:
            result[bag_index] = rows[id_array[start_index:end_index]].mean(axis=0, dtype=np.float32)
    return result


def pool_mean_torch_reference(table, ids, offsets):
    """PyTorch CPU oracle used to compare implementations with different reduction orders."""
    import torch

    rows = torch.as_tensor(table, dtype=torch.float32, device="cpu").contiguous()
    id_tensor = torch.as_tensor(ids, dtype=torch.int64, device="cpu").contiguous()
    offset_tensor = torch.as_tensor(offsets, dtype=torch.int64, device="cpu").contiguous()
    if rows.ndim != 2 or not torch.isfinite(rows).all():
        raise ValueError("table must be a finite two-dimensional matrix")
    if id_tensor.ndim != 1 or offset_tensor.ndim != 1 or offset_tensor.numel() == 0:
        raise ValueError("ids and offsets must be one-dimensional and offsets nonempty")
    if offset_tensor[0].item() != 0 or offset_tensor[-1].item() != id_tensor.numel():
        raise ValueError("offsets must start at zero and end at len(ids)")
    if torch.any(offset_tensor[1:] < offset_tensor[:-1]):
        raise ValueError("offsets must be nondecreasing")
    if id_tensor.numel() and (torch.any(id_tensor < 0) or torch.any(id_tensor >= rows.shape[0])):
        raise ValueError("ids must be within the embedding table")
    output = torch.zeros((offset_tensor.numel() - 1, rows.shape[1]), dtype=torch.float32)
    for bag in range(output.shape[0]):
        start = int(offset_tensor[bag].item())
        end = int(offset_tensor[bag + 1].item())
        if end > start:
            output[bag] = rows.index_select(0, id_tensor[start:end]).mean(dim=0)
    return output

