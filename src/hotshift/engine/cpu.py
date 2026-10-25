from __future__ import annotations

from collections.abc import Sequence

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
    for bag_index, (start, end) in enumerate(zip(offset_array[:-1], offset_array[1:])):
        start_index, end_index = int(start), int(end)
        if end_index > start_index:
            result[bag_index] = rows[id_array[start_index:end_index]].mean(axis=0, dtype=np.float32)
    return result

