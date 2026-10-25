import numpy as np
import pytest

from hotshift.engine.cpu import pool_mean_reference


def test_mean_pooling_keeps_duplicates_and_zeroes_empty_bags():
    table = np.asarray([[1, 3], [3, 5], [7, 9]], dtype=np.float32)
    result = pool_mean_reference(table, [0, 0, 1, 2], [0, 3, 3, 4])
    np.testing.assert_allclose(result, [[5 / 3, 11 / 3], [0, 0], [7, 9]], atol=1e-6)


@pytest.mark.parametrize(
    ("ids", "offsets"),
    [([-1], [0, 1]), ([3], [0, 1]), ([0], [1, 1]), ([0], [0, 2]), ([0], [0, -1])],
)
def test_pool_rejects_invalid_ids_or_offsets(ids, offsets):
    table = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        pool_mean_reference(table, ids, offsets)

