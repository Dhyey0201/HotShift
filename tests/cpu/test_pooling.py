import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hotshift.engine.base import EngineLimits, flatten_histories
from hotshift.engine.cpu import pool_mean_reference, pool_mean_torch_reference

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures" / "hand_pool.json").read_text())


def test_mean_pooling_keeps_duplicates_and_zeroes_empty_bags():
    table = np.asarray([[1, 3], [3, 5], [7, 9]], dtype=np.float32)
    result = pool_mean_reference(table, [0, 0, 1, 2], [0, 3, 3, 4])
    np.testing.assert_allclose(result, [[5 / 3, 11 / 3], [0, 0], [7, 9]], atol=1e-6)


def test_hand_calculated_fixture():
    table = np.asarray(FIXTURE["table"], dtype=np.float32)
    result = pool_mean_reference(table, FIXTURE["ids"], FIXTURE["offsets"])
    np.testing.assert_allclose(result, FIXTURE["expected"], atol=1e-6)


def test_empty_batch_returns_empty_matrix():
    table = np.zeros((3, 5), dtype=np.float32)
    assert pool_mean_reference(table, np.zeros(0, dtype=np.int64), [0]).shape == (0, 5)


@pytest.mark.parametrize(
    ("ids", "offsets"),
    [([-1], [0, 1]), ([3], [0, 1]), ([0], [1, 1]), ([0], [0, 2]), ([0], [0, -1]), ([0, 1], [0, 2, 1, 2])],
)
def test_pool_rejects_invalid_ids_or_offsets(ids, offsets):
    table = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        pool_mean_reference(table, ids, offsets)


def test_pool_rejects_wrong_dtype_and_layout():
    with pytest.raises(ValueError):
        pool_mean_reference(np.zeros((3, 2), dtype=np.float64), [0], [0, 1])
    with pytest.raises(ValueError):
        pool_mean_reference(np.zeros((2, 3), dtype=np.float32).T, [0], [0, 1])
    with pytest.raises(ValueError):
        pool_mean_reference(np.zeros((3, 2), dtype=np.float32), [0.5], [0, 1])


def test_flatten_enforces_request_and_id_limits():
    limits = EngineLimits(max_history_ids=4, max_batch_requests=2, max_batch_ids=6)
    flat, offsets = flatten_histories([[1, 2], [], [3]][:2], 10, limits)
    assert flat.tolist() == [1, 2] and offsets.tolist() == [0, 2, 2]
    with pytest.raises(ValueError, match="request capacity"):
        flatten_histories([[1], [1], [1]], 10, limits)
    with pytest.raises(ValueError, match="request limit"):
        flatten_histories([[1] * 5], 10, limits)
    with pytest.raises(ValueError, match="flattened-ID"):
        flatten_histories([[1] * 4, [1] * 3], 10, limits)
    with pytest.raises(ValueError, match="outside"):
        flatten_histories([[10]], 10, limits)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.lists(st.integers(0, 19), max_size=12), max_size=8),
    st.sampled_from([1, 7, 31, 33]),
)
def test_numpy_reference_matches_torch_oracle(histories, dimensions):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(dimensions)
    table = rng.standard_normal((20, dimensions)).astype(np.float32)
    flat, offsets = flatten_histories(histories, 20, EngineLimits())
    numpy_result = pool_mean_reference(table, flat, offsets)
    torch_result = pool_mean_torch_reference(table, flat, offsets).numpy()
    np.testing.assert_allclose(numpy_result, torch_result, atol=1e-5, rtol=1e-4)
    bag = torch.nn.functional.embedding_bag(
        torch.from_numpy(flat), torch.from_numpy(table), torch.from_numpy(offsets), mode="mean", include_last_offset=True
    ).numpy()
    np.testing.assert_allclose(numpy_result, bag, atol=1e-5, rtol=1e-4)


def test_extreme_magnitudes_against_fp64_accumulation():
    table = np.asarray([[1e30, -1e-30], [1e30, 1e-30], [-3e29, 7.0]], dtype=np.float32)
    ids, offsets = [0, 1, 2, 2], [0, 4]
    result = pool_mean_reference(table, ids, offsets)
    expected = table.astype(np.float64)[ids].mean(axis=0)
    # FP32 accumulation of 1e30-scale values: relative tolerance documented separately.
    np.testing.assert_allclose(result[0], expected, rtol=1e-6, atol=1e-6)
    assert np.isfinite(result).all()
