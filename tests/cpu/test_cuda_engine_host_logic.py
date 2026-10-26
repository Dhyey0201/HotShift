"""CUDA engine host bookkeeping under a CPU operator shim (see cuda_shim.py).

These tests run the real CUDACacheEngine code paths for planning and maintenance with
reference PyTorch operators. They do not exercise CUDA kernels or synchronization; the
tests under tests/cuda remain the GPU gate.
"""

import sys

import numpy as np
import pytest

pytest.importorskip("torch")

from hotshift.data.generate import generate_embeddings
from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import EngineLimits
from hotshift.engine.cpu import pool_mean_reference
from hotshift.policies.base import CachePolicy, PolicyDecision
from hotshift.policies.factory import POLICY_NAMES, create_policy
from hotshift.policies.static import StaticPolicy


@pytest.fixture
def shim(monkeypatch):
    from cuda_shim import make_torch_shim

    import hotshift.engine.cuda as cuda_module

    monkeypatch.setitem(sys.modules, "torch", make_torch_shim())
    monkeypatch.setattr(cuda_module, "_LOADED", True)
    return cuda_module.CUDACacheEngine


@pytest.fixture
def store(tmp_path):
    generate_embeddings(97, 13, 5, tmp_path)
    return EmbeddingStore(tmp_path)


def _expected(store, histories):
    flat = np.asarray([item for history in histories for item in history], dtype=np.int64)
    offsets = np.zeros(len(histories) + 1, dtype=np.int64)
    np.cumsum([len(history) for history in histories], out=offsets[1:])
    return pool_mean_reference(store, flat, offsets)


@pytest.mark.parametrize("name", POLICY_NAMES)
@pytest.mark.parametrize("variant", [{}, {"pinned_staging": False}, {"admit_from_miss_buffer": False}])
def test_randomized_lifecycle_preserves_invariants_and_results(shim, store, name, variant):
    rng = np.random.default_rng(len(name))
    policy = StaticPolicy(range(0, 97, 3)) if name == "static" else create_policy(name)
    capacity = 0 if name == "no_cache" else 9
    engine = shim(store, capacity, policy, limits=EngineLimits(256, 8, 64), **variant)
    for step in range(60):
        histories = [rng.integers(0, 97, size=rng.integers(0, 9)).tolist() for _ in range(rng.integers(0, 8))]
        result = engine.pool(histories)
        np.testing.assert_allclose(result.outputs, _expected(store, histories), atol=1e-5, rtol=1e-4)
        assert result.h2d_bytes > 0 or not histories or all(not history for history in histories)
        engine.check_invariants()
        if step == 30:
            engine.reset()
            engine.check_invariants()
    assert len(engine.resident) <= capacity


def test_all_hit_all_miss_and_unique_larger_than_capacity(shim, store):
    engine = shim(store, 4, create_policy("lru"))
    miss = engine.pool([[1, 2, 2], [3]])
    assert miss.occurrence_hits == 0 and miss.unique_ids == 3 and miss.host_miss_bytes == 3 * 13 * 4
    hit = engine.pool([[1, 2], [3, 3]])
    assert hit.occurrence_hits == 4 and hit.unique_hits == 3 and hit.host_miss_bytes == 0
    wide = engine.pool([list(range(10, 30))])
    np.testing.assert_allclose(wide.outputs, _expected(store, [list(range(10, 30))]), atol=1e-5)
    assert len(engine.resident) == 4
    engine.check_invariants()


def test_cache_size_one_and_empty_batches(shim, store):
    engine = shim(store, 1, create_policy("decayed_lfu"))
    assert engine.pool([]).outputs.shape == (0, 13)
    np.testing.assert_array_equal(engine.pool([[], []]).outputs, np.zeros((2, 13), dtype=np.float32))
    for item in [5, 5, 6, 5]:
        engine.pool([[item]])
        engine.check_invariants()
    assert engine.resident == {5}


def test_static_preload_larger_than_staging_is_chunked(shim, store):
    engine = shim(store, 40, StaticPolicy(range(50)), limits=EngineLimits(16, 4, 8))
    assert engine.resident == set(range(40))
    engine.check_invariants()
    engine.reset()
    assert engine.resident == set(range(40))


def test_maximum_limits_are_accepted_and_excess_rejected(shim, store):
    engine = shim(store, 8, create_policy("lru"), limits=EngineLimits(256, 64, 16384))
    histories = [[index % 97] * 256 for index in range(64)]
    result = engine.pool(histories)
    assert result.occurrences == 16384
    with pytest.raises(ValueError):
        engine.pool([[0] * 257])
    with pytest.raises(ValueError):
        engine.pool([[98]])
    assert engine.healthy, "validation errors must not mark the engine unhealthy"


class EvictOnly(CachePolicy):
    name = "evict_only"

    def __init__(self):
        self.calls = 0

    def observe(self, occurrences):
        return None

    def choose(self, resident, incoming, capacity):
        self.calls += 1
        if self.calls == 1:
            return PolicyDecision(tuple(sorted(incoming))[:capacity], ())
        return PolicyDecision((), tuple(sorted(resident))[:2])

    def reset(self):
        self.calls = 0


def test_evictions_without_admissions_free_slots(shim, store):
    engine = shim(store, 4, EvictOnly())
    engine.pool([[1, 2, 3, 4]])
    assert engine.resident == {1, 2, 3, 4}
    engine.pool([[50]])
    assert engine.resident == {3, 4}
    assert len(engine.free_slots) == 2
    engine.check_invariants()


class Broken(CachePolicy):
    name = "broken"

    def observe(self, occurrences):
        return None

    def choose(self, resident, incoming, capacity):
        return PolicyDecision((), (999,))

    def reset(self):
        return None


def test_inconsistent_policy_decision_marks_engine_unhealthy(shim, store):
    engine = shim(store, 4, Broken())
    with pytest.raises(RuntimeError, match="non-resident"):
        engine.pool([[1]])
    assert engine.healthy is False
    with pytest.raises(RuntimeError, match="unhealthy"):
        engine.pool([[1]])


def test_memory_report_accounts_for_every_tensor(shim, store):
    engine = shim(store, 10, create_policy("lru"), limits=EngineLimits(256, 4, 32))
    memory = engine.memory()
    assert memory["device_cache_rows_bytes"] == 10 * 13 * 4
    assert memory["device_item_to_slot_bytes"] == 97 * 4
    assert memory["device_tensor_bytes"] == memory["device_cache_rows_bytes"] + memory["device_item_to_slot_bytes"] + memory["device_staging_bytes"]
    assert memory["device_staging_bytes"] > 32 * 13 * 4
