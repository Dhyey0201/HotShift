import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module", autouse=True)
def cuda_extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    from hotshift.engine.cuda import build_cuda_extension

    build_cuda_extension()


def make_store(tmp_path, rows=16, dimensions=7):
    from hotshift.data.generate import generate_embeddings
    from hotshift.data.store import EmbeddingStore

    generate_embeddings(rows, dimensions, 91, tmp_path)
    return EmbeddingStore(tmp_path)


def expected(store, histories):
    from hotshift.engine.cpu import pool_mean_reference

    flat = np.asarray([item for history in histories for item in history], dtype=np.int64)
    offsets = np.zeros(len(histories) + 1, dtype=np.int64)
    np.cumsum([len(history) for history in histories], out=offsets[1:])
    return pool_mean_reference(store, flat, offsets)


def test_cuda_cache_engine_hits_eviction_reset_and_pooling(tmp_path):
    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.lru import LRUPolicy

    store = make_store(tmp_path)
    engine = CUDACacheEngine(store, 2, LRUPolicy())
    first = engine.pool([[1, 1, 2], [], [4]])
    np.testing.assert_allclose(first.outputs, expected(store, [[1, 1, 2], [], [4]]), atol=1e-5, rtol=1e-4)
    assert first.occurrence_hits == 0
    assert engine.resident == {2, 4}

    hit = engine.pool([[2, 2, 4]])
    assert hit.occurrence_hits == 3
    assert hit.unique_hits == 2

    replacement = engine.pool([[1, 1, 1, 1]])
    assert replacement.admissions == 1
    assert replacement.evictions == 1
    assert engine.resident == {1, 4}
    engine.check_invariants()

    generation = engine.generation
    engine.reset()
    assert engine.generation == generation + 1
    assert engine.resident == set()
    after_reset = engine.pool([[1]])
    assert after_reset.occurrence_hits == 0
    assert engine.resident == {1}
    engine.check_invariants()


def test_cuda_static_policy_preloads_and_survives_reset(tmp_path):
    from hotshift.engine.base import EngineLimits
    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.static import StaticPolicy

    store = make_store(tmp_path, rows=64)
    # A preload larger than the staging buffer is copied in chunks.
    engine = CUDACacheEngine(store, 40, StaticPolicy(range(50)), limits=EngineLimits(16, 4, 8))
    assert engine.resident == set(range(40))
    engine.check_invariants()
    assert engine.pool([[1, 3]]).occurrence_hits == 2
    engine.pool([[45]])
    assert engine.resident == set(range(40))
    engine.reset()
    assert engine.resident == set(range(40))
    engine.check_invariants()


def test_cuda_no_cache_never_admits_rows(tmp_path):
    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.no_cache import NoCachePolicy

    store = make_store(tmp_path)
    engine = CUDACacheEngine(store, 0, NoCachePolicy())
    for _ in range(2):
        result = engine.pool([[2], []])
        np.testing.assert_allclose(result.outputs, expected(store, [[2], []]), atol=1e-5, rtol=1e-4)
        assert result.occurrence_hits == 0
        assert result.admissions == 0
        assert engine.resident == set()


@pytest.mark.parametrize("policy_name", ["lru", "decayed_lfu", "adaptive", "static"])
@pytest.mark.parametrize("variant", [{}, {"pinned_staging": False}, {"admit_from_miss_buffer": False}])
def test_randomized_replacement_preserves_invariants(tmp_path, policy_name, variant):
    from hotshift.engine.base import EngineLimits
    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.factory import create_policy
    from hotshift.policies.static import StaticPolicy

    store = make_store(tmp_path, rows=200, dimensions=33)
    rng = np.random.default_rng(len(policy_name))
    policy = StaticPolicy(range(0, 200, 7)) if policy_name == "static" else create_policy(policy_name)
    engine = CUDACacheEngine(store, 12, policy, limits=EngineLimits(256, 8, 128), time_kernels=True, **variant)
    for step in range(80):
        histories = [rng.integers(0, 200, size=rng.integers(0, 16)).tolist() for _ in range(rng.integers(0, 8))]
        result = engine.pool(histories)
        np.testing.assert_allclose(result.outputs, expected(store, histories), atol=1e-5, rtol=1e-4)
        if step % 10 == 0:
            engine.check_invariants()
    # Every slot was reused at least once by the replacing policies.
    engine.check_invariants()
    assert len(engine.resident) <= 12


def test_unique_ids_exceed_capacity_and_maximum_batch(tmp_path):
    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.lru import LRUPolicy

    store = make_store(tmp_path, rows=20000, dimensions=32)
    engine = CUDACacheEngine(store, 1, LRUPolicy())
    histories = [list(range(index * 256, (index + 1) * 256)) for index in range(64)]
    result = engine.pool(histories)
    assert result.occurrences == 16384 and result.unique_ids == 16384
    np.testing.assert_allclose(result.outputs, expected(store, histories), atol=1e-5, rtol=1e-4)
    assert len(engine.resident) == 1
    engine.check_invariants()


def test_pending_device_work_delays_host_planning_not_correctness(tmp_path):
    """Stall the stream before a batch: host readbacks must wait on events, never read early."""
    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.lru import LRUPolicy

    store = make_store(tmp_path, rows=500, dimensions=64)
    engine = CUDACacheEngine(store, 50, LRUPolicy())
    rng = np.random.default_rng(1)
    for _ in range(20):
        torch.cuda._sleep(20_000_000)
        histories = [rng.integers(0, 500, size=32).tolist() for _ in range(8)]
        result = engine.pool(histories)
        np.testing.assert_allclose(result.outputs, expected(store, histories), atol=1e-5, rtol=1e-4)
    engine.check_invariants()


def test_validation_errors_keep_engine_healthy_and_faults_do_not(tmp_path, monkeypatch):
    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.lru import LRUPolicy

    store = make_store(tmp_path)
    engine = CUDACacheEngine(store, 4, LRUPolicy())
    with pytest.raises(ValueError):
        engine.pool([[16]])
    with pytest.raises(ValueError):
        engine.pool([[0] * 257])
    assert engine.healthy

    def fault(*args, **kwargs):
        raise RuntimeError("simulated CUDA error")

    monkeypatch.setattr(engine, "_transaction", fault)
    with pytest.raises(RuntimeError):
        engine.pool([[1, 2]])
    assert engine.healthy is False
    with pytest.raises(RuntimeError, match="unhealthy"):
        engine.pool([[1]])


def test_repeated_runs_do_not_grow_allocated_memory(tmp_path):
    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.factory import create_policy

    store = make_store(tmp_path, rows=1000, dimensions=32)
    engine = CUDACacheEngine(store, 100, create_policy("adaptive"))
    rng = np.random.default_rng(2)
    engine.pool([rng.integers(0, 1000, size=64).tolist() for _ in range(16)])
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    for _ in range(200):
        engine.pool([rng.integers(0, 1000, size=64).tolist() for _ in range(16)])
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= baseline + 1024 * 1024
    memory = engine.memory()
    assert memory["device_tensor_bytes"] <= memory["allocator_allocated_bytes"] + 1024 * 1024


@pytest.mark.parametrize("name", ["torch_cpu", "torch_gpu_resident"])
def test_baselines_match_reference(tmp_path, name):
    from hotshift.engine.baselines import TorchBaselineEngine

    store = make_store(tmp_path, rows=300, dimensions=129)
    engine = TorchBaselineEngine(store, name)
    histories = [[1, 1, 299], [], [7] * 256]
    result = engine.pool(histories)
    np.testing.assert_allclose(result.outputs, expected(store, histories), atol=1e-5, rtol=1e-4)
