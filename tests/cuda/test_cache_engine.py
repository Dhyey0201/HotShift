import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module", autouse=True)
def cuda_extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    from hotshift.engine.cuda import build_cuda_extension

    build_cuda_extension()


def make_store(tmp_path):
    from hotshift.data.generate import generate_embeddings
    from hotshift.data.store import EmbeddingStore

    generate_embeddings(16, 7, 91, tmp_path)
    return EmbeddingStore(tmp_path)


def test_cuda_cache_engine_hits_eviction_reset_and_pooling(tmp_path):
    import numpy as np

    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.lru import LRUPolicy

    store = make_store(tmp_path)
    engine = CUDACacheEngine(store, 2, LRUPolicy())
    first = engine.pool([[1, 1, 2], [], [4]])
    expected = np.stack(
        (store.table[1] * (2 / 3) + store.table[2] * (1 / 3), np.zeros(7), store.table[4])
    ).astype(np.float32)
    np.testing.assert_allclose(first.outputs, expected, atol=1e-5, rtol=1e-4)
    assert first.occurrence_hits == 0
    assert engine.resident == {2, 4}

    hit = engine.pool([[2, 2, 4]])
    assert hit.occurrence_hits == 3
    assert hit.unique_hits == 2

    replacement = engine.pool([[1, 1, 1, 1]])
    assert replacement.admissions == 1
    assert replacement.evictions == 1
    assert engine.resident == {1, 4}

    generation = engine.generation
    engine.reset()
    assert engine.generation == generation + 1
    assert engine.resident == set()
    after_reset = engine.pool([[1]])
    assert after_reset.occurrence_hits == 0
    assert engine.resident == {1}


def test_cuda_static_policy_preloads_and_survives_reset(tmp_path):
    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.static import StaticPolicy

    store = make_store(tmp_path)
    engine = CUDACacheEngine(store, 2, StaticPolicy([1, 3, 5]))
    assert engine.resident == {1, 3}
    assert engine.pool([[1, 3]]).occurrence_hits == 2
    engine.pool([[2]])
    assert engine.resident == {1, 3}
    engine.reset()
    assert engine.resident == {1, 3}
    assert engine.pool([[1]]).occurrence_hits == 1


def test_cuda_no_cache_never_admits_rows(tmp_path):
    import numpy as np

    from hotshift.engine.cuda import CUDACacheEngine
    from hotshift.policies.no_cache import NoCachePolicy

    store = make_store(tmp_path)
    engine = CUDACacheEngine(store, 0, NoCachePolicy())
    expected = np.stack((store.table[2], np.zeros(7))).astype(np.float32)
    for _ in range(2):
        result = engine.pool([[2], []])
        np.testing.assert_allclose(result.outputs, expected, atol=1e-5, rtol=1e-4)
        assert result.occurrence_hits == 0
        assert result.admissions == 0
        assert engine.resident == set()
