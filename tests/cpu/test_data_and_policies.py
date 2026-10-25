import numpy as np

from hotshift.data.generate import generate_embeddings
from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import CPUCacheEngine
from hotshift.policies.adaptive import AdaptivePolicy
from hotshift.policies.lru import LRUPolicy
from hotshift.policies.static import StaticPolicy


def test_seeded_data_regenerates_identically(tmp_path):
    first = generate_embeddings(128, 17, 71, tmp_path / "first", chunk_rows=31)
    second = generate_embeddings(128, 17, 71, tmp_path / "second", chunk_rows=64)
    assert first["embedding_sha256"] == second["embedding_sha256"]
    store = EmbeddingStore(tmp_path / "first")
    assert not store.table.flags.writeable
    assert np.allclose(np.linalg.norm(store.table, axis=1), 1.0, atol=1e-6)


def test_lru_eviction_uses_request_occurrence_order(tmp_path):
    generate_embeddings(8, 3, 5, tmp_path)
    engine = CPUCacheEngine(EmbeddingStore(tmp_path), 2, LRUPolicy())
    engine.pool([[0, 1]])
    engine.pool([[0, 2]])
    assert engine.resident == {0, 2}


def test_lru_batch_uses_last_occurrence_for_recency():
    policy = LRUPolicy()
    policy.observe([4, 9, 4])
    decision = policy.choose(set(), [4, 9], 1)
    assert decision.resident == {4}


def test_static_policy_keeps_explicit_warm_set(tmp_path):
    generate_embeddings(8, 3, 5, tmp_path)
    engine = CPUCacheEngine(EmbeddingStore(tmp_path), 2, StaticPolicy([1, 3]))
    engine.resident = {1, 3}
    engine.pool([[2, 2]])
    assert engine.resident == {1, 3}


def test_static_policy_preloads_and_restores_its_warm_set(tmp_path):
    generate_embeddings(8, 3, 5, tmp_path)
    engine = CPUCacheEngine(EmbeddingStore(tmp_path), 2, StaticPolicy([1, 3, 5]))
    assert engine.resident == {1, 3}
    engine.reset()
    assert engine.resident == {1, 3}


def test_adaptive_shift_detection_is_deterministic():
    policy = AdaptivePolicy(window_occurrences=4, buckets=16, normal_budget=2, shift_budget=4)
    shifted_item = next(item for item in range(1, 1000) if policy._bucket(item) != policy._bucket(0))
    for _ in range(2):
        policy.observe([0, 0, 0, 0])
    for _ in range(2):
        policy.observe([shifted_item] * 4)
    assert policy.shift_score > 0
    assert policy.shift_windows_remaining > 0
    assert policy.mode_transitions


def test_adaptive_admission_margin_and_window_budget():
    policy = AdaptivePolicy(window_occurrences=4, buckets=16, normal_budget=1, shift_budget=2)
    policy.scores.update({10: 10.0, 20: 10.5, 30: 20.0})
    rejected = policy.choose({10}, [20], 1)
    assert rejected.resident == {10}
    admitted = policy.choose({10}, [30], 1)
    assert admitted.resident == {30}
    assert policy.choose({30}, [10], 1).resident == {30}
    policy.observe([1, 1, 1, 1])
    assert policy.maintenance_remaining == policy.normal_budget

