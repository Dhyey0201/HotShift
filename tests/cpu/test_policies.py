"""Policy semantics against brute-force reference simulators on small traces."""

from collections import OrderedDict

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hotshift.data.generate import generate_embeddings
from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import CPUCacheEngine
from hotshift.policies.adaptive import AdaptivePolicy
from hotshift.policies.decayed_lfu import DecayedLFUPolicy
from hotshift.policies.factory import POLICY_NAMES, create_policy
from hotshift.policies.lru import LRUPolicy
from hotshift.policies.no_cache import NoCachePolicy
from hotshift.policies.static import StaticPolicy

batches = st.lists(st.lists(st.integers(0, 29), max_size=24), min_size=1, max_size=25)


def _drive(policy, capacity, trace):
    resident: set[int] = set()
    history = []
    for batch in trace:
        misses = sorted(set(batch) - resident)
        policy.observe(batch)
        decision = policy.choose(resident, misses, capacity)
        assert set(decision.admissions) <= set(misses), "admissions must be batch misses"
        assert set(decision.evictions) <= resident, "evictions must be resident"
        assert len(set(decision.admissions)) == len(decision.admissions)
        decision.apply(resident)
        assert len(resident) <= capacity
        history.append((frozenset(resident), decision))
    return history


class BruteLRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.order: OrderedDict[int, None] = OrderedDict()

    def batch(self, items):
        for item in items:
            self.order.pop(item, None)
            self.order[item] = None
            while len(self.order) > self.capacity:
                self.order.popitem(last=False)
        return set(self.order)


@settings(max_examples=150, deadline=None)
@given(batches, st.integers(1, 12))
def test_lru_matches_per_occurrence_reference(trace, capacity):
    reference = BruteLRU(capacity)
    for resident, _ in _drive(LRUPolicy(), capacity, trace):
        pass
    # Replaying occurrence by occurrence yields the same final state as batched recency.
    for batch in trace:
        expected = reference.batch(batch)
    assert resident == expected


def test_lru_batch_uses_last_occurrence_for_recency():
    policy = LRUPolicy()
    policy.observe([4, 9, 4])
    decision = policy.choose(set(), [4, 9], 1)
    assert decision.admissions == (4,) and decision.evictions == ()


def _brute_scores(trace, window, decay):
    """Inflated decayed-frequency scores after each batch, computed from scratch."""
    growth = 1.0 / decay
    occurrences = [item for batch in trace for item in batch]
    snapshots, position, scores = [], 0, {}
    for batch in trace:
        for item in batch:
            weight = growth ** (position // window)
            scores[item] = scores.get(item, 0.0) + weight
            position += 1
        snapshots.append(dict(scores))
    assert position == len(occurrences)
    return snapshots


@settings(max_examples=150, deadline=None)
@given(batches, st.integers(1, 10), st.integers(1, 9))
def test_decayed_lfu_equals_top_capacity_of_resident_and_misses(trace, capacity, window):
    policy = DecayedLFUPolicy(window_occurrences=window, decay=0.8)
    snapshots = _brute_scores(trace, window, 0.8)
    resident: set[int] = set()
    for batch, scores in zip(trace, snapshots):
        misses = sorted(set(batch) - resident)
        pool = resident | set(misses)
        expected = set(sorted(pool, key=lambda item: (-scores.get(item, 0.0), item))[:capacity])
        policy.observe(batch)
        policy.choose(resident, misses, capacity).apply(resident)
        assert resident == expected


def test_decayed_lfu_decay_and_flush_semantics():
    policy = DecayedLFUPolicy(window_occurrences=2, decay=0.5)
    policy.observe([1, 1])  # window 0 complete: score(1) = 2
    assert policy.score(1) == pytest.approx(1.0)  # viewed from window 1: 2 * 0.5
    policy.observe([2])  # partial window counts at full weight
    assert policy.score(2) == pytest.approx(1.0)
    policy.flush()
    assert policy.score(2) == pytest.approx(0.5)
    assert policy.score(1) == pytest.approx(0.5)


def test_decayed_lfu_renormalizes_without_changing_ranking(monkeypatch):
    import hotshift.policies.decayed_lfu as module

    monkeypatch.setattr(module, "_RENORMALIZE_AT", 10.0)
    policy = DecayedLFUPolicy(window_occurrences=1, decay=0.5)
    resident: set[int] = set()
    for item in [1, 2, 1, 3, 1, 2, 4, 4, 4, 4, 5]:
        misses = [item] if item not in resident else []
        policy.observe([item])
        policy.choose(resident, misses, 2).apply(resident)
        assert policy.weight <= 10.0 * 2
    assert resident == {4, 5}


def test_adaptive_shift_detection_is_deterministic():
    def run():
        policy = AdaptivePolicy(window_occurrences=4, buckets=16, normal_budget=2, shift_budget=4)
        shifted_item = next(item for item in range(1, 1000) if policy._bucket(item) != policy._bucket(0))
        for _ in range(2):
            policy.observe([0, 0, 0, 0])
        for _ in range(2):
            policy.observe([shifted_item] * 4)
        return policy

    first, second = run(), run()
    assert first.shift_score > 0
    assert first.in_shift_mode
    assert first.mode_transitions == second.mode_transitions
    assert first.mode_transitions[0]["mode"] == "shift"


def test_adaptive_scalar_and_vector_hash_agree():
    policy = AdaptivePolicy(buckets=4096)
    items = np.arange(0, 5000, 37, dtype=np.int64)
    assert policy._bucket_array(items, 4096).tolist() == [policy._bucket(int(item)) for item in items]


def test_adaptive_admission_margin_and_window_budget():
    policy = AdaptivePolicy(window_occurrences=4, buckets=16, normal_budget=1, shift_budget=2)
    policy.scores.update({10: 10.0, 20: 10.5, 30: 20.0, 40: 50.0})
    resident = {10}
    rejected = policy.choose(resident, [20], 1)
    assert rejected.admissions == ()  # 10.5 does not exceed 10 by 10 percent
    admitted = policy.choose(resident, [30], 1)
    assert admitted.admissions == (30,) and admitted.evictions == (10,)
    admitted.apply(resident)
    # The window's single replacement is spent, even for a much better candidate.
    assert policy.choose(resident, [40], 1).admissions == ()
    # An item evicted this window cannot return in the same window.
    assert 10 in policy.evicted_this_window
    policy.observe([1, 1, 1, 1])
    assert policy.maintenance_remaining == policy.normal_budget
    assert policy.evicted_this_window == set()


def test_adaptive_byte_budget_limits_replacements():
    policy = AdaptivePolicy(window_occurrences=1000, normal_budget=8, row_bytes=128, normal_budget_bytes=256)
    policy.scores.update({1: 1.0, 2: 1.0, 3: 1.0, 11: 9.0, 12: 9.0, 13: 9.0})
    resident = {1, 2, 3}
    decision = policy.choose(resident, [11, 12, 13], 3)
    assert len(decision.evictions) == 2  # 256 bytes / 128-byte rows


def test_adaptive_fills_empty_slots_without_spending_budget():
    policy = AdaptivePolicy(window_occurrences=1000, normal_budget=1)
    resident: set[int] = set()
    policy.observe([1, 2, 3])
    decision = policy.choose(resident, [1, 2, 3], 3)
    assert set(decision.admissions) == {1, 2, 3}
    assert policy.maintenance_remaining == 1


@settings(max_examples=60, deadline=None)
@given(batches, st.integers(1, 8))
def test_adaptive_never_exceeds_budget_or_readmits_within_window(trace, capacity):
    budget = 2
    policy = AdaptivePolicy(window_occurrences=16, buckets=8, normal_budget=budget, shift_budget=budget)
    resident: set[int] = set()
    replacements_in_window = 0
    window = 0
    evicted_in_window: set[int] = set()
    for batch in trace:
        policy.observe(batch)
        if policy.window_index != window:
            window, replacements_in_window, evicted_in_window = policy.window_index, 0, set()
        misses = sorted(set(batch) - resident)
        decision = policy.choose(resident, misses, capacity)
        assert not set(decision.admissions) & evicted_in_window
        replacements_in_window += len(decision.evictions)
        evicted_in_window |= set(decision.evictions)
        assert replacements_in_window <= budget
        decision.apply(resident)
        assert len(resident) <= capacity


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_every_policy_is_deterministic_and_bounded(name):
    rng = np.random.default_rng(3)
    trace = [rng.integers(0, 40, size=rng.integers(0, 30)).tolist() for _ in range(60)]

    def run():
        policy = StaticPolicy(range(5)) if name == "static" else create_policy(name)
        return [(state, decision) for state, decision in _drive(policy, 6 if name != "no_cache" else 0, trace)]

    assert run() == run()


def test_static_policy_preloads_and_restores_its_warm_set(tmp_path):
    generate_embeddings(8, 3, 5, tmp_path)
    engine = CPUCacheEngine(EmbeddingStore(tmp_path), 2, StaticPolicy([1, 3, 5]))
    assert engine.resident == {1, 3}
    engine.pool([[2, 2]])
    assert engine.resident == {1, 3}
    engine.reset()
    assert engine.resident == {1, 3}


def test_no_cache_evicts_everything():
    decision = NoCachePolicy().choose({3, 1}, [5], 0)
    assert decision.admissions == () and decision.evictions == (1, 3)


def test_cpu_engine_results_do_not_depend_on_policy(tmp_path):
    generate_embeddings(64, 9, 2, tmp_path)
    store = EmbeddingStore(tmp_path)
    rng = np.random.default_rng(8)
    batches_ = [[rng.integers(0, 64, size=rng.integers(0, 12)).tolist() for _ in range(5)] for _ in range(20)]
    outputs = {}
    for name in POLICY_NAMES:
        policy = StaticPolicy(range(10)) if name == "static" else create_policy(name)
        engine = CPUCacheEngine(store, 0 if name == "no_cache" else 7, policy)
        outputs[name] = [engine.pool(batch).outputs for batch in batches_]
    for name in POLICY_NAMES:
        for left, right in zip(outputs["no_cache"], outputs[name]):
            np.testing.assert_array_equal(left, right)


def test_lru_eviction_uses_request_occurrence_order(tmp_path):
    generate_embeddings(8, 3, 5, tmp_path)
    engine = CPUCacheEngine(EmbeddingStore(tmp_path), 2, LRUPolicy())
    engine.pool([[0, 1]])
    engine.pool([[0, 2]])
    assert engine.resident == {0, 2}
