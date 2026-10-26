import json

import numpy as np
import pytest

from hotshift.data.generate import generate_embeddings
from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import CPUCacheEngine
from hotshift.policies.factory import create_policy
from hotshift.traffic.generate import SCENARIOS, TrafficSampler, generate_trace
from hotshift.traffic.replay import (
    BatchRecord,
    load_trace,
    phase_boundary,
    recovery,
    replay_engine,
    replay_trace,
    summarize,
    warm_items_from_trace,
    windows,
)


def test_trace_reproducibility_and_policy_replay(tmp_path):
    generate_embeddings(128, 8, 42, tmp_path / "data")
    first = generate_trace(tmp_path / "a.jsonl", 128, 80, 4, 9, "abrupt")
    second = generate_trace(tmp_path / "b.jsonl", 128, 80, 4, 9, "abrupt")
    assert first["trace_sha256"] == second["trace_sha256"]
    other_seed = generate_trace(tmp_path / "c.jsonl", 128, 80, 4, 10, "abrupt")
    assert other_seed["trace_sha256"] != first["trace_sha256"]
    result = replay_trace(tmp_path / "a.jsonl", EmbeddingStore(tmp_path / "data"), 16, "lru", 8)
    assert result["measured_requests"] == 64
    assert 0 <= result["occurrence_hit_rate"] <= 1


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_generates_valid_bounded_events(tmp_path, scenario):
    manifest = generate_trace(tmp_path / "t.jsonl", 1000, 200, 16, 3, scenario)
    events = load_trace(tmp_path / "t.jsonl")
    assert len(events) == 200 and manifest["requests"] == 200
    assert all(0 <= item < 1000 for event in events for item in event["history_ids"])
    assert all(len(event["history_ids"]) <= 256 for event in events)
    offsets = [event["scheduled_offset_ns"] for event in events]
    assert offsets == sorted(offsets)
    assert sum(event["warmup"] for event in events) == manifest["warmup_requests"] == 40
    assert manifest["phase_boundaries"][0]["first_event_index"] == 0


def test_phase_boundaries_are_known_and_shift_the_hot_set(tmp_path):
    manifest = generate_trace(tmp_path / "abrupt.jsonl", 10000, 400, 32, 1, "abrupt")
    events = load_trace(tmp_path / "abrupt.jsonl")
    assert phase_boundary(events) == 200
    assert [b["phase_id"] for b in manifest["phase_boundaries"]] == [0, 1]
    before = {item for event in events[:200] for item in event["history_ids"]}
    after = {item for event in events[200:] for item in event["history_ids"]}
    assert len(before & after) < 0.2 * len(after)
    burst = generate_trace(tmp_path / "burst.jsonl", 10000, 400, 8, 1, "brief_burst")
    assert [b["phase_id"] for b in burst["phase_boundaries"]] == [0, 1, 2]
    assert [b["first_event_index"] for b in burst["phase_boundaries"]] == [0, 160, 200]


def test_post_shift_reference_uses_the_abrupt_post_shift_mapping():
    abrupt = TrafficSampler(5000, "abrupt", 7)
    reference = TrafficSampler(5000, "post_shift", 7)
    assert np.array_equal(abrupt.permutation, reference.permutation)
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    shifted, _, _ = abrupt.draw(rng_a, 64, 0.9, 0)
    steady, _, _ = reference.draw(rng_b, 64, 0.1, 0)
    assert shifted == steady


def test_overload_compresses_scheduled_arrivals_and_mixed_regions_label_events(tmp_path):
    generate_trace(tmp_path / "o.jsonl", 1000, 400, 4, 2, "overload", request_rate=100.0)
    events = load_trace(tmp_path / "o.jsonl")
    normal_gap = events[10]["scheduled_offset_ns"] - events[9]["scheduled_offset_ns"]
    overload_gap = events[200]["scheduled_offset_ns"] - events[199]["scheduled_offset_ns"]
    assert normal_gap == pytest.approx(10 * overload_gap, rel=0.01)
    generate_trace(tmp_path / "m.jsonl", 1000, 8, 4, 2, "mixed_regions")
    regions = [event["region"] for event in load_trace(tmp_path / "m.jsonl")]
    assert regions[:4] == ["north", "south", "east", "west"]


def test_gradual_mixture_increases_shifted_share():
    sampler = TrafficSampler(100000, "gradual", 5)
    rng = np.random.default_rng(0)
    hot_original = set(sampler.permutation[:50].tolist())

    def original_share(fraction):
        ids, _, _ = sampler.draw(rng, 4000, fraction, 0)
        return np.mean([item in hot_original for item in ids])

    assert original_share(0.1) > original_share(0.6) > original_share(0.95)


def test_static_warm_set_uses_only_the_warmup_prefix(tmp_path):
    generate_trace(tmp_path / "t.jsonl", 1000, 100, 8, 4, "abrupt")
    events = load_trace(tmp_path / "t.jsonl")
    future_only = {item for event in events if not event["warmup"] for item in event["history_ids"]}
    prefix = {item for event in events if event["warmup"] for item in event["history_ids"]}
    warm, warm_requests = warm_items_from_trace(events, 50)
    assert warm_requests == 20
    assert set(warm) <= prefix
    assert not (set(warm) - prefix) & future_only


def test_batch_partitions_respect_phase_warmup_and_window_boundaries(tmp_path):
    generate_embeddings(1000, 4, 1, tmp_path / "data")
    store = EmbeddingStore(tmp_path / "data")
    generate_trace(tmp_path / "t.jsonl", 1000, 300, 4, 4, "abrupt")
    events = load_trace(tmp_path / "t.jsonl")
    partitions = {}
    for name in ("lru", "adaptive", "no_cache"):
        engine = CPUCacheEngine(store, 0 if name == "no_cache" else 50, create_policy(name))
        records = replay_engine(events, engine, 16, 25)
        partitions[name] = [(record.start_event, record.end_event) for record in records]
        for record in records:
            span = events[record.start_event : record.end_event]
            assert len({event["phase_id"] for event in span}) == 1
            assert len({event["warmup"] for event in span}) == 1
            assert record.start_event // 25 == (record.end_event - 1) // 25
    assert partitions["lru"] == partitions["adaptive"] == partitions["no_cache"]


def _record(start, end, hits, occurrences, wall=1.0, transitions=0):
    return BatchRecord(start, end, 0 if start < 100 else 1, False, end - start, start * 1000, end * 1000,
                       occurrences, hits, occurrences, hits, 0, None, None, 0, 0, wall, None, {}, None, None, transitions)


def test_recovery_states():
    records = [_record(start, start + 10, hits, 10, transitions=int(start == 120)) for start, hits in
               zip(range(0, 200, 10), [8] * 10 + [1, 2, 4, 7, 8, 8, 8, 8, 8, 8])]
    rows = windows(records, 10)
    recovered = recovery(records, rows, 100, 0.8)
    assert recovered["status"] == "recovered"
    assert recovered["recovery_requests"] == 50  # windows 140, 150, 160 are the first three at >= 0.72
    assert recovered["detection_requests"] == 20
    assert recovery(records, rows, 100, 0.99)["status"] == "not_recovered"
    assert recovery(records, rows, 100, 0.0)["status"] == "not_applicable"
    assert recovery(records, rows, None, 0.8)["status"] == "not_applicable"


def test_summary_is_json_safe_and_null_when_empty():
    summary = summarize([])
    assert summary["occurrence_hit_rate"] is None and summary["request_throughput"] is None
    json.dumps(summary, allow_nan=False)
