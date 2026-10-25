from hotshift.data.generate import generate_embeddings
from hotshift.data.store import EmbeddingStore
from hotshift.traffic.generate import generate_trace
from hotshift.traffic.replay import replay_trace


def test_trace_reproducibility_and_policy_replay(tmp_path):
    generate_embeddings(128, 8, 42, tmp_path / "data")
    first = generate_trace(tmp_path / "a.jsonl", 128, 80, 4, 9, "abrupt")
    second = generate_trace(tmp_path / "b.jsonl", 128, 80, 4, 9, "abrupt")
    assert first["trace_sha256"] == second["trace_sha256"]
    result = replay_trace(tmp_path / "a.jsonl", EmbeddingStore(tmp_path / "data"), 16, "lru", 8)
    assert result["measured_requests"] == 64
    assert 0 <= result["occurrence_hit_rate"] <= 1

