import json

import numpy as np
import pytest

from hotshift.data.catalog import generate_catalog
from hotshift.data.generate import generate_embeddings
from hotshift.data.store import EmbeddingStore
from hotshift.recommend.scoring import CatalogScorer


def test_seeded_data_regenerates_identically(tmp_path):
    first = generate_embeddings(128, 17, 71, tmp_path / "first", chunk_rows=31)
    second = generate_embeddings(128, 17, 71, tmp_path / "second", chunk_rows=64)
    assert first["embedding_sha256"] == second["embedding_sha256"]
    store = EmbeddingStore(tmp_path / "first")
    assert not store.table.flags.writeable
    assert np.allclose(np.linalg.norm(store.table, axis=1), 1.0, atol=1e-6)
    different = generate_embeddings(128, 17, 72, tmp_path / "third")
    assert different["embedding_sha256"] != first["embedding_sha256"]


def test_store_rejects_corrupt_or_mismatched_data(tmp_path):
    generate_embeddings(16, 4, 1, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps({**manifest, "embedding_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="checksum"):
        EmbeddingStore(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({**manifest, "rows": 17}))
    with pytest.raises(ValueError, match="shape"):
        EmbeddingStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        EmbeddingStore(tmp_path / "missing")


def test_store_rejects_nonfinite_rows(tmp_path):
    generate_embeddings(4, 2, 1, tmp_path)
    table = np.load(tmp_path / "embeddings.npy")
    table[1, 0] = np.nan
    np.save(tmp_path / "embeddings.npy", table)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["embedding_sha256"] = EmbeddingStore._sha256(tmp_path / "embeddings.npy")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonfinite"):
        EmbeddingStore(tmp_path)


def test_catalog_is_bounded_synthetic_and_checksummed(tmp_path):
    catalog = generate_catalog(10000, 8, 3, tmp_path, candidate_count=64)
    assert len(catalog) == 4096
    assert {item["dataset_version"] for item in catalog} == {"synthetic-v1"}
    assert catalog[0]["title"].startswith("Synthetic Track")
    scorer = CatalogScorer(tmp_path)
    vector = np.asarray(scorer.embeddings[5], dtype=np.float32)
    items, ranking = scorer.recommend(vector, [5], 10, exclude_seen=True)
    assert ranking == "similarity" and len(items) == 10 and 5 not in [item["item_id"] for item in items]
    scores = [item["score"] for item in items]
    assert scores == sorted(scores, reverse=True)
    fallback, ranking = scorer.recommend(np.zeros(8, dtype=np.float32), [], 5, exclude_seen=True)
    assert ranking == "popularity_fallback" and [item["item_id"] for item in fallback] == [0, 1, 2, 3, 4]
    everything, _ = scorer.recommend(vector, list(range(60)), 100, exclude_seen=True)
    assert len(everything) == 4  # only 64 - 60 eligible candidates remain
    (tmp_path / "catalog.json").write_text("[]")
    with pytest.raises(ValueError):
        CatalogScorer(tmp_path)
