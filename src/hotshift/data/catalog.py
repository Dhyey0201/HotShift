from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hotshift.data.generate import _sha256

GENRES = ("ambient", "electronic", "folk", "jazz", "pop", "rock", "soul", "classical")
REGIONS = ("north", "south", "east", "west")


def generate_catalog(
    rows: int,
    dimensions: int,
    seed: int,
    output_dir: str | Path,
    candidate_count: int = 512,
) -> list[dict[str, object]]:
    if rows <= 0 or dimensions <= 0 or not 0 < candidate_count <= min(rows, 4096):
        raise ValueError("catalog dimensions and candidate_count are out of range")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    catalog_rows = min(rows, 4096)
    if candidate_count > catalog_rows:
        raise ValueError("candidate_count cannot exceed the bounded catalog size")
    genre_vectors = rng.standard_normal((len(GENRES), dimensions), dtype=np.float32)
    genre_vectors /= np.maximum(np.linalg.norm(genre_vectors, axis=1, keepdims=True), 1e-12)
    genres = rng.integers(0, len(GENRES), size=catalog_rows)
    embeddings = genre_vectors[genres] + rng.standard_normal((catalog_rows, dimensions), dtype=np.float32) * 0.08
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    catalog = [
        {
            "item_id": item_id,
            "title": f"Synthetic Track {item_id:06d}",
            "artist": f"Fictional Artist {item_id % 173:03d}",
            "genre": GENRES[int(genres[item_id])],
            "region": REGIONS[item_id % len(REGIONS)],
            "dataset_version": "synthetic-v1",
        }
        for item_id in range(catalog_rows)
    ]
    embedding_path = target / "catalog_embeddings.npy"
    metadata_path = target / "catalog.json"
    candidates_path = target / "candidate_ids.npy"
    np.save(embedding_path, embeddings.astype(np.float32, copy=False))
    metadata_path.write_text(
        json.dumps(catalog, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    np.save(candidates_path, np.arange(candidate_count, dtype=np.int64))
    manifest = {
        "schema_version": 1,
        "dataset_version": "synthetic-v1",
        "rows": catalog_rows,
        "dimensions": dimensions,
        "candidate_count": candidate_count,
        "embedding_sha256": _sha256(embedding_path),
        "catalog_sha256": _sha256(metadata_path),
        "candidate_ids_sha256": _sha256(candidates_path),
    }
    (target / "catalog_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return catalog

