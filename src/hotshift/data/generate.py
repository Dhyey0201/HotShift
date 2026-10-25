from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from hotshift.config import HotShiftConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_embeddings(
    rows: int,
    dimensions: int,
    seed: int,
    output_dir: str | Path,
    chunk_rows: int = 8192,
) -> dict[str, object]:
    if rows <= 0 or dimensions <= 0 or chunk_rows <= 0:
        raise ValueError("rows, dimensions, and chunk_rows must be positive")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="embeddings-", suffix=".npy", dir=target)
    os.close(fd)
    temporary_path = Path(temporary_name)
    output_path = target / "embeddings.npy"
    try:
        vectors = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.float32, shape=(rows, dimensions)
        )
        rng = np.random.default_rng(seed)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            block = rng.standard_normal((end - start, dimensions), dtype=np.float32)
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            np.divide(block, norms, out=block, where=norms > 0)
            vectors[start:end] = block
        vectors.flush()
        del vectors
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": "synthetic-v1",
        "rows": rows,
        "dimensions": dimensions,
        "dtype": "float32",
        "layout": "row_major",
        "seed": seed,
        "generator": "numpy.default_rng.standard_normal",
        "normalization": "unit_l2_per_row",
        "chunk_rows": chunk_rows,
        "embedding_sha256": _sha256(output_path),
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def generate_from_config(config: HotShiftConfig) -> dict[str, object]:
    return generate_embeddings(
        rows=config.rows,
        dimensions=config.dimensions,
        seed=config.seed,
        output_dir=config.output_dir,
    )

