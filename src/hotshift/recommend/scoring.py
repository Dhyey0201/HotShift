from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class CatalogScorer:
    def __init__(self, directory: str | Path) -> None:
        target = Path(directory)
        manifest = json.loads((target / "catalog_manifest.json").read_text(encoding="utf-8"))
        self.items: list[dict[str, Any]] = json.loads((target / "catalog.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(target / "catalog_embeddings.npy", mmap_mode="r", allow_pickle=False)
        self.candidate_ids = np.load(target / "candidate_ids.npy", mmap_mode="r", allow_pickle=False)
        if self.embeddings.ndim != 2 or self.embeddings.dtype != np.float32:
            raise ValueError("catalog embeddings must be a two-dimensional FP32 array")
        if len(self.items) != self.embeddings.shape[0] or len(self.items) > 4096:
            raise ValueError("catalog row count is invalid or exceeds the 4096-item scoring bound")
        if self.candidate_ids.ndim != 1 or len(self.candidate_ids) > 4096:
            raise ValueError("candidate IDs must be a bounded one-dimensional array")
        if len(np.unique(self.candidate_ids)) != len(self.candidate_ids) or np.any(self.candidate_ids < 0) or np.any(self.candidate_ids >= len(self.items)):
            raise ValueError("candidate IDs contain duplicates or out-of-range values")
        if manifest.get("rows") != len(self.items) or manifest.get("dimensions") != self.embeddings.shape[1]:
            raise ValueError("catalog manifest dimensions do not match catalog data")
        for filename, key in (("catalog_embeddings.npy", "embedding_sha256"), ("catalog.json", "catalog_sha256"), ("candidate_ids.npy", "candidate_ids_sha256")):
            if self._sha256(target / filename) != manifest.get(key):
                raise ValueError(f"catalog checksum mismatch: {filename}")
        self.manifest = manifest

    def recommend(self, vector: np.ndarray, history: list[int], top_k: int, exclude_seen: bool) -> list[dict[str, Any]]:
        if vector.shape != (self.embeddings.shape[1],):
            raise ValueError("history and catalog embedding dimensions differ")
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            normalized = vector / norm
            candidate_scores = np.asarray(self.embeddings[self.candidate_ids] @ normalized, dtype=np.float32)
        else:
            candidate_scores = np.zeros(len(self.candidate_ids), dtype=np.float32)
        candidates = np.asarray(self.candidate_ids, dtype=np.int64)
        if exclude_seen and history:
            keep = ~np.isin(candidates, np.asarray(history, dtype=np.int64))
            candidates = candidates[keep]
            candidate_scores = candidate_scores[keep]
        ordering = np.lexsort((candidates, -candidate_scores))
        return [
            {"item_id": int(candidates[index]), "score": float(candidate_scores[index]), **self.items[int(candidates[index])]}
            for index in ordering[:top_k]
        ]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

