from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class EmbeddingStore:
    """Read-only, memory-mapped FP32 embedding table with manifest validation."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        data_path = self.directory / "embeddings.npy"
        if not manifest_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(f"dataset is incomplete: {self.directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        table = np.load(data_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (int(manifest["rows"]), int(manifest["dimensions"]))
        if table.shape != expected_shape or table.dtype != np.float32 or not table.flags.c_contiguous:
            raise ValueError("embedding table does not match manifest shape, dtype, or layout")
        if manifest.get("dtype") != "float32" or manifest.get("layout") != "row_major":
            raise ValueError("unsupported embedding manifest dtype or layout")
        if self._sha256(data_path) != manifest.get("embedding_sha256"):
            raise ValueError("embedding checksum does not match manifest")
        if not np.isfinite(table).all():
            raise ValueError("embedding table contains nonfinite values")
        table.flags.writeable = False
        self.table = table
        self.manifest = manifest

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @property
    def rows(self) -> int:
        return int(self.table.shape[0])

    @property
    def dimensions(self) -> int:
        return int(self.table.shape[1])

