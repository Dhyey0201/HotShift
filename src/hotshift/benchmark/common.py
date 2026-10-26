"""Artifact, environment, and statistics helpers shared by every benchmark runner."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import numpy as np

INCOMPLETE_MARKER = "INCOMPLETE"
SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_run_id(kind: str | None = None) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{kind}-{uuid4().hex[:8]}" if kind else f"{stamp}-{uuid4().hex[:8]}"


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not len(values):
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100))


def median_or_none(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.median(present)) if present else None


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _run(command: list[str], timeout: float = 5) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_state() -> dict[str, object]:
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        # A dirty tree means the measured source is not exactly the recorded commit.
        "release_eligible": bool(commit) and status == "",
    }


def environment(include_cuda: bool) -> dict[str, object]:
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__,
    }
    try:
        import os

        info["host_memory_total_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        info["host_memory_total_bytes"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        if include_cuda and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info.update({
                "gpu": torch.cuda.get_device_name(0),
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
                "gpu_total_memory_bytes": int(properties.total_memory),
                "driver": _first_line(_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])),
                "nvcc": _last_line(_run(["nvcc", "--version"])),
            })
    except ImportError:
        info["torch"] = None
    return info


def _first_line(text: str | None) -> str | None:
    return text.splitlines()[0] if text else None


def _last_line(text: str | None) -> str | None:
    return text.splitlines()[-1] if text else None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class RunDirectory:
    """Writes into a hidden temporary directory, then atomically renames on completion.

    If the run fails or is interrupted, the directory is renamed into place anyway with an
    ``INCOMPLETE`` marker describing why, so partial evidence is never mistaken for a run.
    """

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root
        self.run_id = run_id
        root.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=root))
        (self.path / INCOMPLETE_MARKER).write_text("run in progress\n", encoding="utf-8")

    @property
    def destination(self) -> Path:
        return self.root / self.run_id

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self._finalize()
        else:
            (self.path / INCOMPLETE_MARKER).write_text(
                f"run did not complete: {exc_type.__name__}: {exc}\n", encoding="utf-8"
            )
            if self.destination.exists():
                shutil.rmtree(self.path, ignore_errors=True)
            else:
                self.path.replace(self.destination)
        return False

    def _finalize(self) -> None:
        (self.path / INCOMPLETE_MARKER).unlink()
        checksums = {
            path.relative_to(self.path).as_posix(): sha256_file(path)
            for path in sorted(self.path.rglob("*"))
            if path.is_file() and path.name != "checksums.json"
        }
        write_json(self.path / "checksums.json", checksums)
        self.path.replace(self.destination)


def is_complete_run(directory: Path) -> bool:
    return (
        directory.is_dir()
        and not directory.name.startswith(".")
        and not (directory / INCOMPLETE_MARKER).exists()
        and (directory / "manifest.json").is_file()
    )
