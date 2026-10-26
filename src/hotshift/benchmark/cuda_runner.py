from __future__ import annotations

from pathlib import Path

from hotshift.benchmark.runner import run_matrix


def run_cuda_matrix(matrix_path: str | Path, smoke: bool = False) -> Path:
    """Run the replay matrix with the custom CUDA cache engine and PyTorch baselines."""
    return run_matrix(matrix_path, smoke=smoke, backend="cuda")
