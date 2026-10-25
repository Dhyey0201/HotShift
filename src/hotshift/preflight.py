from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any


def _probe(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def preflight(backend: str = "cpu") -> dict[str, Any]:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")
    nvidia_smi = shutil.which("nvidia-smi")
    nvcc = shutil.which("nvcc")
    gpu_list = _probe([nvidia_smi, "-L"]) if nvidia_smi else None
    nvcc_version = _probe([nvcc, "--version"]) if nvcc else None
    device_listed = bool(gpu_list and "GPU" in gpu_list)
    toolkit_available = nvcc_version is not None
    profiling_tools = {
        name: shutil.which(name) for name in ("nsys", "ncu", "compute-sanitizer")
    }
    try:
        host_memory_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        host_memory_bytes = None
    cuda_runtime_available = False
    torch_version = None
    torch_cuda_version = None
    device_name = None
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda_version = torch.version.cuda
        cuda_runtime_available = bool(torch.cuda.is_available())
        if cuda_runtime_available:
            device_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    cuda_available = device_listed and toolkit_available and cuda_runtime_available
    result: dict[str, Any] = {
        "requested_backend": backend,
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "host_memory_bytes": host_memory_bytes,
        "cpu_status": "available",
        "cuda_status": "available" if cuda_available else "blocked",
        "cuda_block_reason": None if cuda_available else "CUDA runtime, NVIDIA device, or toolkit is unavailable",
        "gpu_device_status": "available" if device_listed else "unavailable",
        "cuda_toolkit_status": "available" if toolkit_available else "unavailable",
        "cuda_runtime_status": "available" if cuda_runtime_available else "unavailable",
        "profiling_status": (
            "tools_present_permissions_unverified"
            if all(profiling_tools.values())
            else "unavailable_or_incomplete"
        ),
        "nvidia_smi": nvidia_smi,
        "gpu_list": gpu_list,
        "nvcc": nvcc,
        "nvcc_version": nvcc_version,
        "profiling_tools": profiling_tools,
        "torch": torch_version,
        "torch_cuda": torch_cuda_version,
        "device": device_name,
    }
    return result

