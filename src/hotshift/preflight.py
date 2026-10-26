from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _probe(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def preflight(backend: str = "cpu", config_path: str | None = None) -> dict[str, Any]:
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
        host_total_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        host_available_bytes = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        host_total_bytes = None
        host_available_bytes = None
    device_free_bytes = None
    if nvidia_smi:
        free_memory = _probe([nvidia_smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        try:
            device_free_bytes = int(free_memory.splitlines()[0].strip()) * 1024 * 1024 if free_memory else None
        except (IndexError, ValueError):
            device_free_bytes = None
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
        "host_memory_total_bytes": host_total_bytes,
        "host_memory_available_bytes": host_available_bytes,
        "device_free_bytes": device_free_bytes,
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
    selected_config = config_path or os.getenv("HOTSHIFT_CONFIG")
    if selected_config and Path(selected_config).is_file():
        try:
            from hotshift.config import load_config

            config = load_config(selected_config)
            # Mirrors CUDABuffers: per-ID ids/slots/source/inverse plus maintenance keys, sources,
            # and slots; worst-case miss rows; offsets and outputs. Allocator slack is extra.
            per_id_bytes = 8 + 4 + 4 + 8 + 4 + 8 + 8 + 8
            device_estimate = (
                config.cache_bytes
                + config.rows * 4
                + config.max_batch_ids * (per_id_bytes + config.dimensions * 4)
                + (config.max_batch_requests + 1) * 8
                + config.max_batch_requests * config.dimensions * 4
            )
            # Host: table, matching pinned staging, CPU slot mirror, and policy dictionaries
            # (roughly 120 bytes per scored item for the frequency policies).
            host_estimate = (
                config.table_bytes
                + config.max_batch_ids * (per_id_bytes + config.dimensions * 4)
                + config.cache_capacity * 8
                + config.rows * 120
                + config.queue_capacity * config.max_history_ids * 8
            )
            overridden = os.getenv("HOTSHIFT_ALLOW_OVERCAPACITY") == "1"
            device_ok = device_free_bytes is None or device_estimate <= device_free_bytes * 0.70
            host_ok = host_available_bytes is None or host_estimate <= host_available_bytes * 0.60
            result["capacity_estimate"] = {
                "device_bytes": device_estimate,
                "host_bytes": host_estimate,
                "device_limit_bytes": int(device_free_bytes * 0.70) if device_free_bytes else None,
                "host_limit_bytes": int(host_available_bytes * 0.60) if host_available_bytes else None,
                "override_enabled": overridden,
                "status": "available" if (device_ok and host_ok) or overridden else "blocked",
            }
            if backend == "cuda" and not ((device_ok and host_ok) or overridden):
                result["cuda_status"] = "blocked"
                result["cuda_block_reason"] = "projected allocation exceeds the configured free-memory safety limit"
        except (ImportError, OSError, ValueError) as error:
            result["capacity_estimate"] = {"status": "invalid", "reason": str(error)}
    return result



_KERNEL_PROBE = (
    "import torch;from hotshift.engine.cuda import build_cuda_extension;build_cuda_extension();"
    "m=torch.tensor([-1,0],dtype=torch.int32,device='cuda');"
    "r=torch.ops.hotshift.probe_slots(torch.tensor([0,1,5],device='cuda'),m).cpu().tolist();"
    "assert r==[-1,0,-1],r;print('kernel-ok')"
)


def deep_checks(timeout: float = 600.0) -> dict[str, Any]:
    """Compile the extension, execute one owned kernel, and test profiler counter access.

    Profiling status is one of ``available``, ``restricted`` (tools present but counters
    denied, e.g. ERR_NVGPUCTRPERM), or ``unavailable`` (tools missing).
    """
    result: dict[str, Any] = {}
    try:
        kernel = subprocess.run([sys.executable, "-c", _KERNEL_PROBE], capture_output=True, text=True, timeout=timeout, check=False)
        result["extension_kernel"] = "ok" if "kernel-ok" in kernel.stdout else "failed"
        if result["extension_kernel"] != "ok":
            result["extension_error"] = (kernel.stderr or kernel.stdout)[-2000:]
    except subprocess.TimeoutExpired:
        result["extension_kernel"] = "timeout"
    ncu = shutil.which("ncu")
    if ncu is None:
        result["counter_profiling"] = "unavailable"
        result["counter_profiling_reason"] = "ncu not installed"
        return result
    try:
        probe = subprocess.run(
            [ncu, "--metrics", "sm__cycles_elapsed.avg", "--launch-count", "1", sys.executable, "-c",
             "import torch;torch.ones(8,device='cuda').sum().item()"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        output = probe.stdout + probe.stderr
        if "ERR_NVGPUCTRPERM" in output:
            result["counter_profiling"] = "restricted"
            result["counter_profiling_reason"] = "ERR_NVGPUCTRPERM: the host denies GPU performance counters"
        elif probe.returncode == 0 and "sm__cycles_elapsed" in output:
            result["counter_profiling"] = "available"
        else:
            result["counter_profiling"] = "failed"
            result["counter_profiling_reason"] = output[-1000:]
    except subprocess.TimeoutExpired:
        result["counter_profiling"] = "timeout"
    result["timeline_profiling"] = "nsys available" if shutil.which("nsys") else "nsys unavailable; use hotshift profile-torch (CUPTI tracing)"
    return result
