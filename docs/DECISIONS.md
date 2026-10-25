# Engineering decisions

## Initial environment and development scope

- The available host is Apple Silicon without `nvidia-smi` or `nvcc`; implement portable CPU foundations first and explicitly block CUDA execution, sanitizers, profiling, and GPU measurements.
- Declare Python 3.11+ and isolate the service stack as an optional dependency so the dataset and CPU reference path do not require a running web service.
- Use NumPy FP32 for the deterministic CPU reference and policy simulator; PyTorch supplies the optional CPU oracle and CUDA extension interface. Numerical parity between CUDA and reference paths remains a release gate.
- Save generated embeddings as memory-mapped `.npy` data and validate the manifest checksum on load. Generated data is local and ignored by Git.
- No redistribution license is selected in this initial scaffold; confirm ownership and licensing before publishing. No third-party implementation code is copied.
- Keep the initial CUDA engine serialized around host-side miss planning, using bounded staging and the current CUDA stream. Only introduce overlap after event ownership and sanitizer behavior are validated on NVIDIA hardware.

