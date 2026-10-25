# Engineering decisions

## Initial environment and development scope

- The available host is Apple Silicon without `nvidia-smi` or `nvcc`; implement portable CPU foundations first and explicitly block CUDA execution, sanitizers, profiling, and GPU measurements.
- Declare Python 3.11+ and isolate the service stack as an optional dependency so the dataset and CPU reference path do not require a running web service.
- Use NumPy FP32 for the initial reference implementation. PyTorch integration remains planned for scoring and the custom operator extension; numerical parity with the eventual PyTorch reference remains a release gate.
- Save generated embeddings as memory-mapped `.npy` data and validate the manifest checksum on load. Generated data is local and ignored by Git.
- No redistribution license is selected in this initial scaffold; confirm ownership and licensing before publishing. No third-party implementation code is copied.

