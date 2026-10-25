# Development

Create a virtual environment and install `python3 -m pip install -e .`. Generate the smoke fixture with `python3 -m hotshift.cli data generate --config configs/smoke.yaml`. Run `python3 scripts/preflight.py --backend cpu` to inspect the local environment. The GPU preflight is expected to report blocked on machines without both a supported NVIDIA device and CUDA toolchain.

The default smoke table uses deterministic seed 42 and is stored under ignored `artifacts/`. Do not commit generated tables, profiler captures, credentials, or large traces. GPU development requires matching the PyTorch CUDA build, local toolkit, compiler, and driver; record the resolved environment before profiling.

