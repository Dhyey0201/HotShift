# Run HotShift CUDA checks in Kaggle

In Kaggle, create a Python notebook, enable **Internet**, and select **GPU** under **Settings → Session options → Accelerator**. Then copy the entire Python block below into one code cell and run it. It clones or updates HotShift from GitHub, checks for a GPU and `nvcc`, builds the CUDA extension, runs CUDA/cache/API tests, creates a short CUDA benchmark artifact, and runs Compute Sanitizer when available. If the cell reports that `github.com` cannot be resolved, turn Internet on in the notebook settings, restart the session, and rerun the cell.

```python
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

def run(*command):
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)

try:
    urllib.request.urlopen("https://github.com", timeout=15).close()
except Exception as error:
    raise RuntimeError("Kaggle cannot reach GitHub. Enable Internet in notebook Settings, restart the session, and rerun this cell.") from error

repo = "/kaggle/working/HotShift"
if os.path.isdir(os.path.join(repo, ".git")):
    run("git", "-C", repo, "pull", "--ff-only")
else:
    repo = os.path.join(tempfile.mkdtemp(prefix="hotshift-", dir="/kaggle/working"), "HotShift")
    run("git", "clone", "https://github.com/Dhyey0201/HotShift.git", repo)
os.chdir(repo)

import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("nvcc:", shutil.which("nvcc"))

if not torch.cuda.is_available():
    raise RuntimeError("Enable the Kaggle GPU accelerator and restart the notebook session.")
if shutil.which("nvcc") is None:
    raise RuntimeError("This Kaggle image has no nvcc compiler, so HotShift's CUDA extension cannot be built here.")

run(sys.executable, "-m", "pip", "install", "-e", ".[dev,service]")
run(sys.executable, "scripts/preflight.py", "--backend", "cuda")
run(sys.executable, "-m", "hotshift.cli", "data", "generate", "--config", "configs/smoke.yaml")
run(sys.executable, "-m", "hotshift.cli", "data", "catalog", "--config", "configs/smoke.yaml")
run(sys.executable, "-m", "hotshift.cli", "build-extension")
run(sys.executable, "-m", "pytest", "tests/cpu", "tests/integration", "tests/cuda", "-q")
run(sys.executable, "-m", "hotshift.cli", "bench-cuda", "--matrix", "configs/smoke.yaml", "--smoke")

if shutil.which("nsys") or shutil.which("ncu"):
    try:
        run("bash", "scripts/profile.sh", "--matrix", "configs/smoke.yaml", "--smoke")
    except subprocess.CalledProcessError as error:
        print(f"Profiler could not run in this Kaggle session (exit {error.returncode}); benchmark results are still saved.")

if shutil.which("compute-sanitizer"):
    run("bash", "scripts/run_sanitizers.sh")
else:
    print("SKIPPED: Compute Sanitizer is not installed in this Kaggle image.")
```

Kaggle GPU availability and installed tools vary. If the cell reports that `nvcc` is missing, this runtime cannot compile the CUDA extension. Kaggle may also deny Nsight Compute access to GPU performance counters (`ERR_NVGPUCTRPERM`); the benchmark artifact is still saved when that happens. The short `bench-cuda --smoke` run is functional evidence only; use `configs/benchmark.yaml` for the full repeated matrix. `hotshift bench` remains CPU replay only.

