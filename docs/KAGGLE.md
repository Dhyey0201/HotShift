# Run the HotShift GPU gates on Kaggle

Create a Python notebook, turn **Internet** on, and select a **GPU** accelerator (a T4 was used before). Each cell below is self-contained. The cells clone `https://github.com/Dhyey0201/HotShift` and check out the `BRANCH` named at the top of each cell (currently `complete-gpu-evidence`; set it to `main` once that branch is merged).

## Cell 1: gates (about 15–30 minutes)

Builds the extension, runs every test with CUDA skips counted as failures, runs Compute Sanitizer, the smoke CUDA benchmark, the kernel microbenchmark, and a `torch.profiler` capture, then packages the results.

```python
import os, subprocess, sys, urllib.request

def run(*command):
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)

BRANCH = "complete-gpu-evidence"
urllib.request.urlopen("https://github.com", timeout=15).close()  # fails fast if Internet is off
repo = "/kaggle/working/HotShift"
if not os.path.isdir(os.path.join(repo, ".git")):
    run("git", "clone", "https://github.com/Dhyey0201/HotShift.git", repo)
run("git", "-C", repo, "fetch", "origin", BRANCH)
run("git", "-C", repo, "checkout", "-B", BRANCH, f"origin/{BRANCH}")
os.chdir(repo)
run(sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev,service]")
run("bash", "scripts/gpu_evidence.sh", "--quick")
```

If `compute-sanitizer` is missing from the image, the script logs `BLOCKED` for that gate instead of passing it. Nsight Compute is attempted when present. Kaggle has denied its counters (`ERR_NVGPUCTRPERM`), and the `torch.profiler` capture under `artifacts/profiles/` is the fallback.

## Cell 2: full evidence (several hours; run in a fresh session)

Generates the 1M-row datasets and runs the standard matrix, the ablation, the cache and dimension sweeps, and the open-loop load sweep against a CUDA server. Kaggle sessions are limited to 12 hours. If a run is cut off, its directory is marked `INCOMPLETE` and ignored, and the individual `bench-cuda --matrix configs/<name>.yaml` commands in `scripts/gpu_evidence.sh` can be run in separate sessions.

```python
import os, subprocess, sys
BRANCH = "complete-gpu-evidence"
os.chdir("/kaggle/working/HotShift")
subprocess.run(["git", "fetch", "origin", BRANCH], check=True)
subprocess.run(["git", "checkout", "-B", BRANCH, f"origin/{BRANCH}"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev,service]"], check=True)
subprocess.run(["bash", "scripts/gpu_evidence.sh"], check=True)
```

## Cell 3: load sweep only (about 15 minutes)

Reruns just the open-loop load sweep (warmup, stepped capacity ramp, offered-load fractions, overload) against a CUDA server. It writes its own log, `artifacts/logs/gpu_evidence-load.log`, so it never replaces the gate evidence.

```python
import os, subprocess, sys
BRANCH = "complete-gpu-evidence"
os.chdir("/kaggle/working/HotShift")
subprocess.run(["git", "fetch", "origin", BRANCH], check=True)
subprocess.run(["git", "checkout", "-B", BRANCH, f"origin/{BRANCH}"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev,service]"], check=True)
subprocess.run(["bash", "scripts/gpu_evidence.sh", "--load"], check=True)
```

## Bringing results back

Every mode finishes with `artifacts/hotshift-gpu-evidence-<timestamp>.tar.gz` (raw traces are excluded because they are regenerable from the seeds in each manifest). Download it from the notebook's output panel, then locally:

```bash
tar -xzf hotshift-gpu-evidence-*.tar.gz        # restores artifacts/runs, profiles, sanitizers, logs
python3 -m hotshift.cli report --runs artifacts/runs
python3 scripts/release_check.py               # CPU checks plus validation of the imported GPU evidence
```

Then copy the measured results, with run IDs, into `docs/BENCHMARKS.md` and `docs/PERFORMANCE.md`, and commit small summaries (manifest, summary, aggregate, and REPORT) as representative evidence.
