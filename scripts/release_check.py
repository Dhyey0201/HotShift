from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hotshift.preflight import preflight

required = [
    "README.md",
    "docs/STATUS.md",
    "src/hotshift/engine/cpu.py",
    "src/hotshift/serving/app.py",
    "csrc/probe.cu",
    "csrc/pool.cu",
    "frontend/package.json",
]
missing = [name for name in required if not (ROOT / name).exists()]
gpu = preflight("cuda")
report = {"missing_files": missing, "cuda_status": gpu["cuda_status"], "cuda_block_reason": gpu["cuda_block_reason"]}
print(json.dumps(report, indent=2))
if missing:
    raise SystemExit(1)
commands = [
    [sys.executable, "-m", "pytest", "tests/cpu", "tests/integration"],
    ["npm", "run", "test", "--prefix", "frontend"],
    ["npm", "run", "build", "--prefix", "frontend"],
]
for command in commands:
    subprocess.run(command, cwd=ROOT, check=True)
if gpu["cuda_status"] != "available":
    print("RELEASE BLOCKED: mandatory CUDA validation requires an NVIDIA CUDA host", file=sys.stderr)
    raise SystemExit(2)
subprocess.run([sys.executable, "-m", "hotshift.cli", "data", "generate", "--config", "configs/smoke.yaml"], cwd=ROOT, check=True)
subprocess.run([sys.executable, "-m", "hotshift.cli", "data", "catalog", "--config", "configs/smoke.yaml"], cwd=ROOT, check=True)
subprocess.run([sys.executable, "-m", "hotshift.cli", "build-extension"], cwd=ROOT, check=True)
subprocess.run([sys.executable, "-m", "pytest", "tests/cuda", "-m", "cuda"], cwd=ROOT, check=True)
subprocess.run(["bash", "scripts/run_sanitizers.sh"], cwd=ROOT, check=True)
subprocess.run([sys.executable, "-m", "hotshift.cli", "bench-cuda", "--matrix", "configs/smoke.yaml", "--smoke"], cwd=ROOT, check=True)
