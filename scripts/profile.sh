#!/usr/bin/env bash
set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
  echo "BLOCKED: Nsight Systems (nsys) is not installed" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/profile.sh --matrix configs/benchmark.yaml [--smoke]" >&2
  exit 2
fi
mkdir -p artifacts/profiles
python3 -m hotshift.cli preflight --backend cuda
python3 -m hotshift.cli build-extension
profile_name="artifacts/profiles/hotshift-$(date -u +%Y%m%dT%H%M%SZ)"
nsys profile --trace=cuda,nvtx,osrt --output="$profile_name" \
  python3 -m hotshift.cli bench-cuda "$@"
