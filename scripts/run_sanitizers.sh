#!/usr/bin/env bash
set -euo pipefail

if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo "BLOCKED: compute-sanitizer is not installed" >&2
  exit 2
fi
if ! python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "BLOCKED: PyTorch reports no CUDA device" >&2
  exit 2
fi

mkdir -p artifacts/sanitizers
for tool in memcheck initcheck synccheck racecheck; do
  compute-sanitizer --error-exitcode 1 --tool "$tool" \
    python3 -m pytest tests/cuda -m cuda -q \
    2>&1 | tee "artifacts/sanitizers/${tool}.log"
done
