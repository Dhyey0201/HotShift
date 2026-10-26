#!/usr/bin/env bash
# Compute Sanitizer on the owned kernels with small deterministic fixtures. Each tool must
# exit zero; logs are retained under artifacts/sanitizers. racecheck covers shared-memory
# hazards only; it does not prove the cache slot lifetime protocol (see event-ordering tests).
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
python3 -m hotshift.cli build-extension
# Operator and engine tests only; the API test spins up threads that add no kernel coverage.
targets="tests/cuda/test_operators.py tests/cuda/test_cache_engine.py"
status=0
for tool in memcheck initcheck synccheck racecheck; do
  echo "== $tool =="
  if ! HOTSHIFT_REQUIRE_CUDA=1 compute-sanitizer --error-exitcode 1 --tool "$tool" --target-processes all \
      python3 -m pytest $targets -m cuda -q -p no:cacheprovider -k "not repeated_runs and not pending_device_work" \
      >"artifacts/sanitizers/${tool}.log" 2>&1; then
    echo "FAILED: $tool (see artifacts/sanitizers/${tool}.log)" >&2
    status=1
  fi
  tail -n 3 "artifacts/sanitizers/${tool}.log"
done
exit $status
