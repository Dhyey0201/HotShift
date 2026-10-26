#!/usr/bin/env bash
# Runs every GPU release gate and benchmark on an NVIDIA host, then packages the evidence.
# Usage: bash scripts/gpu_evidence.sh [--quick]
#   --quick runs only the gates and smoke benchmarks (minutes); the default runs the full
#   standard matrix, sweeps, ablation, kernel benchmark, load sweep, and profiles (hours).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1
mkdir -p artifacts/logs
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a artifacts/logs/gpu_evidence.log; }
step() { log "RUN: $*"; "$@" 2>&1 | tee -a artifacts/logs/gpu_evidence.log; }

step $PY scripts/preflight.py --backend cuda --config configs/benchmark.yaml --deep
step $PY -m hotshift.cli data generate --config configs/smoke.yaml --if-missing
step $PY -m hotshift.cli data catalog --config configs/smoke.yaml
step $PY -m hotshift.cli build-extension
log "GATE: CPU, integration, and CUDA tests with zero allowed CUDA skips"
HOTSHIFT_REQUIRE_CUDA=1 step $PY -m pytest tests/cpu tests/integration tests/cuda -q
log "GATE: Compute Sanitizer on owned kernels"
if command -v compute-sanitizer >/dev/null 2>&1; then step bash scripts/run_sanitizers.sh; else log "BLOCKED: compute-sanitizer not installed"; fi
step $PY -m hotshift.cli bench-cuda --matrix configs/smoke.yaml --smoke
step $PY -m hotshift.cli bench-kernels --repeats 30
step $PY -m hotshift.cli profile-torch --config configs/smoke.yaml
if command -v nsys >/dev/null 2>&1 || command -v ncu >/dev/null 2>&1; then
  bash scripts/profile.sh --matrix configs/smoke.yaml --smoke 2>&1 | tee -a artifacts/logs/gpu_evidence.log || log "Nsight capture failed (permissions?); torch.profiler capture is still saved"
fi

if [[ $QUICK -eq 0 ]]; then
  step $PY -m hotshift.cli data generate --config configs/benchmark.yaml --if-missing
  step $PY -m hotshift.cli bench-cuda --matrix configs/benchmark.yaml
  step $PY -m hotshift.cli bench-cuda --matrix configs/ablation.yaml
  step $PY -m hotshift.cli bench-cuda --matrix configs/sweep_cache.yaml
  for d in 32 256; do
    step $PY -m hotshift.cli data generate --config configs/sweep_dim$d.yaml --if-missing
    step $PY -m hotshift.cli bench-cuda --matrix configs/sweep_dim$d.yaml
  done
  step $PY -m hotshift.cli data generate --config configs/demo.yaml --if-missing
  step $PY -m hotshift.cli data catalog --config configs/demo.yaml
  $PY -m hotshift.cli serve --config configs/demo.yaml --backend cuda >artifacts/logs/serve.log 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null || true' EXIT
  for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health/ready >/dev/null && break; sleep 2; done
  step $PY scripts/load_sweep.py --config configs/demo.yaml
  kill -TERM $SERVER; wait $SERVER || true
  trap - EXIT
fi

step $PY -m hotshift.cli report --runs artifacts/runs --output artifacts/runs/REPORT.md
ARCHIVE=artifacts/hotshift-gpu-evidence-$(date -u +%Y%m%dT%H%M%SZ).tar.gz
# Raw traces are regenerable from seeds and are excluded to keep the archive small.
tar --exclude='*/traces/*.jsonl' -czf "$ARCHIVE" artifacts/runs artifacts/profiles artifacts/sanitizers artifacts/logs artifacts/live 2>/dev/null || \
  tar --exclude='*/traces/*.jsonl' -czf "$ARCHIVE" artifacts/runs artifacts/logs
log "DONE: evidence archive $ARCHIVE"
