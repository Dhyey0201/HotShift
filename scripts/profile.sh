#!/usr/bin/env bash
# Nsight capture of a short CUDA replay. Nsight Systems gives the timeline (NVTX stage
# ranges: deduplicate, probe, probe_readback, pack_misses, transfer_misses, pool,
# output_copy, policy_update, maintenance). Nsight Compute samples a few owned kernels and
# needs GPU performance-counter permission. When neither is usable, run
# `python3 -m hotshift.cli profile-torch`, which traces kernels via CUPTI without counters.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/profile.sh --matrix configs/smoke.yaml [--smoke]" >&2
  exit 2
fi
if ! command -v nsys >/dev/null 2>&1 && ! command -v ncu >/dev/null 2>&1; then
  echo "BLOCKED: neither Nsight Systems (nsys) nor Nsight Compute (ncu) is installed; use hotshift profile-torch" >&2
  exit 2
fi
mkdir -p artifacts/profiles
python3 -m hotshift.cli preflight --backend cuda
python3 -m hotshift.cli build-extension
profile_name="artifacts/profiles/hotshift-$(date -u +%Y%m%dT%H%M%SZ)"
if command -v nsys >/dev/null 2>&1; then
  nsys profile --trace=cuda,nvtx,osrt --sample=none --output="$profile_name" \
    python3 -m hotshift.cli bench-cuda "$@"
  nsys stats --report cuda_gpu_kern_sum,nvtx_sum --format csv --output "$profile_name" "$profile_name.nsys-rep" || true
fi
if command -v ncu >/dev/null 2>&1; then
  # A small representative sample of owned kernels; not a whole-system timeline.
  ncu --set basic --kernel-name regex:"probe_kernel|pool_kernel|scatter_rows_kernel" --launch-count 12 \
    --force-overwrite --export="$profile_name-compute" python3 -m hotshift.cli bench-cuda "$@" \
    || echo "Nsight Compute failed; ERR_NVGPUCTRPERM means the host denies performance counters" >&2
fi
