# Benchmarks

## Current evidence

CUDA operators have been built and tested on a user-provided Kaggle runtime, including Compute Sanitizer. No GPU benchmark results have been collected yet. CPU replay runs the numerical reference and policy simulator; CPU timings are not GPU timings, and hit rate alone does not establish latency improvement. The dashboard labels CPU source-miss bytes as an estimate.

## Reproducible CPU replay

Generate traces with `python3 -m hotshift.cli trace generate --config configs/smoke.yaml --scenario abrupt --seed 42 --requests 1000`. A trace records scheduled offsets independent of backend completion, event IDs, histories, phase, warmup membership, generator parameters, and SHA-256. Replay selected policies using `python3 -m hotshift.cli bench --matrix configs/benchmark.yaml --smoke`. Output directories under `artifacts/runs/{run_id}` include manifest, summary, policy CSV, trace data, and checksums.

The smoke matrix is functional evidence only. Larger CPU replays are useful for policy semantics but cannot support GPU performance or p99 claims. Static cache residency comes from the trace warmup prefix. Other policy metrics exclude warmup while retaining the prefix to initialize state.

## GPU protocol still required

The CUDA runner is `python3 -m hotshift.cli bench-cuda --matrix configs/benchmark.yaml`; use `--smoke` first. It records deterministic trace hashes, repeated seeds, rotated policy order, cache hit rates, host batch wall time, CUDA event spans, miss payload bytes, peak device allocations, and GPU/toolchain metadata. Its request throughput divides measured requests by the sum of measured batch wall times, excluding trace file parsing and gaps between batches; it describes serialized active-batch capacity, not offered-load service throughput. `bash scripts/profile.sh --matrix configs/smoke.yaml --smoke` uses Nsight Systems when installed, or falls back to a bounded 12-kernel Nsight Compute capture. Nsight Compute reports kernel metrics, not a whole-system timeline.

Validate the runner and artifact schema on Kaggle before starting a full matrix. The initial runner compares no-cache, static, LRU, decayed-LFU, and adaptive cache modes. A separately implemented CPU or full GPU-resident reference baseline and service load benchmark remain follow-up work. Use identical immutable data, traces, batch partitions, initial cache state, and declared memory budgets when adding baselines. CUDA event intervals are not pure kernel time: they can include device idle gaps, while batch wall time includes host planning, transfer, synchronization, kernels, and policy maintenance. The implementation is serialized and does not overlap host planning with GPU work.

Required initial profiles cover stable, abrupt, brief-burst, and uniform traffic at microbatches 1/16/64 with at least three seeds, then a targeted capacity/dimension/load matrix. Preserve workloads in which caching loses or provides little benefit. There is no required speedup. Do not publish performance claims until smoke and full matrices are reviewed, correctness is maintained, and Nsight captures explain the measured behavior.

