# Benchmarks

## Current evidence

There are no measured GPU results. CPU replay runs the numerical reference and policy simulator; cache hit rates describe simulated residency. CPU timings are not GPU kernel timings, and CPU cache policy hit rate does not imply latency improvement. The dashboard labels CPU source-miss bytes as an estimate.

## Reproducible CPU replay

Generate traces with `python3 -m hotshift.cli trace generate --config configs/smoke.yaml --scenario abrupt --seed 42 --requests 1000`. A trace records scheduled offsets independent of backend completion, event IDs, histories, phase, warmup membership, generator parameters, and SHA-256. Replay selected policies using `python3 -m hotshift.cli bench --matrix configs/benchmark.yaml --smoke`. Output directories under `artifacts/runs/{run_id}` include manifest, summary, policy CSV, trace data, and checksums.

The smoke matrix is functional evidence only. Larger CPU replays are useful for policy semantics but cannot support GPU performance or p99 claims. Static cache residency comes from the trace warmup prefix. Other policy metrics exclude warmup while retaining the prefix to initialize state.

## GPU protocol still required

Compare CPU PyTorch reference, no-cache transfer, static, LRU, decayed frequency, adaptive, and full GPU-resident reference. Use identical immutable data, traces, batch partitions, initial cache state, and declared memory budgets. Measure host end-to-end time and CUDA event spans separately; include miss planning, synchronization, policy work, maintenance, and cache preload. Report transfer payloads, device memory, successful requests, dispatch lag, errors, p50/p95/p99 sample counts, and each run. Repeat at least three times with fixed seeds and rotate policy order.

Required initial profiles cover stable, abrupt, brief-burst, and uniform traffic at microbatches 1/16/64, then a targeted capacity/dimension/load matrix. Preserve workloads in which caching loses or provides little benefit. There is no required speedup. GPU reports remain blocked until the extension executes on a real NVIDIA GPU and sanitizer findings are resolved.

