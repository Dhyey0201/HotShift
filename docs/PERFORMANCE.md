# Performance log

Each entry records a hypothesis, the baseline, the change, the correctness result, and the measurement. Entries without GPU measurements are marked **pending** and must not be cited as improvements. Measurements below are from a Tesla T4 at commit `f395eef` (`57cbca1` in run manifests, before the history rewrite); run IDs refer to `artifacts/runs/`.

## P1: Per-batch work proportional to cache size (code analysis)

- **Hypothesis.** The first engine did O(C) Python work per batch (free-slot list comprehension, `list.index` per eviction, rebuilding the resident set), and the policies did O(C) to O(N) work per batch (LRU walked its entire recency history; decayed LFU sorted every resident item). At C = 100,000 with batch size 1 (80,000 measured batches), that is billions of Python operations per replay, enough to dominate any kernel cost.
- **Change.** A free-slot stack, an item→slot dictionary, incremental residency, LRU tracking residents only, decayed LFU with inflated weights and a lazy min-heap, and vectorized sketch hashing.
- **Correctness.** Property tests against brute-force LRU and top-C decayed-frequency simulators (`tests/cpu/test_policies.py`); engine invariants under randomized replacement (`test_cuda_engine_host_logic.py`, `tests/cuda/test_cache_engine.py`).
- **Measurement.** The full 1M-row matrix (252 configurations × 100k requests) finished in 2 h 43 min (`20261026T013249Z-cuda-67ce68c2`), so the change made it practical. It did not make the dynamic policies cheap: at batch 64 on stable traffic, `policy_ms` averages 3.18 ms per batch for LRU, 4.45 ms for decayed LFU, and 1.59 ms for adaptive, against a 1.52 ms *total* batch time for `no_cache`. Python policy bookkeeping is now the largest cost in the engine (see P5).

## P2: Pinned staging (preserved baseline: `pageable_staging`)

- **Hypothesis.** Pageable host buffers make every copy synchronous and staged by the driver, and per-request `torch.as_tensor(..., device=...)` allocations add allocator and copy overhead.
- **Change.** Preallocated pinned host and device buffers sized for the worst case of a microbatch, with event-guarded reuse.
- **Measurement.** `20261026T041525Z-cuda-cf672a53`, abrupt, 3 seeds. Pinned staging cut the miss-transfer stage from 0.481 to 0.381 ms per batch for LRU at batch 64 (−21%), and raised throughput by 1.4% to 3.8% (LRU batch 64: 10,788 → 11,035 req/s; adaptive batch 1: 1,017 → 1,055). Seed ranges do not overlap. A real but small gain, because transfers are a small share of batch time.

## P3: Device-to-device admission (preserved baseline: `host_admission`)

- **Hypothesis.** Admitted rows were copied host→device a second time even though they had just been staged as misses.
- **Change.** The owned `scatter_rows_` kernel copies from the device miss buffer, removing `admissions × D × 4` bytes of H2D per batch (visible as `h2d_bytes` in summaries and windows).
- **Measurement.** Same run. Device-to-device admission removed 43% of LRU's H2D bytes (215 → 123 MB at batch 1; 210 → 118 MB at batch 64) and 12–14% of adaptive's, with `maintenance_ms` down 4–9%. Throughput changed by +1.0% (LRU batch 1) to −0.1% (adaptive batch 64), within noise for adaptive. Correct and cheaper on the bus, but not a throughput lever on this GPU.

## P4: Fused pooling vs gather plus library pooling

- **Hypothesis.** Fusing descriptor lookup and the mean avoids materializing `[U, D]` unique rows. The current kernel uses one thread per (bag, dimension) with 128 threads per block, so at D = 32, three quarters of each block is idle. Tuning that is deliberately deferred until measurements show the pool kernel matters next to the host boundary.
- **Measurement.** `20261026T013214Z-kernels-2a46fa6b`, 64 bags. The fused kernel is 1.1× to 2.0× faster than gather plus `embedding_bag` for bags of 1 to 32 at every D. It is slower for long bags at D = 32 (0.65× at 256) and D = 128 (0.90× at 256), and PyTorch's `embedding_bag` on a resident table beats it by about 2× at bag length 256. The profile (`20261026T013217Z-torchprof-5639e316`) shows `pool_kernel` as 79% of GPU kernel time, but GPU time is a small share of the batch: host ranges `hotshift.maintenance` (26% of CPU self time) and `hotshift.policy_update` (21%) dominate. Tuning the pool kernel is worth doing only after the host costs in P5.

## P5: Host policy and maintenance cost (open; next optimization target)

- **Finding.** Kernel time per batch is 0.24–0.35 ms; batch wall time is 1.3 ms (`static`) to 7.1 ms (decayed LFU) at batch 64. LRU and decayed LFU admit a row on almost every miss (about 500k admissions per 100k-request run), so each batch pays Python policy updates plus a maintenance transaction with two host synchronizations. As a result, no dynamic policy beats `no_cache` on the T4, where PCIe is fast and the table would fit on the device anyway.
- **Candidate changes, one at a time.** Move LRU and decayed-frequency bookkeeping into vectorized NumPy or a C++ extension. Add an admission filter (for example admit only on a second miss) to cut churn. Skip the post-maintenance synchronization when the next batch's probe already orders after it on the same stream.
- **Measurement.** Pending. Compare against `20261026T013249Z-cuda-67ce68c2` with the same seeds; the target is `lru` reaching `no_cache` throughput at batch 64.

## How to add an entry

Run the baseline and changed configurations with identical traces (same seeds, same matrix), confirm `output_check.passed` on both, and cite both run IDs with the median and range from `aggregate.json`. Change one major factor at a time. Record failed attempts and regressions here too.
