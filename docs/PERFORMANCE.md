# Performance log

Each entry records a hypothesis, the baseline, the change, the correctness result, and the measurement. Entries without GPU measurements are marked **pending** and must not be cited as improvements.

## P1: Per-batch work proportional to cache size (code analysis)

- **Hypothesis.** The first engine did O(C) Python work per batch (free-slot list comprehension, `list.index` per eviction, rebuilding the resident set), and the policies did O(C) to O(N) work per batch (LRU walked its entire recency history; decayed LFU sorted every resident item). At C = 100,000 with batch size 1 (80,000 measured batches), that is billions of Python operations per replay, enough to dominate any kernel cost.
- **Change.** A free-slot stack, an item→slot dictionary, incremental residency, LRU tracking residents only, decayed LFU with inflated weights and a lazy min-heap, and vectorized sketch hashing.
- **Correctness.** Property tests against brute-force LRU and top-C decayed-frequency simulators (`tests/cpu/test_policies.py`); engine invariants under randomized replacement (`test_cuda_engine_host_logic.py`, `tests/cuda/test_cache_engine.py`).
- **Measurement.** Pending: the full matrix run time and `policy_ms`/`maintenance_ms` stage medians.

## P2: Pinned staging (preserved baseline: `pageable_staging`)

- **Hypothesis.** Pageable host buffers make every copy synchronous and staged by the driver, and per-request `torch.as_tensor(..., device=...)` allocations add allocator and copy overhead.
- **Change.** Preallocated pinned host and device buffers sized for the worst case of a microbatch, with event-guarded reuse.
- **Measurement.** Pending: `configs/ablation.yaml` (`optimized` vs `pageable_staging`, batch 1 and 64).

## P3: Device-to-device admission (preserved baseline: `host_admission`)

- **Hypothesis.** Admitted rows were copied host→device a second time even though they had just been staged as misses.
- **Change.** The owned `scatter_rows_` kernel copies from the device miss buffer, removing `admissions × D × 4` bytes of H2D per batch (visible as `h2d_bytes` in summaries and windows).
- **Measurement.** Pending: `configs/ablation.yaml` (`optimized` vs `host_admission`), comparing `h2d_bytes` and `maintenance_ms`.

## P4: Fused pooling vs gather plus library pooling

- **Hypothesis.** Fusing descriptor lookup and the mean avoids materializing `[U, D]` unique rows. The current kernel uses one thread per (bag, dimension) with 128 threads per block, so at D = 32, three quarters of each block is idle. Tuning that is deliberately deferred until measurements show the pool kernel matters next to the host boundary.
- **Measurement.** Pending: `hotshift bench-kernels` (D ∈ {32, 128, 256} × bag length ∈ {1, …, 256}), plus `profile-torch` stage totals to establish whether kernels or host planning dominate.

## How to add an entry

Run the baseline and changed configurations with identical traces (same seeds, same matrix), confirm `output_check.passed` on both, and cite both run IDs with the median and range from `aggregate.json`. Change one major factor at a time. Record failed attempts and regressions here too.
