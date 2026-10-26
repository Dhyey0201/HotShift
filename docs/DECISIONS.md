# Engineering decisions

Material design choices and deviations from the original plan. Each lists its reason and the tests that cover it.

## Environment and scope

- **Development host without a GPU.** CPU foundations were built first on Apple Silicon, and the GPU gates run on Kaggle or a GPU runner. CUDA tests skip loudly locally and fail under `HOTSHIFT_REQUIRE_CUDA=1`.
- **License: MIT**, chosen by the project owner (October 2026). No third-party implementation code is included.
- **C++ standard follows PyTorch.** The extension builds with C++17 unless the installed PyTorch headers require C++20 (PyTorch 2.12+), which `cxx_standard()` detects from `torch/all.h`. NVCC needs CUDA 12 or newer for C++20. `ops.cpp` was syntax-checked locally against PyTorch 2.14 headers under C++20; the `.cu` files need `nvcc` and were not compiled locally.
- **CPU lock for Python 3.11 Linux only.** PyTorch is not locked because the wheel must match the host's CUDA toolkit; manifests record the resolved versions instead.

## Engine

- **One stream, explicit events.** The engine uses PyTorch's current CUDA stream for transfers and kernels, with `probe_done`, `output_done`, and `maintenance_done` events for the host waits, instead of separate compute and transfer streams. The serialized design gains nothing from a second stream until batches overlap, and one stream removes cross-stream allocator lifetime hazards. Overlap is future work and must be proven in Nsight first. Tests: `test_pending_device_work_delays_host_planning_not_correctness`.
- **Synchronous maintenance completion.** The engine waits on `maintenance_done` before publishing the CPU mirror, so the mirror never describes a transaction that has not finished. The cost is one extra sync per batch that changes residency, measured in `maintenance_ms`.
- **Slot metadata on the host.** `slot_to_item[C]` and the free-slot stack live only in the CPU mirror; the device holds `item_to_slot[N]` and `cache_rows`. Nothing on the device needs slot keys in this design.
- **Device-to-device admission.** Admitted rows are always this batch's misses, already staged on the device, so a custom `scatter_rows_` kernel copies them into slots. `admit_from_miss_buffer=False` keeps the host re-copy as a measurable baseline (`configs/ablation.yaml`). Pinned staging similarly has a `pinned_staging=False` pageable baseline.
- **Policies return deltas and track their own residency.** This makes a batch cost O(batch) instead of O(C) or O(N). The previous version scanned all slots, all of LRU's history, and every resident score on every batch, which would make the 1M-row matrix impractically slow. The engine validates every delta, and an invalid one marks it unhealthy. Tests: `test_policies.py`, `test_cuda_engine_host_logic.py`.
- **Validation errors are not faults.** Input rejected before device work (a `ValueError`) leaves the engine healthy. Every other failure marks it unhealthy with no backend fallback.
- **Defensive kernel bounds.** Kernels bounds-check descriptors even though the host validates first. An inconsistent descriptor yields zeros instead of an out-of-bounds read, and it is caught by the output verification.

## Policies

- **Decayed frequency uses inflated weights.** Each occurrence adds `(1/0.8)^w`, which is exactly equivalent to multiplying all scores by 0.8 at each window boundary but needs no O(N) pass; scores renormalize before overflow. A partial window counts at full weight (as if it completed now), and `flush()` closes it at end of run.
- **Adaptive empty-slot fills are unbudgeted.** The planned per-window budget (256 normal, 1024 shift) is applied to replacements. Filling an empty slot evicts nothing and copies a row already on the device, while budgeting fills would leave a 100k-row cache cold for hundreds of windows. Row and byte limits both apply to replacements.
- **Static warm set in the live service** comes from a seeded 2000-request prefix of the stable distribution, separate from any live traffic.

## Traffic and measurement

- **Trace version 2.** Ranks map to IDs through a seeded permutation (instead of contiguous ID offsets, which would favor CPU cache locality), gradual drift is a probability mixture, and the generator adds `mixed_regions`, `overload`, and the `post_shift` reference. Older traces and runs are schema 1; the report and dashboard skip them.
- **Batch partitions split at warmup, phase, and window boundaries.** This keeps windows exact and makes partitions identical across engines.
- **Recovery reference** is a half-length `post_shift` trace with a 50% warmup, replayed with the same policy, variant, budget, and batch size. Using half the length bounds the extra cost.
- **Request-level samples.** For offline replay, `batch_samples.csv.gz` (with per-sample request counts) substitutes for `requests.csv`; open-loop runs write `requests.csv`.
- **Open-loop client drops** a dispatch when its in-flight limit is reached, rather than delaying it, and reports the count.
- **Profiling fallback.** Kaggle denies GPU performance counters, so `hotshift profile-torch` (torch.profiler with CUPTI activity tracing) supplies timeline and kernel-duration evidence, plus `bench-kernels` for operator comparisons. Nsight remains the first choice where permitted.

## Serving and UI

- **Recommendation scoring stays on the CPU** (NumPy, at most 512 candidates) for both backends, and is timed separately as `scoring`. Running it on the GPU would mean either a second GPU owner outside the engine thread or routing scoring through the engine; neither fits the single-owner design, and scoring is outside the cache experiment.
- **Busy responses.** While a reset or policy switch drains work, requests get 503 with `busy: true` and `Retry-After: 1`, and a second concurrent command is refused.
- **Live metrics** add a 5-second rolling window, since cumulative hit rates hide shifts. Percentiles are exact over at most 4096 recent batch samples, and both scopes are labeled in the API and UI.
