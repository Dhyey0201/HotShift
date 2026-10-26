# HotShift implementation status

Updated 2026-10-26.

## Summary

Every milestone is implemented and validated, on the CPU locally and on a Tesla T4 (Kaggle, CUDA 12.8, PyTorch 2.11) at commit `f395eef` (`57cbca1` in run manifests, before the history rewrite):

- Locally: 110 Python tests, 11 frontend unit tests, 5 Playwright tests against the real CPU API, the production frontend build, ruff, and the Docker Compose CPU stack.
- On the T4: 157 tests with `HOTSHIFT_REQUIRE_CUDA=1` and no CUDA skips. Compute Sanitizer memcheck, racecheck, initcheck, and synccheck are clean. The standard matrix, ablation, cache-fraction sweep, dimension sweeps, kernel microbenchmark, and `torch.profiler` capture are complete, with every output check passing.
- `python3 scripts/release_check.py` passes, accepting that imported evidence because `src`, `csrc`, and `tests` are unchanged since the tested commit.

The measured result is unfavorable to the dynamic policies: on the T4 they are slower than no caching (see [BENCHMARKS.md](BENCHMARKS.md) and [PERFORMANCE.md](PERFORMANCE.md) P5).

## Requirement map

| ID | Requirement | State | Evidence or remaining gate |
| --- | --- | --- | --- |
| R01 | Correct mean pooling | complete | `tests/cpu/test_pooling.py`; `tests/cuda/test_operators.py` on the T4; max abs error 0 in `20261026T013214Z-kernels-2a46fa6b` |
| R02 | Custom CUDA operations | complete | `csrc/` probe, gather, pool, scatter, mapping kernels; `tests/cuda`; clean logs in `artifacts/sanitizers/` |
| R03 | Bounded GPU cache | complete | 64.5 MB device memory for every cache mode vs 512 MB full-table reference, in every summary's `memory` |
| R04 | Safe slot reuse | complete | Maintenance protocol in ARCHITECTURE; randomized GPU invariants and racecheck/synccheck clean on the T4 |
| R05 | Adaptive behavior | complete (measured, unfavorable) | Shift detection and bounded budgets work; abrupt recovery 20,000 requests vs LRU 2,000 (`20261026T013249Z-cuda-67ce68c2`) |
| R06 | Fair comparison | complete | Shared trace checksums, identical partitions, budgets in manifests, rotated order, both baselines |
| R07 | Bounded serving | complete | `tests/cpu/test_scheduler.py`, `tests/integration`, and `tests/cuda/test_cuda_api.py` on the T4 (the latter found and fixed the backlog batching bug in `f395eef`) |
| R08 | Actual live dashboard | complete (CPU) | `npm run test:e2e --prefix frontend`; `docs/dashboard.png`. A GPU-backed recording is optional |
| R09 | Reproducible build | in_progress | Locks and the CPU image pinned by digest. `Dockerfile.cuda` not yet built or pinned |
| R10 | Performance explanation | complete | PERFORMANCE P1 to P4 measured; P5 names the host-side bottleneck and the next experiment |
| R11 | Honest reporting | complete | Claim scan; unfavorable cases lead BENCHMARKS; the invalid load probe is reported as invalid |
| R12 | Usable handoff | complete | README, ARCHITECTURE, DEVELOPMENT, BENCHMARKS, DECISIONS, PERFORMANCE, DEMO, RELEASE, KAGGLE; `artifacts/release/release_summary.json` |

## Milestones

| Milestone | State |
| --- | --- |
| 0 Repository and environment | complete |
| 1 Dataset and CPU reference | complete |
| 2 Extension and GPU operators | complete (T4) |
| 3 Bounded cache engine | complete (T4) |
| 4 Policies and traces | complete |
| 5 Benchmark foundation | complete; full GPU comparison recorded |
| 6 Serving and scheduling | complete; T4 load sweep recorded |
| 7 Music demo and dashboard | complete on CPU |
| 8 Profiling and optimization | P1 to P4 measured; P5 open |
| 9 Evidence and release | release check passes |

## Known issues

- **Service capacity is bounded by the HTTP worker, not the engine.** It sustains 250 req/s and fails at 400 req/s on the T4, about 2% of the engine's offline throughput ([BENCHMARKS.md](BENCHMARKS.md), load sweep).
- **Dynamic policies are slower than no caching on the T4.** Python policy and maintenance work dominates (PERFORMANCE P5).
- `Dockerfile.cuda` has not been built; its base image must be validated and pinned by digest.
- Nsight Compute is denied on Kaggle (`ERR_NVGPUCTRPERM`); profiling evidence is `torch.profiler` only.
