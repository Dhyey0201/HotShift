# HotShift implementation status

Updated 2026-10-25.

## Summary

Every milestone is implemented. Everything that can run without an NVIDIA GPU has been run and passes:

- 109 Python tests: 87 CPU and integration tests plus 22 CUDA-engine host-logic tests using the operator shim.
- 11 frontend unit tests and 5 Playwright tests against the real CPU API.
- The production frontend build, ruff, the CPU smoke benchmark and report, and the Docker Compose CPU stack (health checks, recommendations through the dashboard proxy, and a clean SIGTERM exit 0).

The CUDA engine was substantially rewritten after the earlier Kaggle run. Changes include pinned staging, the new `scatter_rows_` kernel, the maintenance transaction, gradient guards, and O(batch) bookkeeping. **The earlier Kaggle pass (27 tests, all four sanitizer tools clean, Tesla T4) therefore does not cover the current code.** Each GPU-dependent item below is `blocked` until `bash scripts/gpu_evidence.sh` has run on an NVIDIA host.

## Requirement map

| ID | Requirement | State | Evidence or remaining gate |
| --- | --- | --- | --- |
| R01 | Correct mean pooling | in_progress | CPU: `pytest tests/cpu/test_pooling.py` (hand fixture, Torch oracle, Hypothesis, FP64 extremes). GPU: `tests/cuda/test_operators.py` must pass on the current code |
| R02 | Custom CUDA operations | in_progress | `csrc/` probe, gather, pool, scatter, mapping kernels; `tests/cuda`; sanitizer logs under `artifacts/sanitizers` (rerun required) |
| R03 | Bounded GPU cache | in_progress | Memory accounting in `/api/v1/status` and every summary; capacity tests in the host-logic shim pass; `tests/cuda/test_cache_engine.py` (U > C, max batch, memory stability) must run on a GPU |
| R04 | Safe slot reuse | in_progress | Maintenance protocol in ARCHITECTURE; randomized invariants pass under the shim; GPU randomized, stalled-stream, and sanitizer runs pending |
| R05 | Adaptive behavior | in_progress | Policy property tests pass; recovery and detection metrics implemented; abrupt-shift GPU results pending |
| R06 | Fair comparison | in_progress | Shared trace checksums, identical partitions, budgets in manifests, rotated order, baselines; `configs/benchmark.yaml` run pending |
| R07 | Bounded serving | complete (CPU) | `pytest tests/cpu/test_scheduler.py tests/integration`: queue full, timeout, cancellation before and after launch, exclusive drain, unhealthy engine, shutdown. GPU concurrent test `tests/cuda/test_cuda_api.py` pending |
| R08 | Actual live dashboard | complete (CPU) | `npm run test:e2e --prefix frontend` with a real API; `docs/dashboard.png`. Optional GPU journey and demo recording pending |
| R09 | Reproducible build | in_progress | `requirements/cpu-py311-linux.lock`, `frontend/package-lock.json`, CPU Docker image pinned by digest and validated. `Dockerfile.cuda` digest pin pending GPU validation |
| R10 | Performance explanation | blocked | `docs/PERFORMANCE.md` hypotheses P1 to P4 with preserved baselines; needs the `ablation`, `bench-kernels`, and `profile-torch` runs |
| R11 | Honest reporting | complete | Claim scan in `scripts/release_check.py`; the report lists unfavorable cases automatically; null-with-reason measurements |
| R12 | Usable handoff | complete | README, ARCHITECTURE, DEVELOPMENT, BENCHMARKS, DECISIONS, PERFORMANCE, DEMO, RELEASE, KAGGLE; `scripts/release_check.py` writes `artifacts/release/release_summary.json` |

## Milestones

| Milestone | State |
| --- | --- |
| 0 Repository and environment | complete (deep preflight distinguishes available, unavailable, and restricted profiling) |
| 1 Dataset and CPU reference | complete |
| 2 Extension and GPU operators | implemented; GPU rerun pending |
| 3 Bounded cache engine | implemented; host logic validated; GPU rerun pending |
| 4 Policies and traces | complete |
| 5 Benchmark foundation | implemented; first full GPU comparison pending |
| 6 Serving and scheduling | complete on CPU; one GPU concurrency test pending |
| 7 Music demo and dashboard | complete on CPU |
| 8 Profiling and optimization | blocked on GPU measurements |
| 9 Evidence and release | blocked on milestones 2, 5, and 8 GPU runs |

## Next exact commands

On an NVIDIA host (Kaggle cells in [KAGGLE.md](KAGGLE.md)):

```bash
python3 -m pip install -e '.[dev,service]'
bash scripts/gpu_evidence.sh --quick      # gates: tests with zero CUDA skips, sanitizers, smoke, kernels, profile
bash scripts/gpu_evidence.sh              # full evidence: matrix, sweeps, ablation, load sweep (hours)
```

Then copy the evidence archive back, extract it into `artifacts/`, run `python3 -m hotshift.cli report --runs artifacts/runs`, record results in BENCHMARKS.md and PERFORMANCE.md, and run `python3 scripts/release_check.py`.

## Known issues

- Not yet executed on a GPU: the new `scatter_rows_` kernel, the gradient checks in `pool.cu` and `gather.cu`, and every CUDA test written for the rewritten engine. A compile or test failure there is the first thing to fix.
- `Dockerfile.cuda` has not been built. Its base image tag must be validated and pinned by digest.
- Smoke-size traces never complete a policy window, so adaptive behavior is only meaningful in full-size runs.
