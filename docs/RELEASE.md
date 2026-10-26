# Release checklist

Blueprint section 28. `python3 scripts/release_check.py` automates the checks marked ⚙ and writes `artifacts/release/release_summary.json` (commit, dirty flag, environment, results, remaining optional work). On a host without CUDA it accepts imported evidence from `scripts/gpu_evidence.sh` (the step log with every step passed, four clean sanitizer logs, and a verified full CUDA matrix whose commit has the same `src`, `csrc`, and `tests` as `HEAD`); without that evidence it exits 2 (BLOCKED).

| Item | State | How it is verified |
| --- | --- | --- |
| Clean setup builds the extension and frontend with documented dependencies | ✅ | ⚙ frontend build; lockfile; `build-extension` on a Kaggle T4 (CUDA 12.8, PyTorch 2.11) |
| CPU mode works and is labeled; CUDA mode exercised on a real device | ✅ | Playwright CPU journey; `tests/cuda/test_cuda_api.py` on the T4 |
| All numerical, cache, scheduler, API gates pass; zero skipped GPU tests | ✅ | ⚙ `pytest`; `HOTSHIFT_REQUIRE_CUDA=1`: 157 passed on the T4, no CUDA skips |
| Owned-kernel sanitizer findings resolved | ✅ (none found) | ⚙ `scripts/run_sanitizers.sh` (logs retained) |
| Reset and policy switching cannot race active reads | ✅ | Exclusive drain on the owner thread; `test_exclusive_drains_queue_and_blocks_admission`, `test_reset_and_policy_switch_during_queued_work_drain_first` |
| Host, GPU, staging, metadata memory bounded and reported | ✅ | Preflight estimate; `memory` in status and summaries; `test_memory_report_accounts_for_every_tensor` |
| Requests cannot exceed capacities or inject artifact paths | ✅ | 422 tests; run ID regex and resolved-path checks |
| Stable, abrupt, brief-burst, uniform experiments complete | ✅ (`20261026T013249Z-cuda-67ce68c2`) | ⚙ release check looks for a full non-smoke CUDA run |
| Comparisons use identical trace checksums, semantics, budgets | ✅ (by construction) | Manifest trace hashes; dashboard compatibility check |
| Every performance number traces to raw artifacts | ✅ | BENCHMARKS and PERFORMANCE cite a run ID for every number |
| At least one limitation and one losing or low-benefit workload | ✅ | BENCHMARKS headline: dynamic policies lose to `no_cache` on the T4; uniform traffic; adaptive recovers 10× slower than LRU |
| Dashboard uses real metrics with units and timing scope | ✅ | Playwright; scope labels in the API |
| Shutdown drains or cancels safely; interrupted artifacts marked | ✅ | `test_graceful_shutdown_cancels_work_beyond_the_drain_timeout`; `test_interrupted_run_is_marked_incomplete`; Docker SIGTERM exit 0 |
| Docs current | ✅ | ⚙ required-file check and claim scan |
| Release summary with commit, commands, hardware, remaining work | ⚙ | `artifacts/release/release_summary.json` |

Open items outside the checklist: the load sweep's capacity measurement needs a rerun (`gpu_evidence.sh --load`), and `Dockerfile.cuda` has not been built or pinned.

Legend: ✅ done, ⏳ waiting on a GPU run.
