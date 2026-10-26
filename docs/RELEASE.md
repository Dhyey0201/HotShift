# Release checklist

Blueprint section 28. `python3 scripts/release_check.py` automates the checks marked ⚙ and writes `artifacts/release/release_summary.json` (commit, dirty flag, environment, results, remaining optional work). It exits 2 (BLOCKED) on a host without CUDA.

| Item | State | How it is verified |
| --- | --- | --- |
| Clean setup builds the extension and frontend with documented dependencies | CPU ✅ / GPU ⏳ | ⚙ frontend build; lockfile; `build-extension` on a GPU host |
| CPU mode works and is labeled; CUDA mode exercised on a real device | CPU ✅ / GPU ⏳ | Playwright CPU journey; `tests/cuda/test_cuda_api.py` |
| All numerical, cache, scheduler, API gates pass; zero skipped GPU tests | CPU ✅ / GPU ⏳ | ⚙ `pytest`; ⚙ `HOTSHIFT_REQUIRE_CUDA=1 pytest tests/cuda` |
| Owned-kernel sanitizer findings resolved | ⏳ | ⚙ `scripts/run_sanitizers.sh` (logs retained) |
| Reset and policy switching cannot race active reads | ✅ | Exclusive drain on the owner thread; `test_exclusive_drains_queue_and_blocks_admission`, `test_reset_and_policy_switch_during_queued_work_drain_first` |
| Host, GPU, staging, metadata memory bounded and reported | ✅ (GPU numbers ⏳) | Preflight estimate; `memory` in status and summaries; `test_memory_report_accounts_for_every_tensor` |
| Requests cannot exceed capacities or inject artifact paths | ✅ | 422 tests; run ID regex and resolved-path checks |
| Stable, abrupt, brief-burst, uniform experiments complete | ⏳ | ⚙ release check looks for a full non-smoke CUDA run |
| Comparisons use identical trace checksums, semantics, budgets | ✅ (by construction) | Manifest trace hashes; dashboard compatibility check |
| Every performance number traces to raw artifacts | ✅ | Report links each run; no numbers in docs yet |
| At least one limitation and one losing or low-benefit workload | ✅ / ⏳ | BENCHMARKS limitations; report's unfavorable-case section (GPU results pending) |
| Dashboard uses real metrics with units and timing scope | ✅ | Playwright; scope labels in the API |
| Shutdown drains or cancels safely; interrupted artifacts marked | ✅ | `test_graceful_shutdown_cancels_work_beyond_the_drain_timeout`; `test_interrupted_run_is_marked_incomplete`; Docker SIGTERM exit 0 |
| Docs current | ✅ | ⚙ required-file check and claim scan |
| Release summary with commit, commands, hardware, remaining work | ⚙ | `artifacts/release/release_summary.json` |

Legend: ✅ done, ⏳ waiting on a GPU run.
