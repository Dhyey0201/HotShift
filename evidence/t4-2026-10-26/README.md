# T4 evidence, 2026-10-26

Results behind the numbers in [BENCHMARKS.md](../../docs/BENCHMARKS.md) and [PERFORMANCE.md](../../docs/PERFORMANCE.md). Produced on a Kaggle Tesla T4 by `bash scripts/gpu_evidence.sh` (full run) and `bash scripts/gpu_evidence.sh --load` (load sweep). Hardware, software, and commit details are in [provenance.json](provenance.json).

| Path | Contents |
| --- | --- |
| [REPORT.md](REPORT.md) | Generated report: every configuration's median, range, p99, H2D bytes, recovery, output check, and unfavorable cases, with charts |
| `runs/<run_id>/` | `manifest.json` (command, commit, environment, dataset and trace checksums), `summary.json` (one row per configuration and seed), `aggregate.json` (medians across seeds) |
| `logs/gpu_evidence.log` | Full step log: preflight, build, 157 tests with `HOTSHIFT_REQUIRE_CUDA=1`, sanitizers, benchmarks |
| `sanitizers/` | Compute Sanitizer memcheck, racecheck, initcheck, synccheck logs (all clean) |
| `profile/` | `torch.profiler` key averages and stage summary (Nsight Compute is denied on Kaggle) |
| `load_sweep.jsonl` | One line per load-sweep step, transcribed from the console output of the `--load` run |

| Run | What it measures |
| --- | --- |
| `20261026T013249Z-cuda-67ce68c2` | Standard matrix: 1M × 128, C/N = 0.10, 4 scenarios, 3 batch sizes, 5 policies + 2 references, 3 seeds |
| `20261026T041525Z-cuda-cf672a53` | Ablation: pinned vs pageable staging, device vs host admission |
| `20261026T043300Z-cuda-a80b131d` | Cache fractions 0.01, 0.05, 0.10, 0.25 |
| `20261026T044419Z-cuda-70844ddc`, `20261026T044721Z-cuda-f852b250` | D = 32 and D = 256 |
| `20261026T013214Z-kernels-2a46fa6b` | Fused pool vs gather + `embedding_bag` microbenchmark |

Omitted to keep the repository small, and regenerable from the seeds in each manifest: traces, `windows.csv`, and `batch_samples.csv.gz` (about 230 MB). The per-request files of the load sweep were not copied off the Kaggle session.

`python3 scripts/release_check.py` validates this folder: the log must report every step passed, the sanitizer logs must be clean, the standard matrix must verify every output, and `src`, `csrc`, and `tests` must match the tested commit.
