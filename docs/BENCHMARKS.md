# Benchmarks

## Current evidence

| Evidence | Status |
| --- | --- |
| CPU replay matrix (policy semantics, CPU timings) | Generated locally; `hotshift bench`. Not GPU evidence. |
| CUDA correctness gates | Passed on Kaggle T4 for the **pre-rewrite** engine; the current engine must be revalidated. |
| CUDA replay matrix, sweeps, ablation, kernel microbenchmark, load sweep | Implemented; **not yet run**. One command: `bash scripts/gpu_evidence.sh`. |
| Profiler evidence | Nsight Compute denied on Kaggle (`ERR_NVGPUCTRPERM`); `hotshift profile-torch` (CUPTI tracing) is the fallback. |

No GPU performance claim is made until those runs exist. This file is updated from `artifacts/runs/REPORT.md` once they do. Every number must link to its run directory.

## Workloads

Traces are immutable JSONL (`trace_version` 2) with `event_index`, `scheduled_offset_ns`, `history_ids`, `profile_id`, `phase_id`, an optional `region`, and a `warmup` flag, plus a manifest holding the generator parameters, phase boundaries, and SHA-256. Popularity is truncated Zipf (α = 1.1) over N ranks; a seeded permutation maps ranks to IDs, and a shift moves that mapping. History length is fixed (default 32), except every fourth request draws a length uniformly from [0, 256]. The first 20% of requests are a warmup prefix.

| Scenario | Behavior | Purpose |
| --- | --- | --- |
| `stable` | One fixed skewed distribution | Steady-state benefit |
| `abrupt` | Hot set replaced at 50% | Adaptation and recovery |
| `gradual` | Mixture weight ramps 0→1 over 40–80% | Smooth drift |
| `brief_burst` | New hot set only during 40–50% | Overreaction to bursts |
| `alternating` | Hot set alternates in 8 segments | Thrashing |
| `uniform` | Equal access probability | Low-locality counterexample |
| `mixed_regions` | Four interleaved regional mappings | Shared cache pressure |
| `overload` | Arrival rate ×10 during 25–75% | Queue limits and errors (service replay) |
| `post_shift` | Stable on the post-shift mapping | Recovery reference only |

## Baselines and fairness

Equal-budget cache modes (`no_cache`, `static`, `lru`, `decayed_lfu`, `adaptive`) share the same engine, row dtype, capacity, map, staging limits, trace, batch partitions, and output contract. Two library references are included and labeled as such: `torch_cpu`, which is `embedding_bag(mode="mean")` over the host table with no GPU tier, and `torch_gpu_resident`, the whole table on the GPU. The latter is a **higher-memory reference, not an equal-budget competitor**. An external FBGEMM or NVIDIA cache baseline is optional and not integrated.

## Methodology

- **Fixed-batch offline replay** (`hotshift bench-cuda`, `hotshift bench`) isolates engine cost. A batch ends when it is full, or when the warmup flag, phase, or trace window changes, so every engine sees identical partitions.
- Before measuring, each engine runs a context, allocator, and kernel warmup and is then reset to its declared initial condition. In the `warm` condition, the trace's warmup prefix is replayed and excluded from metrics; in `cold`, everything is measured from an empty cache. `static` is preloaded from the warmup prefix only.
- Batch wall time spans the whole engine call (host planning, transfers, synchronizations, kernels, policy, maintenance). Maintenance is always inside the measured time. CUDA event spans are recorded separately and labeled. Throughput is measured requests divided by summed batch wall time.
- Outputs are verified against the CPU reference every `verify_every` batches. Any mismatch is recorded in the summary and fails the release check.
- Seeds repeat 3 times, and policy order rotates per seed. `aggregate.json` reports the median, min, and max across seeds, and `summary.json` keeps every run. Percentiles come from raw batch samples. A median of p99 values is not a confidence interval.
- **Recovery** (abrupt scenario): windows of `requests/100` trace requests. The target is the hit rate of a separate warmed run on the `post_shift` distribution with the same policy and budget. Recovery is the requests (and processing and scheduled time) from the known boundary until three consecutive windows reach 90% of that target, or `not_recovered`. Detection (the adaptive shift transition) is reported separately.
- **Open-loop service replay** (`hotshift loadtest`, `scripts/load_sweep.py`) dispatches on the trace timeline. It records dispatch lag, drops dispatches when the client's in-flight limit is hit (rather than silently closing the loop), and counts 429/504/503 by cause. Offered loads are fractions of a measured capacity probe.

## Experiment matrix

| Config | Purpose |
| --- | --- |
| `configs/benchmark.yaml` | N = 1M, D = 128, C/N = 0.10, history 32, batch {1, 16, 64}; stable, abrupt, brief burst, uniform; all policies plus both baselines; 3 seeds; 100k requests |
| `configs/sweep_cache.yaml` | Cache fractions 0.01, 0.05, 0.10, 0.25 |
| `configs/sweep_dim32.yaml`, `configs/sweep_dim256.yaml` | D = 32 and D = 256 |
| `configs/ablation.yaml` | Optimized engine vs preserved baselines (pageable staging, host re-copy admission) |
| `scripts/load_sweep.py` | Offered load at 0.25 to 1.25 of the measured capacity, plus the overload trace |

## Artifacts

Each run writes `artifacts/runs/{run_id}/`: `manifest.json` (schema 2, command, Git commit and dirty flag, environment, dataset and trace checksums, budget, selected matrix, memory per engine, sample counts), `summary.json`, `aggregate.json`, `windows.csv` (trace-request windows: hits, unique hits, H2D/D2H bytes, admissions, evictions, shift score, policy mode, policy and maintenance time), `batch_samples.csv.gz`, `events.jsonl`, `traces/`, and `checksums.json`.

In offline replay a request's latency is its batch's wall time, so `batch_samples.csv.gz`, with a request count per sample, is the documented equivalent of `requests.csv`. Open-loop runs write a real `requests.csv`. JSON never contains NaN or Infinity, and a missing measurement is `null` with a reason. `hotshift report` turns completed runs into `REPORT.md` with SVG charts and an automatic **unfavorable and low-benefit cases** section.

## Limitations

- Results describe one GPU, driver, toolkit, dataset, and workload. The serialized engine has a host synchronization at probe readback and after maintenance, and it does not overlap batches.
- Smoke traces (512 requests, about 7k occurrences) never complete a 65,536-occurrence policy window, so `decayed_lfu` and `adaptive` behave like plain frequency counters there, and adaptive cannot detect a shift. Smoke runs cannot support policy or p99 claims.
- CPU-mode hit rates come from a residency model and its timings are CPU timings. Live dashboard runs are illustrative; formal comparisons come from replay.
