# Benchmarks

## Current evidence

All GPU evidence below comes from one Kaggle session on a **Tesla T4** (driver 580.178.04, CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.13) at commit `4232294` (recorded in run manifests as `57cbca1`, its ID before a history rewrite), with a clean tree. Run summaries, logs, and the full generated report are committed in [evidence/t4-2026-10-26/](../evidence/t4-2026-10-26/); its [REPORT.md](../evidence/t4-2026-10-26/REPORT.md) holds every row. Large per-sample files are omitted and regenerable with `bash scripts/gpu_evidence.sh`.

| Evidence | Status | Run |
| --- | --- | --- |
| CUDA tests with `HOTSHIFT_REQUIRE_CUDA=1` | 157 passed; the 2 skips are CPU-only tests that skip when a GPU is present | `evidence/t4-2026-10-26/logs/gpu_evidence.log` |
| Compute Sanitizer (memcheck, racecheck, initcheck, synccheck) | 0 errors, 0 hazards on all 45 operator and engine tests | `evidence/t4-2026-10-26/sanitizers/` |
| Standard matrix (1M × 128, C/N = 0.10, 3 seeds, 100k requests) | Complete; every output check passed | `20261026T013249Z-cuda-67ce68c2` |
| Ablation (staging and admission baselines) | Complete | `20261026T041525Z-cuda-cf672a53` |
| Cache-fraction sweep | Complete | `20261026T043300Z-cuda-a80b131d` |
| Dimension sweeps D = 32 and D = 256 | Complete | `20261026T044419Z-cuda-70844ddc`, `20261026T044721Z-cuda-f852b250` |
| Kernel microbenchmark | Complete | `20261026T013214Z-kernels-2a46fa6b` |
| `torch.profiler` capture | Complete; Nsight Compute denied by the host (`ERR_NVGPUCTRPERM`) | `evidence/t4-2026-10-26/profile/` (`20261026T013217Z-torchprof-5639e316`) |
| Open-loop load sweep | Complete: sustains 250 req/s, fails at 400 req/s | `evidence/t4-2026-10-26/load_sweep.jsonl` (runs `20261026T060043Z` … `060646Z`; an earlier sweep used an invalid probe and is superseded) |

## Results (Tesla T4)

### Headline: on this GPU, the bounded cache saves memory but not time

The 1M × 128 FP32 table is 512 MB, which fits easily in the T4's 15.6 GB. At that size the full-table reference is the fastest option, and among the equal-budget cache modes, **no cache or a static cache beats every adaptive policy on throughput**. The cache's measured benefits are memory and transfer volume, not throughput:

- Device memory: 64.5 MB for every cache mode (51.2 MB of cached rows, 4 MB map, 9.3 MB staging) vs 512 MB for `torch_gpu_resident`.
- Host→device bytes, stable traffic, batch 16: 1.29 GB with `no_cache` vs 0.32 GB with `static` (−75%) and 0.35 GB with `lru` (−73%).

Stable traffic, warm, median of 3 seeds (requests/s; hit rate in parentheses):

| Engine | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `no_cache` | 1,459 | 16,178 | 41,108 |
| `static` (0.890) | 1,487 | **17,669** | **47,968** |
| `lru` (0.887) | 875 | 6,229 | 10,502 |
| `decayed_lfu` (0.889) | 815 | 5,477 | 8,613 |
| `adaptive` (0.889) | 1,174 | 10,438 | 20,712 |
| `torch_cpu` (reference) | 5,574 | 44,869 | 98,856 |
| `torch_gpu_resident` (reference, 8× the memory) | 4,336 | 42,414 | 98,260 |

`static` is the only cache mode faster than `no_cache`: +2% at batch 1, +9% at batch 16, +17% at batch 64. Every dynamic policy is slower than `no_cache`, from 1.2× (adaptive, batch 1) to 4.8× (decayed LFU, batch 64). The same ordering holds in the abrupt and brief-burst scenarios, at every cache fraction from 0.01 to 0.25, and at D = 32 and D = 256.

### Why: host work, not the GPU, sets the batch time

Per-batch stage means, stable, batch 64, seed 42 (ms):

| Stage | `no_cache` | `static` | `lru` | `adaptive` |
| --- | ---: | ---: | ---: | ---: |
| Policy update (Python) | 0.06 | 0.02 | **3.18** | **1.59** |
| Maintenance (host side of invalidate, scatter, publish) | 0.02 | 0.02 | **1.41** | 0.10 |
| Deduplicate + pack (host) | 0.80 | 0.63 | 0.67 | 0.66 |
| Miss transfer + pool + output copy | 0.38 | 0.37 | 0.38 | 0.39 |
| Probe (incl. readback sync) | 0.23 | 0.22 | 0.23 | 0.24 |
| **Batch wall time p50** | **1.52** | **1.30** | **5.79** | **2.76** |
| GPU kernels (probe + pool + maintenance) | 0.24 | 0.25 | 0.35 | 0.26 |

GPU kernel time is a small fraction of every batch. LRU and decayed LFU replace a row on nearly every miss (about 508,000 and 496,000 admissions per run), so their Python bookkeeping and the maintenance transaction dominate. The adaptive policy's replacement budget limits admissions to 17,408, but its sketch update still costs 1.6 ms per batch. PCIe is not the bottleneck: removing three quarters of the H2D bytes saves less time than the policy adds.

### Shift behavior (abrupt scenario, batch 16)

| Policy | Hit rate | Recovery to 90% of the post-shift reference |
| --- | ---: | --- |
| `lru` | 0.881 | 2,000 requests (all 3 seeds) |
| `decayed_lfu` | 0.883 | 2,000 requests (all 3 seeds) |
| `adaptive` | 0.823 | 20,000 requests (all 3 seeds) |
| `static` | 0.350 | not recovered |

The adaptive policy recovers **10× slower** than LRU, because its per-window replacement budget (1,024 rows in shift mode) caps how fast the hot set can move. It does not overreact to the brief burst (hit rate 0.856 vs LRU 0.876), but it does not beat LRU there either. In the uniform scenario every policy reaches the expected ≈0.10 hit rate, and the dynamic policies are 1.3× to 18.7× slower than `no_cache`.

### Kernel microbenchmark (D = 128, 64 bags, CUDA event median)

| Bag length | Fused pool (µs) | Gather + `embedding_bag` (µs) | Resident `embedding_bag` (µs) |
| ---: | ---: | ---: | ---: |
| 1 | 24.5 | 43.0 | 29.1 |
| 8 | 24.5 | 42.9 | 27.0 |
| 32 | 38.4 | 43.5 | 30.1 |
| 128 | 122.9 | 127.0 | 68.3 |
| 256 | 258.1 | 232.3 | 128.4 |

The fused kernel beats gather-then-pool for bags up to 32 (1.1× to 1.8×), the case the serving path uses. It loses at 256, and PyTorch's `embedding_bag` on a resident table is about 2× faster for long bags. Its one-thread-per-(bag, dimension) mapping leaves room for improvement. Maximum absolute error was 0 in every case.

### Load sweep (demo profile, 250k × 128, adaptive policy, one server worker)

Open-loop, stable traffic, history length 32, client on the same Kaggle host. Objective: at most 1% of requests unsuccessful and client p99 at most 50 ms.

| Offered rate (req/s) | Successful | Client p99 (ms) | Run |
| ---: | ---: | ---: | --- |
| 50 (ramp) | 1,000 / 1,000 | 8.9 | `20261026T060104Z-load-b7de9850` |
| 100 | 6,000 / 6,000 | 7.6 | `20261026T060437Z-load-66550d89` |
| 150 | 6,000 / 6,000 | 8.8 | `20261026T060518Z-load-cad5220e` |
| 200 | 6,000 / 6,000 | 9.9 | `20261026T060548Z-load-7e3dfb06` |
| 250 | 6,000 / 6,000 | 13.0 | `20261026T060612Z-load-6c140df0` |
| 400 (ramp) | 3,457 / 4,000 | 5,031 | `20261026T060136Z-load-408469bb` |
| Overload trace (10× bursts on 100) | 3,915 / 6,000 | 7,020 | `20261026T060646Z-load-a3d5c6d9` |

The service **sustains 250 req/s and fails its objective at 400 req/s**. At 400 the queue backed up until 512 requests were in flight, the client dropped 543 dispatches at that limit, and p99 reached 5 s. The client kept to its schedule (dispatch lag p99 31 ms vs 5 s of latency), so the server, not the client, saturated. No request was rejected with 429 and none timed out, even though the server's queue timeout is 2 s; the most likely reading is that the backlog formed before requests reached the bounded scheduler queue (in the HTTP worker's connection and request handling), which is therefore not protecting latency at this load.

This is about 2% of what the engine sustains offline at similar batch sizes (10,000 to 20,000 req/s), so the single Python HTTP worker (`/api/v1/embeddings/pool`: request parsing, validation, JSON encoding of 128-float vectors), not the cache engine, bounds the service. Profiling the worker is the next step before changing it. The ramp doubled from 50, so capacity lies between 250 and 400 req/s. `scripts/load_sweep.py` now bisects that bracket (3 steps by default) on its next run.

An earlier sweep (`20261026T045029Z` … `050224Z`) measured "capacity" by flooding a cold server with 6,000 requests in about 0.1 s; the client dropped 5,488 and the result (86 req/s) was invalid. It is kept only as a record.

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
- **Open-loop service replay** (`hotshift loadtest`, `scripts/load_sweep.py`) dispatches on the trace timeline. It records dispatch lag, drops dispatches when the client's in-flight limit is hit (rather than silently closing the loop), and counts 429/504/503 by cause. Offered loads are fractions of a capacity measured by a stepped ramp after a warmup.

## Experiment matrix

| Config | Purpose |
| --- | --- |
| `configs/benchmark.yaml` | N = 1M, D = 128, C/N = 0.10, history 32, batch {1, 16, 64}; stable, abrupt, brief burst, uniform; all policies plus both baselines; 3 seeds; 100k requests |
| `configs/sweep_cache.yaml` | Cache fractions 0.01, 0.05, 0.10, 0.25 |
| `configs/sweep_dim32.yaml`, `configs/sweep_dim256.yaml` | D = 32 and D = 256 |
| `configs/ablation.yaml` | Optimized engine vs preserved baselines (pageable staging, host re-copy admission) |
| `scripts/load_sweep.py` | Warmup, stepped ramp to the highest rate meeting the objective, then 0.25 to 1.25 of that rate, plus the overload trace |

## Artifacts

Each run writes `artifacts/runs/{run_id}/`: `manifest.json` (schema 2, command, Git commit and dirty flag, environment, dataset and trace checksums, budget, selected matrix, memory per engine, sample counts), `summary.json`, `aggregate.json`, `windows.csv` (trace-request windows: hits, unique hits, H2D/D2H bytes, admissions, evictions, shift score, policy mode, policy and maintenance time), `batch_samples.csv.gz`, `events.jsonl`, `traces/`, and `checksums.json`.

In offline replay a request's latency is its batch's wall time, so `batch_samples.csv.gz`, with a request count per sample, is the documented equivalent of `requests.csv`. Open-loop runs write a real `requests.csv`. JSON never contains NaN or Infinity, and a missing measurement is `null` with a reason. `hotshift report` turns completed runs into `REPORT.md` with SVG charts and an automatic **unfavorable and low-benefit cases** section.

## Limitations

- Results describe one GPU, driver, toolkit, dataset, and workload. The serialized engine has a host synchronization at probe readback and after maintenance, and it does not overlap batches.
- Smoke traces (512 requests, about 7k occurrences) never complete a 65,536-occurrence policy window, so `decayed_lfu` and `adaptive` behave like plain frequency counters there, and adaptive cannot detect a shift. Smoke runs cannot support policy or p99 claims.
- CPU-mode hit rates come from a residency model and its timings are CPU timings. Live dashboard runs are illustrative; formal comparisons come from replay.
