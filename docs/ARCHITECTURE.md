# Architecture

## Components and ownership

| Component | Code | Owns | Never does |
| --- | --- | --- | --- |
| API event loop | `serving/app.py` | Validation, request futures, status | Run blocking engine or GPU work |
| Batch scheduler | `serving/scheduler.py` | FIFO queue, deadlines, microbatching, exclusive commands | Touch cache arrays |
| Engine worker | `serving/worker.py` | The single thread that calls the engine | Admit concurrent callers into the cache |
| CUDA engine | `engine/cuda.py`, `engine/buffers.py`, `csrc/` | Cache tensors, CPU mirror, staging, every CUDA launch | Publish a slot before its row is copied |
| CPU engine | `engine/base.py` | Same contract with a residency model and the NumPy reference | Claim GPU performance |
| Embedding store | `data/store.py` | Immutable memory-mapped table, checksum validation | Change rows during a run |
| Policies | `policies/` | Scores and residency deltas | Mutate engine state directly |
| Metrics | `serving/metrics.py` | Counters and bounded recent samples | Synchronize the device on a scrape |
| Frontend | `frontend/src` | Display and controls | Fabricate metrics (mock data is test-only) |

## Request path

1. FastAPI validates the body (Pydantic, unknown keys rejected, at most 256 IDs, IDs in range, timeout 10 to 10000 ms) and submits it to the scheduler. Errors map to 422 (malformed), 429 with `Retry-After` (queue full), 503 (unhealthy engine, shutdown, or a running maintenance command), and 504 (queued past its deadline).
2. The scheduler takes the oldest request and keeps collecting until it reaches 64 requests or 16384 flattened IDs, or until 2 ms have passed since that oldest request was enqueued. A request that would exceed the ID limit becomes the head of the next batch, so it is not starved. Expired and cancelled requests are skipped before launch and never train a policy.
3. The worker thread runs `engine.pool(histories)`; results return to the loop through `run_in_executor`, so futures are only resolved on the loop thread. A caller that disconnects after launch has its result dropped, and the batch still completes before buffers are reused.
4. `/api/v1/recommend` normalizes the pooled vector and scores at most 512 synthetic candidates (dot product, ties by ID, seen items excluded). An empty history uses a deterministic popularity fallback; `/api/v1/embeddings/pool` returns zeros.

Policy switches and cache resets are exclusive. Admission pauses (callers get 503 `busy`), queued and in-flight work drains, the command runs on the owner thread, and `state_generation` increments. Both are refused while a live scenario runs.

## CUDA batch order

| Step | Host or device | Synchronization |
| --- | --- | --- |
| Flatten, validate, `np.unique` with inverse (sorted unique IDs) | host | — |
| Copy unique IDs (pinned → device); `hotshift::probe_slots` | device | — |
| Copy slots back; **wait on `probe_done`** | both | intentional, measured host boundary |
| Pack miss rows from the memory-mapped table into pinned staging; build `source_index` (`s ≥ 0` cache slot, `-(m+1)` miss row) | host | — |
| Copy misses, descriptors, inverse, offsets; `hotshift::pool_mean` | device | same stream |
| Copy output back; **wait on `output_done`** | both | response data ready |
| Policy `observe` (original occurrence order) and `choose` | host | — |
| Maintenance transaction; **wait on `maintenance_done`**; publish the CPU mirror | both | next probe is ordered after it |

Everything runs on the current PyTorch CUDA stream. The pipeline is serialized: no host planning overlaps device work, and no overlap is claimed. Pinned staging is reused only after the event that ends the copy reading it has completed. The CUDA event spans recorded by the benchmark include device idle time while the host plans.

## Cache representation and slot lifetime

- `item_to_slot[N]` (int32, device) maps an item to its slot, or `-1`. `cache_rows[C, D]` (FP32, device) holds the cached rows.
- The CPU mirror holds `slot_of` (item → slot), `slot_to_item[C]`, the resident set, and a free-slot stack. There are no per-slot keys on the device.
- A maintenance transaction runs only after the batch's pooling and output copy have completed. It (1) invalidates evicted items' map entries, (2) runs `scatter_rows_` to copy admitted rows into their reserved slots, and (3) publishes the admitted items' map entries. It then waits on `maintenance_done` before updating the mirror. The next batch's probe is stream-ordered after the transaction (and explicitly waits on the event), so it can never observe a published slot whose row is incomplete.
- Admitted rows are this batch's misses and are already staged on the device, so admission is a device-to-device copy. The `admit_from_miss_buffer=False` variant keeps the simpler host re-copy as a profiling baseline.
- Policies return deltas, with admissions always drawn from the batch's misses. The engine rejects duplicate admissions, evictions of non-resident items, and admissions that exceed capacity. Any such inconsistency, or any device error, marks the engine unhealthy. It never keeps serving from a partially updated cache. The cache is rebuilt from the immutable table on restart; it is not checkpointed.

Invariants (`CUDACacheEngine.check_invariants`, used in tests): the map and the mirror are a bijection, slots are in `[0, C)`, cached rows equal the immutable table rows, and free plus resident slots equal `C`.

## Data and numerical contract

The table is `[N, D]` row-major contiguous FP32 with dense IDs in `[0, N)`, written by a seeded chunked generator with unit-normalized rows, a SHA-256, and a dataset version in `manifest.json`. A batch is `ids[T]` plus `offsets[B+1]`. Duplicates count per occurrence, an empty bag pools to zeros, and an empty batch returns `[0, D]`. IDs are int64 and slots int32 (capacity must fit in int32). Kernel row offsets use 64-bit arithmetic, and kernels bounds-check every descriptor even though the host validates first. Correctness is checked against PyTorch and NumPy CPU references with `atol=1e-5, rtol=1e-4`; extreme magnitudes are checked against FP64 accumulation.

## Policy semantics

All policies observe processed occurrences in original request order and keep their own residency bookkeeping, so a decision costs time proportional to the batch size. That cost scales neither with `C` nor with `N`.

- **static**: the most frequent IDs of the trace's warmup prefix (never the evaluation suffix), preloaded and then fixed. The live service draws its warm set from a separate seeded stable-distribution prefix.
- **lru**: recency is replayed occurrence by occurrence, so the batch's final state equals exact per-occurrence LRU. Misses are served from staging and residency changes apply after pooling.
- **decayed_lfu**: windows of 65536 occurrences. At each complete window, scores are multiplied by 0.8 and the window's counts are added. Scores are stored in inflated units (`1.25^w` per occurrence in window `w`), so no O(N) decay pass is needed, and they are renormalized before FP64 overflow. The in-progress window counts at full weight, and `flush()` closes a partial window at end of run. Selection is the exact top-C of resident plus misses, using a lazy min-heap of residents and ID tie-breaking.
- **adaptive**: decayed-frequency candidates plus fast (0.5) and slow (0.95) decayed distributions over a 4096-bucket sketch (fixed 32-bit mixer, seed `0x9E3779B9`). The shift score is their total variation distance. Two consecutive windows above 0.20 enter shift mode for four windows. Replacements require the candidate's score to exceed the victim's by 10% plus epsilon, and are capped per window at `min(256, C)` rows (normal) or `min(1024, C)` rows (shift), with an equivalent byte limit. Unused allowance is discarded at each boundary, and an item evicted in a window cannot return in the same window. Filling empty slots evicts nothing and is not budgeted (see DECISIONS).

## Memory accounting

- Host: table `N·D·4`, pinned staging (`max_ids·(52 + 4D)` bytes), slot mirror `C·8`, and policy dictionaries (about 120 bytes per scored item).
- Device: cache rows `C·D·4`, map `N·4`, and staging `max_ids·(52 + 4D)` plus offsets and outputs. Probe and pool outputs come from the PyTorch caching allocator.
- Standard profile (N = 1,000,000, D = 128, C = 100,000): table 512,000,000 B (488.3 MiB), cache 51,200,000 B (48.8 MiB, an imposed 10% budget), map 4,000,000 B (3.8 MiB), staging about 9.2 MB.

`/api/v1/status` and every benchmark summary report this accounting along with the allocator's allocated, reserved, and peak bytes. Preflight refuses a configuration whose device estimate exceeds 70% of free device memory, or whose host estimate exceeds 60% of available RAM, unless `HOTSHIFT_ALLOW_OVERCAPACITY=1` is set.
