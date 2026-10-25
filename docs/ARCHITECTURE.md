# Architecture

## Request path

The FastAPI layer validates bounded histories and submits them to a FIFO scheduler. A single scheduler task groups requests by count, flattened ID limit, and a 2 ms default batch window. Engine calls run on one dedicated executor thread. Queue overflow returns HTTP 429, queued expiry returns 504, invalid input returns 422, and an unhealthy engine returns 503. Cancellation skips queued work; launched work completes before its buffers may be reused.

The CUDA engine flattens histories and computes sorted unique IDs on the CPU, probes the GPU map with `hotshift::probe_slots`, copies probe results to the host for miss planning, copies unique misses into reusable GPU staging, and calls the fused `hotshift::pool_mean` kernel. It copies pooled output back and performs serialized replacement on the same current stream. The current implementation synchronizes output and maintenance boundaries; it does not claim compute/transfer overlap. CPU mode uses the same input contract and the NumPy reference path.

## Data contract

The embedding table is a contiguous row-major `[N,D]` FP32 matrix, dense item IDs `[0,N)`, immutable for the run. A request batch has flattened IDs and offsets `[B+1]`. Duplicates count once per occurrence, and empty histories yield zero vectors. The generator uses a seeded NumPy generator, normalizes each row, writes `.npy`, and records a checksum and dataset version in `manifest.json`. `EmbeddingStore` memory maps the file read-only and validates metadata, checksum, and finiteness.

The demo catalog contains at most 4096 synthetic candidate items. Catalog vectors create fictional genre similarity and are not trained recommendation embeddings or quality evidence.

## Cache and policy ownership

The engine owns residency and all GPU mutation. Each resident ID maps to one slot; slots identify one ID. Pooling completes before replacement. Maintenance copies rows before publishing mappings and then synchronizes. A kernel failure makes the engine unhealthy; it never silently changes from CUDA to CPU.

CPU policies include static warmup, LRU based on occurrence order, decayed frequency windows, and a 4096-bucket adaptive frequency sketch. Adaptive shift detection uses total variation distance, two consecutive threshold windows, a four-window elevated admission budget, and stable ID tie-breaking. These are engineering baselines, not established optimal policies. CPU simulator results do not demonstrate that GPU cache latencies improve.

## Memory accounting

- Table: `N * D * 4` host bytes.
- Cache rows: `C * D * 4` device bytes.
- Dense item map: `N * 4` device bytes.
- Slot keys and metadata: approximately `C * 8` device bytes.
- Miss staging: `max_batch_ids * D * 4` device bytes plus bounded host packing.
- Descriptors, inverse indices, offsets, output buffers, candidates, and allocator reserves add further capacity-dependent overhead.

Preflight estimates device and host allocations and applies 70% free-device and 60% available-host limits. `HOTSHIFT_ALLOW_OVERCAPACITY=1` is an explicit local override; use it only after reviewing the full estimate.

