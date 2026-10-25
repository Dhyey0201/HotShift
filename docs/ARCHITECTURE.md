# Architecture

## Target execution path

The API validates IDs, history length, and deadlines, then submits requests to a bounded FIFO scheduler. One engine owner combines a microbatch, computes sorted unique IDs and inverse positions, probes the GPU direct map with a custom operator, and synchronizes the probe result for host miss planning. Unique misses are packed from the immutable CPU table into bounded staging buffers. Custom pooling consumes cache rows and miss rows; results return to the API worker. Policy updates and cache replacement happen only after the active pool has completed.

## Data contract

The table is a contiguous row-major `[N,D]` FP32 matrix indexed by dense IDs in `[0,N)`. Histories use flattened IDs and `[B+1]` offsets. Each output is the mean of its bag, duplicates count as separate occurrences, and empty bags produce zero vectors. `EmbeddingStore` memory maps the generated table read-only and checks shape, dtype, layout, finiteness, and SHA-256 against `manifest.json`.

## Ownership and cache safety

The API owns request validation and futures; the scheduler owns queue order and deadlines; one engine worker owns all device operations and cache mutation. A cache mapping is published only after its row is ready. A slot is not reused until the prior batch has completed. CUDA failure makes the engine unhealthy; a partially mutated cache is never served.

## Capacity accounting

Table payload is `N * D * 4` bytes; cache row payload is `C * D * 4` bytes; the direct item map adds `N * 4` bytes. Miss rows, slot metadata, descriptors, inverse indices, outputs, candidate data, pinned staging, and allocator reserve must be added before selecting a profile. These values are planning estimates until measured on the target device.

