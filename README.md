# HotShift

HotShift is a single-GPU embedding cache and recommendation serving demonstration. A full, read-only FP32 embedding table stays in CPU memory; a bounded cache of selected rows lives in GPU memory. Pooled embedding requests (mean of a listener's history rows) are served through custom CUDA kernels while cache policies adapt as item popularity shifts. A small synthetic music catalog and a live dashboard make the behavior visible.

This is an engineering portfolio project, not a trained recommender or a production service. The catalog and listeners are fictional. The adaptive policy is a practical heuristic, not a novel algorithm, and no policy is claimed to be universally better. Performance statements appear only next to the recorded runs that support them.

![HotShift dashboard in CPU development mode, showing live API metrics](docs/dashboard.png)

## Architecture

```text
 HTTP clients ──► FastAPI (validation, 422/429/503/504) ──► bounded FIFO queue (1024)
                                                                │ microbatch: ≤64 requests,
                                                                │ ≤16384 IDs, ≤2 ms wait
                                                                ▼
                                          single engine worker thread (owns all GPU state)
   host: dedupe IDs ─► probe readback ─► pack misses (pinned) ─► policy ─► maintenance plan
   GPU:   hotshift::probe_slots ─────────► hotshift::pool_mean ─────────► scatter_rows_ + update_mapping_
                     item_to_slot[N]        cache_rows[C,D] + miss rows      (invalidate → copy → publish)
```

Custom CUDA components (all in `csrc/`): `probe_slots` (dense item→slot map lookup), `gather_rows` (validation and comparison path), `pool_mean` (fused mean pooling over cache and miss rows through descriptors), `scatter_rows_` (copies admitted rows into reserved slots), and `update_mapping_` (invalidates and publishes map entries). Transfers use standard CUDA copies through PyTorch. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the batch order, event ordering, slot lifetime protocol, and memory equations.

Cache modes: `no_cache`, `static` (fixed set from a warmup prefix), `lru`, `decayed_lfu`, and `adaptive` (decayed frequency plus a sketch-based shift detector with bounded replacement budgets). Baselines: PyTorch `embedding_bag` on CPU, and a fully GPU-resident PyTorch table (a higher-memory reference, not an equal-budget competitor).

## Status

Every milestone is implemented and validated. On a Tesla T4 (Kaggle), 157 tests pass with no CUDA skips, Compute Sanitizer finds no errors or hazards, and the full benchmark matrix, ablation, and sweeps completed with every output verified. `scripts/release_check.py` passes against that evidence.

**Main finding:** on the T4, the bounded cache uses 8× less GPU memory than a full-table copy (64.5 MB vs 512 MB) and cuts host→device traffic by up to 75%. It does not raise throughput, though: only the static cache beats no caching (+9% to +17% at batch 16–64), and the LRU, decayed-frequency, and adaptive policies are 1.2× to 4.8× slower than no caching, because their Python bookkeeping costs more than the transfers they save. The adaptive policy recovers from an abrupt shift 10× slower than LRU. Details and run IDs are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [docs/PERFORMANCE.md](docs/PERFORMANCE.md). To reproduce, run `bash scripts/gpu_evidence.sh` on an NVIDIA host (see [docs/KAGGLE.md](docs/KAGGLE.md)). [docs/STATUS.md](docs/STATUS.md) maps every requirement to its evidence, and [docs/RELEASE.md](docs/RELEASE.md) tracks the release checklist.

## Quick start (CPU development mode)

Requires Python 3.11+ and Node 22. CPU mode runs everything except the CUDA kernels and is labeled as CPU in the API and UI.

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements/cpu-py311-linux.lock   # exact pins (Linux, Python 3.11)
python3 -m pip install -e '.[dev,service]'                     # or this alone on other platforms
python3 -m hotshift.cli data generate --config configs/smoke.yaml
python3 -m hotshift.cli data catalog --config configs/smoke.yaml
python3 -m hotshift.cli test-reference --config configs/smoke.yaml
python3 -m hotshift.cli serve --config configs/smoke.yaml --backend cpu
```

In a second terminal: `npm ci --prefix frontend && npm run dev --prefix frontend`, then open http://127.0.0.1:5173. Or run both with Docker: `docker compose --profile cpu up --build`.

Reproduce a CPU replay benchmark and report:

```bash
python3 -m hotshift.cli trace generate --config configs/smoke.yaml --scenario abrupt --seed 42
python3 -m hotshift.cli bench --matrix configs/smoke.yaml --smoke
python3 -m hotshift.cli report --runs artifacts/runs
```

## GPU mode

Requirements: Linux, an NVIDIA GPU and driver, a CUDA toolkit with `nvcc`, a host compiler, and a PyTorch CUDA build matched to that toolkit. An 8 GB GPU handles the smoke and demo profiles; the standard benchmark needs about 0.6 GB of device memory plus allocator slack (see the capacity estimate from `scripts/preflight.py --config configs/benchmark.yaml`). Apple Silicon cannot run CUDA.

```bash
python3 scripts/preflight.py --backend cuda --deep      # device, toolkit, kernel run, profiler counters
python3 -m hotshift.cli build-extension
python3 -m pytest tests/cuda -m cuda
bash scripts/run_sanitizers.sh
python3 -m hotshift.cli serve --config configs/demo.yaml --backend cuda
python3 -m hotshift.cli bench-cuda --matrix configs/benchmark.yaml
```

GPU mode fails clearly when CUDA is unavailable and never falls back to CPU. Run one worker only: each worker would own a separate cache and GPU context.

## Documents

[Architecture](docs/ARCHITECTURE.md) · [Development](docs/DEVELOPMENT.md) · [Benchmarks](docs/BENCHMARKS.md) · [Performance log](docs/PERFORMANCE.md) · [Decisions](docs/DECISIONS.md) · [Status](docs/STATUS.md) · [Demo script](docs/DEMO.md) · [Release checklist](docs/RELEASE.md) · [Kaggle GPU runs](docs/KAGGLE.md)

## Credits and license

Related systems that informed the design (no code was copied): NVIDIA's [nv-embedding-cache](https://github.com/NVIDIA/nv-embedding-cache), Meta's [FBGEMM](https://docs.pytorch.org/FBGEMM/) embedding operators, and the PyTorch custom C++/CUDA operator tutorial. HotShift is not endorsed by NVIDIA, Meta, or PyTorch. The dense direct map is a simple array, not a hash table. Released under the [MIT License](LICENSE).
