# Development

## Environments

| Environment | Python | PyTorch | CUDA | Status |
| --- | --- | --- | --- | --- |
| Local development (Apple Silicon) | 3.14.3 | 2.14.1 CPU | none | CPU, API, UI, Docker CPU validated |
| CPU lock / CI / Docker | 3.11 | CPU wheel in CI only | none | `requirements/cpu-py311-linux.lock` |
| Kaggle GPU | 3.13.15 | 2.11.0+cu128 | toolkit 12.8.93, Tesla T4 | all GPU gates and benchmarks ([evidence](../evidence/t4-2026-10-26/)) |

The lock (`requirements/cpu-py311-linux.lock`) pins the CPU service and test stack for Linux, Python 3.11, generated with `uv pip compile pyproject.toml --extra dev --extra service --python-version 3.11 --python-platform x86_64-manylinux_2_28`. PyTorch is deliberately not locked: install the build that matches the host's CUDA toolkit and driver first, then the project. Every benchmark manifest records the resolved Python, NumPy, PyTorch, CUDA, driver, `nvcc`, and GPU versions.

## CPU setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e '.[dev,service]'
python3 -m pip install torch                       # optional on CPU: enables the torch baselines and oracle tests
make data-smoke                                    # generate, catalog, reference check
make test-cpu                                      # tests/cpu + tests/integration
npm ci --prefix frontend
npm run test --prefix frontend && npm run build --prefix frontend
npx --prefix frontend playwright install chromium && npm run test:e2e --prefix frontend
```

The API runs one worker at `127.0.0.1:8000`. Vite runs at `127.0.0.1:5173` and proxies only `/api` and `/health`. The Playwright config starts both, creates a CPU smoke replay so the comparison view loads a real artifact, and uses `HOTSHIFT_PYTHON` (default `.venv/bin/python`). Docker: `docker compose --profile cpu up --build` runs the API (Python 3.11 image pinned by digest) and the dashboard, bound to localhost.

## Test layout and markers

- `tests/cpu` (marker `cpu`): the reference, Hypothesis property tests of every policy against brute-force simulators, traces, replay, recovery, the benchmark runner, the scheduler with an injected fake engine, and `test_cuda_engine_host_logic.py`. That last file runs the real `CUDACacheEngine` bookkeeping against a CPU operator shim. It validates planning and maintenance logic only, not kernels or synchronization.
- `tests/integration` (marker `integration`): the API end to end, including 429/504/503 paths, startup failures, scenarios, artifact path safety, and an open-loop overload replay.
- `tests/cuda` (marker `cuda`): owned operators, input validation, the engine lifecycle against the reference, event ordering under a stalled stream, memory stability, baselines, and the CUDA-backed API with concurrent batching.

CUDA tests skip without a device, and pytest then prints `GPU SKIP SUMMARY ... this is NOT a GPU pass`. Set `HOTSHIFT_REQUIRE_CUDA=1` (as `make test-cuda`, the release check, and the evidence script do) to turn any CUDA skip into a failure.

## CUDA workflow

```bash
python3 scripts/preflight.py --backend cuda --deep   # runs one owned kernel; reports counter profiling as available/restricted/unavailable
python3 -m hotshift.cli build-extension              # JIT build via torch.utils.cpp_extension + ninja, cached per torch version
HOTSHIFT_REQUIRE_CUDA=1 python3 -m pytest tests/cuda
bash scripts/run_sanitizers.sh                       # memcheck, initcheck, synccheck, racecheck; logs in artifacts/sanitizers
bash scripts/gpu_evidence.sh --quick                 # every gate plus smoke benchmarks; drop --quick for the full evidence set
```

Build the extension before any timed run; the CUDA Docker image compiles it at build time (`build-extension --no-device-check`). If compilation fails, check the PyTorch CUDA version against `nvcc --version`, the host compiler, `TORCH_CUDA_ARCH_LIST`, and stale builds under `~/.cache/torch_extensions`. Never download prebuilt binaries from unknown sources.

If a device fault occurs, the engine marks itself unhealthy and `/health/ready` returns 503. Restart the process, reproduce on the smallest fixture, and run memcheck. `CUDA_LAUNCH_BLOCKING=1` is for debugging only and must never be set in a published run.

## Profiling

- Nsight Systems (`scripts/profile.sh`) gives the timeline. NVTX ranges mark each stage: `hotshift.deduplicate`, `probe`, `probe_readback`, `pack_misses`, `transfer_misses`, `pool`, `output_copy`, `policy_update`, `maintenance`.
- Nsight Compute samples `probe_kernel`, `pool_kernel`, and `scatter_rows_kernel`, and needs GPU performance-counter permission. Kaggle denies it (`ERR_NVGPUCTRPERM`); preflight `--deep` reports this as `restricted`.
- `python3 -m hotshift.cli profile-torch` replays a short trace under `torch.profiler`. CUPTI activity tracing needs no counter permission, and it exports a Chrome trace, a key-averages table, and per-stage CPU and device totals under `artifacts/profiles/`.
- `python3 -m hotshift.cli bench-kernels` microbenchmarks the fused pool against gather plus `embedding_bag` and a resident `embedding_bag`, across D ∈ {32, 128, 256} and bag lengths {1, 8, 32, 128, 256}.

Profiler-instrumented numbers are diagnostic; headline numbers come from uninstrumented `bench-cuda` runs. Record each optimization attempt in [PERFORMANCE.md](PERFORMANCE.md).

## Operations

- Configuration is strict YAML (unknown keys fail startup). Environment variables: see `.env.example`. `HOTSHIFT_SHUTDOWN_TIMEOUT` bounds the queue drain on shutdown.
- On SIGTERM, uvicorn stops accepting connections. The app then cancels any live scenario (persisting it as `interrupted`), drains queued work within the timeout, cancels whatever remains, lets an already-launched batch finish, and closes the engine. Uvicorn's graceful timeout (`serve --shutdown-timeout`, default 10 s) bounds the whole sequence; when it expires, open connections are closed. Benchmark runs write to a hidden temporary directory, and an interrupted run is renamed with an `INCOMPLETE` marker that the API, report, and release check all ignore.
- Logs carry backend, generation, batch size, outcome, and failure cause. Per-batch logs are DEBUG (`--log-level DEBUG`), and nothing is logged per token.
- Demo controls are unauthenticated and bound to localhost. A public deployment would need authentication for state-changing routes, which is out of scope.
