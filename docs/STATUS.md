# HotShift implementation status

## Environment and milestone

CPU product path and CUDA source are implemented. Local CPU/API, dashboard unit/build, and browser validation pass on Apple Silicon with Python 3.14. The Kaggle GPU run used a Tesla T4 runtime with Python 3.13.15, PyTorch 2.11.0+cu128, and CUDA toolkit 12.8.93. The extension built; all 27 CPU/API/CUDA tests passed; Compute Sanitizer memcheck, initcheck, synccheck, and racecheck each passed all 10 CUDA tests with zero reported errors/hazards. A CUDA smoke benchmark artifact was generated. The local Apple Silicon host still has no `nvidia-smi`/`nvcc`; Nsight Systems is unavailable in the Kaggle image, though Nsight Compute (`ncu`) is present.

## Implemented

- Strict YAML profiles, package metadata, environment sample, preflight estimates, and local CPU/GPU status reporting.
- Deterministic chunked FP32 embeddings, manifests/checksums, immutable store, and bounded synthetic catalog.
- NumPy CPU pooling reference and PyTorch CPU oracle; CPU cache simulator with static warmup, LRU, decayed LFU, and adaptive sketch policy.
- Deterministic stable/shift/burst/alternating/uniform traces and offline replay with persisted manifests, summaries, tables, and checksums; artifacts are discoverable from the API and dashboard comparison view.
- FastAPI schemas/routes, bounded batching, queue limits, deadlines, cancellation, exclusive controls, recommendations, scenarios, status, and metrics.
- React/Vite dashboard with backend badge, real metrics polling, music demo, scenario/policy controls, and charts.
- C++/CUDA custom probe, gather, fused pooling, mapping update kernels, and a Python CUDA cache engine.
- CUDA cache engine static preloading, reset, cache lifecycle/API tests, and a repeated CUDA benchmark runner are implemented; GPU-only additions await rerun on Kaggle.
- CPU, integration, CUDA, frontend unit, and Playwright test sources; CI workflows; CPU Docker Compose; sanitizer/profile/release scripts.

## Not yet proven or incomplete

- Only the CUDA smoke benchmark has been run; the full repeated scenario/policy/batch matrix and performance analysis remain.
- Nsight Systems profiling has not been run because `nsys` is absent from the Kaggle image. Nsight Compute is present but no capture has been collected.
- Docker image and Compose flow have not been validated in this environment.
- External baseline adapters and a license decision remain outstanding. A license must be selected by the project owner before code reuse is authorized.

## Requirement map

| ID | Requirement | State | Evidence/remaining gate |
| --- | --- | --- | --- |
| R01 | Correct mean pooling | validated for tested operator cases | CPU reference plus CUDA pooling tests passed on Kaggle, including awkward dimensions and empty histories |
| R02 | Custom CUDA operations | validated for tested operators | Extension build succeeded; 6 CUDA tests passed; Compute Sanitizer reported zero errors/hazards |
| R03 | Bounded GPU cache | validated for tested cases | Cache lifecycle, static preload/reset, no-cache, and eviction tests passed on Kaggle |
| R04 | Safe slot reuse | validated for tested cases | Cache replacement tests passed under all four Compute Sanitizer tools; broader long-running stress remains |
| R05 | Adaptive behavior | implemented on CPU | Policy and deterministic shift tests pass; GPU deployment unverified |
| R06 | Fair comparison | partial | CPU/GPU replay runners and repeated trace matrix implemented; GPU matrix and additional reference baselines remain |
| R07 | Bounded serving | validated on CPU and CUDA smoke path | CPU integration and CUDA-backed API tests passed |
| R08 | Live dashboard | implemented | Frontend tests/build and Playwright journey pass |
| R09 | Reproducible build | partial | CPU/frontend validation and CUDA extension build pass; Docker image/Compose execution remains unverified |
| R10 | Performance explanation | blocked | Requires full repeated GPU measurements and a profiler capture; `nsys` unavailable in current Kaggle image |
| R11 | Honest reporting | implemented | CPU/GPU limits and unverified GPU evidence are called out |
| R12 | Usable handoff | partial | Guides and commands present; owner license decision and CUDA release gate remain |

## Validation performed

- `pytest tests/cpu tests/integration -q`: 17 passed on the latest local run.
- `ruff check src tests scripts`: passed.
- `npm run test --prefix frontend`: 2 passed; `npm run build --prefix frontend`: passed (Vite reports a large bundle warning).
- `npm run test:e2e --prefix frontend`: 1 passed.
- CPU smoke benchmark generated a run directory under `artifacts/runs/` with manifest, summary, trace, CSVs, events, and checksums.
- User-reported Kaggle validation: Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, CUDA toolkit 12.8.93; extension build succeeded; all 27 tests passed; each sanitizer tool passed 10 CUDA tests with zero errors/hazards; CUDA smoke benchmark artifact `20261025T050501Z-cuda-cc1c7439` was generated. `ncu` is available; `nsys` is not.
- Local follow-up: 17 CPU/API tests pass; 3 CUDA modules are skipped locally because PyTorch/CUDA is unavailable. Ruff, Python compile checks, CPU smoke benchmark, dashboard build, and Playwright pass.
- Local preflight remains blocked because this Apple Silicon host lacks an NVIDIA GPU, CUDA toolkit, and CUDA-enabled PyTorch runtime.

## Next commands

```bash
python3 -m pip install -e '.[dev,service]'
python3 -m hotshift.cli data generate --config configs/smoke.yaml
python3 -m hotshift.cli data catalog --config configs/smoke.yaml
python3 -m pytest tests/cpu tests/integration
npm install --prefix frontend
npm run test --prefix frontend && npm run build --prefix frontend
```

Next, inspect the generated smoke artifact, run `python3 -m hotshift.cli bench-cuda --matrix configs/benchmark.yaml` for the full repeated matrix, and capture a GPU profile with Nsight Systems or Compute. Run the Docker/Compose flow and resolve the project license choice before calling this a complete reusable public release. No assistant attribution has been added.
