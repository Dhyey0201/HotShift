# HotShift implementation status

## Environment and milestone

CPU product path and CUDA source are implemented. Local CPU/API, dashboard unit/build, and browser validation pass on Apple Silicon with Python 3.14. This host has no `nvidia-smi`/`nvcc`. CUDA compilation, execution, sanitizers, Nsight profiling, and GPU benchmarks remain blocked on an NVIDIA CUDA host.

## Implemented

- Strict YAML profiles, package metadata, environment sample, preflight estimates, and local CPU/GPU status reporting.
- Deterministic chunked FP32 embeddings, manifests/checksums, immutable store, and bounded synthetic catalog.
- NumPy CPU pooling reference and PyTorch CPU oracle; CPU cache simulator with static warmup, LRU, decayed LFU, and adaptive sketch policy.
- Deterministic stable/shift/burst/alternating/uniform traces and offline replay with persisted manifests, summaries, tables, and checksums; artifacts are discoverable from the API and dashboard comparison view.
- FastAPI schemas/routes, bounded batching, queue limits, deadlines, cancellation, exclusive controls, recommendations, scenarios, status, and metrics.
- React/Vite dashboard with backend badge, real metrics polling, music demo, scenario/policy controls, and charts.
- C++/CUDA custom probe, gather, fused pooling, mapping update kernels, and a Python CUDA cache engine.
- CPU, integration, CUDA, frontend unit, and Playwright test sources; CI workflows; CPU Docker Compose; sanitizer/profile/release scripts.

## Not yet proven or incomplete

- No CUDA compiler/device is available locally. CUDA source has not been compiled; kernel behavior, cache lifetime, sanitizer, and GPU performance gates are unverified.
- CUDA kernels are uncompiled and unexecuted. Nsight captures and GPU benchmark evidence do not exist.
- Docker image and Compose flow have not been validated in this environment.
- External baseline adapters and a license decision remain outstanding. A license must be selected by the project owner before code reuse is authorized.
- External baseline adapters and a license decision remain outstanding. Do not claim a complete release until GPU gates pass and the owner chooses a project license.

## Requirement map

| ID | Requirement | State | Evidence/remaining gate |
| --- | --- | --- | --- |
| R01 | Correct mean pooling | partial | CPU reference and API pooling pass; CUDA parity unrun |
| R02 | Custom CUDA operations | blocked | `csrc/` implemented; compile and execute on NVIDIA host |
| R03 | Bounded GPU cache | blocked | `src/hotshift/engine/cuda.py`; GPU capacity/residency tests required |
| R04 | Safe slot reuse | blocked | Serialized implementation; event/sanitizer stress unrun |
| R05 | Adaptive behavior | implemented on CPU | Policy and deterministic shift tests pass; GPU deployment unverified |
| R06 | Fair comparison | partial | Deterministic CPU replay and run artifacts; formal GPU matrix absent |
| R07 | Bounded serving | implemented on CPU | API integration tests pass |
| R08 | Live dashboard | implemented | Frontend tests/build and Playwright journey pass |
| R09 | Reproducible build | partial | Python/frontend dependencies and CPU validation pass; Docker and CUDA builds unverified |
| R10 | Performance explanation | blocked | Requires actual Nsight captures and measurements |
| R11 | Honest reporting | implemented | CPU/GPU limits and unverified GPU evidence are called out |
| R12 | Usable handoff | partial | Guides and commands present; owner license decision and CUDA release gate remain |

## Validation performed

- `pytest tests/cpu tests/integration -q`: 16 passed.
- `ruff check src tests scripts`: passed.
- `npm run test --prefix frontend`: 2 passed; `npm run build --prefix frontend`: passed (Vite reports a large bundle warning).
- `npm run test:e2e --prefix frontend`: 1 passed.
- CPU smoke benchmark generated a run directory under `artifacts/runs/` with manifest, summary, trace, CSVs, events, and checksums.
- CUDA preflight is blocked because the host lacks an NVIDIA GPU, CUDA toolkit, and CUDA-enabled PyTorch runtime.

## Next commands

```bash
python3 -m pip install -e '.[dev,service]'
python3 -m hotshift.cli data generate --config configs/smoke.yaml
python3 -m hotshift.cli data catalog --config configs/smoke.yaml
python3 -m pytest tests/cpu tests/integration
npm install --prefix frontend
npm run test --prefix frontend && npm run build --prefix frontend
```

Run `python3 scripts/release_check.py` on the owner’s supported environment. On an NVIDIA Linux machine install its matched PyTorch CUDA build, run preflight, build the extension, execute CUDA tests and sanitizers, then collect the required benchmark matrix. No assistant attribution has been added.
