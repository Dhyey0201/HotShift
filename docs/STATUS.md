# HotShift implementation status

## Current milestone

Milestones 0 and 1 are in progress. The current environment is Apple Silicon with Python 3.14 and no NVIDIA tools. CUDA execution, Compute Sanitizer, Nsight profiling, and GPU benchmarks are blocked here. No CUDA results or performance claims exist.

## Implemented

- Initial Python package metadata, strict YAML configuration loader, three resource profiles, and environment example.
- Seeded chunked FP32 embedding generation, unit normalization, checksummed manifest, and read-only memory-mapped store.
- CPU mean-pooling reference contract with duplicate and empty-bag semantics and input bounds checks.
- Basic command-line generation, reference smoke command, and CPU/GPU preflight report.
- Minimal synthetic catalog generation and repository ignore rules for generated artifacts.
- CPU foundation workflow and initial architecture, development, benchmark, decisions, and demo handoff docs.

## Not started

Custom CUDA extension and owned operators; bounded cache lifecycle and policy implementations; test suites; trace generation and replay; benchmark runner and artifacts; service scheduler and API; recommendation scoring; frontend; containers; CI; sanitizer and profiling scripts; release checklist.

## Requirement map

| Requirement | State | Evidence |
| --- | --- | --- |
| R01 numerical pooling | in_progress | `src/hotshift/engine/cpu.py`; CUDA validation not started |
| R02 custom CUDA operations | blocked | No CUDA toolkit or NVIDIA GPU in this environment |
| R03 bounded GPU cache | not_started | — |
| R04 safe slot reuse | not_started | — |
| R05 adaptive behavior | not_started | — |
| R06 fair comparison | not_started | — |
| R07 bounded serving | not_started | — |
| R08 live dashboard | not_started | — |
| R09 reproducible build | in_progress | `pyproject.toml`, config and generator; clean environment not validated |
| R10 performance explanation | blocked | Requires profiled GPU evidence |
| R11 honest reporting | in_progress | No performance claims made |
| R12 usable handoff | in_progress | This status file and CPU quick start |

## Next reproducible commands

```bash
python3 -m pip install -e .
python3 scripts/preflight.py --backend cpu
python3 -m hotshift.cli data generate --config configs/smoke.yaml
python3 -m hotshift.cli test-reference --config configs/smoke.yaml
```

After the CPU dataset path is validated, continue with catalog/reference behavior and policy simulators. Complete remaining CPU, API, and UI work on this machine; run CUDA release gates on an NVIDIA Linux host.

