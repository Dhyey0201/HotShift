# HotShift

HotShift is an engineering demonstration of serving pooled embedding requests from an immutable CPU table and a bounded GPU cache. The first implementation milestone provides deterministic synthetic data and a CPU reference path; custom CUDA operators, bounded serving, benchmarks, and the dashboard are still in progress.

## Current status

This checkout is being developed on Apple Silicon without `nvidia-smi` or `nvcc`. CPU development can proceed here. CUDA execution, sanitizer runs, profiling, and GPU performance evidence are blocked until a suitable Linux NVIDIA environment is available. No performance results are claimed.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 scripts/preflight.py --backend cpu
python3 -m hotshift.cli data generate --config configs/smoke.yaml
python3 -m hotshift.cli test-reference --config configs/smoke.yaml
```

The default dataset is fictional and generated locally. Embedding tables and generated artifacts are excluded from Git. See [docs/STATUS.md](docs/STATUS.md) for implemented work and the next command.

## Architecture plan

The target request path validates bounded histories, batches requests, probes a custom GPU item-to-slot map, stages unique cache misses from host memory, and runs owned CUDA pooling. Cache reads and serialized maintenance share one engine owner. A CPU backend will provide the same mean-pooling contract for development and comparisons. The music catalog is synthetic and demonstrates systems behavior; it does not represent learned recommendation quality.

## Hardware and scope

CUDA development requires a compatible Linux host, NVIDIA driver and GPU, CUDA toolkit with `nvcc`, and a PyTorch build matched to that toolkit. Apple Silicon supports the CPU development path only. The target release includes static, LRU, decayed-frequency, and adaptive policies, bounded FastAPI serving, a React dashboard, reproducible traces, and raw benchmark artifacts. GPU claims require a real device and measured evidence.

