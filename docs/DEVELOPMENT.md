# Development

## CPU environment

Use Python 3.11 or newer for CPU work, create a virtual environment, and install `python3 -m pip install -e '.[dev,service]'`. On a GPU host, select a Python release supported by the chosen PyTorch CUDA wheel. Generate smoke data and catalog with `python3 -m hotshift.cli data generate --config configs/smoke.yaml` and `python3 -m hotshift.cli data catalog --config configs/smoke.yaml`. Then run `python3 -m pytest tests/cpu tests/integration` and `npm install --prefix frontend && npm run test --prefix frontend && npm run build --prefix frontend`.

The app runs one worker at `127.0.0.1:8000`; Vite runs at `127.0.0.1:5173` and proxies API calls. For the CPU UI journey, run `npm run test:e2e --prefix frontend` after installing Playwright browsers with `npx playwright install chromium`. Do not expose unauthenticated state-changing controls to a public network.

## CUDA environment

Choose a compatible Linux driver, GPU, local CUDA development toolkit, host compiler, PyTorch CUDA wheel, and GPU architecture. Record exact versions before profiling. Run `python3 scripts/preflight.py --backend cuda`, build the extension before timed serving with `python3 -m hotshift.cli build-extension`, then run the CUDA tests and `bash scripts/run_sanitizers.sh`. The code uses PyTorch's current CUDA stream. The CPU planning boundary and serialized maintenance are intentional; Nsight evidence is required before claiming overlap or performance improvements.

GPU errors mark the engine unhealthy. Reproduce illegal accesses on the smallest fixture, run Compute Sanitizer, and restart the worker after a fault. The normal CPU CI workflow does not claim GPU coverage. The manual GPU workflow runs only on an explicitly configured trusted self-hosted runner.

## Artifacts

Keep generated data and formal run output under `artifacts/`. Each replay writes immutable manifests, summary data, a policy table, trace files, and checksums. Never commit personal history data or credentials. No project redistribution license is selected yet.

