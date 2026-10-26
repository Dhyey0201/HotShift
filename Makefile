.PHONY: setup-check data-smoke test-cpu test-cuda lint api ui bench-smoke bench-cuda-smoke bench-gpu-evidence sanitize profile release-check

PYTHON ?= python3

setup-check:
	$(PYTHON) scripts/preflight.py --backend cpu

data-smoke:
	$(PYTHON) -m hotshift.cli data generate --config configs/smoke.yaml --if-missing
	$(PYTHON) -m hotshift.cli data catalog --config configs/smoke.yaml
	$(PYTHON) -m hotshift.cli test-reference --config configs/smoke.yaml

lint:
	$(PYTHON) -m ruff check src tests scripts

test-cpu:
	$(PYTHON) -m pytest tests/cpu tests/integration -m 'cpu or integration'

test-cuda:
	HOTSHIFT_REQUIRE_CUDA=1 $(PYTHON) -m pytest tests/cuda -m cuda

api:
	$(PYTHON) -m hotshift.cli serve --config configs/smoke.yaml --backend cpu

ui:
	npm run dev --prefix frontend

bench-smoke:
	$(PYTHON) -m hotshift.cli bench --matrix configs/smoke.yaml --smoke

bench-cuda-smoke:
	$(PYTHON) -m hotshift.cli bench-cuda --matrix configs/smoke.yaml --smoke

# Full GPU evidence set: standard matrix, sweeps, kernels, profiler capture, report.
bench-gpu-evidence:
	bash scripts/gpu_evidence.sh

sanitize:
	bash scripts/run_sanitizers.sh

profile:
	bash scripts/profile.sh --matrix configs/smoke.yaml --smoke

release-check:
	$(PYTHON) scripts/release_check.py
