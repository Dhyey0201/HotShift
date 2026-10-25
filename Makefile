.PHONY: setup-check data-smoke test-cpu test-cuda api ui bench-smoke sanitize release-check

setup-check:
	python3 scripts/preflight.py --backend cpu

data-smoke:
	python3 -m hotshift.cli data generate --config configs/smoke.yaml
	python3 -m hotshift.cli test-reference --config configs/smoke.yaml

test-cpu:
	python3 -m pytest tests/cpu tests/integration -m 'cpu or integration'

test-cuda:
	python3 -m pytest tests/cuda -m cuda

api:
	python3 -m hotshift.cli serve --config configs/demo.yaml --backend cpu

ui:
	cd frontend && npm run dev

bench-smoke:
	python3 -m hotshift.cli bench --matrix configs/benchmark.yaml --smoke

sanitize:
	bash scripts/run_sanitizers.sh

release-check:
	python3 scripts/release_check.py

