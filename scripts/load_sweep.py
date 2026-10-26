"""Open-loop offered-load sweep against a running service (blueprint 17.4).

First measures sustainable throughput with an overload trace, then offers fractions of that
measured rate. Offered loads are derived from measurement, never from invented targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotshift.benchmark.loadtest import run_loadtest
from hotshift.config import load_config
from hotshift.traffic.generate import generate_trace

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/demo.yaml")
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--requests", type=int, default=6000)
parser.add_argument("--fractions", default="0.25,0.5,0.75,1.0,1.25")
parser.add_argument("--output", default="artifacts/traces/load")
arguments = parser.parse_args()

config = load_config(arguments.config)
out = Path(arguments.output)
probe_trace = out / "capacity-probe.jsonl"
generate_trace(probe_trace, config.rows, requests=arguments.requests, history_length=32, seed=config.seed, scenario="stable", request_rate=50000.0)
probe = run_loadtest(probe_trace, arguments.url, max_in_flight=512)
summary = json.loads((probe / "summary.json").read_text())[0]
capacity = summary["successful_throughput"]
print(json.dumps({"capacity_probe_run": probe.name, "measured_successful_throughput": capacity}))
if not capacity:
    raise SystemExit("capacity probe completed no requests")
for fraction in (float(value) for value in arguments.fractions.split(",")):
    rate = capacity * fraction
    trace = out / f"stable-load{fraction:.2f}.jsonl"
    generate_trace(trace, config.rows, requests=arguments.requests, history_length=32, seed=config.seed, scenario="stable", request_rate=rate)
    run = run_loadtest(trace, arguments.url, max_in_flight=512)
    result = json.loads((run / "summary.json").read_text())[0]
    print(json.dumps({"fraction": fraction, "offered_rate": rate, "run": run.name, "successful": result["successful_requests"], "rejected": result["rejected_requests"], "p99_ms": result["client_latency_ms_p99"]}))
overload = out / "overload.jsonl"
generate_trace(overload, config.rows, requests=arguments.requests, history_length=32, seed=config.seed, scenario="overload", request_rate=capacity * 0.5)
print(json.dumps({"overload_run": run_loadtest(overload, arguments.url, max_in_flight=512).name}))
