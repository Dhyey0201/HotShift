"""Open-loop offered-load sweep against a running service (blueprint 17.4).

Finds sustainable throughput with a stepped ramp, then offers fractions of that measured rate.
Offered loads are derived from measurement, never from invented targets.

The ramp warms the server first, then doubles the offered rate until a step misses its
service objective: more than 1% of requests rejected, expired, failed, or dropped, or a p99
above ``--slo-ms``. A step that meets the objective while its client dispatch lag p99 exceeds
``--max-lag-ms`` stops the ramp as client-limited: the client, not the server, set the load.
After a miss, ``--refine-steps`` bisections narrow the bracket between passing and failing rates.
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
parser.add_argument("--start-rate", type=float, default=50.0)
parser.add_argument("--max-rate", type=float, default=6400.0)
parser.add_argument("--slo-ms", type=float, default=50.0)
parser.add_argument("--max-lag-ms", type=float, default=20.0)
parser.add_argument("--refine-steps", type=int, default=3)
parser.add_argument("--output", default="artifacts/traces/load")
arguments = parser.parse_args()

config = load_config(arguments.config)
out = Path(arguments.output)


def offer(name: str, rate: float, requests: int, scenario: str = "stable") -> dict:
    trace = out / f"{name}.jsonl"
    generate_trace(trace, config.rows, requests=requests, history_length=32, seed=config.seed, scenario=scenario, request_rate=rate)
    run = run_loadtest(trace, arguments.url, max_in_flight=512)
    result = json.loads((run / "summary.json").read_text())[0]
    unsuccessful = result["offered_requests"] - result["successful_requests"]
    return {
        "run": run.name, "offered_rate": rate, "offered": result["offered_requests"], "successful": result["successful_requests"],
        "unsuccessful_fraction": unsuccessful / max(result["offered_requests"], 1), "rejected": result["rejected_requests"],
        "dropped": result["dropped_dispatches"], "p99_ms": result["client_latency_ms_p99"], "lag_p99_ms": result["dispatch_lag_ms_p99"],
    }


# Warmup: first requests pay connection setup, allocator growth, and lazy CUDA initialization.
print(json.dumps({"warmup": offer("warmup", arguments.start_rate, 500)}))
capacity = None
stop_reason = f"reached --max-rate {arguments.max_rate:g} without missing the objective"
rate = arguments.start_rate
while rate <= arguments.max_rate:
    step = offer(f"ramp-{rate:g}", rate, int(min(max(rate * 10, 1000), 20000)))
    print(json.dumps({"ramp": step}))
    if step["unsuccessful_fraction"] > 0.01 or step["p99_ms"] is None or step["p99_ms"] > arguments.slo_ms:
        stop_reason = f"objective missed at {rate:g} req/s"
        break
    # The objective held, but the client fell behind its schedule, so the offered rate is unproven.
    if step["lag_p99_ms"] is not None and step["lag_p99_ms"] > arguments.max_lag_ms:
        stop_reason = f"client-limited at {rate:g} req/s (dispatch lag p99 {step['lag_p99_ms']:.1f} ms)"
        break
    capacity = rate
    rate *= 2
# Doubling only brackets capacity; bisect between the last passing and first failing rate.
failing = rate if stop_reason.startswith("objective missed") else None
for _ in range(arguments.refine_steps if capacity is not None and failing is not None else 0):
    middle = (capacity + failing) / 2
    step = offer(f"refine-{middle:g}", middle, int(min(max(middle * 10, 1000), 20000)))
    print(json.dumps({"refine": step}))
    if step["unsuccessful_fraction"] > 0.01 or step["p99_ms"] is None or step["p99_ms"] > arguments.slo_ms:
        failing = middle
    elif step["lag_p99_ms"] is not None and step["lag_p99_ms"] > arguments.max_lag_ms:
        break
    else:
        capacity = middle
if failing is not None:
    stop_reason += f"; capacity lies between {capacity:g} and {failing:g} req/s"
print(json.dumps({"measured_capacity_rate": capacity, "slo_ms": arguments.slo_ms, "stop_reason": stop_reason}))
if capacity is None:
    raise SystemExit("the starting rate already missed the objective; lower --start-rate")
for fraction in (float(value) for value in arguments.fractions.split(",")):
    print(json.dumps({"fraction": fraction, **offer(f"stable-load{fraction:.2f}", capacity * fraction, arguments.requests)}))
print(json.dumps({"overload": offer("overload", capacity * 0.5, arguments.requests, scenario="overload")}))
