"""Open-loop service replay.

Requests are dispatched on the trace's scheduled timeline regardless of how fast earlier
responses return, so queueing, rejection, and timeouts appear instead of being hidden by
coordinated omission. Dispatch lag is recorded separately from latency. When the client's
connection limit is saturated at dispatch time, the request is dropped and counted rather
than delayed, which would silently turn the run into a closed loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from hotshift.benchmark.common import (
    SCHEMA_VERSION,
    RunDirectory,
    environment,
    git_state,
    new_run_id,
    percentile,
    ratio,
    sha256_file,
    write_csv,
    write_json,
)
from hotshift.traffic.replay import load_trace

REQUEST_COLUMNS = [
    "event_index", "phase_id", "scheduled_ms", "dispatch_lag_ms", "outcome", "status_code",
    "client_latency_ms", "send_to_response_ms", "server_queue_ms", "server_batch_engine_ms",
    "server_total_ms",
]
OUTCOMES = {200: "success", 422: "rejected_validation", 429: "rejected_queue_full", 503: "unavailable", 504: "timeout"}


async def replay_open_loop(
    client,
    events: list[dict[str, Any]],
    rate_scale: float = 1.0,
    max_in_flight: int = 256,
    timeout_ms: int = 2000,
    endpoint: str = "/api/v1/embeddings/pool",
) -> list[dict[str, Any]]:
    if rate_scale <= 0 or max_in_flight <= 0:
        raise ValueError("rate_scale and max_in_flight must be positive")
    results: list[dict[str, Any]] = []
    in_flight = 0
    tasks: list[asyncio.Task] = []
    origin = perf_counter()

    async def send(event: dict[str, Any], scheduled: float, dispatched: float) -> None:
        nonlocal in_flight
        row: dict[str, Any] = {
            "event_index": event["event_index"],
            "phase_id": event["phase_id"],
            "scheduled_ms": scheduled * 1000,
            "dispatch_lag_ms": (dispatched - scheduled) * 1000,
        }
        try:
            response = await client.post(endpoint, json={"history_ids": event["history_ids"], "timeout_ms": timeout_ms})
            finished = perf_counter() - origin
            row["status_code"] = response.status_code
            row["outcome"] = OUTCOMES.get(response.status_code, "error")
            if response.status_code == 200:
                timing = response.json().get("timing_ms", {})
                row["server_queue_ms"] = timing.get("queue")
                row["server_batch_engine_ms"] = timing.get("batch_engine")
                row["server_total_ms"] = timing.get("server_total")
        except Exception as error:  # every client failure is recorded, not raised
            finished = perf_counter() - origin
            row["status_code"] = None
            row["outcome"] = f"client_error:{type(error).__name__}"
        row["client_latency_ms"] = (finished - scheduled) * 1000
        row["send_to_response_ms"] = (finished - dispatched) * 1000
        results.append(row)
        in_flight -= 1

    for event in events:
        scheduled = event["scheduled_offset_ns"] / 1e9 / rate_scale
        delay = scheduled - (perf_counter() - origin)
        if delay > 0:
            await asyncio.sleep(delay)
        dispatched = perf_counter() - origin
        if in_flight >= max_in_flight:
            results.append({
                "event_index": event["event_index"], "phase_id": event["phase_id"],
                "scheduled_ms": scheduled * 1000, "dispatch_lag_ms": (dispatched - scheduled) * 1000,
                "outcome": "dropped_client_limit", "status_code": None,
            })
            continue
        in_flight += 1
        tasks.append(asyncio.create_task(send(event, scheduled, dispatched)))
    if tasks:
        await asyncio.gather(*tasks)
    results.sort(key=lambda row: row["event_index"])
    return results


def summarize_requests(rows: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    success = [row for row in rows if row["outcome"] == "success"]
    latency = [row["client_latency_ms"] for row in success]
    lag = [row["dispatch_lag_ms"] for row in rows]
    by_phase: dict[str, Any] = {}
    for phase in sorted({row["phase_id"] for row in rows}):
        phase_rows = [row for row in rows if row["phase_id"] == phase]
        phase_success = [row["client_latency_ms"] for row in phase_rows if row["outcome"] == "success"]
        by_phase[str(phase)] = {
            "offered": len(phase_rows),
            "successful": len(phase_success),
            "client_latency_ms_p50": percentile(phase_success, 0.50),
            "client_latency_ms_p99": percentile(phase_success, 0.99),
        }
    return {
        "offered_requests": len(rows),
        "successful_requests": len(success),
        "rejected_requests": counts.get("rejected_queue_full", 0) + counts.get("rejected_validation", 0),
        "expired_requests": counts.get("timeout", 0),
        "cancelled_requests": 0,
        "failed_requests": sum(count for outcome, count in counts.items() if outcome in {"error", "unavailable"} or outcome.startswith("client_error")),
        "dropped_dispatches": counts.get("dropped_client_limit", 0),
        "outcome_counts": counts,
        "wall_seconds": wall_seconds,
        "successful_throughput": ratio(len(success), wall_seconds),
        "client_latency_ms_p50": percentile(latency, 0.50),
        "client_latency_ms_p95": percentile(latency, 0.95),
        "client_latency_ms_p99": percentile(latency, 0.99),
        "dispatch_lag_ms_p50": percentile(lag, 0.50),
        "dispatch_lag_ms_p99": percentile(lag, 0.99),
        "dispatch_lag_ms_max": max(lag) if lag else None,
        "latency_sample_count": len(latency),
        "by_phase": by_phase,
        "latency_scope": "scheduled dispatch to received response for successful requests; dispatch lag reported separately",
    }


async def _run(trace_path: Path, url: str, rate_scale: float, max_in_flight: int, timeout_ms: int, limit: int | None):
    import httpx

    events = load_trace(trace_path)
    if limit is not None:
        events = events[:limit]
    limits = httpx.Limits(max_connections=max_in_flight, max_keepalive_connections=max_in_flight)
    async with httpx.AsyncClient(base_url=url, limits=limits, timeout=timeout_ms / 1000 + 5) as client:
        status = (await client.get("/api/v1/status")).json()
        started = perf_counter()
        rows = await replay_open_loop(client, events, rate_scale, max_in_flight, timeout_ms)
        wall = perf_counter() - started
        metrics = (await client.get("/api/v1/metrics")).json()
    return events, rows, wall, status, metrics


def run_loadtest(
    trace_path: str | Path,
    url: str = "http://127.0.0.1:8000",
    rate_scale: float = 1.0,
    max_in_flight: int = 256,
    timeout_ms: int = 2000,
    limit: int | None = None,
) -> Path:
    trace_path = Path(trace_path)
    _, rows, wall, status, metrics = asyncio.run(
        _run(trace_path, url, rate_scale, max_in_flight, timeout_ms, limit)
    )
    root = Path(os.getenv("HOTSHIFT_ARTIFACT_DIR", "artifacts")) / "runs"
    run_id = new_run_id("load")
    summary = summarize_requests(rows, wall)
    summary.update({"run_id": run_id, "scenario": _scenario(trace_path), "policy": status.get("policy"), "backend": status.get("backend")})
    with RunDirectory(root, run_id) as run:
        write_json(run.path / "manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "kind": "open_loop_service_replay",
            "backend": status.get("backend"),
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            **git_state(),
            "client_environment": {**environment(include_cuda=False), "hostname": socket.gethostname()},
            "server_status": status,
            "server_metrics_after": metrics,
            "trace_path": str(trace_path),
            "trace_sha256": sha256_file(trace_path),
            "rate_scale": rate_scale,
            "max_in_flight": max_in_flight,
            "timeout_ms": timeout_ms,
            "request_limit": limit,
            "sample_counts": {"requests": len(rows)},
            "limitations": [
                "open-loop timeline with a bounded client; dropped dispatches mean the client limit, not the server, bounded the offered load",
                "a single local client process; record client CPU saturation before claiming server capacity",
            ],
        })
        write_json(run.path / "summary.json", [summary])
        write_csv(run.path / "requests.csv", rows, REQUEST_COLUMNS)
        (run.path / "events.jsonl").write_text(
            json.dumps({"event": "loadtest_complete", "offered": len(rows)}) + "\n", encoding="utf-8"
        )
    return run.destination


def _scenario(trace_path: Path) -> str | None:
    manifest = trace_path.with_suffix(trace_path.suffix + ".manifest.json")
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8")).get("scenario")
    return None
