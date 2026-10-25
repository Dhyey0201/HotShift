from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

SCENARIOS = {"stable", "abrupt", "gradual", "brief_burst", "alternating", "uniform"}


def _phase_offset(scenario: str, fraction: float, rows: int) -> int:
    if scenario == "abrupt" and fraction >= 0.5:
        return rows // 2
    if scenario == "brief_burst" and 0.4 <= fraction < 0.5:
        return rows // 2
    if scenario == "alternating":
        return 0 if int(fraction * 8) % 2 == 0 else rows // 2
    if scenario == "gradual":
        return int(fraction * rows / 2)
    return 0


def generate_trace(
    output_path: str | Path,
    rows: int,
    requests: int = 10000,
    history_length: int = 32,
    seed: int = 42,
    scenario: str = "stable",
    request_rate: float = 1000.0,
    warmup_fraction: float = 0.2,
) -> dict[str, object]:
    if rows <= 0 or requests <= 0 or not 0 <= history_length <= 256 or request_rate <= 0:
        raise ValueError("rows, requests, history length, or request rate is out of range")
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {', '.join(sorted(SCENARIOS))}")
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="trace-", suffix=".jsonl", dir=target.parent)
    digest = hashlib.sha256()
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, rows + 1, dtype=np.float64)
    alpha = 0.0 if scenario == "uniform" else 1.1
    weights = np.ones(rows, dtype=np.float64) if alpha == 0 else ranks ** (-alpha)
    cdf = np.cumsum(weights / weights.sum())
    cdf[-1] = 1.0
    stride_ns = int(1_000_000_000 / request_rate)
    warmup_requests = int(requests * warmup_fraction)
    with os.fdopen(fd, "wb") as stream:
        for event_index in range(requests):
            profile_id = event_index % 4
            length = history_length if profile_id != 3 else int(rng.integers(0, 257))
            sample = np.searchsorted(cdf, rng.random(length), side="right")
            fraction = event_index / max(requests - 1, 1)
            phase_id = 0 if scenario == "stable" else min(int(fraction * 2), 1)
            offset = _phase_offset(scenario, fraction, rows)
            ids = ((sample + offset) % rows).astype(np.int64).tolist()
            event = {
                "trace_version": 1,
                "event_index": event_index,
                "scheduled_offset_ns": event_index * stride_ns,
                "history_ids": ids,
                "profile_id": profile_id,
                "phase_id": phase_id,
                "warmup": event_index < warmup_requests,
            }
            encoded = (json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n").encode()
            stream.write(encoded)
            digest.update(encoded)
    Path(temporary_name).replace(target)
    manifest = {
        "trace_version": 1,
        "scenario": scenario,
        "seed": seed,
        "rows": rows,
        "requests": requests,
        "history_length": history_length,
        "history_distribution": "fixed_with_every_fourth_request_mixed_0_to_256",
        "request_rate": request_rate,
        "warmup_requests": warmup_requests,
        "trace_sha256": digest.hexdigest(),
    }
    target.with_suffix(target.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest

