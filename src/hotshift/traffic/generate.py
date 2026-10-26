"""Deterministic request traces.

Popularity follows a truncated Zipf distribution over N ranks, sampled by inverse-CDF search
over a CDF precomputed once per trace. A fixed seeded permutation maps ranks to item IDs.
Popularity shifts move the rank-to-ID mapping by ``rows // 2`` (or ``rows // 4`` per region);
embedding rows never change.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

SCENARIOS = (
    "stable", "abrupt", "gradual", "brief_burst", "alternating", "uniform",
    "mixed_regions", "overload", "post_shift",
)
REGIONS = ("north", "south", "east", "west")
ZIPF_ALPHA = 1.1
OVERLOAD_RATE_MULTIPLIER = 10.0


class TrafficSampler:
    """Draws histories for one scenario; shared by offline traces and live service scenarios."""

    def __init__(self, rows: int, scenario: str, seed: int) -> None:
        if rows <= 0:
            raise ValueError("rows must be positive")
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {', '.join(SCENARIOS)}")
        self.rows = rows
        self.scenario = scenario
        self.seed = seed
        self.alpha = 0.0 if scenario == "uniform" else ZIPF_ALPHA
        ranks = np.arange(1, rows + 1, dtype=np.float64)
        weights = np.ones(rows, dtype=np.float64) if self.alpha == 0 else ranks ** (-self.alpha)
        self.cdf = np.cumsum(weights / weights.sum())
        self.cdf[-1] = 1.0
        self.permutation = np.random.default_rng([seed, 1]).permutation(rows).astype(np.int64)
        self.shift = rows // 2

    def phase(self, fraction: float, event_index: int = 0) -> tuple[int, float]:
        """Return (phase_id, probability that an occurrence uses the shifted mapping)."""
        scenario = self.scenario
        if scenario == "abrupt":
            return (1, 1.0) if fraction >= 0.5 else (0, 0.0)
        if scenario == "brief_burst":
            if fraction < 0.4:
                return 0, 0.0
            return (1, 1.0) if fraction < 0.5 else (2, 0.0)
        if scenario == "alternating":
            segment = min(int(fraction * 8), 7)
            return segment, float(segment % 2)
        if scenario == "gradual":
            if fraction < 0.4:
                return 0, 0.0
            if fraction < 0.8:
                return 1, (fraction - 0.4) / 0.4
            return 2, 1.0
        if scenario == "overload":
            if fraction < 0.25:
                return 0, 0.0
            return (1, 0.0) if fraction < 0.75 else (2, 0.0)
        if scenario == "post_shift":
            return 0, 1.0
        return 0, 0.0

    def draw(self, rng: np.random.Generator, length: int, fraction: float, event_index: int) -> tuple[list[int], int, str | None]:
        phase_id, shifted = self.phase(fraction, event_index)
        ranks = np.searchsorted(self.cdf, rng.random(length), side="right")
        region = None
        if self.scenario == "mixed_regions":
            region_index = event_index % len(REGIONS)
            offset: np.ndarray | int = region_index * (self.rows // len(REGIONS))
            region = REGIONS[region_index]
        elif 0.0 < shifted < 1.0:
            offset = np.where(rng.random(length) < shifted, self.shift, 0)
        else:
            offset = self.shift if shifted >= 1.0 else 0
        ids = self.permutation[(ranks + offset) % self.rows]
        return ids.tolist(), phase_id, region


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
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    sampler = TrafficSampler(rows, scenario, seed)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="trace-", suffix=".jsonl", dir=target.parent)
    digest = hashlib.sha256()
    rng = np.random.default_rng([seed, 0])
    base_stride_ns = 1_000_000_000 / request_rate
    warmup_requests = int(requests * warmup_fraction)
    boundaries: list[dict[str, int]] = []
    scheduled_ns = 0.0
    last_phase: int | None = None
    try:
        with os.fdopen(fd, "wb") as stream:
            for event_index in range(requests):
                fraction = event_index / requests
                profile_id = event_index % 4
                length = history_length if profile_id != 3 else int(rng.integers(0, 257))
                ids, phase_id, region = sampler.draw(rng, length, fraction, event_index)
                if phase_id != last_phase:
                    boundaries.append({"phase_id": phase_id, "first_event_index": event_index})
                    last_phase = phase_id
                event: dict[str, object] = {
                    "trace_version": 2,
                    "event_index": event_index,
                    "scheduled_offset_ns": int(scheduled_ns),
                    "history_ids": ids,
                    "profile_id": profile_id,
                    "phase_id": phase_id,
                    "warmup": event_index < warmup_requests,
                }
                if region is not None:
                    event["region"] = region
                stride = base_stride_ns
                if scenario == "overload" and phase_id == 1:
                    stride /= OVERLOAD_RATE_MULTIPLIER
                scheduled_ns += stride
                encoded = (json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n").encode()
                stream.write(encoded)
                digest.update(encoded)
        Path(temporary_name).replace(target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    manifest = {
        "trace_version": 2,
        "scenario": scenario,
        "seed": seed,
        "rows": rows,
        "requests": requests,
        "history_length": history_length,
        "history_distribution": "fixed length, except every fourth request (profile_id 3) draws a uniform length in [0, 256]",
        "popularity": {
            "distribution": "truncated_zipf" if sampler.alpha else "uniform",
            "alpha": sampler.alpha,
            "sampler": "inverse CDF via searchsorted over a precomputed CDF",
            "rank_to_id": "numpy default_rng([seed, 1]).permutation(rows)",
            "shift_offset_ranks": sampler.shift,
        },
        "request_rate": request_rate,
        "overload_rate_multiplier": OVERLOAD_RATE_MULTIPLIER if scenario == "overload" else None,
        "warmup_requests": warmup_requests,
        "phase_boundaries": boundaries,
        "trace_sha256": digest.hexdigest(),
    }
    target.with_suffix(target.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest
