from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class HotShiftConfig:
    profile: str
    rows: int
    dimensions: int
    seed: int
    output_dir: Path
    cache_capacity: int
    policy: str
    max_history_ids: int
    max_batch_requests: int
    max_batch_ids: int
    queue_capacity: int
    max_batch_delay_ms: float

    @property
    def table_bytes(self) -> int:
        return self.rows * self.dimensions * 4

    @property
    def cache_bytes(self) -> int:
        return self.cache_capacity * self.dimensions * 4


def _require_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    unexpected = set(value) - expected
    missing = expected - set(value)
    if unexpected or missing:
        details = []
        if unexpected:
            details.append(f"unknown keys: {', '.join(sorted(unexpected))}")
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        raise ValueError(f"invalid {path}: {'; '.join(details)}")


def load_config(path: str | Path) -> HotShiftConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a YAML mapping")

    unexpected = set(raw) - {"profile", "data", "cache", "limits", "benchmark"}
    missing = {"profile", "data", "cache", "limits"} - set(raw)
    if unexpected or missing:
        details = []
        if unexpected:
            details.append(f"unknown keys: {', '.join(sorted(unexpected))}")
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        raise ValueError(f"invalid configuration: {'; '.join(details)}")
    for key in ("data", "cache", "limits"):
        if not isinstance(raw[key], dict):
            raise TypeError(f"{key} must be a YAML mapping")
    _require_keys(raw["data"], {"rows", "dimensions", "seed", "output_dir"}, "data")
    _require_keys(raw["cache"], {"capacity", "policy"}, "cache")
    _require_keys(
        raw["limits"],
        {
            "max_history_ids",
            "max_batch_requests",
            "max_batch_ids",
            "queue_capacity",
            "max_batch_delay_ms",
        },
        "limits",
    )
    if "benchmark" in raw:
        if not isinstance(raw["benchmark"], dict):
            raise ValueError("benchmark must be a YAML mapping")
        extra_benchmark_keys = set(raw["benchmark"]) - {
            "scenarios", "policies", "microbatch_sizes", "history_length", "requests", "repeats"
        }
        if extra_benchmark_keys:
            raise ValueError(f"invalid benchmark: unknown keys: {', '.join(sorted(extra_benchmark_keys))}")
        benchmark = raw["benchmark"]
        for name in ("scenarios", "policies", "microbatch_sizes"):
            if name in benchmark and (not isinstance(benchmark[name], list) or not benchmark[name]):
                raise ValueError(f"benchmark.{name} must be a non-empty list")
        if set(benchmark.get("scenarios", [])) - {
            "stable", "abrupt", "gradual", "brief_burst", "alternating", "uniform"
        }:
            raise ValueError("benchmark.scenarios contains an unsupported scenario")
        if set(benchmark.get("policies", [])) - {
            "no_cache", "static", "lru", "decayed_lfu", "adaptive"
        }:
            raise ValueError("benchmark.policies contains an unsupported policy")
        if any(int(size) <= 0 or int(size) > 64 for size in benchmark.get("microbatch_sizes", [])):
            raise ValueError("benchmark microbatch sizes must be between 1 and 64")
        if int(benchmark.get("requests", 1)) <= 0 or not 1 <= int(benchmark.get("repeats", 1)) <= 10:
            raise ValueError("benchmark requests must be positive and repeats must be between 1 and 10")
        if not 0 <= int(benchmark.get("history_length", 0)) <= 256:
            raise ValueError("benchmark.history_length must be between 0 and 256")

    data = raw["data"]
    cache = raw["cache"]
    limits = raw["limits"]
    result = HotShiftConfig(
        profile=str(raw["profile"]),
        rows=int(data["rows"]),
        dimensions=int(data["dimensions"]),
        seed=int(data["seed"]),
        output_dir=Path(data["output_dir"]),
        cache_capacity=int(cache["capacity"]),
        policy=str(cache["policy"]),
        max_history_ids=int(limits["max_history_ids"]),
        max_batch_requests=int(limits["max_batch_requests"]),
        max_batch_ids=int(limits["max_batch_ids"]),
        queue_capacity=int(limits["queue_capacity"]),
        max_batch_delay_ms=float(limits["max_batch_delay_ms"]),
    )
    if result.rows <= 0 or result.dimensions <= 0:
        raise ValueError("data.rows and data.dimensions must be positive")
    if result.rows >= 2**31 or result.dimensions > 4096:
        raise ValueError("data rows or dimensions exceed supported bounds")
    if not 0 < result.cache_capacity <= result.rows:
        raise ValueError("cache.capacity must be between 1 and data.rows")
    if result.policy not in {"no_cache", "static", "lru", "decayed_lfu", "adaptive"}:
        raise ValueError("cache.policy must be no_cache, static, lru, decayed_lfu, or adaptive")
    if min(
        result.max_history_ids,
        result.max_batch_requests,
        result.max_batch_ids,
        result.queue_capacity,
    ) <= 0:
        raise ValueError("request and queue limits must be positive")
    if result.max_batch_delay_ms <= 0:
        raise ValueError("limits.max_batch_delay_ms must be positive")
    if result.max_history_ids > 256 or result.max_batch_requests > 64 or result.max_batch_ids > 16384 or result.max_batch_delay_ms > 100:
        raise ValueError("request and batch limits exceed supported bounds")
    return result

