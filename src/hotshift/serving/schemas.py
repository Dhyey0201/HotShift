from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PoolRequest(StrictModel):
    history_ids: list[int] = Field(max_length=256)
    timeout_ms: int = Field(default=2000, ge=10, le=10000)


class RecommendRequest(PoolRequest):
    top_k: int = Field(default=10, ge=1, le=100)
    exclude_seen: bool = True


class PolicyRequest(StrictModel):
    policy: Literal["no_cache", "static", "lru", "decayed_lfu", "adaptive"]


class ScenarioStartRequest(StrictModel):
    scenario: Literal["stable", "viral_release", "gradual", "brief_burst", "uniform"]
    duration_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    request_rate: float = Field(default=10.0, ge=0.1, le=100.0)
    history_length: int = Field(default=32, ge=0, le=256)
    seed: int = 42

