from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyDecision:
    """Residency delta for one batch. Admissions are always IDs that missed in that batch."""

    admissions: tuple[int, ...]
    evictions: tuple[int, ...]

    def apply(self, resident: set[int]) -> set[int]:
        resident.difference_update(self.evictions)
        resident.update(self.admissions)
        return resident


class CachePolicy:
    """Replacement policy contract shared by the CPU and CUDA engines.

    The engine calls ``observe`` with the batch's flattened occurrences in original request
    order, then ``choose`` with the current resident set and the batch's unique misses. A
    policy keeps its own residency bookkeeping so a decision costs time proportional to the
    batch, not the cache capacity; the engine must therefore apply every returned decision.
    """

    name = "base"

    def observe(self, occurrences: Sequence[int] | np.ndarray) -> None:
        raise NotImplementedError

    def choose(self, resident: set[int], incoming: Iterable[int], capacity: int) -> PolicyDecision:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def configure(self, row_bytes: int) -> None:
        """Receive the engine's row size so byte budgets can be enforced. Optional."""

    def state(self) -> dict[str, object]:
        """Small, JSON-safe snapshot used for window logs and dashboards."""
        return {"mode": "fixed", "shift_score": None}
