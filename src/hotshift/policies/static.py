from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from hotshift.policies.base import CachePolicy, PolicyDecision


class StaticPolicy(CachePolicy):
    """Keeps the explicit warmup set fixed and never admits evaluation misses."""

    name = "static"

    def __init__(self, warm_items: Iterable[int] = ()) -> None:
        self.warm_items = tuple(dict.fromkeys(int(item) for item in warm_items))

    def observe(self, occurrences: Sequence[int] | np.ndarray) -> None:
        return None

    def choose(self, resident: set[int], incoming: Iterable[int], capacity: int) -> PolicyDecision:
        return PolicyDecision((), ())

    def reset(self) -> None:
        return None
