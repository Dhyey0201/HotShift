from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence

from hotshift.policies.base import CachePolicy, PolicyDecision


class LRUPolicy(CachePolicy):
    name = "lru"

    def __init__(self) -> None:
        self.recency: OrderedDict[int, None] = OrderedDict()

    def observe(self, occurrences: Sequence[int]) -> None:
        for item in occurrences:
            self.recency.pop(int(item), None)
            self.recency[int(item)] = None

    def choose(self, resident: set[int], incoming: Iterable[int], capacity: int) -> PolicyDecision:
        candidates = resident | {int(value) for value in incoming}
        ordered = [item for item in self.recency if item in candidates]
        ordered.extend(sorted(candidates - set(ordered)))
        selected_order = ordered[-capacity:] if capacity else []
        selected = set(selected_order)
        admissions = tuple(item for item in selected_order if item not in resident)
        evictions = tuple(item for item in ordered if item in resident and item not in selected)
        return PolicyDecision(frozenset(selected), admissions, evictions)

    def reset(self) -> None:
        self.recency.clear()

