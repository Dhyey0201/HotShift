from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence

import numpy as np

from hotshift.policies.base import CachePolicy, PolicyDecision


class LRUPolicy(CachePolicy):
    """Least-recently-used residency driven by original request occurrence order.

    Recency is replayed occurrence by occurrence, so the final state after a batch equals
    an exact per-occurrence LRU over that batch. The only batching approximation is timing:
    admissions take effect after pooling, and a batch's own misses are served from staging.
    """

    name = "lru"

    def __init__(self) -> None:
        # Resident items only, least recent first.
        self.recency: OrderedDict[int, None] = OrderedDict()
        self._batch: list[int] = []

    def observe(self, occurrences: Sequence[int] | np.ndarray) -> None:
        self._batch.extend(int(item) for item in occurrences)

    def choose(self, resident: set[int], incoming: Iterable[int], capacity: int) -> PolicyDecision:
        batch, self._batch = self._batch, []
        if len(self.recency) != len(resident):
            # Residency was set outside this policy; recency among those items is unknown.
            self.recency = OrderedDict((item, None) for item in sorted(resident))
        if capacity <= 0:
            evicted = tuple(self.recency)
            self.recency.clear()
            return PolicyDecision((), evicted)
        # Keep each item's last occurrence position; earlier duplicates do not change order.
        last_order = list(dict.fromkeys(reversed(batch)))
        last_order.reverse()
        for item in last_order:
            self.recency.pop(item, None)
            self.recency[item] = None
        evictions: list[int] = []
        while len(self.recency) > capacity:
            item, _ = self.recency.popitem(last=False)
            if item in resident:
                evictions.append(item)
        admissions = tuple(item for item in last_order if item in self.recency and item not in resident)
        return PolicyDecision(admissions, tuple(evictions))

    def reset(self) -> None:
        self.recency.clear()
        self._batch.clear()
