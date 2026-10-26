from collections.abc import Iterable, Sequence

import numpy as np

from hotshift.policies.base import CachePolicy, PolicyDecision


class NoCachePolicy(CachePolicy):
    name = "no_cache"

    def observe(self, occurrences: Sequence[int] | np.ndarray) -> None:
        return None

    def choose(self, resident: set[int], incoming: Iterable[int], capacity: int) -> PolicyDecision:
        return PolicyDecision((), tuple(sorted(resident)))

    def reset(self) -> None:
        return None
