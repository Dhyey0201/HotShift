from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    resident: frozenset[int]
    admissions: tuple[int, ...]
    evictions: tuple[int, ...]


class CachePolicy:
    name = "base"

    def observe(self, occurrences: Sequence[int]) -> None:
        raise NotImplementedError

    def choose(self, resident: set[int], incoming: Iterable[int], capacity: int) -> PolicyDecision:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

