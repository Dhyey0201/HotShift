from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from hotshift.policies.base import CachePolicy, PolicyDecision


class DecayedLFUPolicy(CachePolicy):
    name = "decayed_lfu"

    def __init__(self, window_occurrences: int = 65536, decay: float = 0.8) -> None:
        if window_occurrences <= 0 or not 0 < decay < 1:
            raise ValueError("window_occurrences must be positive and decay must be in (0, 1)")
        self.window_occurrences = window_occurrences
        self.decay = decay
        self.scores: dict[int, float] = defaultdict(float)
        self.pending: dict[int, int] = defaultdict(int)
        self.position = 0

    def observe(self, occurrences: Sequence[int]) -> None:
        for item in occurrences:
            self.pending[int(item)] += 1
            self.position += 1
            if self.position % self.window_occurrences == 0:
                self._finish_window()

    def _finish_window(self) -> None:
        for item in list(self.scores):
            self.scores[item] *= self.decay
            if self.scores[item] < 1e-12:
                del self.scores[item]
        for item, count in self.pending.items():
            self.scores[item] = self.scores.get(item, 0.0) + count
        self.pending.clear()

    def flush(self) -> None:
        if self.pending:
            self._finish_window()

    def score(self, item: int) -> float:
        return self.scores.get(item, 0.0) + self.pending.get(item, 0)

    def choose(self, resident: set[int], incoming: Iterable[int], capacity: int) -> PolicyDecision:
        if capacity <= 0:
            return PolicyDecision(frozenset(), (), tuple(sorted(resident)))
        candidate_ids = resident | {int(item) for item in incoming}
        ranked = sorted(candidate_ids, key=lambda item: (-self.score(item), item))
        selected = set(ranked[:capacity])
        admissions = tuple(sorted(selected - resident))
        evictions = tuple(sorted(resident - selected))
        return PolicyDecision(frozenset(selected), admissions, evictions)

    def reset(self) -> None:
        self.scores.clear()
        self.pending.clear()
        self.position = 0

