from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence

import numpy as np

from hotshift.policies.base import CachePolicy, PolicyDecision

# Rescale stored scores before the growing occurrence weight can approach FP64 overflow.
_RENORMALIZE_AT = 1e200


class DecayedLFUPolicy(CachePolicy):
    """Windowed decayed-frequency policy.

    At every complete window of ``window_occurrences`` processed occurrences, each score is
    multiplied by ``decay`` and that window's counts are added. Scores are stored in
    "inflated" units: an occurrence in window ``w`` adds ``(1/decay)**w``. Dividing every
    stored score by the current weight gives the decayed score, and because that division is
    uniform, ranking never needs an O(N) decay pass. An in-progress window counts at full
    weight, i.e. as though it completed now; ``flush`` closes a partial window at end of run.

    Resident victims live in a lazy min-heap keyed by (score, -item). Inflated scores only
    increase, so an entry whose stored score is stale can be refreshed when it reaches the
    top. Ranking ties break by item ID: a lower ID ranks higher.
    """

    name = "decayed_lfu"

    def __init__(self, window_occurrences: int = 65536, decay: float = 0.8) -> None:
        if window_occurrences <= 0 or not 0 < decay < 1:
            raise ValueError("window_occurrences must be positive and decay must be in (0, 1)")
        self.window_occurrences = window_occurrences
        self.decay = decay
        self.growth = 1.0 / decay
        self.weight = 1.0
        self.scores: dict[int, float] = {}
        self.position = 0
        self.window_index = 0
        self._heap: list[tuple[float, int]] = []
        self._resident: set[int] = set()

    def observe(self, occurrences: Sequence[int] | np.ndarray) -> None:
        ids = np.asarray(occurrences, dtype=np.int64)
        start = 0
        while start < ids.size:
            room = self.window_occurrences - self.position % self.window_occurrences
            chunk = ids[start : start + room]
            self._accumulate(chunk)
            self.position += int(chunk.size)
            start += int(chunk.size)
            if self.position % self.window_occurrences == 0:
                self._finish_window()

    def _accumulate(self, chunk: np.ndarray) -> None:
        unique, counts = np.unique(chunk, return_counts=True)
        scores = self.scores
        weight = self.weight
        for item, count in zip(unique.tolist(), counts.tolist()):
            scores[item] = scores.get(item, 0.0) + count * weight

    def _finish_window(self) -> None:
        self.window_index += 1
        self.weight *= self.growth
        if self.weight > _RENORMALIZE_AT:
            factor = 1.0 / self.weight
            for item in self.scores:
                self.scores[item] *= factor
            # Positive scaling preserves heap order, and each entry stays equal to its score
            # when it was current, because both sides undergo the same multiplication.
            self._heap = [(score * factor, key) for score, key in self._heap]
            self.weight = 1.0

    def flush(self) -> None:
        if self.position % self.window_occurrences:
            self.position += self.window_occurrences - self.position % self.window_occurrences
            self._finish_window()

    def score(self, item: int) -> float:
        """Decayed score in occurrence units."""
        return self.scores.get(int(item), 0.0) / self.weight

    def _raw(self, item: int) -> float:
        return self.scores.get(item, 0.0)

    def _admit(self, item: int) -> None:
        self._resident.add(item)
        heapq.heappush(self._heap, (self._raw(item), -item))

    def _peek_victim(self) -> int | None:
        heap = self._heap
        while heap:
            stored, negative = heap[0]
            item = -negative
            if item not in self._resident:
                heapq.heappop(heap)
                continue
            current = self._raw(item)
            if current != stored:
                heapq.heapreplace(heap, (current, negative))
                continue
            return item
        return None

    def _pop_victim(self) -> int:
        _, negative = heapq.heappop(self._heap)
        item = -negative
        self._resident.discard(item)
        return item

    def _beats(self, candidate: int, victim: int) -> bool:
        candidate_score, victim_score = self._raw(candidate), self._raw(victim)
        return candidate_score > victim_score or (candidate_score == victim_score and candidate < victim)

    def _ranked(self, incoming: Iterable[int], excluded: set[int] | None = None) -> list[int]:
        candidates = {int(item) for item in incoming}
        candidates -= self._resident
        if excluded:
            candidates -= excluded
        return sorted(candidates, key=lambda item: (-self._raw(item), item))

    def _sync(self, resident: set[int]) -> None:
        # Engines apply every decision, so this only triggers when residency was set externally.
        if len(resident) != len(self._resident):
            self._resident = {int(item) for item in resident}
            self._heap = [(self._raw(item), -item) for item in self._resident]
            heapq.heapify(self._heap)

    def _evict_all(self) -> PolicyDecision:
        evicted = tuple(sorted(self._resident))
        self._resident.clear()
        self._heap.clear()
        return PolicyDecision((), evicted)

    def choose(self, resident: set[int], incoming: Iterable[int], capacity: int) -> PolicyDecision:
        if capacity <= 0:
            return self._evict_all()
        self._sync(resident)
        admissions: list[int] = []
        evictions: list[int] = []
        for item in self._ranked(incoming):
            if len(self._resident) < capacity:
                self._admit(item)
                admissions.append(item)
                continue
            victim = self._peek_victim()
            # Candidates arrive best first, so once one loses every later one loses too.
            if victim is None or not self._beats(item, victim):
                break
            evictions.append(self._pop_victim())
            self._admit(item)
            admissions.append(item)
        return PolicyDecision(tuple(admissions), tuple(evictions))

    def reset(self) -> None:
        self.scores.clear()
        self.position = 0
        self.window_index = 0
        self.weight = 1.0
        self._heap.clear()
        self._resident.clear()

    def state(self) -> dict[str, object]:
        return {"mode": "frequency", "shift_score": None}
