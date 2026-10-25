from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from hotshift.policies.base import PolicyDecision
from hotshift.policies.decayed_lfu import DecayedLFUPolicy


class AdaptivePolicy(DecayedLFUPolicy):
    """Frequency policy with fixed-size fast/slow sketches and bounded shift mode."""

    name = "adaptive"

    def __init__(
        self,
        window_occurrences: int = 65536,
        buckets: int = 4096,
        threshold: float = 0.20,
        normal_budget: int = 256,
        shift_budget: int = 1024,
    ) -> None:
        super().__init__(window_occurrences=window_occurrences, decay=0.8)
        if buckets <= 0 or not 0 < threshold < 1 or min(normal_budget, shift_budget) <= 0:
            raise ValueError("invalid adaptive policy parameters")
        self.buckets = buckets
        self.threshold = threshold
        self.normal_budget = normal_budget
        self.shift_budget = shift_budget
        self.fast = [0.0] * buckets
        self.slow = [0.0] * buckets
        self.window_sketch: dict[int, int] = defaultdict(int)
        self.windows_over_threshold = 0
        self.shift_windows_remaining = 0
        self.shift_score = 0.0
        self.mode_transitions: list[dict[str, float | str | int]] = []
        self.evicted_this_window: set[int] = set()
        self.maintenance_remaining = normal_budget

    def observe(self, occurrences: Sequence[int]) -> None:
        for item in occurrences:
            item_id = int(item)
            self.pending[item_id] += 1
            self.position += 1
            self.window_sketch[self._bucket(item_id)] += 1
            if self.position % self.window_occurrences == 0:
                self._finish_window()

    def _bucket(self, item: int) -> int:
        mixed = (item + 0x9E3779B9) & 0xFFFFFFFF
        mixed ^= mixed >> 16
        mixed = (mixed * 0x7FEB352D) & 0xFFFFFFFF
        mixed ^= mixed >> 15
        return mixed % self.buckets

    def _finish_window(self) -> None:
        super()._finish_window()
        for index in range(self.buckets):
            self.fast[index] *= 0.5
            self.slow[index] *= 0.95
        for index, count in self.window_sketch.items():
            self.fast[index] += float(count)
            self.slow[index] += float(count)
        self.window_sketch.clear()
        self.evicted_this_window.clear()
        fast_sum, slow_sum = sum(self.fast), sum(self.slow)
        if fast_sum and slow_sum:
            self.shift_score = 0.5 * sum(
                abs(a / fast_sum - b / slow_sum) for a, b in zip(self.fast, self.slow)
            )
        else:
            self.shift_score = 0.0
        self.windows_over_threshold = (
            self.windows_over_threshold + 1 if self.shift_score > self.threshold else 0
        )
        if self.windows_over_threshold >= 2:
            self.shift_windows_remaining = 4
            self.windows_over_threshold = 0
            self.mode_transitions.append(
                {"position": self.position, "mode": "shift", "reason": "sketch_tv", "score": self.shift_score}
            )
        elif self.shift_windows_remaining:
            self.shift_windows_remaining -= 1
            if self.shift_windows_remaining == 0:
                self.mode_transitions.append(
                    {"position": self.position, "mode": "normal", "reason": "shift_budget_elapsed", "score": self.shift_score}
                )
        self.maintenance_remaining = (
            self.shift_budget if self.shift_windows_remaining else self.normal_budget
        )

    def choose(self, resident: set[int], incoming: Iterable[int], capacity: int) -> PolicyDecision:
        if capacity <= 0:
            return PolicyDecision(frozenset(), (), tuple(sorted(resident)))
        row_budget = self.shift_budget if self.shift_windows_remaining else self.normal_budget
        budget = min(self.maintenance_remaining, row_budget, capacity)
        candidates = sorted(
            {
                int(item)
                for item in incoming
                if int(item) not in resident and int(item) not in self.evicted_this_window
            },
            key=lambda item: (-self.score(item), item),
        )
        selected: set[int] = set()
        evicted: list[int] = []
        available = max(0, capacity - len(resident))
        for item in candidates[: min(available, budget)]:
            selected.add(item)
        victims = sorted(resident, key=lambda item: (self.score(item), item))
        while candidates and len(selected) < budget and len(evicted) < budget:
            candidate = next((item for item in candidates if item not in selected), None)
            if candidate is None or not victims:
                break
            victim = next((item for item in victims if item not in evicted), None)
            if victim is None or self.score(candidate) <= self.score(victim) * 1.1 + 1e-6:
                break
            selected.add(candidate)
            evicted.append(victim)
            self.evicted_this_window.add(victim)
        self.maintenance_remaining = max(0, self.maintenance_remaining - len(selected))
        return PolicyDecision(
            frozenset((resident - set(evicted)) | selected),
            tuple(item for item in candidates if item in selected),
            tuple(evicted),
        )

    def reset(self) -> None:
        super().reset()
        self.fast = [0.0] * self.buckets
        self.slow = [0.0] * self.buckets
        self.window_sketch.clear()
        self.windows_over_threshold = 0
        self.shift_windows_remaining = 0
        self.shift_score = 0.0
        self.mode_transitions.clear()
        self.evicted_this_window.clear()
        self.maintenance_remaining = self.normal_budget

