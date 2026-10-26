"""Adaptive frequency policy with sketch-based popularity shift detection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from hotshift.policies.base import PolicyDecision
from hotshift.policies.decayed_lfu import DecayedLFUPolicy

_MASK = 0xFFFFFFFF


class AdaptivePolicy(DecayedLFUPolicy):
    """Decayed-frequency candidates, a fixed-size fast/slow sketch, and bounded shift mode.

    Shift detection: occurrences are hashed into ``buckets`` with a fixed 32-bit mixer
    (seed 0x9E3779B9). At each window boundary the fast (decay 0.5) and slow (decay 0.95)
    sketches decay and absorb the window. Their total variation distance is the shift score;
    two consecutive windows above ``threshold`` enter shift mode for four windows.

    Maintenance: filling an empty slot evicts nothing and copies a row already staged on the
    device, so it is not budgeted. Replacements are budgeted per window, in rows and bytes:
    ``normal_budget`` rows normally and ``shift_budget`` rows in shift mode, both capped at
    capacity. Unused allowance is discarded at the next window boundary. A replacement
    requires the candidate score to exceed the victim score by 10 percent plus epsilon, and
    an item evicted in a window cannot be readmitted in that same window.
    """

    name = "adaptive"

    def __init__(
        self,
        window_occurrences: int = 65536,
        buckets: int = 4096,
        threshold: float = 0.20,
        normal_budget: int = 256,
        shift_budget: int = 1024,
        fast_decay: float = 0.5,
        slow_decay: float = 0.95,
        margin: float = 0.10,
        epsilon: float = 1e-6,
        row_bytes: int | None = None,
        normal_budget_bytes: int | None = None,
        shift_budget_bytes: int | None = None,
    ) -> None:
        super().__init__(window_occurrences=window_occurrences, decay=0.8)
        if buckets <= 0 or not 0 < threshold < 1 or min(normal_budget, shift_budget) <= 0:
            raise ValueError("invalid adaptive policy parameters")
        if not 0 < fast_decay < 1 or not 0 < slow_decay < 1 or margin < 0 or epsilon <= 0:
            raise ValueError("invalid adaptive decay or admission margin")
        self.buckets = buckets
        self.threshold = threshold
        self.normal_budget = normal_budget
        self.shift_budget = shift_budget
        self.fast_decay = fast_decay
        self.slow_decay = slow_decay
        self.margin = margin
        self.epsilon = epsilon
        self.row_bytes = row_bytes
        self.normal_budget_bytes = normal_budget_bytes
        self.shift_budget_bytes = shift_budget_bytes
        self.fast = np.zeros(buckets, dtype=np.float64)
        self.slow = np.zeros(buckets, dtype=np.float64)
        self.window_sketch = np.zeros(buckets, dtype=np.float64)
        self.windows_over_threshold = 0
        self.shift_windows_remaining = 0
        self.shift_score = 0.0
        self.mode_transitions: list[dict[str, float | str | int]] = []
        self.evicted_this_window: set[int] = set()
        self.capacity_hint: int | None = None
        self.maintenance_remaining = normal_budget
        self.bytes_remaining: int | None = None
        self._reset_allowance()

    @staticmethod
    def _bucket_array(items: np.ndarray, buckets: int) -> np.ndarray:
        mixed = (items.astype(np.uint64) + np.uint64(0x9E3779B9)) & np.uint64(_MASK)
        mixed ^= mixed >> np.uint64(16)
        mixed = (mixed * np.uint64(0x7FEB352D)) & np.uint64(_MASK)
        mixed ^= mixed >> np.uint64(15)
        return (mixed % np.uint64(buckets)).astype(np.int64)

    def _bucket(self, item: int) -> int:
        return int(self._bucket_array(np.asarray([item], dtype=np.int64), self.buckets)[0])

    def _accumulate(self, chunk: np.ndarray) -> None:
        unique, counts = np.unique(chunk, return_counts=True)
        scores = self.scores
        weight = self.weight
        for item, count in zip(unique.tolist(), counts.tolist()):
            scores[item] = scores.get(item, 0.0) + count * weight
        if unique.size:
            self.window_sketch += np.bincount(
                self._bucket_array(unique, self.buckets), weights=counts, minlength=self.buckets
            )

    @property
    def in_shift_mode(self) -> bool:
        return self.shift_windows_remaining > 0

    def _row_budget(self) -> int:
        budget = self.shift_budget if self.in_shift_mode else self.normal_budget
        return min(budget, self.capacity_hint) if self.capacity_hint is not None else budget

    def _reset_allowance(self) -> None:
        self.maintenance_remaining = self._row_budget()
        if self.row_bytes:
            explicit = self.shift_budget_bytes if self.in_shift_mode else self.normal_budget_bytes
            self.bytes_remaining = explicit if explicit is not None else self._row_budget() * self.row_bytes
        else:
            self.bytes_remaining = None

    def configure(self, row_bytes: int) -> None:
        if self.row_bytes is None:
            self.row_bytes = int(row_bytes)
            self._reset_allowance()

    def _finish_window(self) -> None:
        super()._finish_window()
        self.fast *= self.fast_decay
        self.slow *= self.slow_decay
        self.fast += self.window_sketch
        self.slow += self.window_sketch
        self.window_sketch[:] = 0.0
        self.evicted_this_window.clear()
        fast_sum, slow_sum = float(self.fast.sum()), float(self.slow.sum())
        if fast_sum > 0 and slow_sum > 0:
            self.shift_score = 0.5 * float(np.abs(self.fast / fast_sum - self.slow / slow_sum).sum())
        else:
            self.shift_score = 0.0
        self.windows_over_threshold = (
            self.windows_over_threshold + 1 if self.shift_score > self.threshold else 0
        )
        if self.windows_over_threshold >= 2:
            if not self.in_shift_mode:
                self.mode_transitions.append(
                    {"position": self.position, "mode": "shift", "reason": "sketch_tv_above_threshold", "score": self.shift_score}
                )
            self.shift_windows_remaining = 4
            self.windows_over_threshold = 0
        elif self.shift_windows_remaining:
            self.shift_windows_remaining -= 1
            if self.shift_windows_remaining == 0:
                self.mode_transitions.append(
                    {"position": self.position, "mode": "normal", "reason": "shift_budget_elapsed", "score": self.shift_score}
                )
        self._reset_allowance()

    def _beats(self, candidate: int, victim: int) -> bool:
        return self._raw(candidate) > self._raw(victim) * (1.0 + self.margin) + self.epsilon * self.weight

    def choose(self, resident: set[int], incoming: Iterable[int], capacity: int) -> PolicyDecision:
        if capacity <= 0:
            return self._evict_all()
        self._sync(resident)
        if self.capacity_hint != capacity:
            self.capacity_hint = capacity
            self.maintenance_remaining = min(self.maintenance_remaining, capacity)
        admissions: list[int] = []
        evictions: list[int] = []
        for item in self._ranked(incoming, self.evicted_this_window):
            if len(self._resident) < capacity:
                self._admit(item)
                admissions.append(item)
                continue
            if self.maintenance_remaining <= 0:
                break
            if self.bytes_remaining is not None and self.row_bytes and self.bytes_remaining < self.row_bytes:
                break
            victim = self._peek_victim()
            if victim is None or not self._beats(item, victim):
                break
            evicted = self._pop_victim()
            evictions.append(evicted)
            self.evicted_this_window.add(evicted)
            self._admit(item)
            admissions.append(item)
            self.maintenance_remaining -= 1
            if self.bytes_remaining is not None and self.row_bytes:
                self.bytes_remaining -= self.row_bytes
        return PolicyDecision(tuple(admissions), tuple(evictions))

    def reset(self) -> None:
        super().reset()
        self.fast[:] = 0.0
        self.slow[:] = 0.0
        self.window_sketch[:] = 0.0
        self.windows_over_threshold = 0
        self.shift_windows_remaining = 0
        self.shift_score = 0.0
        self.mode_transitions.clear()
        self.evicted_this_window.clear()
        self._reset_allowance()

    def state(self) -> dict[str, object]:
        return {
            "mode": "shift" if self.in_shift_mode else "normal",
            "shift_score": self.shift_score,
            "transitions": len(self.mode_transitions),
        }
