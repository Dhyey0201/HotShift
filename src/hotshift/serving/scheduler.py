from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from hotshift.serving.metrics import MetricsCollector
from hotshift.serving.worker import EngineWorker

logger = logging.getLogger("hotshift.scheduler")


@dataclass
class QueuedRequest:
    history: list[int]
    future: asyncio.Future[Any]
    enqueued_at: float
    deadline: float
    request_id: str = ""
    started_at: float | None = None


@dataclass(frozen=True)
class SubmitResult:
    vector: list[float]
    queue_ms: float
    batch_engine_ms: float
    server_ms: float
    batch_size: int
    generation: int


@dataclass
class _BatchOutcome:
    vectors: list[list[float]] = field(default_factory=list)
    engine_ms: float = 0.0
    generation: int = 0


class QueueFullError(RuntimeError):
    pass


class SchedulerUnavailableError(RuntimeError):
    pass


class EngineBusyError(SchedulerUnavailableError):
    """Admission is paused while an exclusive maintenance command runs."""


class BatchScheduler:
    """FIFO microbatching in front of a single engine worker.

    A batch starts when it reaches ``max_batch_requests`` or ``max_batch_ids`` flattened IDs,
    or when ``max_batch_delay_ms`` has elapsed since its oldest request was enqueued. A
    request that would exceed the ID limit is held as the head of the next batch, so large
    requests are never starved. Expired or cancelled queued requests are skipped before
    launch and never reach the policy.
    """

    def __init__(
        self,
        engine: Any,
        queue_capacity: int = 1024,
        max_batch_requests: int = 64,
        max_batch_ids: int = 16384,
        max_batch_delay_ms: float = 2.0,
    ) -> None:
        self.engine = engine
        self.worker = EngineWorker(engine)
        self.queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(maxsize=queue_capacity)
        self.queue_capacity = queue_capacity
        self.max_batch_requests = max_batch_requests
        self.max_batch_ids = max_batch_ids
        self.max_batch_delay = max_batch_delay_ms / 1000
        self.metrics = MetricsCollector()
        self.task: asyncio.Task[None] | None = None
        self.accepting = False
        self.closed = False
        self.exclusive_active = False
        self._pending: QueuedRequest | None = None
        self._exclusive_lock = asyncio.Lock()

    async def start(self) -> None:
        if self.task is None:
            self.accepting = True
            self.task = asyncio.create_task(self._run(), name="hotshift-batch-scheduler")

    async def submit(self, history: Sequence[int], timeout_ms: int = 2000, request_id: str = "") -> SubmitResult:
        if self.closed:
            raise SchedulerUnavailableError("engine is shutting down")
        if self.exclusive_active:
            raise EngineBusyError("engine is running a maintenance command; retry shortly")
        if not self.accepting:
            raise SchedulerUnavailableError("engine is not accepting requests")
        if not self.worker.healthy:
            raise SchedulerUnavailableError("engine is unhealthy")
        loop = asyncio.get_running_loop()
        now = monotonic()
        future: asyncio.Future[Any] = loop.create_future()
        request = QueuedRequest(list(history), future, now, now + timeout_ms / 1000, request_id)
        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull as error:
            self.metrics.record_error("queue_full")
            raise QueueFullError("request queue is full") from error
        self._update_occupancy()
        try:
            outcome = await asyncio.wait_for(asyncio.shield(future), timeout_ms / 1000)
        except TimeoutError:
            if not future.done():
                future.cancel()
            self.metrics.record_error("timeout")
            raise
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            self.metrics.record_error("cancelled_client")
            raise
        vector, engine_ms, generation, batch_size = outcome
        started = request.started_at if request.started_at is not None else request.enqueued_at
        return SubmitResult(
            vector=vector,
            queue_ms=(started - request.enqueued_at) * 1000,
            batch_engine_ms=engine_ms,
            server_ms=(monotonic() - now) * 1000,
            batch_size=batch_size,
            generation=generation,
        )

    def _update_occupancy(self) -> None:
        self.metrics.set_queue_occupancy(self.queue.qsize() + int(self._pending is not None), self.queue_capacity)

    async def _get_next(self) -> QueuedRequest | None:
        if self._pending is not None:
            item, self._pending = self._pending, None
            return item
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=0.1)
        except TimeoutError:
            return None

    async def _collect(self, first: QueuedRequest) -> list[QueuedRequest]:
        batch = [first]
        token_count = len(first.history)
        flush_at = first.enqueued_at + self.max_batch_delay
        while len(batch) < self.max_batch_requests and token_count < self.max_batch_ids:
            # Requests already queued always join; the delay only bounds waiting for new ones.
            # Under a backlog the oldest request is past its delay, yet the batch still fills.
            try:
                candidate = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                remaining = flush_at - monotonic()
                if remaining <= 0:
                    break
                try:
                    candidate = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except TimeoutError:
                    break
            if token_count + len(candidate.history) > self.max_batch_ids:
                self._pending = candidate
                break
            batch.append(candidate)
            token_count += len(candidate.history)
        return batch

    def _admit(self, batch: list[QueuedRequest]) -> list[QueuedRequest]:
        active: list[QueuedRequest] = []
        now = monotonic()
        for request in batch:
            if request.future.done():
                self.metrics.record_error("skipped_queued")
            elif now >= request.deadline:
                request.future.set_exception(TimeoutError())
                self.metrics.record_error("expired_queued")
            else:
                request.started_at = now
                active.append(request)
        return active

    async def _run(self) -> None:
        while not self.closed:
            first = await self._get_next()
            if first is None:
                continue
            batch = await self._collect(first)
            self._update_occupancy()
            try:
                active = self._admit(batch)
                if active:
                    await self._execute(active)
            finally:
                for _ in batch:
                    self.queue.task_done()
                self._update_occupancy()

    async def _execute(self, active: list[QueuedRequest]) -> None:
        if not self.worker.healthy:
            for request in active:
                if not request.future.done():
                    request.future.set_exception(SchedulerUnavailableError("engine is unhealthy"))
            self.metrics.record_error("unhealthy_engine")
            return
        engine = self.engine
        try:
            result = await self.worker.run(engine.pool, [request.history for request in active])
        except ValueError as error:
            # Input rejected by the engine before any device work; the engine stays healthy.
            self.metrics.record_error("engine_validation")
            for request in active:
                if not request.future.done():
                    request.future.set_exception(error)
            return
        except Exception as error:  # any other engine failure is fatal to the engine
            engine.healthy = False
            self.metrics.record_error("engine_failure")
            logger.error(
                "batch failed backend=%s generation=%s batch_size=%d cause=%s",
                getattr(engine, "backend", "?"), getattr(engine, "generation", "?"), len(active), error,
            )
            for request in active:
                if not request.future.done():
                    request.future.set_exception(SchedulerUnavailableError(f"engine failure: {error}"))
            return
        queue_delays = [((request.started_at or request.enqueued_at) - request.enqueued_at) * 1000 for request in active]
        self.metrics.record_batch(result, queue_delays, len(active))
        generation = getattr(engine, "generation", 0)
        logger.debug(
            "batch backend=%s generation=%s batch_size=%d occurrences=%d engine_ms=%.3f",
            getattr(engine, "backend", "?"), generation, len(active), result.occurrences, result.engine_ms,
        )
        for index, request in enumerate(active):
            # A caller that disconnected after launch simply never reads its result.
            if not request.future.done():
                request.future.set_result((result.outputs[index].tolist(), result.engine_ms, generation, len(active)))

    async def exclusive(self, operation: Callable[[], Any]) -> None:
        """Pause admission, drain queued and in-flight work, run ``operation`` on the owner thread."""
        if self.task is None or self.closed:
            raise SchedulerUnavailableError("scheduler is not running")
        if self._exclusive_lock.locked():
            raise EngineBusyError("another maintenance command is running")
        async with self._exclusive_lock:
            self.exclusive_active = True
            try:
                await self.queue.join()
                await self.worker.run(operation)
            finally:
                self.exclusive_active = False

    async def close(self, timeout_seconds: float = 5.0) -> None:
        """Stop admission, drain bounded work, cancel what remains, and release the engine."""
        self.accepting = False
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout_seconds)
        except TimeoutError:
            logger.warning("shutdown drain timed out after %.1fs; cancelling queued requests", timeout_seconds)
            if self._pending is not None and not self._pending.future.done():
                self._pending.future.cancel()
            while not self.queue.empty():
                request = self.queue.get_nowait()
                if not request.future.done():
                    request.future.cancel()
                self.queue.task_done()
        self.closed = True
        if self.task is not None:
            # The loop exits after the batch it is executing; launched work is allowed to finish
            # and resolve its callers. Only a loop stuck beyond the bound is cancelled.
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=max(timeout_seconds, 1.0))
            except TimeoutError:
                logger.warning("in-flight batch exceeded the shutdown bound; cancelling the scheduler loop")
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
        await asyncio.get_running_loop().run_in_executor(None, self.worker.close)
