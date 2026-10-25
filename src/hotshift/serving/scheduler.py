from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic

from hotshift.engine.base import CPUCacheEngine
from hotshift.serving.metrics import MetricsCollector


@dataclass
class QueuedRequest:
    history: list[int]
    future: asyncio.Future[list[float]]
    enqueued_at: float
    deadline: float


class QueueFullError(RuntimeError):
    pass


class SchedulerUnavailableError(RuntimeError):
    pass


class BatchScheduler:
    def __init__(
        self,
        engine: CPUCacheEngine,
        queue_capacity: int = 1024,
        max_batch_requests: int = 64,
        max_batch_ids: int = 16384,
        max_batch_delay_ms: float = 2.0,
    ) -> None:
        self.engine = engine
        self.queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(maxsize=queue_capacity)
        self.max_batch_requests = max_batch_requests
        self.max_batch_ids = max_batch_ids
        self.max_batch_delay = max_batch_delay_ms / 1000
        self.metrics = MetricsCollector()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hotshift-engine")
        self.task: asyncio.Task[None] | None = None
        self.accepting = False
        self.closed = False
        self._pending: QueuedRequest | None = None

    async def start(self) -> None:
        if self.task is None:
            self.accepting = True
            self.task = asyncio.create_task(self._run(), name="hotshift-batch-scheduler")

    async def submit(self, history: Sequence[int], timeout_ms: int = 2000) -> tuple[list[float], float, float]:
        if not self.accepting or self.closed:
            raise SchedulerUnavailableError("engine is not accepting requests")
        loop = asyncio.get_running_loop()
        now = monotonic()
        future: asyncio.Future[list[float]] = loop.create_future()
        request = QueuedRequest(list(history), future, now, now + timeout_ms / 1000)
        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull as error:
            self.metrics.record_error("queue_full")
            raise QueueFullError("request queue is full") from error
        self.metrics.set_queue_occupancy(self.queue.qsize())
        try:
            result = await asyncio.wait_for(asyncio.shield(future), timeout_ms / 1000)
            return result, (request.started_at - request.enqueued_at) * 1000, (monotonic() - now) * 1000
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

    async def _get_next(self) -> QueuedRequest | None:
        if self._pending is not None:
            item, self._pending = self._pending, None
            return item
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=0.1)
        except TimeoutError:
            return None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while not self.closed:
            first = await self._get_next()
            if first is None:
                continue
            batch = [first]
            token_count = len(first.history)
            flush_at = first.enqueued_at + self.max_batch_delay
            while len(batch) < self.max_batch_requests and monotonic() < flush_at:
                remaining = flush_at - monotonic()
                try:
                    candidate = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                if token_count + len(candidate.history) > self.max_batch_ids:
                    self._pending = candidate
                    break
                batch.append(candidate)
                token_count += len(candidate.history)
            self.metrics.set_queue_occupancy(self.queue.qsize() + int(self._pending is not None))
            active: list[QueuedRequest] = []
            now = monotonic()
            for request in batch:
                if request.future.cancelled():
                    self.metrics.record_error("cancelled_queued")
                elif now >= request.deadline:
                    if not request.future.done():
                        request.future.set_exception(TimeoutError())
                    self.metrics.record_error("expired_queued")
                else:
                    request.started_at = now
                    active.append(request)
            if not active:
                for _ in batch:
                    self.queue.task_done()
                continue
            if not self.engine.healthy:
                for request in active:
                    if not request.future.done():
                        request.future.set_exception(SchedulerUnavailableError("engine is unhealthy"))
                self.metrics.record_error("unhealthy_engine")
                for _ in batch:
                    self.queue.task_done()
                continue
            try:
                result = await loop.run_in_executor(
                    self.executor,
                    self.engine.pool,
                    [request.history for request in active],
                )
                queue_delays = [(request.started_at - request.enqueued_at) * 1000 for request in active]
                self.metrics.record_batch(result, queue_delays, len(active))
                for index, request in enumerate(active):
                    if not request.future.done():
                        request.future.set_result(result.outputs[index].tolist())
            except Exception as error:
                self.engine.healthy = False
                self.metrics.record_error("engine_failure")
                for request in active:
                    if not request.future.done():
                        request.future.set_exception(error)
            finally:
                for _ in batch:
                    self.queue.task_done()
                self.metrics.set_queue_occupancy(self.queue.qsize() + int(self._pending is not None))

    async def exclusive(self, operation) -> None:
        if self.task is None or self.closed:
            raise SchedulerUnavailableError("scheduler is not running")
        self.accepting = False
        try:
            await self.queue.join()
            await asyncio.get_running_loop().run_in_executor(self.executor, operation)
        finally:
            if not self.closed:
                self.accepting = True

    async def close(self, timeout_seconds: float = 5.0) -> None:
        self.accepting = False
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout_seconds)
        except TimeoutError:
            while not self.queue.empty():
                request = self.queue.get_nowait()
                if not request.future.done():
                    request.future.cancel()
                self.queue.task_done()
        self.closed = True
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True, cancel_futures=True)

