"""Scheduler behavior with an injected fake engine. Valid for queueing, not CUDA concurrency."""

import asyncio
import threading
import time

import numpy as np
import pytest

from hotshift.data.generate import generate_embeddings
from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import BatchResult, CPUCacheEngine
from hotshift.policies.lru import LRUPolicy
from hotshift.serving.scheduler import (
    BatchScheduler,
    EngineBusyError,
    QueueFullError,
    SchedulerUnavailableError,
)


class FakeEngine:
    backend = "fake"

    def __init__(self, delay: float = 0.0, fail_with: Exception | None = None) -> None:
        self.delay = delay
        self.fail_with = fail_with
        self.healthy = True
        self.generation = 0
        self.batches: list[list[list[int]]] = []
        self.thread_names: set[str] = set()
        self.release = threading.Event()
        self.release.set()
        self.active = 0
        self.max_active = 0

    def pool(self, histories):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.thread_names.add(threading.current_thread().name)
        try:
            self.release.wait(5)
            time.sleep(self.delay)
            if self.fail_with is not None:
                raise self.fail_with
            self.batches.append([list(history) for history in histories])
            outputs = np.asarray([[float(sum(history)), float(len(history))] for history in histories], dtype=np.float32)
            occurrences = sum(len(history) for history in histories)
            return BatchResult(outputs, 0, occurrences, 0, occurrences, 0, 1.0, 0, 0)
        finally:
            self.active -= 1

    def reset(self):
        self.generation += 1


def run(coroutine):
    return asyncio.run(coroutine)


def test_scheduler_batches_and_records_metrics(tmp_path):
    generate_embeddings(32, 4, 4, tmp_path)
    engine = CPUCacheEngine(EmbeddingStore(tmp_path), 8, LRUPolicy())

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=8, max_batch_requests=4, max_batch_delay_ms=5)
        await scheduler.start()
        first, second = await asyncio.gather(scheduler.submit([1, 2]), scheduler.submit([2, 3]))
        assert len(first.vector) == 4 and len(second.vector) == 4
        assert first.batch_size == 2
        snapshot = scheduler.metrics.snapshot()
        assert snapshot["counts"]["completed_requests"] == 2
        assert snapshot["occurrences"] == 4
        assert snapshot["window"]["requests"] == 2
        await scheduler.close()

    run(exercise())


def test_fifo_order_token_limit_and_single_owner_thread():
    engine = FakeEngine()

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=32, max_batch_requests=8, max_batch_ids=10, max_batch_delay_ms=20)
        await scheduler.start()
        histories = [[index] * 4 for index in range(6)]
        results = await asyncio.gather(*(scheduler.submit(history) for history in histories))
        assert [result.vector[0] for result in results] == [sum(history) for history in histories]
        await scheduler.close()

    run(exercise())
    flattened = [history[0] for batch in engine.batches for history in batch]
    assert flattened == list(range(6))
    assert all(sum(len(history) for history in batch) <= 10 for batch in engine.batches)
    assert len(engine.thread_names) == 1 and engine.max_active == 1


def test_oversized_request_is_not_starved():
    engine = FakeEngine()

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=32, max_batch_requests=8, max_batch_ids=8, max_batch_delay_ms=20)
        await scheduler.start()
        small = scheduler.submit([1] * 6)
        large = scheduler.submit([2] * 8)
        tail = scheduler.submit([3])
        await asyncio.gather(small, large, tail)
        await scheduler.close()

    run(exercise())
    assert [[history[0] for history in batch] for batch in engine.batches] == [[1], [2], [3]]


def test_queue_full_rejects_immediately():
    engine = FakeEngine()
    engine.release.clear()

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=2, max_batch_requests=1, max_batch_delay_ms=1)
        await scheduler.start()
        tasks = [asyncio.create_task(scheduler.submit([1], 3000))]
        await asyncio.sleep(0.05)  # the first request is launched and blocked in the engine
        tasks += [asyncio.create_task(scheduler.submit([1], 3000)) for _ in range(2)]
        await asyncio.sleep(0)  # two more fill the queue
        with pytest.raises(QueueFullError):
            await scheduler.submit([1])
        engine.release.set()
        await asyncio.gather(*tasks)
        assert scheduler.metrics.snapshot()["counts"]["errors_queue_full"] == 1
        await scheduler.close()

    run(exercise())


def test_queued_timeout_and_cancellation_are_skipped_before_launch():
    engine = FakeEngine()
    engine.release.clear()

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=8, max_batch_requests=1, max_batch_delay_ms=1)
        await scheduler.start()
        blocker = asyncio.create_task(scheduler.submit([0], 3000))
        await asyncio.sleep(0.05)
        with pytest.raises(TimeoutError):
            await scheduler.submit([7], 50)
        cancelled = asyncio.create_task(scheduler.submit([8], 3000))
        await asyncio.sleep(0.01)
        cancelled.cancel()
        engine.release.set()
        await blocker
        await asyncio.sleep(0.05)
        counts = scheduler.metrics.snapshot()["counts"]
        assert counts["errors_timeout"] == 1 and counts["errors_cancelled_client"] == 1
        await scheduler.close()

    run(exercise())
    executed = [history[0] for batch in engine.batches for history in batch]
    assert executed == [0], "expired and cancelled queued requests must never reach the engine"


def test_cancellation_after_launch_finishes_batch_and_drops_response():
    engine = FakeEngine()
    engine.release.clear()

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=8, max_batch_requests=2, max_batch_delay_ms=1)
        await scheduler.start()
        task = asyncio.create_task(scheduler.submit([5], 3000))
        await asyncio.sleep(0.05)
        task.cancel()
        engine.release.set()
        follow_up = await scheduler.submit([6])
        assert follow_up.vector[0] == 6
        await scheduler.close()

    run(exercise())
    assert [history[0] for batch in engine.batches for history in batch] == [5, 6]


def test_exclusive_drains_queue_and_blocks_admission():
    engine = FakeEngine(delay=0.02)

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=16, max_batch_requests=1, max_batch_delay_ms=1)
        await scheduler.start()
        queued = [asyncio.create_task(scheduler.submit([index])) for index in range(4)]
        await asyncio.sleep(0)
        order: list[str] = []
        exclusive = asyncio.create_task(scheduler.exclusive(lambda: order.append(f"reset-after-{len(engine.batches)}")))
        await asyncio.sleep(0.005)
        with pytest.raises(EngineBusyError):
            await scheduler.submit([99])
        with pytest.raises(EngineBusyError):
            await scheduler.exclusive(lambda: None)
        await exclusive
        await asyncio.gather(*queued)
        assert order == ["reset-after-4"]
        assert (await scheduler.submit([5])).vector[0] == 5
        await scheduler.close()

    run(exercise())


def test_engine_failure_marks_unhealthy_without_fallback():
    engine = FakeEngine(fail_with=RuntimeError("illegal memory access"))

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=8, max_batch_delay_ms=1)
        await scheduler.start()
        with pytest.raises(SchedulerUnavailableError, match="engine failure"):
            await scheduler.submit([1])
        assert engine.healthy is False
        with pytest.raises(SchedulerUnavailableError, match="unhealthy"):
            await scheduler.submit([1])
        await scheduler.close()

    run(exercise())


def test_engine_validation_error_keeps_engine_healthy():
    engine = FakeEngine(fail_with=ValueError("history contains an ID outside the embedding table"))

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=8, max_batch_delay_ms=1)
        await scheduler.start()
        with pytest.raises(ValueError):
            await scheduler.submit([1])
        assert engine.healthy is True
        await scheduler.close()

    run(exercise())


def test_graceful_shutdown_cancels_work_beyond_the_drain_timeout():
    engine = FakeEngine()
    engine.release.clear()

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=8, max_batch_requests=1, max_batch_delay_ms=1)
        await scheduler.start()
        launched = asyncio.create_task(scheduler.submit([1], 5000))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(scheduler.submit([2], 5000))
        await asyncio.sleep(0.01)
        threading.Timer(0.3, engine.release.set).start()
        await scheduler.close(timeout_seconds=0.1)
        with pytest.raises(SchedulerUnavailableError):
            await scheduler.submit([3])
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert (await launched).vector[0] == 1, "launched work must finish and resolve its caller"

    run(exercise())
    assert [history[0] for batch in engine.batches for history in batch] == [1]
