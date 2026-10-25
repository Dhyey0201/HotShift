import asyncio

from hotshift.data.generate import generate_embeddings
from hotshift.data.store import EmbeddingStore
from hotshift.engine.base import CPUCacheEngine
from hotshift.policies.lru import LRUPolicy
from hotshift.serving.scheduler import BatchScheduler


def test_scheduler_batches_and_records_metrics(tmp_path):
    generate_embeddings(32, 4, 4, tmp_path)
    engine = CPUCacheEngine(EmbeddingStore(tmp_path), 8, LRUPolicy())

    async def exercise():
        scheduler = BatchScheduler(engine, queue_capacity=8, max_batch_requests=4, max_batch_delay_ms=5)
        await scheduler.start()
        first, second = await asyncio.gather(scheduler.submit([1, 2]), scheduler.submit([2, 3]))
        assert len(first[0]) == 4
        assert len(second[0]) == 4
        snapshot = scheduler.metrics.snapshot()
        assert snapshot["counts"]["completed_requests"] == 2
        assert snapshot["occurrences"] == 4
        await scheduler.close()

    asyncio.run(exercise())

