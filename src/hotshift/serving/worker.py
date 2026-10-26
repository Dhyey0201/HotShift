"""The single thread that owns the engine.

All engine calls, including cache resets and policy switches, run on this one thread, so
cache state and CUDA launches never see concurrent callers. Results return to the asyncio
event loop through ``run_in_executor``, which resolves futures on the loop thread.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")
logger = logging.getLogger("hotshift.worker")


class EngineWorker:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hotshift-engine")
        self.closed = False

    @property
    def healthy(self) -> bool:
        return bool(getattr(self.engine, "healthy", True))

    async def run(self, function: Callable[..., T], *args: Any) -> T:
        if self.closed:
            raise RuntimeError("engine worker is closed")
        return await asyncio.get_running_loop().run_in_executor(self._executor, function, *args)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        # Waits for an already-launched batch; launched device work is never abandoned.
        self._executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(self.engine, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # shutdown must finish even if teardown fails
                logger.exception("engine close failed")
