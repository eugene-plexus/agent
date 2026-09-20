"""Serialize model-launch commits and benchmark admission on a node."""

import asyncio
from collections.abc import AsyncIterator

from fastapi import HTTPException, Request


def launch_lock(request: Request) -> asyncio.Lock:
    lock = getattr(request.app.state, "model_launch_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.model_launch_lock = lock
    return lock


async def runtime_launch(request: Request) -> AsyncIterator[None]:
    async with launch_lock(request):
        manager = getattr(request.app.state, "benchmarks", None)
        if manager is not None and manager.active:
            raise HTTPException(
                409,
                "A benchmark is running on this node. Cancel it or wait before starting a model.",
            )
        yield
