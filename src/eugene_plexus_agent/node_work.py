"""Serialize model-launch commits and benchmark admission on a node."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request

from .runtime_context import RuntimeContext


def launch_lock(request: RuntimeContext) -> asyncio.Lock:
    lock = getattr(request.app.state, "model_launch_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.model_launch_lock = lock
    return lock


async def runtime_launch(request: Request) -> AsyncIterator[None]:
    async with launch_guard(request):
        yield


@asynccontextmanager
async def launch_guard(request: RuntimeContext) -> AsyncIterator[None]:
    from .measurement_node import active_measurement

    async with launch_lock(request):
        if kind := active_measurement(request.app):
            raise HTTPException(
                409,
                f"{kind} is running on this node. Cancel it or wait before starting a model; "
                "models it stopped start again when it ends.",
            )
        yield
