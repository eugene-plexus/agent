"""Serialize model-launch commits and benchmark admission on a node."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request

from .runtime_context import RuntimeContext

JOB_SITE_REFUSAL = (
    "This machine is a job site: it serves its owner's files and runs no models, engines, "
    "benchmarks or profile builds (remote-nodes.md §3.2). Nothing was started."
)


def is_job_site(app: object) -> bool:
    identity = getattr(getattr(app, "state", None), "node_identity", None)
    return bool(identity is not None and identity.record.job_site)


def refuse_on_job_site(request: Request) -> None:
    """A route dependency: 409 on a Job Site, whoever asks."""
    if is_job_site(request.app):
        raise HTTPException(409, JOB_SITE_REFUSAL)


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

    if is_job_site(request.app):
        raise HTTPException(409, JOB_SITE_REFUSAL)
    async with launch_lock(request):
        if kind := active_measurement(request.app):
            raise HTTPException(
                409,
                f"{kind} is running on this node. Cancel it or wait before starting a model; "
                "models it stopped start again when it ends.",
            )
        yield
