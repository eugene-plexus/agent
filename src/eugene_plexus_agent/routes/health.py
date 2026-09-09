"""GET /healthz — liveness / readiness probe."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .._generated.common_models import Health, Status

router = APIRouter(tags=["meta"])


@router.get("/healthz", response_model=Health)
async def healthz(request: Request) -> Health:
    safe_mode = bool(getattr(request.app.state, "safe_mode", False))
    if safe_mode:
        return Health(
            status=Status.degraded,
            version=__version__,
            component="agent",
            safeMode=True,
        )
    return Health(
        status=Status.ok,
        version=__version__,
        component="agent",
        safeMode=False,
    )
