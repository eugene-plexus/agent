"""GET /healthz — liveness / readiness probe."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .._generated.common_models import Health, Status
from ..state import UNREADABLE_SUFFIX

router = APIRouter(tags=["meta"])


@router.get("/healthz", response_model=Health)
async def healthz(request: Request) -> Health:
    """Liveness, plus the one thing that can be wrong and still answer.

    **`configError` is the reason on the wire** (review §6.1 #6). An
    `agent.yaml` that will not parse no longer stops the process; it
    leaves it running on defaults with no topology, which from outside
    looks exactly like a fresh install. Something has to say which, and
    it has to be readable without a token, because this is the probe the
    supervisor and every uptime check already poll. `configFilePreserved`
    names the copy, so the remedy is a path rather than a search.

    `details` is a free-form object on `Health` and needs no contract
    change; the library already uses it for `unreadableRoots` and
    `scanError`, which is the same answer to the same question.
    """
    safe_mode = bool(getattr(request.app.state, "safe_mode", False))
    if safe_mode:
        return Health(
            status=Status.degraded,
            version=__version__,
            component="agent",
            safeMode=True,
        )

    state = getattr(request.app.state, "agent_state", None)
    reason = state.degraded_reason if state is not None else None
    if reason:
        return Health(
            status=Status.degraded,
            version=__version__,
            component="agent",
            safeMode=False,
            details={
                "configError": reason,
                "configFile": str(state.path) if state is not None else None,
                "configFilePreserved": (
                    str(state.path.with_suffix(state.path.suffix + UNREADABLE_SUFFIX))
                    if state is not None
                    else None
                ),
            },
        )

    return Health(
        status=Status.ok,
        version=__version__,
        component="agent",
        safeMode=False,
    )
