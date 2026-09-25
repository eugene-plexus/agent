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

    **`trustBundleAgeSeconds`** rides on every answer from a node that
    has joined an install: seconds since it last took the control root's
    bundle, which it pulls every minute. Reported, never a status: a dead
    root must not make every node look down.

    **`installPermissions`** rides on every answer, degraded or not:
    one sentence per account outside this one, SYSTEM and Administrators
    that can read `node.yaml` or add files to the install
    (`install_permissions`). It does not degrade the status -- the
    install works, and a status that flips for a permissions problem
    would hide the next real outage behind a known one.
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
                **_permissions(request),
                **_trust(request),
                "configError": reason,
                "configFile": str(state.path) if state is not None else None,
                "configFilePreserved": (
                    str(state.path.with_suffix(state.path.suffix + UNREADABLE_SUFFIX))
                    if state is not None
                    else None
                ),
            },
        )

    # `apps.yaml` degrades on its own (docs/design/apps-and-spokes.md),
    # and says so here the same way: the topology is fine, the apps are
    # not, and the copy that would not load is named.
    apps = getattr(request.app.state, "apps", None)
    if apps is not None and apps.store.degraded_reason:
        apps_reason = apps.store.degraded_reason
        return Health(
            status=Status.degraded,
            version=__version__,
            component="agent",
            safeMode=False,
            details={
                **_permissions(request),
                **_trust(request),
                "appsError": apps_reason,
                "appsFile": str(apps.store.path),
                "appsFilePreserved": str(
                    apps.store.path.with_suffix(apps.store.path.suffix + UNREADABLE_SUFFIX)
                ),
            },
        )

    return Health(
        status=Status.ok,
        version=__version__,
        component="agent",
        safeMode=False,
        details={**_permissions(request), **_trust(request)} or None,
    )


def _permissions(request: Request) -> dict[str, list[str]]:
    sentences = list(getattr(request.app.state, "install_permissions", None) or [])
    return {"installPermissions": sentences} if sentences else {}


def _trust(request: Request) -> dict[str, int]:
    auth = getattr(request.app.state, "auth_state", None)
    age = auth.trust.heard_age_seconds() if auth is not None else None
    return {"trustBundleAgeSeconds": age} if age is not None else {}
