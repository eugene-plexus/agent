"""`POST /v1/node/update/check` and `POST /v1/node/update`.

Plus `node_update_view`, which `GET /v1/node` calls: what is installed
and whether it is behind, from the last check and the records an update
leaves. No network on that path -- every console polls it.

Design: `specs/docs/design/in-app-updates.md`.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status

from .. import install_info, update_apply
from .._generated.common_models import Problem
from .._generated.models import (
    NodeInstall,
    NodeUpdate,
    UpdateChannel,
    UpdateRequest,
    UpdateRun,
)
from ..dependencies import require_operator_session
from ..state import default_update_channel
from ..updates import UpdateChecker, valid_ref

log = logging.getLogger(__name__)

router = APIRouter(tags=["node"])
_write_auth = [Depends(require_operator_session)]


def checker_for(app: FastAPI) -> UpdateChecker:
    """This agent's one checker, made on first use."""
    found = getattr(app.state, "update_checker", None)
    if found is None:
        state = app.state.agent_state
        found = UpdateChecker(
            setting=state.get_config,
            default_channel=lambda: UpdateChannel(default_update_channel()),
            settling=state.update_channel_settling,
            settle=lambda channel: state.settle_update_channel(channel.value),
        )
        app.state.update_checker = found
    return found


def checker(request: Request) -> UpdateChecker:
    return checker_for(request.app)


def _prefix(request: Request):  # type: ignore[no-untyped-def]
    return install_info.install_prefix(request.app.state.settings.config_file)


def _view(request: Request, install: NodeInstall) -> NodeUpdate:
    check = checker(request)
    result = check.current()
    target = result.newest if result is not None else None
    prefix = _prefix(request)
    ready = install.mechanism.value == "systemd_system" and update_apply.system_unit_ready()
    return check.view(
        install,
        apply=update_apply.plan(install, target, system_unit_ready=ready),
        running=update_apply.running(prefix),
        last=update_apply.last(prefix) or update_apply.expired_run(prefix),
    )


def node_update_view(request: Request) -> tuple[NodeInstall, NodeUpdate]:
    """What `GET /v1/node` reports as `install` and `update`. Blocking reads
    (a service or task lookup on Windows): call from a worker thread."""
    install = install_info.describe()
    return install, _view(request, install)


def _problem(code: int, slug: str, title: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/agent#{slug}",
            title=title,
            status=code,
            detail=detail,
            component="agent",
        ).model_dump(exclude_none=True),
    )


@router.post(
    "/v1/node/update/check",
    response_model=NodeUpdate,
    response_model_exclude_none=True,
    dependencies=_write_auth,
)
async def check_now(request: Request) -> NodeUpdate:
    install = await asyncio.to_thread(install_info.describe)
    await checker(request).check(install)
    return await asyncio.to_thread(_view, request, install)


@router.post(
    "/v1/node/update",
    response_model=UpdateRun,
    response_model_exclude_none=True,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=_write_auth,
)
async def update_now(request: Request, body: UpdateRequest) -> UpdateRun:
    install = await asyncio.to_thread(install_info.describe)
    view = await asyncio.to_thread(_view, request, install)
    if not view.apply.possible:
        steps = " ".join(
            s.text + (f" {s.command}" if s.command else "") for s in view.apply.steps or []
        )
        raise _problem(
            409,
            "update-not-possible-here",
            "This install cannot update itself",
            f"{view.apply.reason or ''} {steps}".strip(),
        )
    if view.running is not None:
        raise _problem(
            409,
            "update-running",
            "An update is already running",
            f"The update to {view.running.target[:12]} started at "
            f"{view.running.startedAt:%H:%M} UTC is still running on this machine.",
        )
    result = checker(request).current()
    target = result.newest if result is not None else None
    if target is None:
        raise _problem(
            409,
            "update-not-checked",
            "Nothing to update to yet",
            result.error
            if result is not None and result.error
            else "This machine has not checked its channel yet. Check for updates first.",
        )
    # **Only what this agent itself found.** The caller names the target it
    # was shown, so a click is never an update to something the person did
    # not see -- and a caller cannot name anything else at all.
    if body.target != target.ref or not valid_ref(body.target):
        raise _problem(
            409,
            "update-target-moved",
            "That is not the newest version any more",
            f"The newest on {target.channel.value} is now {target.release or target.ref[:12]}. "
            "Check again and look before updating.",
        )
    if not view.available:
        name = target.release or target.ref[:12]
        if view.ahead:
            # Never a downgrade dressed as an update (2026-09-30).
            raise _problem(
                409,
                "update-nothing-newer",
                "This machine is newer",
                f"{name} is older than what this machine runs in "
                f"{', '.join(view.ahead)}, so installing it would move those back. "
                "Nothing is installed.",
            )
        raise _problem(
            409,
            "update-nothing-newer",
            "Already up to date",
            f"This machine already runs {name}.",
        )
    try:
        return await asyncio.to_thread(
            update_apply.start, install=install, target=target, prefix=_prefix(request)
        )
    except update_apply.UpdateRefused as exc:
        raise _problem(
            409, "update-not-started", "The update could not be started", str(exc)
        ) from exc
