"""GET /v1/node — this host, as its own agent sees it.

The install-wide `Node` view lives on the control root; this is the
half only the host itself can answer: what it can compute on, with live
memory, and (from M5) whether it is enrolled and to whom. The
enrollment half is still M5 debt — this agent reports `enrolled: false`
and no control root — but the device inventory is real, because M6's
admission needs it on the host that spawns, and the control root's
node poll has been asking for it since M5 against agents that 404'd.

Detection shells out to a vendor tool, so it runs in a worker thread
rather than on the event loop. Tests inject a detector on
`app.state.device_detector`.
"""

from __future__ import annotations

import asyncio
import platform
import sys

from fastapi import APIRouter, Depends, Request

from .. import __version__
from .._generated.models import Arch, NodeIdentity, Os
from ..dependencies import require_operator_or_service
from ..engines.devices import detect_devices

router = APIRouter(tags=["node"])

_read_auth = [Depends(require_operator_or_service)]


def _os() -> Os | None:
    if sys.platform == "win32":
        return Os.windows
    if sys.platform == "darwin":
        return Os.macos
    if sys.platform.startswith("linux"):
        return Os.linux
    return None


def _arch() -> Arch | None:
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        return Arch.x64
    if machine in ("arm64", "aarch64"):
        return Arch.arm64
    return None


@router.get(
    "/v1/node",
    response_model=NodeIdentity,
    response_model_exclude_none=True,
    dependencies=_read_auth,
)
async def get_node(request: Request) -> NodeIdentity:
    detector = getattr(request.app.state, "device_detector", None) or detect_devices
    snapshot = await asyncio.to_thread(detector)
    return NodeIdentity(
        enrolled=False,
        os=_os(),
        arch=_arch(),
        devices=list(snapshot.devices),
        agentVersion=__version__,
    )
