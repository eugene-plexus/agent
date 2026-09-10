"""This host in an install: `GET /v1/node`, `POST /v1/node/enroll`,
`POST /v1/node/rekey`.

The agent's half of M5's exchange, built at M7. The control root's half
has existed since the `control` repo did; its 85 tests ran against fake
agents that did all of this, and no real agent did any of it.

**Enroll** (operator-only): generate the node keypair if there is none,
work out where other hosts reach this agent, present the join token and
the *public* key to the control root, and take back a name, the epoch,
the install's signing key and the root's identity. Then adopt that key
for this agent's own token verification and restart every supervised
component so they pick it up — from that moment a token minted anywhere
in the install verifies here, which is the property M5 §1 named as the
whole problem.

**Re-key** (no bearer; the credential is a signature): the control root
rotates the install key, or announces a new epoch after a promotion, by
sending a message signed with the identity this node recorded at
enrollment. A lower epoch is refused — that refusal *is* epoch fencing —
and so is a replayed key generation.

Device detection shells out to a vendor tool, so it runs in a worker
thread rather than on the event loop. Tests inject a detector on
`app.state.device_detector` and an httpx transport on
`app.state.control_transport`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import platform
import socket
import sys
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status

from .. import __version__
from .._generated.common_models import Problem
from .._generated.models import Arch, EnrollRequest, NodeIdentity, Os, RekeyRequest
from ..auth_state import AuthState
from ..dependencies import require_operator_or_service, require_operator_session
from ..engines.devices import DeviceSnapshot, detect_devices
from ..node_identity import (
    FencedError,
    NodeIdentityStore,
    derive_advertise_host,
    effective_advertise_url,
    format_url,
    rekey_message,
    verify_rekey_signature,
)
from ..state import AgentState

log = logging.getLogger(__name__)

router = APIRouter(tags=["node"])

_read_auth = [Depends(require_operator_or_service)]
_write_auth = [Depends(require_operator_session)]

# The control root answers enrollment from memory plus one log append;
# anything slower is the root being down, which is a 502 here either way.
_ENROLL_TIMEOUT_SECONDS = 15.0


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


def _store(request: Request) -> NodeIdentityStore:
    store: NodeIdentityStore = request.app.state.node_identity
    return store


async def _devices(request: Request) -> DeviceSnapshot:
    detector = getattr(request.app.state, "device_detector", None) or detect_devices
    snapshot: DeviceSnapshot = await asyncio.to_thread(detector)
    return snapshot


def _identity(request: Request, snapshot: DeviceSnapshot) -> NodeIdentity:
    """The identity as it stands, plus the live inventory."""
    record = _store(request).record
    state: AgentState = request.app.state.agent_state
    advertise = effective_advertise_url(state.get_config("advertiseUrl"), record.advertise_url)
    return NodeIdentity(
        enrolled=record.enrolled,
        name=record.name if record.enrolled else None,
        publicKey=record.public_key,
        controlUrl=record.control_url if record.enrolled else None,  # type: ignore[arg-type]
        epoch=record.epoch if record.enrolled else None,
        advertiseUrl=advertise,  # type: ignore[arg-type]
        signingKeyId=record.signing_key_id if record.enrolled else None,
        controlPublicKey=record.control_public_key if record.enrolled else None,
        os=_os(),
        arch=_arch(),
        devices=list(snapshot.devices),
        agentVersion=__version__,
    )


async def _restart_children(request: Request, *, why: str) -> None:
    """Every supervised component reads the signing key from its
    environment at spawn, so a key this agent just adopted reaches them
    only through a respawn. Engines are not touched: they have no auth."""
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None:
        return
    try:
        restarted = await supervisor.restart_all()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("could not restart supervised components after %s: %s", why, exc)
        return
    if restarted:
        log.info(
            "%s; restarting %d supervised component(s) so they pick up the signing key: %s",
            why,
            len(restarted),
            ", ".join(restarted),
        )


def _decode_signing_key(value: str) -> bytes | None:
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) == 32 else None


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


@router.get(
    "/v1/node",
    response_model=NodeIdentity,
    response_model_exclude_none=True,
    dependencies=_read_auth,
)
async def get_node(request: Request) -> NodeIdentity:
    return _identity(request, await _devices(request))


@router.post(
    "/v1/node/enroll",
    response_model=NodeIdentity,
    response_model_exclude_none=True,
    dependencies=_write_auth,
)
async def enroll_with_control(request: Request, body: EnrollRequest) -> NodeIdentity:
    """Join an install. See the module docstring for the order and why.

    **The operator's own session on this agent stops verifying the moment
    this succeeds**, because the key it was signed with has been replaced
    by the install's. Log in again — here or at the control root; both
    now mint tokens this agent accepts. The same price a rotation charges
    at the control root, for the same reason.

    The advertise URL is the `advertiseUrl` config field when the
    operator set one; otherwise it is derived from the local end of a TCP
    connection to the control root, which on a mesh VPN is the interface
    the root can reach back on. When neither is available the enrollment
    still goes ahead **without** a URL rather than with a loopback one: a
    loopback address recorded at the control root for a remote node
    would make the root probe itself, which is worse than a node it
    knows it cannot reach.
    """
    store = _store(request)
    state: AgentState = request.app.state.agent_state
    auth: AuthState = request.app.state.auth_state
    settings = request.app.state.settings

    if store.record.enrolled:
        held = store.record
        raise _problem(
            status.HTTP_409_CONFLICT,
            "already-enrolled",
            "Already enrolled",
            f"This agent is enrolled as {held.name!r} with the control root at "
            f"{held.control_url}. Re-enrolling elsewhere is a deliberate act: revoke it there "
            f"first (DELETE /v1/nodes/{held.name}), which rotates the install's signing key, "
            f"then remove node.yaml beside agent.yaml and restart this agent.",
        )

    record = store.ensure_keypair()
    control_url = str(body.controlUrl).rstrip("/")

    advertise = effective_advertise_url(state.get_config("advertiseUrl"), None)
    if advertise is None:
        host = await asyncio.to_thread(derive_advertise_host, control_url)
        if host is not None:
            advertise = format_url(host, int(settings.bind_port))
            log.info("derived advertise address %s from the route to %s", advertise, control_url)
        else:
            log.warning(
                "enrolling with no advertise address: the control root will record this node "
                "without a URL and report it unreachable. Set `advertiseUrl` in the agent "
                "config and re-enroll."
            )

    snapshot = await _devices(request)
    name = (body.name or "").strip() or socket.gethostname()
    host_os = _os()
    host_arch = _arch()
    payload: dict[str, Any] = {
        "token": body.token,
        "name": name,
        "publicKey": record.public_key,
        "agentVersion": __version__,
        "os": host_os.value if host_os is not None else None,
        "arch": host_arch.value if host_arch is not None else None,
        "devices": [d.model_dump(exclude_none=True, mode="json") for d in snapshot.devices],
    }
    if advertise is not None:
        payload["url"] = advertise

    transport = getattr(request.app.state, "control_transport", None)
    try:
        async with httpx.AsyncClient(
            timeout=_ENROLL_TIMEOUT_SECONDS, transport=transport
        ) as client:
            response = await client.post(f"{control_url}/v1/nodes/enroll", json=payload)
    except httpx.HTTPError as exc:
        raise _problem(
            status.HTTP_502_BAD_GATEWAY,
            "control-root-unreachable",
            "Control root unreachable",
            f"Could not reach the control root at {control_url}: {exc}. Nothing was recorded; "
            f"this agent is still unenrolled.",
        ) from exc

    if response.status_code != 201:
        raise _problem(
            status.HTTP_502_BAD_GATEWAY,
            "control-root-refused",
            "Control root refused the enrollment",
            f"The control root at {control_url} answered {response.status_code}: "
            f"{_detail(response)}. Nothing was recorded; this agent is still unenrolled.",
        )

    try:
        enrollment = response.json()
    except ValueError:
        enrollment = None
    if not isinstance(enrollment, dict):
        raise _malformed(control_url, "the body was not a JSON object")
    granted_name = enrollment.get("name")
    epoch = enrollment.get("epoch")
    signing_key_b64 = enrollment.get("signingKey")
    if not isinstance(granted_name, str) or not granted_name:
        raise _malformed(control_url, "`name` is missing")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise _malformed(control_url, f"`epoch` is {epoch!r}")
    signing_key = _decode_signing_key(signing_key_b64) if isinstance(signing_key_b64, str) else None
    if signing_key is None:
        raise _malformed(control_url, "`signingKey` is not 32 base64 bytes")

    store.record_enrollment(
        name=granted_name,
        control_url=control_url,
        epoch=epoch,
        signing_key=str(signing_key_b64),
        signing_key_id=_str_or_none(enrollment.get("signingKeyId")),
        control_public_key=_str_or_none(enrollment.get("controlPublicKey")),
        recovery_public_key=_str_or_none(enrollment.get("recoveryPublicKey")),
        advertise_url=advertise,
    )
    auth.set_signing_key(signing_key)
    log.info(
        "enrolled as %r with the control root at %s at epoch %d; adopted the install's signing "
        "key (generation %s)",
        granted_name,
        control_url,
        epoch,
        enrollment.get("signingKeyId"),
    )
    await _restart_children(request, why="enrolled and adopted the install's signing key")
    return _identity(request, snapshot)


@router.post(
    "/v1/node/rekey",
    response_model=NodeIdentity,
    response_model_exclude_none=True,
)
async def rekey_node(request: Request, body: RekeyRequest) -> NodeIdentity:
    """Take a new signing key, or a new epoch, from the control root.

    No bearer dependency, deliberately: the credential is the signature,
    verified against the `controlPublicKey` recorded at enrollment. See
    `agent.yaml` for why a bearer cannot do this job.
    """
    store = _store(request)
    auth: AuthState = request.app.state.auth_state
    record = store.record

    if not record.enrolled or not record.control_public_key:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "not-enrolled",
            "Not enrolled",
            "This agent has not enrolled with a control root, so it has no root identity to "
            "verify a re-key against.",
        )

    message = rekey_message(
        signing_key=body.signingKey, signing_key_id=body.signingKeyId, epoch=int(body.epoch)
    )
    if not verify_rekey_signature(
        control_public_key=record.control_public_key, message=message, signature=body.signature
    ):
        log.warning("refused a re-key whose signature did not verify against the control root")
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "signature-rejected",
            "Signature rejected",
            "The re-key is not signed by the control root this agent enrolled with.",
        )

    signing_key = _decode_signing_key(body.signingKey)
    if signing_key is None:
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "malformed-signing-key",
            "Malformed signing key",
            "`signingKey` must be 32 bytes, base64.",
        )

    try:
        changed = store.accept_rekey(
            signing_key=body.signingKey, signing_key_id=body.signingKeyId, epoch=int(body.epoch)
        )
    except FencedError as exc:
        log.warning("fenced a re-key: %s", exc)
        raise _problem(status.HTTP_409_CONFLICT, "fenced", "Fenced", str(exc)) from exc

    if changed:
        auth.set_signing_key(signing_key)
        log.warning(
            "signing key rotated to generation %s at epoch %d", body.signingKeyId, body.epoch
        )
        await _restart_children(
            request, why=f"the install's signing key rotated to generation {body.signingKeyId}"
        )
    else:
        log.info("epoch %d acknowledged; signing key unchanged", body.epoch)

    return _identity(request, await _devices(request))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict):
            return str(detail.get("detail") or detail.get("title") or detail)
        if detail:
            return str(detail)
    return response.text[:300]


def _malformed(control_url: str, what: str) -> HTTPException:
    return _problem(
        status.HTTP_502_BAD_GATEWAY,
        "control-root-malformed",
        "Control root answered with a malformed enrollment",
        f"The control root at {control_url} answered 201 but {what}. Nothing was recorded.",
    )


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
