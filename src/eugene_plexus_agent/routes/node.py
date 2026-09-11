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
import sys

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status

from .. import __version__, security
from .._generated.common_models import Problem
from .._generated.models import (
    Arch,
    EnrollRequest,
    NodeIdentity,
    Os,
    RekeyRequest,
    UnenrollRequest,
    UnenrollResult,
)
from ..auth_state import AuthState
from ..dependencies import require_operator_or_service, require_operator_session
from ..engines.devices import DeviceSnapshot, detect_devices
from ..enrollment import (
    EnrollmentError,
    perform_enrollment,
    problem_detail,
    resolve_advertise_url,
)
from ..node_identity import (
    FencedError,
    NodeIdentityStore,
    effective_advertise_url,
    rekey_message,
    verify_rekey_signature,
)
from ..state import AgentState

log = logging.getLogger(__name__)

router = APIRouter(tags=["node"])

_read_auth = [Depends(require_operator_or_service)]
_write_auth = [Depends(require_operator_session)]

# Revoking at the root rotates the install's signing key across every
# remaining node, so it is slower than a read and worth waiting for —
# but not worth blocking a detach on, which is why exceeding it still
# leaves this node un-enrolled with `controlNotified: false`.
_REVOKE_TIMEOUT_SECONDS = 20.0


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
        signingPublicKey=record.signing_public_key,
        advertiseSequence=record.advertise_sequence,
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
    by the install's. Log in again - here or at the control root; both
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

    The protocol itself lives in `enrollment.py`, because
    `eugene-plexus-agent join` runs it with no app around it.
    """
    store = _store(request)
    state: AgentState = request.app.state.agent_state
    auth: AuthState = request.app.state.auth_state
    settings = request.app.state.settings

    control_url = str(body.controlUrl).rstrip("/")
    advertise = await resolve_advertise_url(
        configured=state.get_config("advertiseUrl"),
        control_url=control_url,
        bind_port=int(settings.bind_port),
    )
    if advertise is None:
        log.warning(
            "enrolling with no advertise address: the control root will record this node "
            "without a URL and report it unreachable. Set `advertiseUrl` in the agent "
            "config and re-enroll."
        )

    snapshot = await _devices(request)
    try:
        outcome = await perform_enrollment(
            store=store,
            control_url=control_url,
            token=body.token,
            name=body.name,
            advertise_url=advertise,
            devices=[d.model_dump(exclude_none=True, mode="json") for d in snapshot.devices],
            transport=getattr(request.app.state, "control_transport", None),
        )
    except EnrollmentError as exc:
        raise _from_enrollment_error(exc) from exc

    auth.set_signing_key(outcome.signing_key)
    log.info(
        "enrolled as %r with the control root at %s at epoch %d; adopted the install's signing "
        "key (generation %s)",
        outcome.name,
        control_url,
        outcome.epoch,
        outcome.signing_key_id,
    )
    await _restart_children(request, why="enrolled and adopted the install's signing key")
    return _identity(request, snapshot)


@router.post(
    "/v1/node/unenroll",
    response_model=UnenrollResult,
    response_model_exclude_none=True,
    dependencies=_write_auth,
)
async def unenroll_node(request: Request, body: UnenrollRequest | None = None) -> UnenrollResult:
    """Leave an install - the exact inverse of enrolling.

    **Why this is safe to allow from the node**, which is the only part
    that needs an argument: revocation exists because a node that still
    *holds* the signing key can still authenticate, so removing a registry
    entry alone does nothing. Un-enrolling **discards** that key. A node
    cannot escape revocation this way; it can only disarm itself.

    **It proceeds when the root is unreachable.** An operator detaching a
    node from a dead install is precisely the case where refusing is
    useless - `degraded-mode-required`, applied to a trust operation.
    `controlNotified` is how the caller learns the install still lists
    this node and still trusts the key it just threw away.

    The root is told first, forwarding the caller's own bearer: one
    install, one signing key, so the operator session that authorized
    this call is an operator session at the root too. Revoking there
    rotates the key for every remaining node, which is the entire point
    of revoking rather than deleting.

    Like enrollment, this logs out every session on this node, because
    the key they were signed with is gone.
    """
    store = _store(request)
    auth: AuthState = request.app.state.auth_state
    record = store.record

    if not record.enrolled:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "not-enrolled",
            "Not enrolled",
            "This agent is not enrolled with any control root, so there is nothing to leave.",
        )

    previous_name = str(record.name)
    previous_control_url = str(record.control_url)
    notify = True if body is None or body.notifyControl is None else bool(body.notifyControl)

    notified = False
    detail: str | None = None
    if not notify:
        detail = (
            "notifyControl was false, so the control root was not told. It still lists this "
            f"node as {previous_name!r} and still trusts the signing key this node just "
            f"discarded; revoke it there (DELETE /v1/nodes/{previous_name}) when you can."
        )
    else:
        notified, detail = await _revoke_at_root(
            request, control_url=previous_control_url, name=previous_name
        )

    store.unenroll()
    auth.set_signing_key(security.generate_signing_key())
    log.warning(
        "left the install at %s (was %r); discarded its signing key and minted a local one%s",
        previous_control_url,
        previous_name,
        "" if notified else "; THE CONTROL ROOT WAS NOT TOLD",
    )
    await _restart_children(request, why="left the install and returned to a local signing key")

    return UnenrollResult(
        identity=_identity(request, await _devices(request)),
        controlNotified=notified,
        previousName=previous_name,
        previousControlUrl=previous_control_url,  # type: ignore[arg-type]
        detail=detail,
    )


async def _revoke_at_root(
    request: Request, *, control_url: str, name: str
) -> tuple[bool, str | None]:
    """Ask the root to revoke this node, with the caller's own credential.

    Returns `(notified, detail)` and never raises: every failure here is
    survivable, and the one thing that must not happen is a node that
    stays attached because its dead root could not be reached.
    """
    credentials = request.headers.get("authorization")
    if not credentials:
        return False, (
            "No Authorization header to forward, so the control root was not told. "
            f"Revoke this node there: DELETE /v1/nodes/{name}."
        )
    transport = getattr(request.app.state, "control_transport", None)
    target = f"{control_url.rstrip('/')}/v1/nodes/{name}"
    try:
        async with httpx.AsyncClient(
            timeout=_REVOKE_TIMEOUT_SECONDS, transport=transport
        ) as client:
            response = await client.delete(target, headers={"Authorization": credentials})
    except httpx.HTTPError as exc:
        return False, (
            f"Could not reach the control root at {control_url}: {exc}. This node has left "
            f"anyway; revoke it there (DELETE /v1/nodes/{name}) so the install rotates its "
            f"signing key."
        )
    if response.status_code == 404:
        # Already gone from the registry. Nothing is owed, and reporting
        # this as "not notified" would send an operator chasing a node
        # the root has never heard of.
        return True, None
    if response.status_code not in (200, 202, 204):
        return False, (
            f"The control root answered {response.status_code}: {problem_detail(response)}. "
            f"This node has left anyway; revoke it there (DELETE /v1/nodes/{name})."
        )
    return True, None


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


def _from_enrollment_error(exc: EnrollmentError) -> HTTPException:
    """One mapping from the protocol module's failures to HTTP.

    `already-enrolled` is the only 409; everything else is the control
    root being unreachable, refusing, or answering something we cannot
    read, which are all 502 - the failure is upstream of this agent, and
    this agent recorded nothing either way.
    """
    code = (
        status.HTTP_409_CONFLICT if exc.slug == "already-enrolled" else status.HTTP_502_BAD_GATEWAY
    )
    return _problem(code, exc.slug, exc.title, exc.detail)
