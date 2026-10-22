"""This host in an install: `GET /v1/node`, `POST /v1/node/enroll`,
`POST /v1/node/trust-bundle`.

The agent's half of M5's exchange, built at M7, and since 2026-09-25 an
exchange of public keys only (`specs/docs/design/per-node-token-keys.md`).

**Enroll** (operator-only): generate the node's keys if there are none,
work out where other hosts reach this agent, present the join token and
the *public* keys to the control root, and take back a name, the epoch,
the root's identity and a trust bundle that already lists this node's
token key. Then restart every supervised component so they verify
against the install's bundle instead of this node's own.

**Trust bundle** (no bearer; the credential is the signature): the
control root pushes a newer bundle whenever one changes, and this agent
pulls it every minute too. A lower epoch or version is refused — that
refusal *is* epoch fencing and rollback protection.

Device detection shells out to a vendor tool, so it runs in a worker
thread rather than on the event loop. Tests inject a detector on
`app.state.device_detector` and an httpx transport on
`app.state.control_transport`.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from typing import Annotated, Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials

from .. import (
    __version__,
    default_topology,
    host_history,
    leaving,
    machine_port,
    peer,
    reach,
    security,
    standby,
    tokens,
)
from .._generated.common_models import ConfigUpdateRequest, Problem, SignedTrustBundle
from .._generated.models import (
    Arch,
    ClientTls,
    ComponentStatus,
    EnrollRequest,
    HostHistory,
    HostHistoryStepChoice,
    HostMetric,
    LocalStandby,
    MachineKeyChange,
    MachinePort,
    NodeIdentity,
    NodeReach,
    NodeReachRequest,
    NodeReachResult,
    Os,
    ReachStep,
    SignedMachineList,
    Step,
    UnenrollRequest,
    UnenrollResult,
)
from .._http import internal_client
from ..auth_state import AuthState
from ..dependencies import require_operator_or_service, require_operator_session
from ..engines.devices import DeviceSnapshot, detect_devices
from ..enrollment import (
    EnrollmentError,
    announce_address,
    perform_enrollment,
    problem_detail,
    resolve_advertise_url,
)
from ..firewall import FirewallQuery, read_firewall
from ..machine_trust import MachineListError, MachineListFenced, MachineListUnsigned
from ..machine_trust import pull as pull_machine_list
from ..node_identity import (
    FencedError,
    NodeIdentityStore,
    advertise_host,
    effective_advertise_url,
    is_loopback_host,
)
from ..state import AgentState
from ..trust import BundleRollback
from .updates import node_update_view

log = logging.getLogger(__name__)

router = APIRouter(tags=["node"])

_read_auth = [Depends(require_operator_or_service)]
_write_auth = [Depends(require_operator_session)]

# Revoking at the root is one log entry and a bundle push, but it is a
# write on another host and worth waiting for — not worth blocking a
# detach on, which is why exceeding it still leaves this node
# un-enrolled with `controlNotified: false`.
_REVOKE_TIMEOUT_SECONDS = 20.0

# Leaving by passphrase is limited as sign-in is, and in the same count
# (`routes/auth.py`), so the two cannot be alternated for more guesses.
_LEAVE_RATE_WINDOW_SECONDS = 60
_LEAVE_RATE_MAX_FAILURES = 5


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


async def _refresh_machines(app: Any, control_url: str) -> None:
    async with internal_client(timeout=10.0) as client:
        await pull_machine_list(app.state.machine_trust, client, control_url)


def _machine(request: Request) -> MachinePort | None:
    """`NodeIdentity.machine`: this node's machine port, as it stands."""
    door = getattr(request.app.state, "machine_port", None)
    if door is None:
        return None
    trust = getattr(request.app.state, "machine_trust", None)
    from .. import machine_key_change

    running = getattr(request.app.state, "machine_key_change", None)
    # The running change's own view, which keeps a failure node.yaml could
    # not hold (a full disk); the record's otherwise.
    change = (
        running.status()
        if running is not None
        else machine_key_change.status(_store(request).record)
    )
    return MachinePort(
        url=machine_port.machine_url(request.app),  # type: ignore[arg-type]
        port=door.status.port,
        pin=door.status.pin,
        listening=door.status.listening,
        error=door.status.error,
        required=bool(trust is not None and trust.required),
        listAgeSeconds=trust.age_seconds if trust is not None else None,
        listVersion=trust.version if trust is not None else None,
        keyChange=MachineKeyChange.model_validate(change) if change else None,
    )


def _client_tls(request: Request) -> ClientTls | None:
    """`NodeIdentity.clientTls`: this node's client doors and their leaf (LT2)."""
    tls = getattr(request.app.state, "client_tls", None)
    return ClientTls.model_validate(tls.status()) if tls is not None else None


def _identity(request: Request, snapshot: DeviceSnapshot) -> NodeIdentity:
    """The identity as it stands, plus the live inventory."""
    record = _store(request).record
    state: AgentState = request.app.state.agent_state
    advertise = effective_advertise_url(state.get_config("advertiseUrl"), record.advertise_url)
    trust = request.app.state.auth_state.trust
    bundle = trust.bundle if record.enrolled else None
    token_public = tokens.public_b64(trust.signer().key) if record.token_private_key else None
    entry = getattr(request.app.state, "entrypoint_config", None)
    return NodeIdentity(
        entrypoint=entry.public_urls() if entry else None,
        enrolled=record.enrolled,
        hostedSites=(
            [site]
            if (site := getattr(getattr(request.app.state, "site_host", None), "site", None))
            else None
        ),
        name=record.name if record.enrolled else None,
        publicKey=record.public_key,
        controlUrl=record.control_url if record.enrolled else None,  # type: ignore[arg-type]
        controlUrlEpoch=record.control_url_epoch if record.enrolled else None,
        epoch=record.epoch if record.enrolled else None,
        advertiseUrl=advertise,  # type: ignore[arg-type]
        machine=_machine(request),
        clientTls=_client_tls(request),
        trustBundleVersion=bundle.version if bundle is not None else None,
        trustBundleAgeSeconds=trust.heard_age_seconds() if record.enrolled else None,
        tokenPublicKey=token_public,
        controlPublicKey=record.control_public_key if record.enrolled else None,
        signingPublicKey=record.signing_public_key,
        advertiseSequence=record.advertise_sequence,
        os=_os(),
        arch=_arch(),
        devices=list(snapshot.devices),
        agentVersion=__version__,
        hostsControl=standby.hosts_active_control(state),
        standby=_local_standby(request, state, record.control_url),
        # Read at the moment of answering, not cached and not derived
        # from anything: the point of the field is that a console can
        # compare two hosts and see a drift neither host can see about
        # itself. `tokens.note_clock_skew` observes the same quantity
        # precisely and can only write it to a log.
        time=datetime.now(UTC),
    )


def _local_standby(request: Request, state: AgentState, control_url: Any) -> LocalStandby | None:
    """The warm standby this agent runs, if its node holds the grant. A
    promoted copy is not a standby: it is the root, and `hostsControl`
    says so (row 4)."""
    entry = state.get_topology_entry(standby.STANDBY_COMPONENT)
    if entry is None or not standby.is_standby(entry) or standby.promoted(state):
        return None
    supervisor = getattr(request.app.state, "supervisor", None)
    live = (
        supervisor.status_for(entry.name, has_spawn=True)[0]
        if supervisor is not None
        else ComponentStatus.unreachable
    )
    return LocalStandby(
        component=entry.name,
        url=entry.url,
        status=live,
        following=control_url,
    )


async def _restart_children(request: Request, *, why: str) -> None:
    await leaving.restart_children(request.app, why=why)


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
    """This host, its devices, and whether anything else can get to it.

    `reach` is assembled here rather than on the identity helper because
    the other three callers of `_identity` -- enroll, unenroll, trust-bundle --
    are each in the middle of a trust operation and none of them wants a
    firewall read on the way out. It goes in a thread: the firewall read
    is a COM enumeration on Windows and two subprocesses on POSIX, and
    neither belongs on the event loop.
    """
    identity = _identity(request, await _devices(request))
    identity.reach = await asyncio.to_thread(
        _reach_view,
        request,
        advertise=str(identity.advertiseUrl) if identity.advertiseUrl else None,
    )
    # What is installed and whether it is behind: from the code itself and
    # the last check, never a network call here -- every console polls this.
    identity.install, identity.update = await asyncio.to_thread(node_update_view, request)
    return identity


@router.post(
    "/v1/node/enroll",
    response_model=NodeIdentity,
    response_model_exclude_none=True,
    dependencies=_write_auth,
)
async def enroll_with_control(request: Request, body: EnrollRequest) -> NodeIdentity:
    """Join an install. See the module docstring for the order and why.

    **The operator's own session on this agent stops verifying the moment
    this succeeds**, because this agent now trusts the install's bundle
    and not its own. Sign in again: the sign-in forwards to the control
    root, which is the only thing that mints sessions in an install.

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
            label=body.label,
            advertise_url=advertise,
            devices=[d.model_dump(exclude_none=True, mode="json") for d in snapshot.devices],
            transport=getattr(request.app.state, "control_transport", None),
            machine=machine_port.announced(request.app),
        )
    except EnrollmentError as exc:
        raise _from_enrollment_error(exc) from exc

    auth.trust.load()
    # The table at once: the root's key from the join token carries the
    # calls that follow, and the root's signed list comes with the first.
    trust_machines = request.app.state.machine_trust
    trust_machines.rebuild()
    async with internal_client(timeout=10.0) as client:
        await pull_machine_list(trust_machines, client, control_url)
    # Joining answers the one onboarding question, so nothing is left
    # for the wizard to do. Set here rather than only at the next boot
    # because enrolling does not restart this process -- without it the
    # operator who just enrolled gets sent to first-run setup.
    default_topology.mark_onboarded(state)
    log.info(
        "enrolled as %r with the control root at %s at epoch %d; trust bundle %d lists %d keys",
        outcome.name,
        control_url,
        outcome.epoch,
        outcome.bundle.version,
        len(outcome.bundle.keys),
    )
    await _restart_children(request, why="enrolled and now trusts the install's bundle")
    return _identity(request, snapshot)


@router.post(
    "/v1/node/unenroll",
    response_model=UnenrollResult,
    response_model_exclude_none=True,
)
async def unenroll_node(request: Request, body: UnenrollRequest | None = None) -> UnenrollResult:
    """Leave an install - the exact inverse of enrolling.

    **Why this is safe to allow from the node**: the install stops
    trusting this node when the root drops its key from the bundle.
    Un-enrolling only makes this node stop trusting the install. A node
    cannot escape revocation this way; it can only disarm itself.

    **It proceeds when the root is unreachable.** An operator detaching a
    node from a dead install is precisely the case where refusing is
    useless - `degraded-mode-required`, applied to a trust operation.
    `controlNotified` is how the caller learns the install still lists
    this node and still trusts its key.

    The root is told first, forwarding the caller's own session, which is
    addressed to the root as well as to this machine. Revoking there drops
    this node's key from the bundle every other machine holds.

    Like enrollment, this logs out every session on this node, because
    this node no longer trusts the authority that signed them.

    **With no `Authorization` header and a `passphrase`**, this machine's
    own passphrase stands in for the session a removed machine cannot get
    (agent#21; `_leave_by_passphrase`).
    """
    passphrase = body.passphrase if body is not None else None
    if passphrase is not None and not request.headers.get("authorization"):
        return await _leave_by_passphrase(request, passphrase)
    require_operator_session(request, _credentials(request))

    store = _store(request)
    record = store.record

    if not record.enrolled:
        raise _not_enrolled()
    if standby.hosts_active_control(request.app.state.agent_state):
        raise _runs_the_root()

    previous_name = str(record.name)
    previous_control_url = str(record.control_url)
    notify = True if body is None or body.notifyControl is None else bool(body.notifyControl)

    notified = False
    detail: str | None = None
    if not notify:
        detail = (
            "notifyControl was false, so the control root was not told. It still lists this "
            f"node as {previous_name!r} and still trusts this node's key; revoke it there "
            f"(DELETE /v1/nodes/{previous_name}) when you can."
        )
    else:
        notified, detail = await _revoke_at_root(
            request, control_url=previous_control_url, name=previous_name
        )

    left = await leaving.leave(request.app, notified=notified, why="an operator asked")
    if left is None and store.record.enrolled:
        raise _not_enrolled()
    # None and not enrolled: a trust pull left at the same moment.

    return UnenrollResult(
        identity=_identity(request, await _devices(request)),
        controlNotified=notified,
        previousName=previous_name,
        previousControlUrl=previous_control_url,  # type: ignore[arg-type]
        detail=detail,
    )


def _runs_the_root() -> HTTPException:
    return _problem(
        status.HTTP_409_CONFLICT,
        "runs-the-root",
        "This machine runs the install's control root",
        "It does not leave the install: that would take the install's root off its own "
        "authority. Move the root to another machine first.",
    )


def _not_enrolled() -> HTTPException:
    return _problem(
        status.HTTP_409_CONFLICT,
        "not-enrolled",
        "Not enrolled",
        "This agent is not enrolled with any control root, so there is nothing to leave.",
    )


def _credentials(request: Request) -> HTTPAuthorizationCredentials | None:
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=value.strip())


async def _leave_by_passphrase(request: Request, passphrase: str) -> UnenrollResult:
    """Leave with this machine's own passphrase and no session (agent#21, A).

    A removed machine, or one whose root is gone, has no session its own
    agent accepts, because sessions come from the root. Its own stored
    passphrase is what this machine still trusts, so it stands in, under
    the sign-in rate limit. Then the root is asked for its bundle: a
    bundle that no longer lists this machine makes it leave (as the
    minute's pull would), no answer lets it leave, and a bundle that still
    lists it is refused, because then a session can still be had.

    Leaving can only disarm this machine, never make the install trust
    it; that is why a passphrase is enough.
    """
    state: AgentState = request.app.state.agent_state
    auth: AuthState = request.app.state.auth_state
    remote = (
        peer.peer_of(
            request.client.host if request.client else None,
            request.headers.get(peer.PEER_HEADER),
        )
        or "unknown"
    )
    record = _store(request).record
    if not record.enrolled:
        raise _not_enrolled()
    if auth.is_login_rate_limited(
        remote,
        window_seconds=_LEAVE_RATE_WINDOW_SECONDS,
        max_in_window=_LEAVE_RATE_MAX_FAILURES,
    ):
        raise _problem(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate-limited",
            "Rate limited",
            f"Too many failed attempts from {remote}. Wait {_LEAVE_RATE_WINDOW_SECONDS} "
            "seconds and try again.",
        )
    stored = state.get_passphrase_hash()
    if stored is None:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "no-own-passphrase",
            "This machine has no passphrase of its own",
            "It joined the install without one, so it cannot leave by passphrase. It leaves "
            "by itself within a minute of the control root no longer listing it. While the "
            "install still lists it, sign in and leave with that session.",
        )
    if not security.verify_passphrase(passphrase, stored):
        auth.record_login_failure(
            remote,
            window_seconds=_LEAVE_RATE_WINDOW_SECONDS,
            max_in_window=_LEAVE_RATE_MAX_FAILURES,
        )
        log.warning("failed leave-by-passphrase attempt from %s", remote)
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "wrong-passphrase",
            "Wrong passphrase",
            "That is not this machine's passphrase. Repeated failures from one source are "
            "rate-limited.",
        )
    auth.clear_login_failures(remote)

    previous_name = str(record.name)
    previous_control_url = str(record.control_url)
    if standby.hosts_active_control(state):
        raise _runs_the_root()
    heard = await leaving.ask_now(request.app)
    if heard is leaving.Heard.UNTRUSTED:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "untrusted-answer",
            "The control root's answer was refused",
            f"The control root at {previous_control_url} answered with a trust bundle this "
            "machine does not take (the agent's log says why), so it cannot tell whether the "
            "install still lists this machine and does not leave by passphrase.",
        )
    if heard is leaving.Heard.LISTED:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "still-listed",
            "The install still lists this machine",
            f"The control root at {previous_control_url} still lists this machine as "
            f"{previous_name!r}. Sign in and leave with that session, or remove the machine "
            "from Machines on the install first.",
        )
    if heard is leaving.Heard.LEFT:
        notified, detail = True, "The install had already removed this machine."
    else:
        left = await leaving.leave(
            request.app,
            notified=False,
            why="its own passphrase was given while the control root did not answer",
        )
        if left is None and _store(request).record.enrolled:
            raise _not_enrolled()
        notified = False
        detail = (
            f"The control root at {previous_control_url} did not answer, so it was not told. "
            f"If the install still lists this machine as {previous_name!r}, remove it there "
            f"(DELETE /v1/nodes/{previous_name})."
        )
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
        async with internal_client(timeout=_REVOKE_TIMEOUT_SECONDS, transport=transport) as client:
            response = await client.delete(target, headers={"Authorization": credentials})
    except httpx.HTTPError as exc:
        return False, (
            f"Could not reach the control root at {control_url}: {exc}. This node has left "
            f"anyway; revoke it there (DELETE /v1/nodes/{name}) so the install drops its key."
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
    "/v1/node/trust-bundle",
    response_model=NodeIdentity,
    response_model_exclude_none=True,
)
async def take_trust_bundle(request: Request, body: SignedTrustBundle) -> NodeIdentity:
    """Take a newer trust bundle, or a new epoch, from the control root.

    No bearer dependency, deliberately: the credential is the signature,
    checked against the `controlPublicKey` pinned at enrollment. A newer
    bundle from the same authority restarts nothing; children reload the
    file.
    """
    auth: AuthState = request.app.state.auth_state
    if not _store(request).record.enrolled:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "not-enrolled",
            "Not enrolled",
            "This agent has not enrolled with a control root, so it has no root identity to "
            "check a trust bundle against.",
        )
    try:
        bundle = auth.trust.accept(body.jws)
    except tokens.BundleError as exc:
        log.warning("refused a trust bundle: %s", exc)
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "signature-rejected",
            "Signature rejected",
            f"The trust bundle is not signed by the control root this agent enrolled with: {exc}",
        ) from exc
    except (FencedError, BundleRollback) as exc:
        log.warning("fenced a trust bundle: %s", exc)
        raise _problem(status.HTTP_409_CONFLICT, "fenced", "Fenced", str(exc)) from exc
    log.info("took trust bundle %d at epoch %d", bundle.version, bundle.epoch)
    await standby.reconcile(request.app)
    # A new bundle is a machine joining or leaving, so the machine list is
    # fetched at once rather than at the next minute (LT1). In the
    # background: the root is waiting for this answer.
    record = _store(request).record
    if record.control_url:
        asyncio.get_running_loop().create_task(_refresh_machines(request.app, record.control_url))
    return _identity(request, await _devices(request))


@router.post(
    "/v1/node/machine-list",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def take_machine_list(request: Request, body: SignedMachineList) -> Response:
    """Take a newer machine list from the control root, and follow a
    promoted root to its address (row 4).

    The mirror of `take_trust_bundle`: no bearer, because the credential is
    the signature, checked against the root identity pinned at enrolment.
    `MachineTrust.accept` judges it: 409 for a list older than the one held
    or from a superseded root, and a move only for a list at an epoch above
    the one this node's root address came with (R4-MOVE). The move's own
    calls (announcing here, the bundle there) run beside this answer, which
    the root is waiting for."""
    if not _store(request).record.enrolled:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "not-enrolled",
            "Not enrolled",
            "This agent has not enrolled with a control root, so it has no root identity to "
            "check a machine list against.",
        )
    try:
        request.app.state.machine_trust.accept(body.jws)
    except MachineListFenced as exc:
        log.warning("refused a machine list: %s", exc)
        raise _problem(
            status.HTTP_409_CONFLICT, "fenced", "Fenced", f"Refused machine list: {exc}."
        ) from exc
    except FencedError as exc:
        raise _problem(status.HTTP_409_CONFLICT, "fenced", "Fenced", str(exc)) from exc
    except MachineListUnsigned as exc:
        log.warning("refused a machine list: %s", exc)
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "signature-rejected",
            "Signature rejected",
            f"Refused machine list: {exc}.",
        ) from exc
    except MachineListError as exc:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "older",
            "Older than the one held",
            f"Refused machine list: {exc}.",
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------- #
# the machine key: change it with no stop in traffic (row 3)
# --------------------------------------------------------------------------- #


def _key_change(request: Request) -> Any:
    change = getattr(request.app.state, "machine_key_change", None)
    if change is None:
        door = getattr(request.app.state, "machine_port", None)
        raise _problem(
            status.HTTP_409_CONFLICT,
            "machine-port-down",
            "Machine port not running",
            "This machine runs no machine port"
            + (f" ({door.status.error})" if door is not None and door.status.error else "")
            + ", so it presents no key to change.",
        )
    return change


@router.post(
    "/v1/node/machine-key",
    response_model=MachineKeyChange,
    response_model_exclude_none=True,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=_write_auth,
)
async def start_machine_key_change(request: Request) -> MachineKeyChange:
    """Give this machine a new machine key, with no stop in traffic. See
    `machine_key_change` for the order of the steps."""
    from ..machine_key_change import KeyChangeRefused

    try:
        started = await _key_change(request).start()
    except KeyChangeRefused as exc:
        raise _problem(exc.status, exc.slug, exc.title, exc.detail) from exc
    return MachineKeyChange.model_validate(started)


@router.delete(
    "/v1/node/machine-key",
    response_model=MachineKeyChange,
    response_model_exclude_none=True,
    dependencies=_write_auth,
)
async def stop_machine_key_change(request: Request) -> MachineKeyChange:
    """Stop a machine key change before the switch: the next key is deleted."""
    from ..machine_key_change import KeyChangeRefused

    try:
        stopped = await _key_change(request).stop()
    except KeyChangeRefused as exc:
        raise _problem(exc.status, exc.slug, exc.title, exc.detail) from exc
    return MachineKeyChange.model_validate(stopped)


# --------------------------------------------------------------------------- #
# reach: can anything else on the network get here
# --------------------------------------------------------------------------- #


def _listeners(request: Request) -> list[reach.Listener]:
    return listeners_of(request.app)


def listeners_of(app: Any) -> list[reach.Listener]:
    """This install's processes on this host, and the interface each one
    was actually started on.

    The agent's own bind is the value `build_server` handed uvicorn; a
    component's is what the supervisor spawned it with. A component this
    agent declares but is not currently running contributes nothing —
    "what it would bind if it started" is a guess, and this object
    exists to hold facts.
    """
    settings = app.state.settings
    out = [
        reach.Listener(
            process="agent",
            port=int(settings.bind_port),
            bind_host=getattr(app.state, "bind_host", None) or settings.bind_host,
        )
    ]
    # The machine port (LT1): what every other machine reaches this one on.
    door = getattr(app.state, "machine_port", None)
    if door is not None and door.status.listening:
        out.append(
            reach.Listener(
                process="machine port",
                port=door.status.port,
                bind_host=machine_port.bind_host_for(app),
            )
        )
    state: AgentState = app.state.agent_state
    # Apps listen too, on their own ports, and a phone opening a chat app
    # needs the same three things a phone opening the console does.
    manager = getattr(app.state, "apps", None)
    if manager is not None:
        for record in manager.store.installed():
            if not manager.supervisor.is_running(record.id):
                continue
            out.append(
                reach.Listener(
                    process=f"app:{record.id}",
                    port=record.port,
                    bind_host=manager.supervisor.bind_host(record.id),
                )
            )
    supervisor = getattr(app.state, "supervisor", None)
    if supervisor is None:
        return out
    for entry in state.list_topology_entries():
        if entry.spawn is None or not supervisor.is_supervised(entry.name):
            continue
        port = urlparse(str(entry.url)).port
        if port is None:
            continue
        out.append(
            reach.Listener(
                process=str(entry.kind.value if hasattr(entry.kind, "value") else entry.kind),
                port=port,
                bind_host=supervisor.bind_host_for(entry.name),
            )
        )
    return out


def _firewall_program() -> str | None:
    """The executable the firewall would have a rule about.

    `sys.executable`, because that is what listens: every component here
    is `python -m <module>` under this agent's own interpreter, and it is
    that path Windows' *Windows Security Alert* dialog names when
    somebody clicks Allow.
    """
    return sys.executable or None


def _reach_view(request: Request, *, advertise: str | None) -> NodeReach:
    """`NodeReach` as it stands right now.

    Assembled rather than cached. The firewall read is 78 ms on the
    machine it was measured on, the bind values are in memory, and the
    proposed address is half a millisecond of routing-table lookup — so
    there is nothing here worth serving stale, and a stale reach answer
    is worse than a slow one by the same argument that makes the object
    evidence rather than configuration.
    """
    settings = request.app.state.settings
    listeners = _listeners(request)
    bound = reach.bound_addresses(listeners)
    agent_bound = next((b for b in bound if b.process == "agent"), None)
    restart = (getattr(request.app.state, "restart_describer", None) or reach.describe_restart)()
    # A seam, for the reason the library learned the hard way at
    # `test_detail_groups_candidates_and_attaches_the_fit`: a test that
    # reads live hardware asserts about the developer's machine. This
    # one would read the developer's own firewall rules and answer
    # differently on every box and in CI. `conftest` pins it.
    reader = getattr(request.app.state, "firewall_reader", None) or read_firewall
    fw = reader(
        FirewallQuery(
            ports=tuple(b.port for b in bound) or (int(settings.bind_port),),
            program=_firewall_program(),
        )
    )
    witness = getattr(request.app.state, "off_host", None)
    return NodeReach(
        enabled=not is_loopback_host(advertise_host(advertise)),
        advertiseUrl=advertise,  # type: ignore[arg-type]
        proposedUrl=reach.proposed_url(int(settings.bind_port)),  # type: ignore[arg-type]
        boundAddresses=bound,
        restartRequired=reach.restart_required(advertise_url=advertise, agent_bound=agent_bound),
        restart=restart,
        firewall=fw,
        lastReachedFrom=witness.address if witness is not None else None,
        lastReachedAt=witness.at if witness is not None else None,
    )


@router.post(
    "/v1/node/reach",
    response_model=NodeReachResult,
    response_model_exclude_none=True,
    dependencies=_write_auth,
)
async def set_node_reach(request: Request, body: NodeReachRequest) -> NodeReachResult:
    """Turn "reach it from other devices" on or off. See `agent.yaml`.

    Four steps, in this order, each recorded whether or not it worked:
    write the address, tell the control root, restart the components,
    change the firewall. **A step that fails does not roll back the ones
    before it** — a firewall rule that could not be added is not a
    reason to stop advertising, and an operator who is told "nothing
    happened" about a change that half happened is worse off than one
    who is told which half.

    The agent's own restart is a fifth step and is opt-in, because the
    browser making this call is talking to the process that would go
    away.
    """
    state: AgentState = request.app.state.agent_state
    settings = request.app.state.settings
    steps: list[ReachStep] = []

    url: str | None = None
    if body.enabled:
        url = str(body.url).rstrip("/") if body.url else reach.proposed_url(int(settings.bind_port))
        if url is None:
            raise _problem(
                status.HTTP_400_BAD_REQUEST,
                "no-network-address",
                "No address to offer",
                "This machine has no address on a network other than its own loopback, so "
                "there is nothing for another device to connect to. Connect it to a network, "
                "or set the address yourself in the agent's config.",
            )
        if is_loopback_host(advertise_host(url)):
            raise _problem(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "loopback-address",
                "That address is this machine only",
                f"{url} is a loopback address, which is what turning this off means. Give an "
                "address other devices can reach, or turn it off instead.",
            )

    # 1. the setting.
    result = state.apply_config_patch(
        ConfigUpdateRequest(advertiseUrl=url if body.enabled else None)
    )
    if result.rejected:
        steps.append(
            ReachStep(step=Step.advertise, ok=False, detail="; ".join(map(str, result.rejected)))
        )
    else:
        steps.append(
            ReachStep(
                step=Step.advertise,
                ok=True,
                detail=(
                    f"This machine now tells the rest of the install it is at {url}."
                    if body.enabled
                    else "This machine is back to answering only itself."
                ),
            )
        )

    # 2. the control root, if there is one. Same call the config route
    #    makes when the field is edited by hand: an install whose root
    #    holds the old address routes to the old address.
    identity = getattr(request.app.state, "node_identity", None)
    if identity is not None and identity.record.enrolled:
        try:
            await _announce_reach(request)
            steps.append(ReachStep(step=Step.announce, ok=True))
        except Exception as exc:  # pragma: no cover - the announce path logs its own failures
            steps.append(ReachStep(step=Step.announce, ok=False, detail=str(exc)))

    # 3. the components. They take the bind host from their environment
    #    at spawn, so this is the whole of their half.
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is not None:
        try:
            restarted = await supervisor.restart_all()
            steps.append(
                ReachStep(
                    step=Step.restart_components,
                    ok=True,
                    detail=(
                        f"Restarted {', '.join(restarted)}." if restarted else "Nothing to restart."
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            steps.append(ReachStep(step=Step.restart_components, ok=False, detail=str(exc)))

    # 3b. the apps, for the same reason: they take their bind host from
    #     their environment at spawn.
    manager = getattr(request.app.state, "apps", None)
    if manager is not None:
        running = [r for r in manager.store.installed() if manager.supervisor.is_running(r.id)]
        for record in running:
            try:
                await manager.restart(record)
            except Exception:  # pragma: no cover - defensive
                log.warning("could not restart app %s for reach", record.id, exc_info=True)

    # 4. the firewall.
    if body.allowFirewall:
        ports = tuple(sorted({listener.port for listener in _listeners(request)}))
        ok, detail = await asyncio.to_thread(_change_firewall, body.enabled, ports)
        steps.append(ReachStep(step=Step.firewall, ok=ok, detail=detail))

    advertise = effective_advertise_url(
        state.get_config("advertiseUrl"),
        identity.record.advertise_url if identity is not None else None,
    )
    view = await asyncio.to_thread(_reach_view, request, advertise=advertise)

    # 5. this agent, last and only when asked.
    restarted_self = False
    if body.restartAgent:
        restarted_self, detail = reach.spawn_restart(view.restart or reach.describe_restart())
        steps.append(ReachStep(step=Step.restart_agent, ok=restarted_self, detail=detail))

    log.info(
        "reach turned %s: %s",
        "on" if body.enabled else "off",
        "; ".join(f"{s.step.value}={'ok' if s.ok else 'failed'}" for s in steps),
    )
    return NodeReachResult(reach=view, steps=steps, restarted=restarted_self)


def _change_firewall(enabled: bool, ports: tuple[int, ...]) -> tuple[bool, str]:
    """Add or remove this install's firewall allowance, per platform.

    Windows can do it in place when elevated and through a prompt on the
    desktop when it is not. Linux and macOS print the command instead:
    both need root, and the two ways for a web server to have root are a
    password prompt it has no terminal for and a permanent sudoers
    entry. Neither is worth a switch, and
    `easy-default-expert-override` asks for the explanation when the
    easy path is not safely available.
    """
    if sys.platform == "win32":
        from ..firewall import windows

        return windows.add_rule(ports) if enabled else windows.remove_rule()
    if sys.platform.startswith("linux"):
        from ..firewall import linux

        return linux.add_rule(ports) if enabled else linux.remove_rule()
    if sys.platform == "darwin":
        from ..firewall import macos

        program = _firewall_program()
        return macos.add_rule(program) if enabled else macos.remove_rule(program)
    return False, f"This agent cannot change the firewall on {sys.platform}."


async def _announce_reach(request: Request) -> None:
    """Tell the control root this node's address, now rather than at the
    next boot. The same thing `PATCH /v1/config` does when the field is
    edited by hand — kept here rather than imported from the config
    route because the two are one behaviour with two entry points and
    the route module is not the place either of them belongs."""
    identity = _store(request)
    state: AgentState = request.app.state.agent_state
    settings = request.app.state.settings
    if not identity.record.enrolled:
        return
    url = await resolve_advertise_url(
        configured=state.get_config("advertiseUrl"),
        control_url=identity.record.control_url,
        bind_port=int(settings.bind_port),
        persisted=identity.record.advertise_url,
    )
    if url is None:
        return
    identity.record_advertise_url(url)
    await announce_address(
        store=identity,
        url=url,
        transport=getattr(request.app.state, "control_transport", None),
    )


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


# --- host history (specs docs/design/host-history.md) -------------------------


def _history_store(request: Request) -> host_history.HostHistoryStore:
    store = getattr(request.app.state, "host_history", None)
    if not isinstance(store, host_history.HostHistoryStore):
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "host-history-unavailable",
            "Host history is not available",
            "This agent has no history store (it is running in safe mode).",
        )
    return store


@router.get(
    "/v1/node/history",
    response_model=HostHistory,
    response_model_exclude_none=True,
    dependencies=_read_auth,
)
async def get_node_history(
    request: Request,
    from_: Annotated[datetime | None, Query(alias="from")] = None,
    to: datetime | None = None,
    step: HostHistoryStepChoice = HostHistoryStepChoice.auto,
    metric: Annotated[list[HostMetric] | None, Query()] = None,
) -> HostHistory:
    """What this host's hardware did over a window. See `agent.yaml`."""
    store = _history_store(request)
    if not await asyncio.to_thread(store.open):
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "host-history-unavailable",
            "Host history could not be read",
            store.error or "the history store could not be opened",
        )
    state: AgentState = request.app.state.agent_state
    settings = host_history.read_settings(state.get_config)
    try:
        return await asyncio.to_thread(
            store.query,
            now=time.time(),
            settings=settings,
            start=from_.timestamp() if from_ is not None else None,
            end=to.timestamp() if to is not None else None,
            step=step,
            metrics=metric,
        )
    except host_history.HistoryWindowError as exc:
        raise _problem(
            status.HTTP_400_BAD_REQUEST, "host-history-window", "Window refused", str(exc)
        ) from exc


@router.delete("/v1/node/history", status_code=204, dependencies=_write_auth)
async def delete_node_history(request: Request) -> Response:
    """Delete everything recorded. Recording, if on, starts again from empty."""
    store = _history_store(request)
    if not await asyncio.to_thread(store.open):
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "host-history-unavailable",
            "Host history could not be deleted",
            store.error or "the history store could not be opened",
        )
    await asyncio.to_thread(store.clear)
    log.info("host history deleted at an operator's request")
    return Response(status_code=204)


@router.get("/v1/trust/ca", response_class=Response)
async def get_install_ca(
    request: Request, format: str = Query(default="pem", pattern="^(pem|der|bundle)$")
) -> Response:
    """The install CA that signed this node's client leaf (LT2), for a device
    to trust. Public: a certificate is not a secret, and a device fetches it
    before it trusts anything."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    tls = getattr(request.app.state, "client_tls", None)
    pem = tls.ca_pem() if tls is not None else None
    if not pem:
        raise _problem(
            404,
            "no-install-ca",
            "No install CA here yet",
            "This machine holds no certificate from the install's certificate authority: "
            "it is not joined to an install, or the root has not issued it one yet.",
        )
    if format == "der":
        der = x509.load_pem_x509_certificate(pem.encode("ascii")).public_bytes(
            serialization.Encoding.DER
        )
        return Response(
            der,
            media_type="application/pkix-cert",
            headers={"Content-Disposition": 'attachment; filename="eugene-plexus-ca.cer"'},
        )
    if format == "bundle":
        from ..client_tls import bundle_text

        return Response(
            bundle_text(pem),
            media_type="application/x-pem-file",
            headers={"Content-Disposition": 'attachment; filename="eugene-plexus-ca-bundle.pem"'},
        )
    return Response(
        pem,
        media_type="application/x-pem-file",
        headers={"Content-Disposition": 'inline; filename="eugene-plexus-ca.pem"'},
    )
