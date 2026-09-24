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
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status

from .. import __version__, default_topology, reach, security
from .._generated.common_models import ConfigUpdateRequest, Problem
from .._generated.models import (
    Arch,
    EnrollRequest,
    NodeIdentity,
    NodeReach,
    NodeReachRequest,
    NodeReachResult,
    Os,
    ReachStep,
    RekeyRequest,
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
from ..node_identity import (
    FencedError,
    NodeIdentityStore,
    advertise_host,
    effective_advertise_url,
    is_loopback_host,
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
        # Read at the moment of answering, not cached and not derived
        # from anything: the point of the field is that a console can
        # compare two hosts and see a drift neither host can see about
        # itself. `_note_clock_skew` in security.py observes the same
        # quantity precisely and can only write it to a log.
        time=datetime.now(UTC),
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
    try:
        security.validate_signing_key(raw)
    except (ValueError, TypeError):
        return None
    return raw


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
    the other three callers of `_identity` -- enroll, unenroll, rekey --
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
    # Joining answers the one onboarding question, so nothing is left
    # for the wizard to do. Set here rather than only at the next boot
    # because enrolling does not restart this process -- without it the
    # operator who just enrolled gets sent to first-run setup.
    default_topology.mark_onboarded(state)
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
        async with internal_client(timeout=_REVOKE_TIMEOUT_SECONDS, transport=transport) as client:
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
            "`signingKey` must be base64 Ed25519 private PEM or a legacy 32-byte key.",
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
# reach: can anything else on the network get here
# --------------------------------------------------------------------------- #


def _listeners(request: Request) -> list[reach.Listener]:
    """This install's processes on this host, and the interface each one
    was actually started on.

    The agent's own bind is the value `build_server` handed uvicorn; a
    component's is what the supervisor spawned it with. A component this
    agent declares but is not currently running contributes nothing —
    "what it would bind if it started" is a guess, and this object
    exists to hold facts.
    """
    settings = request.app.state.settings
    out = [
        reach.Listener(
            process="agent",
            port=int(settings.bind_port),
            bind_host=getattr(request.app.state, "bind_host", None) or settings.bind_host,
        )
    ]
    state: AgentState = request.app.state.agent_state
    # Apps listen too, on their own ports, and a phone opening a chat app
    # needs the same three things a phone opening the console does.
    manager = getattr(request.app.state, "apps", None)
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
    supervisor = getattr(request.app.state, "supervisor", None)
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
