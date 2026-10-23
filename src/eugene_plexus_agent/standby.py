"""The warm standby this agent runs when its node holds the grant.

`specs/docs/design/warm-standby.md`, SB3. An owner makes one machine the
install's standby at the control root (`PUT /v1/nodes/{name}/standby`).
The root's new trust bundle lists the `standby` grant on that node's key,
and this agent, on taking a bundle, reconciles: it declares and starts a
control root in the standby role, following the root this node joined,
or stops it and **deletes its copy of the replication set** when the
grant goes. That copy holds the sealed keys, the salt and the passphrase
verifier, which is why it does not outlive the grant.

Nothing is typed by anyone: no `spawn.env`, no URL. The standby's only
credential is a token for this machine (`sub: standby`), which it trades
at `POST /v1/auth/service-token` for the 15-minute token the active root
accepts on its replication routes and nowhere else.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from . import ports, tokens
from ._generated.models import ComponentEntry, ComponentKind, SpawnConfig
from ._http import shared_internal_client
from .default_topology import RESERVED_PORTS
from .state import AgentState

log = logging.getLogger(__name__)

#: The standby's name among this agent's components. The agent owns it.
STANDBY_COMPONENT = "standby"
#: Where its copy of the replication set lives, beside agent.yaml.
STATE_DIR_NAME = "standby-state"
#: Where a standby control looks for a free port: above the machine port,
#: which takes 8083 on every node (LT1).
CONTROL_PORT = 8084
#: What a control root's `/healthz` says it is (`details.role`).
ROLE_ACTIVE = "control"
ROLE_STANDBY = "standby"


def is_standby(entry: ComponentEntry) -> bool:
    """The copy this agent owns: its standby, or, once `promoted`, the
    install's root it became."""
    return entry.name == STANDBY_COMPONENT and entry.kind == ComponentKind.control


def promoted(state: AgentState) -> bool:
    """Row 4: this agent's copy was promoted and is the install's root.

    Marked by `_reconcile` only when the copy says it is a root and a
    machine list signed at its epoch names this machine the root; kept in
    `agent.yaml`, and cleared with the copy. While it holds, the copy is
    this machine's control root (`hosts_active_control`, the machine port's
    `/control`, the door file), with no restart."""
    entry = state.get_topology_entry(STANDBY_COMPONENT)
    return entry is not None and is_standby(entry) and state.standby_promoted()


def is_root(entry: ComponentEntry, state: AgentState) -> bool:
    """`entry` is the install's control root on this machine: a control
    entry other than the copy, or the copy once promoted."""
    return entry.kind == ComponentKind.control and (not is_standby(entry) or promoted(state))


def hosts_active_control(state: AgentState) -> bool:
    """This agent supervises the install's active control root. The
    standby it may run does not count, until it is promoted (row 4)."""
    return any(is_root(e, state) for e in state.list_topology_entries())


def config_dir(state: AgentState) -> Path:
    """The directory agent.yaml is in, where the standby's files go."""
    return Path(state.path).resolve().parent


def state_dir(config_dir: Path) -> Path:
    return config_dir / STATE_DIR_NAME


def child_env(
    home: Path,
    control_url: str,
    *,
    node: str | None,
    following: bool,
    epoch: int | None = None,
) -> dict[str, str]:
    """Suffix -> value for the standby's spawn, over everything else: its
    data directory, its node's name (so a promotion names it), loopback,
    since it pulls and is never dialled, and, **while the grant holds**,
    the standby role, the root it follows and the epoch this node holds.
    Without the grant it was promoted (the promotion takes the grant), so
    it starts as the root it now is. With the grant, a copy whose own log
    says it was promoted above `epoch` starts as the root too (row 4
    review): its agent had not heard of the promotion yet."""
    env = {
        "STATE_DIR": str(state_dir(home)),
        "BIND_HOST": "127.0.0.1",
    }
    if node:
        env["NODE_NAME"] = node
    if following:
        env["ROLE"] = "standby"
        env["ACTIVE_URL"] = control_url
        env["FOLLOW_EPOCH"] = str(epoch or 0)
    return env


def spawn_env(app: Any) -> dict[str, str]:
    """`child_env` for this agent, read at every spawn."""
    record = app.state.node_identity.record
    return child_env(
        config_dir(app.state.agent_state),
        str(record.control_url or ""),
        node=record.name,
        # A signed list naming this machine the root is promotion evidence
        # the bundle may not carry yet (row 4 review): the copy is not
        # started to follow the root it replaced.
        following=wanted(app) and listed_root(app) != record.name,
        epoch=record.epoch,
    )


def wanted(app: Any) -> bool:
    trust = app.state.auth_state.trust
    return bool(trust.enrolled) and tokens.GRANT_STANDBY in trust.grants()


async def reconcile(app: Any) -> None:
    """Run the standby exactly when this node holds the grant.

    Called after every bundle this agent takes, and once at start. Cheap
    and idempotent, so a minute's pull also retries a deletion that a
    locked file stopped. Never raises: the trust path it rides on must not
    fail because of it."""
    lock = getattr(app.state, "standby_lock", None)
    if lock is None:
        lock = app.state.standby_lock = asyncio.Lock()
    async with lock:
        try:
            await _reconcile(app)
        except Exception:
            log.exception("could not bring the standby in line with this node's grant")


async def _reconcile(app: Any) -> None:
    state: AgentState = app.state.agent_state
    supervisor = getattr(app.state, "supervisor", None)
    home = config_dir(state)
    entry = state.get_topology_entry(STANDBY_COMPONENT)
    me = app.state.node_identity.record.name
    if promoted(state) and entry is not None:
        # Row 4 review: the mark holds only while signed evidence names this
        # machine the root: a list that names it, or (re-review) while no
        # list names any root at the epoch held, a bundle signed at the
        # epoch the copy says it is the root at, with no standby grant here.
        # A mark nothing backs (written by hand, or kept after another
        # machine was promoted) is dropped.
        root = listed_root(app)
        keep = root == me
        if root is None:
            role, epoch = await _local_health(entry)
            keep = (
                role == ROLE_ACTIVE
                and _bundle_names_promotion(app, epoch)
                and (_promoted_unannounced(app, epoch) or _follows_own_copy(app, entry, epoch))
            )
        if not keep:
            state.set_standby_promoted(False)
            log.warning(
                "no signed list or bundle this node holds says this machine is the root, so "
                "its copy is not served as the root"
            )
    want = wanted(app)
    if want and entry is not None and not promoted(state):
        # Row 4 re-review: a copy promoted here, and restarted before this
        # agent heard (a crash before any push), runs as the root and may be
        # locked. Its own signed bundle is the evidence: taken, it ends the
        # grant here, and the copy is marked below.
        await _take_promotion_from_the_copy(app, entry)
        want = wanted(app)
    if want and hosts_active_control(state):
        if entry is None:
            log.error(
                "this node holds the standby grant but runs the active control root; a "
                "standby here would stop with it, so none is started. Make another machine "
                "the standby."
            )
        want = False
    if want and entry is None:
        record = app.state.node_identity.record
        if not record.control_url:
            log.error("this node holds the standby grant but records no control root to follow")
            return
        taken = {_port_of(e) for e in state.list_topology_entries()}
        port = ports.first_free(
            CONTROL_PORT,
            reserved=(RESERVED_PORTS - {CONTROL_PORT})
            | taken
            | {getattr(getattr(app.state, "settings", None), "machine_port", 8083)},
        )
        entry = state.add_topology_entry(
            ComponentEntry(
                name=STANDBY_COMPONENT,
                kind=ComponentKind.control,
                url=f"http://127.0.0.1:{port}",  # type: ignore[arg-type]
                spawn=SpawnConfig(configFile=str(home / f"{STANDBY_COMPONENT}.yaml")),
                safeMode=False,
            )
        )
        if supervisor is not None:
            supervisor.add_and_start(entry)
        log.warning(
            "this node is now the install's warm standby: its control root listens on %d "
            "and follows %s",
            port,
            record.control_url,
        )
        return
    if not want and entry is not None:
        # The grant goes two ways: an owner removed it, or a promotion
        # took it because this copy is the install's root now. Only the
        # control itself can say which, so it is asked, and nothing is
        # stopped or deleted until it says it is still a standby.
        role, epoch = await _local_health(entry)
        if role == ROLE_ACTIVE:
            if not getattr(app.state, "standby_promoted_said", False):
                log.warning(
                    "this node's standby was promoted and is the install's control "
                    "root now; it keeps running here"
                )
                app.state.standby_promoted_said = True
            if not promoted(state) and _named_root(app, epoch):
                _mark_promoted(app, state, epoch)
            elif not promoted(state) and _promoted_unannounced(app, epoch):
                _mark_promoted(app, state, epoch)
                _follow_own_copy(app, entry, int(epoch or 0))
            await _note_root_lock(app)
            return
        if role != ROLE_STANDBY:
            log.warning(
                "this node is no longer the standby, but its control root did not "
                "say it is still one (%s); it and its copy are kept until it does",
                role or "no answer",
            )
            return
        if supervisor is not None:
            await supervisor.remove_and_stop(STANDBY_COMPONENT)
        state.remove_topology_entry(STANDBY_COMPONENT)
        log.warning("this node is no longer the standby: its control root is stopped")
    if not want and state.get_topology_entry(STANDBY_COMPONENT) is None:
        if state.standby_promoted():
            state.set_standby_promoted(False)
        await asyncio.to_thread(_delete_copy, home)
    await _note_root_lock(app)


def _bundle_names_promotion(app: Any, epoch: int | None) -> bool:
    """The bundle this node holds, signed by the install's identity, is at
    `epoch`, at least the epoch held, and grants this node no standby: the
    signed side of a promotion of the copy that says it is the root at
    `epoch` (row 4 re-review)."""
    bundle = app.state.auth_state.trust.bundle
    held = app.state.node_identity.record.epoch or 0
    return (
        isinstance(epoch, int)
        and bundle is not None
        and bundle.epoch == epoch
        and epoch >= held
        and not wanted(app)
    )


def _promoted_unannounced(app: Any, epoch: int | None) -> bool:
    """`_bundle_names_promotion`, at an epoch above the one this node's root
    address came with: the copy was promoted, and no list has moved this
    node yet. A copy that only says `control` at the epoch this node already
    follows (one started without its role) is not this."""
    record = app.state.node_identity.record
    return (
        _bundle_names_promotion(app, epoch)
        and isinstance(epoch, int)
        and epoch > int(record.control_url_epoch or 0)
        and listed_root(app) in (None, record.name)
    )


def _own_address(app: Any, entry: ComponentEntry) -> str:
    """Where this node reaches its own copy: this machine's port in its
    plain form, or the copy's own loopback address without one."""
    from . import machine_port

    door = machine_port.machine_url(app)
    return door.replace("https://", "http://", 1) if door else str(entry.url).rstrip("/")


def _follows_own_copy(app: Any, entry: ComponentEntry, epoch: int | None) -> bool:
    """This node already follows its own copy at `epoch` (`_follow_own_copy`)."""
    record = app.state.node_identity.record
    return (
        isinstance(epoch, int)
        and record.control_url_epoch == epoch
        and (record.control_url or "").rstrip("/") == _own_address(app, entry)
    )


async def _take_promotion_from_the_copy(app: Any, entry: ComponentEntry) -> None:
    """When the local copy says it is the root at an epoch above any this
    node holds, take the bundle it signed at its promotion (it serves it
    locked too), verified against the install's identity. Nothing else
    changes here; the bundle ends the grant, and `_reconcile` goes on."""
    role, epoch = await _local_health(entry)
    record = app.state.node_identity.record
    if role != ROLE_ACTIVE or not isinstance(epoch, int):
        return
    if epoch <= max(int(record.epoch or 0), int(record.control_url_epoch or 0)):
        return
    client = shared_internal_client("standby-health", timeout=5.0)
    try:
        response = await client.get(f"{str(entry.url).rstrip('/')}/v1/trust/bundle")
        jws = str(response.json()["jws"])
        taken = app.state.auth_state.trust.accept(jws)
    except Exception as exc:
        log.warning(
            "this machine's copy says it is the root at epoch %d, but its bundle could not "
            "be taken: %s",
            epoch,
            exc,
        )
        return
    log.warning(
        "this machine's copy was promoted to the root at epoch %d before this agent heard: "
        "took the bundle it signed",
        taken.epoch,
    )


def _follow_own_copy(app: Any, entry: ComponentEntry, epoch: int) -> None:
    """This node follows the root its own copy now is: the other machines
    learn its address from the list it signs once it is unlocked, and this
    one, from its own copy. Through this machine's port when it has one."""
    url = _own_address(app, entry)
    app.state.node_identity.follow_root(url, epoch)
    machines = getattr(app.state, "machine_trust", None)
    if machines is not None:
        machines.rebuild()
    log.warning("this node follows its own copy, the install's root at epoch %d, at %s", epoch, url)


async def _note_root_lock(app: Any) -> None:
    """`NodeIdentity.rootLocked`: whether the root this machine runs is
    locked, from its `/healthz` (row 4 re-review). Home says so."""
    from . import machine_port

    state: AgentState = app.state.agent_state
    entry = machine_port.active_control_entry(state)
    if entry is None:
        app.state.root_locked = None
        return
    details = await _local_details(entry)
    locked = details.get("locked") if details is not None else None
    was = getattr(app.state, "root_locked", None)
    app.state.root_locked = locked if isinstance(locked, bool) else None
    if app.state.root_locked and not was:
        log.warning(
            "the install's root on this machine is locked: sign in on this machine's console "
            "to unlock it"
        )


def listed_root(app: Any) -> str | None:
    """The machine the signed machine list held names the root, at an epoch
    this node still holds (row 4); None when it names none."""
    machines = getattr(app.state, "machine_trust", None)
    control = (machines.claims or {}).get("control") if machines is not None else None
    if not isinstance(control, dict):
        return None
    epoch, node = control.get("epoch"), control.get("node")
    held = app.state.node_identity.record.epoch or 0
    if not isinstance(epoch, int) or epoch < held or not isinstance(node, str):
        return None
    return node


def unmarked_root(app: Any) -> bool:
    """The machine list held names this machine the root, and the copy it
    runs is not marked promoted yet: `reconcile` has something to decide."""
    state: AgentState = app.state.agent_state
    if state.get_topology_entry(STANDBY_COMPONENT) is None or promoted(state):
        return False
    return bool(listed_root(app) == app.state.node_identity.record.name)


def _named_root(app: Any, epoch: int | None) -> bool:
    """Row 4: the root's signed machine list held here names this machine
    the root, at the epoch the copy says it is at. The list is signed by
    the install's identity, so a copy that only says `control` (one started
    without its role, say) is not taken for the root: the epoch it says
    must be the one the list names, and no copy reaches a higher epoch
    without a promotion entry."""
    machines = getattr(app.state, "machine_trust", None)
    control = (machines.claims or {}).get("control") if machines is not None else None
    name = app.state.node_identity.record.name
    return (
        isinstance(control, dict)
        and isinstance(epoch, int)
        and control.get("node") == name
        and control.get("epoch") == epoch
    )


def _mark_promoted(app: Any, state: AgentState, epoch: int | None) -> None:
    """The copy is this machine's control root now: kept in `agent.yaml`,
    and the door file (the machine port's keys and address, which control
    reads) is written for it. Nothing restarts, so it does not come back
    sealed."""
    state.set_standby_promoted(True)
    log.warning(
        "this machine's copy is the install's control root at epoch %s: other machines "
        "reach it through this machine's port",
        epoch,
    )
    key_change = getattr(app.state, "machine_key_change", None)
    if key_change is not None:
        try:
            key_change.write_door()
        except Exception:  # pragma: no cover - defensive
            log.warning("could not write the machine door file for the promoted copy")


async def _local_details(entry: ComponentEntry) -> dict[str, Any] | None:
    """The local control root's `/healthz` details, or None without an
    answer."""
    client = shared_internal_client("standby-health", timeout=5.0)
    try:
        response = await client.get(f"{str(entry.url).rstrip('/')}/healthz")
        details = response.json().get("details") or {}
    except Exception:
        return None
    return details if isinstance(details, dict) else None


async def _local_health(entry: ComponentEntry) -> tuple[str | None, int | None]:
    """What the local control root says it is, and at which epoch, from its
    `/healthz` (`details.role`, `details.epoch`)."""
    details = await _local_details(entry)
    if details is None:
        return None, None
    role, epoch = details.get("role"), details.get("epoch")
    return (
        role if isinstance(role, str) else None,
        epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else None,
    )


def _delete_copy(home: Path) -> None:
    """Delete the replica and its config. A file still held is said and
    retried at the next reconcile."""
    copy = state_dir(home)
    config = home / f"{STANDBY_COMPONENT}.yaml"
    if copy.exists():
        try:
            shutil.rmtree(copy)
            log.warning("deleted the standby's copy of the replication set at %s", copy)
        except OSError as exc:
            log.warning("could not delete the standby's copy at %s yet: %s", copy, exc)
    config.unlink(missing_ok=True)


def _port_of(entry: ComponentEntry) -> int:
    try:
        return int(str(entry.url).rstrip("/").rsplit(":", 1)[-1])
    except ValueError:
        return 0
