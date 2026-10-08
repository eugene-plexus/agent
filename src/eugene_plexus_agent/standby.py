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
#: A control root's port; a worker runs no other control.
CONTROL_PORT = 8083
#: What a control root's `/healthz` says it is (`details.role`).
ROLE_ACTIVE = "control"
ROLE_STANDBY = "standby"


def is_standby(entry: ComponentEntry) -> bool:
    return entry.name == STANDBY_COMPONENT and entry.kind == ComponentKind.control


def hosts_active_control(state: AgentState) -> bool:
    """This agent supervises the install's active control root. The
    standby it may run does not count."""
    return any(
        e.kind == ComponentKind.control and not is_standby(e) for e in state.list_topology_entries()
    )


def config_dir(state: AgentState) -> Path:
    """The directory agent.yaml is in, where the standby's files go."""
    return Path(state.path).resolve().parent


def state_dir(config_dir: Path) -> Path:
    return config_dir / STATE_DIR_NAME


def child_env(home: Path, control_url: str, *, node: str | None, following: bool) -> dict[str, str]:
    """Suffix -> value for the standby's spawn, over everything else: its
    data directory, its node's name (so a promotion names it), loopback,
    since it pulls and is never dialled, and, **while the grant holds**,
    the standby role and the root it follows. Without the grant it was
    promoted (the promotion takes the grant), so it starts as the root
    it now is."""
    env = {
        "STATE_DIR": str(state_dir(home)),
        "BIND_HOST": "127.0.0.1",
    }
    if node:
        env["NODE_NAME"] = node
    if following:
        env["ROLE"] = "standby"
        env["ACTIVE_URL"] = control_url
    return env


def spawn_env(app: Any) -> dict[str, str]:
    """`child_env` for this agent, read at every spawn."""
    record = app.state.node_identity.record
    return child_env(
        config_dir(app.state.agent_state),
        str(record.control_url or ""),
        node=record.name,
        following=wanted(app),
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
        port = ports.first_free(CONTROL_PORT, reserved=(RESERVED_PORTS - {CONTROL_PORT}) | taken)
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
        role = await _local_role(entry)
        if role == ROLE_ACTIVE:
            if not getattr(app.state, "standby_promoted_said", False):
                log.warning(
                    "this node's standby was promoted and is the install's control "
                    "root now; it keeps running here"
                )
                app.state.standby_promoted_said = True
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
        await asyncio.to_thread(_delete_copy, home)


async def _local_role(entry: ComponentEntry) -> str | None:
    """What the local control root says it is, from its `/healthz`."""
    client = shared_internal_client("standby-health", timeout=5.0)
    try:
        response = await client.get(f"{str(entry.url).rstrip('/')}/healthz")
        details = response.json().get("details") or {}
    except Exception:
        return None
    role = details.get("role") if isinstance(details, dict) else None
    return role if isinstance(role, str) else None


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
