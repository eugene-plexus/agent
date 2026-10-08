"""The warm standby this agent runs (specs docs/design/warm-standby.md, SB3).

The agent starts a standby control root exactly when its node's key holds
the `standby` grant, gives it one credential and nothing else, and stops
it and deletes its copy of the replication set when the grant goes.
Before control#5 none of this existed: a standby was started by hand
through `spawn.env` and had no credential at all.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import standby, tokens
from eugene_plexus_agent._generated.models import (
    ComponentEntry,
    ComponentKind,
    ComponentStatus,
    SpawnConfig,
)
from eugene_plexus_agent.auth_state import AuthState
from eugene_plexus_agent.node_identity import NodeIdentityStore
from eugene_plexus_agent.state import AgentState
from eugene_plexus_agent.supervisor import SupervisedProcess
from eugene_plexus_agent.trust import BUNDLE_FILE, MintRefused, NodeTrust

from .conftest import FakeRoot, enroll_store

ROOT_URL = "http://root.invalid:8083"


def _enrolled(home: Path, *, grants: tuple[str, ...] = ()) -> tuple[NodeTrust, FakeRoot]:
    root = FakeRoot()
    store = NodeIdentityStore(home / "node.yaml")
    enroll_store(store, root, "spare", control_url=ROOT_URL, grants=grants)
    trust = NodeTrust(store, home / BUNDLE_FILE)
    trust.load()
    return trust, root


# --------------------------------------------------------------------------- minting


def test_a_standby_token_leaves_this_machine_for_control_only_with_the_grant(
    tmp_path: Path,
) -> None:
    plain, _ = _enrolled(tmp_path / "plain")
    with pytest.raises(MintRefused, match="not the standby"):
        plain.mint_service(sub=tokens.SUB_STANDBY, audience=tokens.RECIPIENT_CONTROL)
    granted, _ = _enrolled(tmp_path / "granted", grants=(tokens.GRANT_STANDBY,))
    token, _ = granted.mint_service(sub=tokens.SUB_STANDBY, audience=tokens.RECIPIENT_CONTROL)
    claims = tokens.verify(
        token,
        bundle=granted.bundle,  # type: ignore[arg-type]
        recipient=tokens.RECIPIENT_CONTROL,
        classes=(tokens.TYP_SERVICE,),
    )
    assert (claims.sub, claims.aud) == ("standby", ("control",))
    with pytest.raises(MintRefused, match="control only"):
        granted.mint_service(sub=tokens.SUB_STANDBY, audience="node:gpu-box")


# --------------------------------------------------------------------------- spawning


def _spawned_env(
    monkeypatch: pytest.MonkeyPatch, entry: ComponentEntry, **kwargs: Any
) -> dict[str, str]:
    captured: dict[str, str] = {}

    class _Done:
        returncode = 0
        stdout = None
        pid = 1

        async def wait(self) -> int:
            await asyncio.sleep(3600)
            return 0

        def terminate(self) -> None: ...

        def kill(self) -> None: ...

    async def fake_create(*_args: Any, **create: Any) -> _Done:
        captured.update(create.get("env") or {})
        return _Done()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    async def go() -> None:
        sp = SupervisedProcess.for_component(entry, logging.getLogger("test"), **kwargs)
        sp.start()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if captured:
                break
        await sp.stop()

    asyncio.run(go())
    return captured


def _standby_entry(home: Path) -> ComponentEntry:
    return ComponentEntry(
        name=standby.STANDBY_COMPONENT,
        kind=ComponentKind.control,
        url="http://127.0.0.1:8083",  # type: ignore[arg-type]
        spawn=SpawnConfig(configFile=str(home / "standby.yaml")),
        safeMode=False,
    )


def test_the_standby_is_the_one_control_process_given_a_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trust, _ = _enrolled(tmp_path, grants=(tokens.GRANT_STANDBY,))
    auth = AuthState(trust=trust, master_key=b"m" * 32)
    env = _spawned_env(
        monkeypatch,
        _standby_entry(tmp_path),
        auth_state=auth,
        standby_env=lambda: standby.child_env(tmp_path, ROOT_URL, node="spare", following=True),
    )
    assert env["EUGENE_PLEXUS_CONTROL_ROLE"] == "standby"
    assert env["EUGENE_PLEXUS_CONTROL_ACTIVE_URL"] == ROOT_URL
    assert env["EUGENE_PLEXUS_CONTROL_STATE_DIR"] == str(tmp_path / "standby-state")
    assert env["EUGENE_PLEXUS_CONTROL_BIND_HOST"] == "127.0.0.1"
    assert env["EUGENE_PLEXUS_CONTROL_NODE_NAME"] == "spare", "so a promotion names it"
    claims = trust.verify(env["EUGENE_PLEXUS_CONTROL_SERVICE_TOKEN"], classes=("ep-service+jwt",))
    assert (claims.sub, claims.aud) == ("standby", (trust.recipient,)), "this machine only"
    for absent in (
        "EUGENE_PLEXUS_CONTROL_AUTH_SIGNING_KEY",
        "EUGENE_PLEXUS_CONTROL_MASTER_KEY",
        "EUGENE_PLEXUS_CONTROL_TRUST_BUNDLE_FILE",
        "EUGENE_PLEXUS_CONTROL_TRUST_AUTHORITY",
        "EUGENE_PLEXUS_CONTROL_AUTH_RECIPIENT",
    ):
        assert absent not in env, absent

    # The install's own control root still gets nothing, standby wiring or not.
    root_entry = _standby_entry(tmp_path).model_copy(update={"name": "control"})
    root_env = _spawned_env(
        monkeypatch,
        root_entry,
        auth_state=auth,
        standby_env=lambda: standby.child_env(tmp_path, ROOT_URL, node="spare", following=True),
    )
    assert "EUGENE_PLEXUS_CONTROL_SERVICE_TOKEN" not in root_env
    assert "EUGENE_PLEXUS_CONTROL_ROLE" not in root_env

    # Promoted: the grant is gone, so it starts as the root it now is,
    # holding no standby token, on the copy it kept.
    promoted = _spawned_env(
        monkeypatch,
        _standby_entry(tmp_path),
        auth_state=auth,
        standby_env=lambda: standby.child_env(tmp_path, ROOT_URL, node="spare", following=False),
    )
    assert "EUGENE_PLEXUS_CONTROL_ROLE" not in promoted
    assert "EUGENE_PLEXUS_CONTROL_SERVICE_TOKEN" not in promoted
    assert promoted["EUGENE_PLEXUS_CONTROL_STATE_DIR"] == str(tmp_path / "standby-state")


# --------------------------------------------------------------------------- reconciling


class _Supervisor:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []

    def add_and_start(self, entry: ComponentEntry) -> None:
        self.started.append(entry.name)

    async def remove_and_stop(self, name: str) -> None:
        self.stopped.append(name)


def _app(home: Path, trust: NodeTrust) -> Any:
    state = AgentState(home / "agent.yaml")
    state.load()
    return SimpleNamespace(
        state=SimpleNamespace(
            agent_state=state,
            supervisor=_Supervisor(),
            auth_state=AuthState(trust=trust),
            node_identity=trust._identity,
        )
    )


def _set_grants(trust: NodeTrust, root: FakeRoot, grants: tuple[str, ...]) -> None:
    key = trust.signer().key.public_key()
    root.register("spare", key, grants)
    trust.accept(root.bundle().jws)


def test_the_grant_starts_the_standby_and_its_removal_deletes_the_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trust, root = _enrolled(tmp_path)
    app = _app(tmp_path, trust)
    asyncio.run(standby.reconcile(app))
    assert app.state.agent_state.get_topology_entry("standby") is None, "no grant, no standby"

    _set_grants(trust, root, (tokens.GRANT_STANDBY,))
    asyncio.run(standby.reconcile(app))
    entry = app.state.agent_state.get_topology_entry("standby")
    assert entry is not None and entry.kind == ComponentKind.control
    assert str(entry.url).startswith("http://127.0.0.1:")
    assert app.state.supervisor.started == ["standby"]
    asyncio.run(standby.reconcile(app))
    assert app.state.supervisor.started == ["standby"], "idempotent"

    copy = tmp_path / "standby-state"
    copy.mkdir()
    (copy / "snapshot.json").write_text("{}", encoding="utf-8")
    _set_grants(trust, root, ())
    _says(monkeypatch, None)
    asyncio.run(standby.reconcile(app))
    assert app.state.supervisor.stopped == [], "it did not say it is still a standby"
    assert copy.exists()
    _says(monkeypatch, "standby")
    asyncio.run(standby.reconcile(app))
    assert app.state.supervisor.stopped == ["standby"]
    assert app.state.agent_state.get_topology_entry("standby") is None
    assert not copy.exists(), "the sealed copy does not outlive the grant"


def _says(monkeypatch: pytest.MonkeyPatch, role: str | None) -> None:
    """What the local control root's `/healthz` reports as its role."""

    async def answer(_entry: ComponentEntry) -> str | None:
        return role

    monkeypatch.setattr(standby, "_local_role", answer)


def test_a_promoted_standby_is_never_stopped_or_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The promotion takes the grant. The copy is the install's root now,
    and the log says so once, not that the copy failed to answer."""
    trust, root = _enrolled(tmp_path, grants=(tokens.GRANT_STANDBY,))
    app = _app(tmp_path, trust)
    asyncio.run(standby.reconcile(app))
    copy = tmp_path / "standby-state"
    copy.mkdir()
    (copy / "log.jsonl").write_text("{}", encoding="utf-8")
    _set_grants(trust, root, ())
    _says(monkeypatch, "control")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=standby.log.name):
        for _ in range(2):
            asyncio.run(standby.reconcile(app))
    assert app.state.supervisor.stopped == []
    assert app.state.agent_state.get_topology_entry("standby") is not None
    assert (copy / "log.jsonl").exists()
    said = [r.getMessage() for r in caplog.records if r.name == standby.log.name]
    assert said == [
        "this node's standby was promoted and is the install's control root now; "
        "it keeps running here"
    ], said


def test_no_standby_on_the_machine_that_runs_the_active_root(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trust, _ = _enrolled(tmp_path, grants=(tokens.GRANT_STANDBY,))
    app = _app(tmp_path, trust)
    app.state.agent_state.add_topology_entry(
        _standby_entry(tmp_path).model_copy(update={"name": "control"})
    )
    with caplog.at_level(logging.ERROR):
        asyncio.run(standby.reconcile(app))
    assert app.state.agent_state.get_topology_entry("standby") is None
    assert "runs the active control root" in caplog.text


def test_the_node_report_says_whether_this_machine_hosts_the_root(
    authed_client: TestClient,
) -> None:
    state: AgentState = authed_client.app.state.agent_state  # type: ignore[attr-defined]
    stub = authed_client.app.state.supervisor  # type: ignore[attr-defined]
    stub.is_supervised = lambda _name: False
    stub.status_for = lambda _name, has_spawn: (ComponentStatus.running, None, None, None)
    for name in [e.name for e in state.list_topology_entries()]:
        state.remove_topology_entry(name)
    assert authed_client.get("/v1/node").json().get("hostsControl") is False
    state.add_topology_entry(_standby_entry(Path(state.path).parent))
    body = authed_client.get("/v1/node").json()
    assert body.get("hostsControl") is False, "a standby is not the root"
    assert body["standby"]["component"] == "standby"
    assert body["standby"]["status"] == "running"
    state.add_topology_entry(
        _standby_entry(Path(state.path).parent).model_copy(
            update={"name": "control", "url": "http://127.0.0.1:8183"}
        )
    )
    assert authed_client.get("/v1/node").json()["hostsControl"] is True
