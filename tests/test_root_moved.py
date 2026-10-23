"""Row 4: a node follows a promoted root, and only a promoted one.

`specs/docs/design/rows/row-04.md`. Before this row `node.yaml`'s
`controlUrl` was set at enrolment and never changed: a promotion raised the
epoch on every node it reached, and every node went on pulling from the
root that had stopped.

The root's signed machine list now names the root (`control`: its epoch,
its node and its address). A node writes that address as its `controlUrl`
only for a list newer than the one it holds, at an epoch above the one its
`controlUrl` came with (`controlUrlEpoch`) and at least the one it holds
(R4-MOVE, Troy 2026-10-21: only a promotion moves a node). Then it
announces itself to the new root and pulls the bundle there. A node that
cannot reach its root for 2 minutes asks the machine that held the standby
grant, through that machine's port and pinned to its key.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import _http, leaving, machine_trust, tokens
from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest
from eugene_plexus_agent.node_identity import NodeIdentityStore

from .conftest import FakeRoot, enroll_app, enroll_store

ROOT_A = "http://10.0.0.5:8083"
ROOT_B = "http://10.0.0.9:8083"
PIN_B = "B" * 43
#: B as a list names it: a list's root is one of the machines it lists, at
#: the machine port and with the key it registered (row 4 review).
B_NODE = {
    "name": "spare",
    "url": "http://10.0.0.9:8079",
    "machineUrl": "https://10.0.0.9:8083",
    "pins": [PIN_B],
}


@pytest.fixture(autouse=True)
def _no_table() -> Any:
    yield
    _http.set_machine_table(None)


def _list(root: FakeRoot, version: int, **claims: Any) -> str:
    payload: dict[str, Any] = {
        "iat": 1,
        "version": version,
        "required": False,
        "controlPins": [],
        "nodes": [],
    }
    payload.update(claims)
    return jwt.encode(
        payload, root.identity, algorithm="EdDSA", headers={"typ": machine_trust.TYP_MACHINES}
    )


def _naming(epoch: int, url: str | None, node: str | None = "spare") -> dict[str, Any]:
    return {"epoch": epoch, "node": node, "url": url}


class _Root:
    """Every root this node may call, as one mock transport: what it was
    asked, and a bundle for whoever asks."""

    def __init__(self, root: FakeRoot) -> None:
        self.root = root
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append((request.method, url))
        if request.url.path == "/v1/trust/bundle":
            return httpx.Response(200, json={"jws": self.root.bundle().jws})
        if request.method == "PATCH":
            return httpx.Response(200, json={"name": "me"})
        return httpx.Response(404, json={"detail": "nothing here"})

    def asked(self, method: str, prefix: str) -> bool:
        return any(m == method and u.startswith(prefix) for m, u in self.calls)


def _wait(check: Callable[[], Any], timeout: float = 10.0) -> Any:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        found = check()
        if found:
            return found
        time.sleep(0.05)
    raise AssertionError("timed out")


def _push(client: TestClient, jws: str) -> httpx.Response:
    return client.post("/v1/node/machine-list", json={"jws": jws}, headers={"Authorization": ""})


def test_higher_epoch_list_moves_control_url_and_reannounces(app: FastAPI) -> None:
    root = FakeRoot()
    root.epoch = 2
    seen = _Root(root)
    app.state.control_transport = httpx.MockTransport(seen)
    with TestClient(app) as client:
        app.state.agent_state.apply_config_patch(
            ConfigUpdateRequest(advertiseUrl="http://10.0.0.6:8079")
        )
        enroll_app(app, root, "me", control_url=ROOT_A)
        record = app.state.node_identity.record
        assert (record.control_url, record.epoch, record.control_url_epoch) == (ROOT_A, 2, 2)

        # The same epoch with another address: the node keeps the root it
        # reaches (it may know it by an address the root does not).
        same = _push(client, _list(root, 10, control=_naming(2, ROOT_B)))
        assert same.status_code == 204, same.text
        assert app.state.node_identity.record.control_url == ROOT_A

        root.epoch = 3
        # A promotion: epoch 3, at B. The node moves, holds epoch 3, carries
        # B's address to B's machine port, announces to B and pulls there.
        moved = _push(
            client,
            _list(root, 11, control=_naming(3, ROOT_B), controlPins=[PIN_B], nodes=[B_NODE]),
        )
        assert moved.status_code == 204, moved.text
        record = app.state.node_identity.record
        assert (record.control_url, record.epoch, record.control_url_epoch) == (ROOT_B, 3, 3)
        assert record.control_pins is None, "the join token's key was the old root's"
        assert _http.machine_route(ROOT_B + "/v1/trust/bundle") == (
            "https://10.0.0.9:8083/control/v1/trust/bundle"
        )
        _wait(lambda: seen.asked("PATCH", ROOT_B + "/v1/nodes/me"))
        _wait(lambda: seen.asked("GET", ROOT_B + "/v1/trust/bundle"))
        assert not seen.asked("PATCH", ROOT_A), "nothing more is said to the old root"
        node = client.get("/v1/node", headers=_operator(app, root)).json()
        assert (node["controlUrl"].rstrip("/"), node["controlUrlEpoch"]) == (ROOT_B, 3)
        on_disk = yaml.safe_load(app.state.node_identity.path.read_text(encoding="utf-8"))
        assert (on_disk["controlUrl"], on_disk["controlUrlEpoch"]) == (ROOT_B, 3)

        # The old root, still signing at epoch 2, is refused outright, even
        # with a higher version: it is a superseded root.
        stale = _push(client, _list(root, 50, control=_naming(2, ROOT_A, node="nas")))
        assert stale.status_code == 409, stale.text
        assert "epoch 2 is below the 3 this node holds" in stale.text
        assert app.state.node_identity.record.control_url == ROOT_B

        # Signed by another root: 401, and nothing moves.
        stranger = _push(client, _list(FakeRoot(), 99, control=_naming(9, ROOT_A)))
        assert stranger.status_code == 401
        assert app.state.node_identity.record.control_url == ROOT_B


def _operator(app: FastAPI, root: FakeRoot) -> dict[str, str]:
    name = app.state.node_identity.record.name
    return {"Authorization": f"Bearer {root.session(f'node:{name}', 'control')}"}


def test_a_node_updated_to_this_row_keeps_its_root(tmp_path: Path) -> None:
    """`controlUrlEpoch` missing at load is the epoch held: an update never
    moves a node, and a list at that epoch does not either."""
    root = FakeRoot()
    root.epoch = 4
    store = NodeIdentityStore(tmp_path / "node.yaml")
    store.load()
    store.ensure_keypair()
    enroll_store(store, root, "me", control_url=ROOT_A)
    raw = yaml.safe_load(store.path.read_text(encoding="utf-8"))
    raw.pop("controlUrlEpoch")
    store.path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    again = NodeIdentityStore(tmp_path / "node.yaml")
    again.load()
    assert again.record.control_url_epoch == 4
    trust = machine_trust.MachineTrust(tmp_path, again)
    trust.accept(_list(root, 3, control=_naming(4, ROOT_B)))
    assert again.record.control_url == ROOT_A


def test_a_list_older_root_without_a_control_block_is_taken_as_before(tmp_path: Path) -> None:
    root = FakeRoot()
    store = NodeIdentityStore(tmp_path / "node.yaml")
    store.load()
    store.ensure_keypair()
    enroll_store(store, root, "me", control_url=ROOT_A)
    trust = machine_trust.MachineTrust(tmp_path, store)
    assert trust.accept(_list(root, 3)) is True
    assert store.record.control_url == ROOT_A


async def test_a_node_that_cannot_reach_its_root_asks_the_standby_s_machine(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2 minutes of failed pulls, then the machine that held the standby
    grant: `https://<its door>/control/v1/trust/machines`, judged by the same
    rules. Before then, and while that machine's copy is still a standby (no
    route), nothing moves."""
    root = FakeRoot()
    with TestClient(app):
        enroll_app(app, root, "me", control_url=ROOT_A)
        spare = tokens.generate_private_key().public_key()
        root.register("spare", spare, (tokens.GRANT_STANDBY,))
        app.state.auth_state.trust.accept(root.bundle().jws)
        app.state.machine_trust.accept(
            _list(
                root,
                2,
                control=_naming(1, ROOT_A, node="nas"),
                nodes=[
                    {
                        "name": "spare",
                        "url": "http://10.0.0.9:8079",
                        "machineUrl": "https://10.0.0.9:8083",
                        "pins": [PIN_B],
                    }
                ],
            )
        )
        promoted = {"yes": False}
        asked: list[str] = []

        def answer(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            asked.append(url)
            if url.startswith(ROOT_A):
                raise httpx.ConnectError("connection refused", request=request)
            if url == "https://10.0.0.9:8083/control/v1/trust/machines":
                if not promoted["yes"]:
                    return httpx.Response(404, json={"title": "No control root here"})
                jws = _list(
                    root, 9, control=_naming(2, ROOT_B), controlPins=[PIN_B], nodes=[B_NODE]
                )
                return httpx.Response(200, json={"jws": jws})
            return httpx.Response(404)

        clock = [1000.0]
        monkeypatch.setattr(leaving, "_clock", lambda: clock[0])
        app.state.control_transport = httpx.MockTransport(answer)
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
            assert await leaving.pull_once(app, client) is leaving.Heard.NO_ANSWER
            clock[0] += 119
            await leaving.pull_once(app, client)
            assert not [u for u in asked if "10.0.0.9" in u], "not before 2 minutes"
            clock[0] += 2
            await leaving.pull_once(app, client)
            assert "https://10.0.0.9:8083/control/v1/trust/machines" in asked
            assert app.state.node_identity.record.control_url == ROOT_A, "still a standby there"
            promoted["yes"] = True
            clock[0] += 60
            await leaving.pull_once(app, client)
            moving = app.state.root_moved_task
            assert moving is not None
            await moving
        record = app.state.node_identity.record
        assert (record.control_url, record.control_url_epoch, record.epoch) == (ROOT_B, 2, 2)


def test_the_list_is_kept_with_its_control_block(tmp_path: Path) -> None:
    root = FakeRoot()
    store = NodeIdentityStore(tmp_path / "node.yaml")
    store.load()
    store.ensure_keypair()
    enroll_store(store, root, "me", control_url=ROOT_A)
    trust = machine_trust.MachineTrust(tmp_path, store)
    trust.accept(_list(root, 5, control=_naming(2, ROOT_B)))
    kept = json.loads((tmp_path / machine_trust.LIST_FILE).read_text(encoding="utf-8"))
    assert jwt.decode(kept["jws"], options={"verify_signature": False})["control"]["epoch"] == 2


def test_a_root_address_no_other_machine_reaches_is_not_followed(tmp_path: Path) -> None:
    """A list at a new epoch that names a loopback or link-local address, or
    none, does not move the node: it would leave it with a root it cannot
    reach. It still holds the epoch, which fences the old root."""
    root = FakeRoot()
    store = NodeIdentityStore(tmp_path / "node.yaml")
    store.load()
    store.ensure_keypair()
    enroll_store(store, root, "me", control_url=ROOT_A)
    trust = machine_trust.MachineTrust(tmp_path, store)
    for version, url in (
        (2, "http://127.0.0.1:8083"),
        (3, "http://169.254.169.254:8083"),
        (4, None),
    ):
        trust.accept(_list(root, version, control=_naming(2, url)))
        assert store.record.control_url == ROOT_A, url
    assert (store.record.epoch, store.record.control_url_epoch) == (2, 1)


def test_a_list_naming_this_machine_marks_its_copy_with_no_address(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A promoted copy on a machine whose list names no address (no machine
    port, no nodes name): nothing moves, but the copy is this machine's root
    once a list signed at its epoch names this machine and the copy says it
    is a root. The push route brings the copy in line, as the bundle's does."""
    from eugene_plexus_agent import standby
    from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind, SpawnConfig

    root = FakeRoot()
    with TestClient(app) as client:
        enroll_app(app, root, "me", control_url=ROOT_A, grants=(tokens.GRANT_STANDBY,))
        state = app.state.agent_state
        state.add_topology_entry(
            ComponentEntry(
                name=standby.STANDBY_COMPONENT,
                kind=ComponentKind.control,
                url="http://127.0.0.1:9",  # type: ignore[arg-type]
                spawn=SpawnConfig(configFile=str(Path(state.path).parent / "standby.yaml")),
            )
        )

        async def says(_entry: Any) -> tuple[str, int]:
            return "control", 2

        monkeypatch.setattr(standby, "_local_health", says)
        root.epoch = 2
        root.register("me", app.state.auth_state.trust.signer().key.public_key(), ())
        took = client.post("/v1/node/trust-bundle", json={"jws": root.bundle().jws})
        assert took.status_code == 200, took.text
        assert not standby.promoted(state), "no list names this machine yet"

        listed = _push(client, _list(root, 3, control=_naming(2, None, node="me")))
        assert listed.status_code == 204, listed.text
        assert standby.promoted(state)
        assert app.state.node_identity.record.control_url == ROOT_A, "no address to move to"


# --------------------------------------------------------------------------- the review


def _moved_to_b(app: FastAPI, root: FakeRoot, client: TestClient) -> None:
    """This node takes B's list at epoch 3, before B's bundle arrives."""
    taken = _push(
        client, _list(root, 11, control=_naming(3, ROOT_B), controlPins=[PIN_B], nodes=[B_NODE])
    )
    assert taken.status_code == 204, taken.text
    assert app.state.node_identity.record.epoch == 3


def test_a_command_from_a_root_this_machine_no_longer_follows_is_refused(
    app: FastAPI,
) -> None:
    """A promotion keeps the root's token key: the old root's tokens carry
    the epoch they were signed at, and a machine that holds a higher one
    refuses them, its bundle not yet moved included (row 4 review,
    blocker 1)."""
    root = FakeRoot()
    root.epoch = 2
    app.state.control_transport = httpx.MockTransport(_Root(root))
    with TestClient(app) as client:
        enroll_app(app, root, "me", control_url=ROOT_A)
        _moved_to_b(app, root, client)
        assert app.state.auth_state.trust.bundle.epoch == 2, "B's bundle has not arrived"
        old = tokens.Signer(key=root.token.key, issuer="control", epoch=2)
        new = tokens.Signer(key=root.token.key, issuer="control", epoch=3)
        command, _ = old.mint(
            typ=tokens.TYP_SERVICE, sub="control", aud=["node:me"], ttl_seconds=300
        )
        refused = client.post(
            "/v1/runtimes",
            json={"name": "x", "engine": "llama_cpp", "modelPath": "/m.gguf", "autoStart": False},
            headers={"Authorization": f"Bearer {command}"},
        )
        assert refused.status_code == 401, refused.text
        assert (
            "This command came from a root this machine no longer follows "
            "(epoch 2; it follows epoch 3)."
        ) in refused.text
        session, _ = old.mint(
            typ=tokens.TYP_SESSION, sub="operator", aud=["node:me"], ttl_seconds=600
        )
        stale = client.get("/v1/node", headers={"Authorization": f"Bearer {session}"})
        assert stale.status_code == 401
        fresh, _ = new.mint(
            typ=tokens.TYP_SESSION, sub="operator", aud=["node:me"], ttl_seconds=600
        )
        took = client.get("/v1/node", headers={"Authorization": f"Bearer {fresh}"})
        assert took.status_code == 200, took.text


def _store_at(tmp_path: Path, root: FakeRoot, name: str = "me") -> NodeIdentityStore:
    store = NodeIdentityStore(tmp_path / "node.yaml")
    store.load()
    store.ensure_keypair()
    enroll_store(store, root, name, control_url=ROOT_A)
    return store


def test_a_list_that_names_no_root_epoch_is_refused_once_one_that_does_is_held(
    tmp_path: Path,
) -> None:
    """A replay of an older root's list (no `control`) after the node holds
    an epoch-aware one, and a `control` block that does not hold together,
    are refused; nothing of them is kept (row 4 review, blocker 2)."""
    root = FakeRoot()
    store = _store_at(tmp_path, root)
    trust = machine_trust.MachineTrust(tmp_path, store)
    held = _list(root, 10, control=_naming(1, ROOT_A, node=None), controlPins=[PIN_B])
    trust.accept(held)
    replay = _list(root, 11, controlPins=["O" * 43])
    with pytest.raises(machine_trust.MachineListFenced, match="names no root epoch"):
        trust.accept(replay)
    with pytest.raises(machine_trust.MachineListError, match="cannot read"):
        trust.accept(_list(root, 12, control={"epoch": "three"}))
    kept = json.loads((tmp_path / machine_trust.LIST_FILE).read_text(encoding="utf-8"))
    assert kept["jws"] == held
    again = machine_trust.MachineTrust(tmp_path, store)
    again.load()
    assert again.claims is not None and again.claims["version"] == 10


def test_another_list_under_the_same_version_is_refused_and_not_kept(tmp_path: Path) -> None:
    """One version is one list: another one under the same number is
    refused in memory and on disk alike, so a restart cannot take it (row 4
    review, blocker 2)."""
    root = FakeRoot()
    store = _store_at(tmp_path, root)
    trust = machine_trust.MachineTrust(tmp_path, store)
    held = _list(root, 10, control=_naming(1, ROOT_A, node=None), controlPins=[PIN_B])
    assert trust.accept(held)
    assert trust.accept(held) is False, "the same list again is fine"
    other = _list(root, 10, control=_naming(1, ROOT_A, node=None), controlPins=["O" * 43])
    with pytest.raises(machine_trust.MachineListError, match="other content"):
        trust.accept(other)
    again = machine_trust.MachineTrust(tmp_path, store)
    again.load()
    assert again.claims is not None and again.claims["controlPins"] == [PIN_B]


def test_a_kept_list_of_a_root_below_the_epoch_held_is_not_loaded(tmp_path: Path) -> None:
    root = FakeRoot()
    store = _store_at(tmp_path, root)
    machine_trust.MachineTrust(tmp_path, store).accept(
        _list(root, 10, control=_naming(1, ROOT_A, node=None))
    )
    store.accept_epoch(3)
    again = machine_trust.MachineTrust(tmp_path, store)
    again.load()
    assert again.claims is None


def test_a_root_that_is_not_one_of_the_listed_machines_moves_no_one(tmp_path: Path) -> None:
    """A list's root is one of the machines it lists, at the machine port
    and with the key that machine registered (row 4 review)."""
    root = FakeRoot()
    store = _store_at(tmp_path, root)
    trust = machine_trust.MachineTrust(tmp_path, store)
    cases = [
        {"control": _naming(2, ROOT_B, node="unknown"), "nodes": []},
        {"control": _naming(2, "http://10.0.0.66:8083"), "nodes": [B_NODE]},
        {"control": _naming(2, ROOT_B), "nodes": [B_NODE], "controlPins": ["O" * 43]},
    ]
    for version, claims in enumerate(cases, start=2):
        trust.accept(_list(root, version, **claims))
        assert store.record.control_url == ROOT_A, claims
    trust.accept(_list(root, 9, control=_naming(2, ROOT_B), nodes=[B_NODE], controlPins=[PIN_B]))
    assert store.record.control_url == ROOT_B


async def test_a_root_that_answers_for_an_older_epoch_counts_as_silent(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old root that comes back answers with bundles this node fences:
    that is no root, and the 2 minutes run as if it said nothing; and when
    the standby's machine changed its key, its root's signed list is taken
    without that key, verified (row 4 review)."""
    from eugene_plexus_agent import _http as http

    root = FakeRoot()
    root.epoch = 2
    with TestClient(app):
        enroll_app(app, root, "me", control_url=ROOT_A)
        spare = tokens.generate_private_key().public_key()
        root.register("spare", spare, (tokens.GRANT_STANDBY,))
        app.state.auth_state.trust.accept(root.bundle().jws)
        app.state.machine_trust.accept(
            _list(root, 2, control=_naming(2, ROOT_A, node=None), nodes=[B_NODE])
        )
        old = FakeRoot(identity=root.identity, token=root.token, epoch=1)
        old.members = dict(root.members)
        stale = old.bundle().jws
        moved = _list(root, 9, control=_naming(3, ROOT_B), controlPins=[PIN_B], nodes=[B_NODE])
        asked: list[str] = []

        def answer(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            asked.append(url)
            if url.startswith(ROOT_A):
                return httpx.Response(200, json={"jws": stale})
            if url.startswith("https://10.0.0.9:8083/"):
                raise httpx.ConnectError("refused") from http.PinRefused(
                    "10.0.0.9:8083", "N" * 43, "spare"
                )
            return httpx.Response(404)

        async def unpinned(url: str) -> httpx.Response:
            asked.append("unpinned " + url)
            return httpx.Response(200, json={"jws": moved})

        clock = [1000.0]
        monkeypatch.setattr(leaving, "_clock", lambda: clock[0])
        monkeypatch.setattr(leaving, "_unpinned", unpinned)
        app.state.control_transport = httpx.MockTransport(answer)
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
            assert await leaving.pull_once(app, client) is leaving.Heard.UNTRUSTED
            clock[0] += 121
            await leaving.pull_once(app, client)
            if app.state.root_moved_task is not None:
                await app.state.root_moved_task
        assert "unpinned http://10.0.0.9:8083/v1/trust/machines" in asked
        record = app.state.node_identity.record
        assert (record.control_url, record.control_url_epoch) == (ROOT_B, 3)


def _spare_app(tmp_path: Path, root: FakeRoot, grants: tuple[str, ...]) -> Any:
    from types import SimpleNamespace

    from eugene_plexus_agent.auth_state import AuthState
    from eugene_plexus_agent.state import AgentState
    from eugene_plexus_agent.trust import BUNDLE_FILE, NodeTrust

    store = NodeIdentityStore(tmp_path / "node.yaml")
    store.load()
    store.ensure_keypair()
    enroll_store(store, root, "spare", control_url=ROOT_A, grants=grants)
    trust = NodeTrust(store, tmp_path / BUNDLE_FILE)
    trust.load()
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    return SimpleNamespace(
        state=SimpleNamespace(
            agent_state=state,
            auth_state=AuthState(trust=trust),
            node_identity=store,
            machine_trust=machine_trust.MachineTrust(tmp_path, store),
            supervisor=None,
        )
    )


def test_a_copy_the_list_names_the_root_is_not_started_to_follow(tmp_path: Path) -> None:
    """The standby's spawn: while the grant holds it follows, and is told
    the epoch this node holds; once a signed list names this machine the
    root, it is started as the root, though the bundle still grants it
    (row 4 review, blocker 3)."""
    from eugene_plexus_agent import standby

    root = FakeRoot()
    app = _spare_app(tmp_path, root, (tokens.GRANT_STANDBY,))
    env = standby.spawn_env(app)
    assert (env["ROLE"], env["FOLLOW_EPOCH"]) == ("standby", "1")
    app.state.machine_trust.accept(_list(root, 5, control=_naming(2, None)))
    env = standby.spawn_env(app)
    assert "ROLE" not in env and "FOLLOW_EPOCH" not in env


def test_a_promoted_mark_no_list_backs_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`standbyPromoted` holds only while a signed list names this machine
    the root (row 4 review)."""
    import asyncio

    from eugene_plexus_agent import standby
    from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind, SpawnConfig

    app = _spare_app(tmp_path, FakeRoot(), ())
    state = app.state.agent_state
    state.add_topology_entry(
        ComponentEntry(
            name=standby.STANDBY_COMPONENT,
            kind=ComponentKind.control,
            url="http://127.0.0.1:9",  # type: ignore[arg-type]
            spawn=SpawnConfig(configFile=str(tmp_path / "standby.yaml")),
        )
    )
    state.set_standby_promoted(True)
    assert standby.promoted(state), "written by hand"

    async def says(_entry: Any) -> tuple[str, int]:
        return "control", 1

    monkeypatch.setattr(standby, "_local_health", says)
    asyncio.run(standby.reconcile(app))
    assert not standby.promoted(state)
    assert not standby.hosts_active_control(state)
