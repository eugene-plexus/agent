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
            _list(root, 11, control=_naming(3, ROOT_B), controlPins=[PIN_B]),
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
                jws = _list(root, 9, control=_naming(2, ROOT_B), controlPins=[PIN_B])
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
