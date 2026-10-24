"""Row 53: every install has a store, on the root, supervised like the gateway.

The named test of the row (`test_a_root_install_starts_the_store_by_default`)
is in `test_components.py`; these are the rest: the upgrade that adds a
store to a root set up before it existed, the spawn plan, the proxy target,
and `POST /v1/people/known`, through which the store learns who was removed.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import default_topology
from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind, SpawnConfig
from eugene_plexus_agent.state import AgentState

from .conftest import FakeRoot, enroll_app, local_service_token


@pytest.fixture(autouse=True)
def _store_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(default_topology, "is_installed", lambda module: True)


def _entry(name: str, kind: ComponentKind, port: int, tmp_path: Path) -> ComponentEntry:
    return ComponentEntry(
        name=name,
        kind=kind,
        url=f"http://127.0.0.1:{port}",  # type: ignore[arg-type]
        spawn=SpawnConfig(configFile=str(tmp_path / f"{name}.yaml")),
        safeMode=False,
    )


def _root(tmp_path: Path, *, control_port: int = 8084) -> AgentState:
    """A root set up before the store existed: control moved behind the
    machine port to 8084, the first free port above it then."""
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    state.add_topology_entry(_entry("control", ComponentKind.control, control_port, tmp_path))
    state.add_topology_entry(_entry("gateway", ComponentKind.gateway, 8080, tmp_path))
    return state


def test_a_root_set_up_before_the_store_gets_one_once(tmp_path: Path) -> None:
    state = _root(tmp_path)
    assert default_topology.add_store(state) == "store"
    store = state.get_topology_entry("store")
    assert store is not None and store.kind == ComponentKind.store
    # 8084 is control's here, so the store takes the next free port.
    assert str(store.url).rstrip("/").endswith(":8085")
    assert default_topology.add_store(state) is None


def test_a_store_the_owner_removed_stays_removed(tmp_path: Path) -> None:
    state = _root(tmp_path)
    assert default_topology.add_store(state) == "store"
    state.remove_topology_entry("store")
    assert default_topology.add_store(state) is None
    assert state.get_topology_entry("store") is None


def test_a_worker_never_declares_the_store(tmp_path: Path) -> None:
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    state.add_topology_entry(_entry("left", ComponentKind.inference_driver, 8090, tmp_path))
    assert default_topology.add_store(state) is None
    assert state.get_topology_entry("store") is None
    assert not (tmp_path / default_topology.STORE_DECLARED_MARKER).exists()


def test_a_store_that_does_not_import_is_not_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(default_topology, "is_installed", lambda module: False)
    state = _root(tmp_path)
    assert default_topology.add_store(state) is None
    # Not marked: the next boot, after an update that installs it, declares it.
    assert not (tmp_path / default_topology.STORE_DECLARED_MARKER).exists()


def test_the_store_is_spawned_with_its_own_prefix_and_the_auth_wiring(tmp_path: Path) -> None:
    from eugene_plexus_agent.supervisor import _COMPONENT_SPECS

    spec = _COMPONENT_SPECS[ComponentKind.store]
    assert spec.module == "eugene_plexus_store"
    assert spec.env_prefix == "EUGENE_PLEXUS_STORE"


def test_the_proxy_finds_the_store_by_kind(authed_client: TestClient, tmp_path: Path) -> None:
    from eugene_plexus_agent.routes import proxy

    state: AgentState = authed_client.app.state.agent_state  # type: ignore[attr-defined]
    state.add_topology_entry(_entry("my-store", ComponentKind.store, 18084, tmp_path))
    assert proxy._SINGLETON_KINDS["store"] == "store"

    class _Request:
        app = authed_client.app

    assert proxy.resolve_local(_Request(), "store") == "http://127.0.0.1:18084/"  # type: ignore[arg-type]


def _people_app(authed_client: TestClient, root: FakeRoot, answer: httpx.Response) -> list[dict]:
    app = authed_client.app
    enroll_app(app, root, "nas")  # type: ignore[arg-type]
    asked: list[dict] = []

    def control(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/people/known"
        asked.append(
            {"body": json.loads(request.content), "auth": request.headers["authorization"]}
        )
        return answer

    app.state.control_transport = httpx.MockTransport(control)  # type: ignore[attr-defined]
    return asked


def test_the_store_asks_the_root_who_was_removed_through_its_agent(
    authed_client: TestClient,
) -> None:
    root = FakeRoot()
    asked = _people_app(
        authed_client, root, httpx.Response(200, json={"known": ["ann"], "unknown": ["ben"]})
    )
    token = local_service_token(authed_client.app, "store")  # type: ignore[arg-type]
    answer = authed_client.post(
        "/v1/people/known",
        json={"subjects": ["ann", "ben"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"known": ["ann"], "unknown": ["ben"]}
    # The agent asked with its own token, not the store's.
    assert asked[0]["body"] == {"subjects": ["ann", "ben"]}
    assert asked[0]["auth"] != f"Bearer {token}"


def test_only_this_machines_store_may_ask(authed_client: TestClient) -> None:
    root = FakeRoot()
    _people_app(authed_client, root, httpx.Response(200, json={"known": [], "unknown": []}))
    for sub in ("gateway", "library"):
        token = local_service_token(authed_client.app, sub)  # type: ignore[arg-type]
        refused = authed_client.post(
            "/v1/people/known",
            json={"subjects": ["ann"]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert refused.status_code == 401
    # The operator's own session is not the store either.
    assert authed_client.post("/v1/people/known", json={"subjects": ["ann"]}).status_code == 401


def test_a_root_that_cannot_say_is_a_503_naming_it(authed_client: TestClient) -> None:
    root = FakeRoot()
    _people_app(
        authed_client,
        root,
        httpx.Response(503, json={"detail": {"detail": "Eugene is locked."}}),
    )
    token = local_service_token(authed_client.app, "store")  # type: ignore[arg-type]
    answer = authed_client.post(
        "/v1/people/known",
        json={"subjects": ["ann"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert answer.status_code == 503
    assert "Eugene is locked." in answer.text


def test_a_machine_that_joined_nothing_cannot_say(authed_client: TestClient) -> None:
    token = local_service_token(authed_client.app, "store")  # type: ignore[arg-type]
    answer = authed_client.post(
        "/v1/people/known",
        json={"subjects": ["ann"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert answer.status_code == 503
    assert "has not joined a control root" in answer.text


def test_control_moving_behind_the_machine_port_never_takes_the_stores_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh root declares control on 8083 and the store on 8084 before
    either runs; control then leaves 8083 for the machine port. Every port
    is free to the operating system at that moment, so the walk must skip
    the declared ones itself."""
    from eugene_plexus_agent import machine_port, ports

    monkeypatch.setattr(ports, "is_free", lambda port, host="127.0.0.1": True)
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    state.add_topology_entry(_entry("control", ComponentKind.control, 8083, tmp_path))
    state.add_topology_entry(_entry("store", ComponentKind.store, 8084, tmp_path))
    moved = machine_port.move_control_behind(state, 8083)
    assert moved is not None and moved not in (8083, 8084)
    store = state.get_topology_entry("store")
    assert store is not None and str(store.url).rstrip("/").endswith(":8084")
