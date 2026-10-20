"""The ports the agent opens in front of a child bind where that child would
have: the image check (compose-acceptance 13 and the NAS entry point's 31)
caught the gateway's door on loopback in a container and the machine port
wide behind the HTTPS entry point."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eugene_plexus_agent import client_tls, machine_port
from eugene_plexus_agent.node_identity import NodeIdentityStore
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.state import AgentState

_IMAGE_BINDS = {
    "EUGENE_PLEXUS_CONTROL_BIND_HOST": "0.0.0.0",
    "EUGENE_PLEXUS_GATEWAY_BIND_HOST": "0.0.0.0",
    "EUGENE_PLEXUS_DRIVER_BIND_HOST": "0.0.0.0",
}


def _app(tmp_path: Path, *, entrypoint: bool = False, nodes: bool = False) -> Any:
    config = tmp_path / "agent.yaml"
    settings = Settings(config_file=config, bind_host="127.0.0.1", bind_port=8079)
    if entrypoint:
        settings.entrypoint_config = tmp_path / "entrypoint.json"
        settings._entrypoint_nodes = nodes
    state = AgentState(config)
    state.load()
    identity = NodeIdentityStore(tmp_path / "node.yaml")
    identity.load()
    return SimpleNamespace(
        state=SimpleNamespace(
            settings=settings, agent_state=state, node_identity=identity, bind_host="127.0.0.1"
        )
    )


class _Doors:
    def __init__(self) -> None:
        self.doors: dict[str, Any] = {}
        self.opened: list[str] = []

    async def open_pump(self, name: str, *, bind_host: str, port: int, plain_target: Any) -> None:
        self.opened.append(bind_host)

    async def close(self, name: str) -> None:
        pass


async def _gateway_door_bind(app: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    gateway = SimpleNamespace(url="http://127.0.0.1:8080")
    monkeypatch.setattr(client_tls, "gateway_entry", lambda state: gateway)
    doors = _Doors()
    await client_tls.GatewayDoor(doors).ensure(app)  # type: ignore[arg-type]
    assert doors.opened, "the topology has no gateway"
    return doors.opened[-1]


@pytest.mark.parametrize("image", [False, True])
async def test_the_gateway_door_binds_where_the_gateway_would(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image: bool
) -> None:
    for name in _IMAGE_BINDS:
        monkeypatch.delenv(name, raising=False)
    if image:
        for name, value in _IMAGE_BINDS.items():
            monkeypatch.setenv(name, value)
    assert await _gateway_door_bind(_app(tmp_path), monkeypatch) == (
        "0.0.0.0" if image else "127.0.0.1"
    )


async def test_behind_the_entry_point_the_gateway_door_stays_on_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in _IMAGE_BINDS.items():
        monkeypatch.setenv(name, value)
    assert await _gateway_door_bind(_app(tmp_path, entrypoint=True), monkeypatch) == "127.0.0.1"


def test_the_machine_port_binds_where_control_would(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _IMAGE_BINDS:
        monkeypatch.delenv(name, raising=False)
    # A plain install that advertises nothing.
    assert machine_port.bind_host_for(_app(tmp_path / "a")) == "127.0.0.1"
    for name, value in _IMAGE_BINDS.items():
        monkeypatch.setenv(name, value)
    # The image, direct ports: control's 8083 is published.
    assert machine_port.bind_host_for(_app(tmp_path / "b")) == "0.0.0.0"
    # Behind the entry point, which serves the node name: control is private.
    assert machine_port.bind_host_for(_app(tmp_path / "c", entrypoint=True, nodes=True)) == (
        "127.0.0.1"
    )
    # Behind the entry point without a node name: control keeps its port.
    assert machine_port.bind_host_for(_app(tmp_path / "d", entrypoint=True)) == "0.0.0.0"


def test_the_machine_port_leaves_the_agent_s_own_server_logging_alone() -> None:
    """compose-acceptance 18: the agent's shutdown lines went missing once the
    machine port built its inner servers at WARNING."""
    import logging

    error, access = logging.getLogger("uvicorn.error"), logging.getLogger("uvicorn.access")
    handler = logging.NullHandler()
    access.addHandler(handler)
    saved = (error.level, access.level, access.propagate)
    error.setLevel(logging.INFO)
    access.setLevel(logging.INFO)
    try:
        inner = machine_port._Inner(lambda *_: None, None)
        inner.sock.close()
        assert error.level == logging.INFO and access.level == logging.INFO
        assert handler in access.handlers
        assert inner.server.config.access_log is False
    finally:
        access.removeHandler(handler)
        error.setLevel(saved[0])
        access.setLevel(saved[1])
        access.propagate = saved[2]
