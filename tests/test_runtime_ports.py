"""Runtime and companion-driver ports skip what the machine already holds.

Found by the per-node m7 run on 2026-09-25: the agent assigned 8090 up
from its own records alone, so on a box where another install's engine
held 8090 and its driver 8091, a second agent put its own there -- and on
Windows a second bind can share a port rather than fail. First-boot
seeding already asked the socket (`ports.first_free`); this path did not.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from eugene_plexus_agent import state as state_module
from eugene_plexus_agent._generated.models import EngineKind, RuntimeSpec
from eugene_plexus_agent.state import AgentState

# Captured at import, before the suite-wide fixture pins it for every test.
_real_held_elsewhere = state_module._held_elsewhere


def _state(tmp_path: Path) -> AgentState:
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    return state


def _spec(name: str, port: int | None = None) -> RuntimeSpec:
    return RuntimeSpec(
        name=name, engine=EngineKind.llama_cpp, modelPath=f"/m/{name}.gguf", port=port
    )


def _holding(monkeypatch: pytest.MonkeyPatch, *held: int) -> None:
    monkeypatch.setattr(state_module, "_held_elsewhere", lambda port: port in held)


def test_a_runtime_skips_ports_something_else_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _holding(monkeypatch, 8090, 8091)
    assert _state(tmp_path).add_runtime(_spec("qwen")).port == 8092


def test_a_companion_skips_them_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _holding(monkeypatch, 8090, 8091)
    state = _state(tmp_path)
    runtime = state.add_runtime(_spec("qwen"))
    assert runtime.port == 8092
    assert state.allocate_component_port() == 8093


def test_nothing_held_keeps_the_documented_start(tmp_path: Path) -> None:
    """The probe only moves a port off 8090 when 8090 is actually held."""
    assert _state(tmp_path).add_runtime(_spec("qwen")).port == 8090


def test_a_port_the_operator_picks_is_kept_even_when_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit value wins; the engine's own failure says what holds it."""
    _holding(monkeypatch, 8090)
    assert _state(tmp_path).add_runtime(_spec("qwen", port=8090)).port == 8090


def test_the_real_probe_sees_a_real_listener() -> None:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        assert _real_held_elsewhere(port) is True
    assert _real_held_elsewhere(port) is False
