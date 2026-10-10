"""A person's Stop outlives the agent (agent#11).

Troy, 2026-10-09 and 2026-10-10: a model he had stopped to free the GPU
was running again after every update and every reboot, and took the VRAM
Strata needed. The stop reason lived only in memory, and boot started
every runtime whose `autoStart` was not false.

Now a stop for `operator` is kept in `agent.yaml` until something starts
the runtime; `idle` and `measurement` are not. `autoStart` is changed on
its own, without restarting the engine.
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_agent import runtimes as runtimes_module
from eugene_plexus_agent._generated.models import RuntimeSpec, RuntimeStatus, StopReason
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.runtimes import RuntimeSupervisor
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.state import AgentState

from .conftest import TEST_PASSPHRASE, StubRuntimeSupervisor, StubSupervisor, local_service_token
from .test_runtime_end_to_end import _FAKE_ENGINE, _await_status, _FakeEngineAdapter


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _spec(**overrides: Any) -> RuntimeSpec:
    body: dict[str, Any] = {
        "name": "huihui",
        "engine": "llama_cpp",
        "modelPath": "/models/Huihui-Q6_K_L.gguf",
        "host": "127.0.0.1",
    }
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


def _state(path: Path, *specs: RuntimeSpec) -> AgentState:
    state = AgentState(path)
    state.load()
    for spec in specs:
        state.add_runtime(spec)
    return state


def _reloaded(path: Path) -> AgentState:
    state = AgentState(path)
    state.load()
    return state


# --- what agent.yaml keeps ----------------------------------------------------


def test_a_remembered_stop_is_in_agent_yaml_and_survives_a_reload(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    _state(path, _spec()).remember_stop("huihui")
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["stoppedRuntimes"] == ["huihui"]
    assert _reloaded(path).remembered_stops() == {"huihui"}


def test_forgetting_it_takes_it_out_of_the_file(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    state = _state(path, _spec())
    state.remember_stop("huihui")
    state.forget_stop("huihui")
    assert "stoppedRuntimes" not in yaml.safe_load(path.read_text(encoding="utf-8"))
    assert _reloaded(path).remembered_stops() == set()


def test_only_declared_runtimes_are_remembered(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    state = _state(path, _spec())
    state.remember_stop("never-declared")
    assert state.remembered_stops() == set()
    # And a name in the file with no declaration beside it is dropped.
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["stoppedRuntimes"] = ["huihui", "gone"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert _reloaded(path).remembered_stops() == {"huihui"}


def test_removing_or_renaming_a_runtime_forgets_its_stop(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    state = _state(path, _spec(), _spec(name="other"))
    state.remember_stop("huihui")
    state.remember_stop("other")
    state.remove_runtime("huihui")
    renamed = state.get_runtime_spec("other")
    assert renamed is not None
    state.update_runtime("other", renamed.model_copy(update={"name": "renamed"}))
    assert _reloaded(path).remembered_stops() == set()
    # A runtime declared later under the old name does not inherit it.
    state.add_runtime(_spec())
    assert state.remembered_stops() == set()


def test_auto_start_changes_alone(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    state = _state(path, _spec(port=8123, flags={"ctx-size": 8192}))
    updated = state.set_runtime_auto_start("huihui", False)
    assert updated is not None and updated.autoStart is False
    kept = _reloaded(path).get_runtime_spec("huihui")
    assert kept is not None
    assert kept.autoStart is False
    assert kept.port == 8123
    assert kept.flags == {"ctx-size": 8192}
    assert state.set_runtime_auto_start("nope", False) is None


# --- the supervisor -----------------------------------------------------------


class _Recording(RuntimeSupervisor):
    """The real supervisor, recording `add_and_start` instead of spawning."""

    def __init__(self, state: AgentState) -> None:
        super().__init__(log=logging.getLogger("test"), stop_memory=state)
        self.started: list[str] = []

    def add_and_start(self, spec: RuntimeSpec) -> None:
        self.started.append(spec.name)


async def test_only_a_persons_stop_is_remembered(tmp_path: Path) -> None:
    state = _state(tmp_path / "agent.yaml", _spec(), _spec(name="b"), _spec(name="c"))
    supervisor = _Recording(state)
    await supervisor.stop_one("huihui")
    await supervisor.stop_one("b", reason=StopReason.idle)
    await supervisor.stop_one("c", reason=StopReason.measurement)
    assert state.remembered_stops() == {"huihui"}


async def test_boot_starts_what_nobody_stopped(tmp_path: Path) -> None:
    state = _state(tmp_path / "agent.yaml", _spec(), _spec(name="kept"))
    state.remember_stop("kept")
    supervisor = _Recording(state)
    for spec in state.list_runtime_specs():
        supervisor.start_at_boot(spec)
    assert supervisor.started == ["huihui"]
    kept = state.get_runtime_spec("kept")
    assert kept is not None
    runtime = supervisor.compose(kept)
    assert runtime.status is RuntimeStatus.stopped
    assert runtime.stopReason is StopReason.operator


@pytest.fixture
def fake_engine(tmp_path: Path) -> Path:
    script = tmp_path / "fake_llama_server.py"
    script.write_text(_FAKE_ENGINE, encoding="utf-8")
    return script


async def test_a_stopped_model_stays_stopped_across_an_agent_restart_until_started(
    tmp_path: Path, fake_engine: Path
) -> None:
    """THE BUG, with a real engine process: stop it, restart the agent,
    and it must not come back; Start brings it back and forgets the stop."""
    path = tmp_path / "agent.yaml"
    spec = _spec(port=_free_port())
    original = runtimes_module.ADAPTERS.copy()
    runtimes_module.ADAPTERS[spec.engine] = _FakeEngineAdapter(fake_engine, ready_after=0.0)
    try:
        first = RuntimeSupervisor(log=logging.getLogger("test"), stop_memory=_state(path, spec))
        first.add_and_start(spec)
        await first.start_readiness_loop(lambda: [spec])
        assert await _await_status(first, spec, RuntimeStatus.ready) is RuntimeStatus.ready
        await first.stop_one(spec.name)
        # The agent shuts down: an update, a reboot.
        await first.stop_all()

        state = _reloaded(path)
        second = RuntimeSupervisor(log=logging.getLogger("test"), stop_memory=state)
        try:
            for declared in state.list_runtime_specs():
                second.start_at_boot(declared)
            await second.start_readiness_loop(state.list_runtime_specs)
            runtime = second.compose(spec)
            assert runtime.status is RuntimeStatus.stopped
            assert runtime.stopReason is StopReason.operator
            assert runtime.pid is None
            assert not second.is_running(spec.name)

            # Start: what the console's button and a gateway wake both do.
            second.add_and_start(spec.model_copy(update={"autoStart": True}))
            assert await _await_status(second, spec, RuntimeStatus.ready) is RuntimeStatus.ready
            assert state.remembered_stops() == set()
            assert _reloaded(path).remembered_stops() == set()
        finally:
            await second.stop_all()
    finally:
        runtimes_module.ADAPTERS.clear()
        runtimes_module.ADAPTERS.update(original)


# --- wiring: the agent's own boot reads agent.yaml ------------------------------


def test_the_agent_boots_a_stopped_runtime_stopped(tmp_path: Path) -> None:
    """`create_app` with the real runtime supervisor: what was stopped
    before the restart is reported stopped by someone, and nothing ran."""
    path = tmp_path / "agent.yaml"
    state = _state(path, _spec(port=_free_port(), autoDriver=False))
    state.remember_stop("huihui")
    app = create_app(settings=Settings(config_file=path, default_topology=False))
    app.state.supervisor = StubSupervisor()
    with TestClient(app) as client:
        init = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
        client.headers["Authorization"] = f"Bearer {init.json()['sessionToken']}"
        runtime = client.get("/v1/runtimes/huihui").json()
        assert runtime["status"] == "stopped", runtime
        assert runtime["stopReason"] == "operator"
        assert runtime.get("pid") is None
    # Initializing rewrote agent.yaml; the stop is still in it.
    assert _reloaded(path).remembered_stops() == {"huihui"}


# --- PUT /v1/runtimes/{name}/auto-start -----------------------------------------


def _declare(client: TestClient) -> None:
    created = client.post(
        "/v1/runtimes",
        json={"name": "huihui", "engine": "llama_cpp", "modelPath": "/models/h.gguf"},
    )
    assert created.status_code in (200, 201), created.text


def test_auto_start_is_saved_without_touching_the_engine(
    authed_client: TestClient, stub_runtime_supervisor: StubRuntimeSupervisor, settings: Settings
) -> None:
    _declare(authed_client)
    before = list(stub_runtime_supervisor.calls)
    response = authed_client.put("/v1/runtimes/huihui/auto-start", json={"autoStart": False})
    assert response.status_code == 200, response.text
    assert response.json()["autoStart"] is False
    # Running, and still running: no stop, no restart, no start.
    assert response.json()["status"] == "starting"
    assert stub_runtime_supervisor.calls == before
    assert authed_client.get("/v1/runtimes/huihui").json()["autoStart"] is False
    saved = _reloaded(settings.config_file).get_runtime_spec("huihui")
    assert saved is not None and saved.autoStart is False

    back = authed_client.put("/v1/runtimes/huihui/auto-start", json={"autoStart": True})
    assert back.json()["autoStart"] is True


def test_auto_start_for_a_runtime_that_is_not_there_is_a_404(authed_client: TestClient) -> None:
    response = authed_client.put("/v1/runtimes/nope/auto-start", json={"autoStart": False})
    assert response.status_code == 404


def test_auto_start_is_the_operators_not_the_gateways(client: TestClient) -> None:
    init = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    operator = {"Authorization": f"Bearer {init.json()['sessionToken']}"}
    created = client.post(
        "/v1/runtimes",
        json={"name": "huihui", "engine": "llama_cpp", "modelPath": "/models/h.gguf"},
        headers=operator,
    )
    assert created.status_code in (200, 201), created.text
    gateway = {"Authorization": f"Bearer {local_service_token(client.app, 'gateway')}"}  # type: ignore[arg-type]
    refused = client.put(
        "/v1/runtimes/huihui/auto-start", json={"autoStart": False}, headers=gateway
    )
    assert refused.status_code == 401
