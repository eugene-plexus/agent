"""Pause models on this machine, and keep serving (row 35).

Troy, 2026-09-18: an RTX owner wants the graphics card back to play a
game, and Stop Eugene took the console and this machine's place in the
install with it. Pause stops every model here and nothing else; while it
lasts nothing starts a model here, not a request, a Run, a benchmark or
a reboot; Resume starts again what Pause stopped, through admission.

The console's half is `POST /v1/node/pause` and `/v1/node/resume`, with
an operator session only. The tray's half is the named pipe in
`tray_pipe.py`, tested in `test_tray_pipe.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_agent._generated.models import RuntimeSpec
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.runtimes import RuntimeSupervisor
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.state import AgentState

from .conftest import TEST_PASSPHRASE, StubRuntimeSupervisor, StubSupervisor, local_service_token


def _declare(client: TestClient, tmp_path: Path, name: str, **extra: Any) -> None:
    path = tmp_path / f"{name}.gguf"
    path.write_bytes(b"fixture")
    spec = {"name": name, "engine": "llama_cpp", "modelPath": str(path), "autoDriver": False}
    response = client.post("/v1/runtimes", json={**spec, **extra})
    assert response.status_code == 201, response.text


@pytest.fixture
def operator(
    authed_client: TestClient, app: Any, stub_runtime_supervisor: StubRuntimeSupervisor
) -> TestClient:
    """An operator's client, with the stub supervisor reading this agent's
    stop memory as the real one does (`app.py` wires `AgentState` there)."""
    stub_runtime_supervisor._stop_memory = app.state.agent_state
    return authed_client


def _pause_of(body: dict[str, Any]) -> dict[str, Any]:
    """`NodePause` without `trayChannel`, which only Windows reports."""
    return {k: v for k, v in body.items() if k != "trayChannel"}


def _gateway(app: Any) -> dict[str, str]:
    return {"Authorization": f"Bearer {local_service_token(app, 'gateway')}"}


def _paused_detail(response: Any) -> str:
    """The sentence a refusal carries, from the Problem the agent answers."""
    detail = response.json()["detail"]
    return str(detail["detail"] if isinstance(detail, dict) else detail)


# --- the design's test: Pause stops, and a wake is refused -------------------


def test_pause_stops_and_refuses_wake(
    operator: TestClient, app: Any, stub_runtime_supervisor: StubRuntimeSupervisor, tmp_path
) -> None:
    _declare(operator, tmp_path, "busy")
    assert stub_runtime_supervisor.is_running("busy")

    paused = operator.post("/v1/node/pause")
    assert paused.status_code == 200, paused.text
    body = paused.json()
    assert body["pause"]["paused"] is True
    assert body["pause"]["by"] == "console"
    assert body["pause"]["runtimes"] == ["busy"]
    assert [(r["name"], r["state"]) for r in body["results"]] == [("busy", "stopped")]

    runtime = operator.get("/v1/runtimes/busy").json()
    assert runtime["status"] == "stopped"
    assert runtime["stopReason"] == "paused"
    assert not stub_runtime_supervisor.is_running("busy")

    # The gateway's wake: refused, and the refusal names the pause.
    woken = operator.post("/v1/runtimes/busy/start", headers=_gateway(app))
    assert woken.status_code == 409, woken.text
    sentence = _paused_detail(woken)
    assert "paused" in sentence and "Resume" in sentence
    assert not stub_runtime_supervisor.is_running("busy")

    node = operator.get("/v1/node").json()
    assert node["pause"]["paused"] is True
    assert node["pause"]["runtimes"] == ["busy"]


@pytest.mark.parametrize(
    "verb,path",
    [
        ("post", "/v1/runtimes"),
        ("patch", "/v1/runtimes/busy"),
        ("post", "/v1/runtimes/busy/start"),
        ("post", "/v1/runtimes/busy/restart"),
        ("post", "/v1/benchmarks"),
        ("post", "/v1/profile-builds"),
        ("post", "/v1/serving-measurements"),
    ],
)
def test_nothing_starts_a_model_while_paused(
    operator: TestClient,
    stub_runtime_supervisor: StubRuntimeSupervisor,
    tmp_path: Path,
    verb: str,
    path: str,
) -> None:
    _declare(operator, tmp_path, "busy")
    assert operator.post("/v1/node/pause").status_code == 200
    before = list(stub_runtime_supervisor.calls)
    model = tmp_path / "other.gguf"
    model.write_bytes(b"fixture")
    runtime = {"name": "other" if path == "/v1/runtimes" else "busy", "engine": "llama_cpp"}
    runtime["modelPath"] = str(model)
    body: dict[str, Any] = runtime
    if path == "/v1/benchmarks":
        body = {"modelId": "m", "profileId": "p", "profileName": "t", "runtime": runtime}
    elif path == "/v1/profile-builds":
        body = {"modelId": "m", "runtime": runtime, "accuracy": "high"}
    elif path == "/v1/serving-measurements":
        body = {"runtime": "busy"}
    response = getattr(operator, verb)(path, json=body)
    assert response.status_code == 409, response.text
    assert "paused" in _paused_detail(response)
    assert stub_runtime_supervisor.calls == before


def test_a_second_pause_changes_nothing(operator: TestClient, tmp_path: Path) -> None:
    _declare(operator, tmp_path, "busy")
    first = operator.post("/v1/node/pause").json()
    second = operator.post("/v1/node/pause")
    assert second.status_code == 200
    assert second.json()["pause"] == first["pause"]
    assert second.json()["results"] == []


def test_a_stop_request_cannot_claim_a_pause(
    operator: TestClient, stub_runtime_supervisor: StubRuntimeSupervisor, tmp_path: Path
) -> None:
    """`paused` is Pause's alone: a stop that names it is a person's stop,
    and the runtime never reads Paused on a machine that is not paused."""
    _declare(operator, tmp_path, "busy")
    stopped = operator.post("/v1/runtimes/busy/stop", json={"reason": "paused"})
    assert stopped.status_code == 202
    assert operator.get("/v1/runtimes/busy").json()["stopReason"] == "operator"
    assert _pause_of(operator.get("/v1/node").json()["pause"]) == {"paused": False}


# --- who may pause -----------------------------------------------------------


def test_pause_refuses_gateway_and_client_tokens(
    operator: TestClient, app: Any, stub_runtime_supervisor: StubRuntimeSupervisor, tmp_path
) -> None:
    _declare(operator, tmp_path, "busy")
    made = operator.post("/v1/auth/client-keys", json={"name": "app"})
    assert made.status_code in (200, 201), made.text
    key = {"Authorization": f"Bearer {made.json()['token']}"}
    control = {"Authorization": f"Bearer {local_service_token(app, 'control')}"}
    before = list(stub_runtime_supervisor.calls)
    for path in ("/v1/node/pause", "/v1/node/resume"):
        for headers in (_gateway(app), key, control):
            refused = operator.post(path, headers=headers)
            assert refused.status_code in (401, 403), (path, refused.text)
    assert stub_runtime_supervisor.calls == before
    assert stub_runtime_supervisor.is_running("busy")
    assert operator.get("/v1/node").json()["pause"]["paused"] is False


def test_the_routes_are_changes_for_frozen_trust() -> None:
    """Row 46: another machine's token is refused on them past the limit."""
    from eugene_plexus_agent.dependencies import FROZEN_TRUST_CHANGES

    assert ("POST", "/v1/node/pause") in FROZEN_TRUST_CHANGES
    assert ("POST", "/v1/node/resume") in FROZEN_TRUST_CHANGES


# --- Resume --------------------------------------------------------------------


def test_resume_restarts_through_admission(
    operator: TestClient,
    stub_runtime_supervisor: StubRuntimeSupervisor,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from eugene_plexus_agent._generated.models import AdmissionDecision
    from eugene_plexus_agent.routes import runtimes as runtime_routes

    _declare(operator, tmp_path, "small")
    _declare(operator, tmp_path, "big")
    assert operator.post("/v1/node/pause").json()["pause"]["runtimes"] == ["small", "big"]

    original = runtime_routes._measure_launch
    asked: list[str] = []

    async def measured(request: Any, spec: RuntimeSpec) -> Any:
        asked.append(spec.name)
        admission, places = await original(request, spec)
        if spec.name == "big":
            refused = "not enough GPU memory (21.4 GB needed, 9.8 GB free)"
            admission = admission.model_copy(
                update={"decision": AdmissionDecision.refuse, "reason": refused}
            )
        return admission, places

    monkeypatch.setattr(runtime_routes, "_measure_launch", measured)
    resumed = operator.post("/v1/node/resume")
    assert resumed.status_code == 200, resumed.text
    body = resumed.json()
    assert _pause_of(body["pause"]) == {"paused": False}
    states = {r["name"]: (r["state"], r.get("detail")) for r in body["results"]}
    assert states["small"][0] == "started"
    assert states["big"] == ("refused", "not enough GPU memory (21.4 GB needed, 9.8 GB free)")
    # Through admission, both of them, and never forced.
    assert sorted(asked) == ["big", "small"]
    assert stub_runtime_supervisor.is_running("small")
    assert not stub_runtime_supervisor.is_running("big")
    # The refused one is a person's stop now: it says so, and stays stopped.
    big = operator.get("/v1/runtimes/big").json()
    assert big["status"] == "stopped" and big["stopReason"] == "operator"
    # Starts are allowed again.
    assert operator.post("/v1/runtimes/big/start").status_code in (202, 422)


def test_a_runtime_removed_during_a_pause_leaves_it(operator: TestClient, tmp_path: Path) -> None:
    _declare(operator, tmp_path, "kept")
    _declare(operator, tmp_path, "removed")
    operator.post("/v1/node/pause")
    assert operator.delete("/v1/runtimes/removed").status_code == 204
    assert operator.get("/v1/node").json()["pause"]["runtimes"] == ["kept"]
    results = operator.post("/v1/node/resume").json()["results"]
    assert [(r["name"], r["state"]) for r in results] == [("kept", "started")]


def test_resume_when_not_paused_does_nothing(
    operator: TestClient, stub_runtime_supervisor: StubRuntimeSupervisor
) -> None:
    before = list(stub_runtime_supervisor.calls)
    response = operator.post("/v1/node/resume")
    assert response.status_code == 200
    assert _pause_of(response.json()["pause"]) == {"paused": False}
    assert response.json()["results"] == []
    assert stub_runtime_supervisor.calls == before


# --- a measurement that is running when Pause comes ----------------------------


def test_pause_cancels_a_running_benchmark_and_keeps_its_models_stopped(
    operator: TestClient,
    stub_runtime_supervisor: StubRuntimeSupervisor,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from .test_benchmark_routes import _startable

    body = _startable(monkeypatch, tmp_path)
    _declare(operator, tmp_path, "busy")
    body["stopRuntimes"] = ["busy"]
    started = operator.post("/v1/benchmarks", json=body)
    assert started.status_code == 202, started.text
    assert not stub_runtime_supervisor.is_running("busy")

    paused = operator.post("/v1/node/pause")
    assert paused.status_code == 200, paused.text
    assert paused.json()["pause"]["runtimes"] == ["busy"]
    job = operator.get("/v1/benchmarks").json()["benchmarks"][0]
    assert job["state"] == "cancelled"
    assert [r["state"] for r in job["restarts"]] == ["skipped"]
    assert "paused" in job["restarts"][0]["detail"]
    # The benchmark's own restart did not bring it back.
    assert not stub_runtime_supervisor.is_running("busy")
    assert operator.get("/v1/runtimes/busy").json()["stopReason"] == "paused"


# --- across a restart ------------------------------------------------------------


class _Recording(RuntimeSupervisor):
    """The real supervisor, recording `add_and_start` instead of spawning."""

    def __init__(self, state: AgentState) -> None:
        super().__init__(log=logging.getLogger("test"), stop_memory=state)
        self.started: list[str] = []

    def add_and_start(self, spec: RuntimeSpec) -> None:
        self.started.append(spec.name)


def _state(path: Path, *names: str) -> AgentState:
    state = AgentState(path)
    state.load()
    for name in names:
        state.add_runtime(
            RuntimeSpec.model_validate(
                {
                    "name": name,
                    "engine": "llama_cpp",
                    "modelPath": f"/models/{name}.gguf",
                    "autoDriver": False,
                }
            )
        )
    return state


def test_pause_survives_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reboot during a game: nothing loads the card again. A runtime the
    gateway had unloaded before the pause is held too, and joins the list
    Resume starts, because a boot would have started it."""
    path = tmp_path / "agent.yaml"
    state = _state(path, "busy", "unloaded", "kept-off")
    state.remember_stop("kept-off")
    from eugene_plexus_agent import pause

    state.set_pause(pause.new_record(by="tray", runtimes=["busy"]))

    started: list[str] = []
    monkeypatch.setattr(
        RuntimeSupervisor, "add_and_start", lambda self, spec: started.append(spec.name)
    )
    app = create_app(settings=Settings(config_file=path, default_topology=False))
    app.state.supervisor = StubSupervisor()
    with TestClient(app) as client:
        init = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
        client.headers["Authorization"] = f"Bearer {init.json()['sessionToken']}"
        assert started == []
        for name in ("busy", "unloaded"):
            runtime = client.get(f"/v1/runtimes/{name}").json()
            assert runtime["status"] == "stopped", runtime
            assert runtime["stopReason"] == "paused", runtime
        kept = client.get("/v1/runtimes/kept-off").json()
        assert kept["stopReason"] == "operator"
        node = client.get("/v1/node").json()["pause"]
        assert node["paused"] is True and node["by"] == "tray"
        assert node["runtimes"] == ["busy", "unloaded"]
        woken = client.post("/v1/runtimes/busy/start", headers=_gateway(app))
        assert woken.status_code == 409
        assert started == []
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))["pausedModels"]
    assert saved["runtimes"] == ["busy", "unloaded"]


def test_the_supervisor_boots_paused_runtimes_stopped(tmp_path: Path) -> None:
    state = _state(tmp_path / "agent.yaml", "busy", "free")
    from eugene_plexus_agent import pause

    state.set_pause(pause.new_record(by="console", runtimes=["busy"]))
    supervisor = _Recording(state)
    for spec in state.list_runtime_specs():
        supervisor.start_at_boot(spec)
    assert supervisor.started == ["free"]


def test_a_damaged_pause_record_keeps_the_machine_paused(tmp_path: Path) -> None:
    """A bad file never stops the agent, and never loads the card either."""
    path = tmp_path / "agent.yaml"
    _state(path, "busy")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["pausedModels"] = {"since": "not a time", "by": 7, "runtimes": "busy"}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    state = AgentState(path)
    assert state.load_or_degrade() is None
    assert state.pause_record() is not None
    # The boot holds every runtime that would start, and Resume starts them.
    assert state.hold_paused_at_boot() == ["busy"]
    assert state.remembered_stops() == {"busy"}


def test_the_pause_sentence_names_the_machine_and_the_way_back() -> None:
    from eugene_plexus_agent import pause

    tray = pause.new_record(by="tray", runtimes=[])
    assert pause.sentence(tray, machine="Amish_Station") == (
        "Models on Amish_Station are paused from its tray icon. "
        "Resume them there or in Eugene's console."
    )
    console = pause.new_record(by="console", runtimes=[])
    assert pause.sentence(console, machine="nas").startswith(
        "Models on nas are paused from Eugene's console. Resume them"
    )
