"""The companion driver: declared with the runtime, gone with it.

Through the routes, with the stub supervisors, because the whole point
is the coupling between two collections that used to know nothing of
each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_agent import companions
from eugene_plexus_agent._generated.models import RuntimeSpec
from eugene_plexus_agent.state import AgentState

from .conftest import StubSupervisor


def _runtime(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "qwen3-a",
        "engine": "llama_cpp",
        "modelPath": "/models/Qwen3-1.7B-Q8_0.gguf",
    }
    body.update(overrides)
    return body


def _components(client: TestClient) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in client.get("/v1/components").json()["components"]}


def test_declaring_a_runtime_declares_its_companion(
    authed_client: TestClient, settings: Any, stub_supervisor: StubSupervisor
) -> None:
    created = authed_client.post("/v1/runtimes", json=_runtime())
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["driver"] == "qwen3-a-driver"

    comps = _components(authed_client)
    assert "qwen3-a-driver" in comps
    companion = comps["qwen3-a-driver"]
    assert companion["kind"] == "inference-driver"
    # A port from the runtime range, distinct from the runtime's own.
    assert companion["url"].startswith("http://127.0.0.1:")
    assert urlparse(companion["url"]).port != body["port"]

    # Its config is three lines: follow the runtime by name, serve its alias.
    config_path = Path(companion["spawn"]["configFile"])
    assert config_path.parent == settings.config_file.parent / "drivers"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert document == {
        "provider": "openai_compat_custom",
        "runtimeName": "qwen3-a",
        "modelId": "Qwen3-1.7B-Q8_0",
    }
    # And it was handed to the component supervisor to spawn.
    assert ("add_and_start", "qwen3-a-driver") in stub_supervisor.calls


def test_the_companion_port_is_not_reused_by_a_later_runtime(authed_client: TestClient) -> None:
    first = authed_client.post("/v1/runtimes", json=_runtime(name="a")).json()
    companion = _components(authed_client)["a-driver"]
    companion_port = urlparse(companion["url"]).port
    second = authed_client.post("/v1/runtimes", json=_runtime(name="b")).json()
    assert len({first["port"], companion_port, second["port"]}) == 3


def test_auto_driver_false_declares_no_companion(authed_client: TestClient) -> None:
    body = authed_client.post("/v1/runtimes", json=_runtime(autoDriver=False)).json()
    assert body.get("driver") is None
    assert "qwen3-a-driver" not in _components(authed_client)


def test_deleting_the_runtime_removes_the_companion(
    authed_client: TestClient, stub_supervisor: StubSupervisor
) -> None:
    authed_client.post("/v1/runtimes", json=_runtime())
    assert authed_client.delete("/v1/runtimes/qwen3-a").status_code == 204
    assert "qwen3-a-driver" not in _components(authed_client)
    assert ("remove_and_stop", "qwen3-a-driver") in stub_supervisor.calls


def test_changing_the_alias_retargets_and_restarts_the_companion(
    authed_client: TestClient, stub_supervisor: StubSupervisor
) -> None:
    authed_client.post("/v1/runtimes", json=_runtime())
    stub_supervisor.calls.clear()
    updated = authed_client.patch("/v1/runtimes/qwen3-a", json=_runtime(modelAlias="fast"))
    assert updated.status_code == 200, updated.text
    config_path = Path(_components(authed_client)["qwen3-a-driver"]["spawn"]["configFile"])
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["modelId"] == "fast"
    assert ("restart", "qwen3-a-driver") in stub_supervisor.calls


def test_renaming_the_runtime_renames_the_companion(authed_client: TestClient) -> None:
    authed_client.post("/v1/runtimes", json=_runtime())
    authed_client.patch("/v1/runtimes/qwen3-a", json=_runtime(name="qwen3-b"))
    comps = _components(authed_client)
    assert "qwen3-a-driver" not in comps
    assert "qwen3-b-driver" in comps


def test_a_hand_made_component_holding_the_name_is_a_conflict(authed_client: TestClient) -> None:
    authed_client.post(
        "/v1/components",
        json={
            "name": "qwen3-a-driver",
            "kind": "inference-driver",
            "url": "http://127.0.0.1:8081",
            "spawn": {"configFile": "/somewhere/else.yaml"},
        },
    )
    refused = authed_client.post("/v1/runtimes", json=_runtime())
    assert refused.status_code == 409
    assert "not a companion" in refused.json()["detail"]["detail"]
    # Nothing was declared.
    assert authed_client.get("/v1/runtimes").json()["runtimes"] == []

    # Opting out makes the hand-made driver the way to front it.
    assert authed_client.post("/v1/runtimes", json=_runtime(autoDriver=False)).status_code == 201


@pytest.mark.anyio
async def test_reconcile_creates_missing_companions_and_leaves_the_rest(tmp_path: Path) -> None:
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    state.add_runtime(RuntimeSpec.model_validate(_runtime(name="with")))
    state.add_runtime(RuntimeSpec.model_validate(_runtime(name="without", autoDriver=False)))
    # A companion that already exists is left alone.
    existing = RuntimeSpec.model_validate(_runtime(name="already"))
    state.add_runtime(existing)
    await companions.ensure_companion(state, None, existing)

    supervisor = StubSupervisor()
    created = await companions.reconcile(state, supervisor)
    assert created == ["with-driver"]
    names = {e.name for e in state.list_topology_entries()}
    assert names == {"with-driver", "already-driver"}
    assert ("add_and_start", "with-driver") in supervisor.calls
    assert ("add_and_start", "already-driver") not in supervisor.calls


@pytest.mark.anyio
async def test_reconcile_survives_a_conflict(tmp_path: Path) -> None:
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    state.add_runtime(RuntimeSpec.model_validate(_runtime(name="taken")))
    state.add_topology_entry(
        companions.ComponentEntry.model_validate(
            {
                "name": "taken-driver",
                "kind": "inference-driver",
                "url": "http://127.0.0.1:8081",
                "spawn": {"configFile": "/elsewhere.yaml"},
            }
        )
    )
    state.add_runtime(RuntimeSpec.model_validate(_runtime(name="fine")))
    created = await companions.reconcile(state, None)
    assert created == ["fine-driver"]


def test_is_companion_is_about_the_config_path_not_the_name(tmp_path: Path) -> None:
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    ours = companions.ComponentEntry.model_validate(
        {
            "name": "x-driver",
            "kind": "inference-driver",
            "url": "http://127.0.0.1:8091",
            "spawn": {"configFile": str(companions.companion_config_path(state, "x-driver"))},
        }
    )
    theirs = ours.model_copy(
        update={"spawn": ours.spawn.model_copy(update={"configFile": "/o.yaml"})}
    )  # type: ignore[union-attr]
    assert companions.is_companion(ours, state) is True
    assert companions.is_companion(theirs, state) is False
