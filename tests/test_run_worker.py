"""The durable worker uses the same launch guard and admission as HTTP routes."""

from types import SimpleNamespace

import pytest

from eugene_plexus_agent.companions import companion_name
from eugene_plexus_agent.routes import runtimes
from eugene_plexus_agent.run_worker import NodeActions, RunWorker, runtime_spec
from eugene_plexus_agent.runtime_context import NodeContext

MODEL = {
    "id": "m",
    "name": "Example 8B",
    "path": "/models/m.gguf",
    "format": "gguf",
    "contextLength": 40960,
}
PROFILE = {
    "id": "p",
    "name": "default",
    "engine": "llama_cpp",
    "default": True,
    "flags": {"contextSize": 4096},
    "extraArgs": [],
    "env": {},
}


@pytest.mark.asyncio
async def test_replay_after_declaration_starts_only_once(authed_client, stub_runtime_supervisor):
    actions = NodeActions(authed_client.app)
    first = await actions.launch(MODEL, PROFILE, start=True)
    second = await actions.launch(MODEL, PROFILE, start=True)
    assert first["name"] == second["name"] == "example-8b"
    assert stub_runtime_supervisor.calls.count(("add_and_start", "example-8b")) == 1
    assert len(authed_client.app.state.agent_state.list_runtime_specs()) == 1


@pytest.mark.asyncio
async def test_skip_replay_keeps_a_stopped_declaration(authed_client, stub_runtime_supervisor):
    actions = NodeActions(authed_client.app)
    await actions.launch(MODEL, PROFILE, start=False)
    await actions.launch(MODEL, PROFILE, start=False)
    assert not stub_runtime_supervisor.is_running("example-8b")
    assert stub_runtime_supervisor.calls.count(("add_and_start", "example-8b")) == 1
    await actions.launch(MODEL, PROFILE, start=True)
    assert stub_runtime_supervisor.is_running("example-8b")


@pytest.mark.asyncio
async def test_a_name_collision_never_adopts_different_settings(authed_client):
    actions = NodeActions(authed_client.app)
    await actions.launch(MODEL, PROFILE, start=False)
    with pytest.raises(ValueError, match="different flags"):
        await actions.launch(MODEL, {**PROFILE, "flags": {"contextSize": 8192}}, start=True)


@pytest.mark.asyncio
async def test_replay_repairs_a_declaration_missing_its_companion(authed_client):
    state = authed_client.app.state.agent_state
    state.add_runtime(runtime_spec(MODEL, PROFILE, start=False))
    name = companion_name("example-8b")
    assert state.get_topology_entry(name) is None
    await NodeActions(authed_client.app).launch(MODEL, PROFILE, start=False)
    assert state.get_topology_entry(name) is not None


@pytest.mark.asyncio
async def test_a_new_approved_run_can_retry_an_old_failed_install(app, monkeypatch):
    from eugene_plexus_agent import run_worker

    failed = SimpleNamespace(model_dump=lambda **_: {"state": "failed", "error": "Disk full"})
    started = SimpleNamespace(model_dump=lambda **_: {"state": "downloading"})
    monkeypatch.setattr(
        run_worker, "installer_for", lambda _: SimpleNamespace(snapshot=lambda: failed)
    )
    attempts = []

    async def install(engine):
        attempts.append(engine)
        return started

    class FakeActions(NodeActions):
        async def engines(self):
            return [{"engine": "llama_cpp", "available": False}]

    monkeypatch.setattr(runtimes, "install_engine", install)
    actions = FakeActions(app)
    assert await actions.install("llama_cpp") == {"state": "downloading"}
    with pytest.raises(ValueError, match="Disk full"):
        await actions.install("llama_cpp", previous={"state": "downloading"})
    assert attempts == ["llama_cpp"]


@pytest.mark.asyncio
async def test_worker_preserves_mlx_quantization_compatibility(app):
    class FakeActions(NodeActions):
        async def engines(self):
            return [
                {"engine": e, "available": True, "modelFormats": ["safetensors"]}
                for e in ["vllm", "mlx"]
            ]

    worker = RunWorker(app, node_actions=FakeActions(app))
    job = {
        "step": "checking",
        "engine": None,
        "model": {
            **MODEL,
            "format": "safetensors",
            "safetensors": {"mlxQuantization": {"bits": 4}},
        },
    }
    result = await worker.advance(None, "/unused", job, {})
    assert result == {"step": "settings", "engine": "mlx"}


@pytest.mark.asyncio
async def test_experts_context_uses_free_memory_on_the_target_node(app, monkeypatch):
    async def admission(context, spec):
        assert isinstance(context, NodeContext)
        return SimpleNamespace(maxContextLength=0, fit="too_large")

    monkeypatch.setattr(runtimes, "_admission_for", admission)

    class Library:
        async def operation_request(self, method, path, **kwargs):
            assert kwargs["params"]["vramBytes"] == 24 * 1024**3
            assert kwargs["params"]["gpuCount"] == 1
            return {"fit": {"offload": "experts"}, "maxContextExpertsInRam": 61440}

    assert await NodeActions(app).context_size(Library(), MODEL, "llama_cpp") == 40960


def test_runtime_composition_keeps_profile_flags_env_and_arguments():
    profile = {
        **PROFILE,
        "default": False,
        "name": "Long Context",
        "extraArgs": ["--verbose"],
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
    }
    spec = runtime_spec(MODEL, profile, start=False)
    assert spec.name == "example-8b-long-context"
    assert spec.flags["contextSize"] == 4096
    assert spec.env == profile["env"]
    assert spec.extraArgs == profile["extraArgs"]
    assert spec.autoStart is False
