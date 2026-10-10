"""The durable worker uses the same launch guard and admission as HTTP routes."""

from types import SimpleNamespace

import httpx
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


MLX_MODEL = {**MODEL, "format": "safetensors", "safetensors": {"mlxQuantization": {"bits": 4}}}


class _BothSafetensorsEngines(NodeActions):
    async def engines(self):
        return [
            {"engine": e, "available": True, "modelFormats": ["safetensors"]}
            for e in ["vllm", "mlx"]
        ]


class _Judge:
    """A library that answers `POST /v1/eligibility` with given verdicts."""

    def __init__(self, verdicts=None, *, status=None):
        self.verdicts, self.status, self.asked = verdicts, status, []

    async def operation_request(self, method, path, **kwargs):
        self.asked.append((method, path, kwargs.get("json")))
        if self.status:
            request = httpx.Request(method, "http://library" + path)
            raise httpx.HTTPStatusError(
                "refused", request=request, response=httpx.Response(self.status, request=request)
            )
        return {"models": [{"modelId": "m", "level": "works_here", "engines": self.verdicts}]}


def _verdict(engine, verdict, *, available=True, reason="runs it as it is"):
    return {"engine": engine, "verdict": verdict, "available": available, "reason": reason}


@pytest.mark.asyncio
async def test_run_takes_the_engine_the_library_judges_best(app):
    # The MLX rule lives in the library now (LS1); Run follows its order.
    library = _Judge([_verdict("mlx", "runs"), _verdict("vllm", "no", reason="cannot load MLX")])
    worker = RunWorker(app, node_actions=_BothSafetensorsEngines(app))
    job = {"step": "checking", "engine": None, "model": MLX_MODEL}
    result = await worker.advance(library, "/unused", job, {})
    assert result == {"step": "settings", "engine": "mlx"}
    method, path, body = library.asked[0]
    assert (method, path, body["models"]) == ("POST", "/v1/eligibility", ["m"])
    # An engine reported without `accepts` is judged on its formats alone.
    assert body["engines"][0]["accepts"] == [{"format": "safetensors"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 422])
async def test_a_library_older_than_eligibility_keeps_the_mlx_rule(app, status):
    # A container root updated after its workers answers 404 (2026-10-09);
    # one older than LS3 refuses Strata's `prepared` requirement with 422.
    worker = RunWorker(app, node_actions=_BothSafetensorsEngines(app))
    job = {"step": "checking", "engine": None, "model": MLX_MODEL}
    result = await worker.advance(_Judge(status=status), "/unused", job, {})
    assert result == {"step": "settings", "engine": "mlx"}


@pytest.mark.asyncio
async def test_when_nothing_can_run_it_each_engine_says_why(app):
    library = _Judge(
        [
            _verdict("vllm", "no", reason="cannot load MLX-quantized weights; only MLX reads them"),
            _verdict("mlx", "runs", available=False),
        ]
    )
    worker = RunWorker(app, node_actions=_BothSafetensorsEngines(app))
    job = {"step": "checking", "engine": None, "model": MLX_MODEL}
    with pytest.raises(ValueError) as refused:
        await worker.advance(library, "/unused", job, {})
    assert "vllm cannot load MLX-quantized weights" in str(refused.value)
    assert "mlx is not installed here" in str(refused.value)


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


@pytest.mark.asyncio
async def test_a_run_suggests_vllm_no_context_it_would_refuse(app, monkeypatch):
    """A profile carrying `contextSize` is refused for vLLM at launch (LS6,
    library#9): Run measures nothing and suggests nothing for it."""

    async def admission(context, spec):
        raise AssertionError("vLLM's context was measured for a flag it refuses")

    monkeypatch.setattr(runtimes, "_admission_for", admission)
    assert await NodeActions(app).context_size(object(), MODEL, "vllm") is None


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
