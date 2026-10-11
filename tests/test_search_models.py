"""A search model, as far as the agent can take it (MR3).

An embedding or reranking model has to be started the way llama-server needs
(`test_run_mode.py` is the command line), and everything the agent says about it
afterwards has to be true of what it started:

* what is running is reported from the command line it was started with, because
  llama-server's `/props` says nothing of its task (measured on b11552);
* a running search model has no idle timeout and is not on offer for eviction,
  because the embeddings door never wakes a stopped runtime: an embedder unloaded
  for being idle would simply stop answering;
* it is admitted and reserved at what it will take (a 4,096-token window), not at
  the model's own 32,768;
* an engine that cannot start one refuses to be declared as one.

The supervision here is real: a real subprocess standing in for llama-server, the
real adapter's real command line, the real readiness probe and the real status
mapping. Only the engine's HTTP answers are faked.
"""

from __future__ import annotations

import logging
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_agent import runtimes as runtimes_module
from eugene_plexus_agent._generated.models import (
    EngineKind,
    Origin,
    RuntimeCapabilities,
    RuntimeSpec,
    RuntimeStatus,
)
from eugene_plexus_agent.admission import reservation_bytes
from eugene_plexus_agent.engines import LlamaCppAdapter
from eugene_plexus_agent.engines.base import DiscoveredBinary
from eugene_plexus_agent.routes import runtimes as routes
from eugene_plexus_agent.run_worker import NodeActions, runtime_spec
from eugene_plexus_agent.runtimes import RuntimeSupervisor, validate_spec
from eugene_plexus_agent.state import AgentState

from . import gguf_files as g
from .conftest import StubRuntimeSupervisor, StubSupervisor
from .test_runtime_end_to_end import _FAKE_ENGINE, _await_status


class _RealArgvAdapter(LlamaCppAdapter):
    """The real adapter's real command line, behind a stand-in that answers `/health`.

    The stand-in is launched through the interpreter with the real flags after it,
    so the argv the supervisor records (and the capabilities are read from) is the
    one the adapter built.
    """

    def __init__(self, script: Path, ready_after: float = 0.0) -> None:
        self._script = script
        self._ready_after = ready_after

    def build_argv(self, spec: RuntimeSpec, binary: DiscoveredBinary, port: int) -> list[str]:
        real = super().build_argv(spec, binary, port)
        return [
            sys.executable,
            str(self._script),
            "--ready-after",
            str(self._ready_after),
            *real[1:],
        ]

    def working_directory(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> str | None:
        return None

    def resolve_binary(
        self, spec: RuntimeSpec, *, configured: str | None = None
    ) -> DiscoveredBinary:
        return DiscoveredBinary(path=Path(sys.executable), origin=Origin.configured, version="test")


@pytest.fixture
def stand_in(tmp_path: Path) -> Path:
    script = tmp_path / "fake_llama_server.py"
    script.write_text(_FAKE_ENGINE, encoding="utf-8")
    return script


@pytest.fixture
def adapters() -> Iterator[None]:
    """Whatever a test swaps into the registry is put back."""
    original = runtimes_module.ADAPTERS.copy()
    try:
        yield
    finally:
        runtimes_module.ADAPTERS.clear()
        runtimes_module.ADAPTERS.update(original)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _spec(model: Path, **overrides: Any) -> RuntimeSpec:
    body: dict[str, Any] = {
        "name": "found-in-the-file",
        "engine": EngineKind.llama_cpp,
        "modelPath": str(model),
        "host": "127.0.0.1",
        "port": _free_port(),
    }
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


async def _ready(
    stand_in: Path, spec: RuntimeSpec, *, ready_after: float = 0.0
) -> tuple[RuntimeSupervisor, Any]:
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))
    runtimes_module.ADAPTERS[spec.engine] = _RealArgvAdapter(stand_in, ready_after)
    supervisor.add_and_start(spec)
    await supervisor.start_readiness_loop(lambda: [spec])
    assert await _await_status(supervisor, spec, RuntimeStatus.ready) == RuntimeStatus.ready
    return supervisor, supervisor.compose(spec)


# --- what is running is reported from what it was started with -------------------------


async def test_a_running_embedder_says_it_embeds_and_has_no_idle_timeout(
    stand_in: Path, tmp_path: Path, adapters: None
) -> None:
    """Declared with an idle timeout and start-on-demand, as a hand-written declaration
    or an old placement might: what the gateway reads is what is in effect."""
    spec = _spec(
        g.qwen3_embedding(tmp_path / "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        idleUnloadSeconds=600,
        startOnDemand=True,
    )
    supervisor, view = await _ready(stand_in, spec)
    try:
        assert view.argv is not None and "--embeddings" in view.argv
        assert view.capabilities is not None
        assert view.capabilities.embeddings is True
        assert view.capabilities.reranking is False
        assert view.idleUnloadSeconds is None, "the gateway never unloads an embedder"
        assert view.startOnDemand is None
        # What /props said is still there beside what the command line says.
        assert view.capabilities.contextLength == 4096
        assert view.capabilities.parallelSlots == 2
    finally:
        await supervisor.stop_all()


async def test_a_running_reranker_says_it_reranks_and_does_not_say_it_embeds(
    stand_in: Path, tmp_path: Path, adapters: None
) -> None:
    spec = _spec(g.qwen3_reranker(tmp_path / "reranker.gguf"), idleUnloadSeconds=600)
    supervisor, view = await _ready(stand_in, spec)
    try:
        assert view.argv is not None and "--reranking" in view.argv
        assert view.capabilities is not None
        assert view.capabilities.reranking is True
        assert view.capabilities.embeddings is False, "it answers embeddings with zeros"
        assert view.idleUnloadSeconds is None
    finally:
        await supervisor.stop_all()


async def test_a_chat_model_keeps_everything_it_declared(
    stand_in: Path, tmp_path: Path, adapters: None
) -> None:
    spec = _spec(g.qwen3_chat(tmp_path / "chat.gguf"), idleUnloadSeconds=600, startOnDemand=True)
    supervisor, view = await _ready(stand_in, spec)
    try:
        assert view.capabilities is not None
        assert view.capabilities.embeddings is False
        assert view.capabilities.reranking is False
        assert view.idleUnloadSeconds == 600
        assert view.startOnDemand is True
    finally:
        await supervisor.stop_all()


async def test_chat_declared_on_an_embedding_file_runs_as_chat_and_reports_it(
    stand_in: Path, tmp_path: Path, adapters: None
) -> None:
    """The person's override: the runtime that was told `chat` reports chat, and
    keeps its idle timeout, exactly as a chat model does."""
    spec = _spec(g.qwen3_embedding(tmp_path / "m.gguf"), mode="chat", idleUnloadSeconds=600)
    supervisor, view = await _ready(stand_in, spec)
    try:
        assert view.argv is not None and "--embeddings" not in view.argv
        assert view.capabilities is not None and view.capabilities.embeddings is False
        assert view.idleUnloadSeconds == 600
        assert view.mode is not None and view.mode.value == "chat"
    finally:
        await supervisor.stop_all()


async def test_a_flag_typed_into_extra_args_is_what_it_was_started_as(
    stand_in: Path, tmp_path: Path, adapters: None
) -> None:
    """The old workaround. A chat-looking file started with `--embeddings` typed
    into the profile embeds, and says so: the report is the command line's."""
    spec = _spec(g.qwen3_chat(tmp_path / "m.gguf"), extraArgs=["--embeddings", "--pooling", "last"])
    supervisor = RuntimeSupervisor(
        log=logging.getLogger("test"), get_config={"allowUnrestrictedEngineLaunch": True}.get
    )
    runtimes_module.ADAPTERS[spec.engine] = _RealArgvAdapter(stand_in)
    supervisor.add_and_start(spec)
    await supervisor.start_readiness_loop(lambda: [spec])
    try:
        assert await _await_status(supervisor, spec, RuntimeStatus.ready) == RuntimeStatus.ready
        view = supervisor.compose(spec)
        assert view.capabilities is not None and view.capabilities.embeddings is True
        assert view.idleUnloadSeconds is None
    finally:
        await supervisor.stop_all()


async def test_a_search_model_is_not_idle_unloadable_while_it_is_still_loading(
    stand_in: Path, tmp_path: Path, adapters: None
) -> None:
    """The command line exists from the moment of the spawn, so the minutes before it
    answers are covered too: nothing may count it evictable in them."""
    spec = _spec(g.qwen3_embedding(tmp_path / "m.gguf"), idleUnloadSeconds=600)
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))
    runtimes_module.ADAPTERS[spec.engine] = _RealArgvAdapter(stand_in, ready_after=8.0)
    supervisor.add_and_start(spec)
    await supervisor.start_readiness_loop(lambda: [spec])
    try:
        assert await _await_status(supervisor, spec, RuntimeStatus.loading) == RuntimeStatus.loading
        view = supervisor.compose(spec)
        assert view.idleUnloadSeconds is None
        assert view.capabilities is None, "nothing is read back until it is ready"
    finally:
        await supervisor.stop_all()


def test_a_stopped_runtime_shows_what_was_declared() -> None:
    """With no process there is no command line, so nothing is claimed beyond the
    declaration; the gateway acts only on a ready runtime."""
    spec = RuntimeSpec(
        name="r", engine=EngineKind.llama_cpp, modelPath="/m/a.gguf", idleUnloadSeconds=600
    )
    view = RuntimeSupervisor().compose(spec)
    assert view.status is RuntimeStatus.stopped
    assert view.idleUnloadSeconds == 600


def test_the_declared_mode_is_echoed_and_absent_means_absent() -> None:
    plain = RuntimeSupervisor().compose(
        RuntimeSpec(name="r", engine="llama_cpp", modelPath="/m/a.gguf")
    )
    assert plain.mode is None
    said = RuntimeSupervisor().compose(
        RuntimeSpec(name="r", engine="llama_cpp", modelPath="/m/a.gguf", mode="reranking")
    )
    assert said.mode is not None and said.mode.value == "reranking"


def test_capabilities_with_no_props_still_say_what_it_was_started_as() -> None:
    """`/props` is best-effort: a runtime that serves but will not describe itself is
    still ready, and its task is still known."""
    supervisor = RuntimeSupervisor()
    spec = RuntimeSpec(name="r", engine="llama_cpp", modelPath="/m/a.gguf", idleUnloadSeconds=60)
    from eugene_plexus_agent.engines.base import Ready

    supervisor._processes["r"] = SimpleNamespace(  # type: ignore[assignment]
        state=runtimes_module.ProcessState.starting,
        last_error=None,
        last_restart=None,
        pid=1,
        last_argv=["llama-server", "--embeddings"],
    )
    supervisor._readiness["r"] = Ready(capabilities=None)
    view = supervisor.compose(spec)
    assert view.status is RuntimeStatus.ready
    assert view.capabilities is not None and view.capabilities.embeddings is True
    assert view.idleUnloadSeconds is None


# --- an engine that cannot start one refuses to be declared as one ----------------------


@pytest.mark.parametrize("engine", ["vllm", "mlx", "strata"])
@pytest.mark.parametrize("mode", ["embedding", "reranking"])
def test_another_engine_refuses_a_search_mode(engine: str, mode: str) -> None:
    path = "/m/prepared.eugene-prepared.json" if engine == "strata" else "/m/model"
    reason = validate_spec(RuntimeSpec(name="r", engine=engine, modelPath=path, mode=mode))
    assert reason is not None
    assert engine in reason and mode in reason and "only llama.cpp" in reason


@pytest.mark.parametrize("engine", ["vllm", "mlx", "strata"])
@pytest.mark.parametrize("mode", [None, "auto", "chat"])
def test_another_engine_accepts_auto_and_chat(engine: str, mode: str | None) -> None:
    path = "/m/prepared.eugene-prepared.json" if engine == "strata" else "/m/model"
    body: dict[str, Any] = {"name": "r", "engine": engine, "modelPath": path}
    if mode is not None:
        body["mode"] = mode
    assert validate_spec(RuntimeSpec.model_validate(body)) is None


@pytest.mark.parametrize("mode", [None, "auto", "chat", "embedding", "reranking"])
def test_llama_cpp_accepts_every_mode(mode: str | None) -> None:
    body: dict[str, Any] = {"name": "r", "engine": "llama_cpp", "modelPath": "/m/a.gguf"}
    if mode is not None:
        body["mode"] = mode
    assert validate_spec(RuntimeSpec.model_validate(body)) is None


def _runtime(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "search",
        "engine": "llama_cpp",
        "modelPath": "/models/Qwen3-Embedding-0.6B-Q8_0.gguf",
        "autoStart": False,
    }
    body.update(overrides)
    return body


def test_the_route_refuses_a_search_mode_for_vllm_and_says_why(authed_client: TestClient) -> None:
    refused = authed_client.post(
        "/v1/runtimes", json=_runtime(engine="vllm", modelPath="/models/q", mode="embedding")
    )
    assert refused.status_code == 400
    assert "cannot start a model as 'embedding'" in refused.json()["detail"]["detail"]
    assert authed_client.get("/v1/runtimes").json()["runtimes"] == []


def test_the_route_takes_a_mode_and_reports_it(authed_client: TestClient, settings: Any) -> None:
    created = authed_client.post("/v1/runtimes", json=_runtime(mode="embedding"))
    assert created.status_code == 201, created.text
    assert created.json()["mode"] == "embedding"

    reloaded = AgentState(settings.config_file)
    reloaded.load()
    [spec] = reloaded.list_runtime_specs()
    assert spec.mode is not None and spec.mode.value == "embedding"


def test_a_runtime_that_says_no_mode_writes_none(authed_client: TestClient, settings: Any) -> None:
    """Absent means auto, so the file is not filled with `mode: auto` for every runtime."""
    authed_client.post("/v1/runtimes", json=_runtime())
    [entry] = yaml.safe_load(settings.config_file.read_text(encoding="utf-8"))["runtimes"]
    assert "mode" not in entry


def test_a_mode_the_contract_does_not_know_is_refused_by_the_schema(
    authed_client: TestClient,
) -> None:
    assert authed_client.post("/v1/runtimes", json=_runtime(mode="summarising")).status_code == 422


# --- the driver asks again when a runtime will start as something else ------------------
#
# A companion driver learns whether its engine embeds by trying it, and keeps the
# answer for its life. A runtime that is edited to start as another thing (a person
# choosing Chat for a file that was an embedder, or Search for one that was chat)
# keeps its driver, which would go on listing the model as what it was.


def _driver_restarts(stub: StubSupervisor) -> list[str]:
    return [name for call, name in stub.calls if call == "restart"]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("embedding", "chat"),
        ("chat", "embedding"),
        ("embedding", "reranking"),
        ("reranking", "embedding"),
        # `auto` on a file that is not there is a chat model, so this is a change too.
        ("auto", "embedding"),
    ],
)
def test_a_runtime_that_will_start_as_something_else_gets_a_driver_that_asks_again(
    authed_client: TestClient, stub_supervisor: StubSupervisor, before: str, after: str
) -> None:
    authed_client.post("/v1/runtimes", json=_runtime(mode=before))
    stub_supervisor.calls.clear()
    patched = authed_client.patch("/v1/runtimes/search", json=_runtime(mode=after))
    assert patched.status_code == 200, patched.text
    assert _driver_restarts(stub_supervisor) == ["search-driver"]


@pytest.mark.parametrize("mode", ["auto", "chat", "embedding", "reranking"])
def test_a_change_that_leaves_what_it_starts_as_alone_leaves_its_driver_alone(
    authed_client: TestClient, stub_supervisor: StubSupervisor, mode: str
) -> None:
    authed_client.post("/v1/runtimes", json=_runtime(mode=mode))
    stub_supervisor.calls.clear()
    patched = authed_client.patch(
        "/v1/runtimes/search", json=_runtime(mode=mode, flags={"threads": 4})
    )
    assert patched.status_code == 200, patched.text
    assert _driver_restarts(stub_supervisor) == []


def test_auto_and_the_mode_it_resolves_to_are_one_thing_to_the_driver(
    authed_client: TestClient, stub_supervisor: StubSupervisor
) -> None:
    """The file is not there, so `auto` is chat, and `chat` says the same: no change
    in what the engine is, so nothing for the driver to ask again."""
    authed_client.post("/v1/runtimes", json=_runtime(mode="auto"))
    stub_supervisor.calls.clear()
    authed_client.patch("/v1/runtimes/search", json=_runtime(mode="chat"))
    assert _driver_restarts(stub_supervisor) == []


def test_declaring_and_renaming_restart_no_driver_that_is_not_already_running(
    authed_client: TestClient, stub_supervisor: StubSupervisor
) -> None:
    authed_client.post("/v1/runtimes", json=_runtime(mode="chat"))
    assert _driver_restarts(stub_supervisor) == []
    # A rename makes a new driver, which has learned nothing to unlearn.
    authed_client.patch("/v1/runtimes/search", json=_runtime(name="other", mode="embedding"))
    assert _driver_restarts(stub_supervisor) == []
    assert ("add_and_start", "other-driver") in stub_supervisor.calls


def test_a_runtime_fronted_by_hand_has_no_companion_to_restart(
    authed_client: TestClient, stub_supervisor: StubSupervisor
) -> None:
    authed_client.post("/v1/runtimes", json=_runtime(mode="chat", autoDriver=False))
    stub_supervisor.calls.clear()
    patched = authed_client.patch(
        "/v1/runtimes/search", json=_runtime(mode="embedding", autoDriver=False)
    )
    assert patched.status_code == 200, patched.text
    assert _driver_restarts(stub_supervisor) == []


def test_a_flag_typed_into_extra_args_that_changes_what_it_starts_as_is_a_change_too(
    authed_client: TestClient, stub_supervisor: StubSupervisor, settings: Any
) -> None:
    authed_client.patch("/v1/config", json={"allowUnrestrictedEngineLaunch": True})
    authed_client.post("/v1/runtimes", json=_runtime(mode="chat"))
    stub_supervisor.calls.clear()
    patched = authed_client.patch(
        "/v1/runtimes/search", json=_runtime(mode="chat", extraArgs=["--embeddings"])
    )
    assert patched.status_code == 200, patched.text
    assert _driver_restarts(stub_supervisor) == ["search-driver"]


# --- admitted and reserved at what it will take ----------------------------------------


def _context(client: TestClient) -> Any:
    """What the routes take: anything with a started application on it."""
    return SimpleNamespace(app=client.app, state=SimpleNamespace())


def test_admission_measures_a_search_model_at_its_own_window(
    authed_client: TestClient, tmp_path: Path
) -> None:
    """Left as declared, an embedder would be sized at the model's own 32,768 tokens
    (3.5 GiB of cache for Qwen3-Embedding)."""
    model = g.qwen3_embedding(tmp_path / "Qwen3-Embedding-0.6B-Q8_0.gguf")
    declared = RuntimeSpec(name="e", engine="llama_cpp", modelPath=str(model))
    seen = routes.launch_view(_context(authed_client), declared)
    assert seen.flags == {
        "contextSize": 4096,
        "parallelSlots": 4,
        "batchSize": 2048,
        "ubatchSize": 2048,
    }
    assert declared.flags is None, "the declaration is not edited"


def test_admission_leaves_the_operators_flags_and_a_chat_model_alone(
    authed_client: TestClient, tmp_path: Path
) -> None:
    embedder = g.qwen3_embedding(tmp_path / "e.gguf")
    chat = g.qwen3_chat(tmp_path / "c.gguf")
    own = RuntimeSpec(
        name="e", engine="llama_cpp", modelPath=str(embedder), flags={"contextSize": 16384}
    )
    assert routes.launch_view(_context(authed_client), own).flags is not None
    assert routes.launch_view(_context(authed_client), own).flags["contextSize"] == 16384  # type: ignore[index]

    plain = RuntimeSpec(name="c", engine="llama_cpp", modelPath=str(chat))
    assert routes.launch_view(_context(authed_client), plain) is plain
    forced_chat = RuntimeSpec(name="e", engine="llama_cpp", modelPath=str(embedder), mode="chat")
    assert routes.launch_view(_context(authed_client), forced_chat) is forced_chat

    missing = RuntimeSpec(name="m", engine="llama_cpp", modelPath=str(tmp_path / "nope.gguf"))
    assert routes.launch_view(_context(authed_client), missing) is missing


def test_other_engines_are_not_looked_at(authed_client: TestClient) -> None:
    vllm = RuntimeSpec(name="v", engine="vllm", modelPath="/m/model")
    assert routes.launch_view(_context(authed_client), vllm) is vllm


def test_the_dry_run_reports_the_window_a_search_model_will_have(
    authed_client: TestClient, tmp_path: Path
) -> None:
    model = g.qwen3_embedding(tmp_path / "Qwen3-Embedding-0.6B-Q8_0.gguf")
    answer = authed_client.post(
        "/v1/runtimes/admission",
        json={"name": "e", "engine": "llama_cpp", "modelPath": str(model)},
    )
    assert answer.status_code == 200, answer.text
    assert answer.json()["contextLength"] == 4096

    chat = g.qwen3_chat(tmp_path / "chat.gguf")
    answer = authed_client.post(
        "/v1/runtimes/admission",
        json={"name": "c", "engine": "llama_cpp", "modelPath": str(chat)},
    )
    assert answer.json()["contextLength"] != 4096


def test_a_search_models_reservation_is_what_it_takes_and_not_the_whole_card() -> None:
    """With the context unset on a build whose fit is on, llama.cpp fills the card to
    its margin, and the ledger would promise it all of it. A search model has its
    window set, so it is promised what was measured."""
    from eugene_plexus_agent._generated.models import (
        Admission,
        AdmissionBasis,
        AdmissionDecision,
        AdmissionFit,
    )

    declared = RuntimeSpec(name="e", engine="llama_cpp", modelPath="/m/e.gguf")
    measured = Admission(
        decision=AdmissionDecision.admit,
        fit=AdmissionFit.fits,
        basis=AdmissionBasis.file_size,
        reason="fits",
        requiredBytes=1_200_000_000,
        freeBytes=24 * 1024**3,
    )
    room = reservation_bytes(declared, measured, engine_places=True)
    assert room > measured.requiredBytes, "unset context: the engine takes what it is given"
    view = declared.model_copy(update={"flags": {"contextSize": 4096}})
    assert reservation_bytes(view, measured, engine_places=True) == 1_200_000_000


async def test_a_running_search_model_is_not_offered_for_eviction(
    authed_client: TestClient, tmp_path: Path
) -> None:
    """The gateway evicts only runtimes that opted in with an idle timeout, to make
    room for a start-on-demand model. A search model that was declared with one has
    not opted in: unloaded, it would stop answering."""

    class Supervisor(StubRuntimeSupervisor):
        def compose(self, spec: RuntimeSpec) -> Any:
            view = super().compose(spec)
            update: dict[str, Any] = {"status": RuntimeStatus.ready}
            if spec.name == "embedder":
                update["idleUnloadSeconds"] = None  # what a running embedder reports
            return view.model_copy(update=update)

    app = authed_client.app
    app.state.runtime_supervisor = Supervisor()
    state: AgentState = app.state.agent_state
    for name in ("embedder", "chat"):
        state.add_runtime(
            RuntimeSpec(
                name=name,
                engine="llama_cpp",
                modelPath=f"/models/{name}.gguf",
                idleUnloadSeconds=300,
                port=_free_port(),
            )
        )
    # One small card, held by both.
    from .conftest import fake_devices

    app.state.device_detector = lambda: fake_devices(free=2 * 1024**3, total=24 * 1024**3)

    big = RuntimeSpec(name="big", engine="llama_cpp", modelPath="/models/big.gguf", autoStart=False)
    admission, _places = await routes._measure_launch(_context(authed_client), big)
    held = {b.name: b for b in admission.blockers}
    assert held["embedder"].evictable is False
    assert held["embedder"].idleUnloadSeconds is None
    assert held["chat"].evictable is True and held["chat"].idleUnloadSeconds == 300


# --- the one-click Run composes the same launch, and does not size a search model to the card


def test_a_run_copies_the_profiles_mode_onto_the_runtime() -> None:
    model = {"name": "Qwen3-Embedding-0.6B-Q8_0", "path": "/m/e.gguf"}
    profile = {
        "id": "p1",
        "name": "default",
        "default": True,
        "engine": "llama_cpp",
        "mode": "chat",
    }
    assert runtime_spec(model, profile, start=True).mode is not None
    assert runtime_spec(model, profile, start=True).mode.value == "chat"  # type: ignore[union-attr]
    without = {k: v for k, v in profile.items() if k != "mode"}
    spec = runtime_spec(model, without, start=True)
    assert spec.mode is not None and spec.mode.value == "auto", "a profile older than MR3 is auto"


async def test_a_run_does_not_size_a_search_model_to_the_card(
    authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The largest context that fits would be a curated flag, and a curated flag beats
    the search model's own 4,096: a gigabyte of cache nobody asked for."""
    from eugene_plexus_agent.routes import runtimes as actions

    async def should_not_be_asked(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a search model is not sized to the card")

    monkeypatch.setattr(actions, "_admission_for", should_not_be_asked)
    node = NodeActions(authed_client.app)  # type: ignore[arg-type]
    for capabilities in ({"embedding": True}, {"rerank": True}):
        assert (
            await node.context_size(
                None, {"capabilities": capabilities, "path": "/m/e.gguf"}, "llama_cpp"
            )
            is None
        )  # type: ignore[arg-type]

    async def asked(*_a: Any, **_k: Any) -> Any:
        return SimpleNamespace(maxContextLength=2048, fit="fits")

    monkeypatch.setattr(actions, "_admission_for", asked)
    chat = {"capabilities": {"chat": True}, "path": "/m/c.gguf", "contextLength": 40960}
    assert await node.context_size(None, chat, "llama_cpp") == 2048  # type: ignore[arg-type]


def test_the_flag_schema_says_what_a_search_model_starts_with_when_these_are_unset() -> None:
    fields = {f.key: f.description for f in LlamaCppAdapter().flag_schema().fields}
    for key in ("contextSize", "parallelSlots", "batchSize", "ubatchSize"):
        assert "search model" in fields[key], key
    assert "4,096" in fields["contextSize"] and "2,048" in fields["batchSize"]


def test_capabilities_carry_reranking() -> None:
    assert RuntimeCapabilities(reranking=True).reranking is True
    assert RuntimeCapabilities().reranking is None, "absent for an engine that does not say"
