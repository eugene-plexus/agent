"""Row 9 (PC7): a runtime's prompt cache is a setting, and what runs says what it has.

llama.cpp keeps recent conversations in system memory (`--cache-ram`, 8 GiB unless
told otherwise) and vLLM can offload its KV cache there (`--kv-offloading-size`).
Before this row the agent set neither, so the size could not be changed, was never
shown, and was never counted against the memory the next model needs.

The view is read from the command line the engine was launched with: the value in
effect, never the profile's wish (settings never lie).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import eugene_plexus_agent.admission as admission_module
from eugene_plexus_agent import runtimes as runtimes_module
from eugene_plexus_agent._generated.models import (
    EngineKind,
    Origin,
    PromptCacheSource,
    RuntimeSpec,
    RuntimeStatus,
)
from eugene_plexus_agent.engines import LlamaCppAdapter, MlxAdapter, adapter_for
from eugene_plexus_agent.engines.base import DiscoveredBinary, Ready
from eugene_plexus_agent.routes import runtimes as routes
from eugene_plexus_agent.runtimes import RuntimeSupervisor

from .conftest import StubRuntimeSupervisor, fake_devices

GIB = 1024**3
MIB = 1024**2


@pytest.fixture
def adapter() -> LlamaCppAdapter:
    return LlamaCppAdapter()


@pytest.fixture
def binary(tmp_path: Path) -> DiscoveredBinary:
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    return DiscoveredBinary(path=exe, origin=Origin.path, version="4589")


def _spec(**overrides: Any) -> RuntimeSpec:
    base: dict[str, Any] = {
        "name": "qwen3-30b",
        "engine": EngineKind.llama_cpp,
        "modelPath": "/models/Qwen3-30B-A3B-Q4_K_M.gguf",
    }
    base.update(overrides)
    return RuntimeSpec.model_validate(base)


def _started(adapter: LlamaCppAdapter, binary: DiscoveredBinary, **overrides: Any):
    """The spec, and the command line the adapter really builds for it."""
    spec = _spec(**overrides)
    return spec, adapter.build_argv(spec, binary, port=8090)


# --- what a running llama.cpp says ----------------------------------------------


def test_a_default_start_says_the_default_it_ran_with(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    spec, argv = _started(adapter, binary)
    view = adapter.prompt_cache_view(spec, argv)
    assert view is not None
    assert view.bytes == 8 * GIB
    assert view.source is PromptCacheSource.engine_default
    assert view.note is None


def test_a_profile_size_is_what_the_view_reports(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    spec, argv = _started(adapter, binary, flags={"cacheRam": 24})
    view = adapter.prompt_cache_view(spec, argv)
    assert view is not None
    assert (view.bytes, view.source) == (24 * GIB, PromptCacheSource.profile)

    spec, argv = _started(adapter, binary, flags={"cacheRam": 0})
    view = adapter.prompt_cache_view(spec, argv)
    assert view is not None
    assert (view.bytes, view.source) == (0, PromptCacheSource.off)


def test_a_size_typed_into_extra_args_wins_and_is_said(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    """The command line takes the last `--cache-ram`: the profile's setting does not apply."""
    spec, argv = _started(
        adapter, binary, flags={"cacheRam": 24}, extraArgs=["--cache-ram", "2048"]
    )
    view = adapter.prompt_cache_view(spec, argv)
    assert view is not None
    assert (view.bytes, view.source) == (2 * GIB, PromptCacheSource.extra_args)
    assert view.note is not None and "extraArgs" in view.note and "24" in view.note

    spec, argv = _started(adapter, binary, extraArgs=["-cram=512"])
    view = adapter.prompt_cache_view(spec, argv)
    assert view is not None and view.bytes == 512 * MIB

    spec, argv = _started(adapter, binary, extraArgs=["--cache-ram", "-1"])
    view = adapter.prompt_cache_view(spec, argv)
    assert view is not None
    assert view.bytes is None and view.source is PromptCacheSource.extra_args
    assert view.note is not None and "no limit" in view.note.lower()


def test_a_build_without_the_flag_says_so_instead_of_a_size(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_agent.engines import llama_cpp

    monkeypatch.setattr(llama_cpp, "_supported_long_flags", lambda path: frozenset({"--ctx-size"}))
    spec, argv = _started(adapter, binary, flags={"cacheRam": 24})
    view = adapter.prompt_cache_view(spec, argv)
    assert view is not None
    assert view.bytes is None and view.source is PromptCacheSource.unsupported
    assert view.note is not None and "--cache-ram" in view.note

    # Only `--cache-reuse` missing: the size is real, and the note names what is not passed.
    monkeypatch.setattr(llama_cpp, "_supported_long_flags", lambda path: frozenset({"--cache-ram"}))
    spec, argv = _started(adapter, binary, flags={"cacheReuse": 128})
    view = adapter.prompt_cache_view(spec, argv)
    assert view is not None and view.bytes == 8 * GIB
    assert view.note is not None and "--cache-reuse" in view.note


def test_a_search_model_has_no_prompt_cache(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    spec, argv = _started(adapter, binary, mode="embedding")
    assert adapter.prompt_cache_view(spec, argv) is None
    assert adapter.prompt_cache_view(_spec(mode="reranking"), []) is None


def test_before_there_is_a_command_line_the_view_is_what_the_launch_will_pass(
    adapter: LlamaCppAdapter,
) -> None:
    """A runtime still copying its file has no process, and already holds its cache."""
    view = adapter.prompt_cache_view(_spec(flags={"cacheRam": 12}), [])
    assert view is not None
    assert (view.bytes, view.source) == (12 * GIB, PromptCacheSource.profile)
    typed = adapter.prompt_cache_view(_spec(extraArgs=["--cache-ram", "1024"]), [])
    assert typed is not None
    assert (typed.bytes, typed.source) == (GIB, PromptCacheSource.extra_args)


def test_an_engine_with_no_prompt_cache_says_nothing() -> None:
    spec = RuntimeSpec.model_validate({"name": "m", "engine": "mlx", "modelPath": "/m/model"})
    assert MlxAdapter().prompt_cache_view(spec, ["mlx_lm.server"]) is None


# --- what the runtime list says ---------------------------------------------------


def _running(name: str, argv: list[str] | None) -> tuple[RuntimeSupervisor, RuntimeSpec]:
    supervisor = RuntimeSupervisor()
    spec = RuntimeSpec(name=name, engine=EngineKind.llama_cpp, modelPath="/m/a.gguf")
    supervisor._processes[name] = SimpleNamespace(  # type: ignore[assignment]
        state=runtimes_module.ProcessState.starting,
        last_error=None,
        last_restart=None,
        pid=1,
        last_argv=argv,
    )
    supervisor._readiness[name] = Ready(capabilities=None)
    return supervisor, spec


def test_a_running_runtime_carries_the_cache_it_was_started_with() -> None:
    supervisor, spec = _running("r", ["llama-server", "--cache-ram", "24576"])
    view = supervisor.compose(spec)
    assert view.status is RuntimeStatus.ready
    assert view.promptCache is not None
    # The value in effect is the command line's. The profile says nothing, so it reads as the
    # default it shows, and the note says the two differ.
    assert view.promptCache.bytes == 24 * GIB
    assert view.promptCache.source is PromptCacheSource.engine_default
    assert view.promptCache.note is not None and "next start" in view.promptCache.note


def test_a_stopped_runtime_carries_none() -> None:
    spec = RuntimeSpec(name="r", engine=EngineKind.llama_cpp, modelPath="/m/a.gguf")
    assert RuntimeSupervisor().compose(spec).promptCache is None


def test_a_running_search_model_carries_none() -> None:
    supervisor, spec = _running("r", ["llama-server", "--embeddings", "--cache-ram", "8192"])
    assert supervisor.compose(spec).promptCache is None


# --- what the next launch is told ---------------------------------------------------

_SIZES: dict[str, int] = {}


@pytest.fixture
def _fake_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission_module, "model_size_bytes", lambda p: _SIZES.get(p))
    monkeypatch.setattr(admission_module, "path_exists", lambda p: True)
    monkeypatch.setattr(routes, "engine_places", lambda spec, get_config: False)


def _model(tmp_path: Path, size: int, tag: str) -> str:
    path = str(tmp_path / f"model-{tag}.gguf")
    _SIZES[path] = size
    return path


def _body(name: str, path: str, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "name": name,
        "engine": "llama_cpp",
        "modelPath": path,
        # Drafting's allowance would move the numbers these tests compare.
        "flags": {"reuseRepeatedText": False},
    }
    body.update(overrides)
    return body


@pytest.mark.usefixtures("_fake_sizes")
def test_a_started_runtimes_cache_is_counted_by_the_next_launch(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the routes, on a host with no card: the first runtime's 16 GiB cache is
    promised to it from the moment it starts, and the second model no longer fits beside
    it. With the first one's cache off, the same second model is admitted."""
    original = StubRuntimeSupervisor.compose

    def compose(self: StubRuntimeSupervisor, spec: RuntimeSpec):  # type: ignore[no-untyped-def]
        """The stub flips a started runtime to `starting` after the real `compose` has
        decided it holds nothing; the view of a runtime on its way up is the adapter's."""
        runtime = original(self, spec)
        if runtime.status is RuntimeStatus.starting:
            adapter = adapter_for(spec.engine)
            assert adapter is not None
            runtime = runtime.model_copy(
                update={"promptCache": adapter.prompt_cache_view(spec, [])}
            )
        return runtime

    monkeypatch.setattr(StubRuntimeSupervisor, "compose", compose)
    authed_client.app.state.device_detector = lambda: fake_devices(count=0)  # type: ignore[attr-defined]
    big = _model(tmp_path, 25 * GIB, "big")

    first = _body(
        "first",
        _model(tmp_path, 2 * GIB, "small"),
        flags={"reuseRepeatedText": False, "cacheRam": 0},
    )
    assert authed_client.post("/v1/runtimes", json=first).status_code == 201
    quiet = authed_client.post("/v1/runtimes/admission", json=_body("second", big))
    assert quiet.status_code == 200, quiet.text
    assert quiet.json()["decision"] == "admit", quiet.json()["reason"]

    authed_client.delete("/v1/runtimes/first")
    first["flags"] = {"reuseRepeatedText": False, "cacheRam": 16}
    assert authed_client.post("/v1/runtimes", json=first).status_code == 201
    loud = authed_client.post("/v1/runtimes/admission", json=_body("second", big))
    assert loud.status_code == 200, loud.text
    answer = loud.json()
    assert answer["decision"] == "refuse", answer["reason"]
    assert "first" in answer["reason"] and "16.0 GiB" in answer["reason"]
    assert answer["promptCacheBytes"] == 8 * GIB


def test_the_dry_run_says_what_a_chat_model_keeps_and_a_search_model_does_not(
    authed_client: TestClient, tmp_path: Path
) -> None:
    from . import gguf_files as g

    chat = g.qwen3_chat(tmp_path / "chat.gguf")
    asked = authed_client.post(
        "/v1/runtimes/admission",
        json={"name": "c", "engine": "llama_cpp", "modelPath": str(chat)},
    )
    assert asked.status_code == 200, asked.text
    assert asked.json()["promptCacheBytes"] == 8 * GIB

    embedder = g.qwen3_embedding(tmp_path / "Qwen3-Embedding-0.6B-Q8_0.gguf")
    asked = authed_client.post(
        "/v1/runtimes/admission",
        json={"name": "e", "engine": "llama_cpp", "modelPath": str(embedder)},
    )
    assert asked.status_code == 200, asked.text
    assert asked.json().get("promptCacheBytes") is None
