"""The MLX adapter, ported to main (roadmap B1).

Written the way the vLLM suite is written, and for the same reason
doubled: this engine has never met a real host, so every claim is pinned
to upstream source (mlx-lm v0.31.3, re-read at the tag 2026-09-22) and
the fixtures are labeled simulated. What only a Mac can prove is listed
in docs/design/mlx-engine.md, not asserted here.

The three behaviors unique to this adapter, each with its own section:

  * readiness costs a token — but only ONCE per process. `established`
    downgrades the probe to a plain /health read, and the first Ready is
    always paid for with a real generation.
  * upstream main's 503 `unavailable` /health reads as Loading, so the
    next release pin gets the cheap probe for free — while any other
    non-200 stays NotAnswering (a 503 from something that is not
    mlx_lm.server must not be promoted to a load that never ends).
  * the model-name blocker: `upstream_model_id()` hands the companion
    the `default_model` sentinel and the alias never reaches the argv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent import _http
from eugene_plexus_agent._generated.models import (
    Arch,
    EngineKind,
    HostAccelerator,
    ModelFormat,
    Origin,
    Os,
    Policy,
    RuntimeSpec,
)
from eugene_plexus_agent.companions import render_config, upstream_model_id
from eugene_plexus_agent.engines import ADAPTERS, adapter_for
from eugene_plexus_agent.engines.base import (
    DiscoveredBinary,
    Loading,
    NotAnswering,
    Ready,
    interpret_readiness,
)
from eugene_plexus_agent.engines.mlx import (
    DEFAULT_MODEL_SENTINEL,
    UPSTREAM_VERSION_PINNED,
    MlxAdapter,
)


@pytest.fixture
def adapter() -> MlxAdapter:
    return MlxAdapter()


@pytest.fixture
def binary(tmp_path: Path) -> DiscoveredBinary:
    exe = tmp_path / "eugene-mlx" / "bin" / "mlx_lm.server"
    exe.parent.mkdir(parents=True)
    exe.write_text(f"#!{tmp_path}/eugene-mlx/bin/python3.12\n", encoding="utf-8")
    return DiscoveredBinary(path=exe, origin=Origin.configured, version="0.31.3")


def _spec(**overrides: Any) -> RuntimeSpec:
    base: dict[str, Any] = {
        "name": "qwen3-tiny",
        "engine": EngineKind.mlx,
        "modelPath": "/Users/troy/models/mlx-community/Qwen3-0.6B-4bit",
    }
    base.update(overrides)
    return RuntimeSpec.model_validate(base)


def _patch_client(handler: Any) -> None:
    """Route the adapter's probes through a MockTransport, via the
    shared `engine-probe` slot — the seam every adapter probe uses since
    R1.1. `conftest.py` clears it between tests."""
    _http.set_shared_client(
        "engine-probe", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_mlx_is_registered_and_experimental() -> None:
    adapter = adapter_for(EngineKind.mlx)
    assert isinstance(adapter, MlxAdapter)
    assert ADAPTERS[EngineKind.mlx] is adapter
    # Experimental until a physical Apple silicon run is recorded — the
    # UI badges off this instead of hardcoding a list.
    assert adapter.experimental is True
    assert MlxAdapter.model_formats == (ModelFormat.safetensors,)
    assert MlxAdapter.install_policy is Policy.manual
    assert MlxAdapter.configured_binary_key == "mlxBinary"
    # The other two stay non-experimental: both are live-verified.
    assert adapter_for(EngineKind.llama_cpp).experimental is False
    assert adapter_for(EngineKind.vllm).experimental is False


# --------------------------------------------------------------------------- #
# argv
# --------------------------------------------------------------------------- #


def test_argv_is_model_host_port(adapter: MlxAdapter, binary: DiscoveredBinary) -> None:
    argv = adapter.build_argv(_spec(), binary, 8101)
    assert argv[0] == str(binary.path)
    assert argv[1:] == [
        "--model",
        "/Users/troy/models/mlx-community/Qwen3-0.6B-4bit",
        "--host",
        "127.0.0.1",
        "--port",
        "8101",
    ]


def test_the_alias_never_reaches_the_argv(adapter: MlxAdapter, binary: DiscoveredBinary) -> None:
    """mlx_lm.server has no --served-model-name; the alias travels via
    the companion's upstreamModelId split instead."""
    argv = adapter.build_argv(_spec(modelAlias="my-public-alias"), binary, 8101)
    assert "my-public-alias" not in argv


def test_curated_flags_map_to_the_v0_31_3_cli_names(
    adapter: MlxAdapter, binary: DiscoveredBinary
) -> None:
    spec = _spec(
        flags={
            "maxTokens": 1024,
            "promptCacheSize": 4,
            "promptCacheBytes": "8GB",
            "decodeConcurrency": 16,
            "promptConcurrency": 4,
            "prefillStepSize": 4096,
            "draftModel": "/models/draft",
            "numDraftTokens": 5,
            "adapterPath": "/models/lora",
            "chatTemplate": "{{ messages }}",
        }
    )
    argv = adapter.build_argv(spec, binary, 8101)
    joined = " ".join(argv)
    assert "--max-tokens 1024" in joined
    assert "--prompt-cache-size 4" in joined
    assert "--prompt-cache-bytes 8GB" in joined
    assert "--decode-concurrency 16" in joined
    assert "--prompt-concurrency 4" in joined
    assert "--prefill-step-size 4096" in joined
    assert "--draft-model /models/draft" in joined
    assert "--num-draft-tokens 5" in joined
    assert "--adapter-path /models/lora" in joined
    assert "--chat-template {{ messages }}" in joined


def test_boolean_flags_are_presence_only(adapter: MlxAdapter, binary: DiscoveredBinary) -> None:
    on = adapter.build_argv(_spec(flags={"trustRemoteCode": True}), binary, 8101)
    off = adapter.build_argv(_spec(flags={"trustRemoteCode": False}), binary, 8101)
    # `store_true` upstream: presence only, no value token ever.
    assert "--trust-remote-code" in on
    assert "True" not in on
    assert "--trust-remote-code" not in off


def test_extra_args_come_last_so_they_can_override(
    adapter: MlxAdapter, binary: DiscoveredBinary
) -> None:
    spec = _spec(flags={"maxTokens": 256}, extraArgs=["--max-tokens", "99"])
    argv = adapter.build_argv(spec, binary, 8101)
    assert argv[-2:] == ["--max-tokens", "99"]


def test_working_directory_is_inherited_not_the_venv_bin(
    adapter: MlxAdapter, binary: DiscoveredBinary
) -> None:
    assert adapter.working_directory(_spec(), binary) is None
    assert adapter.working_directory(_spec(workingDirectory="/tmp/run"), binary) == "/tmp/run"


def test_unknown_flags_are_rejected_not_dropped(adapter: MlxAdapter) -> None:
    assert adapter.validate_flags({"gpuLayers": 99}) == ["gpuLayers"]


# --------------------------------------------------------------------------- #
# the model-name split
# --------------------------------------------------------------------------- #


def test_upstream_model_id_is_the_sentinel(adapter: MlxAdapter) -> None:
    assert adapter.upstream_model_id(_spec()) == DEFAULT_MODEL_SENTINEL


def test_other_engines_have_no_upstream_id() -> None:
    """They are launched WITH the alias, so their companions translate
    nothing — and the key must come out None, not absent, so a runtime
    that changes engine has it cleared in the companion's config."""
    llama_spec = _spec(engine=EngineKind.llama_cpp, modelPath="/models/q.gguf")
    assert upstream_model_id(llama_spec) is None


def test_companion_config_carries_the_sentinel_for_mlx() -> None:
    doc = render_config(
        runtime_name="qwen3-tiny",
        alias="qwen3-0.6b-4bit",
        upstream=upstream_model_id(_spec()),
    )
    assert doc["modelId"] == "qwen3-0.6b-4bit"
    assert doc["upstreamModelId"] == DEFAULT_MODEL_SENTINEL


def test_companion_config_nulls_the_key_for_other_engines() -> None:
    doc = render_config(runtime_name="qwen", alias="qwen3-8b", upstream=None)
    assert "upstreamModelId" in doc
    assert doc["upstreamModelId"] is None


# --------------------------------------------------------------------------- #
# readiness: the token is paid once per process
# --------------------------------------------------------------------------- #


def _health_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok"})


async def test_first_ready_requires_a_real_token(adapter: MlxAdapter) -> None:
    """A listening HTTP port alone must not enable Send: /health up but
    generation timing out is Loading, never Ready."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/health":
            return _health_ok(request)
        raise httpx.ReadTimeout("model still loading", request=request)

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8101")
    assert isinstance(outcome, Loading)
    assert "/v1/chat/completions" in calls


async def test_a_token_back_is_ready(adapter: MlxAdapter) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_ok(request)
        body = json.loads(request.content)
        # The probe asks for one token of the sentinel, at temperature 0.
        assert body["model"] == DEFAULT_MODEL_SENTINEL
        assert body["max_tokens"] == 1
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "o"}}],
                "model": DEFAULT_MODEL_SENTINEL,
            },
        )

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8101")
    assert isinstance(outcome, Ready)


async def test_established_downgrades_to_a_health_read(adapter: MlxAdapter) -> None:
    """Once this process proved residency, the poll stops generating —
    the fix the branch document asked for, as the base change it asked
    for. A wedged generation path after that shows ready until requests
    fail; the trade is recorded in the module docstring."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/health":
            return _health_ok(request)
        raise AssertionError("an established probe must not generate")

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8101", established=True)
    assert isinstance(outcome, Ready)
    assert calls == ["/health"]


async def test_not_established_never_trusts_health_alone(adapter: MlxAdapter) -> None:
    """The complement of the downgrade: established=False (a fresh or
    respawned process) always pays the token, even though /health is
    answering ok — a respawned mlx binds its port before loading."""
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/health":
            return _health_ok(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "o"}}]})

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8101", established=False)
    assert isinstance(outcome, Ready)
    assert "/v1/chat/completions" in paths


# --------------------------------------------------------------------------- #
# readiness: the health endpoint's three shapes
# --------------------------------------------------------------------------- #


async def test_connection_refused_is_not_answering_and_not_reached(
    adapter: MlxAdapter,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8101")
    assert isinstance(outcome, NotAnswering)
    assert outcome.reached is False


async def test_upstream_mains_unavailable_health_reads_as_loading(
    adapter: MlxAdapter,
) -> None:
    """Unreleased upstream (c69d128) answers 503 unavailable while
    loading. Reading it now means the next release pin gets the cheap
    probe with no adapter change — and it must short-circuit BEFORE the
    token probe, or the load would cost a queued generation anyway."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health", "must not fall through to generation"
        return httpx.Response(503, json={"status": "unavailable"})

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8101")
    assert isinstance(outcome, Loading)


async def test_a_strange_503_is_not_promoted_to_loading(adapter: MlxAdapter) -> None:
    """A 503 from something that is not mlx_lm.server — a proxy, a
    stacked stale process — must stay NotAnswering(reached=True), not
    become a load that never ends."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8101")
    assert isinstance(outcome, NotAnswering)
    assert outcome.reached is True


async def test_a_served_error_is_a_launch_mistake_not_a_load(
    adapter: MlxAdapter,
) -> None:
    """A 4xx from the completion probe is a served answer — the likeliest
    cause is `--model` never passed, and saying so beats an endless
    `loading`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _health_ok(request)
        return httpx.Response(400, json={"error": "model default_model not found"})

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8101")
    assert isinstance(outcome, NotAnswering)
    assert outcome.reached is True
    assert "400" in (outcome.detail or "")


# --------------------------------------------------------------------------- #
# readiness: the budget bounds a narrated load
# --------------------------------------------------------------------------- #


def test_a_long_load_is_flagged_past_budget(adapter: MlxAdapter) -> None:
    """A wedged mlx_lm.server keeps answering /health and never
    generates — indistinguishable from a slow load except by the clock,
    which is why the adapter declares a budget and interpret_readiness
    now applies it to narrated loads too."""
    outcome = interpret_readiness(
        adapter,
        Loading(detail="no token within 8s"),
        process_alive=True,
        elapsed_seconds=adapter.startup_budget_seconds + 60,
    )
    assert isinstance(outcome, Loading)
    assert outcome.past_budget is True
    assert "600" in (outcome.detail or "")


def test_a_load_inside_the_budget_is_not_flagged(adapter: MlxAdapter) -> None:
    outcome = interpret_readiness(
        adapter,
        Loading(detail="no token within 8s"),
        process_alive=True,
        elapsed_seconds=30.0,
    )
    assert isinstance(outcome, Loading)
    assert outcome.past_budget is False


def test_llama_cpp_narrated_loads_stay_unflagged() -> None:
    """llama.cpp declares no budget, so the widened rule changes nothing
    for it — the flag exists only where an adapter opted in."""
    llama = adapter_for(EngineKind.llama_cpp)
    assert llama.startup_budget_seconds is None
    outcome = interpret_readiness(
        llama,
        Loading(detail="loading model"),
        process_alive=True,
        elapsed_seconds=100_000.0,
    )
    assert isinstance(outcome, Loading)
    assert outcome.past_budget is False


# --------------------------------------------------------------------------- #
# discovery and installation
# --------------------------------------------------------------------------- #


def test_manual_install_pins_the_verified_release() -> None:
    """The recipe installs the exact release the claims were read
    against, into an environment of its own — never Eugene's venv."""
    adapter = MlxAdapter()
    install = adapter.manual_install(HostAccelerator(os=Os.macos, arch=Arch.arm64))
    assert install.command is not None
    assert f"mlx-lm=={UPSTREAM_VERSION_PINNED}" in install.command
    assert "uv venv" in install.command
    assert "eugene-mlx" in install.command
    # The Rosetta trap: the notes tell the tester how to verify the
    # environment is native ARM before trusting it.
    assert "arm64" in (install.notes or "")
    assert "mlx_lm.server" in (install.notes or "")


def test_manual_install_refuses_a_command_that_would_not_work() -> None:
    """`pip install mlx-lm` succeeds on any platform and installs no
    engine outside Apple silicon — so everywhere else gets the reason
    and no command."""
    adapter = MlxAdapter()
    for host in (
        HostAccelerator(os=Os.windows, arch=Arch.x64),
        HostAccelerator(os=Os.linux, arch=Arch.x64),
        HostAccelerator(os=Os.macos, arch=Arch.x64),  # Intel Mac
        HostAccelerator(os=Os.linux, arch=Arch.arm64),
    ):
        install = adapter.manual_install(host)
        assert install.command is None, host
        assert "Apple silicon" in (install.notes or ""), host
        assert install.docsUrl


def test_exit_explanations_name_the_darwin_marker_trap(adapter: MlxAdapter) -> None:
    explained = adapter.explain_exit(1, "ImportError: No module named 'mlx'")
    assert explained is not None
    assert "Apple" in explained

    unexplained = adapter.explain_exit(1, "Segmentation fault")
    assert unexplained is None
