"""The llama.cpp adapter: argv construction, readiness, flag validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent._generated.models import (
    ConfigValueType,
    EngineKind,
    Origin,
    RuntimeSpec,
)
from eugene_plexus_agent.engines import EngineUnavailableError, LlamaCppAdapter, adapter_for
from eugene_plexus_agent.engines.base import DiscoveredBinary, Loading, NotAnswering, Ready
from eugene_plexus_agent.engines.llama_cpp import default_model_alias


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


# --------------------------------------------------------------------------- #
# argv
# --------------------------------------------------------------------------- #


def test_argv_carries_model_host_port_and_alias(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    argv = adapter.build_argv(_spec(), binary, port=8090)

    assert argv[0] == str(binary.path)
    assert argv[argv.index("--model") + 1] == "/models/Qwen3-30B-A3B-Q4_K_M.gguf"
    assert argv[argv.index("--port") + 1] == "8090"
    # Loopback by default: an engine has no auth of its own, so it must
    # not be exposed directly. Reaching it from elsewhere is the
    # gateway's job, and the gateway has auth.
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    # The alias defaults to the filename, which is what makes "the name
    # you ask the gateway for is the name of the file you downloaded"
    # actually true.
    assert argv[argv.index("--alias") + 1] == "Qwen3-30B-A3B-Q4_K_M"


def test_argv_uses_the_resolved_port_not_the_declared_one(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    """The agent assigns ports, so the caller's resolved value wins."""
    argv = adapter.build_argv(_spec(port=9999), binary, port=8090)
    assert argv[argv.index("--port") + 1] == "8090"
    assert "9999" not in argv


def test_curated_flags_become_cli_arguments(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    spec = _spec(
        flags={
            "contextSize": 8192,
            "gpuLayers": 99,
            "parallelSlots": 4,
            "tensorSplit": "0.6,0.4",
        }
    )
    argv = adapter.build_argv(spec, binary, port=8090)
    assert argv[argv.index("--ctx-size") + 1] == "8192"
    assert argv[argv.index("--n-gpu-layers") + 1] == "99"
    assert argv[argv.index("--parallel") + 1] == "4"
    assert argv[argv.index("--tensor-split") + 1] == "0.6,0.4"


def test_boolean_flags_are_presence_only(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    """`--flash-attn false` is not a thing llama-server understands, so a
    false boolean must be absent rather than passed with a value."""
    on = adapter.build_argv(_spec(flags={"flashAttention": True}), binary, port=8090)
    assert "--flash-attn" in on
    # Nothing follows it, or the next token is another flag —
    # never a value.
    following = on[on.index("--flash-attn") + 1 :]
    assert following == [] or following[0].startswith("--")

    off = adapter.build_argv(_spec(flags={"flashAttention": False}), binary, port=8090)
    assert "--flash-attn" not in off
    assert "false" not in off


def test_unset_flags_are_absent_entirely(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    """A flag the operator didn't set must not be sent at all — not sent
    as the schema's default. The engine's own default is the honest one,
    and for contextSize it's the model's trained context."""
    argv = adapter.build_argv(_spec(), binary, port=8090)
    assert "--ctx-size" not in argv
    assert "--n-gpu-layers" not in argv
    # `parallelSlots` has a schema default of 1, which is UI guidance —
    # it must not leak into argv when the operator left it alone.
    assert "--parallel" not in argv


def test_extra_args_come_last_so_they_can_override(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    """The escape hatch for the long tail of flags a curated surface will
    always miss. Last position is what makes it an override."""
    spec = _spec(flags={"contextSize": 8192}, extraArgs=["--ctx-size", "4096", "--verbose"])
    argv = adapter.build_argv(spec, binary, port=8090)
    assert argv[-3:] == ["--ctx-size", "4096", "--verbose"]
    # Both occurrences present; llama-server takes the later one.
    assert argv.count("--ctx-size") == 2


def test_explicit_alias_overrides_the_filename(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    argv = adapter.build_argv(_spec(modelAlias="qwen"), binary, port=8090)
    assert argv[argv.index("--alias") + 1] == "qwen"


def test_default_alias_strips_gguf_but_keeps_directory_names() -> None:
    assert default_model_alias("/models/Qwen3-30B-Q4_K_M.gguf") == "Qwen3-30B-Q4_K_M"
    # Multi-file formats point at a directory, whose name IS the model name.
    assert default_model_alias("/models/Llama-3.1-8B-Instruct") == "Llama-3.1-8B-Instruct"


# --------------------------------------------------------------------------- #
# working directory + binary resolution
# --------------------------------------------------------------------------- #


def test_working_directory_defaults_to_the_binary_directory(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    """Prebuilt llama.cpp releases ship their shared libraries next to the
    executable and won't start from an unrelated cwd."""
    assert adapter.working_directory(_spec(), binary) == str(binary.path.parent)


def test_explicit_working_directory_wins(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    spec = _spec(workingDirectory="/somewhere/else")
    assert adapter.working_directory(spec, binary) == "/somewhere/else"


def test_explicit_binary_beats_discovery(
    adapter: LlamaCppAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator who built llama.cpp themselves for one model must not
    have that choice silently overridden by whatever is on PATH."""
    custom = tmp_path / "my-llama-server"
    custom.write_text("#!/bin/sh\n", encoding="utf-8")
    on_path = DiscoveredBinary(Path("/usr/bin/llama-server"), Origin.path)
    monkeypatch.setattr(adapter, "discover", lambda: on_path)

    resolved = adapter.resolve_binary(_spec(binary=str(custom)))
    assert resolved.path == custom
    assert resolved.origin == Origin.configured


def test_missing_explicit_binary_is_an_error_not_a_fallback(adapter: LlamaCppAdapter) -> None:
    with pytest.raises(EngineUnavailableError, match="does not exist"):
        adapter.resolve_binary(_spec(binary="/nope/llama-server"))


def test_no_binary_anywhere_names_the_fix(
    adapter: LlamaCppAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "discover", lambda: None)
    with pytest.raises(EngineUnavailableError) as caught:
        adapter.resolve_binary(_spec())
    # The message has to name the fix, since until engine
    # acquisition lands, setting `binary` by hand IS the fix.
    message = " ".join(str(caught.value).split())
    assert "llama-server" in message
    assert "`binary`" in message
    assert "PATH" in message


# --------------------------------------------------------------------------- #
# readiness
#
# The starting/loading/ready split is the concrete payoff of a per-engine
# probe: a generic TCP connect cannot tell "reading 20GB off disk" from
# "wrong argv, never coming up", and those need different responses.
# --------------------------------------------------------------------------- #


async def test_health_503_loading_model_reports_loading(
    adapter: LlamaCppAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(503, json={"status": "loading model"})

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8090")
    assert isinstance(outcome, Loading)
    assert outcome.detail == "loading model"


async def test_health_ok_reports_ready_with_capabilities(
    adapter: LlamaCppAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200,
            json={
                "default_generation_settings": {"n_ctx": 8192},
                "total_slots": 4,
                "modalities": {"vision": True},
            },
        )

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8090")
    assert isinstance(outcome, Ready)
    assert outcome.capabilities is not None
    # Read back off the engine, not inferred from the spec: a requested
    # context larger than the model gets clamped, and the clamped value
    # is the true one.
    assert outcome.capabilities.contextLength == 8192
    assert outcome.capabilities.parallelSlots == 4
    assert outcome.capabilities.multimodal is True


async def test_ready_survives_an_unreadable_props(
    adapter: LlamaCppAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runtime that is serving but won't describe itself is still
    serving. Capabilities are best-effort."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(500, text="nope")

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8090")
    assert isinstance(outcome, Ready)
    assert outcome.capabilities is None


async def test_connection_refused_reports_not_answering(
    adapter: LlamaCppAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8090")
    assert isinstance(outcome, NotAnswering)


async def test_unknown_non_ok_status_is_treated_as_loading(
    adapter: LlamaCppAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Don't assert readiness we can't vouch for."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "something-new-upstream"})

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8090")
    assert isinstance(outcome, Loading)


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """Route the adapter's httpx calls through a MockTransport."""
    real_init = httpx.AsyncClient.__init__

    def init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)


# --------------------------------------------------------------------------- #
# flag schema
# --------------------------------------------------------------------------- #


def test_flag_schema_is_a_standard_config_schema(adapter: LlamaCppAdapter) -> None:
    """Returning a ConfigSchema is what lets the generic config editor
    render engine flags with no engine-specific UI code."""
    schema = adapter.flag_schema()
    assert schema.component == "engine:llama_cpp"
    assert schema.fields
    for field in schema.fields:
        assert field.label, f"{field.key} needs a label for the form"
        assert field.description, f"{field.key} needs help text — that is the point"
        assert field.category in (schema.categories or {})


def test_every_schema_flag_has_a_cli_mapping(adapter: LlamaCppAdapter) -> None:
    """A flag in the schema with no CLI name would render in the UI and
    then KeyError at spawn."""
    from eugene_plexus_agent.engines.llama_cpp import _FLAG_CLI_NAMES

    assert {f.key for f in adapter.flag_schema().fields} == set(_FLAG_CLI_NAMES)


def test_unknown_flags_are_reported_not_dropped(adapter: LlamaCppAdapter) -> None:
    """A typo'd flag that silently vanishes is far worse than one that
    errors: the engine starts, behaves differently from what was asked
    for, and nothing says why."""
    unknown = adapter.validate_flags({"contextSize": 8192, "ctxSize": 4096, "nonsense": 1})
    assert unknown == ["ctxSize", "nonsense"]
    assert adapter.validate_flags({"contextSize": 8192}) == []


def test_context_size_has_no_default(adapter: LlamaCppAdapter) -> None:
    """Leaving it unset means "the model's trained context", which the
    engine resolves. A default here would silently cap models."""
    field = next(f for f in adapter.flag_schema().fields if f.key == "contextSize")
    assert field.default is None
    assert field.valueType == ConfigValueType.integer


# Engine kinds whose contract has landed but whose adapter has not. Empty is
# the correct steady state; an entry here is a debt with a name.
#
# `vllm` arrived in EngineKind with specs 811112b (M4), whose implementation
# was paused for M5's multi-host and trust work. Registering the vLLM adapter
# is the first thing M4's implementation has to do, and deleting the entry
# below is how that gets proved.
CONTRACTED_WITHOUT_ADAPTER = {EngineKind.vllm}


def test_registry_covers_every_engine_kind() -> None:
    """EngineKind is a closed enum precisely because an engine is
    supported when an adapter exists. The enum and the registry are two
    views of the same fact, so they must not drift.

    Drift is tolerated only for kinds named in
    `CONTRACTED_WITHOUT_ADAPTER`, so a gap has to be written down
    deliberately rather than discovered by a red build.
    """
    for kind in EngineKind:
        if kind in CONTRACTED_WITHOUT_ADAPTER:
            continue
        assert adapter_for(kind) is not None, f"no adapter registered for {kind}"


def test_the_contracted_without_adapter_list_is_honest() -> None:
    """Every excused kind must actually be missing an adapter.

    Without this, the allowlist above would silently keep excusing an
    engine after its adapter landed, and the parity check it exists to
    weaken would stay weakened forever.
    """
    for kind in CONTRACTED_WITHOUT_ADAPTER:
        assert adapter_for(kind) is None, (
            f"{kind} has an adapter now — remove it from "
            "CONTRACTED_WITHOUT_ADAPTER so parity is enforced again"
        )
