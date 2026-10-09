"""The llama.cpp adapter: argv construction, readiness, flag validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent import _http
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


def test_cache_type_is_one_choice_for_k_and_v(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    # The flash-attention kernels handle matching pairs only, so the one
    # setting writes both flags; a quantised cache brings flash attention.
    argv = adapter.build_argv(_spec(flags={"cacheType": "q8_0"}), binary, port=8090)
    assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
    assert argv[argv.index("--cache-type-v") + 1] == "q8_0"
    assert argv[argv.index("--flash-attn") + 1] == "on"

    full = adapter.build_argv(_spec(flags={"cacheType": "f16"}), binary, port=8090)
    assert full[full.index("--cache-type-k") + 1] == "f16" and "--flash-attn" not in full

    # Flash attention already asked for: one switch, not two.
    both = adapter.build_argv(
        _spec(flags={"cacheType": "q4_0", "flashAttention": True}), binary, port=8090
    )
    assert both.count("--flash-attn") == 1

    unset = adapter.build_argv(_spec(flags={}), binary, port=8090)
    assert "--cache-type-k" not in unset and "--cache-type-v" not in unset


def test_memory_margin_is_fits_target(adapter: LlamaCppAdapter, binary: DiscoveredBinary) -> None:
    argv = adapter.build_argv(_spec(flags={"memoryMargin": 4096}), binary, port=8090)
    assert argv[argv.index("--fit-target") + 1] == "4096"
    assert "--fit-target" not in adapter.build_argv(_spec(flags={}), binary, port=8090)


def test_new_memory_settings_say_what_unset_means(adapter: LlamaCppAdapter) -> None:
    fields = {f.key: f for f in adapter.flag_schema().fields}
    for key in ("cacheType", "memoryMargin"):
        assert fields[key].unsetMeans, key
    assert fields["cacheType"].enumValues == ["f16", "q8_0", "q4_0"]


def test_flash_attention_takes_a_value(adapter: LlamaCppAdapter, binary: DiscoveredBinary) -> None:
    """`-fa` takes on|off|auto. This test used to require the opposite --
    that nothing but another flag follow `--flash-attn` -- which locked in
    the argv that crashed the first profile a person built (2026-10-01):
    llama-server read the next flag as the value and refused to start.
    Asserted with a flag after it, because that is the shape that broke."""
    on = adapter.build_argv(
        _spec(flags={"flashAttention": True, "cacheType": "q8_0"}), binary, port=8090
    )
    assert on[on.index("--flash-attn") + 1] == "on"
    assert on.count("--flash-attn") == 1

    alone = adapter.build_argv(_spec(flags={"flashAttention": True}), binary, port=8090)
    assert alone[alone.index("--flash-attn") + 1] == "on"

    # False leaves the engine's own default (auto) alone.
    off = adapter.build_argv(_spec(flags={"flashAttention": False}), binary, port=8090)
    assert "--flash-attn" not in off
    assert "false" not in off


def _flash(argv: list[str]) -> str | None:
    return argv[argv.index("--flash-attn") + 1] if "--flash-attn" in argv else None


def test_flash_attention_is_on_off_or_the_engines_own_choice(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    """agent#6: three states, and an unticked box never meant off."""

    def argv(**flags: object) -> list[str]:
        return adapter.build_argv(_spec(flags=flags), binary, port=8090)

    assert _flash(argv(flashAttention="on")) == "on"
    assert _flash(argv(flashAttention="off")) == "off"
    assert _flash(argv()) is None, "unset: llama.cpp decides (auto)"
    # Profiles saved before three states: True sent on, False sent nothing.
    assert _flash(argv(flashAttention=True)) == "on"
    assert _flash(argv(flashAttention=False)) is None, "an old False is not off"


def test_a_quantised_cache_turns_flash_attention_on_and_says_so(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, caplog: pytest.LogCaptureFixture
) -> None:
    for choice in (None, "on", "off", False):
        flags: dict[str, object] = {"cacheType": "q4_0"}
        if choice is not None:
            flags["flashAttention"] = choice
        built = adapter.build_argv(_spec(flags=flags), binary, port=8090)
        assert _flash(built) == "on" and built.count("--flash-attn") == 1, choice
    assert "needs it" in caplog.text, "an overridden off is said, not silently ignored"


def test_the_flash_attention_setting_says_what_unset_means(adapter: LlamaCppAdapter) -> None:
    field = next(f for f in adapter.flag_schema().fields if f.key == "flashAttention")
    assert field.enumValues == ["on", "off"]
    assert field.default is None
    assert "auto" in (field.unsetMeans or "")


def test_boolean_switches_are_presence_only(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    """The switches that really are presence-only stay so."""
    on = adapter.build_argv(_spec(flags={"continuousBatching": True}), binary, port=8090)
    assert "--cont-batching" in on
    following = on[on.index("--cont-batching") + 1 :]
    assert following == [] or following[0].startswith("--")
    off = adapter.build_argv(_spec(flags={"continuousBatching": False}), binary, port=8090)
    assert "--cont-batching" not in off


def test_unset_flags_are_absent_entirely(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    """A flag the operator didn't set must not be sent at all — not sent
    as the schema's default. The engine's own default is the honest one,
    and for contextSize it's the model's trained context."""
    argv = adapter.build_argv(_spec(), binary, port=8090)
    assert "--ctx-size" not in argv
    assert "--n-gpu-layers" not in argv
    # `parallelSlots` unset is llama-server's own automatic slots; nothing
    # of ours may leak into argv when the operator left it alone.
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
    monkeypatch.setattr(adapter, "discover", lambda **_: on_path)

    resolved = adapter.resolve_binary(_spec(binary=str(custom)))
    assert resolved.path == custom
    assert resolved.origin == Origin.configured


def test_missing_explicit_binary_is_an_error_not_a_fallback(adapter: LlamaCppAdapter) -> None:
    with pytest.raises(EngineUnavailableError, match="does not exist"):
        adapter.resolve_binary(_spec(binary="/nope/llama-server"))


def test_no_binary_anywhere_names_the_fix(
    adapter: LlamaCppAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter, "discover", lambda **_: None)
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
    assert outcome.capabilities.vision is True


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
    """Route the adapter's probes through a MockTransport.

    **Injection, not a patched constructor.** This used to monkeypatch
    `httpx.AsyncClient.__init__`, which only worked while every probe
    built its own client inside the call under test. Adapters now share
    one client for the life of the process (that construction cost
    ~104 ms of synchronous CPU on the event loop, every 2 s, per
    runtime), so a patched constructor would miss an already-built
    client -- and, worse, leak this test's transport into every later
    test in the process. The shared slot is the seam; `conftest.py`
    clears it between tests.
    """
    del monkeypatch  # kept in the signature so call sites do not all change
    _http.set_shared_client(
        "engine-probe", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


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


def test_every_schema_flag_reaches_the_command_line_somehow(
    adapter: LlamaCppAdapter,
) -> None:
    """A flag in the schema that nothing translates would render in the UI
    and then KeyError at spawn.

    Three routes now: most keys are a one-to-one CLI name, `noMmap` /
    `mlock` collapse into a single `--load-mode` value whose spelling
    depends on the binary, and `cacheType` expands into the K/V pair of
    flags. A key in no set is the defect this guards; a key in two would
    be built twice.
    """
    from eugene_plexus_agent.engines.llama_cpp import (
        _FLAG_CLI_NAMES,
        _LOAD_MODE_KEYS,
        CACHE_TYPE_KEY,
    )

    direct = set(_FLAG_CLI_NAMES)
    collapsed = set(_LOAD_MODE_KEYS)
    expanded = {CACHE_TYPE_KEY}
    assert not (direct & collapsed), "a key translated twice would be passed twice"
    assert not (direct & expanded) and not (collapsed & expanded)
    assert {f.key for f in adapter.flag_schema().fields} == direct | collapsed | expanded


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
# `vllm` sat here from specs 811112b (M4's contracts, implementation paused
# for M5) until its adapter landed, at which point the honesty test below
# forced the entry out. That is the mechanism working as intended: the
# allowlist cannot outlive the gap it excuses.
CONTRACTED_WITHOUT_ADAPTER: set[EngineKind] = set()


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


# --------------------------------------------------------------------------- #
# load mode: --load-mode since ~b10900, --no-mmap/--mlock before it
#
# Found live 2026-09-17 on b10948: ticking `noMmap` in this project's own
# schema spawned `llama-server --no-mmap`, which that build rejects, and the
# runtime crash-looped. The spelling is read off the binary rather than gated
# on a build number.
# --------------------------------------------------------------------------- #


def _binary_whose_help_says(tmp_path: Any, help_text: str) -> DiscoveredBinary:
    """A stand-in binary that really is executed for `--help`."""
    script = tmp_path / "llama-server-help.py"
    script.write_text(
        f"import sys; sys.stdout.write({help_text!r})",
        encoding="utf-8",
    )
    exe = tmp_path / ("fake-llama-server" + (".bat" if os.name == "nt" else ""))
    if os.name == "nt":
        exe.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    return DiscoveredBinary(path=exe, origin=Origin.path, version="10948")


_NEW_HELP = "-lm,   --load-mode MODE   model loading mode (default: auto)\n"
_OLD_HELP = "       --no-mmap          do not memory-map model\n       --mlock   force RAM\n"


def test_no_mmap_becomes_load_mode_on_a_build_that_has_it(
    adapter: LlamaCppAdapter, tmp_path: Any
) -> None:
    binary = _binary_whose_help_says(tmp_path, _NEW_HELP)

    argv = adapter.build_argv(_spec(flags={"noMmap": True}), binary, port=8090)

    assert "--no-mmap" not in argv, "the old spelling is what the engine rejects"
    assert argv[argv.index("--load-mode") + 1] == "none"


def test_no_mmap_keeps_the_old_spelling_on_a_build_that_only_has_that(
    adapter: LlamaCppAdapter, tmp_path: Any
) -> None:
    binary = _binary_whose_help_says(tmp_path, _OLD_HELP)

    argv = adapter.build_argv(_spec(flags={"noMmap": True}), binary, port=8090)

    assert "--no-mmap" in argv
    assert "--load-mode" not in argv


def test_mlock_alone_still_means_mmap_and_pin(adapter: LlamaCppAdapter, tmp_path: Any) -> None:
    # `--mlock` used to leave mmap alone because mmap was the default, so
    # the faithful translation is mmap+mlock, NOT bare mlock.
    binary = _binary_whose_help_says(tmp_path, _NEW_HELP)

    argv = adapter.build_argv(_spec(flags={"mlock": True}), binary, port=8090)

    assert argv[argv.index("--load-mode") + 1] == "mmap+mlock"


def test_both_flags_collapse_into_one_value(adapter: LlamaCppAdapter, tmp_path: Any) -> None:
    binary = _binary_whose_help_says(tmp_path, _NEW_HELP)

    argv = adapter.build_argv(_spec(flags={"noMmap": True, "mlock": True}), binary, port=8090)

    assert argv.count("--load-mode") == 1, "two --load-mode values would be a conflict"
    assert argv[argv.index("--load-mode") + 1] == "mlock"


def test_neither_flag_names_no_mode_at_all(adapter: LlamaCppAdapter, tmp_path: Any) -> None:
    # The engine's own default is the right answer; naming it would be us
    # deciding something the operator did not.
    binary = _binary_whose_help_says(tmp_path, _NEW_HELP)

    argv = adapter.build_argv(_spec(flags={"contextSize": 4096}), binary, port=8090)

    assert "--load-mode" not in argv
    assert "--no-mmap" not in argv


def test_an_unaskable_binary_still_builds_an_argv(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary
) -> None:
    # `binary` is an empty stub that cannot answer --help. Refusing to
    # launch over a failed cosmetic probe would turn it into an outage.
    argv = adapter.build_argv(_spec(flags={"noMmap": True}), binary, port=8090)

    assert argv[argv.index("--load-mode") + 1] == "none"


def test_a_rejected_argument_is_explained_rather_than_left_as_an_exit_code(
    adapter: LlamaCppAdapter,
) -> None:
    explained = adapter.explain_exit(1, "error: invalid argument: --no-mmap\n")

    assert explained is not None
    assert "--no-mmap" in explained
    assert adapter.explain_exit(1, "ggml_cuda_init: failed\n") is None


async def test_audio_projector_does_not_advertise_vision(adapter, monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"modalities": {"vision": False, "audio": True}})

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8090")
    assert outcome.capabilities.multimodal is True
    assert outcome.capabilities.vision is False


def test_projector_uses_curated_flags_with_spaces(adapter, binary):
    argv = adapter.build_argv(
        _spec(flags={"projectorPath": "/models/my vision/mmproj.gguf", "projectorOnCpu": True}),
        binary,
        port=8090,
    )
    assert argv[argv.index("--mmproj") + 1] == "/models/my vision/mmproj.gguf"
    assert "--no-mmproj-offload" in argv


def test_every_adapter_declares_what_it_accepts_and_old_formats_agree() -> None:
    """`modelFormats` stays for consoles older than `accepts` (LS1): it is
    the formats needing no preparation, so an older console never offers
    Strata for every GGUF."""
    from eugene_plexus_agent.engines import ADAPTERS

    for kind, adapter in ADAPTERS.items():
        assert adapter.accepts, f"{kind.value} declares nothing it accepts"
        plain = {r.format for r in adapter.accepts if r.preparation is None}
        assert set(adapter.model_formats) == plain, kind.value


def test_the_engine_list_carries_accepts(tmp_path, monkeypatch) -> None:
    from eugene_plexus_agent.engines import llama_architectures
    from eugene_plexus_agent.runtimes import describe_engines

    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_ENGINE_ROOT", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    by_kind = {e.engine.value: e for e in describe_engines()}
    llama = by_kind["llama_cpp"].accepts or []
    assert llama[0].architectures == list(llama_architectures.shipped().names)
    strata = by_kind["strata"].accepts or []
    assert [r.architectures for r in strata] == [["qwen4exp"]]
    assert strata[0].preparation is not None
    vllm = by_kind["vllm"].accepts or []
    assert vllm[0].mlxQuantization is not None and vllm[0].authority is not None
