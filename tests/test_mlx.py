"""The MLX adapter: argv, the refusal, readiness, exit explanations.

**Read this header before trusting a green run here.** Everything below
is a fixture — a fake console script, a fake HTTP transport. For
llama.cpp and vLLM that was a reasonable half of the story, with a live
acceptance run supplying the other half. For MLX there is no other half
yet: `mlx` is Apple-silicon-only and this project's box is Windows with
an RTX 5090, so **no line of this adapter has ever driven a real
engine.** That is the entire reason it lives on a branch.

What these tests genuinely verify is the adapter's *logic*: that argv is
built from the flags upstream actually accepts, that the refusal on a
non-Apple host omits a command rather than inventing one, and that the
three readiness outcomes come out of the three observable situations.

What they cannot verify is every claim about mlx_lm.server itself. Those
claims are cited to v0.31.3 source in the adapter's docstrings, and
listed with a way to check each one in
`specs/docs/design/mlx-engine-unverified.md`. The honest expectation,
from M4: the fixtures will be vindicated on the points read off source
and surprised on the points about the *host*, because that is exactly
how vLLM went.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent._generated.models import (
    Arch,
    ConfigValueType,
    EngineKind,
    HostAccelerator,
    Origin,
    Os,
    Policy,
    RuntimeSpec,
)
from eugene_plexus_agent.engines import DiscoveredBinary, adapter_for
from eugene_plexus_agent.engines.base import Loading, NotAnswering, Ready
from eugene_plexus_agent.engines.mlx import (
    DEFAULT_MODEL_SENTINEL,
    MlxAdapter,
)


@pytest.fixture
def adapter() -> MlxAdapter:
    return MlxAdapter()


@pytest.fixture
def binary(tmp_path: Path) -> DiscoveredBinary:
    exe = tmp_path / "venv" / "bin" / "mlx_lm.server"
    exe.parent.mkdir(parents=True)
    exe.write_text(f"#!{tmp_path}/venv/bin/python3.12\n", encoding="utf-8")
    return DiscoveredBinary(path=exe, origin=Origin.configured, version="0.31.3")


def _spec(**overrides: Any) -> RuntimeSpec:
    base: dict[str, Any] = {
        "name": "qwen3-8b-mlx",
        "engine": EngineKind.mlx,
        "modelPath": "/Users/troy/models/Qwen3-8B-MLX-4bit",
    }
    base.update(overrides)
    return RuntimeSpec.model_validate(base)


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real_init = httpx.AsyncClient.__init__

    def init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)


APPLE = HostAccelerator(os=Os.macos, arch=Arch.arm64)
NOT_APPLE = HostAccelerator(os=Os.linux, arch=Arch.x64)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_mlx_is_registered_and_is_a_manual_engine() -> None:
    """The enum and the registry are two views of one fact, and MLX is
    now in both. `manual` for vLLM's reason: the unit of installation is
    a Python environment, which is not a thing we can fetch and hash."""
    adapter = adapter_for(EngineKind.mlx)
    assert isinstance(adapter, MlxAdapter)
    assert adapter.install_policy is Policy.manual
    assert adapter.configured_binary_key == "mlxBinary"
    # The dot belongs to the console script's name
    # (`setup.py`: `mlx_lm.server = mlx_lm.server:main`), and reading it
    # as an extension is the obvious way to get discovery wrong.
    assert adapter.binary_name == "mlx_lm.server"


# --------------------------------------------------------------------------- #
# argv
# --------------------------------------------------------------------------- #


def test_argv_passes_model_host_and_port(adapter: MlxAdapter, binary: DiscoveredBinary) -> None:
    argv = adapter.build_argv(_spec(), binary, 8130)
    assert argv[0] == str(binary.path)
    assert argv[1:3] == ["--model", "/Users/troy/models/Qwen3-8B-MLX-4bit"]
    assert "--host" in argv and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8130"


def test_no_served_model_name_is_passed_because_there_is_no_such_flag(
    adapter: MlxAdapter, binary: DiscoveredBinary
) -> None:
    """The one place this adapter cannot obey the base class.

    Every other adapter passes the alias explicitly, because an engine
    that names itself after its `--model` argument publishes the
    operator's directory layout as an OpenAI model id. `mlx_lm.server`
    has **no such flag** — `main()` at v0.31.3 defines no
    `--served-model-name` and no synonym — so there is nothing to pass
    and pretending otherwise would put an unknown argument on the
    command line.

    Pinned as a test so that the day upstream adds one, this fails and
    someone goes and uses it.
    """
    argv = adapter.build_argv(_spec(modelAlias="qwen3-8b"), binary, 8130)
    # Flags only. argv[0] is the console script, whose path here is a
    # pytest tmp_path built from this test's own (long) name -- the
    # first version of this assertion searched the whole list and
    # matched that, which is the same trap as looking for a subject
    # somewhere it never was.
    flags = [part for part in argv[1:] if part.startswith("--")]
    assert not any(part.startswith("--served") for part in flags)
    assert "qwen3-8b" not in argv[1:]


def test_curated_flags_map_to_the_v0_31_3_cli_names(
    adapter: MlxAdapter, binary: DiscoveredBinary
) -> None:
    """Each name is an `add_argument` in `server.py`'s `main()`.

    The value of this test is not that the mapping is applied — it is
    that the *names* are written down somewhere a person can diff
    against upstream when a flag stops working after an upgrade.
    """
    spec = _spec(
        flags={
            "maxTokens": 2048,
            "promptCacheSize": 4,
            "promptCacheBytes": "8GB",
            "decodeConcurrency": 16,
            "promptConcurrency": 4,
            "prefillStepSize": 1024,
            "draftModel": "/Users/troy/models/Qwen3-0.6B-MLX",
            "numDraftTokens": 5,
            "adapterPath": "/Users/troy/adapters/lora",
            "chatTemplate": "{{ messages }}",
        }
    )
    argv = adapter.build_argv(spec, binary, 8130)
    for flag, value in [
        ("--max-tokens", "2048"),
        ("--prompt-cache-size", "4"),
        ("--prompt-cache-bytes", "8GB"),
        ("--decode-concurrency", "16"),
        ("--prompt-concurrency", "4"),
        ("--prefill-step-size", "1024"),
        ("--draft-model", "/Users/troy/models/Qwen3-0.6B-MLX"),
        ("--num-draft-tokens", "5"),
        ("--adapter-path", "/Users/troy/adapters/lora"),
        ("--chat-template", "{{ messages }}"),
    ]:
        assert flag in argv, f"{flag} missing"
        assert argv[argv.index(flag) + 1] == value


def test_boolean_flags_are_presence_only(adapter: MlxAdapter, binary: DiscoveredBinary) -> None:
    """`--trust-remote-code` is `action="store_true"` upstream, so it
    takes no value and absent means false. Passing `--trust-remote-code
    false` would enable it, which is the failure mode this pins."""
    on = adapter.build_argv(_spec(flags={"trustRemoteCode": True}), binary, 8130)
    assert "--trust-remote-code" in on
    assert "True" not in on and "true" not in on

    off = adapter.build_argv(_spec(flags={"trustRemoteCode": False}), binary, 8130)
    assert "--trust-remote-code" not in off


def test_extra_args_come_last_so_they_can_override(
    adapter: MlxAdapter, binary: DiscoveredBinary
) -> None:
    argv = adapter.build_argv(
        _spec(flags={"maxTokens": 512}, extraArgs=["--max-tokens", "99"]), binary, 8130
    )
    assert argv[-2:] == ["--max-tokens", "99"]


def test_unknown_flags_are_rejected_rather_than_dropped(adapter: MlxAdapter) -> None:
    """A typo that silently vanishes is worse than one that errors: the
    engine starts, behaves differently from what was asked, and nothing
    says why."""
    assert adapter.validate_flags({"maxTokens": 1, "nonsense": 2}) == ["nonsense"]


def test_working_directory_is_inherited_not_the_venv_bin(
    adapter: MlxAdapter, binary: DiscoveredBinary
) -> None:
    """The base default is the binary's directory, which exists for
    prebuilt llama.cpp releases that keep shared libraries beside the
    executable. A console script has no such need."""
    assert adapter.working_directory(_spec(), binary) is None
    assert adapter.working_directory(_spec(workingDirectory="/tmp/x"), binary) == "/tmp/x"


# --------------------------------------------------------------------------- #
# the refusal
# --------------------------------------------------------------------------- #


def test_a_non_apple_host_gets_no_command_only_an_explanation(adapter: MlxAdapter) -> None:
    """The contract's rule is that a command which does not work is
    worse than no command. MLX makes that unusually sharp: `pip install
    mlx-lm` **succeeds** on Linux and Windows, because upstream guards
    only its `mlx` dependency with `platform_system == 'Darwin'`. So an
    install line here would appear to work and then fail at launch."""
    refusal = adapter.manual_install(NOT_APPLE)
    assert refusal.command is None
    assert refusal.docsUrl
    assert "Apple silicon" in (refusal.notes or "")
    assert "succeed" in (refusal.notes or "")


def test_apple_silicon_gets_the_install_line_and_the_next_step(adapter: MlxAdapter) -> None:
    install = adapter.manual_install(APPLE)
    assert install.command == "uv pip install mlx-lm"
    assert "mlxBinary" in (install.notes or "")


def test_an_intel_mac_is_refused_too(adapter: MlxAdapter) -> None:
    """macOS is not the condition; Apple silicon is. Metal on an Intel
    Mac will not run this engine."""
    assert adapter.manual_install(HostAccelerator(os=Os.macos, arch=Arch.x64)).command is None


# --------------------------------------------------------------------------- #
# readiness — the interesting part
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_nothing_listening_is_not_answering(
    adapter: MlxAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8130")
    assert isinstance(outcome, NotAnswering)
    # And `reached=False`, which is what separates "nothing there" from
    # "something answered unusably". The server binds almost at once, so
    # this window is genuinely short.
    assert outcome.reached is False


@pytest.mark.asyncio
async def test_health_ok_but_no_token_is_loading(
    adapter: MlxAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The claim this whole adapter turns on.**

    `handle_health_check` writes a hardcoded `{"status": "ok"}` with no
    reference to the model, and `run()` starts the HTTP server while
    `load_default()` is still going on the generator thread. So a 200
    from `/health` proves only that the server exists. Treating it as
    readiness would hand the gateway a runtime whose first request pays
    the entire model load — reported as backend slowness, with
    `swappedIn: false`, which is precisely the misattribution M8 found
    for external backends.

    Loading is *stated* here rather than inferred. Unlike vLLM this
    needs no process handle: the server answered, and it will not
    generate, and there is nothing else that can mean.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        raise httpx.ReadTimeout("still loading", request=request)

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8130")
    assert isinstance(outcome, Loading)
    assert "another thread" in (outcome.detail or "")


@pytest.mark.asyncio
async def test_a_token_comes_back_and_that_is_ready(
    adapter: MlxAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        import json as _json

        seen.append(_json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8130")
    assert isinstance(outcome, Ready)

    # The probe asks for exactly one token, and names upstream's own
    # sentinel rather than restating the model path: `APIHandler` reads
    # `body.get("model", "default_model")` and `ModelProvider` seeds
    # `_model_map["default_model"] = cli_args.model`.
    assert seen and seen[0]["max_tokens"] == 1
    assert seen[0]["model"] == DEFAULT_MODEL_SENTINEL


@pytest.mark.asyncio
async def test_no_capabilities_are_invented(
    adapter: MlxAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MLX publishes no context length anywhere. The library computes
    fit verdicts against whatever it is told, so an invented context
    window is worse than an absent one."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8130")
    assert isinstance(outcome, Ready)
    assert outcome.capabilities is None


@pytest.mark.asyncio
async def test_a_served_error_is_not_reported_as_an_endless_load(
    adapter: MlxAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4xx from the completion is an answer, not a load.

    The likeliest cause is a launch with no `--model`: `load_default()`
    is a no-op when `cli_args.model` is None, so `default_model` maps to
    None and every request fails. Reporting that as `loading` would mean
    a runtime that sits at "loading" forever with nothing wrong that
    anyone can see.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(400, text="model not found")

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8130")
    assert isinstance(outcome, NotAnswering)
    assert outcome.reached is True
    assert "400" in (outcome.detail or "")


@pytest.mark.asyncio
async def test_an_unrecognised_health_status_is_not_readiness(
    adapter: MlxAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream only ever writes 200 there, so anything else is a server
    we do not recognise — and must not be followed by an expensive
    generation against something that may not be MLX at all."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(503, text="nope")

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8130")
    assert isinstance(outcome, NotAnswering)
    assert calls == ["/health"]


# --------------------------------------------------------------------------- #
# exit explanations
# --------------------------------------------------------------------------- #


def test_the_darwin_marker_trap_is_explained(adapter: MlxAdapter) -> None:
    """The failure this engine is most likely to produce in the wild,
    and the one whose traceback names none of the cause: mlx-lm installed
    on a non-Apple host, which upstream's environment marker permits."""
    explanation = adapter.explain_exit(1, "ModuleNotFoundError: No module named 'mlx'\n")
    assert explanation is not None
    assert "Darwin" in explanation
    assert "Apple silicon" in explanation


def test_an_unknown_death_gets_no_invented_explanation(adapter: MlxAdapter) -> None:
    """The generic message stands rather than a guess replacing it."""
    assert adapter.explain_exit(1, "Segmentation fault\n") is None


def test_nothing_is_refused_before_launch(adapter: MlxAdapter) -> None:
    """`default_env` stays empty and there is no pre-launch host check.

    The standing rule from 2026-09-11: an eager refusal can be wrong and
    an explanation of a real failure cannot. vLLM proved it — a warm
    Triton cache runs with `CC=/nonexistent`, so refusing on a missing
    compiler would have rejected a launch that works.
    """
    assert adapter.default_env(_spec(), _fake_binary()) == {}


def _fake_binary() -> DiscoveredBinary:
    return DiscoveredBinary(
        path=Path("/opt/mlx/bin/mlx_lm.server"), origin=Origin.configured, version="0.31.3"
    )


# --------------------------------------------------------------------------- #
# flag schema
# --------------------------------------------------------------------------- #


def test_flag_schema_is_a_standard_config_schema(adapter: MlxAdapter) -> None:
    """Returning a ConfigSchema is what lets the generic config editor
    render engine flags with no engine-specific UI code."""
    schema = adapter.flag_schema()
    assert schema.component == "engine:mlx"
    assert schema.fields
    for field in schema.fields:
        assert field.category in schema.categories
        assert field.label and field.description
        assert isinstance(field.valueType, ConfigValueType)
