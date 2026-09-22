"""The Kev adapter — the decision engine, from measured claims.

Pinned to `kev/serve.py` at commit 1c35199 and a live kev-0.8b run on
WSL CPU (2026-09-22). The three shapes that make this engine unlike the
chat three, each with its section: the "binary" is an interpreter and a
bare `python` on PATH is never evidence of a Kev environment; the model
loads before the port binds, so readiness is vLLM's shape with Kev's
`/v1/models` as the probe; and the companion it declares speaks the
System One protocol with a one-request concurrency ceiling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent import _http
from eugene_plexus_agent._generated.models import (
    EngineKind,
    ModelFormat,
    Origin,
    Policy,
    RuntimeSpec,
)
from eugene_plexus_agent.companions import companion_overrides, render_config
from eugene_plexus_agent.engines import adapter_for
from eugene_plexus_agent.engines.base import (
    DiscoveredBinary,
    EngineUnavailableError,
    Loading,
    NotAnswering,
    Ready,
    interpret_readiness,
)
from eugene_plexus_agent.engines.kev import UPSTREAM_COMMIT_PINNED, KevAdapter
from eugene_plexus_agent.runtimes import validate_spec


@pytest.fixture
def adapter() -> KevAdapter:
    return KevAdapter()


@pytest.fixture
def binary(tmp_path: Path) -> DiscoveredBinary:
    exe = tmp_path / "eugene-kev" / ".venv" / "bin" / "python"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    return DiscoveredBinary(path=exe, origin=Origin.configured, version="0.4.0")


def _spec(**overrides: Any) -> RuntimeSpec:
    base: dict[str, Any] = {
        "name": "tickets",
        "engine": EngineKind.kev,
        "modelPath": "/home/troy/checkpoints/kev-0.8b",
    }
    base.update(overrides)
    return RuntimeSpec.model_validate(base)


def _patch_client(handler: Any) -> None:
    _http.set_shared_client(
        "engine-probe", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


# --------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------- #


def test_kev_is_registered_experimental_and_decision_only() -> None:
    adapter = adapter_for(EngineKind.kev)
    assert isinstance(adapter, KevAdapter)
    assert adapter.experimental is True
    assert KevAdapter.model_formats == (ModelFormat.kev_checkpoint,)
    assert KevAdapter.install_policy is Policy.manual
    assert KevAdapter.configured_binary_key == "kevPython"
    # vLLM's readiness shape, measured: the load precedes the bind.
    assert KevAdapter.answers_while_loading is False
    assert KevAdapter.startup_budget_seconds == 900.0


# --------------------------------------------------------------------- #
# argv — the pinned flags and nothing else
# --------------------------------------------------------------------- #


def test_argv_is_module_run_port(adapter: KevAdapter, binary: DiscoveredBinary) -> None:
    argv = adapter.build_argv(_spec(), binary, 8409)
    assert argv == [
        str(binary.path),
        "-m",
        "kev.serve",
        "--run",
        "/home/troy/checkpoints/kev-0.8b",
        "--port",
        "8409",
    ]


def test_no_host_flag_exists_and_a_lan_host_is_refused() -> None:
    """Upstream binds loopback unconditionally (no --host at the pinned
    commit). A spec asking for a LAN bind would silently not get one, so
    it is refused at declaration with the reason."""
    reason = validate_spec(_spec(host="0.0.0.0"))
    assert reason is not None
    assert "127.0.0.1" in reason and "--host" in reason
    assert validate_spec(_spec(host="127.0.0.1")) is None or "kev.serve" not in str(
        validate_spec(_spec(host="127.0.0.1"))
    )


def test_the_fallback_flag_maps_and_unknown_flags_are_rejected(
    adapter: KevAdapter, binary: DiscoveredBinary
) -> None:
    argv = adapter.build_argv(_spec(flags={"fallback": "runs/smoke"}), binary, 8409)
    assert "--fallback" in argv
    assert adapter.validate_flags({"gpuLayers": 99}) == ["gpuLayers"]


def test_working_directory_is_the_checkout(adapter: KevAdapter, binary: DiscoveredBinary) -> None:
    """`python -m kev.serve` resolves relative artifacts against the
    cwd, and the interpreter lives at <checkout>/.venv/bin/python."""
    assert adapter.working_directory(_spec(), binary) == str(binary.path.parent.parent.parent)


# --------------------------------------------------------------------- #
# discovery — never PATH
# --------------------------------------------------------------------- #


def test_discover_never_falls_back_to_path(adapter: KevAdapter) -> None:
    """A bare `python` on PATH is never evidence of a Kev environment;
    the base-class which() fallback would 'find' the engine on every
    machine that has Python and not Kev — which is every machine."""
    assert adapter.discover(configured=None) is None


def test_a_missing_configured_interpreter_is_an_error(adapter: KevAdapter) -> None:
    with pytest.raises(EngineUnavailableError):
        adapter.discover(configured="/nowhere/.venv/bin/python")


def test_manual_install_pins_the_commit(adapter: KevAdapter) -> None:
    from eugene_plexus_agent._generated.models import Arch, HostAccelerator, Os

    install = adapter.manual_install(HostAccelerator(os=Os.linux, arch=Arch.x64))
    assert install.command is not None
    assert UPSTREAM_COMMIT_PINNED in install.command
    assert "uv sync --extra serve" in install.command
    assert "kevPython" in (install.notes or "")
    assert "loopback" in (install.notes or "")


# --------------------------------------------------------------------- #
# readiness — /v1/models once serving; silence + live pid = loading
# --------------------------------------------------------------------- #


async def test_models_answering_is_ready(adapter: KevAdapter) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"models": [{"id": "kev-latest", "run": "kev-0.8b", "device": "cpu"}]},
        )

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8409")
    assert isinstance(outcome, Ready)


async def test_alive_and_refusing_is_loading(adapter: KevAdapter) -> None:
    """The measured shape: `ck.load(...)` — and on a first run the base
    model download — completes before uvicorn binds. Only the supervisor
    can turn that silence into `loading`."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8409")
    assert isinstance(outcome, NotAnswering)
    assert outcome.reached is False

    interpreted = interpret_readiness(adapter, outcome, process_alive=True, elapsed_seconds=30.0)
    assert isinstance(interpreted, Loading)

    dead = interpret_readiness(adapter, outcome, process_alive=False, elapsed_seconds=30.0)
    assert isinstance(dead, NotAnswering)


async def test_a_long_first_launch_is_flagged_past_budget(adapter: KevAdapter) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    _patch_client(handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8409")
    interpreted = interpret_readiness(
        adapter, outcome, process_alive=True, elapsed_seconds=adapter.startup_budget_seconds + 60
    )
    assert isinstance(interpreted, Loading)
    assert interpreted.past_budget is True


# --------------------------------------------------------------------- #
# the companion speaks decisions
# --------------------------------------------------------------------- #


def test_kev_companion_gets_the_decision_provider_and_the_ceiling() -> None:
    overrides = companion_overrides(_spec())
    document = render_config(
        runtime_name="tickets", alias="kev-0.8b", upstream=None, overrides=overrides
    )
    assert document["provider"] == "systemone_custom"
    # The measured fact the gateway enforces: the pinned server holds
    # one request at a time and cannot shed work.
    assert document["decisionMaxConcurrent"] == 1
    assert document["modelId"] == "kev-0.8b"


def test_chat_engine_companions_are_unchanged() -> None:
    llama = _spec(engine=EngineKind.llama_cpp, modelPath="/models/q.gguf")
    document = render_config(
        runtime_name="qwen", alias="q", upstream=None, overrides=companion_overrides(llama)
    )
    assert document["provider"] == "openai_compat_custom"
    assert document["decisionMaxConcurrent"] is None


def test_an_override_outside_the_managed_keys_is_refused() -> None:
    """The merge writes only managed keys into an operator's file; an
    adapter inventing a new one would leave strays the clearing rule
    never clears."""
    with pytest.raises(ValueError):
        render_config(runtime_name="x", alias="y", overrides={"apiKey": "nope"})
