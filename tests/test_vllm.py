"""The vLLM adapter: argv, discovery, environment readout, readiness.

Everything here runs against fixtures — a fake console script, a fake
interpreter probe, a fake HTTP transport, a fake process handle. That is
real verification of the adapter's *logic* and the readiness state
machine. What it cannot verify is vLLM itself: no vLLM process has ever
run for this project (the dev box is Windows and vLLM has no Windows
build), so the wall-clock startup budget and the "connections are
refused, not accepted-and-hung" claim wait for the first Linux run.
`scripts/m4-acceptance.sh` in the specs repo is that run, unexecuted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent._generated.models import (
    Accelerator,
    Arch,
    ConfigValueType,
    EngineKind,
    FrameworkAccelerator,
    HostAccelerator,
    Origin,
    Os,
    Policy,
    RuntimeSpec,
)
from eugene_plexus_agent.engines import (
    EngineUnavailableError,
    LlamaCppAdapter,
    VllmAdapter,
    adapter_for,
    interpret_readiness,
)
from eugene_plexus_agent.engines.base import (
    DiscoveredBinary,
    Loading,
    NotAnswering,
    Ready,
)
from eugene_plexus_agent.engines.vllm import (
    _FLAG_CLI_NAMES,
    STARTUP_BUDGET_SECONDS,
    accelerator_from_torch_version,
    inspect_python_environment,
    interpreter_from_shebang,
    manual_install_for,
)


@pytest.fixture
def adapter() -> VllmAdapter:
    return VllmAdapter()


@pytest.fixture
def binary(tmp_path: Path) -> DiscoveredBinary:
    exe = tmp_path / "venv" / "bin" / "vllm"
    exe.parent.mkdir(parents=True)
    exe.write_text(f"#!{tmp_path}/venv/bin/python3.12\n", encoding="utf-8")
    return DiscoveredBinary(path=exe, origin=Origin.configured, version="0.29.0")


def _spec(**overrides: Any) -> RuntimeSpec:
    base: dict[str, Any] = {
        "name": "qwen3-8b",
        "engine": EngineKind.vllm,
        "modelPath": "/home/troy/models/Qwen3-8B",
    }
    base.update(overrides)
    return RuntimeSpec.model_validate(base)


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real_init = httpx.AsyncClient.__init__

    def init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_vllm_is_registered_and_loads_safetensors_only() -> None:
    adapter = adapter_for(EngineKind.vllm)
    assert isinstance(adapter, VllmAdapter)
    # GGUF through vLLM is experimental upstream and needs a second
    # tokenizer model; claiming it would light up Launch across the whole
    # GGUF population llama.cpp already serves properly.
    assert [f.value for f in adapter.model_formats] == ["safetensors"]
    assert adapter.install_policy is Policy.manual
    assert adapter.configured_binary_key == "vllmBinary"


# --------------------------------------------------------------------------- #
# argv
# --------------------------------------------------------------------------- #


def test_argv_is_vllm_serve_with_a_positional_model(
    adapter: VllmAdapter, binary: DiscoveredBinary
) -> None:
    argv = adapter.build_argv(_spec(), binary, port=8090)

    # `vllm serve [model_tag] [options]` — upstream copies the positional
    # onto `args.model` and hides `--model` from `vllm serve --help`.
    assert argv[:3] == [str(binary.path), "serve", "/home/troy/models/Qwen3-8B"]
    assert "--model" not in argv
    assert argv[argv.index("--port") + 1] == "8090"
    # Loopback by default: an engine has no auth of its own.
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


def test_served_model_name_is_always_passed(adapter: VllmAdapter, binary: DiscoveredBinary) -> None:
    """vLLM's default served name is the `--model` argument verbatim, and
    we launch by absolute path — so leaving it unset would publish
    `/home/troy/models/Qwen3-8B` as an OpenAI model id: the operator's
    layout leaked to every client, and a routing key that differs per
    host for the same model."""
    argv = adapter.build_argv(_spec(), binary, port=8090)
    assert argv[argv.index("--served-model-name") + 1] == "Qwen3-8B"
    assert "/home/troy/models/Qwen3-8B" not in argv[argv.index("--served-model-name") + 1]

    explicit = adapter.build_argv(_spec(modelAlias="qwen"), binary, port=8090)
    assert explicit[explicit.index("--served-model-name") + 1] == "qwen"


def test_curated_flags_map_to_the_v0_29_0_cli_names(
    adapter: VllmAdapter, binary: DiscoveredBinary
) -> None:
    spec = _spec(
        flags={
            "maxModelLen": 32768,
            "gpuMemoryUtilization": 0.45,
            "tensorParallelSize": 2,
            "pipelineParallelSize": 1,
            "maxNumSeqs": 64,
            "maxNumBatchedTokens": 8192,
            "dtype": "bfloat16",
            "quantization": "awq",
            "kvCacheDtype": "fp8",
            "tokenizer": "/models/tok",
        }
    )
    argv = adapter.build_argv(spec, binary, port=8090)
    assert argv[argv.index("--max-model-len") + 1] == "32768"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.45"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--pipeline-parallel-size") + 1] == "1"
    assert argv[argv.index("--max-num-seqs") + 1] == "64"
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "8192"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--quantization") + 1] == "awq"
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"
    assert argv[argv.index("--tokenizer") + 1] == "/models/tok"


def test_boolean_flags_are_presence_only(adapter: VllmAdapter, binary: DiscoveredBinary) -> None:
    """vLLM's booleans are `argparse.BooleanOptionalAction`: `--enforce-eager`
    on, `--no-enforce-eager` off, neither takes a value. Every curated
    boolean defaults to off upstream, so absent means false."""
    on = adapter.build_argv(
        _spec(flags={"enforceEager": True, "trustRemoteCode": True}), binary, port=8090
    )
    assert "--enforce-eager" in on
    assert "--trust-remote-code" in on
    for flag in ("--enforce-eager", "--trust-remote-code"):
        following = on[on.index(flag) + 1 :]
        assert following == [] or following[0].startswith("--")

    off = adapter.build_argv(
        _spec(flags={"enforceEager": False, "trustRemoteCode": False}), binary, port=8090
    )
    assert "--enforce-eager" not in off
    assert "--trust-remote-code" not in off
    assert "false" not in off


def test_unset_flags_are_absent_entirely(adapter: VllmAdapter, binary: DiscoveredBinary) -> None:
    """The engine's own default is the honest one — 0.92 for GPU memory,
    the model's configured length for context."""
    argv = adapter.build_argv(_spec(), binary, port=8090)
    for cli in _FLAG_CLI_NAMES.values():
        assert cli not in argv


def test_extra_args_come_last_so_they_can_override(
    adapter: VllmAdapter, binary: DiscoveredBinary
) -> None:
    spec = _spec(flags={"maxModelLen": 4096}, extraArgs=["--max-model-len", "auto"])
    argv = adapter.build_argv(spec, binary, port=8090)
    assert argv[-2:] == ["--max-model-len", "auto"]
    assert argv.count("--max-model-len") == 2


def test_working_directory_is_inherited_not_the_venv_bin(
    adapter: VllmAdapter, binary: DiscoveredBinary
) -> None:
    """The base default (the binary's own directory) exists for prebuilt
    llama.cpp releases with shared libraries beside the executable. A
    console script has no such need."""
    assert adapter.working_directory(_spec(), binary) is None
    assert adapter.working_directory(_spec(workingDirectory="/work"), binary) == "/work"


# --------------------------------------------------------------------------- #
# discovery: explicit binary > configured (install-wide) > managed > PATH
# --------------------------------------------------------------------------- #


def _fake_console_script(root: Path, name: str = "vllm") -> Path:
    exe = root / "bin" / name
    exe.parent.mkdir(parents=True, exist_ok=True)
    # Point the shebang at the real interpreter so the environment probe
    # runs for real (and finds no vllm, which is the honest answer here).
    exe.write_text(f"#!{sys.executable}\n", encoding="utf-8")
    return exe


def test_configured_path_is_found_with_origin_configured(
    adapter: VllmAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`vllmBinary` is how an engine we never install becomes
    discoverable at all — without it a perfectly good venv reads as
    `available: false`."""
    exe = _fake_console_script(tmp_path / "venv")
    monkeypatch.setattr("shutil.which", lambda name: None)

    found = adapter.discover(configured=str(exe))
    assert found is not None
    assert found.path == exe
    assert found.origin == Origin.configured
    # The environment was inspected, once, and the interpreter is the
    # one the shebang named.
    assert found.python is not None
    assert found.python.interpreter == sys.executable


def test_configured_path_beats_path(
    adapter: VllmAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _fake_console_script(tmp_path / "venv")
    on_path = _fake_console_script(tmp_path / "system")
    monkeypatch.setattr("shutil.which", lambda name: str(on_path))

    found = adapter.discover(configured=str(configured))
    assert found is not None and found.path == configured

    fallback = adapter.discover(configured=None)
    assert fallback is not None and fallback.path == on_path
    assert fallback.origin == Origin.path


def test_missing_configured_path_is_an_error_not_a_fallback(
    adapter: VllmAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator said where the engine is. Quietly using whatever is
    on PATH would run something they did not choose."""
    on_path = _fake_console_script(tmp_path / "system")
    monkeypatch.setattr("shutil.which", lambda name: str(on_path))

    with pytest.raises(EngineUnavailableError, match="vllmBinary") as caught:
        adapter.discover(configured=str(tmp_path / "nope" / "vllm"))
    assert "does not exist" in str(caught.value)


def test_explicit_runtime_binary_still_beats_the_configured_path(
    adapter: VllmAdapter, tmp_path: Path
) -> None:
    configured = _fake_console_script(tmp_path / "venv")
    per_runtime = _fake_console_script(tmp_path / "special")

    resolved = adapter.resolve_binary(_spec(binary=str(per_runtime)), configured=str(configured))
    assert resolved.path == per_runtime
    assert resolved.origin == Origin.configured


def test_nothing_found_names_vllm_binary_as_the_fix(
    adapter: VllmAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(EngineUnavailableError) as caught:
        adapter.resolve_binary(_spec())
    message = " ".join(str(caught.value).split())
    assert "vllmBinary" in message
    assert "manualInstall" in message
    # And it must not point at the install endpoint, which 422s for vLLM.
    assert "/install" not in message


# --------------------------------------------------------------------------- #
# environment readout
# --------------------------------------------------------------------------- #


def test_shebang_parsing_handles_absolute_and_env_forms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    absolute = tmp_path / "a"
    absolute.write_text("#!/opt/venv/bin/python3.12\nimport re\n", encoding="utf-8")
    # The literal string, not a Path: what the script will execute.
    assert interpreter_from_shebang(absolute) == "/opt/venv/bin/python3.12"

    via_env = tmp_path / "b"
    via_env.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/python3")
    assert interpreter_from_shebang(via_env) == "/usr/local/bin/python3"

    # A Windows console-script launcher is a PE binary: no shebang, no
    # environment readout, and no invented one either.
    binary = tmp_path / "c.exe"
    binary.write_bytes(b"MZ\x90\x00" + b"\x00" * 64)
    assert interpreter_from_shebang(binary) is None
    assert inspect_python_environment(binary) is None


def test_environment_probe_reads_metadata_without_importing_vllm(tmp_path: Path) -> None:
    """Against the real interpreter running this test: the probe answers
    with its Python version and, honestly, no vllm — this venv has none.
    Whether torch is present depends on the box (the agent venv is the
    install's runtime venv and may carry one), so only the consistency
    between the torch version and the accelerator is asserted."""
    exe = _fake_console_script(tmp_path)
    env = inspect_python_environment(exe)
    assert env is not None
    assert env.interpreter == sys.executable
    assert env.pythonVersion == ".".join(str(part) for part in sys.version_info[:3])
    assert env.packageVersion is None
    if env.torchVersion is None:
        assert env.accelerator is None
    else:
        assert isinstance(env.accelerator, FrameworkAccelerator)


def test_environment_probe_reports_versions_and_accelerator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "vllm"
    exe.write_text("#!/opt/venv/bin/python3.12\n", encoding="utf-8")

    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["argv"] = argv
        payload = {"python": "3.12.8", "vllm": "0.29.0", "torch": "2.9.0+cu129"}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    env = inspect_python_environment(exe)
    assert env is not None
    # Isolated mode, so the operator's site customisations cannot break
    # the probe; and the probe is the interpreter, never `vllm --version`.
    assert seen["argv"][:2] == ["/opt/venv/bin/python3.12", "-I"]
    assert env.interpreter == "/opt/venv/bin/python3.12"
    assert env.pythonVersion == "3.12.8"
    assert env.packageVersion == "0.29.0"
    assert env.torchVersion == "2.9.0+cu129"
    assert env.accelerator is FrameworkAccelerator.cuda


def test_torch_build_tag_maps_to_framework_accelerator() -> None:
    assert accelerator_from_torch_version("2.9.0+cu129") is FrameworkAccelerator.cuda
    assert accelerator_from_torch_version("2.9.0+rocm7.0") is FrameworkAccelerator.rocm
    assert accelerator_from_torch_version("2.9.0+xpu") is FrameworkAccelerator.xpu
    assert accelerator_from_torch_version("2.9.0+cpu") is FrameworkAccelerator.none
    # PyPI forbids local version tags, so PyPI's Linux wheel — a CUDA
    # build — reports a bare version, as does the CPU-only macOS wheel.
    # `unknown` is the honest answer; `none` would claim an absence.
    assert accelerator_from_torch_version("2.9.0") is FrameworkAccelerator.unknown
    assert accelerator_from_torch_version("2.9.0+weird") is FrameworkAccelerator.unknown
    assert accelerator_from_torch_version(None) is None


def test_describe_inspects_the_environment_once(
    adapter: VllmAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "vllm"
    exe.write_text("#!/opt/venv/bin/python3.12\n", encoding="utf-8")
    calls = {"n": 0}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        payload = {"python": "3.12.8", "vllm": "0.29.0", "torch": "2.9.0+cu129"}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    found = adapter.describe(exe, Origin.configured)
    assert calls["n"] == 1
    assert found.version == "0.29.0"
    assert found.python is not None and found.python.torchVersion == "2.9.0+cu129"


# --------------------------------------------------------------------------- #
# manual install: the refusal names the command
# --------------------------------------------------------------------------- #


def _host(os_: Os | None, accelerator: Accelerator | None, arch: Arch | None = Arch.x64) -> Any:
    return HostAccelerator(os=os_, arch=arch, accelerator=accelerator, acceleratorVersion=None)


def test_manual_install_names_upstreams_command_per_accelerator() -> None:
    cuda = manual_install_for(_host(Os.linux, Accelerator.cuda))
    assert cuda.command == "uv pip install vllm --torch-backend=auto"
    assert cuda.docsUrl.endswith("/installation/gpu/")
    assert cuda.notes and "vllmBinary" in cuda.notes

    rocm = manual_install_for(_host(Os.linux, Accelerator.rocm))
    assert rocm.command is not None and "wheels.vllm.ai/rocm/" in rocm.command
    assert rocm.notes and "3.12" in rocm.notes

    xpu = manual_install_for(_host(Os.linux, Accelerator.sycl))
    assert xpu.command is not None and "download.pytorch.org/whl/xpu" in xpu.command

    cpu = manual_install_for(_host(Os.linux, Accelerator.none, Arch.arm64))
    assert (
        cpu.command is not None
        and "aarch64" in cpu.command
        and "--torch-backend cpu" in cpu.command
    )
    assert cpu.docsUrl.endswith("/installation/cpu/")


def test_manual_install_refuses_to_guess_where_upstream_has_no_command() -> None:
    """A command that does not work is worse than no command, because the
    operator will believe it."""
    windows = manual_install_for(_host(Os.windows, Accelerator.cuda))
    assert windows.command is None
    assert windows.notes and "WSL" in windows.notes

    mac = manual_install_for(_host(Os.macos, Accelerator.metal, Arch.arm64))
    assert mac.command is None
    assert mac.notes and "vLLM-Metal" in mac.notes

    unknown = manual_install_for(_host(None, None, None))
    assert unknown.command is None
    assert unknown.docsUrl


def test_every_manual_install_has_a_docs_url() -> None:
    """The one field that always exists: upstream's instructions are
    correct for longer than any copy we make of them."""
    for os_ in (*Os, None):
        for acc in (*Accelerator, None):
            manual = manual_install_for(_host(os_, acc))
            assert manual.docsUrl.startswith("https://")


# --------------------------------------------------------------------------- #
# readiness — the network half
#
# vLLM binds its port before loading and answers nothing until the model
# is resident, so the probe alone cannot tell loading from dead. It
# reports what it saw; the supervisor adds the process handle below.
# --------------------------------------------------------------------------- #


async def test_connection_refused_is_not_answering_and_not_reached(
    adapter: VllmAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8090")
    assert isinstance(outcome, NotAnswering)
    # The probe must NOT call this loading on its own; it does not know
    # whether the process is alive.
    assert outcome.reached is False


async def test_health_503_is_a_dead_engine_not_a_load(
    adapter: VllmAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream's `/health` returns 503 only on `EngineDeadError`. Reading
    that as `loading` — llama-server's meaning for 503 — would show a
    dead engine as "working on it"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8090")
    assert isinstance(outcome, NotAnswering)
    assert outcome.reached is True
    assert outcome.detail and "dead" in outcome.detail


async def test_health_200_is_ready_with_version_and_context_read_back(
    adapter: VllmAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)  # empty body, as upstream sends it
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.29.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "Qwen3-8B",
                            "object": "model",
                            "max_model_len": 40960,
                            "root": "/home/troy/models/Qwen3-8B",
                        }
                    ],
                },
            )
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8090")
    assert isinstance(outcome, Ready)
    assert outcome.version == "0.29.0"
    assert outcome.capabilities is not None
    # Read back off the engine, not inferred from the spec.
    assert outcome.capabilities.contextLength == 40960
    # Nothing reports the sequence budget back, so it is not invented.
    assert outcome.capabilities.parallelSlots is None


async def test_ready_survives_an_unreadable_models_endpoint(
    adapter: VllmAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/v1/models` is behind `--api-key` if the operator set one through
    extraArgs. A runtime that is serving but will not describe itself is
    still serving."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(401, json={"error": "Unauthorized"})

    _patch_client(monkeypatch, handler)
    outcome = await adapter.probe_readiness("http://127.0.0.1:8090")
    assert isinstance(outcome, Ready)
    assert outcome.capabilities is None
    assert outcome.version is None


# --------------------------------------------------------------------------- #
# readiness — the supervisor's half
#
# This is M4's architectural result: "the process is alive and nothing
# answers" IS loading for an engine that does not answer while it loads.
# It needs the pid, which the supervisor has and a network probe does not.
# --------------------------------------------------------------------------- #


def test_alive_and_refusing_connections_is_loading_for_vllm(adapter: VllmAdapter) -> None:
    silent = NotAnswering(detail="connection refused", reached=False)
    outcome = interpret_readiness(adapter, silent, process_alive=True, elapsed_seconds=45.0)
    assert isinstance(outcome, Loading)
    assert outcome.past_budget is False
    assert outcome.detail and "45s" in outcome.detail


def test_a_dead_process_stays_not_answering(adapter: VllmAdapter) -> None:
    """No pid, nothing to be loading. The supervisor's own crashed/exited
    mapping takes over from here."""
    silent = NotAnswering(detail="connection refused", reached=False)
    outcome = interpret_readiness(adapter, silent, process_alive=False, elapsed_seconds=45.0)
    assert outcome is silent


def test_an_answer_that_was_reached_is_not_reinterpreted(adapter: VllmAdapter) -> None:
    """A 503 from a listening server is information, not silence."""
    dead = NotAnswering(detail="engine dead", reached=True)
    assert interpret_readiness(adapter, dead, process_alive=True, elapsed_seconds=1.0) is dead


def test_llama_cpp_is_unaffected_by_the_rule() -> None:
    """llama-server narrates its own load. Its silence really is silence."""
    silent = NotAnswering(detail="connection refused", reached=False)
    outcome = interpret_readiness(
        LlamaCppAdapter(), silent, process_alive=True, elapsed_seconds=45.0
    )
    assert outcome is silent


def test_past_the_startup_budget_is_still_loading_but_flagged(adapter: VllmAdapter) -> None:
    """The state stays `loading` — a live, silent process cannot be
    anything else — but the operator gets the elapsed time and a pointer
    at the captured output. Evidence, not a guess at "wedged"."""
    silent = NotAnswering(detail="connection refused", reached=False)
    outcome = interpret_readiness(
        adapter, silent, process_alive=True, elapsed_seconds=STARTUP_BUDGET_SECONDS + 12
    )
    assert isinstance(outcome, Loading)
    assert outcome.past_budget is True
    assert outcome.detail and "612s" in outcome.detail
    assert "captured engine output" in outcome.detail

    within = interpret_readiness(
        adapter, silent, process_alive=True, elapsed_seconds=STARTUP_BUDGET_SECONDS - 1
    )
    assert isinstance(within, Loading) and within.past_budget is False


def test_ready_and_loading_pass_straight_through(adapter: VllmAdapter) -> None:
    ready = Ready()
    assert interpret_readiness(adapter, ready, process_alive=True, elapsed_seconds=1.0) is ready
    loading = Loading(detail="x")
    assert interpret_readiness(adapter, loading, process_alive=True, elapsed_seconds=1.0) is loading


# --------------------------------------------------------------------------- #
# flag schema
# --------------------------------------------------------------------------- #


def test_flag_schema_is_a_standard_config_schema(adapter: VllmAdapter) -> None:
    schema = adapter.flag_schema()
    assert schema.component == "engine:vllm"
    assert schema.fields
    for field in schema.fields:
        assert field.label, f"{field.key} needs a label for the form"
        assert field.description, f"{field.key} needs help text — that is the point"
        assert field.category in (schema.categories or {})


def test_every_schema_flag_has_a_cli_mapping(adapter: VllmAdapter) -> None:
    assert {f.key for f in adapter.flag_schema().fields} == set(_FLAG_CLI_NAMES)


def test_the_curated_surface_is_the_designed_twelve(adapter: VllmAdapter) -> None:
    """The M4 design's flag table, checked against `vllm/engine/arg_utils.py`
    at v0.29.0 — every one is an `add_argument` there. Model path, host,
    port and `--served-model-name` are the adapter's, not the operator's,
    and are deliberately absent."""
    keys = {f.key for f in adapter.flag_schema().fields}
    assert keys == {
        "maxModelLen",
        "gpuMemoryUtilization",
        "tensorParallelSize",
        "pipelineParallelSize",
        "maxNumSeqs",
        "maxNumBatchedTokens",
        "dtype",
        "quantization",
        "kvCacheDtype",
        "enforceEager",
        "trustRemoteCode",
        "tokenizer",
    }
    assert "servedModelName" not in keys
    assert "model" not in keys


def test_dtype_and_kv_cache_dtype_enums_match_upstream_literals(adapter: VllmAdapter) -> None:
    fields = {f.key: f for f in adapter.flag_schema().fields}
    # `ModelDType` in vllm/config/model.py at v0.29.0.
    assert fields["dtype"].enumValues == ["auto", "half", "float16", "bfloat16", "float", "float32"]
    # A curated subset of `CacheDType` in vllm/config/cache.py: the
    # hardware-specific ones go through extraArgs.
    assert fields["kvCacheDtype"].enumValues is not None
    assert set(fields["kvCacheDtype"].enumValues) <= {
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_inc",
        "fp8_ds_mla",
    }
    assert fields["gpuMemoryUtilization"].valueType == ConfigValueType.number
    assert fields["gpuMemoryUtilization"].maximum == 1.0
    # No default leaks into argv — the engine's 0.92 is the honest one.
    assert fields["gpuMemoryUtilization"].default is None
    assert fields["maxModelLen"].default is None


def test_trust_remote_code_says_what_it_is(adapter: VllmAdapter) -> None:
    """It executes code shipped inside the model folder. The UI presents
    it as what it is."""
    field = next(f for f in adapter.flag_schema().fields if f.key == "trustRemoteCode")
    assert field.valueType == ConfigValueType.boolean
    assert field.default is False
    assert "executes code" in field.description.lower()


def test_unknown_flags_are_reported_not_dropped(adapter: VllmAdapter) -> None:
    unknown = adapter.validate_flags({"maxModelLen": 8192, "ctxSize": 4096, "gpuLayers": 99})
    assert unknown == ["ctxSize", "gpuLayers"]
