"""Prepared models, truthful readiness and ownership, without a GPU."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import httpx
import pytest

from eugene_plexus_agent._generated.models import EngineKind, HostAccelerator, Origin, RuntimeSpec
from eugene_plexus_agent.engines import strata, strata_models
from eugene_plexus_agent.engines.acquisition import (
    AcquisitionError,
    AcquisitionPlan,
    EngineInstaller,
    ManagedStore,
    Unavailable,
)
from eugene_plexus_agent.engines.base import DiscoveredBinary, Loading, NotAnswering, Ready
from eugene_plexus_agent.engines.strata_install import plan
from eugene_plexus_agent.runtimes import validate_spec
from eugene_plexus_agent.supervisor import SpawnPlanError


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_ENGINE_ROOT", str(tmp_path / "owned"))
    root = tmp_path / "borrowed"
    server = root / "serve/server.py"
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    native = root / "engine" / ("strata.exe" if os.name == "nt" else "strata")
    for f in (server, python, native):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
    dll = root / ".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/cublas64_13.dll"
    dll.parent.mkdir(parents=True)
    dll.touch()
    models = tmp_path / "models"
    models.mkdir()
    (models / "pack").mkdir()
    (models / "shard.gguf").touch()
    tokenizer = models / "tokenizer"
    tokenizer.mkdir()
    for f in ("vocab.json", "merges.txt", "token_type.json"):
        (tokenizer / f).touch()
    cfg = {
        "exe": "do-not-execute.exe",
        "args": ["--pack", "pack", "--native", "shard.gguf", "--max-context", "8192"],
        "tokenizer": "tokenizer",
    }
    config = models / "qwen.json"
    config.write_text(json.dumps(cfg))
    return root, server, config, cfg


def test_private_config_preserves_assets_and_assigns_distinct_aliases(prepared):
    root, server, config, _ = prepared
    original = config.read_bytes()
    adapter = strata.StrataAdapter()
    binary = DiscoveredBinary(server, Origin.configured)
    for name in ("small", "large"):
        spec = RuntimeSpec(name=name, engine="strata", modelPath=str(config), modelAlias=name)
        argv = adapter.build_argv(spec, binary, 8123)
        saved = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
        assert argv[argv.index("--engine") + 1] == "strata"
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        assert saved["model_name"] == name
        assert Path(saved["exe"]).parent == root / "engine"
        assert Path(saved["args"][1]) == config.parent / "pack"
        assert saved["parallel"] == 1
        assert saved["lib_dirs"] == [str(root / ".venv/Lib/site-packages/nvidia/cu13/bin/x86_64")]
    assert config.read_bytes() == original


def test_prepared_setup_tuning_and_metadata_are_accepted(prepared):
    root, _, config, cfg = prepared
    cfg.update(gpu=0, gpus_asked=True, draft_vocab="en")
    cfg["args"] += ["--pool-workers", "8", "--pcie-frac", "0.5"]
    config.write_text(json.dumps(cfg))
    saved = strata.prepared_config(config, alias="a", root=root)
    assert saved["gpu"] == 0
    assert saved["args"][-4:] == ["--pool-workers", "8", "--pcie-frac", "0.5"]
    assert "draft_vocab" not in saved and "gpus_asked" not in saved


@pytest.mark.parametrize(
    "change",
    [
        {"before_load": "whoami"},
        {"mcp_servers": {"x": {"command": "whoami"}}},
        {"parallel": 4},
        {"vision": {"exe": "x"}},
        {"args": ["--batch", "8"]},
        {"tokenizer": "missing"},
        {"env": {"PYTHONPATH": "untrusted"}},
    ],
)
def test_unsupported_or_missing_assets_refuse_before_spawn(prepared, change):
    root, _, config, cfg = prepared
    config.write_text(json.dumps({**cfg, **change}))
    with pytest.raises(SpawnPlanError):
        strata.prepared_config(config, alias="a", root=root)


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"service": "strata", "loaded": False}, Loading),
        ({"service": "strata", "loaded": True, "max_context": 4096}, Ready),
        ({"status": "ok"}, NotAnswering),
    ],
)
@pytest.mark.anyio
async def test_health_200_alone_is_not_ready(monkeypatch, body, expected):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    ) as client:
        monkeypatch.setattr(strata, "probe_client", lambda: client)
        result = await strata.StrataAdapter().probe_readiness("http://strata")
    assert isinstance(result, expected)
    if isinstance(result, Ready):
        assert result.capabilities.parallelSlots == 1
        assert result.capabilities.contextPoolTokens is None


def test_recipe_is_pinned_and_platform_gated():
    host = HostAccelerator(os="windows", arch="x64", accelerator="cuda", acceleratorVersion="13.3")
    selected = plan(host, None, None)
    assert isinstance(selected, AcquisitionPlan)
    assert all(a.sha256 for a in selected.assets)
    assert isinstance(plan(host.model_copy(update={"os": "macos"}), None, None), Unavailable)
    assert isinstance(plan(host, "latest", None), Unavailable)


def test_no_automatic_eviction_or_arbitrary_gguf():
    spec = RuntimeSpec(
        name="strata", engine="strata", modelPath="prepared.json", startOnDemand=True
    )
    assert "explicit start/stop" in validate_spec(spec)
    assert "Strata loads a prepared model" in validate_spec(
        spec.model_copy(update={"startOnDemand": False, "modelPath": "file.gguf"})
    )
    library = spec.model_copy(
        update={"startOnDemand": False, "modelPath": "/m/qwen.eugene-prepared.json"}
    )
    assert validate_spec(library) is None


# --- a prepared model from the Library (LS3) -----------------------------------


def _provenance(folder: Path, **body) -> Path:
    path = folder / "qwen-flash.eugene-prepared.json"
    path.write_text(json.dumps({"engine": "strata", **body}), encoding="utf-8")
    return path


def test_a_library_model_launches_its_entry_under_its_own_name(prepared, tmp_path):
    _, server, config, _ = prepared
    library = tmp_path / "library"
    library.mkdir()
    provenance = _provenance(library, entry=str(config))
    beside = _provenance(config.parent, entry=config.name)
    for path in (provenance, beside):
        spec = RuntimeSpec(name="q", engine="strata", modelPath=str(path))
        argv = strata.StrataAdapter().build_argv(
            spec, DiscoveredBinary(server, Origin.configured), 1
        )
        saved = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
        # The Library's name for it, not the provenance file's.
        assert saved["model_name"] == "qwen-flash"
        assert Path(saved["args"][1]) == config.parent / "pack"


@pytest.mark.parametrize(
    "body,said",
    [
        ({"engine": "kev", "entry": "qwen.json"}, "prepared for kev, not Strata"),
        ({"engine": "strata", "entry": "qwen.json", "formatVersion": 2}, "newer Eugene"),
        ({"engine": "strata"}, "not valid"),
        ({"engine": "strata", "entry": "elsewhere.json"}, "elsewhere.json"),
    ],
)
def test_a_provenance_file_that_cannot_launch_says_why(prepared, body, said):
    root, _, config, _ = prepared
    path = config.parent / "qwen-flash.eugene-prepared.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    spec = RuntimeSpec(name="q", engine="strata", modelPath=str(path))
    with pytest.raises(SpawnPlanError, match=said):
        strata.StrataAdapter().prepare_config(
            spec, DiscoveredBinary(root / "serve/server.py", Origin.configured)
        )


def test_strata_declares_the_prepared_models_it_loads():
    loads = [r for r in strata.StrataAdapter.accepts if r.format.value == "prepared"]
    assert len(loads) == 1 and loads[0].preparedFor is EngineKind.strata
    assert loads[0].preparation is None
    # Older consoles see only `prepared`, so never Strata for every GGUF (B6).
    assert [f.value for f in strata.StrataAdapter.model_formats] == ["prepared"]


def test_stratas_list_is_upstream_setups_nine_choices():
    """LS4: read off setup.py at the pinned commit. Nine choices, each a
    named first shard at a pinned revision, needing preparation."""
    models = strata_models.SUPPORTED_MODELS
    assert [m.id for m in models] == [
        "Q2_0",
        "IQ2_XS",
        "IQ3_XXS",
        "IQ3_S",
        "swift-IQ2_XS",
        "swift-IQ3_XXS",
        "coder-IQ1_M",
        "unsloth-UD-IQ4_XS",
        "unsloth-UD-Q4_K_XL",
    ]
    for m in models:
        assert m.format.value == "gguf" and m.architecture == "qwen4exp"
        assert m.source.repoId and m.source.file
        assert "-00001-of-0000" in m.source.file and m.source.file.endswith(".gguf")
        assert m.source.revision and len(m.source.revision) == 40
        assert m.preparation is not None and m.preparation.recipe == "strata-prepare"
        assert m.sizeBytes and m.sizeBytes > 50_000_000_000
        assert m.quantization and m.quantization in m.source.file
    assert [m.id for m in models if m.recommended] == ["IQ2_XS"]
    assert [m.id for m in models if m.experimental] == ["unsloth-UD-Q4_K_XL"]
    assert [m.id for m in models if m.license] == ["swift-IQ2_XS", "swift-IQ3_XXS"]


def test_strata_prepares_only_the_ggufs_on_its_list():
    """Upstream's setup accepts a GGUF by name: the requirement names the
    list's first shards, so another publisher's qwen4exp GGUF is not one."""
    gguf = next(r for r in strata.StrataAdapter.accepts if r.format.value == "gguf")
    assert gguf.files == [
        Path(m.source.file).name for m in strata_models.SUPPORTED_MODELS if m.source.file
    ]
    assert "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf" not in gguf.files
    assert gguf.preparation == strata_models.PREPARATION
    assert strata.StrataAdapter.supported_models == strata_models.SUPPORTED_MODELS


def test_admission_reads_the_entry_through_the_nodes_mapping(authed_client, prepared):
    _, server, config, cfg = prepared
    _provenance(config.parent, entry=config.name)
    authed_client.patch(
        "/v1/config",
        json={
            "strataServer": str(server),
            "pathMappings": [{"from": "/remote/models", "to": str(config.parent)}],
        },
    )
    model = "/remote/models/qwen-flash.eugene-prepared.json"
    request = {"name": "a", "engine": "strata", "modelPath": model}
    body = authed_client.post("/v1/runtimes/admission", json=request).json()
    assert body["decision"] == "admit", body
    assert body["fit"] == "unknown"
    cfg["args"] += ["--mtp", "missing-mtp-pack"]
    config.write_text(json.dumps(cfg))
    body = authed_client.post("/v1/runtimes/admission", json=request).json()
    assert body["decision"] == "refuse"
    assert "missing-mtp-pack" in body["reason"]


def test_uninstall_preserves_borrowed_models_and_ignores_unowned_dirs(tmp_path):
    store = ManagedStore(tmp_path / "owned", EngineKind.strata)
    build = store.build_dir("v1")
    build.mkdir(parents=True)
    binary = build / "server.py"
    binary.touch()
    store.write_metadata(build, version="v1", variant="test", binary=binary)
    unknown = store.directory / "borrowed"
    unknown.mkdir()
    (unknown / "model.gguf").touch()
    model = tmp_path / "model.gguf"
    model.touch()
    store.remove_builds()
    store.remove_builds()
    assert not build.exists()
    assert model.exists() and (unknown / "model.gguf").exists()
    with pytest.raises(AcquisitionError):
        store.build_dir("../escape")


@pytest.mark.anyio
async def test_cancel_waits_for_worker_before_cleanup_and_retry(tmp_path):
    began = threading.Event()
    ended = threading.Event()

    class SlowInstaller(EngineInstaller):
        def _install(self, plan, progress, staging):
            staging.mkdir(parents=True)
            began.set()
            while not progress.cancelled.is_set():
                time.sleep(0.005)
            time.sleep(0.025)
            (staging / "last-write").touch()
            ended.set()
            progress.check_cancelled()

    store = ManagedStore(tmp_path, EngineKind.strata)
    installer = SlowInstaller(store, EngineKind.strata)
    installer.start(AcquisitionPlan("v1", "test", (), "server.py"))
    await asyncio.to_thread(began.wait, 2)
    result = await installer.cancel()
    assert ended.is_set()
    assert result.state == "cancelled"
    assert not installer.running and not store.build_dir(".staging-v1").exists()


def test_uninstall_requires_operator(client):
    assert client.post("/v1/engines/strata/uninstall").status_code == 401


def test_prepared_config_size_is_never_a_memory_estimate(authed_client, prepared):
    _, server, config, _ = prepared
    authed_client.patch("/v1/config", json={"strataServer": str(server)})
    response = authed_client.post(
        "/v1/runtimes/admission",
        json={
            "name": "a",
            "engine": "strata",
            "modelPath": str(config),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "admit" and body["fit"] == "unknown"
    assert body.get("requiredBytes") is None
    assert "Config size is not model size" in body["warning"]
    assert not (strata.StrataAdapter().managed_store().directory / ".launch").exists()


def test_admission_checks_prepared_assets_before_disrupting_a_running_model(
    authed_client, prepared
):
    _, server, config, cfg = prepared
    authed_client.patch(
        "/v1/config",
        json={
            "strataServer": str(server),
            "pathMappings": [{"from": "/remote/models", "to": str(config.parent)}],
        },
    )
    cfg["args"] += ["--mtp", "missing-mtp-pack"]
    config.write_text(json.dumps(cfg))
    response = authed_client.post(
        "/v1/runtimes/admission",
        json={"name": "a", "engine": "strata", "modelPath": "/remote/models/qwen.json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "refuse"
    assert "prepared model asset is missing" in body["reason"]
    assert "missing-mtp-pack" in body["reason"]


def test_uninstall_refuses_a_running_runtime_then_keeps_its_declaration(
    authed_client,
    stub_runtime_supervisor,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_ENGINE_ROOT", str(tmp_path))
    store = ManagedStore(tmp_path, EngineKind.strata)
    build = store.build_dir("v1")
    build.mkdir(parents=True)
    binary = build / "server.py"
    binary.touch()
    store.write_metadata(build, version="v1", variant="test", binary=binary)
    state = authed_client.app.state.agent_state
    state.add_runtime(
        RuntimeSpec(name="a", engine="strata", modelPath="model.json", autoStart=False)
    )
    stub_runtime_supervisor.started.add("a")
    assert authed_client.post("/v1/engines/strata/uninstall").status_code == 409
    assert binary.exists()
    stub_runtime_supervisor.started.clear()
    assert authed_client.post("/v1/engines/strata/uninstall").status_code == 204
    assert state.get_runtime_spec("a") is not None and not build.exists()


def test_uninstall_can_finish_a_partial_removal(tmp_path):
    store = ManagedStore(tmp_path, EngineKind.strata)
    build = store.build_dir("v1")
    build.mkdir(parents=True)
    (build / "install.json").write_text(
        json.dumps({"binary": "missing-server.py", "version": "v1"})
    )
    (build / "leftover.dll").touch()
    store.remove_builds()
    assert not build.exists()


@pytest.mark.anyio
async def test_immediate_cancel_is_reported(tmp_path):
    installer = EngineInstaller(ManagedStore(tmp_path, EngineKind.strata), EngineKind.strata)
    installer.start(AcquisitionPlan("v1", "test", (), "server.py"))
    assert (await installer.cancel()).state == "cancelled"


def test_the_service_makes_stratas_venv_with_an_interpreter_that_can_run_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under the Windows service `sys.executable` is pythonservice.exe, which
    prints its usage for `-m venv` and exits: Strata's install failed on a
    service install (Amish_Station, 2026-10-09). The venv is made, for real,
    with the service venv's own python.exe."""
    import sys

    if sys.platform != "win32":
        pytest.skip("Windows service interpreter")
    from eugene_plexus_agent.engines import strata_install
    from eugene_plexus_agent.engines.acquisition import _Progress

    monkeypatch.setattr(sys, "executable", str(Path(sys.prefix) / "pythonservice.exe"))
    command = strata_install._venv_command(tmp_path / ".venv")
    assert Path(command[0]) == Path(sys.prefix) / "Scripts" / "python.exe"
    strata_install._run_command(command, _Progress(EngineKind.strata), tmp_path)
    assert (tmp_path / ".venv" / "Scripts" / "python.exe").is_file()


def test_a_service_without_its_interpreter_says_which_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from eugene_plexus_agent.engines import strata_install
    from eugene_plexus_agent.engines.acquisition import AcquisitionError

    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonservice.exe"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    with pytest.raises(AcquisitionError, match="interpreter is missing"):
        strata_install._venv_command(tmp_path / ".venv")


def test_a_failed_setup_command_is_named_with_its_exit_code(tmp_path: Path) -> None:
    import sys

    from eugene_plexus_agent.engines import strata_install
    from eugene_plexus_agent.engines.acquisition import AcquisitionError, _Progress

    command = [sys.executable, "-c", "print('some usage text'); raise SystemExit(3)"]
    with pytest.raises(AcquisitionError) as failed:
        strata_install._run_command(command, _Progress(EngineKind.strata), tmp_path)
    said = str(failed.value)
    assert f"`{Path(sys.executable).name} -c" in said and "exited 3" in said, said
    assert "some usage text" in said


def test_a_failed_setup_command_that_speaks_utf16_is_readable(tmp_path: Path) -> None:
    """pythonservice.exe writes its usage in UTF-16; read as UTF-8 it put a
    NUL (a box on screen) between every letter of the path (Troy, 2026-10-09)."""
    import sys

    from eugene_plexus_agent.engines import strata_install
    from eugene_plexus_agent.engines.acquisition import _Progress

    said_by = "import sys; sys.stdout.buffer.write('usage: -debug servicename'.encode('utf-16-le'))"
    command = [sys.executable, "-c", said_by + "; raise SystemExit(2)"]
    with pytest.raises(AcquisitionError) as failed:
        strata_install._run_command(command, _Progress(EngineKind.strata), tmp_path)
    said = str(failed.value)
    assert "usage: -debug servicename" in said and "\x00" not in said, repr(said)
    # UTF-8 output, the ordinary case, is unchanged.
    assert strata_install._readable("pip: ok — done".encode()) == "pip: ok — done"
