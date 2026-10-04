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
from eugene_plexus_agent.engines import strata
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
    assert "prepared Strata JSON" in validate_spec(
        spec.model_copy(update={"startOnDemand": False, "modelPath": "file.gguf"})
    )


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
