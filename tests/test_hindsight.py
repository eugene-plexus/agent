"""MV5b: a private catalogue link and explicit launcher settings; no real services."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import app_accounts, app_launcher, apps, logs
from eugene_plexus_agent._generated.models import AppOrigin, AppSettingsUpdate
from eugene_plexus_agent.dependencies import require_operator_session
from eugene_plexus_agent.routes import apps as routes

from .test_apps import _fake_python, _manager, _manifest, _record


def hindsight() -> Any:
    return _manifest(
        id="hindsight",
        name="Hindsight",
        source="pypi",
        version="1.0.0",
        package="hindsight-api-slim[embedded-db]",
        entry="hindsight_api.main:main",
        provides="memory-server",
        providesPath="/mcp/",
        ui=False,
        experimental=True,
        environment={
            "MODEL": "{memoryModel}",
            "MCP_TOKEN": "{linkSecret}",
            "DATABASE": "pg0://hindsight:{databaseSecret}@eugene-plexus-hindsight:{auxPort}",
        },
    )


def test_catalogue_has_slim_hindsight_and_linked_workbench() -> None:
    catalogue = {m.id: m for m in apps.load_catalogue()}
    hs = catalogue["hindsight"]
    assert apps.pip_requirement(hs) == "hindsight-api-slim[embedded-db]==1.0.0"
    assert hs.experimental and not hs.ui and not hs.localActions
    assert hs.environment["HINDSIGHT_API_DATABASE_URL"] == (
        "pg0://hindsight:{databaseSecret}@eugene-plexus-hindsight:{auxPort}"
    )
    assert hs.environment["HINDSIGHT_API_TENANT_API_KEY"] == "{appSecret}"
    assert hs.environment["HINDSIGHT_API_MCP_AUTH_TOKEN"] == "{linkSecret}"
    assert "memory-server" in catalogue["workbench"].uses


@pytest.mark.parametrize(
    "platform,kind,allowed",
    [
        ("linux", None, True),
        ("linux", "systemd", True),
        ("win32", None, False),
        ("win32", "windows_service", False),
        ("darwin", None, False),
    ],
)
def test_entry_availability_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, kind: Any, allowed: bool
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    manager.accounts = app_accounts.AccountSupport(kind)
    monkeypatch.setattr(apps.sys, "platform", platform)
    monkeypatch.setattr(manager, "uv", lambda: Path("uv"))
    entry = manager.as_catalogue(enrolled=True).apps[0]
    assert entry.installable is allowed
    assert bool(entry.reason) is not allowed
    if not allowed:
        assert "pg0" in entry.reason or "Linux" in entry.reason


@pytest.mark.parametrize(
    "claim",
    [
        {"provides": "memory-server", "providesPath": "/mcp/"},
        {"uses": ["memory-server"]},
        {"environment": {"TOKEN": "{linkSecret}"}},
    ],
)
def test_custom_link_claim_is_refused(tmp_path: Path, claim: dict[str, Any]) -> None:
    store = apps.AppStore(tmp_path / "apps.yaml")
    with pytest.raises(ValueError, match="catalogue"):
        store.add_custom(_manifest(**claim))


@pytest.mark.asyncio
async def test_install_waits_for_choice_and_reserves_aux_port(tmp_path: Path) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")
    assert record.settings.memoryModel is None
    assert not manager.supervisor.is_running("hindsight")
    assert manager.view(record).detail == "Choose the model Hindsight reads your chats with"
    assert record.aux_port != record.port
    assert record.aux_port in manager.store.taken_ports()
    assert manager.allocate_port() not in {record.port, record.aux_port}
    reopened = apps.AppStore(manager.store.path)
    reopened.load()
    assert reopened.get("hindsight").aux_port == record.aux_port
    assert reopened.get("hindsight").settings.memoryModel is None


@pytest.mark.asyncio
async def test_settings_stop_old_process_before_publish_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")
    _fake_python(manager.store, record)
    events = []

    async def pg0(record: Any, action: str) -> None:
        events.append((action, record.settings.memoryModel))

    monkeypatch.setattr(manager, "pg0", pg0)
    await manager.update_settings(record, AppSettingsUpdate(memoryModel="expert-untested"))
    assert manager.supervisor.is_running(record.id)
    assert record.settings.memoryModel == "expert-untested"
    original_stop = manager.supervisor.stop

    async def stop(ident: str) -> None:
        assert record.settings.memoryModel == "expert-untested"
        await original_stop(ident)

    monkeypatch.setattr(manager.supervisor, "stop", stop)
    await manager.update_settings(record, AppSettingsUpdate(memoryModel="another"))
    assert ("stop", "expert-untested") in events
    metadata = json.loads(manager.store.link_model_file(record.id).read_text())
    assert metadata == {"memoryModel": "another"}
    assert manager.view(record).settings.memoryModel == "another"


@pytest.mark.asyncio
async def test_settings_failure_preserves_effective_value_and_names_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")

    async def fail(ident: str) -> None:
        raise OSError("process 123 refused termination")

    monkeypatch.setattr(manager, "stop", fail)
    with pytest.raises(apps.AppInstallError, match="process 123 refused termination"):
        await manager.update_settings(record, AppSettingsUpdate(memoryModel="new"))
    assert record.settings.memoryModel is None
    assert "process 123" in manager.view(record).lastError


@pytest.mark.asyncio
async def test_shutdown_stop_and_purge_use_named_pg0_before_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    manager.supervisor = apps.AppSupervisor()
    events = []

    async def pg0(record: Any, action: str) -> None:
        assert manager.store.app_dir(record.id).exists()
        events.append(action)

    monkeypatch.setattr(manager, "pg0", pg0)
    await manager.aclose()
    assert "stop" in events
    await manager.uninstall("hindsight", purge=True)
    assert events[-2:] == ["stop", "drop"]
    assert not manager.store.app_dir("hindsight").exists()


def test_render_never_starts_with_missing_model_or_link_secret(tmp_path: Path) -> None:
    spec = {
        "argv": ["python"],
        "dataDir": str(tmp_path),
        "environment": {"MODEL": "{memoryModel}", "TOKEN": "{linkSecret}"},
    }
    with pytest.raises(ValueError, match="Choose the model"):
        app_launcher.render_start(spec, {})
    spec["values"] = {"memoryModel": "expert"}
    with pytest.raises(ValueError, match=r"link.*(missing|read)"):
        app_launcher.render_start(spec, {})


@pytest.mark.asyncio
async def test_only_catalogue_consumers_receive_link_and_secret_rotates(tmp_path: Path) -> None:
    consumer = _manifest(id="workbench", uses=["inference", "memory-server"])
    manager = _manager(tmp_path, catalogue=[hindsight(), consumer])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    hs = manager.store.get("hindsight")
    secret = manager.store.link_secret_file(hs.id).read_text()
    for origin, manifest, linked in [
        (AppOrigin.catalogue, consumer, True),
        (AppOrigin.custom, consumer, False),
        (AppOrigin.catalogue, _manifest(), False),
    ]:
        record = _record(manifest)
        record.origin = origin
        if origin is AppOrigin.custom:
            assert manager.memory_link(record) == {}
            continue
        manager.store.put(record)
        _fake_python(manager.store, record)
        await manager.start(record)
        env = manager.supervisor.planners[record.id].base_plan().env
        assert ("EUGENE_PLEXUS_APP_MEMORY_SERVER_URL" in env) is linked
        if linked:
            assert env["EUGENE_PLEXUS_APP_MEMORY_SERVER_URL"] == f"http://127.0.0.1:{hs.port}/mcp/"
            assert secret not in json.dumps(env)
    # An unset provider has no pg0 process or installed interpreter to clean up.
    await manager.uninstall(hs.id, purge=True)
    assert not manager.store.link_secret_file(hs.id).exists()
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    assert manager.store.link_secret_file(hs.id).read_text() != secret


def test_settings_routes_return_explicit_null_and_fail_with_effective_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    record = _record(hindsight())
    record.origin = AppOrigin.catalogue
    manager.store.put(record)
    app = FastAPI()
    app.state.apps = manager
    app.include_router(routes.router)
    app.dependency_overrides[require_operator_session] = lambda: None
    with TestClient(app) as client:
        assert client.get("/v1/apps/hindsight/settings").json() == {"memoryModel": None}
        assert client.get("/v1/apps/hindsight").json()["settings"] == {"memoryModel": None}
        assert (
            client.patch("/v1/apps/hindsight/settings", json={"linkSecret": "no"}).status_code
            == 422
        )
        assert (
            client.patch("/v1/apps/hindsight/settings", json={"memoryModel": ""}).status_code == 422
        )

        async def fail(ident: str) -> None:
            raise OSError("process 123 refused termination")

        monkeypatch.setattr(manager, "stop", fail)
        failed = client.patch("/v1/apps/hindsight/settings", json={"memoryModel": "new"})
        assert failed.status_code == 503
        assert "process 123" in failed.text
        assert client.get("/v1/apps/hindsight/settings").json() == {"memoryModel": None}


@pytest.mark.parametrize("action", ["stop", "drop"])
def test_pg0_command_is_named_and_drop_is_noninteractive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    calls = []

    def run(argv: list[str], **kw: Any) -> Any:
        calls.append((argv, kw))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(app_launcher.subprocess, "run", run)
    app_launcher.pg0_cleanup("app-python", action, {"HOME": str(tmp_path)})
    argv, kwargs = calls[0]
    assert argv[0] == "app-python" and argv[3:6] == [action, "--name", "eugene-plexus-hindsight"]
    assert kwargs["env"]["HOME"] == str(tmp_path)
    if action == "drop":
        assert "--force" in argv


def test_pg0_does_not_treat_an_unrelated_missing_file_as_an_absent_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_launcher.subprocess,
        "run",
        lambda *a, **kw: type(
            "Result",
            (),
            {"returncode": 1, "stdout": "", "stderr": "pg_ctl: library not found: libxml2.so"},
        )(),
    )
    with pytest.raises(ValueError, match=r"libxml2\.so"):
        app_launcher.pg0_cleanup("app-python", "stop", {})


def test_systemd_link_uses_credentials_and_no_other_app_receives_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials = tmp_path / "credentials"
    credentials.mkdir()
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(credentials))
    spec_file = tmp_path / "launch.json"
    spec_file.write_text(
        json.dumps({"memoryServer": {"url": "http://127.0.0.1:8192/mcp/"}, "linkProvider": True})
    )
    spec = app_launcher.load_spec(str(spec_file))
    env = app_launcher.child_environment(spec)
    assert env["EUGENE_PLEXUS_APP_MEMORY_SERVER_TOKEN_FILE"] == str(
        credentials / "memory_server_token"
    )
    assert env["EUGENE_PLEXUS_APP_MEMORY_SERVER_MODEL_FILE"] == str(
        tmp_path / "memory_server_model.json"
    )
    assert spec["linkSecretFile"] == str(credentials / "memory_server_token")
    assert "EUGENE_PLEXUS_APP_MEMORY_SERVER_TOKEN_FILE" not in app_launcher.child_environment({})


@pytest.mark.asyncio
async def test_stopped_settings_stay_stopped_and_failed_save_keeps_old_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")
    record.enabled = False
    await manager.update_settings(record, AppSettingsUpdate(memoryModel="expert"))
    assert record.settings.memoryModel == "expert"
    assert not manager.supervisor.is_running(record.id)
    await manager.update_settings(record, AppSettingsUpdate())
    assert record.settings.memoryModel == "expert"

    def fail() -> None:
        raise OSError("disk full")

    monkeypatch.setattr(manager.store, "_write", fail)
    with pytest.raises(apps.AppInstallError, match="disk full"):
        await manager.update_settings(record, AppSettingsUpdate(memoryModel="new"))
    assert record.settings.memoryModel == "expert"


@pytest.mark.asyncio
async def test_aux_port_is_reserved_for_any_manifest_that_uses_it(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest = _manifest(environment={"SECOND_PORT": "{auxPort}"})
    await manager.installed_callback(manifest, AppOrigin.custom)
    record = manager.store.get(manifest.id)
    assert record.aux_port is not None and record.aux_port != record.port


@pytest.mark.asyncio
async def test_failed_process_stop_keeps_the_supervisor_observation() -> None:
    supervisor = apps.AppSupervisor()

    class Process:
        async def stop(self) -> None:
            raise OSError("process 42 still alive")

    process = Process()
    supervisor._processes["hindsight"] = process
    with pytest.raises(OSError, match="process 42"):
        await supervisor.stop("hindsight")
    assert supervisor.is_running("hindsight")


@pytest.mark.asyncio
async def test_every_automatic_spawn_cleans_pg0_before_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")
    record.settings.memoryModel = "expert"
    _fake_python(manager.store, record)
    planner = apps._AppPlanner(
        record, store=manager.store, gateway_url=lambda: None, bind_host=lambda: "0.0.0.0"
    )
    events = []
    monkeypatch.setattr(app_launcher, "pg0_cleanup", lambda *a: events.append("stop"))

    def render(spec: Any, env: Any) -> Any:
        events.append("render")
        return spec["argv"], env, []

    monkeypatch.setattr(app_launcher, "render_start", render)
    await planner.plan_async()
    await planner.plan_async()
    assert events == ["stop", "render", "stop", "render"]


@pytest.mark.asyncio
async def test_uninstall_withdraws_the_consumers_credential_before_restart(tmp_path: Path) -> None:
    consumer = _manifest(id="workbench", uses=["inference", "memory-server"])
    manager = _manager(tmp_path, catalogue=[hindsight(), consumer])
    manager.accounts = app_accounts.AccountSupport("systemd")
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = _record(consumer)
    record.origin = AppOrigin.catalogue
    manager.store.put(record)
    _fake_python(manager.store, record)
    await manager.start(record)
    assert manager.store.link_secret_file(record.id).read_text()
    seen = []
    original = manager.supervisor.start

    def start(planner: Any) -> None:
        seen.append(manager.store.link_secret_file(record.id).read_text())
        original(planner)

    manager.supervisor.start = start
    await manager.uninstall("hindsight", purge=False)
    assert seen == [""]
    assert manager.store.link_secret_file("hindsight").exists()


def test_provider_does_not_fall_back_to_downloaded_embedding_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = {
        "argv": ["python"],
        "dataDir": str(tmp_path),
        "linkProvider": True,
        "values": {"gatewayUrl": "http://127.0.0.1:8490", "memoryModel": "chosen"},
        "environment": {"EMBEDDINGS": "{embeddingModel}", "RERANKER": "{rerankModel}"},
    }
    monkeypatch.setattr(app_launcher, "ask_gateway_for_models", lambda *a: ([], None))
    with pytest.raises(ValueError, match="no local search model"):
        app_launcher.render_start(spec, {})


@pytest.mark.asyncio
async def test_metadata_write_failure_restores_effective_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")
    publish = manager.publish_memory_model

    def fail_once(record: Any) -> None:
        if record.settings.memoryModel == "new":
            raise OSError("model metadata disk is full")
        publish(record)

    monkeypatch.setattr(manager, "publish_memory_model", fail_once)
    with pytest.raises(apps.AppInstallError, match="model metadata disk is full"):
        await manager.update_settings(record, AppSettingsUpdate(memoryModel="new"))
    assert record.settings.memoryModel is None
    reopened = apps.AppStore(manager.store.path)
    reopened.load()
    assert reopened.get(record.id).settings.memoryModel is None
    assert json.loads(manager.store.link_model_file(record.id).read_text()) == {"memoryModel": None}


@pytest.mark.asyncio
async def test_uninstall_waits_for_a_settings_change_before_deleting_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")
    entered, release = asyncio.Event(), asyncio.Event()

    async def stop(ident: str) -> None:
        if not entered.is_set():
            entered.set()
            await release.wait()

    monkeypatch.setattr(manager, "stop", stop)
    changing = asyncio.create_task(
        manager.update_settings(record, AppSettingsUpdate(memoryModel="new"))
    )
    await entered.wait()
    removing = asyncio.create_task(manager.uninstall(record.id, purge=True))
    await asyncio.sleep(0)
    try:
        assert manager.store.get(record.id) is record
    finally:
        release.set()
        await changing
        await removing
    assert manager.store.get(record.id) is None
    assert not manager.store.app_dir(record.id).exists()


@pytest.mark.asyncio
async def test_failed_purge_names_the_filesystem_cause_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")

    def fail(path: Path) -> None:
        raise PermissionError("memory folder is locked")

    monkeypatch.setattr(apps.shutil, "rmtree", fail)
    with pytest.raises(PermissionError, match="memory folder is locked"):
        await manager.uninstall(record.id, purge=True)
    assert manager.store.get(record.id) is record
    assert not record.enabled


@pytest.mark.asyncio
async def test_systemd_model_metadata_changes_before_the_provider_can_restart(
    tmp_path: Path,
) -> None:
    consumer = _manifest(id="workbench", uses=["inference", "memory-server"])
    manager = _manager(tmp_path, catalogue=[hindsight(), consumer])
    manager.accounts = app_accounts.AccountSupport("systemd")
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = _record(consumer)
    record.origin = AppOrigin.catalogue
    manager.store.put(record)
    _fake_python(manager.store, record)
    await manager.start(record)
    planner = manager.supervisor.planners[record.id]
    spawn = planner.base_plan()
    plan = app_accounts.LaunchPlan(
        app_id=record.id,
        argv=spawn.argv,
        env=spawn.env,
        data_dir=manager.store.data_dir(record.id),
        key_file=manager.store.key_file(record.id),
        admin_token="fake",
        port=record.port,
        bind_host=None,
        launch=planner.launch(),
    )
    spec = app_accounts.launch_spec(plan, kind="systemd", ingress="")
    assert spec["memoryServer"]["modelFile"] == str(manager.store.link_model_file(record.id))
    provider = manager.store.get("hindsight")
    provider.settings.memoryModel = "new-outside-model"
    manager.publish_memory_model(provider)
    assert json.loads(manager.store.link_model_file(record.id).read_text()) == {
        "memoryModel": "new-outside-model"
    }


def test_database_password_is_private_kept_and_never_taken_from_spec(tmp_path: Path) -> None:
    spec = {
        "argv": ["python"],
        "dataDir": str(tmp_path),
        "values": {"auxPort": "8499", "databaseSecret": "forged"},
        "environment": {"DATABASE": hindsight().environment["DATABASE"]},
    }
    _, first, notes = app_launcher.render_start(spec, {})
    _, second, _ = app_launcher.render_start(spec, {})
    password = (tmp_path / "database_secret").read_text()
    assert len(password) >= 32 and password != "forged"
    assert first == second
    assert first["DATABASE"] == f"pg0://hindsight:{password}@eugene-plexus-hindsight:8499"
    assert password not in json.dumps(spec) + json.dumps(notes)
    assert app_launcher.app_secret(str(tmp_path)) != password
    assert "databaseSecret" in app_launcher.SECRET_PLACEHOLDERS
    with pytest.raises(ValueError, match=r"database.*data folder"):
        app_launcher.render_start({**spec, "dataDir": ""}, {})


@pytest.mark.asyncio
async def test_systemd_provider_start_preserves_credentials_and_repairs_empty_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    manager.accounts = app_accounts.AccountSupport("systemd")
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")
    secret = manager.store.link_secret_file(record.id)
    secret.write_text("")
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    token = secret.read_text()
    assert token
    record = manager.store.get("hindsight")
    _fake_python(manager.store, record)
    record.settings.memoryModel = "chosen"
    manager.publish_memory_model(record)
    monkeypatch.setattr(manager, "pg0", lambda *a: asyncio.sleep(0))
    await manager.start(record)
    assert secret.read_text() == token
    assert json.loads(manager.store.link_model_file(record.id).read_text()) == {
        "memoryModel": "chosen"
    }
    planner = manager.supervisor.planners[record.id]
    _, env, _ = app_launcher.render_start(planner._render_spec(planner.base_plan()), {})
    assert env["MCP_TOKEN"] == token


@pytest.mark.asyncio
async def test_start_repairs_empty_provider_token_after_agent_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    manager.accounts = app_accounts.AccountSupport("systemd")
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")
    record.settings.memoryModel = "chosen"
    _fake_python(manager.store, record)
    secret = manager.store.link_secret_file(record.id)
    secret.write_text("")
    monkeypatch.setattr(manager, "pg0", lambda *a: asyncio.sleep(0))
    await manager.start(record)
    assert secret.read_text(), "An already installed provider must repair its empty token"


@pytest.mark.asyncio
async def test_own_account_start_failure_redacts_log_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runner = app_accounts.SystemdRunner(tmp_path / "apps")
    supervisor = app_accounts.OwnAccountSupervisor(
        runner,
        ingress=lambda: "",
        data_dir=lambda ident: tmp_path / ident,
        key_file=lambda ident: tmp_path / ident / "key",
    )
    manager = _manager(tmp_path)
    record = _record(hindsight())
    _fake_python(manager.store, record)
    planner = apps._AppPlanner(
        record, store=manager.store, gateway_url=lambda: None, bind_host=lambda: None
    )

    def state(ident: str) -> Any:
        raise OSError("database failed: postgresql://hindsight:must-hide@127.0.0.1:8499/hindsight")

    monkeypatch.setattr(runner, "state", state)
    with caplog.at_level(logging.ERROR):
        await supervisor._start(planner)
    assert "must-hide" not in caplog.text
    assert "must-hide" not in supervisor._errors[record.id]


@pytest.mark.parametrize("action", ["start", "restart"])
@pytest.mark.asyncio
async def test_start_routes_wait_for_model_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    from starlette.requests import Request

    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    record = manager.store.get("hindsight")
    entered, release = asyncio.Event(), asyncio.Event()
    observed = []
    stops = []

    async def stop(ident: str) -> None:
        stops.append(ident)
        entered.set()
        await release.wait()

    async def start(record: Any) -> None:
        observed.append(record.settings.memoryModel)

    monkeypatch.setattr(manager, "stop", stop)
    monkeypatch.setattr(manager, "start", start)
    monkeypatch.setattr(routes, "_ensure_sign_in", lambda *a: asyncio.sleep(0))
    app = FastAPI()
    app.state.apps = manager
    request = Request({"type": "http", "app": app})
    changing = asyncio.create_task(
        manager.update_settings(record, AppSettingsUpdate(memoryModel="B"))
    )
    await entered.wait()
    starting = asyncio.create_task(getattr(routes, action)(request, record.id))
    await asyncio.sleep(0.03)
    try:
        assert observed == [], "A route launched the old model while PATCH held the lock"
        assert stops == ["hindsight"], "Restart stopped the app during the model change"
    finally:
        release.set()
        await changing
        await starting
    assert all(model == "B" for model in observed)


@pytest.mark.asyncio
async def test_shutdown_keeps_own_account_apps_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    manager.accounts = app_accounts.AccountSupport("systemd")
    stops = []
    monkeypatch.setattr(manager, "stop", lambda ident: stops.append(ident))
    await manager.aclose()
    assert stops == []


@pytest.mark.asyncio
async def test_shutdown_continues_after_each_pg0_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manager = _manager(tmp_path, catalogue=[hindsight()])
    await manager.installed_callback(hindsight(), AppOrigin.catalogue)
    second = _record(hindsight())
    second.manifest = second.manifest.model_copy(update={"id": "second"})
    manager.store.put(second)
    monkeypatch.setattr(apps, "memory_provider", lambda record: True)
    manager.supervisor = apps.AppSupervisor()
    stopped = []

    async def fail(ident: str) -> None:
        stopped.append(ident)
        raise OSError("pg0 permission denied")

    monkeypatch.setattr(manager, "stop", fail)
    with caplog.at_level(logging.WARNING):
        await manager.aclose()
    assert stopped == ["hindsight", "second"]
    assert "pg0 permission denied" in caplog.text


@pytest.mark.parametrize("scheme", ["pg0", "postgres", "postgresql"])
def test_database_urls_redacted_in_forwarding_logs_and_cleanup_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], scheme: str
) -> None:
    raw = f"failed: {scheme}://hindsight:secret%21@127.0.0.1:8499/hindsight"
    assert "secret%21" not in logs.redact(raw)
    forwarder = app_launcher.Forwarder(None, None, "hindsight")
    monkeypatch.setattr(app_launcher, "WINDOWS", False)
    forwarder.add(raw)
    assert "secret%21" not in capsys.readouterr().out
    assert "secret%21" not in "".join(forwarder._take())
    monkeypatch.setattr(
        app_launcher.subprocess,
        "run",
        lambda *a, **kw: type("Result", (), {"returncode": 1, "stdout": raw, "stderr": ""})(),
    )
    with pytest.raises(ValueError) as error:
        app_launcher.pg0_cleanup("python", "stop", {})
    assert "secret%21" not in str(error.value)


def test_provider_refuses_an_occupied_database_port(tmp_path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(ValueError, match=rf"database port {port}.*in use"):
            app_launcher.render_start(
                {
                    "argv": [sys.executable],
                    "linkProvider": True,
                    "dataDir": str(tmp_path),
                    "values": {"auxPort": str(port)},
                },
                {},
            )


def test_provider_replaces_existing_trust_rules_before_spawn(tmp_path: Path) -> None:
    root = tmp_path / ".pg0"
    hba = root / "instances/eugene-plexus-hindsight/data/pg_hba.conf"
    hba.parent.mkdir(parents=True)
    hba.write_text("local all all trust\nhost all all 127.0.0.1/32 trust\n")
    app_launcher.prepare_database(str(tmp_path), None)
    assert hba.read_text().splitlines() == [
        "local all all password",
        "host all all 127.0.0.1/32 password",
        "host all all ::1/128 password",
    ]


@pytest.mark.skipif(
    sys.platform == "win32", reason="PostgreSQL's POSIX port reuse; Windows Hindsight is refused"
)
def test_provider_can_restart_with_closed_connections_in_time_wait(tmp_path: Path) -> None:
    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        client.connect(("127.0.0.1", port))
        connection, _ = listener.accept()
        connection.close()  # The server actively closes and owns TIME_WAIT.
        assert client.recv(1) == b""
    app_launcher.prepare_database(str(tmp_path), str(port))


def test_provider_restores_the_callers_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_launcher, "WINDOWS", False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(app_launcher, "pg0_cleanup", lambda *a: None)
    current = [0o022]

    def umask(mask: int) -> int:
        previous, current[0] = current[0], mask
        return previous

    monkeypatch.setattr(app_launcher.os, "umask", umask)
    assert app_launcher.watch_memory_provider([sys.executable, "-c", "pass"]) == 0
    assert current == [0o022]


@pytest.mark.asyncio
async def test_supervisor_and_file_logs_remove_database_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from eugene_plexus_agent.supervisor import SupervisedProcess

    raw = "postgresql://hindsight:password-to-hide@127.0.0.1:8499/hindsight\n"
    planner = apps._AppPlanner(
        _record(hindsight()),
        store=apps.AppStore(tmp_path / "apps.yaml"),
        gateway_url=lambda: None,
        bind_host=lambda: None,
    )
    process = SupervisedProcess(planner, logging.getLogger("test"))
    reader = asyncio.StreamReader()
    reader.feed_data(raw.encode())
    reader.feed_eof()
    await process._pipe_child_output(reader)
    assert "password-to-hide" not in capsys.readouterr().out
    assert "password-to-hide" not in "".join(process._output_tail)
    assert "password-to-hide" not in logs.stamp(raw)


def test_app_reports_redact_database_credentials(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    record = _record(hindsight())
    manager.store.put(record)
    manager._errors[record.id] = (
        "failed: pg0://hindsight:password-to-hide@eugene-plexus-hindsight:8499"
    )
    assert "password-to-hide" not in manager.view(record).model_dump_json()


def test_provider_port_move_stops_child_and_reports_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(app_launcher, "pg0_cleanup", lambda *a: calls.append(a))
    command = [
        sys.executable,
        "-u",
        "-c",
        (
            "import time;print('postgresql://hindsight:secret@127.0.0.1:8499/hindsight');"
            "print('Port 8499 is in use, using port 8500 instead.');time.sleep(30)"
        ),
    ]
    assert app_launcher.watch_memory_provider(command) != 0
    output = capsys.readouterr().out
    assert "8499" in output and "in use" in output and "not started" in output
    assert "secret@" not in output
    assert calls and calls[-1][1] == "stop"


def test_provider_detects_a_silent_port_move_in_pg0_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        "HINDSIGHT_API_DATABASE_URL", "pg0://hindsight:secret@eugene-plexus-hindsight:8499"
    )
    instance = tmp_path / ".pg0/instances/eugene-plexus-hindsight/instance.json"
    instance.parent.mkdir(parents=True)
    instance.write_text(json.dumps({"port": 8500}))
    (tmp_path / "database_secret").write_text("a-raw-password")
    monkeypatch.setattr(app_launcher, "pg0_cleanup", lambda *a: None)
    code = app_launcher.watch_memory_provider(
        [sys.executable, "-u", "-c", "import time;print('Password: a-raw-password');time.sleep(30)"]
    )
    output = capsys.readouterr().out
    assert code != 0 and "pg0 selected port 8500" in output
    assert "a-raw-password" not in output


@pytest.mark.asyncio
async def test_catalogue_install_pins_pg0_in_the_same_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    async def tool(argv: list[str], **kw: Any) -> tuple[int, str]:
        calls.append(argv)
        return 0, ""

    monkeypatch.setattr(apps, "_run_tool", tool)
    manifest = next(m for m in apps.load_catalogue() if m.id == "hindsight")
    await apps.AppInstaller()._install(
        manifest,
        apps._Progress(app=manifest.id, version=manifest.version),
        target=tmp_path / "version",
        uv=Path("uv"),
    )
    install = next(argv for argv in calls if "install" in argv)
    assert "pg0-embedded==0.15.2" in install
