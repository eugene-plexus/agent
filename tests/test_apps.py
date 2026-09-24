"""Optional apps (specs docs/design/apps-and-spokes.md).

The checks here are about the properties the design rests on, each one a
thing that would be true of a plausible wrong implementation:

* an app is handed **no hub credential** -- a planner that reused
  `child_environment(component_prefix=...)` would pass one through;
* an unreadable `apps.yaml` costs **the apps and nothing else**;
* the key is minted with **the caller's** credential, and an uninstall
  whose revocation fails **removes nothing**;
* an app's settings are reached with **its admin token, never the
  operator's bearer**.

The last test builds a real environment with `uv` from a stdlib-only
package, starts it and stops it, because "the environment is committed
by its install.json" and "the verify step refuses a package python -m
cannot run" are claims about files and processes, not about calls.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from eugene_plexus_agent import _http, apps
from eugene_plexus_agent._generated.models import (
    AppManifest,
    AppOrigin,
    ClientKey,
    ClientKeyCreated,
    ComponentStatus,
)
from eugene_plexus_agent.routes import apps as apps_routes

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_app"


def _manifest(**overrides: Any) -> AppManifest:
    base: dict[str, Any] = {
        "id": "tiny",
        "name": "Tiny",
        "source": str(FIXTURE.resolve()),
        "version": "v1",
        "package": "tiny-app",
        "entry": "tiny_app",
    }
    base.update(overrides)
    return apps.normalized(AppManifest.model_validate(base))


def _record(manifest: AppManifest | None = None, *, port: int = 8190) -> apps.InstalledApp:
    return apps.InstalledApp(
        manifest=manifest or _manifest(),
        origin=AppOrigin.custom,
        port=port,
        installed_at=datetime.now(UTC),
    )


def _fake_python(store: apps.AppStore, record: apps.InstalledApp) -> Path:
    python = apps.venv_python(store.version_dir(record.id, record.version) / "venv")
    python.parent.mkdir(parents=True)
    python.write_text("")
    return python


class FakeAppSupervisor:
    """Records starts and stops; spawns nothing."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.planners: dict[str, Any] = {}
        self.token = "admin-token-from-this-spawn"

    def start(self, planner: Any) -> None:
        self.started.append(planner.record.id)
        self.planners[planner.record.id] = planner

    async def stop(self, app_id: str) -> None:
        self.planners.pop(app_id, None)

    def is_running(self, app_id: str) -> bool:
        return app_id in self.planners

    def admin_token(self, app_id: str) -> str | None:
        return self.token if app_id in self.planners else None

    def status(self, app_id: str) -> tuple[Any, ...]:
        return ComponentStatus.running, None, None, None, None

    def bind_host(self, app_id: str) -> str | None:
        return "0.0.0.0" if app_id in self.planners else None

    async def stop_all(self) -> None:
        self.planners.clear()


def _manager(tmp_path: Path, **kw: Any) -> apps.AppManager:
    async def gateway() -> tuple[str | None, str | None]:
        return "http://127.0.0.1:8080", None

    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    manager = apps.AppManager(
        store=store,
        catalogue=kw.pop("catalogue", []),
        get_config=kw.pop("get_config", lambda key: None),
        bind_host=lambda: None,
        advertise_host=lambda: None,
        node_name=lambda: "node-a",
        resolve_gateway=gateway,
    )
    manager.supervisor = FakeAppSupervisor()  # type: ignore[assignment]
    return manager


# --------------------------------------------------------------------------- #
# what an app is handed
# --------------------------------------------------------------------------- #


def test_an_app_is_handed_no_hub_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Everything a component could be handed, present in the agent's own
    # environment the way a supervisor's children see it.
    for name in (
        "EUGENE_PLEXUS_GATEWAY_SERVICE_TOKEN",
        "EUGENE_PLEXUS_DRIVER_MASTER_KEY",
        "EUGENE_PLEXUS_LIBRARY_AUTH_VERIFY_KEY",
        "EUGENE_PLEXUS_AGENT_CONFIG_FILE",
        "EUGENE_PLEXUS_APP_ADMIN_TOKEN",
    ):
        monkeypatch.setenv(name, "leaked")
    monkeypatch.setenv("SOME_HOST_SETTING", "kept")

    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    record = _record()
    python = _fake_python(store, record)
    planner = apps._AppPlanner(
        record,
        store=store,
        gateway_url=lambda: "http://10.0.0.5:8079/api/proxy/gateway",
        bind_host=lambda: "0.0.0.0",
    )
    plan = planner.plan()

    ours = {k for k in plan.env if k.startswith("EUGENE_PLEXUS_")}
    assert ours == {
        "EUGENE_PLEXUS_APP_ID",
        "EUGENE_PLEXUS_APP_BIND_PORT",
        "EUGENE_PLEXUS_APP_BIND_HOST",
        "EUGENE_PLEXUS_APP_DATA_DIR",
        "EUGENE_PLEXUS_APP_KEY_FILE",
        "EUGENE_PLEXUS_APP_ADMIN_TOKEN",
        "EUGENE_PLEXUS_APP_GATEWAY_URL",
    }
    assert "leaked" not in plan.env.values()
    assert plan.env["SOME_HOST_SETTING"] == "kept"
    assert plan.env["EUGENE_PLEXUS_APP_GATEWAY_URL"] == "http://10.0.0.5:8079/api/proxy/gateway"
    assert plan.argv == [str(python), "-m", "tiny_app"]

    # A fresh admin token per spawn, and the planner is where the agent
    # reads the current one.
    first = planner.admin_token
    assert first and plan.env["EUGENE_PLEXUS_APP_ADMIN_TOKEN"] == first
    assert planner.plan().env["EUGENE_PLEXUS_APP_ADMIN_TOKEN"] != first


def test_no_gateway_means_no_gateway_variable_rather_than_a_guess(tmp_path: Path) -> None:
    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    record = _record()
    _fake_python(store, record)
    plan = apps._AppPlanner(
        record, store=store, gateway_url=lambda: None, bind_host=lambda: None
    ).plan()
    assert "EUGENE_PLEXUS_APP_GATEWAY_URL" not in plan.env
    assert "EUGENE_PLEXUS_APP_BIND_HOST" not in plan.env


def test_a_missing_environment_is_refused_with_the_fix(tmp_path: Path) -> None:
    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    planner = apps._AppPlanner(
        _record(), store=store, gateway_url=lambda: None, bind_host=lambda: None
    )
    with pytest.raises(apps.SpawnPlanError, match="install the app again"):
        planner.plan()


# --------------------------------------------------------------------------- #
# apps.yaml degrades alone
# --------------------------------------------------------------------------- #


def test_an_unreadable_apps_file_costs_the_apps_and_nothing_else(
    app: FastAPI, tmp_path: Path
) -> None:
    (tmp_path / "agent.yaml").write_text(
        yaml.safe_dump(
            {
                "components": [
                    {
                        "name": "gateway",
                        "kind": "gateway",
                        "url": "http://127.0.0.1:8080",
                        "spawn": {"configFile": str(tmp_path / "gateway.yaml")},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / apps.APPS_FILE).write_text("installed: [ {manifest: 3}", encoding="utf-8")

    with TestClient(app) as client:
        token = client.post("/v1/auth/initialize", json={"passphrase": "correct horse battery"})
        client.headers["Authorization"] = f"Bearer {token.json()['sessionToken']}"

        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert "appsError" in health["details"]
        assert (tmp_path / (apps.APPS_FILE + apps.UNREADABLE_SUFFIX)).is_file()

        # The topology is untouched: this is the whole point of a file of
        # its own.
        names = [c["name"] for c in client.get("/v1/components").json()["components"]]
        assert names == ["gateway"]
        assert client.get("/v1/apps").json() == {"apps": []}


# --------------------------------------------------------------------------- #
# the catalogue
# --------------------------------------------------------------------------- #


def test_the_shipped_catalogue_parses() -> None:
    raw = yaml.safe_load(
        (Path(apps.__file__).parent / apps.CATALOGUE_RESOURCE).read_text(encoding="utf-8")
    )
    assert isinstance(raw, list)
    assert apps.load_catalogue() == [apps.normalized(AppManifest.model_validate(i)) for i in raw]


def test_the_catalogue_says_why_it_cannot_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, catalogue=[_manifest()])
    monkeypatch.setattr(apps, "find_uv", lambda configured=None: None)
    catalogue = manager.as_catalogue(enrolled=True)
    assert not catalogue.installable and "cannot find uv" in (catalogue.reason or "")

    monkeypatch.setattr(apps, "find_uv", lambda configured=None: Path("uv"))
    catalogue = manager.as_catalogue(enrolled=False)
    assert not catalogue.installable
    assert "key that changes every time it starts" in (catalogue.reason or "")

    catalogue = manager.as_catalogue(enrolled=True)
    assert catalogue.installable and catalogue.reason is None
    assert [e.manifest.id for e in catalogue.apps] == ["tiny"]


def test_custom_apps_are_off_until_someone_turns_them_on(
    authed_client: TestClient, tmp_path: Path
) -> None:
    body = _manifest().model_dump(mode="json", exclude_none=True)
    refused = authed_client.post("/v1/app-catalogue/custom", json=body)
    assert refused.status_code == 403
    assert "Allow apps not in the catalogue" in refused.json()["detail"]["detail"]

    patched = authed_client.patch("/v1/config", json={"allowCustomApps": True})
    assert patched.status_code == 200, patched.text
    added = authed_client.post("/v1/app-catalogue/custom", json=body)
    assert added.status_code == 201, added.text
    assert added.json()["origin"] == "custom"

    assert authed_client.post("/v1/app-catalogue/custom", json=body).status_code == 409
    relative = dict(body, id="relative", source="some/checkout")
    assert authed_client.post("/v1/app-catalogue/custom", json=relative).status_code == 422
    missing = dict(body, id="missing", source=str(tmp_path / "not-here"))
    assert authed_client.post("/v1/app-catalogue/custom", json=missing).status_code == 422


# --------------------------------------------------------------------------- #
# the key
# --------------------------------------------------------------------------- #


def _created(key_id: str = "k1", name: str = "app:tiny@node-a") -> ClientKeyCreated:
    return ClientKeyCreated(
        key=ClientKey.model_validate(
            {
                "id": key_id,
                "name": name,
                "tail": "abcd",
                "createdAt": "2026-09-23T00:00:00Z",
                "expiresAt": "2027-09-23T00:00:00Z",
            }
        ),
        token="the-token-shown-once",
    )


def test_install_mints_the_key_with_the_callers_credential_once(
    app: FastAPI, authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager: apps.AppManager = app.state.apps
    manager.catalogue = {"tiny": _manifest()}
    monkeypatch.setattr(apps_routes, "_enrolled", lambda request: True)
    monkeypatch.setattr(apps, "find_uv", lambda configured=None: Path("uv"))
    minted: list[str | None] = []

    async def mint(request: Any, body: Any, *, authorization: str | None) -> ClientKeyCreated:
        minted.append(authorization)
        return _created(name=body.name)

    started: list[str] = []

    def start(manifest: AppManifest, **_kw: Any) -> Any:
        started.append(manifest.id)
        return apps._Progress(app=manifest.id, version=manifest.version).snapshot()

    monkeypatch.setattr(apps_routes, "mint_client_key", mint)
    monkeypatch.setattr(manager.installer, "start", start)

    first = authed_client.post("/v1/apps/tiny/install")
    assert first.status_code == 202, first.text
    assert minted == [authed_client.headers["Authorization"]]
    assert manager.store.key_file("tiny").read_text() == "the-token-shown-once"
    key = manager.store.key("tiny")
    # Named for the app and the machine, so the install's key list says
    # what it is; this test node is not enrolled, so it has no name yet.
    assert key is not None and key.key_name == "app:tiny@this-node"

    # A retry after a failed install reuses the key rather than minting a
    # second one nobody would ever revoke.
    assert authed_client.post("/v1/apps/tiny/install").status_code == 202
    assert len(minted) == 1
    assert started == ["tiny", "tiny"]


def test_an_install_needs_an_enrolled_node(
    app: FastAPI, authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    app.state.apps.catalogue = {"tiny": _manifest()}
    monkeypatch.setattr(apps, "find_uv", lambda configured=None: Path("uv"))
    refused = authed_client.post("/v1/apps/tiny/install")
    assert refused.status_code == 422
    assert "first-run setup" in refused.json()["detail"]["detail"]


def test_an_uninstall_whose_revocation_fails_removes_nothing(
    app: FastAPI, authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager: apps.AppManager = app.state.apps
    manager.supervisor = FakeAppSupervisor()  # type: ignore[assignment]
    manager.store.put(_record())
    manager.store.put_key("tiny", apps.AppKey(key_id="k1", key_name="app:tiny@node-a"))
    outcome: dict[str, Any] = {"raise": HTTPException(503, detail="root down")}
    revoked: list[str] = []

    async def revoke(request: Any, key_id: str, *, authorization: str | None) -> None:
        if outcome["raise"] is not None:
            raise outcome["raise"]
        revoked.append(key_id)

    monkeypatch.setattr(apps_routes, "revoke_client_key_at_authority", revoke)

    assert authed_client.delete("/v1/apps/tiny").status_code == 503
    assert manager.store.get("tiny") is not None
    assert manager.store.key("tiny") is not None

    # A key the registry has never heard of is as revoked as it gets.
    outcome["raise"] = HTTPException(404, detail="no such key")
    assert authed_client.delete("/v1/apps/tiny").status_code == 204
    assert manager.store.get("tiny") is None and manager.store.key("tiny") is None
    assert authed_client.delete("/v1/apps/tiny").status_code == 404

    manager.store.put(_record())
    manager.store.put_key("tiny", apps.AppKey(key_id="k2", key_name="app:tiny@node-a"))
    outcome["raise"] = None
    assert authed_client.delete("/v1/apps/tiny").status_code == 204
    assert revoked == ["k2"]


# --------------------------------------------------------------------------- #
# updates, stop and start
# --------------------------------------------------------------------------- #


def _commit(store: apps.AppStore, app_id: str, version: str, when: str) -> None:
    target = store.version_dir(app_id, version)
    target.mkdir(parents=True)
    (target / apps.INSTALL_METADATA).write_text(json.dumps({"installedAt": when}))


def test_an_update_keeps_the_previous_version_and_prunes_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    monkeypatch.setattr(apps.ports, "is_free", lambda port, host="127.0.0.1": port != 8190)
    store = manager.store
    _commit(store, "tiny", "v1", "2026-09-01")
    _commit(store, "tiny", "v2", "2026-09-02")
    asyncio.run(manager.installed_callback(_manifest(version="v2"), AppOrigin.custom))
    first = store.get("tiny")
    assert first is not None and first.port == 8191  # 8190 is held by something else
    assert first.previous_version is None
    assert apps.installed_versions(store, "tiny") == ["v2"]

    _commit(store, "tiny", "v3", "2026-09-03")
    stray = store.version_dir("tiny", "half-built")
    stray.mkdir(parents=True)

    asyncio.run(manager.installed_callback(_manifest(version="v3"), AppOrigin.custom))
    record = store.get("tiny")
    assert record is not None
    assert (record.version, record.previous_version, record.port) == ("v3", "v2", 8191)
    assert apps.installed_versions(store, "tiny") == ["v3", "v2"]
    assert not stray.exists()


def test_stop_is_remembered_and_boot_leaves_it_stopped(
    app: FastAPI, authed_client: TestClient, tmp_path: Path
) -> None:
    manager: apps.AppManager = app.state.apps
    fake = FakeAppSupervisor()
    manager.supervisor = fake  # type: ignore[assignment]
    manager.store.put(_record())

    stopped = authed_client.post("/v1/apps/tiny/stop").json()
    assert stopped["enabled"] is False and stopped["status"] == "exited"
    assert authed_client.post("/v1/apps/tiny/restart").status_code == 409

    reloaded = apps.AppStore(tmp_path / apps.APPS_FILE)
    reloaded.load()
    record = reloaded.get("tiny")
    assert record is not None and record.enabled is False

    fake.started.clear()
    asyncio.run(manager.start_enabled())
    assert fake.started == []

    started = authed_client.post("/v1/apps/tiny/start").json()
    assert started["enabled"] is True and fake.started == ["tiny"]


# --------------------------------------------------------------------------- #
# an app's settings, through the agent
# --------------------------------------------------------------------------- #


def test_settings_go_through_the_admin_token_not_the_operators_bearer(
    app: FastAPI, authed_client: TestClient
) -> None:
    manager: apps.AppManager = app.state.apps
    fake = FakeAppSupervisor()
    manager.supervisor = fake  # type: ignore[assignment]
    manager.store.put(_record(_manifest(configTrio=True)))
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json={"component": "tiny", "values": {"answer": 42}})

    _http.set_shared_client(
        "apps-config", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    try:
        # Not running: the agent has no admin token to present.
        assert authed_client.get("/v1/apps/tiny/config").status_code == 503

        fake.planners["tiny"] = object()
        got = authed_client.get("/v1/apps/tiny/config")
        assert got.status_code == 200, got.text
        assert seen == [f"Bearer {fake.token}"]
        assert authed_client.headers["Authorization"] not in seen
    finally:
        _http.set_shared_client("apps-config", None)

    manager.store.put(_record(_manifest(configTrio=False)))
    assert authed_client.get("/v1/apps/tiny/config").status_code == 409


def test_reach_counts_a_running_apps_port(app: FastAPI, authed_client: TestClient) -> None:
    manager: apps.AppManager = app.state.apps
    fake = FakeAppSupervisor()
    manager.supervisor = fake  # type: ignore[assignment]
    manager.store.put(_record(port=8195))
    fake.planners["tiny"] = object()
    bound = authed_client.get("/v1/node").json()["reach"]["boundAddresses"]
    assert {"process": "app:tiny", "port": 8195} in [
        {"process": b["process"], "port": b["port"]} for b in bound
    ]


# --------------------------------------------------------------------------- #
# for real: uv builds it, the verify step judges it, a process runs it
# --------------------------------------------------------------------------- #


def _real_uv() -> Path | None:
    beside = Path(sys.executable).parent / ("uv.exe" if sys.platform == "win32" else "uv")
    if beside.is_file():
        return beside
    found = shutil.which("uv")
    return Path(found) if found else None


@pytest.mark.skipif(_real_uv() is None, reason="uv is not installed in this environment")
def test_a_real_install_runs_carries_no_credential_and_refuses_what_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uv = _real_uv()
    assert uv is not None
    monkeypatch.setenv("EUGENE_PLEXUS_GATEWAY_SERVICE_TOKEN", "leaked")

    async def scenario() -> None:
        store = apps.AppStore(tmp_path / apps.APPS_FILE)
        supervisor = apps.AppSupervisor()

        async def gateway() -> tuple[str | None, str | None]:
            return None, "no gateway in this test"

        manager = apps.AppManager(
            store=store,
            catalogue=[],
            get_config=lambda key: None,
            bind_host=lambda: None,
            advertise_host=lambda: None,
            node_name=lambda: "node-a",
            resolve_gateway=gateway,
        )
        manager.supervisor = supervisor
        installer = apps.AppInstaller()
        try:
            # A package whose -m target has no __main__: installed, refused
            # at verify, and its directory gone.
            bad = _manifest(id="nomain", entry="tiny_app.nomain", version="bad")
            installer.start(
                bad,
                store=store,
                uv=uv,
                on_installed=lambda m: manager.installed_callback(m, AppOrigin.custom),
            )
            outcome = await installer.wait("nomain")
            assert outcome is not None and outcome.state.value == "failed", outcome
            assert "no __main__" in (outcome.error or "")
            assert not store.version_dir("nomain", "bad").exists()
            assert store.get("nomain") is None

            good = _manifest()
            installer.start(
                good,
                store=store,
                uv=uv,
                on_installed=lambda m: manager.installed_callback(m, AppOrigin.custom),
            )
            outcome = await installer.wait("tiny")
            assert outcome is not None and outcome.state.value == "done", outcome
            assert (store.version_dir("tiny", "v1") / apps.INSTALL_METADATA).is_file()

            for _ in range(150):
                if manager.view(store.get("tiny")).status == ComponentStatus.running:  # type: ignore[arg-type]
                    break
                await asyncio.sleep(0.2)
            view = manager.view(store.get("tiny"))  # type: ignore[arg-type]
            assert view.status == ComponentStatus.running, view
            assert view.detail == "no gateway in this test"

            names = json.loads((store.data_dir("tiny") / "env.json").read_text())
            assert names and all(n.startswith("EUGENE_PLEXUS_APP_") for n in names), names

            await manager.uninstall("tiny", purge=False)
            assert not (store.app_dir("tiny") / "versions").exists()
            assert (store.data_dir("tiny") / "env.json").is_file()
        finally:
            await installer.aclose()
            await supervisor.stop_all()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# the small rules
# --------------------------------------------------------------------------- #


def test_a_reset_reaches_the_app_as_a_null(app: FastAPI, authed_client: TestClient) -> None:
    manager: apps.AppManager = app.state.apps
    fake = FakeAppSupervisor()
    manager.supervisor = fake  # type: ignore[assignment]
    manager.store.put(_record(_manifest(configTrio=True)))
    fake.planners["tiny"] = object()
    bodies: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, json={"applied": ["searchProvider"], "rejected": [], "requiresRestart": False}
        )

    _http.set_shared_client(
        "apps-config", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    try:
        sent = authed_client.patch("/v1/apps/tiny/config", json={"searchProvider": None})
        assert sent.status_code == 200, sent.text
    finally:
        _http.set_shared_client("apps-config", None)
    assert bodies == [{"searchProvider": None}]


def test_an_app_that_answers_nonsense_is_named_not_a_500(
    app: FastAPI, authed_client: TestClient
) -> None:
    manager: apps.AppManager = app.state.apps
    fake = FakeAppSupervisor()
    manager.supervisor = fake  # type: ignore[assignment]
    manager.store.put(_record(_manifest(configTrio=True)))
    fake.planners["tiny"] = object()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"what": 1}))
    _http.set_shared_client("apps-config", httpx.AsyncClient(transport=transport))
    try:
        got = authed_client.get("/v1/apps/tiny/config/schema")
    finally:
        _http.set_shared_client("apps-config", None)
    assert got.status_code == 502
    assert "Tiny" in got.json()["detail"]["detail"]


def test_uv_is_found_where_the_installer_left_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "uv.exe" if sys.platform == "win32" else "uv"
    prefix = tmp_path / "EugenePlexus"
    (prefix / "bin").mkdir(parents=True)
    beside = prefix / "bin" / name
    beside.write_text("")
    on_path = tmp_path / "elsewhere" / name
    on_path.parent.mkdir()
    on_path.write_text("")
    monkeypatch.setattr(apps.sys, "prefix", str(prefix / "venv"))
    monkeypatch.setattr(apps.shutil, "which", lambda _name: str(on_path))

    assert apps.find_uv() == beside
    configured = tmp_path / "mine" / name
    configured.parent.mkdir()
    configured.write_text("")
    assert apps.find_uv(str(configured)) == configured
    # A configured path that is wrong is wrong, not a reason to guess.
    assert apps.find_uv(str(tmp_path / "typo")) is None
    beside.unlink()
    assert apps.find_uv() == on_path


def test_a_ui_opens_on_the_advertised_host(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    ui = _record(_manifest(ui=True), port=8193)
    assert manager.ui_url(ui) == "http://127.0.0.1:8193/"
    manager._advertise_host = lambda: "fd7a::5"
    assert manager.ui_url(ui) == "http://[fd7a::5]:8193/"
    manager._advertise_host = lambda: "192.168.16.75"
    assert manager.ui_url(ui) == "http://192.168.16.75:8193/"
    assert manager.ui_url(_record(_manifest(ui=False))) is None


class _Identity:
    def __init__(self, *, enrolled: bool) -> None:
        self.record = type(
            "R",
            (),
            {
                "enrolled": enrolled,
                "control_url": "http://192.168.16.252:8283" if enrolled else None,
            },
        )()


def test_the_gateway_is_found_on_this_node_or_through_its_owners_agent(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from eugene_plexus_agent import install_proxy, security
    from eugene_plexus_agent._generated.models import ComponentEntry
    from eugene_plexus_agent.app import resolve_gateway_for_apps
    from eugene_plexus_agent.auth_state import AuthState
    from eugene_plexus_agent.state import AgentState

    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    asked: list[str | None] = []

    class Topology:
        async def owner_of(self, target: str, **kw: Any) -> install_proxy.RemoteNode:
            asked.append(kw.get("authorization"))
            assert target == "gateway"
            return install_proxy.RemoteNode(name="nas", agent_url="http://192.168.16.252:8279/")

    fake = SimpleNamespace(
        state=SimpleNamespace(
            agent_state=state,
            node_identity=_Identity(enrolled=False),
            auth_state=AuthState(signing_key=security.generate_signing_key()),
            install_topology=Topology(),
        )
    )
    url, detail = asyncio.run(resolve_gateway_for_apps(fake))  # type: ignore[arg-type]
    assert url is None and detail and "not part of an install" in detail

    fake.state.node_identity = _Identity(enrolled=True)
    url, detail = asyncio.run(resolve_gateway_for_apps(fake))  # type: ignore[arg-type]
    # The owning node's public proxy, at the address it announced -- not
    # the gateway's own URL, which a port remap makes wrong everywhere
    # but on its host.
    assert (url, detail) == ("http://192.168.16.252:8279/api/proxy/gateway", None)
    assert asked and (asked[0] or "").startswith("Bearer ")

    state.add_topology_entry(
        ComponentEntry.model_validate(
            {
                "name": "gateway",
                "kind": "gateway",
                "url": "http://127.0.0.1:8084",
                "spawn": {"configFile": str(tmp_path / "gateway.yaml")},
            }
        )
    )
    url, detail = asyncio.run(resolve_gateway_for_apps(fake))  # type: ignore[arg-type]
    assert (url, detail) == ("http://127.0.0.1:8084", None)


def test_the_install_tools_get_no_hub_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package's build step is its author's code, run by `uv` at install
    time -- so the tools get the boundary a spawned app gets."""
    monkeypatch.setenv("EUGENE_PLEXUS_GATEWAY_SERVICE_TOKEN", "leaked")
    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_CONFIG_FILE", "leaked")
    seen: list[dict[str, str]] = []

    async def tool(argv: list[str], *, env: dict[str, str]) -> tuple[int, str]:
        seen.append(env)
        return 0, ""

    monkeypatch.setattr(apps, "_run_tool", tool)
    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    target = store.version_dir("tiny", "v1")
    progress = apps._Progress(app="tiny", version="v1")
    asyncio.run(apps.AppInstaller()._install(_manifest(), progress, target=target, uv=Path("uv")))
    assert len(seen) == 3  # venv, install, verify
    assert all(not any(k.startswith("EUGENE_PLEXUS_") for k in env) for env in seen)
    assert (target / apps.INSTALL_METADATA).is_file()
