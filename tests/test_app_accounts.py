"""Each app in an OS account of its own (C1, `workbench.md` §2).

What a unit test can prove: which installs give apps accounts and why the
others do not; that an app which runs what a model chooses is refused
where it would run as the agent; that its key may send logs; that the
spec the launcher reads holds no secret; that the supervisor starts
through the service manager, takes back a running app after an agent
restart, and leaves apps running when the agent stops; and that the
launcher forwards what an app prints and stops it when asked.

What only a runner can prove -- that the account really cannot open
`node.yaml` -- is `specs/scripts/c1-app-accounts-acceptance.py`.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import app_accounts, app_launcher, apps
from eugene_plexus_agent._generated.models import (
    AppManifest,
    AppOrigin,
    ClientKey,
    ClientKeyCreated,
    ComponentStatus,
    InstallMechanism,
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


def _created(name: str) -> ClientKeyCreated:
    now = datetime.now(UTC)
    return ClientKeyCreated(
        key=ClientKey(
            id="k1", name=name, tail="abcdef", createdAt=now, expiresAt=now + timedelta(days=1)
        ),
        token="the-token-shown-once",
    )


@pytest.fixture
def installing(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The install route with minting and the installer replaced."""
    seen: dict[str, Any] = {"bodies": [], "started": []}

    async def mint(request: Any, body: Any, *, authorization: str | None) -> ClientKeyCreated:
        seen["bodies"].append(body)
        return _created(name=body.name)

    def start(manifest: AppManifest, **_kw: Any) -> Any:
        seen["started"].append(manifest.id)
        return apps._Progress(app=manifest.id, version=manifest.version).snapshot()

    manager: apps.AppManager = app.state.apps
    monkeypatch.setattr(apps_routes, "_enrolled", lambda request: True)
    monkeypatch.setattr(apps, "find_uv", lambda configured=None: Path("uv"))
    monkeypatch.setattr(apps_routes, "mint_client_key", mint)
    monkeypatch.setattr(manager.installer, "start", start)
    return seen


# --------------------------------------------------------------------- #
# which installs, and the refusal
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mechanism,kind",
    [
        (InstallMechanism.windows_task, None),
        (InstallMechanism.systemd_user, None),
        (InstallMechanism.launchd, None),
        (InstallMechanism.container, None),
        (InstallMechanism.none, None),
    ],
)
def test_an_install_that_runs_as_the_person_makes_no_accounts_and_says_why(
    mechanism: InstallMechanism, kind: str | None
) -> None:
    support = app_accounts.detect(mechanism)
    assert support.kind is kind
    assert support.reason and len(support.reason) > 30


def test_the_linux_system_install_needs_its_unit_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_accounts, "UNIT_FILE", tmp_path / "eugene-plexus-app@.service")
    monkeypatch.setattr(app_accounts, "CTL_PATH_UNIT", tmp_path / "eugene-plexus-apps-ctl.path")
    missing = app_accounts.detect(InstallMechanism.systemd_system)
    assert missing.kind is None and "install.sh again" in (missing.reason or "")
    (tmp_path / "eugene-plexus-app@.service").write_text("")
    (tmp_path / "eugene-plexus-apps-ctl.path").write_text("")
    assert app_accounts.detect(InstallMechanism.systemd_system).kind == "systemd"


def test_an_app_that_runs_model_chosen_actions_is_refused_without_an_account(
    app: FastAPI, authed_client: TestClient, installing: dict[str, Any]
) -> None:
    manager: apps.AppManager = app.state.apps
    manager.accounts = app_accounts.AccountSupport(
        None, "This install runs as you, from a sign-in task."
    )
    # An entry that does not say is treated as one that does.
    manager.catalogue = {"tiny": _manifest()}
    refused = authed_client.post("/v1/apps/tiny/install")
    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]["detail"]
    assert "account of its own" in detail and "sign-in task" in detail
    assert installing["bodies"] == [] and installing["started"] == []

    manager.catalogue = {"tiny": _manifest(localActions=False)}
    assert authed_client.post("/v1/apps/tiny/install").status_code == 202


def test_where_accounts_exist_the_same_app_installs(
    app: FastAPI, authed_client: TestClient, installing: dict[str, Any]
) -> None:
    manager: apps.AppManager = app.state.apps
    manager.accounts = app_accounts.AccountSupport("systemd")
    manager.catalogue = {"tiny": _manifest()}
    assert authed_client.post("/v1/apps/tiny/install").status_code == 202


def test_an_apps_key_may_send_logs(
    app: FastAPI, authed_client: TestClient, installing: dict[str, Any]
) -> None:
    app.state.apps.catalogue = {"tiny": _manifest(localActions=False)}
    assert authed_client.post("/v1/apps/tiny/install").status_code == 202
    [body] = installing["bodies"]
    assert body.limits is not None and body.limits.writeLogs is True


def test_the_catalogue_says_whether_this_node_makes_accounts(
    app: FastAPI, authed_client: TestClient
) -> None:
    manager: apps.AppManager = app.state.apps
    manager.accounts = app_accounts.AccountSupport(None, "This install runs as you.")
    body = authed_client.get("/v1/app-catalogue").json()
    assert body["ownAccounts"] is False and body["ownAccountsReason"] == "This install runs as you."
    manager.accounts = app_accounts.AccountSupport("windows_service")
    body = authed_client.get("/v1/app-catalogue").json()
    assert body["ownAccounts"] is True and "ownAccountsReason" not in body


# --------------------------------------------------------------------- #
# what the launcher is given
# --------------------------------------------------------------------- #


def _plan(tmp_path: Path) -> app_accounts.LaunchPlan:
    return app_accounts.LaunchPlan(
        app_id="tiny",
        argv=["python", "-m", "tiny_app"],
        env={
            "EUGENE_PLEXUS_APP_ID": "tiny",
            "EUGENE_PLEXUS_APP_BIND_PORT": "8190",
            "EUGENE_PLEXUS_APP_ADMIN_TOKEN": "secret-admin",
            "EUGENE_PLEXUS_APP_KEY_FILE": str(tmp_path / "data" / "client_key"),
            "EUGENE_PLEXUS_APP_DATA_DIR": str(tmp_path / "data"),
            "HTTPS_PROXY": "http://proxy:3128",
            "PATH": "/the/agents/path",
            "EUGENE_PLEXUS_AGENT_CONFIG_FILE": "/nope",
        },
        data_dir=tmp_path / "data",
        key_file=tmp_path / "data" / "client_key",
        admin_token="secret-admin",
        port=8190,
        bind_host=None,
    )


def test_the_spec_holds_no_secret_and_none_of_the_agents_environment(tmp_path: Path) -> None:
    for kind in ("windows_service", "systemd"):
        spec = app_accounts.launch_spec(
            _plan(tmp_path), kind=kind, ingress="http://127.0.0.1:8079/v1/logs"
        )
        text = json.dumps(spec)
        assert "secret-admin" not in text
        assert "EUGENE_PLEXUS_AGENT" not in text and "/the/agents/path" not in text
        assert spec["env"]["EUGENE_PLEXUS_APP_BIND_PORT"] == "8190"
        assert spec["env"]["HTTPS_PROXY"] == "http://proxy:3128"
    windows = app_accounts.launch_spec(_plan(tmp_path), kind="windows_service", ingress="x")
    assert (
        windows["keyFile"].endswith("client_key") and windows["service"] == "EugenePlexusApp-tiny"
    )
    # On Linux systemd hands the key over; a path in the spec would be one
    # the app's account cannot open anyway.
    systemd = app_accounts.launch_spec(_plan(tmp_path), kind="systemd", ingress="x")
    assert "keyFile" not in systemd and "adminTokenFile" not in systemd


def test_the_launcher_fills_in_what_systemd_hands_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    creds = tmp_path / "creds"
    creds.mkdir()
    (creds / "admin_token").write_text("from-systemd\n")
    spec_file = tmp_path / "launch.json"
    spec_file.write_text(
        json.dumps({"app": "tiny", "argv": ["x"], "env": {"EUGENE_PLEXUS_APP_ID": "tiny"}})
    )
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path / "state"))
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(creds))
    spec = app_launcher.load_spec(str(spec_file))
    env = app_launcher.child_environment(spec)
    assert env["EUGENE_PLEXUS_APP_DATA_DIR"] == str(tmp_path / "state")
    assert env["EUGENE_PLEXUS_APP_KEY_FILE"] == str(creds / "client_key")
    assert env["EUGENE_PLEXUS_APP_ADMIN_TOKEN"] == "from-systemd"
    assert env["HOME"] == str(tmp_path / "state")
    # What systemd told the launcher about itself is not the app's.
    assert "CREDENTIALS_DIRECTORY" not in env and "STATE_DIRECTORY" not in env


def test_account_signal_comes_from_service_plan_not_ambient_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "EUGENE_PLEXUS_APP_ACCOUNT_KIND"
    monkeypatch.setenv(name, "windows_service")
    assert name not in app_launcher.child_environment({"env": {name: "systemd"}})
    for kind in ("windows_service", "systemd"):
        spec = app_accounts.launch_spec(_plan(tmp_path), kind=kind, ingress="x")
        assert app_launcher.child_environment(spec)[name] == kind
    assert name not in _planner(tmp_path).plan().env


# --------------------------------------------------------------------- #
# the supervisor
# --------------------------------------------------------------------- #


class FakeRunner:
    kind = "systemd"

    def __init__(self, apps_root: Path, *, running: bool = False) -> None:
        self.apps_root = apps_root
        self.calls: list[str] = []
        self.current = app_accounts.ServiceState("running" if running else "stopped")

    def prepare(self, plan: app_accounts.LaunchPlan, spec: dict) -> None:
        self.calls.append("prepare")
        self.spec = spec
        token = self.token_file(plan.app_id, plan.data_dir)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(plan.admin_token)

    def start(self, app_id: str) -> None:
        self.calls.append("start")
        self.current = app_accounts.ServiceState("running", pid=4242)

    def stop(self, app_id: str) -> None:
        self.calls.append("stop")
        self.current = app_accounts.ServiceState("stopped")

    def state(self, app_id: str) -> app_accounts.ServiceState:
        return self.current

    def remove(self, app_id: str, *, purge: bool) -> None:
        self.calls.append(f"remove purge={purge}")

    def token_file(self, app_id: str, data_dir: Path) -> Path:
        return self.apps_root / app_id / "admin_token"


def _free_port() -> int:
    """A port nothing answers on. Not 8190: that is the first app port, so
    on a machine running a real app (Workbench, on the developer's own box
    since 2026-10-01) the poll below found a healthy app there."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _planner(tmp_path: Path) -> apps._AppPlanner:
    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    record = apps.InstalledApp(
        manifest=_manifest(),
        origin=AppOrigin.custom,
        port=_free_port(),
        installed_at=datetime.now(UTC),
    )
    python = apps.venv_python(store.version_dir(record.id, record.version) / "venv")
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
    return apps._AppPlanner(
        record, store=store, gateway_url=lambda: "http://gw", bind_host=lambda: None
    )


def _supervisor(runner: FakeRunner, tmp_path: Path) -> app_accounts.OwnAccountSupervisor:
    return app_accounts.OwnAccountSupervisor(
        runner,  # type: ignore[arg-type]
        ingress=lambda: "http://127.0.0.1:8079/v1/logs",
        data_dir=lambda app_id: tmp_path / "apps" / app_id / "data",
        key_file=lambda app_id: tmp_path / "apps" / app_id / "data" / "client_key",
    )


def test_a_start_goes_through_the_service_manager_with_the_ingress_in_the_spec(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runner = FakeRunner(tmp_path / "apps")
        supervisor = _supervisor(runner, tmp_path)
        supervisor.start(_planner(tmp_path))
        await supervisor._starting["tiny"]
        assert runner.calls == ["prepare", "start"]
        assert runner.spec["ingress"] == "http://127.0.0.1:8079/v1/logs"
        assert supervisor.admin_token("tiny")
        await supervisor._poll("tiny")
        # The service runs but the app has not answered /healthz yet.
        assert supervisor.status("tiny")[0] == ComponentStatus.starting
        await supervisor.stop_all()

    asyncio.run(scenario())


def test_an_app_running_before_the_agent_started_is_kept_not_restarted(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = FakeRunner(tmp_path / "apps", running=True)
        token = runner.token_file("tiny", tmp_path)
        token.parent.mkdir(parents=True)
        token.write_text("the-token-it-was-started-with")
        supervisor = _supervisor(runner, tmp_path)
        supervisor.start(_planner(tmp_path))
        await supervisor._starting["tiny"]
        assert runner.calls == []
        assert supervisor.admin_token("tiny") == "the-token-it-was-started-with"
        await supervisor.stop_all()

    asyncio.run(scenario())


def test_the_agent_stopping_does_not_stop_the_apps_but_a_stop_does(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = FakeRunner(tmp_path / "apps")
        supervisor = _supervisor(runner, tmp_path)
        supervisor.start(_planner(tmp_path))
        await supervisor._starting["tiny"]
        await supervisor.stop_all()
        assert "stop" not in runner.calls

        supervisor = _supervisor(runner, tmp_path)
        supervisor.start(_planner(tmp_path))
        await supervisor._starting["tiny"]
        await supervisor.stop("tiny")
        assert runner.calls[-1] == "stop"
        await supervisor.remove("tiny", purge=True)
        assert runner.calls[-1] == "remove purge=True"

    asyncio.run(scenario())


def test_a_failed_service_is_crashed_with_a_sentence(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = FakeRunner(tmp_path / "apps")
        supervisor = _supervisor(runner, tmp_path)
        supervisor.start(_planner(tmp_path))
        await supervisor._starting["tiny"]
        runner.current = app_accounts.ServiceState("failed", exit_code=3)
        await supervisor._poll("tiny")
        status, error, *_ = supervisor.status("tiny")
        assert status == ComponentStatus.crashed
        assert error and "code 3" in error and "Logs page" in error
        await supervisor.stop_all()

    asyncio.run(scenario())


def test_a_start_the_service_manager_refuses_is_the_apps_last_error(tmp_path: Path) -> None:
    class Refusing(FakeRunner):
        def start(self, app_id: str) -> None:
            raise OSError("systemctl start eugene-plexus-app@tiny.service failed: no such unit")

    async def scenario() -> None:
        supervisor = _supervisor(Refusing(tmp_path / "apps"), tmp_path)
        supervisor.start(_planner(tmp_path))
        await supervisor._starting["tiny"]
        status, error, *_ = supervisor.status("tiny")
        assert status == ComponentStatus.crashed and "no such unit" in (error or "")
        await supervisor.stop_all()

    asyncio.run(scenario())


# --------------------------------------------------------------------- #
# the launcher, for real: a child that prints, a fake ingress, a stop
# --------------------------------------------------------------------- #


class _Ingress:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.auth: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.auth.append(self.headers.get("authorization", ""))
                for resource in body["resourceLogs"]:
                    for scope in resource["scopeLogs"]:
                        outer.lines.extend(r["body"]["stringValue"] for r in scope["logRecords"])
                self.send_response(200)
                self.send_header("content-length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *_a: object) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/logs"


def test_the_launcher_forwards_what_the_app_prints_and_stops_it(tmp_path: Path) -> None:
    ingress = _Ingress()
    key = tmp_path / "client_key"
    key.write_text("app-key\n")
    child = tmp_path / "child.py"
    child.write_text(
        "import sys, time\n"
        "print('hello from the app', flush=True)\n"
        "print('second line', flush=True)\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    spec = {
        "app": "tiny",
        "argv": [sys.executable, str(child)],
        "env": {},
        "dataDir": str(tmp_path / "data"),
        "keyFile": str(key),
        "ingress": ingress.url,
    }
    (tmp_path / "data").mkdir()
    app_launcher._STOP.clear()
    result: dict[str, int] = {}
    runner = threading.Thread(target=lambda: result.setdefault("code", app_launcher.run(spec)))
    runner.start()
    try:
        deadline = time.perf_counter() + 20
        while "second line" not in ingress.lines and time.perf_counter() < deadline:
            time.sleep(0.1)
        assert ingress.lines[:2] == ["hello from the app", "second line"]
        assert ingress.auth[0] == "Bearer app-key"
    finally:
        app_launcher._STOP.set()
        runner.join(40)
        ingress.server.shutdown()
    assert not runner.is_alive()
    # A stop that was asked for is not a failure.
    assert result["code"] == 0


def test_an_app_that_crashes_hands_its_code_to_the_service_manager(tmp_path: Path) -> None:
    ingress = _Ingress()
    child = tmp_path / "child.py"
    child.write_text("print('about to fail', flush=True)\nraise SystemExit(7)\n")
    key = tmp_path / "client_key"
    key.write_text("app-key")
    spec = {
        "app": "tiny",
        "argv": [sys.executable, str(child)],
        "env": {},
        "keyFile": str(key),
        "ingress": ingress.url,
    }
    app_launcher._STOP.clear()
    try:
        assert app_launcher.run(spec) == 7
    finally:
        ingress.server.shutdown()
    assert "about to fail" in ingress.lines
    assert any("exited with code 7" in line for line in ingress.lines)


def test_lines_wait_for_an_agent_that_is_down_and_say_what_was_lost(tmp_path: Path) -> None:
    forwarder = app_launcher.Forwarder("http://127.0.0.1:9/v1/logs", None, "tiny")
    for n in range(app_launcher.BUFFER_LINES + 5):
        forwarder._lines.append(str(n))
    forwarder.add("one more")
    assert len(forwarder._lines) == app_launcher.BUFFER_LINES
    batch = forwarder._take()
    assert "lines were dropped" in batch[0]


def test_a_service_that_will_not_go_fails_the_uninstall(tmp_path: Path) -> None:
    """C1's first Windows run: removal failed, was logged, and the uninstall
    said it had worked while the service lived on."""

    class Stuck(FakeRunner):
        def remove(self, app_id: str, *, purge: bool) -> None:
            raise OSError("DeleteService: access denied")

    async def scenario() -> None:
        supervisor = _supervisor(Stuck(tmp_path / "apps"), tmp_path)
        supervisor.start(_planner(tmp_path))
        await supervisor._starting["tiny"]
        with pytest.raises(OSError, match="access denied"):
            await supervisor.remove("tiny", purge=False)
        await supervisor.stop_all()

    asyncio.run(scenario())


def test_each_linux_app_gets_a_user_of_its_own_within_systemds_limit() -> None:
    import hashlib

    names = {app_accounts.dynamic_user(i) for i in ("probe-a", "probe-b", "workbench", "a" * 40)}
    assert len(names) == 4
    assert all(len(n) <= 31 and n.startswith("eapp-") for n in names)
    # The same name install.sh's helper computes: sha256 of the id, hex.
    assert app_accounts.dynamic_user("workbench") == (
        "eapp-" + hashlib.sha256(b"workbench").hexdigest()[:12]
    )
    assert app_accounts.dynamic_user("workbench") in app_accounts.account_name(
        "systemd", "workbench"
    )


def test_a_stop_the_service_manager_refuses_raises_and_the_app_is_still_reported(
    tmp_path: Path,
) -> None:
    class Unstoppable(FakeRunner):
        def stop(self, app_id: str) -> None:
            raise OSError("nothing answered the request to stop")

    async def scenario() -> None:
        runner = Unstoppable(tmp_path / "apps")
        supervisor = _supervisor(runner, tmp_path)
        supervisor.start(_planner(tmp_path))
        await supervisor._starting["tiny"]
        with pytest.raises(OSError, match="nothing answered"):
            await supervisor.stop("tiny")
        # Not "exited": the service is still running, so the page says so.
        assert supervisor.is_running("tiny")
        await supervisor.stop_all()

    asyncio.run(scenario())


def test_removing_asks_the_service_manager_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = FakeRunner(tmp_path / "apps")
        supervisor = _supervisor(runner, tmp_path)
        supervisor.start(_planner(tmp_path))
        await supervisor._starting["tiny"]
        await supervisor.remove("tiny", purge=False)
        assert runner.calls == ["prepare", "start", "remove purge=False"]
        assert not supervisor.is_running("tiny")
        await supervisor.stop_all()

    asyncio.run(scenario())
