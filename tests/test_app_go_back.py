"""Go back on Windows, with the service manager's own recovery in the way (agent#34).

C4's Windows run failed at step 101 in about one run in seven: after Go back, Open
WebUI's old version never ran again. Windows starts a crashed app's service again five
seconds after it stops (`WindowsServiceRunner.prepare`'s failure actions), from the
`launch.json` on disk at that moment. When that start came between Go back's stop and
its start, the supervisor kept the service it found running -- the failed new version,
from the new version's spec -- and never wrote the restored one.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_agent import app_accounts, apps
from eugene_plexus_agent._generated.models import AppOrigin
from tests.test_apps import _commit, _fake_python, _manager, _manifest, _record


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RecoveringScm:
    """A runner whose service manager starts a crashed service again on its own.

    `crash()` is the app's start failing. Windows then starts the service again from
    whatever `launch.json` holds when its recovery delay ends; here that is the first
    question about the service after Eugene's stop, which is the window C4 lost.
    `hold()` does nothing here: these tests are about what the supervisor does with a
    service it did not start, whatever let it run.
    """

    kind = "windows_service"

    def __init__(self, apps_root: Path) -> None:
        self.apps_root = apps_root
        self.calls: list[str] = []
        self.current = app_accounts.ServiceState("stopped")
        #: What the running service read when it started, or None.
        self.loaded: dict[str, Any] | None = None
        self._pending_recovery = False
        self._stopped_since_crash = False
        self.recovered = 0

    def _spec_file(self, app_id: str) -> Path:
        return self.apps_root / app_id / app_accounts.SPEC_FILE

    def prepare(self, plan: app_accounts.LaunchPlan, spec: dict) -> None:
        self.calls.append("prepare")
        path = self._spec_file(plan.app_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec), encoding="utf-8")
        self.token_file(plan.app_id, plan.data_dir).write_text(plan.admin_token)

    def _run_from_disk(self, app_id: str) -> None:
        self.loaded = json.loads(self._spec_file(app_id).read_text(encoding="utf-8"))
        self.current = app_accounts.ServiceState("running", pid=4000 + len(self.calls))

    def start(self, app_id: str) -> None:
        self.calls.append("start")
        if self.current.state not in ("running", "starting"):
            self._run_from_disk(app_id)

    def crash(self) -> None:
        self.current = app_accounts.ServiceState("failed", exit_code=1)
        self.loaded = None
        self._pending_recovery = True
        self._stopped_since_crash = False

    def hold(self, app_id: str) -> None:
        self.calls.append("hold")

    def stop(self, app_id: str) -> None:
        self.calls.append("stop")
        if self.current.state in ("running", "starting", "stopping"):
            self.current = app_accounts.ServiceState("stopped")
            self.loaded = None
        self._stopped_since_crash = True

    def state(self, app_id: str) -> app_accounts.ServiceState:
        if self._pending_recovery and self._stopped_since_crash:
            self._pending_recovery = False
            self.recovered += 1
            self.calls.append("recovery")
            self._run_from_disk(app_id)
        return self.current

    def launched(self, app_id: str) -> dict | None:
        try:
            return json.loads(self._spec_file(app_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def remove(self, app_id: str, *, purge: bool) -> None:
        self.calls.append("remove")

    def token_file(self, app_id: str, data_dir: Path) -> Path:
        return self.apps_root / app_id / app_accounts.ADMIN_TOKEN_FILE


def _updated(tmp_path: Path) -> tuple[apps.AppManager, RecoveringScm, Any]:
    """v1 installed in its own account, v2 ready to update to; both environments exist."""
    manager = _manager(tmp_path)
    old = _record(_manifest(updateBackup=["webui.db"]), port=_free_port())
    manager.store.put(old)
    _commit(manager.store, "tiny", "v1", "2026-10-01T00:00:00Z")
    _fake_python(manager.store, old)
    new = _manifest(version="v2", updateBackup=["webui.db"])
    _commit(manager.store, "tiny", "v2", "2026-10-21T00:00:00Z")
    _fake_python(manager.store, _record(new))
    data = manager.store.data_dir("tiny")
    data.mkdir(parents=True)
    (data / "webui.db").write_text("old chats")
    scm = RecoveringScm(manager.store.root)
    manager.supervisor = app_accounts.OwnAccountSupervisor(
        scm,  # type: ignore[arg-type]
        ingress=lambda: "http://127.0.0.1:8079/v1/logs",
        data_dir=manager.store.data_dir,
        key_file=manager.store.key_file,
    )
    return manager, scm, new


def _version_of(spec: dict | None) -> str | None:
    if not spec:
        return None
    parts = Path(spec["argv"][0]).parts
    return parts[parts.index("versions") + 1]


def test_a_restart_by_windows_recovery_during_go_back_gives_way_to_the_restored_version(
    tmp_path: Path,
) -> None:
    manager, scm, new = _updated(tmp_path)
    supervisor = manager.supervisor
    assert isinstance(supervisor, app_accounts.OwnAccountSupervisor)

    async def scenario() -> None:
        await manager.installed_callback(new, AppOrigin.custom)
        await supervisor._starting["tiny"]
        assert _version_of(scm.loaded) == "v2"
        # v2's start fails, as C4's broken update does; Windows will start it again.
        scm.crash()
        record = manager.store.get("tiny")
        assert record is not None
        await manager.rollback(record)
        await supervisor._starting["tiny"]
        await supervisor.stop_all()

    asyncio.run(scenario())
    assert scm.recovered == 1, "the recovery start must fall between Go back's stop and start"
    # What runs is the version Go back restored, from a spec written for it...
    assert _version_of(scm.loaded) == "v1", scm.calls
    # ...and the service Windows started from v2's spec was stopped before it was written.
    after = scm.calls[scm.calls.index("recovery") :]
    assert after[:4] == ["recovery", "stop", "prepare", "start"], scm.calls


# --------------------------------------------------------------------- #
# the Windows runner, against a service manager made of pywin32's calls
# --------------------------------------------------------------------- #

RUNNING, START_PENDING, STOP_PENDING, STOPPED = 4, 2, 3, 1
DISABLED, DEMAND, NO_CHANGE = 4, 3, 0xFFFFFFFF


class _WinError(Exception):
    """`pywintypes.error`: (winerror, funcname, strerror)."""

    def __init__(self, winerror: int, funcname: str = "", strerror: str = "") -> None:
        super().__init__(winerror, funcname, strerror)
        self.winerror, self.funcname, self.strerror = winerror, funcname, strerror


class FakeScm:
    """One app's service as Windows' service control manager keeps it.

    A crash queues a restart `recover_in` seconds later, which nothing cancels but a
    start type that forbids it. A stop takes `stop_queries` status queries to finish,
    or never (`stop_takes_forever`). The service's process may outlive the service's
    own STOPPED (`process_outlives`).
    """

    def __init__(self) -> None:
        self.state, self.pid, self.exit_code = STOPPED, 0, 0
        self.start_type = DEMAND
        self.recover_at: float | None = None
        self.events: list[str] = []
        self.stop_queries = 1
        self.stop_takes_forever = False
        self.start_queries = 0
        self.process_outlives = False
        self.alive: set[int] = set()
        self._stopping = 0
        self._next_pid = 4000

    def run(self) -> None:
        self._next_pid += 1
        self.state, self.pid, self.exit_code = RUNNING, self._next_pid, 0
        self.alive.add(self.pid)

    def crash(self, recover_in: float) -> None:
        self.alive.discard(self.pid)
        self.state, self.pid, self.exit_code = STOPPED, 0, 1
        self.recover_at = time.perf_counter() + recover_in

    def tick(self) -> None:
        if self.recover_at is not None and time.perf_counter() >= self.recover_at:
            self.recover_at = None
            if self.start_type == DISABLED:
                self.events.append("recovery refused: disabled")
            else:
                self.events.append("recovery start")
                self.run()
        if self.state == START_PENDING:
            self.start_queries -= 1
            if self.start_queries <= 0:
                self.state = RUNNING
        if self.state == STOP_PENDING and not self.stop_takes_forever:
            self._stopping -= 1
            if self._stopping <= 0:
                if not self.process_outlives:
                    self.alive.discard(self.pid)
                self.state, self.pid = STOPPED, 0

    def modules(self) -> dict[str, types.ModuleType]:
        scm = self
        pywintypes = types.ModuleType("pywintypes")
        pywintypes.error = _WinError  # type: ignore[attr-defined]

        service = types.ModuleType("win32service")
        for name, value in {
            "SC_MANAGER_CONNECT": 1,
            "SC_MANAGER_ALL_ACCESS": 0xF003F,
            "SERVICE_QUERY_STATUS": 4,
            "SERVICE_CHANGE_CONFIG": 2,
            "SERVICE_ALL_ACCESS": 0xF01FF,
            "SERVICE_WIN32_OWN_PROCESS": 0x10,
            "SERVICE_ERROR_NORMAL": 1,
            "SERVICE_NO_CHANGE": NO_CHANGE,
            "SERVICE_DISABLED": DISABLED,
            "SERVICE_DEMAND_START": DEMAND,
            "SERVICE_RUNNING": RUNNING,
            "SERVICE_START_PENDING": START_PENDING,
            "SERVICE_CONTINUE_PENDING": 5,
            "SERVICE_STOP_PENDING": STOP_PENDING,
            "SERVICE_CONTROL_STOP": 1,
            "SERVICE_CONFIG_DESCRIPTION": 1,
            "SERVICE_CONFIG_FAILURE_ACTIONS": 2,
            "SERVICE_CONFIG_FAILURE_ACTIONS_FLAG": 4,
            "SC_ACTION_RESTART": 1,
        }.items():
            setattr(service, name, value)

        def change_config(_handle: object, _kind: int, start: int, *_rest: object) -> None:
            if start != NO_CHANGE:
                scm.start_type = start
                scm.events.append("disabled" if start == DISABLED else "demand start")

        def create_service(*_args: object) -> str:
            raise _WinError(1073, "CreateService", "The specified service already exists.")

        def query(_handle: object) -> dict[str, int]:
            scm.tick()
            return {
                "CurrentState": scm.state,
                "ProcessId": scm.pid,
                "ServiceSpecificExitCode": scm.exit_code,
                "Win32ExitCode": 1066 if scm.exit_code else 0,
            }

        service.OpenSCManager = lambda *a: "manager"  # type: ignore[attr-defined]
        service.OpenService = lambda *a: "service"  # type: ignore[attr-defined]
        service.CloseServiceHandle = lambda h: None  # type: ignore[attr-defined]
        service.CreateService = create_service  # type: ignore[attr-defined]
        service.ChangeServiceConfig = change_config  # type: ignore[attr-defined]
        service.ChangeServiceConfig2 = lambda *a: None  # type: ignore[attr-defined]
        service.QueryServiceStatusEx = query  # type: ignore[attr-defined]

        util = types.ModuleType("win32serviceutil")

        def control(_name: str, _code: int) -> None:
            scm.tick()
            if scm.state == STOPPED:
                raise _WinError(1062, "ControlService", "The service has not been started.")
            if scm.state in (START_PENDING, STOP_PENDING):
                raise _WinError(1061, "ControlService", "The service cannot accept control.")
            scm.state, scm._stopping = STOP_PENDING, scm.stop_queries
            scm.events.append("stop")

        def start(_name: str) -> None:
            if scm.start_type == DISABLED:
                raise _WinError(1058, "StartService", "The service is disabled.")
            if scm.state != STOPPED:
                raise _WinError(1056, "StartService", "An instance is already running.")
            scm.events.append("start")
            scm.run()

        util.ControlService = control  # type: ignore[attr-defined]
        util.StartService = start  # type: ignore[attr-defined]

        class _Process:
            def __init__(self, pid: int) -> None:
                self.pid = pid

            def Close(self) -> None:
                return None

        api = types.ModuleType("win32api")

        def open_process(_access: int, _inherit: bool, pid: int) -> _Process:
            if pid not in scm.alive:
                raise _WinError(87, "OpenProcess", "The parameter is incorrect.")
            return _Process(pid)

        api.OpenProcess = open_process  # type: ignore[attr-defined]
        con = types.ModuleType("win32con")
        con.SYNCHRONIZE = 0x100000  # type: ignore[attr-defined]
        event = types.ModuleType("win32event")
        event.WAIT_OBJECT_0 = 0  # type: ignore[attr-defined]

        def wait(process: _Process, milliseconds: int) -> int:
            deadline = time.perf_counter() + milliseconds / 1000
            while process.pid in scm.alive and time.perf_counter() < deadline:
                time.sleep(0.01)
            return 0 if process.pid not in scm.alive else 258  # WAIT_TIMEOUT

        event.WaitForSingleObject = wait  # type: ignore[attr-defined]
        return {
            "pywintypes": pywintypes,
            "win32service": service,
            "win32serviceutil": util,
            "win32api": api,
            "win32con": con,
            "win32event": event,
        }


@pytest.fixture
def scm(monkeypatch: pytest.MonkeyPatch) -> FakeScm:
    fake = FakeScm()
    for name, module in fake.modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    return fake


def test_a_crashed_app_that_eugene_stops_is_not_started_again_by_windows(
    tmp_path: Path, scm: FakeScm
) -> None:
    """The race C4 lost, on the service manager's side: the stop of a service
    that has already crashed is refused (1062), and Windows' restart comes
    anyway, from whatever spec is on disk by then."""
    runner = app_accounts.WindowsServiceRunner(tmp_path)
    scm.run()
    scm.crash(recover_in=0.3)
    runner.stop("tiny")
    time.sleep(0.4)
    assert runner.state("tiny").state in ("stopped", "failed"), scm.events
    assert "recovery refused: disabled" in scm.events


def test_a_stop_that_does_not_finish_raises_and_names_the_state(
    tmp_path: Path, scm: FakeScm, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_accounts, "_STOP_TIMEOUT_SECONDS", 1.0)
    runner = app_accounts.WindowsServiceRunner(tmp_path)
    scm.run()
    scm.stop_takes_forever = True
    with pytest.raises(OSError, match=r"did not stop within 1 s \(Windows says stopping, process"):
        runner.stop("tiny")


def test_a_stop_returns_only_once_the_services_process_has_ended(
    tmp_path: Path, scm: FakeScm, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_accounts, "_STOP_TIMEOUT_SECONDS", 1.0)
    runner = app_accounts.WindowsServiceRunner(tmp_path)
    scm.run()
    pid = scm.pid
    scm.process_outlives = True
    with pytest.raises(OSError, match=rf"its process {pid} was still running"):
        runner.stop("tiny")

    # The same process ending a moment after Windows said STOPPED is a stop.
    scm.state, scm.pid = RUNNING, pid
    threading.Timer(0.3, lambda: scm.alive.discard(pid)).start()
    runner.stop("tiny")
    assert pid not in scm.alive


def test_a_stop_of_a_service_still_starting_waits_until_it_takes_the_stop(
    tmp_path: Path, scm: FakeScm
) -> None:
    runner = app_accounts.WindowsServiceRunner(tmp_path)
    scm.run()
    scm.state, scm.start_queries = START_PENDING, 3
    runner.stop("tiny")
    assert runner.state("tiny").state == "stopped"
    assert scm.events.count("stop") == 1


def test_a_service_is_held_until_its_spec_and_grants_are_in_place(
    tmp_path: Path, scm: FakeScm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing -- Windows' recovery included -- starts the service between
    the spec being written and the account being granted what it runs."""
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(
        app_accounts, "_icacls", lambda path, *args: seen.append((Path(path).name, scm.start_type))
    )
    apps_root = tmp_path / "apps"
    venv = apps_root / "tiny" / "versions" / "v1" / "venv"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {tmp_path / 'python'}\n", encoding="utf-8")
    plan = app_accounts.LaunchPlan(
        app_id="tiny",
        argv=[str(venv / "Scripts" / "python.exe"), "-m", "tiny_app"],
        env={},
        data_dir=apps_root / "tiny" / "data",
        key_file=apps_root / "tiny" / "data" / "client_key",
        admin_token="token",
        port=8190,
        bind_host=None,
    )
    runner = app_accounts.WindowsServiceRunner(apps_root)
    runner.prepare(plan, {"app": "tiny", "argv": plan.argv})
    assert seen and all(start_type == DISABLED for _, start_type in seen), seen
    assert scm.start_type == DEMAND
    assert runner.launched("tiny") == {"app": "tiny", "argv": plan.argv}


def test_a_start_windows_refuses_names_why(tmp_path: Path, scm: FakeScm) -> None:
    runner = app_accounts.WindowsServiceRunner(tmp_path)
    scm.start_type = DISABLED
    with pytest.raises(OSError, match=r"Windows would not start EugenePlexusApp-tiny: .*1058"):
        runner.start("tiny")


# --------------------------------------------------------------------- #
# an app kept across an agent restart, and what Go back says when it fails
# --------------------------------------------------------------------- #


class FakeDoors:
    """`client_doors.Doors`, recording where each door leads."""

    def __init__(self) -> None:
        self.opened: dict[str, tuple[int | None, int | None]] = {}

    async def open_pump(self, name: str, **kw: Any) -> None:
        self.opened[name] = (kw["plain_target"](), kw["tls_target"]())

    async def close(self, name: str) -> None:
        self.opened.pop(name, None)

    def plain_allowed(self) -> bool:
        return True


def test_an_app_kept_across_an_agent_restart_is_behind_a_door_that_leads_to_it(
    tmp_path: Path,
) -> None:
    """agent#32: the door was opened on two fresh ports and the kept app
    listened on the old ones, so it read `starting` for good."""
    from tests.test_app_accounts import FakeRunner

    manager = _manager(tmp_path)
    record = _record(port=_free_port())
    manager.store.put(record)
    python = _fake_python(manager.store, record)
    plain, tls = _free_port(), _free_port()
    runner = FakeRunner(manager.store.root, running=True)
    runner.token_file("tiny", tmp_path).parent.mkdir(parents=True, exist_ok=True)
    runner.token_file("tiny", tmp_path).write_text("the-token-it-was-started-with")
    runner.running_spec = {
        "argv": apps.entry_argv(python, record.manifest),
        "env": {
            "EUGENE_PLEXUS_APP_BIND_PORT": str(plain),
            "EUGENE_PLEXUS_APP_TLS_PORT": str(tls),
            "EUGENE_PLEXUS_APP_BIND_HOST": "127.0.0.1",
        },
        "values": {"port": str(plain), "bindHost": "127.0.0.1"},
    }
    supervisor = app_accounts.OwnAccountSupervisor(
        runner,  # type: ignore[arg-type]
        ingress=lambda: "",
        data_dir=manager.store.data_dir,
        key_file=manager.store.key_file,
    )
    manager.supervisor = supervisor
    doors = FakeDoors()
    manager._doors = lambda: doors

    async def scenario() -> None:
        await manager.start(record)
        await supervisor._starting["tiny"]
        assert supervisor.admin_token("tiny") == "the-token-it-was-started-with"
        await supervisor.stop_all()

    asyncio.run(scenario())
    assert doors.opened["app:tiny"] == (plain, tls)
    assert runner.calls == [], "kept, not restarted"


def test_a_restore_that_fails_is_the_apps_last_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from tests.test_app_update_backup import installed

    manager, _data, new = installed(tmp_path)
    asyncio.run(manager.installed_callback(new, AppOrigin.custom))

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk write failed")

    monkeypatch.setattr(shutil, "copy2", fail)
    record = manager.store.get("tiny")
    assert record is not None
    with pytest.raises(apps.AppInstallError, match="disk write failed"):
        asyncio.run(manager.rollback(record))
    error = manager.view(manager.store.get("tiny")).lastError  # type: ignore[arg-type]
    assert error and "not restored" in error and "disk write failed" in error


def test_a_stop_that_fails_during_go_back_is_the_apps_last_error(tmp_path: Path) -> None:
    manager, scm, new = _updated(tmp_path)
    supervisor = manager.supervisor
    assert isinstance(supervisor, app_accounts.OwnAccountSupervisor)

    def stuck(app_id: str) -> None:
        raise OSError("EugenePlexusApp-tiny did not stop within 45 s (Windows says stopping)")

    async def scenario() -> None:
        await manager.installed_callback(new, AppOrigin.custom)
        await supervisor._starting["tiny"]
        scm.stop = stuck  # type: ignore[method-assign]
        record = manager.store.get("tiny")
        assert record is not None
        with pytest.raises(apps.AppInstallError, match="did not stop within 45 s"):
            await manager.rollback(record)
        await supervisor.stop_all()

    asyncio.run(scenario())
    record = manager.store.get("tiny")
    assert record is not None and record.version == "v2", "nothing was restored over it"
    error = manager.view(record).lastError
    assert error and "did not stop within 45 s" in error


# --------------------------------------------------------------------- #
# the launcher: nothing the app started outlives it, and it says why it ends
# --------------------------------------------------------------------- #


def _process_alive(pid: int) -> bool:
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    try:
        return bool(kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0) != 0)
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


@pytest.mark.skipif(sys.platform != "win32", reason="systemd ends a unit's cgroup on Linux")
def test_a_process_the_app_started_ends_with_the_launcher(tmp_path: Path) -> None:
    """agent#34: a process the app leaves behind holds its port and its
    database after the service has stopped. It ends with the launcher."""
    import subprocess

    from eugene_plexus_agent import app_launcher
    from tests.test_app_accounts import _Ingress

    ingress = _Ingress()
    pid_file = tmp_path / "left.pid"
    child = tmp_path / "child.py"
    child.write_text(
        "import subprocess, sys\n"
        "left = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)\n"
        f"open({str(pid_file)!r}, 'w').write(str(left.pid))\n"
        "print('started a helper and left', flush=True)\n"
    )
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
    left = 0
    try:
        assert app_launcher.run(spec) == 0
        left = int(pid_file.read_text())
        deadline = time.perf_counter() + 10
        while _process_alive(left) and time.perf_counter() < deadline:
            time.sleep(0.1)
        assert not _process_alive(left), "the helper the app started outlived the launcher"
    finally:
        ingress.server.shutdown()
        if left and _process_alive(left):
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(left)], capture_output=True)
    assert any("still running after it; ended" in line for line in ingress.lines), ingress.lines


def test_an_unexpected_error_in_the_launcher_is_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_agent import app_launcher
    from tests.test_app_accounts import _Ingress

    ingress = _Ingress()
    key = tmp_path / "client_key"
    key.write_text("app-key")

    def boom(*_args: object) -> None:
        raise RuntimeError("a case nobody planned for")

    monkeypatch.setattr(app_launcher, "render_start", boom)
    app_launcher._STOP.clear()
    try:
        code = app_launcher.run(
            {"app": "tiny", "argv": ["x"], "env": {}, "keyFile": str(key), "ingress": ingress.url}
        )
    finally:
        ingress.server.shutdown()
    assert code == 1
    assert any(
        "unexpected error" in line and "a case nobody planned for" in line for line in ingress.lines
    ), ingress.lines


def test_a_taskkill_that_fails_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from eugene_plexus_agent import app_launcher

    class Stuck:
        pid = 4242

        def poll(self) -> None:
            return None

    monkeypatch.setattr(app_launcher, "WINDOWS", True)
    monkeypatch.setattr(
        app_launcher.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 128, "", "ERROR: Access is denied."),
    )
    note = app_launcher._force(Stuck())  # type: ignore[arg-type]
    assert note and "code 128" in note and "Access is denied" in note


def test_the_launchers_first_line_names_the_launch(tmp_path: Path) -> None:
    from eugene_plexus_agent import app_launcher
    from tests.test_app_accounts import _Ingress

    ingress = _Ingress()
    child = tmp_path / "child.py"
    child.write_text("print('hello', flush=True)\n")
    key = tmp_path / "client_key"
    key.write_text("app-key")
    spec = {
        "app": "tiny",
        "argv": [sys.executable, str(child)],
        "env": {},
        "keyFile": str(key),
        "ingress": ingress.url,
        "version": "0.11.4",
        "generation": "abcdef0123456789",
    }
    app_launcher._STOP.clear()
    try:
        assert app_launcher.run(spec) == 0
    finally:
        ingress.server.shutdown()
    assert ingress.lines[0] == "[launcher] starting tiny 0.11.4 (launch abcdef01)", ingress.lines


# --------------------------------------------------------------------- #
# the review of agent#34: schedules for the two races, and adoption
# --------------------------------------------------------------------- #


class WindowsLike:
    """A runner with Windows' rules, for the schedules the review named.

    A disabled service is one nothing starts. A crash queues a restart that
    fires at a chosen moment (`recover_at`: "hold", "state" or "prepare", as
    that call begins), from the spec on disk then. Start of a service that
    runs does nothing (1056). `prepare` writes the spec first and enables the
    service last; `prepare_gate` holds it in the middle.
    """

    kind = "windows_service"

    def __init__(self, apps_root: Path) -> None:
        self.apps_root = apps_root
        self.events: list[str] = []
        self.enabled = True
        self.current = app_accounts.ServiceState("stopped")
        self.loaded: dict[str, Any] | None = None
        self.pending_recovery = False
        self.recover_at: str | None = None
        self.prepare_gate: threading.Event | None = None
        self.prepare_entered = threading.Event()

    def _spec_file(self, app_id: str) -> Path:
        return self.apps_root / app_id / app_accounts.SPEC_FILE

    def _moment(self, name: str, app_id: str) -> None:
        if self.pending_recovery and self.recover_at == name:
            self.pending_recovery = False
            if self.enabled and self.current.state not in ("running", "starting"):
                self.events.append("recovery start")
                self._run(app_id)
            else:
                self.events.append("recovery refused")

    def _run(self, app_id: str) -> None:
        self.loaded = json.loads(self._spec_file(app_id).read_text(encoding="utf-8"))
        self.current = app_accounts.ServiceState("running", pid=4242)

    def crash(self) -> None:
        self.current = app_accounts.ServiceState("failed", exit_code=1)
        self.loaded = None
        self.pending_recovery = True

    def hold(self, app_id: str) -> None:
        self._moment("hold", app_id)
        self.enabled = False
        self.events.append("disabled")

    def release(self, app_id: str) -> None:
        self.enabled = True
        self.events.append("enabled")

    def state(self, app_id: str) -> app_accounts.ServiceState:
        self._moment("state", app_id)
        return self.current

    def launched(self, app_id: str) -> dict | None:
        try:
            return json.loads(self._spec_file(app_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def launched_at(self, app_id: str) -> float | None:
        try:
            return self._spec_file(app_id).stat().st_mtime
        except OSError:
            return None

    def prepare(self, plan: app_accounts.LaunchPlan, spec: dict) -> None:
        self._moment("prepare", plan.app_id)
        self.events.append("prepare begins")
        self.prepare_entered.set()
        if self.prepare_gate is not None:
            self.prepare_gate.wait(10)
        path = self._spec_file(plan.app_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec), encoding="utf-8")
        self.token_file(plan.app_id, plan.data_dir).write_text(plan.admin_token)
        self.enabled = True
        self.events.append("prepare ends, enabled")

    def start(self, app_id: str) -> None:
        if self.current.state in ("running", "starting"):
            self.events.append("start: it runs already")
            return
        if not self.enabled:
            raise OSError("Windows would not start it: the service is disabled (error 1058)")
        self.events.append("start")
        self._run(app_id)

    def stop(self, app_id: str) -> None:
        self.hold(app_id)
        self.events.append("stop")
        self.current = app_accounts.ServiceState("stopped")
        self.loaded = None

    def remove(self, app_id: str, *, purge: bool) -> None:
        self.stop(app_id)
        self.events.append("remove")

    def token_file(self, app_id: str, data_dir: Path) -> Path:
        return self.apps_root / app_id / app_accounts.ADMIN_TOKEN_FILE


def _two_versions(tmp_path: Path) -> tuple[apps.AppStore, apps.InstalledApp, apps.InstalledApp]:
    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    old = _record(_manifest(), port=_free_port())
    new = _record(_manifest(version="v2"), port=old.port)
    _fake_python(store, old)
    _fake_python(store, new)
    return store, old, new


def _windows_like(
    tmp_path: Path, store: apps.AppStore
) -> tuple[WindowsLike, app_accounts.OwnAccountSupervisor]:
    scm = WindowsLike(store.root)
    supervisor = app_accounts.OwnAccountSupervisor(
        scm,  # type: ignore[arg-type]
        ingress=lambda: "",
        data_dir=store.data_dir,
        key_file=store.key_file,
    )
    return scm, supervisor


def _as_started(store: apps.AppStore, record: apps.InstalledApp, *, token: str = "kept") -> None:
    """The spec and token an earlier start of `record` left, as that start wrote them."""
    port = str(record.port)
    spec = {
        "argv": apps.entry_argv(
            apps.venv_python(store.version_dir(record.id, record.version) / "venv"),
            record.manifest,
        ),
        "version": record.version,
        "env": {"EUGENE_PLEXUS_APP_BIND_PORT": port},
        "values": {"port": port, "bindHost": "127.0.0.1"},
    }
    path = store.root / record.id / app_accounts.SPEC_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec), encoding="utf-8")
    (store.root / record.id / app_accounts.ADMIN_TOKEN_FILE).write_text(token)


def _planner_for(store: apps.AppStore, record: apps.InstalledApp) -> apps._AppPlanner:
    return apps._AppPlanner(
        record, store=store, gateway_url=lambda: "http://gw", bind_host=lambda: None
    )


@pytest.mark.parametrize("recover_at", ["hold", "state", "prepare"])
def test_a_restart_windows_queued_before_a_start_never_runs_the_old_spec(
    tmp_path: Path, recover_at: str
) -> None:
    """Review blocker 2a: an older agent left v1's spec and a crashed service
    with a restart queued; this start is v2's. Wherever the restart lands --
    before the agent looks, as it looks, or as preparation begins -- what runs
    at the end is v2, started from v2's spec."""
    store, old, new = _two_versions(tmp_path)
    _as_started(store, old)
    scm, supervisor = _windows_like(tmp_path, store)
    scm.crash()
    scm.recover_at = recover_at

    async def scenario() -> None:
        supervisor.start(_planner_for(store, new))
        await supervisor._starting["tiny"]
        await supervisor.stop_all()

    asyncio.run(scenario())
    assert _version_of(scm.loaded) == "v2", scm.events
    assert "start: it runs already" not in scm.events, scm.events


@pytest.mark.parametrize("verb", ["stop", "remove"])
def test_a_stop_during_preparation_waits_for_it_and_leaves_the_app_stopped(
    tmp_path: Path, verb: str
) -> None:
    """Review blocker 2b: a stop (or an uninstall) that comes while a start is
    preparing waits for that step, thread and all, and the start goes no
    further. Before, the preparation finished after the stop and enabled the
    service again."""
    store, old, _new = _two_versions(tmp_path)
    scm, supervisor = _windows_like(tmp_path, store)
    gate = threading.Event()
    scm.prepare_gate = gate

    async def scenario() -> None:
        supervisor.start(_planner_for(store, old))
        await asyncio.to_thread(scm.prepare_entered.wait, 10)
        stopping = asyncio.create_task(
            supervisor.stop("tiny") if verb == "stop" else supervisor.remove("tiny", purge=False)
        )
        await asyncio.sleep(0.3)
        waited = not stopping.done()
        gate.set()
        await stopping
        await supervisor.stop_all()
        assert waited, f"the {verb} did not wait for the preparation: {scm.events}"

    try:
        asyncio.run(scenario())
    finally:
        gate.set()
    assert scm.events.index("prepare ends, enabled") < scm.events.index("stop"), scm.events
    assert "start" not in scm.events, scm.events
    assert not scm.enabled and scm.current.state == "stopped", scm.events


def test_the_agents_own_stop_waits_for_a_start_in_flight_then_says_what_it_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app_accounts, "_SHUTDOWN_WAIT_SECONDS", 0.3, raising=False)
    store, old, _new = _two_versions(tmp_path)
    scm, supervisor = _windows_like(tmp_path, store)
    gate = threading.Event()
    scm.prepare_gate = gate

    async def scenario() -> None:
        supervisor.start(_planner_for(store, old))
        await asyncio.to_thread(scm.prepare_entered.wait, 10)
        with caplog.at_level("WARNING"):
            await supervisor.stop_all()
        gate.set()
        await asyncio.sleep(0.3)

    try:
        asyncio.run(scenario())
    finally:
        gate.set()
    assert "still in prepare" in caplog.text, caplog.text
    assert "start" not in scm.events, scm.events


def test_a_service_still_stopping_is_never_kept(tmp_path: Path) -> None:
    """Review should-fix 1: an agent killed in a stop left the service
    stopping; the next start stops it and starts this one."""
    store, old, _new = _two_versions(tmp_path)
    _as_started(store, old)
    scm, supervisor = _windows_like(tmp_path, store)
    scm.current = app_accounts.ServiceState("stopping", pid=4242)

    async def scenario() -> None:
        supervisor.start(_planner_for(store, old))
        await supervisor._starting["tiny"]
        await supervisor.stop_all()

    asyncio.run(scenario())
    assert "stop" in scm.events and "prepare begins" in scm.events and "start" in scm.events


def test_a_kept_service_an_earlier_agent_held_is_released(tmp_path: Path) -> None:
    """Review should-fix 1: an agent killed after holding a service that still
    ran left it disabled. Kept, it is enabled again, so a crash of it is
    restarted like any other app's."""
    store, old, _new = _two_versions(tmp_path)
    _as_started(store, old)
    scm, supervisor = _windows_like(tmp_path, store)
    scm.current = app_accounts.ServiceState("running", pid=4242)
    scm.enabled = False

    async def scenario() -> None:
        supervisor.start(_planner_for(store, old))
        await supervisor._starting["tiny"]
        assert supervisor.admin_token("tiny") == "kept"
        await supervisor.stop_all()

    asyncio.run(scenario())
    assert "prepare begins" not in scm.events, "kept, not restarted"
    assert scm.enabled, scm.events


def test_start_starts_a_kept_app_whose_service_has_since_ended(tmp_path: Path) -> None:
    """Review should-fix 1: Start was a no-op for an app the agent supervised
    whose service had ended, since it asked only whether the agent had a
    planner for it."""
    store, old, _new = _two_versions(tmp_path)
    _as_started(store, old)
    scm, supervisor = _windows_like(tmp_path, store)
    scm.current = app_accounts.ServiceState("running", pid=4242)

    async def scenario() -> None:
        planner = _planner_for(store, old)
        supervisor.start(planner)
        await supervisor._starting["tiny"]
        scm.current = app_accounts.ServiceState("stopped")
        await supervisor._poll("tiny")
        assert not supervisor.is_running("tiny")
        supervisor.start(planner)
        await supervisor._starting["tiny"]
        await supervisor.stop_all()

    asyncio.run(scenario())
    assert scm.events[-2:] == ["prepare ends, enabled", "start"], scm.events


@pytest.mark.parametrize(
    "owners,started_before_spec,kept",
    [
        ({4243}, False, True),
        ({99999}, False, False),
        ({4243}, True, False),
        (set(), False, True),
    ],
)
def test_a_running_service_is_kept_only_if_it_is_what_listens_and_runs_this_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owners: set[int],
    started_before_spec: bool,
    kept: bool,
) -> None:
    """Review should-fix 3: the spec on disk is not proof of what runs. Kept
    only if the service started after its spec was written, and no process
    outside the service's own tree listens on its port (nothing listening yet
    is an app still starting)."""
    store, old, _new = _two_versions(tmp_path)
    _as_started(store, old)
    scm, supervisor = _windows_like(tmp_path, store)
    scm.current = app_accounts.ServiceState("running", pid=4242)
    written = scm.launched_at("tiny") or 0.0
    monkeypatch.setattr(app_accounts, "_process_tree", lambda pid: {4242, 4243}, raising=False)
    monkeypatch.setattr(app_accounts, "_listener_owners", lambda port: set(owners), raising=False)
    monkeypatch.setattr(
        app_accounts,
        "_process_started",
        lambda pid: written - 60 if started_before_spec else written + 1,
        raising=False,
    )

    async def scenario() -> None:
        supervisor.start(_planner_for(store, old))
        await supervisor._starting["tiny"]
        await supervisor.stop_all()

    asyncio.run(scenario())
    assert ("prepare begins" not in scm.events) is kept, scm.events


def test_a_launcher_that_cannot_hold_the_app_in_a_job_does_not_start_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review should-fix 4: without the job, a process the app starts can
    outlive a stop and hold its database while Go back restores it."""
    from eugene_plexus_agent import app_launcher

    class Suspended:
        pid = 4242
        killed = False

        def kill(self) -> None:
            self.killed = True

    child = Suspended()

    class NoJob:
        def __init__(self) -> None:
            raise OSError("CreateJobObjectW: access is denied")

    monkeypatch.setattr(app_launcher, "WINDOWS", True)
    monkeypatch.setattr(app_launcher, "_spawn", lambda *a, **k: child)
    monkeypatch.setattr(app_launcher, "_resume", lambda proc: None)
    monkeypatch.setattr(app_launcher, "_Job", NoJob)
    with pytest.raises(OSError, match="access is denied"):
        app_launcher._spawn_contained({}, ["x"], {})
    assert child.killed
