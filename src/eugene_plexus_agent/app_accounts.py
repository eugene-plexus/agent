"""Each app in an OS account of its own (C1).

Design: `specs/docs/design/workbench.md` §1-§2. Measured 2026-10-01 on
GitHub's runners (`specs/scripts/c1-app-accounts-acceptance.py`): an app
run by the agent's own supervisor runs as the agent's account --
LocalSystem on the Windows service install, `eugene-plexus` on the Linux
system install -- and reads `node.yaml`, `agent.yaml`, the passphrase or
keyring, the control root's log and snapshot, and every other app's key.
Rows 1-3 of the key-exposure work protect Eugene from *other* accounts;
an app is not another account until this module makes it one.

**The OS service manager runs the app, in an account it creates for it.**
Neither needs a password kept anywhere:

* **Windows service install:** one service per app, `EugenePlexusApp-<id>`,
  running as its virtual account `NT SERVICE\\EugenePlexusApp-<id>`. The
  agent is LocalSystem, so it creates, starts and stops services itself,
  and grants that account its own app directory and nothing else.
* **Linux system install:** the template unit `eugene-plexus-app@.service`
  (written by `install.sh`) runs each app as a systemd dynamic user, in a
  namespace where the install's prefix is an empty, read-only tmpfs with
  only the interpreters, the launcher and the app's own directory bound
  back in. Secrets arrive through `LoadCredential=`. The agent cannot
  start a system unit itself (it is unprivileged, and `NoNewPrivileges`
  rules out sudo), so it asks through a request file a root path unit
  watches -- the shape the in-app updater already uses.

The program the service manager runs is `app_launcher.py`, copied out of
this package to `<apps>/launcher/`, which forwards the app's output to
`POST /v1/logs`. Where no account can be made (a per-user Windows install,
Linux `--user`, macOS), `detect()` says why, the agent's own supervisor
runs the app as before, and `AppManager` installs only apps that declare
`localActions: false`.

**An app keeps running when the agent restarts.** It belongs to the
service manager, not to this process; the agent stopping is not the app
stopping, and the agent coming back finds it running.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.resources
import json
import logging
import os
import secrets
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx
import psutil

from ._generated.models import ComponentStatus
from ._http import internal_client
from ._private_files import write_private

if TYPE_CHECKING:
    from .apps import _AppPlanner

log = logging.getLogger(__name__)

LAUNCHER_DIR = "launcher"
LAUNCHER_FILE = "app_launcher.py"
#: Interpreters for app environments: apart from the agent's own, so an
#: app's account is granted these and never the agent's.
APP_PYTHONS = "pythons"
SPEC_FILE = "launch.json"
ADMIN_TOKEN_FILE = "admin_token"
#: Mirrors pps.OIDC_SECRET_FILE: the app's sign-in secret (C2), empty when it
#: signs nobody in. The Linux unit loads it as a credential.
OIDC_SECRET_FILE = "oidc_secret"
CTL_DIR = "ctl"

SERVICE_PREFIX = "EugenePlexusApp-"
UNIT_TEMPLATE = "eugene-plexus-app@"
UNIT_FILE = Path("/etc/systemd/system/eugene-plexus-app@.service")
CTL_PATH_UNIT = Path("/etc/systemd/system/eugene-plexus-apps-ctl.path")

_POLL_SECONDS = 1.5
#: The standard DELETE access right (winnt.h), for DeleteService.
_DELETE = 0x00010000
_CTL_TIMEOUT_SECONDS = 60.0
_STOP_TIMEOUT_SECONDS = 45.0
#: How long the agent's own stop waits for a start in flight to finish its step.
_SHUTDOWN_WAIT_SECONDS = 10.0
#: How long one grant may take. A grant on an app's directory is carried to
#: every file inside it, and its environment is most of them: Workbench's is
#: a few thousand files, Open WebUI's 245 packages and 1.8 GB (C4), with an
#: antivirus scanner looking at each change. Two minutes was sized for the
#: first and is a start that fails on a slow disk for the second; fifteen is
#: a wait no grant should reach, and a hung icacls still ends.
_GRANT_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True)
class AccountSupport:
    """Whether this node gives apps accounts of their own, and how."""

    kind: Literal["windows_service", "systemd"] | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.kind is not None


def detect(mechanism: object | None = None) -> AccountSupport:
    """From how this agent was started, which is what decides it."""
    from ._generated.models import InstallMechanism
    from .install_info import mechanism as current

    found = mechanism if mechanism is not None else current()
    if found == InstallMechanism.windows_service:
        try:
            import win32service  # noqa: F401
        except ImportError:
            return AccountSupport(
                None,
                "This Windows service install is missing pywin32, which the agent uses to give "
                "apps their own accounts. Run install.ps1 again as Administrator to repair it.",
            )
        return AccountSupport("windows_service")
    if found == InstallMechanism.systemd_system:
        if not UNIT_FILE.is_file() or not CTL_PATH_UNIT.is_file():
            return AccountSupport(
                None,
                f"This system install has no {UNIT_FILE.name}, which an older install.sh did not "
                "write. Run install.sh again with sudo to give apps their own accounts.",
            )
        return AccountSupport("systemd")
    if found == InstallMechanism.windows_task:
        return AccountSupport(
            None,
            "This install runs as you, from a sign-in task, so it cannot create accounts. "
            "Install Eugene as the machine's service (run install.ps1 as Administrator) to give "
            "apps accounts of their own.",
        )
    if found == InstallMechanism.systemd_user:
        return AccountSupport(
            None,
            "This install runs as you (install.sh --user), so it cannot create accounts. "
            "Install Eugene as the machine's service (install.sh with sudo) to give apps "
            "accounts of their own.",
        )
    if found == InstallMechanism.launchd:
        return AccountSupport(
            None,
            "On macOS, Eugene runs as you and cannot run apps under accounts of their own yet.",
        )
    if found == InstallMechanism.container:
        return AccountSupport(
            None,
            "In a container, apps run as the container's account; there is no other "
            "account to hand them.",
        )
    return AccountSupport(
        None,
        "This agent was not started by an installed service, so it cannot create "
        "accounts for apps.",
    )


def account_name(kind: str, app_id: str) -> str:
    if kind == "windows_service":
        return f"NT SERVICE\\{SERVICE_PREFIX}{app_id}"
    return f"{dynamic_user(app_id)} (the dynamic user of {UNIT_TEMPLATE}{app_id}.service)"


def dynamic_user(app_id: str) -> str:
    """The user `install.sh`'s helper gives one app's unit: `eapp-` and a
    hash of the id, because systemd takes names of at most 31 characters
    and an id may be 40. One per app, never the template's shared name."""
    return "eapp-" + hashlib.sha256(app_id.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# what the launcher is given
# --------------------------------------------------------------------------- #


def install_launcher(apps_root: Path) -> Path:
    """`<apps>/launcher/app_launcher.py`, rewritten only when it differs."""
    text = (
        importlib.resources.files("eugene_plexus_agent").joinpath(LAUNCHER_FILE).read_text("utf-8")
    )
    target = apps_root / LAUNCHER_DIR / LAUNCHER_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.read_text(encoding="utf-8") != text:
        temp = target.with_suffix(".tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, target)
    return target


def base_interpreter(venv: Path) -> Path:
    """The interpreter an app's venv was built from (`pyvenv.cfg`'s `home`).

    The Windows service runs the launcher with this rather than the venv's
    own `python.exe`, which is a redirector that starts the real one as a
    child: a service must be the process the service manager started.
    """
    home: str | None = None
    with contextlib.suppress(OSError):
        for line in (venv / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip().lower() == "home":
                home = value.strip()
    if not home:
        raise FileNotFoundError(f"{venv / 'pyvenv.cfg'} names no base interpreter")
    name = "python.exe" if sys.platform == "win32" else "python3"
    return Path(home) / name


@dataclass
class LaunchPlan:
    """What one start of one app needs, from `_AppPlanner.plan()`."""

    app_id: str
    argv: list[str]
    env: dict[str, str]
    data_dir: Path
    key_file: Path
    admin_token: str
    port: int
    bind_host: str | None
    #: The manifest's own start, not filled in (`_AppPlanner.launch`): what
    #: the launcher fills in inside the account. Holds no secret.
    launch: dict = field(default_factory=dict)
    #: The app's version, for the spec and the launcher's first line.
    version: str | None = None


#: Variables the launcher is given in `launch.json`. Everything else comes
#: from the service manager's own environment for the account: the agent's
#: environment says where the agent's things are, which is the point.
_PASSED = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")


#: The variables that say where an app listens. A running service whose spec
#: differs in one of them is not the one the agent's door and health poll
#: expect (agent#32, agent#34).
_PLACE = (
    "EUGENE_PLEXUS_APP_BIND_PORT",
    "EUGENE_PLEXUS_APP_TLS_PORT",
    "EUGENE_PLEXUS_APP_BIND_HOST",
)


def launch_spec(
    plan: LaunchPlan, *, kind: str, ingress: str, generation: str | None = None
) -> dict:
    """What `launch.json` holds. `generation` names this one launch: the
    launcher's first line says it, so the Logs page tells two starts apart."""
    env = {k: v for k, v in plan.env.items() if k.startswith("EUGENE_PLEXUS_APP_") or k in _PASSED}
    # A fixed local transport for app-only release calls; never inferred from
    # the control root's gateway or a browser's address.
    env["EUGENE_PLEXUS_APP_AGENT_URL"] = ingress.removesuffix("/v1/logs")
    # The secrets never go in a file the account can read beside its
    # spec: on Windows they are files in its own data directory, on Linux
    # systemd hands them over as credentials.
    for name in (
        "EUGENE_PLEXUS_APP_ADMIN_TOKEN",
        "EUGENE_PLEXUS_APP_KEY_FILE",
        "EUGENE_PLEXUS_APP_DATA_DIR",
        "EUGENE_PLEXUS_APP_OIDC_SECRET_FILE",
    ):
        env.pop(name, None)
    spec: dict = {
        **plan.launch,
        "app": plan.app_id,
        "accountKind": kind,
        "argv": plan.argv,
        "env": env,
        "ingress": ingress,
    }
    if plan.version:
        spec["version"] = plan.version
    if generation:
        spec["generation"] = generation
    link_url = env.pop("EUGENE_PLEXUS_APP_MEMORY_SERVER_URL", None)
    if link_url:
        token_path = env.pop("EUGENE_PLEXUS_APP_MEMORY_SERVER_TOKEN_FILE")
        model_path = env.pop("EUGENE_PLEXUS_APP_MEMORY_SERVER_MODEL_FILE")
        spec["memoryServer"] = {"url": link_url, "modelFile": model_path}
        if kind == "windows_service":
            spec["memoryServer"]["tokenFile"] = token_path
    if kind == "windows_service":
        spec["service"] = f"{SERVICE_PREFIX}{plan.app_id}"
        spec["dataDir"] = str(plan.data_dir)
        spec["keyFile"] = str(plan.key_file)
        spec["adminTokenFile"] = str(plan.data_dir / ADMIN_TOKEN_FILE)
        spec["oidcSecretFile"] = str(plan.data_dir / OIDC_SECRET_FILE)
        if plan.launch.get("linkProvider"):
            spec["linkSecretFile"] = str(plan.data_dir.parent / "memory_server_token")
    return spec


# --------------------------------------------------------------------------- #
# the two service managers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServiceState:
    """What the service manager says, read on every poll."""

    state: Literal["running", "starting", "stopping", "stopped", "failed", "missing"]
    pid: int | None = None
    exit_code: int | None = None
    detail: str | None = None


class Runner(Protocol):
    kind: str

    def prepare(self, plan: LaunchPlan, spec: dict) -> None: ...
    def start(self, app_id: str) -> None: ...
    def hold(self, app_id: str) -> None: ...
    def release(self, app_id: str) -> None: ...
    def stop(self, app_id: str) -> None: ...
    def state(self, app_id: str) -> ServiceState: ...
    def launched(self, app_id: str) -> dict | None: ...
    def launched_at(self, app_id: str) -> float | None: ...
    def remove(self, app_id: str, *, purge: bool) -> None: ...
    def token_file(self, app_id: str, data_dir: Path) -> Path: ...


def _read_spec(path: Path) -> dict | None:
    """The `launch.json` a service starts from, or None when there is none."""
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return spec if isinstance(spec, dict) else None


def _written_at(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# What runs, as the OS says it: for keeping a service found running only when
# it is what its spec says (agent#34's review). Module functions, so a test
# can stand in for the OS.


def _process_tree(pid: int) -> set[int]:
    """`pid` and every process under it; empty when it has gone."""
    try:
        root = psutil.Process(pid)
        return {pid, *(child.pid for child in root.children(recursive=True))}
    except psutil.Error:
        return set()


def _listener_owners(port: int) -> set[int | None]:
    """The processes listening on `port`. None for one whose owner this
    account cannot see: on Linux, another account's socket."""
    try:
        connections = psutil.net_connections(kind="tcp")
    except psutil.Error:
        return set()
    return {
        conn.pid
        for conn in connections
        if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port
    }


def _process_started(pid: int) -> float | None:
    """When `pid` started, in the clock file times use; None when gone."""
    try:
        return float(psutil.Process(pid).create_time())
    except psutil.Error:
        return None


def _why(exc: Exception) -> str:
    """A pywin32 error in words: its message and its Windows error number."""
    words = getattr(exc, "strerror", None) or str(exc)
    number = getattr(exc, "winerror", None)
    return f"{words} (error {number})" if number is not None else words


def _icacls(path: Path, *args: str) -> None:
    out = subprocess.run(
        ["icacls", str(path), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=_GRANT_TIMEOUT_SECONDS,
    )
    if out.returncode != 0:
        raise OSError(f"icacls {path} {' '.join(args)} failed: {(out.stdout + out.stderr).strip()}")


class WindowsServiceRunner:
    """One Windows service per app, as its virtual account."""

    kind = "windows_service"

    def __init__(self, apps_root: Path) -> None:
        self._apps = apps_root

    def _name(self, app_id: str) -> str:
        return f"{SERVICE_PREFIX}{app_id}"

    def prepare(self, plan: LaunchPlan, spec: dict) -> None:
        import pywintypes
        import win32service

        name = self._name(plan.app_id)
        launcher = install_launcher(self._apps)
        app_dir = self._apps / plan.app_id
        spec_path = app_dir / SPEC_FILE
        python = base_interpreter(Path(plan.argv[0]).parent.parent)
        command = f'"{python}" -I -u "{launcher}" "{spec_path}"'
        write_private(spec_path, json.dumps(spec, indent=2))
        plan.data_dir.mkdir(parents=True, exist_ok=True)
        write_private(plan.data_dir / ADMIN_TOKEN_FILE, plan.admin_token)
        _ensure_secret_file(plan.data_dir)

        # Disabled until the end of this, so nothing starts the service --
        # Windows' recovery included -- before its spec, its account and its
        # grants are all in place for this start (agent#34).
        manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
        try:
            try:
                handle = win32service.CreateService(
                    manager,
                    name,
                    f"Eugene Plexus app: {plan.app_id}",
                    win32service.SERVICE_ALL_ACCESS,
                    win32service.SERVICE_WIN32_OWN_PROCESS,
                    win32service.SERVICE_DISABLED,
                    win32service.SERVICE_ERROR_NORMAL,
                    command,
                    None,
                    0,
                    None,
                    account_name(self.kind, plan.app_id),
                    None,
                )
            except pywintypes.error as exc:
                if exc.winerror != 1073:  # ERROR_SERVICE_EXISTS
                    raise
                handle = win32service.OpenService(manager, name, win32service.SERVICE_ALL_ACCESS)
                win32service.ChangeServiceConfig(
                    handle,
                    win32service.SERVICE_NO_CHANGE,
                    win32service.SERVICE_DISABLED,
                    win32service.SERVICE_NO_CHANGE,
                    command,
                    None,
                    0,
                    None,
                    account_name(self.kind, plan.app_id),
                    None,
                    None,
                )
            try:
                win32service.ChangeServiceConfig2(
                    handle,
                    win32service.SERVICE_CONFIG_DESCRIPTION,
                    "An optional app Eugene Plexus installed, run in an account of its own.",
                )
                # Restart a crash after 5 s. Windows repeats the last action
                # for every later failure, so an app that keeps crashing is
                # started again every few seconds until the agent holds it
                # (`hold`); the app's page shows it crashed meanwhile.
                win32service.ChangeServiceConfig2(
                    handle,
                    win32service.SERVICE_CONFIG_FAILURE_ACTIONS,
                    {
                        "ResetPeriod": 86400,
                        "RebootMsg": None,
                        "Command": None,
                        "Actions": [
                            (win32service.SC_ACTION_RESTART, 5000),
                            (win32service.SC_ACTION_RESTART, 5000),
                            (win32service.SC_ACTION_RESTART, 5000),
                        ],
                    },
                )
                # A launcher that stops with the app's non-zero code is a
                # failure too, not only a process that vanished.
                win32service.ChangeServiceConfig2(
                    handle, win32service.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG, True
                )
            finally:
                win32service.CloseServiceHandle(handle)
        finally:
            win32service.CloseServiceHandle(manager)

        # The account exists now; give it its own directory and the shared
        # code it runs, and nothing else.
        account = account_name(self.kind, plan.app_id)
        _icacls(app_dir, "/grant:r", f"{account}:(OI)(CI)(RX)")
        _icacls(plan.data_dir, "/grant:r", f"{account}:(OI)(CI)(M)")
        _icacls(launcher.parent, "/grant", f"{account}:(OI)(CI)(RX)")
        pythons = self._apps / APP_PYTHONS
        if pythons.is_dir():
            _icacls(pythons, "/grant", f"{account}:(OI)(CI)(RX)")
        # Prepared: from here a start, Windows' recovery's too, starts this.
        self._set_start_type(plan.app_id, win32service.SERVICE_DEMAND_START)

    def _set_start_type(self, app_id: str, start_type: int) -> bool:
        """Change only how the service may be started. False when there is
        no such service."""
        import pywintypes
        import win32service

        manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
        try:
            try:
                handle = win32service.OpenService(
                    manager, self._name(app_id), win32service.SERVICE_CHANGE_CONFIG
                )
            except pywintypes.error as exc:
                if exc.winerror == 1060:  # ERROR_SERVICE_DOES_NOT_EXIST
                    return False
                raise
            try:
                win32service.ChangeServiceConfig(
                    handle,
                    win32service.SERVICE_NO_CHANGE,
                    start_type,
                    win32service.SERVICE_NO_CHANGE,
                    None,
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                )
            finally:
                win32service.CloseServiceHandle(handle)
        finally:
            win32service.CloseServiceHandle(manager)
        return True

    def hold(self, app_id: str) -> None:
        """Keep the service from starting until the next `prepare`.

        Windows starts a crashed app's service again 5 s after it stops (the
        failure actions above), from whatever `launch.json` says by then. A
        stop does not cancel that: the stop of a service that has already
        stopped is refused (error 1062) and the restart comes anyway. During
        an update or Go back it started the version being replaced, and the
        agent kept it (agent#34). A disabled service is one nothing starts.
        """
        import pywintypes
        import win32service

        try:
            self._set_start_type(app_id, win32service.SERVICE_DISABLED)
        except pywintypes.error as exc:
            raise OSError(
                f"Windows would not keep {self._name(app_id)} from starting: {_why(exc)}"
            ) from exc

    def start(self, app_id: str) -> None:
        import pywintypes
        import win32serviceutil

        try:
            win32serviceutil.StartService(self._name(app_id))
        except pywintypes.error as exc:
            if exc.winerror != 1056:  # ERROR_SERVICE_ALREADY_RUNNING
                raise OSError(f"Windows would not start {self._name(app_id)}: {_why(exc)}") from exc

    def stop(self, app_id: str) -> None:
        """Stop the service; return only when it and its process are gone.

        Held first, so nothing starts it again meanwhile. Raises when it is
        not gone within `_STOP_TIMEOUT_SECONDS`: what comes next -- Go back's
        restore, an update's backup, the next version's start -- must not
        work over the files and ports of a process that still runs.
        """
        import pywintypes
        import win32service
        import win32serviceutil

        name = self._name(app_id)
        self.hold(app_id)
        deadline = time.perf_counter() + _STOP_TIMEOUT_SECONDS
        processes: dict[int, object | None] = {}
        try:
            while True:
                current = self.state(app_id)
                if current.pid and current.pid not in processes:
                    # Held open from here, so the wait below is for this
                    # process and never for another that takes its number.
                    processes[current.pid] = self._open_process(current.pid)
                if current.state in ("stopped", "failed", "missing"):
                    break
                if current.state in ("running", "starting"):
                    try:
                        win32serviceutil.ControlService(name, win32service.SERVICE_CONTROL_STOP)
                    except pywintypes.error as exc:
                        if exc.winerror == 1060:  # deleted meanwhile
                            return
                        # 1061: still starting, not taking a stop yet; asked
                        # again next time round. 1062: it has just stopped.
                        if exc.winerror not in (1061, 1062):
                            raise OSError(f"Windows would not stop {name}: {_why(exc)}") from exc
                if time.perf_counter() >= deadline:
                    which = f", process {current.pid}" if current.pid else ""
                    raise OSError(
                        f"{name} did not stop within {_STOP_TIMEOUT_SECONDS:g} s "
                        f"(Windows says {current.state}{which})"
                    )
                time.sleep(0.5)
            for pid, process in processes.items():
                remaining = max(0.0, deadline - time.perf_counter())
                if process is not None and not self._wait_process(process, remaining):
                    raise OSError(
                        f"{name} stopped, but its process {pid} was still running "
                        f"{_STOP_TIMEOUT_SECONDS:g} s after the stop was asked for"
                    )
        finally:
            for process in processes.values():
                self._close_process(process)

    @staticmethod
    def _open_process(pid: int) -> object | None:
        """A handle to wait on `pid` with, or None when it has gone already."""
        import pywintypes
        import win32api
        import win32con

        try:
            handle: object = win32api.OpenProcess(win32con.SYNCHRONIZE, False, pid)
        except pywintypes.error:
            return None
        return handle

    @staticmethod
    def _wait_process(process: object, seconds: float) -> bool:
        """True once the process has ended; False if it still runs after `seconds`."""
        import win32event

        waited = win32event.WaitForSingleObject(process, int(seconds * 1000))
        return bool(waited == win32event.WAIT_OBJECT_0)

    @staticmethod
    def _close_process(process: object | None) -> None:
        if process is not None:
            with contextlib.suppress(Exception):
                process.Close()  # type: ignore[attr-defined]

    def launched(self, app_id: str) -> dict | None:
        return _read_spec(self._apps / app_id / SPEC_FILE)

    def launched_at(self, app_id: str) -> float | None:
        return _written_at(self._apps / app_id / SPEC_FILE)

    def release(self, app_id: str) -> None:
        """Undo `hold` for a service the agent keeps running: an agent that
        ended between holding a service and stopping it left it disabled,
        and a crash of a disabled service is never restarted."""
        import pywintypes
        import win32service

        try:
            self._set_start_type(app_id, win32service.SERVICE_DEMAND_START)
        except pywintypes.error as exc:
            raise OSError(
                f"Windows would not let {self._name(app_id)} be started again: {_why(exc)}"
            ) from exc

    def state(self, app_id: str) -> ServiceState:
        import pywintypes
        import win32service

        try:
            manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
        except pywintypes.error as exc:
            return ServiceState("missing", detail=str(exc))
        try:
            try:
                handle = win32service.OpenService(
                    manager, self._name(app_id), win32service.SERVICE_QUERY_STATUS
                )
            except pywintypes.error:
                return ServiceState("missing")
            try:
                status = win32service.QueryServiceStatusEx(handle)
            finally:
                win32service.CloseServiceHandle(handle)
        finally:
            win32service.CloseServiceHandle(manager)
        current = status["CurrentState"]
        pid = status.get("ProcessId") or None
        code = status.get("ServiceSpecificExitCode") or status.get("Win32ExitCode") or 0
        if current == win32service.SERVICE_RUNNING:
            return ServiceState("running", pid)
        if current in (win32service.SERVICE_START_PENDING, win32service.SERVICE_CONTINUE_PENDING):
            return ServiceState("starting", pid)
        if current == win32service.SERVICE_STOP_PENDING:
            return ServiceState("stopping", pid)
        return ServiceState("failed" if code else "stopped", exit_code=code or None)

    def token_file(self, app_id: str, data_dir: Path) -> Path:
        return data_dir / ADMIN_TOKEN_FILE

    def remove(self, app_id: str, *, purge: bool) -> None:
        import pywintypes
        import win32service

        self.stop(app_id)
        account = account_name(self.kind, app_id)
        # Shared folders keep no grant for an account that is gone.
        for shared in (self._apps / LAUNCHER_DIR, self._apps / APP_PYTHONS):
            if shared.is_dir():
                with contextlib.suppress(OSError):
                    _icacls(shared, "/remove:g", account)
        manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
        try:
            try:
                # DELETE is a standard right, not a service one: win32con has it,
                # win32service does not (C1's first Windows run left the service).
                handle = win32service.OpenService(manager, self._name(app_id), _DELETE)
            except pywintypes.error:
                return
            try:
                win32service.DeleteService(handle)
            finally:
                win32service.CloseServiceHandle(handle)
        finally:
            win32service.CloseServiceHandle(manager)


class SystemdRunner:
    """`eugene-plexus-app@<id>.service`, asked for through the root broker."""

    kind = "systemd"

    def __init__(self, apps_root: Path) -> None:
        self._apps = apps_root

    def _unit(self, app_id: str) -> str:
        return f"{UNIT_TEMPLATE}{app_id}.service"

    def _ask(self, verb: str, app_id: str) -> None:
        """One request to `eugene-plexus-apps-ctl`, waited for.

        The helper reads the request as this account, checks the verb and
        the id itself, and writes its answer back as this account; see
        `install.sh`.
        """
        ctl = self._apps / CTL_DIR
        ctl.mkdir(parents=True, exist_ok=True)
        ident = uuid.uuid4().hex
        request = ctl / f"{ident}.req"
        answer = ctl / f"{ident}.res"
        temp = ctl / f".{ident}.tmp"
        temp.write_text(f"{verb} {app_id}\n", encoding="utf-8")
        os.replace(temp, request)
        deadline = time.perf_counter() + _CTL_TIMEOUT_SECONDS
        while time.perf_counter() < deadline:
            if answer.is_file():
                try:
                    result = json.loads(answer.read_text(encoding="utf-8") or "{}")
                finally:
                    with contextlib.suppress(OSError):
                        answer.unlink()
                if int(result.get("code", 1)) != 0:
                    raise OSError(
                        f"systemctl {verb} {self._unit(app_id)} failed: "
                        f"{str(result.get('output') or '').strip()[-400:]}"
                    )
                return
            time.sleep(0.2)
        with contextlib.suppress(OSError):
            request.unlink()
        raise OSError(
            f"nothing answered the request to {verb} {self._unit(app_id)} within "
            f"{int(_CTL_TIMEOUT_SECONDS)} s. Check that eugene-plexus-apps-ctl.path is enabled "
            "(systemctl status eugene-plexus-apps-ctl.path)."
        )

    def prepare(self, plan: LaunchPlan, spec: dict) -> None:
        launcher = install_launcher(self._apps)
        app_dir = self._apps / plan.app_id
        # Every unit loads these, empty for an app with no authorised link.
        for name in ("memory_server_token", "memory_server_model.json"):
            path = app_dir / name
            if not path.exists():
                write_private(path, "")
        # Readable by the app's dynamic user, which is no owner or group of
        # anything here. Safe because the prefix itself is 0750: only the
        # unit's bind mounts reach inside it.
        (app_dir / SPEC_FILE).write_text(json.dumps(spec, indent=2), encoding="utf-8")
        os.chmod(app_dir / SPEC_FILE, 0o644)
        write_private(app_dir / ADMIN_TOKEN_FILE, plan.admin_token)
        # The unit loads it as a credential and will not start without it.
        _ensure_secret_file(plan.data_dir)
        # The template unit runs `<apps>/<id>/python`: this version's
        # interpreter, swapped in one rename when the version changes.
        link = app_dir / "python"
        temp = app_dir / ".python.tmp"
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()
        temp.symlink_to(Path(plan.argv[0]).relative_to(app_dir))
        os.replace(temp, link)
        for tree in (app_dir / "versions", launcher.parent, self._apps / APP_PYTHONS):
            if tree.exists():
                _open_for_others(tree)
        # The app's own directory is bound into its namespace as itself, so
        # its dynamic user must be able to pass through it. `data/` inside
        # stays 0700: the key reaches the app as a credential instead.
        os.chmod(app_dir, os.stat(app_dir).st_mode | 0o011)

    def start(self, app_id: str) -> None:
        self._ask("start", app_id)

    def hold(self, app_id: str) -> None:
        """Nothing to do: `systemctl stop` cancels a restart `Restart=` has
        queued, and nothing but the agent starts the unit."""
        return None

    def stop(self, app_id: str) -> None:
        # `systemctl stop` returns when the unit has stopped, and systemd
        # ends the rest of its cgroup with it (KillMode=mixed).
        self._ask("stop", app_id)

    def launched(self, app_id: str) -> dict | None:
        return _read_spec(self._apps / app_id / SPEC_FILE)

    def launched_at(self, app_id: str) -> float | None:
        return _written_at(self._apps / app_id / SPEC_FILE)

    def release(self, app_id: str) -> None:
        return None

    def state(self, app_id: str) -> ServiceState:
        out = subprocess.run(
            [
                "systemctl",
                "show",
                self._unit(app_id),
                "--property=LoadState,ActiveState,SubState,MainPID,ExecMainStatus,Result",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        props = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
        if props.get("LoadState") == "not-found":
            return ServiceState("missing")
        active = props.get("ActiveState", "")
        pid = int(props.get("MainPID") or 0) or None
        code = int(props.get("ExecMainStatus") or 0)
        if active == "active":
            return ServiceState("running", pid)
        if active in ("activating", "reloading"):
            return ServiceState("starting", pid)
        if active == "deactivating":
            return ServiceState("stopping", pid)
        if active == "failed" or props.get("Result", "success") != "success":
            return ServiceState("failed", exit_code=code or None, detail=props.get("Result"))
        return ServiceState("stopped", exit_code=code or None)

    def remove(self, app_id: str, *, purge: bool) -> None:
        self.stop(app_id)
        if purge:
            self._ask("clean", app_id)

    def token_file(self, app_id: str, data_dir: Path) -> Path:
        return self._apps / app_id / ADMIN_TOKEN_FILE


def _ensure_secret_file(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if not (data_dir / OIDC_SECRET_FILE).exists():
        write_private(data_dir / OIDC_SECRET_FILE, "")


def _open_for_others(tree: Path) -> None:
    """Directories passable and listable, files readable, through a tree."""
    for root, _dirs, files in os.walk(tree):
        os.chmod(root, os.stat(root).st_mode | 0o055)
        for name in files:
            path = os.path.join(root, name)
            if os.path.islink(path):
                continue
            mode = os.stat(path).st_mode
            os.chmod(path, mode | 0o044 | (0o011 if mode & 0o100 else 0))


def runner_for(support: AccountSupport, apps_root: Path) -> Runner:
    if support.kind == "windows_service":
        return WindowsServiceRunner(apps_root)
    if support.kind == "systemd":
        return SystemdRunner(apps_root)
    raise ValueError(f"no runner for {support!r}")


# --------------------------------------------------------------------------- #
# the supervisor the apps code talks to
# --------------------------------------------------------------------------- #


class _Withdrawn(Exception):
    """A stop, a removal or the agent's own stop took a start back."""


class OwnAccountSupervisor:
    """`AppSupervisor`'s interface, for apps the service manager runs.

    It starts and stops through the runner, polls the service manager and
    the app's `healthPath`, and reports in the same five fields. What it does
    not do is stop apps when the agent stops: they are the service
    manager's, and an agent restart is not an app restart.
    """

    def __init__(
        self,
        runner: Runner,
        *,
        ingress: Callable[[], str],
        data_dir: Callable[[str], Path],
        key_file: Callable[[str], Path],
    ) -> None:
        self.runner = runner
        self._ingress = ingress
        self._data_dir = data_dir
        self._key_file = key_file
        self._planners: dict[str, _AppPlanner] = {}
        self._state: dict[str, ServiceState] = {}
        self._reachable: dict[str, bool] = {}
        self._errors: dict[str, str | None] = {}
        self._started_at: dict[str, datetime] = {}
        self._hosts: dict[str, str | None] = {}
        self._tokens: dict[str, str | None] = {}
        self._starting: dict[str, asyncio.Task[None]] = {}
        self._poll_task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        #: Apps this agent stopped and has not started since: an update, Go
        #: back, a restore, a restart or a person's Stop. What runs of one of
        #: them was not started by the agent, so a start never keeps it.
        self._held: set[str] = set()
        #: The `generation` of the spec each app was last started or kept with.
        self._generations: dict[str, str | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        #: The start each app's start task is for; a stop takes it back.
        self._tickets: dict[str, object] = {}
        #: The call to the service manager a start is in, for the log.
        self._doing: dict[str, str] = {}

    @property
    def kind(self) -> str:
        return self.runner.kind

    def account(self, app_id: str) -> str:
        return account_name(self.runner.kind, app_id)

    # --- lifecycle --------------------------------------------------------

    def start(self, planner: _AppPlanner) -> None:
        app_id = planner.record.id
        if app_id in self._planners and not self._exited(app_id):
            return
        self._planners[app_id] = planner
        self._reachable[app_id] = False
        self._errors[app_id] = None
        # What the service manager said of the last run is not this start's.
        self._state.pop(app_id, None)
        ticket = object()
        self._tickets[app_id] = ticket
        self._starting[app_id] = asyncio.create_task(
            self._start(planner, ticket), name=f"app-start-{app_id}"
        )
        if self._poll_task is None:
            self._client = internal_client(timeout=2.0)
            self._poll_task = asyncio.create_task(self._poll_loop(), name="app-accounts-poll")

    def _lock(self, app_id: str) -> asyncio.Lock:
        """One lifecycle operation at a time per app: a start, a stop, a
        removal, a pg0 cleanup. Each waits for the one before it, the call
        that one has in a thread included (agent#34's review: a preparation
        cancelled by a stop finished after it, and enabled the service again)."""
        return self._locks.setdefault(app_id, asyncio.Lock())

    async def _step(self, app_id: str, ticket: object, what: str, call: Any, *args: Any) -> Any:
        """One call of a start to the service manager, in a thread, if the
        start is still wanted. A start taken back stops between steps, never
        inside one: a thread cannot be stopped, so the step is let finish."""
        if self._tickets.get(app_id) is not ticket:
            raise _Withdrawn(what)
        self._doing[app_id] = what
        try:
            return await asyncio.to_thread(call, *args)
        finally:
            self._doing.pop(app_id, None)

    async def _start(self, planner: _AppPlanner, ticket: object) -> None:
        app_id = planner.record.id
        async with self._lock(app_id):
            try:
                await self._start_held(planner, ticket)
            except _Withdrawn as exc:
                log.info("app %s: its start was taken back before %s", app_id, exc)
            except Exception as exc:
                from .app_launcher import redact_database

                cause = redact_database(str(exc))
                self._errors[app_id] = f"could not start in an account of its own: {cause}"
                log.error("app %s could not start in its own account: %s", app_id, cause)

    async def _start_held(self, planner: _AppPlanner, ticket: object) -> None:
        app_id = planner.record.id
        if self._tickets.get(app_id) is not ticket:
            raise _Withdrawn("it began")
        # Not `plan()`: that fills in the app's secrets, which only the
        # launcher does here, inside the app's account (C4).
        spawn = planner.base_plan()
        plan = LaunchPlan(
            app_id=app_id,
            argv=list(spawn.argv),
            env=dict(spawn.env),
            data_dir=self._data_dir(app_id),
            key_file=self._key_file(app_id),
            admin_token=planner.admin_token or secrets.token_urlsafe(32),
            port=planner.record.port,
            bind_host=spawn.env.get("EUGENE_PLEXUS_APP_BIND_HOST"),
            launch=planner.launch(),
            version=planner.record.version,
        )
        self._hosts[app_id] = plan.bind_host or "127.0.0.1"
        generation = uuid.uuid4().hex
        spec = launch_spec(
            plan, kind=self.runner.kind, ingress=self._ingress(), generation=generation
        )
        # Held before anything: from here nothing starts the service but
        # this start -- a restart Windows queued for an earlier crash
        # included. One that came before is seen running below, and is kept
        # or stopped before a file of this start is written (agent#34).
        await self._step(app_id, ticket, "hold", self.runner.hold, app_id)
        current = await self._step(app_id, ticket, "state", self.runner.state, app_id)
        why = "it is still stopping"
        if current.state in ("running", "starting"):
            running = await self._step(app_id, ticket, "launched", self.runner.launched, app_id)
            why_not = self._why_not_keep(app_id, running, spec)
            if why_not is None:
                why_not = await self._step(
                    app_id, ticket, "check", self._why_not_this_process, app_id, current, running
                )
            if why_not is None:
                # Running since before this agent started, as this start
                # would run it: keep it, take back its admin token, and let
                # it be restarted after a crash again.
                token_file = self.runner.token_file(app_id, plan.data_dir)
                with contextlib.suppress(OSError):
                    planner.admin_token = token_file.read_text(encoding="utf-8").strip() or None
                if planner.admin_token:
                    await self._step(app_id, ticket, "release", self.runner.release, app_id)
                    self._tokens[app_id] = planner.admin_token
                    self._started_at.setdefault(app_id, datetime.now(UTC))
                    self._generations[app_id] = (running or {}).get("generation")
                    return
                why_not = "its admin token could not be read"
            why = why_not
        if current.state in ("running", "starting", "stopping"):
            log.warning(
                "app %s: the service that runs is not the one to keep (%s); stopping it "
                "before starting %s",
                app_id,
                why,
                planner.record.version,
            )
            await self._step(app_id, ticket, "stop", self.runner.stop, app_id)
        planner.admin_token = plan.admin_token
        self._tokens[app_id] = plan.admin_token
        await self._step(app_id, ticket, "prepare", self.runner.prepare, plan, spec)
        await self._step(app_id, ticket, "start", self.runner.start, app_id)
        self._held.discard(app_id)
        self._generations[app_id] = generation
        self._started_at[app_id] = datetime.now(UTC)
        log.info("app %s started in its own account (%s)", app_id, self.account(app_id))

    def _why_not_keep(self, app_id: str, running: dict | None, expected: dict) -> str | None:
        """Why a service found running is not the start the agent wants, or
        None to keep it. Kept only when the agent did not stop the app since
        it started, and the spec the service started from runs this version
        where the agent's door and health poll look for it."""
        if app_id in self._held:
            return (
                "the agent stopped this app, so what runs now was started by the service "
                "manager's own restart"
            )
        if running is None:
            return "the spec it was started with is missing"
        known = self._generations.get(app_id)
        if known and running.get("generation") != known:
            return "it was started from another launch"
        if running.get("argv") != expected.get("argv"):
            if running.get("version") and running.get("version") != expected.get("version"):
                return f"it runs version {running['version']}"
            return "it runs another version or command"
        now_env, want_env = running.get("env") or {}, expected.get("env") or {}
        now_values, want_values = running.get("values") or {}, expected.get("values") or {}
        if any(now_env.get(name) != want_env.get(name) for name in _PLACE) or any(
            now_values.get(name) != want_values.get(name) for name in ("port", "bindHost")
        ):
            return "it listens where the agent does not look for it"
        return None

    def _why_not_this_process(
        self, app_id: str, current: ServiceState, running: dict | None
    ) -> str | None:
        """Why the spec on disk is not proof of what runs, or None.

        The service must have started after its spec was written, so it read
        that spec; and no process outside the service's own tree may listen
        on the app's port, or the door and the health poll would reach that
        process instead. Nothing listening yet is an app still starting. On
        Linux the owner of another account's socket cannot be read by the
        agent's account; systemd's word that the unit runs stands there.
        """
        if not current.pid:
            return None
        started = _process_started(current.pid)
        written = self.runner.launched_at(app_id)
        if started is not None and written is not None and written > started + 1.0:
            return "its spec was written after it started, so it runs an older one"
        port = str(((running or {}).get("env") or {}).get("EUGENE_PLEXUS_APP_BIND_PORT") or "")
        if port.isdigit():
            tree = _process_tree(current.pid)
            strangers = {owner for owner in _listener_owners(int(port)) if owner is not None} - tree
            if strangers:
                return f"port {port} is held by process {min(strangers)}, which is not the app's"
        return None

    async def ports_in_use(self, app_id: str) -> tuple[int, int] | None:
        """The two loopback ports behind the app's door that a service of it
        found running listens on, from the spec it started with.

        The agent opens the door before the start that keeps such a service.
        With fresh ports, the door led nowhere and the app read `starting`
        for good after an agent restart (agent#32). None when nothing runs,
        when the agent stopped the app since (that start keeps nothing), or
        when the app listens without a door.
        """
        if app_id in self._held or (app_id in self._planners and not self._exited(app_id)):
            return None
        try:
            current = await asyncio.to_thread(self.runner.state, app_id)
        except Exception:
            return None
        if current.state not in ("running", "starting"):
            return None
        env = (await asyncio.to_thread(self.runner.launched, app_id) or {}).get("env") or {}
        try:
            return int(env["EUGENE_PLEXUS_APP_BIND_PORT"]), int(env["EUGENE_PLEXUS_APP_TLS_PORT"])
        except (KeyError, TypeError, ValueError):
            return None

    async def _withdraw(self, app_id: str) -> None:
        """Take back a start in flight, and wait until it has finished the
        step it is in: its thread is not stopped by a cancel, and what it
        writes after a stop is what the review of agent#34 found."""
        self._tickets.pop(app_id, None)
        task = self._starting.pop(app_id, None)
        if task is not None and not task.done():
            with contextlib.suppress(Exception):
                await asyncio.shield(task)

    def _forget(self, app_id: str) -> None:
        self._planners.pop(app_id, None)
        self._reachable.pop(app_id, None)
        self._tokens.pop(app_id, None)
        self._state.pop(app_id, None)

    async def _alive(self, app_id: str) -> bool:
        try:
            current = await asyncio.to_thread(self.runner.state, app_id)
        except Exception:
            return False
        return current.state in ("running", "starting", "stopping")

    async def stop(self, app_id: str) -> None:
        """Stop the app's service, and hold it stopped until the agent starts
        it: the service manager's own restart of a crash does not count.
        Raises when the service manager will not, and keeps reporting the app
        as it is: a stop that failed is not an app that stopped."""
        await self._withdraw(app_id)
        async with self._lock(app_id):
            self._held.add(app_id)
            await asyncio.to_thread(self.runner.hold, app_id)
            if app_id in self._planners or await self._alive(app_id):
                await asyncio.to_thread(self.runner.stop, app_id)
            self._forget(app_id)

    async def cleanup_pg0(self, planner: _AppPlanner, action: str) -> None:
        """Run named cleanup inside the same service account and private state directory."""
        app_id = planner.record.id
        await self._withdraw(app_id)
        async with self._lock(app_id):
            self._held.add(app_id)
            await self._cleanup_pg0(planner, action)

    async def _cleanup_pg0(self, planner: _AppPlanner, action: str) -> None:
        app_id = planner.record.id
        await asyncio.to_thread(self.runner.stop, app_id)
        spawn = planner.base_plan()
        plan = LaunchPlan(
            app_id=app_id,
            argv=spawn.argv,
            env=spawn.env,
            data_dir=self._data_dir(app_id),
            key_file=self._key_file(app_id),
            admin_token=planner.admin_token or "",
            port=planner.record.port,
            bind_host="127.0.0.1",
            launch={**planner.launch(), "pg0Action": action},
        )
        spec = launch_spec(plan, kind=self.runner.kind, ingress=self._ingress())
        await asyncio.to_thread(self.runner.prepare, plan, spec)
        await asyncio.to_thread(self.runner.start, app_id)
        deadline = time.perf_counter() + _STOP_TIMEOUT_SECONDS
        while time.perf_counter() < deadline:
            state = await asyncio.to_thread(self.runner.state, app_id)
            if state.state == "stopped" and not state.exit_code:
                return
            if state.state in {"failed", "missing"}:
                cause = state.detail or state.exit_code or state.state
                from .app_launcher import PG0_INSTANCE, redact_database

                raise OSError(
                    redact_database(f"pg0 {action} --name {PG0_INSTANCE} failed: {cause}")
                )
            await asyncio.sleep(0.1)
        await asyncio.to_thread(self.runner.stop, app_id)
        raise OSError(
            f"pg0 {action} --name eugene-plexus-hindsight did not finish within "
            f"{_STOP_TIMEOUT_SECONDS:g} seconds"
        )

    async def remove(self, app_id: str, *, purge: bool) -> None:
        """Stop the app and remove its service, in one request to the
        service manager's side. Raises when the service stays: an uninstall
        that reported success while the service lived on is what C1's first
        Windows run found."""
        await self._withdraw(app_id)
        async with self._lock(app_id):
            self._held.add(app_id)
            await asyncio.to_thread(self.runner.remove, app_id, purge=purge)
            self._forget(app_id)

    async def stop_all(self) -> None:
        """The agent is stopping. The apps are not.

        A start in flight is taken back, and waited for until it has finished
        the step it is in, for up to `_SHUTDOWN_WAIT_SECONDS`. One that is
        still busy then (a grant over a large environment can take minutes)
        is left to finish that step on its own, and the log says so: it goes
        no further, and the next agent's start holds the service before it
        looks at what that step left.
        """
        self._tickets.clear()
        busy = {app_id: task for app_id, task in self._starting.items() if not task.done()}
        if busy:
            _, late = await asyncio.wait(busy.values(), timeout=_SHUTDOWN_WAIT_SECONDS)
            for app_id, task in busy.items():
                if task in late:
                    log.warning(
                        "app %s: its start was still in %s when the agent stopped; that step "
                        "is left to finish on its own, and the start goes no further",
                        app_id,
                        self._doing.get(app_id, "a step"),
                    )
                    task.cancel()
        self._starting.clear()
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._poll_task
            self._poll_task = None
        if self._client is not None:
            with contextlib.suppress(BaseException):
                await self._client.aclose()
            self._client = None
        self._planners.clear()

    # --- what it reports --------------------------------------------------

    def _exited(self, app_id: str) -> bool:
        """The agent supervises the app, but nothing of it runs: its start
        failed, or its service has ended since. A Start then starts it again
        (agent#34's review: Start asked only whether there was a planner)."""
        task = self._starting.get(app_id)
        if task is not None and not task.done():
            return False
        if self._errors.get(app_id):
            return True
        state = self._state.get(app_id)
        # Not a crash: Windows' recovery or systemd's Restart= starts that again.
        return state is not None and state.state in ("stopped", "missing")

    def is_running(self, app_id: str) -> bool:
        return app_id in self._planners and not self._exited(app_id)

    def admin_token(self, app_id: str) -> str | None:
        return self._tokens.get(app_id)

    def bind_host(self, app_id: str) -> str | None:
        return self._hosts.get(app_id) if app_id in self._planners else None

    def status(
        self, app_id: str
    ) -> tuple[ComponentStatus, str | None, datetime | None, int | None, str | None]:
        if app_id not in self._planners:
            return ComponentStatus.exited, None, None, None, None
        state = self._state.get(app_id)
        error = self._errors.get(app_id)
        restarted = self._started_at.get(app_id)
        host = self._hosts.get(app_id)
        if error:
            return ComponentStatus.crashed, error, restarted, None, host
        if state is None or state.state in ("missing", "starting", "stopping"):
            return ComponentStatus.starting, None, restarted, state.pid if state else None, host
        if state.state == "running":
            live = (
                ComponentStatus.running if self._reachable.get(app_id) else ComponentStatus.starting
            )
            return live, None, restarted, state.pid, host
        if state.state == "failed":
            code = f" with code {state.exit_code}" if state.exit_code else ""
            return (
                ComponentStatus.crashed,
                f"its service stopped{code}; what it printed is on the Logs page",
                restarted,
                None,
                host,
            )
        return ComponentStatus.exited, None, restarted, None, host

    # --- polling ----------------------------------------------------------

    async def _poll_loop(self) -> None:
        try:
            while True:
                await asyncio.gather(
                    *(self._poll(app_id) for app_id in list(self._planners)),
                    return_exceptions=True,
                )
                await asyncio.sleep(_POLL_SECONDS)
        except asyncio.CancelledError:
            return

    async def _poll(self, app_id: str) -> None:
        planner = self._planners.get(app_id)
        if planner is None:
            return
        task = self._starting.get(app_id)
        if task is not None and not task.done():
            # What the service manager says before a start has finished is
            # the run before it: the page says starting meanwhile, and Start
            # must not read it as an app that has ended.
            return
        ticket = self._tickets.get(app_id)
        try:
            observed = await asyncio.to_thread(self.runner.state, app_id)
        except Exception as exc:
            observed = ServiceState("missing", detail=str(exc))
        if self._tickets.get(app_id) is not ticket or app_id not in self._planners:
            return  # a start or a stop came while this was asked
        self._state[app_id] = observed
        ready = False
        client = self._client
        if client is not None:
            try:
                response = await client.get(planner.health_url)
                ready = response.is_success
            except httpx.HTTPError:
                ready = False
        if app_id in self._planners:
            self._reachable[app_id] = ready
