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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import httpx

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


#: Variables the launcher is given in `launch.json`. Everything else comes
#: from the service manager's own environment for the account: the agent's
#: environment says where the agent's things are, which is the point.
_PASSED = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")


def launch_spec(plan: LaunchPlan, *, kind: str, ingress: str) -> dict:
    env = {k: v for k, v in plan.env.items() if k.startswith("EUGENE_PLEXUS_APP_") or k in _PASSED}
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
        "app": plan.app_id,
        "argv": plan.argv,
        "env": env,
        "ingress": ingress,
    }
    if kind == "windows_service":
        spec["service"] = f"{SERVICE_PREFIX}{plan.app_id}"
        spec["dataDir"] = str(plan.data_dir)
        spec["keyFile"] = str(plan.key_file)
        spec["adminTokenFile"] = str(plan.data_dir / ADMIN_TOKEN_FILE)
        spec["oidcSecretFile"] = str(plan.data_dir / OIDC_SECRET_FILE)
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
    def stop(self, app_id: str) -> None: ...
    def state(self, app_id: str) -> ServiceState: ...
    def remove(self, app_id: str, *, purge: bool) -> None: ...
    def token_file(self, app_id: str, data_dir: Path) -> Path: ...


def _icacls(path: Path, *args: str) -> None:
    out = subprocess.run(
        ["icacls", str(path), *args], capture_output=True, text=True, check=False, timeout=120
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

        manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
        try:
            try:
                handle = win32service.CreateService(
                    manager,
                    name,
                    f"Eugene Plexus app: {plan.app_id}",
                    win32service.SERVICE_ALL_ACCESS,
                    win32service.SERVICE_WIN32_OWN_PROCESS,
                    win32service.SERVICE_DEMAND_START,
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
                    win32service.SERVICE_DEMAND_START,
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
                # Restart a crash after 5 s, three times a day; the
                # agent's page says when it has given up.
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

    def start(self, app_id: str) -> None:
        import pywintypes
        import win32serviceutil

        try:
            win32serviceutil.StartService(self._name(app_id))
        except pywintypes.error as exc:
            if exc.winerror != 1056:  # ERROR_SERVICE_ALREADY_RUNNING
                raise

    def stop(self, app_id: str) -> None:
        import pywintypes
        import win32service
        import win32serviceutil

        name = self._name(app_id)
        try:
            win32serviceutil.ControlService(name, win32service.SERVICE_CONTROL_STOP)
        except pywintypes.error as exc:
            if exc.winerror in (1062, 1060):  # not started; does not exist
                return
            raise
        deadline = time.perf_counter() + _STOP_TIMEOUT_SECONDS
        while time.perf_counter() < deadline:
            if self.state(app_id).state in ("stopped", "failed", "missing"):
                return
            time.sleep(0.5)

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

    def stop(self, app_id: str) -> None:
        self._ask("stop", app_id)

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


class OwnAccountSupervisor:
    """`AppSupervisor`'s interface, for apps the service manager runs.

    It starts and stops through the runner, polls the service manager and
    the app's `/healthz`, and reports in the same five fields. What it does
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

    @property
    def kind(self) -> str:
        return self.runner.kind

    def account(self, app_id: str) -> str:
        return account_name(self.runner.kind, app_id)

    # --- lifecycle --------------------------------------------------------

    def start(self, planner: _AppPlanner) -> None:
        app_id = planner.record.id
        if app_id in self._planners:
            return
        self._planners[app_id] = planner
        self._reachable[app_id] = False
        self._errors[app_id] = None
        self._starting[app_id] = asyncio.create_task(
            self._start(planner), name=f"app-start-{app_id}"
        )
        if self._poll_task is None:
            self._client = internal_client(timeout=2.0)
            self._poll_task = asyncio.create_task(self._poll_loop(), name="app-accounts-poll")

    async def _start(self, planner: _AppPlanner) -> None:
        app_id = planner.record.id
        try:
            spawn = planner.plan()
            plan = LaunchPlan(
                app_id=app_id,
                argv=list(spawn.argv),
                env=dict(spawn.env),
                data_dir=self._data_dir(app_id),
                key_file=self._key_file(app_id),
                admin_token=planner.admin_token or secrets.token_urlsafe(32),
                port=planner.record.port,
                bind_host=spawn.env.get("EUGENE_PLEXUS_APP_BIND_HOST"),
            )
            self._hosts[app_id] = plan.bind_host or "127.0.0.1"
            current = await asyncio.to_thread(self.runner.state, app_id)
            if current.state in ("running", "starting"):
                # Running since before this agent started: keep it, and
                # take back the admin token it was started with.
                token_file = self.runner.token_file(app_id, plan.data_dir)
                with contextlib.suppress(OSError):
                    planner.admin_token = token_file.read_text(encoding="utf-8").strip() or None
                if planner.admin_token:
                    self._tokens[app_id] = planner.admin_token
                    self._started_at.setdefault(app_id, datetime.now(UTC))
                    return
            spec = launch_spec(plan, kind=self.runner.kind, ingress=self._ingress())
            planner.admin_token = plan.admin_token
            self._tokens[app_id] = plan.admin_token
            await asyncio.to_thread(self.runner.prepare, plan, spec)
            await asyncio.to_thread(self.runner.start, app_id)
            self._started_at[app_id] = datetime.now(UTC)
            log.info("app %s started in its own account (%s)", app_id, self.account(app_id))
        except Exception as exc:
            self._errors[app_id] = f"could not start in an account of its own: {exc}"
            log.error("app %s could not start in its own account: %s", app_id, exc)

    async def _settle_start(self, app_id: str) -> None:
        task = self._starting.pop(app_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def _forget(self, app_id: str) -> None:
        self._planners.pop(app_id, None)
        self._reachable.pop(app_id, None)
        self._tokens.pop(app_id, None)

    async def stop(self, app_id: str) -> None:
        """Stop the app's service. Raises when the service manager will
        not, and keeps reporting the app as it is: a stop that failed is
        not an app that stopped."""
        await self._settle_start(app_id)
        if app_id in self._planners:
            await asyncio.to_thread(self.runner.stop, app_id)
        self._forget(app_id)

    async def remove(self, app_id: str, *, purge: bool) -> None:
        """Stop the app and remove its service, in one request to the
        service manager's side. Raises when the service stays: an uninstall
        that reported success while the service lived on is what C1's first
        Windows run found."""
        await self._settle_start(app_id)
        await asyncio.to_thread(self.runner.remove, app_id, purge=purge)
        self._forget(app_id)

    async def stop_all(self) -> None:
        """The agent is stopping. The apps are not."""
        for task in self._starting.values():
            task.cancel()
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

    def is_running(self, app_id: str) -> bool:
        return app_id in self._planners

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
        try:
            self._state[app_id] = await asyncio.to_thread(self.runner.state, app_id)
        except Exception as exc:
            self._state[app_id] = ServiceState("missing", detail=str(exc))
        ready = False
        client = self._client
        if client is not None:
            try:
                response = await client.get(f"http://127.0.0.1:{planner.record.port}/healthz")
                ready = response.is_success
            except httpx.HTTPError:
                ready = False
        if app_id in self._planners:
            self._reachable[app_id] = ready
