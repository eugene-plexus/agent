"""The agent as a Windows service.

Written for install-paths §9 step 3, which writes an autostart unit on
every platform: a systemd user unit on Linux, a launchd agent on macOS,
and this on Windows. Call #4 made Windows a first-class target and call
#7 chose `pywin32` over NSSM — a normal wheel in the venv that already
*is* the component runtime, against a third-party binary that would have
to be downloaded, checksummed and trusted by a script whose whole virtue
is that it is short enough to read.

## What a service costs, and what it does not

**Its children get a hard kill and the agent itself does not.** A
service has no console, so `GenerateConsoleCtrlEvent` fails with
`WinError 6` and `process_signals` falls back to `TerminateProcess` for
every supervised child — measured at step 2, decided by Troy, and
announced at boot by `app.py` rather than discovered at the first stop.

But the *agent* stops cleanly, and that is not a detail. `SvcStop` sets
`should_exit` on the uvicorn server, so the ASGI lifespan shutdown runs:
the agent closes its own state and asks the supervisor to stop each
child in turn. What the children lose is their own lifespan shutdown,
not the agent's. This is why the class holds a `uvicorn.Server` rather
than calling `uvicorn.run()` — `run()` installs signal handlers a
service will never receive.

## Registration

Installed and removed by `install.ps1`, which is the only thing that
should be calling this module:

    python -m eugene_plexus_agent.winservice install   (elevated)
    python -m eugene_plexus_agent.winservice remove    (elevated)

Both need Administrator: the Service Control Manager grants
`SC_MANAGER_CREATE_SERVICE` to nobody else. A non-elevated install gets
a per-user scheduled task instead, which keeps the graceful path because
a task *does* have a console — see `install.ps1`.

**VERIFICATION STATUS.** Everything in this module that can be checked
without Administrator is checked by `tests/test_winservice.py` and by
`scripts/install-acceptance.sh`: the module imports, the class carries
the names the installer registers, the server is built and stoppable,
and an unelevated `install` fails with a sentence rather than a
traceback. **Registering and starting the service itself is unverified**
— it needs an elevated session, which the session that wrote this did
not have. `install.ps1 -Verify` prints the two commands that close it.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

SERVICE_NAME = "EugenePlexusAgent"
SERVICE_DISPLAY_NAME = "Eugene Plexus node agent"
SERVICE_DESCRIPTION = (
    "Supervises Eugene Plexus components and local inference engines on this host, "
    "and serves the web UI."
)

log = logging.getLogger(__name__)

# **The class is defined at module scope, and that is load-bearing.**
# The first draft built it inside a factory so pywin32 could be imported
# lazily, and registration failed outright:
#
#     _pickle.PicklingError: Can't pickle
#     <class '...build_service_class.<locals>.EugenePlexusAgentService'>
#
# `win32serviceutil.InstallService` records *where to find the class
# again* — it calls `pickle.whichmodule`, writes "module.ClassName" into
# the registry, and `PythonService.exe` later imports that module and
# getattrs that name. A class that only exists after someone calls a
# factory satisfies neither half: it has no importable location, and
# importing its module would not define it. Found by installing pywin32
# and running `install` unelevated, which got nowhere near the Service
# Control Manager before failing here.
#
# The guard keeps the module importable off-Windows, which is the other
# constraint: the test suite and CI are Linux.
try:  # pragma: no cover - the branch taken depends on the platform
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil

    PYWIN32_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYWIN32_AVAILABLE = False


def _require_pywin32() -> None:
    """Explain what to install, rather than let an ImportError speak.

    "No module named 'win32serviceutil'" does not name its own fix, and
    the overwhelmingly common cause is installing the extra into a
    different interpreter than the agent runs from — so the message
    names the interpreter.
    """
    if not PYWIN32_AVAILABLE:
        raise SystemExit(
            "The Windows service needs pywin32, which is not installed in "
            f"{sys.executable}.\n"
            "    pip install 'eugene-plexus-agent[service]'\n"
            "...into that interpreter, then try again."
        )


if PYWIN32_AVAILABLE:

    class EugenePlexusAgentService(win32serviceutil.ServiceFramework):
        _svc_name_ = SERVICE_NAME
        _svc_display_name_ = SERVICE_DISPLAY_NAME
        _svc_description_ = SERVICE_DESCRIPTION

        def __init__(self, args: list[str]) -> None:
            super().__init__(args)
            self._stopped = win32event.CreateEvent(None, 0, 0, None)
            self._server: Any = None

        def SvcStop(self) -> None:
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            server = self._server
            if server is not None:
                # The graceful path for the agent itself: uvicorn's own
                # loop notices the flag, runs the lifespan shutdown, and
                # returns. The supervisor's stop_all() happens inside
                # that shutdown, which is what makes this different from
                # simply killing the process.
                server.should_exit = True
            win32event.SetEvent(self._stopped)

        def SvcDoRun(self) -> None:
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, ""),
            )
            from .__main__ import build_server
            from .settings import load_settings

            settings = load_settings()
            # A service has no console at all, so `has_tty()` would
            # already be False -- but the unattended path is declared
            # rather than inferred everywhere else this project starts
            # the agent, and a service is the least-watched of the lot.
            self._server = build_server(settings, unattended=True)
            # The run loop goes on a thread so SvcStop, which the SCM
            # calls on *its* thread, is not waiting behind it.
            thread = threading.Thread(target=self._server.run, name="agent", daemon=False)
            thread.start()
            win32event.WaitForSingleObject(self._stopped, win32event.INFINITE)
            # systemd's TimeoutStopSec has no Windows equivalent we
            # control, and the SCM kills a service that takes too long.
            # Bound the wait rather than hang a `sc stop` forever on a
            # child that will not go.
            thread.join(timeout=90)


def service_class() -> type:
    """The registered service class, or a sentence saying why there is none.

    Callers must go through this rather than reaching for the name
    directly: off-Windows and on a Windows install without the `service`
    extra there is no class at all, and `AttributeError` is not an
    explanation.
    """
    _require_pywin32()
    return EugenePlexusAgentService


def main(argv: list[str] | None = None) -> None:
    """`python -m eugene_plexus_agent.winservice <install|remove|start|stop>`.

    Thin by design — `HandleCommandLine` is pywin32's own argument
    surface and reimplementing it would only add ways to disagree with
    the SCM.
    """
    if sys.platform != "win32":
        raise SystemExit("The Eugene Plexus Windows service only exists on Windows.")

    _require_pywin32()
    argv = list(sys.argv if argv is None else ["winservice", *argv])

    # **Take the class from the canonically-imported module, not from
    # this one.** `python -m eugene_plexus_agent.winservice` runs this
    # file as `__main__`, so the class defined above carries
    # `__module__ == "__main__"` -- and that string is what pywin32
    # writes into the registry for `PythonService.exe` to import later.
    # Re-importing under the dotted name gives the SCM a location that
    # still exists when nothing is running it as a script.
    from eugene_plexus_agent import winservice as canonical

    cls = canonical.service_class()

    if len(argv) == 1:
        # No arguments is how the SCM starts us: it expects the process
        # to connect back to it, not to print usage.
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(cls)
        servicemanager.StartServiceCtrlDispatcher()
        return

    command = argv[1]

    # **Refuse before pywin32 starts moving files.** `InstallService`
    # relocates `pythonservice.exe` and copies a helper DLL beside the
    # base interpreter *before* it ever touches the Service Control
    # Manager, so an unelevated attempt leaves those side effects behind
    # and then fails anyway. Checking first turns that into a sentence
    # and no writes at all.
    if command in {"install", "update", "remove"} and not _is_elevated():
        raise SystemExit(_elevation_message(command))

    win32serviceutil.HandleCommandLine(cls, argv=argv)

    # **`HandleCommandLine` reports failure by printing and exiting 0.**
    # Measured, 2026-09-11: an unelevated `install` prints "Error
    # installing service: Access is denied. (5)", registers nothing, and
    # returns success. `install.ps1` tests `$LASTEXITCODE`, so without
    # this the installer would have called a machine with no service
    # installed. Ask the SCM instead of believing the exit code.
    if command == "install" and not _service_exists():
        raise SystemExit(_elevation_message(command))
    if command == "remove" and _service_exists():
        raise SystemExit(f"The {SERVICE_NAME} service is still registered after 'remove'.")


def _is_elevated() -> bool:
    """Whether this process can create a service.

    The `sys.platform` guard is for mypy as much as for correctness:
    `ctypes.windll` does not exist on the Linux that CI type-checks on,
    and narrowing makes the line unreachable there rather than needing
    an ignore that would then be unused on Windows.
    """
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # pragma: no cover - defensive
        return False


def _service_exists() -> bool:
    """Ask the SCM, because the exit code cannot be trusted."""
    try:
        win32serviceutil.QueryServiceStatus(SERVICE_NAME)
    except Exception:
        return False
    return True


def _elevation_message(command: str) -> str:
    return (
        f"Registering a Windows service needs Administrator, and '{command}' did not "
        f"take effect.\n"
        "    Start an elevated PowerShell, then:\n"
        f'    & "{sys.executable}" -m eugene_plexus_agent.winservice {command}\n'
        "Or leave it unelevated: install.ps1 registers a per-user logon task instead, "
        "which also keeps the graceful stop for supervised children."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
