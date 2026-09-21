"""What can be asserted about the Windows service without Administrator.

Which is less than the whole thing, and the gap is stated rather than
papered over: registering a service needs `SC_MANAGER_CREATE_SERVICE`,
the SCM grants it to nobody else, and CI is Linux. So these tests cover
the parts that fail *quietly* if they rot — the module being importable
off-Windows, the missing-dependency message naming its own fix, and
`build_server` actually returning something stoppable — and
`scripts/install-acceptance.sh` covers the rest of the reachable
surface on a real Windows box.

Two defects here were found by installing pywin32 and *running* it, not
by reading the API: the service class cannot live inside a factory
(`InstallService` records where to import it from), and
`HandleCommandLine` reports failure by printing and exiting 0.

`build_server` is the load-bearing one. It exists only so `SvcStop` has
a `should_exit` flag to set; if someone later collapses it back into
`uvicorn.run()`, nothing on Linux notices and the service silently
becomes un-stoppable except by the SCM's kill.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import uvicorn

from eugene_plexus_agent import winservice
from eugene_plexus_agent.__main__ import build_server
from eugene_plexus_agent.settings import Settings


@pytest.mark.skipif(not winservice.PYWIN32_AVAILABLE, reason="Windows service loader")
def test_service_host_loads_from_a_venv_without_python_on_path(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """Load the real host, but name a nonexistent service: no SCM writes/start."""
    import ctypes
    import os
    import shutil
    import subprocess

    import win32service

    prefix = tmp_path / "isolated venv"
    site = prefix / "Lib" / "site-packages"
    site.mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text(
        f"home = {sys.base_prefix}\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
    installed_site = Path(win32service.__file__).parent.parent
    imported = prefix / "crypto-imported.txt"
    (site / "probe.pth").write_text(
        "\n".join(
            str(p) for p in (installed_site, installed_site / "win32", installed_site / "win32/lib")
        )
        + "\n"
        + f"import nacl.public; import cryptography.hazmat.bindings._rust; open({str(imported)!r}, 'w').write('ok')\n",
        encoding="utf-8",
    )
    # Earlier pywin32 registration can move the wheel's host to the venv root.
    packaged = Path(win32service.__file__).with_name("pythonservice.exe")
    if not packaged.is_file():
        packaged = Path(sys.prefix) / "pythonservice.exe"
    shutil.copy2(packaged, prefix / "pythonservice.exe")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    host = winservice._prepare_service_host()
    assert host.parent == prefix / "Scripts"
    env = {
        k: v for k, v in os.environ.items() if not k.upper().startswith(("PYTHON", "EUGENE_PLEXUS"))
    }
    env["PATH"] = str(Path(os.environ["SYSTEMROOT"]) / "System32")
    old_mode = ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    try:
        result = subprocess.run(
            [str(host), "-debug", "__EP_NONEXISTENT_LOADER_TEST__"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    finally:
        ctypes.windll.kernel32.SetErrorMode(old_mode)
    output = result.stdout.decode("utf-16-le", errors="replace")
    assert result.returncode == 0, (hex(result.returncode & 0xFFFFFFFF), result.stderr)
    assert "Debugging service __EP_NONEXISTENT_LOADER_TEST__" in output
    assert "PythonClass" in output  # reaches the expected missing registry entry
    assert imported.is_file(), result.stderr


@pytest.mark.skipif(not winservice.PYWIN32_AVAILABLE, reason="Windows service registration")
def test_registration_uses_prepared_host_and_propagates_scm_error(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    import win32serviceutil

    host = tmp_path / "Scripts" / "pythonservice.exe"
    monkeypatch.setattr(winservice.service_class(), "_exe_name_", None, raising=False)
    monkeypatch.setattr(winservice, "_is_elevated", lambda: True)
    monkeypatch.setattr(winservice, "_prepare_service_host", lambda: host)
    observed = []

    def handle(cls, argv):
        observed.append((cls._exe_name_, argv[-1]))
        return 5

    monkeypatch.setattr(win32serviceutil, "HandleCommandLine", handle)
    monkeypatch.setattr(winservice, "_service_exists", lambda: True)
    with pytest.raises(SystemExit) as exc:
        winservice.main(["update"])
    assert exc.value.code == 5
    assert observed == [(str(host), "update")]


def test_module_imports_off_windows() -> None:
    """Importable everywhere, so it can be checked everywhere.

    The pywin32 import is guarded rather than absent: the class it
    defines must be a module-level attribute, because that is what
    `InstallService` records and `PythonService.exe` later imports. A
    module that can only be imported on Windows could only be tested on
    Windows.
    """
    assert importlib.import_module("eugene_plexus_agent.winservice") is winservice
    assert winservice.SERVICE_NAME == "EugenePlexusAgent"
    assert winservice.SERVICE_DISPLAY_NAME
    assert winservice.SERVICE_DESCRIPTION


@pytest.mark.skipif(
    winservice.PYWIN32_AVAILABLE,
    reason="pywin32 is installed here, so the absent-dependency path cannot be taken",
)
def test_missing_pywin32_names_its_own_fix() -> None:
    with pytest.raises(SystemExit) as exc:
        winservice._require_pywin32()
    text = str(exc.value)
    assert "pywin32" in text
    assert "eugene-plexus-agent[service]" in text
    # The interpreter matters: the overwhelmingly common cause is
    # installing into a different one than the agent runs from.
    assert sys.executable in text


def test_main_refuses_off_windows() -> None:
    if sys.platform == "win32":
        pytest.skip("this refusal only fires off Windows")
    with pytest.raises(SystemExit) as exc:
        winservice.main(["install"])
    assert "only exists on Windows" in str(exc.value)


def test_build_server_is_stoppable(tmp_path: Path) -> None:
    """The whole reason `build_server` was split out of `_serve`.

    `uvicorn.run()` builds one of these and calls `.run()`, which
    installs SIGINT/SIGTERM handlers. A Windows service receives
    neither; it stops by setting `should_exit` on the object. So the
    contract this asserts is: we get the object, unstarted, with the
    flag.
    """
    settings = Settings(
        config_file=tmp_path / "agent.yaml",
        bind_port=8179,
        default_topology=False,
    )
    server = build_server(settings)
    assert isinstance(server, uvicorn.Server)
    assert server.should_exit is False
    assert server.started is False
    assert server.config.port == 8179
    # Setting the flag is all SvcStop does; it must be plain attribute
    # assignment and not a method that needs a running loop.
    server.should_exit = True
    assert server.should_exit is True


@pytest.mark.skipif(
    not winservice.PYWIN32_AVAILABLE,
    reason="needs pywin32, which only installs on Windows",
)
def test_service_class_is_importable_by_name() -> None:
    """The defect the factory version had, asserted rather than retried.

    `win32serviceutil.InstallService` calls `pickle.whichmodule` on the
    class and writes "module.ClassName" into the registry;
    `PythonService.exe` then imports that module and getattrs that name.
    A class built inside a function satisfies neither half — it raises
    PicklingError at registration, which is a step no unelevated test
    can reach, so this asserts the property the registration needs
    instead of the registration itself.
    """
    import importlib
    import pickle

    cls = winservice.service_class()
    module_name = pickle.whichmodule(cls, cls.__name__)
    assert module_name != "__main__"
    resolved = getattr(importlib.import_module(module_name), cls.__name__)
    assert resolved is cls
    assert cls._svc_name_ == winservice.SERVICE_NAME


def test_unattended_skips_the_first_boot_question(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The defect a Windows scheduled task found, asserted directly.

    A task's process has BOTH stdin and stdout as a console -- measured
    2026-09-11 -- so `has_tty()` is True and the agent printed the
    first-boot question into a console nobody can see, then blocked on
    `input()`. Nothing listened, nothing was logged, and the task
    reported Running. The fix is that every unit file declares
    `--unattended` rather than relying on the absence of a terminal, so
    this asserts the flag beats a TTY rather than that a TTY is absent.
    """
    from eugene_plexus_agent import __main__ as entry

    monkeypatch.setattr(entry, "has_tty", lambda: True)
    monkeypatch.setattr(entry, "is_fresh_boot", lambda _settings: True)

    asked = []
    monkeypatch.setattr(entry, "ask", lambda _settings: asked.append(1) or None)

    settings = Settings(config_file=tmp_path / "agent.yaml", bind_port=8180)
    entry.build_server(settings, unattended=True)
    assert asked == [], "the question was asked with --unattended and a TTY present"

    entry.build_server(settings, unattended=False)
    assert asked == [1], "without --unattended and with a TTY, the question must still be asked"


@pytest.mark.skipif(not winservice.PYWIN32_AVAILABLE, reason="needs Windows service host")
def test_event_log_denial_does_not_prevent_service_start(monkeypatch, caplog):  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from eugene_plexus_agent import __main__, process_signals, settings

    observed = []

    def denied(*args):
        raise PermissionError("RegisterEventSource: access denied")

    monkeypatch.setattr(winservice.servicemanager, "LogMsg", denied)
    monkeypatch.setattr(process_signals, "ensure_console", lambda: None)
    monkeypatch.setattr(winservice, "_chdir_to_prefix", lambda: None)
    monkeypatch.setattr(settings, "load_settings", lambda: None)
    monkeypatch.setattr(
        __main__,
        "build_server",
        lambda *args, **kwargs: SimpleNamespace(run=lambda: observed.append("ran")),
    )
    monkeypatch.setattr(winservice.win32event, "WaitForSingleObject", lambda *args: None)
    cls = winservice.service_class()
    instance = cls.__new__(cls)
    instance._stopped = object()
    instance.SvcDoRun()
    assert observed == ["ran"]
    assert "event log" in caplog.text.lower()
