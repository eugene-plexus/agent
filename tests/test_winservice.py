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
