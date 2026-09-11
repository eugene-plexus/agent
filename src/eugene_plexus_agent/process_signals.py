"""How a supervised child is asked to stop, per platform.

Split out of `supervisor.py` at install-paths §9 step 2, when Windows
became a first-class target and "graceful shutdown is a hard kill there"
stopped being an acceptable footnote.

## What was measured, because the design's premises were inherited

On this box, 2026-09-11, with a uvicorn child — which is what every
Eugene Plexus component is:

| how it was stopped                         | rc  | ASGI lifespan shutdown |
| ------------------------------------------ | --- | ---------------------- |
| `TerminateProcess` (`proc.terminate()`)     | 1   | **NO**                 |
| `CTRL_BREAK_EVENT`, no handler installed    | 3   | **yes**                |
| `CTRL_BREAK_EVENT`, handler → `SIGINT`      | 0   | **yes**                |

**`TerminateProcess` is the only one of the three that skips it**, and
the plain no-handler case already works — because CPython gives SIGBREAK
the same default handler as SIGINT, so a console event becomes a
`KeyboardInterrupt` and uvicorn unwinds normally. That is why this is an
agent-only change: **no component needed a line of code.** The
expectation going in was a five-repo edit.

## Three constraints the measurements imposed

**The child must be in its own process group.** `CTRL_C_EVENT` cannot be
aimed at one — Windows delivers it to every process sharing the console
or to none — so `CTRL_BREAK_EVENT` into a group created with
`CREATE_NEW_PROCESS_GROUP` is the only way to stop one child without
stopping the agent as well.

**Escalation is mandatory, not a nicety.** A child that installs a
console handler and returns TRUE from it survives the event indefinitely
— measured, with a stand-in that ignored it for six seconds and was
still running. `TerminateProcess` had no such failure mode, so adding a
graceful path *introduces* a hang this module has to close.

**A parent with no console cannot send the event at all.**
`GenerateConsoleCtrlEvent` fails with `WinError 6` (invalid handle)
after `FreeConsole()` — verified directly. **That is exactly what a
Windows service is**, which matters because §9 step 3 writes one. We do
not try to conjure a console: `AllocConsole()` can reassign the
process's standard handles, and an agent whose logs vanish is a worse
outcome than a hard kill. We fall back, and we say so once, naming the
reason — an explanation of a real failure beats a prediction.

POSIX is untouched: `SIGTERM`, then `SIGKILL`, exactly as before.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from enum import StrEnum
from typing import Any

from . import orphan_kill

log = logging.getLogger(__name__)


# **Resolved by name with a literal fallback, so the Windows paths stay
# testable on the only CI this project has — Linux.** `signal` has no
# `CTRL_BREAK_EVENT` and `subprocess` no `CREATE_NEW_PROCESS_GROUP`
# there, so a test that monkeypatches `sys.platform` to "win32" walks
# straight into an AttributeError raised by production code, and the
# behaviour that matters most on a first-class target ends up asserted
# nowhere. These are fixed Win32 ABI constants, not Python details:
# CTRL_BREAK_EVENT is 1 and CREATE_NEW_PROCESS_GROUP is 0x200, and they
# cannot change without breaking every program on the platform.
CTRL_BREAK_EVENT: int = getattr(signal, "CTRL_BREAK_EVENT", 1)
CREATE_NEW_PROCESS_GROUP: int = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)


class StopSignal(StrEnum):
    """What was actually sent, for logs and for tests that must be able
    to tell a graceful request from a hard kill."""

    sigterm = "SIGTERM"
    ctrl_break = "CTRL_BREAK_EVENT"
    terminate_process = "TerminateProcess"


#: Set once, the first time a console event cannot be sent. A service
#: stopping twelve children must not emit twelve identical warnings.
_console_warning_emitted = False


class _Stoppable:
    """Structural type for what this module needs off a child process.

    `asyncio.subprocess.Process` satisfies it, and so does a test double
    that records rather than signals.
    """

    pid: int
    returncode: int | None

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


def spawn_kwargs() -> dict[str, Any]:
    """Every platform-specific kwarg a supervised spawn needs.

    One entry point on purpose. Orphan-prevention (`prctl` on Linux) and
    stop-signalling (a process group on Windows) are different concerns
    that both reach `create_subprocess_exec`, and two call sites
    contributing kwargs is how one of them quietly stops being applied.
    """
    out: dict[str, Any] = dict(orphan_kill.kwargs_for_platform())
    if sys.platform == "win32":
        # The child needs its own group to be a target for
        # CTRL_BREAK_EVENT. Side effect worth knowing: it no longer
        # receives Ctrl+C from the agent's console, which is correct —
        # children are stopped by their supervisor, not by whoever is
        # looking at the terminal.
        out["creationflags"] = out.get("creationflags", 0) | CREATE_NEW_PROCESS_GROUP
    return out


def graceful_stop_kind() -> StopSignal:
    """What `request_stop` will attempt on this platform."""
    return StopSignal.ctrl_break if sys.platform == "win32" else StopSignal.sigterm


def console_attached() -> bool:
    """Is this process attached to a console? Windows only; True elsewhere.

    **`GetConsoleProcessList`, not `GetConsoleWindow`.** The obvious
    probe returns a null HWND for a process attached to a ConPTY — which
    is every modern terminal — so it reports "no console" for a process
    that has one and can send console events perfectly well. That lie
    cost a probe run here: it claimed a console-less parent had a
    console, and the conclusion had to be reached another way (calling
    `FreeConsole()` and watching a working call start failing).
    `GetConsoleProcessList` returns 4 and 0 for those two cases.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        buffer = (ctypes.c_uint32 * 1)()
        return int(kernel32.GetConsoleProcessList(buffer, 1)) > 0
    except Exception:  # pragma: no cover - defensive
        return False


def describe_stop_capability() -> tuple[bool, str]:
    """`(graceful, why)` — for an announcement at startup, not at first stop.

    **The degradation is accepted and therefore has to be visible.**
    Troy's call, 2026-09-11: Windows ships a real service, and a service
    has no console, so children there are hard-killed. That is the same
    shape as the Vulkan decision (§7 of the install-paths design) —
    ship it, and badge it permanently — and the badge is worth more at
    boot than at the first stop, because at the first stop the operator
    is already watching something else go wrong.
    """
    if sys.platform != "win32":
        return True, "children are stopped with SIGTERM, then SIGKILL if they hang"
    if console_attached():
        return True, (
            "children are stopped with CTRL_BREAK_EVENT, so they run their "
            "shutdown hooks; a child that ignores it is killed"
        )
    return False, (
        "this agent has no console, so children are hard-killed with "
        "TerminateProcess: in-flight requests are dropped and shutdown hooks "
        "do not run. That is what a Windows service looks like, and it is a "
        "known, accepted limitation of running as one rather than a fault"
    )


def request_stop(proc: Any, *, name: str, logger: logging.Logger | None = None) -> StopSignal:
    """Ask a child to shut down, gracefully where the platform allows.

    Returns what was actually sent — never what was intended. A caller
    that logs the intent instead would report a graceful stop for a hard
    kill, which is the precise shape of misreport this codebase keeps
    finding.

    Always followed by an escalation deadline at the call site: this
    function only *asks*.
    """
    global _console_warning_emitted
    active = logger or log

    if sys.platform != "win32":
        proc.terminate()
        return StopSignal.sigterm

    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        # Guard, not paranoia: os.kill(0, CTRL_BREAK_EVENT) signals the
        # agent's OWN process group. Getting here with a falsy pid would
        # stop the supervisor along with the child.
        proc.terminate()
        return StopSignal.terminate_process

    try:
        os.kill(pid, CTRL_BREAK_EVENT)
        return StopSignal.ctrl_break
    except OSError as exc:
        if not _console_warning_emitted:
            _console_warning_emitted = True
            active.warning(
                "cannot send a console event to %s (%s), so children get "
                "TerminateProcess instead of a graceful stop — in-flight "
                "requests are dropped and shutdown hooks do not run. This "
                "agent has no console, which is what a Windows service "
                "looks like; run it with one to restore graceful stops.",
                name,
                exc,
            )
        proc.terminate()
        return StopSignal.terminate_process


def reset_console_warning_for_tests() -> None:
    """Clear the once-only latch so a test can observe it firing."""
    global _console_warning_emitted
    _console_warning_emitted = False
