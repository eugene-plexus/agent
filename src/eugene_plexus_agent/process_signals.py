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


def ensure_console() -> bool:
    """Give this process a console if it has none. Windows only.

    **This is what makes a Windows service stop its engines gracefully,
    and the cost that ruled it out was never measured.**
    `install-paths-and-distribution.md` §7 lists *"Real service +
    `AllocConsole()`"* with the cost *"can reassign the agent's std
    handles — logs vanish"*, and concludes *"we do not conjure a
    console"*. Measured 2026-09-18, three arms, each in its own process,
    child spawned with `CREATE_NEW_PROCESS_GROUP` and a SIGBREAK
    handler:

        inherited console            event sent, child out in 0.034 s
        after FreeConsole()          WinError 6, child never signalled
        after FreeConsole()+Alloc    event sent, child out in 0.036 s

    and after the third, the `RotatingFileHandler` kept writing and
    neither `print` nor `sys.stderr.write` raised. It was never going to:
    the durable sink is a file handler, children are `stdout=PIPE`
    (`supervisor.py`), and under pywin32's service host `sys.__stdout__`
    is already `None` — a case `console_logging` handles today. A
    session-0 console is invisible and valid, which is all
    `GenerateConsoleCtrlEvent` needs.

    **Call it before the first child is spawned.** `spawn_kwargs` sets no
    console flag, so a child inherits whatever console the parent holds
    *at spawn time*; a console allocated afterwards is one that child is
    not attached to, and the event would still fail for it alone —
    which is worse than failing for all of them, because it fails for
    some.

    Returns True when this process has a console when the call returns,
    whether or not this call is what gave it one. Never raises: an agent
    that cannot allocate a console still supervises, it just goes back
    to hard-killing children and says so at boot.
    """
    if sys.platform != "win32":
        return True
    if console_attached():
        return True
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        # AllocConsole returns 0 and sets ERROR_ACCESS_DENIED (5) when
        # the process already has one, which `console_attached()` above
        # has ruled out — so a 0 here is a real refusal. Ask
        # `console_attached()` again rather than trusting the return
        # value, for the same reason `winservice.main` re-asks the SCM:
        # the question is "is there a console now", not "did the call
        # say yes".
        kernel32.AllocConsole()
    except Exception:  # pragma: no cover - defensive
        log.warning("could not allocate a console; children will be hard-killed")
        return False
    got = console_attached()
    if got:
        log.info("allocated a console, so supervised children can be stopped gracefully")
    else:
        log.warning("could not allocate a console; children will be hard-killed")
    return got


def describe_stop_capability() -> tuple[bool, str]:
    """`(graceful, why)` — for an announcement at startup, not at first stop.

    **The degradation was accepted on 2026-09-11 and is no longer
    necessary, but the badge stays.** Troy's call then: Windows ships a
    real service, a service has no console, so children there are
    hard-killed, and that is badged permanently rather than hidden.
    R2.6 removed the cause — `ensure_console()` above — so the service
    now takes the first branch. The console-less string stays because it
    is still the truth for anything that reaches that state another way:
    an `AllocConsole` that is refused, or a future host that starts the
    agent without one and without going through `winservice`.
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
