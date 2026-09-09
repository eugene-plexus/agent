"""Cross-platform orphan-prevention for supervised children.

Without this, hard-killing the agent (SIGKILL on POSIX,
TerminateProcess on Windows, OS crash, power loss) leaves spawned
children orphaned. They keep running until manually terminated, which
on a personal-use install is at best annoying (port still bound) and
at worst confusing ("I killed Eugene but he's still answering").

Two approaches, picked per-platform:

- **Linux**: each spawned child registers `prctl(PR_SET_PDEATHSIG,
  SIGTERM)` in its `preexec_fn` so the kernel sends SIGTERM to the
  child whenever its parent dies, regardless of how. Implemented via
  `ctypes` to avoid pulling in a third-party prctl wrapper.

- **Windows**: the agent creates a Job Object with the
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` flag and assigns each spawned
  child to it. When the agent process exits (any way), the kernel
  closes the job handle, which kills every assigned child. This
  generalizes cleanly to "the agent is the root of a process tree
  the OS will tear down for us."

- **macOS**: macOS has no direct equivalent of pdeathsig. We could use
  kqueue's NOTE_EXIT but that requires polling; for v0.1 we accept the
  same orphan-on-hard-kill risk POSIX-without-prctl always had. Add
  better handling here when v0.2 hardens supervision.

Tests for the platform-specific paths are deliberately light — actually
exercising prctl or Job Objects requires forking real processes. The
core test asserts the right cross-platform `setup_for_kind()` was
selected based on `sys.platform`.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

# Linux's <sys/prctl.h> defines PR_SET_PDEATHSIG = 1. SIGTERM (15) is
# what the kernel sends to the child when the parent dies.
_PR_SET_PDEATHSIG = 1
_SIGTERM = 15


def linux_pdeathsig_preexec() -> None:
    """preexec_fn for asyncio.create_subprocess_exec on Linux.

    Runs in the forked child between fork() and exec(). Registers
    PR_SET_PDEATHSIG so the kernel sends SIGTERM to this child if its
    parent (the agent) dies. Best-effort: prctl failures are logged
    but don't block the spawn.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        result = libc.prctl(_PR_SET_PDEATHSIG, _SIGTERM, 0, 0, 0)
        if result != 0:
            err = ctypes.get_errno()
            log.warning("prctl(PR_SET_PDEATHSIG) failed: errno=%d", err)
    except OSError as e:
        log.warning("could not load libc for prctl: %s", e)


class WindowsJobObject:
    """Wraps a Windows Job Object configured to kill its members on close.

    The agent creates one of these at startup and assigns every
    spawned child to it. When the agent process exits (graceful,
    crash, hard kill, OS reboot — any reason), the kernel closes the
    last reference to the job handle, which triggers KILL_ON_JOB_CLOSE
    and reaps every child.

    Initialization is best-effort. If Job Object creation fails (which
    shouldn't happen on any supported Windows version) we log and
    proceed without — orphan-on-hard-kill is the existing behavior, so
    we're never worse off than before.
    """

    # Win32 constants. Keep them inline rather than depending on
    # pywin32 since `ctypes` is stdlib and we don't want to pull a
    # native dependency in just for this.
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    def __init__(self) -> None:
        self.handle: int | None = None
        if sys.platform != "win32":
            return
        try:
            self.handle = self._create_kill_on_close_job()
        except OSError as e:
            log.warning("could not create Windows Job Object: %s", e)
            self.handle = None

    def assign(self, pid: int) -> None:
        """Assign a process to this job. No-op if the job didn't init."""
        if self.handle is None or sys.platform != "win32":
            return
        try:
            self._assign_pid_to_job(self.handle, pid)
        except OSError as e:
            log.warning("could not assign pid %d to Job Object: %s", pid, e)

    @staticmethod
    def _create_kill_on_close_job() -> int:
        """CreateJobObjectW + SetInformationJobObject(KILL_ON_JOB_CLOSE)."""
        # ctypes.WinDLL is Windows-only. The double-tagged ignore keeps
        # mypy quiet both ways: attr-defined on Linux/Mac CI, and
        # unused-ignore on Windows where the attribute does exist.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")  # type: ignore[attr-defined,unused-ignore]

        # Build JOBOBJECT_EXTENDED_LIMIT_INFORMATION with the kill-on-close flag.
        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BasicLimitInfo(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class _ExtendedLimitInfo(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInfo),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = _ExtendedLimitInfo()
        info.BasicLimitInformation.LimitFlags = WindowsJobObject._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        kernel32.SetInformationJobObject.restype = ctypes.c_int
        kernel32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        ok = kernel32.SetInformationJobObject(
            handle,
            WindowsJobObject._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")  # type: ignore[attr-defined,unused-ignore]

        return int(handle)

    @staticmethod
    def _assign_pid_to_job(job_handle: int, pid: int) -> None:
        """Open the process by pid, AssignProcessToJobObject, close
        the process handle. The job retains its membership."""
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        # PROCESS_SET_QUOTA | PROCESS_TERMINATE = 0x0100 | 0x0001
        process_access = 0x0100 | 0x0001
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        proc_handle = kernel32.OpenProcess(process_access, False, pid)
        if not proc_handle:
            raise OSError(ctypes.get_last_error(), f"OpenProcess({pid}) failed")  # type: ignore[attr-defined,unused-ignore]
        try:
            kernel32.AssignProcessToJobObject.restype = ctypes.c_int
            kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            ok = kernel32.AssignProcessToJobObject(job_handle, proc_handle)
            if not ok:
                raise OSError(
                    ctypes.get_last_error(),  # type: ignore[attr-defined,unused-ignore]
                    f"AssignProcessToJobObject(pid={pid}) failed",
                )
        finally:
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle(proc_handle)


def preexec_for_platform() -> Callable[[], None] | None:
    """Returns a `preexec_fn` for `asyncio.create_subprocess_exec`, or
    None if the current platform doesn't have one. Linux: pdeathsig.
    Other POSIX: not yet implemented (best-effort behavior matches
    pre-orphan-handling). Windows: handled via Job Object, returns None
    here because Win32 doesn't have preexec hooks."""
    if sys.platform.startswith("linux"):
        return linux_pdeathsig_preexec
    return None


def kwargs_for_platform() -> dict[str, Any]:
    """Extra `asyncio.create_subprocess_exec` kwargs needed for
    orphan-prevention on this platform.

    On POSIX with a preexec available, threads it through. On Windows
    the Job Object is applied AFTER spawn (we need the pid first), so
    no kwargs here — see Supervisor.add_and_start.
    """
    out: dict[str, Any] = {}
    preexec = preexec_for_platform()
    if preexec is not None:
        out["preexec_fn"] = preexec
    return out


def is_orphan_kill_supported() -> bool:
    """True if we expect children to be reaped automatically when the
    agent dies hard. Useful for log lines and the operator-facing
    docs that explain the v0.1 behavior."""
    if sys.platform.startswith("linux"):
        return True
    if sys.platform == "win32":
        return True
    return False


# Module-level singleton: created once when the agent process boots.
# `Supervisor` reaches into this for the assign-pid call after each
# spawn. None on non-Windows or when init failed.
_windows_job: WindowsJobObject | None = None


def windows_job() -> WindowsJobObject | None:
    """Return the process-wide Windows Job Object, creating it on first
    call. None on non-Windows or when init failed."""
    global _windows_job
    if sys.platform != "win32":
        return None
    if _windows_job is None:
        _windows_job = WindowsJobObject()
    return _windows_job


def _reset_for_tests() -> None:
    """Drop the cached Windows job object so tests can re-init under
    monkeypatched ctypes. Not for production use."""
    global _windows_job
    _windows_job = None


# Initialize on import — getting an early "couldn't create Job Object"
# warning is more useful than failing on the first spawn. No-op on
# non-Windows.
if sys.platform == "win32":
    log.info(
        "creating Windows Job Object for orphan-prevention (KILL_ON_JOB_CLOSE)",
    )
    windows_job()
