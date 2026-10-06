"""Each linked person's worker, started as that person (§2.4, §3.2, J24, J25).

The machine's privileged starter runs one worker per linked account. This
agent plays that part on two kinds of install:

- **a Windows service install** (LocalSystem): for each linked account that
  has a session on this machine, the session's own token
  (`WTSQueryUserToken`), started into that session with
  `CreateProcessAsUser`. While the person is signed out there is no worker,
  and their calls are refused saying so (J25, revised after the S4U
  measurement, §2.4.1). The token is UAC's filtered one; where it is not
  (UAC off, the built-in Administrator) it is filtered here by hand, as UAC
  does: Administrators deny-only, privileges trimmed, Medium integrity, and
  the token's owner and default DACL set to the person (without that last
  step the process cannot open its own objects and dies at start).
- **a per-user install**: one worker, for the installing person, as this
  agent's own child.

On a Linux system install root runs the workers (`eugene-plexus-site-
worker@<uid>.service`), never this unprivileged agent.

Workers run the site host's own installed program, from the administrator-
only install directory. Each is put in a job of its own that closes with
this agent, so an agent restart never leaves one running beside its
replacement.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .site_links import LinkStore, not_a_person

log = logging.getLogger(__name__)

RESTART_SECONDS = (2.0, 5.0, 15.0, 60.0)


@dataclass
class WorkerProgram:
    """What a worker runs: the site host's interpreter and the arguments
    every worker shares."""

    python: Path
    channel: str
    host: str
    servers: Path
    protect: tuple[Path, ...] = ()
    #: A per-user install (J38): the site host runs as the one person it
    #: serves, so their worker shares its account, and says so.
    shared: bool = False

    def argv(self, account: str) -> list[str]:
        args = [
            str(self.python),
            "-I",
            "-m",
            "eugene_plexus_site_host.worker",
            "--account",
            account,
            "--channel",
            self.channel,
            "--host",
            self.host,
            "--servers",
            str(self.servers),
        ]
        for path in self.protect:
            args += ["--protect", str(path)]
        if self.shared:
            args.append("--shared-account")
        return args


@dataclass
class _Running:
    handle: Any
    pid: int
    session: int | None
    program: WorkerProgram
    #: Windows: the worker's own job, which kills it when this agent goes.
    job: Any = None
    started: float = field(default_factory=time.perf_counter)


class _Backoff:
    def __init__(self) -> None:
        self._next: dict[str, tuple[float, int]] = {}

    def ready(self, account: str) -> bool:
        due = self._next.get(account)
        return due is None or time.perf_counter() >= due[0]

    def failed(self, account: str, ran: float) -> None:
        tries = 0 if ran > 300 else self._next.get(account, (0.0, -1))[1] + 1
        delay = RESTART_SECONDS[min(tries, len(RESTART_SECONDS) - 1)]
        self._next[account] = (time.perf_counter() + delay, tries)


# --- Windows: a token for the person -------------------------------------------------


def _limited(token: Any) -> Any:
    """The token a worker runs with: UAC's filtered one when Windows gave it,
    else the same filter by hand. Never a token with administrator power."""
    import win32security

    kind = win32security.GetTokenInformation(token, win32security.TokenElevationType)
    if kind == 3:  # TokenElevationTypeLimited: already UAC's filtered token.
        return token
    if kind == 2:  # Full: its linked token is the filtered one.
        return win32security.GetTokenInformation(token, win32security.TokenLinkedToken)
    if not _admin_enabled(token):
        return token
    return _filter_by_hand(token)


def _admin_enabled(token: Any) -> bool:
    import win32security

    admins = win32security.ConvertStringSidToSid("S-1-5-32-544")
    for sid, attributes in win32security.GetTokenInformation(token, win32security.TokenGroups):
        if sid == admins and attributes & 0x4:  # SE_GROUP_ENABLED
            return True
    return False


def _filter_by_hand(token: Any) -> Any:
    """CreateRestrictedToken(DISABLE_MAX_PRIVILEGE | LUA_TOKEN), Administrators
    deny-only, Medium integrity, owner and default DACL the person's: the
    shape §2.4.1 measured working, on Amish_Station, 2026-10-06."""
    # For the type checker as much as the runtime: CI type-checks on Linux,
    # where ctypes has no WinDLL (process_io.py says why not `type: ignore`).
    if sys.platform != "win32":
        raise OSError("filtering a Windows token is a Windows-only API")
    import ctypes
    from ctypes import wintypes

    import pywintypes
    import win32security

    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class SidAndAttributes(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    adv.CreateRestrictedToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(SidAndAttributes),
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    adv.SetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    adv.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    adv.GetSecurityDescriptorDacl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    adv.GetLengthSid.argtypes = [ctypes.c_void_p]
    k32.LocalFree.argtypes = [ctypes.c_void_p]

    def sid(text: str) -> ctypes.c_void_p:
        value = ctypes.c_void_p()
        if not adv.ConvertStringSidToSidW(text, ctypes.byref(value)):
            raise OSError(ctypes.get_last_error(), "ConvertStringSidToSid")
        return value

    user_sid = win32security.ConvertSidToStringSid(
        win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    )
    admins, medium, user = sid("S-1-5-32-544"), sid("S-1-16-8192"), sid(user_sid)
    restricted = wintypes.HANDLE()
    sd = ctypes.c_void_p()
    try:
        disable = (SidAndAttributes * 1)(SidAndAttributes(admins.value, 0))
        if not adv.CreateRestrictedToken(
            int(token), 0x1 | 0x4, 1, disable, 0, None, 0, None, ctypes.byref(restricted)
        ):
            raise OSError(ctypes.get_last_error(), "CreateRestrictedToken")

        class Label(ctypes.Structure):
            _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

        label = Label(medium.value, 0x20)  # SE_GROUP_INTEGRITY
        size = ctypes.sizeof(label) + adv.GetLengthSid(medium)
        if not adv.SetTokenInformation(restricted, 25, ctypes.byref(label), size):
            raise OSError(ctypes.get_last_error(), "SetTokenInformation(IntegrityLevel)")
        owner = ctypes.c_void_p(user.value)
        if not adv.SetTokenInformation(restricted, 4, ctypes.byref(owner), ctypes.sizeof(owner)):
            raise OSError(ctypes.get_last_error(), "SetTokenInformation(Owner)")
        if not adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f"D:(A;;GA;;;SY)(A;;GA;;;{user_sid})", 1, ctypes.byref(sd), None
        ):
            raise OSError(ctypes.get_last_error(), "ConvertStringSecurityDescriptor")
        present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
        adv.GetSecurityDescriptorDacl(
            sd, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)
        )
        holder = ctypes.c_void_p(dacl.value)
        if not adv.SetTokenInformation(restricted, 6, ctypes.byref(holder), ctypes.sizeof(holder)):
            raise OSError(ctypes.get_last_error(), "SetTokenInformation(DefaultDacl)")
        return pywintypes.HANDLE(restricted.value)
    finally:
        for value in (admins, medium, user, sd):
            if value.value:
                k32.LocalFree(value)


def _worker_job() -> Any:
    """A job for one worker, which kills it when its last handle closes.

    One per worker, never one for all: a process already in a job joins
    another only while that one is empty or sits on its own job's chain.
    An agent that is itself in a job (Task Scheduler puts every task in
    one) passes it to each child, so a shared job took the first worker
    and refused the second with *Access is denied* (the first Windows run).
    An empty job is always accepted."""
    import win32job

    job = win32job.CreateJobObject(None, "")
    info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
    return job


def _close(handle: Any) -> None:
    if handle is not None:
        with contextlib.suppress(Exception):
            handle.Close()


def _session_users() -> dict[str, int]:
    """Each signed-in account's SID and one session of theirs, an active
    session preferred over a disconnected one."""
    import win32security
    import win32ts

    found: dict[str, tuple[int, int]] = {}
    for session in win32ts.WTSEnumerateSessions(win32ts.WTS_CURRENT_SERVER_HANDLE):
        ident, state = session["SessionId"], session["State"]
        if state not in (win32ts.WTSActive, win32ts.WTSDisconnected):
            continue
        try:
            token = win32ts.WTSQueryUserToken(ident)
        except Exception:
            continue
        try:
            user = win32security.ConvertSidToStringSid(
                win32security.GetTokenInformation(token, win32security.TokenUser)[0]
            )
        finally:
            token.Close()
        rank = 0 if state == win32ts.WTSActive else 1
        if user not in found or rank < found[user][1]:
            found[user] = (ident, rank)
    return {user: ident for user, (ident, _) in found.items()}


class WindowsStarter:
    """Workers for linked accounts that are signed in, as LocalSystem."""

    def __init__(self, links: LinkStore) -> None:
        self.links = links
        self.running: dict[str, _Running] = {}
        self.backoff = _Backoff()
        self.program: WorkerProgram | None = None

    def _exited(self, entry: _Running) -> bool:
        import win32event

        return bool(win32event.WaitForSingleObject(entry.handle, 0) == win32event.WAIT_OBJECT_0)

    def _stop(self, account: str) -> None:
        import win32process

        entry = self.running.pop(account, None)
        if entry is not None:
            with contextlib.suppress(Exception):
                win32process.TerminateProcess(entry.handle, 1)
            _close(entry.job)

    def step(self) -> None:
        program = self.program
        linked = {link.account for link in self.links.load() if not not_a_person(link.account)}
        for account, entry in list(self.running.items()):
            if self._exited(entry):
                self.running.pop(account)
                _close(entry.job)
                self.backoff.failed(account, time.perf_counter() - entry.started)
                log.info("a site worker for %s ended", account)
            elif account not in linked or program is None or entry.program != program:
                self._stop(account)
        if program is None or not linked:
            return
        sessions = _session_users()
        for account in linked:
            if account in self.running or account not in sessions:
                continue
            if not self.backoff.ready(account):
                continue
            try:
                self._start(account, sessions[account], program)
            except Exception as exc:
                self.backoff.failed(account, 0.0)
                log.warning("could not start the site worker for %s: %s", account, exc)

    def _start(self, account: str, session: int, program: WorkerProgram) -> None:
        import win32con
        import win32job
        import win32process
        import win32profile
        import win32security
        import win32ts

        token = win32ts.WTSQueryUserToken(session)
        user = win32security.ConvertSidToStringSid(
            win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        )
        if user != account:
            raise RuntimeError("the session's account is not the linked one")
        limited = _limited(token)
        environment = win32profile.CreateEnvironmentBlock(limited, False)
        startup = win32process.STARTUPINFO()
        startup.lpDesktop = "winsta0\\default"
        startup.dwFlags = win32con.STARTF_USESHOWWINDOW
        startup.wShowWindow = win32con.SW_HIDE
        flags = (
            win32con.CREATE_SUSPENDED
            | win32process.CREATE_NO_WINDOW
            | win32con.CREATE_UNICODE_ENVIRONMENT
            | win32process.CREATE_NEW_PROCESS_GROUP
        )
        job = _worker_job()
        try:
            process, thread, pid, _tid = win32process.CreateProcessAsUser(
                limited,
                None,
                subprocess.list2cmdline(program.argv(account)),
                None,
                None,
                False,
                flags,
                environment,
                environment.get("USERPROFILE"),
                startup,
            )
        except Exception:
            _close(job)
            raise
        try:
            win32job.AssignProcessToJobObject(job, process)
        except Exception:
            win32process.TerminateProcess(process, 1)
            _close(job)
            raise
        win32process.ResumeThread(thread)
        thread.Close()
        self.running[account] = _Running(process, pid, session, program, job)
        log.info("started the site worker for %s in session %s (pid %s)", account, session, pid)

    def connected_accounts(self) -> set[str]:
        return set(self.running)

    def close(self) -> None:
        for account in list(self.running):
            self._stop(account)


class ChildStarter:
    """A per-user install's one worker: the installing person's, as this
    agent's own child."""

    def __init__(self, account: str) -> None:
        self.account = account
        self.program: WorkerProgram | None = None
        self.process: asyncio.subprocess.Process | None = None
        self._program_running: WorkerProgram | None = None
        self.backoff = _Backoff()
        self._started = 0.0

    async def step(self) -> None:
        process = self.process
        if process is not None and process.returncode is not None:
            self.process = None
            self.backoff.failed(self.account, time.perf_counter() - self._started)
        if self.process is not None and self._program_running != self.program:
            await self.stop()
        if self.program is None or self.process is not None:
            return
        if not self.backoff.ready(self.account):
            return
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.process = await asyncio.create_subprocess_exec(
            *self.program.argv(self.account),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            **kwargs,
        )
        self._program_running, self._started = self.program, time.perf_counter()

    async def stop(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
