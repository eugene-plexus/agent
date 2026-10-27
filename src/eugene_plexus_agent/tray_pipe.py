r"""The tray icon's local channel: a named pipe and the caller's Windows account (row 35).

The tray asks the agent to pause and resume this machine's models. It
holds **no Eugene credential** to do it (TRAY-1, Troy 2026-10-21): a token
on disk in a person's session is something to steal, and a client key
that reached Pause would widen what every client key may do. Windows'
own sign-in decides instead, with the rule Stop Eugene already uses: **an
account that may stop the Eugene service may pause its models.** That
grants nothing new; those accounts can already stop the whole service.

## The server (in the agent, on Windows)

* `\\.\pipe\eugene-plexus-agent-<port>`, the port keeping two installs on
  one machine apart. Created with `FILE_FLAG_FIRST_PIPE_INSTANCE`, one
  instance and no more, and `PIPE_REJECT_REMOTE_CLIENTS`; its DACL denies
  `NETWORK` too. Nothing another machine can reach.
* **Who may open it** is read from the service's own DACL
  (`QueryServiceObjectSecurity`): SYSTEM, Administrators, and each named
  account or group (`S-1-5-21-…`, `S-1-12-1-…`) an allow ACE gives
  `SERVICE_STOP`. Broad groups (`WD`, `AU`, `BU`, `IU`, `SU`, …) are never
  added, whatever the service grants them; every ACE that denies the stop
  right is copied as a deny. Clients get read and `FILE_WRITE_DATA` only --
  never `FILE_APPEND_DATA`, which on a pipe is `FILE_CREATE_PIPE_INSTANCE`.
  Without the service the agent's own account takes the third place.
  The DACL is read again before each connection and applied in place.
* **Each connection is checked again, and that check decides.** After
  the request is read, the agent impersonates the caller, opens the
  thread's token, reverts at once (a failed revert ends the channel), and
  asks Windows' `AccessCheck` whether that token may stop the service --
  groups, deny ACEs and all, with the DACL read now. A grant removed with
  `sc sdset` takes effect at the next connection.
* One JSON line in and one out, at most 4 KiB in, a 5-second read
  timeout; one connection at a time. `state`, `pause` and `resume`; any
  other verb gets `{"ok": false, "error": "unknown verb"}`.
* **A name already taken** is reported (`NodeTrayChannel.taken`) and
  logged with the holder's process, and tried again every 30 seconds.
  The holder is read over an anonymous connection, so it cannot act as
  the agent.

## The client (in the tray)

`call` opens the pipe at identification level -- the agent learns who
asks and cannot act as them -- and, before it sends anything, checks
`GetNamedPipeServerProcessId` against the service's own process
(`QueryServiceStatusEx`). Any other server is an imposter and gets
nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

PIPE_PREFIX = "\\\\.\\pipe\\eugene-plexus-agent-"
MAX_REQUEST_BYTES = 4096
READ_TIMEOUT_SECONDS = 5.0
#: Pause and Resume stop and start engines; a state read is quick.
ANSWER_TIMEOUT_SECONDS = 180.0
RETRY_SECONDS = 30.0

SERVING = "serving"
TAKEN = "taken"
UNAVAILABLE = "unavailable"

# --- access masks -------------------------------------------------------------

SERVICE_STOP = 0x0020
GENERIC_ALL = 0x10000000
GENERIC_EXECUTE = 0x20000000
#: FILE_ALL_ACCESS: SYSTEM and Administrators.
PIPE_ALL = 0x001F01FF
#: FILE_GENERIC_READ | FILE_WRITE_DATA: what a client of the pipe may do.
#: Not FILE_APPEND_DATA, which on a pipe is FILE_CREATE_PIPE_INSTANCE.
CLIENT_ACCESS = 0x0012008B
#: The generic mapping of a service object (winsvc.h): read, write, execute, all.
SERVICE_GENERIC_MAPPING = (0x0002008D, 0x00020002, 0x00020170, 0x000F01FF)

#: SDDL's two-letter rights, as they apply to a service's DACL.
_SDDL_RIGHTS = {
    "GA": GENERIC_ALL,
    "GX": GENERIC_EXECUTE,
    "GW": 0x40000000,
    "GR": 0x80000000,
    "SD": 0x00010000,
    "RC": 0x00020000,
    "WD": 0x00040000,
    "WO": 0x00080000,
    "CC": 0x0001,
    "DC": 0x0002,
    "LC": 0x0004,
    "SW": 0x0008,
    "RP": 0x0010,
    "WP": 0x0020,
    "DT": 0x0040,
    "LO": 0x0080,
    "CR": 0x0100,
    "FA": 0x001F01FF,
    "FR": 0x00120089,
    "FW": 0x00120116,
    "FX": 0x001200A0,
    "KA": 0x000F003F,
    "KR": 0x00020019,
    "KW": 0x00020006,
    "KX": 0x00020019,
}

#: The two the pipe always admits, in full.
_ALWAYS = ("SY", "BA")
#: A named account (this machine's or a domain's), or one of their own groups;
#: an Entra ID account. Never a well-known broad group.
_SPECIFIC_PREFIXES = ("S-1-5-21-", "S-1-12-1-")


def pipe_name(port: int) -> str:
    return f"{PIPE_PREFIX}{port}"


# --- the DACL, as data (testable anywhere) ------------------------------------


@dataclass(frozen=True)
class Ace:
    """One ACE of a DACL in SDDL. `allow` is None for a kind not understood
    (an object or callback allow); `mask` is None for rights not understood."""

    allow: bool | None
    mask: int | None
    sid: str


def _mask(rights: str) -> int | None:
    text = rights.strip()
    if text.lower().startswith("0x"):
        try:
            return int(text, 16)
        except ValueError:
            return None
    if text.isdigit():
        return int(text)
    if len(text) % 2:
        return None
    mask = 0
    for index in range(0, len(text), 2):
        bit = _SDDL_RIGHTS.get(text[index : index + 2].upper())
        if bit is None:
            return None
        mask |= bit
    return mask


def parse_dacl(sddl: str) -> list[Ace]:
    """The ACEs of the `D:` part of `sddl`, in order. Inherit-only ACEs are
    left out: they do not apply to the object itself."""
    start = sddl.find("D:")
    if start < 0:
        return []
    index = start + 2
    while index < len(sddl) and sddl[index] != "(" and not sddl.startswith("S:", index):
        index += 1
    aces: list[Ace] = []
    while index < len(sddl) and sddl[index] == "(":
        depth, end = 0, index
        while end < len(sddl):
            if sddl[end] == "(":
                depth += 1
            elif sddl[end] == ")":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        fields = sddl[index + 1 : end].split(";")
        index = end + 1
        if len(fields) < 6:
            continue
        kind, flags, rights, sid = fields[0].upper(), fields[1].upper(), fields[2], fields[5]
        if "IO" in [flags[i : i + 2] for i in range(0, len(flags), 2)]:
            continue
        allow = True if kind == "A" else False if kind in ("D", "OD", "XD") else None
        aces.append(Ace(allow=allow, mask=_mask(rights), sid=sid.strip()))
    return aces


def _stops(mask: int | None) -> bool:
    return mask is not None and bool(mask & (SERVICE_STOP | GENERIC_ALL | GENERIC_EXECUTE))


@dataclass(frozen=True)
class PipeGrants:
    """Who the pipe admits (`allowed`, SYSTEM and Administrators first) and
    who it refuses outright (`denied`)."""

    allowed: tuple[str, ...]
    denied: tuple[str, ...] = field(default=())


def grants_from_service_dacl(sddl: str) -> PipeGrants:
    """The pipe's grants from the service's DACL: the accounts that may stop it.

    A deny ACE whose rights are not understood counts as denying the stop:
    every doubt refuses.
    """
    allowed: list[str] = list(_ALWAYS)
    denied: list[str] = []
    for ace in parse_dacl(sddl):
        if ace.allow is False and (ace.mask is None or _stops(ace.mask)):
            denied.append(ace.sid)
        elif ace.allow is True and _stops(ace.mask) and ace.sid.startswith(_SPECIFIC_PREFIXES):
            allowed.append(ace.sid)
    return PipeGrants(allowed=tuple(dict.fromkeys(allowed)), denied=tuple(dict.fromkeys(denied)))


def grants_for_own_account(sid: str) -> PipeGrants:
    """An install without the service: the agent's own account."""
    return PipeGrants(allowed=(*_ALWAYS, sid))


def pipe_sddl(grants: PipeGrants) -> str:
    """The pipe's protected DACL: NETWORK and every denied SID refused first."""
    parts = ["D:P", f"(D;;{PIPE_ALL:#x};;;NU)"]
    parts += [f"(D;;{PIPE_ALL:#x};;;{sid})" for sid in grants.denied]
    parts += [f"(A;;{PIPE_ALL:#x};;;{sid})" for sid in _ALWAYS if sid in grants.allowed]
    parts += [f"(A;;{CLIENT_ACCESS:#x};;;{sid})" for sid in grants.allowed if sid not in _ALWAYS]
    return "".join(parts)


# --- the protocol, as data -------------------------------------------------------


VERBS = ("state", "pause", "resume")
DENIED = "This account may not pause models on this machine."


def parse_request(line: bytes) -> tuple[str | None, dict[str, Any] | None]:
    """`(verb, None)` for a request to answer, `(None, reply)` for one refused."""
    try:
        request = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, {"ok": False, "error": "bad request"}
    verb = request.get("verb") if isinstance(request, dict) else None
    if verb not in VERBS:
        return None, {"ok": False, "error": "unknown verb"}
    return str(verb), None


# --- Windows -----------------------------------------------------------------------


def _win32() -> Any:
    """pywin32's modules, imported only where they exist."""
    import pywintypes
    import win32api
    import win32event
    import win32file
    import win32pipe
    import win32security
    import win32service
    import winerror

    return _Win(
        pywintypes=pywintypes,
        api=win32api,
        event=win32event,
        file=win32file,
        pipe=win32pipe,
        security=win32security,
        service=win32service,
        error=winerror,
    )


@dataclass
class _Win:
    pywintypes: Any
    api: Any
    event: Any
    file: Any
    pipe: Any
    security: Any
    service: Any
    error: Any


class _Fatal(Exception):
    """The channel cannot go on safely: it ends."""


def own_service() -> str | None:
    """The service this process is, or None (an install without one)."""
    if sys.platform != "win32":
        return None
    try:
        w = _win32()
        scm = w.service.OpenSCManager(None, None, w.service.SC_MANAGER_ENUMERATE_SERVICE)
        try:
            for entry in w.service.EnumServicesStatusEx(
                scm, w.service.SERVICE_WIN32, w.service.SERVICE_ACTIVE, None
            ):
                if entry.get("ProcessId") == os.getpid():
                    return str(entry["ServiceName"])
        finally:
            w.service.CloseServiceHandle(scm)
    except Exception:
        log.debug("could not ask which service this process is", exc_info=True)
    return None


def service_dacl_sddl(name: str) -> str:
    """The service's DACL, as SDDL."""
    w = _win32()
    scm = w.service.OpenSCManager(None, None, w.service.SC_MANAGER_CONNECT)
    try:
        handle = w.service.OpenService(scm, name, 0x00020000)  # READ_CONTROL
        try:
            sd = w.service.QueryServiceObjectSecurity(handle, w.security.DACL_SECURITY_INFORMATION)
        finally:
            w.service.CloseServiceHandle(handle)
    finally:
        w.service.CloseServiceHandle(scm)
    return str(
        w.security.ConvertSecurityDescriptorToStringSecurityDescriptor(
            sd, w.security.SDDL_REVISION_1, w.security.DACL_SECURITY_INFORMATION
        )
    )


def service_pid(name: str) -> int | None:
    """The process of a running service, or None."""
    if sys.platform != "win32":
        return None
    try:
        w = _win32()
        scm = w.service.OpenSCManager(None, None, w.service.SC_MANAGER_CONNECT)
        try:
            handle = w.service.OpenService(scm, name, w.service.SERVICE_QUERY_STATUS)
            try:
                status = w.service.QueryServiceStatusEx(handle)
            finally:
                w.service.CloseServiceHandle(handle)
        finally:
            w.service.CloseServiceHandle(scm)
    except Exception:
        return None
    if status.get("CurrentState") != w.service.SERVICE_RUNNING:
        return None
    pid = int(status.get("ProcessId") or 0)
    return pid or None


def own_sid() -> str:
    w = _win32()
    token = w.security.OpenProcessToken(w.api.GetCurrentProcess(), w.security.TOKEN_QUERY)
    try:
        user = w.security.GetTokenInformation(token, w.security.TokenUser)[0]
        return str(w.security.ConvertSidToStringSid(user))
    finally:
        w.api.CloseHandle(token)


def may_stop(dacl_sddl: str, token: int) -> bool:
    """Windows' own `AccessCheck`: may this token stop a service with this DACL?"""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class GenericMapping(ctypes.Structure):
        _fields_ = [
            ("read", wintypes.DWORD),
            ("write", wintypes.DWORD),
            ("execute", wintypes.DWORD),
            ("all", wintypes.DWORD),
        ]

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.ULONG),
    ]
    convert.restype = wintypes.BOOL
    check = advapi.AccessCheck
    check.argtypes = [
        ctypes.c_void_p,
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(GenericMapping),
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.BOOL),
    ]
    check.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor = ctypes.c_void_p()
    # AccessCheck needs an owner and a group; neither bears on SERVICE_STOP.
    if not convert("O:SYG:SY" + dacl_sddl, 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        mapping = GenericMapping(*SERVICE_GENERIC_MAPPING)
        privileges = ctypes.create_string_buffer(256)
        size = wintypes.DWORD(256)
        granted = wintypes.DWORD(0)
        status = wintypes.BOOL(0)
        if not check(
            descriptor,
            wintypes.HANDLE(token),
            SERVICE_STOP,
            ctypes.byref(mapping),
            privileges,
            ctypes.byref(size),
            ctypes.byref(granted),
            ctypes.byref(status),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(status.value) and bool(granted.value & SERVICE_STOP)
    finally:
        kernel.LocalFree(descriptor)


def _wait(w: Any, overlapped: Any, handle: Any, seconds: float, stop: Any = None) -> int | None:
    """Bytes moved by an overlapped call, or None when it timed out or the
    channel is stopping (the call is cancelled)."""
    events = [overlapped.hEvent] + ([stop] if stop is not None else [])
    which = w.event.WaitForMultipleObjects(events, False, max(0, int(seconds * 1000)))
    if which != w.event.WAIT_OBJECT_0:
        w.file.CancelIo(handle)
        with contextlib.suppress(w.pywintypes.error):
            w.file.GetOverlappedResult(handle, overlapped, True)
        return None
    return int(w.file.GetOverlappedResult(handle, overlapped, False))


def _overlapped(w: Any) -> Any:
    overlapped = w.pywintypes.OVERLAPPED()
    overlapped.hEvent = w.event.CreateEvent(None, True, False, None)
    return overlapped


def _read_line(w: Any, handle: Any, seconds: float, stop: Any = None) -> bytes | None:
    """One line, at most MAX_REQUEST_BYTES, within `seconds`; None otherwise."""
    deadline = time.perf_counter() + seconds
    data = b""
    while b"\n" not in data:
        if len(data) > MAX_REQUEST_BYTES:
            return None
        left = deadline - time.perf_counter()
        if left <= 0:
            return None
        overlapped = _overlapped(w)
        buffer = w.file.AllocateReadBuffer(4096)
        try:
            w.file.ReadFile(handle, buffer, overlapped)
        except w.pywintypes.error:
            return None
        try:
            count = _wait(w, overlapped, handle, left, stop)
        except w.pywintypes.error:
            return None
        if not count:
            return None
        data += bytes(buffer[:count])
    line = data.split(b"\n", 1)[0]
    return line if len(line) <= MAX_REQUEST_BYTES else None


def _write(w: Any, handle: Any, data: bytes, seconds: float) -> bool:
    overlapped = _overlapped(w)
    try:
        w.file.WriteFile(handle, data, overlapped)
        return _wait(w, overlapped, handle, seconds) == len(data)
    except w.pywintypes.error:
        return False


class TrayPipe:
    """The agent's end of the tray's channel. `start`, then `stop` at shutdown."""

    def __init__(
        self,
        *,
        port: int,
        answer: Callable[[str], Coroutine[Any, Any, dict[str, Any]]],
        loop: asyncio.AbstractEventLoop,
        read_timeout: float = READ_TIMEOUT_SECONDS,
    ) -> None:
        self.name = pipe_name(port)
        self.status = UNAVAILABLE
        self.holder: int | None = None
        self._answer = answer
        self._loop = loop
        self._read_timeout = read_timeout
        self._service: str | None = None
        self._own: str | None = None
        self._handle: Any = None
        self._sddl: str | None = None
        self._stop: Any = None
        self._thread: threading.Thread | None = None
        self._w: Any = None
        self._reported: str | None = None

    # --- life ---------------------------------------------------------------

    def start(self) -> None:
        try:
            self._w = _win32()
        except ImportError:
            # `sys.prefix`, not the executable: under the service that is
            # pythonservice.exe, which names no environment.
            log.warning(
                "The tray's channel %s is not served: pywin32 is not installed in %s, so "
                "the tray cannot pause models.",
                self.name,
                sys.prefix,
            )
            return
        self._stop = self._w.event.CreateEvent(None, True, False, None)
        self._service = own_service()
        try:
            self._own = own_sid()
        except Exception:
            log.warning("could not read this agent's own account", exc_info=True)
        self._create()
        self._thread = threading.Thread(target=self._run, name="tray-channel", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._w.event.SetEvent(self._stop)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._close()

    def _close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                self._w.api.CloseHandle(handle)
            except Exception:
                log.debug("closing the tray channel", exc_info=True)

    # --- who may connect ----------------------------------------------------------

    def _grants(self) -> tuple[PipeGrants, str | None]:
        """The pipe's grants, and the service DACL they came from (None
        without the service, or when it could not be read: then only
        SYSTEM and Administrators get in)."""
        if self._service is not None:
            try:
                dacl = service_dacl_sddl(self._service)
                return grants_from_service_dacl(dacl), dacl
            except Exception:
                log.warning(
                    "could not read who may stop the %s service; the tray channel admits "
                    "SYSTEM and Administrators only",
                    self._service,
                    exc_info=True,
                )
                return PipeGrants(allowed=_ALWAYS), None
        if self._own is None:
            return PipeGrants(allowed=_ALWAYS), None
        return grants_for_own_account(self._own), None

    def _descriptor(self, sddl: str) -> Any:
        return self._w.security.ConvertStringSecurityDescriptorToSecurityDescriptor(
            sddl, self._w.security.SDDL_REVISION_1
        )

    def _create(self) -> None:
        w = self._w
        grants, _ = self._grants()
        sddl = pipe_sddl(grants)
        attributes = w.security.SECURITY_ATTRIBUTES()
        attributes.SECURITY_DESCRIPTOR = self._descriptor(sddl)
        attributes.bInheritHandle = False
        open_mode = (
            w.pipe.PIPE_ACCESS_DUPLEX
            | w.file.FILE_FLAG_OVERLAPPED
            | 0x00080000  # FILE_FLAG_FIRST_PIPE_INSTANCE
            | 0x00040000  # WRITE_DAC: the DACL is refreshed in place
        )
        pipe_mode = (
            w.pipe.PIPE_TYPE_BYTE
            | w.pipe.PIPE_READMODE_BYTE
            | w.pipe.PIPE_WAIT
            | w.pipe.PIPE_REJECT_REMOTE_CLIENTS
        )
        try:
            handle = w.pipe.CreateNamedPipe(
                self.name, open_mode, pipe_mode, 1, 65536, 65536, 0, attributes
            )
        except w.pywintypes.error as exc:
            if exc.winerror in (w.error.ERROR_ACCESS_DENIED, w.error.ERROR_PIPE_BUSY):
                holder = self._holder()
                if self.status != TAKEN or holder != self.holder:
                    log.warning(
                        "The tray's channel %s is held by another program (process %s), so "
                        "the tray cannot pause models. Trying again every %.0f seconds.",
                        self.name,
                        holder if holder is not None else "unknown",
                        RETRY_SECONDS,
                    )
                self.status, self.holder = TAKEN, holder
            else:
                if self._reported != str(exc):
                    log.warning(
                        "The tray's channel %s could not be opened (%s); trying again every "
                        "%.0f seconds.",
                        self.name,
                        exc,
                        RETRY_SECONDS,
                    )
                self._reported = str(exc)
                self.status = UNAVAILABLE
            return
        self._handle, self._sddl = handle, sddl
        self.status, self.holder = SERVING, None
        log.info(
            "The tray's channel %s admits the accounts that may stop %s",
            self.name,
            f"the {self._service} service" if self._service else "this agent's own account",
        )

    def _holder(self) -> int | None:
        """The process holding the name, asked anonymously so it learns nothing."""
        w = self._w
        try:
            handle = w.file.CreateFile(
                self.name,
                w.file.GENERIC_READ,
                0,
                None,
                w.file.OPEN_EXISTING,
                0x00100000 | w.file.SECURITY_ANONYMOUS,  # SECURITY_SQOS_PRESENT
                None,
            )
        except w.pywintypes.error:
            return None
        try:
            return int(w.pipe.GetNamedPipeServerProcessId(handle))
        except w.pywintypes.error:
            return None
        finally:
            w.api.CloseHandle(handle)

    def _refresh(self) -> None:
        """Apply the service's DACL as it is now, if it changed."""
        w = self._w
        grants, _ = self._grants()
        sddl = pipe_sddl(grants)
        if sddl == self._sddl:
            return
        dacl = self._descriptor(sddl).GetSecurityDescriptorDacl()
        w.security.SetSecurityInfo(
            self._handle,
            w.security.SE_KERNEL_OBJECT,
            w.security.DACL_SECURITY_INFORMATION | w.security.PROTECTED_DACL_SECURITY_INFORMATION,
            None,
            None,
            dacl,
            None,
        )
        self._sddl = sddl
        log.info("The tray's channel follows a change to who may stop the service")

    # --- serving -------------------------------------------------------------------------

    def _stopping(self) -> bool:
        w = self._w
        return bool(w.event.WaitForSingleObject(self._stop, 0) == w.event.WAIT_OBJECT_0)

    def _run(self) -> None:
        w = self._w
        while not self._stopping():
            if self._handle is None:
                if w.event.WaitForSingleObject(self._stop, int(RETRY_SECONDS * 1000)) == 0:
                    break
                self._create()
                continue
            connected = False
            try:
                self._refresh()
                connected = self._connect()
                if not connected:
                    break
                self._serve_one()
            except _Fatal as exc:
                log.error("The tray's channel %s stopped: %s", self.name, exc)
                self.status = UNAVAILABLE
                break
            except Exception:
                log.exception("The tray's channel %s: a connection failed", self.name)
            finally:
                if connected and self._handle is not None:
                    with contextlib.suppress(w.pywintypes.error):
                        w.pipe.DisconnectNamedPipe(self._handle)
        self._close()

    def _connect(self) -> bool:
        w = self._w
        overlapped = _overlapped(w)
        result = w.pipe.ConnectNamedPipe(self._handle, overlapped)
        if result == w.error.ERROR_PIPE_CONNECTED:
            return True
        which = w.event.WaitForMultipleObjects(
            [overlapped.hEvent, self._stop], False, w.event.INFINITE
        )
        if which != w.event.WAIT_OBJECT_0:
            w.file.CancelIo(self._handle)
            return False
        w.file.GetOverlappedResult(self._handle, overlapped, False)
        return True

    def _caller(self) -> tuple[bool, str]:
        """May the client on the pipe pause models here? And its SID, for the log.

        Impersonate only to open the thread's token, and revert before
        anything else. A revert that fails ends the channel: the thread
        must never go on as somebody else.
        """
        w = self._w
        try:
            w.security.ImpersonateNamedPipeClient(self._handle)
        except w.pywintypes.error as exc:
            log.warning("The tray's channel could not tell who is asking: %s", exc)
            return False, "a caller it could not identify"
        token = None
        try:
            token = w.security.OpenThreadToken(
                w.api.GetCurrentThread(), w.security.TOKEN_QUERY, True
            )
        except w.pywintypes.error as exc:
            log.warning("The tray's channel could not read the caller's account: %s", exc)
        finally:
            try:
                w.security.RevertToSelf()
            except Exception as exc:
                raise _Fatal(f"could not stop acting as the caller ({exc})") from exc
        if token is None:
            return False, "a caller it could not identify"
        try:
            user = w.security.GetTokenInformation(token, w.security.TokenUser)[0]
            sid = str(w.security.ConvertSidToStringSid(user))
            if self._service is not None:
                try:
                    dacl = service_dacl_sddl(self._service)
                except Exception:
                    log.warning("could not read who may stop %s", self._service, exc_info=True)
                    return False, sid
                return may_stop(dacl, int(token)), sid
            if sid in (self._own, "S-1-5-18"):
                return True, sid
            administrators = w.security.ConvertStringSidToSid("S-1-5-32-544")
            return bool(w.security.CheckTokenMembership(token, administrators)), sid
        finally:
            w.api.CloseHandle(token)

    def _serve_one(self) -> None:
        w = self._w
        line = _read_line(w, self._handle, self._read_timeout, self._stop)
        if line is None:
            log.info("The tray's channel dropped a client that sent no request in time")
            return
        allowed, sid = self._caller()
        reply: dict[str, Any] | None
        if not allowed:
            log.warning("The tray's channel refused %s: it may not stop the service", sid)
            reply = {"ok": False, "error": DENIED, "denied": True}
        else:
            verb, reply = parse_request(line)
            if verb is not None:
                if verb != "state":
                    log.info("The tray's channel: %s asks to %s models", sid, verb)
                reply = self._ask(verb)
        data = (json.dumps(reply) + "\n").encode("utf-8")
        if _write(w, self._handle, data, self._read_timeout):
            # Wait for the client to read and close; a disconnect before it
            # has read would throw the answer away.
            _read_line(w, self._handle, self._read_timeout, self._stop)

    def _ask(self, verb: str) -> dict[str, Any]:
        try:
            future = asyncio.run_coroutine_threadsafe(self._answer(verb), self._loop)
            answer: dict[str, Any] = future.result(timeout=ANSWER_TIMEOUT_SECONDS)
            return answer
        except Exception as exc:
            log.exception("The tray's channel could not %s models", verb)
            return {"ok": False, "error": f"Eugene could not {verb} the models: {exc}"}


# --- the tray's end ---------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """What the channel said. `status`: ok, denied, absent, imposter, busy, error."""

    status: str
    body: dict[str, Any]
    why: str = ""


def call(
    verb: str,
    *,
    port: int,
    service_name: str,
    timeout: float = READ_TIMEOUT_SECONDS,
    expected_pid: int | None = None,
) -> Answer:
    """Ask the agent over its channel, as the account this process runs as.

    Nothing is sent until the server is the service's own process
    (`expected_pid` overrides it, for a test).
    """
    if sys.platform != "win32":
        return Answer("absent", {}, "The tray's channel is on Windows only.")
    try:
        w = _win32()
    except ImportError:
        return Answer("error", {}, "The tray needs pywin32.")
    pid = expected_pid if expected_pid is not None else service_pid(service_name)
    if pid is None:
        return Answer("absent", {}, "Eugene is not running.")
    name = pipe_name(port)
    handle = None
    for attempt in range(2):
        try:
            handle = w.file.CreateFile(
                name,
                w.file.GENERIC_READ | 0x0002,  # FILE_WRITE_DATA, never FILE_APPEND_DATA
                0,
                None,
                w.file.OPEN_EXISTING,
                w.file.FILE_FLAG_OVERLAPPED | 0x00100000 | w.file.SECURITY_IDENTIFICATION,
                None,
            )
            break
        except w.pywintypes.error as exc:
            if exc.winerror == w.error.ERROR_ACCESS_DENIED:
                return Answer("denied", {}, DENIED)
            if exc.winerror == w.error.ERROR_PIPE_BUSY and attempt == 0:
                with contextlib.suppress(w.pywintypes.error):
                    w.pipe.WaitNamedPipe(name, 2000)
                continue
            if exc.winerror == w.error.ERROR_PIPE_BUSY:
                return Answer("busy", {}, "Eugene is busy with another request. Try again.")
            return Answer("absent", {}, "Eugene's tray channel is not open.")
    if handle is None:
        return Answer("busy", {}, "Eugene is busy with another request. Try again.")
    try:
        try:
            server = int(w.pipe.GetNamedPipeServerProcessId(handle))
        except w.pywintypes.error:
            server = -1
        if server != pid:
            return Answer("imposter", {}, "Eugene's tray channel is held by another program.")
        request = (json.dumps({"verb": verb}) + "\n").encode("utf-8")
        if not _write(w, handle, request, READ_TIMEOUT_SECONDS):
            return Answer("error", {}, "Eugene did not take the request.")
        line = _read_line_any(w, handle, timeout)
        if line is None:
            return Answer("error", {}, "Eugene did not answer in time.")
        try:
            body = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return Answer("error", {}, "Eugene's answer could not be read.")
        if not isinstance(body, dict):
            return Answer("error", {}, "Eugene's answer could not be read.")
        if body.get("denied"):
            return Answer("denied", body, DENIED)
        if not body.get("ok"):
            return Answer("error", body, str(body.get("error") or "Eugene refused."))
        return Answer("ok", body)
    finally:
        w.api.CloseHandle(handle)


def _read_line_any(w: Any, handle: Any, seconds: float) -> bytes | None:
    """An answer line of any length (a resume lists every runtime)."""
    deadline = time.perf_counter() + seconds
    data = b""
    while b"\n" not in data:
        left = deadline - time.perf_counter()
        if left <= 0:
            return None
        overlapped = _overlapped(w)
        buffer = w.file.AllocateReadBuffer(65536)
        try:
            w.file.ReadFile(handle, buffer, overlapped)
            count = _wait(w, overlapped, handle, left)
        except w.pywintypes.error:
            return None
        if not count:
            return None
        data += bytes(buffer[:count])
    return data.split(b"\n", 1)[0]
