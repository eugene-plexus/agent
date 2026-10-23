"""Operator-confirmed OS access, with private, write-ahead permission deltas.

The Workbench account can release a recorded delta, never supply one. Paths
and directory identities are checked again by the backend at the mutation.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import re
import stat
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Protocol

from ._generated.models import FolderAccess
from ._private_files import write_private

FILE = "folder-access.json"
_LOCK = threading.RLock()
_RUNNING: set[str] = set()
UNAVAILABLE = (
    "Workbench runs in Eugene's own account here, so it already reaches what Eugene reaches. "
    "Folder access needs a Windows service or a Linux system install."
)
MESSAGES = {
    "not_a_folder": "Choose an existing folder, not a link.",
    "link": "Choose an existing folder, not a link.",
    "drive_root": "Choose a folder inside the drive, not the whole drive.",
    "network": "Network folders are not supported.",
    "system": "Eugene does not give apps system folders.",
    "eugene": "This is Eugene's own folder.",
    "home": "This is inside a person's home. Use a Job Site to work in your own folders.",
    "unavailable": UNAVAILABLE,
}
WINDOWS_HOMES = ("Users",)
LINUX_SYSTEM = tuple(
    "/" + p
    for p in [
        "bin",
        "boot",
        "dev",
        "etc",
        "lib",
        "lib32",
        "lib64",
        "libx32",
        "opt",
        "proc",
        "root",
        "run",
        "sbin",
        "snap",
        "sys",
        "tmp",
        "usr",
        "var",
    ]
)


@dataclass(frozen=True)
class Facts:
    path: str
    is_dir: bool = True
    link: bool = False
    network: bool = False
    mount_root: bool = False
    eugene: tuple[str, ...] = ()
    homes: tuple[str, ...] = ()
    system: tuple[str, ...] = ()


def _path(value: str) -> PurePath:
    return (
        PureWindowsPath(value)
        if value.startswith("\\\\") or re.match(r"^[A-Za-z]:", value)
        else PurePosixPath(value)
    )


def _overlaps(path: PurePath, roots: tuple[str, ...]) -> bool:
    return any(
        path.is_relative_to(_path(root)) or _path(root).is_relative_to(path) for root in roots
    )


def refusal(facts: Facts) -> str | None:
    """Own prefix/data first; otherwise system, home, then other install ancestors."""
    path = _path(facts.path)
    if facts.path.startswith(("\\\\?\\", "\\\\.\\")):
        return "not_a_folder"
    if facts.network or facts.path.startswith(("\\\\", "//")):
        return "network"
    if facts.link:
        return "link"
    if (
        not facts.is_dir
        or not path.is_absolute()
        or any(ord(c) < 32 or ord(c) == 127 for c in facts.path)
    ):
        return "not_a_folder"
    if facts.mount_root or str(path) == path.anchor:
        return "drive_root"
    if any(path.is_relative_to(_path(root)) for root in facts.eugene):
        return "eugene"
    if isinstance(path, PureWindowsPath):
        drive = path.drive
        system = tuple(
            drive + "\\" + name
            for name in (
                "Windows",
                "Windows.old",
                "Program Files",
                "Program Files (x86)",
                "ProgramData",
                "Recovery",
                "System Volume Information",
                "$Recycle.Bin",
            )
        )
        homes = tuple(drive + "\\" + name for name in WINDOWS_HOMES)
    else:
        system, homes = LINUX_SYSTEM, ("/home",)
        if path.parts[1].startswith("lib"):
            return "system"
    if _overlaps(path, facts.system + system):
        return "system"
    if _overlaps(path, facts.homes + homes):
        return "home"
    if _overlaps(path, facts.eugene):
        return "eugene"
    return None


class Refused(ValueError):
    def __init__(self, cause: str, detail: str = "") -> None:
        self.cause = cause
        super().__init__(f"{MESSAGES[cause]} {detail}".strip())


class RemovalIncomplete(OSError):
    """Removal changed some permissions and named the items it could not change."""


class InspectionFailed(ValueError):
    """A folder could not be checked; its permissions must stay unchanged."""


class Permissions(Protocol):
    account: str

    def inspect(self, path: str) -> tuple[Facts, list[int]]: ...
    def grant(
        self, path: str, mode: str, save: Callable[[dict], None]
    ) -> tuple[dict, int, bool]: ...
    def revoke(self, path: str, delta: dict) -> bool: ...


def _same_identity(left: list[int], right: list[int]) -> bool:
    # Phase 2 Linux records included device and mount ids, both transient.
    return (left[1:4] if len(left) == 5 else left) == (right[1:4] if len(right) == 5 else right)


class Access:
    def __init__(
        self,
        file: Path,
        permissions: Permissions,
        *,
        process: Callable[[], int | None] = lambda: None,
    ) -> None:
        self.file, self.permissions = file, permissions
        self.process = process

    def _read(self) -> dict[str, dict]:
        try:
            data: dict[str, dict] = json.loads(self.file.read_text("utf-8"))
            return data
        except FileNotFoundError:
            return {}

    def _write(self, rows: dict[str, dict]) -> None:
        write_private(self.file, json.dumps(rows, indent=2))

    def _inspect(self, path: str) -> tuple[Facts, list[int]]:
        try:
            return self.permissions.inspect(path)
        except OSError as exc:
            cause = (
                f"Eugene's account cannot open {path}"
                if isinstance(exc, PermissionError)
                else f"Eugene cannot check {path}"
            )
            raise InspectionFailed(
                f"{cause}: {exc}. Check this folder's permissions and availability, then try again."
            ) from exc

    def listing(self, app: str) -> list[FolderAccess]:
        with _LOCK:
            rows = self._read()
            before = json.dumps(rows)
            result = []
            for row in rows.values():
                if row["app"] != app:
                    continue
                if row["state"] == "ready":
                    try:
                        facts, identity = self._inspect(row["path"])
                        if facts.link or not _same_identity(identity, row["identity"]):
                            row.update(
                                state="changed_outside",
                                message=(
                                    "The folder was replaced; "
                                    "its permissions were left as they are."
                                ),
                            )
                    except InspectionFailed as exc:
                        transient = dict(row)
                        transient.update(
                            state="changed_outside",
                            message=(
                                f"The folder cannot be checked: {exc}. "
                                "Its permissions were left as they are."
                            ),
                        )
                        result.append(FolderAccess.model_validate(transient))
                        continue
                if row["state"] in ("applying", "removing") and row["id"] not in _RUNNING:
                    row.update(
                        state="failed",
                        message=(
                            "The agent stopped during the permission change. "
                            "Remove this record to release its recorded additions."
                        ),
                    )
                if (
                    row["state"] == "ready"
                    and row["restartNeeded"]
                    and (pid := self.process())
                    and pid != row.get("process")
                ):
                    row.update(
                        restartNeeded=False, message=f"Workbench can read and write {row['path']}."
                    )
                result.append(FolderAccess.model_validate(row))
            if json.dumps(rows) != before:
                self._write(rows)
            return result

    def add(
        self,
        app: str,
        path: str,
        mode: str,
        operator: str,
        *,
        schedule: Callable[[Callable[[], None]], None] | None = None,
    ) -> FolderAccess:
        with _LOCK:
            facts, identity = self._inspect(path)
            if cause := refusal(facts):
                raise Refused(cause, path)
            rows = self._read()
            for existing in rows.values():
                if (
                    existing["app"] == app
                    and _path(existing["path"]) == _path(facts.path)
                    and existing["state"] != "removed"
                ):
                    if (
                        not _same_identity(existing["identity"], identity)
                        and existing["id"] not in _RUNNING
                    ):
                        existing.update(
                            state="changed_outside",
                            message=(
                                "The folder was replaced; "
                                "its old permissions were left as they are."
                            ),
                        )
                        continue
                    if existing["mode"] == mode and existing["state"] == "ready":
                        return FolderAccess.model_validate(existing)
                    raise ValueError(
                        "Remove the existing access record before changing this folder's mode."
                    )
            ident = uuid.uuid4().hex
            row: dict[str, Any] = dict(
                id=ident,
                app=app,
                path=facts.path,
                mode=mode,
                account=self.permissions.account,
                state="applying",
                preexisting=False,
                notInheriting=0,
                restartNeeded=False,
                message="Applying folder access.",
                addedAt=datetime.now(UTC).isoformat(),
                addedBy=operator,
                identity=identity,
                delta=None,
                process=self.process(),
            )
            rows[ident] = row
            self._write(rows)

            _RUNNING.add(ident)
            snapshot = FolderAccess.model_validate(row)

        def persist() -> None:
            with _LOCK:
                current = self._read()
                current[ident] = row
                self._write(current)

        def save(delta: dict) -> None:
            if "identity" in delta and not _same_identity(delta["identity"], identity):
                raise ValueError("The folder was replaced before its permission could be changed.")
            row["delta"] = delta
            persist()

        def apply() -> None:
            try:
                delta, count, restart = self.permissions.grant(facts.path, mode, save)
                ability = "read and write" if mode == "write" else "read"
                row.update(
                    delta=delta,
                    process=delta.get("process", row["process"]),
                    state="ready",
                    preexisting=not delta["added"],
                    notInheriting=count,
                    restartNeeded=restart,
                    message=(
                        f"Workbench restarts to write in {path}. Answers running in it stop."
                        if restart
                        else f"Workbench can {ability} {path}. "
                        f"{count} items inside keep their own permissions and stay closed."
                    ),
                )
            except Exception as exc:
                row.update(state="failed", message=str(exc) or type(exc).__name__)
            finally:
                try:
                    persist()
                finally:
                    with _LOCK:
                        _RUNNING.discard(ident)

        if schedule is not None:
            schedule(apply)
            return snapshot
        apply()
        return FolderAccess.model_validate(row)

    def remove(self, app: str, ident: str) -> FolderAccess:
        with _LOCK:
            rows = self._read()
            row = rows.get(ident)
            if row is None or row["app"] != app:
                raise KeyError(ident)
            if row["state"] == "removed":
                return FolderAccess.model_validate(row)
            if ident in _RUNNING:
                raise ValueError(
                    "This folder's permissions are still changing. Wait for it to finish."
                )
            row.update(state="removing", message="Removing only Eugene's additions.")
            _RUNNING.add(ident)
            self._write(rows)
        try:
            facts, identity = self._inspect(row["path"])
            if facts.link or (facts.is_dir and not _same_identity(identity, row["identity"])):
                raise ValueError("The folder was replaced.")
            if row["delta"] is not None and not self.permissions.revoke(row["path"], row["delta"]):
                raise ValueError("The recorded permissions were changed outside Eugene.")
            row.update(
                state="removed",
                restartNeeded=False,
                message="Eugene's additions were removed. Preexisting permissions stay.",
            )
        except (RemovalIncomplete, InspectionFailed) as exc:
            row.update(state="failed", message=str(exc))
        except ValueError as exc:
            row.update(
                state="changed_outside", message=f"{exc} Its permissions were left as they are."
            )
        except Exception as exc:
            row.update(state="failed", message=f"Access could not be removed: {exc}.")
        finally:
            with _LOCK:
                try:
                    current = self._read()
                    reapplied = set((row["delta"] or {}).get("reapplied", []))
                    for other in current.values():
                        delta = other.get("delta")
                        if other["app"] == app and delta and delta.get("id") in reapplied:
                            other["preexisting"] = False
                            delta["added"] = True
                    current[ident] = row
                    self._write(current)
                finally:
                    _RUNNING.discard(ident)
        return FolderAccess.model_validate(row)

    def lookup(self, app: str, path: str) -> list[FolderAccess]:
        facts, identity = self._inspect(path)
        candidate = _path(facts.path)
        with _LOCK:
            matching = [
                FolderAccess.model_validate(row)
                for row in self._read().values()
                if row["app"] == app
                and row["state"] != "removed"
                and (
                    candidate.is_relative_to(_path(row["path"]))
                    or (identity and _same_identity(identity, row["identity"]))
                )
            ]
        if not matching:
            raise KeyError("No recorded folder access covers this folder for this app.")
        return matching

    def release(self, app: str, ident: str) -> list[FolderAccess]:
        return [self.remove(app, ident)]


def windows_roots() -> tuple[tuple[str, ...], tuple[str, ...]]:
    if sys.platform != "win32":
        raise OSError("Windows profile detection needs Windows.")
    import winreg

    homes = []
    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList"
    ) as key:
        for index in range(winreg.QueryInfoKey(key)[0]):
            with winreg.OpenKey(key, winreg.EnumKey(key, index)) as profile:
                homes.append(
                    os.path.expandvars(winreg.QueryValueEx(profile, "ProfileImagePath")[0])
                )
    system = tuple(
        os.environ[name]
        for name in ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData")
        if name in os.environ
    )
    return tuple(homes), system


class WindowsPermissions:
    """Hold every parent against rename; read and set the DACL on the same handle."""

    READ = 0x1200A9
    WRITE = READ | 0x116  # FILE_GENERIC_WRITE without DELETE or DELETE_CHILD

    def __init__(
        self, account: str, *, eugene: tuple[str, ...], homes: tuple[str, ...] | None = None
    ) -> None:
        self.account, self.eugene = account, eugene
        detected_homes, self.system = windows_roots()
        self.homes = detected_homes if homes is None else homes

    @contextlib.contextmanager
    def _open(
        self, path: str, *, change: bool = False, _single: bool = False, revoke: bool = False
    ) -> Iterator[tuple[int, Facts, list[int]]]:
        if sys.platform != "win32":
            raise OSError("Windows folder permissions need Windows.")
        import msvcrt
        from ctypes import wintypes as w

        value = PureWindowsPath(path)
        if path.startswith(("\\\\?\\", "\\\\.\\")):
            raise Refused("not_a_folder", "Device paths are not supported. " + path)
        if path.startswith(("\\\\", "//")):
            raise Refused("network", path)
        if not value.is_absolute() or not re.fullmatch("[A-Za-z]:", value.drive):
            raise Refused("not_a_folder", path)
        if any(
            p in (".", "..") or p.endswith((".", " ")) or ":" in p or any(ord(c) < 32 for c in p)
            for p in value.parts[1:]
        ):
            raise Refused("not_a_folder", path)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [
            w.LPCWSTR,
            w.DWORD,
            w.DWORD,
            w.LPVOID,
            w.DWORD,
            w.DWORD,
            w.HANDLE,
        ]
        kernel.CreateFileW.restype = w.HANDLE
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.GetDriveTypeW.argtypes = [w.LPCWSTR]
        kernel.GetDriveTypeW.restype = w.UINT
        kernel.GetFinalPathNameByHandleW.argtypes = [w.HANDLE, w.LPWSTR, w.DWORD, w.DWORD]
        kernel.GetFinalPathNameByHandleW.restype = w.DWORD
        if kernel.GetDriveTypeW(value.anchor) == 4:
            raise Refused("network", path)
        held = []
        try:
            current = PureWindowsPath(value.anchor)
            components = (str(value),) if _single else (value.anchor, *value.parts[1:])
            for index, part in enumerate(components):
                if index or _single:
                    current /= part
                access = 0x60000 if change and index == len(components) - 1 else 0x20080
                # Files need no data access and share read, write and delete.
                # Retained directory handles keep the traversal parents in place.
                directory = stat.S_ISDIR(os.lstat(current).st_mode)
                share = 3 if directory else 7
                handle = kernel.CreateFileW(str(current), access, share, None, 3, 0x02200000, None)
                if handle == w.HANDLE(-1).value:
                    error = ctypes.WinError(ctypes.get_last_error())
                    error.filename = str(current)
                    raise error
                fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
                held.append(fd)
                info = os.fstat(fd)
                if info.st_file_attributes & 0x400 and not revoke:
                    raise Refused("link", str(current))
                if not stat.S_ISDIR(info.st_mode) and not _single:
                    raise Refused("not_a_folder", str(current))
                if not stat.S_ISDIR(info.st_mode) and info.st_nlink != 1 and not revoke:
                    raise Refused("link", f"{current} has more than one hard link.")
            buffer = ctypes.create_unicode_buffer(32768)
            length = kernel.GetFinalPathNameByHandleW(
                msvcrt.get_osfhandle(held[-1]), buffer, len(buffer), 0
            )
            if not length or length >= len(buffer):
                raise ctypes.WinError(ctypes.get_last_error())
            final = buffer.value.removeprefix("\\\\?\\")
            facts = Facts(
                final,
                is_dir=stat.S_ISDIR(info.st_mode),
                eugene=self.eugene,
                homes=self.homes,
                system=self.system,
                link=bool(info.st_file_attributes & 0x400),
            )
            yield (
                msvcrt.get_osfhandle(held[-1]),
                facts,
                [info.st_dev, info.st_ino, info.st_birthtime_ns],
            )
        finally:
            for fd in reversed(held):
                os.close(fd)

    def inspect(self, path: str) -> tuple[Facts, list[int]]:
        # Refuse reserved trees even when they are absent on this machine.
        lexical = Facts(path, eugene=self.eugene, homes=self.homes, system=self.system)
        if refusal(lexical):
            return lexical, []
        try:
            with self._open(path) as (_, facts, identity):
                return facts, identity
        except Refused as exc:
            return Facts(
                path,
                link=exc.cause == "link",
                network=exc.cause == "network",
                is_dir=exc.cause not in ("not_a_folder",),
            ), []
        except (FileNotFoundError, NotADirectoryError):
            return Facts(path, is_dir=False), []

    @staticmethod
    def _acl(ws: Any, handle: int) -> Any:
        return ws.GetSecurityInfo(
            handle, ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION
        ).GetSecurityDescriptorDacl()

    def _tree(
        self,
        ws: Any,
        path: str,
        handle: int,
        identity: list[int],
        *,
        revoke: bool = False,
        selected: set[tuple[str, ...]] | None = None,
    ) -> Iterator[tuple[tuple[str, ...], int, list[int], Any, bool, bool]]:
        """Yield one live file handle; only traversal directories remain open."""

        def walk(
            current: Path,
            names: tuple[str, ...],
            held: int,
            ident: list[int],
            directory: bool,
            link: bool = False,
        ) -> Iterator[tuple[tuple[str, ...], int, list[int], Any, bool, bool]]:
            descriptor = ws.GetSecurityInfo(held, ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION)
            protected = bool(names and descriptor.GetSecurityDescriptorControl()[0] & 0x1000)
            yield names, held, ident, descriptor.GetSecurityDescriptorDacl(), directory, protected
            if directory and not protected and not link:
                for entry in current.iterdir():
                    if selected is not None and (*names, entry.name) not in selected:
                        continue
                    try:
                        with self._open(
                            str(entry), change=not revoke, _single=True, revoke=revoke
                        ) as (
                            child,
                            facts,
                            child_id,
                        ):
                            yield from walk(
                                entry,
                                (*names, entry.name),
                                child,
                                child_id,
                                facts.is_dir,
                                facts.link,
                            )
                    except FileNotFoundError:
                        if not revoke:
                            raise

        yield from walk(Path(path), (), handle, identity, True)

    @staticmethod
    def _set_acl(ws: Any, handle: int, acl: Any) -> None:
        # The kernel API changes this object only. SetSecurityInfo propagates
        # to unchecked descendants when opened with READ_CONTROL | WRITE_DAC.
        descriptor = ws.GetSecurityInfo(handle, ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION)
        control = descriptor.GetSecurityDescriptorControl()[0]
        descriptor.SetSecurityDescriptorDacl(1, acl, 0)
        # The kernel consumes AUTO_INHERIT_REQ and sets AUTO_INHERITED. Without
        # the request it clears that state and turns inherited ACEs explicit.
        descriptor.SetSecurityDescriptorControl(0x100, 0x100 if control & 0x400 else 0)
        protection = (
            ws.PROTECTED_DACL_SECURITY_INFORMATION
            if control & 0x1000
            else ws.UNPROTECTED_DACL_SECURITY_INFORMATION
        )
        ws.SetKernelObjectSecurity(handle, ws.DACL_SECURITY_INFORMATION | protection, descriptor)

    @staticmethod
    def _insert(ws: Any, acl: Any, sid: Any, mask: int, flags: int) -> Any:
        entries = [acl.GetAce(i) for i in range(acl.GetAceCount())]
        index = (
            len(entries)
            if flags & 16
            else next((i for i, e in enumerate(entries) if e[0][1] & 16), len(entries))
        )
        ordered = ws.ACL()
        for entry in (*entries[:index], ((0, flags), mask, sid), *entries[index:]):
            kind, entry_flags = entry[0]
            if kind in (0, 1):
                method = (
                    ordered.AddAccessAllowedAceEx if kind == 0 else ordered.AddAccessDeniedAceEx
                )
                method(ws.ACL_REVISION, entry_flags, entry[1], entry[-1])
            elif kind in (5, 6):
                method = (
                    ordered.AddAccessAllowedObjectAce
                    if kind == 5
                    else ordered.AddAccessDeniedObjectAce
                )
                method(ws.ACL_REVISION_DS, entry_flags, entry[1], entry[2], entry[3], entry[-1])
            else:
                raise ValueError(
                    f"The folder has an unsupported ACE type ({kind}); "
                    "its permissions were left as they are."
                )
        return ordered

    @staticmethod
    def _matches(ws: Any, acl: Any, delta: dict, flags: int) -> list[int]:
        if acl is None:
            return []
        return [
            i
            for i in range(acl.GetAceCount())
            if (
                acl.GetAce(i)[0] == (0, flags)
                and acl.GetAce(i)[1] == delta["mask"]
                and ws.ConvertSidToStringSid(acl.GetAce(i)[-1]) == delta["sid"]
            )
        ]

    def grant(self, path: str, mode: str, save: Callable[[dict], None]) -> tuple[dict, int, bool]:
        import win32security as ws

        sid = (
            ws.ConvertStringSidToSid(self.account)
            if self.account.startswith("S-1-")
            else ws.LookupAccountName(None, self.account)[0]
        )
        mask = self.WRITE if mode == "write" else self.READ
        sid_text = ws.ConvertSidToStringSid(sid)
        with self._open(path, change=True) as (handle, facts, identity):
            if cause := refusal(facts):
                raise Refused(cause, path)
            plans: dict[tuple[str, ...], dict[str, Any]] = {}
            count = 0
            for names, _, ident, acl, directory, protected in self._tree(
                ws, path, handle, identity
            ):
                if protected:
                    count += 1
                    continue
                if acl is None:
                    continue
                matching = [
                    acl.GetAce(i)
                    for i in range(acl.GetAceCount())
                    if ws.ConvertSidToStringSid(acl.GetAce(i)[-1]) == sid_text
                    and not acl.GetAce(i)[0][1] & 8
                ]
                if any(e[0][0] != 0 for e in matching):
                    raise ValueError(
                        "An existing deny or special ACE names this account; "
                        "its permissions were left as they are."
                    )
                rights = 0
                for entry in matching:
                    if not directory or entry[0][1] & 3 == 3:
                        rights |= entry[1]
                if rights & mask == mask:
                    continue
                flags = 3 if not names else (19 if directory else 16)
                plans[names] = dict(names=names, identity=ident, flags=flags, applied=False)
            delta = dict(
                added=bool(plans),
                sid=sid_text,
                mask=mask,
                identity=identity,
                items=list(plans.values()),
            )
            save(delta)  # One complete write-ahead plan, independent of tree size.
            selected = {names[:i] for names in plans for i in range(len(names) + 1)}
            for names, held, ident, acl, _, _ in self._tree(
                ws, path, handle, identity, selected=selected
            ):
                item = plans.get(names)
                if item is None:
                    continue
                if ident != item["identity"]:
                    raise ValueError(f"The folder item was replaced: {Path(path).joinpath(*names)}")
                self._set_acl(ws, held, self._insert(ws, acl, sid, mask, item["flags"]))
                item["applied"] = True
            save(delta)
            return delta, count, False

    def revoke(self, path: str, delta: dict) -> bool:
        import win32security as ws

        if not delta["added"]:
            return True
        with contextlib.ExitStack() as opened:
            try:
                handle, _, identity = opened.enter_context(self._open(path, change=True))
            except FileNotFoundError:
                return True
            if identity != delta["identity"]:
                return False
            recorded = {tuple(item["names"]): item for item in delta["items"]}
            owned = set()
            for names, _held, ident, acl, directory, protected in self._tree(
                ws, path, handle, identity, revoke=True
            ):
                item = recorded.get(names)
                if item is not None:
                    if item.get("removed"):
                        owned.add(names)
                        continue
                    if ident != item["identity"]:
                        # A replacement never inherits the recorded explicit ACE.
                        continue
                    flags = item["flags"]
                elif names and names[:-1] in owned and not protected:
                    flags = 19 if directory else 16
                else:
                    continue
                matches = self._matches(ws, acl, delta, flags)
                if len(matches) > 1:
                    return False
                if matches:
                    # Files/links were inspected with READ_CONTROL only.
                    # Request WRITE_DAC only for an exact owned ACE.
                    try:
                        with self._open(
                            str(Path(path).joinpath(*names)), change=True, _single=True, revoke=True
                        ) as (change, _, current_id):
                            if current_id != ident:
                                continue
                            acl = self._acl(ws, change)
                            matches = self._matches(ws, acl, delta, flags)
                            if len(matches) > 1:
                                return False
                            if matches:
                                acl.DeleteAce(matches[0])
                                self._set_acl(ws, change, acl)
                    except FileNotFoundError:
                        continue
                elif item is not None and item.get("applied"):
                    # It was already removed externally; no owned rights remain.
                    pass
                owned.add(names)
                if item is not None:
                    item["removed"] = True
            # Detect files created in the narrow interval before a parent's
            # ACE was removed. A retry uses the same recorded plan.
            for names, _, _, acl, directory, _ in self._tree(
                ws, path, handle, identity, revoke=True
            ):
                if (
                    names
                    and names[:-1] in owned
                    and self._matches(ws, acl, delta, 19 if directory else 16)
                ):
                    raise OSError(
                        "New files inherited access during removal. "
                        "Some additions remain; try Remove access again."
                    )
            return True


class LinuxPermissions:
    """Ask the installed root broker; its root-owned ledger is authoritative."""

    def __init__(self, app: str, apps: Path, *, eugene: tuple[str, ...]) -> None:
        from .app_accounts import dynamic_user

        self.app, self.ctl, self.eugene = app, apps / "ctl", eugene
        self.account = dynamic_user(app) + "-files"

    def inspect(self, path: str) -> tuple[Facts, list[int]]:
        if sys.platform != "linux":
            raise OSError("Linux folder permissions need Linux.")
        # Eugene's account need not be able to traverse a shared folder.
        # The root broker validates the path and opens it without following links,
        # just as it does at the mutation. Never probe the person's folder here.
        ident = uuid.uuid4().hex
        self.ctl.mkdir(parents=True, exist_ok=True)
        request = self.ctl / f"{ident}.path"
        write_private(request, path)
        try:
            result = self._ask("folder-inspect", ident)
        finally:
            request.unlink(missing_ok=True)
        facts = result["facts"]
        for key in ("eugene", "homes", "system"):
            facts[key] = tuple(facts.get(key, ()))
        facts["eugene"] += self.eugene
        return Facts(**facts), result["identity"]

    def _ask(self, verb: str, ident: str, mode: str = "read") -> dict:
        self.ctl.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex
        request = self.ctl / f"{request_id}.req"
        result = self.ctl / f"{request_id}.res"
        write_private(request, f"{verb} {self.app} {mode} {ident}\n")
        # A root helper may still be applying a large plan after 900 seconds.
        # Only its result can declare success or failure.
        while True:
            if result.is_file():
                try:
                    answer: dict = json.loads(result.read_text("utf-8"))
                finally:
                    result.unlink(missing_ok=True)
                if answer.get("code") != 0:
                    raise OSError(
                        str(
                            answer.get("output")
                            or "The folder permission helper refused the request."
                        )
                    )
                result_body: dict = json.loads(answer["output"])
                return result_body
            time.sleep(0.2)

    def grant(self, path: str, mode: str, save: Callable[[dict], None]) -> tuple[dict, int, bool]:
        facts, identity = self.inspect(path)
        if cause := refusal(facts):
            raise Refused(cause, path)
        ident = uuid.uuid4().hex
        delta = dict(added=True, id=ident, identity=identity)
        save(delta)
        self.ctl.mkdir(parents=True, exist_ok=True)
        write_private(self.ctl / f"{ident}.path", path)
        write_private(self.ctl / f"{ident}.identity", json.dumps(identity))
        try:
            result = self._ask("folder-grant", ident, mode)
        finally:
            (self.ctl / f"{ident}.path").unlink(missing_ok=True)
            (self.ctl / f"{ident}.identity").unlink(missing_ok=True)
        delta["added"] = result["added"]
        if "process" in result:
            delta["process"] = result["process"]
        return delta, 0, bool(result.get("restartNeeded", mode == "write"))

    def revoke(self, path: str, delta: dict) -> bool:
        result = self._ask("folder-revoke", delta["id"])
        delta["reapplied"] = result.get("reapplied", [])
        if not result["removed"] and result.get("message"):
            raise RemovalIncomplete(
                f"Some items were not changed: {str(result['message']).rstrip('.')}. "
                "Try Remove access again."
            )
        return bool(result["removed"])
