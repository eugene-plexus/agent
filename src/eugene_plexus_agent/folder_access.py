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
import struct
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
    """One pure decision, including ancestors that would expose a refused tree."""
    path = _path(facts.path)
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
    if _overlaps(path, facts.eugene):
        return "eugene"
    if isinstance(path, PureWindowsPath):
        drive = path.drive
        system = tuple(
            drive + "\\" + name
            for name in (
                "Windows",
                "Program Files",
                "Program Files (x86)",
                "ProgramData",
                "Recovery",
                "System Volume Information",
                "$Recycle.Bin",
            )
        )
        homes = (drive + "\\Users",)
    else:
        system, homes = LINUX_SYSTEM, ("/home",)
        if path.parts[1].startswith("lib"):
            return "system"
    if _overlaps(path, facts.homes + homes):
        return "home"
    if _overlaps(path, facts.system + system):
        return "system"
    return None


class Refused(ValueError):
    def __init__(self, cause: str, detail: str = "") -> None:
        self.cause = cause
        super().__init__(f"{MESSAGES[cause]} {detail}".strip())


class Permissions(Protocol):
    account: str

    def inspect(self, path: str) -> tuple[Facts, list[int]]: ...
    def grant(
        self, path: str, mode: str, save: Callable[[dict], None]
    ) -> tuple[dict, int, bool]: ...
    def revoke(self, path: str, delta: dict) -> bool: ...


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

    def listing(self, app: str) -> list[FolderAccess]:
        with _LOCK:
            rows = self._read()
            result = []
            for row in rows.values():
                if row["app"] != app:
                    continue
                if row["state"] == "ready":
                    try:
                        facts, identity = self.permissions.inspect(row["path"])
                        if facts.link or identity != row["identity"]:
                            row.update(
                                state="changed_outside",
                                message=(
                                    "The folder was replaced; "
                                    "its permissions were left as they are."
                                ),
                            )
                    except OSError as exc:
                        row.update(
                            state="changed_outside",
                            message=(
                                f"The folder cannot be checked: {exc}. "
                                "Its permissions were left as they are."
                            ),
                        )
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
            facts, identity = self.permissions.inspect(path)
            if cause := refusal(facts):
                raise Refused(cause, path)
            rows = self._read()
            for existing in rows.values():
                if (
                    existing["app"] == app
                    and _path(existing["path"]) == _path(facts.path)
                    and existing["state"] != "removed"
                ):
                    if existing["identity"] != identity and existing["id"] not in _RUNNING:
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
            if "identity" in delta and delta["identity"] != identity:
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
            facts, identity = self.permissions.inspect(row["path"])
            if facts.link or identity != row["identity"]:
                raise ValueError("The folder was replaced.")
            if row["delta"] is not None and not self.permissions.revoke(row["path"], row["delta"]):
                raise ValueError("The recorded permissions were changed outside Eugene.")
            row.update(
                state="removed",
                restartNeeded=False,
                message="Eugene's additions were removed. Preexisting permissions stay.",
            )
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
                    current[ident] = row
                    self._write(current)
                finally:
                    _RUNNING.discard(ident)
        return FolderAccess.model_validate(row)

    def release(self, app: str, path: str) -> list[FolderAccess]:
        with _LOCK:
            matching = [
                ident
                for ident, row in self._read().items()
                if row["app"] == app
                and _path(row["path"]) == _path(path)
                and row["state"] != "removed"
            ]
        return [self.remove(app, ident) for ident in matching]


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
        self, path: str, *, change: bool = False, _single: bool = False
    ) -> Iterator[tuple[int, Facts, list[int]]]:
        if sys.platform != "win32":
            raise OSError("Windows folder permissions need Windows.")
        import msvcrt
        from ctypes import wintypes as w

        value = PureWindowsPath(path)
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
                # MAXIMUM_ALLOWED deliberately suppresses SetSecurityInfo's
                # automatic propagation. Every child is checked and changed
                # through its own held handle instead (including hardlinks).
                access = 0x02000000 if change and index == len(components) - 1 else 0x20080
                handle = kernel.CreateFileW(str(current), access, 3, None, 3, 0x02200000, None)
                if handle == w.HANDLE(-1).value:
                    raise ctypes.WinError(ctypes.get_last_error())
                fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
                held.append(fd)
                info = os.fstat(fd)
                if info.st_file_attributes & 0x400:
                    raise Refused("link", str(current))
                if not stat.S_ISDIR(info.st_mode) and not _single:
                    raise Refused("not_a_folder", str(current))
                if not stat.S_ISDIR(info.st_mode) and info.st_nlink != 1:
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
        stack: contextlib.ExitStack,
        reuse: dict[tuple[str, ...], tuple[int, list[int], Any, bool]] | None = None,
    ) -> tuple[list[tuple[tuple[str, ...], int, list[int], Any, bool]], int]:
        nodes = []
        protected = 0

        def walk(
            current: Path, names: tuple[str, ...], held: int, ident: list[int], directory: bool
        ) -> None:
            nonlocal protected
            descriptor = ws.GetSecurityInfo(held, ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION)
            nodes.append((names, held, ident, descriptor.GetSecurityDescriptorDacl(), directory))
            if names and descriptor.GetSecurityDescriptorControl()[0] & 0x1000:
                protected += 1
                return
            if directory:
                for entry in current.iterdir():
                    child_names = (*names, entry.name)
                    previous = reuse.get(child_names) if reuse is not None else None
                    if previous is not None:
                        child, child_id, _, is_dir = previous
                    else:
                        child, facts, child_id = stack.enter_context(
                            self._open(str(entry), change=True, _single=True)
                        )
                        is_dir = facts.is_dir
                    walk(entry, child_names, child, child_id, is_dir)

        walk(Path(path), (), handle, identity, True)
        return nodes, protected

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
        with (
            self._open(path, change=True) as (handle, facts, identity),
            contextlib.ExitStack() as stack,
        ):
            if cause := refusal(facts):
                raise Refused(cause, path)
            nodes, count = self._tree(ws, path, handle, identity, stack)
            plans = []
            for names, held, ident, acl, directory in nodes:
                descriptor = ws.GetSecurityInfo(
                    held, ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION
                )
                if names and descriptor.GetSecurityDescriptorControl()[0] & 0x1000:
                    continue
                if acl is None:  # NULL DACL already grants everything
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
                plans.append((names, held, ident, flags, self._insert(ws, acl, sid, mask, flags)))
            delta = dict(added=bool(plans), sid=sid_text, mask=mask, identity=identity, items=[])
            save(delta)
            for names, held, ident, flags, acl in plans:
                item = dict(names=names, identity=ident, flags=flags, applied=False)
                delta["items"].append(item)
                save(delta)
                ws.SetSecurityInfo(
                    held, ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION, None, None, acl, None
                )
                item["applied"] = True
                save(delta)
            return delta, count, False

    def revoke(self, path: str, delta: dict) -> bool:
        import win32security as ws

        if not delta["added"]:
            return True
        with (
            self._open(path, change=True) as (handle, _, identity),
            contextlib.ExitStack() as stack,
        ):
            if identity != delta["identity"]:
                return False
            nodes, _ = self._tree(ws, path, handle, identity, stack)
            current = {
                names: (held, ident, acl, directory) for names, held, ident, acl, directory in nodes
            }
            changes = []
            owned = set()
            recorded = {tuple(item["names"]) for item in delta["items"]}
            for item in delta["items"]:
                if item.get("removed"):
                    continue
                names = tuple(item["names"])
                found = current.get(names)
                if found is None or found[1] != item["identity"]:
                    return False
                held, _, acl, _ = found
                matches = self._matches(ws, acl, delta, item["flags"])
                if not matches and not item["applied"]:
                    continue
                if len(matches) != 1:
                    return False
                owned.add(names)
                acl.DeleteAce(matches[0])
                changes.append((held, acl, item))
            # A new object's inherited bit and its held parent establish its
            # provenance on Windows. Explicit by-hand entries are never removed.
            for names, held, _, acl, directory in nodes:
                if not names or names in recorded or names[:-1] not in owned:
                    continue
                matches = self._matches(ws, acl, delta, 19 if directory else 16)
                if len(matches) > 1:
                    return False
                if matches:
                    owned.add(names)
                    acl.DeleteAce(matches[0])
                    item = dict(
                        names=names,
                        identity=current[names][1],
                        flags=19 if directory else 16,
                        applied=True,
                    )
                    delta["items"].append(item)
                    changes.append((held, acl, item))
            for held, acl, item in changes:
                ws.SetSecurityInfo(
                    held, ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION, None, None, acl, None
                )
                item["removed"] = True
            # A file can inherit the old ACE after the inspection and before
            # its parent's ACE is removed. Record it for a safe GUI retry;
            # never report that all additions are gone while it remains.
            known = set(current)
            try:
                final_nodes, _ = self._tree(ws, path, handle, identity, stack, reuse=current)
            except Refused as exc:
                raise OSError(f"Folder contents changed during removal: {exc}") from exc
            leftovers = False
            for names, _, ident, acl, directory in final_nodes:
                if not names or names in known or names[:-1] not in owned:
                    continue
                flags = 19 if directory else 16
                if self._matches(ws, acl, delta, flags):
                    delta["items"].append(
                        dict(names=names, identity=ident, flags=flags, applied=True)
                    )
                    leftovers = True
            if leftovers:
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
        import pwd

        if path.startswith("//"):
            return Facts(path, network=True), []

        value = Path(path)
        homes = tuple(
            p.pw_dir
            for p in pwd.getpwall()
            if p.pw_dir.startswith("/") and p.pw_dir not in ("/", "/nonexistent")
        )
        if any(p.is_symlink() for p in (value, *value.parents)):
            return Facts(path, link=True), []
        try:
            info = value.stat()
        except (FileNotFoundError, NotADirectoryError):
            return Facts(path, is_dir=False), []
        if not stat.S_ISDIR(info.st_mode):
            return Facts(path, is_dir=False), []
        # Metadata must be readable before the requested access is granted.
        fd = os.open(path, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            data = ctypes.create_string_buffer(256)
            libc = ctypes.CDLL(None, use_errno=True)
            if (
                libc.statx(fd, b"", 0x1000, 0x1800, data)
                or struct.unpack_from("=I", data)[0] & 0x1800 != 0x1800
            ):
                raise OSError(
                    "The filesystem does not expose folder creation time and mount identity."
                )
            seconds, nanos = struct.unpack_from("=qI", data, 80)
            identity = [
                info.st_dev,
                info.st_ino,
                seconds,
                nanos,
                struct.unpack_from("=Q", data, 144)[0],
            ]
        finally:
            os.close(fd)
        return Facts(
            str(value),
            is_dir=value.is_dir(),
            mount_root=os.path.ismount(value),
            eugene=self.eugene,
            homes=homes,
        ), identity

    def _ask(self, verb: str, ident: str, mode: str = "read") -> dict:
        self.ctl.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex
        request = self.ctl / f"{request_id}.req"
        result = self.ctl / f"{request_id}.res"
        write_private(request, f"{verb} {self.app} {mode} {ident}\n")
        deadline = time.perf_counter() + 900
        while time.perf_counter() < deadline:
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
        raise TimeoutError(
            "The installed folder permission helper did not answer within 900 seconds."
        )

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
        if not result["removed"] and result.get("message"):
            raise ValueError(result["message"])
        return bool(result["removed"])
