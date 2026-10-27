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
    "unsafe_parent": "can be changed by other accounts.",
    "unavailable": UNAVAILABLE,
}
# What an unknown entry is, in words (R23-UNKNOWN).
NULL_DACL = "it has no permission list (a null DACL), so every account can open it"
OWNED = "{account} owns it, so it can change its permissions"
UNREADABLE = "Eugene cannot read its permissions"
HARDLINK = "it has another name outside the folder, so changing it here changes it there too"
SPECIAL = "it is a pipe, socket or device, which Eugene never changes on its own"
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
    unsafe_parent: str = ""
    #: Names a file has. Above one, a change here reaches every other name.
    links: int = 1


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
    """Own prefix/data first; otherwise system, home, then other install ancestors.

    An unsafe parent comes last. On Linux, Eugene's prefix and a person's home
    belong to their own accounts, so every folder inside them also has such a
    parent; the person is told the rule that refuses the folder itself.
    """
    cause = _policy_refusal(facts)
    if cause is None and facts.unsafe_parent:
        return "unsafe_parent"
    return cause


def _policy_refusal(facts: Facts) -> str | None:
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
        super().__init__(
            (
                f"{detail} {MESSAGES[cause]}"
                if cause == "unsafe_parent"
                else f"{MESSAGES[cause]} {detail}"
            ).strip()
        )


@contextlib.contextmanager
def _privileges(ws: Any, *names: str) -> Iterator[bool]:
    """Enable privileges for this thread alone; yield whether all were granted.

    The thread takes a copy of the process token (ImpersonateSelf), so no other
    thread of the agent ever runs with them, and gives it back afterwards.
    """
    import win32api

    ws.ImpersonateSelf(ws.SecurityImpersonation)
    try:
        token = ws.OpenThreadToken(
            win32api.GetCurrentThread(), ws.TOKEN_ADJUST_PRIVILEGES | ws.TOKEN_QUERY, True
        )
        try:
            ws.AdjustTokenPrivileges(
                token,
                False,
                [(ws.LookupPrivilegeValue(None, name), ws.SE_PRIVILEGE_ENABLED) for name in names],
            )
            granted = win32api.GetLastError() != 1300  # ERROR_NOT_ALL_ASSIGNED
            yield granted
        finally:
            token.Close()
    finally:
        ws.RevertToSelf()


class RemovalIncomplete(OSError):
    """Removal changed some permissions and named the items it could not change."""


class UnknownAccess(RemovalIncomplete):
    """Removal found access for the app's account that Eugene did not add.

    The owner decides each entry (R23-UNKNOWN). An entry is a dict: kind,
    path, entry (in words) and fingerprint, which the backend uses to check
    the entry again; the fingerprint stays in the agent's private record.
    """

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries
        paths = list(dict.fromkeys(str(e["path"]) for e in entries))
        shown = ", ".join(paths[:3]) + (f" and {len(paths) - 3} more" if len(paths) > 3 else "")
        if all(e["kind"] == "hardlink" for e in entries):
            what = "a file with another name outside the folder"
        elif all(e["kind"] in ("hardlink", "special") for e in entries):
            what = "files Eugene never changes on its own"
        else:
            what = f"{'an entry' if len(entries) == 1 else 'entries'} Eugene did not add"
        super().__init__(
            f"Workbench's account can still open {shown} through {what}. "
            "Choose Remove this entry too or Keep it."
        )


class NotWaiting(ValueError):
    """The entry is not waiting for the owner's decision."""


class StaleEntry(ValueError):
    """What the owner decided about has changed since Eugene asked.

    Nothing is applied on an old answer: the removal runs again and asks about
    what is there now."""


class SharedFile(StaleEntry):
    """The file has more names than removal may change it through.

    Read on the handle that would change it: a link made after the prompt, or
    after the owner allowed one name less, stops the change there."""

    def __init__(self, links: int) -> None:
        self.links = links
        super().__init__(
            "The file has another name outside the folder now, so it was left as it is."
        )


def _file_key(identity: list[int]) -> str:
    """A file by what Windows calls it: volume serial and file ID.

    A rename keeps both and a replacement has neither, so exemptions and
    inventories keyed by this follow the file, never its name or a link."""
    return f"{identity[0]}:{identity[1]}"


def _answer_key(fingerprint: dict[str, Any]) -> str:
    """What an owner's answer is about: the item and the entry, not its name or state."""
    return json.dumps(
        {k: v for k, v in fingerprint.items() if k not in ("names", "state")}, sort_keys=True
    )


def _kept(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The entries the owner kept, as the backend's own record reports them.

    A backend that holds the owner's answers (the Linux helper's ledger) is the
    truth: an answer it committed shows even when this record never heard of it.
    The report is a one-time message from the last removal, never saved.
    """
    delta = row.get("delta") or {}
    retained = delta.pop("retained", None)
    if retained is None:
        return [u for u in row.get("unknown") or [] if u["decision"] == "kept"]
    return [
        dict(
            id=_entry_id(found["fingerprint"]),
            path=str(found["path"]),
            kind=found["kind"],
            entry=found["entry"],
            decision="kept",
        )
        for found in retained
    ]


def _entry_id(fingerprint: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]


class InspectionFailed(ValueError):
    """A folder could not be checked; its permissions must stay unchanged."""


class Permissions(Protocol):
    account: str

    def inspect(self, path: str) -> tuple[Facts, list[int]]: ...
    def grant(
        self, path: str, mode: str, save: Callable[[dict], None]
    ) -> tuple[dict, int, bool]: ...
    def revoke(self, path: str, delta: dict) -> bool: ...
    def decide(self, path: str, delta: dict, fingerprint: dict, action: str) -> None: ...


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
                        if facts.unsafe_parent and not identity:
                            raise InspectionFailed(
                                f"{facts.unsafe_parent} can be changed by other accounts"
                            )
                        if not _same_identity(identity, row["identity"]):
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
                raise Refused(cause, facts.unsafe_parent if cause == "unsafe_parent" else path)
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
            if not getattr(self.permissions, "overlap", True):
                self._refuse_overlap(rows, app, facts.path)
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

    @staticmethod
    def _refuse_overlap(rows: dict[str, dict], app: str, path: str) -> None:
        """One grant per item where a backend cannot tell two grants' rights apart.

        On Windows a grant inside another finds the outer grant's inherited
        rights and adds nothing, so removing the outer grant would also take
        the inner one's access while its record still said ready.
        """
        new = _path(path)
        for existing in rows.values():
            if existing["app"] != app or existing["state"] == "removed":
                continue
            other = _path(existing["path"])
            if other == new:
                continue
            if new.is_relative_to(other):
                raise ValueError(
                    f"{path} is inside {existing['path']}, which Workbench can already open. "
                    f"Remove access to {existing['path']} first, then give access again."
                )
            if other.is_relative_to(new):
                raise ValueError(
                    f"{path} holds {existing['path']}, which Workbench can already open. "
                    f"Remove access to {existing['path']} first, then give access again."
                )

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
            # The Linux broker opens the root-owned ledger's path and validates
            # its saved identity. New-grant inspection must never gate revoke.
            if not isinstance(self.permissions, LinuxPermissions):
                facts, identity = self._inspect(row["path"])
                if facts.link or (facts.is_dir and not _same_identity(identity, row["identity"])):
                    raise ValueError("The folder was replaced.")
            if row["delta"] is not None and not self.permissions.revoke(row["path"], row["delta"]):
                raise ValueError("The recorded permissions were changed outside Eugene.")
            kept = _kept(row)
            paths = ", ".join(dict.fromkeys(u["path"] for u in kept))
            row.update(
                state="removed",
                restartNeeded=False,
                unknown=kept,
                message=(
                    "Eugene's additions were removed. "
                    + (
                        f"Workbench can still open {paths} through "
                        f"{'an entry' if len(kept) == 1 else 'entries'} Eugene did not add."
                        if kept
                        else "Preexisting permissions stay."
                    )
                ),
            )
        except UnknownAccess as exc:
            # R23-UNKNOWN: the owner decides each entry. Kept ones stay listed.
            keys = dict(row.get("unknownKeys") or {})
            listed = _kept(row)
            for found in exc.entries:
                entry = _entry_id(found["fingerprint"])
                keys[entry] = found["fingerprint"]
                listed.append(
                    dict(
                        id=entry,
                        path=str(found["path"]),
                        kind=found["kind"],
                        entry=found["entry"],
                        decision="pending",
                    )
                )
            row.update(state="incomplete", unknown=listed, unknownKeys=keys, message=str(exc))
        except (RemovalIncomplete, InspectionFailed) as exc:
            row.update(state="failed", message=str(exc))
        except ValueError as exc:
            row.update(
                state="changed_outside", message=f"{exc} Its permissions were left as they are."
            )
        except Exception as exc:
            row.update(state="failed", message=f"Access could not be removed: {exc}.")
        finally:
            (row["delta"] or {}).pop("retained", None)
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

    def decide(self, app: str, ident: str, entry: str, action: str) -> FolderAccess:
        """The owner's decision about one entry Eugene did not add, then the removal again."""
        with _LOCK:
            rows = self._read()
            row = rows.get(ident)
            if row is None or row["app"] != app:
                raise KeyError(ident)
            found = next((u for u in row.get("unknown") or [] if u["id"] == entry), None)
            if found is None:
                raise KeyError(entry)
            if row["state"] != "incomplete" or found["decision"] != "pending" or ident in _RUNNING:
                raise NotWaiting(
                    "This entry is not waiting for a decision. "
                    "Choose Remove access again to check the folder."
                )
            fingerprint = (row.get("unknownKeys") or {})[entry]
            _RUNNING.add(ident)
        try:
            # Checks the entry again, and removes only it (or records Keep).
            try:
                self.permissions.decide(row["path"], row["delta"], fingerprint, action)
            except StaleEntry:
                # What the owner saw is gone: apply nothing, drop the old question
                # and let the removal below ask about what is there now.
                action = "stale"
            if action == "keep":
                found["decision"] = "kept"
            else:
                row["unknown"] = [u for u in row["unknown"] if u["id"] != entry]
                row["unknownKeys"].pop(entry, None)
            with _LOCK:
                current = self._read()
                current[ident] = row
                self._write(current)
        finally:
            with _LOCK:
                _RUNNING.discard(ident)
        return self.remove(app, ident)

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
    #: A grant inside another cannot tell its own rights from the outer one's.
    overlap = False

    def __init__(
        self, account: str, *, eugene: tuple[str, ...], homes: tuple[str, ...] | None = None
    ) -> None:
        self.account, self.eugene = account, eugene
        detected_homes, self.system = windows_roots()
        self.homes = detected_homes if homes is None else homes

    @contextlib.contextmanager
    def _open(
        self,
        path: str,
        *,
        change: bool = False,
        _single: bool = False,
        revoke: bool = False,
        owner: bool = False,
        names: tuple[str, ...] = (),
        root_identity: list[int] | None = None,
        max_links: int | None = None,
    ) -> Iterator[tuple[int, Facts, list[int]]]:
        """Open `path`, or the item `names` below it, one component at a time.

        Every component is opened from the one before, which stays held: no
        component may be a link (only the last item below the grant, when
        `revoke` asks for links), and `root_identity` must be the grant's own
        folder. Never reopen an item by its full path: a folder swapped for a
        junction would be followed.
        """
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
            p in (".", "..")
            or p.endswith((".", " "))
            or ":" in p
            or "/" in p
            or "\\" in p
            or any(ord(c) < 32 for c in p)
            for p in (*value.parts[1:], *names)
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
            components = (str(value),) if _single else (value.anchor, *value.parts[1:], *names)
            for index, part in enumerate(components):
                last = index == len(components) - 1
                if index or _single:
                    current /= part
                access = 0x60000 if change and last else 0x20080
                if owner and last:
                    access |= 0x80000  # WRITE_OWNER
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
                if info.st_file_attributes & 0x400 and not (revoke and last and (_single or names)):
                    raise Refused("link", str(current))
                if not stat.S_ISDIR(info.st_mode) and not _single and not (names and last):
                    raise Refused("not_a_folder", str(current))
                if not stat.S_ISDIR(info.st_mode) and info.st_nlink != 1 and not revoke:
                    raise Refused("link", f"{current} has more than one hard link.")
                if (
                    root_identity is not None
                    and not _single
                    and index == len(value.parts) - 1
                    and [info.st_dev, info.st_ino, info.st_birthtime_ns] != root_identity
                ):
                    raise ValueError("The folder was replaced.")
                if (
                    last
                    and max_links is not None
                    and not stat.S_ISDIR(info.st_mode)
                    and info.st_nlink > max_links
                ):
                    raise SharedFile(info.st_nlink)
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
                links=info.st_nlink,
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
        descend: Callable[[tuple[str, ...], list[int]], bool] | None = None,
        unreadable: list[tuple[str, ...]] | None = None,
    ) -> Iterator[tuple[tuple[str, ...], int, list[int], Any, bool, bool, int]]:
        """Yield one live file handle; only traversal directories remain open.

        A protected folder is entered only when `descend` says so. An item that
        cannot be opened is listed in `unreadable` when that list is given.
        """

        def walk(
            current: Path,
            names: tuple[str, ...],
            held: int,
            ident: list[int],
            directory: bool,
            link: bool = False,
            links: int = 1,
        ) -> Iterator[tuple[tuple[str, ...], int, list[int], Any, bool, bool, int]]:
            descriptor = ws.GetSecurityInfo(held, ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION)
            protected = bool(names and descriptor.GetSecurityDescriptorControl()[0] & 0x1000)
            yield (
                names,
                held,
                ident,
                descriptor.GetSecurityDescriptorDacl(),
                directory,
                protected,
                links,
            )
            if (
                directory
                and not link
                and (not protected or (descend is not None and descend(names, ident)))
            ):
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
                                facts.links,
                            )
                    except FileNotFoundError:
                        if not revoke:
                            raise
                    except PermissionError:
                        if unreadable is None:
                            raise
                        unreadable.append((*names, entry.name))

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
    def _matches(ws: Any, acl: Any, delta: dict, flags: set[int]) -> list[int]:
        if acl is None:
            return []
        return [
            i
            for i in range(acl.GetAceCount())
            if (
                acl.GetAce(i)[0][0] == 0
                and acl.GetAce(i)[0][1] in flags
                and acl.GetAce(i)[1] == delta["mask"]
                and ws.ConvertSidToStringSid(acl.GetAce(i)[-1]) == delta["sid"]
            )
        ]

    @staticmethod
    def _explicit(ws: Any, acl: Any, sid_text: str) -> list[list[int]]:
        """[flags, mask] of each explicit allow ACE that names the account."""
        if acl is None:
            return []
        return [
            [ace[0][1], ace[1]]
            for ace in (acl.GetAce(i) for i in range(acl.GetAceCount()))
            if ace[0][0] in (0, 5)
            and not ace[0][1] & 0x10
            and ws.ConvertSidToStringSid(ace[-1]) == sid_text
        ]

    @staticmethod
    def _unrecorded(acl: Any, matches: list[int], recorded: list[list[int]]) -> list[int]:
        """`matches` less the entries the grant found already there (by hand)."""
        left = [list(ace) for ace in recorded]
        result = []
        for index in matches:
            entry = [acl.GetAce(index)[0][1], acl.GetAce(index)[1]]
            if entry in left:
                left.remove(entry)
            else:
                result.append(index)
        return result

    @staticmethod
    def _owner(ws: Any, handle: int) -> str:
        descriptor = ws.GetSecurityInfo(handle, ws.SE_FILE_OBJECT, ws.OWNER_SECURITY_INFORMATION)
        return str(ws.ConvertSidToStringSid(descriptor.GetSecurityDescriptorOwner()))

    @staticmethod
    def _state(ws: Any, handle: int) -> str:
        """The item's owner and DACL as one fingerprint.

        An owner's answer is about this state; if either changed since Eugene
        asked, the answer is not applied (StaleEntry) and Eugene asks again."""
        import hashlib

        descriptor = ws.GetSecurityInfo(handle, ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION)
        dacl_text = ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
            descriptor, ws.SDDL_REVISION_1, ws.DACL_SECURITY_INFORMATION
        )
        owner = WindowsPermissions._owner(ws, handle)
        return hashlib.sha256(f"{owner}|{dacl_text}".encode()).hexdigest()

    @staticmethod
    def _unchanged(ws: Any, handle: int, state: str | None) -> None:
        if state is not None and WindowsPermissions._state(ws, handle) != state:
            raise StaleEntry(
                "The item's permissions or owner changed since Eugene found it, so the "
                "answer was not applied. Eugene asks again about what is there now."
            )

    def _give(
        self,
        ws: Any,
        path: str,
        names: tuple[str, ...],
        ident: list[int],
        owner: str | None,
        *,
        root: list[int] | None = None,
        state: str | None = None,
        max_links: int = 1,
    ) -> None:
        """Make the granted folder's owner the owner of an item the app owns.

        An owner can always change an item's DACL, so the app could give itself
        its access back. Needs SeRestorePrivilege (LocalSystem has it).
        """
        if owner is None:
            raise PermissionError("the granted folder itself belongs to the app's account")
        with (
            _privileges(ws, "SeRestorePrivilege", "SeTakeOwnershipPrivilege"),
            self._open(
                path,
                change=True,
                owner=True,
                revoke=True,
                names=names,
                root_identity=root,
                max_links=max_links,
            ) as (held, _, current),
        ):
            if current != ident:
                raise ValueError("The folder item was replaced.")
            self._unchanged(ws, held, state)
            ws.SetSecurityInfo(
                held,
                ws.SE_FILE_OBJECT,
                ws.OWNER_SECURITY_INFORMATION,
                ws.ConvertStringSidToSid(owner),
                None,
                None,
                None,
            )

    def _inherit(
        self,
        ws: Any,
        path: str,
        names: tuple[str, ...],
        ident: list[int],
        *,
        root: list[int] | None = None,
        state: str | None = None,
        max_links: int = 1,
    ) -> Any:
        """Replace a null DACL with what the item inherits from its parent.

        Done through the item's own handle, reopened component by component from
        the grant's folder; the parents are held open, so the item cannot be
        swapped and its parent cannot move meanwhile.
        """
        with self._open(
            path, change=True, revoke=True, names=names, root_identity=root, max_links=max_links
        ) as (held, _, current):
            if current != ident:
                raise ValueError("The folder item was replaced.")
            self._unchanged(ws, held, state)
            ws.SetSecurityInfo(
                held,
                ws.SE_FILE_OBJECT,
                ws.DACL_SECURITY_INFORMATION | ws.UNPROTECTED_DACL_SECURITY_INFORMATION,
                None,
                None,
                ws.ACL(),
                None,
            )
            acl = self._acl(ws, held)
            if acl is None:
                raise PermissionError("its DACL is still null")
            return acl

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
            # What removal must leave: the account's own explicit entries from
            # before this grant, and the folders already closed to inheritance.
            # Every list is keyed by file identity (volume serial and file ID),
            # never by name: a rename keeps the exemption, a replacement has none.
            manual: list[list[Any]] = []
            protected_ids: list[str] = []
            # Items the account already owned, and null DACLs: both already
            # gave it everything, so removal leaves them as they are.
            owners: list[str] = []
            open_dacls: list[str] = []
            for names, held, ident, acl, directory, protected, _links in self._tree(
                ws, path, handle, identity
            ):
                key = _file_key(ident)
                if own := self._explicit(ws, acl, sid_text):
                    manual.append([key, own])
                if self._owner(ws, held) == sid_text:
                    owners.append(key)
                if acl is None:
                    open_dacls.append(key)
                if protected:
                    count += 1
                    protected_ids.append(key)
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
                manual=manual,
                protected=protected_ids,
                owners=owners,
                open=open_dacls,
            )
            save(delta)  # One complete write-ahead plan, independent of tree size.
            selected = {names[:i] for names in plans for i in range(len(names) + 1)}
            for names, held, ident, acl, _, _, _links in self._tree(
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
        import pywintypes
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
            # The person the granted folder belongs to receives what the app
            # owns, since an owner can always rewrite an item's DACL.
            person: str | None = self._owner(ws, handle)
            if person == delta["sid"]:
                person = None
            # Every exemption is keyed by file identity, never by name.
            owners_before = set(delta.get("owners", []))
            open_before = set(delta.get("open", []))
            recorded = {_file_key(item["identity"]): item for item in delta["items"]}
            manual = {key: [list(ace) for ace in aces] for key, aces in delta.get("manual", [])}
            # Records from before protected folders were listed keep the old
            # walk, which never entered a protected folder.
            closed = set(delta["protected"]) if "protected" in delta else None
            # Files with another name that the owner let removal change, each
            # at no more names than it had when the owner chose Remove.
            allowed: dict[str, int] = {key: int(links) for key, links in delta.get("allowed", [])}

            def limit(key: str) -> int:
                return max(1, allowed.get(key, 1))

            def descend(names: tuple[str, ...], ident: list[int]) -> bool:
                return closed is not None and _file_key(ident) not in closed

            owned: set[tuple[str, ...]] = set()
            # Eugene's own entries it could not remove: a failure, retried.
            kept: list[tuple[str, ...]] = []
            unreadable: list[tuple[str, ...]] = []
            # Access Eugene did not add: the owner decides (R23-UNKNOWN). An
            # entry the owner chose to keep no longer holds the removal.
            accepted = {_answer_key(fp) for fp in delta.get("accepted", [])}
            asked: dict[str, dict[str, Any]] = {}
            retained: dict[str, dict[str, Any]] = {}

            def ask(
                kind: str,
                names: tuple[str, ...],
                ident: list[int],
                words: str,
                held: int | None = None,
                **extra: int,
            ) -> bool:
                """List one entry for the owner; False when the owner already kept it.

                The entry carries the item's owner and DACL as they are now, so a
                later answer is applied only to the state it was asked about."""
                fingerprint = dict(kind=kind, names=list(names), identity=list(ident), **extra)
                if held is not None:
                    fingerprint["state"] = self._state(ws, held)
                key = _answer_key(fingerprint)
                found = dict(
                    kind=kind,
                    path=str(Path(path).joinpath(*names)),
                    entry=words,
                    fingerprint=fingerprint,
                )
                if key in accepted:
                    retained.setdefault(key, found)
                    return False
                asked.setdefault(key, found)
                return True

            for names, held, ident, acl, directory, protected, links in self._tree(
                ws, path, handle, identity, revoke=True, descend=descend, unreadable=unreadable
            ):
                key = _file_key(ident)
                if not directory and links > limit(key):
                    # One rule, on both systems: a file with another name is never
                    # changed here, since the change would reach that name too.
                    ask("hardlink", names, ident, HARDLINK, held, links=links)
                    continue
                null = acl is None and key not in open_before
                if (
                    null
                    and _answer_key(dict(kind="null_dacl", identity=list(ident))) not in accepted
                ):
                    try:
                        # A null DACL lets everyone do everything; it inherits again.
                        acl = self._inherit(
                            ws, path, names, ident, root=identity, max_links=limit(key)
                        )
                    except FileNotFoundError:
                        continue
                    except SharedFile as exc:
                        ask("hardlink", names, ident, HARDLINK, held, links=exc.links)
                        continue
                    except (OSError, ValueError, pywintypes.error):
                        ask("null_dacl", names, ident, NULL_DACL, held)
                if (
                    key not in owners_before
                    and self._owner(ws, held) == delta["sid"]
                    and _answer_key(dict(kind="owner", identity=list(ident))) not in accepted
                ):
                    try:
                        self._give(
                            ws, path, names, ident, person, root=identity, max_links=limit(key)
                        )
                    except FileNotFoundError:
                        continue
                    except SharedFile as exc:
                        ask("hardlink", names, ident, HARDLINK, held, links=exc.links)
                        continue
                    except (OSError, ValueError, pywintypes.error):
                        ask("owner", names, ident, OWNED.format(account=self.account), held)
                item = recorded.get(key)
                # Protected since the grant: its inherited copy of Eugene's ACE
                # became an explicit one with the same rights.
                changed = protected and descend(names, ident)
                if item is not None:
                    if item.get("removed"):
                        owned.add(names)
                        continue
                    if ident != item["identity"]:
                        # A replacement never inherits the recorded explicit ACE.
                        continue
                    flags = {item["flags"]} | ({item["flags"] & ~0x10} if changed else set())
                elif names and names[:-1] in owned and not protected:
                    flags = {19 if directory else 16}
                elif changed and names[:-1] in owned:
                    flags = {3 if directory else 0}
                else:
                    continue
                # An entry the grant found already there is left as it is.
                aces = manual.get(key, [])
                matches = self._unrecorded(acl, self._matches(ws, acl, delta, flags), aces)
                if len(matches) > 1:
                    return False
                if matches:
                    # Files/links were inspected with READ_CONTROL only. Request
                    # WRITE_DAC only for an exact owned ACE, and reopen the item
                    # below the grant, never by its full path.
                    try:
                        with self._open(
                            path,
                            change=True,
                            revoke=True,
                            names=names,
                            root_identity=identity,
                            max_links=limit(key),
                        ) as (change, _, current_id):
                            if current_id != ident:
                                continue
                            acl = self._acl(ws, change)
                            matches = self._unrecorded(
                                acl, self._matches(ws, acl, delta, flags), aces
                            )
                            if len(matches) > 1:
                                return False
                            if matches:
                                acl.DeleteAce(matches[0])
                                self._set_acl(ws, change, acl)
                    except FileNotFoundError:
                        continue
                    except SharedFile as exc:
                        ask("hardlink", names, ident, HARDLINK, held, links=exc.links)
                        continue
                    except (PermissionError, Refused):
                        kept.append(names)
                        continue
                elif item is not None and item.get("applied"):
                    # It was already removed externally; no owned rights remain.
                    pass
                owned.add(names)
                if item is not None:
                    item["removed"] = True
            # Detect files created in the narrow interval before a parent's
            # ACE was removed, and any entry for the account that was not
            # there before the grant. A retry uses the same recorded plan.
            for names, held, ident, acl, directory, _, links in self._tree(
                ws, path, handle, identity, revoke=True, descend=descend, unreadable=unreadable
            ):
                key = _file_key(ident)
                if not directory and links > limit(key):
                    ask("hardlink", names, ident, HARDLINK, held, links=links)
                    continue
                if acl is None and key not in open_before:
                    ask("null_dacl", names, ident, NULL_DACL, held)
                if key not in owners_before and self._owner(ws, held) == delta["sid"]:
                    ask("owner", names, ident, OWNED.format(account=self.account), held)
                if (
                    names
                    and names[:-1] in owned
                    and self._matches(ws, acl, delta, {19 if directory else 16})
                ):
                    raise OSError(
                        "New files inherited access during removal. "
                        "Some additions remain; try Remove access again."
                    )
                if names in kept:
                    continue
                left = [list(ace) for ace in manual.get(key, [])]
                for ace_flags, ace_mask in self._explicit(ws, acl, delta["sid"]):
                    if [ace_flags, ace_mask] in left:
                        left.remove([ace_flags, ace_mask])
                        continue
                    ask(
                        "entry",
                        names,
                        ident,
                        self._words(ace_mask, ace_flags),
                        held,
                        flags=ace_flags,
                        mask=ace_mask,
                    )
            for names in dict.fromkeys(unreadable):
                ask("unreadable", names, [], UNREADABLE)
            # What the owner kept and Eugene met again: reported on every
            # removal, so the record never shows `removed` without it.
            delta["retained"] = list(retained.values())
            if kept:
                shown = [str(Path(path).joinpath(*names)) for names in kept[:3]]
                more = f" and {len(kept) - 3} more" if len(kept) > 3 else ""
                raise RemovalIncomplete(
                    f"Eugene could not remove its own permission from {', '.join(shown)}{more}: "
                    "the item's permissions were changed outside Eugene. Remove that access "
                    "in the folder's Security settings, then try Remove access again."
                )
            if asked:
                raise UnknownAccess(list(asked.values()))
            return True

    def _words(self, mask: int, flags: int) -> str:
        rights = {
            self.READ: "read",
            self.WRITE: "read and write",
            0x1301BF: "modify",
            0x1F01FF: "full control",
        }.get(mask, f"the rights 0x{mask:X}")
        reach = " on this item and everything in it" if flags & 3 == 3 else " on this item"
        return f"{self.account} has {rights}{reach}, set outside Eugene"

    def decide(self, path: str, delta: dict, fingerprint: dict, action: str) -> None:
        """Keep one unknown entry, or check it again and remove only it.

        Every step reopens the item below the grant's own folder, one component
        at a time, refusing a link in any of them, and acts only on the owner and
        DACL it was asked about (StaleEntry otherwise)."""
        import pywintypes
        import win32security as ws

        if action == "keep":
            accepted = delta.setdefault("accepted", [])
            if _answer_key(fingerprint) not in {_answer_key(a) for a in accepted}:
                accepted.append(fingerprint)
            return
        kind, names = fingerprint["kind"], tuple(fingerprint["names"])
        ident = fingerprint["identity"]
        state = fingerprint.get("state", "")  # none known: it can never match
        root = delta["identity"]
        if kind == "unreadable":
            raise ValueError(
                "Eugene cannot read this item's permissions. Change them in the folder's "
                "Security settings, then choose Remove access again."
            )
        replaced = StaleEntry(
            "The item changed since Eugene found it, so the entry was left as it is. "
            "Eugene asks again about what is there now."
        )
        try:
            allowed = {key: links for key, links in delta.get("allowed", [])}
            most = max(1, allowed.get(_file_key(ident), 1)) if ident else 1
            if kind == "hardlink":
                with self._open(path, revoke=True, names=names, root_identity=root) as (
                    held,
                    facts,
                    now,
                ):
                    if now != ident:
                        raise replaced
                    self._unchanged(ws, held, state)
                    if facts.links > fingerprint.get("links", facts.links):
                        raise SharedFile(facts.links)
                    links = facts.links
                # The removal that follows changes it, at no more names than now.
                delta["allowed"] = [
                    [key, count]
                    for key, count in delta.get("allowed", [])
                    if key != _file_key(ident)
                ] + [[_file_key(ident), links]]
                return
            if kind == "null_dacl":
                self._inherit(ws, path, names, ident, root=root, state=state, max_links=most)
                return
            if kind == "owner":
                with self._open(path, root_identity=root) as (handle, _, _):
                    person: str | None = self._owner(ws, handle)
                self._give(
                    ws,
                    path,
                    names,
                    ident,
                    None if person == delta["sid"] else person,
                    root=root,
                    state=state,
                    max_links=most,
                )
                return
            with self._open(
                path,
                change=True,
                revoke=True,
                names=names,
                root_identity=root,
                max_links=most,
            ) as (held, _, current):
                if current != ident:
                    raise replaced
                self._unchanged(ws, held, state)
                descriptor = ws.GetSecurityInfo(
                    held, ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION
                )
                acl = descriptor.GetSecurityDescriptorDacl()
                index = next(
                    (
                        i
                        for i in range(acl.GetAceCount() if acl is not None else 0)
                        if acl.GetAce(i)[0][0] in (0, 5)
                        and acl.GetAce(i)[0][1] == fingerprint["flags"]
                        and acl.GetAce(i)[1] == fingerprint["mask"]
                        and ws.ConvertSidToStringSid(acl.GetAce(i)[-1]) == delta["sid"]
                    ),
                    None,
                )
                if index is None:
                    raise replaced
                acl.DeleteAce(index)
                protected = descriptor.GetSecurityDescriptorControl()[0] & 0x1000
                # SetSecurityInfo also takes the entry's inherited copies
                # off the items below this one.
                ws.SetSecurityInfo(
                    held,
                    ws.SE_FILE_OBJECT,
                    ws.DACL_SECURITY_INFORMATION
                    | (
                        ws.PROTECTED_DACL_SECURITY_INFORMATION
                        if protected
                        else ws.UNPROTECTED_DACL_SECURITY_INFORMATION
                    ),
                    None,
                    None,
                    acl,
                    None,
                )
        except Refused as exc:
            # A link now stands in the path: nothing outside the grant is touched.
            raise StaleEntry(
                "The item changed since Eugene found it (a link now stands in its path), "
                "so the entry was left as it is. Eugene asks again about what is there now."
            ) from exc
        except FileNotFoundError:
            return  # Gone: nothing left to remove.
        except pywintypes.error as exc:
            raise OSError(f"The entry could not be removed: {exc.strerror}.") from exc


class LinuxPermissions:
    """Ask the installed root broker; its root-owned ledger is authoritative."""

    #: An inspection changes nothing and takes the helper well under a second.
    #: It runs under the module lock, so a helper that never answers (not
    #: installed, its path unit stopped, or busy with a large grant) must end
    #: the wait and name itself, not hold every folder-access request forever.
    INSPECT_TIMEOUT_SECONDS = 30.0
    #: The root helper records each grant's own entries and restores the
    #: overlapping grants that remain when one is removed.
    overlap = True

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
        # Only its result can declare a grant or a revoke done, so those wait.
        # An inspection changes nothing and may give up.
        timeout = self.INSPECT_TIMEOUT_SECONDS if verb == "folder-inspect" else None
        started = time.perf_counter() if timeout is not None else 0.0
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
            if timeout is not None and time.perf_counter() - started >= timeout:
                # Withdraw the request, so a helper that starts later does not
                # act on it for nobody.
                request.unlink(missing_ok=True)
                raise TimeoutError(
                    f"the folder permission helper did not answer in {timeout:g} seconds "
                    "(it is busy changing a large folder, or it is not running)"
                )
            time.sleep(0.2)

    def grant(self, path: str, mode: str, save: Callable[[dict], None]) -> tuple[dict, int, bool]:
        facts, identity = self.inspect(path)
        if cause := refusal(facts):
            raise Refused(cause, facts.unsafe_parent if cause == "unsafe_parent" else path)
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
        # What the owner kept, from the helper's ledger. Nothing is shown as
        # removed without it (an old helper that does not say keeps the saved list).
        if "kept" in result:
            delta["retained"] = [
                dict(
                    kind=found["fingerprint"]["kind"],
                    path=found["path"],
                    entry=found["entry"],
                    fingerprint=found["fingerprint"],
                )
                for found in result["kept"]
            ]
        if result.get("unknown"):
            # The helper left these as they are; the owner decides (R23-UNKNOWN).
            raise UnknownAccess(
                [
                    dict(
                        kind=found["fingerprint"]["kind"],
                        path=found["path"],
                        entry=found["entry"],
                        fingerprint=found["fingerprint"],
                    )
                    for found in result["unknown"]
                ]
            )
        if not result["removed"] and result.get("message"):
            raise RemovalIncomplete(
                f"Some items were not changed: {str(result['message']).rstrip('.')}. "
                "Try Remove access again."
            )
        return bool(result["removed"])

    def decide(self, path: str, delta: dict, fingerprint: dict, action: str) -> None:
        """The root helper checks the entry against its own ledger, then acts."""
        self.ctl.mkdir(parents=True, exist_ok=True)
        request = self.ctl / f"{delta['id']}.entry"
        write_private(request, json.dumps(fingerprint))
        try:
            result = self._ask("folder-decide", delta["id"], action)
        except OSError as exc:
            raise ValueError(str(exc).rstrip(".") + ".") from exc
        finally:
            request.unlink(missing_ok=True)
        if result.get("stale"):
            # The helper applied nothing: the file gained another name, or its
            # entry changed. The removal runs again and asks about what is there.
            raise StaleEntry(str(result.get("message") or "The item changed since Eugene asked."))
