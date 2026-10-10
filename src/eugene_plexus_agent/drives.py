"""What kind of drive a folder is on: an SSD, a spinning disk, or a network share (LS7).

Strata reads its experts and tables from disk while it answers (upstream
#605: a spinning disk stalls prompts for minutes; a share is worse), and a
node's model copy exists to put the model on a fast local drive. So the
console says which drive a copy folder is on, warns when Strata is
installed with its files on a slow one, and names the drive when a Strata
model is about to run from one (Troy, reviewing B51).

The operating system's own answer, never a guess from a name or a speed:

* Windows: `GetDriveTypeW` says a share (`DRIVE_REMOTE`, and any UNC path);
  for a local volume, `IOCTL_STORAGE_QUERY_PROPERTY` with
  `StorageDeviceSeekPenaltyProperty` says whether the disk incurs a seek
  penalty, which is how Windows itself tells a spinning disk from an SSD.
  No administrator rights and no subprocess.
* Linux: the mount's file-system type says a share (NFS, SMB, SSHFS); the
  block device's `queue/rotational` says spinning or not.
* Anything else, or any question the system will not answer: `unknown`,
  which never warns.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from typing import Literal

DriveKind = Literal["ssd", "hdd", "network", "unknown"]

#: How long an answer is reused: the console asks on every read of the
#: settings and every launch, and a drive's kind does not change.
CACHE_SECONDS = 300.0
_cache: dict[str, tuple[float, DriveKind]] = {}

#: File systems that are someone else's disk across a network.
NETWORK_FILESYSTEMS = frozenset(
    {"nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse.sshfs", "sshfs", "9p", "afs", "ceph"}
)

WORDS: dict[DriveKind, str] = {
    "ssd": "an SSD",
    "hdd": "a spinning disk",
    "network": "a network share",
    "unknown": "a drive whose kind this machine did not say",
}


def drive_kind(path: str) -> DriveKind:
    """The kind of drive `path` is on (it need not exist yet: the nearest
    existing parent decides)."""
    if not path:
        return "unknown"
    if path.startswith(("\\\\", "//")):
        return "network"
    target = _existing(os.path.abspath(os.path.expanduser(path)))
    key = _volume_key(target)
    now = time.perf_counter()
    cached = _cache.get(key)
    if cached is not None and now - cached[0] <= CACHE_SECONDS:
        return cached[1]
    try:
        kind = _windows_kind(target) if sys.platform == "win32" else _linux_kind(target)
    except Exception:
        kind = "unknown"
    _cache[key] = (now, kind)
    return kind


def slow(kind: DriveKind) -> bool:
    return kind in ("hdd", "network")


def _existing(path: str) -> str:
    current = path
    while not os.path.exists(current):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return current


def _volume_key(path: str) -> str:
    if sys.platform == "win32":
        drive, _ = os.path.splitdrive(path)
        return drive.upper() or path
    try:
        return f"dev:{os.stat(path).st_dev}"
    except OSError:
        return path


# --- Windows -----------------------------------------------------------------

_DRIVE_REMOTE = 4
_DRIVE_FIXED = 3
_IOCTL_STORAGE_QUERY_PROPERTY = 0x002D1400
_STORAGE_DEVICE_SEEK_PENALTY_PROPERTY = 7
_PROPERTY_STANDARD_QUERY = 0
_FILE_SHARE_READ_WRITE = 0x00000001 | 0x00000002
_OPEN_EXISTING = 3


class _StoragePropertyQuery(ctypes.Structure):
    _fields_ = [
        ("PropertyId", ctypes.c_int),
        ("QueryType", ctypes.c_int),
        ("AdditionalParameters", ctypes.c_byte * 1),
    ]


class _SeekPenaltyDescriptor(ctypes.Structure):
    _fields_ = [
        ("Version", ctypes.c_ulong),
        ("Size", ctypes.c_ulong),
        ("IncursSeekPenalty", ctypes.c_ubyte),
    ]


def _windows_kind(path: str) -> DriveKind:
    from ctypes import wintypes

    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
    drive, _ = os.path.splitdrive(path)
    if not drive:
        return "unknown"
    if drive.startswith(("\\\\", "//")):
        return "network"
    kind = kernel32.GetDriveTypeW(ctypes.c_wchar_p(drive + "\\"))
    if kind == _DRIVE_REMOTE:
        return "network"
    if kind != _DRIVE_FIXED:
        return "unknown"
    create = kernel32.CreateFileW
    create.restype = wintypes.HANDLE
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    handle = create(f"\\\\.\\{drive}", 0, _FILE_SHARE_READ_WRITE, None, _OPEN_EXISTING, 0, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        return "unknown"
    try:
        query = _StoragePropertyQuery(
            _STORAGE_DEVICE_SEEK_PENALTY_PROPERTY, _PROPERTY_STANDARD_QUERY
        )
        answer = _SeekPenaltyDescriptor()
        returned = wintypes.DWORD(0)
        ok = kernel32.DeviceIoControl(
            handle,
            _IOCTL_STORAGE_QUERY_PROPERTY,
            ctypes.byref(query),
            ctypes.sizeof(query),
            ctypes.byref(answer),
            ctypes.sizeof(answer),
            ctypes.byref(returned),
            None,
        )
        if not ok or returned.value < ctypes.sizeof(answer):
            return "unknown"
        return "hdd" if answer.IncursSeekPenalty else "ssd"
    finally:
        kernel32.CloseHandle(handle)


# --- Linux -------------------------------------------------------------------


def _linux_kind(path: str) -> DriveKind:
    fstype = _mount_type(path)
    if fstype is not None and fstype in NETWORK_FILESYSTEMS:
        return "network"
    st = os.stat(path)
    # Through getattr: Windows' `os` has neither, and this runs only on Linux.
    major, minor = getattr(os, "major")(st.st_dev), getattr(os, "minor")(st.st_dev)  # noqa: B009
    device = os.path.realpath(f"/sys/dev/block/{major}:{minor}")
    for candidate in (device, os.path.dirname(device)):
        flag = os.path.join(candidate, "queue", "rotational")
        if os.path.isfile(flag):
            with open(flag, encoding="ascii") as handle:
                return "hdd" if handle.read().strip() == "1" else "ssd"
    return "unknown"


def _mount_type(path: str) -> str | None:
    """The file-system type of the mount holding `path` (the longest
    mount point that contains it), from `/proc/self/mountinfo`."""
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    best: tuple[int, str] | None = None
    for line in lines:
        left, _, right = line.partition(" - ")
        fields = left.split()
        if len(fields) < 5 or not right:
            continue
        point = fields[4].replace("\\040", " ")
        if path == point or path.startswith(point.rstrip("/") + "/") or point == "/":
            fstype = right.split()[0]
            if best is None or len(point) > best[0]:
                best = (len(point), fstype)
    return best[1] if best is not None else None


def copy_folder_status(folder: str) -> tuple[str, str] | None:
    """`(level, sentence)` for the Settings line beside a copy folder, or
    None when the drive's kind is not known."""
    kind = drive_kind(folder)
    if kind == "ssd":
        return ("info", "On an SSD.")
    if kind == "hdd":
        return (
            "warning",
            "On a spinning disk: models copied here load slower, and Strata, which reads "
            "its model from disk while it answers, stalls on one. Pick a folder on an SSD.",
        )
    if kind == "network":
        return (
            "warning",
            "On a network share: a copy here is read over the network like the Library "
            "itself, which is what copying exists to avoid. Pick a folder on this "
            "machine's own SSD.",
        )
    return None


def strata_install_note(copy_folder: str | None) -> str | None:
    """What Strata's install says when it finishes, when this node's copy
    folder is on a slow drive (Troy, reviewing B51: say so up front, so a
    folder chosen earlier gets moved)."""
    if not copy_folder:
        return None
    kind = drive_kind(copy_folder)
    if not slow(kind):
        return None
    return (
        f"this machine copies models to {copy_folder}, on {WORDS[kind]}; Strata reads its "
        "model from disk while it answers and needs an SSD: move the copy folder in "
        "Settings, Model storage on each machine"
    )


def strata_run_warning(path: str) -> str | None:
    """A Strata start's warning when the files it would open are on a slow
    drive, naming the drive."""
    kind = drive_kind(path)
    if not slow(kind):
        return None
    drive = os.path.splitdrive(os.path.abspath(path))[0] or path
    return (
        f"Strata will read this model from {drive}, {WORDS[kind]}, while it answers, and "
        "stalls on one (upstream #605). Turn on this machine's model copies with a folder "
        "on an SSD (Settings, Model storage on each machine)."
    )
