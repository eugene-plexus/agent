"""Linux held handles and a Landlock boundary for each built-in file call.

DynamicUser implies RestrictSUIDSGID, which blocks openat2. Landlock keeps
kernel-enforced containment without weakening that service policy. Only a
fresh, disposable thread is restricted; it never executes untrusted code.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import platform
import struct
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import Self

from .folder_io import FolderError

_fcntl = importlib.import_module("fcntl")


def _flag(name: str) -> int:
    return int(getattr(os, name))


def isolated[T](operation: Callable[[], T]) -> T:
    # A pooled worker must never retain the previous grant's Landlock policy.
    result: Future[T] = Future()

    def run() -> None:
        try:
            result.set_result(operation())
        except BaseException as exc:
            result.set_exception(exc)

    thread = threading.Thread(target=run, name="workbench-file-call")
    thread.start()
    thread.join()
    return result.result()


def _libc() -> ctypes.CDLL:
    if platform.machine().lower() not in {"x86_64", "aarch64", "amd64", "arm64"}:
        raise FolderError("Folder tools currently support 64-bit x86 and ARM Linux hosts.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def _supported() -> None:
    # Linux UAPI: landlock_create_ruleset(NULL, 0, CREATE_RULESET_VERSION).
    if _libc().syscall(ctypes.c_long(444), None, ctypes.c_size_t(0), ctypes.c_uint(1)) < 3:
        raise FolderError("Linux folder tools require enabled Landlock ABI 3 or newer.")


def _statx(fd: int) -> bytes:
    libc = _libc()
    buffer = ctypes.create_string_buffer(256)
    try:
        statx = libc.statx
    except AttributeError:
        raise FolderError("This Linux host needs statx support for folder tools.") from None
    statx.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
    statx.restype = ctypes.c_int
    # AT_EMPTY_PATH; request BTIME and MNT_ID. Both are required, not optional.
    if (
        statx(fd, b"", 0x1000, 0x1800, buffer) < 0
        or struct.unpack_from("=I", buffer)[0] & 0x1800 != 0x1800
    ):
        raise FolderError("This filesystem must expose folder creation time and mount identity.")
    return buffer.raw


def birth_time(fd: int) -> str:
    seconds, nanos = struct.unpack_from("=qI", _statx(fd), 80)
    return f"{seconds}:{nanos}"


def _mount(fd: int) -> int:
    return int(struct.unpack_from("=Q", _statx(fd), 144)[0])


def _restrict(fd: int, writable: bool) -> None:
    # Handle all ABI-3 filesystem rights. Allow only directory/text reads,
    # plus regular-file creation, writes and truncate for an approved write.
    handled = ctypes.c_uint64((1 << 15) - 1)
    allowed = (1 << 2) | (1 << 3)
    if writable:
        allowed |= (1 << 1) | (1 << 8) | (1 << 14)
    libc = _libc()
    rules = libc.syscall(
        ctypes.c_long(444), ctypes.byref(handled), ctypes.c_size_t(8), ctypes.c_uint(0)
    )
    if rules < 0:
        raise FolderError("The host could not create the required file boundary.")
    try:
        # struct landlock_path_beneath_attr is packed: u64 rights, s32 parent_fd.
        rule = ctypes.create_string_buffer(struct.pack("=Qi", allowed, fd))
        added = libc.syscall(
            ctypes.c_long(445),
            ctypes.c_int(rules),
            ctypes.c_int(1),
            ctypes.byref(rule),
            ctypes.c_uint(0),
        )
        if added < 0 or libc.prctl(38, 1, 0, 0, 0) < 0:
            raise FolderError("The host could not prepare the required file boundary.")
        if libc.syscall(ctypes.c_long(446), ctypes.c_int(rules), ctypes.c_uint(0)) < 0:
            raise FolderError("The host could not enforce the required file boundary.")
    finally:
        os.close(rules)


def _open(parent: int | None, name: str, flags: int) -> int:
    return os.open(name, flags | _flag("O_CLOEXEC") | _flag("O_NOFOLLOW"), 0o660, dir_fd=parent)


class Root:
    def __init__(self, path: str) -> None:
        _supported()
        value = Path(path)
        if not value.is_absolute() or ".." in value.parts:
            raise FolderError("Use the full path to an existing folder.")
        fd = _open(None, "/", os.O_RDONLY | _flag("O_DIRECTORY"))
        try:
            for name in value.parts[1:]:
                following = _open(fd, name, os.O_RDONLY | _flag("O_DIRECTORY"))
                os.close(fd)
                fd = following
            self.path = os.readlink(f"/proc/self/fd/{fd}")
            self.mount = _mount(fd)
            self.fd = fd
        except BaseException:
            os.close(fd)
            raise

    def restrict(self, writable: bool) -> None:
        _restrict(self.fd, writable)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        os.close(self.fd)

    def _same_mount(self, fd: int) -> None:
        if _mount(fd) != self.mount:
            raise FolderError("Mounted subfolders and files are not supported.")

    @contextmanager
    def _directory(self, names: list[str]) -> Iterator[int]:
        held = [self.fd]
        try:
            for name in names:
                fd = _open(held[-1], name, os.O_RDONLY | _flag("O_DIRECTORY"))
                held.append(fd)
                self._same_mount(fd)
            yield held[-1]
        finally:
            for fd in reversed(held[1:]):
                os.close(fd)

    @contextmanager
    def file(self, names: list[str], *, write: bool = False, create: bool = False) -> Iterator[int]:
        flags = (os.O_RDWR if write else os.O_RDONLY) | _flag("O_NONBLOCK")
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        with self._directory(names[:-1]) as parent:
            fd = _open(parent, names[-1], flags)
            try:
                self._same_mount(fd)
                _fcntl.flock(fd, (_fcntl.LOCK_EX if write else _fcntl.LOCK_SH) | _fcntl.LOCK_NB)
                yield fd
            finally:
                os.close(fd)

    def names(self, names: list[str], limit: int) -> list[str]:
        with self._directory(names) as parent, os.scandir(parent) as entries:
            out = []
            for entry in entries:
                out.append(entry.name)
                if len(out) >= limit:
                    break
            return out
