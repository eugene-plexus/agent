"""Windows file handles relative to held parents; no Win32 path reopening.

NtCreateFile opens each single component with FILE_OPEN_REPARSE_POINT.
Parent handles remain open without FILE_SHARE_DELETE throughout the call.
NtQueryDirectoryFile enumerates the directory handle, not its pathname.
"""

from __future__ import annotations

import ctypes
import os
import re
import struct
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes as w
from pathlib import Path
from typing import Any, Self

from .folder_io import FolderError, parts

if sys.platform == "win32":

    class _Unicode(ctypes.Structure):
        _fields_ = [("Length", w.USHORT), ("MaximumLength", w.USHORT), ("Buffer", w.LPWSTR)]

    class _Attributes(ctypes.Structure):
        _fields_ = [
            ("Length", w.ULONG),
            ("RootDirectory", w.HANDLE),
            ("ObjectName", ctypes.POINTER(_Unicode)),
            ("Attributes", w.ULONG),
            ("SecurityDescriptor", w.LPVOID),
            ("SecurityQualityOfService", w.LPVOID),
        ]

    class _Status(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

    def _apis() -> tuple[Any, Any]:
        nt = ctypes.WinDLL("ntdll", use_last_error=True)
        nt.NtCreateFile.argtypes = [
            ctypes.POINTER(w.HANDLE),
            w.ULONG,
            ctypes.POINTER(_Attributes),
            ctypes.POINTER(_Status),
            w.LPVOID,
            w.ULONG,
            w.ULONG,
            w.ULONG,
            w.ULONG,
            w.LPVOID,
            w.ULONG,
        ]
        nt.NtCreateFile.restype = w.LONG
        nt.NtQueryDirectoryFile.argtypes = [
            w.HANDLE,
            w.HANDLE,
            w.LPVOID,
            w.LPVOID,
            ctypes.POINTER(_Status),
            w.LPVOID,
            w.ULONG,
            w.ULONG,
            w.BOOLEAN,
            w.LPVOID,
            w.BOOLEAN,
        ]
        nt.NtQueryDirectoryFile.restype = w.LONG
        nt.RtlNtStatusToDosError.argtypes = [w.LONG]
        nt.RtlNtStatusToDosError.restype = w.ULONG
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.GetFileType.argtypes = [w.HANDLE]
        kernel.GetFileType.restype = w.DWORD
        kernel.GetDriveTypeW.argtypes = [w.LPCWSTR]
        kernel.GetDriveTypeW.restype = w.UINT
        kernel.GetFinalPathNameByHandleW.argtypes = [w.HANDLE, w.LPWSTR, w.DWORD, w.DWORD]
        kernel.GetFinalPathNameByHandleW.restype = w.DWORD
        return nt, kernel

    def _handle(fd: int) -> int:
        import msvcrt

        return int(msvcrt.get_osfhandle(fd))

    def _open(
        parent: int | None, name: str, *, directory: bool, write: bool = False, create: bool = False
    ) -> int:
        import msvcrt

        nt, kernel = _apis()
        buffer = ctypes.create_unicode_buffer(name)
        count = len(name.encode("utf-16-le"))
        unicode = _Unicode(count, count + 2, ctypes.cast(buffer, w.LPWSTR))
        attrs = _Attributes(
            ctypes.sizeof(_Attributes),
            _handle(parent) if parent is not None else None,
            ctypes.pointer(unicode),
            0x40,
            None,
            None,
        )
        handle, status = w.HANDLE(), _Status()
        # READ_ATTRIBUTES | SYNCHRONIZE | READ_DATA/LIST_DIRECTORY; no truncate.
        access = 0x100081 | (0x2 if write else 0)
        # OPEN_REPARSE_POINT | SYNCHRONOUS_IO_NONALERT | DIRECTORY/NON_DIRECTORY.
        options = 0x200020 | (0x1 if directory else 0x40)
        result = nt.NtCreateFile(
            ctypes.byref(handle),
            access,
            ctypes.byref(attrs),
            ctypes.byref(status),
            None,
            0,
            3 if directory else 1,
            2 if create else 1,
            options,
            None,
            0,
        )
        if result < 0:
            raise ctypes.WinError(nt.RtlNtStatusToDosError(result))
        try:
            if kernel.GetFileType(handle) != 1:  # FILE_TYPE_DISK
                raise FolderError("Only ordinary files and folders on a local disk are supported.")
            assert handle.value is not None
            fd = msvcrt.open_osfhandle(
                handle.value, os.O_BINARY | (os.O_RDWR if write else os.O_RDONLY)
            )
        except BaseException:
            kernel.CloseHandle(handle)
            raise
        try:
            if os.fstat(fd).st_file_attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
                raise FolderError("Links, junctions and other reparse points are not allowed.")
            return fd
        except BaseException:
            os.close(fd)
            raise

    class Root:
        def __init__(self, path: str) -> None:
            value = Path(path)
            if not re.fullmatch(r"[A-Za-z]:", value.drive) or not value.is_absolute():
                raise FolderError(
                    "Use a local drive path. Network and device paths are not supported."
                )
            _, kernel = _apis()
            if kernel.GetDriveTypeW(value.anchor) != 3:
                raise FolderError("Use a folder on a local fixed disk.")
            self.parents: list[int] = []
            try:
                self.parents.append(_open(None, "\\??\\" + value.anchor, directory=True))
                for name in value.parts[1:]:
                    parts(name)
                    self.parents.append(_open(self.parents[-1], name, directory=True))
                self.fd = self.parents[-1]
                buffer = ctypes.create_unicode_buffer(32768)
                length = kernel.GetFinalPathNameByHandleW(_handle(self.fd), buffer, len(buffer), 0)
                if not length or length >= len(buffer):
                    raise FolderError("Could not establish this folder's local disk path.")
                self.path = buffer.value.removeprefix("\\\\?\\")
            except BaseException:
                self.__exit__()
                raise

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            for fd in reversed(self.parents):
                os.close(fd)
            self.parents.clear()

        @contextmanager
        def _directory(self, names: list[str]) -> Iterator[int]:
            held = [self.fd]
            try:
                for name in names:
                    held.append(_open(held[-1], name, directory=True))
                yield held[-1]
            finally:
                for fd in reversed(held[1:]):
                    os.close(fd)

        @contextmanager
        def file(
            self, names: list[str], *, write: bool = False, create: bool = False
        ) -> Iterator[int]:
            with self._directory(names[:-1]) as parent:
                fd = _open(parent, names[-1], directory=False, write=write, create=create)
                try:
                    yield fd
                finally:
                    os.close(fd)

        def names(self, names: list[str], limit: int) -> list[str]:
            nt, _ = _apis()
            with self._directory(names) as parent:
                result: list[str] = []
                buffer = ctypes.create_string_buffer(65536)
                status = _Status()
                first = True
                while len(result) < limit:
                    code = nt.NtQueryDirectoryFile(
                        _handle(parent),
                        None,
                        None,
                        None,
                        ctypes.byref(status),
                        buffer,
                        len(buffer),
                        12,
                        False,
                        None,
                        first,
                    )
                    first = False
                    if code & 0xFFFFFFFF == 0x80000006:  # STATUS_NO_MORE_FILES
                        break
                    if code < 0:
                        raise ctypes.WinError(nt.RtlNtStatusToDosError(code))
                    offset = 0
                    while offset < status.Information:
                        next_offset, _, size = struct.unpack_from("<III", buffer, offset)
                        name = buffer.raw[offset + 12 : offset + 12 + size].decode("utf-16-le")
                        if name not in {".", ".."}:
                            result.append(name)
                        if not next_offset or len(result) >= limit:
                            break
                        offset += next_offset
                return result
