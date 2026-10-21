"""Retain one app update's data, outside the app account's writable directory."""

from __future__ import annotations

import json
import math
import os
import secrets
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

METADATA = "backup.json"


def relative_path(value: str) -> Path:
    """One portable relative name, with no alternate Windows spelling."""
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if (
        not value
        or value in {".", ".."}
        or posix.is_absolute()
        or windows.root
        or windows.drive
        or "\\" in value
        or ":" in value
        or any(p in {"", ".", ".."} or p.endswith((".", " ")) for p in value.split("/"))
        or any(
            p.upper().split(".")[0]
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *[f"COM{i}" for i in range(1, 10)],
                *[f"LPT{i}" for i in range(1, 10)],
            }
            for p in posix.parts
        )
    ):
        raise ValueError(f"updateBackup path {value!r} must stay inside the app's data directory")
    return Path(*posix.parts)


def validate_paths(paths: list[str]) -> None:
    checked: list[Path] = []
    for value in paths:
        path = relative_path(value)
        if any(path == old or path in old.parents or old in path.parents for old in checked):
            raise ValueError("updateBackup paths must not overlap")
        checked.append(path)
    if any(p.parts[0] in {METADATA, "given.json", "connection.sha256"} for p in checked):
        raise ValueError("updateBackup names an agent settings record")


def _plain(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ValueError(f"updateBackup refuses a link or reparse point: {path.name}")
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise ValueError(f"updateBackup refuses a special file: {path.name}")


def inside(root: Path, value: str) -> Path:
    path = root / relative_path(value)
    _plain(root)
    for part in [*reversed(path.parents), path]:
        if part == root or root in part.parents:
            _plain(part)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("updateBackup path escapes its directory")
    return path


def _size(path: Path) -> int:
    _plain(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(_size(child) for child in path.iterdir())


def _copy(source: Path, target: Path) -> None:
    _size(source)  # Check every child before following any directory.
    if source.is_dir():
        shutil.copytree(source, target)
    elif source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _remove(path: Path) -> None:
    _size(path)  # No recursive operation through an app-controlled reparse point.
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def directory(app_dir: Path, version: str) -> Path:
    relative_path(version)
    if "/" in version:
        raise ValueError("updateBackup version must be one directory name")
    _plain(app_dir)
    root = app_dir / "backups"
    _plain(root)
    return inside(root, version)


def check_space(app_dir: Path, data: Path, paths: list[str], name: str) -> None:
    validate_paths(paths)
    total = sum(_size(inside(data, p)) for p in _paths(paths))
    required = math.ceil(total * 1.2)
    free = shutil.disk_usage(app_dir).free
    if free < required:
        volume = app_dir.anchor or str(app_dir)
        raise ValueError(
            f"The update was not started: {name}'s data needs {required / 1e9:.1f} GB "
            f"for its backup, and {volume} has {free / 1e9:.1f} GB free."
        )


def create(app_dir: Path, data: Path, manifest: dict[str, Any], paths: list[str]) -> None:
    """Caller has stopped the writer. Publish metadata only after every copy."""
    target = directory(app_dir, manifest["version"])
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = inside(target.parent, ".pending-" + secrets.token_hex(12))
    staging.mkdir(mode=0o700)
    try:
        present = {}
        for name in _paths(paths):
            source = inside(data, name)
            present[name] = source.exists()
            if present[name]:
                _copy(source, inside(staging, name))
        metadata = {
            "version": manifest["version"],
            "createdAt": datetime.now(UTC).isoformat(),
            "manifest": manifest,
            "paths": present,
        }
        (staging / METADATA).write_text(json.dumps(metadata), encoding="utf-8")
        _remove(target)
        os.replace(staging, target)
        # A completed update replaces the previously retained backup.
        for child in target.parent.iterdir():
            if child != target:
                _remove(child)
    finally:
        _remove(staging)


def _paths(paths: list[str]) -> list[str]:
    # A stopped/crashed SQLite writer can leave transactions in its WAL.
    # Restore the whole database, including absence of newly created sidecars.
    extra = [
        p + suffix for p in paths if p.endswith(".db") for suffix in ("-wal", "-shm", "-journal")
    ]
    return list(dict.fromkeys([*paths, *extra, "given.json", "connection.sha256"]))


def read(app_dir: Path, version: str | None) -> dict[str, Any] | None:
    if not version:
        return None
    target = directory(app_dir, version)
    metadata = inside(target, METADATA)
    if not metadata.is_file():
        return None
    if metadata.stat().st_size > 128 * 1024:
        raise ValueError("The update backup record is too large")
    record = json.loads(metadata.read_text(encoding="utf-8"))
    if record.get("version") != version or not isinstance(record.get("paths"), dict):
        raise ValueError("The update backup record does not match the previous version")
    for name, present in record["paths"].items():
        if not isinstance(present, bool):
            raise ValueError("The update backup has an invalid path record")
        path = inside(target, name)
        if present and not path.exists():
            raise ValueError(f"The update backup is missing {name}")
        _size(path)
    return record


def restore(app_dir: Path, data: Path, record: dict[str, Any]) -> None:
    target = directory(app_dir, record["version"])
    # Validate all sources and destinations before removing the first file.
    for name in record["paths"]:
        _size(inside(target, name))
        _size(inside(data, name))
    for name, present in record["paths"].items():
        destination = inside(data, name)
        _remove(destination)
        if present:
            _copy(inside(target, name), destination)


def consume(app_dir: Path, version: str) -> None:
    _remove(directory(app_dir, version))
