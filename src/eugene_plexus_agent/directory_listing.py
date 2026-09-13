"""One directory on this host, listed for a path picker (M11).

Behind `GET /v1/directories`: the picker `path_list` has promised since
M2 -- *"an add/remove list of directory pickers"* -- finally has
something to list. On the agent that is a path mapping's `to` and an
engine binary; the library carries the same module for its model roots.
**Two copies, one schema**, which is this project's rule: components
share schemas, not code, and eighty lines duplicated is cheaper than a
shared package that every component would have to install into the
agent's venv.

What it is not: a file browser. It lists what an operator could already
type into a path field, on request, to the strongest credential there
is -- and nothing else. A directory the component may not read is a
403, an entry it may not stat is skipped, and a dead network share
blocks for as long as the OS takes to give up, which is why the route
that calls this is a plain `def` and runs in a worker thread.
"""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from ._generated.common_models import DirectoryEntry, DirectoryEntryKind, DirectoryListing

# Windows' hidden attribute bit. Named here rather than read off `stat`
# because typeshed declares those constants for Windows only, so a
# reference to `stat.FILE_ATTRIBUTE_HIDDEN` fails the Linux type check
# that CI runs.
_HIDDEN_ATTRIBUTE = 0x2


@dataclass(frozen=True)
class ListingError(Exception):
    """Why a directory could not be listed, with the status it maps to."""

    status: int
    title: str
    detail: str

    def __str__(self) -> str:
        return self.detail


def host_name() -> str:
    return socket.gethostname()


def _is_hidden(entry: os.DirEntry[str]) -> bool:
    if entry.name.startswith("."):
        return True
    try:
        attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & _HIDDEN_ATTRIBUTE)


def starting_points(*, host: str | None = None) -> DirectoryListing:
    """Where a picker begins with no path: every drive on Windows, `/`
    on POSIX, and the home directory either way."""
    entries: list[DirectoryEntry] = []
    listdrives = getattr(os, "listdrives", None)
    if sys.platform == "win32" and listdrives is not None:
        for drive in listdrives():
            entries.append(
                DirectoryEntry(name=drive, path=drive, kind=DirectoryEntryKind.directory)
            )
    else:
        entries.append(DirectoryEntry(name="/", path="/", kind=DirectoryEntryKind.directory))
    home = Path.home()
    entries.append(DirectoryEntry(name="Home", path=str(home), kind=DirectoryEntryKind.directory))
    return DirectoryListing(host=host or host_name(), entries=entries)


def list_directory(
    path: str | None,
    *,
    include_files: bool = False,
    show_hidden: bool = False,
    host: str | None = None,
) -> DirectoryListing:
    """The children of `path` on this host, directories first.

    Raises `ListingError` with the status the route should answer: 404
    for a path that is not there, 400 for one that is not a directory,
    403 for one this process may not read.
    """
    if path is None or not path.strip():
        return starting_points(host=host)

    machine = host or host_name()
    target = Path(os.path.abspath(os.path.expanduser(path.strip())))
    try:
        if not target.exists():
            raise ListingError(404, "No such directory", f"{target} does not exist on {machine}.")
        if not target.is_dir():
            raise ListingError(400, "Not a directory", f"{target} is a file, not a directory.")
        with os.scandir(target) as scan:
            found = list(scan)
    except PermissionError as exc:
        raise ListingError(
            403, "Permission denied", f"{machine} may not read {target}: {exc}"
        ) from exc
    except OSError as exc:
        # A network path that is not there, a drive that went away: the
        # OS's own words are the useful part.
        raise ListingError(
            404, "Directory could not be read", f"{target} could not be read on {machine}: {exc}"
        ) from exc

    entries: list[DirectoryEntry] = []
    for entry in found:
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if not is_dir and not include_files:
            continue
        hidden = _is_hidden(entry)
        if hidden and not show_hidden:
            continue
        entries.append(
            DirectoryEntry(
                name=entry.name,
                path=str(target / entry.name),
                kind=DirectoryEntryKind.directory if is_dir else DirectoryEntryKind.file,
                hidden=hidden if show_hidden else None,
            )
        )
    entries.sort(key=lambda e: (e.kind is not DirectoryEntryKind.directory, e.name.casefold()))

    parent = target.parent
    return DirectoryListing(
        host=machine,
        path=str(target),
        parent=str(parent) if parent != target else None,
        entries=entries,
    )


__all__ = ["ListingError", "host_name", "list_directory", "starting_points"]
