"""Which programs Eugene's own account runs (row 8).

Run as an account of its own -- the Mac's LaunchDaemon (`_eugeneplexus`)
or a Linux system install (`eugene-plexus`) -- the agent holds the
machine's keys out of reach of the people who use the machine. A program
it runs must then be one nobody else can change: otherwise whatever runs as
the person who owns it runs as Eugene's account the next time an engine
starts. So an executable a setting names (a runtime's `binary`,
`vllmBinary`, `mlxBinary`, `strataServer`, `kevPython`, `uvBinary`), and one
found on PATH, runs only when the file and every folder above it belong to
root or to Eugene's account and none of them can be written by its group or
by others. Checked before anything is executed, a `--version` probe
included, on the path as given and on the path its links resolve to.

A person's own install (the account is the person's) is not limited: the
person can change their own files anyway. Windows is not covered: its
service accounts are guarded by ACLs, which this does not read. Neither are
macOS ACL entries on those folders.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

#: The accounts the installers make for Eugene: the Mac's daemon, and a
#: Linux system install.
SERVICE_ACCOUNTS = frozenset({"_eugeneplexus", "eugene-plexus"})

#: What the check reads, kept apart so a test can stand in for the disk.
_lstat = os.lstat
_realpath = os.path.realpath


def _euid() -> int | None:
    geteuid = getattr(os, "geteuid", None)
    return geteuid() if geteuid is not None else None


def _account_name(uid: int) -> str | None:
    if sys.platform == "win32":
        return None
    import pwd

    try:
        return str(pwd.getpwuid(uid).pw_name)
    except KeyError:
        return None


def own_account() -> bool:
    """This agent runs as an account the installer made for Eugene."""
    uid = _euid()
    return uid is not None and _account_name(uid) in SERVICE_ACCOUNTS


def untrusted(path: Path | str) -> str | None:
    """Why Eugene's own account must not run PATH, in words a person acts
    on, or None when it may (or when this agent is not in its own account).
    """
    if not own_account():
        return None
    uid = _euid()
    given = Path(os.path.abspath(os.path.expanduser(str(path))))
    real = Path(_realpath(given))
    # The file and the folders its links resolve to; and the folders the
    # path as given passes through, which hold its links (a link's own
    # owner and mode do not matter: the folder it is in decides who can
    # change it).
    parts = [real, *real.parents]
    parts += [p for p in given.parents if p not in parts]
    for part in parts:
        try:
            info = _lstat(part)
        except OSError as e:
            return (
                f"{path} cannot be checked ({e.strerror or e}), so Eugene's own account "
                "does not run it"
            )
        if stat.S_ISLNK(info.st_mode):
            continue
        if info.st_uid not in (0, uid):
            owner = _account_name(info.st_uid) or f"user {info.st_uid}"
            where = f"{path} belongs" if part == real else f"{path} is in {part}, which belongs"
            return (
                f"{where} to {owner}; Eugene's own account runs programs only from folders it "
                "or root owns. Move it, or let Eugene fetch its own build"
            )
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            what = path if part == real else part
            return (
                f"{what} can be changed by others (mode {stat.S_IMODE(info.st_mode):04o}); "
                f"Eugene's own account runs {path} only from folders nobody else can change. "
                "Move it, or let Eugene fetch its own build"
            )
    return None
