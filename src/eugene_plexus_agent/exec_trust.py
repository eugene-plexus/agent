"""Which programs Eugene's own account runs (row 8).

Run as an account of its own -- the Mac's LaunchDaemon (`_eugeneplexus`)
or a Linux system install (`eugene-plexus`) -- the agent holds the
machine's keys out of reach of the people who use the machine. A program
it runs must then be one nobody else can change: otherwise whatever runs as
the person who could change it runs as Eugene's account the next time it
starts. So a program the agent runs -- an engine binary a setting names (a
runtime's `binary`, `vllmBinary`, `mlxBinary`, `strataServer`,
`kevPython`), one found on PATH, a managed build, the `uv` it installs apps
with, an app's interpreter -- runs only when:

- the path, walked from `/` one name at a time with every link followed by
  hand, steps only into folders, on every hop, and ends at a file, that
  belong to root or to Eugene's account, can be written by neither their
  group nor others, and (on macOS) carry no ACL entry that lets anyone
  else write, append, delete, add or change them, on a volume that does
  not ignore ownership;
- a script's interpreter (`#!`, and the `exec` line uv writes under
  `#!/bin/sh` for a path too long for a shebang) passes the same walk;
  `#!/usr/bin/env` is refused, because PATH, not the script, would choose
  the program;
- a Python environment the program belongs to passes too: its root, its
  `site-packages` folders, and the interpreter it was made from (`home` in
  `pyvenv.cfg`) with that interpreter's standard library.

The walk returns the resolved absolute path, with no link left in it, and
that path is what the agent executes, a `--version` probe included: a
relative path or a bare name means nothing else at the time it runs.

Not covered, and why: the files inside a trusted folder are not each
checked (a folder only root or Eugene's account can change holds only what
they put there); Windows (its service accounts are guarded by ACLs, which
this does not read); what a program the agent runs starts in turn (an
app's own children, a tool server's) -- those run inside the app's own
environment, checked when the app starts. A person's own install (the
account is the person's) is not limited: they can change their own files
anyway.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

#: The accounts the installers make for Eugene: the Mac's daemon, and a
#: Linux system install.
SERVICE_ACCOUNTS = frozenset({"_eugeneplexus", "eugene-plexus"})

#: macOS ACL rights that let a principal change what runs.
_MUTATING = frozenset(
    {
        "write",
        "append",
        "delete",
        "add_file",
        "add_subdirectory",
        "delete_child",
        "writeattr",
        "writeextattr",
        "writesecurity",
        "chown",
    }
)
_LINK_HOPS = 40
_INTERPRETER_DEPTH = 4
_ADVICE = "Move it, or let Eugene fetch its own build"


class Untrusted(Exception):
    """A program Eugene's own account will not run, and why."""


# --- what the check reads, kept apart so a test can stand in for the disk ---

_lstat = os.lstat
_readlink = os.readlink
_listdir = os.listdir


def _head(path: PurePosixPath | Path) -> bytes:
    with open(str(path), "rb") as handle:
        return handle.read(4096)


def _platform() -> str:
    return sys.platform


def _euid() -> int | None:
    geteuid = getattr(os, "geteuid", None)
    return geteuid() if geteuid is not None else None


def _cwd() -> str:
    return os.getcwd()


def _account_name(uid: int) -> str | None:
    if sys.platform == "win32":
        return None
    import pwd

    try:
        return str(pwd.getpwuid(uid).pw_name)
    except KeyError:
        return None


def _acl_entries(path: Path) -> list[str]:
    """macOS: the ACL entries `ls -le` lists under PATH, by name."""
    done = subprocess.run(
        ["/bin/ls", "-led", str(path)], capture_output=True, text=True, timeout=10, check=False
    )
    if done.returncode != 0:
        raise OSError(done.stderr.strip() or f"ls -le {path} failed")
    return [line.strip() for line in done.stdout.splitlines()[1:] if line.strip()]


_mounts: tuple[float, list[tuple[str, frozenset[str]]]] | None = None


def _mount_options(path: Path) -> frozenset[str]:
    """macOS: the options of the volume PATH is on (`mount`), read at most
    once a minute."""
    global _mounts
    now = time.perf_counter()
    if _mounts is None or now - _mounts[0] > 60:
        done = subprocess.run(
            ["/sbin/mount"], capture_output=True, text=True, timeout=10, check=False
        )
        table = []
        for line in done.stdout.splitlines():
            found = re.match(r"^.+? on (.+) \((.*)\)$", line)
            if found:
                options = frozenset(o.strip() for o in found.group(2).split(","))
                table.append((found.group(1), options))
        _mounts = (now, table)
    best: tuple[str, frozenset[str]] | None = None
    text = str(path)
    for point, options in _mounts[1]:
        inside = point == "/" or text == point or text.startswith(point.rstrip("/") + "/")
        if inside and (best is None or len(point) > len(best[0])):
            best = (point, options)
    return best[1] if best else frozenset()


#: macOS ACL answers, by what identifies a file's metadata: a change of
#: owner, mode or ACL changes its ctime, so a stale answer is never used.
_acl_seen: dict[tuple[str, int, int, int], list[str]] = {}


def _acl_cached(path: PurePosixPath, info: os.stat_result) -> list[str]:
    key = (str(path), info.st_dev, info.st_ino, info.st_ctime_ns)
    found = _acl_seen.get(key)
    if found is None:
        found = _acl_entries(Path(str(path)))
        if len(_acl_seen) > 4096:
            _acl_seen.clear()
        _acl_seen[key] = found
    return found


# --- the check ----------------------------------------------------------------


def own_account() -> bool:
    """This agent runs as an account the installer made for Eugene."""
    uid = _euid()
    return uid is not None and _account_name(uid) in SERVICE_ACCOUNTS


def _check(path: PurePosixPath, info: os.stat_result, uid: int, given: str) -> None:
    """One file or folder on the walk: raise Untrusted when anyone but root
    or Eugene's account could change it."""
    if info.st_uid not in (0, uid):
        owner = _account_name(info.st_uid) or f"user {info.st_uid}"
        where = f"{given} belongs" if str(path) == given else f"{given} is in {path}, which belongs"
        raise Untrusted(
            f"{where} to {owner}; Eugene's own account runs programs only from folders it "
            f"or root owns. {_ADVICE}"
        )
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise Untrusted(
            f"{path} can be changed by others (mode {stat.S_IMODE(info.st_mode):04o}); Eugene's "
            f"own account runs {given} only from folders nobody else can change. {_ADVICE}"
        )
    if _platform() != "darwin":
        return
    if "noowners" in _mount_options(Path(str(path))):
        raise Untrusted(
            f"{path} is on a volume that ignores ownership, so nobody can tell who may change "
            f"{given}; Eugene's own account does not run it. {_ADVICE}"
        )
    try:
        entries = _acl_cached(path, info)
    except OSError as e:
        raise Untrusted(f"{path}'s ACL cannot be read ({e}), so {given} is not run") from e
    me = _account_name(uid)
    for entry in entries:
        found = re.match(r"^(?:\d+:\s*)?(user|group):(\S+)\s+(?:inherited\s+)?allow\s+(\S+)", entry)
        if not found:
            continue
        kind, who, rights = found.groups()
        if kind == "user" and who in ("root", me):
            continue
        granted = set(rights.split(",")) & _MUTATING
        if granted:
            raise Untrusted(
                f"{path} carries an ACL entry that lets {kind} {who} "
                f"{', '.join(sorted(granted))}; Eugene's own account runs {given} only from "
                f"folders nobody else can change. {_ADVICE}"
            )


def _absolute(given: str) -> str:
    spelled = os.path.expanduser(given)
    if not spelled.startswith("/"):
        spelled = f"{_cwd().rstrip('/')}/{spelled}"
    return spelled


def trusted_path(path: Path | str) -> Path:
    """PATH walked by hand from `/` -- every link followed, every folder
    stepped into on every hop checked, and the file itself. Raises
    Untrusted. Returns the path to execute: absolute, every folder in it
    resolved, and its last name kept when that name is a link (a venv's
    `bin/python` is one, and Python finds its environment by the name it
    was started as). Such a link sits in a folder that was checked, and
    where it points was walked, so nobody else can change what runs.

    Outside Eugene's own account, PATH as it was given."""
    if not own_account():
        return Path(path)
    return Path(str(_walk(str(path))))


def _walk(given: str) -> PurePosixPath:
    """`trusted_path`'s walk, on POSIX paths whatever the platform."""
    uid = _euid()
    assert uid is not None
    pending = list(PurePosixPath(_absolute(given)).parts[1:])
    # The names still to walk from the path as given; those a link's
    # target adds go in front of them.
    given_left = len(pending)
    current = PurePosixPath("/")
    run_as: PurePosixPath | None = None
    try:
        _check(current, _lstat(current), uid, given)
    except OSError as e:
        raise Untrusted(f"{given} cannot be checked ({e.strerror or e}), so it is not run") from e
    hops = 0
    while pending:
        last_given = len(pending) == given_left and given_left == 1
        if len(pending) == given_left:
            given_left -= 1
        name = pending.pop(0)
        if name in ("", "."):
            continue
        if name == "..":
            current = current.parent  # an ancestor: checked on the way in
            continue
        step = current / name
        try:
            info = _lstat(step)
        except OSError as e:
            raise Untrusted(
                f"{given} cannot be checked at {step} ({e.strerror or e}), so it is not run"
            ) from e
        if stat.S_ISLNK(info.st_mode):
            hops += 1
            if hops > _LINK_HOPS:
                raise Untrusted(f"{given} goes through too many links, so it is not run")
            if last_given:
                run_as = step
            # The link sits in `current`, which was checked: only root or
            # Eugene's account can change where it points.
            target = PurePosixPath(_readlink(step))
            if target.is_absolute():
                current = PurePosixPath("/")
                pending = list(target.parts[1:]) + pending
            else:
                pending = list(target.parts) + pending
            continue
        _check(step, info, uid, given)
        current = step
    return run_as or current


def _interpreters(path: PurePosixPath) -> list[str] | None:
    """What runs a script, or None for a binary: the program `#!` names
    (refused when it is `env`), and under `#!/bin/sh` the python uv's
    `'''exec' "<python>" "$0" "$@"` line names."""
    try:
        head = _head(path)
    except OSError:
        return None
    if not head.startswith(b"#!"):
        return None
    lines = head.decode("utf-8", errors="replace").splitlines()
    words = lines[0][2:].split()
    if not words:
        return None
    if PurePosixPath(words[0]).name == "env":
        raise Untrusted(
            f"{path} names its interpreter through PATH (#!{' '.join(words)}), so the program "
            f"that would run is not known; Eugene's own account does not run it. {_ADVICE}"
        )
    found = [words[0]]
    if len(lines) > 1:
        trampoline = re.match(r"""^'''exec' "(.+?)" "\$0" "\$@"$""", lines[1].strip())
        if trampoline:
            found.append(trampoline.group(1))
    return found


def _exists(path: PurePosixPath) -> bool:
    try:
        _lstat(path)
    except OSError:
        return False
    return True


def _environment(root: PurePosixPath) -> None:
    """The Python environment at ROOT (it has a `pyvenv.cfg`): the root,
    its site-packages, and the interpreter it was made from with its
    standard library."""
    root = _walk(str(root))
    for lib in ("lib", "lib64"):
        try:
            names = _listdir(root / lib)
        except OSError:
            continue
        for name in names:
            packages = root / lib / name / "site-packages"
            if name.startswith("python") and _exists(packages):
                _walk(str(packages))
    try:
        text = _head(root / "pyvenv.cfg").decode("utf-8", errors="replace")
    except OSError:
        return
    home = re.search(r"(?m)^home\s*=\s*(.+?)\s*$", text)
    if home:
        base = _walk(home.group(1))
        if _exists(base.parent / "lib"):
            _walk(str(base.parent / "lib"))


def trusted_program(path: Path | str, *, _depth: int = 0) -> Path:
    """`trusted_path`, then what runs a script, and the Python environment
    the program belongs to. Returns the resolved path to execute. Raises
    Untrusted. Outside Eugene's own account, PATH as it was given."""
    if not own_account():
        return Path(path)
    return Path(str(_program(str(path), _depth)))


def _program(given: str, depth: int) -> PurePosixPath:
    resolved = _walk(given)
    interpreters = _interpreters(resolved)
    if interpreters:
        if depth >= _INTERPRETER_DEPTH:
            raise Untrusted(f"{given} goes through too many interpreters, so it is not run")
        for interpreter in interpreters:
            _program(interpreter, depth + 1)
    # The environment by the path as spelled (a venv's `bin/python` is a
    # link out of it) and by the path it resolves to.
    for program in (PurePosixPath(_absolute(given)), resolved):
        if _exists(program.parent.parent / "pyvenv.cfg"):
            _environment(program.parent.parent)
    return resolved


def untrusted(path: Path | str) -> str | None:
    """Why Eugene's own account must not run PATH, or None when it may."""
    try:
        trusted_program(path)
    except Untrusted as e:
        return str(e)
    return None
