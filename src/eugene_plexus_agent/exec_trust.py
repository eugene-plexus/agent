"""Which programs Eugene's own account runs (row 8, part 8c).

Run as an account of its own -- the Mac's LaunchDaemon (`_eugeneplexus`)
or a Linux system install (`eugene-plexus`) -- the agent holds the
machine's keys out of reach of the people who use the machine. Whatever
can change a program it runs can act as Eugene's account the next time
that program starts. So a program the agent runs -- an engine binary a
setting names (a runtime's `binary`, `vllmBinary`, `mlxBinary`,
`strataServer`, `kevPython`), one found on PATH, a managed build, the `uv`
it installs apps with, an app's interpreter -- is judged first, by
`assess`, the one place the decision is made.

What is walked: the path, from `/`, one name at a time with every link
followed by hand -- every folder stepped into on every hop, and the file;
a script's interpreter (`#!`, and the `exec` line uv writes under
`#!/bin/sh` for a path too long for a shebang) the same way, and every path
in what follows the interpreter, which it may run (`env` by any name, `sh`
with a script); and the Python environment the program belongs to (its
root, its `site-packages` folders, the interpreter it was made from, `home`
in `pyvenv.cfg`, with that interpreter's standard library).

What it decides (call R8-EXEC-TRUST, Choice 3):

- `ok`: every item belongs to root or to Eugene's account and nobody else
  can change it. It runs.
- `needs_approval`: every item belongs to root, Eugene's account or ONE
  other account, and no group or others can change any of them. It runs
  once an owner approves it (`exec_approvals`): the approval binds the
  path that runs, that account, the file's SHA-256 and every item walked
  (path, owner, mode), and lapses when any of them differ. An approval
  trusts that account: the hash catches a change, it is not a boundary
  against the account that owns the file.
- `refused`: a group or others can change an item (mode bits; on macOS an
  ACL entry that lets someone else write, append, delete, add or change
  it, or a volume that ignores ownership), two or more other accounts own
  items, a script lets PATH choose its interpreter (`env` under any name,
  a link, a copy or a hard link of it; a relative path; a bare program name
  after the interpreter), an ACL entry is not understood, or something
  cannot be checked or read (a script whose first line cannot be read, an
  environment's settings; any error but "no such file" is not absence).
  It never runs, and every approval its refusal touches is dropped.

An engine Eugene's own account starts gets a PATH of checked folders only
(`search_path`), never the one the agent inherited.

The walk returns the path to execute: absolute, every folder in it
resolved, and its last name kept when that name is a link (a venv's
`bin/python` is one, and Python finds its environment by the name it was
started as). That path is what the agent executes, a `--version` probe
included: a relative path or a bare name means nothing else when it runs.

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

import dataclasses
import hashlib
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
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
#: One entry as `ls -le` lists it, the principal's name whole.
_ACL_ENTRY = re.compile(
    r"^(?:\d+:\s*)?(user|group):(.+?)\s+(?:inherited\s+)?(allow|deny)\s+([a-z_]+(?:,[a-z_]+)*)$"
)
#: Programs that run whatever program PATH finds by the name they are given.
_PATH_RUNNERS = frozenset({"env"})
_LINK_HOPS = 40
_INTERPRETER_DEPTH = 4
#: The largest program hashed for an approval.
_HASH_LIMIT = 1 << 30
#: What a refusal offers besides copying the program: an engine's.
INSTALL_ENGINE = "let Eugene install the engine itself"


class Kind(StrEnum):
    """What `assess` decided; see the module's docstring."""

    ok = "ok"
    needs_approval = "needs_approval"
    refused = "refused"


@dataclass(frozen=True)
class ChainItem:
    """One file, folder or link the walk met, as it was then."""

    path: str
    uid: int
    mode: int


@dataclass(frozen=True)
class Finding:
    """One thing on a program's way that root and Eugene's account do not
    control alone. `problem`: `owner` (another account owns it), `others`
    (a group, others, an ACL entry or the volume let someone else change
    it), or `unknown` (it cannot be judged). `text` names the item and who
    can change it; `fix` picks the advice."""

    problem: str
    path: str
    uid: int | None
    text: str
    owner: str | None = None
    fix: str = "copy"


@dataclass(frozen=True)
class Verdict:
    """`assess`'s answer, with everything a caller acts on, so none of it is
    worked out twice: the kind, the path to execute, the one other account
    and the file's SHA-256 (`needs_approval`), and the items walked."""

    kind: Kind
    given: str
    program: str | None = None
    #: GIVEN made absolute, its links not followed: the path a setting
    #: looks the program up by. An approval is shown, made and kept by it,
    #: so a link in it (`current -> /opt/real`) stays part of what was
    #: approved (8c review).
    lookup: str | None = None
    account: str | None = None
    account_uid: int | None = None
    sha256: str | None = None
    chain: tuple[ChainItem, ...] = ()
    findings: tuple[Finding, ...] = ()
    home: str = "Eugene's folder"
    #: The file the program reached, every link followed, and its (device,
    #: inode): set even when the check stopped there, so a refusal can drop
    #: every approval that reaches the same file by another name.
    real: str | None = None
    real_id: tuple[int, int] | None = None

    def message(self, instead: str = INSTALL_ENGINE) -> str | None:
        """Why it did not run, naming the path, who can change it, and what
        to do. None for `ok`."""
        if self.kind is Kind.ok:
            return None
        lead = self.why()
        if self.kind is Kind.needs_approval:
            return (
                f"Eugene's own account runs {self.given} only once an owner approves it: "
                f"{lead}. Approve it, copy its folder into {self.home} as an administrator, "
                f"or {instead}."
            )
        fix = next((f.fix for f in self.findings if f.problem != "owner"), "copy")
        advice = {
            "shebang": "Name the interpreter by its full path on the script's first line, or "
            f"{instead}.",
            "check": f"Check the path, or {instead}.",
        }.get(fix, f"Copy its folder into {self.home} as an administrator, or {instead}.")
        return f"Eugene's own account will not run {self.given}: {lead}. {advice}"

    def why(self) -> str:
        """Who or what can change it, naming the item: the first such thing
        found, or the accounts when more than one owns a part of it."""
        others = [f for f in self.findings if f.problem != "owner"]
        if others:
            return others[0].text
        names = sorted({f.owner or f"user {f.uid}" for f in self.findings})
        if len(names) > 1:
            return f"more than one account can change it ({', '.join(names)})"
        return self.findings[0].text if self.findings else "it cannot be checked"


class Untrusted(Exception):
    """A program Eugene's own account will not run, and why."""

    def __init__(self, message: str, verdict: Verdict | None = None) -> None:
        super().__init__(message)
        self.verdict = verdict


# --- what the check reads, kept apart so a test can stand in for the disk ---

_lstat = os.lstat
_readlink = os.readlink
_listdir = os.listdir
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


def _head(path: PurePosixPath | Path) -> bytes:
    """The first bytes of a plain file. Opened without waiting, and refused
    when it is not a plain file, so a pipe a person put in a program's
    place cannot hold the agent."""
    fd = os.open(str(path), _OPEN_FLAGS)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"{path} is not a file")
        return os.read(fd, 4096)
    finally:
        os.close(fd)


def _sha256(path: PurePosixPath | Path) -> str:
    """The file's SHA-256, read every time it is asked (8c review): no
    answer is kept by size and times, because a file's change time moves on
    a coarse clock (a second on HFS+, a jiffy on Linux), so two writes can
    share one. Only a program that needs an approval is hashed."""
    fd = os.open(str(path), _OPEN_FLAGS)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(f"{path} is not a file")
        if info.st_size > _HASH_LIMIT:
            raise OSError(f"{path} is larger than {_HASH_LIMIT >> 30} GiB")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1 << 20):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


#: Where a system keeps `env`: one followed link apart on usr-merged Linux.
_ENV_PATHS = ("/usr/bin/env", "/bin/env")


def _system_env() -> list[tuple[int, int, int, str]]:
    """The system's `env`, by what identifies it whatever it is called:
    (device, inode, size, SHA-256) of each file the usual paths reach."""
    found: list[tuple[int, int, int, str]] = []
    for spelled in _ENV_PATHS:
        try:
            info = os.stat(spelled)
        except OSError:
            continue
        if any((info.st_dev, info.st_ino) == (dev, ino) for dev, ino, _, _ in found):
            continue
        try:
            digest = _sha256(PurePosixPath(spelled))
        except OSError:
            digest = ""
        found.append((info.st_dev, info.st_ino, info.st_size, digest))
    return found


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


def _account_home(uid: int) -> str | None:
    """Eugene's folder: its account's home, which the installers make the
    install's own folder."""
    if sys.platform == "win32":
        return None
    import pwd

    try:
        return str(pwd.getpwuid(uid).pw_dir)
    except KeyError:
        return None


def _group_name(gid: int) -> str | None:
    if sys.platform == "win32":
        return None
    import grp

    try:
        return str(grp.getgrgid(gid).gr_name)
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


#: macOS volume options, by device: what a file's `st_dev` names, read when
#: a device is first seen and again after a minute.
_volumes: dict[int, tuple[float, frozenset[str]]] = {}


def _mount_options(path: Path, dev: int) -> frozenset[str]:
    """macOS: the options of the volume PATH is on. `df` names the volume's
    own mount point (a firmlinked `/usr/local` is on the Data volume, not
    `/`), and `mount` its options; the answer is kept by device, so a volume
    mounted a moment ago is read the first time it is met."""
    now = time.perf_counter()
    seen = _volumes.get(dev)
    if seen is not None and now - seen[0] <= 60:
        return seen[1]
    df = subprocess.run(
        ["/bin/df", "-P", str(path)], capture_output=True, text=True, timeout=10, check=False
    )
    lines = df.stdout.splitlines()
    fields = lines[1].split(None, 5) if len(lines) > 1 else []
    if len(fields) < 6:
        raise OSError(f"df named no volume for {path}")
    point = fields[5]
    table = subprocess.run(
        ["/sbin/mount"], capture_output=True, text=True, timeout=10, check=False
    ).stdout
    options: frozenset[str] = frozenset()
    for line in table.splitlines():
        found = re.match(r"^.+? on (.+) \((.*)\)$", line)
        if found and found.group(1) == point:
            options = frozenset(o.strip() for o in found.group(2).split(","))
    if len(_volumes) > 256:
        _volumes.clear()
    _volumes[dev] = (now, options)
    return options


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


def _owner_name(uid: int) -> str:
    return _account_name(uid) or f"user {uid}"


class _Walk:
    """One check: every item it met, and what it found. It goes on past an
    item another account owns, so the verdict knows whether anything else
    could change the program too."""

    def __init__(self, uid: int) -> None:
        self.uid = uid
        self.me = _account_name(uid)
        self.chain: list[ChainItem] = []
        self.findings: list[Finding] = []
        #: The file the program reached (path, device, inode), once known.
        self.real: tuple[str, int, int] | None = None

    def find(
        self,
        problem: str,
        path: PurePosixPath | str,
        text: str,
        *,
        uid: int | None = None,
        fix: str = "copy",
    ) -> None:
        owner = _owner_name(uid) if uid is not None else None
        self.findings.append(Finding(problem, str(path), uid, text, owner, fix))


def _check(w: _Walk, path: PurePosixPath, info: os.stat_result, role: str, final: bool) -> None:
    """One file or folder on the way."""
    w.chain.append(ChainItem(str(path), info.st_uid, info.st_mode))
    owner = _owner_name(info.st_uid)

    def said(what: str) -> str:
        return f"{role} {what}" if final else f"{role} is reached through {path}, which {what}"

    if info.st_uid not in (0, w.uid):
        w.find("owner", path, said(f"belongs to {owner}"), uid=info.st_uid)
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        if info.st_mode & stat.S_IWOTH:
            who = "anyone"
        else:
            gid = getattr(info, "st_gid", 0)
            who = f"its owner {owner} and group {_group_name(gid) or gid}"
        mode = stat.S_IMODE(info.st_mode)
        w.find(
            "others",
            path,
            said(f"can be changed by others (mode {mode:04o}: {who})"),
            uid=info.st_uid,
        )
    if _platform() != "darwin":
        return
    try:
        options = _mount_options(Path(str(path)), info.st_dev)
    except OSError as e:
        w.find("unknown", path, said(f"is on a volume that cannot be read ({e})"), fix="check")
        return
    if "noowners" in options:
        w.find(
            "others",
            path,
            said("is on a volume that ignores ownership, so anyone can change it"),
            uid=info.st_uid,
        )
        return
    try:
        entries = _acl_cached(path, info)
    except OSError as e:
        w.find("unknown", path, said(f"has an ACL that cannot be read ({e})"), fix="check")
        return
    for entry in entries:
        # `N: user:<name> [inherited] allow|deny <rights>`; a name may hold
        # spaces (`group:Domain Users`). An entry not in that form is not
        # understood, so it is refused rather than passed over (8c review).
        found = _ACL_ENTRY.match(entry)
        if not found:
            w.find(
                "others",
                path,
                said(f"carries an ACL entry Eugene cannot read ({entry})"),
                uid=info.st_uid,
            )
            continue
        kind, who, verdict, rights = found.groups()
        if verdict == "deny":
            continue
        # Root, Eugene's account, and the item's own owner, who can change
        # it anyway (and was found above when it is someone else).
        if kind == "user" and who in ("root", w.me, owner):
            continue
        granted = set(rights.split(",")) & _MUTATING
        if granted:
            w.find(
                "others",
                path,
                said(f"carries an ACL entry that lets {kind} {who} {', '.join(sorted(granted))}"),
                uid=info.st_uid,
            )


def _absolute(given: str) -> str:
    spelled = os.path.expanduser(given)
    if not spelled.startswith("/"):
        spelled = f"{_cwd().rstrip('/')}/{spelled}"
    return spelled


def _walk(
    w: _Walk, given: str, role: str
) -> tuple[PurePosixPath, PurePosixPath, os.stat_result] | None:
    """GIVEN walked by hand from `/`: every link followed, every folder
    stepped into on every hop checked, and the last item. Returns the path
    to execute, the file it reaches with no link left, and what that file
    is; or None when the walk could not finish (a finding says why)."""
    pending = list(PurePosixPath(_absolute(given)).parts[1:])
    # The names still to walk from the path as given; those a link's
    # target adds go in front of them.
    given_left = len(pending)
    current = PurePosixPath("/")
    run_as: PurePosixPath | None = None
    try:
        info = _lstat(current)
    except OSError as e:
        w.find("unknown", current, f"{role} cannot be checked ({e.strerror or e})", fix="check")
        return None
    _check(w, current, info, role, final=not pending)
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
            found = _lstat(step)
        except OSError as e:
            w.find(
                "unknown",
                step,
                f"{role} cannot be checked at {step} ({e.strerror or e})",
                fix="check",
            )
            return None
        if stat.S_ISLNK(found.st_mode):
            hops += 1
            if hops > _LINK_HOPS:
                w.find("unknown", step, f"{role} goes through too many links", fix="check")
                return None
            if last_given:
                run_as = step
            # The link sits in `current`, which was checked: whoever can
            # change where it points was found there.
            w.chain.append(ChainItem(str(step), found.st_uid, found.st_mode))
            try:
                target = PurePosixPath(_readlink(step))
            except OSError as e:
                w.find("unknown", step, f"{role} cannot be read at {step} ({e})", fix="check")
                return None
            if target.is_absolute():
                current = PurePosixPath("/")
                pending = list(target.parts[1:]) + pending
            else:
                pending = list(target.parts) + pending
            continue
        _check(w, step, found, role, final=not pending)
        current, info = step, found
    return run_as or current, current, info


def _interpreters(w: _Walk, path: PurePosixPath, role: str) -> tuple[list[str], list[str]]:
    """What runs a script, or nothing for a binary: the program `#!` names,
    and under `#!/bin/sh` the python uv's `'''exec' "<python>" "$0" "$@"`
    line names. Each must be an absolute path (a bare name is found from
    the folder the script starts in, or by PATH, not where this looks), and
    `env` is refused (PATH would choose). A file it cannot read is refused:
    what runs it is not known (8c review).

    Also the paths in what follows the interpreter, which it may run (`env`
    by any name, `sh` with a script, `nice`, `timeout`): every word with a
    `/` in it, from its first `/` (so `--file=/x` and `-f/x` too), to be
    checked like a program. The kernel hands Linux's interpreter all of it
    as one argument and macOS's as words; the paths are the same either
    way. A word with no `/` is an option, or a bare name PATH would find,
    which is refused (8c re-reviews)."""
    try:
        head = _head(path)
    except OSError as e:
        w.find(
            "unknown",
            path,
            f"{role} cannot be read to find what runs it ({e.strerror or e})",
            fix="check",
        )
        return [], []
    if not head.startswith(b"#!"):
        return [], []
    lines = head.decode("utf-8", errors="replace").splitlines()
    words = lines[0][2:].split()
    if not words:
        return [], []
    if PurePosixPath(words[0]).name in _PATH_RUNNERS:
        w.find(
            "unknown",
            path,
            f"{role} names its interpreter through PATH (#!{' '.join(words)}), so what "
            "would run is not known",
            fix="shebang",
        )
        return [], []
    arguments: list[str] = []
    for word in words[1:]:
        if "/" in word:
            arguments.append(word[word.index("/") :])
        elif not word.startswith("-"):
            w.find(
                "unknown",
                path,
                f"{role} names a program by a bare name after its interpreter ({word}), which "
                "PATH would find",
                fix="shebang",
            )
    found = [words[0]]
    if len(lines) > 1:
        trampoline = re.match(r"""^'''exec' "(.+?)" "\$0" "\$@"$""", lines[1].strip())
        if trampoline:
            found.append(trampoline.group(1))
    absolute = []
    for interpreter in found:
        if interpreter.startswith("/"):
            absolute.append(interpreter)
            continue
        shown = f"#!{' '.join(words)}" if interpreter == words[0] else interpreter
        w.find(
            "unknown",
            path,
            f"{role} names its interpreter by a relative path ({shown}), which the folder it "
            "starts in or PATH would choose",
            fix="shebang",
        )
    return absolute, arguments


def _exists(path: PurePosixPath, w: _Walk | None = None, role: str = "it") -> bool:
    """Something is at PATH (a link counts). Only "no such file" and "not a
    folder" mean absent (8c re-review): any other error is not known to be
    absence, so on a walk it is a finding (refused, naming the cause) and
    the branch stops; asked alone, it counts as present, so what follows
    checks it and fails closed."""
    try:
        _lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as e:
        if w is None:
            return True
        w.find(
            "unknown", path, f"{role} cannot be checked at {path} ({e.strerror or e})", fix="check"
        )
        return False
    return True


def present(path: str) -> bool:
    """Something is at PATH, a POSIX path (a link counts), as the check
    reads the disk; an error other than absence counts as present."""
    return _exists(PurePosixPath(path))


def _is_system_env(path: PurePosixPath, info: os.stat_result) -> bool:
    """PATH is the system's `env` under another name: the same file (its
    device and inode, a hard link) or the same bytes (a copy)."""
    for dev, ino, size, digest in _system_env():
        if (info.st_dev, info.st_ino) == (dev, ino):
            return True
        if digest and getattr(info, "st_size", None) == size and _sha256(path) == digest:
            return True
    return False


def _environment(w: _Walk, root: PurePosixPath) -> None:
    """The Python environment at ROOT (it has a `pyvenv.cfg`): the root,
    its site-packages, and the interpreter it was made from with its
    standard library."""
    role = f"its Python environment {root}"
    walked = _walk(w, str(root), role)
    if walked is None:
        return
    root = walked[0]
    for lib in ("lib", "lib64"):
        try:
            names = sorted(_listdir(root / lib))
        except FileNotFoundError:
            continue
        except OSError as e:
            # What is in it cannot be known (8c review): refused.
            w.find(
                "unknown", root / lib, f"{role}: {root / lib} cannot be listed ({e})", fix="check"
            )
            continue
        for name in names:
            packages = root / lib / name / "site-packages"
            if name.startswith("python") and _exists(packages, w, role):
                _walk(w, str(packages), f"its packages {packages}")
    try:
        text = _head(root / "pyvenv.cfg").decode("utf-8", errors="replace")
    except OSError as e:
        w.find(
            "unknown",
            root / "pyvenv.cfg",
            f"{role}: {root / 'pyvenv.cfg'} cannot be read ({e})",
            fix="check",
        )
        return
    home = re.search(r"(?m)^home\s*=\s*(.+?)\s*$", text)
    if home:
        base = _walk(w, home.group(1), f"the Python it was made from ({home.group(1)})")
        if base is not None and _exists(base[0].parent / "lib", w, role):
            lib_dir = base[0].parent / "lib"
            _walk(w, str(lib_dir), f"that Python's standard library ({lib_dir})")


def _program(
    w: _Walk, given: str, depth: int, role: str, *, argument: bool = False
) -> PurePosixPath | None:
    """GIVEN, as a program: the walk, then what runs it. ARGUMENT: a path an
    interpreter was given, which may also be a folder (`-I/usr/share/perl`)."""
    walked = _walk(w, given, role)
    if walked is None:
        return None
    resolved, real, info = walked
    if depth == 0:
        # The file it reaches, whatever it was looked up by and whatever
        # the check finds next: a refusal drops every approval that reaches
        # it (8c re-review 2).
        w.real = (str(real), info.st_dev, info.st_ino)
    if argument and stat.S_ISDIR(info.st_mode):
        return resolved
    if not stat.S_ISREG(info.st_mode):
        w.find("unknown", resolved, f"{role} is not a file", fix="check")
        return None
    if real.name in _PATH_RUNNERS:
        # `env` by another name runs whatever PATH finds (8c review).
        w.find(
            "unknown",
            resolved,
            f"{role} is {real.name}, which lets PATH choose the program",
            fix="shebang" if depth else "check",
        )
        return None
    try:
        system_env = _is_system_env(real, info)
    except OSError as e:
        w.find("unknown", resolved, f"{role} cannot be read ({e.strerror or e})", fix="check")
        return None
    if system_env:
        # A copy or a hard link of `env` (8c re-review).
        w.find(
            "unknown",
            resolved,
            f"{role} is the system's env under another name, which lets PATH choose the program",
            fix="shebang" if depth else "check",
        )
        return None
    interpreters, arguments = _interpreters(w, resolved, role)
    if (interpreters or arguments) and depth >= _INTERPRETER_DEPTH:
        w.find("unknown", resolved, f"{role} goes through too many interpreters", fix="check")
        interpreters, arguments = [], []
    for interpreter in interpreters:
        _program(w, interpreter, depth + 1, f"its interpreter {interpreter}")
    # What follows the interpreter may be a program it runs: each path in
    # it is checked like one, and the verdict is the worst of them all.
    for path in arguments:
        _program(w, path, depth + 1, f"its interpreter's argument {path}", argument=True)
    # The environment by the path as spelled (a venv's `bin/python` is a
    # link out of it) and by the path it resolves to.
    for program in (PurePosixPath(_absolute(given)), resolved):
        if _exists(program.parent.parent / "pyvenv.cfg", w, role):
            _environment(w, program.parent.parent)
    return resolved


def _decide(w: _Walk, given: str, resolved: PurePosixPath | None, *, hashed: bool) -> Verdict:
    """The kind, from what the walk found."""
    home = _account_home(w.uid) or "Eugene's folder"
    program = str(resolved) if resolved is not None else None
    if resolved is not None and not w.findings:
        return Verdict(Kind.ok, given, program, chain=tuple(w.chain), home=home)
    owners = {f.uid for f in w.findings if f.problem == "owner"}
    single = (
        resolved is not None and len(owners) == 1 and all(f.problem == "owner" for f in w.findings)
    )
    digest: str | None = None
    if single and hashed:
        assert resolved is not None
        try:
            digest = _sha256(resolved)
        except OSError as e:
            w.find("unknown", resolved, f"it cannot be read to be checked ({e})", fix="check")
            single = False
    if not single:
        return Verdict(
            Kind.refused,
            given,
            program,
            chain=tuple(w.chain),
            findings=tuple(w.findings),
            home=home,
        )
    (uid,) = owners
    return Verdict(
        Kind.needs_approval,
        given,
        program,
        account=_owner_name(uid) if uid is not None else None,
        account_uid=uid,
        sha256=digest,
        chain=tuple(w.chain),
        findings=tuple(w.findings),
        home=home,
    )


def assess(path: Path | str) -> Verdict:
    """The one decision: whether Eugene's own account may run PATH, and the
    path to execute. Outside Eugene's own account, `ok` with PATH as given."""
    given = str(path)
    if not own_account():
        return Verdict(Kind.ok, given, given)
    uid = _euid()
    assert uid is not None
    w = _Walk(uid)
    verdict = _decide(w, given, _program(w, given, 0, "it"), hashed=True)
    return dataclasses.replace(
        verdict,
        lookup=_absolute(given),
        real=w.real[0] if w.real else None,
        real_id=(w.real[1], w.real[2]) if w.real else None,
    )


def file_id(path: str) -> tuple[int, int] | None:
    """PATH's (device, inode), as the check reads the disk, or None."""
    try:
        info = _lstat(PurePosixPath(path))
    except OSError:
        return None
    return info.st_dev, info.st_ino


def assess_path(path: Path | str) -> Verdict:
    """`assess` for a file the agent keeps rather than runs (the approvals
    file): the walk alone, nothing hashed."""
    given = str(path)
    if not own_account():
        return Verdict(Kind.ok, given, given)
    uid = _euid()
    assert uid is not None
    w = _Walk(uid)
    walked = _walk(w, given, "it")
    return _decide(w, given, walked[0] if walked else None, hashed=False)


#: The folders a program Eugene's own account starts may find programs in,
#: before each is checked (sudo's `secure_path`).
_SYSTEM_PATH = ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin")


def search_path(configured: str | None = None) -> str | None:
    """PATH for an engine Eugene's own account starts, never the one it
    inherited (8c re-review): CONFIGURED (a runtime's own `PATH`) and then
    the system's folders, each kept only when the check finds it `ok` (root
    or Eugene's account alone can change it). Outside Eugene's own account,
    None: the process keeps the PATH it was given.

    Never empty, nor with an empty folder in it: either is a search of the
    folder the engine starts in. When no folder passes, Untrusted, naming
    the first folder's cause, and the engine is not started (8c re-review
    2)."""
    if not own_account():
        return None
    kept: list[str] = []
    first: str | None = None
    for folder in [*(configured.split(":") if configured else []), *_SYSTEM_PATH]:
        if not folder.startswith("/") or folder in kept:
            continue
        verdict = assess_path(folder)
        if verdict.kind is Kind.ok:
            kept.append(folder)
        elif first is None:
            first = f"{folder}: {verdict.why()}"
    if not kept:
        raise Untrusted(
            "Eugene's own account starts no engine without a folder it may find programs in, "
            f"and no folder to find programs in passed the check ({first})"
        )
    return ":".join(kept)


def runs(verdict: Verdict) -> bool:
    """The rule: `ok` runs; `needs_approval` runs while an owner's approval
    of exactly this holds; `refused` never runs, and drops any approval of
    the program, so putting things back afterwards does not revive it (8c
    review)."""
    if verdict.kind is Kind.ok:
        return True
    from . import exec_approvals

    if verdict.kind is Kind.needs_approval:
        return exec_approvals.holds(verdict)
    exec_approvals.forget(verdict)
    return False


def trusted_program(path: Path | str, *, instead: str = INSTALL_ENGINE) -> Path:
    """The path to execute PATH by, when Eugene's own account may run it.
    Raises Untrusted, carrying the verdict, otherwise. INSTEAD is what the
    refusal offers besides copying the program."""
    verdict = assess(path)
    if runs(verdict) and verdict.program is not None:
        return Path(verdict.program)
    raise Untrusted(verdict.message(instead) or f"{path} is not run", verdict)


def untrusted(path: Path | str, *, instead: str = INSTALL_ENGINE) -> str | None:
    """Why Eugene's own account must not run PATH now, or None when it may."""
    verdict = assess(path)
    return None if runs(verdict) else verdict.message(instead)
