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
`#!/bin/sh` for a path too long for a shebang) the same way; and the Python
environment the program belongs to (its root, its `site-packages` folders,
the interpreter it was made from, `home` in `pyvenv.cfg`, with that
interpreter's standard library).

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
  items, a script lets PATH choose its interpreter (`#!/usr/bin/env`), or
  something cannot be checked. It never runs.

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
    account: str | None = None
    account_uid: int | None = None
    sha256: str | None = None
    chain: tuple[ChainItem, ...] = ()
    findings: tuple[Finding, ...] = ()
    home: str = "Eugene's folder"

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


#: SHA-256 answers, by what identifies a file's bytes: a write changes its
#: ctime, which its owner cannot set back.
_hashes: dict[tuple[str, int, int, int, int, int], str] = {}


def _sha256(path: PurePosixPath | Path) -> str:
    fd = os.open(str(path), _OPEN_FLAGS)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(f"{path} is not a file")
        if info.st_size > _HASH_LIMIT:
            raise OSError(f"{path} is larger than {_HASH_LIMIT >> 30} GiB")
        key = (
            str(path),
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        found = _hashes.get(key)
        if found is None:
            digest = hashlib.sha256()
            while chunk := os.read(fd, 1 << 20):
                digest.update(chunk)
            found = digest.hexdigest()
            if len(_hashes) > 256:
                _hashes.clear()
            _hashes[key] = found
        return found
    finally:
        os.close(fd)


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
        found = re.match(r"^(?:\d+:\s*)?(user|group):(\S+)\s+(?:inherited\s+)?allow\s+(\S+)", entry)
        if not found:
            continue
        kind, who, rights = found.groups()
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


def _walk(w: _Walk, given: str, role: str) -> tuple[PurePosixPath, os.stat_result] | None:
    """GIVEN walked by hand from `/`: every link followed, every folder
    stepped into on every hop checked, and the last item. Returns the path
    to execute and what the last item is, or None when the walk could not
    finish (a finding says why)."""
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
    return run_as or current, info


def _interpreters(w: _Walk, path: PurePosixPath, role: str) -> list[str]:
    """What runs a script, or nothing for a binary: the program `#!` names
    (refused when it is `env`), and under `#!/bin/sh` the python uv's
    `'''exec' "<python>" "$0" "$@"` line names."""
    try:
        head = _head(path)
    except OSError:
        return []
    if not head.startswith(b"#!"):
        return []
    lines = head.decode("utf-8", errors="replace").splitlines()
    words = lines[0][2:].split()
    if not words:
        return []
    if PurePosixPath(words[0]).name == "env":
        w.find(
            "unknown",
            path,
            f"{role} names its interpreter through PATH (#!{' '.join(words)}), so what "
            "would run is not known",
            fix="shebang",
        )
        return []
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


def present(path: str) -> bool:
    """Something is at PATH, a POSIX path (a link counts), as the check
    reads the disk."""
    return _exists(PurePosixPath(path))


def _environment(w: _Walk, root: PurePosixPath) -> None:
    """The Python environment at ROOT (it has a `pyvenv.cfg`): the root,
    its site-packages, and the interpreter it was made from with its
    standard library."""
    walked = _walk(w, str(root), f"its Python environment {root}")
    if walked is None:
        return
    root = walked[0]
    for lib in ("lib", "lib64"):
        try:
            names = sorted(_listdir(root / lib))
        except OSError:
            continue
        for name in names:
            packages = root / lib / name / "site-packages"
            if name.startswith("python") and _exists(packages):
                _walk(w, str(packages), f"its packages {packages}")
    try:
        text = _head(root / "pyvenv.cfg").decode("utf-8", errors="replace")
    except OSError:
        return
    home = re.search(r"(?m)^home\s*=\s*(.+?)\s*$", text)
    if home:
        base = _walk(w, home.group(1), f"the Python it was made from ({home.group(1)})")
        if base is not None and _exists(base[0].parent / "lib"):
            lib_dir = base[0].parent / "lib"
            _walk(w, str(lib_dir), f"that Python's standard library ({lib_dir})")


def _program(w: _Walk, given: str, depth: int, role: str) -> PurePosixPath | None:
    walked = _walk(w, given, role)
    if walked is None:
        return None
    resolved, info = walked
    if not stat.S_ISREG(info.st_mode):
        w.find("unknown", resolved, f"{role} is not a file", fix="check")
        return None
    for interpreter in _interpreters(w, resolved, role):
        if depth >= _INTERPRETER_DEPTH:
            w.find("unknown", resolved, f"{role} goes through too many interpreters", fix="check")
            break
        _program(w, interpreter, depth + 1, f"its interpreter {interpreter}")
    # The environment by the path as spelled (a venv's `bin/python` is a
    # link out of it) and by the path it resolves to.
    for program in (PurePosixPath(_absolute(given)), resolved):
        if _exists(program.parent.parent / "pyvenv.cfg"):
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
    return _decide(w, given, _program(w, given, 0, "it"), hashed=True)


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


def runs(verdict: Verdict) -> bool:
    """The rule: `ok` runs; `needs_approval` runs while an owner's approval
    of exactly this holds; `refused` never runs."""
    if verdict.kind is Kind.ok:
        return True
    if verdict.kind is Kind.needs_approval:
        from . import exec_approvals

        return exec_approvals.holds(verdict)
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
