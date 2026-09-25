"""Whether another account on this machine can read this install's secrets
or plant files in it (Windows).

`node.yaml` holds this node's private keys in the clear (`node_identity`),
and `_private_files` makes it owner-only on POSIX from its first byte. On
Windows the mode bits mean nothing and a file takes the ACL its directory
hands down -- which `_private_files` calls "the installer's business" and
which, until 2026-09-24, no installer did. Measured on a live service
install then: `%ProgramData%\\EugenePlexus` inherits
`BUILTIN\\Users:(OI)(CI)(RX)` and `BUILTIN\\Users:(CI)(WD,AD)` from
`%ProgramData%`, so every local account could read `node.yaml` (and mint
operator tokens for the whole install) and add files anywhere under it
-- a `.pth` in the venv's site-packages runs as LocalSystem at the next
service start. A per-user install inherits whatever the profile grants:
on the same box, `CodexSandboxUsers:(M)`.

`install.ps1` now sets the directory's ACL. This module is the other
half: it reads what the ACL actually is at startup and says so when an
account other than this one, SYSTEM or Administrators can read the
secrets or add files, because an install from before the fix, or a
directory someone re-permissioned by hand, looks healthy from every
other angle. It reports and never repairs: which subfolders the tray
icon needs to read is the installer's knowledge, and two copies of that
list are two chances to disagree.

Deliberately out of scope: an account with Administrator rights, which
can take any file whatever its ACL, and a program running as the same
account as the agent, which can read the agent's memory as easily as
its files.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["Ace", "Grant", "check", "foreign_grants"]

# ACE types and flags, from winnt.h.
ACCESS_ALLOWED_ACE_TYPE = 0x0
OBJECT_INHERIT_ACE = 0x1
INHERIT_ONLY_ACE = 0x8

# Rights that let an account read a file's contents.
_READ = 0x0001 | 0x80000000 | 0x10000000  # FILE_READ_DATA, GENERIC_READ, GENERIC_ALL
# Rights that let an account put something into a directory, or take
# its ACL over. FILE_ADD_FILE and FILE_ADD_SUBDIRECTORY are the same bits
# as FILE_WRITE_DATA and FILE_APPEND_DATA; on a file they change it.
_WRITE = (
    0x0002  # FILE_ADD_FILE / FILE_WRITE_DATA
    | 0x0004  # FILE_ADD_SUBDIRECTORY / FILE_APPEND_DATA
    | 0x00040000  # WRITE_DAC
    | 0x00080000  # WRITE_OWNER
    | 0x40000000  # GENERIC_WRITE
    | 0x10000000  # GENERIC_ALL
)

# Accounts that may hold these rights without it being a finding. The
# process's own account is added at check time.
_TRUSTED = frozenset(
    {
        "S-1-5-18",  # LocalSystem
        "S-1-5-32-544",  # BUILTIN\Administrators
        "S-1-3-4",  # OWNER RIGHTS
        # CREATOR OWNER: inherit-only on a directory, and replaced on each
        # new file by whoever created it -- this agent, for its own files.
        # A file somebody else created is judged by their real SID.
        "S-1-3-0",
        # NT SERVICE\TrustedInstaller
        "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464",
    }
)

# Files whose contents are the finding if another account can read them.
# `agent.yaml` carries the passphrase's Argon2id hash, the master-key
# salt and every sealed share password.
_SECRET_FILES = ("node.yaml", "agent.yaml")


@dataclass(frozen=True)
class Ace:
    """One ACE as `GetSecurityDescriptorDacl` reports it."""

    type: int
    flags: int
    mask: int
    sid: str
    account: str


@dataclass(frozen=True)
class Grant:
    """An account outside the trusted set holding a right it should not."""

    path: Path
    account: str
    right: str  # "read" or "add files to" / "change"

    def sentence(self) -> str:
        return f"{self.account} can {self.right} {self.path}"


def foreign_grants(
    aces: list[Ace], *, own_sid: str, is_directory: bool, path: Path, holds_secrets: bool = True
) -> list[Grant]:
    """The grants in `aces` that another account should not have.

    Pure, so it is tested on every platform. On a file, an effective
    allow of any read right is a finding, and so is any write right. On
    a directory, listing it is not a secret; adding to it is, and so is
    an object-inherit read, because that decides every file created in
    it from now on -- including the temp file `_private_files` writes a
    secret into before renaming it over the old one. A directory that
    holds only code (`holds_secrets=False`, the interpreter's) is judged
    on adding alone: anyone may read Python.

    Deny ACEs are ignored, which can only over-report: an allow that a
    deny cancels is flagged anyway. That is the safe direction for a
    warning.
    """
    trusted = _TRUSTED | {own_sid}
    grants: list[Grant] = []
    for ace in aces:
        if ace.type != ACCESS_ALLOWED_ACE_TYPE or ace.sid in trusted:
            continue
        effective = not ace.flags & INHERIT_ONLY_ACE
        if is_directory:
            if effective and ace.mask & _WRITE:
                grants.append(Grant(path, ace.account, "add files to"))
            if holds_secrets and ace.flags & OBJECT_INHERIT_ACE and ace.mask & _READ:
                grants.append(Grant(path, ace.account, "read new files in"))
        elif effective:
            if ace.mask & _READ:
                grants.append(Grant(path, ace.account, "read"))
            if ace.mask & _WRITE:
                grants.append(Grant(path, ace.account, "change"))
    # One sentence per (account, right): an account named in two ACEs
    # (one inherited, one explicit) is still one finding.
    return list(dict.fromkeys(grants))


def check(config_dir: Path) -> list[Grant]:
    """What another account can do to this install, or `[]`.

    `[]` also means *could not tell*: off Windows, where `_private_files`
    already makes the secrets owner-only, and on a Windows venv without
    pywin32. A check that cannot run must not claim a problem.
    """
    if sys.platform != "win32":
        return []
    try:
        import win32api
        import win32security
    except ImportError:
        log.debug("pywin32 is not installed; install permissions were not checked")
        return []

    try:
        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), win32security.TOKEN_QUERY
        )
        own_sid = win32security.ConvertSidToStringSid(
            win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        )
    except Exception as exc:  # a probe, never fatal
        log.debug("could not read this process's account (%s); permissions not checked", exc)
        return []

    # (path, is a directory, holds secrets)
    targets: list[tuple[Path, bool, bool]] = [(config_dir, True, True)]
    targets += [(config_dir / name, False, True) for name in _SECRET_FILES]
    # The interpreter this agent runs, and the one its venv was built
    # from: a file added to either runs as this account at the next start.
    for prefix in sorted({Path(sys.prefix), Path(sys.base_prefix)}):
        targets.append((prefix, True, False))

    grants: list[Grant] = []
    for path, is_directory, holds_secrets in targets:
        if not path.exists():
            continue
        aces = _read_aces(win32security, path)
        if aces is None:
            continue
        grants += foreign_grants(
            aces,
            own_sid=own_sid,
            is_directory=is_directory,
            path=path,
            holds_secrets=holds_secrets,
        )
    return list(dict.fromkeys(grants))


def _read_aces(ws: Any, path: Path) -> list[Ace] | None:
    try:
        descriptor = ws.GetFileSecurity(str(path), ws.DACL_SECURITY_INFORMATION)
        dacl = descriptor.GetSecurityDescriptorDacl()
    except Exception as exc:  # a probe, never fatal
        log.debug("could not read the ACL of %s (%s)", path, exc)
        return None
    if dacl is None:
        # A NULL DACL grants everyone everything.
        return [Ace(ACCESS_ALLOWED_ACE_TYPE, OBJECT_INHERIT_ACE, 0x10000000, "S-1-1-0", "Everyone")]
    aces: list[Ace] = []
    for index in range(dacl.GetAceCount()):
        # A standard ACE is ((type, flags), mask, sid); an object ACE puts
        # two GUIDs before the SID. The SID is last either way.
        entry = dacl.GetAce(index)
        (ace_type, flags), mask, sid = entry[0], entry[1], entry[-1]
        sid_string = ws.ConvertSidToStringSid(sid)
        try:
            name, domain, _ = ws.LookupAccountSid(None, sid)
            account = f"{domain}\\{name}" if domain else name
        except Exception:  # an orphaned SID has no name
            account = sid_string
        aces.append(Ace(ace_type, flags, mask, sid_string, account))
    return aces
