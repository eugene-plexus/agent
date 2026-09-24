"""The passphrase, kept in a file only this agent's account can read.

Used only when `securityMode == "passphrase_file"`, the sibling of
`keyring_store` for an agent that has no keyring to use: **the Linux
system install** (2026-09-24), where the agent runs as its own account
so that a program running as the person -- an AI agent they started,
say -- cannot read its files, its environment or its memory. A system
account has no desktop session and so no Secret Service, and without
this every reboot would leave the install locked until someone signed
in. The installer points `EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE` here and
writes `securityMode: passphrase_file` before the first start; the
control root on the same host reads the same file through its own
`passphrase_file` mode.

**The difference from the control root's mode is who writes the file.**
Control's is the operator's -- a container secret, mounted -- and it
never writes it. On a system install nobody supplies one: the
passphrase is typed into the browser wizard. So this agent writes it,
at `initialize` and at every sign-in, which are the only moments it
holds the passphrase itself. We made it, so we manage it.

**The passphrase, not the derived key**, because the control root needs
the passphrase: it derives its own master key from it with its own
salt, and one file serves both.

What this trades away is the keyring's trade exactly: whoever can read
the file can unlock the install. It is 0400 and owned by the service
account, so that is the account itself and root -- neither of which any
file permission could keep out anyway.

Every call degrades rather than raises: a missing, unreadable or wrong
file leaves the agent locked and asking, the behaviour with the mode
unset.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

from . import _private_files, security

log = logging.getLogger(__name__)

__all__ = ["FILE_MODE", "MAX_BYTES", "read_passphrase", "store_passphrase", "unlock_key"]

FILE_MODE = 0o400
"""Owner read, nothing else. Every write is a new file renamed over it."""

MAX_BYTES = 4096
"""A passphrase, not a file full of them."""


def read_passphrase(path: Path | None) -> str | None:
    """The passphrase held in `path`, or None if it cannot be had.

    Never raises, and every None is logged with its reason: the visible
    symptom of each is the same locked agent.
    """
    if path is None:
        log.warning(
            "securityMode is passphrase_file but EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE is "
            "not set; this agent is locked until someone signs in"
        )
        return None
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        log.warning(
            "passphrase file %s does not exist yet; this agent is locked until someone "
            "signs in, which writes it",
            path,
        )
        return None
    except OSError as exc:
        log.warning("passphrase file %s could not be read (%s); this agent stays locked", path, exc)
        return None
    if len(raw) > MAX_BYTES:
        log.warning(
            "passphrase file %s is %d bytes, over the %d-byte limit; not treating it as a "
            "passphrase",
            path,
            len(raw),
            MAX_BYTES,
        )
        return None
    _warn_if_widely_readable(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        log.warning(
            "passphrase file %s is not valid UTF-8 (%s); this agent stays locked", path, exc
        )
        return None
    # One trailing newline, the one `echo` adds, and nothing else: a
    # passphrase may begin or end with a space. The control root reads
    # the same file by the same rule.
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith("\n"):
        text = text[:-1]
    if not text:
        log.warning("passphrase file %s is empty; this agent stays locked", path)
        return None
    return text


def store_passphrase(path: Path | None, passphrase: str) -> bool:
    """Write `passphrase` to `path`, owner-read-only, unless it is already there.

    Called with a passphrase that was just set or just verified, so
    what lands is always one this install accepts. Returns whether the
    file now holds it. Never raises: a sign-in must not fail because a
    convenience did.
    """
    if path is None:
        log.warning(
            "securityMode is passphrase_file but EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE is not "
            "set, so there is nowhere to keep the passphrase; this agent will be locked "
            "after its next restart"
        )
        return False
    try:
        if path.is_file() and path.read_bytes() == passphrase.encode("utf-8"):
            return True
    except OSError:
        pass  # unreadable is a reason to write it, not to give up
    try:
        _private_files.write_private(path, passphrase, mode=FILE_MODE)
    except OSError as exc:
        log.warning(
            "could not write the passphrase file %s (%s); this agent will be locked after "
            "its next restart",
            path,
            exc,
        )
        return False
    log.info("passphrase kept in %s for unlocking after a restart", path)
    return True


def unlock_key(path: Path | None, *, passphrase_hash: str, salt: bytes) -> bytes | None:
    """The master key, from the passphrase in `path`, or None.

    Verified against the install's own hash before anything is derived:
    a file holding some other passphrase would otherwise derive a key
    that opens nothing, and every sealed value would read as corrupt
    rather than locked. Two Argon2id runs; call it off the event loop.
    """
    passphrase = read_passphrase(path)
    if passphrase is None:
        return None
    if not security.verify_passphrase(passphrase, passphrase_hash):
        log.warning(
            "the passphrase in %s is not this install's; this agent stays locked until "
            "someone signs in, which replaces it",
            path,
        )
        return None
    return security.derive_master_key(passphrase, salt)


def _warn_if_widely_readable(path: Path) -> None:
    """Say so when another account could read it. A warning, not a refusal."""
    if os.name == "nt":
        return
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        log.warning(
            "passphrase file %s is readable by group or other (mode %o); anyone who can "
            "read it can unlock this install. chmod 400 it.",
            path,
            stat.S_IMODE(mode),
        )
