"""Thin wrapper around the `keyring` library for OS-managed secret storage.

Used only when `securityMode == "os_keyring"`. The agent stores the
derived master key under (`service`, `username`) so a power outage or
service restart auto-recovers without the operator re-typing the
passphrase. The OS-level boundary on that store is whatever the
underlying backend provides:

  * Windows: WinVault / Credential Manager — per-Windows-user
  * macOS: Keychain — per-macOS-user, may prompt on first access
  * Linux: Secret Service (gnome-keyring / KWallet) — per-session,
    requires a running daemon

The library auto-selects the highest-priority available backend; on
headless Linux without an unlocked secret service the active backend
becomes a `fail` one. Every call here is wrapped in a broad except
so a missing/locked backend never crashes the agent — the
operator just falls through to the passphrase-prompt path.

Storage shape: master key is 32 random bytes (derived via Argon2id);
keyring backends take strings, so we base64-encode at store time
and decode at load time.

**The entry is scoped per install (S0 of the hobbyist UX plan,
2026-09-15).** Two installs share one machine more often than the
original single slot assumed: the live worker and a `.dev-install`,
the live worker and every acceptance run, a reinstall under a new
prefix beside the old state. With `os_keyring` becoming the desktop
default, a second install's wizard would have overwritten the first
install's stored key under the one shared username, and the first
would have come back locked on its next start with a warning nobody
was watching for. So the username carries a fingerprint of the
install's master-key salt — the one value that is unique per install,
exists before the master key does, and is stored beside it in
`agent.yaml`. A legacy single-slot entry is read once, moved to its
scoped name and deleted, so an install that predates this keeps
auto-unlocking across the upgrade.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import threading

import keyring
import keyring.errors

log = logging.getLogger(__name__)

# Service name shown in the OS keyring UI (Credential Manager / Keychain
# / Secret Service). Stable across installs of the same machine so an
# operator can recognize what's storing the key.
SERVICE = "eugene-plexus-agent"

# The pre-scoping username. Read as a fallback and migrated; never
# written to again.
LEGACY_USERNAME = "master-key"

# Kept for readers of the old name. Nothing in this module writes to it.
USERNAME = LEGACY_USERNAME


def install_id_for(master_salt_b64: str) -> str:
    """A short, stable fingerprint of one install, from its master-key salt.

    The salt is 16+ random bytes minted at `POST /v1/auth/initialize`
    and persisted in the clear in `agent.yaml`; hashing it keeps the
    keyring entry name from being the salt itself while staying
    deterministic across restarts. Twelve hex characters is 48 bits —
    collision between two installs on one machine is not a real risk,
    and the name stays readable in Credential Manager.
    """
    raw = base64.b64decode(master_salt_b64)
    return hashlib.sha256(raw).hexdigest()[:12]


def username_for(install_id: str) -> str:
    return f"{LEGACY_USERNAME}:{install_id}"


def _read(username: str) -> bytes | None:
    try:
        encoded = keyring.get_password(SERVICE, username)
    except keyring.errors.KeyringError as e:
        log.warning("keyring read failed (%s); falling back to passphrase prompt", e)
        return None
    except Exception as e:
        # Some backends raise non-KeyringError exceptions on headless
        # systems (RuntimeError from the dbus probe, etc.). Defensive
        # broad catch — keyring failure is never fatal.
        log.warning("keyring read raised unexpected %s (%s)", type(e).__name__, e)
        return None
    if not encoded:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as e:
        log.warning("stored keyring value is not valid base64 (%s); ignoring", e)
        return None
    if len(raw) != 32:
        log.warning("stored keyring value is %d bytes, expected 32; ignoring", len(raw))
        return None
    return raw


def _write(username: str, master_key: bytes) -> bool:
    encoded = base64.b64encode(master_key).decode("ascii")
    try:
        keyring.set_password(SERVICE, username, encoded)
        return True
    except keyring.errors.KeyringError as e:
        log.warning("keyring write failed (%s); master key NOT persisted", e)
        return False
    except Exception as e:
        log.warning("keyring write raised unexpected %s (%s)", type(e).__name__, e)
        return False


def _delete(username: str) -> bool:
    try:
        keyring.delete_password(SERVICE, username)
        return True
    except keyring.errors.PasswordDeleteError:
        # No value stored — not an error from our perspective.
        return False
    except keyring.errors.KeyringError as e:
        log.warning("keyring delete failed (%s)", e)
        return False
    except Exception as e:
        log.warning("keyring delete raised unexpected %s (%s)", type(e).__name__, e)
        return False


def get_master_key(install_id: str) -> bytes | None:
    """Return this install's stored master key, or None if not present /
    backend unavailable / decode failed.

    Never raises. Any backend hiccup logs at warning level and falls
    back to None so the lifespan can move on to the passphrase prompt.

    Reads the scoped entry first. Finding nothing there, it reads the
    legacy single-slot entry an older agent wrote, and if that holds a
    key it is moved to the scoped name — written first, deleted second,
    so a failure between the two leaves a duplicate rather than
    nothing. Whether the legacy key actually opens THIS install's
    sealed values is the caller's to find out, exactly as before.
    """
    scoped = username_for(install_id)
    found = _read(scoped)
    if found is not None:
        return found
    legacy = _read(LEGACY_USERNAME)
    if legacy is None:
        return None
    if _write(scoped, legacy):
        _delete(LEGACY_USERNAME)
        log.info("moved the stored master key to its per-install keyring entry")
    return legacy


def set_master_key(master_key: bytes, install_id: str) -> bool:
    """Persist the master key under this install's entry. Returns True on
    success.

    Best-effort: on failure logs a warning and returns False — the
    operator can still finish the unlock, they just won't auto-recover
    next time. Wizard UI can surface the False return as a "couldn't
    enable auto-unlock; you'll need to enter your passphrase on each
    restart" notice."""
    if len(master_key) != 32:
        raise ValueError("master key must be 32 bytes")
    return _write(username_for(install_id), master_key)


def delete_master_key(install_id: str) -> bool:
    """Remove this install's stored master key. Returns True if a value
    was deleted, False if there was nothing stored or the delete failed.

    Called when the operator switches `securityMode` from
    `os_keyring` to `prompt_on_startup` — the install promises a
    stronger boundary, so the old auto-unlock secret must be wiped.
    The legacy slot is cleared too: an install that was never migrated
    (it switched modes before its first `os_keyring` start) must not
    leave a key behind under the old name.
    """
    scoped = _delete(username_for(install_id))
    legacy = _delete(LEGACY_USERNAME)
    return scoped or legacy


# --------------------------------------------------------------------------- #
# Availability probe
# --------------------------------------------------------------------------- #

_PROBE_LOCK = threading.Lock()
_probe_result: bool | None = None
_probe_done = False


def probe_sync() -> bool:
    """Whether this host's keyring can hold a secret for us: write, read
    back and delete a throwaway entry.

    Measured, not assumed from the platform — a Windows service account
    without a credential store, a container, and a headless Linux box
    with no unlocked Secret Service all report False here and would all
    have been "Linux/Windows, so probably fine" by inference. Memoised
    for the life of the process: a desktop keyring may put up an unlock
    dialog on first contact, and once is the right number of times to
    do that from a status endpoint the UI polls on every page load.

    Runs synchronously; the route wraps it in a thread with a deadline,
    because a Secret Service that is present but locked can block on a
    prompt nobody will answer.
    """
    global _probe_result, _probe_done
    with _PROBE_LOCK:
        if _probe_done:
            return bool(_probe_result)
        username = f"probe:{secrets.token_hex(6)}"
        payload = secrets.token_bytes(32)
        ok = False
        try:
            if _write(username, payload):
                ok = _read(username) == payload
        finally:
            _delete(username)
        _probe_result = ok
        _probe_done = True
        if not ok:
            log.info(
                "this host's OS keyring did not accept a probe entry; securityMode "
                "os_keyring would not auto-unlock here"
            )
        return ok


def reset_probe_cache() -> None:
    """Tests only: forget the memoised probe result."""
    global _probe_result, _probe_done
    with _PROBE_LOCK:
        _probe_result = None
        _probe_done = False
