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
"""

from __future__ import annotations

import base64
import logging

import keyring
import keyring.errors

log = logging.getLogger(__name__)

# Service name shown in the OS keyring UI (Credential Manager / Keychain
# / Secret Service). Stable across installs of the same machine so an
# operator can recognize what's storing the key.
SERVICE = "eugene-plexus-agent"

# Username slot under that service. v0.2 is single-operator; v0.3+
# multi-operator would shift to per-operator usernames here.
USERNAME = "master-key"


def get_master_key() -> bytes | None:
    """Return the stored master key, or None if not present / backend
    unavailable / decode failed.

    Never raises. Any backend hiccup logs at warning level and falls
    back to None so the lifespan can move on to the passphrase prompt.
    """
    try:
        encoded = keyring.get_password(SERVICE, USERNAME)
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


def set_master_key(master_key: bytes) -> bool:
    """Persist the master key. Returns True on success.

    Best-effort: on failure logs a warning and returns False — the
    operator can still finish the unlock, they just won't auto-recover
    next time. Wizard UI can surface the False return as a "couldn't
    enable auto-unlock; you'll need to enter your passphrase on each
    restart" notice."""
    if len(master_key) != 32:
        raise ValueError("master key must be 32 bytes")
    encoded = base64.b64encode(master_key).decode("ascii")
    try:
        keyring.set_password(SERVICE, USERNAME, encoded)
        return True
    except keyring.errors.KeyringError as e:
        log.warning("keyring write failed (%s); master key NOT persisted", e)
        return False
    except Exception as e:
        log.warning("keyring write raised unexpected %s (%s)", type(e).__name__, e)
        return False


def delete_master_key() -> bool:
    """Remove the stored master key. Returns True if a value was
    deleted, False if there was nothing stored or the delete failed.

    Called when the operator switches `securityMode` from
    `os_keyring` to `prompt_on_startup` — the install promises a
    stronger boundary, so the old auto-unlock secret must be wiped.
    """
    try:
        keyring.delete_password(SERVICE, USERNAME)
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
