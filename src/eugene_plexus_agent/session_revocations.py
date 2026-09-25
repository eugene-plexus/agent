"""The sessions an operator signed out of, remembered across restarts.

`DELETE /v1/auth/sessions/current` used to put the token in a set in
this process's memory. Two things were wrong with that, and they are
why this is a file and why the proxy reads it:

* **A restart forgot every sign-out.** An unenrolled agent mints a new
  signing key per restart, so its old tokens died with it -- but an
  enrolled node adopts THE INSTALL'S key from `node.yaml` at boot, and
  there a restart turned every signed-out token back into a live one
  for the rest of its 14 days.
* **Only this agent's own routes consulted the set.** The gateway, the
  library, the drivers and the control root verify tokens themselves
  and had never heard of it. (Since 2026-09-25 the control root also
  replicates a sign-out and the trust bundle carries it everywhere.) The browser
  reaches all of them through this agent's `/api/proxy`, so "Sign out"
  signed out of the agent's pages and of nothing else. The proxy now
  refuses a revoked token before it forwards anything (`routes/proxy.py`).

**Keyed by a SHA-256 of the token, never the token.** Not by `jti`:
this agent's operator sessions gained a random one only on 2026-09-22
(`tokens.Signer.mint` says why), every session minted before
that has none and is valid for up to 14 days more, and an enrolled node
also accepts the control root's sessions, which carry a `jti` of their
own. The whole token identifies all three alike. And not the token
itself, because a list of usable bearer tokens on disk would be a worse
thing to leave lying around than the problem this solves; a token is
300-odd bytes of signed, high-entropy JSON, and its hash says nothing
about it to anyone who does not already hold it.

**An entry is kept for the token's `exp` plus the clock-skew leeway**
(`tokens.LEEWAY_SECONDS`), because that is how long a
decoder in this install would still accept it. After that the token is
refused as expired wherever it goes, and the entry is only litter; so
the file is bounded by the sign-outs of the last fortnight.

**Degraded, not dead** (`degraded-mode-required`). A file that will not
parse costs the sign-outs it held and nothing else: it is kept beside
itself as `.unreadable`, the reason is logged at ERROR with what it
means, and the next sign-out writes a good file. Refusing to boot over
it would take an install down to protect sessions a person can end
again by signing in and out.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from . import tokens
from ._private_files import write_private

log = logging.getLogger(__name__)

__all__ = ["REVOKED_SESSIONS_FILE", "RevokedSessions", "session_id"]

REVOKED_SESSIONS_FILE = "revoked_sessions.json"
"""Beside `agent.yaml`, like `node.yaml` and `client_keys.json`."""

_UNREADABLE_SUFFIX = ".unreadable"
_VERSION = 1


def session_id(token: str) -> str:
    """What is stored and compared: a hash of the token, never the token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class RevokedSessions:
    """A set of signed-out sessions, each held until its token could no
    longer verify anyway. In memory only until `bind` gives it a file."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until: dict[str, float] = {}
        self._path: Path | None = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._until)

    def bind(self, path: Path, *, now: float | None = None) -> None:
        """Adopt `path`: read what it holds, keep what is already revoked
        here, and write every later sign-out to it.

        Called once, in the lifespan, whether or not a test pre-built the
        auth state -- a sign-out that was not written down is the defect
        this module exists to remove.
        """
        stamp = time.time() if now is None else now
        loaded = self._read(path)
        with self._lock:
            self._path = path
            for key, until in loaded.items():
                self._until[key] = max(until, self._until.get(key, 0.0))
            pruned = self._prune_locked(stamp)
            if pruned or len(self._until) != len(loaded):
                self._persist_locked()

    def revoke(self, token: str, *, expires_at: int, now: float | None = None) -> None:
        """Refuse `token` from now until it could no longer verify.

        **A failed write does not undo the sign-out.** The revocation holds
        in this process either way; what a full disk costs is its survival
        across a restart, and that is logged rather than turned into a
        sign-out button that errors -- which would leave the person who
        pressed it believing they were still signed in, or unable to leave.
        """
        stamp = time.time() if now is None else now
        with self._lock:
            self._until[session_id(token)] = float(expires_at + tokens.LEEWAY_SECONDS)
            self._prune_locked(stamp)
            self._persist_locked()

    def is_revoked(self, token: str, *, now: float | None = None) -> bool:
        stamp = time.time() if now is None else now
        key = session_id(token)
        with self._lock:
            until = self._until.get(key)
            return until is not None and until > stamp

    # ----- internals --------------------------------------------------

    def _prune_locked(self, now: float) -> int:
        expired = [key for key, until in self._until.items() if until <= now]
        for key in expired:
            del self._until[key]
        return len(expired)

    def _persist_locked(self) -> None:
        if self._path is None:
            return
        document = {
            "version": _VERSION,
            "sessions": [{"id": key, "until": until} for key, until in sorted(self._until.items())],
        }
        try:
            write_private(self._path, json.dumps(document, indent=2))
        except OSError as exc:
            log.error(
                "could not record a sign-out in %s (%s); it holds until this agent restarts, "
                "and after that the signed-out session is accepted again until it expires",
                self._path,
                exc,
            )

    @staticmethod
    def _read(path: Path) -> dict[str, float]:
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            _preserve(path)
            log.error(
                "%s could not be read (%s); the sign-outs it recorded are forgotten, so any "
                "browser signed out before this restart is signed in again until its session "
                "expires. A copy is at %s. Sign out again on any browser that should not be.",
                path,
                exc,
                path.with_name(path.name + _UNREADABLE_SUFFIX),
            )
            return {}
        sessions = raw.get("sessions") if isinstance(raw, dict) else None
        if not isinstance(sessions, list):
            _preserve(path)
            log.error(
                "%s is not a list of sign-outs; starting with none. A copy is kept beside it.",
                path,
            )
            return {}
        out: dict[str, float] = {}
        for entry in sessions:
            # One bad row costs that row, not the file: every other entry
            # is a sign-out somebody asked for.
            if not isinstance(entry, dict):
                continue
            key, until = entry.get("id"), entry.get("until")
            if (
                isinstance(key, str)
                and isinstance(until, int | float)
                and not isinstance(until, bool)
            ):
                out[key] = float(until)
        return out


def _preserve(path: Path) -> None:
    """Keep a file we could not read, beside itself. Never raises."""
    target = path.with_name(path.name + _UNREADABLE_SUFFIX)
    try:
        write_private(target, path.read_bytes())
    except OSError as exc:  # pragma: no cover - defensive
        log.warning("could not preserve %s as %s: %s", path, target, exc)
