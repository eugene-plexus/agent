"""In-memory auth state for the agent process.

Holds runtime secrets the agent should never persist:

  * `signing_key` — Ed25519 private PEM, or the retained legacy HMAC
    key until rotation. Enrolled agents restore the install's key from
    node.yaml; only unenrolled agents generate a key at startup.
  * `master_key` — 32 bytes derived from the operator's passphrase
    via Argon2id. Encrypts apiKey-style fields on each child's disk.
    Threaded to spawned children via env var at startup.

And one thing that is not a secret and IS persisted:

  * `revoked` — the sessions the operator signed out of, as hashes.
    "Cleared at restart along with the signing key" was the design
    until 2026-09-22, and it was wrong on an enrolled node, whose
    signing key is NOT cleared at restart: every signed-out token came
    back to life. See `session_revocations`.

This state lives in `app.state.auth_state` after the lifespan
initializes it. The supervisor reaches into it to read the master
key + service tokens at spawn time. The routes layer reads it to
validate session tokens. Nothing else touches it.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from . import security
from .session_revocations import RevokedSessions


@dataclass
class AuthState:
    """Per-process auth state. Rebuilt at every startup; only `revoked`
    is read back from disk, by the lifespan."""

    # Private signing material for all JWTs. New per restart on an agent that
    # has not enrolled; the INSTALL'S key, persisted in node.yaml and
    # adopted at boot, on one that has (M7). Replaced in place by
    # enrollment and by a signed re-key from the control root — the
    # supervisor reads it at every spawn, so children restarted after
    # either pick up the current one.
    signing_key: bytes
    # 32-byte master key derived from the operator's passphrase. None
    # until the passphrase has been verified (login) or recovered from
    # the OS keyring; the agent refuses to spawn children that need
    # encrypted secrets until it has one.
    master_key: bytes | None = None
    # Signed-out sessions. Logout adds; checked on every auth-protected
    # request AND by the browser proxy before it forwards anything. In
    # memory until the lifespan binds it to its file beside agent.yaml.
    revoked: RevokedSessions = field(default_factory=RevokedSessions)
    # Per-source-IP sliding-window log of failed login attempts. Kept
    # in AuthState (rather than module-global) so tests get a clean
    # rate-limit state with each fresh app fixture.
    login_failures: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def has_master_key(self) -> bool:
        return self.master_key is not None

    def set_master_key(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("master key must be 32 bytes")
        with self._lock:
            self.master_key = key

    def set_signing_key(self, key: bytes) -> None:
        """Adopt the install's signing key — at enrollment, or when the
        control root rotates it. Every token this agent minted under the
        previous key stops verifying here, which is the point of a
        rotation and the price of enrollment."""
        security.validate_signing_key(key)
        with self._lock:
            self.signing_key = key

    def revoke(self, token: str, *, expires_at: int) -> None:
        """Sign `token` out until it would have expired anyway."""
        self.revoked.revoke(token, expires_at=expires_at)

    def is_revoked(self, token: str) -> bool:
        return self.revoked.is_revoked(token)

    def record_login_failure(self, source: str, *, window_seconds: int, max_in_window: int) -> bool:
        """Append a failure for this source; return True iff the source
        is now at or above the rate limit."""
        now = time.time()
        with self._lock:
            window = self.login_failures[source]
            while window and window[0] < now - window_seconds:
                window.popleft()
            window.append(now)
            return len(window) >= max_in_window

    def is_login_rate_limited(
        self, source: str, *, window_seconds: int, max_in_window: int
    ) -> bool:
        now = time.time()
        with self._lock:
            window = self.login_failures[source]
            while window and window[0] < now - window_seconds:
                window.popleft()
            return len(window) >= max_in_window

    def clear_login_failures(self, source: str) -> None:
        with self._lock:
            self.login_failures.pop(source, None)
