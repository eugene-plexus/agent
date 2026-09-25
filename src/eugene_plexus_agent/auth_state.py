"""In-memory auth state for the agent process.

Holds runtime secrets the agent should never persist:

  * `trust` — this node's token key, the authority it trusts, and the
    trust bundle (`trust.NodeTrust`). The key is this node's own, from
    `node.yaml`; no install-wide key exists any more (2026-09-25).
  * `master_key` — 32 bytes derived from the operator's passphrase
    via Argon2id. Encrypts apiKey-style fields on each child's disk.
    Threaded to spawned children via env var at startup.

And one thing that is not a secret and IS persisted:

  * `revoked` — the sessions the operator signed out of, as hashes.
    "Cleared at restart along with the signing key" was the design
    until 2026-09-22, and it was wrong on an enrolled node, whose key
    is NOT cleared at restart: every signed-out token came back to life.
    See `session_revocations`. Enrolled, a sign-out is also replicated
    by the control root and reaches every machine through the bundle.

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

from .session_revocations import RevokedSessions
from .trust import NodeTrust


@dataclass
class AuthState:
    """Per-process auth state. Rebuilt at every startup; only `revoked`
    is read back from disk, by the lifespan."""

    # This node's token key and the bundle it verifies against. The
    # supervisor reads it at every spawn, so children restarted after
    # enrollment or un-enrollment pick up the current authority.
    trust: NodeTrust
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
