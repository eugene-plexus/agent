"""In-memory auth state for the agent process.

Holds runtime secrets the agent should never persist:

  * `signing_key` — 32 random bytes generated at every startup. HMAC
    signs all JWTs (session + service tokens). Rotating at every
    restart effectively revokes all outstanding tokens (good-enough
    v0.2 revocation).
  * `master_key` — 32 bytes derived from the operator's passphrase
    via Argon2id. Encrypts apiKey-style fields on each child's disk.
    Threaded to spawned children via env var at startup.
  * `revoked_tokens` — set of JWT IDs (or full token strings) the
    operator has logged out. Cleared at restart along with the
    signing key.

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


@dataclass
class AuthState:
    """Per-process auth state. NOT persisted; rebuilt at every startup."""

    # Per-restart HMAC signing key for all JWTs.
    signing_key: bytes
    # 32-byte master key derived from the operator's passphrase. None
    # until the passphrase has been verified (login) or recovered from
    # the OS keyring; the agent refuses to spawn children that need
    # encrypted secrets until it has one.
    master_key: bytes | None = None
    # Revoked session tokens — set of full token strings. Logout
    # appends; checked on every auth-protected request.
    revoked_tokens: set[str] = field(default_factory=set)
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

    def revoke(self, token: str) -> None:
        with self._lock:
            self.revoked_tokens.add(token)

    def is_revoked(self, token: str) -> bool:
        with self._lock:
            return token in self.revoked_tokens

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
