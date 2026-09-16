"""Long-lived keys for apps outside the install (hobbyist UX S4, 2026-09-15).

Before this module the only bearer a person could paste into Continue,
Cline, Open WebUI or an OpenAI SDK was the **operator session token**:
authority over every component in the install, reachable only from the
playground's diagnostic panel, and dead fourteen days after sign-in.
`tailnet.md` said so in as many words -- *"There is no long-lived client
key yet."* A client key replaces it with something a person can hand
out: named, a year long by default, scoped to the gateway's three
OpenAI-compatible paths by its `aud: client` claim, and revocable one at
a time.

## What is stored, and what is not

**Not the token.** It is on the wire once, from the POST that minted it,
and then forgotten. There is nothing in `client_keys.json` worth
stealing and nothing to re-read: a lost key is re-minted. What is kept
is a record -- the name the operator gave it, when it was made, when it
expires, whether it has been revoked, and the token's **tail**.

The tail and not a prefix, and that is not a style choice: the token is
a JWT, so every key this install mints begins
`eyJhbGciOiJIUzI1NiIs...`. A prefix would identify nothing. Six
characters off the end are enough to match a record against a key
already pasted into an app, and too few to be worth anything alone.

## Where the file lives, and why it is its own file

`client_keys.json` beside `agent.yaml`, the same shape as
`library_folders.json`. Not inside `agent.yaml`: `AgentState` writes its
`auth` block by replacing it whole (`set_passphrase`), a growing list
does not belong in a config document the config trio serves, and a
record file that can be deleted by hand is a recovery path.

## Revocation, and its honest bound

A revoked record is **kept**, marked with a timestamp, until the key's
own expiry passes. A list that forgets cannot tell "never minted here"
from "turned off". Past the expiry the record is pruned and the id
leaves the revoked set -- the signature check refuses an expired token
with no list to consult, and a revocation list that only grows is a
leak.

The gateway polls `revoked()` and caches it for about its routing
refresh interval, so revoking is bounded rather than instant. The
instant revocation is the one that has existed since M7: rotate the
install's signing key and every token everywhere stops verifying.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KEYS_FILE = "client_keys.json"

# How much of the token's end is recorded. Enough to pick one key out of
# a list of five; not enough to help anyone holding it.
TAIL_LENGTH = 6

# The default life of a key, in days. A year: long enough that someone
# who set Continue up once does not come back to a dead key, short
# enough that a key forgotten in a config file eventually stops working.
DEFAULT_TTL_DAYS = 365
MAX_TTL_DAYS = 3650


@dataclass(frozen=True)
class ClientKeyRecord:
    """One key, as it is stored. Never carries the token."""

    id: str
    name: str
    tail: str
    created_at: float
    expires_at: float
    revoked_at: float | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "tail": self.tail,
            "createdAt": self.created_at,
            "expiresAt": self.expires_at,
        }
        if self.revoked_at is not None:
            out["revokedAt"] = self.revoked_at
        return out

    @classmethod
    def from_json(cls, raw: Any) -> ClientKeyRecord | None:
        """Parse one record, or None when it is not one.

        A record file edited by hand, or written by a future version,
        must not stop the agent: an unreadable entry is dropped with a
        warning and the rest of the list still loads
        (`degraded-mode-required`, applied to the agent's own files).
        """
        if not isinstance(raw, dict):
            return None
        try:
            key_id = str(raw["id"])
            name = str(raw["name"])
            tail = str(raw.get("tail", ""))
            created = float(raw["createdAt"])
            expires = float(raw["expiresAt"])
        except (KeyError, TypeError, ValueError):
            return None
        revoked_raw = raw.get("revokedAt")
        try:
            revoked = float(revoked_raw) if revoked_raw is not None else None
        except (TypeError, ValueError):
            revoked = None
        if not key_id or not name:
            return None
        return cls(
            id=key_id,
            name=name,
            tail=tail,
            created_at=created,
            expires_at=expires,
            revoked_at=revoked,
        )


def new_key_id() -> str:
    """The token's `jti`, and the record's id in the URL.

    128 bits of randomness as hex. Not a counter: the id travels inside
    a token an outside app holds, and a guessable one would let someone
    revoke a key they were never given.
    """
    return secrets.token_hex(16)


def tail_of(token: str) -> str:
    return token[-TAIL_LENGTH:] if len(token) > TAIL_LENGTH else token


def as_datetime(unix_seconds: float) -> datetime:
    return datetime.fromtimestamp(unix_seconds, tz=UTC)


class ClientKeyStore:
    """Threadsafe owner of `client_keys.json`. One lock, one file write.

    `revision` increments whenever the *revoked set* could have changed
    -- a revoke, or a prune that dropped an expired revoked record. A
    mint does not change it: the gateway reads only the revoked set, and
    a revision that moved for a reason the reader cannot see is a
    revision that teaches the reader nothing.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._records: dict[str, ClientKeyRecord] = {}
        self._revision = 0

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        """Read the file, tolerating everything a file can be.

        A missing file is an install with no client keys, which is every
        install until someone mints one -- not an error, and not a
        reason to write an empty file nobody asked for.
        """
        with self._lock:
            self._records = {}
            self._revision = 0
            if not self._path.exists():
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning(
                    "could not read %s (%s); this agent starts with no client keys and will "
                    "overwrite the file on the next mint",
                    self._path,
                    e,
                )
                return
            if not isinstance(raw, dict):
                log.warning("%s is not a JSON object; ignoring it", self._path)
                return
            try:
                self._revision = int(raw.get("revision", 0))
            except (TypeError, ValueError):
                self._revision = 0
            dropped = 0
            for entry in raw.get("keys") or []:
                record = ClientKeyRecord.from_json(entry)
                if record is None:
                    dropped += 1
                    continue
                self._records[record.id] = record
            if dropped:
                log.warning(
                    "%s had %d unreadable client-key record(s); dropped", self._path, dropped
                )
            self._prune_locked()

    def records(self, *, now: float | None = None) -> list[ClientKeyRecord]:
        """Every live record, newest first.

        Named `records` rather than `list` because inside this class body
        `list` would then name the method and `list[ClientKeyRecord]`
        would stop being a type -- which mypy says out loud and a reader
        would find bewildering.
        """
        with self._lock:
            self._prune_locked(now=now)
            return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def revoked(self, *, now: float | None = None) -> tuple[list[str], int]:
        """The ids the gateway must refuse, and the revision they are at.

        Expired records have already left, so this set is bounded by how
        many *unexpired* keys an operator has revoked.
        """
        with self._lock:
            self._prune_locked(now=now)
            ids = sorted(r.id for r in self._records.values() if r.revoked_at is not None)
            return ids, self._revision

    def add(self, record: ClientKeyRecord) -> ClientKeyRecord:
        with self._lock:
            self._records[record.id] = record
            self._prune_locked()
            self._write_locked()
            return record

    def revoke(self, key_id: str, *, now: float | None = None) -> ClientKeyRecord | None:
        """Mark one key revoked. Returns None when no such key is here.

        Revoking a key that is already revoked is a no-op that still
        answers with the record: a `DELETE` repeated is not an error,
        and re-stamping the time would lose when it actually happened.
        """
        stamp = time.time() if now is None else now
        with self._lock:
            record = self._records.get(key_id)
            if record is None:
                return None
            if record.revoked_at is None:
                record = ClientKeyRecord(
                    id=record.id,
                    name=record.name,
                    tail=record.tail,
                    created_at=record.created_at,
                    expires_at=record.expires_at,
                    revoked_at=stamp,
                )
                self._records[key_id] = record
                self._revision += 1
                self._write_locked()
            return record

    # ----- internals --------------------------------------------------

    def _prune_locked(self, *, now: float | None = None) -> None:
        """Drop records whose key has expired.

        An expired token is refused on its own claims, so keeping it in
        the revoked set buys nothing and costs a list that only grows.
        Dropping a *revoked* one changes what the gateway would read, so
        that bumps the revision; dropping an unrevoked one does not.
        """
        stamp = time.time() if now is None else now
        gone = [r for r in self._records.values() if r.expires_at <= stamp]
        if not gone:
            return
        for record in gone:
            del self._records[record.id]
            if record.revoked_at is not None:
                self._revision += 1
        log.info("pruned %d expired client-key record(s)", len(gone))
        self._write_locked()

    def _write_locked(self) -> None:
        payload = {
            "revision": self._revision,
            "keys": [r.to_json() for r in self._records.values()],
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as e:
            # The same rule `install_console_capture` learned on the
            # first container boot: a convenience that cannot write must
            # not take the process down. The keys stay live in memory for
            # this process run; they are gone on the next start, which is
            # worse than persisting and far better than a 500 on a mint.
            log.warning(
                "could not write %s (%s); client keys will not survive a restart", self._path, e
            )
