"""Atomic standalone registry and source records for install-wide migration.

Only metadata is stored; bearer tokens are returned once at creation. A corrupt
or unwritable registry is unavailable, never silently replaced with an empty one.
"""

from __future__ import annotations

import json
import logging
import math
import os
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
        """Parse metadata strictly; corruption must never clear a revocation."""
        if not isinstance(raw, dict):
            return None
        try:
            key_id = raw["id"]
            name = raw["name"]
            tail = raw.get("tail", "")
            if not all(isinstance(v, str) for v in (key_id, name, tail)):
                return None
            created = float(raw["createdAt"])
            expires = float(raw["expiresAt"])
        except (KeyError, TypeError, ValueError):
            return None
        revoked_raw = raw.get("revokedAt")
        try:
            revoked = float(revoked_raw) if revoked_raw is not None else None
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (created, expires)) or (
            revoked is not None and not math.isfinite(revoked)
        ):
            return None
        if not key_id or not name or len(key_id) > 128 or len(name) > 64 or len(tail) > 6:
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
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._records: dict[str, ClientKeyRecord] = {}
        self._revision = 0
        self.error: str | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def revision(self) -> int:
        return self._revision

    def load(self) -> None:
        with self._lock:
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict) or not isinstance(raw.get("keys"), list):
                    raise ValueError("invalid registry")
                revision = raw.get("revision", 0)
                if type(revision) is not int or revision < 0:
                    raise ValueError("invalid revision")
                records = {}
                for value in raw["keys"]:
                    record = ClientKeyRecord.from_json(value)
                    if record is None or record.id in records:
                        raise ValueError("invalid or duplicate record")
                    records[record.id] = record
                self._records, self._revision, self.error = records, revision, None
            except FileNotFoundError:
                self._records, self._revision, self.error = {}, 0, None
            except (OSError, ValueError, TypeError):
                self.error = (
                    "Client-key registry is unreadable. Restore client_keys.json from backup."
                )
                log.warning(self.error)

    def _check(self) -> None:
        if self.error:
            raise OSError(self.error)

    def records(self, *, now: float | None = None) -> list[ClientKeyRecord]:
        with self._lock:
            self._check()
            stamp = time.time() if now is None else now
            return sorted(
                (r for r in self._records.values() if r.expires_at > stamp),
                key=lambda r: r.created_at,
                reverse=True,
            )

    def revoked(self, *, now: float | None = None) -> tuple[list[str], int]:
        with self._lock:
            return sorted(
                r.id for r in self.records(now=now) if r.revoked_at is not None
            ), self._revision

    def add(self, record: ClientKeyRecord) -> ClientKeyRecord:
        with self._lock:
            self._check()
            if record.id in self._records:
                raise ValueError("duplicate client key")
            self._commit({**self._records, record.id: record})
            return record

    def revoke(self, key_id: str, *, now: float | None = None) -> ClientKeyRecord | None:
        with self._lock:
            self._check()
            record = self._records.get(key_id)
            if record is not None and record.revoked_at is None:
                record = ClientKeyRecord(
                    record.id,
                    record.name,
                    record.tail,
                    record.created_at,
                    record.expires_at,
                    time.time() if now is None else now,
                )
                self._commit({**self._records, key_id: record})
            return record

    def _commit(self, records: dict[str, ClientKeyRecord]) -> None:
        # Commit to disk before making the new state visible or returning success.
        records = {key: value for key, value in records.items() if value.expires_at > time.time()}
        revision = self._revision + 1
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        with os.fdopen(
            os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8"
        ) as output:
            json.dump(
                {"revision": revision, "keys": [r.to_json() for r in records.values()]}, output
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self._path)
        self._records, self._revision = records, revision
