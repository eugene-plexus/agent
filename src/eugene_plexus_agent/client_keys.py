"""The standalone registry: a machine that has joined no install keeps its keys here.

Only metadata is stored; bearer tokens are returned once at creation. A corrupt
or unwritable registry is unavailable, never silently replaced with an empty one.
"""

from __future__ import annotations

import json
import logging
import math
import secrets
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._private_files import write_private
from .client_admission import AdmissionClock, decide, validate_ledger, validate_limits

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
    limits: dict[str, Any] | None = None
    # Row 24. `lineage` names the bucket the key counts in; a replacement joins
    # the key it replaced. `last_used_at` is to the minute.
    last_used_at: float | None = None
    retires_at: float | None = None
    replaces: str | None = None
    replaced_by: str | None = None
    lineage: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "tail": self.tail,
            "createdAt": self.created_at,
            "expiresAt": self.expires_at,
        }
        if self.limits is not None:
            out["limits"] = self.limits
        if self.revoked_at is not None:
            out["revokedAt"] = self.revoked_at
        for name, value in (
            ("lastUsedAt", self.last_used_at),
            ("retiresAt", self.retires_at),
            ("replaces", self.replaces),
            ("replacedBy", self.replaced_by),
            ("lineage", self.lineage),
        ):
            if value is not None:
                out[name] = value
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
        stamps: list[float | None] = []
        for field in ("lastUsedAt", "retiresAt"):
            try:
                stamps.append(float(raw[field]) if raw.get(field) is not None else None)
            except (TypeError, ValueError):
                return None
        if any(s is not None and not math.isfinite(s) for s in stamps):
            return None
        links = [raw.get(n) for n in ("replaces", "replacedBy", "lineage")]
        if any(v is not None and (not isinstance(v, str) or not 0 < len(v) <= 128) for v in links):
            return None
        if not key_id or not name or len(key_id) > 128 or len(name) > 64 or len(tail) > 6:
            return None
        try:
            limits = validate_limits(raw.get("limits"))
        except ValueError:
            return None
        return cls(
            id=key_id,
            name=name,
            tail=tail,
            created_at=created,
            expires_at=expires,
            revoked_at=revoked,
            limits=limits,
            last_used_at=stamps[0],
            retires_at=stamps[1],
            replaces=links[0],
            replaced_by=links[1],
            lineage=links[2],
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
        self._admission = validate_ledger(None)
        self._admission_clock = AdmissionClock()
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
                self._admission = validate_ledger(raw.get("admission"))
                self._admission_clock = AdmissionClock(self._admission["clock"])
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
                record = replace(record, revoked_at=time.time() if now is None else now)
                self._commit({**self._records, key_id: record})
            return record

    def set_limits(self, key_id: str, limits: dict[str, Any]) -> ClientKeyRecord | None:
        with self._lock:
            self._check()
            record = self._records.get(key_id)
            if record is not None:
                clean = validate_limits(limits)
                record = replace(record, limits=clean)
                records = {**self._records, key_id: record}
                # A rotated key's tokens share one allowance, so one policy.
                for other_id, other in self._records.items():
                    if (
                        other_id != key_id
                        and other.revoked_at is None
                        and (other.lineage or other_id) == (record.lineage or key_id)
                    ):
                        records[other_id] = replace(other, limits=clean)
                self._commit(records)
            return record

    def rotate(
        self, old_id: str, new: ClientKeyRecord, *, retires_at: float
    ) -> ClientKeyRecord | None:
        """Add `new` as `old_id`'s replacement. None for an unknown key.

        Raises ValueError, naming the cause, for a key that is off or replaced.
        The replacement takes the old key's name and limits, and joins its
        allowance bucket.
        """
        with self._lock:
            self._check()
            old = self._records.get(old_id)
            if old is None:
                return None
            if old.revoked_at is not None:
                raise ValueError("This key is turned off.")
            if old.expires_at <= time.time():
                raise ValueError("This key has expired.")
            if old.replaced_by is not None:
                raise ValueError(f"This key is already replaced by ...{old.replaced_by[-6:]}.")
            new = replace(
                new,
                name=old.name,
                limits=old.limits,
                replaces=old_id,
                lineage=old.lineage or old_id,
                revoked_at=None,
                last_used_at=None,
                retires_at=None,
                replaced_by=None,
            )
            self._commit(
                {
                    **self._records,
                    old_id: replace(old, retires_at=retires_at, replaced_by=new.id),
                    new.id: new,
                }
            )
            return new

    def _minute_after(self, key: ClientKeyRecord | None, action: str) -> float | None:
        if key is None or action not in ("check", "acquire"):
            return None
        minute = float(int(time.time() // 60) * 60)
        return minute if key.last_used_at is None or key.last_used_at < minute else None

    def admit(
        self,
        *,
        key_id: str,
        action: str,
        request_id: str,
        model: str | None,
        purpose: str = "inference",
    ) -> dict[str, Any]:
        with self._lock:
            self._check()
            key = self._records.get(key_id)
            result, candidate = decide(
                self._admission,
                key.to_json() if key else None,
                key_id=key_id,
                action=action,
                request_id=request_id,
                model=model,
                now=self._admission_clock.now(self._admission["clock"]),
                purpose=purpose,
            )
            used = self._minute_after(key, action)
            records = self._records
            if used is not None and key is not None:
                records = {**records, key_id: replace(key, last_used_at=used)}
            if candidate is not None:
                self._commit(records, admission=candidate)
            elif used is not None:
                # A check writes the registry at most once a minute for each key.
                self._commit(records)
            return result

    def _commit(
        self, records: dict[str, ClientKeyRecord], *, admission: dict[str, Any] | None = None
    ) -> None:
        # Commit to disk before making the new state visible or returning success.
        records = {key: value for key, value in records.items() if value.expires_at > time.time()}
        revision = self._revision + 1
        # `write_private` rather than the `os.open(..., O_TRUNC, 0o600)`
        # this used to spell out: a fixed `.tmp` name opened with
        # `O_TRUNC` keeps the mode of a leftover from a crash, so one
        # stray 0644 temp made every later registry 0644.
        write_private(
            self._path,
            json.dumps(
                {
                    "revision": revision,
                    "keys": [r.to_json() for r in records.values()],
                    "admission": self._admission if admission is None else admission,
                }
            ),
        )
        self._records, self._revision = records, revision
        if admission is not None:
            self._admission = admission
