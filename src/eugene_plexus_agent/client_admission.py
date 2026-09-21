"""Durable shared admission. Mirrored in the standalone agent; no request content."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime
from typing import Any

LEASE_SECONDS = 30.0
WINDOW_SECONDS = 60.0
MAX_ENTRIES = 50_000


class AdmissionRefusal(Exception):
    def __init__(self, status: int, detail: str, retry: int | None = None):
        self.status, self.detail, self.retry = status, detail, retry
        super().__init__(detail)


def validate_limits(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.keys() - {
        "allowedModels",
        "maxConcurrentRequests",
        "requestsPerMinute",
        "localOnly",
    }:
        raise ValueError("invalid client limits")
    allowed = value.get("allowedModels")
    if "localOnly" in value and not isinstance(value["localOnly"], bool):
        raise ValueError("localOnly must be a boolean")
    if allowed is not None and (
        not isinstance(allowed, list)
        or len(allowed) > 100
        or any(not isinstance(s, str) or not s.strip() or len(s) > 256 for s in allowed)
        or len(set(allowed)) != len(allowed)
    ):
        raise ValueError("invalid allowed models")
    concurrency, rate = value.get("maxConcurrentRequests", 2), value.get("requestsPerMinute", 60)
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("invalid concurrent request limit")
    if type(rate) is not int or not 1 <= rate <= 10_000:
        raise ValueError("invalid request rate limit")
    return {
        **({"localOnly": True} if value.get("localOnly") else {}),
        "allowedModels": allowed,
        "maxConcurrentRequests": concurrency,
        "requestsPerMinute": rate,
    }


def clean_buckets(buckets: dict[str, Any], now: float) -> dict[str, Any]:
    return {
        key: kept
        for key, entries in buckets.items()
        if (
            kept := {
                rid: dict(e)
                for rid, e in entries.items()
                if e["started"] > now - WINDOW_SECONDS or (e["active"] and e["until"] > now)
            }
        )
    }


def validate_ledger(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"clock": 0.0, "buckets": {}}
    if not isinstance(raw, dict) or set(raw) != {"clock", "buckets"}:
        raise ValueError("invalid admission state")
    now, buckets = raw["clock"], raw["buckets"]
    if (
        type(now) not in (int, float)
        or not math.isfinite(now)
        or now < 0
        or not isinstance(buckets, dict)
    ):
        raise ValueError("invalid admission clock/buckets")
    count = 0
    for key, entries in buckets.items():
        if not isinstance(key, str) or not isinstance(entries, dict):
            raise ValueError("invalid admission bucket")
        for rid, entry in entries.items():
            count += 1
            if (
                not isinstance(rid, str)
                or not isinstance(entry, dict)
                or set(entry) != {"started", "until", "active", "charged", "model", "policyDigest"}
            ):
                raise ValueError("invalid admission reservation")
            if any(
                type(entry[k]) not in (int, float) or not math.isfinite(entry[k]) or entry[k] < 0
                for k in ("started", "until")
            ):
                raise ValueError("invalid reservation time")
            if (
                type(entry["active"]) is not bool
                or type(entry["charged"]) is not bool
                or not isinstance(entry["model"], str)
                or not isinstance(entry["policyDigest"], str)
            ):
                raise ValueError("invalid reservation fields")
    if count > MAX_ENTRIES:
        raise ValueError("admission state exceeds its bound")
    return raw


class AdmissionClock:
    """A restart pauses the logical clock; UTC jumps cannot replenish allowance."""

    def __init__(self, persisted: float = 0.0) -> None:
        self.base, self.started = persisted, time.perf_counter()

    def now(self, persisted: float) -> float:
        current = self.base + time.perf_counter() - self.started
        if persisted > current:  # A promoted replica may have replayed newer state.
            self.base, self.started = persisted, time.perf_counter()
            return persisted
        return current


def decide(
    ledger: dict[str, Any],
    key: dict[str, Any] | None,
    *,
    key_id: str,
    action: str,
    request_id: str,
    model: str | None,
    now: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """No mutation until the caller persists the returned candidate atomically."""
    limits = validate_limits(key.get("limits")) if key else None
    result: dict[str, Any] = {"keyId": key_id, "keyName": key["name"] if key else ""}
    if limits is not None:
        result["limits"] = limits
    buckets = clean_buckets(ledger["buckets"], now)
    bucket = buckets.setdefault(key_id, {})
    existing = bucket.get(request_id)
    if action != "release":
        expiry = key.get("expiresAt") if key else None
        expires = (
            datetime.fromisoformat(expiry.replace("Z", "+00:00")).timestamp()
            if isinstance(expiry, str)
            else expiry
        )
        if not key or key.get("revokedAt") is not None or not expires or expires <= time.time():
            raise AdmissionRefusal(401, "This client key is unregistered, expired or revoked.")
    if action == "check":
        return result, None
    digest = hashlib.sha256(json.dumps(limits, sort_keys=True).encode()).hexdigest()
    if action == "release":
        if existing is not None:
            existing["active"] = False
        elif sum(len(b) for b in buckets.values()) < MAX_ENTRIES:
            # A cancelled acquisition can reach the authority after its release.
            bucket[request_id] = {
                "started": now,
                "until": now,
                "active": False,
                "charged": False,
                "model": "",
                "policyDigest": "",
            }
    elif action == "renew":
        if not existing or not existing["active"] or existing["until"] <= now:
            raise AdmissionRefusal(409, "The request reservation expired; start a new request.")
        if existing["policyDigest"] != digest:
            raise AdmissionRefusal(
                409, "The key's permissions or limits changed; start a new request."
            )
        existing["until"] = now + LEASE_SECONDS
        result["leaseSeconds"] = LEASE_SECONDS
    elif action == "acquire":
        if not model:
            raise AdmissionRefusal(422, "A model is required for inference admission.")
        allowed = limits.get("allowedModels") if limits else None
        if allowed is not None and model not in allowed:
            raise AdmissionRefusal(403, "This client key does not permit the requested model.")
        if existing is not None:
            if (
                not existing["active"]
                or existing["until"] <= now
                or existing["model"] != model
                or existing["policyDigest"] != digest
            ):
                raise AdmissionRefusal(409, "This request identifier is no longer usable.")
            result["leaseSeconds"] = min(LEASE_SECONDS, existing["until"] - now)
            return result, None
        if limits is not None:
            active = [e for e in bucket.values() if e["active"] and e["until"] > now]
            recent = [
                e for e in bucket.values() if e["charged"] and e["started"] > now - WINDOW_SECONDS
            ]
            if len(active) >= limits["maxConcurrentRequests"]:
                retry = max(1, math.ceil(min(e["until"] for e in active) - now))
                raise AdmissionRefusal(
                    429,
                    "This key's shared concurrent request limit is reached. "
                    "Wait for an active request to finish or cancel it.",
                    retry,
                )
            if len(recent) >= limits["requestsPerMinute"]:
                retry = max(1, math.ceil(min(e["started"] for e in recent) + WINDOW_SECONDS - now))
                raise AdmissionRefusal(
                    429, "This key's shared rolling-minute request limit is reached.", retry
                )
        if sum(len(b) for b in buckets.values()) >= MAX_ENTRIES:
            raise AdmissionRefusal(503, "Admission is at its bounded capacity; retry shortly.", 30)
        bucket[request_id] = {
            "started": now,
            "until": now + LEASE_SECONDS,
            "active": True,
            "charged": True,
            "model": model,
            "policyDigest": digest,
        }
        result["leaseSeconds"] = LEASE_SECONDS
    else:
        raise AdmissionRefusal(422, "Unknown admission operation.")
    return result, {"clock": now, "buckets": buckets}
