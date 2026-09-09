"""v0.2 security primitives for the agent.

The agent is the install's trust root. This module owns:

  - **Passphrase verification** via Argon2id. The operator's passphrase
    set in the first-run wizard is hashed (Argon2id) and stored in
    `agent.yaml`. Login compares against this hash.

  - **Master-key derivation** via Argon2id raw mode. The same
    passphrase, with a separately-stored salt, deterministically
    derives a 32-byte key the agent uses to encrypt sensitive
    config fields on each child component (libsodium secretbox).
    The master key NEVER lands on disk in plaintext — it lives in
    process memory after derivation, and (optionally) in the OS
    secret store when `securityMode == os_keyring`.

  - **Session token issuance + validation** via JWT-HS256. The
    agent generates a per-restart signing key, distributes it to
    spawned children via env var, and every component validates
    bearer tokens independently using the shared key. Restarting
    the agent rotates signing keys which invalidates all
    existing tokens — good-enough revocation for v0.2.

  - **Service token issuance** for component-internal calls.
    Same JWT shape as session tokens; different `aud` claim
    (`service:<kind>`) so a leaked service token can't be used
    against the UI surface and vice versa.

  - **At-rest envelope encryption** via libsodium secretbox.
    `seal()` produces `MasterKeyEnvelope` records; `open()` inverts.
    Used by children to encrypt apiKey-style fields on disk; the
    agent itself doesn't store anything sensitive that needs
    this (the master key is in memory, the passphrase hash isn't
    reversible).
"""

from __future__ import annotations

import base64
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any

import argon2
import argon2.low_level
import jwt
import nacl.exceptions
import nacl.secret
import nacl.utils

log = logging.getLogger(__name__)

# Argon2id parameters — current (2026) OWASP recommendations for
# interactive logins. Bumped from defaults to err on the side of
# expensive: this only runs on login (once per session start) and
# install setup. Memory is the dominant cost.
_ARGON2_TIME_COST = 3
_ARGON2_MEMORY_COST = 65_536  # KiB
_ARGON2_PARALLELISM = 4
_ARGON2_HASH_LEN = 32  # bytes — drives the secretbox key length

# JWT algorithm + lifetime. HS256 because the signing key is shared
# with children that need to validate independently; asymmetric
# would force every child to do a agent roundtrip or hold a
# public key, neither of which simplifies the v0.2 model.
_JWT_ALG = "HS256"
_DEFAULT_SESSION_TTL_SECONDS = 14 * 24 * 3600  # 14 days

# Audience claim values.
AUDIENCE_OPERATOR = "operator"
SERVICE_AUDIENCE_PREFIX = "service:"


# --------------------------------------------------------------------------- #
# Passphrase
# --------------------------------------------------------------------------- #


_password_hasher = argon2.PasswordHasher(
    time_cost=_ARGON2_TIME_COST,
    memory_cost=_ARGON2_MEMORY_COST,
    parallelism=_ARGON2_PARALLELISM,
    hash_len=_ARGON2_HASH_LEN,
)


def hash_passphrase(passphrase: str) -> str:
    """Return an Argon2id hash of the passphrase suitable for storage.

    The returned string is the full self-describing PHC format
    (`$argon2id$v=19$m=...,t=...,p=...$salt$hash`) — parameters and
    salt are embedded so future verifications don't need any
    out-of-band context."""
    if not passphrase:
        raise ValueError("passphrase must not be empty")
    return _password_hasher.hash(passphrase)


def verify_passphrase(passphrase: str, stored_hash: str) -> bool:
    """Constant-time verification. Returns True on match."""
    try:
        _password_hasher.verify(stored_hash, passphrase)
        return True
    except argon2.exceptions.VerifyMismatchError:
        return False
    except argon2.exceptions.InvalidHashError:
        log.warning("stored passphrase hash is malformed; treating as no-match")
        return False


# --------------------------------------------------------------------------- #
# Master-key derivation
# --------------------------------------------------------------------------- #


def generate_master_key_salt() -> bytes:
    """16 random bytes. Stored alongside the passphrase hash in
    `agent.yaml` as base64. Per-install; persists across
    restarts so the master key derived from the same passphrase is
    stable."""
    return secrets.token_bytes(16)


def derive_master_key(passphrase: str, salt: bytes) -> bytes:
    """Deterministic 32-byte key from passphrase + salt via Argon2id.

    Same input always produces the same output, so the master key
    is recoverable from the operator's passphrase without anything
    secret on disk. The agent runs this once at startup and
    holds the result in memory.
    """
    if not passphrase:
        raise ValueError("passphrase must not be empty")
    if len(salt) < 8:
        raise ValueError("salt must be at least 8 bytes")
    return argon2.low_level.hash_secret_raw(
        secret=passphrase.encode("utf-8"),
        salt=salt,
        time_cost=_ARGON2_TIME_COST,
        memory_cost=_ARGON2_MEMORY_COST,
        parallelism=_ARGON2_PARALLELISM,
        hash_len=_ARGON2_HASH_LEN,
        type=argon2.low_level.Type.ID,
    )


# --------------------------------------------------------------------------- #
# At-rest envelope (libsodium secretbox)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Envelope:
    """Canonical shape of an at-rest encrypted secret. Matches the
    `MasterKeyEnvelope` schema in common.yaml."""

    alg: str
    nonce: str  # base64
    ciphertext: str  # base64

    def to_dict(self) -> dict[str, str]:
        return {"alg": self.alg, "nonce": self.nonce, "ciphertext": self.ciphertext}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Envelope:
        alg = raw.get("alg")
        nonce = raw.get("nonce")
        ciphertext = raw.get("ciphertext")
        if alg != "secretbox-xsalsa20poly1305":
            raise ValueError(f"unsupported envelope alg: {alg!r}")
        if not isinstance(nonce, str) or not isinstance(ciphertext, str):
            raise ValueError("envelope nonce/ciphertext must be base64 strings")
        return cls(alg=alg, nonce=nonce, ciphertext=ciphertext)


def is_envelope(value: Any) -> bool:
    """Quick check — is this a `dict` shaped like an Envelope? Used by
    config loaders that need to decide "decrypt vs. take as plaintext"
    on each field read."""
    return (
        isinstance(value, dict)
        and value.get("alg") == "secretbox-xsalsa20poly1305"
        and "nonce" in value
        and "ciphertext" in value
    )


def seal(plaintext: str, master_key: bytes) -> Envelope:
    """Encrypt a plaintext value into a v0.2 envelope.

    Generates a fresh 24-byte nonce per call (never reuse). The
    returned `Envelope` is JSON-serializable via `to_dict()`."""
    if len(master_key) != 32:
        raise ValueError("master key must be 32 bytes")
    box = nacl.secret.SecretBox(master_key)
    nonce = nacl.utils.random(nacl.secret.SecretBox.NONCE_SIZE)
    ciphertext = box.encrypt(plaintext.encode("utf-8"), nonce).ciphertext
    return Envelope(
        alg="secretbox-xsalsa20poly1305",
        nonce=base64.b64encode(nonce).decode("ascii"),
        ciphertext=base64.b64encode(ciphertext).decode("ascii"),
    )


def open_envelope(envelope: Envelope, master_key: bytes) -> str:
    """Decrypt back to plaintext. Raises ValueError on bad key /
    tampered ciphertext."""
    if len(master_key) != 32:
        raise ValueError("master key must be 32 bytes")
    try:
        nonce = base64.b64decode(envelope.nonce, validate=True)
        ciphertext = base64.b64decode(envelope.ciphertext, validate=True)
    except Exception as e:
        raise ValueError(f"envelope decoding failed: {e}") from e
    box = nacl.secret.SecretBox(master_key)
    try:
        return box.decrypt(ciphertext, nonce).decode("utf-8")
    except nacl.exceptions.CryptoError as e:
        raise ValueError(f"envelope decryption failed: {e}") from e


# --------------------------------------------------------------------------- #
# JWT session + service tokens
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TokenPayload:
    """Decoded JWT claims. `iat` / `exp` are unix seconds."""

    sub: str  # "operator" for operator sessions; component kind for service tokens
    aud: str
    iat: int
    exp: int


def generate_signing_key() -> bytes:
    """32 random bytes — used as the HMAC key for JWT signing.

    The agent generates one of these at every startup and
    distributes it to children via env var. Restarting the
    agent rotates the key, invalidating all existing tokens
    (good-enough v0.2 revocation)."""
    return secrets.token_bytes(32)


def issue_operator_token(
    *,
    signing_key: bytes,
    ttl_seconds: int = _DEFAULT_SESSION_TTL_SECONDS,
    now: int | None = None,
) -> tuple[str, int]:
    """Issue a session token for the UI-authenticated operator.

    Returns `(token, exp_unix_seconds)`. The operator's `sub` is the
    literal string `"operator"` — v0.2 is single-user, but the claim
    is structured so v0.3+ can add `sub: "operator:<id>"` for
    multi-user without re-shaping the token.
    """
    issued_at = now if now is not None else int(time.time())
    expires_at = issued_at + ttl_seconds
    claims = {
        "sub": "operator",
        "aud": AUDIENCE_OPERATOR,
        "iat": issued_at,
        "exp": expires_at,
    }
    token = jwt.encode(claims, signing_key, algorithm=_JWT_ALG)
    return token, expires_at


def issue_service_token(
    *,
    signing_key: bytes,
    kind: str,
    ttl_seconds: int | None = None,
    now: int | None = None,
) -> str:
    """Issue a long-lived service token for one supervised component.

    Threaded via env var to spawned children. Lifetime defaults to
    one year — long enough that a child can run without forced
    re-auth, short enough that a leaked one expires. Agent
    restart rotates the signing key anyway, so the effective
    lifetime is bounded by agent uptime.

    The `kind` is the component class (`gateway`, `inference-driver`).
    Encoded as `aud: "service:<kind>"` so components can additionally
    check the audience matches their own kind on inbound calls — a
    leaked driver service token can't be used against the gateway.
    """
    issued_at = now if now is not None else int(time.time())
    ttl = ttl_seconds if ttl_seconds is not None else 365 * 24 * 3600
    expires_at = issued_at + ttl
    claims = {
        "sub": kind,
        "aud": f"{SERVICE_AUDIENCE_PREFIX}{kind}",
        "iat": issued_at,
        "exp": expires_at,
    }
    return jwt.encode(claims, signing_key, algorithm=_JWT_ALG)


def decode_token(
    *,
    token: str,
    signing_key: bytes,
    expected_audience: str | None = None,
    now: int | None = None,
) -> TokenPayload:
    """Verify a token's signature + expiry and return its claims.

    Raises:
      jwt.ExpiredSignatureError — token expired
      jwt.InvalidAudienceError  — audience mismatch
      jwt.InvalidTokenError     — signature failure or malformed claims
    """
    options: dict[str, Any] = {"require": ["sub", "aud", "iat", "exp"]}
    decode_kwargs: dict[str, Any] = {
        "key": signing_key,
        "algorithms": [_JWT_ALG],
        "options": options,
    }
    if expected_audience is not None:
        decode_kwargs["audience"] = expected_audience
    else:
        # No expected audience to match against — the caller validates the
        # `aud` claim itself (e.g. "operator OR any service:*"). PyJWT would
        # otherwise raise InvalidAudienceError for a token that carries an
        # `aud` claim when no `audience` is supplied, so disable its check.
        # `aud` is still required-present via the `require` list above.
        options["verify_aud"] = False
    if now is not None:
        # leeway is in seconds; we pass `now` via `leeway` is awkward —
        # PyJWT validates against time.time() internally. Tests using
        # the `now` parameter validate by setting the iat/exp explicitly.
        pass
    claims = jwt.decode(token, **decode_kwargs)
    return TokenPayload(
        sub=str(claims["sub"]),
        aud=str(claims["aud"]),
        iat=int(claims["iat"]),
        exp=int(claims["exp"]),
    )
