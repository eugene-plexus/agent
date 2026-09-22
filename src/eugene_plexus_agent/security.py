"""Token minter for the install: Ed25519 for new keys and rotations.
Trusted agents/control retain private PEM; children receive public PEM.
Existing 32-byte HS256 keys remain usable until an explicit rotation.
The token-signing key is independent of the master encryption key.
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
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

log = logging.getLogger(__name__)

# Argon2id parameters — current (2026) OWASP recommendations for
# interactive logins. Bumped from defaults to err on the side of
# expensive: this only runs on login (once per session start) and
# install setup. Memory is the dominant cost.
_ARGON2_TIME_COST = 3
_ARGON2_MEMORY_COST = 65_536  # KiB
_ARGON2_PARALLELISM = 4
_ARGON2_HASH_LEN = 32  # bytes — drives the secretbox key length


def validate_signing_key(key: bytes) -> None:
    """Minters accept Ed25519 PKCS8 PEM or the existing legacy HMAC key."""
    if len(key) == 32:
        return
    parsed = serialization.load_pem_private_key(key, password=None)
    if not isinstance(parsed, Ed25519PrivateKey):
        raise ValueError("token signing requires an Ed25519 private key")


def signing_algorithm(key: bytes) -> str:
    validate_signing_key(key)
    return "HS256" if len(key) == 32 else "EdDSA"


def verification_key(key: bytes) -> bytes:
    """Validate public Ed25519 PEM, or an explicitly legacy 32-byte HMAC key."""
    if len(key) == 32:
        return key
    if key.startswith(b"-----BEGIN PRIVATE KEY-----"):
        parsed_private = serialization.load_pem_private_key(key, password=None)
        if not isinstance(parsed_private, Ed25519PrivateKey):
            raise ValueError("token signing requires an Ed25519 private key")
        key = parsed_private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    parsed = serialization.load_pem_public_key(key)
    if not isinstance(parsed, Ed25519PublicKey):
        raise ValueError("token verification requires an Ed25519 public key")
    return key


def verification_algorithm(key: bytes) -> str:
    """Select from trusted key material, never an untrusted JWT header."""
    return "HS256" if len(key) == 32 else "EdDSA"


_DEFAULT_SESSION_TTL_SECONDS = 14 * 24 * 3600  # 14 days

# Audience claim values.
AUDIENCE_OPERATOR = "operator"
SERVICE_AUDIENCE_PREFIX = "service:"

AUDIENCE_CLIENT = "client"
"""A long-lived key an app outside the install holds (S4, 2026-09-15).

Deliberately neither `operator` nor a `service:` audience, because every
check in every component tests for one of those two. So a client key is
refused by this agent, by the control root, by the library and by the
gateway's own config, admin and metrics paths **without any of them
being taught about it** -- the narrowing comes from the shape of the
claim, not from a list someone has to remember to update. Exactly one
place opts in: the gateway's three OpenAI-compatible paths.
"""

_DEFAULT_CLIENT_TTL_SECONDS = 365 * 24 * 3600


# --------------------------------------------------------------------------- #
# Passphrase
# --------------------------------------------------------------------------- #


MIN_PASSPHRASE_LENGTH = 12
"""The shortest passphrase `POST /v1/auth/initialize` will set, in characters.

One character was enough until 2026-09-22. This is the one secret that
stands between the network and every key the install holds -- the
Argon2id hash in `agent.yaml` is an offline guessing target for anyone
who ever reads that file -- and it cannot be recovered, so the only time
to ask for a real one is when it is chosen. Twelve because a few
ordinary words clear it without anyone reaching for symbols.

**Enforced where a passphrase is chosen, never where one is used.**
Login does not check it: an install set up before this existed has a
shorter passphrase, and refusing it would lock its operator out of their
own machine. The UI's wizard and the control root use the same number;
if it moves, it moves in all three.
"""


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
    jti: str | None = None
    """The key's id, on a client key. Absent on every other token.

    Not in the `require` list: operator sessions and service tokens have
    never carried one, and demanding it would refuse every token minted
    before 2026-09-15 -- including the one the caller is holding while
    they read this.
    """


def generate_signing_key() -> bytes:
    """New installs and explicit rotations use Ed25519, never a new HMAC key."""
    return Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


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
    token = jwt.encode(claims, signing_key, algorithm=signing_algorithm(signing_key))
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
    return jwt.encode(claims, signing_key, algorithm=signing_algorithm(signing_key))


def issue_client_token(
    *,
    signing_key: bytes,
    key_id: str,
    name: str,
    ttl_seconds: int = _DEFAULT_CLIENT_TTL_SECONDS,
    now: int | None = None,
) -> tuple[str, int]:
    """Mint the bearer an OpenAI-compatible client outside the install holds.

    Returns `(token, exp_unix_seconds)`. Signed with the same install
    key as everything else, so no component needs a second key to verify
    it; what makes it safe to hand out is the `aud`, which only the
    gateway's front door accepts.

    `sub` is the operator's name for the key, so a token decoded by hand
    during a support conversation says what it is for. `jti` is the
    record's id: the only claim the gateway needs in order to refuse a
    revoked one.
    """
    if not key_id:
        raise ValueError("key_id must not be empty")
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    issued_at = now if now is not None else int(time.time())
    expires_at = issued_at + ttl_seconds
    claims = {
        "sub": name or "client",
        "aud": AUDIENCE_CLIENT,
        "iat": issued_at,
        "exp": expires_at,
        "jti": key_id,
    }
    token = jwt.encode(claims, signing_key, algorithm=signing_algorithm(signing_key))
    return token, expires_at


CLOCK_SKEW_LEEWAY_SECONDS = 300
"""How far apart two hosts' clocks may drift before a token is refused.

Five minutes: Kerberos's `MaxClockSkew`, and the window Entra and most
OAuth validators apply to `iat`, `nbf` and `exp`. **It was zero until
2026-09-15.** On the live two-machine install the control root's clock
ran half a second ahead of a worker whose Windows Time service had
stopped, and every token the root minted in the first half of each
second was refused by that worker as "not yet valid (iat)" a few
milliseconds later. Long-lived tokens (the gateway's, the operator's
session) passed, every health check said ok, and the root listed the
node `down` with no reason -- so it read as a key or enrollment fault
and was neither. A skew large enough to matter for security is a
broken clock; a broken clock is *reported* (`_note_clock_skew`), not
enforced by refusing traffic between two healthy hosts.
"""

_SKEW_WARN_AFTER_SECONDS = 2.0
_SKEW_WARN_INTERVAL_SECONDS = 60.0
_last_skew_warning = 0.0


def _note_clock_skew(iat: int, *, now: float | None = None) -> None:
    """Warn, at most once a minute, when a token was issued in this host's future.

    Accepted within `CLOCK_SKEW_LEEWAY_SECONDS`, so nothing breaks. Logged
    so a wrong clock on either host is visible long before the skew grows
    past the leeway and starts refusing traffic.
    """
    global _last_skew_warning
    current = time.time() if now is None else now
    ahead = iat - current
    if ahead <= _SKEW_WARN_AFTER_SECONDS:
        return
    if current - _last_skew_warning < _SKEW_WARN_INTERVAL_SECONDS:
        return
    _last_skew_warning = current
    log.warning(
        "accepted a token issued %.1f s in this host's future: the issuer's clock or "
        "this host's is wrong (tolerated up to %d s, then tokens are refused)",
        ahead,
        CLOCK_SKEW_LEEWAY_SECONDS,
    )


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
        "key": verification_key(signing_key),
        "algorithms": [verification_algorithm(signing_key)],
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
    decode_kwargs["leeway"] = CLOCK_SKEW_LEEWAY_SECONDS
    claims = jwt.decode(token, **decode_kwargs)
    _note_clock_skew(int(claims["iat"]))
    raw_jti = claims.get("jti")
    return TokenPayload(
        sub=str(claims["sub"]),
        aud=str(claims["aud"]),
        iat=int(claims["iat"]),
        exp=int(claims["exp"]),
        jti=str(raw_jti) if raw_jti is not None else None,
    )
