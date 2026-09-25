"""The agent's secrets: the passphrase, the master key, and at-rest envelopes.

Tokens are minted and verified in `tokens` and `trust` (per-node token
keys, 2026-09-25). The token key is independent of the master key.
"""

from __future__ import annotations

import base64
import logging
import secrets
from dataclasses import dataclass
from typing import Any

import argon2
import argon2.low_level
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
