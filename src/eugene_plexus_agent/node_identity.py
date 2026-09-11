"""This host's identity in an install: its keypair, its enrollment, the
install's signing key, and where other hosts reach it.

M5 contracted the exchange and built the control root's half; the agent
never had a half. This is it. One file, `node.yaml`, beside `agent.yaml`:

    name: gpu-box
    privateKey: <base64 X25519>        never leaves this host
    publicKey: <base64 X25519>
    signingPrivateKey: <base64 Ed25519 seed>   signs address announcements
    signingPublicKey: <base64 Ed25519>
    advertiseSequence: 3               strictly increasing, mirrored at the root
    controlUrl: http://100.64.0.1:8083
    controlPublicKey: <base64 Ed25519>  the root's identity, checked on every re-key
    recoveryPublicKey: <base64 X25519>  the second recipient of anything sealed here
    signingKey: <base64>                THE INSTALL'S service-token signing key
    signingKeyId: "1"
    epoch: 1                            highest control-root epoch accepted
    advertiseUrl: http://100.64.0.7:8079
    enrolledAt: 2026-09-10T22:14:03+00:00

**The signing key is stored in the clear, mode 0600, on purpose.** The
agent hands it to every component it spawns in their environment, so a
process compromise on this host already yields it; sealing it on disk
would protect it from nothing that matters while making a headless GPU
box unable to spawn a verifiable companion at boot until someone typed a
passphrase *at that host*. The node's private key lives here for the
same reason — what it protects is secrets sealed *to this node*, whose
plaintext also rides in children's environments. This file is therefore
exactly as sensitive as the agent's process environment, and the mesh VPN
remains the network boundary. Separate from `agent.yaml` because that
file is topology an operator edits (and has already leaked once via
`git add -A`); identity is never edited by hand.

**Epoch fencing lives here.** `accept_rekey` refuses an epoch below the
highest recorded, and an equal epoch with a lower key generation — a
superseded control root, or a replayed rotation, is refused on this host
alone, with no election and no agreement with any other agent. That is
the whole of M5 §6's mechanism, on the side that does the fencing.

**The node signs too, and for the mirror-image reason.** An address
announcement (`PATCH /v1/nodes/{name}` at the root) is signed with this
node's Ed25519 key, because a service token names a *kind* and not a
host, and because the case that matters is an unattended reboot with no
operator to authenticate. So: the root proves itself to a node with its
identity key, a node proves itself to the root with its own, and neither
uses a bearer. That needs a **second** keypair — the one above is X25519,
minted so secrets can be sealed *to* this node, and X25519 does not sign;
the derivation between them only runs Ed25519 to X25519, which is the
direction we do not have.

**The re-key's credential is a signature, not a bearer.** A rotation
invalidates every service token in the install, and a re-run of an
interrupted one cannot know which key each node still holds; the control
root's identity key does not rotate, which is what makes it the one
credential that survives the operation. The canonical message is three
fields through one serializer, stated in `agent.yaml`'s `RekeyRequest`
so both sides implement it from the same sentence.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import logging
import os
import socket
import stat
import sys
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import nacl.exceptions
import nacl.public
import nacl.signing
import yaml

log = logging.getLogger(__name__)

NODE_FILE = "node.yaml"

# How long the advertise-host derivation waits on a TCP connect to the
# control root. It is a routing question, not a request; anything slower
# than this is the root being unreachable, and enrollment is about to
# discover that anyway.
_DERIVE_TIMEOUT_SECONDS = 3.0


class FencedError(Exception):
    """A re-key was refused because it would move the epoch backwards, or
    replay a superseded key generation. The caller answers 409."""


# --------------------------------------------------------------------------- #
# The signed re-key message
# --------------------------------------------------------------------------- #


def rekey_message(*, signing_key: str, signing_key_id: str, epoch: int) -> bytes:
    """The canonical bytes both sides sign and verify.

    Exactly the form `RekeyRequest.signature` describes: the JSON object
    with keys sorted and no whitespace. Three fields, one serializer — a
    fourth field or a space anywhere and every rotation in the install
    fails to verify.
    """
    return json.dumps(
        {"epoch": epoch, "signingKey": signing_key, "signingKeyId": signing_key_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def address_message(*, name: str, sequence: int, url: str) -> bytes:
    """The canonical bytes an address announcement is signed over —
    `NodeAddressAnnouncement.signature` in `control.yaml`, byte for byte.

    The mirror of `rekey_message`: three fields, keys sorted, no
    whitespace. `url` goes in **exactly as it will be sent**, before any
    normalization, because the root verifies against the raw request body
    for precisely this reason. Sign a parsed URL and every announcement
    fails with what looks like a crypto error.
    """
    return json.dumps(
        {"name": name, "sequence": sequence, "url": url},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sign_address(*, signing_private_key: str, message: bytes) -> str:
    """Detached Ed25519 signature by this node's identity signing key."""
    seed = base64.b64decode(signing_private_key, validate=True)
    return base64.b64encode(nacl.signing.SigningKey(seed).sign(message).signature).decode("ascii")


def verify_rekey_signature(*, control_public_key: str, message: bytes, signature: str) -> bool:
    """True iff `signature` is the control identity's detached Ed25519
    signature over `message`. Every malformed input is simply False —
    the caller has one answer for "not from a root I recognise"."""
    try:
        verify_key = nacl.signing.VerifyKey(base64.b64decode(control_public_key, validate=True))
        verify_key.verify(message, base64.b64decode(signature, validate=True))
    except (nacl.exceptions.CryptoError, binascii.Error, ValueError, TypeError):
        return False
    return True


def sign_rekey_message(*, control_private_key: str, message: bytes) -> str:
    """The control root's side of the exchange.

    Here as well as in `control` because the agent's tests need to forge
    messages with a key they generated, and because "components share
    schemas, not code" means the two implementations are checked against
    each other by the acceptance run rather than by an import.
    """
    seed = base64.b64decode(control_private_key, validate=True)
    signed = nacl.signing.SigningKey(seed).sign(message)
    return base64.b64encode(signed.signature).decode("ascii")


def generate_control_identity_for_tests() -> tuple[str, str]:
    """`(private, public)` base64 Ed25519 — the shape `control` mints.
    Named for what it is for; nothing on the agent mints a control
    identity in production."""
    signing = nacl.signing.SigningKey.generate()
    return (
        base64.b64encode(bytes(signing)).decode("ascii"),
        base64.b64encode(bytes(signing.verify_key)).decode("ascii"),
    )


# --------------------------------------------------------------------------- #
# Addresses
# --------------------------------------------------------------------------- #


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return True
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def format_url(host: str, port: int, *, scheme: str = "http") -> str:
    """`http://host:port`, bracketing an IPv6 literal."""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


def derive_advertise_host(
    control_url: str, *, timeout: float = _DERIVE_TIMEOUT_SECONDS
) -> str | None:
    """The local address this host uses to reach the control root.

    Opens a TCP connection to the root's host and port and reads the
    local end of the socket. On a mesh VPN that is precisely the
    interface the root can reach back on; it is a guess about routing
    symmetry, so the result is reported on `GET /v1/node` and the
    `advertiseUrl` config field overrides it. None when the root cannot
    be reached, which enrollment is about to report anyway.

    Deliberately not the hostname: it does not resolve across a tailnet
    unless MagicDNS happens to be on, and a value that works on some
    networks is a bug report waiting on the others.
    """
    parsed = urlparse(control_url)
    host = parsed.hostname
    if not host:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            local = sock.getsockname()[0]
    except OSError as exc:
        log.warning("could not derive an advertise address by reaching %s: %s", control_url, exc)
        return None
    return str(local)


def local_agent_url(bind_host: str, bind_port: int) -> str:
    """Where a component this agent spawns reaches *this* agent.

    Local by design: a companion's agent is always on the same host, and
    the loopback default that has been correct since M0 was only ever
    wrong about the port. A wildcard or loopback bind resolves to
    loopback; a bind to one specific interface has to be reached there.
    """
    host = bind_host.strip()
    if host in ("", "0.0.0.0", "::", "127.0.0.1", "localhost"):
        host = "127.0.0.1"
    return format_url(host, bind_port)


def effective_advertise_url(configured: Any, persisted: str | None) -> str | None:
    """The operator's setting wins; otherwise what enrollment derived."""
    if isinstance(configured, str) and configured.strip():
        return configured.strip().rstrip("/")
    return persisted.rstrip("/") if persisted else None


def advertise_host(url: str | None) -> str | None:
    if not url:
        return None
    return urlparse(url).hostname


# --------------------------------------------------------------------------- #
# The record and its store
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IdentityRecord:
    """What `node.yaml` holds. Every field optional so a fresh agent is a
    record with nothing in it rather than a missing file to special-case."""

    name: str | None = None
    private_key: str | None = None
    public_key: str | None = None
    signing_private_key: str | None = None
    signing_public_key: str | None = None
    advertise_sequence: int = 0
    control_url: str | None = None
    control_public_key: str | None = None
    recovery_public_key: str | None = None
    signing_key: str | None = None
    signing_key_id: str | None = None
    epoch: int | None = None
    advertise_url: str | None = None
    enrolled_at: str | None = None

    @property
    def enrolled(self) -> bool:
        return bool(self.name and self.control_url and self.signing_key)

    @property
    def signing_key_bytes(self) -> bytes | None:
        if not self.signing_key:
            return None
        try:
            raw = base64.b64decode(self.signing_key, validate=True)
        except (binascii.Error, ValueError):
            return None
        return raw if len(raw) == 32 else None


_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "name"),
    ("privateKey", "private_key"),
    ("publicKey", "public_key"),
    ("signingPrivateKey", "signing_private_key"),
    ("signingPublicKey", "signing_public_key"),
    ("advertiseSequence", "advertise_sequence"),
    ("controlUrl", "control_url"),
    ("controlPublicKey", "control_public_key"),
    ("recoveryPublicKey", "recovery_public_key"),
    ("signingKey", "signing_key"),
    ("signingKeyId", "signing_key_id"),
    ("epoch", "epoch"),
    ("advertiseUrl", "advertise_url"),
    ("enrolledAt", "enrolled_at"),
)


def _key_id_order(value: str | None) -> int | None:
    """Key generations are compared as integers when both are digit
    strings, which is what the control root mints. Anything else is not
    ordered, and an unordered pair is never called a replay."""
    if value is None or not value.isdigit():
        return None
    return int(value)


class NodeIdentityStore:
    """Threadsafe owner of `node.yaml`. Single lock, single file write."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._record = IdentityRecord()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def record(self) -> IdentityRecord:
        with self._lock:
            return self._record

    # ----- lifecycle --------------------------------------------------

    def load(self) -> None:
        with self._lock:
            if not self._path.exists():
                self._record = IdentityRecord()
                return
            raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"{self._path} must be a YAML mapping at the root")
            values: dict[str, Any] = {}
            for wire, attr in _FIELDS:
                value = raw.get(wire)
                if attr == "epoch":
                    values[attr] = int(value) if isinstance(value, int) else None
                elif attr == "advertise_sequence":
                    values[attr] = int(value) if isinstance(value, int) else 0
                else:
                    values[attr] = (
                        str(value) if isinstance(value, str | int) and value != "" else None
                    )
            self._record = IdentityRecord(**values)
            if self._record.enrolled:
                log.info(
                    "node identity loaded: enrolled as %r with %s at epoch %s, "
                    "signing key generation %s",
                    self._record.name,
                    self._record.control_url,
                    self._record.epoch,
                    self._record.signing_key_id,
                )

    # ----- mutations --------------------------------------------------

    def ensure_keypair(self) -> IdentityRecord:
        """Generate this node's two keypairs if they are missing. Both
        private halves are written here and read by nothing but this
        process.

        **Two, not one, and they are not interchangeable.** The X25519
        pair exists so the control root can seal secrets *to* this node;
        the Ed25519 pair exists so this node can sign an address
        announcement. One key cannot do both jobs — sealing is
        Diffie-Hellman, signing is Ed25519 — and a key that did both
        would be a key whose compromise costs twice.

        Each is filled in independently, so a node that enrolled before
        the signing key existed grows one on its next start. That alone
        does not let it re-advertise: the control root has to hold the
        public half, which only enrollment gives it.
        """
        with self._lock:
            changed = False
            if not (self._record.private_key and self._record.public_key):
                private = nacl.public.PrivateKey.generate()
                self._record = replace(
                    self._record,
                    private_key=base64.b64encode(bytes(private)).decode("ascii"),
                    public_key=base64.b64encode(bytes(private.public_key)).decode("ascii"),
                )
                changed = True
                log.info("generated this node's sealing keypair")
            if not (self._record.signing_private_key and self._record.signing_public_key):
                signing = nacl.signing.SigningKey.generate()
                self._record = replace(
                    self._record,
                    signing_private_key=base64.b64encode(bytes(signing)).decode("ascii"),
                    signing_public_key=base64.b64encode(bytes(signing.verify_key)).decode("ascii"),
                )
                changed = True
                log.info("generated this node's signing keypair")
            if changed:
                self._write_locked()
            return self._record

    def record_enrollment(
        self,
        *,
        name: str,
        control_url: str,
        epoch: int,
        signing_key: str,
        signing_key_id: str | None,
        control_public_key: str | None,
        recovery_public_key: str | None,
        advertise_url: str | None,
    ) -> IdentityRecord:
        with self._lock:
            self._record = replace(
                self._record,
                name=name,
                control_url=control_url.rstrip("/"),
                epoch=epoch,
                signing_key=signing_key,
                signing_key_id=signing_key_id,
                control_public_key=control_public_key,
                recovery_public_key=recovery_public_key,
                advertise_url=advertise_url,
                # The root's `enrollNode` replaces the node record
                # wholesale, so its high-water mark goes to zero here
                # too. Keeping a stale counter on either side is how a
                # rebuilt host ends up unable to re-advertise.
                advertise_sequence=0,
                enrolled_at=datetime.now(UTC).isoformat(),
            )
            self._write_locked()
            return self._record

    def record_advertise_url(self, url: str | None) -> IdentityRecord:
        """Persist what this node currently believes its address to be.

        Written whether or not the root has been told, because it is what
        `GET /v1/node` reports and what decides whether children bind
        wide. Announcing is a separate step that can fail.
        """
        with self._lock:
            if url == self._record.advertise_url:
                return self._record
            self._record = replace(self._record, advertise_url=url)
            self._write_locked()
            return self._record

    def next_advertise_sequence(self) -> int:
        """Claim the next announcement number, persisting it first.

        Incremented **before** the call goes out and never rolled back on
        failure. A gap in the sequence costs nothing — the root only
        requires the next one to be higher — while re-using a number
        after a timeout that actually succeeded would look exactly like
        a replay and be refused forever.
        """
        with self._lock:
            claimed = int(self._record.advertise_sequence or 0) + 1
            self._record = replace(self._record, advertise_sequence=claimed)
            self._write_locked()
            return claimed

    def unenroll(self) -> IdentityRecord:
        """Leave the install: discard its signing key, epoch, root URL and
        root identity, and return this node to its own.

        **The node's own keypairs are kept.** They are this host's
        identity, not the install's — the same node re-joining anywhere
        is still the same node, and nothing the departing install holds
        can be read with a public half alone.

        **The derived advertise URL is discarded**, because it was
        derived from the route to a control root this node no longer
        answers to. An operator-configured `advertiseUrl` lives in
        `agent.yaml` and is untouched, so it keeps winning.
        """
        with self._lock:
            self._record = replace(
                self._record,
                name=None,
                control_url=None,
                control_public_key=None,
                recovery_public_key=None,
                signing_key=None,
                signing_key_id=None,
                epoch=None,
                advertise_url=None,
                advertise_sequence=0,
                enrolled_at=None,
            )
            self._write_locked()
            return self._record

    def accept_rekey(self, *, signing_key: str, signing_key_id: str, epoch: int) -> bool:
        """Record a re-key or an epoch announcement. Returns True iff the
        signing key changed — the caller restarts children only then.

        Raises `FencedError` on a lower epoch, or an equal epoch with a
        lower key generation. Never moves the recorded epoch backwards.
        """
        with self._lock:
            held_epoch = self._record.epoch or 0
            if epoch < held_epoch:
                raise FencedError(
                    f"refusing epoch {epoch}: this node has already acknowledged epoch "
                    f"{held_epoch}. A control root presenting a lower epoch is a superseded "
                    f"root, and this node fences it without an election."
                )
            held_generation = _key_id_order(self._record.signing_key_id)
            offered_generation = _key_id_order(signing_key_id)
            if (
                epoch == held_epoch
                and held_generation is not None
                and offered_generation is not None
                and offered_generation < held_generation
            ):
                raise FencedError(
                    f"refusing signing key generation {signing_key_id!r} at epoch {epoch}: this "
                    f"node holds generation {self._record.signing_key_id!r}. A lower generation at "
                    f"the same epoch is a replayed rotation."
                )
            changed = signing_key != self._record.signing_key
            self._record = replace(
                self._record,
                signing_key=signing_key,
                signing_key_id=signing_key_id,
                epoch=epoch,
            )
            self._write_locked()
            return changed

    # ----- internals --------------------------------------------------

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out: dict[str, Any] = {}
        for wire, attr in _FIELDS:
            value = getattr(self._record, attr)
            if value is not None:
                out[wire] = value
        rendered = yaml.safe_dump(out, sort_keys=True, default_flow_style=False)
        # Written to a sibling and renamed, so a crash mid-write leaves
        # the previous identity rather than half of a new one.
        tmp = self._path.with_suffix(".yaml.tmp")
        tmp.write_text(rendered, encoding="utf-8")
        if sys.platform != "win32":
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, self._path)


__all__ = [
    "NODE_FILE",
    "FencedError",
    "IdentityRecord",
    "NodeIdentityStore",
    "address_message",
    "advertise_host",
    "derive_advertise_host",
    "effective_advertise_url",
    "format_url",
    "generate_control_identity_for_tests",
    "is_loopback_host",
    "local_agent_url",
    "rekey_message",
    "sign_address",
    "sign_rekey_message",
    "verify_rekey_signature",
]
