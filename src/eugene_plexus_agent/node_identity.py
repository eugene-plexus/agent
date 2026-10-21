"""This host's identity in an install: its keys, its enrollment, and
where other hosts reach it.

One file, `node.yaml`, beside `agent.yaml`:

    name: gpu-box
    privateKey: <base64 X25519>        never leaves this host
    publicKey: <base64 X25519>
    signingPrivateKey: <base64 Ed25519 seed>   signs address announcements
    signingPublicKey: <base64 Ed25519>
    tokenPrivateKey: <base64 Ed25519 seed>     signs this node's tokens
    tlsPrivateKey: <PEM P-256>          the machine port's TLS key (LT1)
    tlsCertificate: <PEM>               self-signed; other machines pin its key
    tlsNextPrivateKey: <PEM P-256>      the next machine key, during a change (row 3)
    tlsNextCertificate: <PEM>
    machineKeyChange: {state: ..., ...} the last machine key change (row 3)
    controlPins: <pin>[,<pin>]          the root's machine key, from the join token
    advertiseSequence: 3               strictly increasing, mirrored at the root
    controlUrl: http://100.64.0.1:8083
    controlPublicKey: <base64 Ed25519>  the root's identity: signs every trust bundle
    recoveryPublicKey: <base64 X25519>  the second recipient of anything sealed here
    epoch: 1                            highest control-root epoch accepted
    advertiseUrl: http://100.64.0.7:8079
    enrolledAt: 2026-09-10T22:14:03+00:00

**Every private key here was generated here and has never been anywhere
else** (per-node token keys, 2026-09-25;
`specs/docs/design/per-node-token-keys.md`). Until then this file also
held the install's signing key, sent by the control root at enrollment,
which made every node's copy of this file the whole install. The token
key here signs tokens for this machine and `agent` tokens for reads
elsewhere, and the trust bundle is what says so to everyone else. So
this file, leaked, costs this machine.

It is stored in the clear, mode 0600 (plus the installer's protections),
because a headless GPU box has to mint its children's tokens at boot
with nobody there to type a passphrase. Separate from `agent.yaml`
because that file is topology an operator edits (and has already leaked
once via `git add -A`); identity is never edited by hand.

**Epoch fencing lives here.** `accept_epoch` refuses an epoch below the
highest recorded, so a superseded control root is refused on this host
alone, with no election and no agreement with any other agent. The
epoch arrives on the trust bundle, and the bundle's own version refuses
a rollback (`trust.py`).

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

**The bundle's credential is a signature, not a bearer.** The root
signs it with its identity key, which does not rotate, and this node
checks it against the key it pinned at enrollment.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import socket
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit

import nacl.exceptions
import nacl.public
import nacl.signing
import yaml

from ._private_files import write_private

log = logging.getLogger(__name__)

NODE_FILE = "node.yaml"

# How long the advertise-host derivation waits on a TCP connect to the
# control root. It is a routing question, not a request; anything slower
# than this is the root being unreachable, and enrollment is about to
# discover that anyway.
_DERIVE_TIMEOUT_SECONDS = 3.0


class FencedError(Exception):
    """A trust bundle was refused because it would move the epoch
    backwards. The caller answers 409."""


# --------------------------------------------------------------------------- #
# The signed re-key message
# --------------------------------------------------------------------------- #


def address_message(
    *, name: str, sequence: int, url: str, machine: dict[str, str] | None = None
) -> bytes:
    """The canonical bytes an address announcement is signed over —
    `NodeAddressAnnouncement.signature` in `control.yaml`, byte for byte.

    Three fields, keys sorted, no whitespace. `url` goes in **exactly as
    it will be sent**, before any normalization, because the root
    verifies against the raw request body for precisely this reason. Sign
    a parsed URL and every announcement fails with what looks like a
    crypto error.

    `machine` is `machineUrl` and `tlsPin` when the body carries them
    (LT1): they are signed under their own names, so the key this node
    registers is covered by the same signature as its address.
    """
    signed: dict[str, Any] = {"name": name, "sequence": sequence, "url": url}
    signed.update(machine or {})
    return json.dumps(signed, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_address(*, signing_private_key: str, message: bytes) -> str:
    """Detached Ed25519 signature by this node's identity signing key."""
    seed = base64.b64decode(signing_private_key, validate=True)
    return base64.b64encode(nacl.signing.SigningKey(seed).sign(message).signature).decode("ascii")


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


def not_a_node_address(url: str | None) -> str | None:
    """Why this URL can never be a node's address, or None if it can be.

    R2.4 / review §6.2 #15. The control root refuses to *record* these
    now, which is where the rule belongs; this is the second copy, at
    the point of **use**, because that is where the credential is spent.
    A registry written before that fix still holds whatever a node told
    it, and `install_proxy` forwards the operator's own bearer to
    whatever address it finds there -- so a link-local address in the
    registry is the operator's token handed to a cloud
    instance-metadata service.

    Two copies rather than a shared helper, deliberately: components
    share schemas, not code, and this one answers "may I dial it"
    where the root's answers "may I write it down".
    """
    if not url or not url.strip():
        return "the address is empty"
    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        return f"it does not parse as a URL ({exc})"
    if parts.scheme.lower() not in ("http", "https"):
        return (
            f"the scheme is {parts.scheme or '(none)'!r}; an agent is reached over "
            "http or https and nothing else"
        )
    host = parts.hostname
    if not host:
        return "it names no host"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    if address.is_unspecified:
        return f"{host} is a bind wildcard, not an address anything can dial"
    if address.is_multicast:
        return f"{host} is a multicast group, not a host"
    if address.is_link_local:
        return (
            f"{host} is link-local -- not routable between hosts, and where cloud "
            "instance-metadata services listen"
        )
    if address.is_reserved and not address.is_loopback:
        return f"{host} is in a reserved range and cannot be a host on this network"
    return None


def format_url(host: str, port: int, *, scheme: str = "http") -> str:
    """`http://host:port`, bracketing an IPv6 literal."""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


def derive_advertise_host(
    control_url: str, *, timeout: float = _DERIVE_TIMEOUT_SECONDS, quiet: bool = False
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
        # Quiet for the running node's minute-by-minute recheck (agent#8):
        # a root that is down is said once by the announcement's own warning.
        (log.debug if quiet else log.warning)(
            "could not derive an advertise address by reaching %s: %s", control_url, exc
        )
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


def bind_host_for_advertised_node(
    *,
    configured: str,
    configured_explicitly: bool,
    advertise_url: str | None,
) -> str:
    """Which interface *this agent* binds, given what its node advertises.

    The same rule `shared_child_env` applies to every component this
    agent spawns, applied to the agent itself: **a node that advertises
    a non-loopback address binds one.** Advertising an address is what
    enrolling does, so a node cannot end up telling the control root to
    call back on an address nothing is listening on.

    It could, until 2026-09-11. Enrollment is outbound, so `201 Created`
    proves the node reached the root and nothing at all about the return
    path; a node that joined without `EUGENE_PLEXUS_AGENT_BIND_HOST` set
    came up on loopback, advertised its LAN address, and looked healthy
    from every direction except the one that mattered. Union topology,
    idle unload and start-on-demand would all have failed silently. The
    installers learned to widen the bind themselves; this is the half
    that covers `eugene-plexus-agent join` run by hand, which is the path
    `docs/deployment/tailnet.md` documents.

    **The env var still wins, including with a value that will fail** —
    `easy-default-expert-override`. An operator who sets
    `EUGENE_PLEXUS_AGENT_BIND_HOST=127.0.0.1` on an enrolled node gets
    loopback and the unreachable install they asked for; pydantic tracks
    that in `model_fields_set` even though the value equals the default,
    so "explicitly loopback" and "never mentioned it" are distinguishable.
    """
    if configured_explicitly:
        return configured
    if is_loopback_host(advertise_host(advertise_url)):
        return configured
    return "0.0.0.0"


# --------------------------------------------------------------------------- #
# The machine key (LT1)
# --------------------------------------------------------------------------- #

#: A machine key's certificate outlives any install: its key is what other
#: machines pin, and an expiry would only stop a working install.
_MACHINE_CERT_YEARS = 50


def generate_machine_key() -> tuple[str, str]:
    """`(key PEM, certificate PEM)`: this node's machine port key, ECDSA P-256.

    **A fourth key, separate from the other three**, so TLS never signs with
    the key that signs address announcements or tokens; P-256 because every
    TLS stack takes it. The certificate is self-signed and names nothing:
    other machines check its key against the pin this node registered under
    its signature, never a chain or a name (addresses move with DHCP).
    """
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "eugene-plexus-machine")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365 * _MACHINE_CERT_YEARS))
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    return key_pem, certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")


def machine_pin(certificate_pem: str) -> str:
    """The pin of a machine key: base64url SHA-256 of its SubjectPublicKeyInfo."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    from ._http import spki_pin

    der = x509.load_pem_x509_certificate(certificate_pem.encode("ascii")).public_bytes(
        serialization.Encoding.DER
    )
    return spki_pin(der)


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
    token_private_key: str | None = None
    advertise_sequence: int = 0
    control_url: str | None = None
    control_public_key: str | None = None
    recovery_public_key: str | None = None
    epoch: int | None = None
    advertise_url: str | None = None
    enrolled_at: str | None = None
    tls_private_key: str | None = None
    tls_certificate: str | None = None
    control_pins: str | None = None
    """Comma-separated pins of the root's machine key, from the join token
    this node enrolled with (LT1): what it checks the root against before
    the first signed machine list arrives."""
    tls_next_private_key: str | None = None
    tls_next_certificate: str | None = None
    """The next machine key while it changes (row 3,
    `machine_key_change`): made at the start, swapped in at the switch,
    deleted when the change stops or fails."""
    machine_key_change: dict[str, Any] | None = None
    """The last machine key change, as `machine_key_change` keeps it."""

    @property
    def enrolled(self) -> bool:
        return bool(self.name and self.control_url and self.control_public_key)


_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "name"),
    ("privateKey", "private_key"),
    ("publicKey", "public_key"),
    ("signingPrivateKey", "signing_private_key"),
    ("signingPublicKey", "signing_public_key"),
    ("tokenPrivateKey", "token_private_key"),
    ("advertiseSequence", "advertise_sequence"),
    ("controlUrl", "control_url"),
    ("controlPublicKey", "control_public_key"),
    ("recoveryPublicKey", "recovery_public_key"),
    ("epoch", "epoch"),
    ("advertiseUrl", "advertise_url"),
    ("enrolledAt", "enrolled_at"),
    ("tlsPrivateKey", "tls_private_key"),
    ("tlsCertificate", "tls_certificate"),
    ("controlPins", "control_pins"),
    ("tlsNextPrivateKey", "tls_next_private_key"),
    ("tlsNextCertificate", "tls_next_certificate"),
    ("machineKeyChange", "machine_key_change"),
)


class NodeIdentityStore:
    """Threadsafe owner of `node.yaml`. Single lock, single file write."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._record = IdentityRecord()
        #: Held while an address announcement is in flight (`enrollment`).
        self.announcing = asyncio.Lock()

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
                elif attr == "machine_key_change":
                    values[attr] = dict(value) if isinstance(value, dict) else None
                elif attr == "advertise_sequence":
                    values[attr] = int(value) if isinstance(value, int) else 0
                else:
                    values[attr] = (
                        str(value) if isinstance(value, str | int) and value != "" else None
                    )
            self._record = IdentityRecord(**values)
            if self._record.enrolled:
                log.info(
                    "node identity loaded: enrolled as %r with %s at epoch %s",
                    self._record.name,
                    self._record.control_url,
                    self._record.epoch,
                )

    # ----- mutations --------------------------------------------------

    def ensure_keypair(self) -> IdentityRecord:
        """Generate this node's three keys if they are missing. Every
        private half is written here and read by nothing but this process.

        **Three, and they are not interchangeable.** The X25519 pair
        exists so the control root can seal secrets *to* this node; the
        Ed25519 identity pair signs an address announcement; the Ed25519
        token key signs this node's tokens (2026-09-25). One key per
        job, because a key that did two would be a key whose compromise
        costs twice (NIST SP 800-57 §5.2).

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
            if not self._record.token_private_key:
                token = nacl.signing.SigningKey.generate()
                self._record = replace(
                    self._record,
                    token_private_key=base64.b64encode(bytes(token)).decode("ascii"),
                )
                changed = True
                log.info("generated this node's token key")
            if not (self._record.tls_private_key and self._record.tls_certificate):
                key_pem, certificate_pem = generate_machine_key()
                self._record = replace(
                    self._record, tls_private_key=key_pem, tls_certificate=certificate_pem
                )
                changed = True
                log.info("generated this node's machine key (pin %s)", machine_pin(certificate_pem))
            if changed:
                self._write_locked()
            return self._record

    def record_enrollment(
        self,
        *,
        name: str,
        control_url: str,
        epoch: int,
        control_public_key: str,
        recovery_public_key: str | None,
        advertise_url: str | None,
    ) -> IdentityRecord:
        with self._lock:
            self._record = replace(
                self._record,
                name=name,
                control_url=control_url.rstrip("/"),
                epoch=epoch,
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

    def record_control_pins(self, pins: list[str]) -> IdentityRecord:
        """Keep the root's machine key from the join token (LT1)."""
        with self._lock:
            value = ",".join(pins) or None
            if value != self._record.control_pins:
                self._record = replace(self._record, control_pins=value)
                self._write_locked()
            return self._record

    # ----- the machine key change (row 3) -----------------------------------

    def begin_machine_key_change(self, change: dict[str, Any]) -> IdentityRecord:
        """Make the next machine key and record the change, in one write.

        `change` is the change's state as `machine_key_change` keeps it;
        `nextPin` is filled in here, from the key made."""
        key_pem, certificate_pem = generate_machine_key()
        with self._lock:
            self._record = replace(
                self._record,
                tls_next_private_key=key_pem,
                tls_next_certificate=certificate_pem,
                machine_key_change={**change, "nextPin": machine_pin(certificate_pem)},
            )
            self._write_locked()
            log.info("made this node's next machine key (pin %s)", machine_pin(certificate_pem))
            return self._record

    def update_machine_key_change(self, **fields: Any) -> IdentityRecord:
        """Merge `fields` into the change's state; written only if it moved."""
        with self._lock:
            change = dict(self._record.machine_key_change or {})
            merged = {**change, **fields}
            if merged != change:
                self._record = replace(self._record, machine_key_change=merged)
                self._write_locked()
            return self._record

    def switch_machine_key(self, **fields: Any) -> IdentityRecord:
        """The switch: the next key becomes the machine key, and the old
        private key is gone from this file, in one write. `fields` go into
        the change's state in the same write."""
        with self._lock:
            record = self._record
            if not (record.tls_next_private_key and record.tls_next_certificate):
                raise ValueError("there is no next machine key to switch to")
            self._record = replace(
                record,
                tls_private_key=record.tls_next_private_key,
                tls_certificate=record.tls_next_certificate,
                tls_next_private_key=None,
                tls_next_certificate=None,
                machine_key_change={**(record.machine_key_change or {}), **fields},
            )
            self._write_locked()
            return self._record

    def drop_next_machine_key(self, **fields: Any) -> IdentityRecord:
        """A change that stopped or failed: the next key is deleted, the
        machine key stays, and `fields` go into the change's state."""
        with self._lock:
            self._record = replace(
                self._record,
                tls_next_private_key=None,
                tls_next_certificate=None,
                machine_key_change={**(self._record.machine_key_change or {}), **fields},
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
        """Leave the install: discard its epoch, root URL and root
        identity, and return this node to its own.

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
                epoch=None,
                advertise_url=None,
                advertise_sequence=0,
                enrolled_at=None,
                control_pins=None,
                # A machine key change belongs to the install it was for.
                tls_next_private_key=None,
                tls_next_certificate=None,
                machine_key_change=None,
            )
            self._write_locked()
            return self._record

    def accept_epoch(self, epoch: int) -> None:
        """Record a newer control-root epoch. Raises `FencedError` on a lower one.

        Never moves the recorded epoch backwards: a superseded root is
        refused on this host alone, with no election.
        """
        with self._lock:
            held_epoch = self._record.epoch or 0
            if epoch < held_epoch:
                raise FencedError(
                    f"refusing epoch {epoch}: this node has already acknowledged epoch "
                    f"{held_epoch}. A control root presenting a lower epoch is a superseded "
                    f"root, and this node fences it without an election."
                )
            if epoch != held_epoch:
                self._record = replace(self._record, epoch=epoch)
                self._write_locked()

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
        # the previous identity rather than half of a new one. **Created
        # 0600, not chmodded to it** (2026-09-22): `write_text` then
        # `chmod` left the install's signing key on disk at the umask's
        # mode for as long as the two calls took, and a leftover temp at
        # that fixed name kept its mode through the next `write_text`.
        # (The install's signing key is no longer here; the node's own
        # three private keys are.)
        write_private(self._path, rendered)


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
    "generate_machine_key",
    "is_loopback_host",
    "local_agent_url",
    "machine_pin",
    "not_a_node_address",
    "sign_address",
]
