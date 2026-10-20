"""This node's client certificate and doors (LT2).

`specs/docs/design/tls-for-clients.md` §4-§7. The doors themselves are
`client_doors`; this module keeps what they present and what they report:

- **The leaf.** One per node, for the machine key LT1 made (`node.yaml`
  `tls_private_key`), issued by the root's install CA on request
  (`POST /v1/nodes/{name}/certificate`, with this node's own token). Asked
  for when there is none, when it is two thirds through its life, and as soon
  as this node's address changes. Kept beside `node.yaml` (`client_tls.json`)
  so the doors keep serving with the root down, and loaded into every door in
  place: open connections keep the one they started with.
- **Plain HTTP from other machines**, counted per door and client address for
  7 days (`client_plain.json`), and refused once the owner turns it off; the
  root's signed machine list carries the switch (`clientPlainHttp`).
- **The gateway's door.** The gateway this agent spawns listens on a loopback
  port of its own, and the door holds its published port (8080), in front.

A node alone (not enrolled) has no CA: its doors take plain HTTP only, as
before LT2.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import socket
import ssl
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from . import client_doors
from ._http import internal_client
from ._private_files import write_private

log = logging.getLogger(__name__)

FILE = "client_tls.json"
CHAIN_FILE = "client_tls.pem"
PLAIN_FILE = "client_plain.json"
CERTIFICATE_PATH = "/v1/nodes/{name}/certificate"
#: How often the leaf is looked at once it is held.
CHECK_SECONDS = 60.0
#: How soon to ask again when the root did not answer or is sealed.
RETRY_SECONDS = 15.0
#: How soon to ask again after the root refused, for the same address.
REFUSED_RETRY_SECONDS = 3600.0
PLAIN_DAYS = 7


def free_loopback_port() -> int:
    """A port nothing holds on loopback now, for a child behind a door."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------- #
# Plain connections, by door and client, for 7 days
# --------------------------------------------------------------------------- #


class PlainCounts:
    """`{door: {address: {day: count}}}`, days as ISO dates, pruned past 7."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict[str, dict[str, int]]] = {}
        self._last: dict[tuple[str, str], float] = {}
        self._dirty = False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._data = {
                    str(door): {
                        str(address): {str(d): int(c) for d, c in days.items()}
                        for address, days in (clients or {}).items()
                        if isinstance(days, dict)
                    }
                    for door, clients in raw.get("doors", {}).items()
                    if isinstance(clients, dict)
                }
                for key, seen in (raw.get("lastSeen") or {}).items():
                    door, _, address = str(key).partition("|")
                    self._last[(door, address)] = float(seen)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, AttributeError) as exc:
            log.warning("the plain connection counts at %s are not usable: %s", path, exc)

    def add(self, door: str, address: str) -> None:
        today = datetime.date.today().isoformat()
        days = self._data.setdefault(door, {}).setdefault(address, {})
        days[today] = days.get(today, 0) + 1
        self._last[(door, address)] = time.time()
        self._dirty = True

    def _prune(self) -> None:
        oldest = (datetime.date.today() - datetime.timedelta(days=PLAIN_DAYS - 1)).isoformat()
        for door in list(self._data):
            for address in list(self._data[door]):
                days = {d: c for d, c in self._data[door][address].items() if d >= oldest}
                if days:
                    self._data[door][address] = days
                else:
                    del self._data[door][address]
                    self._last.pop((door, address), None)
            if not self._data[door]:
                del self._data[door]

    def clients(self, door: str) -> list[dict[str, Any]]:
        self._prune()
        rows = [
            {
                "address": address,
                "count": sum(days.values()),
                "lastSeen": datetime.datetime.fromtimestamp(
                    self._last.get((door, address), 0), datetime.UTC
                ),
            }
            for address, days in (self._data.get(door) or {}).items()
        ]
        return sorted(rows, key=lambda r: r["lastSeen"], reverse=True)

    def save(self) -> None:
        if not self._dirty:
            return
        self._prune()
        document = {
            "doors": self._data,
            "lastSeen": {f"{d}|{a}": seen for (d, a), seen in self._last.items()},
        }
        try:
            write_private(self.path, json.dumps(document))
            self._dirty = False
        except OSError as exc:
            log.warning("could not keep the plain connection counts in %s: %s", self.path, exc)


# --------------------------------------------------------------------------- #
# The leaf
# --------------------------------------------------------------------------- #


def _csr(key_pem: str, node_name: str) -> str:
    """A CSR for the machine key with no names: the root signs every name of
    this node's own that its CA may sign for (`tls-for-clients.md` §4)."""
    key = serialization.load_pem_private_key(key_pem.encode("ascii"), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError("the machine key is not an EC key")
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([]))
        .sign(key, hashes.SHA256())
    )
    return csr.public_bytes(serialization.Encoding.PEM).decode("ascii")


def _same_key(certificate_pem: str, key_pem: str) -> bool:
    try:
        cert = x509.load_pem_x509_certificate(certificate_pem.encode("ascii"))
        key = serialization.load_pem_private_key(key_pem.encode("ascii"), password=None)
    except ValueError:
        return False
    spki = serialization.PublicFormat.SubjectPublicKeyInfo
    der = serialization.Encoding.DER
    return cert.public_key().public_bytes(der, spki) == key.public_key().public_bytes(der, spki)


class ClientTls:
    """The leaf this node holds, its state, and the loop that keeps it."""

    def __init__(self, directory: Path, app: Any, doors: client_doors.Doors) -> None:
        self.directory = directory
        self.path = directory / FILE
        self._app = app
        self.doors = doors
        self.held: dict[str, Any] | None = None
        self.state = "none"
        self.error: str | None = None
        self._refused_for: tuple[str | None, float] | None = None

    # ---- what is held ----------------------------------------------------------

    @property
    def _record(self) -> Any:
        return self._app.state.node_identity.record

    def _advertise(self) -> str | None:
        from . import node_identity

        return node_identity.effective_advertise_url(
            self._app.state.agent_state.get_config("advertiseUrl"), self._record.advertise_url
        )

    def load(self) -> None:
        try:
            held = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.warning("the client certificate at %s is not usable: %s", self.path, exc)
            return
        if not isinstance(held, dict) or not held.get("certificate"):
            return
        self._install(held)

    def _install(self, held: dict[str, Any]) -> None:
        key = self._record.tls_private_key
        if not key or not _same_key(str(held["certificate"]), key):
            log.warning("the client certificate held is for another key; asking for a new one")
            return
        chain = self.directory / CHAIN_FILE
        write_private(chain, key.rstrip() + "\n" + str(held["certificate"]).rstrip() + "\n")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_alpn_protocols(["http/1.1"])
        context.load_cert_chain(certfile=str(chain), keyfile=str(chain))
        self.held = held
        self.state, self.error = "issued", None
        self.doors.set_context(context)

    def ca_pem(self) -> str | None:
        return str(self.held["ca"]) if self.held and self.held.get("ca") else None

    def _not_after(self) -> datetime.datetime | None:
        if not self.held:
            return None
        return datetime.datetime.fromisoformat(str(self.held["notAfter"]).replace("Z", "+00:00"))

    def renew_at(self) -> datetime.datetime | None:
        if not self.held:
            return None
        issued = datetime.datetime.fromisoformat(str(self.held["issuedAt"]).replace("Z", "+00:00"))
        not_after = self._not_after()
        assert not_after is not None
        return issued + (not_after - issued) * 2 / 3

    def wanted(self) -> str | None:
        """Why this node should ask for a leaf now, or None."""
        record = self._record
        if not record.enrolled or not record.name or not record.tls_private_key:
            return None
        advertise = self._advertise()
        if self._refused_for is not None:
            address, at = self._refused_for
            if address == advertise and time.time() - at < REFUSED_RETRY_SECONDS:
                return None
        if self.held is None:
            return "this node holds none"
        if not _same_key(str(self.held["certificate"]), record.tls_private_key):
            return "the machine key changed"
        if self.held.get("issuedFor") != advertise:
            return "this node's address changed"
        renew = self.renew_at()
        if renew is not None and datetime.datetime.now(datetime.UTC) >= renew:
            return "it is two thirds through its life"
        return None

    # ---- asking the root -------------------------------------------------------

    async def request(self, reason: str, transport: Any = None) -> None:
        record = self._record
        advertise = self._advertise()
        url = str(record.control_url).rstrip("/") + CERTIFICATE_PATH.format(name=record.name)
        try:
            token = self._app.state.auth_state.trust.agent_token("control")
            body = {"csr": _csr(str(record.tls_private_key), str(record.name))}
        except Exception as exc:  # a key or token this node cannot make
            self.state, self.error = ("issued" if self.held else "error"), str(exc)
            return
        log.info("asking the root for a client certificate: %s", reason)
        try:
            async with internal_client(timeout=30.0, transport=transport) as client:
                response = await client.post(
                    url, json=body, headers={"Authorization": f"Bearer {token}"}
                )
        except httpx.HTTPError as exc:
            if self.held is None:
                self.state = "pending"
            self.error = f"the root did not answer: {exc}"
            return
        if response.status_code == 200:
            answer = response.json()
            held = {
                "certificate": answer["certificate"],
                "ca": answer["ca"],
                "names": list(answer.get("names") or []),
                "notAfter": answer["notAfter"],
                "issuedAt": datetime.datetime.now(datetime.UTC).isoformat(),
                "issuedFor": advertise,
            }
            write_private(self.path, json.dumps(held))
            self._install(held)
            self._refused_for = None
            log.info(
                "this node's doors present a certificate from the install CA for %s",
                ", ".join(held["names"]),
            )
            return
        try:
            detail = str(response.json().get("detail") or response.text)
        except ValueError:
            detail = response.text[:300]
        if response.status_code in (503, 404) or response.status_code >= 500:
            if self.held is None:
                self.state = "pending"
            self.error = detail
            return
        log.warning("the root refused this node a client certificate: %s", detail)
        self._refused_for = (advertise, time.time())
        if self.held is None:
            self.state = "error"
        self.error = detail

    async def step(self) -> None:
        gateway = getattr(self._app.state, "gateway_door", None)
        if gateway is not None:
            await gateway.ensure(self._app)
        reason = self.wanted()
        if reason is not None:
            await self.request(reason)
        if self.doors.counts is not None:
            self.doors.counts.save()

    async def run(self) -> None:
        while True:
            try:
                await self.step()
            except Exception:  # never let the loop die; the doors keep what they hold
                log.exception("the client certificate check failed")
            # Soon while there is no leaf (a node that has just joined, or a
            # root still sealed); every minute once one is held.
            await asyncio.sleep(CHECK_SECONDS if self.held else RETRY_SECONDS)

    # ---- what GET /v1/node says -------------------------------------------------

    def status(self) -> dict[str, Any]:
        record = self._record
        state = self.state if record.enrolled else "none"
        counts = self.doors.counts
        doors = []
        for door in self.doors.doors.values():
            clients = counts.clients(door.name) if counts is not None else []
            doors.append(
                {
                    "name": door.name,
                    "port": door.port,
                    "listening": door.listening,
                    "tls": door.listening and self.doors.context() is not None,
                    "error": door.error,
                    "plainConnections": sum(c["count"] for c in clients),
                    "plainClients": clients,
                }
            )
        not_after = self._not_after()
        renew = self.renew_at()
        return {
            "state": state,
            "names": list((self.held or {}).get("names") or []),
            "notAfter": not_after,
            "renewAt": renew,
            "error": self.error,
            "plainAllowed": self.doors.plain_allowed(),
            "doors": doors,
        }


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def gateway_entry(state: Any) -> Any | None:
    """The one gateway this agent spawns, or None (none, or more than one)."""
    found = [
        e
        for e in state.list_topology_entries()
        if e.kind.value == "gateway" and e.spawn is not None
    ]
    return found[0] if len(found) == 1 else None


def _port_of(url: str) -> int | None:
    try:
        return urlsplit(url).port
    except ValueError:
        return None


class GatewayDoor:
    """The gateway's published port, in front of its loopback one."""

    def __init__(self, doors: client_doors.Doors) -> None:
        self.doors = doors
        self.inner: int | None = None
        self.bind_host: str | None = None
        self._failed: str | None = None

    async def ensure(self, app: Any) -> None:
        """Open, move or close the door to match the topology and the bind
        the gateway would have had on its own. Never raises."""
        from .app import _base_child_env

        state = app.state.agent_state
        entry = gateway_entry(state)
        port = _port_of(str(entry.url)) if entry is not None else None
        if entry is None or port is None:
            if self.inner is not None:
                await self.doors.close("gateway")
                self.inner, self.bind_host = None, None
            return
        wanted = _base_child_env(app.state.settings, state, app.state.node_identity, "gateway").get(
            "BIND_HOST", "127.0.0.1"
        )
        door = self.doors.doors.get("gateway")
        if door is not None and door.listening and wanted == self.bind_host and door.port == port:
            return
        inner = self.inner or free_loopback_port()
        try:
            await self.doors.open_pump(
                "gateway", bind_host=wanted, port=port, plain_target=lambda: self.inner
            )
        except OSError as exc:
            if str(exc) != self._failed:
                log.warning(
                    "the gateway's door could not open on %d (%s); the gateway keeps the "
                    "port and takes plain HTTP only until this agent starts again",
                    port,
                    exc,
                )
            self._failed = str(exc)
            self.inner, self.bind_host = None, None
            return
        self._failed = None
        moved = self.inner is not None and self.bind_host != wanted
        self.inner, self.bind_host = inner, wanted
        if moved:
            log.info("the gateway's door now listens on %s:%d", wanted, port)

    def child_env(self) -> dict[str, str]:
        if self.inner is None:
            return {}
        return {"BIND_HOST": "127.0.0.1", "BIND_PORT": str(self.inner)}


async def start(app: Any) -> ClientTls | None:
    """The doors' certificate, counts and the gateway's door, before any child
    starts (the gateway is told its loopback port)."""
    settings = app.state.settings
    if not settings.client_doors_enabled:
        return None
    doors: client_doors.Doors | None = getattr(app.state, "client_doors", None)
    if doors is None:
        doors = client_doors.Doors()
        app.state.client_doors = doors
    directory = settings.config_file.resolve().parent
    doors.counts = PlainCounts(directory / PLAIN_FILE)

    def plain_allowed() -> bool:
        trust = getattr(app.state, "machine_trust", None)
        claims = getattr(trust, "claims", None) or {}
        return claims.get("clientPlainHttp") != "off"

    def fallback_host() -> str | None:
        from . import node_identity

        advertise = node_identity.effective_advertise_url(
            app.state.agent_state.get_config("advertiseUrl"),
            app.state.node_identity.record.advertise_url,
        )
        return node_identity.advertise_host(advertise)

    doors.plain_allowed = plain_allowed
    doors.fallback_host = fallback_host
    tls = ClientTls(directory, app, doors)
    tls.load()
    app.state.client_tls = tls
    gateway = GatewayDoor(doors)
    await gateway.ensure(app)
    app.state.gateway_door = gateway
    return tls
