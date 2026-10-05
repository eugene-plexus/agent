"""How a Job Site trusts its root over the internet (J7a, remote-nodes.md §3.1).

Troy, 2026-10-05: no third party in the path, and the root's certificate is
pinned at join by default. What is pinned is **the root's identity key**, the
one it signs trust bundles with, carried in the join command. The root signs
the list of TLS public keys its nodes name presents (`GET /v1/trust/tls`), and
this machine accepts a TLS connection to the root only when the key it is
shown is on a list that verified against that identity key. So:

* **No public CA and no DNS name are needed.** The entry point's own CA, a
  certificate the owner made, or Nginx Proxy Manager's Let's Encrypt
  certificate all work the same way, and a bare address works too.
* **A renewal needs nobody.** Shown a key it has not seen, this machine
  fetches the list again *over that same connection*, without sending
  anything, checks the signature, and checks the key it was shown is listed.
  Only then is anything secret sent. A party in the middle cannot pass that
  check without a private key the root itself signed for.
* **The list is re-read every minute** with the trust bundle, over a pinned
  connection, so a key the root stops listing stops being accepted.

Verification by certificate chain is off on these connections on purpose:
the pin is the verification, and a chain check would add a CA this design
leaves out.

**Proxies are honoured** (the helper's fix in §3.1): a root that is not on
this machine's network is reached through the system's HTTPS proxy when it
has one, and the pin is checked on the TLS session inside the tunnel.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import ssl
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpcore
import httpx
import jwt
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from . import tokens
from ._http import client_for, egress_ssl_context, is_internal

log = logging.getLogger(__name__)

TYP_ROOT_TLS = "ep-root-tls+jwt"
PINS_FILE = "root_tls.json"
TLS_PATH = "/v1/trust/tls"


class RootTlsError(Exception):
    """The root's TLS key could not be trusted. Nothing secret was sent."""


class PinRefused(httpcore.ConnectError):
    """The connection presented a key no verified list names."""

    def __init__(self, pin: str) -> None:
        super().__init__(f"the root presented a TLS key it has not signed ({pin})")
        self.pin = pin


def spki_pin(der: bytes) -> tuple[str, int]:
    """Base64url SHA-256 of a certificate's SubjectPublicKeyInfo, and its expiry."""
    cert = x509.load_der_x509_certificate(der)
    spki = cert.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    digest = base64.urlsafe_b64encode(hashlib.sha256(spki).digest()).rstrip(b"=").decode()
    return digest, int(cert.not_valid_after_utc.timestamp())


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    port = parts.port
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    return f"https://{host}" + (f":{port}" if port and port != 443 else "")


# --------------------------------------------------------------------------- #
# What this machine has verified
# --------------------------------------------------------------------------- #


@dataclass
class Pins:
    """The root's TLS keys this machine accepts, from the last verified list."""

    path: Path
    origin: str | None = None
    keys: dict[str, int] = field(default_factory=dict)
    iat: int = 0

    @classmethod
    def load(cls, path: Path) -> Pins:
        pins = cls(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return pins
        except (OSError, ValueError) as exc:
            log.warning("the root's TLS keys at %s are unreadable: %s", path, exc)
            return pins
        if isinstance(raw, dict):
            pins.origin = raw.get("origin") if isinstance(raw.get("origin"), str) else None
            keys = raw.get("keys")
            if isinstance(keys, dict):
                pins.keys = {str(k): int(v) for k, v in keys.items() if isinstance(v, int)}
            pins.iat = int(raw.get("iat") or 0)
        return pins

    def accepts(self, pin: str) -> bool:
        return self.keys.get(pin, 0) > time.time()

    def adopt(self, claims: dict[str, Any]) -> None:
        """Replace the keys with a verified list's. A list older than the one
        held is refused, so an old list cannot bring back a dropped key."""
        if int(claims["iat"]) < self.iat:
            raise RootTlsError("the root sent an older TLS key list than this machine holds")
        self.origin = str(claims["origin"])
        self.iat = int(claims["iat"])
        self.keys = {str(k["spki"]): int(k["notAfter"]) for k in claims["keys"]}
        self.save()

    def save(self) -> None:
        from ._private_files import write_private

        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_private(
            self.path, json.dumps({"origin": self.origin, "keys": self.keys, "iat": self.iat})
        )

    def forget(self) -> None:
        self.origin, self.keys, self.iat = None, {}, 0
        self.path.unlink(missing_ok=True)


def verify_list(jws: str, root_key: str, origin: str) -> dict[str, Any]:
    """The claims of a signed TLS key list, checked against the pinned identity key."""
    try:
        header = jwt.get_unverified_header(jws)
        claims: dict[str, Any] = jwt.decode(
            jws, tokens.load_public(root_key), algorithms=["EdDSA"], options={"verify_aud": False}
        )
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise RootTlsError(
            f"the root's TLS key list is not signed by its pinned key: {exc}"
        ) from exc
    if header.get("typ") != TYP_ROOT_TLS:
        raise RootTlsError("that is not a TLS key list")
    if claims.get("origin") != origin:
        raise RootTlsError(f"the list is for {claims.get('origin')!r}, not {origin!r}")
    keys = claims.get("keys")
    if (
        not isinstance(claims.get("iat"), int)
        or not isinstance(keys, list)
        or not all(
            isinstance(k, dict)
            and isinstance(k.get("spki"), str)
            and isinstance(k.get("notAfter"), int)
            for k in keys
        )
    ):
        raise RootTlsError("the root's TLS key list is malformed")
    return claims


# --------------------------------------------------------------------------- #
# A connection that checks the key before anything is written
# --------------------------------------------------------------------------- #


class _Stream(httpcore.AsyncNetworkStream):
    def __init__(self, inner: httpcore.AsyncNetworkStream, check: Callable[[str, Any], None]):
        self._inner, self._check = inner, check

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:  # noqa: ASYNC109
        return await self._inner.read(max_bytes, timeout)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:  # noqa: ASYNC109
        await self._inner.write(buffer, timeout)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's interface
    ) -> httpcore.AsyncNetworkStream:
        stream = await self._inner.start_tls(ssl_context, server_hostname, timeout)
        try:
            self._check(server_hostname or "", stream.get_extra_info("ssl_object"))
        except BaseException:
            await stream.aclose()
            raise
        return _Stream(stream, self._check)

    def get_extra_info(self, info: str) -> Any:
        return self._inner.get_extra_info(info)


class _Backend(httpcore.AsyncNetworkBackend):
    def __init__(self, check: Callable[[str, Any], None]) -> None:
        self._inner = httpcore.AnyIOBackend()
        self._check = check

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's interface
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        stream = await self._inner.connect_tcp(host, port, timeout, local_address, socket_options)
        return _Stream(stream, self._check)

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's interface
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:  # pragma: no cover - never a root
        raise httpcore.ConnectError("a root is not reached over a Unix socket")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def proxy_for(url: str) -> str | None:
    """The system's HTTPS proxy for a root off this network, or None."""
    if is_internal(url):
        return None
    host = urlsplit(url).hostname or ""
    try:
        if urllib.request.proxy_bypass(host):
            return None
    except OSError:  # pragma: no cover - a broken registry read on Windows
        return None
    proxies = urllib.request.getproxies()
    return proxies.get("https") or proxies.get("all") or None


def pinned_client(
    url: str,
    accept: Callable[[str], bool],
    *,
    timeout: float,
    seen: list[str] | None = None,
) -> httpx.AsyncClient:
    """A client for the root whose TLS session must present a key `accept` takes.

    Checked in the handshake's aftermath and before the first request byte,
    on a direct connection or inside a proxy's tunnel alike.
    """
    host = urlsplit(url).hostname or ""

    def check(server_hostname: str, ssl_object: Any) -> None:
        if server_hostname.strip("[]") != host:
            return  # the TLS session to an HTTPS proxy itself, verified normally
        der = ssl_object.getpeercert(binary_form=True) if ssl_object is not None else None
        if not der:
            raise PinRefused("none")
        pin, _ = spki_pin(der)
        if seen is not None:
            seen.append(pin)
        if not accept(pin):
            raise PinRefused(pin)

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    transport = httpx.AsyncHTTPTransport(verify=context, trust_env=False)
    backend = _Backend(check)
    proxy = proxy_for(url)
    if proxy:
        transport._pool = httpcore.AsyncHTTPProxy(
            proxy_url=proxy,
            ssl_context=context,
            proxy_ssl_context=egress_ssl_context() if proxy.startswith("https:") else None,
            network_backend=backend,
        )
    else:
        transport._pool = httpcore.AsyncConnectionPool(ssl_context=context, network_backend=backend)
    return httpx.AsyncClient(
        transport=transport, timeout=timeout, trust_env=False, follow_redirects=False
    )


def refused_pin(exc: BaseException) -> str | None:
    """The key a connection was refused for, when that is why it failed."""
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, PinRefused):
            return cause.pin
        cause = cause.__cause__ or cause.__context__
    return None


async def fetch_list(url: str, root_key: str, *, seconds: float = 15.0) -> dict[str, Any]:
    """Fetch and verify the root's signed TLS key list, sending nothing.

    Over an unchecked connection, deliberately; what makes it safe is that
    the list must verify against the pinned identity key **and** name the
    key this very connection presented.
    """
    origin = origin_of(url)
    seen: list[str] = []
    async with pinned_client(url, lambda _pin: True, timeout=seconds, seen=seen) as client:
        try:
            answer = await client.get(url.rstrip("/") + TLS_PATH)
        except httpx.HTTPError as exc:
            raise RootTlsError(f"could not reach the root at {origin}: {exc}") from exc
    if answer.status_code != 200:
        detail = answer.text[:300]
        raise RootTlsError(f"the root at {origin} answered {answer.status_code}: {detail}")
    try:
        jws = str(answer.json()["jws"])
    except (ValueError, KeyError, TypeError) as exc:
        raise RootTlsError("the root's answer was not a TLS key list") from exc
    claims = verify_list(jws, root_key, origin)
    now = time.time()
    listed = {k["spki"] for k in claims["keys"] if k["notAfter"] > now}
    if not seen or seen[-1] not in listed:
        raise RootTlsError(
            f"the key {origin} presented ({seen[-1] if seen else 'none'}) is not one its "
            "root signed. Something between this machine and the root is presenting its own "
            "certificate; nothing was sent."
        )
    return claims


# --------------------------------------------------------------------------- #
# The root, as a Job Site reaches it
# --------------------------------------------------------------------------- #


class RootLink:
    """One client to the root for this machine's own calls, pinned on a Job Site.

    An ordinary node keeps what it had: `client_for` its root's address,
    whose proxy rule is per address (a LAN root is dialled direct).
    """

    def __init__(self, store: Any) -> None:
        self.store = store
        self.pins = Pins.load(store.path.parent / PINS_FILE)
        self._client: httpx.AsyncClient | None = None
        self._for: tuple[str, bool] | None = None

    def _pinned(self) -> bool:
        record = self.store.record
        url = str(record.control_url or "")
        return bool(record.job_site and url.startswith("https://"))

    def client(self, timeout: float = 15.0) -> httpx.AsyncClient:
        record = self.store.record
        url = str(record.control_url or "").rstrip("/")
        key = (url, self._pinned())
        if self._client is None or self._for != key:
            if self._client is not None:
                old = self._client
                self._client = None
                _close_later(old)
            if key[1]:
                self._client = pinned_client(url, self.pins.accepts, timeout=timeout)
            else:
                self._client = client_for(url, timeout=timeout, follow_redirects=False)
            self._for = key
        return self._client

    async def refresh(self) -> None:
        """Verify and adopt the root's current TLS key list (a Job Site only)."""
        record = self.store.record
        if not self._pinned() or not record.control_public_key:
            return
        claims = await fetch_list(str(record.control_url), record.control_public_key)
        self.pins.adopt(claims)
        log.info("adopted the root's TLS key list (%d keys)", len(self.pins.keys))

    async def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = str(self.store.record.control_url or "").rstrip("/") + path
        try:
            return await self.client().request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            pin = refused_pin(exc)
            if pin is None:
                raise
            log.info("the root presented a key this machine has not seen (%s); checking", pin)
            await self.refresh()
            return await self.client().request(method, url, **kwargs)

    async def recheck(self) -> None:
        """Re-read the list over a pinned connection, so a key the root stopped
        listing stops being accepted. Never raises: a root that is down is a
        normal state."""
        if not self._pinned():
            return
        record = self.store.record
        try:
            answer = await self.request("GET", TLS_PATH)
            if answer.status_code == 200:
                claims = verify_list(
                    str(answer.json()["jws"]),
                    str(record.control_public_key),
                    origin_of(str(record.control_url)),
                )
                if int(claims["iat"]) > self.pins.iat:
                    self.pins.adopt(claims)
        except (httpx.HTTPError, RootTlsError, ValueError, KeyError) as exc:
            log.debug("could not re-read the root's TLS key list: %s", exc)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _close_later(client: httpx.AsyncClient) -> None:
    import asyncio
    import contextlib

    # No running loop means nothing was ever opened on it.
    with contextlib.suppress(RuntimeError):
        asyncio.get_running_loop().create_task(client.aclose())
