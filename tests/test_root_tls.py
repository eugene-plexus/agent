"""A Job Site trusts its root by the root's identity key (J7a).

Real TLS servers on loopback with throwaway certificates: the root's, and an
impostor's presenting a different key. The impostor serves the root's own
signed list, which is the strongest thing a party in the middle could replay.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import socket
import ssl
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from eugene_plexus_agent import root_tls, tokens


def _certificate(tmp: Path, name: str) -> tuple[Path, Path, bytes]:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=1))
        .not_valid_after(now + dt.timedelta(days=2))
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp / f"{name}.crt", tmp / f"{name}.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path, cert.public_bytes(serialization.Encoding.DER)


class _Server:
    def __init__(self, cert: Path, key: Path, routes: dict[str, Any]) -> None:
        self.hits: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.hits.append(self.path)
                body = json.dumps(routes.get(self.path, {"missing": self.path})).encode()
                self.send_response(200 if self.path in routes else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_POST = do_GET

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _signed_list(identity: Any, origin: str, ders: list[bytes], iat: int | None = None) -> str:
    keys = [{"spki": root_tls.spki_pin(d)[0], "notAfter": root_tls.spki_pin(d)[1]} for d in ders]
    return jwt.encode(
        {"iat": iat or int(time.time()), "origin": origin, "keys": keys},
        identity,
        algorithm="EdDSA",
        headers={"typ": root_tls.TYP_ROOT_TLS},
    )


@pytest.fixture
def root(tmp_path: Path) -> Iterator[dict[str, Any]]:
    identity = tokens.generate_private_key()
    cert, key, der = _certificate(tmp_path, "root")
    holder: dict[str, Any] = {}
    real = _Server(cert, key, holder)
    url = f"https://127.0.0.1:{real.port}"
    holder["/v1/trust/tls"] = {"jws": _signed_list(identity, url, [der])}
    holder["/ping"] = {"ok": True}
    yield {
        "url": url,
        "key": tokens.public_b64(identity),
        "identity": identity,
        "der": der,
        "server": real,
        "routes": holder,
    }
    real.close()


def test_the_list_from_the_root_verifies_and_names_the_key_shown(root: dict[str, Any]) -> None:
    claims = asyncio.run(root_tls.fetch_list(root["url"], root["key"]))
    assert claims["keys"][0]["spki"] == root_tls.spki_pin(root["der"])[0]


def test_an_impostor_replaying_the_roots_own_list_is_refused(
    root: dict[str, Any], tmp_path: Path
) -> None:
    """A party in the middle at the root's address, with the root's own signed
    list (which names the root's key, not the impostor's)."""
    cert, key, _ = _certificate(tmp_path, "middle")
    routes: dict[str, Any] = {}
    middle = _Server(cert, key, routes)
    url = f"https://127.0.0.1:{middle.port}"
    routes["/v1/trust/tls"] = {"jws": _signed_list(root["identity"], url, [root["der"]])}
    try:
        with pytest.raises(root_tls.RootTlsError, match="not one its root signed"):
            asyncio.run(root_tls.fetch_list(url, root["key"]))
    finally:
        middle.close()


def test_a_list_signed_by_any_other_key_is_refused(root: dict[str, Any]) -> None:
    stranger = tokens.generate_private_key()
    root["routes"]["/v1/trust/tls"] = {"jws": _signed_list(stranger, root["url"], [root["der"]])}
    with pytest.raises(root_tls.RootTlsError, match="pinned key"):
        asyncio.run(root_tls.fetch_list(root["url"], root["key"]))


def test_a_root_whose_clock_runs_ahead_is_still_believed(root: dict[str, Any]) -> None:
    """A root a few seconds ahead of the site (WSL2 behind its NAT, measured
    2.4 s, 2026-10-05) signs a list "issued" in the site's future. Every token
    here allows the same clock skew; so does this list."""
    ahead = int(time.time()) + 30
    root["routes"]["/v1/trust/tls"] = {
        "jws": _signed_list(root["identity"], root["url"], [root["der"]], iat=ahead)
    }
    claims = asyncio.run(root_tls.fetch_list(root["url"], root["key"]))
    assert claims["iat"] == ahead
    far = int(time.time()) + tokens.LEEWAY_SECONDS + 60
    root["routes"]["/v1/trust/tls"] = {
        "jws": _signed_list(root["identity"], root["url"], [root["der"]], iat=far)
    }
    with pytest.raises(root_tls.RootTlsError, match="clock"):
        asyncio.run(root_tls.fetch_list(root["url"], root["key"]))


def test_a_list_for_another_origin_is_refused(root: dict[str, Any]) -> None:
    root["routes"]["/v1/trust/tls"] = {
        "jws": _signed_list(root["identity"], "https://elsewhere.example:8443", [root["der"]])
    }
    with pytest.raises(root_tls.RootTlsError, match="is for"):
        asyncio.run(root_tls.fetch_list(root["url"], root["key"]))


def test_a_pinned_client_refuses_a_key_before_sending_anything(root: dict[str, Any]) -> None:
    async def go() -> None:
        async with root_tls.pinned_client(root["url"], lambda _pin: False, timeout=5) as client:
            with pytest.raises(Exception) as caught:
                await client.post(root["url"] + "/ping", content=b"SECRET")
            assert root_tls.refused_pin(caught.value) == root_tls.spki_pin(root["der"])[0]

    asyncio.run(go())
    assert "/ping" not in root["server"].hits


class _Store:
    def __init__(self, tmp: Path, url: str, key: str) -> None:
        self.path = tmp / "node.yaml"

        class Record:
            control_url = url
            control_public_key = key
            job_site = True

        self.record = Record()


def test_a_site_adopts_a_new_key_the_root_signed_and_carries_on(
    root: dict[str, Any], tmp_path: Path
) -> None:
    link = root_tls.RootLink(_Store(tmp_path, root["url"], root["key"]))
    assert link.pins.keys == {}

    async def go() -> int:
        answer = await link.request("GET", "/ping")
        await link.aclose()
        return answer.status_code

    assert asyncio.run(go()) == 200
    assert link.pins.accepts(root_tls.spki_pin(root["der"])[0])
    saved = json.loads((tmp_path / root_tls.PINS_FILE).read_text())
    assert saved["origin"] == root["url"]


def test_an_older_list_cannot_bring_back_a_dropped_key(tmp_path: Path) -> None:
    pins = root_tls.Pins(tmp_path / root_tls.PINS_FILE)
    pins.adopt({"iat": 200, "origin": "https://a", "keys": [{"spki": "new", "notAfter": 2**40}]})
    with pytest.raises(root_tls.RootTlsError, match="older"):
        pins.adopt(
            {"iat": 100, "origin": "https://a", "keys": [{"spki": "old", "notAfter": 2**40}]}
        )
    assert pins.accepts("new") and not pins.accepts("old")


def test_an_expired_key_is_not_accepted(tmp_path: Path) -> None:
    pins = root_tls.Pins(tmp_path / root_tls.PINS_FILE, keys={"gone": int(time.time()) - 1})
    assert not pins.accepts("gone")


def test_the_pin_is_checked_inside_a_proxys_tunnel(
    root: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper honours proxies (§3.1), and the pin still holds through one."""
    tunnels: list[str] = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    proxy_port = listener.getsockname()[1]

    def pump(a: socket.socket, b: socket.socket) -> None:
        try:
            while data := a.recv(65536):
                b.sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):
                with contextlib.suppress(OSError):
                    s.shutdown(socket.SHUT_RDWR)

    def serve() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            request = b""
            while b"\r\n\r\n" not in request:
                request += conn.recv(4096)
            target = request.split(b" ")[1].decode()
            tunnels.append(target)
            host, port = target.rsplit(":", 1)
            upstream = socket.create_connection((host, int(port)))
            conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            threading.Thread(target=pump, args=(conn, upstream), daemon=True).start()
            threading.Thread(target=pump, args=(upstream, conn), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    # A root off this machine's network: a dotted name is egress, and so
    # gets the proxy. It resolves to loopback through the proxy alone.
    url = root["url"].replace("127.0.0.1", "root.example.test")
    monkeypatch.setattr(root_tls, "proxy_for", lambda _u: f"http://127.0.0.1:{proxy_port}")

    async def go(accept: bool) -> int:
        async with root_tls.pinned_client(url, lambda _pin: accept, timeout=5) as client:
            return (await client.get(url + "/ping")).status_code

    # The proxy resolves the name; point it at loopback.
    monkeypatch.setattr(socket, "getaddrinfo", _resolving(socket.getaddrinfo))
    assert asyncio.run(go(True)) == 200
    assert tunnels and tunnels[0].startswith("root.example.test:")
    with pytest.raises(Exception) as caught:
        asyncio.run(go(False))
    assert root_tls.refused_pin(caught.value)
    listener.close()


def _resolving(real: Any) -> Any:
    def resolve(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host in ("root.example.test", b"root.example.test"):
            host = "127.0.0.1"
        return real(host, *args, **kwargs)

    return resolve
