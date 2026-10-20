"""The client doors (LT2): TLS and plain HTTP on one port, told apart by the
first byte, terminated in this process (`client_doors`).

Real sockets throughout: a uvicorn server with the console's protocol, and a
pump door in front of a loopback echo server, each reached over TLS with the
CA verified and over plain HTTP.
"""

from __future__ import annotations

import asyncio
import datetime
import ipaddress
import socket
import ssl
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from eugene_plexus_agent import client_doors


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def pki(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ssl.SSLContext]:
    """A CA file a client trusts, and a server context with a leaf for 127.0.0.1."""
    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(hours=1))
        .not_valid_after(now + datetime.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), False)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(hours=1))
        .not_valid_after(now + datetime.timedelta(days=30))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=True,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), False
        )
        .sign(ca_key, hashes.SHA256())
    )
    directory = tmp_path_factory.mktemp("pki")
    ca_file = directory / "ca.pem"
    ca_file.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    chain = directory / "leaf.pem"
    chain.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        + leaf.public_bytes(serialization.Encoding.PEM)
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(chain)
    return ca_file, context


class _Counts:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def add(self, door: str, peer: str) -> None:
        self.seen.append((door, peer))


async def _console(doors: client_doors.Doors, port: int) -> Any:
    import uvicorn

    async def app(scope: dict, receive: object, send: object) -> None:
        assert scope["type"] == "http"
        body = f"{scope['scheme']} {scope['client'][0]}".encode()
        await send({"type": "http.response.start", "status": 200, "headers": []})  # type: ignore[operator]
        await send({"type": "http.response.body", "body": body})  # type: ignore[operator]

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        lifespan="off",
        log_level="warning",
        http=doors.console_protocol(port),
    )
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup()
    return server


async def test_the_console_answers_tls_and_plain_on_one_port(
    pki: tuple[Path, ssl.SSLContext],
) -> None:
    ca_file, context = pki
    doors = client_doors.Doors(_Counts())
    port = _free_port()
    server = await _console(doors, port)
    try:
        async with httpx.AsyncClient(trust_env=False) as plain:
            got = await plain.get(f"http://127.0.0.1:{port}/")
        assert got.text == "http 127.0.0.1"
        # Before a leaf is held, a TLS hello reaches the HTTP parser and fails.
        with pytest.raises(httpx.HTTPError):
            async with httpx.AsyncClient(
                trust_env=False, verify=ssl.create_default_context(cafile=str(ca_file))
            ) as client:
                await client.get(f"https://127.0.0.1:{port}/")
        doors.set_context(context)
        async with httpx.AsyncClient(
            trust_env=False, verify=ssl.create_default_context(cafile=str(ca_file))
        ) as client:
            got = await client.get(f"https://127.0.0.1:{port}/")
            again = await client.get(f"https://127.0.0.1:{port}/")
        assert got.text == "https 127.0.0.1" and again.status_code == 200
        async with httpx.AsyncClient(trust_env=False) as plain:
            assert (await plain.get(f"http://127.0.0.1:{port}/")).text == "http 127.0.0.1"
        # Loopback is never counted.
        assert doors.counts.seen == []  # type: ignore[union-attr]
    finally:
        await server.shutdown()


async def _echo_server() -> tuple[asyncio.Server, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        line = head.split(b"\r\n", 1)[0]
        port = writer.get_extra_info("sockname")[1]
        body = line + b" via " + str(port).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-length: "
            + str(len(body)).encode()
            + b"\r\nconnection: close\r\n\r\n"
            + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, int(server.sockets[0].getsockname()[1])


async def test_a_pump_door_carries_both_branches_to_their_loopback_ports(
    pki: tuple[Path, ssl.SSLContext],
) -> None:
    ca_file, context = pki
    doors = client_doors.Doors(_Counts())
    doors.set_context(context)
    plain_server, plain_port = await _echo_server()
    tls_server, tls_port = await _echo_server()
    port = _free_port()
    await doors.open_pump(
        "gateway",
        bind_host="127.0.0.1",
        port=port,
        plain_target=lambda: plain_port,
        tls_target=lambda: tls_port,
    )
    try:
        async with httpx.AsyncClient(
            trust_env=False, verify=ssl.create_default_context(cafile=str(ca_file))
        ) as client:
            tls = await client.get(f"https://127.0.0.1:{port}/v1/models")
        async with httpx.AsyncClient(trust_env=False) as client:
            plain = await client.get(f"http://127.0.0.1:{port}/healthz")
        assert tls.text == f"GET /v1/models HTTP/1.1 via {tls_port}"
        assert plain.text == f"GET /healthz HTTP/1.1 via {plain_port}"
        assert doors.doors["gateway"].listening
    finally:
        await doors.close_all()
        plain_server.close()
        tls_server.close()


async def test_plain_from_another_machine_is_counted_and_refused_once_off(
    pki: tuple[Path, ssl.SSLContext],
) -> None:
    """The peer is what the socket says; a sniffer is driven directly with a
    transport that reports another machine's address."""
    counts = _Counts()
    doors = client_doors.Doors(counts)
    doors.set_context(pki[1])
    allowed = {"value": True}
    doors.plain_allowed = lambda: allowed["value"]

    class Transport(asyncio.Transport):
        def __init__(self, peer: str = "192.168.1.50") -> None:
            super().__init__()
            self.peer = peer
            self.written = b""
            self.closed = False
            self.protocol: object = None

        def get_extra_info(self, name: str, default: object = None) -> object:
            if name == "peername":
                return (self.peer, 50000)
            if name == "sockname":
                return ("192.168.1.10", 8080)
            return default

        def write(self, data: bytes) -> None:  # type: ignore[override]
            self.written += data

        def close(self) -> None:
            self.closed = True

        def is_closing(self) -> bool:
            return self.closed

        def set_protocol(self, protocol: asyncio.BaseProtocol) -> None:
            self.protocol = protocol

    handed: list[bool] = []

    class Inner(asyncio.Protocol):
        def data_received(self, data: bytes) -> None:
            handed.append(True)

    request = b"GET /v1/models HTTP/1.1\r\nHost: 192.168.1.10:8080\r\n\r\n"
    first = Transport()
    sniff = client_doors._Sniff(doors, "gateway", 8080, lambda tls: Inner())
    sniff.connection_made(first)
    sniff.data_received(request)
    assert handed == [True] and counts.seen == [("gateway", "192.168.1.50")]

    allowed["value"] = False
    second = Transport()
    sniff = client_doors._Sniff(doors, "gateway", 8080, lambda tls: Inner())
    sniff.connection_made(second)
    sniff.data_received(request)
    assert second.closed and handed == [True]
    assert second.written.startswith(b"HTTP/1.1 403")
    assert b"https://192.168.1.10:8080/v1/models" in second.written

    # This machine at its own LAN address is local: never counted or refused.
    own = Transport(peer="192.168.1.10")
    sniff = client_doors._Sniff(doors, "gateway", 8080, lambda tls: Inner())
    sniff.connection_made(own)
    sniff.data_received(request)
    assert not own.closed and handed == [True, True]
    assert counts.seen == [("gateway", "192.168.1.50")] * 2
