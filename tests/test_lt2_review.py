"""The adversarial review of LT1 and LT2 (2026-10-20): each agent-side
finding, as a test that fails without its fix."""

from __future__ import annotations

import asyncio
from pathlib import Path

from eugene_plexus_agent import client_doors, client_tls, install_names
from eugene_plexus_agent.machine_trust import build_table

PIN_R, PIN_A, PIN_B = "R" * 43, "A" * 43, "B" * 43


def _node(name: str, url: str, door: str, pin: str) -> dict:
    return {"name": name, "url": url, "machineUrl": door, "pins": [pin]}


def test_a_node_claiming_the_roots_address_or_anothers_is_carried_for_nobody() -> None:
    claims = {
        "version": 1,
        "controlPins": [PIN_R],
        "nodes": [
            _node("a", "http://192.168.1.20:8079", "https://192.168.1.20:8083", PIN_A),
            _node("b", "http://192.168.1.20:8079", "https://192.168.1.21:8083", PIN_B),
            _node("thief", "http://192.168.1.10:8083", "https://192.168.1.10:9083", PIN_B),
        ],
    }
    table = build_table(
        claims, own_name="me", control_url="http://192.168.1.10:8083", token_pins=[]
    )
    assert table["routes"]["http://192.168.1.10:8083"] == "https://192.168.1.10:8083/control"
    assert table["routes"]["http://192.168.1.20:8079"] is None


def test_the_lan_cannot_shut_this_machine_out_and_an_ipv6_slash_64_is_one_address() -> None:
    doors = client_doors.Doors()

    class T:
        def is_closing(self) -> bool:
            return False

    for _ in range(client_doors.MAX_PER_PEER):
        assert doors.admit(T(), f"2001:db8::{_:x}")  # type: ignore[arg-type]
    assert not doors.admit(T(), "2001:db8::ffff")  # type: ignore[arg-type]
    assert doors.admit(T(), "2001:db8:0:1::1")  # type: ignore[arg-type]
    for _ in range(client_doors.MAX_CONNECTIONS):
        doors.admit(T(), f"10.0.{_ // 250}.{_ % 250}")  # type: ignore[arg-type]
    assert not doors.admit(T(), "192.168.9.9")  # type: ignore[arg-type]
    assert doors.admit(T(), "127.0.0.1")  # type: ignore[arg-type]


def test_the_refusal_echoes_no_malformed_host_and_brackets_ipv6() -> None:
    # A bare LF ends the Host line: what follows is never echoed with it.
    split = client_doors.refusal(
        b"GET /x HTTP/1.1\r\nHost: a.lan\nSet-Cookie: x=1\r\n\r\n", 8080, None
    )
    assert b"\r\nSet-Cookie" not in split and b"location: https://a.lan/x\r\n" in split
    # A malformed Host is dropped; this machine's IPv6 address is bracketed.
    body = client_doors.refusal(b"GET /x HTTP/1.1\r\nHost: a.lan x=1\r\n\r\n", 8080, "fd00::5")
    assert b"x=1" not in body
    assert b"location: https://[fd00::5]:8080/x\r\n" in body
    good = client_doors.refusal(b"GET /v1 HTTP/1.1\r\nHost: 192.168.1.10:8080\r\n\r\n", 8080, None)
    assert b"location: https://192.168.1.10:8080/v1\r\n" in good


async def test_once_an_app_answered_on_its_tls_port_tls_never_goes_to_the_plain_one() -> None:
    plain = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    plain_port = int(plain.sockets[0].getsockname()[1])
    tls = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    tls_port = int(tls.sockets[0].getsockname()[1])
    doors = client_doors.Doors()
    pump = client_doors._PumpDoor(
        doors,
        client_doors.Door("app:x", 0),
        "127.0.0.1",
        lambda: plain_port,
        lambda: tls_port,
        lambda: plain_port,
    )
    seen: list[int] = []

    async def connect(fallback: object) -> None:
        p = client_doors._Pump(
            lambda: tls_port,
            "app:x",
            fallback,  # type: ignore[arg-type]
            on_connected=pump._tls_connected,
        )
        real = asyncio.get_running_loop().create_connection

        async def spy(factory, host, port):  # type: ignore[no-untyped-def]
            seen.append(port)
            return await real(factory, host, port)

        asyncio.get_running_loop().create_connection = spy  # type: ignore[method-assign]
        try:
            await p._connect(tls_port)
        finally:
            del asyncio.get_running_loop().create_connection  # type: ignore[attr-defined]

    await connect(lambda: plain_port)
    assert pump.tls_answered and seen == [tls_port]
    tls.close()
    await tls.wait_closed()
    seen.clear()
    await connect(None if pump.tls_answered else (lambda: plain_port))
    assert seen == [tls_port]  # refused, and not carried to the plain port
    plain.close()


def test_the_agent_lists_only_names_the_root_signs() -> None:
    assert install_names.signable("eugene.lan")
    assert install_names.signable("192.168.1.5")
    assert not install_names.signable("169.254.1.1")
    assert not install_names.signable("*.home.arpa")
    assert not install_names.signable("eugene.example.com")


def test_the_bundle_served_to_devices_names_no_inherited_ca(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    other = tmp_path / "caddy.pem"
    other.write_text("-----BEGIN CERTIFICATE-----\nCADDY\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv("SSL_CERT_FILE", str(other))
    served = client_tls.bundle_text("INSTALL CA")
    local = client_tls.bundle_text("INSTALL CA", inherited=True)
    assert "CADDY" not in served and "CADDY" in local
    assert served.rstrip().endswith("INSTALL CA")
