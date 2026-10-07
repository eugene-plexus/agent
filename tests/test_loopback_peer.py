"""Who is at the other end of a loopback connection (J14a.2,
`person-held-keys.md` §12.4): each platform's reader, against what the
platform really produced when it was measured, and against a real
connection on the platform the tests run on."""

from __future__ import annotations

import socket
import struct
import sys
import threading
from pathlib import Path

import pytest

from eugene_plexus_agent import loopback_peer
from eugene_plexus_agent.loopback_peer import PeerUnknown

# --- Linux: /proc/net/tcp, as WSL2's kernel 6.18 wrote it ------------------------------
# A server on 127.0.0.1:47143 (uid 1000); a client from uid 1000 on 37348 and
# from a second account, uid 1001, on 37350. Each connection is two rows.
PROC_NET_TCP = """\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:B827 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 30230 1 0000000000000000 100 0 0 10 0
   1: 0100007F:91E4 0100007F:B827 01 00000000:00000000 00:00000000 00000000  1000        0 30240 1 0000000000000000 20 4 30 10 -1
   2: 0100007F:B827 0100007F:91E4 01 00000000:00000000 00:00000000 00000000  1000        0 30241 1 0000000000000000 20 4 30 10 -1
   3: 0100007F:91E6 0100007F:B827 01 00000000:00000000 00:00000000 00000000  1001        0 20783 1 0000000000000000 20 4 30 10 -1
   4: 0100007F:B827 0100007F:91E6 01 00000000:00000000 00:00000000 00000000  1000        0 20784 1 0000000000000000 20 4 30 10 -1
"""


@pytest.fixture
def proc_table(tmp_path: Path) -> Path:
    table = tmp_path / "tcp"
    table.write_text(PROC_NET_TCP, encoding="ascii")
    return table


@pytest.mark.skipif(sys.byteorder != "little", reason="the fixture is a little-endian kernel's")
def test_linux_reads_the_uid_on_the_connecting_socket(proc_table: Path) -> None:
    assert loopback_peer._linux(37348, 47143, proc_table) == "1000"
    assert loopback_peer._linux(37350, 47143, proc_table) == "1001"


@pytest.mark.skipif(sys.byteorder != "little", reason="the fixture is a little-endian kernel's")
def test_linux_never_takes_the_servers_own_row(proc_table: Path) -> None:
    """The accepted socket (server port first) is the agent's own, uid 1000:
    asking with the ports swapped must not find the client's."""
    assert loopback_peer._linux(47143, 37350, proc_table) == "1000"


def test_linux_refuses_what_it_cannot_find_or_cannot_tell(tmp_path: Path, proc_table: Path) -> None:
    with pytest.raises(PeerUnknown, match="could not be found"):
        loopback_peer._linux(40000, 47143, proc_table)
    doubled = tmp_path / "doubled"
    doubled.write_text(
        PROC_NET_TCP + "   5: 0100007F:91E6 0100007F:B827 01 00000000:00000000 00:00000000 00000000"
        "  1000        0 99 1 0 20 4 30 10 -1\n",
        encoding="ascii",
    )
    with pytest.raises(PeerUnknown):
        loopback_peer._linux(37350, 47143, doubled)
    with pytest.raises(PeerUnknown, match="could not be read"):
        loopback_peer._linux(37350, 47143, tmp_path / "missing")


# --- macOS: net.inet.tcp.pcblist_n, as macOS 15.7.9 wrote it -------------------------------
# A server on 49163 (uid 501); a client from a second account, uid 502, on
# 49165. Two groups, one per end, each opening with its inpcb record.

SERVER_END_SOCKET = [
    104,
    1,
    3646113971,
    893287475,
    1,
    0,
    131072,
    3378478627,
    2934898789,
    6,
    2,
    0,
    0,
    0,
    0,
    0,
    501,
    1177,
    0,
    3267,
    0,
    128,
    16779520,
    2,
    0,
    0,
]
SERVER_END_INPCB = (
    "6800000010000000237e5fc96500efaec00dc00b9a4a730c742ea3a1360000000000000000008000000c0b00"
    "014000000000000000000000000000007f0000010000000000000000000000007f0000010000000000000000"
    "0000000000000000bb6c3f9216000000"
)
CLIENT_END_SOCKET = [
    104,
    1,
    3372127673,
    2958381970,
    1,
    0,
    131072,
    3274588751,
    2707123998,
    6,
    2,
    0,
    0,
    0,
    0,
    0,
    502,
    1235,
    0,
    3266,
    0,
    128,
    67111168,
    2,
    0,
    0,
]
CLIENT_END_INPCB = (
    "68000000100000004f422ec31e6f5ba1c00bc00d60b2ffd91caf782a340000000000000040088000000000"
    "00014000000000000000000000000000007f0000010000000000000000000000007f0000010000000000000000"
    "0000000000000000831029b616000000"
)


def _record(kind: int, body: bytes) -> bytes:
    data = struct.pack("<II", 8 + len(body), kind) + body
    return data + b"\0" * (-len(data) % 8)


def _group(inpcb_hex: str, socket_words: list[int]) -> bytes:
    inpcb = bytes.fromhex(inpcb_hex)
    inpcb += b"\0" * (104 - len(inpcb))
    sock = struct.pack(f"<{len(socket_words)}I", *socket_words)
    return (
        inpcb
        + sock
        + _record(0x002, b"\0" * 16)
        + _record(0x004, b"\0" * 16)
        + _record(0x008, b"\0" * 8)
        + _record(0x020, b"\0" * 24)
    )


def _pcblist(*groups: bytes) -> bytes:
    header = struct.pack("<IIQQ", 24, len(groups), 7, 9)
    return header + b"".join(groups)


def test_macos_reads_the_uid_on_the_connecting_socket() -> None:
    data = _pcblist(
        _group(SERVER_END_INPCB, SERVER_END_SOCKET), _group(CLIENT_END_INPCB, CLIENT_END_SOCKET)
    )
    assert loopback_peer._macos_from(data, 49165, 49163) == "502"
    # The other end of the same connection is the server's own socket.
    assert loopback_peer._macos_from(data, 49163, 49165) == "501"


def test_macos_refuses_what_it_cannot_find_or_cannot_tell() -> None:
    data = _pcblist(_group(SERVER_END_INPCB, SERVER_END_SOCKET))
    with pytest.raises(PeerUnknown, match="could not be found"):
        loopback_peer._macos_from(data, 49165, 49163)
    # The same ports between addresses other than 127.0.0.1 are not this connection.
    elsewhere = CLIENT_END_INPCB.replace("7f000001", "0a000001")
    with pytest.raises(PeerUnknown, match="could not be found"):
        loopback_peer._macos_from(_pcblist(_group(elsewhere, CLIENT_END_SOCKET)), 49165, 49163)
    # A socket record of another size is a layout this reader does not know.
    longer = [112, *CLIENT_END_SOCKET[1:], 0, 0]
    with pytest.raises(PeerUnknown, match="lays its connection table out differently"):
        loopback_peer._macos_from(_pcblist(_group(CLIENT_END_INPCB, longer)), 49165, 49163)
    with pytest.raises(PeerUnknown):
        loopback_peer._macos_from(b"\x18\0\0", 49165, 49163)


# --- this platform, for real ------------------------------------------------------------


def test_a_real_loopback_connection_is_this_account() -> None:
    """The reader for the platform the tests run on, against a connection
    this process makes to itself."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    accepted, address = server.accept()
    try:
        assert address[0] == "127.0.0.1"
        assert loopback_peer.peer_account(address[1], port) == loopback_peer.own_account()
    finally:
        for s in (client, accepted, server):
            s.close()


def test_a_port_with_no_connection_is_refused() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    try:
        with pytest.raises(PeerUnknown):
            loopback_peer.peer_account(1, port)
    finally:
        server.close()


def test_threads_do_not_change_the_answer() -> None:
    """A connection made from another thread of this process is still this account."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    made: list[socket.socket] = []
    thread = threading.Thread(
        target=lambda: made.append(socket.create_connection(("127.0.0.1", port)))
    )
    thread.start()
    accepted, address = server.accept()
    thread.join()
    try:
        assert loopback_peer.peer_account(address[1], port) == loopback_peer.own_account()
    finally:
        for s in (*made, accepted, server):
            s.close()
