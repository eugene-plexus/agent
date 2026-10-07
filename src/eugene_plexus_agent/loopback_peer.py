"""Who is at the other end of a loopback TCP connection to this agent.

The pages at the machine (`routes/site_link.py`) serve a person by the OS
account that owns the connecting socket, read from the operating system and
never from the request (J14a, `person-held-keys.md` §12). Each platform's
answer was measured before it was trusted (§12.4, 2026-10-06):

- **Windows**: the TCP table names the process at the far end
  (`GetExtendedTcpTable`) and its token names the account. Unelevated, a
  process of another account cannot be opened, which is a refusal.
- **Linux**: `/proc/net/tcp` carries each socket's uid, readable by any
  account; a second account's connection reads as its own uid.
- **macOS**: `sysctl net.inet.tcp.pcblist_n` carries each socket's uid
  (`xsocket_n.so_uid`), readable by any account, on macOS 14, 15 and 26.
  An unprivileged `lsof` sees only its own processes and cannot answer.

Anything unexpected — no row, two rows, a layout this code does not know —
is `PeerUnknown`: refused, never guessed. The account is a SID on Windows
and a uid in decimal elsewhere, as the links file records it.
"""

from __future__ import annotations

import os
import socket
import struct
import sys
from pathlib import Path

LOOPBACK = "127.0.0.1"


class PeerUnknown(Exception):
    """The connection's account could not be read; the message says why."""


def own_account() -> str:
    """This process's account, in the links file's form."""
    if sys.platform != "win32":
        return str(os.getuid())
    import win32api
    import win32con
    import win32security

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    return str(win32security.ConvertSidToStringSid(user))


def peer_account(client_port: int, server_port: int) -> str:
    """The account that owns the loopback connection from `client_port` to
    this agent's `server_port`."""
    if sys.platform == "win32":
        return _windows(client_port, server_port)
    if sys.platform == "linux":
        return _linux(client_port, server_port, Path("/proc/net/tcp"))
    if sys.platform == "darwin":
        return _macos(client_port, server_port)
    raise PeerUnknown("This system cannot say which account opened this page.")


# --- Windows -----------------------------------------------------------------------


def _windows(client_port: int, server_port: int) -> str:
    # For the type checker as much as the runtime (process_io.py says why).
    if sys.platform != "win32":
        raise PeerUnknown("a connection's owner is read with a Windows-only API")
    import ctypes
    from ctypes import wintypes

    import win32api
    import win32con
    import win32security

    iphlpapi = ctypes.WinDLL("iphlpapi")

    class Row(ctypes.Structure):
        _fields_ = [
            ("state", wintypes.DWORD),
            ("local_addr", wintypes.DWORD),
            ("local_port", wintypes.DWORD),
            ("remote_addr", wintypes.DWORD),
            ("remote_port", wintypes.DWORD),
            ("pid", wintypes.DWORD),
        ]

    size = wintypes.DWORD(0)
    # AF_INET (2), TCP_TABLE_OWNER_PID_CONNECTIONS (4).
    iphlpapi.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 4, 0)
    buffer = ctypes.create_string_buffer(size.value + 4096)
    size = wintypes.DWORD(len(buffer))
    if iphlpapi.GetExtendedTcpTable(buffer, ctypes.byref(size), False, 2, 4, 0) != 0:
        raise PeerUnknown("This machine's connection table could not be read.")
    count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD))[0]
    rows = ctypes.cast(
        ctypes.addressof(buffer) + ctypes.sizeof(wintypes.DWORD), ctypes.POINTER(Row * count)
    ).contents
    loopback = int.from_bytes(socket.inet_aton(LOOPBACK), "little")

    def port(value: int) -> int:
        return ((value & 0xFF) << 8) | ((value >> 8) & 0xFF)

    owners = {
        row.pid
        for row in rows
        if row.local_addr == loopback
        and row.remote_addr == loopback
        and port(row.local_port) == client_port
        and port(row.remote_port) == server_port
    }
    if len(owners) != 1:
        raise PeerUnknown("The program that opened this page could not be found.")
    pid = owners.pop()
    try:
        process = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        token = win32security.OpenProcessToken(process, win32con.TOKEN_QUERY)
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        return str(win32security.ConvertSidToStringSid(user))
    except Exception:
        raise PeerUnknown(
            "The account of the program that opened this page could not be read."
        ) from None


# --- Linux -------------------------------------------------------------------------


def _hex_endpoint(port: int) -> str:
    """`/proc/net/tcp`'s spelling of 127.0.0.1:port: the address as the
    kernel's 32-bit word in this machine's byte order, then the port."""
    address = struct.unpack("=I", socket.inet_aton(LOOPBACK))[0]
    return f"{address:08X}:{port:04X}"


def _linux(client_port: int, server_port: int, table: Path) -> str:
    try:
        lines = table.read_text(encoding="ascii").splitlines()[1:]
    except OSError:
        raise PeerUnknown("This machine's connection table could not be read.") from None
    local, remote = _hex_endpoint(client_port), _hex_endpoint(server_port)
    uids = set()
    for line in lines:
        fields = line.split()
        # sl local_address rem_address st tx:rx tr:when retrnsmt uid ...
        if len(fields) < 8:
            continue
        if fields[1] == local and fields[2] == remote:
            uids.add(fields[7])
    if len(uids) != 1:
        raise PeerUnknown("The program that opened this page could not be found.")
    uid = uids.pop()
    if not uid.isdigit():
        raise PeerUnknown("The account of the program that opened this page could not be read.")
    return uid


# --- macOS -------------------------------------------------------------------------

#: `net.inet.tcp.pcblist_n` (xnu `bsd/netinet/in_pcblist.c`): after an
#: `xinpgen` header, one group of records per connection, each record
#: `u_int32 len, u_int32 kind` and padded to 8 bytes.
_XSO_SOCKET = 0x001
_XSO_INPCB = 0x010
_XSO_TCPCB = 0x020
#: `xsocket_n` (`#pragma pack(4)`): its length, and `so_uid`'s offset.
_XSOCKET_N_LEN = 104
_SO_UID_AT = 64
#: `xinpcb_n`: `inp_fport` and `inp_lport` (network order) follow `xi_inpp`;
#: `inp_vflag` (INP_IPV4 = 1); the foreign and local `in_addr_4in6`, each an
#: IPv4 address after twelve bytes of padding.
_PORTS_AT = 16
_VFLAG_AT = 44
_FADDR_AT = 60
_LADDR_AT = 76


def _pcblist() -> bytes:  # pragma: no cover - macOS only, measured on GitHub's runners
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    name = b"net.inet.tcp.pcblist_n"
    for _ in range(4):
        size = ctypes.c_size_t(0)
        if libc.sysctlbyname(name, None, ctypes.byref(size), None, 0) != 0:
            break
        buffer = ctypes.create_string_buffer(size.value + 65536)
        size = ctypes.c_size_t(len(buffer))
        if libc.sysctlbyname(name, buffer, ctypes.byref(size), None, 0) == 0:
            return buffer.raw[: size.value]
    raise PeerUnknown("This machine's connection table could not be read.")


def _macos(client_port: int, server_port: int) -> str:
    return _macos_from(_pcblist(), client_port, server_port)


def _macos_from(data: bytes, client_port: int, server_port: int) -> str:
    """The uid on the one socket whose local port is `client_port` and whose
    foreign port is `server_port`."""
    try:
        offset = (struct.unpack_from("<I", data, 0)[0] + 7) & ~7
        uids = set()
        group: dict[int, bytes] = {}
        while offset + 8 <= len(data):
            length, kind = struct.unpack_from("<II", data, offset)
            if length < 8:
                break
            group[kind] = data[offset : offset + length]
            offset += (length + 7) & ~7
            if kind != _XSO_TCPCB:
                continue
            sock, inp = group.get(_XSO_SOCKET), group.get(_XSO_INPCB)
            group = {}
            if sock is None or inp is None or len(inp) < _LADDR_AT + 4:
                continue
            fport, lport = struct.unpack_from(">HH", inp, _PORTS_AT)
            if (lport, fport) != (client_port, server_port):
                continue
            loopback = socket.inet_aton(LOOPBACK)
            if (
                not inp[_VFLAG_AT] & 0x1
                or inp[_FADDR_AT : _FADDR_AT + 4] != loopback
                or inp[_LADDR_AT : _LADDR_AT + 4] != loopback
            ):
                continue  # the same ports between other addresses: not this connection
            if len(sock) != _XSOCKET_N_LEN:
                raise PeerUnknown(
                    "This version of macOS lays its connection table out differently."
                )
            uids.add(struct.unpack_from("<I", sock, _SO_UID_AT)[0])
    except struct.error:
        raise PeerUnknown("This machine's connection table could not be read.") from None
    if len(uids) != 1:
        raise PeerUnknown("The program that opened this page could not be found.")
    return str(uids.pop())
