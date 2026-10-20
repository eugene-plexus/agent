"""The client doors: TLS and plain HTTP on the same ports (LT2).

`specs/docs/design/tls-for-clients.md` §5 (LT2-C3: no new port). A browser,
Claude Code, Open WebUI or a script reaches Eugene at three kinds of door:

    8079        the console: this agent's own app
    8080        the gateway, a child process
    app ports   each app (Workbench from 8190), a child process

Each takes TLS and plain HTTP on the same port, told apart by the first byte
of a connection (a TLS handshake starts with `0x16`, an HTTP request with a
letter), as the machine port does (LT1). The leaf they present comes from the
install CA (`client_tls`).

**Terminated in this process, on the connection itself.** The first bytes
are read by a small protocol (`_Sniff`); a TLS connection then gets an
`asyncio.sslproto.SSLProtocol` over the same transport, a plain one the inner
protocol directly. So:

- **8079** hands either branch to uvicorn's own HTTP protocol, inside the
  agent's one server: the app sees the real socket, the real peer and
  `scheme: https` on the TLS branch. Nothing stands in for the caller, so the
  `/link` pages still read the owner of the caller's own loopback socket
  (`loopback_peer`, LT1 §7.2's first trap), and there is no hop to strip
  headers on.
- **8080 and app ports** pump either branch, decrypted, to the child on a
  loopback port of its own (`_Pump`). The gateway reads no peer address; an
  app is told the scheme by which of its two loopback ports a connection
  arrives at (the TLS one, `EUGENE_PLEXUS_APP_TLS_PORT`).

**Plain HTTP from another machine** is counted per door and client address
(`client_plain`), and refused, with the `https://` URL named, once the owner
turns it off (LT2-C4). Loopback is always plain and never counted.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import ssl
from asyncio import sslproto
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import peer as peer_module

log = logging.getLogger(__name__)

#: How long a connection may take to send its first byte.
FIRST_BYTE_SECONDS = 10.0
HANDSHAKE_SECONDS = 30
SHUTDOWN_SECONDS = 30
#: A pumped connection that moves no byte either way for this long is closed.
#: Long, because a model can think for minutes before its first token.
IDLE_SECONDS = 900.0
MAX_CONNECTIONS = 2048
MAX_PER_PEER = 512
TLS_RECORD = 0x16
_HIGH_WATER = 256 * 1024


@dataclass
class Door:
    """One client door, as `GET /v1/node` reports it."""

    name: str
    port: int
    listening: bool = False
    error: str | None = None


def feed(protocol: asyncio.BaseProtocol, data: bytes) -> None:
    """Hand bytes already read to a protocol, buffered or not."""
    if isinstance(protocol, asyncio.BufferedProtocol):
        view = memoryview(data)
        while view:
            buffer = memoryview(protocol.get_buffer(len(view)))
            n = min(len(buffer), len(view))
            buffer[:n] = view[:n]
            protocol.buffer_updated(n)
            view = view[n:]
    else:
        protocol.data_received(data)  # type: ignore[attr-defined]


def _request_target(data: bytes) -> tuple[str | None, str]:
    """The `Host` and the path of a plain request's first bytes, when they hold them."""
    head = data.split(b"\r\n\r\n", 1)[0].decode("latin-1", "replace")
    lines = head.split("\r\n")
    parts = lines[0].split(" ")
    path = parts[1] if len(parts) >= 2 and parts[1].startswith("/") else "/"
    host = None
    for line in lines[1:]:
        name, _, value = line.partition(":")
        if name.strip().lower() == "host":
            host = value.strip() or None
    return host, path


def refusal(data: bytes, port: int, fallback_host: str | None) -> bytes:
    """A plain request from another machine, while plain is off: a 403 that
    names the same address over `https://`."""
    host, path = _request_target(data)
    authority = host or (f"{fallback_host}:{port}" if fallback_host else None)
    url = f"https://{authority}{path}" if authority else f"https on port {port}"
    body = (
        f"This install takes only encrypted connections from other machines. Open {url} instead.\n"
    ).encode()
    return (
        b"HTTP/1.1 403 Forbidden\r\n"
        b"content-type: text/plain; charset=utf-8\r\n"
        + f"content-length: {len(body)}\r\n".encode()
        + (f"location: {url}\r\n".encode() if authority else b"")
        + b"connection: close\r\n\r\n"
        + body
    )


class Doors:
    """The doors of this node, the context they present and their counts."""

    def __init__(self, counts: Any = None) -> None:
        self._context: ssl.SSLContext | None = None
        self.plain_allowed: Callable[[], bool] = lambda: True
        self.fallback_host: Callable[[], str | None] = lambda: None
        self.counts = counts
        self.doors: dict[str, Door] = {}
        self._pumps: dict[str, _PumpDoor] = {}
        self._pending: set[_Sniff] = set()
        self._live: dict[asyncio.BaseTransport, str] = {}

    # ---- the certificate -----------------------------------------------------

    def set_context(self, context: ssl.SSLContext | None) -> None:
        """Every connection from now on presents this; open ones keep theirs."""
        self._context = context

    def context(self) -> ssl.SSLContext | None:
        return self._context

    # ---- bookkeeping shared by every door ------------------------------------

    def admit(self, transport: asyncio.BaseTransport, peer: str) -> bool:
        """Room for one more connection, in all and from this address.

        Counted by transport, not by protocol: uvicorn replaces the protocol
        on a WebSocket upgrade, and a count kept by the protocol would leak.
        Loopback has no per-address cap; every child of this agent is there.
        """
        for live in [t for t in self._live if t.is_closing()]:
            del self._live[live]
        if len(self._live) >= MAX_CONNECTIONS:
            return False
        if not peer_module.is_loopback(peer) and (
            sum(1 for p in self._live.values() if p == peer) >= MAX_PER_PEER
        ):
            return False
        self._live[transport] = peer
        return True

    def count_plain(self, door: str, peer: str) -> None:
        if self.counts is not None:
            self.counts.add(door, peer)

    def close_pending(self) -> None:
        """Connections that have not sent a byte yet, closed at shutdown."""
        for sniff in list(self._pending):
            sniff.close()

    # ---- 8079 ----------------------------------------------------------------

    def console_protocol(self, port: int) -> type[asyncio.Protocol]:
        """The class uvicorn builds for every connection to the console
        (`uvicorn.Config(http=...)`)."""
        from uvicorn.protocols.http.auto import AutoHTTPProtocol

        doors = self
        self.doors["console"] = Door("console", port, listening=True)

        class ConsoleProtocol(_Sniff):
            def __init__(
                self,
                config: Any,
                server_state: Any,
                app_state: Any,
                _loop: asyncio.AbstractEventLoop | None = None,
            ) -> None:
                def inner(_tls: bool) -> asyncio.BaseProtocol:
                    return AutoHTTPProtocol(  # type: ignore[call-arg,unused-ignore]
                        config=config, server_state=server_state, app_state=app_state, _loop=_loop
                    )

                super().__init__(doors, "console", port, inner)
                self._server_state = server_state

            def connection_made(self, transport: asyncio.BaseTransport) -> None:
                # Counted as uvicorn's own connection until it hands over, so
                # a shutdown closes it rather than waiting out its first byte.
                self._server_state.connections.add(self)
                super().connection_made(transport)

            def handed_over(self) -> None:
                self._server_state.connections.discard(self)

            def shutdown(self) -> None:
                self.close()

        return ConsoleProtocol

    # ---- 8080 and app ports --------------------------------------------------

    async def open_pump(
        self,
        name: str,
        *,
        bind_host: str,
        port: int,
        plain_target: Callable[[], int | None],
        tls_target: Callable[[], int | None] | None = None,
        fallback_target: Callable[[], int | None] | None = None,
    ) -> Door:
        """A door on `port` in front of a child on loopback. TLS connections
        go to `tls_target` when given, else to `plain_target`; when nothing
        answers at `tls_target` (an app that does not listen on a TLS port),
        to `fallback_target`."""
        await self.close(name)
        door = Door(name, port)
        self.doors[name] = door
        pump = _PumpDoor(
            self, door, bind_host, plain_target, tls_target or plain_target, fallback_target
        )
        self._pumps[name] = pump
        await pump.start()
        return door

    async def close(self, name: str) -> None:
        pump = self._pumps.pop(name, None)
        if pump is not None:
            await pump.stop()
        if name != "console":
            self.doors.pop(name, None)

    async def close_all(self) -> None:
        for name in list(self._pumps):
            await self.close(name)
        self.close_pending()


class _Sniff(asyncio.Protocol):
    """A connection's first bytes: TLS, plain, or refused."""

    def __init__(
        self,
        doors: Doors,
        door: str,
        port: int,
        inner: Callable[[bool], asyncio.BaseProtocol],
    ) -> None:
        self._doors, self._door, self._port, self._inner = doors, door, port, inner
        self._transport: asyncio.Transport | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._peer = "unknown"
        self._local = False
        self._done = False

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport  # type: ignore[assignment]
        peername = transport.get_extra_info("peername")
        self._peer = str(peername[0]) if peername else "unknown"
        # This machine: loopback, or one of its own addresses (a connection
        # whose source is the address it arrived at). A process here that
        # opens the console at the LAN address is not another machine.
        sockname = transport.get_extra_info("sockname")
        self._local = peer_module.is_loopback(self._peer) or bool(
            sockname and str(sockname[0]) == self._peer
        )
        if not self._doors.admit(transport, self._peer):
            log.warning("%s door: too many connections; closed one from %s", self._door, self._peer)
            self._done = True
            transport.close()
            return
        self._doors._pending.add(self)
        self._timer = asyncio.get_running_loop().call_later(FIRST_BYTE_SECONDS, self.close)

    def handed_over(self) -> None:
        """Called once the connection belongs to its inner protocol."""

    def data_received(self, data: bytes) -> None:
        transport = self._transport
        if transport is None or self._done or not data:
            return
        self._settle()
        context = self._doors.context()
        if data[0] == TLS_RECORD and context is not None:
            loop = asyncio.get_running_loop()
            # A failed handshake is the client's business, said in no log:
            # the waiter's exception is taken so nothing reports it unread.
            waiter = loop.create_future()
            waiter.add_done_callback(lambda f: f.cancelled() or f.exception())
            tls = sslproto.SSLProtocol(
                loop,
                self._inner(True),
                context,
                waiter,
                server_side=True,
                ssl_handshake_timeout=HANDSHAKE_SECONDS,
                ssl_shutdown_timeout=SHUTDOWN_SECONDS,
            )
            self._hand(transport, tls, data)
            return
        if not self._local:
            self._doors.count_plain(self._door, self._peer)
            if not self._doors.plain_allowed():
                log.info("%s door: refused plain HTTP from %s", self._door, self._peer)
                transport.write(refusal(data, self._port, self._doors.fallback_host()))
                transport.close()
                return
        self._hand(transport, self._inner(False), data)

    def _hand(self, transport: asyncio.Transport, protocol: Any, data: bytes) -> None:
        self.handed_over()
        transport.set_protocol(protocol)
        protocol.connection_made(transport)
        feed(protocol, data)

    def _settle(self) -> None:
        self._done = True
        if self._timer is not None:
            self._timer.cancel()
        self._doors._pending.discard(self)

    def _finish(self) -> None:
        self._settle()
        self.handed_over()

    def close(self) -> None:
        if not self._done:
            self._finish()
            if self._transport is not None:
                self._transport.close()

    def connection_lost(self, exc: Exception | None) -> None:
        if not self._done:
            self._finish()


# --------------------------------------------------------------------------- #
# The pump: 8080 and app ports
# --------------------------------------------------------------------------- #


class _PumpDoor:
    def __init__(
        self,
        doors: Doors,
        door: Door,
        bind_host: str,
        plain_target: Callable[[], int | None],
        tls_target: Callable[[], int | None],
        fallback_target: Callable[[], int | None] | None = None,
    ) -> None:
        self._doors, self._door, self._bind_host = doors, door, bind_host
        self._plain_target, self._tls_target = plain_target, tls_target
        self._fallback = fallback_target
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()

        def factory() -> asyncio.Protocol:
            def inner(tls: bool) -> asyncio.BaseProtocol:
                if tls:
                    return _Pump(self._tls_target, self._door.name, self._fallback)
                return _Pump(self._plain_target, self._door.name)

            return _Sniff(self._doors, self._door.name, self._door.port, inner)

        try:
            self._server = await loop.create_server(
                factory, host=self._bind_host, port=self._door.port, reuse_address=True
            )
        except OSError as exc:
            from . import ports

            holder = ports.describe_holder(self._door.port)
            self._door.error = (
                f"port {self._door.port} is in use{f' by {holder}' if holder else ''}"
                if exc.errno in (98, 10048, 48)
                else f"could not listen on {self._bind_host}:{self._door.port}: {exc}"
            )
            self._door.listening = False
            raise
        self._door.listening, self._door.error = True, None
        log.info(
            "%s door listening on %s:%d (TLS and plain HTTP)",
            self._door.name,
            self._bind_host,
            self._door.port,
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        self._door.listening = False


class _Pump(asyncio.Protocol):
    """One client connection, piped to a loopback port of this machine."""

    def __init__(
        self,
        target: Callable[[], int | None],
        door: str,
        fallback: Callable[[], int | None] | None = None,
    ) -> None:
        self._target, self._door, self._fallback = target, door, fallback
        self._client: asyncio.Transport | None = None
        self._upstream: asyncio.Transport | None = None
        self._buffer: list[bytes] = []
        self._idle: asyncio.TimerHandle | None = None
        self._closed = False

    # -- the client side --

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._client = transport  # type: ignore[assignment]
        port = self._target()
        if port is None:
            log.warning("%s door: nothing listens behind it yet", self._door)
            transport.close()
            return
        transport.pause_reading()  # type: ignore[attr-defined]
        asyncio.get_running_loop().create_task(self._connect(port))
        self._touch()

    async def _connect(self, port: int) -> None:
        loop = asyncio.get_running_loop()
        try:
            transport, _ = await loop.create_connection(lambda: _Upstream(self), "127.0.0.1", port)
        except OSError as exc:
            fallback = self._fallback() if self._fallback is not None else None
            if fallback is None or fallback == port:
                # Debug: a child starting (or restarting) is the usual cause,
                # and the supervisor reports a child that stays down.
                log.debug("%s door: 127.0.0.1:%d did not answer: %s", self._door, port, exc)
                self.close()
                return
            self._fallback = None
            await self._connect(fallback)
            return
        if self._closed:
            transport.close()
            return
        self._upstream = transport
        transport.set_write_buffer_limits(high=_HIGH_WATER)
        for chunk in self._buffer:
            transport.write(chunk)
        self._buffer.clear()
        if self._client is not None and not self._client.is_closing():
            self._client.resume_reading()

    def data_received(self, data: bytes) -> None:
        self._touch()
        if self._upstream is None:
            self._buffer.append(data)
        else:
            self._upstream.write(data)

    def eof_received(self) -> bool:
        if self._upstream is not None and self._upstream.can_write_eof():
            self._upstream.write_eof()
            return True
        return False

    def pause_writing(self) -> None:
        if self._upstream is not None:
            self._upstream.pause_reading()

    def resume_writing(self) -> None:
        if self._upstream is not None:
            self._upstream.resume_reading()

    def connection_lost(self, exc: Exception | None) -> None:
        self.close()

    # -- the upstream side, called by _Upstream --

    def from_upstream(self, data: bytes) -> None:
        self._touch()
        if self._client is not None:
            self._client.write(data)

    def upstream_eof(self) -> None:
        if self._client is not None and self._client.can_write_eof():
            self._client.write_eof()
        else:
            self.close()

    def pause_client(self) -> None:
        if self._client is not None:
            with contextlib.suppress(RuntimeError, AttributeError):
                self._client.pause_reading()

    def resume_client(self) -> None:
        if self._client is not None:
            with contextlib.suppress(RuntimeError, AttributeError):
                self._client.resume_reading()

    def _touch(self) -> None:
        if self._idle is not None:
            self._idle.cancel()
        if not self._closed:
            self._idle = asyncio.get_running_loop().call_later(IDLE_SECONDS, self.close)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._idle is not None:
            self._idle.cancel()
        for transport in (self._client, self._upstream):
            if transport is not None:
                transport.close()


class _Upstream(asyncio.Protocol):
    def __init__(self, pump: _Pump) -> None:
        self._pump = pump

    def data_received(self, data: bytes) -> None:
        self._pump.from_upstream(data)

    def eof_received(self) -> bool:
        self._pump.upstream_eof()
        return True

    def pause_writing(self) -> None:
        self._pump.pause_client()

    def resume_writing(self) -> None:
        self._pump.resume_client()

    def connection_lost(self, exc: Exception | None) -> None:
        self._pump.close()
