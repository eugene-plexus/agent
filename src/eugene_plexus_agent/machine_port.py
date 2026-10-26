"""The machine port: this node's one TLS door for the other machines (LT1).

`specs/docs/design/tls-between-machines.md` §3.1. Every connection another
machine of the install opens to this one arrives on 8083 (LT-C1: no new
port), over TLS with this node's machine key, and is forwarded on loopback
to the listener it names:

    https://<node>:8083/agent/...          this agent's own API, in-process
    https://<node>:8083/control/...        control, where this node runs it
    https://<node>:8083/drivers/<name>/... an inference-driver or tool-driver here

**The first-byte switch.** Until the install requires encryption, a machine
that has not updated still speaks plain HTTP to the root's 8083, where
control used to listen. So the front reads each connection's first byte: a
TLS handshake starts with `0x16`, an HTTP request with a letter. TLS goes to
the machine routes; plain HTTP goes to control, as it did, from a machine on
the network only while encryption is not required.

**How the far side keeps the real peer** (§7.2's first trap). Each branch is
a uvicorn on loopback, so without care every remote caller would look local,
and control and the agent trust a loopback peer's `x-eugene-plexus-peer`. The
front records which loopback source port carries which remote peer before
it sends a byte, both inner apps put that peer back into the ASGI scope, and
every hop strips the forwarding and peer headers a caller sent and sets the
peer header itself, as `/api/proxy` does. The agent's own API is served
in-process with the real peer in the scope, so no loopback socket ever
stands in for a remote caller where `/link` identifies a person by the
socket's owner (`loopback_peer`).

**What it never does.** It resolves only against this node's own topology,
never asking control where something is, so inference keeps working with the
root down. It streams both ways with no write timeout (bodies reach 16 MiB),
and it translates no token: every caller already sends a token addressed to
the target, which checks it as before. The port adds encryption and this
node's identity, not a second authority.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import socket
import ssl
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import peer as peer_module
from ._http import internal_client
from ._private_files import write_private

log = logging.getLogger(__name__)

#: LT-C1, taken: the machine port on every node.
MACHINE_PORT = 8083
#: How long a connection may take to send its first byte.
FIRST_BYTE_SECONDS = 15.0
#: How often a port another process holds is tried again.
RETRY_BIND_SECONDS = 15.0
#: A connection that moves no byte either way for this long is closed. Long,
#: because a model can think for minutes before its first token.
IDLE_SECONDS = 900.0
#: Open connections at once: in all, and from one address.
MAX_CONNECTIONS = 2048
MAX_PER_PEER = 512
_PUMP_CHUNK = 64 * 1024

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
#: What a caller may never carry through this port: hop-by-hop headers,
#: `host` (each hop sets its own), and every header that claims to say
#: where a request came from.
STRIPPED_REQUEST = HOP_BY_HOP | {"host", peer_module.PEER_HEADER} | peer_module.FORWARDING_HEADERS


@dataclass
class Status:
    """What `GET /v1/node` reports as `machine`."""

    port: int
    pin: str
    listening: bool = False
    error: str | None = None


@dataclass
class Routes:
    """Where this node's own listeners are, read live from its topology."""

    control: Callable[[], str | None]
    driver: Callable[[str], str | None]
    agent_app: Any
    required: Callable[[], bool]
    node_name: Callable[[], str | None] = field(default=lambda: None)


class _Peers:
    """Loopback source port -> the remote peer whose connection it carries."""

    def __init__(self) -> None:
        self._by_port: dict[int, tuple[str, int]] = {}

    def add(self, port: int, peer: tuple[str, int]) -> None:
        self._by_port[port] = peer

    def drop(self, port: int) -> None:
        self._by_port.pop(port, None)

    def restore(self, scope: dict[str, Any]) -> None:
        client = scope.get("client")
        if client and peer_module.is_loopback(client[0]):
            real = self._by_port.get(int(client[1]))
            if real is not None:
                scope["client"] = real


# --------------------------------------------------------------------------- #
# The routes, as ASGI
# --------------------------------------------------------------------------- #


def _peer_of(scope: dict[str, Any]) -> str:
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


def _clean_headers(scope: dict[str, Any]) -> list[tuple[bytes, bytes]]:
    headers = [
        (k, v)
        for k, v in scope.get("headers") or []
        if k.decode("latin-1").lower() not in STRIPPED_REQUEST
    ]
    headers.append((peer_module.PEER_HEADER.encode("latin-1"), _peer_of(scope).encode("latin-1")))
    return headers


async def _problem(send: Any, status: int, title: str, detail: str) -> None:
    import json

    body = json.dumps(
        {"type": "about:blank", "title": title, "status": status, "detail": detail}
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/problem+json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class _Forwarder:
    """Streams a request to a loopback listener of this node and back."""

    def __init__(self) -> None:
        # Loopback only, so the machine transport never carries these; no
        # write timeout and no read timeout, because a 16 MiB generate body
        # and a long token stream are both normal.
        self._client = internal_client(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __call__(
        self, scope: dict[str, Any], receive: Any, send: Any, base: str, path: str
    ) -> None:
        query = scope.get("query_string") or b""
        url = base.rstrip("/") + (path or "/") + ("?" + query.decode("latin-1") if query else "")
        headers = _clean_headers(scope)
        method = scope["method"]
        declared = {k.lower() for k, _ in headers}
        has_body = (
            method not in ("GET", "HEAD", "OPTIONS", "DELETE")
            or b"content-length" in declared
            or b"transfer-encoding" in {k.lower() for k, _ in scope.get("headers") or []}
        )

        async def body() -> AsyncIterator[bytes]:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if chunk:
                    yield chunk
                if not message.get("more_body", False):
                    return

        request = self._client.build_request(
            method, url, headers=headers, content=body() if has_body else None
        )
        try:
            upstream = await self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            await _problem(
                send, 502, "Not answering", f"{base} on this machine did not answer: {exc}"
            )
            return
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": upstream.status_code,
                    "headers": [
                        (k, v)
                        for k, v in upstream.headers.raw
                        if k.decode("latin-1").lower() not in HOP_BY_HOP
                    ],
                }
            )
            async for chunk in upstream.aiter_raw():
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            await upstream.aclose()


_REFUSED_IN_PATHS = ("%2e", "\\")


def _raw_path(scope: dict[str, Any]) -> str | None:
    """The path as it came on the wire, or None when it is not one this port
    takes: a `.` or `..` segment, an encoded dot, or a backslash
    (adversarial review, 2026-10-20). Routing and the plain branch's Job Site
    rule read this, and the forwarders send it as it is, so no hop can read a
    path another hop normalised differently (`/v1/sites/../nodes/enroll`)."""
    raw = scope.get("raw_path")
    text = raw.decode("latin-1") if isinstance(raw, bytes | bytearray) else str(scope["path"])
    lowered = text.lower()
    if not text.startswith("/") or any(token in lowered for token in _REFUSED_IN_PATHS):
        return None
    if any(segment in (".", "..") for segment in text.split("/")):
        return None
    return text


def _split(path: str, prefix: str) -> str | None:
    """The rest of `path` under `prefix`, or None when it is not under it."""
    if path == prefix:
        return "/"
    if path.startswith(prefix + "/"):
        return path[len(prefix) :]
    return None


class MachineApp:
    """The TLS branch: `/agent`, `/control`, `/drivers/<name>`."""

    def __init__(self, routes: Routes, peers: _Peers, forward: _Forwarder) -> None:
        self.routes, self.peers, self.forward = routes, peers, forward

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        self.peers.restore(scope)
        path = _raw_path(scope)
        if path is None:
            await _problem(send, 400, "Not a path", "This machine port takes no such path.")
            return
        rest = _split(path, "/agent")
        if rest is not None:
            from urllib.parse import unquote

            inner = dict(scope)
            inner["path"] = unquote(rest)
            inner["raw_path"] = rest.encode("latin-1")
            inner["root_path"] = ""
            inner["scheme"] = "https"
            inner["headers"] = _clean_headers(scope)
            await self.routes.agent_app(inner, receive, send)
            return
        rest = _split(path, "/control")
        if rest is not None:
            target = self.routes.control()
            if target is None:
                await _problem(
                    send,
                    404,
                    "No control root here",
                    "This machine does not run the install's control root.",
                )
                return
            await self.forward(scope, receive, send, target, rest)
            return
        rest = _split(path, "/drivers")
        if rest is not None and rest != "/":
            name, _, tail = rest.lstrip("/").partition("/")
            target = self.routes.driver(name)
            if target is None:
                await _problem(
                    send,
                    404,
                    "No such driver here",
                    f"This machine runs no inference-driver or tool-driver named {name!r}.",
                )
                return
            await self.forward(scope, receive, send, target, "/" + tail)
            return
        await _problem(
            send,
            404,
            "Not a machine route",
            "This machine port serves /agent, /control and /drivers/<name>.",
        )


_PUBLIC_PLAIN = re.compile(r"^(/v1/sites/[A-Za-z0-9_/-]+|/v1/trust/tls|/v1/trust/machines)$")


def _job_site_path(path: str) -> bool:
    """What another machine may still ask the root in plain HTTP once
    encryption is required, read from the raw path (`_raw_path`).

    A Job Site's own routes (`/v1/sites/...`, and the TLS key list it pins
    against): Job Sites keep the door they have (design §2, out of LT1's
    scope), so one that joined at a plain LAN address of the root reaches it
    there whatever `machineEncryption` says. And the signed machine list: a
    node whose pinned key for the root no longer matches (the root's machine
    was set up again) fetches it here to learn the root's new key, which it
    takes only if the list verifies against the root's identity key."""
    return bool(_PUBLIC_PLAIN.match(path))


class PlainApp:
    """The plain branch: control, as 8083 always was, while it is allowed."""

    def __init__(self, routes: Routes, peers: _Peers, forward: _Forwarder) -> None:
        self.routes, self.peers, self.forward = routes, peers, forward

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        self.peers.restore(scope)
        path = _raw_path(scope)
        if path is None:
            await _problem(send, 400, "Not a path", "This machine port takes no such path.")
            return
        local = peer_module.is_loopback(_peer_of(scope))
        if not local and self.routes.required() and not _job_site_path(path):
            log.info("refused plain HTTP from %s on the machine port", _peer_of(scope))
            await _problem(
                send,
                403,
                "Encryption between machines is required",
                "This install only talks between machines over TLS. Update the machine "
                "this came from; it runs a version without a machine port.",
            )
            return
        target = self.routes.control()
        if target is None:
            await _problem(
                send,
                400,
                "This is a machine port",
                "Other machines reach this one here over TLS (https).",
            )
            return
        await self.forward(scope, receive, send, target, path)


# --------------------------------------------------------------------------- #
# The listeners
# --------------------------------------------------------------------------- #


class _Inner:
    """A uvicorn on an ephemeral loopback port, run without its signal
    handling: this process's own server owns SIGINT, and uvicorn's `serve()`
    would take it over while this one ran."""

    def __init__(self, app: Any, context: ssl.SSLContext | None) -> None:
        import uvicorn

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = int(self.sock.getsockname()[1])
        # No logging settings: `Config()` applies them to the process-wide
        # `uvicorn.*` loggers, which silenced the agent's own server (its
        # shutdown lines and access log). Access lines are turned off on this
        # config alone, after it is built.
        config = uvicorn.Config(
            app,
            lifespan="off",
            log_config=None,
            log_level=None,
            proxy_headers=False,
            forwarded_allow_ips=[],
            timeout_keep_alive=30,
        )
        config.access_log = False
        config.load()
        config.ssl = context
        self.server = uvicorn.Server(config)
        self.server.lifespan = config.lifespan_class(config)

    async def start(self) -> None:
        await self.server.startup(sockets=[self.sock])

    async def stop(self) -> None:
        with contextlib.suppress(Exception):
            await self.server.shutdown(sockets=[self.sock])
        self.sock.close()


def server_context(
    certificate_pem: str,
    key_pem: str,
    directory: Path,
    *,
    context: ssl.SSLContext | None = None,
) -> ssl.SSLContext:
    """The machine key, as a TLS server. Python loads a chain only from a
    file, so the key and certificate are written owner-only beside
    `node.yaml`, from it, on every start. Given `context`, the key is loaded
    into it in place (`MachinePort.reload`)."""
    path = directory / "machine_tls.pem"
    write_private(path, key_pem.rstrip() + "\n" + certificate_pem.rstrip() + "\n")
    if context is None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=str(path), keyfile=str(path))
    return context


class MachinePort:
    """The front on 8083, its two inner listeners, and its status."""

    def __init__(
        self,
        *,
        routes: Routes,
        context: ssl.SSLContext,
        bind_host: str,
        port: int,
        pin: str,
        directory: Path | None = None,
    ) -> None:
        self.status = Status(port=port, pin=pin)
        self._context = context
        self._directory = directory
        self._bind_host = bind_host
        self._peers = _Peers()
        self._forward = _Forwarder()
        self._tls = _Inner(MachineApp(routes, self._peers, self._forward), context)
        self._plain = _Inner(PlainApp(routes, self._peers, self._forward), None)
        self._front: asyncio.Server | None = None
        self._binder: asyncio.Task[None] | None = None
        self._connections: set[asyncio.Task[None]] = set()
        self._per_peer: dict[str, int] = {}
        self.on_listening: Callable[[], None] | None = None

    async def start(self) -> None:
        await self._tls.start()
        await self._plain.start()
        if not await self._bind():
            self._binder = asyncio.create_task(self._keep_binding(), name="machine-port-bind")

    def reload(self, certificate_pem: str, key_pem: str, pin: str) -> None:
        """Present another machine key from the next handshake on, with no
        restart (row 3). Loaded into the live context: OpenSSL copies the
        certificate and key into each connection when it is made, so a
        handshake under way and every open session keep the key they began
        with, and only new ones get this one. Runs on the event loop, where
        every connection is made, so no handshake sees a certificate without
        its key. Raises when the key does not load, and the old one stays."""
        if self._directory is None:
            raise ValueError("this machine port was started without a key directory")
        server_context(certificate_pem, key_pem, self._directory, context=self._context)
        previous, self.status.pin = self.status.pin, pin
        log.info("the machine port presents key %s now (it presented %s)", pin, previous)

    async def _bind(self) -> bool:
        try:
            self._front = await asyncio.start_server(
                self._handle, host=self._bind_host, port=self.status.port
            )
        except OSError as exc:
            from . import ports

            holder = ports.describe_holder(self.status.port)
            error = (
                f"port {self.status.port} is in use{f' by {holder}' if holder else ''}"
                if exc.errno in (98, 10048, 48)
                else f"could not listen on {self._bind_host}:{self.status.port}: {exc}"
            )
            if error != self.status.error:
                log.warning(
                    "the machine port is not listening: %s. Other machines cannot reach this "
                    "one until it is; trying again every %d s",
                    error,
                    int(RETRY_BIND_SECONDS),
                )
            self.status.error, self.status.listening = error, False
            return False
        self.status.error, self.status.listening = None, True
        log.info(
            "machine port listening on %s:%d (TLS, key %s)",
            self._bind_host,
            self.status.port,
            self.status.pin,
        )
        if self.on_listening is not None:
            self.on_listening()
        return True

    async def _keep_binding(self) -> None:
        # A retry on a timer, not a wait on an event: nothing here learns
        # when another process lets go of the port.
        while True:
            await asyncio.sleep(RETRY_BIND_SECONDS)
            if await self._bind():
                return

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        remote = writer.get_extra_info("peername") or ("unknown", 0)
        source = str(remote[0])
        if (
            len(self._connections) >= MAX_CONNECTIONS
            or self._per_peer.get(source, 0) >= MAX_PER_PEER
        ):
            log.warning("machine port: too many connections; closed one from %s", source)
            writer.close()
            return
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
        self._per_peer[source] = self._per_peer.get(source, 0) + 1
        inner_writer: asyncio.StreamWriter | None = None
        local_port: int | None = None
        try:
            try:
                first = await asyncio.wait_for(reader.read(_PUMP_CHUNK), FIRST_BYTE_SECONDS)
            except TimeoutError:
                return
            if not first:
                return
            target = self._tls if first[0] == 0x16 else self._plain
            inner_reader, inner_writer = await asyncio.open_connection("127.0.0.1", target.port)
            local_port = int(inner_writer.get_extra_info("sockname")[1])
            # Before a byte reaches the inner listener: the request it reads
            # is attributed to this peer, never to loopback.
            self._peers.add(local_port, (source, int(remote[1])))
            inner_writer.write(first)
            upstream = asyncio.create_task(_pump(reader, inner_writer))
            downstream = asyncio.create_task(_pump(inner_reader, writer))
            done, _ = await asyncio.wait(
                {upstream, downstream}, return_when=asyncio.FIRST_COMPLETED
            )
            if upstream in done and not downstream.done():
                # The caller finished sending: let the answer finish, within
                # the idle bound the pump applies itself.
                await asyncio.wait({downstream})
            for pump in (upstream, downstream):
                pump.cancel()
            await asyncio.gather(upstream, downstream, return_exceptions=True)
        except (ConnectionError, OSError) as exc:
            log.debug("machine port connection ended: %s", exc)
        finally:
            if local_port is not None:
                self._peers.drop(local_port)
            for w in (inner_writer, writer):
                if w is not None:
                    w.close()
            left = self._per_peer.get(source, 1) - 1
            if left > 0:
                self._per_peer[source] = left
            else:
                self._per_peer.pop(source, None)
            if task is not None:
                self._connections.discard(task)

    async def stop(self) -> None:
        if self._binder is not None:
            self._binder.cancel()
            await asyncio.gather(self._binder, return_exceptions=True)
        if self._front is not None:
            self._front.close()
        for task in list(self._connections):
            task.cancel()
        await asyncio.gather(*self._connections, return_exceptions=True)
        await self._tls.stop()
        await self._plain.stop()
        await self._forward.aclose()
        self.status.listening = False


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy one direction until it ends or idles for `IDLE_SECONDS`."""
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(reader.read(_PUMP_CHUNK), IDLE_SECONDS)
            except TimeoutError:
                break
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        with contextlib.suppress(Exception):
            if writer.can_write_eof():
                writer.write_eof()


# --------------------------------------------------------------------------- #
# Wiring: what the rest of the agent asks of the machine port
# --------------------------------------------------------------------------- #

_DIALLED_KINDS = frozenset({"inference-driver", "tool-driver"})


def _port_of(url: str) -> int | None:
    from urllib.parse import urlsplit

    try:
        return urlsplit(url).port
    except ValueError:
        return None


def active_control_entry(state: Any) -> Any | None:
    """The active control root this agent spawns, never its standby. A
    standby copy once promoted is the root (row 4): `/control` and the
    plain branch reach it, with no restart."""
    from . import standby

    for entry in state.list_topology_entries():
        if entry.spawn is not None and standby.is_root(entry, state):
            return entry
    return None


def move_control_behind(state: Any, machine_port: int) -> int | None:
    """On the root, control leaves the machine port for the first free
    loopback port above it (LT-C1). Returns the new port, or None when
    nothing moved. Persisted, so it moves once."""
    from . import ports
    from .default_topology import RESERVED_PORTS

    entry = active_control_entry(state)
    if entry is None or _port_of(str(entry.url)) != machine_port:
        return None
    if not ports.is_free(machine_port, host="0.0.0.0"):
        # Something other than control holds it (control is not started yet).
        # Moving control would leave it behind a door that cannot open, so it
        # keeps the port, and the machine port does not start (`start`).
        log.warning(
            "port %d is held by %s; control keeps it and this machine opens no machine "
            "port, so other machines reach it in plain HTTP",
            machine_port,
            ports.describe_holder(machine_port) or "another process",
        )
        return None
    taken = {
        port for e in state.list_topology_entries() if (port := _port_of(str(e.url))) is not None
    }
    # Every declared port is passed: this runs before anything is started, so
    # a port another entry declares (the store's 8084, row 53) is still free
    # to the operating system.
    port = ports.first_free(machine_port + 1, reserved=RESERVED_PORTS | taken | {machine_port})
    state.update_topology_entry(
        entry.name, entry.model_copy(update={"url": f"http://127.0.0.1:{port}"})
    )
    log.warning(
        "control moved from port %d to %d on this machine only: the machine port is on %d "
        "now, and other machines reach control through it",
        machine_port,
        port,
        machine_port,
    )
    return port


def bind_host_for(app: Any) -> str:
    """Wide wherever what it fronts used to be: the agent binds wide, or
    control or a driver the root dials would bind wide on its own (the node
    advertises an address, or a container sets it). Behind a container's
    HTTPS entry point control stays on loopback, and so does this port."""
    from .app import _child_bind_host

    settings = app.state.settings
    bound = getattr(app.state, "bind_host", None) or settings.bind_host
    if bound in ("0.0.0.0", "::"):
        return "0.0.0.0"
    for kind in ("control", "inference-driver"):
        host = _child_bind_host(settings, app.state.agent_state, app.state.node_identity, kind)
        if host in ("0.0.0.0", "::"):
            return "0.0.0.0"
    return "127.0.0.1"


def door_url(*, configured: Any, advertise: str | None, bind_port: int, port: int) -> str | None:
    """Where other machines reach a machine port on `port`, or None when the
    node advertises no address they could use.

    The `machineUrl` config field (`configured`) wins. Otherwise the advertise
    host, on the machine port shifted by the same offset as the advertised
    agent port: a container that publishes 8079 as 8279 publishes 8083 as
    8283, the same guess the console's join command makes (`installPorts`).
    One derivation for the agent and for `join`, which runs with no agent.
    """
    from urllib.parse import urlsplit

    from . import node_identity

    if isinstance(configured, str) and configured.strip():
        return configured.strip().rstrip("/")
    host = node_identity.advertise_host(advertise)
    if not host or node_identity.is_loopback_host(host):
        return None
    if ":" in host:
        # The machine port listens on IPv4 (`bind_host_for`): an IPv6 address
        # would name a door nothing answers at. Set `machineUrl` to override.
        return None
    advertised_port = urlsplit(advertise or "").port
    offset = (advertised_port - int(bind_port)) if advertised_port else 0
    return node_identity.format_url(host, port + offset, scheme="https")


def _door_url(app: Any, door: MachinePort, advertise: str | None = None) -> str | None:
    from . import node_identity

    return door_url(
        configured=app.state.agent_state.get_config("machineUrl"),
        advertise=advertise
        or node_identity.effective_advertise_url(
            app.state.agent_state.get_config("advertiseUrl"),
            app.state.node_identity.record.advertise_url,
        ),
        bind_port=int(app.state.settings.bind_port),
        port=door.status.port,
    )


def machine_url(app: Any) -> str | None:
    """Where other machines reach this one's machine port, or None while it
    is not listening or the node advertises no address (`door_url`)."""
    door: MachinePort | None = getattr(app.state, "machine_port", None)
    if door is None or not door.status.listening:
        return None
    return _door_url(app, door)


def intended_url(app: Any, advertise: str | None = None) -> str | None:
    """Where other machines will reach this one's machine port: whenever this
    node has one, listening or still binding (agent#38).

    **The intent, not the listener.** A machine port that is still binding
    (another process held 8083 a moment longer, at a restart) is no reason
    to tell the root this machine takes plain HTTP only: the root would
    record it as an older version, and in `automatic` every machine would
    then send it tokens in clear until the port opened and it announced
    again. So enrolment and the announcement carry the door this node
    serves TLS on, and the others wait for it to listen. Only a node that
    opens no machine port at all (the setting is off, control kept the port,
    the key does not load) announces none. `advertise` is the address an
    enrolment is about to register, which the record does not hold yet."""
    door: MachinePort | None = getattr(app.state, "machine_port", None)
    if door is None:
        return None
    return _door_url(app, door, advertise)


def announced(app: Any, advertise: str | None = None) -> dict[str, str] | None:
    """`machineUrl` and `tlsPin` for an announcement or an enrolment, or None.
    While the machine key changes, `tlsNextPin` too (row 3). From the door
    this node intends (`intended_url`), listening or not."""
    url = intended_url(app, advertise)
    door: MachinePort | None = getattr(app.state, "machine_port", None)
    if url is None or door is None:
        return None
    from . import machine_key_change

    return machine_key_change.machine_fields(app.state.node_identity.record, url, door.status.pin)


def component_machine_url(app: Any, entry: Any) -> str | None:
    """`Component.machineUrl`: a driver this agent spawns, through the machine port."""
    if entry.spawn is None or entry.kind.value not in _DIALLED_KINDS:
        return None
    base = machine_url(app)
    return f"{base}/drivers/{entry.name}" if base else None


async def start(app: Any) -> MachinePort | None:
    """Open this node's machine port, when the settings ask for one."""
    from . import node_identity

    settings = app.state.settings
    if not settings.machine_port_enabled:
        return None
    control_entry = active_control_entry(app.state.agent_state)
    if control_entry is not None and _port_of(str(control_entry.url)) == int(settings.machine_port):
        return None  # control kept the port (`move_control_behind`)
    record = app.state.node_identity.record
    if not (record.tls_private_key and record.tls_certificate):
        log.error("this node has no machine key, so it opens no machine port")
        return None
    state = app.state.agent_state
    directory = settings.config_file.resolve().parent
    try:
        context = server_context(record.tls_certificate, record.tls_private_key, directory)
    except (OSError, ssl.SSLError, ValueError) as exc:
        log.error("the machine key could not be loaded, so no machine port: %s", exc)
        return None

    def control() -> str | None:
        entry = active_control_entry(state)
        return str(entry.url) if entry is not None else None

    def driver(name: str) -> str | None:
        entry = state.get_topology_entry(name)
        if entry is None or entry.spawn is None or entry.kind.value not in _DIALLED_KINDS:
            return None
        return str(entry.url)

    def required() -> bool:
        trust = getattr(app.state, "machine_trust", None)
        return bool(trust is not None and trust.required)

    door = MachinePort(
        routes=Routes(control=control, driver=driver, agent_app=app, required=required),
        context=context,
        bind_host=bind_host_for(app),
        port=int(settings.machine_port),
        pin=node_identity.machine_pin(record.tls_certificate),
        directory=directory,
    )
    app.state.machine_port = door
    await door.start()
    return door
