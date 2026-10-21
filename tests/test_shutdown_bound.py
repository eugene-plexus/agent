"""A stop never waits for ever on an open request (agent#20).

uvicorn runs the lifespan shutdown, which stops every child, only after
each open request ends. A log follow never ends by itself, and a stopped
service left the agent and all its children running. `build_server` now
bounds the wait, and `open_requests` names what the stop ended.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from eugene_plexus_agent import __main__ as entry
from eugene_plexus_agent import open_requests
from eugene_plexus_agent.__main__ import GRACEFUL_SHUTDOWN_SECONDS, build_server
from eugene_plexus_agent.settings import Settings


def test_the_entrypoint_bounds_the_wait_for_open_requests(settings: Settings) -> None:
    config = build_server(settings, unattended=True).config
    assert config.timeout_graceful_shutdown == GRACEFUL_SHUTDOWN_SECONDS


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_an_endless_stream_does_not_keep_the_children_running(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The lifespan shutdown stands in for `stop_all`: it must run while a
    follow is still open, within the bound, and the follow is named."""
    stopped: list[float] = []
    started = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        started.set()
        yield
        stopped.append(time.perf_counter())

    app = FastAPI(lifespan=lifespan)
    open_requests.install(app)

    @app.get("/v1/logs/stream")
    async def follow() -> StreamingResponse:
        async def frames() -> AsyncIterator[bytes]:
            while True:
                yield b": keepalive\n\n"
                await asyncio.sleep(0.2)

        return StreamingResponse(frames(), media_type="text/event-stream")

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
        )
    )
    app.state.uvicorn_server = server

    async def run() -> float:
        serving = asyncio.create_task(server.serve())
        await asyncio.wait_for(started.wait(), 10)
        # The lifespan starts before the socket listens.
        for _ in range(100):
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                break
            except OSError:
                await asyncio.sleep(0.05)
        else:
            raise AssertionError("the server never listened")
        writer.write(
            b"GET /v1/logs/stream HTTP/1.1\r\nHost: 127.0.0.1\r\nUser-Agent: follower\r\n\r\n"
        )
        await writer.drain()
        assert b"200" in await reader.readline()
        asked = time.perf_counter()
        server.should_exit = True
        await asyncio.wait_for(serving, GRACEFUL_SHUTDOWN_SECONDS + 10)
        writer.close()
        return asked

    with caplog.at_level(logging.WARNING, logger="eugene_plexus_agent.open_requests"):
        asked = asyncio.run(run())
    assert stopped, "the lifespan shutdown never ran"
    assert stopped[0] - asked < GRACEFUL_SHUTDOWN_SECONDS + 3
    named = [r.getMessage() for r in caplog.records if r.name.endswith("open_requests")]
    assert any("GET /v1/logs/stream" in m and "follower" in m for m in named), named


class _Timed(uvicorn.Server):
    """A server that records when something first asked it to exit."""

    exit_asked_at: float | None = None

    @property
    def should_exit(self) -> bool:
        return bool(self.__dict__.get("_should_exit", False))

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        if value and self.exit_asked_at is None:
            self.exit_asked_at = time.perf_counter()
        self.__dict__["_should_exit"] = value


def test_serve_stops_when_should_stop_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service, the command line and the acceptance harness share `serve`.
    The harness has no signal to send on Windows, so it gives `should_stop`
    (a file it creates): the server must start exiting within 0.5 s of that
    turning true, and the lifespan shutdown, which stops the children, runs."""
    stopped: list[float] = []

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        stopped.append(time.perf_counter())

    app = FastAPI(lifespan=lifespan)
    server = _Timed(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=_free_port(),
            log_level="warning",
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
        )
    )
    app.state.uvicorn_server = server
    built: list[bool] = []

    def fake_build_server(settings: Settings, *, unattended: bool = False) -> uvicorn.Server:
        built.append(unattended)
        return server

    monkeypatch.setattr(entry, "build_server", fake_build_server)
    stop = threading.Event()
    turned_true: list[float] = []

    def later() -> None:
        time.sleep(1.0)
        turned_true.append(time.perf_counter())
        stop.set()

    threading.Thread(target=later, daemon=True).start()
    entry.serve(Settings(), unattended=True, should_stop=stop.is_set)

    assert built == [True]
    assert server.exit_asked_at is not None, (
        "should_stop turned true and nothing stopped the server"
    )
    assert server.exit_asked_at - turned_true[0] < 0.5
    assert stopped, "the lifespan shutdown never ran"


def test_serve_without_should_stop_runs_the_server_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `should_stop`: the unchanged path, `server.run()`, then a restart if
    one was asked for."""
    ran: list[str] = []

    class Server:
        config = type("Config", (), {"app": type("App", (), {"state": type("S", (), {})()})()})()

        def run(self) -> None:
            ran.append("run")

    monkeypatch.setattr(entry, "build_server", lambda settings, *, unattended=False: Server())
    entry.serve(Settings(), unattended=True)
    assert ran == ["run"]


def test_a_restart_asked_for_replaces_the_process_after_the_server_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eugene_plexus_agent import entrypoint_setup

    order: list[str] = []

    class State:
        restart_requested = True

    class Server:
        config = type("Config", (), {"app": type("App", (), {"state": State()})()})()

        def run(self) -> None:
            order.append("stopped")

    monkeypatch.setattr(entry, "build_server", lambda settings, *, unattended=False: Server())
    monkeypatch.setattr(entrypoint_setup, "reexec", lambda: order.append("reexec"))
    entry.serve(Settings(), unattended=True, should_stop=lambda: False)
    assert order == ["stopped", "reexec"]
