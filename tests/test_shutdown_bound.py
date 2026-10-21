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
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

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
