"""The browser proxy holds a request body in memory, so it caps it.

`/api/proxy/...` needs no sign-in -- it is the path the login request
travels -- and it read the whole request body with `await
request.body()` before forwarding, with no limit. Anyone who could reach
the port could make this process allocate whatever they sent.

The cap is 32 MiB: above the gateway's own 16 MiB inference cap, so a
chat request with images attached still reaches the gateway and is
judged there, in the gateway's words, rather than refused here in ours.
Tests shrink it to 1 KiB so they do not allocate 32 MiB each.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent.routes import proxy as proxy_routes

from .test_ui_proxy import declare, record_transport

CAP = 1024


@pytest.fixture
def sent(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    # `raising=False` so that, against a proxy with no cap at all, these
    # fail on the status they get rather than on a missing attribute.
    monkeypatch.setattr(proxy_routes, "MAX_PROXY_BODY_BYTES", CAP, raising=False)
    recorder: list[httpx.Request] = []
    app.state.ui_proxy_client = httpx.AsyncClient(transport=record_transport(recorder))
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")
    return recorder


def _chunks(total: int, size: int = 256) -> Iterator[bytes]:
    while total > 0:
        yield b"x" * min(size, total)
        total -= size


def _is_too_large(response: httpx.Response) -> bool:
    if response.status_code != 413:
        return False
    detail = response.json()["detail"]
    return "MiB" in detail["detail"]


async def _drive(
    app: FastAPI, headers: list[tuple[bytes, bytes]], pieces: list[bytes]
) -> tuple[list[dict[str, Any]], int]:
    """Call the app at the ASGI boundary, one body message per piece.

    Returns what it sent and how many body messages it asked for -- the
    one thing `TestClient` cannot show, since it hands the app the whole
    body in a single message.
    """
    consumed = 0

    async def receive() -> dict[str, Any]:
        nonlocal consumed
        if consumed < len(pieces):
            consumed += 1
            return {
                "type": "http.request",
                "body": pieces[consumed - 1],
                "more_body": consumed < len(pieces),
            }
        return {"type": "http.disconnect"}

    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    path = "/api/proxy/gateway/v1/chat/completions"
    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"127.0.0.1:8079"), *headers],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8079),
        },
        receive,
        send,
    )
    return messages, consumed


def test_a_declared_length_over_the_cap_is_refused(
    client: TestClient, sent: list[httpx.Request]
) -> None:
    response = client.post("/api/proxy/gateway/v1/chat/completions", content=b"x" * (CAP + 1))
    assert _is_too_large(response), f"{response.status_code} {response.text[:200]}"
    assert sent == [], "an oversized body was forwarded"


async def test_a_declared_length_over_the_cap_is_refused_before_a_byte_is_read(
    app: FastAPI, client: TestClient, sent: list[httpx.Request]
) -> None:
    """The early check is what spares reading anything at all; without it
    the count below would still refuse, one cap's worth of reading late."""
    messages, consumed = await _drive(
        app, [(b"content-length", str(CAP * 20).encode())], [b"x" * 256] * 20
    )
    assert messages[0]["status"] == 413
    assert consumed == 0, f"read {consumed} body messages before refusing"
    assert sent == []


def test_a_chunked_body_over_the_cap_is_refused_while_it_is_counted(
    client: TestClient, sent: list[httpx.Request]
) -> None:
    """No Content-Length to check: a generator body goes out chunked, so
    only counting what arrives can catch it."""
    response = client.post("/api/proxy/gateway/v1/chat/completions", content=_chunks(CAP + 1))
    assert "content-length" not in {k.lower() for k in response.request.headers}
    assert _is_too_large(response), f"{response.status_code} {response.text[:200]}"
    assert sent == []


@pytest.mark.parametrize("chunked", [False, True])
def test_a_body_at_the_cap_is_forwarded_whole(
    client: TestClient, sent: list[httpx.Request], chunked: bool
) -> None:
    body = b"x" * CAP
    content: Any = _chunks(CAP) if chunked else body
    response = client.post("/api/proxy/gateway/v1/chat/completions", content=content)
    assert response.status_code == 200
    assert sent[-1].content == body


async def test_reading_stops_once_the_cap_is_passed(
    app: FastAPI, client: TestClient, sent: list[httpx.Request]
) -> None:
    """The point of counting is not holding the rest. Driven at the ASGI
    boundary with many small messages, which is what a real chunked
    upload is; the proxy must stop asking for more once it is over."""
    pieces = [b"x" * 256] * 20
    messages, consumed = await _drive(app, [(b"transfer-encoding", b"chunked")], pieces)
    assert messages[0]["status"] == 413
    assert consumed == CAP // 256 + 1, f"read {consumed} of {len(pieces)} messages"
    assert sent == []


def test_the_cap_sits_above_the_gateways_own() -> None:
    """32 MiB, so an image-heavy chat request reaches the gateway's 16 MiB
    check and is refused there, in the gateway's words, if at all."""
    assert proxy_routes.MAX_PROXY_BODY_BYTES == 32 * 1024 * 1024
