"""Every agent response says `nosniff` and `no-referrer`.

Only the UI's static files carried a security header (the two frame
headers, review §6.3 #32). A JSON body a browser was allowed to sniff
into HTML, or a page that sent this install's address -- a tailnet name,
a LAN address -- in the `Referer` of every link it followed out, was
the rest of the surface. These go on everything this process answers:
the UI, the API, the proxy's relayed responses and its own refusals.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from .test_forwarded_peer import _ui_app
from .test_ui_proxy import byte_stream, declare


def _has_both(response: httpx.Response) -> None:
    assert response.headers.get("x-content-type-options") == "nosniff", dict(response.headers)
    assert response.headers.get("referrer-policy") == "no-referrer", dict(response.headers)


def test_the_ui_has_both_and_keeps_its_frame_headers(tmp_path: Path) -> None:
    with TestClient(_ui_app(tmp_path, with_index=True)) as client:
        response = client.get("/")
    assert response.status_code == 200
    _has_both(response)
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_the_no_ui_page_has_both(tmp_path: Path) -> None:
    with TestClient(_ui_app(tmp_path, with_index=False)) as client:
        response = client.get("/")
    assert response.status_code == 503
    _has_both(response)


@pytest.mark.parametrize("path", ["/healthz", "/v1/auth/status", "/v1/config"])
def test_api_answers_and_refusals_have_both(client: TestClient, path: str) -> None:
    _has_both(client.get(path))


def test_a_refused_host_name_has_both(client: TestClient) -> None:
    response = client.get("/healthz", headers={"host": "evil.example.com"})
    assert response.status_code == 403
    _has_both(response)


def test_a_proxied_response_has_ours_and_only_ours(app: FastAPI, client: TestClient) -> None:
    """A component's own values do not survive the hop: this origin has
    one policy, and two `Referrer-Policy` headers are a browser's to
    reconcile however it likes."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "referrer-policy": "unsafe-url",
                "x-content-type-options": "sniff-away",
            },
            stream=byte_stream(b'{"ok": true}'),
        )

    app.state.ui_proxy_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")
    response = client.get("/api/proxy/gateway/v1/models")
    assert response.status_code == 200
    _has_both(response)
    assert response.headers.get_list("referrer-policy") == ["no-referrer"]
    assert response.headers.get_list("x-content-type-options") == ["nosniff"]


async def test_the_layer_passes_a_stream_through_without_holding_it() -> None:
    """Pure ASGI, like `off_host`: the inner app will not send its second
    chunk until the first has reached the outside, so a layer that held
    the body would never finish."""
    from eugene_plexus_agent.response_headers import SecurityHeadersMiddleware

    first_out = asyncio.Event()

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"one", "more_body": True})
        await asyncio.wait_for(first_out.wait(), timeout=5)
        await send({"type": "http.response.body", "body": b"two"})

    seen: list[dict[str, Any]] = []

    async def outer_send(message: dict[str, Any]) -> None:
        seen.append(message)
        if message.get("body") == b"one":
            first_out.set()

    middleware = SecurityHeadersMiddleware(inner)
    await asyncio.wait_for(middleware({"type": "http"}, None, outer_send), timeout=5)
    assert [m.get("body") for m in seen[1:]] == [b"one", b"two"]
    assert (b"x-content-type-options", b"nosniff") in seen[0]["headers"]
