"""A gateway for the tests of what an app is told about the install's models (MR6).

It answers `GET <prefix>/v1/models` the way the real gateway does, on a port
of its own (never 8080, which a developer's live install holds), and can
behave badly on request: refuse with the gateway's own problem document,
redirect, answer with something that is not a list, answer slowly, or answer
one small piece at a time. It records what it was asked, so a test can say how
many requests a start made and with which key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def listed(
    model_id: str,
    *surfaces: str,
    locality: str | None = "local",
    owned_by: str = "openai_compat_custom",
    ready: Any = 1,
) -> dict[str, Any]:
    """One entry of `GET /v1/models`, shaped as the real gateway's.

    Measured against the real gateway app on 2026-10-12: a model of a local
    runtime is `owned_by` the provider key of its driver (every runtime's
    companion driver reports `openai_compat_custom`), never `local`, and
    where it runs is its `x_eugene_plexus.locality`. `owned_by` reads `local`
    only when no driver names a provider or two different ones do.

    `ready` is `x_eugene_plexus.ready_backends`: how many backends run the
    model now. The gateway keeps listing a model whose runtime was stopped,
    with 0. `None` leaves the field out, as a gateway older than it does.
    """
    info: dict[str, Any] = {
        "drivers": [f"{model_id}-driver"],
        "backends": ["openai_compat_http"],
        "surfaces": list(surfaces),
    }
    if ready is not None:
        info["ready_backends"] = ready
    if locality is not None:
        info["locality"] = locality
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": owned_by,
        "x_eugene_plexus": info,
    }


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client that gave up on a slow answer is not an error of this server's.
        return None


class FakeGateway:
    """A gateway that lists `models` for the key it accepts.

    `mode` is what it does with a request that is allowed: `list` answers,
    `problem` refuses with `status` and `said`, `echo` fails with the
    Authorization header in the body, `redirect` sends the client to
    `location`, `garbage` answers 200 with something that is not JSON,
    `hangup` closes the connection without a word, `not-http` answers with
    something that is not HTTP at all, `slow` waits `delay`
    seconds and then answers, and `drip` answers `chunk` bytes at a time,
    `interval` seconds apart.
    """

    def __init__(
        self,
        models: list[dict[str, Any]] | None = None,
        *,
        key: str | None = None,
        prefix: str = "",
        port: int = 0,
    ) -> None:
        """`port` 0 is one the system picks; a fake that comes up late is given
        one that was free a moment ago."""
        self.models = list(models or [])
        #: The key it accepts; None accepts any.
        self.key = key
        self.prefix = prefix
        #: Every request, as `{"path", "authorization"}`.
        self.requests: list[dict[str, str]] = []
        self.mode = "list"
        self.status = 500
        self.said = "the gateway's own words"
        self.delay = 0.0
        self.interval = 0.05
        self.chunk = 16
        self.location = ""
        self._stop = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer._answer(self)

            def log_message(self, *_a: object) -> None:
                return None

        self.server = _Server(("127.0.0.1", port), Handler)
        # A short poll: `shutdown()` waits one, and a test per fake would wait half a second each.
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        ).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}{prefix}"

    def close(self) -> None:
        self._stop.set()
        self.server.shutdown()
        self.server.server_close()

    # ------------------------------------------------------------------ #

    def _send(
        self, handler: BaseHTTPRequestHandler, code: int, body: bytes, **headers: str
    ) -> None:
        try:
            handler.send_response(code)
            handler.send_header("content-type", "application/json")
            handler.send_header("content-length", str(len(body)))
            for name, value in headers.items():
                handler.send_header(name, value)
            handler.end_headers()
            handler.wfile.write(body)
        except OSError:
            return None

    def _problem(self, handler: BaseHTTPRequestHandler, code: int, title: str, text: str) -> None:
        document = {
            "detail": {
                "type": f"https://github.com/eugene-plexus/gateway#{title.replace(' ', '-').lower()}",
                "title": title,
                "status": code,
                "detail": text,
                "component": "gateway",
            }
        }
        self._send(handler, code, json.dumps(document).encode())

    def _answer(self, handler: BaseHTTPRequestHandler) -> None:
        auth = handler.headers.get("authorization", "")
        self.requests.append({"path": handler.path, "authorization": auth})
        if handler.path != f"{self.prefix}/v1/models":
            self._problem(handler, 404, "Not found", f"nothing is served at {handler.path}")
            return
        if self.key is not None and auth != f"Bearer {self.key}":
            self._problem(
                handler,
                401,
                "Invalid token",
                "Bearer token rejected: the signature does not verify",
            )
            return
        listing = json.dumps({"object": "list", "data": self.models}).encode()
        if self.mode == "problem":
            self._problem(handler, self.status, "Gateway fault", self.said)
        elif self.mode == "echo":
            self._problem(handler, 500, "Gateway fault", f"could not handle {auth!r}")
        elif self.mode == "redirect":
            self._send(handler, 302, b"", location=self.location)
        elif self.mode == "garbage":
            self._send(handler, 200, b"<html>this is not the gateway</html>")
        elif self.mode == "hangup":
            handler.close_connection = True
        elif self.mode == "not-http":
            try:
                handler.wfile.write(b"this is not http\r\n\r\n")
            except OSError:
                return
            handler.close_connection = True
        elif self.mode == "slow":
            self._stop.wait(self.delay)
            self._send(handler, 200, listing)
        elif self.mode == "drip":
            try:
                handler.send_response(200)
                handler.send_header("content-type", "application/json")
                handler.send_header("content-length", str(len(listing)))
                handler.end_headers()
                for start in range(0, len(listing), self.chunk):
                    handler.wfile.write(listing[start : start + self.chunk])
                    handler.wfile.flush()
                    if self._stop.wait(self.interval):
                        return
            except OSError:
                return
        else:
            self._send(handler, 200, listing)
