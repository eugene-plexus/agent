"""Two headers on every response this agent sends: `nosniff` and `no-referrer`.

Until 2026-09-22 the only security headers anywhere in this process
were the two frame headers on the UI's static files (`ui_assets`). Two
more are cheap and apply to everything, which is why they live here and
not there:

* **`X-Content-Type-Options: nosniff`.** The API answers JSON, the proxy
  relays whatever a component says, and a browser allowed to sniff can
  decide a body that merely *contains* markup is HTML and render it on
  this origin -- the origin that holds an operator session and an
  unauthenticated proxy to the whole install.
* **`Referrer-Policy: no-referrer`.** Every link the UI follows out --
  a model's page on the hub, the docs, an engine's release notes --
  would otherwise carry this install's address in `Referer`, and on a
  tailnet or a LAN that address is itself something worth not telling
  strangers. Nothing here reads `Referer`, so there is nothing to keep.

**They replace, never add to, whatever an inner layer set.** A proxied
component's own `Referrer-Policy` must not survive the hop: this origin
has one policy, and two conflicting headers are a browser's to reconcile
however it likes.

**No full Content-Security-Policy**, for the reason `ui_assets` gives:
over a Next static export it means enumerating its inline bootstrap and
its chunk origins, which is a real slice with a real chance of shipping
a blank page. The frame headers stay where they are, on the UI's
responses.

**Pure ASGI**, like `off_host` and `host_allowlist`: it rewrites the
`http.response.start` message and forwards every body message the moment
it arrives, so the proxy's token streams pass through unheld.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SECURITY_HEADERS", "SecurityHeadersMiddleware", "install"]

SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
)

_NAMES = frozenset(name for name, _ in SECURITY_HEADERS)


class SecurityHeadersMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Any) -> None:
            if message.get("type") == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers") or ()
                    if bytes(name).lower() not in _NAMES
                ]
                headers.extend(SECURITY_HEADERS)
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


def install(app: Any) -> None:
    """Wrap `app`. Call it last, so it is the outermost layer and covers
    the other layers' own answers -- a refused host name included."""
    app.add_middleware(SecurityHeadersMiddleware)
