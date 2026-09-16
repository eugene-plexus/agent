"""The one piece of evidence about reach that comes from outside.

Static inspection of a firewall says what *should* happen. A connection
that arrived says what did — and on a standalone install, where there is
no control root elsewhere probing this node, it is the only such
evidence there is. The card tells a person to open the address on their
phone; this is what notices that they did.

**Any off-host caller, not the control root specifically.** The root's
own view of this fact is `Node.lastSeenAt` on the trust root, which a
multi-machine console reads there. Duplicating it here would be a second
source of truth for it, and would answer nothing at all for the person
this slice is for.

**Not persisted.** It describes this process. A restart is exactly the
moment somebody wants to know whether reach still works, not whether it
once did, and a timestamp that survived the restart would answer the
wrong one of those two questions.

The cost is one address parse per request, off the ASGI scope's
client tuple, before routing. Pure ASGI rather than
`BaseHTTPMiddleware` for the reason `cors.py` in the gateway gives:
`BaseHTTPMiddleware` buffers, and this agent proxies token streams.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

__all__ = ["OffHostWitness", "install"]


@dataclass
class OffHostWitness:
    """The last connection that came from somewhere other than here."""

    address: str | None = None
    at: datetime | None = None

    def saw(self, host: str) -> None:
        self.address = host
        self.at = datetime.now(UTC)


def _is_off_host(host: str | None) -> bool:
    """Is this client address somewhere other than this machine?

    Loopback is this machine. `::1` and IPv4-mapped loopback are too, and
    `ipaddress` gets both right where a string comparison to
    `"127.0.0.1"` would not. An address we cannot parse — a Unix socket
    peer, a test transport's placeholder — is **not** counted: the whole
    value of this field is that it is evidence, and a caller we cannot
    identify is not evidence of anything.
    """
    if not host:
        return False
    try:
        parsed = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    if parsed.is_loopback:
        return False
    mapped = getattr(parsed, "ipv4_mapped", None)
    return not (mapped is not None and mapped.is_loopback)


class OffHostMiddleware:
    """Pure ASGI, and that is not a style choice.

    `BaseHTTPMiddleware` — what `@app.middleware("http")` installs —
    wraps the response body in a memory stream, and this agent proxies
    token streams: step 1's proxy has a unit test that *deadlocks* when
    the implementation buffers, precisely because a frame count cannot
    tell a buffering proxy from a streaming one. A middleware that
    touches nothing but the scope has no business being in that path at
    all, so it is not.

    The first draft of this file said all of the above in its docstring
    and then used `@app.middleware("http")` anyway.
    """

    def __init__(self, app: Any, witness: OffHostWitness) -> None:
        self.app = app
        self.witness = witness

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            client = scope.get("client")
            if client and _is_off_host(str(client[0])):
                self.witness.saw(str(client[0]))
        await self.app(scope, receive, send)


def install(app: Any, witness: OffHostWitness) -> None:
    """Wrap `app` so every off-host request is noticed."""
    app.add_middleware(OffHostMiddleware, witness=witness)
