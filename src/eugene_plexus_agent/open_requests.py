"""Name each request a shutdown stopped waiting for (agent#20).

uvicorn waits for every open request before it runs the lifespan
shutdown, and that shutdown is what stops this agent's children. One
request that never ends -- a log follow, a proxied stream -- kept a
stopping agent and all its children running for ever. `build_server`
now bounds the wait (`GRACEFUL_SHUTDOWN_SECONDS`), and uvicorn cancels
what is still open when it runs out. This middleware says which requests
those were, so the cause is in the log and not a guess.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


class OpenRequestsMiddleware:
    """Logs an HTTP request that is cancelled while the server stops."""

    def __init__(self, app: Any, *, root: Any) -> None:
        self.app = app
        self._root = root

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        except asyncio.CancelledError:
            server = getattr(self._root.state, "uvicorn_server", None)
            if server is not None and server.should_exit:
                client = scope.get("client")
                agent = dict(scope.get("headers") or ()).get(b"user-agent", b"").decode("latin-1")
                log.warning(
                    "the shutdown stopped waiting for %s %s from %s (%s); it was ended so "
                    "that the components could be stopped",
                    scope.get("method"),
                    scope.get("path"),
                    f"{client[0]}:{client[1]}" if client else "an unknown peer",
                    agent or "no user agent",
                )
            raise


def install(app: Any) -> None:
    app.add_middleware(OpenRequestsMiddleware, root=app)
