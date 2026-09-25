"""Enrolled agents relay key management to the control root.

Keys made on a machine before it joined an install are signed by that
machine alone and are not carried over (per-node token keys,
2026-09-25): their tokens could not verify anywhere in the install, so a
record of them there would only look like a key that works.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import HTTPException, Request

from ._http import internal_client


class ClientKeyRegistry:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.client = internal_client(timeout=3.0)

    @property
    def enrolled(self) -> bool:
        identity = getattr(self.app.state, "node_identity", None)
        return bool(identity and identity.record.enrolled)

    async def forward(
        self, method: str, path: str, *, authorization: str | None = None, body: Any = None
    ) -> Any:
        identity = self.app.state.node_identity.record
        if authorization is None:
            authorization = "Bearer " + self.app.state.auth_state.trust.agent_token("control")
        try:
            async with asyncio.timeout(3.0):
                response = await self.client.request(
                    method,
                    identity.control_url.rstrip("/") + path,
                    headers={"Authorization": authorization},
                    json=body,
                )
            response.raise_for_status()
            return response.json() if response.content else None
        except httpx.HTTPStatusError as exc:
            # Preserve operator mistakes and refusals; outages have one actionable shape.
            if exc.response.status_code in (400, 401, 403, 404, 409, 422, 429):
                try:
                    detail = exc.response.json()
                except ValueError:
                    detail = "Control root refused this operation."
                raise HTTPException(
                    exc.response.status_code,
                    detail=detail,
                    headers={"Retry-After": exc.response.headers["Retry-After"]}
                    if "Retry-After" in exc.response.headers
                    else None,
                ) from exc
            raise self.unavailable() from exc
        except (httpx.HTTPError, ValueError, TimeoutError) as exc:
            raise self.unavailable() from exc

    @staticmethod
    def unavailable() -> HTTPException:
        return HTTPException(
            503,
            detail={
                "title": "Client-key authority unavailable",
                "detail": "The active control root did not provide a key registry. Restore its "
                "connection or sign in to unlock it. Existing policy expires after 60 seconds.",
            },
        )

    @staticmethod
    def local_unavailable() -> HTTPException:
        return HTTPException(
            503,
            detail={
                "title": "Client-key registry unavailable",
                "detail": "Check that this agent can read and write client_keys.json. "
                "If it is damaged, restore a backup and restart the agent; "
                "do not delete it, because it holds existing key and revocation records.",
            },
        )

    async def close(self) -> None:
        await self.client.aclose()


def registry(request: Request) -> ClientKeyRegistry:
    value = getattr(request.app.state, "client_key_registry", None)
    if value is None:
        value = ClientKeyRegistry(request.app)
        request.app.state.client_key_registry = value
    return value
