"""Enrolled agents relay key management and migrate their legacy metadata."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import HTTPException, Request

from . import node_identity, security
from ._http import internal_client
from .client_keys import as_datetime

IMPORT_DOMAIN = b"eugene-plexus/client-keys/import/v1\n"


class ClientKeyRegistry:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.client = internal_client(timeout=3.0)
        self.migration = "pending"
        self.detail = "Existing keys are waiting to be registered with the control root."
        self._digest: str | None = None
        self._next_attempt = 0.0
        self._failures = 0
        self._migration_task: asyncio.Task[None] | None = None

    @property
    def enrolled(self) -> bool:
        identity = getattr(self.app.state, "node_identity", None)
        return bool(identity and identity.record.enrolled)

    async def forward(
        self, method: str, path: str, *, authorization: str | None = None, body: Any = None
    ) -> Any:
        identity = self.app.state.node_identity.record
        if authorization is None:
            authorization = "Bearer " + security.issue_service_token(
                signing_key=self.app.state.auth_state.signing_key, kind="agent"
            )
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

    async def migrate(self) -> None:
        if not self.enrolled:
            return
        if self._migration_task is not None and not self._migration_task.done():
            await asyncio.shield(self._migration_task)
        elif time.perf_counter() >= self._next_attempt:
            self._migration_task = asyncio.create_task(self._import())
            await asyncio.shield(self._migration_task)

    async def _import(self) -> None:
        try:
            identity = self.app.state.node_identity.record
            records = []
            for record in self.app.state.client_keys.records():
                raw = record.to_json()
                for field in ("createdAt", "expiresAt", "revokedAt"):
                    if field in raw:
                        raw[field] = as_datetime(raw[field]).isoformat()
                records.append(raw)
            payload = {"node": identity.name, "keys": records}
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
            digest = hashlib.sha256(canonical + identity.control_url.encode()).hexdigest()
            if digest == self._digest:
                return
            signature = node_identity.sign_address(
                signing_private_key=identity.signing_private_key, message=IMPORT_DOMAIN + canonical
            )
            await self.forward(
                "POST",
                f"/v1/nodes/{quote(identity.name, safe='')}/client-keys/import",
                body={"keys": records, "signature": signature},
            )
            self._digest = digest
            self.migration = "complete"
            self.detail = "Existing keys are registered install-wide. Their tokens are unchanged."
            self._failures = 0
        except (HTTPException, OSError, ValueError, TypeError):
            self._failures += 1
            self.migration = "error"
            self.detail = (
                "Existing keys could not be registered. Restore the control connection "
                "or the local client_keys.json backup; unregistered keys are refused."
            )
        finally:
            self._next_attempt = time.perf_counter() + min(15, 2 ** min(self._failures, 4))

    async def run(self) -> None:
        while True:
            await self.migrate()
            await asyncio.sleep(15)

    async def close(self) -> None:
        if self._migration_task is not None:
            self._migration_task.cancel()
            await asyncio.gather(self._migration_task, return_exceptions=True)
        await self.client.aclose()


def registry(request: Request) -> ClientKeyRegistry:
    value = getattr(request.app.state, "client_key_registry", None)
    if value is None:
        value = ClientKeyRegistry(request.app)
        request.app.state.client_key_registry = value
    return value
