"""Enrolled agents relay key management to the control root.

Keys made on a machine before it joined an install are signed by that
machine alone and are not carried over (per-node token keys,
2026-09-25): their tokens could not verify anywhere in the install, so a
record of them there would only look like a key that works.

## Acting for an operator who reached this machine from another console

Found on the live install (2026-10-01): installing an app on a worker from
the console on another machine signed the operator out, every time. The
console reaches this machine with a five-minute token addressed to it
alone (per-node token keys, D7), and this module sent that token on to the
root, which refused a token not addressed to it. The 401 came back through
two proxies to a browser that reads any 401 as its session ending.

So a caller's token goes to the root unchanged only when it is addressed
there, which a session made by signing in on this machine is. Otherwise
this machine sends its own `agent` token, with the caller's token beside
it as the subject: the root takes that pair only for what an app install
needs, and only for things named for this machine. And a 401 from the root
is reported as the root refusing what this machine sent (502), never
passed on as one, because the caller's own session is not what failed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import HTTPException, Request

from . import tokens
from ._http import internal_client

SUBJECT_TOKEN_HEADER = "X-Eugene-Plexus-Subject-Token"
"""Where this machine puts the operator's token when it acts for them at the
root; its own service token, in `Authorization`, is the actor (RFC 8693)."""


class ClientKeyRegistry:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.client = internal_client(timeout=3.0)

    @property
    def enrolled(self) -> bool:
        identity = getattr(self.app.state, "node_identity", None)
        return bool(identity and identity.record.enrolled)

    def credentials_for_root(self, authorization: str | None) -> dict[str, str]:
        """The headers that carry `authorization`'s authority to the root.

        None is this agent speaking for itself. A bearer the root would take
        as it is -- a session addressed to `control` -- goes unchanged.
        Anything else the caller presented here is addressed to this machine
        alone, so it rides as the subject beside this agent's own token.
        """
        trust = self.app.state.auth_state.trust
        own = "Bearer " + trust.agent_token(tokens.RECIPIENT_CONTROL)
        if authorization is None:
            return {"Authorization": own}
        scheme, _, token = authorization.partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            return {"Authorization": authorization}
        bundle = trust.bundle
        if bundle is not None:
            try:
                tokens.verify(
                    token,
                    bundle=bundle,
                    recipient=tokens.RECIPIENT_CONTROL,
                    classes=(tokens.TYP_SESSION,),
                )
                return {"Authorization": authorization}
            except tokens.TokenError:
                pass
        return {"Authorization": own, SUBJECT_TOKEN_HEADER: token}

    async def forward(
        self, method: str, path: str, *, authorization: str | None = None, body: Any = None
    ) -> Any:
        identity = self.app.state.node_identity.record
        try:
            async with asyncio.timeout(3.0):
                response = await self.client.request(
                    method,
                    identity.control_url.rstrip("/") + path,
                    headers=self.credentials_for_root(authorization),
                    json=body,
                )
            response.raise_for_status()
            return response.json() if response.content else None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401 and authorization is not None:
                # The caller's credential was good here, or this route would
                # not have run; what the root refused is what this machine
                # sent it. Passed on as a 401 it signs the browser out.
                # Only for a caller's authority: on this agent's own calls a
                # 401 is the root's answer about a client key (admission
                # says "revoked" that way), and the gateway reads it so.
                raise HTTPException(
                    502,
                    detail={
                        "title": "Refused at the control root",
                        "detail": "The control root at "
                        f"{identity.control_url} would not take what this machine sent "
                        f"for you: {_said(exc.response)} Your session here is unaffected.",
                    },
                ) from exc
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


def _said(response: httpx.Response) -> str:
    """The root's own words for a refusal, as one sentence."""
    try:
        body = response.json()
    except ValueError:
        return (response.text or f"HTTP {response.status_code}").strip()[:300]
    if isinstance(body, dict):
        inner = body.get("detail", body)
        if isinstance(inner, dict):
            return str(inner.get("detail") or inner.get("title") or inner)[:300]
        return str(inner)[:300]
    return str(body)[:300]


def registry(request: Request) -> ClientKeyRegistry:
    value = getattr(request.app.state, "client_key_registry", None)
    if value is None:
        value = ClientKeyRegistry(request.app)
        request.app.state.client_key_registry = value
    return value
