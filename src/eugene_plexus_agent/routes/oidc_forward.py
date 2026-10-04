"""`/oidc/*`: signing in with Eugene, through this machine (C2).

The OpenID Connect provider is the control root
(`specs/docs/design/sign-in-with-eugene.md`); browsers and apps reach a
machine's agent, so the issuer an app is configured with is this
machine's address and `/oidc`, and this forwards what arrives to the root.

**What this adds, and why the root may believe it.** The root needs three
things only this hop knows: the host the request arrived at (the issuer,
D11), its scheme, and the caller's address (the sign-in limiter's
bucket). They go in headers whose caller-supplied copies are dropped
first, beside this agent's own service token to the root; the root
believes the three only when that token verifies, so a caller reaching
the root's port directly cannot pick the issuer or a fresh limiter
bucket. The host has passed this agent's allowlist (the DNS-rebinding
defence) before any of this runs.

Public, like the pages it forwards: the root authenticates what needs it
(a client's secret at `/oidc/token`, a person's password on the sign-in
page). Nothing of the console's session crosses: the console's bearer
lives in the browser's `sessionStorage`, never in a cookie this could
carry.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from .._http import client_for

log = logging.getLogger(__name__)

router = APIRouter(tags=["oidc"])

FORWARDED_HOST_HEADER = "x-eugene-plexus-forwarded-host"
FORWARDED_PROTO_HEADER = "x-eugene-plexus-forwarded-proto"
FORWARDED_FOR_HEADER = "x-eugene-plexus-forwarded-for"
NODE_TOKEN_HEADER = "x-eugene-plexus-node-token"

#: What the root may set on its answer that the browser or app needs.
_PASSED_BACK = (
    "content-type",
    "location",
    "cache-control",
    "pragma",
    "www-authenticate",
    "content-security-policy",
    "x-frame-options",
    "referrer-policy",
)
#: What the caller sends that the root needs: the client's secret or the
#: bearer, and what the body is.
_PASSED_ON = ("authorization", "content-type", "accept")


def _root_url(request: Request) -> str | None:
    identity = getattr(request.app.state, "node_identity", None)
    if identity is None or not identity.record.enrolled or not identity.record.control_url:
        return None
    return str(identity.record.control_url).rstrip("/")


def _client(request: Request, root: str) -> httpx.AsyncClient:
    client = getattr(request.app.state, "oidc_forward_client", None)
    if client is None or client.is_closed:
        client = client_for(root, timeout=30.0, follow_redirects=False, trust_env=False)
        request.app.state.oidc_forward_client = client
    return client


@router.api_route("/oidc/{path:path}", methods=["GET", "POST"], include_in_schema=True)
async def forward(request: Request, path: str) -> Response:
    root = _root_url(request)
    if root is None:
        return JSONResponse(
            {
                "error": "temporarily_unavailable",
                "error_description": "This machine has not joined an install yet, so there is "
                "no Eugene to sign in with here. Finish first-run setup.",
            },
            status_code=503,
        )
    headers = {k: v for k, v in request.headers.items() if k.lower() in _PASSED_ON}
    headers[FORWARDED_HOST_HEADER] = request.headers.get("host") or request.url.netloc
    headers[FORWARDED_PROTO_HEADER] = request.url.scheme
    entry = getattr(request.app.state, "entrypoint_config", None)
    if entry:
        headers[FORWARDED_HOST_HEADER] = urlsplit(entry.console.origin).netloc
        headers[FORWARDED_PROTO_HEADER] = "https"
    headers[FORWARDED_FOR_HEADER] = request.client.host if request.client else "unknown"
    try:
        headers[NODE_TOKEN_HEADER] = request.app.state.auth_state.trust.agent_token("control")
    except Exception as exc:
        log.warning("cannot forward a sign-in: no token for the control root (%s)", exc)
        return JSONResponse(
            {"error": "temporarily_unavailable", "error_description": str(exc)}, status_code=503
        )
    target = f"{root}/oidc/{path}"
    try:
        answer = await _client(request, root).request(
            request.method,
            target,
            params=str(request.query_params),
            content=await request.body(),
            headers=headers,
        )
    except httpx.HTTPError as exc:
        log.warning("the control root did not answer a sign-in request: %s", exc)
        return JSONResponse(
            {
                "error": "temporarily_unavailable",
                "error_description": f"The control root at {root} did not answer. Try again "
                "once it is back.",
            },
            status_code=503,
        )
    back = {k: v for k, v in answer.headers.items() if k.lower() in _PASSED_BACK}
    return Response(content=answer.content, status_code=answer.status_code, headers=back)
