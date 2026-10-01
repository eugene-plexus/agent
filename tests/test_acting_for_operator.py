"""How this machine carries an operator's authority to the control root.

Found on the live install (2026-10-01): installing an app on a worker from
the console on another machine signed the operator out, every time. The
console reaches the worker with a five-minute token addressed to the worker
alone, and the worker sent that token on to the root to mint the app's key.
The root refused it, the 401 came back through two proxies, and the browser
read it as its session ending. The whole arc runs in
`specs/scripts/c3-workbench-acceptance.py`; pinned here is which headers
leave this machine, and what a refusal at the root turns into.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import tokens
from eugene_plexus_agent.client_key_registry import SUBJECT_TOKEN_HEADER, ClientKeyRegistry

from .conftest import FakeRoot, enroll_app


def _created(name: str) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "key": {
            "id": "k1",
            "name": name,
            "tail": "abcdef",
            "createdAt": now.isoformat(),
            "expiresAt": (now + timedelta(days=365)).isoformat(),
        },
        "token": "client-token",
    }


def _enrolled(app: FastAPI) -> FakeRoot:
    root = FakeRoot()
    # The console the operator is at: a member, so a token naming it as the
    # actor still verifies here.
    root.register("laptop", tokens.generate_private_key().public_key())
    enroll_app(app, root, "gpu-box")
    return root


def _root(app: FastAPI, answer: Any) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answer(request)

    registry = ClientKeyRegistry(app)
    registry.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app.state.client_key_registry = registry
    return seen


def _like_the_root(request: httpx.Request) -> httpx.Response:
    """Takes a session addressed to `control`, or this machine's own token
    with a subject beside it; refuses the rest, as the real root does."""
    authorization = request.headers.get("authorization", "")
    token = authorization.removeprefix("Bearer ")
    claims = _unverified(token)
    if claims.get("sub") == "agent" and SUBJECT_TOKEN_HEADER.lower() in request.headers:
        return httpx.Response(201, json=_created("app:workbench@gpu-box"))
    if "control" in (claims.get("aud") or []):
        return httpx.Response(201, json=_created("app:workbench@gpu-box"))
    return httpx.Response(
        401,
        json={
            "detail": {
                "title": "Invalid token",
                "detail": f"the token is addressed to {claims.get('aud')}, not to 'control'",
            }
        },
    )


def _unverified(token: str) -> dict[str, Any]:
    import jwt

    try:
        return dict(jwt.decode(token, options={"verify_signature": False}))
    except jwt.InvalidTokenError:
        return {}


def test_a_token_the_console_handed_this_machine_reaches_the_root_beside_its_own(
    app: FastAPI, client: TestClient
) -> None:
    """The reproduction: before the fix this was a 401 at the root, passed
    back to the browser, which signed the operator out."""
    root = _enrolled(app)
    seen = _root(app, _like_the_root)
    exchanged = root.session("node:gpu-box", ttl=300, act={"sub": "node:laptop"}, sid="s1")

    answer = client.post(
        "/v1/auth/client-keys",
        json={"name": "app:workbench@gpu-box"},
        headers={"Authorization": f"Bearer {exchanged}"},
    )
    assert answer.status_code == 201, answer.text

    sent = seen[0]
    assert sent.headers[SUBJECT_TOKEN_HEADER] == exchanged
    actor = sent.headers["authorization"].removeprefix("Bearer ")
    assert actor != exchanged
    claims = tokens.verify(
        actor,
        bundle=app.state.auth_state.trust.bundle,
        recipient="control",
        classes=(tokens.TYP_SERVICE,),
    )
    assert claims.sub == "agent" and claims.iss == "node:gpu-box"


def test_a_session_addressed_to_the_root_goes_on_unchanged_and_alone(
    app: FastAPI, client: TestClient
) -> None:
    """A session made by signing in on this machine names the root too:
    nothing to act for, so nothing beside it."""
    root = _enrolled(app)
    seen = _root(app, _like_the_root)
    session = root.session("node:gpu-box", "control")

    answer = client.post(
        "/v1/auth/client-keys",
        json={"name": "laptop"},
        headers={"Authorization": f"Bearer {session}"},
    )
    assert answer.status_code == 201, answer.text
    assert seen[0].headers["authorization"] == f"Bearer {session}"
    assert SUBJECT_TOKEN_HEADER.lower() not in seen[0].headers


def test_a_refusal_at_the_root_is_not_passed_on_as_the_callers_session_ending(
    app: FastAPI, client: TestClient
) -> None:
    root = _enrolled(app)

    def refuses(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "detail": {
                    "title": "Not an operator acting on this machine",
                    "detail": "the root's own reason",
                }
            },
        )

    _root(app, refuses)
    exchanged = root.session("node:gpu-box", ttl=300, act={"sub": "node:laptop"}, sid="s1")
    answer = client.post(
        "/v1/auth/client-keys",
        json={"name": "app:workbench@gpu-box"},
        headers={"Authorization": f"Bearer {exchanged}"},
    )
    assert answer.status_code == 502, answer.text
    assert "the root's own reason" in answer.text
    assert "control.invalid:8083" in answer.text


def test_a_refusal_the_root_makes_about_a_client_key_stays_what_it_said(
    app: FastAPI, client: TestClient
) -> None:
    """This agent's own calls: admission answers a revoked key with a 401,
    and the gateway reads exactly that as "revoked"."""
    import asyncio

    from fastapi import HTTPException

    _enrolled(app)

    def revoked(request: httpx.Request) -> httpx.Response:
        assert SUBJECT_TOKEN_HEADER.lower() not in request.headers
        return httpx.Response(
            401, json={"detail": "This client key is unregistered, expired or revoked."}
        )

    _root(app, revoked)
    registry = app.state.client_key_registry
    try:
        asyncio.run(registry.forward("POST", "/v1/auth/client-keys/admission", body={}))
    except HTTPException as exc:
        assert exc.status_code == 401
        assert "revoked" in str(exc.detail)
    else:  # pragma: no cover - the assertion this test exists for
        raise AssertionError("a refused admission did not raise")
