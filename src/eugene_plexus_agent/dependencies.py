"""FastAPI dependencies for bearer auth, against this node's trust bundle.

Every bearer is checked by `NodeTrust.verify`: the key found by `kid` in
the bundle, the algorithm from the key, `typ`, `iss`, an `aud` naming
this machine, the key's grants and the class's lifetime
(`specs/docs/design/per-node-token-keys.md`, D2 and D5). The route levels:

* `require_operator_session` — an operator session addressed to this
  machine: one signed in here, or one another console exchanged for this
  machine at the control root.
* `require_operator_or_service` — the reads: a session, a service token
  from this machine's own children, the control root's, or a `gateway`
  token (this machine's gateway, or a node the bundle grants `gateway`).
* `require_operator_or_gateway` — starting and stopping a runtime, and
  the client-key policy: a session or a `gateway` token.
* `require_operator_or_control` — declaring a runtime: a session or the
  control root's own token.
* `require_local_service` — a child of this agent speaking for itself,
  and nothing else.

**No `service:*` wildcard anywhere** (2026-09-25). Until then a driver's
token or the library's opened every read on every agent in the install,
because a service token named a kind and every machine shared one key.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import tokens
from ._generated.common_models import Problem
from .auth_state import AuthState
from .state import AgentState

_bearer_scheme = HTTPBearer(auto_error=False)

_SESSION = (tokens.TYP_SESSION,)
_SESSION_OR_SERVICE = (tokens.TYP_SESSION, tokens.TYP_SERVICE)


def _problem(status_code: int, title: str, detail: str) -> HTTPException:
    """Build a 7807-style HTTPException via the generated Problem model."""
    return HTTPException(
        status_code=status_code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/agent#{title.replace(' ', '-').lower()}",
            title=title,
            status=status_code,
            detail=detail,
            component="agent",
        ).model_dump(exclude_none=True),
    )


def revoked_session_problem() -> HTTPException:
    """The one refusal for a signed-out session, wherever it is presented.

    Shared by this module's dependencies and the browser proxy, so a
    token refused at the proxy reads exactly as it does on the agent's
    own routes -- the UI's 401 handler already knows this shape and
    sends the person to sign in.
    """
    return _problem(
        status.HTTP_401_UNAUTHORIZED,
        "Token revoked",
        "Session was explicitly logged out. Login again to obtain a new token.",
    )


def require_initialized(request: Request) -> AgentState:
    """Returns the agent state, ONLY if the operator has set a
    passphrase. Otherwise short-circuits with 503 directing the
    wizard to call POST /v1/auth/initialize first."""
    state: AgentState = request.app.state.agent_state
    if not state.has_passphrase():
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Setup required",
            "This Eugene Plexus install has no passphrase set yet. "
            "Complete the first-run wizard's security screen "
            "(POST /v1/auth/initialize) before using other endpoints.",
        )
    return state


def verify_bearer(request: Request, token: str, *, classes: tuple[str, ...]) -> tokens.Claims:
    """Verify one bearer addressed to this machine, or raise the 401 that says why."""
    auth: AuthState = request.app.state.auth_state
    if auth.is_revoked(token):
        raise revoked_session_problem()
    try:
        return auth.trust.verify(token, classes=classes)
    except tokens.TokenError as exc:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED, "Invalid token", f"Bearer token rejected: {exc}"
        ) from exc


def _bearer(creds: HTTPAuthorizationCredentials | None) -> str:
    if creds is None or not creds.credentials:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide a bearer token via the Authorization: Bearer header.",
        )
    return creds.credentials


def _refuse(claims: tokens.Claims, what: str) -> HTTPException:
    return _problem(
        status.HTTP_401_UNAUTHORIZED,
        "Wrong audience",
        f"A {claims.sub!r} service token from {claims.iss!r} may not {what}.",
    )


def _is_control(claims: tokens.Claims) -> bool:
    return (
        claims.is_service
        and claims.iss == tokens.ISSUER_CONTROL
        and claims.sub == tokens.SUB_CONTROL
    )


def require_operator_session(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """An operator session addressed to this machine."""
    return verify_bearer(request, _bearer(creds), classes=_SESSION)


def require_operator_or_service(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """The reads: a session, this machine's children, the root, or a gateway."""
    claims = verify_bearer(request, _bearer(creds), classes=_SESSION_OR_SERVICE)
    if claims.is_session:
        return claims
    recipient = request.app.state.auth_state.trust.recipient
    if claims.is_local_service(recipient) or _is_control(claims):
        return claims
    if claims.sub == tokens.SUB_GATEWAY:
        return claims
    raise _refuse(claims, "read this agent")


def require_operator_or_gateway(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """Start and stop a runtime, and the client-key policy (M6's lifecycle).

    `gateway` exactly: a driver's or the library's token still cannot
    stop a process holding a GPU, which is why these were operator-only
    through M5. A `gateway` token from another machine verified only if
    the bundle grants that machine `gateway`.
    """
    claims = verify_bearer(request, _bearer(creds), classes=_SESSION_OR_SERVICE)
    if claims.is_session or claims.sub == tokens.SUB_GATEWAY:
        return claims
    raise _refuse(claims, "start or stop a runtime; only the operator or the gateway may")


def require_operator_or_control(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """Declare a runtime: a session, or the control root's own token.

    Signed by the root's key and addressed to this node, so no other
    node can mint one: a leaked worker key cannot run anything here."""
    claims = verify_bearer(request, _bearer(creds), classes=_SESSION_OR_SERVICE)
    if claims.is_session or _is_control(claims):
        return claims
    raise _refuse(claims, "declare a runtime; only the operator or the control root may")


def require_local_service(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """A child of this agent, with the token it was spawned with."""
    claims = verify_bearer(request, _bearer(creds), classes=(tokens.TYP_SERVICE,))
    if claims.is_local_service(request.app.state.auth_state.trust.recipient):
        return claims
    raise _refuse(claims, "ask this agent for a token; only its own children may")
