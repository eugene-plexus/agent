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

**A limit on frozen trust** (row 46). Every bearer passes `verify_bearer`,
and there, past `maxTrustBundleAgeHours` without word from the root, a
token another machine's key signed is refused with 403 on every route that
changes something (`refuse_when_frozen`). Which routes change something is
said once, below: GET, HEAD and OPTIONS read, and every other route is
named in `FROZEN_TRUST_QUESTIONS` or `FROZEN_TRUST_CHANGES`. A route in
neither is refused as a change, and `tests/test_trust.py` fails until it
is named.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import tokens
from ._generated.common_models import Problem
from .auth_state import AuthState
from .state import AgentState
from .trust import MAX_AGE_HOURS_KEY, frozen_refusal, max_age_hours

_bearer_scheme = HTTPBearer(auto_error=False)

_SESSION = (tokens.TYP_SESSION,)
_SESSION_OR_SERVICE = (tokens.TYP_SESSION, tokens.TYP_SERVICE)

#: Methods that read. Every route on them answers past the limit (row 46).
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: Row 46: routes on other methods that change nothing, so another machine's
#: token still reaches them past the limit. Each is a question or part of an
#: answer: refusing one would stop answers, not changes.
FROZEN_TRUST_QUESTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # The dry run a gateway asks before it wakes a runtime.
        ("POST", "/v1/runtimes/admission"),
        # One request's admission for a client key: part of every answer.
        ("POST", "/v1/auth/client-keys/admission"),
        ("POST", "/v1/config/test"),
        ("POST", "/v1/benchmarks/preflight"),
        ("POST", "/v1/profile-builds/preflight"),
        ("POST", "/v1/serving-measurements/preflight"),
        ("POST", "/v1/entrypoint/preview"),
        ("POST", "/v1/engines/{engine}/prepared/inspect"),
        ("POST", "/v1/library/folders/check"),
        ("POST", "/v1/node/update/check"),
        ("POST", "/v1/people/known"),
    }
)

#: Row 46: routes that change something here. Past the limit another
#: machine's token is refused on each. Many take no token from another
#: machine today (an operator session is the root's); they are named so
#: that a route that starts to take one is covered from its first day.
FROZEN_TRUST_CHANGES: frozenset[tuple[str, str]] = frozenset(
    {
        *(
            (method, "/api/proxy/{target}/{path:path}")
            for method in ("DELETE", "PATCH", "POST", "PUT")
        ),
        ("POST", "/link/approve/items/{ident}"),
        ("POST", "/link/approve/items/{ident}/reject"),
        ("POST", "/link/confirm"),
        ("POST", "/link/key"),
        ("POST", "/link/key/remove"),
        ("POST", "/link/passkey/code"),
        ("POST", "/link/passkey/remove"),
        ("POST", "/link/remove"),
        ("POST", "/oidc/{path:path}"),
        ("POST", "/v1/app-catalogue/custom"),
        ("DELETE", "/v1/app-catalogue/custom/{app_id}"),
        ("DELETE", "/v1/apps/{app_id}"),
        ("PATCH", "/v1/apps/{app_id}/config"),
        ("DELETE", "/v1/apps/{app_id}/install"),
        ("POST", "/v1/apps/{app_id}/install"),
        ("POST", "/v1/apps/{app_id}/restart"),
        ("POST", "/v1/apps/{app_id}/rollback"),
        ("PATCH", "/v1/apps/{app_id}/settings"),
        ("POST", "/v1/apps/{app_id}/start"),
        ("POST", "/v1/apps/{app_id}/stop"),
        ("POST", "/v1/apps/{app}/folder-access"),
        ("POST", "/v1/apps/{app}/folder-access/release"),
        ("DELETE", "/v1/apps/{app}/folder-access/{ident}"),
        ("POST", "/v1/apps/{app}/folder-access/{ident}/unknown/{entry}"),
        ("POST", "/v1/auth/client-keys"),
        ("DELETE", "/v1/auth/client-keys/{key_id}"),
        ("PUT", "/v1/auth/client-keys/{key_id}/limits"),
        ("POST", "/v1/auth/client-keys/{key_id}/rotate"),
        ("POST", "/v1/auth/initialize"),
        ("POST", "/v1/auth/login"),
        ("POST", "/v1/auth/passphrase"),
        ("POST", "/v1/auth/service-token"),
        ("DELETE", "/v1/auth/sessions/current"),
        ("POST", "/v1/benchmarks"),
        ("POST", "/v1/benchmarks/{job_id}/cancel"),
        ("POST", "/v1/components"),
        ("DELETE", "/v1/components/{name}"),
        ("PATCH", "/v1/components/{name}"),
        ("POST", "/v1/components/{name}/restart"),
        ("PATCH", "/v1/config"),
        ("DELETE", "/v1/engines/{engine}/install"),
        ("POST", "/v1/engines/{engine}/install"),
        ("POST", "/v1/engines/{engine}/uninstall"),
        # Row 8c: what Eugene's own account may run.
        ("POST", "/v1/exec-approvals"),
        ("DELETE", "/v1/exec-approvals/{id}"),
        ("DELETE", "/v1/entrypoint"),
        ("POST", "/v1/entrypoint/apply"),
        ("POST", "/v1/logs"),
        ("POST", "/v1/model-copies/clear"),
        ("POST", "/v1/node/enroll"),
        ("DELETE", "/v1/node/history"),
        ("DELETE", "/v1/node/machine-key"),
        ("POST", "/v1/node/machine-key"),
        ("POST", "/v1/node/machine-list"),
        ("POST", "/v1/node/passphrase"),
        ("POST", "/v1/node/reach"),
        ("POST", "/v1/node/trust-bundle"),
        ("POST", "/v1/node/unenroll"),
        ("POST", "/v1/node/update"),
        ("POST", "/v1/ollama/send"),
        ("POST", "/v1/profile-builds"),
        ("POST", "/v1/profile-builds/{job_id}/cancel"),
        ("POST", "/v1/runtimes"),
        ("DELETE", "/v1/runtimes/{name}"),
        ("PATCH", "/v1/runtimes/{name}"),
        ("PUT", "/v1/runtimes/{name}/auto-start"),
        ("POST", "/v1/runtimes/{name}/restart"),
        ("POST", "/v1/runtimes/{name}/start"),
        ("POST", "/v1/runtimes/{name}/stop"),
        ("POST", "/v1/serving-measurements"),
        ("POST", "/v1/serving-measurements/{job_id}/cancel"),
        ("DELETE", "/v1/site/links/{subject}"),
        ("POST", "/v1/updates/all"),
    }
)

#: Row 46: through `/api/proxy` to a component on this machine (another
#: machine's components reach the browser proxy through this agent's
#: machine port), the requests on methods other than GET, HEAD and OPTIONS
#: that are answers or questions, by the kind of component the target
#: names. Another machine's token still reaches these past the limit;
#: anything else through the proxy -- a write to the Library or the store,
#: a component's config -- is refused as a change here is, and so is
#: anything sent to a kind not named. The agent's own API (`agent`,
#: `node:<this machine>`) is guarded by its own routes.
FROZEN_TRUST_PROXY_QUESTIONS: dict[str, frozenset[tuple[str, str]]] = {
    "inference-driver": frozenset(
        {
            *(
                ("POST", path)
                for path in (
                    "/v1/generate",
                    "/v1/generate/stream",
                    "/v1/generate/count",
                    "/v1/moderate",
                    "/v1/speak",
                    "/v1/transcribe",
                    "/v1/image",
                    "/v1/image/stream",
                    "/v1/video",
                    "/v1/embed",
                    "/v1/rerank",
                    "/v1/decide",
                )
            ),
            ("POST", "/v1/config/test"),
        }
    ),
    "tool-driver": frozenset({("POST", "/v1/tools/web_search"), ("POST", "/v1/config/test")}),
    # The doors. Another machine's service token is refused there anyway;
    # named so the rule is the same for every kind.
    "gateway": frozenset(
        {
            *(
                ("POST", path)
                for path in (
                    "/v1/chat/completions",
                    "/v1/completions",
                    "/v1/messages",
                    "/v1/messages/count_tokens",
                    "/v1/responses",
                    "/v1/responses/compact",
                    "/v1/responses/input_tokens",
                    "/v1/moderations",
                    "/v1/audio/speech",
                    "/v1/audio/transcriptions",
                    "/v1/audio/translations",
                    "/v1/web_search",
                    "/v1/images/generations",
                    "/v1/images/edits",
                    "/v1/images/variations",
                    "/v1/videos",
                    "/v1/embeddings",
                    "/v1/rerank",
                    "/v1/systemone",
                    "/v1beta/models/{model}:generateContent",
                    "/v1beta/models/{model}:streamGenerateContent",
                    "/v1beta/models/{model}:countTokens",
                    "/v1/models/{model}:generateContent",
                    "/v1/models/{model}:streamGenerateContent",
                    "/v1/models/{model}:countTokens",
                    "/v1/config/test",
                )
            ),
        }
    ),
    "library": frozenset(
        {
            ("POST", "/v1/eligibility"),
            ("POST", "/v1/catalogue/search"),
            ("POST", "/v1/keep-jobs/{id}/files/state"),
            ("POST", "/v1/config/test"),
        }
    ),
    "store": frozenset(),
    "control": frozenset(),
}


def _template(path: str) -> re.Pattern[str]:
    """`/v1/models/{model}:countTokens` as a pattern: one segment per name."""
    parts = re.split(r"\{[^/{}]+\}", path)
    return re.compile("^" + "[^/]+".join(re.escape(part) for part in parts) + "$")


_PROXY_QUESTIONS = {
    kind: tuple((method, _template(path)) for method, path in named)
    for kind, named in FROZEN_TRUST_PROXY_QUESTIONS.items()
}


def proxied_change(kind: str | None, method: str, path: str) -> bool:
    """Whether a request through the proxy to a component of `kind` on this
    machine changes something (row 46). A kind not named, and a request
    in no list, count as changes."""
    method = method.upper()
    if method in READ_METHODS:
        return False
    return not any(
        method == named and pattern.match(path)
        for named, pattern in _PROXY_QUESTIONS.get(kind or "", ())
    )


_ANY_TOKEN = (tokens.TYP_SESSION, tokens.TYP_SERVICE, tokens.TYP_CLIENT, tokens.TYP_ADMINISTRATION)


def refuse_proxied_when_frozen(
    request: Request, kind: str | None, path: str, credentials: list[str]
) -> None:
    """The freeze for a request the proxy carries to a component on this
    machine, which checks its token itself and knows nothing of the root's
    silence (row 46). Refused when the limit is passed, a credential is
    another machine's token addressed here, and the request is a change.
    A credential that does not verify here is the component's to refuse."""
    auth: AuthState | None = getattr(request.app.state, "auth_state", None)
    if auth is None or request.method.upper() in READ_METHODS:
        return
    trust = auth.trust
    age = trust.frozen_age(trust_limit_hours(request))
    if age is None or not proxied_change(kind, request.method, path):
        return
    for token in credentials:
        try:
            claims = trust.verify(token, classes=_ANY_TOKEN)
        except tokens.TokenError:
            continue
        if trust.minted_elsewhere(claims):
            raise _problem(status.HTTP_403_FORBIDDEN, "Root not heard", frozen_refusal(age))


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
        claims = auth.trust.verify(token, classes=classes)
    except tokens.TokenError as exc:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED, "Invalid token", f"Bearer token rejected: {exc}"
        ) from exc
    # Row 4: the bundle refuses a root token below its own epoch; this node
    # may already hold a higher one, from a promoted root's signed list, before
    # that root's bundle arrives. A root it no longer follows commands nothing.
    record = getattr(getattr(request.app.state, "node_identity", None), "record", None)
    held = getattr(record, "epoch", None)
    if claims.epoch is not None and held is not None and claims.epoch < held:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Superseded root",
            tokens.no_longer_followed(claims.epoch, held),
        )
    if _legacy_root_token_after_a_promotion(auth, claims, record):
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Superseded root",
            tokens.no_longer_followed("unknown", int(held or 0)),
        )
    refuse_when_frozen(request, claims)
    # An operator session that arrived through the HTTPS entry point keeps a
    # configuration applied from Settings; without one it goes back
    # (`entrypoint_setup`). A console kept on its own port has not moved, so
    # any operator session there shows the console still answers.
    entry = getattr(request.app.state, "entrypoint_config", None)
    if (
        claims.is_session
        and getattr(request.app.state, "entrypoint_confirm_by", None) is not None
        and (
            getattr(request.state, "via_entrypoint", False)
            or (entry is not None and entry.console_direct)
        )
    ):
        from .entrypoint_setup import confirm

        confirm(request.app)
    return claims


def trust_limit_hours(request: Request) -> int:
    """`maxTrustBundleAgeHours` in effect on this agent (row 46)."""
    state = getattr(request.app.state, "agent_state", None)
    return max_age_hours(state.get_config(MAX_AGE_HOURS_KEY) if state is not None else None)


def changes_something(request: Request) -> bool:
    """Whether the route this request matched changes something (row 46).

    Read off the matched route's template, so `/v1/runtimes/a/stop` and
    `/v1/runtimes/b/stop` are one entry. A request with no matched route
    (a background job re-checking its session) and a route in neither
    list both count as changes."""
    method = str(request.scope.get("method") or "").upper()
    if method in READ_METHODS:
        return False
    return (method, getattr(request.scope.get("route"), "path", None)) not in (
        FROZEN_TRUST_QUESTIONS
    )


def refuse_when_frozen(request: Request, claims: tokens.Claims) -> None:
    """Past the limit without word from the root, refuse another machine's
    token on a route that changes something (row 46).

    The root's tokens are checked against the same old bundle and the
    root is the authority, so they pass; so do this machine's own tokens
    for its children, and every read, so a dead root never stops answers.
    """
    trust = request.app.state.auth_state.trust
    if not trust.minted_elsewhere(claims):
        return
    age = trust.frozen_age(trust_limit_hours(request))
    if age is None or not changes_something(request):
        return
    raise _problem(status.HTTP_403_FORBIDDEN, "Root not heard", frozen_refusal(age))


def _legacy_root_token_after_a_promotion(auth: Any, claims: tokens.Claims, record: Any) -> bool:
    """A root token without `ep` (a root older than row 4), once this node
    knows of a promotion (row 4 re-review): it holds an epoch above its
    bundle's (a promoted root's list came first), or above the epoch it held
    when it first ran row 4 code or joined. Such a token cannot be told
    from one the replaced root signed after it was replaced."""
    if claims.iss != tokens.ISSUER_CONTROL or claims.typ == tokens.TYP_CLIENT:
        return False
    if claims.epoch is not None:
        return False
    held = getattr(record, "epoch", None)
    if held is None:
        return False
    bundle = auth.trust.bundle
    if bundle is not None and held > bundle.epoch:
        return True
    legacy = getattr(record, "legacy_token_epoch", None)
    return legacy is not None and held > legacy


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


def require_local_store(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """This machine's own store, and nothing else (row 53): it asks which
    people the root still has, through `POST /v1/people/known`."""
    claims = verify_bearer(request, _bearer(creds), classes=(tokens.TYP_SERVICE,))
    recipient = request.app.state.auth_state.trust.recipient
    if claims.sub == "store" and claims.is_local_service(recipient):
        return claims
    raise _refuse(claims, "ask about people here; only this machine's store may")


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


def require_control(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """The control root's own token, addressed to this node, and nothing
    else: for what only the root asks, such as removing a person's link to
    an account here when they or their site's owner did so in Workbench."""
    claims = verify_bearer(request, _bearer(creds), classes=_SESSION_OR_SERVICE)
    if _is_control(claims):
        return claims
    raise _refuse(claims, "do this; only the control root may")


def require_local_service(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims:
    """A child of this agent, with the token it was spawned with."""
    claims = verify_bearer(request, _bearer(creds), classes=(tokens.TYP_SERVICE,))
    if claims.is_local_service(request.app.state.auth_state.trust.recipient):
        return claims
    raise _refuse(claims, "ask this agent for a token; only its own children may")
