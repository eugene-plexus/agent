"""Auth endpoints: passphrase initialize, login, logout.

Two-state lifecycle:

  * **Uninitialized** — first run, no passphrase set. Only
    `POST /v1/auth/initialize` and `GET /healthz` work; every other
    endpoint returns 503 "Setup required" until initialize succeeds.

  * **Initialized** — passphrase set. Login is open; everything else
    requires a bearer token.

Login uses Argon2id-PHC verification (constant-time). Successful
login derives the master key from the passphrase + stored salt and
caches it in `AuthState.master_key` so the supervisor can thread it
to spawned children that need to decrypt at-rest secrets.

Rate-limiting: bucket per source IP, 5 failures within 60 seconds
locks the source out for 60 seconds. Cleared on success.
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .. import keyring_store, security
from .._generated.common_models import (
    AuthLoginRequest,
    AuthLoginResponse,
    Problem,
)
from ..auth_state import AuthState
from ..dependencies import require_operator_session
from ..state import AgentState

log = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

_bearer_scheme = HTTPBearer(auto_error=False)

# Rate limit: per-IP, sliding window. Tuned to be friction for an
# attacker and basically invisible to a legitimate operator who
# mistypes once or twice. State lives in AuthState (per-app, not
# module-global) so tests don't poison each other.
_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_FAILURES = 5


def _problem(status_code: int, title: str, detail: str) -> HTTPException:
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


@router.get("/v1/auth/status")
async def auth_status(request: Request) -> dict[str, bool]:
    """Public probe — has a passphrase been set on this install?

    The UI uses this to disambiguate "fresh install, route to /setup"
    from "logged-out, route to /login" without consuming a rate-limited
    login attempt. Returns one boolean only; no secrets, no per-IP
    behavior, no rate limit — safe to call on every page load.
    """
    state: AgentState = request.app.state.agent_state
    return {"initialized": state.has_passphrase()}


class _InitializeRequest(BaseModel):
    """The wizard's payload for first-run passphrase setup."""

    passphrase: str = Field(min_length=1)


@router.post("/v1/auth/initialize", response_model=AuthLoginResponse)
async def initialize(request: Request, body: _InitializeRequest) -> AuthLoginResponse:
    """First-run passphrase setup. Idempotent only as a no-op — refuses
    if a passphrase is already set (use the change-passphrase flow in
    v0.3+, not this endpoint).

    On success: persists the Argon2id-PHC passphrase hash and master-key
    salt to `agent.yaml`, derives the master key into memory, and
    issues an operator session token so the wizard can continue without
    a separate login round-trip.
    """
    state: AgentState = request.app.state.agent_state
    auth: AuthState = request.app.state.auth_state

    if state.has_passphrase():
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Already initialized",
            "This install already has a passphrase set. Use the change-"
            "passphrase flow (planned v0.3) or reset the install by "
            "removing agent.yaml's auth block by hand.",
        )

    # Hash the passphrase (for verification on future logins) and
    # generate the master-key salt (so derivation is deterministic
    # across restarts).
    passphrase_hash = security.hash_passphrase(body.passphrase)
    salt = security.generate_master_key_salt()
    state.set_passphrase(
        passphrase_hash=passphrase_hash,
        master_salt_b64=base64.b64encode(salt).decode("ascii"),
    )

    # Derive + cache the master key so the rest of this process run
    # can encrypt secrets without re-prompting.
    master_key = security.derive_master_key(body.passphrase, salt)
    auth.set_master_key(master_key)

    # If the operator chose OS-keyring mode, persist the master key
    # so the next restart auto-unlocks without re-prompting.
    _persist_master_key_if_keyring_mode(state, master_key)

    # If the supervisor was spawned before this initialize call (in
    # production it is — the lifespan builds it during app startup),
    # any children it already launched ran without MASTER_KEY in their
    # env. Signal a respawn so they pick it up. No-op when nothing is
    # running yet (first-run wizard usually completes before topology
    # is configured), so the operator sees no spurious churn.
    await _restart_supervised_children_if_present(request)

    token, exp = security.issue_operator_token(signing_key=auth.signing_key)
    log.info("first-run passphrase set; operator session issued")
    return AuthLoginResponse(
        sessionToken=token,
        expiresAt=_dt_from_unix(exp),
        operatorName=None,
    )


@router.post("/v1/auth/login", response_model=AuthLoginResponse)
async def login(request: Request, body: AuthLoginRequest) -> AuthLoginResponse:
    """Verify the passphrase, issue a session token, and cache the
    derived master key. Rate-limited per source IP."""
    state: AgentState = request.app.state.agent_state
    auth: AuthState = request.app.state.auth_state
    remote = request.client.host if request.client else "unknown"

    if not state.has_passphrase():
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Setup required",
            "No passphrase set yet. Call POST /v1/auth/initialize first.",
        )

    if auth.is_login_rate_limited(
        remote,
        window_seconds=_RATE_LIMIT_WINDOW_SECONDS,
        max_in_window=_RATE_LIMIT_MAX_FAILURES,
    ):
        raise _problem(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Rate limited",
            f"Too many failed logins from {remote}. Wait "
            f"{_RATE_LIMIT_WINDOW_SECONDS} seconds and try again.",
        )

    stored = state.get_passphrase_hash()
    if stored is None or not security.verify_passphrase(body.passphrase, stored):
        auth.record_login_failure(
            remote,
            window_seconds=_RATE_LIMIT_WINDOW_SECONDS,
            max_in_window=_RATE_LIMIT_MAX_FAILURES,
        )
        log.warning("failed login attempt from %s", remote)
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Wrong passphrase",
            "Passphrase did not match. Repeated failures from one source are rate-limited.",
        )

    auth.clear_login_failures(remote)

    # Re-derive the master key on every login so it's in memory for
    # this process. Idempotent — same passphrase + salt → same key.
    salt_b64 = state.get_master_salt_b64()
    if salt_b64 is None:
        # Shouldn't happen if initialize() was used; defensive nonetheless.
        raise _problem(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Corrupt auth state",
            "Passphrase hash present but master-key salt missing. Restore "
            "agent.yaml from a known-good backup or re-initialize.",
        )
    salt = base64.b64decode(salt_b64)
    had_master_key = auth.has_master_key()
    derived = security.derive_master_key(body.passphrase, salt)
    auth.set_master_key(derived)

    # If the operator's chosen mode is OS-keyring, persist the freshly
    # derived key for next time. Idempotent — same passphrase + salt
    # produces the same key, so re-saving the same value isn't an
    # error. Useful when the operator switches `securityMode` from
    # prompt to keyring without re-initializing.
    _persist_master_key_if_keyring_mode(state, derived)

    # On the FIRST successful login of a process run, children that
    # the supervisor already launched are running without MASTER_KEY in
    # their env (the lifespan starts them before the operator unlocks).
    # Signal a respawn so they pick up the now-available key. Skip the
    # restart on subsequent logins in the same process run — the key
    # is already threaded to live children, repeated logins would just
    # churn for no reason.
    if not had_master_key:
        await _restart_supervised_children_if_present(request)

    token, exp = security.issue_operator_token(signing_key=auth.signing_key)
    log.info("operator login from %s", remote)
    return AuthLoginResponse(
        sessionToken=token,
        expiresAt=_dt_from_unix(exp),
        operatorName=None,
    )


def _persist_master_key_if_keyring_mode(state: AgentState, master_key: bytes) -> None:
    """Save the master key to the OS keyring when the operator opted in.

    Best-effort: keyring write failures log a warning but don't block
    the login flow. Worst case: next restart still works, just needs
    a passphrase prompt. `state.get_config("securityMode")` is the
    canonical source — the operator may have toggled it after
    initialize, so we check on every login.
    """
    if state.get_config("securityMode") != "os_keyring":
        return
    if keyring_store.set_master_key(master_key):
        log.info("master key persisted to OS keyring for auto-unlock")
    else:
        log.warning(
            "securityMode is os_keyring but keyring write failed; "
            "auto-unlock will not work on next restart"
        )


async def _restart_supervised_children_if_present(request: Request) -> None:
    """Ask the supervisor to respawn every child, if one is wired in.

    The supervisor lands on `app.state.supervisor` during the lifespan.
    Tests that don't exercise supervision skip this step. Production
    always has it. Catches and logs any errors so the auth route's
    contract (return a session token on success) isn't broken by a
    misbehaving child.
    """
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None:
        return
    try:
        restarted = await supervisor.restart_all()
        if restarted:
            log.info(
                "signaled %d supervised child(ren) to respawn so they pick "
                "up the now-available MASTER_KEY: %s",
                len(restarted),
                ", ".join(restarted),
            )
    except Exception as e:
        log.warning("restart_all failed; children may run without master key: %s", e)


@router.delete("/v1/auth/sessions/current", status_code=204)
async def logout(
    request: Request,
) -> None:
    """Add the current session token to the in-memory revocation set.

    Validates via the same dependency as protected routes so callers
    can't revoke arbitrary tokens — only the one they're holding.
    """
    creds: HTTPAuthorizationCredentials | None = await _bearer_scheme(request)
    if creds is None or not creds.credentials:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide the session token to revoke via Authorization: Bearer.",
        )
    # Validate before revoking so a bogus token doesn't grow the
    # revocation set unboundedly.
    _ = require_operator_session(request, creds)
    auth: AuthState = request.app.state.auth_state
    auth.revoke(creds.credentials)
    log.info("session revoked")


def _dt_from_unix(unix_seconds: int) -> datetime:
    """Convert a unix-epoch integer to an aware UTC datetime so the
    generated AuthLoginResponse model (which types expiresAt as
    datetime) accepts it."""
    return datetime.fromtimestamp(unix_seconds, tz=UTC)
