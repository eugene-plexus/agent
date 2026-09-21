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

**And "source IP" means the browser's, not the proxy's.** Every login
in this product arrives through this agent's own loopback proxy, so
`request.client.host` is `127.0.0.1` for every caller there has ever
been: one bucket for the whole install, which five mistyped passphrases
from one person emptied for everybody. `peer.peer_of` reads the address
the proxy saw. See `peer.py` for why that header is believable and
`X-Forwarded-For` was not.

**Client keys (S4, 2026-09-15)** live at the bottom of this file. A
different kind of credential: not a session, not a component's service
token, but a long-lived named bearer an operator hands to an app
outside the install. Everything about how they are stored, why the
record keeps a tail rather than a prefix, and why revocation is bounded
rather than instant is in `client_keys.py`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .. import client_keys, keyring_store, peer, security
from .._generated.common_models import (
    AuthLoginRequest,
    AuthLoginResponse,
    Problem,
)
from .._generated.models import (
    AuthStatus,
    ClientKey,
    ClientKeyCreated,
    ClientKeyCreateRequest,
    ClientKeyList,
    ClientKeyPolicy,
    ClientKeyRevocations,
)
from ..auth_state import AuthState
from ..client_key_registry import registry
from ..client_keys import ClientKeyStore
from ..dependencies import require_operator_or_gateway, require_operator_session
from ..state import AgentState
from .config import connect_shares

log = logging.getLogger(__name__)

# How long `GET /v1/auth/status` waits for the keyring probe before
# answering without it. Generous for a desktop keyring's first-contact
# dialog to be dismissed; short enough that a locked headless Secret
# Service does not hang the wizard's first page load.
KEYRING_PROBE_BUDGET_SECONDS = 3.0

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


@router.get("/v1/auth/status", response_model=AuthStatus)
async def auth_status(request: Request) -> AuthStatus:
    """Public probe — has a passphrase been set, is the process unlocked,
    and can this host's keyring keep it that way?

    The UI uses `initialized` to disambiguate "fresh install, route to
    /setup" from "logged-out, route to /login" without consuming a
    rate-limited login attempt. The first-run wizard reads
    `keyringAvailable` to default `securityMode` to `os_keyring` where a
    keyring exists and to say plainly where one does not (S0 of the
    hobbyist UX plan). No secrets, no per-IP behavior, no rate limit —
    safe to call on every page load.

    The probe runs once per process, in a thread, with a deadline: a
    Secret Service that is present but locked can block on a prompt
    nobody will answer, and a status endpoint must not hang with it.
    Past the deadline the field is absent, not False — "we could not
    tell" is a different answer from "no".
    """
    state: AgentState = request.app.state.agent_state
    auth: AuthState = request.app.state.auth_state
    try:
        available: bool | None = await asyncio.wait_for(
            asyncio.to_thread(keyring_store.probe_sync), timeout=KEYRING_PROBE_BUDGET_SECONDS
        )
    except TimeoutError:
        log.warning(
            "the OS keyring probe did not finish within %.0fs; reporting keyringAvailable "
            "as unknown",
            KEYRING_PROBE_BUDGET_SECONDS,
        )
        available = None
    return AuthStatus(
        initialized=state.has_passphrase(),
        unlocked=auth.has_master_key(),
        keyringAvailable=available,
    )


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
        # **The remedy is to sign in, and it used to be to edit a YAML
        # file** (review §6.1 #7). A first-time user whose trust root
        # cannot start reaches this 409 on the wizard's second attempt,
        # and telling them to remove the auth block by hand from the
        # file that holds their install's key material is both the most
        # destructive available instruction and one the browser they are
        # sitting in cannot carry out.
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Already set up",
            "This install already has a passphrase. Sign in with it instead of setting it "
            "up again. If you have forgotten it, there is no recovery: the install's keys "
            "are sealed with it.",
        )

    if state.lost_its_passphrase():
        # **The file held a passphrase and we cannot read it** (review
        # §6.1 #6). `yaml.safe_dump` sorts keys, so a half-written
        # `agent.yaml` keeps its `auth` block and loses the tail: the
        # file comes up degraded, the loaded state has no passphrase,
        # and this endpoint is what the UI offers next. Running the
        # wizard here is the one thing that makes it worse — a fresh
        # `masterSalt` orphans every secret the old one sealed,
        # including the install signing key on every enrolled node, and
        # nothing would say that had happened.
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Set up, but its keys are missing",
            "This install has been set up before, and the part of its configuration that "
            "holds the passphrase is missing. Setting a new one would lock you out of "
            "everything the old one sealed, so it is refused. Restore the configuration "
            "file from a backup, or from the copy kept beside it, and start again.",
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
    remote = (
        peer.peer_of(
            request.client.host if request.client else None,
            request.headers.get(peer.PEER_HEADER),
        )
        or "unknown"
    )

    if not state.has_passphrase():
        # **An enrolled node is a different situation and needs different
        # advice.** A worker onboarded with `eugene-plexus-agent join` has
        # no passphrase of its own and never will: it verifies tokens with
        # the install's signing key, so an operator session minted at the
        # control root already works here. Telling it to run first-run
        # setup would be telling an operator to raise a second install on
        # a machine that is already part of one. Found by M9's acceptance
        # run, which was the first thing to log in at a joined node.
        identity = getattr(request.app.state, "node_identity", None)
        record = identity.record if identity is not None else None
        if record is not None and record.enrolled:
            raise _problem(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "No local passphrase",
                f"This node is enrolled in the install at {record.control_url} and has no "
                f"passphrase of its own. Log in at the control root instead; the session "
                f"token it issues is accepted here, because the whole install shares one "
                f"signing key. Do not run first-run setup on a node that has joined.",
            )
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
        # **And log in to the file servers, for the same reason.** The
        # share passwords are sealed with the master key, so an agent on
        # `prompt_on_startup` had nothing to unseal at boot. This is the
        # moment it does — and it is the moment before the respawned
        # children go looking for models.
        await connect_shares(request)

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
    salt_b64 = state.get_master_salt_b64()
    if salt_b64 is None:
        return
    if keyring_store.set_master_key(master_key, keyring_store.install_id_for(salt_b64)):
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


# --------------------------------------------------------------------- #
# Client keys (hobbyist UX S4, 2026-09-15)
# --------------------------------------------------------------------- #


def _keys(request: Request) -> ClientKeyStore:
    """The record store, or a transient one.

    `app.state.client_keys` is wired in the lifespan. A test app that
    skips it gets an in-memory store rather than a 500, which is the
    same tolerance `library_folders` and `supervisor` already get.
    """
    store: ClientKeyStore | None = getattr(request.app.state, "client_keys", None)
    if store is None:
        store = ClientKeyStore(Path(".") / client_keys.KEYS_FILE)
        request.app.state.client_keys = store
    return store


def _to_model(record: client_keys.ClientKeyRecord) -> ClientKey:
    return ClientKey(
        id=record.id,
        name=record.name,
        tail=record.tail,
        createdAt=client_keys.as_datetime(record.created_at),
        expiresAt=client_keys.as_datetime(record.expires_at),
        revokedAt=(
            client_keys.as_datetime(record.revoked_at) if record.revoked_at is not None else None
        ),
    )


@router.get(
    "/v1/auth/client-keys",
    response_model=ClientKeyList,
    response_model_exclude_none=True,
    dependencies=[Depends(require_operator_session)],
)
async def list_client_keys(request: Request) -> ClientKeyList:
    """The records, newest first. Never the tokens -- see `client_keys`."""
    owner = registry(request)
    if owner.enrolled:
        await owner.migrate()
        data = await owner.forward(
            "GET", "/v1/auth/client-keys", authorization=request.headers.get("authorization")
        )
        data.update(migration=owner.migration, detail=owner.detail)
        return ClientKeyList.model_validate(data)
    try:
        return ClientKeyList.model_validate(
            dict(
                keys=[_to_model(r) for r in _keys(request).records()],
                scope="standalone",
                migration="standalone",
                revision=_keys(request).revision,
            )
        )
    except OSError as exc:
        raise owner.local_unavailable() from exc


@router.post(
    "/v1/auth/client-keys",
    response_model=ClientKeyCreated,
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_operator_session)],
)
async def create_client_key(request: Request, body: ClientKeyCreateRequest) -> ClientKeyCreated:
    """Mint at the control root when enrolled, or in this standalone registry.

    Forward the caller's operator credential; never upgrade a service token.
    The bearer is returned once and only metadata is durably stored.
    """
    owner = registry(request)
    if owner.enrolled:
        return ClientKeyCreated.model_validate(
            await owner.forward(
                "POST",
                "/v1/auth/client-keys",
                authorization=request.headers.get("authorization"),
                body=body.model_dump(mode="json"),
            )
        )
    name = body.name.strip()
    if not name:
        raise _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Name required",
            "Give the key a name so it can be told apart from the others later -- "
            "what app or machine it is for.",
        )
    ttl_days = body.ttlDays if body.ttlDays is not None else client_keys.DEFAULT_TTL_DAYS
    auth: AuthState = request.app.state.auth_state
    key_id = client_keys.new_key_id()
    # One instant for both, so the record's `createdAt` IS the token's
    # `iat`. Two clock reads would put the record a fraction of a second
    # before the token and make "valid a year" read as 364 days in any
    # arithmetic a client does on the two fields.
    issued_at = int(time.time())
    token, expires_at = security.issue_client_token(
        signing_key=auth.signing_key,
        key_id=key_id,
        name=name,
        ttl_seconds=int(ttl_days) * 24 * 3600,
        now=issued_at,
    )
    try:
        record = _keys(request).add(
            client_keys.ClientKeyRecord(
                id=key_id,
                name=name,
                tail=client_keys.tail_of(token),
                created_at=float(issued_at),
                expires_at=float(expires_at),
            )
        )
    except OSError as exc:
        raise owner.local_unavailable() from exc
    log.info("minted client key %r (id %s), valid %d day(s)", name, key_id, ttl_days)
    return ClientKeyCreated(key=_to_model(record), token=token)


@router.get(
    "/v1/auth/client-keys/policy",
    response_model=ClientKeyPolicy,
    response_model_exclude_none=True,
    dependencies=[Depends(require_operator_or_gateway)],
)
async def client_key_policy(request: Request, response: Response) -> ClientKeyPolicy:
    response.headers["Cache-Control"] = "no-store"
    owner = registry(request)
    if owner.enrolled:
        # Migration runs independently; a timeout there must not renew or delay policy.
        return ClientKeyPolicy.model_validate(
            await owner.forward("GET", "/v1/auth/client-keys/policy")
        )
    try:
        store = _keys(request)
        return ClientKeyPolicy.model_validate(
            dict(
                authority="standalone",
                revision=store.revision,
                generatedAt=time.time(),
                keys=[
                    _to_model(r).model_dump(
                        mode="json", include={"id", "expiresAt", "revokedAt"}, exclude_none=True
                    )
                    for r in store.records()
                ],
            )
        )
    except OSError as exc:
        raise owner.local_unavailable() from exc


@router.get(
    "/v1/auth/client-keys/revoked",
    response_model=ClientKeyRevocations,
    response_model_exclude_none=True,
    dependencies=[Depends(require_operator_or_gateway)],
)
async def list_revoked_client_keys(request: Request) -> ClientKeyRevocations:
    """What the gateway polls. Ids only, and the revision they are at.

    `require_operator_or_gateway` and not "any service token": a leaked
    driver or library token learns nothing from here. The same narrowing
    M6 applied to starting and stopping a runtime, for the same reason
    -- the set of components that legitimately need a surface is
    usually one, and "any service" is what makes a leak useful.
    """
    owner = registry(request)
    if owner.enrolled:
        data = await owner.forward("GET", "/v1/auth/client-keys/policy")
        return ClientKeyRevocations(
            ids=[key["id"] for key in data["keys"] if key.get("revokedAt")],
            revision=data["revision"],
        )
    try:
        ids, revision = _keys(request).revoked()
    except OSError as exc:
        raise owner.local_unavailable() from exc
    return ClientKeyRevocations(ids=ids, revision=revision)


@router.delete(
    "/v1/auth/client-keys/{key_id}",
    status_code=204,
    dependencies=[Depends(require_operator_session)],
)
async def revoke_client_key(request: Request, key_id: str) -> None:
    """Durably revoke at the authority; repeating a known revocation is a 204."""
    owner = registry(request)
    if owner.enrolled:
        await owner.migrate()
        await owner.forward(
            "DELETE",
            f"/v1/auth/client-keys/{key_id}",
            authorization=request.headers.get("authorization"),
        )
        return
    try:
        record = _keys(request).revoke(key_id)
    except OSError as exc:
        raise owner.local_unavailable() from exc
    if record is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND,
            "No such key",
            f"No client key with id {key_id!r} is registered on this standalone agent.",
        )
    log.info("revoked client key %r (id %s)", record.name, key_id)
