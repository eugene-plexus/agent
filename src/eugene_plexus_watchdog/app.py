"""FastAPI app factory.

The supervisor is wired into the lifespan: at startup the watchdog reads
its topology config and asks the supervisor to spawn every spawned-mode
child; on shutdown it stops them in turn (SIGTERM with timeout, then
SIGKILL). The /v1/components routes layer delegates real-time status
queries to the supervisor so `Component.status` reflects live process
state instead of the skeleton's hard-coded `unreachable`.

v0.2 also seeds `app.state.auth_state` with a fresh JWT signing key on
every startup. The master encryption key starts None and gets populated
either by (a) the OS keyring auto-unlock at lifespan startup when
`securityMode == "os_keyring"`, or (b) on a successful POST
/v1/auth/initialize or /v1/auth/login. When (a) succeeds, children
spawned in this same lifespan get MASTER_KEY in their env immediately
— no Phase-7 restart-on-login churn needed.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from . import __version__, keyring_store, security
from .auth_state import AuthState
from .dependencies import require_operator_session
from .routes import auth as auth_routes
from .routes import components as components_routes
from .routes import config as config_routes
from .routes import health as health_routes
from .routes import runtimes as runtimes_routes
from .runtimes import RuntimeSupervisor, close_installers
from .settings import Settings, load_settings
from .state import WatchdogState
from .supervisor import Supervisor

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    state = WatchdogState(settings.config_file)
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_WATCHDOG_SAFE_MODE=1); ignoring "
            "%s and running on defaults. Fix watchdog state via /v1/config or "
            "/v1/components, then restart without the env var.",
            settings.config_file,
        )
    else:
        state.load()
    app.state.watchdog_state = state
    app.state.safe_mode = settings.safe_mode

    # v0.2 auth state. Tests can pre-populate before the lifespan runs.
    # Production builds a fresh signing key here; rotating at every
    # startup is the v0.2 revocation story (good-enough for one-operator
    # personal-use installs).
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = AuthState(signing_key=security.generate_signing_key())

    # OS keyring auto-unlock — only when the operator opted into it
    # AND a passphrase has been set (so we know which install's key
    # we're recovering). Safe mode and the no-passphrase first-run
    # window skip this so a broken keyring backend can't block the
    # wizard.
    if (
        state.has_passphrase()
        and state.get_config("securityMode") == "os_keyring"
        and not app.state.auth_state.has_master_key()
    ):
        stored = keyring_store.get_master_key()
        if stored is not None:
            app.state.auth_state.set_master_key(stored)
            log.info("master key recovered from OS keyring; children will auto-unlock")
        else:
            log.info(
                "securityMode is os_keyring but no stored key was retrievable; "
                "operator must log in via POST /v1/auth/login to populate it"
            )

    if state.has_passphrase():
        log.info("watchdog initialized; operator may log in via POST /v1/auth/login")
    else:
        log.info(
            "watchdog has no passphrase set; first-run wizard must call "
            "POST /v1/auth/initialize before other endpoints become available",
        )

    # Supervisor injection: tests can pre-populate `app.state.supervisor`
    # with a stub before the lifespan runs (mirroring the gateway's
    # pattern with its driver clients). Production builds the
    # real one here, sharing the AuthState so each spawn can issue a
    # service token, forward the signing key, and (if logged in)
    # forward the master key.
    if not hasattr(app.state, "supervisor"):
        supervisor = Supervisor(log=log, auth_state=app.state.auth_state)
        owns_supervisor = True
    else:
        supervisor = app.state.supervisor
        owns_supervisor = False
    app.state.supervisor = supervisor

    # Engine runtimes get their own supervisor. Same injection pattern,
    # and deliberately a separate object: it owns the readiness polling
    # that only engines have, and nothing about a component's health
    # story applies to a foreign binary.
    if not hasattr(app.state, "runtime_supervisor"):
        runtime_supervisor = RuntimeSupervisor(log=log)
        owns_runtimes = True
    else:
        runtime_supervisor = app.state.runtime_supervisor
        owns_runtimes = False
    app.state.runtime_supervisor = runtime_supervisor

    if not settings.safe_mode and owns_supervisor:
        for entry in state.list_topology_entries():
            supervisor.add_and_start(entry)
        await supervisor.start_health_loop(state.list_topology_entries)

    if not settings.safe_mode and owns_runtimes:
        for spec in state.list_runtime_specs():
            runtime_supervisor.add_and_start(spec)
        await runtime_supervisor.start_readiness_loop(state.list_runtime_specs)

    try:
        yield
    finally:
        # An in-flight engine download is the cheapest thing here to
        # abandon and the only one holding a half-written directory, so
        # it goes first.
        await close_installers()
        # Engines next: they are the ones holding GPU memory, and a
        # driver briefly outliving its engine is harmless while the
        # reverse leaves requests hitting a dead port.
        if owns_runtimes:
            await runtime_supervisor.stop_all()
        if owns_supervisor:
            await supervisor.stop_all()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — watchdog",
        description="Process supervisor and UI host for an Eugene Plexus install.",
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Public routes (no auth required).
    app.include_router(health_routes.router)
    app.include_router(auth_routes.router)

    # v0.2 protected routes — bearer session token required.
    protected_dependencies = [Depends(require_operator_session)]
    app.include_router(config_routes.router, dependencies=protected_dependencies)
    # The components router declares auth per-route, NOT at the router
    # level: the read endpoints accept operator OR service tokens (so
    # peers can auto-resolve topology), while mutations stay operator-
    # only. A blanket router dependency would force operator-only on the
    # GETs too, which is the v0.2.1 bug we're fixing.
    app.include_router(components_routes.router)
    # Same per-route auth split as components, for the same reason: the
    # gateway resolves what is running with a service token, while
    # starting or stopping a process that holds a GPU stays operator-only.
    app.include_router(runtimes_routes.router)

    return app
