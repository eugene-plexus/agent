"""FastAPI app factory.

The supervisor is wired into the lifespan: at startup the agent reads
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

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from . import (
    __version__,
    companions,
    default_topology,
    enrollment,
    keyring_store,
    node_identity,
    process_signals,
    security,
    ui_assets,
)
from .auth_state import AuthState
from .dependencies import require_operator_session
from .routes import auth as auth_routes
from .routes import components as components_routes
from .routes import config as config_routes
from .routes import health as health_routes
from .routes import node as node_routes
from .routes import proxy as proxy_routes
from .routes import runtimes as runtimes_routes
from .runtimes import RuntimeSupervisor, close_installers
from .settings import Settings, load_settings
from .state import AgentState
from .supervisor import Supervisor

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    state = AgentState(settings.config_file)
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_AGENT_SAFE_MODE=1); ignoring "
            "%s and running on defaults. Fix agent state via /v1/config or "
            "/v1/components, then restart without the env var.",
            settings.config_file,
        )
    else:
        state.load()
    app.state.agent_state = state
    app.state.safe_mode = settings.safe_mode

    # This host's identity in the install (M7): node.yaml beside
    # agent.yaml. Loaded before auth state, because an enrolled agent
    # verifies tokens with THE INSTALL'S signing key from the first
    # request and hands that key to every child it spawns — a restart
    # is not a re-key. An agent that has not enrolled mints a random
    # per-restart key, which is the single-host behaviour unchanged.
    identity = node_identity.NodeIdentityStore(
        settings.config_file.resolve().parent / node_identity.NODE_FILE
    )
    try:
        identity.load()
    except ValueError as exc:
        # Degraded, not dead: supervision needs no identity. The agent
        # comes up unenrolled and says why.
        log.error("node identity file could not be read (%s); running unenrolled", exc)
    app.state.node_identity = identity

    # Tests can pre-populate auth state before the lifespan runs.
    if not hasattr(app.state, "auth_state"):
        install_key = identity.record.signing_key_bytes
        app.state.auth_state = AuthState(signing_key=install_key or security.generate_signing_key())
        if install_key is not None:
            log.info(
                "verifying tokens with the install's signing key (generation %s) as node %r",
                identity.record.signing_key_id,
                identity.record.name,
            )

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
        log.info("agent initialized; operator may log in via POST /v1/auth/login")
    else:
        log.info(
            "agent has no passphrase set; first-run wizard must call "
            "POST /v1/auth/initialize before other endpoints become available",
        )

    # **How this host stops things, said at boot rather than discovered.**
    # On Windows without a console — which is what a service is — the
    # graceful path is unavailable and every child gets a hard kill. That
    # limitation is accepted rather than worked around (see
    # `process_signals.describe_stop_capability`), and an accepted
    # limitation has to be visible or it is indistinguishable from a bug.
    graceful, why = process_signals.describe_stop_capability()
    app.state.graceful_stop = graceful
    (log.info if graceful else log.warning)("child shutdown: %s", why)

    # Supervisor injection: tests can pre-populate `app.state.supervisor`
    # with a stub before the lifespan runs (mirroring the gateway's
    # pattern with its driver clients). Production builds the
    # real one here, sharing the AuthState so each spawn can issue a
    # service token, forward the signing key, and (if logged in)
    # forward the master key.
    if not hasattr(app.state, "supervisor"):
        supervisor = Supervisor(
            log=log,
            auth_state=app.state.auth_state,
            shared_child_env=lambda: shared_child_env(settings, state, identity),
        )
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
        # The agent's config is where an install-wide engine path
        # (`vllmBinary`) lives; the supervisor reads it at each spawn.
        runtime_supervisor = RuntimeSupervisor(log=log, get_config=state.get_config)
        owns_runtimes = True
    else:
        runtime_supervisor = app.state.runtime_supervisor
        owns_runtimes = False
    app.state.runtime_supervisor = runtime_supervisor
    # `Runtime.node`, from the agent's own identity and nowhere else.
    runtime_supervisor.node_name_provider = lambda: (
        identity.record.name if identity.record.enrolled else None
    )

    # The topology every install has, on the one boot where there isn't
    # one yet. Before supervision starts, so the components below are
    # started by the same loop as any operator-declared entry rather
    # than by a second path that could drift from it.
    if (
        not settings.safe_mode
        and settings.default_topology
        and default_topology.should_seed(state, enrolled=identity.record.enrolled)
    ):
        declared = default_topology.seed(state)
        if declared:
            log.info(
                "first boot: declared the default topology (%s). "
                "Edit or add to it from the UI, or set "
                "EUGENE_PLEXUS_AGENT_DEFAULT_TOPOLOGY=0 on a node that will enroll.",
                ", ".join(declared),
            )

    if not settings.safe_mode and owns_supervisor:
        for entry in state.list_topology_entries():
            supervisor.add_and_start(entry)
        await supervisor.start_health_loop(state.list_topology_entries)

    if not settings.safe_mode and owns_runtimes:
        # Companions first, so a runtime that has one gets it whether the
        # install predates M6 or an operator deleted the driver by hand.
        # `autoDriver: true` means the agent keeps one, and boot is where
        # that promise is made good.
        created = await companions.reconcile(state, supervisor)
        if created:
            log.info("declared %d companion driver(s) at boot: %s", len(created), created)
        for spec in state.list_runtime_specs():
            runtime_supervisor.add_and_start(spec)
        await runtime_supervisor.start_readiness_loop(state.list_runtime_specs)

    # **Where this host is, said out loud on every boot.** Before M9 the
    # address was announced once, at enrollment, so a host that rebooted
    # onto a new tailnet IP left the control root holding an address
    # nobody was listening on — and the root could not poll its way out,
    # because the only address it had was the stale one. A background
    # task, not an await: a management-plane call must not hold up
    # supervision, and an unreachable root is a normal state here.
    announce_task: asyncio.Task[None] | None = None
    if not settings.safe_mode and identity.record.enrolled:
        announce_task = asyncio.create_task(_announce_address(app, settings, state, identity))

    try:
        yield
    finally:
        if announce_task is not None and not announce_task.done():
            announce_task.cancel()
        # The browser's upstream connections. Closed first because it is
        # the only thing here holding sockets to processes the next two
        # steps are about to stop.
        proxy_client = getattr(app.state, "ui_proxy_client", None)
        if proxy_client is not None:
            await proxy_client.aclose()
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


async def _announce_address(
    app: FastAPI,
    settings: Settings,
    state: AgentState,
    identity: node_identity.NodeIdentityStore,
) -> None:
    """Re-derive this node's address and tell the control root.

    **Re-derived rather than read back**, which is the whole point: a
    host that rebooted onto a new address has a *stale* persisted value,
    so trusting it would announce the old one. An operator-configured
    `advertiseUrl` still wins, and a root that cannot be reached leaves
    the persisted value alone.
    """
    try:
        url = await enrollment.resolve_advertise_url(
            configured=state.get_config("advertiseUrl"),
            control_url=identity.record.control_url,
            bind_port=int(settings.bind_port),
            persisted=identity.record.advertise_url,
        )
        if url is None:
            log.warning(
                "enrolled with %s but this node has no address to advertise; other hosts "
                "cannot reach it. Set `advertiseUrl` in the agent config.",
                identity.record.control_url,
            )
            return
        identity.record_advertise_url(url)
        await enrollment.announce_address(
            store=identity,
            url=url,
            transport=getattr(app.state, "control_transport", None),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("could not announce this node's address: %s", exc)


def shared_child_env(
    settings: Settings, state: AgentState, identity: node_identity.NodeIdentityStore
) -> dict[str, str]:
    """Env values (suffix -> value) every spawned component gets, prefixed
    per kind by the planner.

    `AGENT_URL` is this agent's own *local* address: a companion's agent
    is always on the same host, and the loopback default was only ever
    wrong about the port. `BIND_HOST` is `0.0.0.0` only when this node
    advertises a non-loopback address — a component that must be
    reached from another host cannot bind only to this one — and is
    otherwise left to each component's own loopback default. Engines are
    not components and are never widened."""
    env = {
        "AGENT_URL": node_identity.local_agent_url(settings.bind_host, int(settings.bind_port)),
    }
    advertise = node_identity.effective_advertise_url(
        state.get_config("advertiseUrl"), identity.record.advertise_url
    )
    if not node_identity.is_loopback_host(node_identity.advertise_host(advertise)):
        env["BIND_HOST"] = "0.0.0.0"
    return env


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — agent",
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
    # starting or stopping a process that holds a GPU is operator-only
    # — or, from M6, the gateway's own token, because it is the one
    # component that sees demand.
    app.include_router(runtimes_routes.router)
    # This host's identity and devices; reads only, operator or service.
    app.include_router(node_routes.router)

    # The browser surface, registered LAST and in this order. The proxy
    # is deliberately unauthenticated — it is the path the login request
    # itself travels — and the static mount answers everything not
    # matched above, so anything registered after it is unreachable.
    app.include_router(proxy_routes.router)
    ui_assets.mount(app, settings.ui_dir)

    return app
