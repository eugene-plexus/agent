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
import base64
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI

from . import (
    __version__,
    apps,
    companions,
    default_topology,
    enrollment,
    host_allowlist,
    install_permissions,
    keyring_store,
    node_identity,
    off_host,
    passphrase_file,
    process_signals,
    response_headers,
    security,
    session_revocations,
    share_credentials,
    ui_assets,
)
from ._http import aclose_shared
from .auth_state import AuthState
from .client_key_registry import ClientKeyRegistry
from .client_keys import KEYS_FILE, ClientKeyStore
from .dependencies import require_operator_session
from .library_folders import FOLDERS_FILE, LibraryFolderCache
from .routes import apps as apps_routes
from .routes import auth as auth_routes
from .routes import benchmarks as benchmark_routes
from .routes import components as components_routes
from .routes import config as config_routes
from .routes import directories as directories_routes
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
    from .recovery_guard import refuse_quarantined

    refuse_quarantined(settings.config_file)
    # This machine's FQDN, one of the names the Host allowlist answers to.
    # A reverse lookup that can wait on DNS, so off the loop and not
    # awaited; until it answers, the plain host name stands in.
    host_allowlist.start_learning_fqdn()
    state = AgentState(settings.config_file)
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_AGENT_SAFE_MODE=1); ignoring "
            "%s and running on defaults. Fix agent state via /v1/config or "
            "/v1/components, then restart without the env var.",
            settings.config_file,
        )
        config_error: str | None = None
    else:
        # **Degraded, not dead** (review §6.1 #6). This was a bare
        # `state.load()` into an uncaught lifespan, so a half-written
        # `agent.yaml` was an install that would not start and could not
        # be repaired from the browser -- the exact failure mode
        # `degraded-mode-required` exists to forbid, unapplied to the
        # component that owns the rule's own file. See
        # `AgentState.load_or_degrade` for what a failure preserves.
        config_error = state.load_or_degrade()
    app.state.agent_state = state
    app.state.safe_mode = settings.safe_mode
    app.state.config_error = config_error

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

    # Whether another account on this machine can read node.yaml or add
    # files to this install (Windows; see `install_permissions`). Off the
    # loop and not awaited: naming a domain account can wait on a domain
    # controller, and startup must not.
    app.state.install_permissions = []
    app.state.install_permissions_task = asyncio.create_task(
        _check_install_permissions(app, settings.config_file.resolve().parent)
    )

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

    # The sessions signed out before this start. Outside the `hasattr`
    # above, and in safe mode too: a sign-out that was not read back is
    # a live token on any node whose signing key survived the restart,
    # which is every enrolled one -- and safe mode ignores `agent.yaml`,
    # not the security state beside it.
    app.state.auth_state.revoked.bind(
        settings.config_file.resolve().parent / session_revocations.REVOKED_SESSIONS_FILE
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
        salt_b64 = state.get_master_salt_b64()
        stored = (
            keyring_store.get_master_key(keyring_store.install_id_for(salt_b64))
            if salt_b64
            else None
        )
        if stored is not None:
            app.state.auth_state.set_master_key(stored)
            log.info("master key recovered from OS keyring; children will auto-unlock")
        else:
            log.info(
                "securityMode is os_keyring but no stored key was retrievable; "
                "operator must log in via POST /v1/auth/login to populate it"
            )

    # The keyring's sibling for an agent under its own account, which has
    # no keyring: the Linux system install (2026-09-24). Here for the same
    # reason the keyring is -- children spawned in this lifespan then start
    # with the master key -- and off the loop, because it is two Argon2id
    # runs (verify, then derive).
    if (
        state.has_passphrase()
        and state.get_config("securityMode") == "passphrase_file"
        and not app.state.auth_state.has_master_key()
    ):
        salt_b64 = state.get_master_salt_b64()
        stored_hash = state.get_passphrase_hash()
        if salt_b64 and stored_hash:
            key = await asyncio.to_thread(
                passphrase_file.unlock_key,
                settings.passphrase_file,
                passphrase_hash=stored_hash,
                salt=base64.b64decode(salt_b64),
            )
            if key is not None:
                app.state.auth_state.set_master_key(key)
                log.info("master key recovered from the passphrase file; children will auto-unlock")

    # **Log in to the file servers before anything opens a model.** This
    # is the whole of R2.6's first measurement: an agent running as a
    # Windows service holds none of the credentials the person who
    # installed it typed into Explorer, so a share that opened yesterday
    # answers `WinError 1272` today with every health check still green.
    # It sits here, after the keyring recovery above, because the
    # passwords are sealed with the master key -- on `prompt_on_startup`
    # there is nothing to unseal yet and the same call runs again at
    # login. Off the event loop: an unreachable server blocks for as long
    # as the network stack allows.
    await _connect_configured_shares(app)

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
    # This node's copy of the Library's folder list (2026-09-14), beside
    # agent.yaml. Every request that talks to the library refreshes it;
    # a spawn reads it, so an agent that restarts with the library down
    # still resolves a folder's mount. Tests may inject their own.
    if not hasattr(app.state, "library_folders"):
        folders_cache = LibraryFolderCache(settings.config_file.resolve().parent / FOLDERS_FILE)
        if not settings.safe_mode:
            folders_cache.load()
        app.state.library_folders = folders_cache
    else:
        folders_cache = app.state.library_folders

    # The long-lived keys this node has minted for apps outside the
    # install (S4). Beside agent.yaml, its own file: `AgentState` writes
    # its `auth` block whole, and a growing list does not belong in a
    # document the config trio serves.
    if not hasattr(app.state, "client_keys"):
        key_store = ClientKeyStore(settings.config_file.resolve().parent / KEYS_FILE)
        key_store.load()
        app.state.client_keys = key_store

    if not hasattr(app.state, "runtime_supervisor"):
        # The agent's config is where an install-wide engine path
        # (`vllmBinary`) lives; the supervisor reads it at each spawn.
        runtime_supervisor = RuntimeSupervisor(
            log=log,
            get_config=state.get_config,
            inherited_rules=folders_cache.inherited_rules,
        )
        owns_runtimes = True
    else:
        runtime_supervisor = app.state.runtime_supervisor
        owns_runtimes = False
    app.state.runtime_supervisor = runtime_supervisor
    # `Runtime.node`, from the agent's own identity and nowhere else.
    runtime_supervisor.node_name_provider = lambda: (
        identity.record.name if identity.record.enrolled else None
    )

    # Optional apps (docs/design/apps-and-spokes.md). None in safe mode,
    # which supervises nothing it was not asked to on the command line.
    # `apps.yaml` degrades on its own: a file this build cannot read
    # costs the apps, never the topology above.
    if not hasattr(app.state, "apps"):
        if settings.safe_mode:
            app.state.apps = None
        else:
            app_store = apps.AppStore(settings.config_file.resolve().parent / apps.APPS_FILE)
            app_store.load_or_degrade()
            app.state.apps = apps.AppManager(
                store=app_store,
                catalogue=apps.load_catalogue(),
                get_config=state.get_config,
                bind_host=lambda: shared_child_env(settings, state, identity).get("BIND_HOST"),
                advertise_host=lambda: _app_advertise_host(state, identity),
                node_name=lambda: identity.record.name if identity.record.enrolled else None,
                resolve_gateway=lambda: resolve_gateway_for_apps(app),
            )
    app_manager: apps.AppManager | None = app.state.apps

    # The topology every install has, on the one boot where there isn't
    # one yet. Before supervision starts, so the components below are
    # started by the same loop as any operator-declared entry rather
    # than by a second path that could drift from it.
    # An enrolled node is onboarded by definition. Done here as well as
    # at the moment of enrolling, because an install that joined before
    # this existed is still carrying `firstRunComplete: false` and would
    # otherwise keep offering its operator a wizard that would raise a
    # second install. Costs one config write, once.
    if (
        not settings.safe_mode
        and identity.record.enrolled
        and default_topology.mark_onboarded(state)
    ):
        log.info(
            "this node is enrolled with %s, so first-run setup is complete; recorded it "
            "(the web UI reads that flag and was offering the first-run wizard)",
            identity.record.control_url or "a control root",
        )

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

    # Apps last, and in the background: each start resolves the gateway,
    # which on a worker is a read from the control root, and a root that
    # is down must not hold up supervision of the hub itself.
    apps_task: asyncio.Task[None] | None = None
    if app_manager is not None:
        apps_task = asyncio.create_task(app_manager.start_enabled(), name="apps-boot")

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

    registry = ClientKeyRegistry(app)
    app.state.client_key_registry = registry
    registry_task = asyncio.create_task(registry.run())
    try:
        yield
    finally:
        # Apps first: they are clients of everything below, and a spoke
        # outliving the hub it talks to only produces errors in its log.
        if apps_task is not None and not apps_task.done():
            apps_task.cancel()
            await asyncio.gather(apps_task, return_exceptions=True)
        if app_manager is not None:
            await app_manager.aclose()
        registry_task.cancel()
        await asyncio.gather(registry_task, return_exceptions=True)
        await registry.close()
        benchmarks = getattr(app.state, "benchmarks", None)
        if benchmarks is not None:
            await benchmarks.close()
        if announce_task is not None and not announce_task.done():
            announce_task.cancel()
        permissions_task = app.state.install_permissions_task
        if not permissions_task.done():
            permissions_task.cancel()
            await asyncio.gather(permissions_task, return_exceptions=True)
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
        # Last: the process-wide clients (`_http.shared_internal_client`)
        # -- the engine readiness probe and the library reads admission
        # makes. Last because everything above may still be probing on
        # the way down, and closing a pool under an in-flight probe
        # turns an orderly shutdown into a traceback.
        await aclose_shared()


async def _check_install_permissions(app: FastAPI, config_dir: Path) -> None:
    """Say once, loudly, if another account can read this install's secrets.

    The finding also rides on `/healthz` as `details.installPermissions`,
    so it is visible without the log. Never fatal: a warning about the
    install must not be what stops the install.
    """
    try:
        grants = await asyncio.to_thread(install_permissions.check, config_dir)
    except Exception:  # pragma: no cover - defensive
        log.warning("could not check this install's permissions", exc_info=True)
        return
    app.state.install_permissions = [grant.sentence() for grant in grants]
    for sentence in app.state.install_permissions:
        log.warning(
            "install permissions: %s. Another account on this machine could read this "
            "install's signing key or add code that runs as this agent. Re-run the "
            "installer to repair the folder's permissions.",
            sentence,
        )


async def _connect_configured_shares(app: FastAPI) -> None:
    """Ask the OS to log this host in to every configured file server.

    Never fatal. `degraded-mode-required` applies as it does everywhere
    else here: a server that is off, a password that has rotated or a
    typo in a host name leaves the rest of the install supervising
    normally, and the operator finds out from the model that will not
    load — which already says which path it tried.
    """
    state: AgentState = app.state.agent_state
    auth = app.state.auth_state
    entries = share_credentials.unseal_entries(
        state.get_config("shareCredentials"),
        auth.master_key if auth.has_master_key() else None,
    )
    if not entries:
        return
    try:
        await asyncio.to_thread(share_credentials.connect_all, entries)
    except Exception:  # pragma: no cover - defensive
        log.warning("could not log in to the configured file servers", exc_info=True)


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


def _app_advertise_host(state: AgentState, identity: node_identity.NodeIdentityStore) -> str | None:
    """The host another device opens an app's UI on, or None for loopback."""
    advertise = node_identity.effective_advertise_url(
        state.get_config("advertiseUrl"), identity.record.advertise_url
    )
    host = node_identity.advertise_host(advertise)
    if not host or node_identity.is_loopback_host(host):
        return None
    return host


async def resolve_gateway_for_apps(app: FastAPI) -> tuple[str | None, str | None]:
    """The gateway address an app on this node is handed, or why there is none.

    **This node's own gateway when it runs one.** Otherwise the owning
    node's agent proxy, `<agent>/api/proxy/gateway`: the same public path
    a browser uses, reached at the address that node announced -- which
    survives the container's port remap that a component's own URL does
    not (see `install_proxy`). The app's client key rides through it
    untouched and the gateway checks it; the proxy adds nothing.

    The lookup spends this agent's own `service:agent` token, which the
    control root accepts for reads. There is no operator at a boot.
    """
    from ._generated.models import ComponentKind

    state: AgentState = app.state.agent_state
    for entry in state.list_topology_entries():
        if entry.kind == ComponentKind.gateway and entry.spawn is not None:
            return str(entry.url).rstrip("/"), None
    identity: node_identity.NodeIdentityStore = app.state.node_identity
    if not identity.record.enrolled or not identity.record.control_url:
        return None, (
            "This machine runs no gateway and is not part of an install yet, so there was no "
            "gateway address to give the app."
        )
    from . import install_proxy

    token = security.issue_service_token(signing_key=app.state.auth_state.signing_key, kind="agent")
    cache = getattr(app.state, "install_topology", None)
    if cache is None:
        cache = install_proxy.InstallTopology()
        app.state.install_topology = cache
    try:
        owner = await cache.owner_of(
            "gateway",
            control_url=identity.record.control_url,
            authorization=f"Bearer {token}",
            transport=getattr(app.state, "control_transport", None),
        )
    except install_proxy.InstallLookupError as exc:
        return None, f"The gateway could not be found, so the app was given none: {exc}"
    return owner.agent_url.rstrip("/") + "/api/proxy/gateway", None


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
    # DNS rebinding: answer only to names this machine could really be
    # opened by, on every route -- UI, API and proxy alike. Added before
    # the witness below, so it sits INSIDE it: a phone that opened a
    # name we refuse still reached this machine, which is what the
    # witness records.
    host_allowlist.install(app)
    # The only evidence about reach that comes from outside this machine:
    # somebody's phone, or another node, actually connected. Pure ASGI
    # and scope-only, so the streaming proxy below is untouched.
    app.state.off_host = off_host.OffHostWitness()
    off_host.install(app, app.state.off_host)
    # `nosniff` and `no-referrer` on every response. Added last, so it is
    # the outermost layer and covers the two above's own answers too --
    # a refused host name is still a page a browser renders.
    response_headers.install(app)

    # Public routes (no auth required).
    app.include_router(health_routes.router)
    app.include_router(auth_routes.router)

    # v0.2 protected routes — bearer session token required.
    protected_dependencies = [Depends(require_operator_session)]
    app.include_router(config_routes.router, dependencies=protected_dependencies)
    # The picker behind every path field (M11). Declares operator-only on
    # its one route; it lists what the operator could already type.
    app.include_router(directories_routes.router)
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
    app.include_router(benchmark_routes.router)
    # Optional apps. Operator-only on every route; declared on the router.
    app.include_router(apps_routes.router)
    # This host's identity and devices; reads only, operator or service.
    app.include_router(node_routes.router)

    # The browser surface, registered LAST and in this order. The proxy
    # is deliberately unauthenticated — it is the path the login request
    # itself travels — and the static mount answers everything not
    # matched above, so anything registered after it is unreachable.
    app.include_router(proxy_routes.router)
    ui_assets.mount(app, settings.ui_dir)

    return app
