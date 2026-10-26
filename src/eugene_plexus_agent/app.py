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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI

from . import (
    __version__,
    apps,
    companions,
    default_topology,
    enrollment,
    firewall_follow,
    host_allowlist,
    install_info,
    install_permissions,
    keyring_store,
    node_identity,
    off_host,
    open_requests,
    passphrase_file,
    process_signals,
    response_headers,
    session_revocations,
    share_credentials,
    standby,
    ui_assets,
    update_all,
)
from ._generated.models import RuntimeSpec
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
from .routes import entrypoint as entrypoint_routes
from .routes import health as health_routes
from .routes import log_ingress as log_ingress_routes
from .routes import logs as logs_routes
from .routes import node as node_routes
from .routes import oidc_forward as oidc_forward_routes
from .routes import people as people_routes
from .routes import profile_builds as profile_build_routes
from .routes import proxy as proxy_routes
from .routes import runtimes as runtimes_routes
from .routes import serving_measurements as serving_measurement_routes
from .routes import site_link as site_link_routes
from .routes import updates as update_routes
from .runtimes import RuntimeSupervisor, close_installers
from .settings import Settings, load_settings
from .state import AgentState
from .supervisor import Supervisor
from .trust import BUNDLE_FILE, NodeTrust

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
    # Strata's install says when this node's copy folder is on a slow drive
    # (LS7): its installer is one per engine, so it learns the config once.
    from ._generated.models import EngineKind as _EngineKind
    from .runtimes import installer_for as _installer_for

    _installer_for(_EngineKind.strata, state.get_config)
    app.state.config_error = config_error

    # This host's identity in the install (M7): node.yaml beside
    # agent.yaml. Loaded before auth state, because this node signs with
    # its own token key from `node.yaml` and verifies against the trust
    # bundle kept beside it from the first request (2026-09-25). An agent
    # that has not enrolled is its own authority, with the same key.
    identity = node_identity.NodeIdentityStore(
        settings.config_file.resolve().parent / node_identity.NODE_FILE
    )
    try:
        identity.load()
    except ValueError as exc:
        # Degraded, not dead: supervision needs no identity. The agent
        # comes up unenrolled and says why.
        log.error("node identity file could not be read (%s); running unenrolled", exc)
    try:
        # The three keys, made once and kept: an unenrolled node needs its
        # token key and identity key as much as an enrolled one does.
        identity.ensure_keypair()
    except OSError as exc:
        log.error("could not write this node's keys to %s: %s", identity.path, exc)
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
        trust = NodeTrust(identity, settings.config_file.resolve().parent / BUNDLE_FILE)
        try:
            trust.load()
        except (OSError, RuntimeError) as exc:
            log.error("could not load this node's trust: %s", exc)
        app.state.auth_state = AuthState(trust=trust)
        log.info(
            "signing as %s; trusting %s",
            trust.recipient,
            "the control root's bundle" if trust.enrolled else "this node alone",
        )

    # What this node holds about the other machines' TLS doors (LT1): the
    # root's signed list kept on disk, re-verified, and the table every
    # client here and every child uses, from before the first request.
    from .machine_trust import MachineTrust

    machine_trust = MachineTrust(settings.config_file.resolve().parent, identity)
    # This machine under the address the others know it by: its own requests
    # to it are not a crossing to another machine.
    machine_trust.own_urls = lambda: (
        node_identity.effective_advertise_url(
            state.get_config("advertiseUrl"), identity.record.advertise_url
        ),
    )
    machine_trust.load()
    app.state.machine_trust = machine_trust
    # Row 4: once this node follows a promoted root to its address, it
    # announces itself there and pulls the bundle, beside whatever took the
    # list (a push or a pull), which must not wait for it.
    app.state.root_moved_task = None

    def _root_moved() -> None:
        from . import leaving

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        app.state.root_moved_task = loop.create_task(
            leaving.after_root_moved(app), name="root-moved"
        )

    machine_trust.on_moved = _root_moved

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
        and state.get_master_salt_b64()
    ):
        # Row 5: `master_key_for` unwraps a key wrapped after a change.
        key = await asyncio.to_thread(passphrase_file.unlock_key, settings.passphrase_file, state)
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
            shared_child_env=lambda kind=None: {
                **shared_child_env(
                    settings,
                    state,
                    identity,
                    kind,
                    door_up=lambda: getattr(app.state, "machine_port", None) is not None,
                ),
                # LT2: a gateway behind its door listens on loopback.
                **(
                    app.state.gateway_door.child_env()
                    if kind == "gateway" and getattr(app.state, "gateway_door", None)
                    else {}
                ),
            },
            standby_env=lambda: standby.spawn_env(app),
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
            library_roots=lambda: [folder.path for folder in folders_cache.folders or []],
            stop_memory=state,
        )
        owns_runtimes = True
    else:
        runtime_supervisor = app.state.runtime_supervisor
        owns_runtimes = False
    app.state.runtime_supervisor = runtime_supervisor
    if owns_runtimes:
        from fastapi import HTTPException

        from .card_launch import CardLaunch, DeviceMemory
        from .routes.runtimes import _measure_launch
        from .runtime_context import NodeContext

        runtime_supervisor.device_memory = DeviceMemory(state.path.parent / "runtime-devices.json")

        async def check_runtime_cards(spec: RuntimeSpec) -> CardLaunch:
            context = NodeContext(app=app)
            try:
                admission, _ = await _measure_launch(context, spec)
            except HTTPException as exc:
                detail = exc.detail
                raise ValueError(
                    detail.get("detail", str(detail)) if isinstance(detail, dict) else str(detail)
                ) from exc
            if (
                admission.decision.value == "refuse"
                and spec.name not in runtime_supervisor.force_launches
            ):
                raise ValueError(admission.reason)
            launch: CardLaunch = context.state.card_launch
            return launch

        runtime_supervisor.launch_check = check_runtime_cards
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
                # The container explicitly binds the agent to all interfaces;
                # apps must honor that even before a LAN address is advertised.
                bind_host=lambda: shared_child_env(settings, state, identity).get(
                    "BIND_HOST", settings.bind_host
                ),
                advertise_host=lambda: _app_advertise_host(state, identity),
                node_name=lambda: identity.record.name if identity.record.enrolled else None,
                resolve_gateway=lambda: resolve_gateway_for_apps(app),
                # C1: where an app's launcher sends what the app prints.
                ingress_url=lambda: f"http://127.0.0.1:{int(settings.bind_port)}/v1/logs",
                # C2: where an app on this machine signs people in -- this
                # agent's /oidc, at the address a browser opens the app at.
                oidc_issuer=lambda: (
                    _oidc_issuer(_app_advertise_host(state, identity), int(settings.bind_port))
                    if app.state.entrypoint_config is None
                    else (app.state.entrypoint_config.console.origin + "/oidc")
                ),
                public_origin=lambda app_id: (
                    app.state.entrypoint_config.workbench.origin
                    if app.state.entrypoint_config and app_id == "workbench"
                    else None
                ),
                oidc_backchannel=(
                    f"http://127.0.0.1:{int(settings.bind_port)}/oidc"
                    if app.state.entrypoint_config
                    else None
                ),
                private_apps=bool(app.state.entrypoint_config),
                # LT2: each app behind its door, once the doors are open, and
                # given the install CA.
                doors=lambda: (
                    getattr(app.state, "client_doors", None)
                    if settings.client_doors_enabled
                    else None
                ),
                ca_bundle=lambda: (
                    tls_.bundle_path()
                    if (tls_ := getattr(app.state, "client_tls", None)) is not None
                    else None
                ),
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
    # Row 53: every install has a store, on the root. A root set up before
    # it existed gets one here, once; one the owner removed or moved stays
    # where they put it (`default_topology.add_store`).
    if not settings.safe_mode and settings.default_topology:
        added = default_topology.add_store(state)
        if added:
            log.info(
                "declared the store on this root (%s): it keeps what apps upload through the "
                "gateway's /v1/files. Settings -> Store changes what it keeps.",
                added,
            )

    # The machine port (LT1): on the root, control leaves 8083 first, so the
    # port is free before anything binds it. Before the components start, so
    # control is told it is behind the port and binds loopback.
    from . import machine_port

    if not settings.safe_mode and settings.machine_port_enabled:
        machine_port.move_control_behind(state, int(settings.machine_port))
    door = None if settings.safe_mode else await machine_port.start(app)
    # Row 3: a change of this machine's key, resumed from `node.yaml`; on the
    # root, the door file control reads the machine port's keys from, written
    # before control starts.
    from . import machine_key_change

    key_change = machine_key_change.from_app(app) if door is not None else None
    app.state.machine_key_change = key_change
    if key_change is not None:
        key_change.write_door()
    # The client doors (LT2): the leaf they present, and the gateway's door,
    # before the gateway starts and is told its loopback port.
    from . import client_tls

    tls = None if settings.safe_mode else await client_tls.start(app)
    tls_task = asyncio.create_task(tls.run(), name="client-tls") if tls is not None else None
    if tls is not None:

        def _new_ca() -> None:
            # Apps read the CA at start: the ones running get it now, once
            # in an install's life (an install updated to LT2).
            manager = getattr(app.state, "apps", None)
            if manager is None:
                return

            async def restart() -> None:
                for record in manager.store.installed():
                    if record.enabled and manager.supervisor.is_running(record.id):
                        log.info("restarting app %s so it trusts the install CA", record.id)
                        await manager.restart(record)

            asyncio.get_running_loop().create_task(restart())

        tls.on_new_ca = _new_ca

    def _rebind_on_required(required: bool) -> None:
        # Required moves the drivers to loopback (LT1b; control is there
        # already, behind the machine port); the bind
        # is fixed when a child starts, so those restart.
        kinds = {"inference-driver", "tool-driver"}
        names = [
            e.name
            for e in state.list_topology_entries()
            if e.spawn is not None and e.kind.value in kinds
        ]
        log.warning(
            "encryption between machines is %s now: restarting %s so %s",
            "required" if required else "no longer required",
            ", ".join(names) or "nothing",
            "they listen on this machine only" if required else "other machines reach them again",
        )
        for name in names:
            asyncio.get_running_loop().create_task(supervisor.restart(name))

    machine_trust.on_required_changed = _rebind_on_required

    if not settings.safe_mode and owns_supervisor:
        for entry in state.list_topology_entries():
            supervisor.add_and_start(entry)
        await supervisor.start_health_loop(state.list_topology_entries)
        # The bundle this node holds decides whether it runs the standby,
        # before the first pull: a grant removed while it was off goes now.
        await standby.reconcile(app)

    if not settings.safe_mode and owns_runtimes:
        # Companions first, so a runtime that has one gets it whether the
        # install predates M6 or an operator deleted the driver by hand.
        # `autoDriver: true` means the agent keeps one, and boot is where
        # that promise is made good.
        created = await companions.reconcile(state, supervisor)
        if created:
            log.info("declared %d companion driver(s) at boot: %s", len(created), created)
        for spec in state.list_runtime_specs():
            runtime_supervisor.start_at_boot(spec)
        await runtime_supervisor.start_readiness_loop(state.list_runtime_specs)

    # Host history (specs docs/design/host-history.md): each agent records
    # its own host. The store is opened by the first sample, so a file that
    # cannot be opened costs only this, with the cause on the endpoint.
    from . import host_history, host_probes
    from .engines.devices import detect_devices

    history_store = host_history.HostHistoryStore(
        settings.config_file.resolve().parent / host_history.FILE_NAME
    )
    app.state.host_history = history_store

    def _history_probe() -> host_probes.HostProbe:
        def counts() -> host_probes.RuntimeCounts:
            ready, loading, pids = runtime_supervisor.host_counts()
            return host_probes.RuntimeCounts(running=ready, loading=loading, pids=pids)

        return host_probes.HostProbe(
            devices=getattr(app.state, "device_detector", None) or detect_devices,
            runtimes=counts,
            folders=lambda: host_history.folders_of(app.state),
        )

    history_task = (
        None
        if settings.safe_mode
        else asyncio.create_task(
            host_history.HostSampler(history_store, _history_probe, state.get_config).run_forever(),
            name="host-history",
        )
    )

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
    # supervision, and an unreachable root is a normal state here. Since
    # agent#8 it keeps trying until the root has heard, and says so again
    # when the address moves while running.
    announce_task: asyncio.Task[None] | None = None
    if not settings.safe_mode:
        announce_task = asyncio.create_task(
            _announce_address(app, settings, state, identity), name="address-announce"
        )
    key_change_task = (
        asyncio.create_task(key_change.run(), name="machine-key-change")
        if key_change is not None
        else None
    )

    registry = ClientKeyRegistry(app)
    app.state.client_key_registry = registry
    # The trust bundle, pulled every minute: how a node that was down
    # during a revocation, a sign-out or a root-key rotation catches up by
    # itself. The push is the fast path; this is the one that cannot miss.
    bundle_task = asyncio.create_task(_pull_trust_bundle(app), name="trust-bundle-pull")
    # Is this install behind its channel? A minute after boot and every six
    # hours after, never on the request path: GET /v1/node reads the last
    # result. See specs docs/design/in-app-updates.md.
    update_task: asyncio.Task[None] | None = None
    if not settings.safe_mode:
        update_task = asyncio.create_task(
            update_routes.checker_for(app).run_forever(install_info.describe), name="update-check"
        )
    from . import ollama_report
    from .run_worker import RunWorker

    ollama_task = (
        None
        if settings.safe_mode
        else asyncio.create_task(ollama_report.run(app), name="ollama-report")
    )

    run_task = (
        None
        if settings.safe_mode
        else asyncio.create_task(RunWorker(app).run_forever(), name="run-operations")
    )
    from .site_host import SiteHostSupervisor

    # A job site on this machine is its own enrollment (J19, J23): the agent
    # installs and keeps its host, and relays nothing for it.
    app.state.site_host = SiteHostSupervisor(app)
    helper_task = (
        None
        if settings.safe_mode
        else asyncio.create_task(app.state.site_host.run(), name="site-host")
    )
    from .entrypoint import EntryPoint

    entry_task = None
    if app.state.entrypoint_config:
        entry_task = asyncio.create_task(
            EntryPoint(
                app,
                app.state.entrypoint_config,
                settings.config_file.resolve().parent / "entrypoint",
                app.state.entrypoint_token,
            ).run(),
            name="https-entrypoint",
        )
    from . import entrypoint_setup, sign_in_refresh

    # A configuration applied from Settings goes back unless an operator
    # signs in through it in time; and every app's sign-in return address
    # follows where it is opened now (2026-10-05).
    approval_task = (
        asyncio.create_task(entrypoint_setup.watch_approval(app), name="entrypoint-approval")
        if entry_task is not None and getattr(app.state, "entrypoint_confirm_by", None)
        else None
    )
    sign_in_task = (
        None
        if settings.safe_mode
        else asyncio.create_task(sign_in_refresh.run(app), name="sign-in-addresses")
    )
    # This install's own firewall rule follows its listeners (2026-10-10).
    firewall_task = (
        None
        if settings.safe_mode
        else asyncio.create_task(
            firewall_follow.run(lambda: node_routes.listeners_of(app)), name="firewall-follow"
        )
    )
    try:
        if not settings.safe_mode:
            update_all.manager(app).resume()
        yield
    finally:
        all_updates = getattr(app.state, "update_all", None)
        if all_updates is not None:
            await all_updates.close()
        for task in (approval_task, sign_in_task, firewall_task, history_task, ollama_task):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if entry_task is not None:
            entry_task.cancel()
            await asyncio.gather(entry_task, return_exceptions=True)
        if helper_task is not None:
            helper_task.cancel()
            await asyncio.gather(helper_task, return_exceptions=True)
        if run_task is not None:
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
        bundle_task.cancel()
        await asyncio.gather(bundle_task, return_exceptions=True)
        if update_task is not None:
            update_task.cancel()
            await asyncio.gather(update_task, return_exceptions=True)
        # Apps first: they are clients of everything below, and a spoke
        # outliving the hub it talks to only produces errors in its log.
        if apps_task is not None and not apps_task.done():
            apps_task.cancel()
            await asyncio.gather(apps_task, return_exceptions=True)
        if app_manager is not None:
            await app_manager.aclose()
        serving_measurements = getattr(app.state, "serving_measurements", None)
        if serving_measurements is not None:
            await serving_measurements.close()
        await registry.close()
        benchmarks = getattr(app.state, "benchmarks", None)
        if benchmarks is not None:
            await benchmarks.close()
        profile_builds = getattr(app.state, "profile_builds", None)
        if profile_builds is not None:
            await profile_builds.close()
        if announce_task is not None and not announce_task.done():
            announce_task.cancel()
            await asyncio.gather(announce_task, return_exceptions=True)
        moving = getattr(app.state, "root_moved_task", None)
        if moving is not None and not moving.done():
            moving.cancel()
            await asyncio.gather(moving, return_exceptions=True)
        if key_change_task is not None:
            key_change_task.cancel()
            await asyncio.gather(key_change_task, return_exceptions=True)
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
        if door is not None:
            await door.stop()
        if tls_task is not None:
            tls_task.cancel()
            await asyncio.gather(tls_task, return_exceptions=True)
        client_doors_ = getattr(app.state, "client_doors", None)
        if client_doors_ is not None:
            await client_doors_.close_all()
            if client_doors_.counts is not None:
                client_doors_.counts.save()
        # Last: the process-wide clients (`_http.shared_internal_client`)
        # -- the engine readiness probe and the library reads admission
        # makes. Last because everything above may still be probing on
        # the way down, and closing a pool under an in-flight probe
        # turns an orderly shutdown into a traceback.
        await aclose_shared()


TRUST_PULL_INTERVAL_SECONDS = 60.0


async def _pull_trust_bundle(app: FastAPI) -> None:
    """Ask the control root for its bundle every minute, while enrolled.

    Never raises and never stops: a root that is down is a normal state,
    and the bundle this node holds keeps working meanwhile (it has no
    hard expiry, by design; its age is on `GET /v1/node`). A bundle that
    no longer lists this machine makes it leave the install (agent#21,
    `leaving`).
    """
    from . import leaving
    from ._http import internal_client

    transport = getattr(app.state, "control_transport", None)
    async with internal_client(timeout=leaving.PULL_TIMEOUT_SECONDS, transport=transport) as client:
        while True:
            await leaving.pull_once(app, client)
            await asyncio.sleep(TRUST_PULL_INTERVAL_SECONDS)


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
            "node's own keys or add code that runs as this agent. Re-run the "
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
    """Re-derive this node's address and keep the control root told
    (`enrollment.keep_address_announced`).

    **Re-derived rather than read back**, which is the whole point: a
    host that rebooted onto a new address has a *stale* persisted value,
    so trusting it would announce the old one. An operator-configured
    `advertiseUrl` still wins, and a root that cannot be reached leaves
    the persisted value alone.
    """

    async def resolve(quiet: bool) -> str | None:
        return await enrollment.resolve_advertise_url(
            configured=state.get_config("advertiseUrl"),
            control_url=identity.record.control_url,
            bind_port=int(settings.bind_port),
            persisted=identity.record.advertise_url,
            quiet=quiet,
        )

    try:
        from . import machine_port

        trust = app.state.machine_trust
        await enrollment.keep_address_announced(
            store=identity,
            resolve=resolve,
            transport=getattr(app.state, "control_transport", None),
            machine=lambda: machine_port.announced(app),
            recorded=trust.own_record,
            old_root=lambda: trust.root_has_list is False,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("stopped announcing this node's address: %s", exc)


def _oidc_issuer(host: str | None, port: int) -> str:
    """This agent's /oidc as a browser reaches it (C2): the advertised host,
    else loopback, bracketed when it is an IPv6 literal."""
    host = host or "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}/oidc"


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

    token = app.state.auth_state.trust.agent_token("control")
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
    return await _over_tls(app, owner.agent_url.rstrip("/")) + "/api/proxy/gateway", None


async def _over_tls(app: FastAPI, agent_url: str) -> str:
    """Another machine's console, over `https://` when its door answers TLS
    with a certificate from the install CA (LT2): an app's prompts to a
    gateway on another machine then cross the network encrypted, and keep
    working once plain HTTP is off. Plain when this node holds no CA yet or
    the other machine runs a version without client TLS."""
    from urllib.parse import urlsplit

    import httpx

    from ._http import internal_client

    tls = getattr(app.state, "client_tls", None)
    parts = urlsplit(agent_url)
    if tls is None or tls.ca_pem() is None or parts.scheme != "http":
        return agent_url
    secure = parts._replace(scheme="https").geturl()
    trust = getattr(app.state, "machine_trust", None)
    if trust is not None and trust.required:
        # Every machine runs a version with a TLS door (the install requires
        # encryption between machines): no probe, so nobody in the middle can
        # turn an app's traffic plain by blocking TLS.
        return secure
    try:
        async with internal_client(timeout=5.0) as client:
            answer = await client.get(secure + "/healthz")
        if answer.status_code == 200:
            return secure
    except httpx.HTTPError as exc:
        log.info("apps reach %s in plain HTTP: its TLS door did not answer (%s)", agent_url, exc)
    return agent_url


#: Component kinds the root's gateway reaches on another machine directly.
_DIALLED_BY_THE_ROOT = frozenset({"inference-driver", "tool-driver"})


#: The machine table this agent writes for its children (`machine_trust`).
MACHINE_PINS_FILE = "machine_pins.json"


def _machine_tls_required(identity: node_identity.NodeIdentityStore) -> bool:
    """The root's signed list, as last taken, says encryption is required."""
    from .machine_trust import LIST_FILE, MachineListError, verify

    record = identity.record
    if not record.enrolled or not record.control_public_key:
        return False
    try:
        import json

        kept = json.loads((identity.path.parent / LIST_FILE).read_text(encoding="utf-8"))
        return bool(verify(str(kept["jws"]), record.control_public_key).get("required"))
    except (OSError, ValueError, KeyError, MachineListError):
        return False


def _hosts_control(state: AgentState) -> bool:
    """This agent supervises the control root: it is the control host. A
    warm standby it runs does not count (warm-standby.md)."""
    from .standby import hosts_active_control

    return hosts_active_control(state)


def shared_child_env(
    settings: Settings,
    state: AgentState,
    identity: node_identity.NodeIdentityStore,
    kind: str | None = None,
    door_up: Callable[[], bool] | None = None,
) -> dict[str, str]:
    """`_base_child_env`, with the machine port's part (LT1).

    Every child reads the machine table this agent writes from the root's
    signed list. Control is behind the machine port, so it listens on
    loopback and is handed the port's key, which it signs into the list
    and puts on join tokens. Once the install requires encryption (LT1b),
    the drivers listen on loopback too: other machines reach them through
    the machine port only.
    """
    env = _base_child_env(settings, state, identity, kind)
    env["MACHINE_PINS_FILE"] = str(identity.path.parent / MACHINE_PINS_FILE)
    # Only behind a machine port that started: one that did not (8083 held
    # by another process, an unreadable key) leaves control where it was.
    if settings.machine_port_enabled and (door_up is None or door_up()):
        if kind == "control":
            if identity.record.tls_certificate:
                env["MACHINE_PINS"] = node_identity.machine_pin(identity.record.tls_certificate)
            # Row 3: the keys again, from a file this agent rewrites while
            # the machine key changes; control reads it while it runs.
            # `MACHINE_PINS` stays for a control older than row 3.
            from .machine_key_change import DOOR_FILE

            env["MACHINE_DOOR_FILE"] = str(identity.path.parent / DOOR_FILE)
            env["BIND_HOST"] = "127.0.0.1"
        elif kind in _DIALLED_BY_THE_ROOT and _machine_tls_required(identity):
            env["BIND_HOST"] = "127.0.0.1"
    return env


def _base_child_env(
    settings: Settings,
    state: AgentState,
    identity: node_identity.NodeIdentityStore,
    kind: str | None = None,
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
    if settings.entrypoint_config:
        if settings._entrypoint_console_origin:
            env["AGENT_PUBLIC_ORIGIN"] = settings._entrypoint_console_origin
        # **The control root keeps its direct port unless the entry point
        # serves the node hostname** (2026-10-05). Enrolled machines know the
        # root by that address and it is all they use; moving it behind the
        # entry point meant editing every machine's node.yaml by hand. Its
        # port takes signed and authenticated calls only, as it did before.
        if kind == "control" and settings._entrypoint_nodes_origin:
            # Job Sites (remote-nodes.md §3.1): invitations name this
            # origin, and the root signs the TLS key it presents (J7a).
            env["NODES_ORIGIN"] = settings._entrypoint_nodes_origin
            if settings._entrypoint_nodes_probe:
                env["NODES_PROBE"] = settings._entrypoint_nodes_probe
        # **An enrolled worker's drivers keep their direct bind too**
        # (agent#9): the root's gateway dials each inference-driver and
        # tool-driver on another machine at its advertised address, so
        # loopback took every model and search account here out of
        # routing. On the control host its own gateway reaches them here.
        direct = (
            kind in _DIALLED_BY_THE_ROOT and identity.record.enrolled and not _hosts_control(state)
        )
        if not (kind == "control" and not settings._entrypoint_nodes) and not direct:
            env["BIND_HOST"] = "127.0.0.1"
            return env
    advertise = node_identity.effective_advertise_url(
        state.get_config("advertiseUrl"), identity.record.advertise_url
    )
    if not node_identity.is_loopback_host(node_identity.advertise_host(advertise)):
        env["BIND_HOST"] = "0.0.0.0"
    return env


def _child_bind_host(
    settings: Settings,
    state: AgentState,
    identity: node_identity.NodeIdentityStore,
    kind: str,
) -> str:
    """The interface a child of `kind` would bind on its own: what the agent
    tells it, else the `<PREFIX>_BIND_HOST` it inherits (the image sets one
    per kind), else its loopback default. A port the agent now opens in
    front of that child (the gateway's door, the machine port) binds here."""
    import os

    from .supervisor import _COMPONENT_SPECS

    told = _base_child_env(settings, state, identity, kind).get("BIND_HOST")
    if told:
        return told
    prefixes = [spec.env_prefix for k, spec in _COMPONENT_SPECS.items() if k.value == kind]
    inherited = os.environ.get(f"{prefixes[0]}_BIND_HOST") if prefixes else None
    return inherited or "127.0.0.1"


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
    # Every HTTP request a stop ends is named in the log (agent#20).
    open_requests.install(app)
    from . import entrypoint

    entrypoint.install(app)

    # Public routes (no auth required).
    app.include_router(health_routes.router)
    app.include_router(auth_routes.router)

    # v0.2 protected routes — bearer session token required.
    protected_dependencies = [Depends(require_operator_session)]
    app.include_router(config_routes.router, dependencies=protected_dependencies)
    app.include_router(entrypoint_routes.router)
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
    app.include_router(profile_build_routes.router)
    app.include_router(serving_measurement_routes.router)
    # Optional apps. Operator-only on every route; declared on the router.
    app.include_router(apps_routes.router)
    # This host's identity and devices; reads only, operator or service.
    app.include_router(node_routes.router)
    from .routes import ollama as ollama_routes

    app.include_router(ollama_routes.router)
    # Checking for and installing a newer version; operator-only.
    app.include_router(update_routes.router)
    app.include_router(logs_routes.router)
    # The log ingress (C1): POST /v1/logs, the one route that takes a client
    # key, write-only and only from a key with writeLogs.
    app.include_router(log_ingress_routes.router)
    # Signing in with Eugene (C2): /oidc/* forwards to the control root.
    app.include_router(oidc_forward_routes.router)
    # Row 53: this machine's store asks which people the root still has.
    app.include_router(people_routes.router)
    # A Job Site: the link page at the machine (J27, J37), and the root's
    # removal of a link (§3.2).
    app.include_router(site_link_routes.router)
    app.include_router(site_link_routes.api)

    # The browser surface, registered LAST and in this order. The proxy
    # is deliberately unauthenticated — it is the path the login request
    # itself travels — and the static mount answers everything not
    # matched above, so anything registered after it is unreachable.
    app.include_router(proxy_routes.router)
    ui_assets.mount(app, settings.ui_dir)

    return app
