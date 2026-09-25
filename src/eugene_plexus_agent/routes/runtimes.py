"""Engine-runtime endpoints: /v1/engines and /v1/runtimes.

Mirrors the shape of the components routes — declaration from
`AgentState`, live state from the supervisor at request time — but
against a separate collection, because an engine binary shares none of a
component's declarative shape. See the agent spec's
components-vs-runtimes table.

M6 adds three things here and they are the whole of lifecycle policy as
the agent sees it: a declared runtime brings its **companion driver**
with it (`companions.py`), a launch is **measured before it spawns**
(`admission.py`), and the gateway — the one component that sees demand
— may **stop and start** a runtime with its own service token, saying
why, so the agent can report `stopReason`.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from .. import install_proxy, library_folders
from .._generated.common_models import Problem, RestartResult
from .._generated.models import (
    Admission,
    AdmissionDecision,
    ComponentKind,
    EngineInstall,
    EngineInstallRequest,
    EngineKind,
    EngineList,
    ModelCopyClearResult,
    ModelCopySkipped,
    Runtime,
    RuntimeList,
    RuntimeSpec,
    StopReason,
    StopRequest,
)
from ..admission import (
    PENDING_STATUSES,
    LibraryFitClient,
    RunningRuntime,
    check_admission,
)
from ..companions import (
    CompanionConflict,
    companion_name,
    ensure_companion,
    is_companion,
    remove_companion,
)
from ..dependencies import (
    require_operator_or_control,
    require_operator_or_gateway,
    require_operator_or_service,
    require_operator_session,
)
from ..engines.acquisition import AcquisitionError, Unavailable
from ..engines.devices import detect_devices
from ..install_proxy import lookup_authorization
from ..model_paths import PathRule, rules_from_config
from ..node_work import runtime_launch
from ..reservations import ReservationLedger
from ..runtimes import (
    RuntimeSupervisor,
    describe_engines,
    installer_for,
    plan_for,
    validate_spec,
)
from ..state import AgentState
from .proxy import install_topology

log = logging.getLogger(__name__)

router = APIRouter()

# Same split as components: reads accept operator OR service tokens so the
# gateway can resolve what is running with its service token; mutations
# stay operator-only, because a leaked service token must not be able to
# start or stop a process holding a GPU.
#
# Except stop and start, from M6: those accept the operator OR the
# gateway's own audience, checked exactly. The gateway is the component
# that sees demand, so it is the one that unloads an idle runtime and
# wakes one on request. A leaked driver or library token still cannot.
_read_auth = [Depends(require_operator_or_service)]
_write_auth = [Depends(require_operator_session)]
_lifecycle_auth = [Depends(require_operator_or_gateway)]
# And declaring one, from M7: the operator OR the control root, because
# the control root forwards declarations to the node that will run them
# and the trust root's token is what every other credential reduces to.
_declare_auth = [Depends(require_operator_or_control)]


def _problem(*, code: int, slug: str, title: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/agent#{slug}",
            title=title,
            status=code,
            detail=detail,
            component="agent",
        ).model_dump(exclude_none=True),
    )


def _not_found(name: str) -> HTTPException:
    return _problem(
        code=status.HTTP_404_NOT_FOUND,
        slug="runtime-not-found",
        title="Runtime not found",
        detail=f"No runtime named {name!r} is declared.",
    )


def _refused(admission: Admission) -> HTTPException:
    return _problem(
        code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        slug="admission-refused",
        title="Admission refused",
        detail=admission.reason,
    )


def _supervisor(request: Request) -> RuntimeSupervisor | None:
    return getattr(request.app.state, "runtime_supervisor", None)


def _component_supervisor(request: Request):  # type: ignore[no-untyped-def]
    return getattr(request.app.state, "supervisor", None)


def _compose(spec: RuntimeSpec, supervisor: RuntimeSupervisor | None) -> Runtime:
    if supervisor is None:
        # No supervisor wired (some tests): report the declaration with a
        # spec-valid `stopped`, which is honest — nothing is running it.
        return RuntimeSupervisor().compose(spec)
    return supervisor.compose(spec)


# --------------------------------------------------------------------------- #
# Admission
# --------------------------------------------------------------------------- #


async def library_client_for(request: Request) -> LibraryFitClient | None:
    """The library this agent can ask, if any.

    The one in this agent's own topology when there is one. Otherwise --
    an enrolled worker declares none, by M9's rule -- the install's,
    found through the control root the way the console hop finds a
    component (`install_proxy`: address nodes, not components) and
    reached through the owning node's agent proxy, with a
    `service:agent` token this node mints. That is what makes admission
    on a worker `metadata`-based at all (M11); before it, every launch
    on the GPU node of the first two-machine install was measured by
    file size and nothing said so.

    Nothing here is fatal. With no library reachable, admission falls
    back to file size exactly as it did before, and logs why.

    Tests may inject `app.state.library_fit_client` (anything with an
    async `fit`, or None to force the file-size path), a
    `library_transport` for the client to speak through, and a
    `control_transport` for the lookup.
    """
    if hasattr(request.app.state, "library_fit_client"):
        return request.app.state.library_fit_client  # type: ignore[no-any-return]
    state: AgentState = request.app.state.agent_state
    auth = getattr(request.app.state, "auth_state", None)
    trust = auth.trust if auth is not None else None
    transport = getattr(request.app.state, "library_transport", None)

    entry = next(
        (e for e in state.list_topology_entries() if e.kind is ComponentKind.library), None
    )
    if entry is not None:
        local = trust.agent_token(trust.recipient) if trust is not None else None
        return LibraryFitClient(str(entry.url), local, transport=transport)

    identity = getattr(request.app.state, "node_identity", None)
    record = identity.record if identity is not None else None
    if record is None or not record.enrolled or not record.control_url:
        return None
    try:
        remote = await install_topology(request).owner_of(
            "library",
            control_url=str(record.control_url),
            authorization=lookup_authorization(request),
            transport=getattr(request.app.state, "control_transport", None),
        )
    except install_proxy.InstallLookupError as exc:
        log.info(
            "no library on this node, and the install's could not be found (%s); "
            "admission measures by file size",
            exc,
        )
        return None
    if record.name and remote.name == record.name:
        # The registry says the library is here and this topology has
        # none. A disagreement the console hop reports; here it means
        # there is nothing to ask.
        return None
    # An `agent` token addressed to the library's machine: the owner's
    # proxy forwards it to its local library unchanged, and it is good
    # nowhere else (2026-09-25).
    token = trust.agent_token(f"node:{remote.name}") if trust is not None else None
    return LibraryFitClient(
        f"{remote.agent_url.rstrip('/')}/api/proxy/library", token, transport=transport
    )


def folder_cache_for(request: Request) -> library_folders.LibraryFolderCache | None:
    """This node's copy of the Library's folders, when the app has one."""
    cache = getattr(request.app.state, "library_folders", None)
    return cache if isinstance(cache, library_folders.LibraryFolderCache) else None


def effective_rules_for(request: Request) -> list[PathRule]:
    """This node's `pathMappings` overrides, then the Library folders'
    mounts for this host -- the rules a spawn uses (2026-09-14)."""
    state: AgentState = request.app.state.agent_state
    cache = folder_cache_for(request)
    return library_folders.effective_rules(
        rules_from_config(state.get_config),
        cache.inherited_rules() if cache is not None else (),
    )


async def refresh_library_folders(request: Request) -> bool:
    """Read the library's folders into this node's copy, once per request.

    Once per request rather than in a background loop, which is why it
    is cheap: the library is asked only when something here is about to
    use its answer. True when the library answered on this request.
    """
    done = getattr(request.state, "library_folders_refreshed", None)
    if done is not None:
        return bool(done)
    cache = folder_cache_for(request)
    answered = False
    if cache is not None:
        answered = await library_folders.refresh(cache, await library_client_for(request))
    request.state.library_folders_refreshed = answered
    return answered


def require_library_folder(request: Request, model_path: str) -> None:
    """A node runs only what the Library catalogues (2026-09-14).

    400 when `model_path` lies under none of the Library's folders. When
    the folder list has never been read from this node, the check is
    skipped with a warning rather than refusing every launch -- a worker
    in its first minute, a single box whose library has not started.
    """
    cache = folder_cache_for(request)
    if cache is None or not cache.known:
        log.warning(
            "%s was not checked against the Library's folders: this node has never read "
            "them (no library reachable). It launches on M11's terms; "
            "POST /v1/library/folders/check says when the list arrives.",
            model_path,
        )
        return
    if cache.folder_for(model_path) is None:
        raise _problem(
            code=status.HTTP_400_BAD_REQUEST,
            slug="model-not-in-library",
            title="Not a Library model",
            detail=library_folders.not_in_library_detail(model_path),
        )


def _ledger(request: Request) -> ReservationLedger:
    """This node's record of memory promised to launches in flight.

    On `app.state` rather than inside the supervisor because the routes
    are what can tell a dry run from a launch, and a supervisor that
    reserved on `add_and_start` would have no bytes to reserve -- the
    arithmetic lives one layer up.
    """
    ledger = getattr(request.app.state, "reservations", None)
    if ledger is None:
        ledger = ReservationLedger()
        request.app.state.reservations = ledger
    return ledger


def _reserve(request: Request, spec: RuntimeSpec, admission: Admission | None) -> None:
    """Promise the memory a launch we just scheduled is about to take.

    Called at the point of committing to a start and nowhere else, so a
    declaration that 409s or a spec that is only being previewed leaves
    nothing behind. `requiredBytes` is absent when nothing could be
    measured -- an `unknown` fit, a file that could not be sized -- and
    a promise of an unknown quantity is not a promise.
    """
    if admission is None or not admission.requiredBytes:
        return
    _ledger(request).reserve(
        spec.name,
        device_index=admission.device.index if admission.device is not None else None,
        size_bytes=admission.requiredBytes,
    )


async def _admission_for(request: Request, spec: RuntimeSpec) -> Admission:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    await refresh_library_folders(request)
    detector = getattr(request.app.state, "device_detector", None) or detect_devices
    snapshot = await asyncio.to_thread(detector)
    observed = [(other, _compose(other, supervisor).status) for other in state.list_runtime_specs()]
    running = [
        RunningRuntime(spec=other, status=status)
        for other, status in observed
        if other.name != spec.name
    ]
    # Read is where the ledger is reconciled: a reservation stands only
    # while its runtime is still on its way up. Past that it either holds
    # the memory for real -- and the snapshot below counts it, so
    # counting both would refuse a third launch that fits -- or holds
    # none. Reconciling here rather than on a loop means the sweep runs
    # exactly when its answer is about to be used.
    ledger = _ledger(request)
    ledger.reconcile(other.name for other, status in observed if status in PENDING_STATUSES)
    identity = getattr(request.app.state, "node_identity", None)
    node_name = identity.record.name if identity is not None and identity.record.enrolled else None
    return await check_admission(
        spec,
        snapshot=snapshot,
        library=await library_client_for(request),
        running=running,
        reservations=ledger.entries(),
        size_of=getattr(request.app.state, "model_size_of", None),
        mappings=effective_rules_for(request),
        node_name=node_name,
        exists=getattr(request.app.state, "model_exists", None),
        # So admission asks about the file a launch would actually open,
        # including this node's own copy when it holds one.
        copy_settings=supervisor.copy_settings() if supervisor is not None else None,
    )


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #


@router.get("/v1/engines", response_model=EngineList, tags=["engines"], dependencies=_read_auth)
async def list_engines(request: Request) -> EngineList:
    """What this agent can start, and what it found on disk.

    **In a thread, because none of it is I/O the loop can await**
    (review §6.1 #5). One call resolves the host with vendor-tool
    subprocesses at a 5 s cap each, walks the managed store on disk, and
    may reach `api.github.com` with a blocking `urlopen`. Home polls this
    every 15 s and the Issues badge every 30 s per node, so on a box
    whose driver stack is unhappy the agent — which is also the process
    serving the browser its own UI and proxying every other component —
    was stalled for seconds at a time. The codebase already does this at
    five comparable sites; `asyncio.to_thread` here is the same move.
    """
    # The agent's own config is where an install-wide engine path
    # (`vllmBinary`) lives, so discovery reads through it. `get_config`
    # takes a `threading.Lock`, so it is safe to call from the worker.
    state: AgentState = request.app.state.agent_state
    engines = await asyncio.to_thread(describe_engines, get_config=state.get_config)
    return EngineList(engines=engines)


def _engine_kind(engine: str) -> EngineKind:
    try:
        return EngineKind(engine)
    except ValueError:
        raise _problem(
            code=status.HTTP_404_NOT_FOUND,
            slug="unknown-engine",
            title="Unknown engine",
            detail=(
                f"{engine!r} is not an engine this agent implements. "
                f"GET /v1/engines lists what it does."
            ),
        ) from None


@router.post(
    "/v1/engines/{engine}/install",
    response_model=EngineInstall,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["engines"],
    dependencies=_write_auth,
)
async def install_engine(engine: str, body: EngineInstallRequest | None = None) -> EngineInstall:
    """Fetch, verify and unpack an engine build.

    Returns immediately. The download is hundreds of megabytes — on
    Windows with CUDA it is two assets totalling about half a gigabyte —
    and holding a request open for it buys nothing while losing the
    progress the operator actually wants to watch.
    """
    kind = _engine_kind(engine)
    installer = installer_for(kind)
    if installer is None:
        raise _problem(
            code=status.HTTP_404_NOT_FOUND,
            slug="unknown-engine",
            title="Unknown engine",
            detail=f"No adapter for engine {engine!r} in this build.",
        )
    if installer.running:
        raise _problem(
            code=status.HTTP_409_CONFLICT,
            slug="install-in-flight",
            title="Install already running",
            detail=(
                f"An install is already in flight for {engine!r}. Wait for it, or "
                f"DELETE this path to cancel it."
            ),
        )

    # Same thread treatment as the listing above, for the same reason:
    # this resolves the host and may reach upstream, and the operator
    # who pressed Install is not the only person using this agent.
    plan = await asyncio.to_thread(plan_for, kind, version=body.version if body else None)
    if isinstance(plan, Unavailable):
        # 422, not 404 or 500: the request was well-formed and the engine
        # exists — this host simply has nothing installable, which on
        # Linux with an NVIDIA GPU is a permanent and legitimate answer
        # rather than a transient failure worth retrying.
        raise _problem(
            code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            slug="nothing-installable",
            title="Nothing installable for this host",
            detail=plan.reason,
        )

    try:
        return installer.start(plan)
    except AcquisitionError as e:
        raise _problem(
            code=status.HTTP_409_CONFLICT,
            slug="install-in-flight",
            title="Install already running",
            detail=str(e),
        ) from e


@router.get(
    "/v1/engines/{engine}/install",
    response_model=EngineInstall,
    tags=["engines"],
    dependencies=_read_auth,
)
async def get_engine_install(engine: str) -> EngineInstall:
    """Progress of the current or most recent install."""
    kind = _engine_kind(engine)
    installer = installer_for(kind)
    snapshot = installer.snapshot() if installer else None
    if snapshot is None:
        raise _problem(
            code=status.HTTP_404_NOT_FOUND,
            slug="no-install",
            title="No install",
            detail=f"No install has been started for {engine!r} in this process.",
        )
    return snapshot


@router.delete(
    "/v1/engines/{engine}/install",
    response_model=EngineInstall,
    tags=["engines"],
    dependencies=_write_auth,
)
async def cancel_engine_install(engine: str) -> EngineInstall:
    """Cancel an install in flight.

    Cancels an *install*; it does not uninstall an engine. A build that
    already finished is untouched.
    """
    kind = _engine_kind(engine)
    installer = installer_for(kind)
    if installer is None or not installer.running:
        raise _problem(
            code=status.HTTP_409_CONFLICT,
            slug="nothing-to-cancel",
            title="Nothing in flight",
            detail=f"No install is running for {engine!r}.",
        )
    snapshot = await installer.cancel()
    assert snapshot is not None  # running implies a snapshot exists
    return snapshot


# --------------------------------------------------------------------------- #
# Runtimes
# --------------------------------------------------------------------------- #


@router.get("/v1/runtimes", response_model=RuntimeList, tags=["runtimes"], dependencies=_read_auth)
async def list_runtimes(request: Request) -> RuntimeList:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    return RuntimeList(
        runtimes=[_compose(s, supervisor) for s in state.list_runtime_specs()],
    )


@router.post(
    "/v1/runtimes/admission",
    response_model=Admission,
    tags=["runtimes"],
    dependencies=_read_auth,
)
async def check_runtime_admission(request: Request, body: RuntimeSpec) -> Admission:
    """Would this runtime fit on the device it targets, right now?

    The dry run behind create and start. Declares nothing, spawns
    nothing; readable with a service token so the gateway can ask it
    before waking a runtime.
    """
    if (reason := validate_spec(body, request.app.state.agent_state.get_config)) is not None:
        raise _problem(
            code=status.HTTP_400_BAD_REQUEST,
            slug="invalid-runtime-spec",
            title="Invalid runtime spec",
            detail=reason,
        )
    return await _admission_for(request, body)


@router.post(
    "/v1/runtimes",
    response_model=Runtime,
    status_code=201,
    tags=["runtimes"],
    dependencies=[*_declare_auth, Depends(runtime_launch)],
)
async def create_runtime(
    request: Request,
    body: RuntimeSpec,
    force: bool = Query(default=False),
) -> Runtime:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)

    # Validate before persisting: a declaration that can only fail at
    # spawn is worse than a 400, because the failure surfaces minutes
    # later in a log rather than in the form the operator is looking at.
    if (reason := validate_spec(body, state.get_config)) is not None:
        raise _problem(
            code=status.HTTP_400_BAD_REQUEST,
            slug="invalid-runtime-spec",
            title="Invalid runtime spec",
            detail=reason,
        )

    # A node runs only what the Library catalogues. Checked before the
    # companion is declared, or a refused runtime would leave a driver
    # behind -- the same ordering the companion-name check keeps.
    await refresh_library_folders(request)
    require_library_folder(request, body.modelPath)

    # The companion's name has to be free before anything is written, or
    # a refused companion would leave a runtime declared with no driver
    # and the operator believing it has one.
    if body.autoDriver is not False:
        held = state.get_topology_entry(companion_name(body.name))
        if held is not None and not is_companion(held, state):
            raise _problem(
                code=status.HTTP_409_CONFLICT,
                slug="companion-name-conflict",
                title="Companion driver name already in use",
                detail=(
                    f"Runtime {body.name!r} would declare a companion driver named "
                    f"{companion_name(body.name)!r}, but a component of that name already "
                    f"exists and is not a companion. Rename the runtime, or set "
                    f"autoDriver: false and front it with that driver by hand."
                ),
            )

    # A launch that will not fit is refused before it spawns. Measured
    # whenever it *is* a launch -- `autoStart: false` is measured when it
    # starts instead -- and `force` is the operator saying they know
    # better, which overrides the refusal and not the arithmetic: a
    # forced launch still spends the memory, so it is still measured and
    # still reserved. Before that, a forced launch was invisible to the
    # next admission.
    admission: Admission | None = None
    if body.autoStart is not False:
        admission = await _admission_for(request, body)
        if admission.decision is AdmissionDecision.refuse and not force:
            raise _refused(admission)

    try:
        spec = state.add_runtime(body)
    except KeyError as e:
        raise _problem(
            code=status.HTTP_409_CONFLICT,
            slug="runtime-name-conflict",
            title="Name already in use",
            detail=f"Runtime {body.name!r} already exists.",
        ) from e
    except ValueError as e:
        raise _problem(
            code=status.HTTP_400_BAD_REQUEST,
            slug="invalid-runtime-spec",
            title="Invalid runtime spec",
            detail=str(e),
        ) from e

    # Runtime first, companion second: the runtime's port is assigned at
    # write time, so when the driver resolves `runtimeName` the answer
    # already exists whatever the engine's state.
    if spec.autoDriver is not False:
        try:
            await ensure_companion(state, _component_supervisor(request), spec)
        except CompanionConflict as e:  # pragma: no cover - checked above
            log.error("companion for %s: %s", spec.name, e)

    if supervisor is not None:
        supervisor.add_and_start(spec)
        # Last, and only once the start is actually scheduled: a
        # declaration that 409'd or 400'd above never promised anything,
        # so there is no failure path here that needs a release.
        _reserve(request, spec, admission)
    return _compose(spec, supervisor)


@router.get(
    "/v1/runtimes/{name}", response_model=Runtime, tags=["runtimes"], dependencies=_read_auth
)
async def get_runtime(request: Request, name: str) -> Runtime:
    state: AgentState = request.app.state.agent_state
    spec = state.get_runtime_spec(name)
    if spec is None:
        raise _not_found(name)
    return _compose(spec, _supervisor(request))


@router.patch(
    "/v1/runtimes/{name}",
    response_model=Runtime,
    tags=["runtimes"],
    dependencies=[*_write_auth, Depends(runtime_launch)],
)
async def update_runtime(request: Request, name: str, body: RuntimeSpec) -> Runtime:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    components = _component_supervisor(request)

    if (reason := validate_spec(body, state.get_config)) is not None:
        raise _problem(
            code=status.HTTP_400_BAD_REQUEST,
            slug="invalid-runtime-spec",
            title="Invalid runtime spec",
            detail=reason,
        )
    await refresh_library_folders(request)
    require_library_folder(request, body.modelPath)
    previous = state.get_runtime_spec(name)
    if previous is None:
        raise _not_found(name)
    try:
        updated = state.update_runtime(name, body)
    except KeyError as e:
        raise _problem(
            code=status.HTTP_409_CONFLICT,
            slug="runtime-name-conflict",
            title="Name already in use",
            detail=f"Runtime {body.name!r} already exists.",
        ) from e
    except ValueError as e:
        raise _problem(
            code=status.HTTP_400_BAD_REQUEST,
            slug="invalid-runtime-spec",
            title="Invalid runtime spec",
            detail=str(e),
        ) from e
    if updated is None:
        raise _not_found(name)

    # The companion follows the declaration: gone with a rename or an
    # opt-out, re-targeted when the alias moved, created when opted in.
    if previous.autoDriver is not False and (
        updated.name != previous.name or updated.autoDriver is False
    ):
        await remove_companion(state, components, previous.name)
    if updated.autoDriver is not False:
        try:
            await ensure_companion(state, components, updated)
        except CompanionConflict as e:
            raise _problem(
                code=status.HTTP_409_CONFLICT,
                slug="companion-name-conflict",
                title="Companion driver name already in use",
                detail=str(e),
            ) from e

    # Every field on a RuntimeSpec is baked into the argv at spawn time —
    # there is no way to re-flag a live llama-server — so any change
    # restarts the engine. The response reports the post-restart state,
    # normally `starting` or `loading` rather than `ready`.
    if supervisor is not None:
        # The reservation is deliberately NOT released here. A PATCH
        # restarts the engine, so the runtime is about to take memory
        # again; the standing promise is stale in size and right in
        # kind, and dropping it would reopen the defect for exactly as
        # long as the restart takes.
        await supervisor.remove_and_stop(name)
        supervisor.add_and_start(updated)
    return _compose(updated, supervisor)


@router.delete("/v1/runtimes/{name}", status_code=204, tags=["runtimes"], dependencies=_write_auth)
async def delete_runtime(request: Request, name: str) -> Response:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    if state.get_runtime_spec(name) is None:
        raise _not_found(name)
    if supervisor is not None:
        await supervisor.remove_and_stop(name)
    # The companion goes with its runtime; the model file is never
    # touched. Removing a runtime un-declares a way of serving a model;
    # it does not delete anything the user owns.
    await remove_companion(state, _component_supervisor(request), name)
    state.remove_runtime(name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/v1/runtimes/{name}/restart",
    response_model=RestartResult,
    status_code=202,
    tags=["runtimes"],
    dependencies=[*_write_auth, Depends(runtime_launch)],
)
async def restart_runtime(request: Request, name: str) -> RestartResult:
    state: AgentState = request.app.state.agent_state
    spec = state.get_runtime_spec(name)
    if spec is None:
        raise _not_found(name)
    supervisor = _supervisor(request)
    if supervisor is not None and not await supervisor.restart(name):
        # Nothing was supervised under that name — it is stopped or was
        # given up on. Start it: a Restart button that no-ops on a stopped
        # engine is a worse answer than just bringing it back.
        supervisor.add_and_start(spec)
    return RestartResult(
        scheduled=True,
        delayMs=0,
        message=(
            f"Sent terminate to engine runtime {name!r}; it will respawn shortly. "
            f"The model load is paid again, so expect `loading` before `ready`."
        ),
    )


@router.post(
    "/v1/runtimes/{name}/stop",
    response_model=RestartResult,
    status_code=202,
    tags=["runtimes"],
    dependencies=_lifecycle_auth,
)
async def stop_runtime(
    request: Request, name: str, body: StopRequest | None = None
) -> RestartResult:
    state: AgentState = request.app.state.agent_state
    if state.get_runtime_spec(name) is None:
        raise _not_found(name)
    reason = body.reason if body is not None and body.reason is not None else StopReason.operator
    supervisor = _supervisor(request)
    if supervisor is not None:
        await supervisor.stop_one(name, reason=reason)
    return RestartResult(
        scheduled=True,
        delayMs=0,
        message=(
            f"Engine runtime {name!r} stopped ({reason.value}) and left declared; its GPU "
            f"memory is released. POST .../start to bring it back."
        ),
    )


@router.post(
    "/v1/runtimes/{name}/start",
    response_model=RestartResult,
    status_code=202,
    tags=["runtimes"],
    dependencies=[*_lifecycle_auth, Depends(runtime_launch)],
)
async def start_runtime(
    request: Request,
    name: str,
    force: bool = Query(default=False),
) -> RestartResult:
    state: AgentState = request.app.state.agent_state
    spec = state.get_runtime_spec(name)
    if spec is None:
        raise _not_found(name)
    supervisor = _supervisor(request)
    already = supervisor is not None and supervisor.is_running(name)
    admission: Admission | None = None
    if not already:
        # Where a runtime declared with `autoStart: false` meets
        # admission — it was not measured at declaration because it was
        # not being launched then. Measured under `force` too, because
        # the reservation comes off this number.
        admission = await _admission_for(request, spec)
        if admission.decision is AdmissionDecision.refuse and not force:
            raise _refused(admission)
    if supervisor is not None and not already:
        # `autoStart: false` means "don't start at boot", not "never
        # start" — an explicit start overrides it for this session.
        supervisor.add_and_start(spec.model_copy(update={"autoStart": True}))
        _reserve(request, spec, admission)
    return RestartResult(
        # `scheduled` reports whether this call caused a start, so an
        # idempotent second press is distinguishable from the first.
        scheduled=not already,
        delayMs=0,
        message=(
            f"Engine runtime {name!r} was already running; nothing to do."
            if already
            else f"Start scheduled for engine runtime {name!r}."
        ),
    )


@router.post(
    "/v1/model-copies/clear",
    response_model=ModelCopyClearResult,
    tags=["runtimes"],
    dependencies=_write_auth,
)
async def clear_model_copies(request: Request) -> ModelCopyClearResult:
    """Delete this node's local copies of its models.

    Operator-only: it deletes files on this host. Deleting nothing is a
    success -- a node with copying switched off answers with two empty
    lists rather than an error, because "there was nothing to clear" is
    the honest answer to "clear this".

    It stops nothing and switches nothing off. A copy a runtime is using
    is reported in `skipped`, and the copies come back the next time
    those runtimes start, which is what the operator asked for by
    leaving the toggle on.
    """
    supervisor = _supervisor(request)
    if supervisor is None:
        return ModelCopyClearResult(deleted=[], bytesFreed=0, skipped=[])
    result = await asyncio.to_thread(supervisor.clear_copies)
    return ModelCopyClearResult(
        deleted=result.deleted,
        bytesFreed=result.bytes_freed,
        skipped=[
            ModelCopySkipped(path=s.path, runtime=s.runtime, reason=s.reason)
            for s in result.skipped
        ],
    )
