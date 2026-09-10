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

from .. import security
from .._generated.common_models import Problem, RestartResult
from .._generated.models import (
    Admission,
    AdmissionDecision,
    ComponentKind,
    EngineInstall,
    EngineInstallRequest,
    EngineKind,
    EngineList,
    Runtime,
    RuntimeList,
    RuntimeSpec,
    StopReason,
    StopRequest,
)
from ..admission import LibraryFitClient, RunningRuntime, check_admission
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
from ..runtimes import (
    RuntimeSupervisor,
    describe_engines,
    installer_for,
    plan_for,
    validate_spec,
)
from ..state import AgentState

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


def _library_client(request: Request) -> LibraryFitClient | None:
    """The library, if this agent's topology has one.

    Tests may inject `app.state.library_fit_client` (anything with an
    async `fit`, or None to force the file-size path).
    """
    if hasattr(request.app.state, "library_fit_client"):
        return request.app.state.library_fit_client  # type: ignore[no-any-return]
    state: AgentState = request.app.state.agent_state
    entry = next(
        (e for e in state.list_topology_entries() if e.kind is ComponentKind.library), None
    )
    if entry is None:
        return None
    auth = getattr(request.app.state, "auth_state", None)
    token = (
        security.issue_service_token(signing_key=auth.signing_key, kind="agent")
        if auth is not None and auth.signing_key is not None
        else None
    )
    return LibraryFitClient(str(entry.url), token)


async def _admission_for(request: Request, spec: RuntimeSpec) -> Admission:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    detector = getattr(request.app.state, "device_detector", None) or detect_devices
    snapshot = await asyncio.to_thread(detector)
    running = [
        RunningRuntime(spec=other, status=_compose(other, supervisor).status)
        for other in state.list_runtime_specs()
        if other.name != spec.name
    ]
    return await check_admission(
        spec,
        snapshot=snapshot,
        library=_library_client(request),
        running=running,
        size_of=getattr(request.app.state, "model_size_of", None),
    )


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #


@router.get("/v1/engines", response_model=EngineList, tags=["engines"], dependencies=_read_auth)
async def list_engines(request: Request) -> EngineList:
    # The agent's own config is where an install-wide engine path
    # (`vllmBinary`) lives, so discovery reads through it.
    state: AgentState = request.app.state.agent_state
    return EngineList(engines=describe_engines(get_config=state.get_config))


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

    plan = plan_for(kind, version=body.version if body else None)
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
    if (reason := validate_spec(body)) is not None:
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
    dependencies=_declare_auth,
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
    if (reason := validate_spec(body)) is not None:
        raise _problem(
            code=status.HTTP_400_BAD_REQUEST,
            slug="invalid-runtime-spec",
            title="Invalid runtime spec",
            detail=reason,
        )

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
    # only when it *is* a launch: `autoStart: false` is measured when it
    # starts, and `force` is the operator saying they know better.
    if body.autoStart is not False and not force:
        admission = await _admission_for(request, body)
        if admission.decision is AdmissionDecision.refuse:
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
    "/v1/runtimes/{name}", response_model=Runtime, tags=["runtimes"], dependencies=_write_auth
)
async def update_runtime(request: Request, name: str, body: RuntimeSpec) -> Runtime:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    components = _component_supervisor(request)

    if (reason := validate_spec(body)) is not None:
        raise _problem(
            code=status.HTTP_400_BAD_REQUEST,
            slug="invalid-runtime-spec",
            title="Invalid runtime spec",
            detail=reason,
        )
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
    dependencies=_write_auth,
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
    dependencies=_lifecycle_auth,
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
    if not already and not force:
        # Where a runtime declared with `autoStart: false` meets
        # admission — it was not measured at declaration because it was
        # not being launched then.
        admission = await _admission_for(request, spec)
        if admission.decision is AdmissionDecision.refuse:
            raise _refused(admission)
    if supervisor is not None and not already:
        # `autoStart: false` means "don't start at boot", not "never
        # start" — an explicit start overrides it for this session.
        supervisor.add_and_start(spec.model_copy(update={"autoStart": True}))
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
