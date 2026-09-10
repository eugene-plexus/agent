"""Engine-runtime endpoints: /v1/engines and /v1/runtimes.

Mirrors the shape of the components routes — declaration from
`AgentState`, live state from the supervisor at request time — but
against a separate collection, because an engine binary shares none of a
component's declarative shape. See the agent spec's
components-vs-runtimes table.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from .._generated.common_models import Problem, RestartResult
from .._generated.models import (
    EngineInstall,
    EngineInstallRequest,
    EngineKind,
    EngineList,
    Runtime,
    RuntimeList,
    RuntimeSpec,
)
from ..dependencies import require_operator_or_service, require_operator_session
from ..engines.acquisition import AcquisitionError, Unavailable
from ..runtimes import (
    RuntimeSupervisor,
    describe_engines,
    installer_for,
    plan_for,
    validate_spec,
)
from ..state import AgentState

router = APIRouter()

# Same split as components: reads accept operator OR service tokens so the
# gateway can resolve what is running with its service token; mutations
# stay operator-only, because a leaked service token must not be able to
# start or stop a process holding a GPU.
_read_auth = [Depends(require_operator_or_service)]
_write_auth = [Depends(require_operator_session)]


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


def _supervisor(request: Request) -> RuntimeSupervisor | None:
    return getattr(request.app.state, "runtime_supervisor", None)


def _compose(spec: RuntimeSpec, supervisor: RuntimeSupervisor | None) -> Runtime:
    if supervisor is None:
        # No supervisor wired (some tests): report the declaration with a
        # spec-valid `stopped`, which is honest — nothing is running it.
        return RuntimeSupervisor().compose(spec)
    return supervisor.compose(spec)


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
    "/v1/runtimes",
    response_model=Runtime,
    status_code=201,
    tags=["runtimes"],
    dependencies=_write_auth,
)
async def create_runtime(request: Request, body: RuntimeSpec) -> Runtime:
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

    if (reason := validate_spec(body)) is not None:
        raise _problem(
            code=status.HTTP_400_BAD_REQUEST,
            slug="invalid-runtime-spec",
            title="Invalid runtime spec",
            detail=reason,
        )
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
    if supervisor is not None:
        await supervisor.remove_and_stop(name)
    # The model file is never touched. Removing a runtime un-declares a
    # way of serving a model; it does not delete anything the user owns.
    if not state.remove_runtime(name):
        raise _not_found(name)
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
    dependencies=_write_auth,
)
async def stop_runtime(request: Request, name: str) -> RestartResult:
    state: AgentState = request.app.state.agent_state
    if state.get_runtime_spec(name) is None:
        raise _not_found(name)
    supervisor = _supervisor(request)
    if supervisor is not None:
        await supervisor.stop_one(name)
    return RestartResult(
        scheduled=True,
        delayMs=0,
        message=(
            f"Engine runtime {name!r} stopped and left declared; its GPU memory "
            f"is released. POST .../start to bring it back."
        ),
    )


@router.post(
    "/v1/runtimes/{name}/start",
    response_model=RestartResult,
    status_code=202,
    tags=["runtimes"],
    dependencies=_write_auth,
)
async def start_runtime(request: Request, name: str) -> RestartResult:
    state: AgentState = request.app.state.agent_state
    spec = state.get_runtime_spec(name)
    if spec is None:
        raise _not_found(name)
    supervisor = _supervisor(request)
    already = supervisor is not None and supervisor.is_running(name)
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
