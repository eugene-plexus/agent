"""Topology endpoints: /v1/components and /v1/components/{name}.

Combines the declarative topology (from `AgentState`) with live state
from the `Supervisor` (status, pid, lastRestart, lastError) at request
time. Topology mutations through PATCH/POST/DELETE keep the supervisor
in sync — adding a component spawns it, removing one stops it,
updating it triggers a restart so the change takes effect.
"""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from .._generated.common_models import Problem, RestartResult
from .._generated.models import Component, ComponentEntry, ComponentList, ComponentStatus
from ..dependencies import require_operator_or_service, require_operator_session
from ..node_identity import advertise_host, effective_advertise_url, format_url
from ..state import AgentState
from ..supervisor import Supervisor

router = APIRouter(tags=["components"])

# Reads accept operator OR service tokens so peer components can
# auto-resolve topology with their service token; mutations stay
# operator-only (a leaked service token must not be able to re-spawn or
# delete components). See dependencies.require_operator_or_service.
_read_auth = [Depends(require_operator_or_service)]
_write_auth = [Depends(require_operator_session)]


def _not_found(name: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=Problem(
            type="https://github.com/eugene-plexus/agent#component-not-found",
            title="Component not found",
            status=404,
            detail=f"No component named {name!r} in the topology.",
            component="agent",
        ).model_dump(exclude_none=True),
    )


def _name_conflict(name: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=Problem(
            type="https://github.com/eugene-plexus/agent#component-name-conflict",
            title="Name already in use",
            status=409,
            detail=f"Component {name!r} already exists.",
            component="agent",
        ).model_dump(exclude_none=True),
    )


def _advertise_host(request: Request) -> str | None:
    """The host other nodes reach this agent at, if it has one — the
    `advertiseUrl` config field, else what enrollment derived. None on a
    single-host install, and then `Component.advertiseUrl` is absent."""
    state: AgentState = request.app.state.agent_state
    identity = getattr(request.app.state, "node_identity", None)
    persisted = identity.record.advertise_url if identity is not None else None
    return advertise_host(effective_advertise_url(state.get_config("advertiseUrl"), persisted))


def _compose(
    entry: ComponentEntry,
    supervisor: Supervisor | None,
    advertise: str | None = None,
) -> Component:
    """Combine the declarative entry with whatever live state the
    Supervisor knows about it. When no Supervisor is wired (some tests),
    falls back to `unreachable` so the wire shape is still spec-valid.

    `advertise` is this node's advertise host. A spawned component gets
    `advertiseUrl` = that host with its own port — where a peer on
    another host reaches it — while `url` keeps its one meaning: what
    this agent binds and probes. A remote entry is already an address
    the operator chose, so it is left alone."""
    if supervisor is None:
        live_status = ComponentStatus.unreachable
        last_error = None
        last_restart = None
        pid = None
    else:
        live_status, last_error, last_restart, pid = supervisor.status_for(
            entry.name, has_spawn=entry.spawn is not None
        )
    advertise_url: str | None = None
    if advertise is not None and entry.spawn is not None:
        port = urlparse(str(entry.url)).port
        if port is not None:
            advertise_url = format_url(advertise, port)
    return Component(
        name=entry.name,
        kind=entry.kind,
        url=entry.url,
        advertiseUrl=advertise_url,  # type: ignore[arg-type]
        spawn=entry.spawn,
        safeMode=entry.safeMode,
        status=live_status,
        pid=pid,
        lastRestart=last_restart,
        lastError=last_error,
    )


def _supervisor(request: Request) -> Supervisor | None:
    return getattr(request.app.state, "supervisor", None)


@router.get("/v1/components", response_model=ComponentList, dependencies=_read_auth)
async def list_components(request: Request) -> ComponentList:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    return ComponentList(
        components=[
            _compose(e, supervisor, _advertise_host(request)) for e in state.list_topology_entries()
        ],
    )


@router.post("/v1/components", response_model=Component, status_code=201, dependencies=_write_auth)
async def create_component(request: Request, body: ComponentEntry) -> Component:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    try:
        entry = state.add_topology_entry(body)
    except KeyError as e:
        raise _name_conflict(body.name) from e
    if supervisor is not None:
        supervisor.add_and_start(entry)
    return _compose(entry, supervisor, _advertise_host(request))


@router.get("/v1/components/{name}", response_model=Component, dependencies=_read_auth)
async def get_component(request: Request, name: str) -> Component:
    state: AgentState = request.app.state.agent_state
    entry = state.get_topology_entry(name)
    if entry is None:
        raise _not_found(name)
    return _compose(entry, _supervisor(request), _advertise_host(request))


@router.patch("/v1/components/{name}", response_model=Component, dependencies=_write_auth)
async def update_component(request: Request, name: str, body: ComponentEntry) -> Component:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    try:
        updated = state.update_topology_entry(name, body)
    except KeyError as e:
        raise _name_conflict(body.name) from e
    if updated is None:
        raise _not_found(name)
    # Supervised process state is keyed by name, so a rename ends one
    # supervision and starts another. Updates to a same-name entry just
    # trigger a restart so the new env vars (port, configFile, safeMode)
    # take effect.
    if supervisor is not None:
        if name != updated.name:
            await supervisor.remove_and_stop(name)
            supervisor.add_and_start(updated)
        else:
            await supervisor.restart(name)
    return _compose(updated, supervisor, _advertise_host(request))


@router.delete("/v1/components/{name}", status_code=204, dependencies=_write_auth)
async def delete_component(request: Request, name: str) -> Response:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    if not state.remove_topology_entry(name):
        raise _not_found(name)
    if supervisor is not None:
        await supervisor.remove_and_stop(name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/v1/components/{name}/restart",
    response_model=RestartResult,
    status_code=202,
    dependencies=_write_auth,
)
async def restart_component(request: Request, name: str) -> RestartResult:
    state: AgentState = request.app.state.agent_state
    supervisor = _supervisor(request)
    entry = state.get_topology_entry(name)
    if entry is None:
        raise _not_found(name)
    # Remote components have no spawn lifecycle the agent can act on.
    # Per the spec: 409 Conflict.
    if entry.spawn is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=Problem(
                type="https://github.com/eugene-plexus/agent#cannot-restart-remote",
                title="Cannot restart remote component",
                status=409,
                detail=(
                    f"Component {name!r} is remote (no spawn block); the "
                    "agent cannot restart what it does not own."
                ),
                component="agent",
            ).model_dump(exclude_none=True),
        )
    if supervisor is None:
        # No supervisor wired (some test harnesses). Spec-valid 202 with a
        # message that makes the situation legible.
        return RestartResult(
            scheduled=False,
            delayMs=0,
            message="No supervisor is attached to this agent instance.",
        )
    restarted = await supervisor.restart(name)
    if not restarted:
        # Topology entry exists but supervisor doesn't know about it —
        # transient state during topology edits. Best-effort start.
        supervisor.add_and_start(entry)
    return RestartResult(
        scheduled=True,
        delayMs=0,
        message=(
            f"Sent terminate to {name!r}; the supervisor's restart-on-exit "
            "loop will respawn it shortly."
        ),
    )
