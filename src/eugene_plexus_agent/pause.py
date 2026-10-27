"""Pause the models on this machine, and keep serving (row 35).

Troy, 2026-09-18: *"many users with RTX graphic cards will want to turn
Eugene off easily to play a video game."* Stop Eugene does that, and takes
the console, the gateway and this machine's place in the install with it.
Pause stops every model here and nothing else.

The tray icon asks for it over a local channel (`tray_pipe`), and the
console through `POST /v1/node/pause`; both call `pause` and `resume`
here.

**Pause** takes the launch lock, records the pause, cancels a measurement
that is running (the models it stopped join the pause), and stops every
runtime that is not stopped with `StopReason.paused`. Answers in progress
on those models stop at once (TRAY-3): a person starting a game wants the
card now.

**While paused, nothing starts a model here.** `node_work.launch_guard`
and the measurement routes refuse with 409 and `sentence`; a measurement's
own restart leaves its models stopped; and at boot `AgentState` holds the
paused runtimes, so a reboot loads nothing (TRAY-2).

**Resume** clears the record, then starts the list again exactly as the
Start button would: through memory admission, never forced. A runtime
admission refuses stays stopped as a person's stop, and its result names
the cause.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import sys
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from ._generated.common_models import Problem
from ._generated.models import (
    MeasurementRestartState,
    NodePause,
    NodePauseBy,
    NodePauseResult,
    NodePauseRuntimeResult,
    NodePauseRuntimeState,
    NodeTrayChannel,
    StopReason,
)
from .runtime_context import NodeContext

log = logging.getLogger(__name__)

#: How long Pause waits for a measurement it cancelled to end. A benchmark
#: ends its process within five seconds; past this the pause goes on, and
#: the measurement's own restart still finds the machine paused.
CANCEL_SECONDS = 15.0


def new_record(*, by: str, runtimes: list[str]) -> dict[str, Any]:
    """What `AgentState.set_pause` keeps: when, from where, and the list."""
    return {"since": datetime.now(UTC).isoformat(), "by": by, "runtimes": list(runtimes)}


def record(app: Any) -> dict[str, Any] | None:
    """The pause in effect on this machine, or None."""
    state = getattr(app.state, "agent_state", None)
    return state.pause_record() if state is not None else None


def machine_name(app: Any) -> str:
    """This machine's name in the install, or its host name before it joins one."""
    identity = getattr(app.state, "node_identity", None)
    held = getattr(identity, "record", None)
    if held is not None and held.enrolled and held.name:
        return str(held.name)
    return socket.gethostname()


def sentence(held: dict[str, Any], *, machine: str) -> str:
    """Why a start is refused, and where the way back is. One sentence each."""
    if held.get("by") == "tray":
        return (
            f"Models on {machine} are paused from its tray icon. "
            "Resume them there or in Eugene's console."
        )
    if sys.platform == "win32":
        return (
            f"Models on {machine} are paused from Eugene's console. "
            "Resume them there or from its tray icon."
        )
    return f"Models on {machine} are paused from Eugene's console. Resume them there."


def refusal(app: Any) -> HTTPException | None:
    """The 409 for any start while paused, or None when not paused."""
    held = record(app)
    if held is None:
        return None
    return HTTPException(
        status_code=409,
        detail=Problem(
            type="https://github.com/eugene-plexus/agent#models-paused",
            title="Models paused",
            status=409,
            detail=sentence(held, machine=machine_name(app)),
            component="agent",
        ).model_dump(exclude_none=True),
    )


def tray_channel(app: Any) -> NodeTrayChannel | None:
    """The tray's channel on Windows; None elsewhere, where there is no tray."""
    if sys.platform != "win32":
        return None
    pipe = getattr(app.state, "tray_pipe", None)
    status = getattr(pipe, "status", None)
    return NodeTrayChannel(status) if status else NodeTrayChannel.unavailable


def view(app: Any) -> NodePause:
    """`NodeIdentity.pause`: present on every agent from row 35 on."""
    held = record(app)
    channel = tray_channel(app)
    if held is None:
        return NodePause(paused=False, trayChannel=channel)
    return NodePause(
        paused=True,
        since=datetime.fromisoformat(held["since"]),
        by=NodePauseBy(held["by"]),
        runtimes=list(held["runtimes"]),
        trayChannel=channel,
    )


def _result(name: str, state: NodePauseRuntimeState, detail: str | None = None) -> Any:
    return NodePauseRuntimeResult(name=name, state=state, detail=detail)


async def _cancel_measurements(app: Any) -> list[str]:
    """Cancel the measurement running here, if any (TRAY-4).

    Returns the runtimes on this machine it had stopped, which join the
    pause. Its own restart runs as it ends, finds the machine paused, and
    starts nothing (`measurement_node.restart_stopped`).
    """
    held: list[str] = []
    for attribute in ("benchmarks", "profile_builds"):
        manager = getattr(app.state, attribute, None)
        if manager is None or not manager.active or not manager.jobs:
            continue
        job = manager.jobs[-1]
        held += [r.name for r in job.restarts or [] if r.state == MeasurementRestartState.pending]
        log.info("Pause cancels the running %s %s", attribute.replace("_", " "), job.id)
        try:
            await asyncio.wait_for(asyncio.shield(manager.cancel(job.id)), CANCEL_SECONDS)
        except TimeoutError:
            log.warning("The %s did not end within %.0f s of Pause", attribute, CANCEL_SECONDS)
    serving = getattr(app.state, "serving_measurements", None)
    if serving is not None and serving.active and serving.task is not None:
        job = next((j for j in reversed(serving.jobs) if j.state == "running"), None)
        if job is not None:
            held += [
                r.runtime
                for r in job.restarts
                if r.state == MeasurementRestartState.pending and r.node == job.node
            ]
        log.info("Pause cancels the running serving measurement")
        serving.task.cancel()
        await asyncio.wait({serving.task}, timeout=CANCEL_SECONDS)
    return held


async def pause(app: Any, *, by: str) -> NodePauseResult:
    """Stop every model on this machine, and keep it stopped (row 35)."""
    from .measurement_node import running_runtimes
    from .node_work import launch_lock
    from .routes.runtimes import _supervisor

    context = NodeContext(app=app)
    state = app.state.agent_state
    async with launch_lock(context):
        if state.pause_record() is not None:
            return NodePauseResult(pause=view(app), results=[])
        # Recorded first. A measurement cancelled below then starts nothing
        # again, and a crash part-way leaves the machine paused rather than
        # half-stopped: the boot holds what a boot would start.
        started = new_record(by=by, runtimes=[])
        state.set_pause(started)
        held = await _cancel_measurements(app)
        supervisor = _supervisor(context)
        running = running_runtimes(context)
        results: list[NodePauseRuntimeResult] = []
        for name in running:
            try:
                if supervisor is not None:
                    await supervisor.stop_one(name, reason=StopReason.paused)
                results.append(_result(name, NodePauseRuntimeState.stopped))
            except Exception as exc:
                log.exception("Pause could not stop runtime %r", name)
                results.append(
                    _result(name, NodePauseRuntimeState.failed, f"Could not stop it: {exc}")
                )
        for name in held:
            if name not in running:
                results.append(
                    _result(
                        name,
                        NodePauseRuntimeState.stopped,
                        "A measurement had stopped it; it stays stopped.",
                    )
                )
        names = list(dict.fromkeys([*running, *held]))
        state.set_pause({**started, "runtimes": names})
        log.info(
            "Models on this machine paused from the %s: %s", by, ", ".join(names) or "none running"
        )
    return NodePauseResult(pause=view(app), results=results)


async def resume(app: Any) -> NodePauseResult:
    """Start again what Pause stopped, through admission, never forced."""
    from .measurement_node import restart_stopped
    from .node_work import launch_lock

    context = NodeContext(app=app)
    state = app.state.agent_state
    async with launch_lock(context):
        held = state.pause_record()
        if held is None:
            return NodePauseResult(pause=view(app), results=[])
        state.set_pause(None)
        names = list(held["runtimes"])
        restarted = await restart_stopped(context, names, enabled=True, hold_lock=False)
        results: list[NodePauseRuntimeResult] = []
        for outcome in restarted:
            if outcome.state == MeasurementRestartState.restarted:
                results.append(_result(outcome.name, NodePauseRuntimeState.started))
                continue
            if state.get_runtime_spec(outcome.name) is None:
                results.append(
                    _result(outcome.name, NodePauseRuntimeState.gone, outcome.detail or None)
                )
                continue
            # It stays stopped, as a person's stop: it reads so, and a reboot
            # does not start it behind the refusal's back.
            with contextlib.suppress(Exception):
                state.remember_stop(outcome.name)
            refused = outcome.state == MeasurementRestartState.refused
            results.append(
                _result(
                    outcome.name,
                    NodePauseRuntimeState.refused if refused else NodePauseRuntimeState.failed,
                    outcome.detail,
                )
            )
        log.info(
            "Models on this machine resumed: %d of %d started",
            sum(r.state is NodePauseRuntimeState.started for r in results),
            len(results),
        )
    return NodePauseResult(pause=view(app), results=results)


async def answer_tray(app: Any, verb: str) -> dict[str, Any]:
    """One answer to the tray's channel (`tray_pipe`): `state`, `pause`, `resume`."""
    if verb == "state":
        return {"ok": True, "pause": view(app).model_dump(mode="json", exclude_none=True)}
    if verb == "pause":
        done = await pause(app, by=NodePauseBy.tray.value)
    elif verb == "resume":
        done = await resume(app)
    else:
        return {"ok": False, "error": "unknown verb"}
    return {"ok": True, **done.model_dump(mode="json", exclude_none=True)}
