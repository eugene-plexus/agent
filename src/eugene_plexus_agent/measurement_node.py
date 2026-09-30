"""The step every job that measures a node shares: ask, stop, restart.

A benchmark or a profile build needs the node's memory to itself. R6.1
refused while anything was running, which sent the operator to another
page to stop their models and back again. Since 2026-09-30 (Troy) the
page asks instead, and this module keeps the three promises that makes:

1. **Only what the operator agreed to is stopped.** The request carries
   the names the preflight showed them (`stopRuntimes`). A runtime that is
   running and NOT in that list refuses the job — a model started between
   the question and the click was never asked about.
2. **What was stopped is started again** when the job ends, whatever way
   it ends, through admission exactly as the Start button does, and never
   forced: a refusal is reported by name.
3. **Nothing is stopped implicitly.** A request with an empty list while
   something runs is the same 409 R6.1 always gave.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from fastapi import HTTPException, Request

from ._generated.models import (
    AdmissionDecision,
    MeasurementRestart,
    MeasurementRestartState,
    RuntimeStatus,
    StopReason,
)

log = logging.getLogger(__name__)

# The app.state attributes that hold measurement-job managers, and how a
# refusal names each. One measurement at a time per node, of either kind.
_MANAGERS = (("benchmarks", "A benchmark"), ("profile_builds", "A settings build"))


def active_measurement(app) -> str | None:  # type: ignore[no-untyped-def]
    """The kind of measurement job running on this node, or None."""
    for attribute, label in _MANAGERS:
        manager = getattr(app.state, attribute, None)
        if manager is not None and manager.active:
            return label
    return None


def running_runtimes(request: Request) -> list[str]:
    """Runtimes on this node that are not stopped: the preflight's question."""
    from .routes.runtimes import _compose, _supervisor

    state = request.app.state.agent_state
    supervisor = _supervisor(request)
    return [
        spec.name
        for spec in state.list_runtime_specs()
        if _compose(spec, supervisor).status != RuntimeStatus.stopped
    ]


def unlisted(running: Iterable[str], agreed: Iterable[str]) -> list[str]:
    """Running runtimes the operator did not agree to stop."""
    allowed = set(agreed)
    return [name for name in running if name not in allowed]


def refusal_for(names: Sequence[str], agreed: Sequence[str]) -> HTTPException:
    """The 409 for running models nobody agreed to stop."""
    listed = ", ".join(names)
    if agreed:
        detail = (
            f"{listed} started after you were asked, so it was not stopped. "
            "Check the list again and agree to stopping it, or stop it first."
        )
    else:
        detail = (
            f"Models are running on this node: {listed}. Agree to stopping them "
            "for this measurement (they start again when it ends), or stop them first."
        )
    return HTTPException(409, detail)


async def stop_agreed(request: Request, names: Sequence[str]) -> list[str]:
    """Stop exactly `names` for a measurement; returns those actually stopped."""
    from .routes.runtimes import _supervisor

    supervisor = _supervisor(request)
    stopped: list[str] = []
    if supervisor is None:
        return stopped
    for name in names:
        await supervisor.stop_one(name, reason=StopReason.measurement)
        stopped.append(name)
        log.info("Stopped runtime %r for a measurement, with the operator's agreement", name)
    return stopped


def pending(names: Sequence[str]) -> list[MeasurementRestart]:
    """The restart record a job starts with: every stopped runtime pending."""
    return [MeasurementRestart(name=n, state=MeasurementRestartState.pending) for n in names]


async def restart_stopped(
    request: Request, names: Sequence[str], *, enabled: bool, hold_lock: bool = True
) -> list[MeasurementRestart]:
    """Start again what a measurement stopped, as the Start button would.

    Holds the launch lock so a restart is serialised with every other
    launch on the node, runs admission (a node whose memory changed while
    the job ran gets a refusal, not an OOM), reserves what it starts, and
    reports each runtime's fate. Never raises: a job's end must not fail
    because one restart did.

    `hold_lock=False` is for a caller already holding the launch lock (a
    start route undoing its own stops after a refusal): asyncio's lock is
    not re-entrant, and taking it again there would wait forever.
    """
    import contextlib

    from .node_work import launch_lock
    from .routes.runtimes import _admission_for, _reserve, _supervisor

    results: list[MeasurementRestart] = []
    if not names:
        return results
    if not enabled:
        return [
            MeasurementRestart(
                name=n,
                state=MeasurementRestartState.skipped,
                detail="Not started again: the request asked to leave it stopped.",
            )
            for n in names
        ]
    state = request.app.state.agent_state
    guard = launch_lock(request) if hold_lock else contextlib.nullcontext()
    async with guard:
        for name in names:
            try:
                spec = state.get_runtime_spec(name)
                supervisor = _supervisor(request)
                if spec is None:
                    results.append(
                        _result(name, "failed", "It is no longer declared on this node.")
                    )
                    continue
                if supervisor is None:
                    results.append(
                        _result(name, "failed", "This agent is not supervising runtimes.")
                    )
                    continue
                if supervisor.is_running(name):
                    results.append(_result(name, "restarted", "It was already running again."))
                    continue
                admission = await _admission_for(request, spec)
                if admission.decision is AdmissionDecision.refuse:
                    results.append(_result(name, "refused", admission.reason or "Memory refused."))
                    continue
                supervisor.add_and_start(spec.model_copy(update={"autoStart": True}))
                _reserve(request, spec, admission)
                results.append(_result(name, "restarted", "Started again."))
            except Exception as exc:
                log.exception("Could not restart runtime %r after a measurement", name)
                results.append(_result(name, "failed", f"Could not start it again: {exc}"))
    return results


def _result(name: str, state: str, detail: str) -> MeasurementRestart:
    return MeasurementRestart(name=name, state=MeasurementRestartState(state), detail=detail)
