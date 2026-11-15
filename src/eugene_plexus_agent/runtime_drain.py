"""The runtime drain (row 13): the agent holds it, every gateway obeys it.

A runtime is taken out of routing before it restarts or stops, and its
answers finish first. The design is `specs/docs/design/rows/row-13.md` §2.1:

1. **Start.** `POST /v1/runtimes/{name}/drain` takes the operator, the agent's
   own rollout, or a gateway token, as `stop` does.
2. **Leave routing.** The runtime reports `status: draining`. Gateways route
   only to `ready` runtimes, so an older gateway obeys too. No agent calls a
   gateway, so no new grant is needed.
3. **Report.** Each gateway posts `.../drain/report` on every refresh while a
   drain is open (every 2 s then): its own count of answers running on the
   runtime, and the drain state its routing table holds.
4. **Drained.** The real subject: every gateway that read `GET /v1/runtimes`
   in the last 330 s reported `running: 0`, after the drain began, from a table
   that already shows the runtime draining. No reader: drained at once. A
   reader silent for 330 s is no longer waited for, and the message names it.
5. **Grace.** After `graceSeconds` the drain is `cut` and `then` runs anyway:
   "2 answers were cut at the 120 s limit."
6. **Then.** `restart` (on the build in use for it), `stop`, or `hold` until
   `DELETE` cancels the drain.

Two waits come before it leaves routing. `when: idle`: every gateway reports
it idle for `idleSeconds` with nothing running. `when: covered` (DI-C3, Troy
2026-10-21): every gateway reports that the model's other ready copies can
carry its current demand; a model with no other running copy goes at once
(DI-C3b).

**Hooks for later rows.** `before_stop` runs before `then` restarts or stops
the engine: where row 15 saves its slots.

Drains are not saved: an agent restart restarts the engines anyway, and a
rollout (which is saved) asks again.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ._generated.models import (
    RuntimeDrain,
    RuntimeDrainReason,
    RuntimeDrainReport,
    RuntimeDrainRequest,
    RuntimeDrainState,
    RuntimeDrainThen,
    RuntimeDrainWhen,
    StopReason,
)

log = logging.getLogger(__name__)

#: A gateway that read this agent's runtimes this recently is waited on.
READER_SECONDS = 330.0
# Longer than the longest `routingRefreshSeconds` a gateway takes (300 s), so a
# gateway that reads at its slowest is still waited on (row 13 review #6). The
# same span bounds a gateway that reads but never reports.
#: How often an open drain looks at its reports.
TICK_SECONDS = 0.25
#: A report older than this does not answer whether a runtime may leave now.
FRESH_SECONDS = 10.0
#: How a drain ends when its runtime was stopped by someone else meanwhile; its
#: `then` does not run.
STOPPED_MEANWHILE = "It was stopped while it drained, so nothing more was done."
#: The states in which the runtime is out of routing.
LEAVING = frozenset({RuntimeDrainState.draining, RuntimeDrainState.drained, RuntimeDrainState.cut})
#: The table states a gateway reports from which its `running: 0` counts.
SEEN_LEAVING = frozenset({"draining", "drained", "cut"})


class DrainError(Exception):
    """A drain cannot start, take a report, or be cancelled; `status` is the
    HTTP status for it."""

    def __init__(self, detail: str, status: int = 409) -> None:
        super().__init__(detail)
        self.status = status


def _who(issuer: str) -> str:
    """`node:tower` as a person says it: "the gateway on tower"."""
    node = issuer.split(":", 1)[1] if issuer.startswith("node:") else issuer
    return f"the gateway on {node}"


def _answers(count: int) -> str:
    return f"{count} answer{'s' if count != 1 else ''}"


@dataclass
class _Report:
    at: float
    body: RuntimeDrainReport


@dataclass(eq=False)
class _Drain:
    name: str
    reason: RuntimeDrainReason
    when: RuntimeDrainWhen
    idle_seconds: int
    grace_seconds: int
    then: RuntimeDrainThen
    requested_at: datetime
    requested: float
    state: RuntimeDrainState
    started_at: datetime | None = None
    started: float | None = None
    deadline: float | None = None
    reports: dict[str, _Report] = field(default_factory=dict)
    #: The gateways it waits on, as of the last look, and those it stopped
    #: waiting on because they went silent.
    waiting_on: set[str] = field(default_factory=set)
    dropped: set[str] = field(default_factory=set)
    #: Of those, the ones that read the runtimes but never reported.
    mute: set[str] = field(default_factory=set)
    idle_for: float | None = None
    other_copies: int | None = None
    cut: int = 0
    message: str | None = None
    finished: asyncio.Future[RuntimeDrain] | None = None
    task: asyncio.Task[None] | None = None
    #: Runs as it leaves routing and may change `then` (the rollout's: restart
    #: on the new build at once, or stop it to measure).
    on_leave: Callable[[RuntimeDrain], RuntimeDrainThen | None] | None = None


class RuntimeDrains:
    """The drains this agent holds, and the gateways that read its runtimes."""

    def __init__(
        self,
        *,
        run_then: Callable[[str, RuntimeDrainThen, RuntimeDrainReason], Awaitable[None]],
        is_running: Callable[[str], bool],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._run_then = run_then
        self._is_running = is_running
        self._clock = clock
        #: Gateway issuer -> when it last read `/v1/runtimes` (this clock).
        self.readers: dict[str, float] = {}
        self._drains: dict[str, _Drain] = {}
        #: Row 15: runs before `then` restarts or stops the engine.
        self.before_stop: Callable[[str, RuntimeDrain], Awaitable[None]] | None = None

    # --- the gateways ------------------------------------------------------

    def saw_reader(self, issuer: str) -> None:
        """A gateway (`sub: gateway`, this issuer) read this agent's runtimes."""
        self.readers[issuer] = self._clock()

    def live_readers(self) -> set[str]:
        now = self._clock()
        for issuer in [i for i, at in self.readers.items() if now - at > READER_SECONDS]:
            del self.readers[issuer]
        return set(self.readers)

    # --- reading -----------------------------------------------------------

    def view(self, name: str) -> RuntimeDrain | None:
        """The drain open on `name`, as `Runtime.drain` shows it."""
        drain = self._drains.get(name)
        return self._view(drain) if drain is not None else None

    def leaving(self, name: str) -> bool:
        """Whether `name` is out of routing (`status: draining`)."""
        drain = self._drains.get(name)
        return drain is not None and drain.state in LEAVING

    def _running(self, drain: _Drain) -> int:
        """Answers still running, by the last report of each gateway waited on."""
        return sum(drain.reports[i].body.running for i in drain.waiting_on if i in drain.reports)

    def _view(self, drain: _Drain) -> RuntimeDrain:
        return RuntimeDrain(
            reason=drain.reason,
            when=drain.when,
            idleSeconds=drain.idle_seconds,
            graceSeconds=drain.grace_seconds,
            then=drain.then,
            state=drain.state,
            requestedAt=drain.requested_at,
            startedAt=drain.started_at,
            deadline=(
                drain.started_at + timedelta(seconds=drain.grace_seconds)
                if drain.started_at is not None
                else None
            ),
            running=self._running(drain),
            gateways=len(drain.waiting_on),
            idleFor=drain.idle_for,
            otherCopies=drain.other_copies,
            message=drain.message,
        )

    # --- changing ----------------------------------------------------------

    def start(
        self,
        name: str,
        request: RuntimeDrainRequest | None = None,
        *,
        on_leave: Callable[[RuntimeDrain], RuntimeDrainThen | None] | None = None,
    ) -> RuntimeDrain:
        """Open a drain on a running runtime. Raises `DrainError` (409).

        `on_leave` runs once, as it leaves routing, with what the gateways
        reported then; a `then` it returns replaces the request's."""
        if name in self._drains:
            raise DrainError(f"{name} is already draining. Cancel that drain first.")
        if not self._is_running(name):
            raise DrainError(f"{name} is not running, so there is nothing to drain.")
        request = request or RuntimeDrainRequest()
        when = request.when or RuntimeDrainWhen.now
        now = self._clock()
        drain = _Drain(
            name=name,
            reason=request.reason or RuntimeDrainReason.operator,
            when=when,
            idle_seconds=int(request.idleSeconds if request.idleSeconds is not None else 600),
            grace_seconds=int(request.graceSeconds if request.graceSeconds is not None else 120),
            then=request.then or RuntimeDrainThen.hold,
            requested_at=datetime.now(UTC),
            requested=now,
            state=(
                RuntimeDrainState.waiting_idle
                if when is RuntimeDrainWhen.idle
                else RuntimeDrainState.waiting_covered
                if when is RuntimeDrainWhen.covered
                else RuntimeDrainState.draining
            ),
            finished=asyncio.get_running_loop().create_future(),
            on_leave=on_leave,
        )
        drain.waiting_on = self.live_readers()
        if drain.state is RuntimeDrainState.draining:
            self._leave(drain)
        else:
            drain.message = (
                f"Waiting until it has been idle {drain.idle_seconds} s."
                if when is RuntimeDrainWhen.idle
                else "Waiting until its other copies can carry the load."
            )
        self._drains[name] = drain
        drain.task = asyncio.create_task(self._watch(drain), name=f"drain:{name}")
        log.info(
            "runtime %s: drain opened (%s, when %s, then %s)",
            name,
            drain.reason.value,
            drain.when.value,
            drain.then.value,
        )
        return self._view(drain)

    def report(self, name: str, issuer: str, body: RuntimeDrainReport) -> RuntimeDrain:
        drain = self._drains.get(name)
        if drain is None:
            raise DrainError(f"No drain is open on {name}.")
        drain.reports[issuer] = _Report(at=self._clock(), body=body)
        self.saw_reader(issuer)
        self._evaluate(drain)
        return self._view(drain)

    async def cancel(self, name: str) -> RuntimeDrain:
        """Back to routing. A drain whose `then` already ran cannot be cancelled."""
        drain = self._drains.get(name)
        if drain is None:
            raise DrainError(f"No drain is open on {name}.")
        if drain.then is not RuntimeDrainThen.hold and drain.state in (
            RuntimeDrainState.drained,
            RuntimeDrainState.cut,
        ):
            # Its `then` runs (or ran): stopping it halfway would leave the
            # model stopped while saying it is back in routing (review #7).
            doing = "restarting" if drain.then is RuntimeDrainThen.restart else "stopping"
            raise DrainError(f"The drain on {name} has ended, and it is {doing} now.")
        drain.state = RuntimeDrainState.cancelled
        drain.message = "Cancelled. It is back in routing."
        if drain.task is not None and not drain.task.done():
            drain.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await drain.task
        self._finish(drain)
        return self._view(drain)

    async def wait(self, name: str) -> RuntimeDrain:
        """Until the drain on `name` has ended (its `then` ran, or it was
        cancelled); how it ended. A `hold` ends when it is drained or cut."""
        drain = self._drains.get(name)
        if drain is None or drain.finished is None:
            raise DrainError(f"No drain is open on {name}.")
        return await asyncio.shield(drain.finished)

    async def aclose(self) -> None:
        for drain in list(self._drains.values()):
            if drain.task is not None and not drain.task.done():
                drain.task.cancel()
                with contextlib.suppress(asyncio.CancelledError, BaseException):
                    await drain.task
            if drain.finished is not None and not drain.finished.done():
                drain.finished.cancel()
        self._drains.clear()

    # --- the drain itself ----------------------------------------------------

    def _leave(self, drain: _Drain) -> None:
        now = self._clock()
        drain.state = RuntimeDrainState.draining
        drain.started = now
        drain.started_at = datetime.now(UTC)
        drain.deadline = now + drain.grace_seconds
        drain.waiting_on = self.live_readers() - drain.dropped
        fresh = [r.body for i, r in drain.reports.items() if i in drain.waiting_on]
        idle = [b.idleSeconds for b in fresh if b.idleSeconds is not None]
        copies = [b.otherCopies for b in fresh if b.otherCopies is not None]
        drain.idle_for = min(idle) if idle else None
        drain.other_copies = min(copies) if copies else None
        drain.message = self._waiting_words(drain)
        if drain.on_leave is not None:
            hook, drain.on_leave = drain.on_leave, None
            try:
                then = hook(self._view(drain))
            except Exception:
                log.exception("runtime %s: the drain's leave hook failed", drain.name)
                then = None
            if then is not None:
                drain.then = then

    def _waiting_words(self, drain: _Drain) -> str:
        if not drain.waiting_on:
            return "No gateway routes to it."
        running = self._running(drain)
        return f"Out of routing; {_answers(running)} running."

    def _fresh(self, drain: _Drain, issuer: str, since: float) -> RuntimeDrainReport | None:
        report = drain.reports.get(issuer)
        if report is None or report.at < since or self._clock() - report.at > FRESH_SECONDS:
            return None
        return report.body

    def _evaluate(self, drain: _Drain) -> bool:
        """Move the drain on if its subject allows. Whether it has ended."""
        if drain.state in (RuntimeDrainState.cancelled,):
            return True
        live = self.live_readers()
        gone = drain.waiting_on - live
        if gone:
            drain.dropped |= gone
            drain.waiting_on &= live
        if drain.state in (RuntimeDrainState.waiting_idle, RuntimeDrainState.waiting_covered):
            # A gateway that reads the runtimes but has not reported once in
            # 330 s (one from before row 13) cannot say idle or covered: it is
            # not waited for, and the message names it, so a wait never
            # outlives the gateways that can end it.
            mute = {
                i
                for i in live
                if i not in drain.reports and self._clock() - drain.requested > READER_SECONDS
            }
            drain.mute |= mute
            drain.dropped |= mute
            live = live - drain.dropped
        if drain.state is RuntimeDrainState.waiting_idle:
            drain.waiting_on = live
            if all(
                (body := self._fresh(drain, i, drain.requested)) is not None
                and body.running == 0
                and (body.idleSeconds or 0) >= drain.idle_seconds
                for i in live
            ):
                self._leave(drain)
            return False
        if drain.state is RuntimeDrainState.waiting_covered:
            drain.waiting_on = live
            if all(
                (body := self._fresh(drain, i, drain.requested)) is not None
                and (body.otherCopies == 0 or bool(body.covered))
                for i in live
            ):
                self._leave(drain)
            return False
        if drain.state is not RuntimeDrainState.draining:
            return True
        assert drain.started is not None and drain.deadline is not None
        done = all(
            (r := drain.reports.get(i)) is not None
            and r.at >= drain.started
            and r.body.running == 0
            and (r.body.state is not None and r.body.state.value in SEEN_LEAVING)
            for i in drain.waiting_on
        )
        if done:
            drain.state = RuntimeDrainState.drained
            drain.message = "Every answer finished." + self._dropped_words(drain)
            return True
        if self._clock() >= drain.deadline:
            drain.cut = self._running(drain)
            drain.state = RuntimeDrainState.cut
            drain.message = (
                f"{_answers(drain.cut).capitalize()} "
                f"{'were' if drain.cut != 1 else 'was'} cut at the {drain.grace_seconds} s limit."
                if drain.cut
                else f"A gateway did not confirm within the {drain.grace_seconds} s limit."
            ) + self._dropped_words(drain)
            return True
        drain.message = self._waiting_words(drain)
        return False

    def _dropped_words(self, drain: _Drain) -> str:
        words = ""
        silent = drain.dropped - drain.mute
        if silent:
            names = ", ".join(sorted(_who(i) for i in silent))
            words += f" Not waited for: {names}, which stopped reading for {READER_SECONDS:.0f} s."
        if drain.mute:
            names = ", ".join(sorted(_who(i) for i in drain.mute))
            words += f" Not waited for: {names}, which never reported (update it)."
        return words

    async def _watch(self, drain: _Drain) -> None:
        try:
            while not self._evaluate(drain):
                if not self._is_running(drain.name):
                    # Stopped meanwhile (an idle unload, an operator's stop):
                    # nothing runs on it, so there is nothing left to wait for,
                    # and its `then` does not run: whoever stopped it decided
                    # (row 13 review #8).
                    drain.state = RuntimeDrainState.drained
                    drain.message = STOPPED_MEANWHILE
                    log.info("runtime %s: %s", drain.name, drain.message)
                    self._finish(drain)
                    return
                await asyncio.sleep(TICK_SECONDS)
            if drain.state is RuntimeDrainState.cancelled:
                return
            log.info("runtime %s: drain %s: %s", drain.name, drain.state.value, drain.message)
            if drain.then is RuntimeDrainThen.hold:
                self._resolve(drain)
                return
            if self.before_stop is not None:
                with contextlib.suppress(Exception):
                    await self.before_stop(drain.name, self._view(drain))
            try:
                await self._run_then(drain.name, drain.then, drain.reason)
            except Exception as exc:
                log.exception("runtime %s: the drain's %s failed", drain.name, drain.then.value)
                drain.message = (drain.message or "") + f" Then {drain.then.value} failed: {exc}"
            self._finish(drain)
        except asyncio.CancelledError:
            raise

    def _resolve(self, drain: _Drain) -> None:
        if drain.finished is not None and not drain.finished.done():
            drain.finished.set_result(self._view(drain))

    def _finish(self, drain: _Drain) -> None:
        self._resolve(drain)
        if self._drains.get(drain.name) is drain:
            del self._drains[drain.name]


def stop_reason_for(reason: RuntimeDrainReason) -> StopReason:
    """How a drain's `then: stop` is recorded: an update or a check stops it
    for a while (`measurement`, not remembered across restarts); an operator
    or a machine's maintenance stops it until someone starts it."""
    if reason in (RuntimeDrainReason.update, RuntimeDrainReason.check):
        return StopReason.measurement
    return StopReason.operator


def of(app: Any) -> RuntimeDrains:
    """The app's drains, made on first use and wired into its runtime
    supervisor so `GET /v1/runtimes` shows them."""
    found = getattr(app.state, "runtime_drains", None)
    if isinstance(found, RuntimeDrains):
        supervisor = getattr(app.state, "runtime_supervisor", None)
        if supervisor is not None and getattr(supervisor, "drains", None) is not found:
            supervisor.drains = found
        return found
    from .runtime_context import NodeContext

    def is_running(name: str) -> bool:
        supervisor = getattr(app.state, "runtime_supervisor", None)
        return bool(supervisor is not None and supervisor.is_running(name))

    async def run_then(name: str, then: RuntimeDrainThen, reason: RuntimeDrainReason) -> None:
        from .measurement_node import restart_stopped

        supervisor = getattr(app.state, "runtime_supervisor", None)
        if supervisor is None:
            return
        if then is RuntimeDrainThen.stop:
            await supervisor.stop_one(name, reason=stop_reason_for(reason))
            return
        # Restart: a launch, so it takes the launch lock and admission as the
        # Start button does, and it starts on the build in use for it.
        await supervisor.stop_one(name, reason=StopReason.measurement)
        result = (await restart_stopped(NodeContext(app), [name], enabled=True))[0]
        if result.state.value != "restarted":
            raise RuntimeError(result.detail or "it did not start again")

    drains = RuntimeDrains(run_then=run_then, is_running=is_running)
    app.state.runtime_drains = drains
    supervisor = getattr(app.state, "runtime_supervisor", None)
    if supervisor is not None:
        supervisor.drains = drains
    return drains


__all__ = ["DrainError", "RuntimeDrains", "of", "stop_reason_for"]
