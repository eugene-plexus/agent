"""Row 13: the runtime drain. The agent holds it; every gateway obeys it.

The drain waits on its real subject: every gateway that read this agent's
runtimes in the last 60 s reports `running: 0`, after the drain began, from a
table that already shows the runtime draining. Then `then` runs.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import runtime_drain
from eugene_plexus_agent._generated.models import (
    RuntimeDrainReason,
    RuntimeDrainReport,
    RuntimeDrainRequest,
    RuntimeDrainState,
    RuntimeDrainThen,
    RuntimeDrainWhen,
    RuntimeSpec,
    RuntimeStatus,
)
from eugene_plexus_agent.runtimes import RuntimeSupervisor

from .conftest import TEST_PASSPHRASE, StubRuntimeSupervisor, local_service_token

GW = "node:tower"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _fast_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_drain, "TICK_SECONDS", 0.01)


def _drains(clock: Clock, running: set[str] | None = None) -> tuple[Any, list[tuple[str, str]]]:
    ran: list[tuple[str, str]] = []
    alive = running if running is not None else {"r"}

    async def run_then(name: str, then: RuntimeDrainThen, reason: RuntimeDrainReason) -> None:
        ran.append((name, then.value))

    drains = runtime_drain.RuntimeDrains(
        run_then=run_then, is_running=lambda n: n in alive, clock=clock
    )
    return drains, ran


def _report(running: int, state: str | None = "draining", **extra: Any) -> RuntimeDrainReport:
    return RuntimeDrainReport.model_validate({"running": running, "state": state, **extra})


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0.02)


async def test_drain_waits_for_the_gateway_report() -> None:
    """`then: restart` runs only after the gateway that read the runtime
    reports 0, from a table that shows it draining."""
    clock = Clock()
    drains, ran = _drains(clock)
    drains.saw_reader(GW)
    view = drains.start(
        "r", RuntimeDrainRequest(reason=RuntimeDrainReason.update, then=RuntimeDrainThen.restart)
    )
    assert view.state is RuntimeDrainState.draining and view.gateways == 1
    assert drains.leaving("r")
    await _settle()
    assert ran == [], "nothing reported yet"

    drains.report("r", GW, _report(2))
    await _settle()
    assert ran == [] and drains.view("r").running == 2

    # A table a moment stale still shows it waiting: its 0 does not count.
    drains.report("r", GW, _report(0, state="waiting_covered"))
    await _settle()
    assert ran == []

    drains.report("r", GW, _report(0))
    final = await asyncio.wait_for(drains.wait("r"), 2)
    assert ran == [("r", "restart")]
    assert final.state is RuntimeDrainState.drained
    assert drains.view("r") is None, "the drain closes once its then ran"


async def test_a_report_from_before_the_drain_began_does_not_count() -> None:
    clock = Clock()
    drains, ran = _drains(clock)
    drains.saw_reader(GW)
    drains.start(
        "r", RuntimeDrainRequest(when=RuntimeDrainWhen.covered, then=RuntimeDrainThen.stop)
    )
    drains.report("r", GW, _report(0, state="waiting_covered", otherCopies=1, covered=True))
    await _settle()
    assert drains.view("r").state is RuntimeDrainState.draining
    await _settle()
    assert ran == [], "the report that let it leave is not the one that says it is empty"


async def test_no_gateway_reads_it_so_it_drains_at_once() -> None:
    drains, ran = _drains(Clock())
    drains.start("r", RuntimeDrainRequest(then=RuntimeDrainThen.stop))
    final = await asyncio.wait_for(drains.wait("r"), 2)
    assert ran == [("r", "stop")]
    assert final.gateways == 0


async def test_cut_at_grace_names_count() -> None:
    clock = Clock()
    drains, ran = _drains(clock)
    drains.saw_reader(GW)
    drains.start("r", RuntimeDrainRequest(graceSeconds=120, then=RuntimeDrainThen.restart))
    drains.report("r", GW, _report(2))
    await _settle()
    clock.now += 119
    drains.saw_reader(GW)
    await _settle()
    assert ran == []
    clock.now += 2
    drains.saw_reader(GW)
    final = await asyncio.wait_for(drains.wait("r"), 2)
    assert final.state is RuntimeDrainState.cut
    assert final.message == "2 answers were cut at the 120 s limit."
    assert ran == [("r", "restart")], "then runs anyway: an update must end"


async def test_a_gateway_with_the_longest_refresh_is_still_waited_for() -> None:
    """Review #6: a gateway reads the runtimes every `routingRefreshSeconds`,
    up to 300 s. One that read 200 s before the drain began is still a reader,
    so the drain waits for its report rather than ending at once."""
    clock = Clock()
    drains, ran = _drains(clock)
    drains.saw_reader(GW)
    clock.now += 200
    view = drains.start("r", RuntimeDrainRequest(then=RuntimeDrainThen.stop))
    assert view.gateways == 1, "the gateway is waited on"
    await _settle()
    assert ran == [], "not drained until it reports"
    drains.report("r", GW, _report(0))
    await asyncio.wait_for(drains.wait("r"), 2)
    assert ran == [("r", "stop")]


async def test_a_silent_gateway_is_not_waited_for_and_is_named() -> None:
    clock = Clock()
    drains, ran = _drains(clock)
    drains.saw_reader(GW)
    drains.saw_reader("node:attic")
    drains.start("r", RuntimeDrainRequest(then=RuntimeDrainThen.stop))
    drains.report("r", GW, _report(0))
    await _settle()
    assert ran == [], "attic has not reported"
    clock.now += runtime_drain.READER_SECONDS + 1
    drains.saw_reader(GW)
    drains.report("r", GW, _report(0))
    final = await asyncio.wait_for(drains.wait("r"), 2)
    assert ran == [("r", "stop")]
    assert "the gateway on attic" in (final.message or "")


async def test_covered_waits_until_the_other_copies_carry_the_load() -> None:
    """DI-C3: one copy leaves only while the others carry the current demand;
    a model with no other running copy goes at once (DI-C3b)."""
    clock = Clock()
    drains, _ran = _drains(clock, {"r", "s"})
    drains.saw_reader(GW)
    drains.start("r", RuntimeDrainRequest(when=RuntimeDrainWhen.covered))
    assert drains.view("r").state is RuntimeDrainState.waiting_covered
    assert not drains.leaving("r"), "still routed while it waits"
    drains.report("r", GW, _report(1, state="waiting_covered", otherCopies=1, covered=False))
    await _settle()
    assert drains.view("r").state is RuntimeDrainState.waiting_covered
    drains.report("r", GW, _report(1, state="waiting_covered", otherCopies=1, covered=True))
    await _settle()
    leaving = drains.view("r")
    assert leaving.state is RuntimeDrainState.draining and leaving.otherCopies == 1

    drains.start("s", RuntimeDrainRequest(when=RuntimeDrainWhen.covered))
    drains.report("s", GW, _report(3, state="waiting_covered", otherCopies=0, covered=False))
    await _settle()
    assert drains.view("s").state is RuntimeDrainState.draining, "one copy: at once"


async def test_a_gateway_that_never_reports_does_not_hold_a_wait_forever() -> None:
    """A gateway from before row 13 reads the runtimes but never reports. It
    cannot say covered or idle, so after 60 s it is not waited for, and the
    message names it: an update ends."""
    clock = Clock()
    drains, ran = _drains(clock)
    drains.saw_reader("node:attic")
    drains.start(
        "r", RuntimeDrainRequest(when=RuntimeDrainWhen.covered, then=RuntimeDrainThen.stop)
    )
    await _settle()
    assert drains.view("r").state is RuntimeDrainState.waiting_covered
    clock.now += runtime_drain.READER_SECONDS + 1
    drains.saw_reader("node:attic")  # still reading, never reporting
    final = await asyncio.wait_for(drains.wait("r"), 2)
    assert ran == [("r", "stop")]
    assert "the gateway on attic, which never reported" in (final.message or "")


async def test_on_leave_chooses_then_from_what_the_gateways_saw() -> None:
    """The rollout's hook: as it leaves routing, the drain's `then` may
    change (restart on the new build at once, or stop it to measure)."""
    clock = Clock()
    drains, ran = _drains(clock)
    drains.saw_reader(GW)
    seen: list[int | None] = []

    def on_leave(view: Any) -> RuntimeDrainThen:
        seen.append(view.otherCopies)
        return RuntimeDrainThen.restart

    drains.start(
        "r",
        RuntimeDrainRequest(when=RuntimeDrainWhen.covered, then=RuntimeDrainThen.stop),
        on_leave=on_leave,
    )
    drains.report("r", GW, _report(1, state="waiting_covered", otherCopies=0))
    await _settle()
    assert seen == [0] and drains.view("r").then is RuntimeDrainThen.restart
    drains.report("r", GW, _report(0))
    await asyncio.wait_for(drains.wait("r"), 2)
    assert ran == [("r", "restart")]


async def test_idle_waits_for_every_gateway_to_report_it_idle() -> None:
    clock = Clock()
    drains, _ran = _drains(clock)
    drains.saw_reader(GW)
    drains.start("r", RuntimeDrainRequest(when=RuntimeDrainWhen.idle, idleSeconds=600))
    drains.report("r", GW, _report(0, state="waiting_idle", idleSeconds=599))
    await _settle()
    assert drains.view("r").state is RuntimeDrainState.waiting_idle
    drains.report("r", GW, _report(0, state="waiting_idle", idleSeconds=600))
    await _settle()
    assert drains.view("r").state is RuntimeDrainState.draining
    assert drains.view("r").idleFor == 600


async def test_hold_stays_out_of_routing_until_cancelled() -> None:
    drains, ran = _drains(Clock())
    drains.start("r", RuntimeDrainRequest(then=RuntimeDrainThen.hold))
    held = await asyncio.wait_for(drains.wait("r"), 2)
    assert held.state is RuntimeDrainState.drained and ran == []
    assert drains.leaving("r")
    cancelled = await drains.cancel("r")
    assert cancelled.state is RuntimeDrainState.cancelled
    assert not drains.leaving("r") and drains.view("r") is None


async def test_a_second_drain_and_a_stopped_runtime_are_refused() -> None:
    drains, _ran = _drains(Clock(), {"r"})
    drains.saw_reader(GW)
    drains.start("r")
    with pytest.raises(runtime_drain.DrainError, match="already draining"):
        drains.start("r")
    with pytest.raises(runtime_drain.DrainError, match="not running"):
        drains.start("stopped")
    await drains.aclose()


def test_compose_says_draining_and_carries_the_drain() -> None:
    supervisor = RuntimeSupervisor()
    supervisor._status_for = lambda *a, **k: RuntimeStatus.ready  # type: ignore[method-assign]
    spec = RuntimeSpec.model_validate(
        {"name": "r", "engine": "llama_cpp", "modelPath": "/m/r.gguf", "port": 9001}
    )

    async def go() -> None:
        drains, _ran = _drains(Clock())
        supervisor.drains = drains
        drains.saw_reader(GW)
        drains.start("r", RuntimeDrainRequest(when=RuntimeDrainWhen.covered))
        waiting = supervisor.compose(spec)
        assert waiting.status is RuntimeStatus.ready and waiting.drain is not None
        drains.report("r", GW, _report(0, state="waiting_covered", otherCopies=0))
        await _settle()
        leaving = supervisor.compose(spec)
        assert leaving.status is RuntimeStatus.draining
        assert leaving.drain is not None and leaving.drain.state is RuntimeDrainState.draining
        await drains.aclose()

    asyncio.run(go())


# --- the routes ---------------------------------------------------------------------


class ReadyRuntimes(StubRuntimeSupervisor):
    """Started runtimes are `ready`, as a loaded engine is, so a drain can
    take one out of routing."""

    def compose(self, spec: RuntimeSpec) -> Any:
        if spec.name in self.started:
            self._status_for = lambda *a, **k: RuntimeStatus.ready  # type: ignore[method-assign]
        else:
            self._status_for = RuntimeSupervisor._status_for.__get__(self)  # type: ignore[method-assign]
        return RuntimeSupervisor.compose(self, spec)


@pytest.fixture
def ready_app(app: Any) -> Any:
    app.state.runtime_supervisor = ReadyRuntimes()
    return app


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_drain_routes_report_and_status(ready_app: Any) -> None:
    with TestClient(ready_app) as c:
        operator = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).json()[
            "sessionToken"
        ]
        made = c.post(
            "/v1/runtimes",
            json={
                "name": "r",
                "engine": "llama_cpp",
                "modelPath": "/m/r.gguf",
                "autoDriver": False,
            },
            headers=_bearer(operator),
        )
        assert made.status_code == 201, made.text
        gateway = local_service_token(ready_app, "gateway")
        # The gateway reads the runtimes: it is now waited on.
        listed = c.get("/v1/runtimes", headers=_bearer(gateway)).json()["runtimes"]
        assert listed[0]["status"] == "ready" and listed[0].get("drain") is None

        opened = c.post(
            "/v1/runtimes/r/drain",
            json={"reason": "update", "then": "hold"},
            headers=_bearer(gateway),
        )
        assert opened.status_code == 202, opened.text
        assert opened.json()["state"] == "draining" and opened.json()["gateways"] == 1
        runtime = c.get("/v1/runtimes/r", headers=_bearer(operator)).json()
        assert runtime["status"] == "draining" and runtime["drain"]["reason"] == "update"

        driver = local_service_token(ready_app, "inference-driver")
        for refused in (driver, operator):
            answer = c.post(
                "/v1/runtimes/r/drain/report",
                json={"running": 0, "state": "draining"},
                headers=_bearer(refused),
            )
            assert answer.status_code == 401, answer.text
        took = c.post(
            "/v1/runtimes/r/drain/report",
            json={"running": 0, "state": "draining"},
            headers=_bearer(gateway),
        )
        assert took.status_code == 200 and took.json()["state"] == "drained", took.text
        assert c.post("/v1/runtimes/r/drain", headers=_bearer(operator)).status_code == 409

        back = c.delete("/v1/runtimes/r/drain", headers=_bearer(operator))
        assert back.status_code == 200 and back.json()["state"] == "cancelled"
        assert c.get("/v1/runtimes/r", headers=_bearer(operator)).json()["status"] == "ready"
        assert c.delete("/v1/runtimes/r/drain", headers=_bearer(operator)).status_code == 409


async def test_a_drain_cannot_be_cancelled_once_its_then_runs() -> None:
    """Review #7: a cancel between the stop and the start of `then: restart`
    said "back in routing" and left the model stopped. Once the drain has
    ended and its `then` runs, a cancel is refused, and the restart finishes."""
    stopped = asyncio.Event()
    release = asyncio.Event()
    running = {"r"}
    log: list[str] = []

    async def run_then(name: str, then: RuntimeDrainThen, reason: RuntimeDrainReason) -> None:
        running.discard(name)
        log.append("stopped")
        stopped.set()
        await release.wait()  # the launch: admission, the lock, the model load
        running.add(name)
        log.append("started")

    drains = runtime_drain.RuntimeDrains(run_then=run_then, is_running=lambda n: n in running)
    drains.start("r", RuntimeDrainRequest(then=RuntimeDrainThen.restart, graceSeconds=0))
    await asyncio.wait_for(stopped.wait(), 2)
    with pytest.raises(runtime_drain.DrainError, match="restarting"):
        await drains.cancel("r")
    release.set()
    final = await asyncio.wait_for(drains.wait("r"), 2)
    assert log == ["stopped", "started"] and running == {"r"}
    assert final.state is RuntimeDrainState.drained


async def test_a_runtime_stopped_by_someone_else_is_not_restarted_by_its_drain() -> None:
    """Review #8: an operator's Stop (or an idle unload) while a drain waits
    ends the drain, and its `then: restart` does not start the model again."""
    alive = {"r"}
    drains, ran = _drains(Clock(), alive)
    drains.saw_reader(GW)
    drains.start(
        "r", RuntimeDrainRequest(when=RuntimeDrainWhen.covered, then=RuntimeDrainThen.restart)
    )
    await _settle()
    alive.discard("r")  # the operator's Stop
    final = await asyncio.wait_for(drains.wait("r"), 2)
    assert ran == [], "the drain did not start it again"
    assert final.message == runtime_drain.STOPPED_MEANWHILE
