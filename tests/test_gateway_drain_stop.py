"""Row 13: the agent drains its gateway before every stop.

Over HTTP (`POST /v1/admin/drain` with this agent's own token), so it works
where a console signal cannot be sent (the Windows service). The gateway is
ended only past its own deadline plus 5 s, and at most 20 s when the agent
itself stops.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from eugene_plexus_agent import _http, process_signals, supervisor, tokens
from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind, SpawnConfig
from eugene_plexus_agent.supervisor import SupervisedProcess, Supervisor

from .test_supervisor import _auth, _FakeProcess


def _entry(kind: ComponentKind = ComponentKind.gateway, name: str = "gateway") -> ComponentEntry:
    return ComponentEntry(
        name=name,
        kind=kind,
        url="http://127.0.0.1:18080",  # type: ignore[arg-type]
        spawn=SpawnConfig(configFile=f"/tmp/{name}/config.yaml"),
        safeMode=False,
    )


class Gateway:
    """The gateway's drain, over a mock transport: it exits by itself
    `finish_after` seconds after the drain starts (its answers finished), or
    never when that is None."""

    def __init__(self, proc: _FakeProcess, finish_after: float | None, deadline: float) -> None:
        self.proc = proc
        self.finish_after = finish_after
        self.deadline = deadline
        self.events: list[str] = []
        self.tokens: list[str] = []
        self.started: float | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/admin/drain":
            return httpx.Response(404)
        if request.method == "POST":
            self.events.append("drain")
            self.tokens.append(request.headers.get("authorization", ""))
            self.started = time.perf_counter()
            if self.finish_after is not None:
                asyncio.get_running_loop().call_later(self.finish_after, self.proc._finish, 0)
        state = "exiting" if self.proc.returncode is not None else "draining"
        deadline = datetime.now(UTC) + timedelta(seconds=self.deadline)
        return httpx.Response(
            202 if request.method == "POST" else 200,
            json={"state": state, "running": 1, "deadline": deadline.isoformat()},
        )


async def _spawned(
    monkeypatch: pytest.MonkeyPatch,
    entry: ComponentEntry,
    finish_after: float | None,
    deadline: float,
) -> tuple[SupervisedProcess, _FakeProcess, Gateway, list[str]]:
    proc = _FakeProcess()

    async def fake_create(*args: Any, **kwargs: Any) -> _FakeProcess:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    gateway = Gateway(proc, finish_after, deadline)
    _http.set_shared_client(
        "gateway-drain", httpx.AsyncClient(transport=httpx.MockTransport(gateway.handler))
    )
    signals: list[str] = []

    def request_stop(p: Any, *, name: str, logger: Any = None) -> Any:
        gateway.events.append("signal")
        signals.append(name)
        p.terminate()
        return process_signals.StopSignal.terminate_process

    monkeypatch.setattr(process_signals, "request_stop", request_stop)
    auth = _auth()
    sp = SupervisedProcess.for_component(entry, logging.getLogger("test"), auth_state=auth)
    sp.start()
    for _ in range(100):
        if sp.pid is not None:
            break
        await asyncio.sleep(0.01)
    assert sp.pid is not None
    return sp, proc, gateway, signals


async def test_gateway_stop_drains_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stop asks the gateway to drain, with this agent's own token, waits
    for it to finish its answers and exit, and does not kill it."""
    sp, proc, gateway, _signals = await _spawned(monkeypatch, _entry(), 0.3, 60)
    began = time.perf_counter()
    await sp.stop()
    assert gateway.events[0] == "drain", gateway.events
    assert proc.returncode == 0 and not proc.killed, "it exited by itself, after its answers"
    assert time.perf_counter() - began >= 0.25
    token = gateway.tokens[0].removeprefix("Bearer ")
    trust = _auth().trust  # a different node: only its own may drain it
    assert token and trust.recipient
    claims = sp._planner._auth_state.trust.verify(token, classes=(tokens.TYP_SERVICE,))  # type: ignore[attr-defined]
    # Review note: its own `sub`, short-lived, not the year-long `sub: agent`.
    assert claims.sub == supervisor.DRAIN_SUB and claims.is_local_service(claims.iss)
    assert claims.exp - claims.iat <= 300


async def test_a_gateway_that_never_exits_is_ended_at_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the agent itself stops it waits at most its cap, then stops the
    gateway as before."""
    sp, proc, gateway, signals = await _spawned(monkeypatch, _entry(), None, 600)
    began = time.perf_counter()
    await sp.stop(drain_cap=0.6)
    waited = time.perf_counter() - began
    assert gateway.events[:2] == ["drain", "signal"], gateway.events
    assert 0.5 <= waited < 3, waited
    assert signals == ["gateway"] and proc.returncode is not None


async def test_the_kill_waits_for_the_gateways_own_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor, "DRAIN_KILL_GRACE_SECONDS", 0.2)
    sp, _proc, gateway, _signals = await _spawned(monkeypatch, _entry(), None, 0.5)
    began = time.perf_counter()
    await sp.stop()
    waited = time.perf_counter() - began
    assert gateway.events[:2] == ["drain", "signal"]
    assert 0.6 <= waited < 3, "its deadline (0.5 s) plus the grace, then the stop"


async def test_only_a_gateway_is_drained(monkeypatch: pytest.MonkeyPatch) -> None:
    sp, _proc, gateway, _signals = await _spawned(
        monkeypatch, _entry(ComponentKind.inference_driver, "left"), 0.1, 60
    )
    await sp.stop()
    assert gateway.events == ["signal"]


async def test_agent_stop_drains_gateways_before_anything_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Supervisor.drain_gateways` is what the agent's own stop calls before it
    stops the engines, so the answers in flight still have an engine."""
    proc = _FakeProcess()

    async def fake_create(*args: Any, **kwargs: Any) -> _FakeProcess:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    gateway = Gateway(proc, 0.2, 60)
    _http.set_shared_client(
        "gateway-drain", httpx.AsyncClient(transport=httpx.MockTransport(gateway.handler))
    )
    owner = Supervisor(logging.getLogger("test"), auth_state=_auth())
    owner.add_and_start(_entry())
    for _ in range(100):
        if owner._processes["gateway"].pid is not None:
            break
        await asyncio.sleep(0.01)
    sp = owner._processes["gateway"]
    await owner.drain_gateways(cap=5)
    assert gateway.events == ["drain"] and proc.returncode == 0
    # Review #2: the agent is stopping, so a gateway that drained and exited 0
    # is not spawned again.
    await asyncio.sleep(1.0)
    assert sp.spawns == 1, f"spawned {sp.spawns} times during the agent's stop"
    await owner.stop_all()


@pytest.mark.parametrize("finish_after", [0.05, 0.2, 0.7])
async def test_one_restart_spawns_once(
    monkeypatch: pytest.MonkeyPatch, finish_after: float
) -> None:
    """Review #3: a gateway that finishes its drain exits by itself and the
    loop spawns it again; the restart then signalled that new process too, so
    one restart spawned up to three times. Only the process drained is ever
    signalled."""
    first = _FakeProcess()
    made: list[_FakeProcess] = []

    async def fake_create(*args: Any, **kwargs: Any) -> _FakeProcess:
        proc = first if not made else _FakeProcess()
        made.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    gateway = Gateway(first, finish_after, 60)
    _http.set_shared_client(
        "gateway-drain", httpx.AsyncClient(transport=httpx.MockTransport(gateway.handler))
    )
    owner = Supervisor(logging.getLogger("test"), auth_state=_auth())
    owner.add_and_start(_entry())
    sp = owner._processes["gateway"]
    for _ in range(100):
        if sp.pid is not None:
            break
        await asyncio.sleep(0.01)
    try:
        await owner.restart("gateway")
        await asyncio.sleep(1.5)
        assert len(made) == 2, f"{len(made)} processes for one restart"
        assert not made[1].terminated and not made[1].killed, "the new gateway was signalled"
    finally:
        await owner.stop_all()


async def test_a_restart_does_not_wait_for_the_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review #4: sign-in, enrollment and `POST /v1/components/{name}/restart`
    await a restart, which waited for the whole drain (up to 125 s, 905 s at
    the most). The restart now returns at once; the drain and the stop run in
    the background."""
    sp, proc, gateway, _signals = await _spawned(monkeypatch, _entry(), None, 60)
    began = time.perf_counter()
    await asyncio.wait_for(sp.restart(), 2)
    assert time.perf_counter() - began < 1.0
    for _ in range(100):
        if gateway.events:
            break
        await asyncio.sleep(0.01)
    assert gateway.events == ["drain"], "the drain still runs, in the background"
    assert proc.returncode is None
    await sp.stop(drain_cap=0.3)
    assert proc.returncode is not None
