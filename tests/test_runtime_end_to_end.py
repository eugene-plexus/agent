"""End-to-end runtime supervision against a real subprocess.

This is M0's agent acceptance test: a runtime is declared, the
supervisor spawns an actual process, the adapter's real `/health` probe
watches it, and `GET /v1/runtimes` reports `loading` and then `ready`.

The engine is a stand-in — a small Python HTTP server that speaks
llama-server's `/health` and `/props` contract, including the 503
"loading model" phase — because the point is to prove OUR chain, not
llama.cpp's. Everything between the declaration and the status is real:
a real subprocess, real pipes, the real supervision loop, the real
readiness probe, the real status mapping.

Only `build_argv` is substituted, so the fake engine can be launched
through `sys.executable` on any platform. argv construction has its own
unit tests in test_engines.py.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eugene_plexus_agent._generated.models import Origin, RuntimeSpec, RuntimeStatus
from eugene_plexus_agent.engines.base import DiscoveredBinary, Loading, Ready
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter
from eugene_plexus_agent.runtimes import RuntimeSupervisor

from .test_supervisor import _FakeProcess

# A minimal llama-server impersonator. Reports `loading model` for its
# first few polls, then `ok` — so the test observes the real transition
# rather than a single terminal state.
_FAKE_ENGINE = """
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default

PORT = int(opt("--port", "0"))
READY_AFTER = float(opt("--ready-after", "0"))
STARTED = time.perf_counter()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        loading = (time.perf_counter() - STARTED) < READY_AFTER
        if self.path == "/health":
            if loading:
                self._send(503, {"status": "loading model"})
            else:
                self._send(200, {"status": "ok"})
        elif self.path == "/props":
            self._send(200, {
                "default_generation_settings": {"n_ctx": 4096},
                "total_slots": 2,
            })
        else:
            self._send(404, {"error": "nope"})

    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

print("fake llama-server listening on", PORT, flush=True)
HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
"""


class _FakeEngineAdapter(LlamaCppAdapter):
    """Real llama.cpp adapter, launched through the interpreter.

    Overrides only argv so a .py stand-in is executable everywhere;
    `probe_readiness`, `_read_capabilities` and `working_directory` are
    the genuine implementations under test here.
    """

    def __init__(self, script: Path, ready_after: float) -> None:
        self._script = script
        self._ready_after = ready_after

    def build_argv(self, spec: RuntimeSpec, binary: DiscoveredBinary, port: int) -> list[str]:
        return [
            sys.executable,
            str(self._script),
            "--port",
            str(port),
            "--ready-after",
            str(self._ready_after),
        ]

    def working_directory(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> str | None:
        return None

    def resolve_binary(
        self, spec: RuntimeSpec, *, configured: str | None = None
    ) -> DiscoveredBinary:
        return DiscoveredBinary(path=Path(sys.executable), origin=Origin.configured, version="test")


@pytest.fixture
def fake_engine(tmp_path: Path) -> Path:
    script = tmp_path / "fake_llama_server.py"
    script.write_text(_FAKE_ENGINE, encoding="utf-8")
    return script


def _spec(port: int, **overrides: Any) -> RuntimeSpec:
    body: dict[str, Any] = {
        "name": "qwen3-30b",
        "engine": "llama_cpp",
        "modelPath": "/models/Qwen3-30B-A3B-Q4_K_M.gguf",
        "host": "127.0.0.1",
        "port": port,
    }
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


# Generous: the readiness loop polls every 2s and the fake engine holds
# `loading` for 3s, so a slow CI box still has plenty of room.
_STATUS_TIMEOUT_SECONDS = 20.0


async def _await_status(
    supervisor: RuntimeSupervisor,
    spec: RuntimeSpec,
    wanted: RuntimeStatus,
) -> RuntimeStatus:
    """Poll until the composed status matches.

    Returns the last status seen rather than raising, so a timeout shows
    up as an assertion naming what it actually got.
    """
    seen = supervisor.compose(spec).status
    try:
        async with asyncio.timeout(_STATUS_TIMEOUT_SECONDS):
            while True:
                seen = supervisor.compose(spec).status
                if seen == wanted:
                    return seen
                await asyncio.sleep(0.1)
    except TimeoutError:
        return seen


async def test_runtime_goes_starting_then_loading_then_ready(fake_engine: Path) -> None:
    """The M0 chain for the agent: declare, spawn, observe.

    Asserts the `loading` state is actually reached rather than skipped,
    because that distinction is the reason readiness is per-engine — an
    operator staring at a dashboard needs to tell "reading 20GB off
    disk" from "wedged".
    """
    port = 8397
    spec = _spec(port)
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))
    adapter = _FakeEngineAdapter(fake_engine, ready_after=3.0)

    # Point the registry at the stand-in for the duration of the test.
    from eugene_plexus_agent import runtimes as runtimes_module

    original = runtimes_module.ADAPTERS.copy()
    runtimes_module.ADAPTERS[spec.engine] = adapter
    try:
        supervisor.add_and_start(spec)
        await supervisor.start_readiness_loop(lambda: [spec])

        # Before the first probe lands, the loop knows only that the
        # child is alive.
        assert supervisor.compose(spec).status in (
            RuntimeStatus.starting,
            RuntimeStatus.loading,
        )

        assert await _await_status(supervisor, spec, RuntimeStatus.loading) == (
            RuntimeStatus.loading
        )
        assert await _await_status(supervisor, spec, RuntimeStatus.ready) == RuntimeStatus.ready

        runtime = supervisor.compose(spec)
        # Capabilities are read back off the running engine, not inferred
        # from the declaration — the engine clamps what it can't honour.
        assert runtime.capabilities is not None
        assert runtime.capabilities.contextLength == 4096
        assert runtime.capabilities.parallelSlots == 2
        # The resolved command line is reported for debugging.
        assert runtime.argv is not None
        assert str(port) in runtime.argv
        assert runtime.pid is not None
        assert str(runtime.url).rstrip("/") == f"http://127.0.0.1:{port}"
    finally:
        await supervisor.stop_all()
        runtimes_module.ADAPTERS.clear()
        runtimes_module.ADAPTERS.update(original)


async def test_stop_releases_the_process(fake_engine: Path) -> None:
    """Stop must actually end the process — the whole reason it exists as
    something distinct from delete is reclaiming GPU memory."""
    port = 8398
    spec = _spec(port)
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))
    adapter = _FakeEngineAdapter(fake_engine, ready_after=0.0)

    from eugene_plexus_agent import runtimes as runtimes_module

    original = runtimes_module.ADAPTERS.copy()
    runtimes_module.ADAPTERS[spec.engine] = adapter
    try:
        supervisor.add_and_start(spec)
        await supervisor.start_readiness_loop(lambda: [spec])
        assert await _await_status(supervisor, spec, RuntimeStatus.ready) == RuntimeStatus.ready

        await supervisor.stop_one("qwen3-30b")

        runtime = supervisor.compose(spec)
        assert runtime.status == RuntimeStatus.stopped
        assert runtime.pid is None
        assert supervisor.is_running("qwen3-30b") is False
    finally:
        await supervisor.stop_all()
        runtimes_module.ADAPTERS.clear()
        runtimes_module.ADAPTERS.update(original)


async def test_a_crash_during_load_is_not_automatically_retried(
    fake_engine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_agent import runtimes as runtimes_module

    spec = _spec(8399)
    adapter = _FakeEngineAdapter(fake_engine, ready_after=0)
    processes: list[_FakeProcess] = []
    original_sleep = asyncio.sleep

    async def loading(_base: str, *, established: bool = False) -> Loading:
        return Loading(detail="reading model from the share")

    async def create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        process = _FakeProcess()
        processes.append(process)
        if len(processes) > 1:
            process._finish(1)
        return process

    async def short_backoff(seconds: float) -> None:
        await original_sleep(min(seconds, 0.001))

    monkeypatch.setattr(adapter, "probe_readiness", loading)
    monkeypatch.setitem(runtimes_module.ADAPTERS, spec.engine, adapter)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(asyncio, "sleep", short_backoff)
    monkeypatch.setattr("eugene_plexus_agent.orphan_kill.windows_job", lambda: None)
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))
    try:
        supervisor.add_and_start(spec)
        async with asyncio.timeout(2):
            while not processes:
                await original_sleep(0)
            await supervisor._probe_one(spec)
            assert supervisor.compose(spec).status == RuntimeStatus.loading
            processes[0]._finish(1)
            await supervisor._processes[spec.name]._task
        assert len(processes) == 1, "a failed model load was repeated without any changed settings"
        runtime = supervisor.compose(spec)
        assert runtime.status == RuntimeStatus.crashed
        assert "before becoming ready" in runtime.lastError
        assert "restart" in runtime.lastError
        assert "retry" in runtime.lastError
        assert runtime.pid is None
    finally:
        await supervisor.stop_all()


@dataclass
class _RuntimeHarness:
    supervisor: RuntimeSupervisor
    spec: RuntimeSpec
    adapter: _FakeEngineAdapter
    spawned: asyncio.Queue[_FakeProcess] = field(default_factory=asyncio.Queue)
    now: float = 0.0
    outcome: Loading | Ready | None = None
    backoffs: list[float] = field(default_factory=list)

    async def next_process(self) -> _FakeProcess:
        return await asyncio.wait_for(self.spawned.get(), 2)

    async def probe(self, outcome: Loading | Ready | None) -> None:
        self.outcome = outcome
        await self.supervisor._probe_one(self.spec)

    async def settled(self) -> None:
        await asyncio.wait_for(self.supervisor._processes[self.spec.name]._task, 2)


@pytest.fixture
async def controlled_runtime(
    fake_engine: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_RuntimeHarness]:
    from eugene_plexus_agent import runtimes as runtimes_module
    from eugene_plexus_agent import supervisor as supervisor_module

    harness = _RuntimeHarness(
        RuntimeSupervisor(log=logging.getLogger("test")),
        _spec(8399),
        _FakeEngineAdapter(fake_engine, ready_after=0),
    )
    original_sleep = asyncio.sleep

    async def create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        process = _FakeProcess()
        harness.spawned.put_nowait(process)
        return process

    async def probe(_base: str, *, established: bool = False) -> Loading | Ready | None:
        return harness.outcome

    async def backoff(seconds: float) -> None:
        harness.backoffs.append(seconds)
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(asyncio, "sleep", backoff)
    monkeypatch.setattr(
        supervisor_module, "time", SimpleNamespace(perf_counter=lambda: harness.now)
    )
    monkeypatch.setattr("eugene_plexus_agent.orphan_kill.windows_job", lambda: None)
    monkeypatch.setattr(harness.adapter, "probe_readiness", probe)
    monkeypatch.setitem(runtimes_module.ADAPTERS, harness.spec.engine, harness.adapter)
    try:
        harness.supervisor.add_and_start(harness.spec)
        yield harness
    finally:
        await harness.supervisor.stop_all()


@pytest.mark.parametrize("seconds", [0.1, 240.0])
async def test_loading_time_is_not_healthy_uptime(
    controlled_runtime: _RuntimeHarness, seconds: float
) -> None:
    harness = controlled_runtime
    process = await harness.next_process()
    await harness.probe(Loading(detail="loading model"))
    harness.now += seconds
    process._finish(1)
    await harness.settled()
    assert harness.spawned.empty()
    assert harness.backoffs == []


async def test_brief_ready_crashes_keep_the_backoff_and_eventually_stop(
    controlled_runtime: _RuntimeHarness,
) -> None:
    harness = controlled_runtime
    for _ in range(5):
        process = await harness.next_process()
        await harness.probe(Ready())
        harness.now += 1
        process._finish(1)
    await harness.settled()
    assert harness.spawned.empty()
    assert harness.backoffs == [2.0, 4.0, 6.0, 8.0]
    assert harness.supervisor.compose(harness.spec).status == RuntimeStatus.crashed


@pytest.mark.parametrize("ready_seconds, expected_backoff", [(59.0, 4.0), (60.0, 2.0)])
async def test_only_stable_ready_time_resets_crash_history(
    controlled_runtime: _RuntimeHarness, ready_seconds: float, expected_backoff: float
) -> None:
    harness = controlled_runtime
    process = await harness.next_process()
    await harness.probe(Ready())
    harness.now += 1
    process._finish(1)
    process = await harness.next_process()
    await harness.probe(Loading(detail="long load"))
    harness.now += 240
    await harness.probe(Ready())
    harness.now += ready_seconds / 2
    await harness.probe(Ready())
    harness.now += ready_seconds / 2
    process._finish(1)
    await harness.next_process()
    assert harness.backoffs == [2.0, expected_backoff]


async def test_lost_readiness_restarts_the_stability_window(
    controlled_runtime: _RuntimeHarness,
) -> None:
    harness = controlled_runtime
    process = await harness.next_process()
    await harness.probe(Ready())
    process._finish(1)
    process = await harness.next_process()
    await harness.probe(Ready())
    harness.now += 50
    await harness.probe(None)
    harness.now += 100
    await harness.probe(Ready())
    harness.now += 20
    process._finish(1)
    await harness.next_process()
    assert harness.backoffs == [2.0, 4.0]


async def test_a_fast_replacement_does_not_inherit_the_stability_clock(
    controlled_runtime: _RuntimeHarness,
) -> None:
    harness = controlled_runtime
    process = await harness.next_process()
    await harness.probe(Ready())
    harness.now += 1
    process._finish(1)
    process = await harness.next_process()
    harness.now += 240
    await harness.probe(Ready())
    harness.now += 1
    process._finish(1)
    await harness.next_process()
    assert harness.backoffs == [2.0, 4.0]


async def test_a_replacement_must_earn_its_own_readiness(
    controlled_runtime: _RuntimeHarness,
) -> None:
    harness = controlled_runtime
    process = await harness.next_process()
    await harness.probe(Ready())
    process._finish(1)
    replacement = await harness.next_process()
    harness.now += 240
    replacement._finish(1)
    await harness.settled()
    assert harness.spawned.empty()
    assert harness.backoffs == [2.0]


async def test_manual_restart_retries_a_terminal_load_failure(
    controlled_runtime: _RuntimeHarness,
) -> None:
    harness = controlled_runtime
    process = await harness.next_process()
    process._finish(1)
    await harness.settled()
    assert await harness.supervisor.restart(harness.spec.name)
    await harness.next_process()
    await harness.probe(Ready())
    assert harness.supervisor.compose(harness.spec).status == RuntimeStatus.ready
    assert harness.supervisor.compose(harness.spec).lastError is None
    assert harness.backoffs == []


async def test_a_clean_exit_still_respawns(controlled_runtime: _RuntimeHarness) -> None:
    harness = controlled_runtime
    process = await harness.next_process()
    process._finish(0)
    await harness.next_process()
    assert harness.backoffs == [0.0]


async def test_late_ready_probe_cannot_belong_to_a_replacement(
    controlled_runtime: _RuntimeHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = controlled_runtime
    process = await harness.next_process()
    await harness.probe(Ready())
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed(_base: str, *, established: bool = False) -> Ready:
        entered.set()
        await release.wait()
        return Ready()

    monkeypatch.setattr(harness.adapter, "probe_readiness", delayed)
    probe = asyncio.create_task(harness.supervisor._probe_one(harness.spec))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        process._finish(1)
        replacement = await harness.next_process()
        release.set()
        await probe
        replacement._finish(1)
        await harness.settled()
        assert harness.spawned.empty()
        assert harness.backoffs == [2.0]
    finally:
        release.set()
        await probe
