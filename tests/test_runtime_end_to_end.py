"""End-to-end runtime supervision against a real subprocess.

This is M0's watchdog acceptance test: a runtime is declared, the
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
from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_watchdog._generated.models import Origin, RuntimeSpec, RuntimeStatus
from eugene_plexus_watchdog.engines.base import DiscoveredBinary
from eugene_plexus_watchdog.engines.llama_cpp import LlamaCppAdapter
from eugene_plexus_watchdog.runtimes import RuntimeSupervisor

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
STARTED = time.monotonic()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        loading = (time.monotonic() - STARTED) < READY_AFTER
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

    def resolve_binary(self, spec: RuntimeSpec) -> DiscoveredBinary:
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
    """The M0 chain for the watchdog: declare, spawn, observe.

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
    from eugene_plexus_watchdog import runtimes as runtimes_module

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

    from eugene_plexus_watchdog import runtimes as runtimes_module

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
