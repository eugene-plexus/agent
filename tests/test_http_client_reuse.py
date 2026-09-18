"""One client, one context, one clock — the agent's half.

The readiness loop is the hottest path in this process: every 2 s, per
ready runtime, each adapter probes `/health` and then reads capabilities
back. Those were two fresh `httpx.AsyncClient()` objects, and building
one parses certifi's PEM bundle — **104-136 ms of synchronous CPU**,
measured in this repo's venv on the Python both installers provision —
on the event loop that also serves the UI and the browser proxy. With
four resident runtimes that is ~0.84 s of every 2 s spent blocking.

Admission made it worse in a place a person watches: three library
reads, three clients, and the UI calls admission every time the
new-profile form opens.

And `trust_env`: the Windows logon task inherits the user environment,
so a corporate `HTTP_PROXY` is applied to every loopback health probe.
The poller does not restart anything — it gates the `starting` ->
`running` promotion — so the whole install sits at *starting* while
actually serving, with no screen that explains it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent import _http
from eugene_plexus_agent.admission import LibraryFitClient
from eugene_plexus_agent.engines.base import probe_client
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter
from eugene_plexus_agent.engines.vllm import VllmAdapter


def _count_constructions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    real = httpx.AsyncClient.__init__

    def counting(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        calls[0] += 1
        real(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", counting)
    return calls


def _mock(handler: Any) -> None:
    _http.set_shared_client(
        "engine-probe", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


# --------------------------------------------------------------------------- #
# one client
# --------------------------------------------------------------------------- #


async def test_ten_readiness_probes_build_one_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """**The 2 s loop.** Ten probes used to be twenty clients — over two
    seconds of synchronous CPU on the loop serving the browser, for ten
    health checks against loopback."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 4096}})

    _mock(handler)
    built = _count_constructions(monkeypatch)
    adapter = LlamaCppAdapter()
    for _ in range(10):
        await adapter.probe_readiness("http://127.0.0.1:8090")
    assert built[0] == 0, f"{built[0]} clients built across ten probes"


async def test_both_adapters_share_the_one_probe_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """llama.cpp and vLLM probe the same way and have no reason to hold
    separate pools; a host running both would otherwise pay twice."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    _mock(handler)
    built = _count_constructions(monkeypatch)
    await LlamaCppAdapter().probe_readiness("http://127.0.0.1:8090")
    await VllmAdapter().probe_readiness("http://127.0.0.1:8091")
    assert built[0] == 0


async def test_an_admission_makes_its_library_reads_on_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three reads, one client. The UI calls admission when the
    new-profile form opens, so this is latency a person sits through."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"folders": []})

    client = LibraryFitClient("http://127.0.0.1:8082", None, transport=httpx.MockTransport(handler))
    built = _count_constructions(monkeypatch)
    await client.list_models()
    await client.folders()
    await client.fit("/models/x.gguf", context_length=None, vram_bytes=None, ram_bytes=None)
    assert len(calls) >= 3
    assert built[0] <= 1, f"{built[0]} clients for one admission's reads"


# --------------------------------------------------------------------------- #
# the deadlock this slice introduced and then caught
# --------------------------------------------------------------------------- #


def test_a_cold_process_can_build_its_first_shared_client() -> None:
    """**A fresh interpreter, because ordering is the whole defect.**

    The first cut of `_http` guarded the context cache with a plain
    `threading.Lock`, and `shared_internal_client` held it while calling
    `ssl_context()`, which takes the same lock — a deadlock on the very
    first call of a process, and only the first, because every later
    caller finds the context already built and never reaches the
    acquire. A full test run always had something else build it first,
    so the suite was green and a cold agent hung on its first engine
    probe. The lock is an `RLock` now.

    Run in a subprocess: no test inside this process can be first.
    """
    root = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import asyncio, sys;"
        f"sys.path.insert(0, {str(root)!r});"
        "from eugene_plexus_agent import _http;"
        "c = _http.shared_internal_client('engine-probe');"
        "assert c is _http.shared_internal_client('engine-probe');"
        "asyncio.run(_http.aclose_shared());"
        "print('ok')"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


# --------------------------------------------------------------------------- #
# no ambient proxy on this install's own traffic
# --------------------------------------------------------------------------- #


def test_the_probe_client_declines_an_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """An engine is always on this machine. A corporate proxy cannot
    dial 127.0.0.1, so every probe would fail and the install would sit
    at `starting` while serving."""
    monkeypatch.setenv("HTTP_PROXY", "http://corp.proxy:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://corp.proxy:3128")
    _http.reset_shared()
    assert probe_client()._mounts == {}


def test_the_library_client_declines_an_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://corp.proxy:3128")
    _http.reset_shared()
    client = LibraryFitClient("http://127.0.0.1:8082", None)
    assert client._client()._mounts == {}


def test_the_browser_proxy_client_declines_an_ambient_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every call the UI makes goes through this one. With the proxy
    applied, a browser on the machine gets nothing at all."""
    from eugene_plexus_agent.routes.proxy import get_client

    monkeypatch.setenv("HTTPS_PROXY", "http://corp.proxy:3128")

    class _App:
        class state:
            pass

    class _Request:
        app = _App()

    assert get_client(_Request())._mounts == {}  # type: ignore[arg-type]


def test_the_github_fetch_keeps_the_users_proxy() -> None:
    """**The half a blanket `trust_env=False` would have broken.**
    Engine acquisition and the model hub are the only egress this
    product has, and for a user behind a corporate proxy that is how
    they reach the internet at all."""
    assert not _http.is_internal("https://github.com")
    assert not _http.is_internal("https://huggingface.co")
    client = _http.egress_client()
    assert client.trust_env is True


# --------------------------------------------------------------------------- #
# one clock
# --------------------------------------------------------------------------- #


def test_no_duration_in_this_component_is_measured_with_monotonic() -> None:
    """On Windows/CPython 3.12 — what both installers provision —
    `monotonic()` is `GetTickCount64`: 20 distinct values in 300 ms, a
    15.6 ms grid. CPython fixed it in 3.13 and this repo's own venv is
    3.14, so the defect is invisible here and present on every shipped
    install. That is `ci-hygiene-and-devenv-mismatch` again.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_agent"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "_generated" not in path.parts and "time.monotonic()" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"these measure with monotonic(): {offenders}"


def test_perf_counter_resolves_finer_than_monotonic() -> None:
    assert time.get_clock_info("perf_counter").resolution <= (
        time.get_clock_info("monotonic").resolution
    )
