"""Row 14 (M4-LOAD): vLLM's `/load` on a runtime, as `engineLoad`.

A gateway counts only the requests it sent. vLLM says how many requests occupy
the engine from every client, and llama.cpp says nothing. The field is the
engine's number, or null: never a 0 for an engine that said nothing.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from eugene_plexus_agent import _http
from eugene_plexus_agent._generated.models import RuntimeSpec
from eugene_plexus_agent.engines import LlamaCppAdapter, vllm
from eugene_plexus_agent.engines.base import Ready
from eugene_plexus_agent.runtimes import RuntimeSupervisor, _RuntimePlanner
from eugene_plexus_agent.supervisor import ProcessState


def serve(body: object, status: int = 200) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/load"
        return httpx.Response(status, json=body)

    _http.set_shared_client(
        "engine-probe", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


@pytest.mark.asyncio
async def test_vllm_reads_server_load() -> None:
    serve({"server_load": 7})
    assert await vllm.VllmAdapter().read_load("http://engine:8100/") == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"server_load": "many"},
        {"server_load": -1},
        {"server_load": 2.5},
        {"server_load": True},
        [1],
    ],
)
async def test_a_load_that_is_no_count_is_unknown(body: object) -> None:
    serve(body)
    assert await vllm.VllmAdapter().read_load("http://engine:8100") is None


@pytest.mark.asyncio
async def test_a_refused_load_is_unknown() -> None:
    serve({"server_load": 3}, status=404)
    assert await vllm.VllmAdapter().read_load("http://engine:8100") is None


@pytest.mark.asyncio
async def test_an_engine_that_does_not_answer_is_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _http.set_shared_client(
        "engine-probe", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    assert await vllm.VllmAdapter().read_load("http://engine:8100") is None


@pytest.mark.asyncio
async def test_llama_cpp_has_no_load_to_read() -> None:
    assert await LlamaCppAdapter().read_load("http://engine:8100") is None


class FakeProcess:
    state = ProcessState.starting
    pid = 4242
    last_error = None
    last_restart = datetime.now(UTC)
    supervising = True
    stopping = False
    spawns = 1
    last_argv = ()

    def observe_readiness(self, ready: bool) -> None:
        pass


def supervisor_for(
    adapter: object, engine: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[RuntimeSupervisor, RuntimeSpec]:
    spec = RuntimeSpec(name="qwen", engine=engine, modelPath=str(tmp_path / "m"), port=18100)
    supervisor = RuntimeSupervisor()
    supervisor._planners["qwen"] = _RuntimePlanner(
        spec,
        adapter,
        logging.getLogger("test"),
        lambda key: False,  # type: ignore[arg-type]
    )
    supervisor._processes["qwen"] = FakeProcess()  # type: ignore[assignment]
    monkeypatch.setattr("eugene_plexus_agent.runtimes.adapter_for", lambda _: adapter)

    async def ready(*args: object, **kwargs: object) -> Ready:
        return Ready()

    monkeypatch.setattr(adapter, "probe_readiness", ready)
    return supervisor, spec


@pytest.mark.asyncio
async def test_a_ready_vllm_runtime_reports_its_engine_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = vllm.VllmAdapter()
    supervisor, spec = supervisor_for(adapter, "vllm", monkeypatch, tmp_path)
    loads = iter([5, 2, None])

    async def read(base: str) -> int | None:
        assert base == "http://127.0.0.1:18100"
        return next(loads)

    monkeypatch.setattr(adapter, "read_load", read)
    assert supervisor.compose(spec).engineLoad is None  # nothing read yet
    await supervisor._probe_one(spec)
    assert supervisor.compose(spec).engineLoad == 5
    await supervisor._probe_one(spec)
    assert supervisor.compose(spec).engineLoad == 2
    # The engine did not answer the last read: unknown, not the last number.
    await supervisor._probe_one(spec)
    assert supervisor.compose(spec).engineLoad is None


@pytest.mark.asyncio
async def test_a_runtime_that_is_no_longer_ready_reports_no_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = vllm.VllmAdapter()
    supervisor, spec = supervisor_for(adapter, "vllm", monkeypatch, tmp_path)

    async def read(base: str) -> int | None:
        return 4

    monkeypatch.setattr(adapter, "read_load", read)
    await supervisor._probe_one(spec)
    assert supervisor.compose(spec).engineLoad == 4
    supervisor._processes["qwen"].spawns += 1  # type: ignore[attr-defined]  # a new run
    assert supervisor.compose(spec).engineLoad is None


@pytest.mark.asyncio
async def test_llama_cpp_runtime_reports_no_engine_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = LlamaCppAdapter()
    supervisor, spec = supervisor_for(adapter, "llama_cpp", monkeypatch, tmp_path)
    await supervisor._probe_one(spec)
    assert supervisor.compose(spec).engineLoad is None
