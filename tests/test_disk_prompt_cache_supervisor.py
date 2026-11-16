"""Row 15 (DI2): when the agent saves a llama.cpp runtime's conversations, and when it reads
them back, against a stand-in llama-server's slot routes.

A planned stop or restart of a ready engine saves each slot that ran a conversation and is
not answering; the next start reads the newest back, one per slot, before the runtime is
reported ready; a file the engine refuses is deleted and its cause named; a deleted runtime
takes its files with it; and the runtime says what this run does.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent import _http
from eugene_plexus_agent import disk_prompt_cache as dpc
from eugene_plexus_agent._generated.models import (
    RuntimeCapabilities,
    RuntimeSpec,
    RuntimeStatus,
    StopReason,
)
from eugene_plexus_agent.engines import LlamaCppAdapter
from eugene_plexus_agent.engines.base import Ready
from eugene_plexus_agent.runtimes import RuntimeSupervisor, _RuntimePlanner
from eugene_plexus_agent.supervisor import ProcessState


class Engine:
    """llama-server's slot routes, writing into the launch folder as the real one does."""

    def __init__(self, folder: Path, slots: list[dict[str, Any]]) -> None:
        self.folder = folder
        self.slots = slots
        self.saved: list[tuple[int, str]] = []
        self.restored: list[tuple[int, str]] = []
        self.events: list[str] = []
        self.slot_reads = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/slots" and request.method == "GET":
            self.slot_reads += 1
            return httpx.Response(200, json=self.slots)
        if path.startswith("/slots/") and request.method == "POST":
            slot = int(path.rsplit("/", 1)[1])
            action = request.url.params.get("action")
            filename = json.loads(request.content)["filename"]
            target = self.folder / filename
            if action == "save":
                target.write_bytes(b"state" * 10)
                self.saved.append((slot, filename))
                self.events.append(f"save {slot}")
                return httpx.Response(
                    200,
                    json={
                        "id_slot": slot,
                        "filename": filename,
                        "n_saved": 120,
                        "n_written": 50,
                        "timings": {"save_ms": 1.5},
                    },
                )
            if action == "restore":
                body = target.read_bytes() if target.exists() else b""
                if body != b"state" * 10:
                    return httpx.Response(
                        400,
                        json={
                            "error": {
                                "code": 400,
                                "type": "invalid_request_error",
                                "message": "Unable to restore slot: Invalid tokens "
                                "in slot save file",
                            }
                        },
                    )
                self.restored.append((slot, filename))
                self.events.append(f"restore {slot}")
                return httpx.Response(
                    200,
                    json={
                        "id_slot": slot,
                        "filename": filename,
                        "n_restored": 120,
                        "n_read": 50,
                        "timings": {"restore_ms": 2.0},
                    },
                )
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})


class FakeProcess:
    state = ProcessState.starting
    pid = 4242
    last_error = None
    supervising = True
    stopping = False
    spawns = 1

    def __init__(self, argv: list[str], events: list[str]) -> None:
        self.last_argv = tuple(argv)
        self.last_restart = datetime.now(UTC)
        self.events = events

    def observe_readiness(self, ready: bool) -> None:
        pass

    async def stop(self) -> None:
        self.events.append("stop")
        self.supervising = False

    async def restart(self) -> None:
        self.events.append("restart")
        self.spawns += 1


def setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: dict[str, Any] | None = None,
    slots: list[dict[str, Any]] | None = None,
    parallel: int = 4,
    flag: bool = True,
) -> tuple[RuntimeSupervisor, RuntimeSpec, Engine, Path]:
    root = tmp_path / "prompt-cache"
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    spec = RuntimeSpec(name="qwen", engine="llama_cpp", modelPath=str(model), port=18100)
    argv = ["/engines/b1/llama-server", "--model", str(model), "--port", "18100"]
    settings = dpc.Settings(enabled=True, days=3, cap_bytes=20 * 1024**3)
    planned, launch = dpc.plan(
        runtime="qwen",
        host=None,
        argv=argv,
        env={},
        extra_args=[],
        root=root,
        settings=settings,
        search=False,
        offered=None,
    )
    assert launch is not None and launch.folder is not None
    folder = launch.folder
    values = {dpc.ENABLED_KEY: True, **(config or {})}
    supervisor = RuntimeSupervisor(get_config=values.get, prompt_cache_root=root)
    adapter = LlamaCppAdapter()
    planner = _RuntimePlanner(spec, adapter, logging.getLogger("test"), values.get)
    planner.disk_launch = launch if flag else dpc.Launch(None, dpc.NOTE_OFF)
    supervisor._planners["qwen"] = planner
    engine = Engine(folder, slots or [])
    process = FakeProcess(planned if flag else argv, engine.events)
    supervisor._processes["qwen"] = process  # type: ignore[assignment]
    monkeypatch.setattr("eugene_plexus_agent.runtimes.adapter_for", lambda _: adapter)

    async def ready(*args: object, **kwargs: object) -> Ready:
        return Ready(capabilities=RuntimeCapabilities(parallelSlots=parallel))

    monkeypatch.setattr(adapter, "probe_readiness", ready)
    _http.set_shared_client(
        "engine-probe", httpx.AsyncClient(transport=httpx.MockTransport(engine.handler))
    )
    return supervisor, spec, engine, folder


def _saved_file(folder: Path, name: str, age: float) -> Path:
    path = folder / name
    path.write_bytes(b"state" * 10)
    when = time.time() - age
    os.utime(path, (when, when))
    return path


async def _ready(supervisor: RuntimeSupervisor, spec: RuntimeSpec) -> None:
    """Probe until ready, as the readiness loop does."""
    for _ in range(50):
        await supervisor._probe_one(spec)
        if supervisor.compose(spec).status is RuntimeStatus.ready:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(supervisor.compose(spec).model_dump(mode="json"))


@pytest.mark.asyncio
async def test_a_planned_stop_saves_each_conversation_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spec, engine, folder = setup(
        tmp_path,
        monkeypatch,
        slots=[
            {"id": 0, "is_processing": False, "n_prompt_tokens": 120},
            {"id": 1, "is_processing": True, "n_prompt_tokens": 50},  # answering: skipped
            {"id": 2, "is_processing": False, "n_prompt_tokens": 0},  # empty
            {"id": 3, "is_processing": False},  # read back and never used since
        ],
    )
    await _ready(supervisor, spec)
    await supervisor.stop_one("qwen", reason=StopReason.idle)
    assert [slot for slot, _ in engine.saved] == [0]
    assert engine.events.index("save 0") < engine.events.index("stop")
    files = dpc.saved(folder)
    assert [f.slot for f in files] == [0]
    assert not list(folder.glob("*.part"))


@pytest.mark.asyncio
async def test_a_restart_saves_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor, spec, engine, _ = setup(
        tmp_path, monkeypatch, slots=[{"id": 1, "is_processing": False, "n_prompt_tokens": 9}]
    )
    await _ready(supervisor, spec)
    await supervisor.restart("qwen")
    assert engine.events[-2:] == ["save 1", "restart"]


@pytest.mark.asyncio
async def test_an_engine_that_is_not_ready_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, _, engine, _ = setup(
        tmp_path, monkeypatch, slots=[{"id": 0, "is_processing": False, "n_prompt_tokens": 9}]
    )
    await supervisor.stop_one("qwen", reason=StopReason.operator)
    assert engine.slot_reads == 0 and engine.saved == []


@pytest.mark.asyncio
async def test_turned_off_it_saves_nothing_even_for_a_run_started_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spec, engine, folder = setup(
        tmp_path,
        monkeypatch,
        config={dpc.ENABLED_KEY: False},
        slots=[{"id": 0, "is_processing": False, "n_prompt_tokens": 9}],
    )
    _saved_file(folder, "1-aaaaaa-s0.bin", age=10)
    await _ready(supervisor, spec)
    assert engine.restored == []
    view = supervisor.compose(spec).diskPromptCache
    assert view is not None and view.saving is False and view.note
    await supervisor.stop_one("qwen", reason=StopReason.idle)
    assert engine.saved == []


@pytest.mark.asyncio
async def test_the_next_start_reads_the_newest_back_before_it_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spec, engine, folder = setup(tmp_path, monkeypatch, parallel=2)
    old = _saved_file(folder, "1-aaaaaa-s0.bin", age=300)
    newer = _saved_file(folder, "2-bbbbbb-s0.bin", age=200)
    newest = _saved_file(folder, "3-cccccc-s1.bin", age=100)
    await supervisor._probe_one(spec)
    reading = supervisor.compose(spec)
    # The engine answers, but the gateway is not told so until the files are read back.
    assert reading.status is RuntimeStatus.loading
    await _ready(supervisor, spec)
    assert sorted(engine.restored) == [(0, newer.name), (1, newest.name)]
    assert time.time() - newest.stat().st_mtime < 60  # used now
    assert time.time() - old.stat().st_mtime > 250  # not read back, not touched
    view = supervisor.compose(spec).diskPromptCache
    assert view is not None and view.saving is True and view.conversations == 2


@pytest.mark.asyncio
async def test_an_expired_file_is_never_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spec, engine, folder = setup(tmp_path, monkeypatch, config={dpc.DAYS_KEY: 1})
    stale = _saved_file(folder, "1-aaaaaa-s0.bin", age=86400 + 60)
    await _ready(supervisor, spec)
    assert engine.restored == [] and not stale.exists()


@pytest.mark.asyncio
async def test_a_file_the_engine_refuses_is_deleted_and_its_cause_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    supervisor, spec, engine, folder = setup(tmp_path, monkeypatch)
    bad = folder / "1-aaaaaa-s0.bin"
    bad.write_bytes(b"not a slot")
    caplog.set_level(logging.WARNING)
    await _ready(supervisor, spec)
    assert engine.restored == [] and not bad.exists()
    assert "Invalid tokens in slot save file" in caplog.text


@pytest.mark.asyncio
async def test_deleting_a_runtime_deletes_its_conversations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, _, _, folder = setup(tmp_path, monkeypatch)
    _saved_file(folder, "1-aaaaaa-s0.bin", age=10)
    await supervisor.remove_and_stop("qwen")
    assert not folder.parent.exists()


@pytest.mark.asyncio
async def test_a_run_started_without_the_folder_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spec, engine, _ = setup(tmp_path, monkeypatch, flag=False)
    await _ready(supervisor, spec)
    view = supervisor.compose(spec).diskPromptCache
    assert view is not None and view.saving is False
    assert view.note and "applies at the next start" in view.note
    await supervisor.stop_one("qwen", reason=StopReason.idle)
    assert engine.slot_reads == 0


@pytest.mark.parametrize("host", [None, "0.0.0.0"])
def test_the_planner_hands_the_engine_its_folder_only_on_loopback(
    tmp_path: Path, host: str | None
) -> None:
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    spec = RuntimeSpec.model_validate(
        {
            "name": "qwen",
            "engine": "llama_cpp",
            "modelPath": str(tmp_path / "q.gguf"),
            "port": 18100,
            "host": host,
            "binary": str(exe),
            "extraArgs": ["--threads", "2"],
        }
    )
    root = tmp_path / "prompt-cache"
    planner = _RuntimePlanner(
        spec,
        LlamaCppAdapter(),
        logging.getLogger("test"),
        get_config={
            "engineBinaryRoots": [str(tmp_path)],
            "allowUnrestrictedEngineLaunch": True,
            dpc.ENABLED_KEY: True,
        }.get,
        prompt_cache_root=root,
    )
    argv = planner.plan().argv
    assert planner.disk_launch is not None
    if host is None:
        folder = planner.disk_launch.folder
        assert folder is not None and folder.parent == root / "qwen"
        assert argv[-4:] == [dpc.FLAG, str(folder), "--threads", "2"]
    else:
        assert dpc.FLAG not in argv
        assert planner.disk_launch.folder is None and "0.0.0.0" in (planner.disk_launch.note or "")


@pytest.mark.asyncio
async def test_an_edited_runtime_saves_and_keeps_its_conversations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, spec, engine, folder = setup(
        tmp_path, monkeypatch, slots=[{"id": 0, "is_processing": False, "n_prompt_tokens": 9}]
    )
    await _ready(supervisor, spec)
    # A PATCH replaces the declaration: the engine stops and starts from the new one.
    await supervisor.remove_and_stop("qwen", keep_conversations=True)
    assert engine.events == ["save 0", "stop"]
    assert [f.slot for f in dpc.saved(folder)] == [0]
