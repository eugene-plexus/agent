"""Real harmless child processes exercise the same pipes and cleanup as llama-bench."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from eugene_plexus_agent._generated.models import BenchmarkRequest
from eugene_plexus_agent.benchmarks import Benchmarks


def body(model: Path) -> BenchmarkRequest:
    return BenchmarkRequest.model_validate(
        {
            "modelId": "m",
            "profileId": "p",
            "profileName": "CPU",
            "tokens": 16,
            "repetitions": 1,
            "runtime": {
                "name": "test",
                "engine": "llama_cpp",
                "modelPath": str(model),
                "flags": {"contextSize": 256, "gpuLayers": 0},
            },
        }
    )


def launch(manager, tmp_path, script):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    return manager.start(
        body(model),
        node="test-node",
        binary=Path(sys.executable),
        version="fixture",
        model=model,
        argv=[sys.executable, "-u", "-c", script],
        depths=[0, 120, 240],
    )


RESULTS = """
import json, os
assert not any(k.startswith('EUGENE_PLEXUS_') for k in os.environ)
for depth in (0, 120, 240):
    print(json.dumps(dict(n_prompt=0, n_gen=16, n_depth=depth,
                         avg_ts=100-depth/4, stddev_ts=0, samples_ts=[100-depth/4],
                         cpu_info='fixture CPU', build_commit='abc123')))
"""


@pytest.mark.asyncio
async def test_real_child_results_persist_and_snapshot_is_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("EUGENE_PLEXUS_MASTER_KEY", "must-not-reach-child")
    manager = Benchmarks(tmp_path / "benchmarks.json")
    job = launch(manager, tmp_path, RESULTS)
    assert manager.active
    await manager.task
    assert not manager.active and manager.process is None
    assert job.state.value == "completed", job.detail
    assert [p.tokensPerSecond for p in job.points] == [100, 70, 40]
    assert job.hardware["build_commit"] == "abc123"
    restored = Benchmarks(manager.path).jobs[0]
    assert restored == job
    assert restored.request.runtime.flags["contextSize"] == 256


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "script,reason",
    [
        ("print('{}')", "does not match"),
        ("print('not json')", "Expecting value"),
        (RESULTS.replace("(0, 120, 240)", "(0, 120)"), "without results"),
        (RESULTS.replace("(0, 120, 240)", "(0, 0, 240)"), "duplicate"),
        (RESULTS.replace("(0, 120, 240)", "(0, 120, 999)"), "does not match"),
        ("import sys; print('fixture error', file=sys.stderr); sys.exit(2)", "fixture error"),
        ("print('x' * 1100000)", "limit"),
    ],
)
async def test_bad_child_output_never_becomes_success(tmp_path, script, reason):
    manager = Benchmarks(tmp_path / "benchmarks.json")
    job = launch(manager, tmp_path, script)
    await manager.task
    assert job.state.value == "failed"
    assert reason.lower() in job.detail.lower()
    assert manager.process is None and not manager.active


@pytest.mark.asyncio
async def test_cancel_waits_for_reaping_and_can_run_again(tmp_path):
    manager = Benchmarks(tmp_path / "benchmarks.json")
    job = launch(manager, tmp_path, "import time; time.sleep(60)")
    for _ in range(100):
        if manager.process:
            break
        await asyncio.sleep(0.01)
    process = manager.process
    assert process is not None
    await manager.cancel(job.id)
    assert process.returncode is not None
    assert job.state.value == "cancelled" and not manager.active
    assert await manager.cancel(job.id) == job
    assert await manager.cancel("missing") is None
    second = launch(manager, tmp_path, RESULTS)
    await manager.task
    assert second.state.value == "completed"


@pytest.mark.asyncio
async def test_timeout_and_close_terminate(tmp_path):
    manager = Benchmarks(tmp_path / "benchmarks.json", timeout=0.15)
    job = launch(manager, tmp_path, "import time; time.sleep(60)")
    await manager.task
    assert job.state.value == "failed" and "time limit" in job.detail
    second = launch(manager, tmp_path, "import time; time.sleep(60)")
    await manager.close()  # cancellation before spawn also settles the record
    assert second.state.value == "cancelled" and not manager.active


@pytest.mark.asyncio
async def test_restart_and_history_write_failure(tmp_path, monkeypatch):
    manager = Benchmarks(tmp_path / "benchmarks.json")
    job = launch(manager, tmp_path, RESULTS)
    await manager.task
    raw = json.loads(manager.path.read_text())
    raw[0]["state"] = "running"
    manager.path.write_text(json.dumps(raw))
    restored = Benchmarks(manager.path)
    assert restored.jobs[0].state.value == "failed"
    assert "restarted" in restored.jobs[0].detail

    def unwritable():
        raise OSError("read-only")

    monkeypatch.setattr(manager, "_save", unwritable)
    with pytest.raises(OSError, match="read-only"):
        launch(manager, tmp_path, RESULTS)
    assert manager.jobs == [job] and not manager.active
