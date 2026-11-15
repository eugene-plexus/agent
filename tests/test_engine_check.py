"""Row 13: the engine check against stand-in tool executables (review #11, #12).

`EngineCheck.measure`, `_answers` and `run_tool` run real child processes here:
a stand-in llama-bench and llama-perplexity per build, which print what the
real tools print. On POSIX each is a script with a shebang; on Windows a `.cmd`
that runs the same script, so `_tool` is pointed at it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
import pytest

from eugene_plexus_agent import engine_rollout
from eugene_plexus_agent.engines.acquisition import InstalledBuild

STANDIN = r"""
import json, os, sys, time
tool, build = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
speeds = {"b1": (1000.0, 100.0), "b2": (1000.0, 88.0)}
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, f"{tool}-{build}.pid"), "w") as f:
    f.write(str(os.getpid()))
if os.environ.get("STANDIN_HANG"):
    time.sleep(600)
if tool == "llama-bench":
    pp, tg = speeds[build]
    print(json.dumps({"n_prompt": 512, "n_gen": 0, "avg_ts": pp}))
    print(json.dumps({"n_prompt": 0, "n_gen": 128, "avg_ts": tg}))
    sys.exit(0)
base = args[args.index("--kl-divergence-base") + 1]
if "--kl-divergence" not in args:
    open(base, "wb").write(b"logits")
    print("computing over 2 chunks, n_ctx=4096")
    sys.exit(0)
if not os.path.exists(base):
    print("cannot open the base file", file=sys.stderr)
    sys.exit(1)
report = (
    "computing over 2 chunks, n_ctx=4096\n"
    "Mean    KLD:   0.003857 ±   0.000260\n"
    "Same top p: 99.612 ± 0.090 %\n"
)
sys.stdout.buffer.write(report.encode("utf-8"))  # as llama.cpp writes, on any code page
"""


def _builds(tmp_path: Path) -> tuple[InstalledBuild, InstalledBuild, dict[tuple[str, str], Path]]:
    script = tmp_path / "standin.py"
    script.write_text(STANDIN, encoding="utf-8")
    tools: dict[tuple[str, str], Path] = {}
    builds = []
    for version in ("b1", "b2"):
        directory = tmp_path / version
        directory.mkdir()
        server = directory / "llama-server"
        server.write_text("", encoding="utf-8")
        for tool in ("llama-bench", "llama-perplexity"):
            if os.name == "nt":
                path = directory / f"{tool}.cmd"
                path.write_text(
                    f'@"{sys.executable}" "{script}" {tool} {version} %*\r\n', encoding="utf-8"
                )
            else:
                path = directory / tool
                path.write_text(
                    f"#!{sys.executable}\nimport runpy, sys\n"
                    f"sys.argv[1:1] = [{tool!r}, {version!r}]\n"
                    f"runpy.run_path({str(script)!r}, run_name='__main__')\n",
                    encoding="utf-8",
                )
                path.chmod(0o755)
            tools[(version, tool)] = path
        builds.append(
            InstalledBuild(
                version=version,
                variant="cpu",
                directory=directory,
                binary=server,
                installed_at=datetime.now(UTC),
                size_bytes=0,
            )
        )
    return builds[0], builds[1], tools


@pytest.fixture
def standins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    old, new, tools = _builds(tmp_path)

    def tool(build: InstalledBuild, name: str) -> Path | None:
        return tools.get((build.version, name))

    monkeypatch.setattr(engine_rollout, "_tool", tool)
    return old, new, tools


async def test_measure_reads_both_tools_on_both_builds(standins, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    old, new, _tools = standins
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a gguf")
    check = engine_rollout.EngineCheck(tmp_path / "work")
    measured, notes = await check.measure(
        model=model, name="qwen3-8b", flags={}, env=dict(os.environ), old=old, new=new
    )
    assert notes == []
    assert measured.measured
    assert measured.promptTps is not None and measured.promptTps.old == 1000.0
    assert measured.generationTps is not None
    assert (measured.generationTps.old, measured.generationTps.new) == (100.0, 88.0)
    assert (measured.sameTopTokenPercent, measured.standardError) == (99.612, 0.09)
    assert engine_rollout.check_words("b2", "qwen3-8b", measured) == (
        "b2 on qwen3-8b: 12% slower at generating, same answers."
    )
    assert list((tmp_path / "work").iterdir()) == [], "the comparison file is removed"


async def test_a_tool_is_ended_when_the_check_is_cancelled(standins, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Review #11: Stop the update during a measurement left llama-bench or
    llama-perplexity running on the GPU. Any exit from the wait ends the tool,
    and whatever it started."""
    _old, new, tools = standins
    env = {**os.environ, "STANDIN_HANG": "1"}
    running = asyncio.create_task(
        engine_rollout.run_tool([str(tools[("b2", "llama-bench")])], env=env)
    )
    pid_file = tmp_path / "llama-bench-b2.pid"
    for _ in range(200):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.05)
    pid = int(pid_file.read_text())
    assert psutil.pid_exists(pid)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("the stand-in tool still runs after the check was cancelled")
    assert new.version == "b2"


async def test_a_tool_that_outruns_its_limit_is_ended(standins, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _old, _new, tools = standins
    env = {**os.environ, "STANDIN_HANG": "1"}
    code, text = await engine_rollout.run_tool(
        [str(tools[("b1", "llama-bench")])], env=env, seconds=1.0
    )
    assert code == -1 and "did not finish within 1 s" in text
    pid = int((tmp_path / "llama-bench-b1.pid").read_text())
    await asyncio.sleep(0.5)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert json  # the stand-in prints jsonl; imported for its readers
