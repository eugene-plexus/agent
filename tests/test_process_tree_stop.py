"""A stopped launcher must not leave its engine computing or holding memory."""

import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path

import psutil
import pytest

from eugene_plexus_agent import process_signals, process_tree
from eugene_plexus_agent.runtimes import RuntimeSupervisor
from eugene_plexus_agent.supervisor import SpawnPlan, SupervisedProcess

real_snapshot = process_tree.snapshot


class Planner:
    name = log_prefix = "tree-test"

    def __init__(self, script: Path, directory: Path) -> None:
        self.script, self.directory = script, directory
        self.spawns = 0

    def plan(self) -> SpawnPlan:
        self.spawns += 1
        return SpawnPlan(
            [
                getattr(sys, "_base_executable", sys.executable),
                str(self.script),
                str(self.directory),
            ],
            dict(os.environ),
        )

    def reset(self) -> None:
        pass

    def on_crash_threshold(self) -> bool:
        return False

    def explain_exit(self, return_code: int, output: list[str]) -> str | None:
        return None


async def until(predicate) -> None:
    async with asyncio.timeout(5):
        while not predicate():  # noqa: ASYNC110 - observing an OS process/file, not an asyncio producer
            await asyncio.sleep(0.01)


def child_pid(path: Path) -> int | None:
    """The pid the child wrote, or None until it can be read.

    Existing is not readable: the file can be seen before its content is
    there, and on Windows opening a file that has only just appeared is
    refused for a moment (PermissionError). Both showed on loaded CI runners.
    """
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


async def wait_for_child_pid(path: Path) -> int:
    async with asyncio.timeout(5):
        while (pid := child_pid(path)) is None:  # noqa: ASYNC110 - observing a file an OS process writes
            await asyncio.sleep(0.01)
    return pid


@pytest.mark.parametrize("action", ["stop", "restart", "kill"])
async def test_launcher_exit_waits_for_its_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    monkeypatch.setattr(process_tree, "snapshot", real_snapshot)
    if action == "kill":
        monkeypatch.setattr(process_tree, "GRACE_SECONDS", 0.3)
    script = tmp_path / "tree.py"
    script.write_text(
        "import os, pathlib, subprocess, sys, time\n"
        "p = pathlib.Path(sys.argv[1])\n"
        "if len(sys.argv) == 2:\n"
        "    child = subprocess.Popen([sys.executable, __file__, str(p), 'child'], "
        "stdin=subprocess.DEVNULL, "
        + (
            "stdout=None, stderr=None)\n"
            if action == "kill"
            else "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        )
        + "    while True: time.sleep(.01)\n"
        "else:\n"
        # Written whole, then renamed: `write_text` creates the file empty first, and
        # the test, which waits for the file to exist, read it empty on a loaded runner.
        "    (p / 'child.tmp').write_text(str(os.getpid()))\n"
        "    os.replace(p / 'child.tmp', p / 'child')\n"
        "    while not (p / 'release').exists(): time.sleep(.01)\n",
        encoding="utf-8",
    )
    planner = Planner(script, tmp_path)
    sp = SupervisedProcess(planner, logging.getLogger("tree-test"))
    descendants: list[psutil.Process] = []
    leaf_pids: list[int] = []

    def stop_parent(proc, **kwargs):
        # Model a launcher exiting while a native engine keeps freeing memory.
        root = psutil.Process(proc.pid)
        descendants.extend(root.children(recursive=True))
        leaf_pid = leaf_pids[0]
        for process in reversed(descendants):
            if process.pid != leaf_pid:
                with contextlib.suppress(psutil.NoSuchProcess):
                    process.kill()
        proc.kill()
        return process_signals.StopSignal.terminate_process

    monkeypatch.setattr(process_signals, "request_stop", stop_parent)
    pending = None
    try:
        sp.start()
        leaf_pids.append(await wait_for_child_pid(tmp_path / "child"))
        leaf = psutil.Process(leaf_pids[0])
        original = sp._proc
        pending = asyncio.create_task(getattr(sp, "stop" if action == "kill" else action)())
        await until(lambda: original.returncode is not None)
        await asyncio.sleep(0.1)
        assert leaf.is_running()
        assert not pending.done(), "shutdown completed while the engine child was still alive"
        assert planner.spawns == 1, "replacement launched before the old engine released resources"
        if action != "kill":
            (tmp_path / "release").touch()
        await asyncio.wait_for(pending, 5)
        assert not leaf.is_running() or leaf.status() == psutil.STATUS_ZOMBIE
    finally:
        (tmp_path / "release").touch()
        for child in descendants:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        if pending is not None:
            with contextlib.suppress(Exception):
                await pending
        await sp.stop()


async def test_runtime_is_tracked_until_stop_finishes() -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldStop:
        async def stop(self):
            entered.set()
            await release.wait()

    supervisor = RuntimeSupervisor()
    process = HeldStop()
    supervisor._processes["held"] = process
    pending = asyncio.create_task(supervisor.stop_one("held"))
    try:
        await entered.wait()
        assert supervisor._processes.get("held") is process
    finally:
        release.set()
        await pending
    assert "held" not in supervisor._processes
