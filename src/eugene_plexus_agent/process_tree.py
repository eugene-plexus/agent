"""Wait for the descendants observed before asking their launcher to stop."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time

import psutil

GRACE_SECONDS = 30.0
KILL_SECONDS = 5.0


def snapshot(pid: int) -> list[psutil.Process]:
    """Retain process identities, not bare PIDs which the OS can reuse."""
    try:
        root = psutil.Process(pid)
        if root.ppid() != os.getpid():
            raise RuntimeError(f"refusing to inspect pid {pid}: it is not this agent's child")
        return root.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def _alive(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


async def finish(children: list[psutil.Process], *, name: str, logger: logging.Logger) -> None:
    """A bounded graceful drain, then kill only identities in this process tree.

    Children can have their own workers. Keep observing those while they exit,
    so workers created during shutdown join the same deadline.
    """
    known = set(children)

    def remaining() -> list[psutil.Process]:
        for child in list(known):
            if _alive(child):
                with contextlib.suppress(psutil.NoSuchProcess):
                    known.update(child.children(recursive=True))
        return [child for child in known if _alive(child)]

    deadline = time.perf_counter() + GRACE_SECONDS
    while alive := remaining():
        if time.perf_counter() >= deadline:
            break
        await asyncio.sleep(0.05)
    if not alive:
        return
    logger.warning(
        "%s descendants did not exit after %.1fs: killing pids %s",
        name,
        GRACE_SECONDS,
        [child.pid for child in alive],
    )
    deadline = time.perf_counter() + KILL_SECONDS
    while alive := remaining():
        for child in alive:
            # psutil checks creation time before kill; an unrelated replacement
            # which reused this PID must never be ended by this stop request.
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        if time.perf_counter() >= deadline:
            raise RuntimeError(
                f"{name}: engine descendants still alive after kill: "
                f"{[child.pid for child in alive]}"
            )
        await asyncio.sleep(0.05)
