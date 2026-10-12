"""An app's start waits for the gateway without stopping the agent (MR6).

`SupervisedProcess._spawn_once` used to call `planner.plan()` on the event loop,
and for an app that names `{embeddingModel}` that call asks the gateway, for up to
`app_launcher.MODELS_TIMEOUT_SECONDS`: every other task in the agent -- the
console, the proxy, every health poll -- stood still for as long. A planner that
can wait now has `plan_async`, which the loop awaits; `_AppPlanner` keeps
everything that reads the agent's own state on the loop and hands only
`render_start` to a thread. The checks:

* another task keeps running while a slow lookup is in progress (the real
  supervisor, the real planner, a gateway that takes its time);
* a start that is stopped while it waits is stopped at once;
* the plan is the one `plan()` makes, and refuses what it refuses;
* two starts of one app at once are given one app secret, because the thread of
  a start that was stopped can still be running when the next one begins.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from eugene_plexus_agent import app_launcher
from eugene_plexus_agent.supervisor import SpawnPlanError, SupervisedProcess

from .fake_gateway import FakeGateway, listed
from .test_app_models import _open_webui
from .test_app_start import KEY, _manifest, _planner, _record, _store


@pytest.fixture
def gateway() -> Iterator[FakeGateway]:
    fake = FakeGateway(key=KEY)
    try:
        yield fake
    finally:
        fake.close()


async def test_a_slow_gateway_does_not_stall_the_agent(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """The whole path: the supervisor's loop, Open WebUI's planner, and a gateway
    that takes a second and a half to say which models it lists. A task that does
    nothing but count must go on counting through it."""
    gateway.models = [listed("embed-a", "embeddings")]
    gateway.mode, gateway.delay = "slow", 1.5
    process = SupervisedProcess(_open_webui(tmp_path, gateway.url), logging.getLogger("test"))
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    counting = asyncio.create_task(tick())
    began = time.perf_counter()
    process.start()
    try:
        # The start ends in a spawn that cannot happen (the app's interpreter here is
        # an empty file), which is where `last_error` is set: the plan was made first.
        while process.last_error is None:
            assert time.perf_counter() - began < 10, "the start never finished"
            await asyncio.sleep(0.01)
        elapsed = time.perf_counter() - began
        counted = ticks
    finally:
        await process.stop()
        counting.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await counting
    assert elapsed >= 1.4, "the lookup was not slow, so nothing was proved"
    # About a hundred. A loop that waited on the lookup counts one or two.
    assert counted >= 20, f"the agent counted {counted} ticks in {elapsed:.1f} s"


async def test_an_app_stopped_while_its_start_waits_is_stopped_at_once(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.mode, gateway.delay = "slow", 3.0
    process = SupervisedProcess(_open_webui(tmp_path, gateway.url), logging.getLogger("test"))
    began = time.perf_counter()
    process.start()
    while not gateway.requests:  # the lookup is under way
        assert time.perf_counter() - began < 10, "the lookup never began"
        await asyncio.sleep(0.01)
    await process.stop()
    assert time.perf_counter() - began < 1.5
    assert process.last_argv is None, "an app that was stopped was started"
    assert process._task is not None and process._task.done()


async def test_plan_async_makes_the_plan_plan_makes(tmp_path: Path, gateway: FakeGateway) -> None:
    gateway.models = [listed("embed-a", "embeddings"), listed("rerank-a", "rerank")]
    planner = _open_webui(tmp_path, gateway.url)
    made = planner.plan()
    awaited = await planner.plan_async()
    assert (awaited.argv, awaited.cwd, awaited.port) == (made.argv, made.cwd, made.port)
    # Each plan mints its own admin token; everything else is the same.
    token = "EUGENE_PLEXUS_APP_ADMIN_TOKEN"
    assert awaited.env[token] != made.env[token]
    assert {k: v for k, v in awaited.env.items() if k != token} == {
        k: v for k, v in made.env.items() if k != token
    }
    assert awaited.env["RAG_EMBEDDING_MODEL"] == "embed-a"


async def test_plan_async_refuses_what_plan_refuses(tmp_path: Path) -> None:
    # `validate_manifest` would never let this in; a planner that is handed it
    # says so the same way on either path.
    planner = _planner(_store(tmp_path), _record(_manifest(environment={"X": "{nope}"})))
    with pytest.raises(SpawnPlanError, match=r"could not fill in how fake starts: .*\{nope\}"):
        planner.plan()
    with pytest.raises(SpawnPlanError, match=r"could not fill in how fake starts: .*\{nope\}"):
        await planner.plan_async()


def test_starts_at_once_are_given_one_app_secret(tmp_path: Path) -> None:
    """`render_start` runs in a thread now, and a start that was stopped while it
    waited leaves its thread to finish beside the next start's: two of them making
    `{appSecret}` together must not each keep their own."""
    for round_ in range(100):
        data = str(tmp_path / f"data{round_}")
        got: list[str] = []
        barrier = threading.Barrier(8)

        def make(
            data: str = data, got: list[str] = got, barrier: threading.Barrier = barrier
        ) -> None:
            barrier.wait()
            got.append(app_launcher.app_secret(data))

        threads = [threading.Thread(target=make) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(got) == 8 and len(set(got)) == 1, got
