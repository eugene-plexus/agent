"""The local copy, WIRED — not the copier itself.

Design: `docs/design/node-local-model-copy.md` §13 step 4.
`test_model_copies.py` proves the copier; every one of those tests
passes against a copier nothing calls. **A component test is not a
wiring test** (S7 produced that lesson twice in one slice), so these
drive `RuntimeSupervisor` and the route, and assert on what a console
would read: the status while it copies, the path the engine is handed,
the note when no copy was made, and that a launch still happens either
way.
"""

from __future__ import annotations

import asyncio
import os.path
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eugene_plexus_agent import model_copies as mc
from eugene_plexus_agent._generated.models import EngineKind, RuntimeSpec, RuntimeStatus
from eugene_plexus_agent.model_paths import PathRule
from eugene_plexus_agent.runtimes import RuntimeSupervisor
from eugene_plexus_agent.supervisor import ProcessState

MODEL = "/models/qwen/Qwen3-27B-Q6_K_L.gguf"


def file_exists(path: str) -> bool:
    """Stat from a sync helper: ASYNC240 rejects filesystem calls inside
    an async def, and these assertions are about a file a worker thread
    just wrote."""
    return os.path.exists(path)


SPIN_TIMEOUT_SECONDS = 10.0


def spin_until(predicate) -> None:  # type: ignore[no-untyped-def]
    """Busy-wait in the copy thread, BOUNDED.

    A fake copy has to block until the test lets it go, and it runs in a
    worker thread `asyncio.to_thread` cannot interrupt. Unbounded, a
    failing assertion leaves it spinning and pytest never exits -- which
    is how a sabotage run turned a red test into a hung suite. Bounded,
    the same case fails.
    """
    deadline = time.perf_counter() + SPIN_TIMEOUT_SECONDS
    while not predicate():
        if time.perf_counter() > deadline:
            raise TimeoutError("the fake copy was never released or cancelled")
        time.sleep(0.005)


class Spawned(Exception):
    """Raised by the stub spawn so a test can see it happened."""


@pytest.fixture
def share(tmp_path: Path) -> PathRule:
    """A Library folder mounted here, with the model really in it."""
    mount = tmp_path / "mount"
    model = mount / "qwen" / "Qwen3-27B-Q6_K_L.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"w" * 4096)
    return PathRule(source="/models", target=str(mount))


def supervisor_for(
    tmp_path: Path,
    share: PathRule,
    *,
    enabled: bool = True,
    min_free_gb: int = 0,
    spawned: list[str] | None = None,
) -> RuntimeSupervisor:
    seen = spawned if spawned is not None else []
    config: dict[str, Any] = {
        "modelCopyEnabled": enabled,
        "modelCopyDir": str(tmp_path / "copies"),
        "modelCopyMinFreeGb": min_free_gb,
    }
    sup = RuntimeSupervisor(
        get_config=config.get,
        inherited_rules=lambda: [share],
    )

    # The spawn is stubbed: this is about what happens BEFORE a process
    # exists, and a real `llama-server` is not available in CI. It still
    # registers the planner and process the real one would, because
    # "which copies are in use" is read off those -- a stub that skipped
    # it would make every copy look unused and quietly prove the wrong
    # thing about Clear.
    def fake_spawn(spec: RuntimeSpec, adapter: Any) -> None:
        seen.append(spec.name)
        sup._planners[spec.name] = SimpleNamespace(spec=spec, binary=None)

        async def stop() -> None:
            return None

        async def restart() -> None:
            seen.append(spec.name)

        sup._processes[spec.name] = SimpleNamespace(
            state=ProcessState.starting,
            last_error=None,
            last_restart=None,
            pid=1234,
            last_argv=None,
            stop=stop,
            restart=restart,
        )

    sup._spawn = fake_spawn  # type: ignore[assignment]
    return sup


def spec() -> RuntimeSpec:
    return RuntimeSpec(name="qwen", engine=EngineKind.llama_cpp, modelPath=MODEL, port=8090)


@pytest.mark.anyio
async def test_a_runtime_reports_copying_before_anything_is_spawned(
    tmp_path: Path, share: PathRule
) -> None:
    """The status this slice added, on the wire, in the state it names.

    `copying` is not `starting`: there is no process, and the contract's
    own word for `starting` is "spawned". Four minutes reported as
    `stopped` or `starting` is the window that got a healthy node
    diagnosed as a broken one.
    """
    spawned: list[str] = []
    sup = supervisor_for(tmp_path, share, spawned=spawned)
    started = asyncio.Event()

    real_copy = mc.copy_file

    def slow_copy(plan, settings, state, **kwargs):  # type: ignore[no-untyped-def]
        started.set()
        cancel = kwargs.get("should_cancel", lambda: False)
        spin_until(lambda: release.is_set() or cancel())
        real_copy(plan, settings, state, **kwargs)

    release = asyncio.Event()
    mc_copy = mc.copy_file
    try:
        mc.copy_file = slow_copy  # type: ignore[assignment]
        sup.add_and_start(spec())
        await asyncio.wait_for(started.wait(), timeout=5)

        view = sup.compose(spec())
        assert view.status is RuntimeStatus.copying
        assert view.copyProgress is not None
        assert view.copyProgress.totalBytes == 4096
        assert spawned == [], "nothing may be spawned while the copy runs"

        release.set()
        await asyncio.wait_for(asyncio.shield(sup._copy_jobs["qwen"].task), timeout=5)
    finally:
        mc.copy_file = mc_copy  # type: ignore[assignment]
    assert spawned == ["qwen"], "the engine starts once the copy lands"


@pytest.mark.anyio
async def test_once_copied_the_engine_is_handed_the_local_file(
    tmp_path: Path, share: PathRule
) -> None:
    sup = supervisor_for(tmp_path, share)
    sup.add_and_start(spec())
    await asyncio.wait_for(asyncio.shield(sup._copy_jobs["qwen"].task), timeout=10)

    view = sup.compose(spec())
    assert view.localPathSource is not None and view.localPathSource.value == "copy"
    assert Path(view.localPath or "").is_relative_to(tmp_path / "copies")
    assert view.modelPath == MODEL, "the declaration keeps the library's spelling"
    assert view.localPathNote is None
    assert view.copyProgress is None, "progress is for a copy in flight, not a finished one"


@pytest.mark.anyio
async def test_with_the_toggle_off_nothing_is_copied_and_the_share_is_opened(
    tmp_path: Path, share: PathRule
) -> None:
    spawned: list[str] = []
    sup = supervisor_for(tmp_path, share, enabled=False, spawned=spawned)
    sup.add_and_start(spec())

    assert spawned == ["qwen"], "no copy means no wait: it spawns immediately"
    view = sup.compose(spec())
    assert view.localPathSource is not None and view.localPathSource.value != "copy"
    assert view.status is not RuntimeStatus.copying


@pytest.mark.anyio
async def test_a_copy_that_will_not_fit_still_launches_and_says_why(
    tmp_path: Path, share: PathRule, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode this feature has: the model still serves and the
    only symptom is a start that is minutes slower than expected. So the
    reason is on the runtime, in words, where the Inference screen reads
    it."""
    spawned: list[str] = []
    sup = supervisor_for(tmp_path, share, min_free_gb=50, spawned=spawned)
    monkeypatch.setattr(mc, "free_bytes", lambda _d: 1 * mc.GIB)

    sup.add_and_start(spec())

    assert spawned == ["qwen"], "a launch never fails because of a copy"
    view = sup.compose(spec())
    assert view.localPathSource is not None and view.localPathSource.value != "copy"
    assert view.localPathNote is not None
    assert "free" in view.localPathNote
    assert str(Path(view.localPath or "")) != ""


@pytest.mark.anyio
async def test_a_copy_that_fails_outright_still_launches_and_says_why(
    tmp_path: Path, share: PathRule, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[str] = []
    sup = supervisor_for(tmp_path, share, spawned=spawned)

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("the share went away")

    monkeypatch.setattr(mc, "copy_file", boom)
    sup.add_and_start(spec())
    await asyncio.wait_for(asyncio.shield(sup._copy_jobs["qwen"].task), timeout=5)

    assert spawned == ["qwen"]
    view = sup.compose(spec())
    assert view.localPathNote is not None and "share went away" in view.localPathNote


@pytest.mark.anyio
async def test_stopping_a_runtime_mid_copy_cancels_it_and_leaves_no_partial(
    tmp_path: Path, share: PathRule
) -> None:
    """A stop must not be answered in four minutes' time by a process
    starting up."""
    spawned: list[str] = []
    sup = supervisor_for(tmp_path, share, spawned=spawned)
    entered = asyncio.Event()

    def blocking_copy(plan, settings, state, **kwargs):  # type: ignore[no-untyped-def]
        entered.set()
        cancel = kwargs.get("should_cancel", lambda: False)
        spin_until(cancel)
        Path(plan.partial).unlink(missing_ok=True)
        raise mc.CopyAborted("the copy was cancelled")

    original = mc.copy_file
    try:
        mc.copy_file = blocking_copy  # type: ignore[assignment]
        sup.add_and_start(spec())
        await asyncio.wait_for(entered.wait(), timeout=5)
        await sup.stop_one("qwen")
    finally:
        mc.copy_file = original  # type: ignore[assignment]

    assert spawned == [], "a cancelled copy does not then start the engine"
    assert "qwen" not in sup._copy_jobs
    assert not list((tmp_path / "copies").rglob("*.ep-partial"))


@pytest.mark.anyio
async def test_a_copying_runtime_counts_as_running(tmp_path: Path, share: PathRule) -> None:
    """Otherwise a second Start begins a second copy of the same 25 GB
    file into the same destination."""
    sup = supervisor_for(tmp_path, share)
    entered = asyncio.Event()

    def blocking_copy(plan, settings, state, **kwargs):  # type: ignore[no-untyped-def]
        entered.set()
        cancel = kwargs.get("should_cancel", lambda: False)
        spin_until(cancel)
        raise mc.CopyAborted("cancelled")

    original = mc.copy_file
    try:
        mc.copy_file = blocking_copy  # type: ignore[assignment]
        sup.add_and_start(spec())
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert sup.is_running("qwen") is True
    finally:
        mc.copy_file = original  # type: ignore[assignment]
        await sup.stop_one("qwen")


@pytest.mark.anyio
async def test_a_declaration_that_goes_away_takes_its_copy_with_it(
    tmp_path: Path, share: PathRule
) -> None:
    sup = supervisor_for(tmp_path, share)
    sup.add_and_start(spec())
    await asyncio.wait_for(asyncio.shield(sup._copy_jobs["qwen"].task), timeout=10)
    copy = sup.compose(spec()).localPath
    assert copy is not None and file_exists(copy)

    # The real sequence: the runtime is stopped and undeclared, and only
    # then is the copy unwanted. Reconciling while it is still running
    # would (correctly) skip it -- nothing here deletes a file an engine
    # has open.
    await sup.remove_and_stop("qwen")
    sup.reconcile_copies([])

    assert not file_exists(copy)


@pytest.mark.anyio
async def test_clear_reports_what_it_skipped_and_stops_nothing(
    tmp_path: Path, share: PathRule
) -> None:
    sup = supervisor_for(tmp_path, share)
    sup.add_and_start(spec())
    await asyncio.wait_for(asyncio.shield(sup._copy_jobs["qwen"].task), timeout=10)
    copy = sup.compose(spec()).localPath
    assert copy is not None

    result = sup.clear_copies()

    assert result.deleted == []
    assert [s.runtime for s in result.skipped] == ["qwen"]
    assert file_exists(copy)


@pytest.mark.anyio
async def test_restart_makes_the_copy_a_plain_restart_would_have_skipped(
    tmp_path: Path, share: PathRule
) -> None:
    """Restart is the gesture right after switching copying on.

    `SupervisedProcess.restart` re-plans in place, which resolves the
    path but never passes through the code that COPIES -- so before
    this, pressing Restart opened the share again and explained
    nothing. Found on the live install, which is the first thing that
    ever pressed it.
    """
    spawned: list[str] = []
    sup = supervisor_for(tmp_path, share, enabled=False, spawned=spawned)
    sup.add_and_start(spec())
    assert spawned == ["qwen"]
    assert sup.compose(spec()).localPathSource is not None
    assert sup.compose(spec()).localPathSource.value != "copy"

    # The operator switches copying on and presses Restart.
    sup._get_config = {  # type: ignore[assignment]
        "modelCopyEnabled": True,
        "modelCopyDir": str(tmp_path / "copies"),
        "modelCopyMinFreeGb": 0,
    }.get
    assert await sup.restart("qwen") is True
    job = sup._copy_jobs.get("qwen")
    assert job is not None, "a restart with copying on must go through the copy path"
    await asyncio.wait_for(asyncio.shield(job.task), timeout=10)

    assert spawned == ["qwen", "qwen"]
    assert sup.compose(spec()).localPathSource.value == "copy"


@pytest.mark.anyio
async def test_restart_with_a_current_copy_stays_a_plain_restart(
    tmp_path: Path, share: PathRule
) -> None:
    """Nothing is re-copied for a restart, which is the whole point:
    the second start is the fast one."""
    sup = supervisor_for(tmp_path, share)
    sup.add_and_start(spec())
    await asyncio.wait_for(asyncio.shield(sup._copy_jobs["qwen"].task), timeout=10)

    assert await sup.restart("qwen") is True
    assert "qwen" not in sup._copy_jobs
