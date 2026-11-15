"""Row 13: checked engine updates.

An install only stages its build; runtimes start from the build in use. A
rollout moves them one at a time, each drained first, measures old against
new when that costs nobody anything (shown, never a gate: R13-CHECK), and
holds a runtime on its old build when the new one cannot run it (R13-CHECK-2).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eugene_plexus_agent import engine_rollout, runtime_drain
from eugene_plexus_agent._generated.models import (
    EngineCheckModel,
    EngineKind,
    EngineRolloutRequest,
    EngineRolloutRuntimeState,
    EngineRolloutState,
    EngineRolloutSwitch,
    EngineSpeed,
    RuntimeDrainReason,
    RuntimeDrainReport,
    RuntimeDrainThen,
    RuntimeSpec,
    RuntimeStatus,
)
from eugene_plexus_agent.engines.acquisition import InUse, ManagedStore

ENGINE = EngineKind.llama_cpp
GW = "node:tower"


def _build(store: ManagedStore, version: str) -> None:
    directory = store.build_dir(version)
    directory.mkdir(parents=True)
    binary = directory / "llama-server"
    binary.write_bytes(version.encode())
    store.write_metadata(directory, version=version, variant="cpu", binary=binary)


def _spec(name: str) -> RuntimeSpec:
    return RuntimeSpec.model_validate(
        {"name": name, "engine": "llama_cpp", "modelPath": f"/m/{name}.gguf", "port": 9001}
    )


class Fleet:
    """The runtimes on one node, as the supervisor would compose them: each
    runs the build its store entry names, and a build in `bad` crashes."""

    def __init__(self, store: ManagedStore, names: list[str], bad: set[str] | None = None):
        self.store = store
        self.specs = {name: _spec(name) for name in names}
        self.bad = bad or set()
        self.running: dict[str, str] = {}
        self.crashed: dict[str, str] = {}
        self.drains: Any = None
        self.starts: list[tuple[str, str]] = []
        #: Runtimes still loading their model (just after a machine update).
        self.loading: set[str] = set()
        for name in names:
            self.start(name)

    # agent_state
    def list_runtime_specs(self) -> list[RuntimeSpec]:
        return list(self.specs.values())

    def get_runtime_spec(self, name: str) -> RuntimeSpec | None:
        return self.specs.get(name)

    def get_config(self) -> Any:
        return None

    # supervisor
    def is_running(self, name: str) -> bool:
        return name in self.running

    def start(self, name: str) -> None:
        build = self.store.current(runtime=name)
        assert build is not None
        self.starts.append((name, build.version))
        self.crashed.pop(name, None)
        if build.version in self.bad or f"{name}@{build.version}" in self.bad:
            self.crashed[name] = f"exit 3221225477 on {build.version}"
        else:
            self.running[name] = build.version

    async def stop_one(self, name: str, **_: Any) -> None:
        self.running.pop(name, None)

    def compose(self, spec: RuntimeSpec) -> Any:
        name = spec.name
        if name in self.crashed:
            status = RuntimeStatus.crashed
        elif name in self.loading:
            status = RuntimeStatus.loading
        elif name in self.running:
            status = RuntimeStatus.ready
            if self.drains is not None and self.drains.leaving(name):
                status = RuntimeStatus.draining
        else:
            status = RuntimeStatus.stopped
        return SimpleNamespace(
            status=status,
            engineVersion=self.running.get(name),
            lastError=self.crashed.get(name),
            url=None,
            capabilities=None,
        )


def _measured(slower: float = 0.0) -> Any:
    async def measure(rollout: Any, runtime: Any, spec: RuntimeSpec, old: Any, new: Any) -> Any:
        check = EngineCheckModel(
            runtime=spec.name,
            model=spec.name,
            passes=False,
            measured=True,
            promptTps=EngineSpeed(old=1000.0, new=1000.0),
            generationTps=EngineSpeed(old=100.0, new=100.0 * (1 - slower)),
            sameTopTokenPercent=99.8,
            standardError=0.1,
        )
        check.reason = engine_rollout.check_words(new.version, spec.name, check)
        return check

    return measure


def _setup(
    tmp_path: Path, names: list[str], bad: set[str] | None = None, slower: float = 0.0
) -> tuple[engine_rollout.EngineRollouts, ManagedStore, Fleet, Any]:
    store = ManagedStore(tmp_path / "engines", ENGINE)
    _build(store, "b1")
    store.set_in_use("b1")
    _build(store, "b2")
    fleet = Fleet(store, names, bad)
    app = SimpleNamespace(state=SimpleNamespace())
    app.state.agent_state = fleet
    app.state.runtime_supervisor = fleet

    async def run_then(name: str, then: RuntimeDrainThen, reason: RuntimeDrainReason) -> None:
        await fleet.stop_one(name)
        if then is RuntimeDrainThen.restart:
            fleet.start(name)

    app.state.runtime_drains = runtime_drain.RuntimeDrains(
        run_then=run_then, is_running=fleet.is_running
    )
    fleet.drains = app.state.runtime_drains
    rollouts = engine_rollout.EngineRollouts(app, tmp_path / "engine-rollouts.json")
    app.state.engine_rollouts = rollouts
    rollouts.store_for = lambda engine: store

    async def launch(name: str) -> str | None:
        fleet.start(name)
        return None

    async def probe(view: Any) -> str | None:
        return None

    rollouts.launch = launch
    rollouts.probe = probe
    rollouts.measure_model = _measured(slower)
    rollouts.poll_seconds = 0.01
    return rollouts, store, fleet, app


async def _ended(rollouts: engine_rollout.EngineRollouts) -> Any:
    task = rollouts._tasks[ENGINE.value]
    await asyncio.wait_for(asyncio.shield(task), 5)
    rollout = rollouts.get(ENGINE)
    assert rollout is not None
    return rollout


@pytest.fixture(autouse=True)
def _fast_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_drain, "TICK_SECONDS", 0.01)


# --- the build in use -------------------------------------------------------------


def test_install_does_not_change_the_build_in_use(tmp_path: Path) -> None:
    """A runtime that restarts while a newer build sits on disk keeps the
    build the others run: no silent mixed fleet. Before row 13 the newest
    build on disk was the one every start used."""
    store = ManagedStore(tmp_path, ENGINE)
    _build(store, "b1")
    store.pin_current()  # what an install does before it commits its build
    _build(store, "b2")

    current = store.current()
    assert current is not None and current.version == "b1"
    assert store.current(runtime="anything").version == "b1"  # type: ignore[union-attr]
    staged = store.staged()
    assert staged is not None and staged.version == "b2"


def test_prune_never_deletes_the_build_in_use_or_a_held_one(tmp_path: Path) -> None:
    store = ManagedStore(tmp_path, ENGINE)
    for version in ("b1", "b2", "b3", "b4", "b5"):
        _build(store, version)
    store.set_in_use("b2")
    store.set_in_use("b1")  # b2 is now the previous build: the way back
    store.set_runtime("held", "b3", "b1 cannot load it")
    removed = store.prune()
    left = {b.version for b in store.list_builds()}
    assert left == {"b1", "b2", "b3", "b5"}, "in use, previous, held, and one staged"
    assert removed == ["b4"]


def test_uninstall_forgets_the_build_in_use(tmp_path: Path) -> None:
    store = ManagedStore(tmp_path, ENGINE)
    _build(store, "b1")
    store.set_in_use("b1")
    store.remove_builds()
    assert store.in_use().version is None


def test_a_record_naming_a_missing_build_falls_back_to_the_newest(tmp_path: Path) -> None:
    store = ManagedStore(tmp_path, ENGINE)
    _build(store, "b1")
    (store.build_dir("b0")).mkdir()
    store.write_in_use(InUse(version="b0"))
    assert store.current().version == "b1"  # type: ignore[union-attr]


# --- the rollout --------------------------------------------------------------------


async def test_one_copy_switches_at_once_and_the_numbers_never_block(tmp_path: Path) -> None:
    """DI-C3b: no other copy, so it leaves routing at once and restarts on
    the new build; R13-CHECK: a 12% slower build is still used, and says so."""
    rollouts, store, fleet, _app = _setup(tmp_path, ["qwen"], slower=0.12)
    started = await rollouts.start(ENGINE, EngineRolloutRequest(version="b2"))
    assert started.from_ == "b1"
    rollout = await _ended(rollouts)
    assert rollout.state is EngineRolloutState.done, rollout.message
    assert store.in_use().version == "b2" and store.in_use().previous == "b1"
    assert fleet.running == {"qwen": "b2"}
    [check] = rollout.check.models
    assert check.passes and check.measured
    assert "12% slower at generating" in (check.reason or "")


async def test_the_only_copy_in_use_restarts_at_once_unmeasured(tmp_path: Path) -> None:
    """DI-C3b with a gateway routing to it: it leaves routing at once and its
    drain restarts it on the new build (a gateway counts that as coming up,
    so requests wait in line). Measuring it would keep it down for minutes,
    so it is not measured."""
    rollouts, store, fleet, app = _setup(tmp_path, ["qwen"])
    drains = app.state.runtime_drains
    drains.saw_reader(GW)

    async def gateway() -> None:
        while True:
            view = drains.view("qwen")
            if view is not None:
                drains.report(
                    "qwen",
                    GW,
                    RuntimeDrainReport(
                        running=0, state=view.state, otherCopies=0, covered=False, idleSeconds=3
                    ),
                )
            drains.saw_reader(GW)
            await asyncio.sleep(0.01)

    reporting = asyncio.create_task(gateway())
    try:
        await rollouts.start(ENGINE, EngineRolloutRequest(version="b2"))
        rollout = await _ended(rollouts)
    finally:
        reporting.cancel()
    assert rollout.state is EngineRolloutState.done, rollout.message
    assert fleet.starts == [("qwen", "b1"), ("qwen", "b2")], "one restart, no measurement stop"
    [check] = rollout.check.models
    assert check.passes and not check.measured
    assert check.reason == "Not measured: it is the only copy and was in use."
    assert store.in_use().version == "b2"


async def test_start_failure_rolls_back(tmp_path: Path) -> None:
    """R13-CHECK-2: a build that cannot run a model is not used for it. The
    runtime starts on its old build again, held there with the reason; a
    build that runs nothing here is kept off, and the old one stays in use."""
    rollouts, store, fleet, _app = _setup(tmp_path, ["qwen"], bad={"b2"})
    await rollouts.start(ENGINE, EngineRolloutRequest(version="b2"))
    rollout = await _ended(rollouts)
    assert rollout.state is EngineRolloutState.kept, rollout.message
    assert "exit 3221225477" in (rollout.message or "")
    assert store.in_use().version == "b1", "the pointer is never left on a failed build"
    assert fleet.running == {"qwen": "b1"}
    assert fleet.starts[-2:] == [("qwen", "b2"), ("qwen", "b1")]


async def test_failed_check_keeps_old_build_for_that_model_only(tmp_path: Path) -> None:
    """The other models move; the one the build cannot run is held on b1."""
    rollouts, store, fleet, _app = _setup(tmp_path, ["good", "odd"], bad={"odd@b2"})
    await rollouts.start(ENGINE, EngineRolloutRequest(version="b2"))
    rollout = await _ended(rollouts)
    assert rollout.state is EngineRolloutState.done, rollout.message
    states = {r.name: r.state for r in rollout.runtimes}
    assert states == {
        "good": EngineRolloutRuntimeState.moved,
        "odd": EngineRolloutRuntimeState.held,
    }
    record = store.in_use()
    assert record.version == "b2"
    assert record.held["odd"].version == "b1" and "b2 could not run odd" in (
        record.held["odd"].reason or ""
    )
    assert fleet.running == {"good": "b2", "odd": "b1"}
    assert "Kept b1 for odd" in (rollout.message or "")


async def test_switch_ask_waits_for_the_answer(tmp_path: Path) -> None:
    rollouts, store, fleet, _app = _setup(tmp_path, ["qwen"])
    await rollouts.start(ENGINE, EngineRolloutRequest(version="b2", switch=EngineRolloutSwitch.ask))
    for _ in range(100):
        if rollouts.get(ENGINE).state is EngineRolloutState.asking:  # type: ignore[union-attr]
            break
        await asyncio.sleep(0.01)
    asking = rollouts.get(ENGINE)
    assert asking is not None and asking.state is EngineRolloutState.asking
    assert asking.question and "Switch now, or when idle?" in asking.question
    assert fleet.running == {"qwen": "b1"}, "nothing moves before the answer"
    rollouts.answer(ENGINE, engine_rollout.EngineRolloutChoice.keep)
    rollout = await _ended(rollouts)
    assert rollout.state is EngineRolloutState.kept and store.in_use().version == "b1"


async def test_rollout_resumes_after_restart(tmp_path: Path) -> None:
    """An agent stop mid-rollout keeps its saved state; the next agent goes
    on from it and checks a runtime it had already started on the new build."""
    rollouts, store, fleet, app = _setup(tmp_path, ["qwen"])
    await rollouts.start(ENGINE, EngineRolloutRequest(version="b2", switch=EngineRolloutSwitch.ask))
    for _ in range(100):
        if rollouts.get(ENGINE).state is EngineRolloutState.asking:  # type: ignore[union-attr]
            break
        await asyncio.sleep(0.01)
    rollouts.answer(ENGINE, engine_rollout.EngineRolloutChoice.switch_now)
    await rollouts.close()
    saved = json.loads((tmp_path / "engine-rollouts.json").read_text(encoding="utf-8"))
    assert saved["rollouts"]["llama_cpp"]["choice"] == "switch_now"

    again = engine_rollout.EngineRollouts(app, tmp_path / "engine-rollouts.json")
    again.store_for = rollouts.store_for
    again.launch = rollouts.launch
    again.probe = rollouts.probe
    again.measure_model = rollouts.measure_model
    again.poll_seconds = 0.01
    app.state.engine_rollouts = again
    again.resume()
    rollout = await _ended(again)
    assert rollout.state is EngineRolloutState.done, rollout.message
    assert store.in_use().version == "b2" and fleet.running == {"qwen": "b2"}


async def test_a_rollout_resumed_mid_start_checks_the_runtime_on_the_new_build(
    tmp_path: Path,
) -> None:
    """Review #12: the agent stops while a runtime is starting on the new
    build (its entry already names it). The next agent does not drain or
    measure it again: it starts it there and checks that it answers."""
    rollouts, store, fleet, app = _setup(tmp_path, ["qwen"])
    hang = asyncio.Event()
    plain = rollouts.launch

    async def launch(name: str) -> str | None:
        await hang.wait()  # the agent stops during this start
        return await plain(name)

    rollouts.launch = launch
    await rollouts.start(ENGINE, EngineRolloutRequest(version="b2"))
    for _ in range(300):
        rollout = rollouts.get(ENGINE)
        if (
            rollout
            and rollout.runtimes
            and rollout.runtimes[0].state is (EngineRolloutRuntimeState.starting)
        ):
            break
        await asyncio.sleep(0.01)
    assert store.in_use().runtimes["qwen"].version == "b2"
    await rollouts.close()
    measured_before = [m.measured for m in rollouts.get(ENGINE).check.models]  # type: ignore[union-attr]

    again = engine_rollout.EngineRollouts(app, tmp_path / "engine-rollouts.json")
    again.store_for = rollouts.store_for
    again.launch = plain
    again.probe = rollouts.probe
    measures: list[str] = []

    async def measure(*args: Any) -> Any:
        measures.append("measured")
        return await _measured()(*args)

    again.measure_model = measure
    again.poll_seconds = 0.01
    app.state.engine_rollouts = again
    again.resume()
    rollout = await _ended(again)
    assert rollout.state is EngineRolloutState.done, rollout.message
    assert measures == [], "not measured again"
    [check] = rollout.check.models
    assert check.passes
    assert "the agent restarted during its check" in (check.reason or ""), (
        check.reason,
        measured_before,
    )
    assert fleet.running == {"qwen": "b2"} and store.in_use().version == "b2"


async def test_a_second_rollout_is_refused_while_one_runs(tmp_path: Path) -> None:
    rollouts, _store, _fleet, _app = _setup(tmp_path, ["qwen"])
    await rollouts.start(ENGINE, EngineRolloutRequest(version="b2", switch=EngineRolloutSwitch.ask))
    with pytest.raises(engine_rollout.RolloutError, match="already running"):
        await rollouts.start(ENGINE, EngineRolloutRequest(version="b2"))
    cancelled = await rollouts.cancel(ENGINE)
    assert cancelled.state is EngineRolloutState.cancelled


async def test_only_llama_cpp_is_rolled_out(tmp_path: Path) -> None:
    rollouts, _store, _fleet, _app = _setup(tmp_path, ["qwen"])
    with pytest.raises(engine_rollout.RolloutError) as refused:
        await rollouts.start(EngineKind.vllm, None)
    assert refused.value.status == 422


# --- the check's words and llama-bench --------------------------------------------


def test_read_bench_averages_pp_and_tg() -> None:
    out = "\n".join(
        [
            "noise",
            json.dumps({"n_prompt": 512, "n_gen": 0, "avg_ts": 1234.5}),
            json.dumps({"n_prompt": 0, "n_gen": 128, "avg_ts": 98.7}),
        ]
    )
    assert engine_rollout.read_bench(out) == (1234.5, 98.7)
    with pytest.raises(ValueError):
        engine_rollout.read_bench("{}")


def test_check_words_say_what_the_numbers_say() -> None:
    check = EngineCheckModel(
        runtime="r",
        model="qwen3-8b",
        passes=True,
        generationTps=EngineSpeed(old=100.0, new=88.0),
        sameTopTokenPercent=97.1,
        standardError=0.2,
    )
    words = engine_rollout.check_words("b11800", "qwen3-8b", check)
    assert (
        words
        == "b11800 on qwen3-8b: 12% slower at generating, 97.1% same next token, 99% expected."
    )


async def test_a_runtime_stopped_while_its_drain_waits_is_left_stopped(tmp_path: Path) -> None:
    """Review #8: an operator stops a runtime while its drain waits for cover.
    It was then measured and started again. Now the rollout leaves it stopped,
    and says so."""
    rollouts, _store, fleet, app = _setup(tmp_path, ["qwen"])
    drains = app.state.runtime_drains
    drains.saw_reader(GW)

    async def gateway() -> None:
        while True:
            view = drains.view("qwen")
            if view is not None:
                drains.report(
                    "qwen",
                    GW,
                    RuntimeDrainReport(
                        running=1, state=view.state, otherCopies=1, covered=False, idleSeconds=1
                    ),
                )
            drains.saw_reader(GW)
            await asyncio.sleep(0.01)

    reporting = asyncio.create_task(gateway())
    try:
        await rollouts.start(ENGINE, EngineRolloutRequest(version="b2"))
        for _ in range(100):
            if drains.view("qwen") is not None:
                break
            await asyncio.sleep(0.01)
        await fleet.stop_one("qwen")  # the operator's Stop
        rollout = await _ended(rollouts)
    finally:
        reporting.cancel()
    assert fleet.starts == [("qwen", "b1")], "it was not started again"
    assert fleet.running == {}
    [runtime] = rollout.runtimes
    assert runtime.state is EngineRolloutRuntimeState.skipped
    assert "stopped during its update" in (runtime.message or "")
    assert rollout.check.models == [], "nothing was measured"
    assert rollouts.measuring is None


async def test_a_model_still_loading_is_waited_for_and_moved(tmp_path: Path) -> None:
    """Review #9: right after a machine's own update its models are still
    loading; a rollout then saw none `ready`, moved the build in use with no
    drain or check, and said "No model was running here". It now waits for
    them (at most its ready limit) and moves them."""
    rollouts, _store, fleet, _app = _setup(tmp_path, ["qwen"])
    fleet.loading.add("qwen")

    async def loaded() -> None:
        await asyncio.sleep(0.3)
        fleet.loading.discard("qwen")

    finishing = asyncio.create_task(loaded())
    await rollouts.start(ENGINE, EngineRolloutRequest(version="b2"))
    rollout = await _ended(rollouts)
    await finishing
    assert rollout.state is EngineRolloutState.done, rollout.message
    assert [r.name for r in rollout.runtimes] == ["qwen"], rollout.message
    assert rollout.runtimes[0].state is EngineRolloutRuntimeState.moved
    assert fleet.running == {"qwen": "b2"}


async def test_a_stopped_update_leaves_no_runtime_pinned_to_its_build(tmp_path: Path) -> None:
    """Review #10: after Stop the update, a runtime already moved kept its own
    entry at the new build while the build in use stayed the old one, so it
    stayed on the new build through every restart and the stale-build advice
    ("Restarting them picks up...") was false forever. Its entry goes: it runs
    on until it restarts, then starts on the build in use."""
    rollouts, store, _fleet, _app = _setup(tmp_path, ["a", "b"])
    gate = asyncio.Event()
    plain = rollouts.launch

    held: list[str] = []

    async def launch(name: str) -> str | None:
        if name == "b" and not held:  # its first start, on b2, hangs
            held.append(name)
            await gate.wait()
        return await plain(name)

    rollouts.launch = launch
    await rollouts.start(ENGINE, EngineRolloutRequest(version="b2"))
    for _ in range(300):
        rollout = rollouts.get(ENGINE)
        if (
            rollout is not None
            and [r.state for r in rollout.runtimes][:1] == [EngineRolloutRuntimeState.moved]
            and rollout.runtimes[1].state is EngineRolloutRuntimeState.starting
        ):
            break
        await asyncio.sleep(0.01)
    assert store.in_use().runtimes.get("a") is not None, "a was moved to b2"
    cancelled = await rollouts.cancel(ENGINE)
    gate.set()
    assert cancelled.state is EngineRolloutState.cancelled
    record = store.in_use()
    assert record.version == "b1" and record.runtimes == {}, record


async def test_a_failed_update_leaves_no_runtime_pinned_to_its_build(tmp_path: Path) -> None:
    rollouts, store, fleet, _app = _setup(tmp_path, ["a", "b"])
    plain = rollouts.probe

    async def probe(view: Any) -> str | None:
        if fleet.running.get("b") == "b2":
            raise RuntimeError("the agent lost its supervisor")
        return await plain(view)

    rollouts.probe = probe
    await rollouts.start(ENGINE, EngineRolloutRequest(version="b2"))
    rollout = await _ended(rollouts)
    assert rollout.state is EngineRolloutState.failed, rollout.message
    record = store.in_use()
    assert record.version == "b1" and record.runtimes == {}, record


def test_install_and_uninstall_wait_for_a_running_update(
    authed_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review note: a second install while an update runs could prune the
    build the update moves to, and an uninstall would take it away. Both are
    refused while one runs, and the refusal names the update."""
    app = authed_client.app
    rollouts = engine_rollout.manager(app)
    monkeypatch.setattr(rollouts, "running", lambda engine: engine is ENGINE)
    rollouts._rollouts[ENGINE.value] = engine_rollout.EngineRollout.model_validate(
        {
            "id": "x",
            "engine": "llama_cpp",
            "version": "b2",
            "from": "b1",
            "state": "switching",
            "switch": "auto",
            "startedAt": "2026-11-15T00:00:00Z",
            "runtimes": [],
            "check": {"models": []},
        }
    )
    for path in ("/v1/engines/llama_cpp/install", "/v1/engines/llama_cpp/uninstall"):
        refused = authed_client.post(path, json={})
        assert refused.status_code == 409, (path, refused.text)
        assert "update of llama.cpp to b2" in refused.text, refused.text
