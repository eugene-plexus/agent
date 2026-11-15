"""Checked engine updates (row 13): move this node's runtimes to a new build.

`specs/docs/design/rows/row-13.md` §2.3-§2.4, with Troy's answers of
2026-10-21, which override the design's body where they differ:

* **DI-C3**: with two or more copies of a model, one copy moves at a time, and
  only while the others carry the current demand (the drain's `when:
  covered`, measured by each gateway from its queue and slots in use).
* **DI-C3b**: a model with one copy moves at once; its requests wait in line.
* **R13-CHECK**: speed and answers are measured and shown, **never a gate**.
* **R13-CHECK-2**: a build that cannot load a model, crashes, or answers with
  an error is not used for that model (it is *held* on its old build, and the
  page says what failed). The other models move. The next rollout tries it
  again.

One runtime at a time: drain it (`then: stop`), measure it when that costs
nobody anything (its other copies carry the load, it was idle 600 s, or no
gateway routes to it; at most 4 models), start it on the new build, and check
that it loads and answers. The build in use moves only at the end, so a crash
mid-rollout leaves every runtime on a build it ran (a runtime's own entry is
written just before its start, and a resumed rollout checks it again).

The state is saved (`engine-rollouts.json` beside the agent's config): an
agent restart resumes it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import orphan_kill, runtime_drain
from ._generated.models import (
    CacheType,
    EngineCheckModel,
    EngineKind,
    EngineRollout,
    EngineRolloutCheck,
    EngineRolloutChoice,
    EngineRolloutRequest,
    EngineRolloutRuntime,
    EngineRolloutRuntimeState,
    EngineRolloutState,
    EngineRolloutSwitch,
    EngineSpeed,
    Runtime,
    RuntimeDrain,
    RuntimeDrainReason,
    RuntimeDrainRequest,
    RuntimeDrainState,
    RuntimeDrainThen,
    RuntimeDrainWhen,
    RuntimeSpec,
    RuntimeStatus,
    State,
    StopReason,
)
from ._http import shared_internal_client
from ._private_files import write_private
from .engines.acquisition import InstalledBuild, ManagedStore, RuntimeBuild

log = logging.getLogger(__name__)

FILE_NAME = "engine-rollouts.json"
#: Models measured in one rollout at most (each costs minutes on one GPU).
CHECK_MODELS = 4
#: How long a runtime has to become ready on the new build.
READY_SECONDS = 600.0
READY_POLL_SECONDS = 1.0
#: A single copy is measured only after this much idleness.
MEASURE_IDLE_SECONDS = 600
#: The drain's grace (R13-DRAIN, Troy 2026-10-21).
GRACE_SECONDS = 120
#: One tool run of the check, at most.
TOOL_SECONDS = 900.0
MAX_OUTPUT = 4 * 1024 * 1024
#: The answers check (the profile builder's instrument): 2 chunks of up to
#: 4,096 tokens of the bundled text.
QUALITY_CONTEXT = 4096
QUALITY_CHUNKS = 2
#: Two speeds this close are the same speed (llama-bench's own spread).
SAME_SPEED = 0.01
#: Same answers: at least this share of the same top token, less its error.
SAME_ANSWERS = 99.0
TERMINAL = frozenset(
    {
        EngineRolloutState.done,
        EngineRolloutState.kept,
        EngineRolloutState.failed,
        EngineRolloutState.cancelled,
    }
)
#: A runtime on its way up, which a rollout waits for before it chooses.
_COMING_UP = frozenset(
    {RuntimeStatus.copying, RuntimeStatus.starting, RuntimeStatus.loading, RuntimeStatus.exited}
)
FINISHED_RUNTIME = frozenset(
    {
        EngineRolloutRuntimeState.moved,
        EngineRolloutRuntimeState.held,
        EngineRolloutRuntimeState.skipped,
    }
)


class RolloutError(Exception):
    def __init__(self, detail: str, status: int = 409) -> None:
        super().__init__(detail)
        self.status = status


def _now() -> datetime:
    return datetime.now(UTC)


def _tail(text: str, limit: int = 300) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    tail = lines[-1] if lines else ""
    return tail[-limit:]


def _model_name(spec: RuntimeSpec) -> str:
    if spec.modelAlias:
        return spec.modelAlias
    stem = Path(spec.modelPath).name
    return stem.rsplit(".", 1)[0] if "." in stem else stem


# --------------------------------------------------------------------------- #
# The check: speed and answers, old build against new
# --------------------------------------------------------------------------- #


async def run_tool(
    argv: list[str], *, env: dict[str, str], seconds: float = TOOL_SECONDS
) -> tuple[int, str]:
    """Run one tool to its end; its exit code and output (stdout, then
    stderr), bounded. A tool that outruns `seconds` is ended (exit -1)."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(Path(argv[0]).parent),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        **orphan_kill.kwargs_for_platform(),
    )
    job = orphan_kill.windows_job()
    if job is not None:
        job.assign(process.pid)
    try:
        out, err = await asyncio.wait_for(process.communicate(), seconds)
    except TimeoutError:
        await _end_tool(process)
        return -1, f"{Path(argv[0]).name} did not finish within {seconds:.0f} s"
    except BaseException:
        # Stop the update, or the agent's own stop, during a measurement: the
        # tool and what it started end too, rather than keep the GPU (review #11).
        await asyncio.shield(_end_tool(process))
        raise
    text = (out[-MAX_OUTPUT:] + b"\n" + err[-MAX_OUTPUT:]).decode("utf-8", errors="replace")
    return int(process.returncode or 0), text


async def _end_tool(process: asyncio.subprocess.Process) -> None:
    """End a tool and every process under it, then reap it."""
    import psutil

    with contextlib.suppress(psutil.Error):
        for child in psutil.Process(process.pid).children(recursive=True):
            with contextlib.suppress(psutil.Error):
                child.kill()
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(process.wait(), 10)


def _tool(build: InstalledBuild, name: str) -> Path | None:
    suffix = ".exe" if os.name == "nt" else ""
    path = build.binary.with_name(name + suffix)
    return path if path.is_file() else None


def read_bench(text: str) -> tuple[float, float]:
    """`(prompt, generation)` tokens a second from llama-bench's jsonl:
    pp512 and tg128, each averaged over its repetitions."""
    prompt = generation = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        rate = row.get("avg_ts")
        if not isinstance(rate, int | float) or not math.isfinite(rate) or rate <= 0:
            continue
        if row.get("n_prompt", 0) > 0 and row.get("n_gen", 0) == 0:
            prompt = float(rate)
        elif row.get("n_gen", 0) > 0 and row.get("n_prompt", 0) == 0:
            generation = float(rate)
    if prompt is None or generation is None:
        raise ValueError("llama-bench did not report pp512 and tg128")
    return prompt, generation


def _change(old: float | None, new: float | None) -> float | None:
    if not old or new is None:
        return None
    return (new - old) / old


def check_words(version: str, model: str, check: EngineCheckModel) -> str:
    """What the numbers say, in one sentence (R13-CHECK: shown, never a gate):
    "b11800 on qwen3-8b: 3% faster at generating, same answers."."""
    parts: list[str] = []
    speed = _change(
        check.generationTps.old if check.generationTps else None,
        check.generationTps.new if check.generationTps else None,
    )
    if speed is not None:
        if abs(speed) < SAME_SPEED:
            parts.append("as fast at generating")
        else:
            parts.append(
                f"{abs(speed) * 100:.0f}% {'faster' if speed > 0 else 'slower'} at generating"
            )
    if check.sameTopTokenPercent is not None:
        same = check.sameTopTokenPercent - (check.standardError or 0)
        parts.append(
            "same answers"
            if same >= SAME_ANSWERS
            else f"{check.sameTopTokenPercent:.1f}% same next token, {SAME_ANSWERS:g}% expected"
        )
    return f"{version} on {model}: {', '.join(parts)}." if parts else ""


class EngineCheck:
    """Measures one model on two builds, with the runtime stopped: speed with
    llama-bench, answers with llama-perplexity's KL divergence and same top
    token on the bundled text (the profile builder's instrument)."""

    def __init__(self, work: Path, *, run: Callable[..., Awaitable[tuple[int, str]]] = run_tool):
        self.work = work
        self.run = run

    async def measure(
        self,
        *,
        model: Path,
        name: str,
        flags: dict[str, Any],
        env: dict[str, str],
        old: InstalledBuild,
        new: InstalledBuild,
    ) -> tuple[EngineCheckModel, list[str]]:
        """The measured fields, and what could not be measured, in words."""
        from .profile_builds import passthrough_args

        notes: list[str] = []
        check = EngineCheckModel(runtime="", model=name, passes=False, measured=False)
        benches = {b: _tool(b, "llama-bench") for b in (old, new)}
        if all(benches.values()):
            speeds: dict[str, tuple[float, float]] = {}
            for build, tool in benches.items():
                assert tool is not None
                code, out = await self.run(
                    [
                        str(tool),
                        "--model",
                        str(model),
                        "--n-prompt",
                        "512",
                        "--n-gen",
                        "128",
                        "--repetitions",
                        "3",
                        "--output",
                        "jsonl",
                        *passthrough_args(flags, bench=True),
                    ],
                    env=env,
                )
                try:
                    if code != 0:
                        raise ValueError(f"exit {code}. {_tail(out)}")
                    speeds[build.version] = read_bench(out)
                except ValueError as exc:
                    notes.append(f"llama-bench of {build.version} failed: {exc}")
                    break
            if len(speeds) == 2:
                check.promptTps = EngineSpeed(
                    old=speeds[old.version][0], new=speeds[new.version][0]
                )
                check.generationTps = EngineSpeed(
                    old=speeds[old.version][1], new=speeds[new.version][1]
                )
                check.measured = True
        else:
            missing = next(b.version for b, t in benches.items() if t is None)
            notes.append(f"{missing} has no llama-bench beside its llama-server")
        quality = await self._answers(model, flags, env, old, new)
        if isinstance(quality, str):
            notes.append(quality)
        else:
            check.sameTopTokenPercent, check.standardError, check.meanKld = quality
            check.measured = True
        return check, notes

    async def _answers(
        self,
        model: Path,
        flags: dict[str, Any],
        env: dict[str, str],
        old: InstalledBuild,
        new: InstalledBuild,
    ) -> tuple[float, float, float] | str:
        from .gguf_context import trained_context
        from .profile_builds import (
            VOCAB_BOUND,
            BuildError,
            bundled_text,
            passthrough_args,
            quality_from_output,
            too_short,
        )

        tools = {b: _tool(b, "llama-perplexity") for b in (old, new)}
        if not all(tools.values()):
            missing = next(b.version for b, t in tools.items() if t is None)
            return f"{missing} has no llama-perplexity beside its llama-server"
        trained = await asyncio.to_thread(trained_context, model)
        context = min(QUALITY_CONTEXT, trained) if trained else QUALITY_CONTEXT
        self.work.mkdir(parents=True, exist_ok=True)
        need = QUALITY_CHUNKS * (context // 2) * VOCAB_BOUND * 2
        free = shutil.disk_usage(self.work).free
        if free < need:
            return (
                f"the answers check needs {need / 2**30:.1f} GiB free beside the agent's "
                f"settings, and there is {free / 2**30:.1f} GiB"
            )
        base = self.work / f"{uuid4().hex}.kld"
        common = [
            "--model",
            str(model),
            "--file",
            str(bundled_text()),
            "--ctx-size",
            str(context),
            "--chunks",
            str(QUALITY_CHUNKS),
            *passthrough_args(flags, bench=False),
        ]
        try:
            old_tool, new_tool = tools[old], tools[new]
            assert old_tool is not None and new_tool is not None
            code, out = await self.run(
                [str(old_tool), *common, "--kl-divergence-base", str(base)], env=env
            )
            if short := too_short(out):
                return f"the bundled text gives {short[1]:,} tokens; {short[0]:,} are needed"
            if code != 0 or not base.is_file():
                return (
                    f"{old.version} could not write its comparison file (exit {code}): {_tail(out)}"
                )
            code, out = await self.run(
                [str(new_tool), *common, "--kl-divergence-base", str(base), "--kl-divergence"],
                env=env,
            )
            if code != 0:
                # The design's expert case: the format changed between builds.
                return f"{new.version} cannot read {old.version}'s comparison file: {_tail(out)}"
            try:
                quality = quality_from_output(out, CacheType.f16, SAME_ANSWERS)
            except BuildError as exc:
                return f"{new.version} gave no comparison: {exc}"
            return quality.sameTopTokenPercent, quality.standardError, quality.meanKld
        finally:
            with contextlib.suppress(OSError):
                base.unlink()


# --------------------------------------------------------------------------- #
# The rollout
# --------------------------------------------------------------------------- #


class EngineRollouts:
    """One rollout per engine on this node, saved across restarts."""

    def __init__(self, app: Any, path: Path) -> None:
        self.app = app
        self.path = path
        self._rollouts: dict[str, EngineRollout] = {}
        self._choices: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._answers: dict[str, asyncio.Future[EngineRolloutChoice]] = {}
        self._closing = False
        #: The runtime being measured now: the node's one measurement job.
        self.measuring: str | None = None
        self.check = EngineCheck(path.parent / "engine-checks")
        #: Seams, which tests replace: the engine's store, how a stopped
        #: runtime is started (as the Start button does), how one model is
        #: measured, whether a started runtime answers, and how its readiness
        #: is polled.
        self.store_for: Callable[[EngineKind], ManagedStore] = _store_for
        self.launch: Callable[[str], Awaitable[str | None]] = self._launch_runtime
        self.measure_model: Callable[..., Awaitable[EngineCheckModel]] = self._measure
        self.probe: Callable[[Runtime], Awaitable[str | None]] = self._probe
        self.ready_seconds = READY_SECONDS
        self.poll_seconds = READY_POLL_SECONDS
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            for engine, entry in (saved.get("rollouts") or {}).items():
                self._rollouts[engine] = EngineRollout.model_validate(entry["rollout"])
                if entry.get("choice"):
                    self._choices[engine] = str(entry["choice"])
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            # A corrupt record must not stop the agent starting; the build in
            # use is in the engine store, which this file never overrides.
            log.exception("Cannot read engine rollouts at %s; starting empty", path)

    # --- reading -----------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether a check is measuring on this node: one measurement at a
        time, as benchmarks and settings builds share (`measurement_node`)."""
        return self.measuring is not None

    def get(self, engine: EngineKind) -> EngineRollout | None:
        return self._rollouts.get(engine.value)

    def running(self, engine: EngineKind) -> bool:
        task = self._tasks.get(engine.value)
        return task is not None and not task.done()

    def _save(self) -> None:
        body = {
            "rollouts": {
                engine: {
                    "rollout": rollout.model_dump(mode="json", by_alias=True, exclude_none=True),
                    **({"choice": self._choices[engine]} if engine in self._choices else {}),
                }
                for engine, rollout in self._rollouts.items()
            }
        }
        write_private(self.path, json.dumps(body, indent=2))

    # --- changing ----------------------------------------------------------

    async def start(self, engine: EngineKind, body: EngineRolloutRequest | None) -> EngineRollout:
        if engine is not EngineKind.llama_cpp:
            raise RolloutError(
                f"Checked updates are for llama.cpp; {engine.value} switches when it installs.", 422
            )
        if self.running(engine):
            raise RolloutError(f"An update of {engine.value} is already running on this node.")
        store = self.store_for(engine)
        if not store.list_builds():
            raise RolloutError(
                f"Eugene does not manage {engine.value} on this node: install it first.", 422
            )
        body = body or EngineRolloutRequest()
        version = body.version
        if version is None:
            staged = store.staged()
            version = staged.version if staged is not None else None
        if version is None:
            from .runtimes import plan_for

            # Upstream is read in a thread: it may take seconds.
            latest = await asyncio.to_thread(plan_for, engine)
            version = getattr(latest, "version", None)
            if version is None:
                raise RolloutError(
                    "No build to move to: nothing is staged here, and upstream did not "
                    f"answer with one ({getattr(latest, 'reason', 'no answer')}).",
                    422,
                )
        if self.running(engine):  # a second start while upstream answered
            raise RolloutError(f"An update of {engine.value} is already running on this node.")
        store.build_dir(version)  # refuses an unsafe name before anything starts
        rollout = EngineRollout.model_validate(
            {
                "id": uuid4().hex,
                "engine": engine,
                "version": version,
                "from": store.in_use().version,
                "state": EngineRolloutState.installing,
                "switch": body.switch or EngineRolloutSwitch.auto,
                "startedAt": _now(),
                "runtimes": [],
                "check": EngineRolloutCheck(models=[]),
                "message": f"Getting {version}.",
            }
        )
        self._rollouts[engine.value] = rollout
        self._choices.pop(engine.value, None)
        self._save()
        self._launch(engine)
        return rollout

    def _launch(self, engine: EngineKind) -> None:
        self._tasks[engine.value] = asyncio.create_task(
            self._run(engine), name=f"engine-rollout:{engine.value}"
        )

    def resume(self) -> None:
        """At boot: carry on with every rollout that had not ended."""
        for engine, rollout in self._rollouts.items():
            if rollout.state not in TERMINAL and not self.running(EngineKind(engine)):
                log.info("resuming the %s rollout to %s", engine, rollout.version)
                self._launch(EngineKind(engine))

    def answer(self, engine: EngineKind, choice: EngineRolloutChoice) -> EngineRollout:
        rollout = self._rollouts.get(engine.value)
        if rollout is None or rollout.state is not EngineRolloutState.asking:
            raise RolloutError("Nothing is waiting for an answer.")
        self._choices[engine.value] = choice.value
        rollout.question = None
        self._save()
        future = self._answers.get(engine.value)
        if future is not None and not future.done():
            future.set_result(choice)
        return rollout

    async def cancel(self, engine: EngineKind) -> EngineRollout:
        rollout = self._rollouts.get(engine.value)
        task = self._tasks.get(engine.value)
        if rollout is None or task is None or task.done():
            raise RolloutError("No update is running for this engine.")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, BaseException):
            await task
        if rollout.state not in TERMINAL:
            # Cancelled before its work began: nothing moved.
            self._end(rollout, EngineRolloutState.cancelled, "Cancelled. Nothing moved.")
        return rollout

    async def close(self) -> None:
        """At shutdown: stop working and keep the saved state for a resume."""
        self._closing = True
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    # --- the work ----------------------------------------------------------

    def _supervisor(self) -> Any:
        return getattr(self.app.state, "runtime_supervisor", None)

    def _state(self) -> Any:
        return self.app.state.agent_state

    def _view(self, spec: RuntimeSpec) -> Runtime | None:
        supervisor = self._supervisor()
        return supervisor.compose(spec) if supervisor is not None else None

    def _set(self, rollout: EngineRollout, **changes: Any) -> None:
        for key, value in changes.items():
            setattr(rollout, key, value)
        self._save()

    def _progress(self, rollout: EngineRollout) -> str:
        moved = sum(1 for r in rollout.runtimes if r.state is EngineRolloutRuntimeState.moved)
        total = len(rollout.runtimes)
        return f"{moved} of {total} runtime{'s' if total != 1 else ''} on {rollout.version}"

    async def _run(self, engine: EngineKind) -> None:
        rollout = self._rollouts[engine.value]
        current: EngineRolloutRuntime | None = None
        try:
            if not await self._install(engine, rollout):
                return
            store = self.store_for(engine)
            builds = {b.version: b for b in store.list_builds()}
            new = builds.get(rollout.version)
            if new is None:
                self._end(rollout, EngineRolloutState.failed, f"{rollout.version} is not on disk.")
                return
            record = store.in_use()
            if record.version is None or (
                record.version == rollout.version and not record.runtimes
            ):
                store.set_in_use(rollout.version)
                self._end(rollout, EngineRolloutState.done, f"{rollout.version} is in use.")
                return
            if not rollout.runtimes:
                await self._settle(engine)
                rollout.runtimes = [
                    EngineRolloutRuntime(name=name, state=EngineRolloutRuntimeState.waiting)
                    for name in self._targets(engine, rollout.version)
                ]
                self._save()
            if not rollout.runtimes:
                store.set_in_use(rollout.version)
                self._end(
                    rollout,
                    EngineRolloutState.done,
                    f"{rollout.version} is in use. No model was running here, so none was checked.",
                )
                return
            choice = await self._ask(engine, rollout)
            if choice is EngineRolloutChoice.keep:
                self._end(
                    rollout,
                    EngineRolloutState.kept,
                    f"Kept {rollout.from_ or 'the build in use'}, as you chose.",
                )
                return
            when = (
                RuntimeDrainWhen.idle
                if choice is EngineRolloutChoice.switch_idle
                else RuntimeDrainWhen.covered
            )
            for current in rollout.runtimes:
                if current.state in FINISHED_RUNTIME:
                    continue
                await self._move(store, rollout, current, new, when)
                current = None
            self._finish(store, rollout)
        except asyncio.CancelledError:
            if not self._closing:
                await self._undo(engine, rollout, current)
            raise
        except Exception as exc:
            log.exception("the %s rollout to %s failed", engine.value, rollout.version)
            with contextlib.suppress(Exception):
                self._forget_moves(self.store_for(engine), rollout)
            self._end(rollout, EngineRolloutState.failed, f"The update stopped: {exc}")

    @staticmethod
    def _forget_moves(store: ManagedStore, rollout: EngineRollout) -> None:
        """A rollout that ends without moving the build in use (stopped, or
        failed) leaves no runtime pinned to its build (review #10): each one
        moved runs on until it restarts, then starts on the build in use, so
        the fleet comes back together. Holds (with a reason) stay."""
        for name, entry in store.in_use().runtimes.items():
            if entry.version == rollout.version and not entry.reason:
                store.set_runtime(name, None)

    def _end(self, rollout: EngineRollout, state: EngineRolloutState, message: str) -> None:
        self._set(rollout, state=state, message=message, finishedAt=_now(), question=None)
        log.info(
            "%s rollout to %s: %s: %s", rollout.engine.value, rollout.version, state.value, message
        )

    async def _install(self, engine: EngineKind, rollout: EngineRollout) -> bool:
        """Get the build onto disk when it is not there. Whether to go on."""
        from .runtimes import installer_for, plan_for

        if any(b.version == rollout.version for b in self.store_for(engine).list_builds()):
            return True
        installer = installer_for(engine)
        assert installer is not None
        self._set(
            rollout, state=EngineRolloutState.installing, message=f"Getting {rollout.version}."
        )
        if not installer.running:
            plan = await asyncio.to_thread(plan_for, engine, version=rollout.version)
            reason = getattr(plan, "reason", None)
            if reason is not None:
                self._end(rollout, EngineRolloutState.failed, f"{rollout.version}: {reason}")
                return False
            installer.start(plan)  # type: ignore[arg-type]
        done = await installer.wait()
        if done is None or done.state is not State.done:
            error = (done.error if done is not None else None) or "the install did not finish"
            self._end(
                rollout, EngineRolloutState.failed, f"{rollout.version} did not install: {error}"
            )
            return False
        return True

    async def _settle(self, engine: EngineKind) -> None:
        """Wait, at most `ready_seconds`, while a runtime of `engine` is on
        its way up (copying, starting, loading, or restarting after an exit):
        right after a machine's own update its models are still loading, and
        a rollout that went on would move the build in use under them with no
        drain or check (review #9)."""
        deadline = time.perf_counter() + self.ready_seconds
        while time.perf_counter() < deadline:
            coming = [
                spec.name
                for spec in self._state().list_runtime_specs()
                if spec.engine is engine
                and not spec.binary
                and (view := self._view(spec)) is not None
                and view.status in _COMING_UP
            ]
            if not coming:
                return
            self._set(
                self._rollouts[engine.value],
                message=f"Waiting for {', '.join(coming)} to finish loading.",
            )
            await asyncio.sleep(self.poll_seconds)

    def _targets(self, engine: EngineKind, version: str) -> list[str]:
        """This node's runtimes of `engine` on a managed build (not those with
        their own `binary`), serving now, and not on `version` already."""
        names: list[str] = []
        for spec in self._state().list_runtime_specs():
            if spec.engine is not engine or spec.binary:
                continue
            view = self._view(spec)
            if view is None or view.status is not RuntimeStatus.ready:
                continue
            if view.engineVersion == version:
                continue
            names.append(spec.name)
        return names

    async def _ask(self, engine: EngineKind, rollout: EngineRollout) -> EngineRolloutChoice:
        """`switch: ask` waits here for the person's answer; `auto` does not."""
        saved = self._choices.get(engine.value)
        if saved is not None:
            return EngineRolloutChoice(saved)
        if rollout.switch is not EngineRolloutSwitch.ask:
            return EngineRolloutChoice.switch_now
        future: asyncio.Future[EngineRolloutChoice] = asyncio.get_running_loop().create_future()
        self._answers[engine.value] = future
        self._set(
            rollout,
            state=EngineRolloutState.asking,
            question=f"{rollout.version} is ready. Switch now, or when idle?",
            message=f"{rollout.version} is ready. Switch now, or when idle?",
        )
        try:
            return await future
        finally:
            self._answers.pop(engine.value, None)

    async def _move(
        self,
        store: ManagedStore,
        rollout: EngineRollout,
        runtime: EngineRolloutRuntime,
        new: InstalledBuild,
        when: RuntimeDrainWhen,
    ) -> None:
        """One runtime: drain, measure when it is free to, start on the new
        build, and keep it there only if it loads and answers."""
        name = runtime.name
        spec = self._state().get_runtime_spec(name)
        if spec is None:
            self._runtime(rollout, runtime, EngineRolloutRuntimeState.skipped, "It was removed.")
            return
        old = store.current(runtime=name)
        if old is None or old.version == new.version:
            old = next((b for b in store.list_builds() if b.version == rollout.from_), old)
        supervisor = self._supervisor()
        resumed = runtime.state is not EngineRolloutRuntimeState.waiting
        on_new = store.in_use().runtimes.get(name)
        if resumed and on_new is not None and on_new.version == new.version and not on_new.reason:
            # The agent restarted while it was starting on the new build: it
            # started on it again at boot (or is stopped); check it from here.
            check = self._unmeasured(spec, "the agent restarted during its check")
            await self._start_and_verify(store, rollout, runtime, spec, new, old, check)
            return
        if supervisor is None or not supervisor.is_running(name):
            self._runtime(
                rollout,
                runtime,
                EngineRolloutRuntimeState.skipped,
                f"It stopped before its turn. It starts on {new.version} next time.",
            )
            return
        self._runtime(
            rollout,
            runtime,
            EngineRolloutRuntimeState.draining,
            "Waiting until its other copies can carry the load."
            if when is RuntimeDrainWhen.covered
            else "Waiting until it is idle.",
        )
        measured = sum(1 for m in rollout.check.models if m.measured)
        skip: list[str] = []

        def on_leave(view: RuntimeDrain) -> RuntimeDrainThen:
            """As it leaves routing: measure it only when that costs nobody
            anything; otherwise restart it on the new build at once (DI-C3b),
            which every gateway counts as coming up, so requests wait in line."""
            from .measurement_node import active_measurement

            free = (
                (view.gateways or 0) == 0
                or (view.otherCopies or 0) >= 1
                or (view.idleFor or 0) >= MEASURE_IDLE_SECONDS
            )
            if old is None:
                skip.append("there is no old build to compare with")
            elif measured >= CHECK_MODELS:
                skip.append(f"{CHECK_MODELS} models were measured already")
            elif not free:
                skip.append("it is the only copy and was in use")
            elif kind := active_measurement(self.app):
                skip.append(f"{kind.lower()} was running on this node")
            if skip:
                store.set_runtime(name, new.version)
                return RuntimeDrainThen.restart
            # Measured: stopped, and no launch on this node until it starts
            # again (one measurement per node), so a wake cannot race it.
            self.measuring = name
            return RuntimeDrainThen.stop

        drains = runtime_drain.of(self.app)
        try:
            drains.start(
                name,
                RuntimeDrainRequest(
                    reason=RuntimeDrainReason.update,
                    when=when,
                    idleSeconds=MEASURE_IDLE_SECONDS,
                    graceSeconds=GRACE_SECONDS,
                    then=RuntimeDrainThen.stop,
                ),
                on_leave=on_leave,
            )
        except runtime_drain.DrainError as exc:
            self._runtime(rollout, runtime, EngineRolloutRuntimeState.skipped, str(exc))
            return
        try:
            drained = await drains.wait(name)
        except BaseException:
            self.measuring = None
            raise
        stopped = drained.message == runtime_drain.STOPPED_MEANWHILE
        if drained.state is RuntimeDrainState.cancelled or stopped:
            # Cancelled, or stopped by someone else while it drained (an
            # operator's Stop, an idle unload): it is not measured or started
            # here (review #8). It starts on the build in use next time.
            self.measuring = None
            if store.in_use().runtimes.get(name) == RuntimeBuild(version=new.version):
                store.set_runtime(name, None)
            self._runtime(
                rollout,
                runtime,
                EngineRolloutRuntimeState.skipped,
                "It was stopped during its update, so it was not moved. It starts on the "
                "build in use."
                if stopped
                else "Its drain was cancelled.",
            )
            return
        if skip:
            # Already restarting on the new build (the drain's `then`).
            check = self._unmeasured(spec, skip[0])
            await self._start_and_verify(
                store, rollout, runtime, spec, new, old, check, started=drained.message or ""
            )
            return
        try:
            assert old is not None
            check = await self.measure_model(rollout, runtime, spec, old, new)
        except Exception as exc:
            # The numbers never block (R13-CHECK), and neither does a check
            # that could not run: the model still moves, and says why.
            log.exception("the check of %s on %s failed", name, new.version)
            check = self._unmeasured(spec, f"the check failed: {exc}")
        finally:
            self.measuring = None
        await self._start_and_verify(store, rollout, runtime, spec, new, old, check)

    def _unmeasured(self, spec: RuntimeSpec, why: str) -> EngineCheckModel:
        return EngineCheckModel(
            runtime=spec.name,
            model=_model_name(spec),
            passes=False,
            measured=False,
            reason=f"Not measured: {why}.",
        )

    async def _measure(
        self,
        rollout: EngineRollout,
        runtime: EngineRolloutRuntime,
        spec: RuntimeSpec,
        old: InstalledBuild,
        new: InstalledBuild,
    ) -> EngineCheckModel:
        from .child_env import child_environment
        from .model_copies import resolve_local_path, settings_from_config
        from .routes.runtimes import effective_rules_for
        from .runtime_context import NodeContext

        self._runtime(
            rollout,
            runtime,
            EngineRolloutRuntimeState.measuring,
            f"Measuring {old.version} and {new.version}.",
        )
        local = await asyncio.to_thread(
            resolve_local_path,
            spec.modelPath,
            effective_rules_for(NodeContext(self.app)),
            settings_from_config(self._state().get_config),
        )
        model = Path(local.path)
        if not await asyncio.to_thread(model.is_file):
            return self._unmeasured(spec, "its model file is not on this machine")
        env = child_environment()
        env.update({k: str(v) for k, v in (spec.env or {}).items()})
        measured, notes = await self.check.measure(
            model=model,
            name=_model_name(spec),
            flags=dict(spec.flags or {}),
            env=env,
            old=old,
            new=new,
        )
        measured.runtime = spec.name
        words = check_words(new.version, measured.model, measured)
        measured.reason = (
            " ".join(part for part in (words, *(n[0].upper() + n[1:] + "." for n in notes)) if part)
            or None
        )
        return measured

    async def _start_and_verify(
        self,
        store: ManagedStore,
        rollout: EngineRollout,
        runtime: EngineRolloutRuntime,
        spec: RuntimeSpec,
        new: InstalledBuild,
        old: InstalledBuild | None,
        check: EngineCheckModel,
        *,
        started: str | None = None,
    ) -> None:
        """Start it on the new build (unless its drain's `then` restarted it
        there already: `started` is that drain's message) and keep it there
        only if it loads and answers."""
        name = spec.name
        model = _model_name(spec)
        self._runtime(
            rollout, runtime, EngineRolloutRuntimeState.starting, f"Starting on {new.version}."
        )
        store.set_runtime(name, new.version)
        failure = await self._start(spec, new.version, launch=started is None, why=started)
        if failure is None:
            check.passes = True
            self._record(rollout, check)
            self._runtime(rollout, runtime, EngineRolloutRuntimeState.moved, f"On {new.version}.")
            return
        # R13-CHECK-2: not used for this model. It goes back to its old build.
        reason = f"{new.version} could not run {model}: {failure}"
        check.passes = False
        check.reason = " ".join(p for p in (reason + ".", check.reason or "") if p)
        self._record(rollout, check)
        back = old.version if old is not None else rollout.from_
        store.set_runtime(name, back, reason)
        supervisor = self._supervisor()
        if supervisor is not None:
            await supervisor.stop_one(name, reason=StopReason.measurement)
        again = await self._start(spec, back) if back else "there is no old build"
        message = f"{reason}. Kept {back}." + (
            f" It did not start again on {back}: {again}." if again else ""
        )
        self._runtime(rollout, runtime, EngineRolloutRuntimeState.held, message)

    def _record(self, rollout: EngineRollout, check: EngineCheckModel) -> None:
        rollout.check.models = [m for m in rollout.check.models if m.runtime != check.runtime]
        rollout.check.models.append(check)
        self._save()

    async def _launch_runtime(self, name: str) -> str | None:
        """Start a stopped runtime as the Start button does (the launch lock
        and admission). None, or why it did not start."""
        from .measurement_node import restart_stopped
        from .runtime_context import NodeContext

        started = (await restart_stopped(NodeContext(self.app), [name], enabled=True))[0]
        if started.state.value != "restarted":
            return started.detail or "it did not start"
        return None

    async def _start(
        self,
        spec: RuntimeSpec,
        version: str | None,
        *,
        launch: bool = True,
        why: str | None = None,
    ) -> str | None:
        """Start `spec` (it starts from its own entry in the store) and wait
        until it is ready on `version` and answers. None, or what failed.
        `launch=False`: a drain's restart started it already; `why` is that
        drain's message, which names a restart that failed."""
        supervisor = self._supervisor()
        if supervisor is None:
            return "this agent is not supervising runtimes"
        if launch:
            failure = await self.launch(spec.name)
            if failure is not None:
                return failure
        deadline = time.perf_counter() + self.ready_seconds
        while True:
            view = self._view(spec)
            status = view.status if view is not None else None
            if view is not None and status is RuntimeStatus.ready:
                if version is not None and view.engineVersion and view.engineVersion != version:
                    return f"it started on {view.engineVersion}, not {version}"
                return await self.probe(view)
            if status in (RuntimeStatus.crashed, RuntimeStatus.exited, RuntimeStatus.stopped):
                marker = " Then restart failed: "
                cause = (view.lastError if view is not None else None) or (
                    why.split(marker, 1)[1] if why and marker in why else f"it is {status.value}"
                )
                return _tail(cause, 400)
            if time.perf_counter() >= deadline:
                return f"it was not ready after {self.ready_seconds:.0f} s"
            await asyncio.sleep(self.poll_seconds)

    async def _probe(self, view: Runtime) -> str | None:
        """Whether a ready engine answers: four tokens from `/completion`. A
        search model (embeddings, reranking) is ready when it is loaded."""
        capabilities = view.capabilities
        if capabilities is not None and (capabilities.embeddings or capabilities.reranking):
            return None
        if view.url is None:
            return None
        client = shared_internal_client("engine-check", timeout=60.0)
        try:
            answer = await client.post(
                f"{str(view.url).rstrip('/')}/completion",
                json={"prompt": "Hello", "n_predict": 4, "temperature": 0},
            )
        except Exception as exc:
            return f"it did not answer: {exc}"
        if answer.status_code >= 400:
            return f"it answered with an error (HTTP {answer.status_code}): {answer.text[:200]}"
        return None

    def _runtime(
        self,
        rollout: EngineRollout,
        runtime: EngineRolloutRuntime,
        state: EngineRolloutRuntimeState,
        message: str,
    ) -> None:
        runtime.state = state
        runtime.message = message
        rollout.state = (
            EngineRolloutState.checking
            if state in (EngineRolloutRuntimeState.draining, EngineRolloutRuntimeState.measuring)
            else EngineRolloutState.switching
        )
        rollout.message = self._progress(rollout)
        self._save()

    def _finish(self, store: ManagedStore, rollout: EngineRollout) -> None:
        moved = [r for r in rollout.runtimes if r.state is EngineRolloutRuntimeState.moved]
        held = [r for r in rollout.runtimes if r.state is EngineRolloutRuntimeState.held]
        rollout.check.passes = bool(moved)
        old = rollout.from_ or "the build in use"
        if moved or not held:
            store.set_in_use(rollout.version)
            message = f"{rollout.version} is in use."
            if held:
                names = ", ".join(r.name for r in held)
                message += f" Kept {old} for {names}: {rollout.version} could not run it."
            self._end(rollout, EngineRolloutState.done, message)
            return
        # It could run none of the checked models: the old build stays in use,
        # and the runtimes need no entries to stay on it.
        for runtime in held:
            entry = store.in_use().runtimes.get(runtime.name)
            if entry is not None and entry.version == store.in_use().version:
                store.set_runtime(runtime.name, None)
        first = held[0].message or ""
        self._end(
            rollout,
            EngineRolloutState.kept,
            f"{rollout.version} could not run any model here; kept {old}. {first}".strip(),
        )

    async def _undo(
        self, engine: EngineKind, rollout: EngineRollout, runtime: EngineRolloutRuntime | None
    ) -> None:
        """A cancelled rollout: the runtime being moved goes back to the build
        in use; each runtime already moved stays where it is."""
        try:
            if runtime is not None and runtime.state not in FINISHED_RUNTIME:
                drains = runtime_drain.of(self.app)
                with contextlib.suppress(runtime_drain.DrainError):
                    await drains.cancel(runtime.name)
                self.measuring = None
                store = self.store_for(engine)
                entry = store.in_use().runtimes.get(runtime.name)
                spec = self._state().get_runtime_spec(runtime.name)
                if entry is not None and entry.version == rollout.version and not entry.reason:
                    store.set_runtime(runtime.name, None)
                    supervisor = self._supervisor()
                    if supervisor is not None and spec is not None:
                        await supervisor.stop_one(runtime.name, reason=StopReason.measurement)
                supervisor = self._supervisor()
                if spec is not None and not (
                    supervisor is not None and supervisor.is_running(spec.name)
                ):
                    await self._start(spec, None)
                runtime.state = EngineRolloutRuntimeState.skipped
                runtime.message = "Cancelled; it stays on the build in use."
        finally:
            with contextlib.suppress(Exception):
                self._forget_moves(self.store_for(engine), rollout)
            self._end(
                rollout,
                EngineRolloutState.cancelled,
                f"Cancelled. {self._progress(rollout)}; they start on "
                f"{rollout.from_ or 'the build in use'} when they next restart.",
            )


def _store_for(engine: EngineKind) -> ManagedStore:
    from .runtimes import adapter_for

    adapter = adapter_for(engine)
    if adapter is None:
        raise RolloutError(f"{engine.value} is not an engine this agent runs.", 422)
    return adapter.managed_store()


def manager(app: Any) -> EngineRollouts:
    found = getattr(app.state, "engine_rollouts", None)
    if isinstance(found, EngineRollouts):
        return found
    found = EngineRollouts(app, app.state.settings.config_file.resolve().parent / FILE_NAME)
    app.state.engine_rollouts = found
    return found


__all__ = [
    "EngineCheck",
    "EngineRollouts",
    "RolloutError",
    "check_words",
    "manager",
    "read_bench",
    "run_tool",
]
