"""Supervision of engine runtimes.

A *runtime* is one third-party engine process — `llama-server` or
`vllm serve` — launched from an argv an `EngineAdapter` builds. It reuses the whole
supervision loop in `supervisor.py` and adds the two things a foreign
binary needs that a Eugene Plexus component does not: an engine-specific
readiness probe, and a status that distinguishes "still loading the
model" from "not answering at all".

Nothing here knows how to *talk* to an engine. That is the driver's job;
see the agent spec's note on where engine knowledge lives.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from . import model_copies
from ._generated.models import (
    CopyProgress,
    EngineAcquisition,
    EngineDescriptor,
    EngineInstall,
    EngineKind,
    HostAccelerator,
    LoadProgress,
    ManagedEngine,
    ManualInstall,
    Policy,
    Runtime,
    RuntimeCapabilities,
    RuntimeLocalPathSource,
    RuntimeSpec,
    RuntimeStatus,
    StopReason,
)
from .companions import companion_name
from .engines import (
    ADAPTERS,
    EngineAdapter,
    EngineUnavailableError,
    Loading,
    Ready,
    adapter_for,
    default_model_alias,
    interpret_readiness,
)
from .engines.acquisition import (
    AcquisitionError,
    AcquisitionPlan,
    EngineInstaller,
    Release,
    Unavailable,
)
from .engines.base import DiscoveredBinary
from .engines.host import detect_host
from .engines.llama_cpp import LlamaCppAdapter
from .library_folders import RulesProvider, effective_rules
from .model_copies import resolve_local_path
from .model_paths import PathRule, rules_from_config
from .process_io import LoadProgressTracker
from .supervisor import (
    ProcessState,
    SpawnPlan,
    SpawnPlanError,
    SupervisedProcess,
)

log = logging.getLogger(__name__)

# Readiness polling cadence. Slower than the component health loop (1.5s)
# on purpose: a model load takes tens of seconds to minutes, so there is
# nothing to learn from probing it four times a second, and `/props` is a
# heavier answer than `/healthz`.
_READINESS_POLL_SECONDS = 2.0

# `key -> value` over the agent's own config. What an adapter's
# `configured_binary_key` is read through: config lives in `AgentState`,
# adapters are stateless singletons, and the two meet here.
ConfigGetter = Callable[[str], Any]


def _configured_binary(adapter: EngineAdapter, get_config: ConfigGetter | None) -> str | None:
    """The install-wide binary path for this engine, if one is configured."""
    if get_config is None or adapter.configured_binary_key is None:
        return None
    value = get_config(adapter.configured_binary_key)
    return str(value) if isinstance(value, str) and value.strip() else None


class _RuntimePlanner:
    """Launch plans for one engine runtime.

    Contrast with `_ComponentPlanner`: no config-file path, no bind-port
    env var, no auth trio (an engine has no notion of our tokens and is
    bound to loopback precisely because it can't authenticate), and no
    recovery — see `on_crash_threshold`.
    """

    def __init__(
        self,
        spec: RuntimeSpec,
        adapter: EngineAdapter,
        log: logging.Logger,
        get_config: ConfigGetter | None = None,
        inherited_rules: RulesProvider | None = None,
    ) -> None:
        self.spec = spec
        self._adapter = adapter
        self._log = log
        self._get_config = get_config
        # The Library folders' mounts for this host, read live like the
        # config (2026-09-14). This node's `pathMappings` are the
        # overrides and come first.
        self._inherited_rules = inherited_rules
        self.binary: DiscoveredBinary | None = None
        """Whatever the last plan resolved. Read for `Runtime.engineVersion`
        so the operator sees the build that is actually running rather than
        whatever is on PATH now."""

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def log_prefix(self) -> str:
        return f"[engine: {self.spec.name}] "

    def plan(self) -> SpawnPlan:
        try:
            binary = self._adapter.resolve_binary(
                self.spec,
                configured=_configured_binary(self._adapter, self._get_config),
            )
        except EngineUnavailableError as e:
            # A declared runtime whose engine is missing is a crash, not a
            # skipped entry: the operator asked for something that cannot
            # be delivered and needs to hear about it.
            raise SpawnPlanError(str(e)) from e

        self.binary = binary
        port = self.spec.port
        if port is None:
            # Should not happen — the store assigns one at write time —
            # but planning is where it would bite, so say so clearly.
            raise SpawnPlanError(
                f"runtime {self.spec.name!r} has no port assigned; "
                f"re-save it so the agent can allocate one"
            )

        # The engine opens the model where THIS host has it (M11). The
        # declaration keeps the library's spelling; the copy handed to
        # the adapter carries the resolved local path, and that is the
        # only place the two differ.
        launch_spec = self._launch_spec()
        argv = self._adapter.build_argv(launch_spec, binary, port)

        # Engines inherit the ambient environment, then the adapter's own
        # defaults for values the engine cannot start without on this
        # host, then the runtime's additions. Accelerator selection
        # (CUDA_VISIBLE_DEVICES and friends) rides in the last of those,
        # which is how a runtime gets pinned to one card and how two
        # replicas end up on two GPUs.
        #
        # The precedence is the point: `setdefault` means an operator who
        # exported the variable in the agent's own shell keeps their
        # value, and `spec.env` last means the runtime's setting beats
        # ours outright — including setting it to something that will
        # fail, which is an expert's prerogative. Anything we actually
        # injected is logged, because an environment variable nobody
        # typed makes a later bug report unreadable.
        env = os.environ.copy()
        injected = {}
        for key, value in self._adapter.default_env(launch_spec, binary).items():
            if key not in env:
                env[key] = value
                injected[key] = value
        if injected:
            log.info(
                "%s: %s set by the %s adapter as this host needs it; "
                "override in the runtime's `env`",
                self.spec.name,
                ", ".join(f"{k}={v}" for k, v in sorted(injected.items())),
                self.spec.engine.value,
            )
        if self.spec.env:
            env.update({k: str(v) for k, v in self.spec.env.items()})

        return SpawnPlan(
            argv=argv,
            env=env,
            cwd=self._adapter.working_directory(launch_spec, binary),
        )

    def _rules(self) -> list[PathRule]:
        return effective_rules(
            rules_from_config(self._get_config),
            self._inherited_rules() if self._inherited_rules is not None else (),
        )

    def _launch_spec(self) -> RuntimeSpec:
        """The declaration as the engine should see it.

        `modelPath` resolved through **this node's own copy when it has a
        current one**, else the Library folder's mount for this host and
        this node's `pathMappings` overrides -- read live, so a rule
        added after the declaration applies at the next start with
        nothing re-declared. Everything else is the declaration verbatim.
        Logged whenever the engine is handed a path nobody typed,
        because an argv naming an unfamiliar file makes a later bug
        report unreadable.

        **The copy is never made here.** This runs synchronously while
        the supervisor builds argv, and a 25 GB transfer would stall it
        for four minutes. Copying belongs to the runtime's own task, in
        `RuntimeSupervisor`, which finishes before this is ever called.
        """
        resolved = resolve_local_path(self.spec.modelPath, self._rules(), self._copy_settings())
        if resolved.path == self.spec.modelPath:
            return self.spec
        self._log.info(
            "%s: opening %s as %s (%s)",
            self.spec.name,
            self.spec.modelPath,
            resolved.path,
            resolved.source,
        )
        return self.spec.model_copy(update={"modelPath": resolved.path})

    def _copy_settings(self) -> model_copies.CopySettings:
        if self._get_config is None:
            return model_copies.CopySettings(enabled=False, directory=None, min_free_bytes=0)
        return model_copies.settings_from_config(self._get_config)

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        """Let the engine's own adapter read the wreckage.

        The adapter is the only thing here that knows what a given
        engine's startup failures look like, so the planner does nothing
        but hand the tail over.
        """
        return self._adapter.explain_exit(return_code, output_tail)

    def on_crash_threshold(self) -> bool:
        """Engines get no second chance, unlike components.

        A component can fall back to safe mode because its value in that
        state is still real — `/v1/config` stays reachable and the
        operator repairs it from the UI. An engine has no such mode: a
        `llama-server` that won't start has nothing to serve and nothing
        to configure. Respawning it forever would just churn.
        """
        self._log.error(
            "engine runtime %s failed to start repeatedly; giving up. Check the "
            "captured engine output above — the resolved argv is on "
            "GET /v1/runtimes/%s. POST /v1/runtimes/%s/restart to retry.",
            self.spec.name,
            self.spec.name,
            self.spec.name,
        )
        return False

    def reset(self) -> None:
        """Nothing latched, so nothing to clear."""


class _CopyJob:
    """One local copy in flight, and the task doing it."""

    def __init__(self, plan: model_copies.CopyPlan, total_bytes: int | None) -> None:
        self.plan = plan
        self.state = model_copies.CopyState(destination=plan.destination, total_bytes=total_bytes)
        self.cancelled = False
        self.task: asyncio.Task[None] | None = None

    def cancel(self) -> None:
        """Ask the worker thread to stop at its next chunk.

        Cancelling the task alone would not do it: the copy runs in a
        thread, and `asyncio.to_thread` cannot interrupt one. The flag is
        what the copy loop reads, and it checks it every 4 MB.
        """
        self.cancelled = True
        if self.task is not None:
            self.task.cancel()


class RuntimeSupervisor:
    """Owns every engine process plus a background readiness-poll task."""

    def __init__(
        self,
        log: logging.Logger | None = None,
        get_config: ConfigGetter | None = None,
        inherited_rules: RulesProvider | None = None,
    ) -> None:
        self._log = log or logging.getLogger(__name__)
        # Read live at plan time, so an operator who sets `vllmBinary` in
        # the UI gets the new path on the next spawn without restarting
        # the agent.
        self._get_config = get_config
        # The Library folders' mounts for this host (2026-09-14), also
        # read live -- this node's copy of the library's folder list,
        # refreshed by every request that talks to the library.
        self._inherited_rules = inherited_rules
        self._processes: dict[str, SupervisedProcess] = {}
        self._planners: dict[str, _RuntimePlanner] = {}
        # Latest readiness per runtime. Held here rather than on the
        # process because it is an *observation*, not loop state — the
        # same reason component safe-mode observations live on Supervisor.
        self._readiness: dict[str, Loading | Ready | None] = {}
        # Why a stopped runtime is stopped — an observation, reported on
        # `Runtime.stopReason` and never persisted or replicated. Cleared
        # the moment the runtime is started.
        self._stop_reasons: dict[str, StopReason] = {}
        # Bytes-read samples per runtime; see `process_io`. Sampled
        # when a view is built rather than on a loop, because a
        # loading runtime is already read every second or two by the
        # control root and every three by a console.
        self._load_progress = LoadProgressTracker()
        # A copy in flight, per runtime, and why the last one did not
        # happen. Both are observations: `_copy_jobs` empties itself when
        # the copy ends, and `_copy_notes` holds the sentence the
        # Inference screen prints under a runtime that is opening the
        # share when the operator asked for a local copy. A skipped copy
        # is the failure mode of this whole feature -- the launch
        # succeeds, the model serves, and the only symptom is four
        # minutes nobody can account for.
        self._copy_jobs: dict[str, _CopyJob] = {}
        self._copy_notes: dict[str, str] = {}
        # The declared model paths this loop last reconciled against.
        # None until the first pass, so a fresh agent reconciles once.
        self._last_declared: set[str] | None = None
        self._poll_task: asyncio.Task[None] | None = None
        # This node's name once enrolled, read at compose time so it is
        # filled from the agent's own identity and can never disagree
        # with it — the contract's reason `Runtime.node` is reported
        # rather than declared. None until enrollment, which is the only
        # state in which "which node" has no answer.
        self.node_name_provider: Callable[[], str | None] | None = None

    # --- collection management --------------------------------------------

    def add_and_start(self, spec: RuntimeSpec) -> None:
        """Begin supervising a runtime. Declared-but-not-started when
        `autoStart` is false — that is how a rarely-used large model stays
        configured without holding VRAM."""
        if spec.name in self._processes:
            return
        if spec.autoStart is False:
            self._planners.pop(spec.name, None)
            self._stop_reasons.setdefault(spec.name, StopReason.autoStart)
            return
        adapter = adapter_for(spec.engine)
        if adapter is None:
            self._log.error(
                "runtime %s names engine %r, which this agent has no adapter for; not starting it.",
                spec.name,
                spec.engine.value,
            )
            return
        job = self._copy_job_for(spec)
        if job is not None:
            # **Copy first, then launch** (design §4.1). The arithmetic
            # says so: a plain copy moves bytes faster than an engine
            # reading them, so copy-then-load is already marginally
            # faster than loading off the share on the very first start,
            # and it halves the wire traffic against copying in the
            # background. The runtime sits in `copying` until it is done
            # -- no process exists yet, which is exactly why that status
            # is not `starting`.
            self._copy_jobs[spec.name] = job
            self._stop_reasons.pop(spec.name, None)
            job.task = asyncio.create_task(
                self._copy_then_spawn(spec, adapter, job), name=f"copy:{spec.name}"
            )
            return
        self._spawn(spec, adapter)

    def _spawn(self, spec: RuntimeSpec, adapter: EngineAdapter) -> None:
        """Start supervising for real. Reached directly when there is no
        copy to make, and after the copy when there was one."""
        planner = _RuntimePlanner(
            spec, adapter, self._log, self._get_config, inherited_rules=self._inherited_rules
        )
        sp = SupervisedProcess(planner, self._log)
        self._planners[spec.name] = planner
        self._processes[spec.name] = sp
        self._stop_reasons.pop(spec.name, None)
        sp.start()

    # --- the local copy ---------------------------------------------------

    def copy_settings(self) -> model_copies.CopySettings:
        """This node's copy trio, read live so a toggle takes effect at
        the next start with no restart."""
        if self._get_config is None:
            return model_copies.CopySettings(enabled=False, directory=None, min_free_bytes=0)
        return model_copies.settings_from_config(self._get_config)

    def _copy_job_for(self, spec: RuntimeSpec) -> _CopyJob | None:
        """A copy to make before this runtime starts, or None.

        None is the common answer and covers every reason not to copy:
        the toggle is off, the copy is already current, the source
        cannot be read, or making it would eat into the headroom the
        operator asked to keep. Each of the last two leaves a note, so
        the screen can say why this start is reading over the network.
        """
        settings = self.copy_settings()
        plan = model_copies.plan_for(spec.modelPath, self._rules(), settings)
        if plan is None:
            self._copy_notes.pop(spec.name, None)
            return None
        if model_copies.copy_is_current(plan):
            self._copy_notes.pop(spec.name, None)
            return None
        try:
            size = os.path.getsize(plan.source)
        except OSError as exc:
            # Not an error here: the model may be on a share that is
            # down, in which case the launch is about to fail for a
            # reason of its own and saying "could not copy" first would
            # bury it.
            self._copy_notes[spec.name] = f"could not read {plan.source} to copy it: {exc}"
            return None

        shortfall = model_copies.headroom_shortfall(plan, settings, size)
        if shortfall > 0:
            # Try to give the space back from copies nothing is using
            # before refusing -- this is the one case eviction exists
            # for. Never evicts to make room for a copy of something
            # else: only copies no runtime here points at any more.
            evicted = model_copies.evict_for_headroom(
                settings.directory, settings, in_use=self._copies_in_use(), keep=[plan.destination]
            )
            if evicted.deleted:
                self._log.info(
                    "removed %d local copies to keep %d GB free",
                    len(evicted.deleted),
                    settings.min_free_bytes // model_copies.GIB,
                )
            shortfall = model_copies.headroom_shortfall(plan, settings, size)
        if shortfall > 0:
            note = (
                f"not copied to this machine: {shortfall / model_copies.GIB:.1f} GB more free "
                f"space is needed to keep {settings.min_free_bytes // model_copies.GIB} GB free. "
                f"Reading it from {plan.source} instead."
            )
            self._copy_notes[spec.name] = note
            self._log.warning("%s: %s", spec.name, note)
            return None
        return _CopyJob(plan, total_bytes=size)

    def _copies_in_use(self) -> dict[str, str]:
        """Destination -> the runtime holding it open. Nothing here ever
        stops a runtime to get at its file, so this is what protects a
        copy from eviction and from Clear."""
        in_use: dict[str, str] = {}
        settings = self.copy_settings()
        if not settings.usable:
            return in_use
        rules = self._rules()
        for name in self._processes:
            planner = self._planners.get(name)
            if planner is None:
                continue
            plan = model_copies.plan_for(planner.spec.modelPath, rules, settings)
            if plan is not None:
                in_use[plan.destination] = name
        # A copy being written counts as in use, and it is not the
        # partial that needs protecting -- that name is skipped
        # everywhere -- but the destination it is about to become. Clear
        # would otherwise delete a file seconds before the rename put it
        # back, and report it as freed space that never came back.
        for name, job in self._copy_jobs.items():
            in_use[job.plan.destination] = name
        return in_use

    async def _copy_then_spawn(
        self, spec: RuntimeSpec, adapter: EngineAdapter, job: _CopyJob
    ) -> None:
        """Make the copy, then start the engine -- **whatever happens**.

        A failed copy is not a failed launch. Every path here ends in a
        spawn that opens whatever `resolve_local_path` then answers,
        which is the share when the copy did not land. The alternative
        would be a node that stops serving because a convenience feature
        could not write a file.
        """
        settings = self.copy_settings()
        note: str | None = None
        try:
            await asyncio.to_thread(
                model_copies.copy_file,
                job.plan,
                settings,
                job.state,
                should_cancel=lambda: job.cancelled,
            )
        except asyncio.CancelledError:
            # The runtime was removed or stopped mid-copy. Nothing to
            # spawn, and `copy_file` has already removed its partial.
            self._copy_jobs.pop(spec.name, None)
            raise
        except model_copies.CopyAborted as exc:
            note = f"{exc}. Reading it from {job.plan.source} instead."
            self._log.warning("%s: %s", spec.name, note)
        except OSError as exc:
            note = f"could not copy the model to this machine: {exc}. Reading it over the network."
            self._log.warning("%s: %s", spec.name, note)
        finally:
            self._copy_jobs.pop(spec.name, None)

        if note is None:
            self._copy_notes.pop(spec.name, None)
        else:
            self._copy_notes[spec.name] = note
        if job.cancelled:
            return
        self._spawn(spec, adapter)

    def reconcile_copies(self, specs: list[RuntimeSpec]) -> None:
        """Drop copies of models no runtime on this node points at.

        The whole eviction policy for the ordinary case, and the reason
        there is no LRU here: the set was never anything but a function
        of the declaration list, so a runtime deleted or repointed takes
        its copy with it.
        """
        settings = self.copy_settings()
        if not settings.usable:
            return
        keep = [
            plan.destination
            for plan in model_copies.wanted(
                [s.modelPath for s in specs], self._rules(), settings
            ).values()
        ]
        model_copies.remove_unwanted(settings.directory, keep, in_use=self._copies_in_use())

    def clear_copies(self) -> model_copies.ClearResult:
        """Delete every copy this node holds. Stops nothing."""
        settings = self.copy_settings()
        return model_copies.clear(settings.directory, in_use=self._copies_in_use())

    async def remove_and_stop(self, name: str) -> None:
        await self._cancel_copy(name)
        sp = self._processes.pop(name, None)
        self._planners.pop(name, None)
        self._readiness.pop(name, None)
        self._stop_reasons.pop(name, None)
        self._copy_notes.pop(name, None)
        self._load_progress.forget(name)
        if sp is not None:
            await sp.stop()

    async def restart(self, name: str) -> bool:
        sp = self._processes.get(name)
        if sp is None:
            return False
        planner = self._planners.get(name)
        if planner is not None and self._copy_wanted(planner.spec):
            # **A restart is a launch, so it copies first.** Without
            # this, the most natural gesture after switching copying on
            # -- press Restart -- re-plans in place, opens the share
            # again and explains nothing, because `sp.restart()` never
            # passes through the path that makes a copy. Found on the
            # live install at step 8 of the design's build order, which
            # is the first thing that ever pressed it.
            await self.stop_one(name, reason=StopReason.operator)
            self.add_and_start(planner.spec.model_copy(update={"autoStart": True}))
            return True
        self._readiness.pop(name, None)
        await sp.restart()
        return True

    def _copy_wanted(self, spec: RuntimeSpec) -> bool:
        """Whether this runtime would copy its model if started now."""
        plan = model_copies.plan_for(spec.modelPath, self._rules(), self.copy_settings())
        return plan is not None and not model_copies.copy_is_current(plan)

    async def stop_one(self, name: str, *, reason: StopReason = StopReason.operator) -> None:
        """Stop the engine but keep the runtime declared.

        Distinct from `remove_and_stop` because an engine holds GPU
        memory: an operator who wants the VRAM back needs a stop that is
        neither a delete nor a crash. `reason` is recorded so the
        dashboard can say *why* — `idle` when the gateway unloaded it.
        """
        await self._cancel_copy(name)
        sp = self._processes.pop(name, None)
        self._planners.pop(name, None)
        self._readiness.pop(name, None)
        self._stop_reasons[name] = reason
        if sp is not None:
            await sp.stop()

    async def stop_all(self) -> None:
        """Shut everything down. Best-effort — never raises."""
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._poll_task
            self._poll_task = None
        for name in list(self._copy_jobs):
            await self._cancel_copy(name)
        await asyncio.gather(
            *(sp.stop() for sp in self._processes.values()),
            return_exceptions=True,
        )
        self._processes.clear()
        self._planners.clear()
        self._readiness.clear()
        self._stop_reasons.clear()

    async def start_readiness_loop(self, get_specs: object) -> None:
        """Start the background readiness poll. `get_specs` is a 0-arg
        callable returning the current RuntimeSpec list, so the loop sees
        topology changes without being restarted."""
        if self._poll_task is not None:
            return
        self._poll_task = asyncio.create_task(
            self._readiness_loop(get_specs), name="runtime-readiness"
        )

    def is_running(self, name: str) -> bool:
        """Running, or on its way there.

        A runtime whose copy is still being made counts: nothing has
        been spawned yet, but a second Start would begin a second copy
        of the same 25 GB file into the same destination.
        """
        return name in self._processes or name in self._copy_jobs

    async def _cancel_copy(self, name: str) -> None:
        """Stop a copy in flight and wait for its thread to notice."""
        job = self._copy_jobs.pop(name, None)
        if job is None:
            return
        job.cancel()
        if job.task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await job.task
        self._log.info("%s: local copy cancelled", name)

    # --- read model -------------------------------------------------------

    def _rules(self) -> list[PathRule]:
        return effective_rules(
            rules_from_config(self._get_config),
            self._inherited_rules() if self._inherited_rules is not None else (),
        )

    def compose(self, spec: RuntimeSpec) -> Runtime:
        """Pair a declaration with what is observed of it."""
        sp = self._processes.get(name := spec.name)
        planner = self._planners.get(name)
        readiness = self._readiness.get(name)

        job = self._copy_jobs.get(name)
        status = self._status_for(sp, readiness, copying=job is not None)
        capabilities: RuntimeCapabilities | None = None
        engine_version: str | None = None
        if isinstance(readiness, Ready):
            capabilities = readiness.capabilities
        if planner is not None and planner.binary is not None:
            engine_version = planner.binary.version

        url = None
        if spec.port is not None:
            url = f"http://{spec.host or '127.0.0.1'}:{spec.port}"

        last_restart: datetime | None = sp.last_restart if sp is not None else None

        # A silent load that has outrun its adapter's startup budget is
        # reported on `lastError` — the contract names "a readiness probe
        # that never passed" as one of its three sources — while `status`
        # stays `loading`, because that is still the only thing a live,
        # silent process can be. Cleared the moment the engine answers.
        last_error = sp.last_error if sp is not None else None
        if last_error is None and isinstance(readiness, Loading) and readiness.past_budget:
            last_error = readiness.detail

        # Why a stopped runtime is stopped. Recorded by `stop_one`; a
        # runtime that was never started in this agent's lifetime was
        # declared `autoStart: false`, which is the only other way to be
        # `stopped` without a recorded reason.
        stop_reason: StopReason | None = None
        if status is RuntimeStatus.stopped:
            stop_reason = self._stop_reasons.get(
                name,
                StopReason.autoStart if spec.autoStart is False else StopReason.operator,
            )

        # **Only while it is loading.** A ready engine's counter keeps
        # climbing as it serves, and a bar that filled once and then
        # crept past 100% would be worse than no bar. `starting` counts:
        # llama.cpp opens its socket before it has read anything, so the
        # first seconds of a read are spent in that status.
        resolved = resolve_local_path(spec.modelPath, self._rules(), self.copy_settings())
        local_path = resolved.path
        load_progress: LoadProgress | None = None
        if status in (RuntimeStatus.loading, RuntimeStatus.starting):
            observed = self._load_progress.sample(
                spec.name, sp.pid if sp is not None else None, local_path
            )
            if observed is not None:
                load_progress = LoadProgress(
                    bytesRead=observed.bytes_read,
                    totalBytes=observed.total_bytes,
                    bytesPerSecond=observed.bytes_per_second,
                    source=observed.source,  # type: ignore[arg-type]
                )
        else:
            self._load_progress.forget(spec.name)

        return Runtime(
            name=spec.name,
            engine=spec.engine,
            modelPath=spec.modelPath,
            # Observed, from the current rules, every time it is read:
            # a stopped runtime shows what its next start would open.
            localPath=local_path,
            localPathSource=RuntimeLocalPathSource(resolved.source),
            # Only when a copy was asked for and is not being used --
            # otherwise there is nothing to explain, and a note on every
            # runtime would train the operator to ignore the one that
            # matters.
            localPathNote=None if resolved.is_copy else self._copy_notes.get(name),
            copyProgress=(
                CopyProgress(
                    bytesCopied=job.state.bytes_copied,
                    totalBytes=job.state.total_bytes,
                    bytesPerSecond=job.state.bytes_per_second,
                    destination=job.plan.destination,
                )
                if job is not None
                else None
            ),
            modelAlias=spec.modelAlias or _default_alias(spec),
            host=spec.host,
            port=spec.port,
            autoStart=spec.autoStart,
            autoDriver=spec.autoDriver,
            idleUnloadSeconds=spec.idleUnloadSeconds,
            startOnDemand=spec.startOnDemand,
            flags=spec.flags,
            extraArgs=spec.extraArgs,
            env=spec.env,
            workingDirectory=spec.workingDirectory,
            binary=spec.binary,
            # Derived from the declaration: `autoDriver` means the agent
            # keeps a companion under this name (reconciled at boot).
            driver=companion_name(spec.name) if spec.autoDriver is not False else None,
            node=self.node_name_provider() if self.node_name_provider is not None else None,
            status=status,
            stopReason=stop_reason,
            url=url,  # type: ignore[arg-type]
            argv=sp.last_argv if sp is not None else None,
            pid=sp.pid if sp is not None else None,
            engineVersion=engine_version,
            capabilities=capabilities,
            lastRestart=last_restart,
            lastError=last_error,
            loadProgress=load_progress,
        )

    def _status_for(
        self,
        sp: SupervisedProcess | None,
        readiness: Loading | Ready | None,
        *,
        copying: bool = False,
    ) -> RuntimeStatus:
        """Map loop state + readiness onto the wire enum.

        The interesting half is `starting` vs `loading` vs `ready`: the
        loop cannot tell those apart — as far as it knows the child is
        simply alive — so the readiness observation supplies the
        distinction. This is the same shape as the component mapping,
        which pairs loop state with a /healthz observation.
        """
        if copying:
            # Before `sp` exists at all: this node is making its local
            # copy of the model file. Reported ahead of everything else
            # because `sp is None` would otherwise read as `stopped`,
            # which says the operator asked for this -- and they asked
            # for the opposite.
            return RuntimeStatus.copying
        if sp is None:
            # Declared but not running: autoStart false, or explicitly
            # stopped. Not an error.
            return RuntimeStatus.stopped
        if sp.state == ProcessState.crashed:
            return RuntimeStatus.crashed
        if sp.state == ProcessState.exited:
            return RuntimeStatus.exited
        if sp.state == ProcessState.not_spawnable:
            # Unreachable for engines — `_RuntimePlanner.plan` never
            # returns None — but mapping it to `stopped` would claim the
            # operator asked for this.
            return RuntimeStatus.crashed
        if isinstance(readiness, Ready):
            return RuntimeStatus.ready
        if isinstance(readiness, Loading):
            return RuntimeStatus.loading
        return RuntimeStatus.starting

    # --- internals --------------------------------------------------------

    async def _readiness_loop(self, get_specs: object) -> None:
        try:
            while True:
                specs: list[RuntimeSpec] = get_specs()  # type: ignore[operator]
                await self._reconcile_copies_if_changed(specs)
                await asyncio.gather(
                    *(self._probe_one(s) for s in specs),
                    return_exceptions=True,
                )
                await asyncio.sleep(_READINESS_POLL_SECONDS)
        except asyncio.CancelledError:
            return

    async def _reconcile_copies_if_changed(self, specs: list[RuntimeSpec]) -> None:
        """Drop copies nothing points at, when the declarations change.

        Hung off this loop because it is the one place that already sees
        the current declaration list. Gated on the set actually changing
        so the common case costs a set comparison rather than a walk of
        the copy directory every two seconds -- which on a spinning disk
        holding 200 GB of models is not free.
        """
        declared = {s.modelPath for s in specs}
        if declared == self._last_declared:
            return
        self._last_declared = declared
        with contextlib.suppress(OSError):
            await asyncio.to_thread(self.reconcile_copies, specs)

    async def _probe_one(self, spec: RuntimeSpec) -> None:
        sp = self._processes.get(spec.name)
        if sp is None or spec.port is None:
            return
        adapter = adapter_for(spec.engine)
        if adapter is None:
            return
        base = f"http://{spec.host or '127.0.0.1'}:{spec.port}"
        outcome = await adapter.probe_readiness(base)
        # The probe saw the network. This is where what the supervisor
        # knows — the pid is alive, and for how long — is added: for an
        # engine that answers nothing while it loads, alive-and-refusing
        # IS loading, and only this side of the split can say so.
        elapsed: float | None = None
        if sp.last_restart is not None:
            elapsed = (datetime.now(UTC) - sp.last_restart).total_seconds()
        outcome = interpret_readiness(
            adapter,
            outcome,
            process_alive=sp.pid is not None,
            elapsed_seconds=elapsed,
        )
        if isinstance(outcome, Ready | Loading):
            self._readiness[spec.name] = outcome
        else:
            self._readiness[spec.name] = None


# One installer per engine, created on demand and kept for the life of the
# process so a terminal install state survives long enough for a UI that
# reconnects afterwards to read how it ended.
_INSTALLERS: dict[EngineKind, EngineInstaller] = {}


def installer_for(kind: EngineKind) -> EngineInstaller | None:
    adapter = adapter_for(kind)
    if adapter is None:
        return None
    existing = _INSTALLERS.get(kind)
    if existing is None:
        existing = EngineInstaller(adapter.managed_store(), kind)
        _INSTALLERS[kind] = existing
    return existing


async def close_installers() -> None:
    """Cancel anything in flight at shutdown."""
    for installer in list(_INSTALLERS.values()):
        await installer.aclose()
    _INSTALLERS.clear()


def _manual_reason(adapter: EngineAdapter, manual: ManualInstall | None) -> str:
    """Why we will not install a `manual` engine, written so the 422 from
    the install endpoint still tells the operator what to do."""
    parts = [
        f"{adapter.kind.value} is installed by the operator, not by this agent: its unit "
        f"of installation is a Python environment, which is not something we can fetch "
        f"and verify."
    ]
    if manual is not None:
        if manual.command:
            parts.append(f"For this host: `{manual.command}`.")
        if manual.notes:
            parts.append(manual.notes)
        parts.append(f"Upstream's install page: {manual.docsUrl}")
    return " ".join(parts)


def plan_for(
    kind: EngineKind,
    *,
    version: str | None = None,
    host: HostAccelerator | None = None,
) -> AcquisitionPlan | Unavailable:
    """What we would fetch for this host, or why we cannot.

    A `manual` engine is always `Unavailable`, on every host, and the
    reason carries the install command — that is what makes the install
    endpoint's 422 an answer rather than a refusal.

    Otherwise adapter-specific by necessity — asset naming is engine
    knowledge, the same kind as argv construction — so this dispatches
    rather than generalising over an interface with one implementation.

    **`host` is passed in by a caller that already resolved it** (review
    §6.1 #5). `detect_host()` shells out to a vendor tool with a 5 s cap
    per probe and is not cached, and `GET /v1/engines` used to reach it
    three times per request — once here, once in `_acquisition_for`, and
    once more for the manual engine's install command. Defaulted rather
    than required so the install route, which resolves nothing else,
    keeps its one call.
    """
    detected = host if host is not None else detect_host()
    adapter = adapter_for(kind)
    if adapter is not None and adapter.install_policy is Policy.manual:
        return Unavailable(reason=_manual_reason(adapter, adapter.manual_install(detected)))
    if not isinstance(adapter, LlamaCppAdapter):
        return Unavailable(
            reason=f"engine {kind.value!r} has no managed-install support in this build"
        )

    if version is None:
        # The newest build that carries THIS host's assets, which since
        # 2026-09-15 is not always the newest build with assets: a release
        # mid-upload has some and not ours.
        return adapter.plan_latest(detected)
    release = next(
        (r for r in adapter.releases.list_releases() if r.version == version),
        None,
    )
    if release is None:
        return Unavailable(
            reason=(
                f"build {version!r} is not among the recent releases of "
                f"{adapter.binary_name}; only recent builds can be installed"
            )
        )
    return adapter.plan_acquisition(detected, release)


def _acquisition_for(kind: EngineKind, host: HostAccelerator) -> EngineAcquisition:
    """The acquisition half of an `EngineDescriptor`.

    Never raises and never blocks on the network beyond the release cache:
    this backs `GET /v1/engines`, which the UI polls, and a failed upstream
    check has to leave the panel stale rather than turn it into an error.

    `host` is required rather than optional here, unlike `plan_for`'s:
    this function has exactly one caller and the whole point is that the
    caller resolved the host once for every adapter.
    """
    adapter = adapter_for(kind)
    detected = host

    if adapter is not None and adapter.install_policy is Policy.manual:
        # Not by us, anywhere. `installable` is false by decision rather
        # than by host, `manualInstall` carries the command for this host,
        # and `reason` says the same thing in prose for a client that
        # reads only that.
        manual = adapter.manual_install(detected)
        return EngineAcquisition(
            policy=Policy.manual,
            installable=False,
            reason=_manual_reason(adapter, manual),
            manualInstall=manual,
            detected=detected,
        )

    latest: Release | None = None
    checked_at: datetime | None = None
    # Inline isinstance rather than a hoisted flag: mypy narrows on the
    # former and not the latter, and `latest_release` lives on the
    # llama.cpp adapter, not the base one.
    if isinstance(adapter, LlamaCppAdapter):
        latest = adapter.latest_release()
        checked_at = adapter.releases.checked_at
    plan = plan_for(kind, host=detected)

    if isinstance(plan, Unavailable):
        return EngineAcquisition(
            policy=Policy.managed,
            installable=False,
            reason=plan.reason,
            detected=detected,
            latestVersion=latest.version if latest else None,
            latestPublishedAt=latest.published_at if latest else None,
            checkedAt=checked_at,
        )
    return EngineAcquisition(
        policy=Policy.managed,
        installable=True,
        variant=plan.variant,
        detected=detected,
        latestVersion=latest.version if latest else None,
        latestPublishedAt=latest.published_at if latest else None,
        checkedAt=checked_at,
    )


def _managed_for(adapter: EngineAdapter) -> ManagedEngine | None:
    builds = adapter.managed_store().list_builds()
    if not builds:
        return None
    current = builds[0]
    return ManagedEngine(
        version=current.version,
        binaryPath=str(current.binary),
        variant=current.variant,
        installedAt=current.installed_at,
        sizeBytes=current.size_bytes,
        previousVersion=builds[1].version if len(builds) > 1 else None,
    )


def describe_engines(get_config: ConfigGetter | None = None) -> list[EngineDescriptor]:
    """What this agent knows how to start, and what it found on disk.

    Backs `GET /v1/engines`. Note what `available` does and does not mean:
    it answers "is a binary discoverable on this host" — by managed
    install, by an install-wide configured path (`vllmBinary`, read
    through `get_config`), or on PATH. A runtime carrying an explicit
    `binary` bypasses discovery entirely and runs happily against an
    engine reported here as unavailable — which reads as a contradiction
    on a dashboard and is worth saying out loud in the `error` text.
    """
    # **Once, for every adapter.** See `plan_for`'s note: this was three
    # `detect_host()` calls per request, each one up to two subprocess
    # probes at a 5 s cap, on a route two pollers hit continuously.
    host = detect_host()
    out: list[EngineDescriptor] = []
    for kind, adapter in ADAPTERS.items():
        acquisition = _acquisition_for(kind, host)
        managed = _managed_for(adapter)

        error: str | None = None
        found: DiscoveredBinary | None
        try:
            found = adapter.discover(configured=_configured_binary(adapter, get_config))
        except EngineUnavailableError as e:
            # A configured path that does not exist. Reported as
            # unavailable with the path named, never quietly replaced by
            # whatever is on PATH.
            found = None
            error = str(e)

        if found is None:
            if error is None:
                if adapter.configured_binary_key is not None:
                    hint = (
                        f"set `{adapter.configured_binary_key}` in the agent config to the "
                        f"{adapter.binary_name!r} console script inside the environment where "
                        f"you installed it, or put it on PATH; `acquisition.manualInstall` "
                        f"has the install command for this host"
                    )
                elif acquisition.installable:
                    hint = f"install one with POST /v1/engines/{kind.value}/install"
                else:
                    hint = "set `binary` on a runtime to point at an existing build"
                error = f"no {adapter.binary_name!r} installed or on PATH — {hint}"
            out.append(
                EngineDescriptor(
                    engine=kind,
                    available=False,
                    modelFormats=list(adapter.model_formats),
                    error=error,
                    flagSchema=adapter.flag_schema(),
                    managed=managed,
                    acquisition=acquisition,
                )
            )
            continue
        out.append(
            EngineDescriptor(
                engine=kind,
                available=True,
                modelFormats=list(adapter.model_formats),
                binaryPath=str(found.path),
                version=found.version,
                origin=found.origin,
                flagSchema=adapter.flag_schema(),
                managed=managed,
                acquisition=acquisition,
                python=found.python,
            )
        )
    return out


def validate_spec(spec: RuntimeSpec) -> str | None:
    """Reject a runtime we cannot honour. Returns a reason, or None.

    Runs before persisting, so `POST /v1/runtimes` fails loudly instead
    of storing a declaration that will only fail at spawn.
    """
    adapter = adapter_for(spec.engine)
    if adapter is None:
        return f"no adapter for engine {spec.engine.value!r}"
    if spec.flags:
        unknown = adapter.validate_flags(spec.flags)
        if unknown:
            known = sorted(f.key for f in adapter.flag_schema().fields)
            return (
                f"unknown flag(s) for {spec.engine.value}: {', '.join(unknown)}. "
                f"Known flags: {', '.join(known)}. Anything not in the curated "
                f"surface goes in extraArgs."
            )
    return None


def _default_alias(spec: RuntimeSpec) -> str:
    """What the runtime serves under when the operator didn't say."""
    return default_model_alias(spec.modelPath)


__all__ = [
    "AcquisitionError",
    "ConfigGetter",
    "EngineInstall",
    "EngineKind",
    "RuntimeSupervisor",
    "close_installers",
    "describe_engines",
    "installer_for",
    "plan_for",
    "validate_spec",
]
