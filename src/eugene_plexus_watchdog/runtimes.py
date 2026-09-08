"""Supervision of engine runtimes.

A *runtime* is one third-party engine process — `llama-server` today —
launched from an argv an `EngineAdapter` builds. It reuses the whole
supervision loop in `supervisor.py` and adds the two things a foreign
binary needs that a Eugene Plexus component does not: an engine-specific
readiness probe, and a status that distinguishes "still loading the
model" from "not answering at all".

Nothing here knows how to *talk* to an engine. That is the driver's job;
see the watchdog spec's note on where engine knowledge lives.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from datetime import datetime

from ._generated.models import (
    EngineAcquisition,
    EngineDescriptor,
    EngineInstall,
    EngineKind,
    ManagedEngine,
    Runtime,
    RuntimeCapabilities,
    RuntimeSpec,
    RuntimeStatus,
)
from .engines import (
    ADAPTERS,
    EngineAdapter,
    EngineUnavailableError,
    Loading,
    Ready,
    adapter_for,
    default_model_alias,
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
    ) -> None:
        self.spec = spec
        self._adapter = adapter
        self._log = log
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
            binary = self._adapter.resolve_binary(self.spec)
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
                f"re-save it so the watchdog can allocate one"
            )

        argv = self._adapter.build_argv(self.spec, binary, port)

        # Engines inherit the ambient environment plus the runtime's own
        # additions. Accelerator selection (CUDA_VISIBLE_DEVICES and
        # friends) rides here, which is how a runtime gets pinned to one
        # card and how two replicas end up on two GPUs.
        env = os.environ.copy()
        if self.spec.env:
            env.update({k: str(v) for k, v in self.spec.env.items()})

        return SpawnPlan(
            argv=argv,
            env=env,
            cwd=self._adapter.working_directory(self.spec, binary),
        )

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


class RuntimeSupervisor:
    """Owns every engine process plus a background readiness-poll task."""

    def __init__(self, log: logging.Logger | None = None) -> None:
        self._log = log or logging.getLogger(__name__)
        self._processes: dict[str, SupervisedProcess] = {}
        self._planners: dict[str, _RuntimePlanner] = {}
        # Latest readiness per runtime. Held here rather than on the
        # process because it is an *observation*, not loop state — the
        # same reason component safe-mode observations live on Supervisor.
        self._readiness: dict[str, Loading | Ready | None] = {}
        self._poll_task: asyncio.Task[None] | None = None

    # --- collection management --------------------------------------------

    def add_and_start(self, spec: RuntimeSpec) -> None:
        """Begin supervising a runtime. Declared-but-not-started when
        `autoStart` is false — that is how a rarely-used large model stays
        configured without holding VRAM."""
        if spec.name in self._processes:
            return
        if spec.autoStart is False:
            self._planners.pop(spec.name, None)
            return
        adapter = adapter_for(spec.engine)
        if adapter is None:
            self._log.error(
                "runtime %s names engine %r, which this watchdog has no adapter "
                "for; not starting it.",
                spec.name,
                spec.engine.value,
            )
            return
        planner = _RuntimePlanner(spec, adapter, self._log)
        sp = SupervisedProcess(planner, self._log)
        self._planners[spec.name] = planner
        self._processes[spec.name] = sp
        sp.start()

    async def remove_and_stop(self, name: str) -> None:
        sp = self._processes.pop(name, None)
        self._planners.pop(name, None)
        self._readiness.pop(name, None)
        if sp is not None:
            await sp.stop()

    async def restart(self, name: str) -> bool:
        sp = self._processes.get(name)
        if sp is None:
            return False
        self._readiness.pop(name, None)
        await sp.restart()
        return True

    async def stop_one(self, name: str) -> None:
        """Stop the engine but keep the runtime declared.

        Distinct from `remove_and_stop` because an engine holds GPU
        memory: an operator who wants the VRAM back needs a stop that is
        neither a delete nor a crash.
        """
        sp = self._processes.pop(name, None)
        self._planners.pop(name, None)
        self._readiness.pop(name, None)
        if sp is not None:
            await sp.stop()

    async def stop_all(self) -> None:
        """Shut everything down. Best-effort — never raises."""
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._poll_task
            self._poll_task = None
        await asyncio.gather(
            *(sp.stop() for sp in self._processes.values()),
            return_exceptions=True,
        )
        self._processes.clear()
        self._planners.clear()
        self._readiness.clear()

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
        return name in self._processes

    # --- read model -------------------------------------------------------

    def compose(self, spec: RuntimeSpec) -> Runtime:
        """Pair a declaration with what is observed of it."""
        sp = self._processes.get(name := spec.name)
        planner = self._planners.get(name)
        readiness = self._readiness.get(name)

        status = self._status_for(sp, readiness)
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

        return Runtime(
            name=spec.name,
            engine=spec.engine,
            modelPath=spec.modelPath,
            modelAlias=spec.modelAlias or _default_alias(spec),
            host=spec.host,
            port=spec.port,
            autoStart=spec.autoStart,
            flags=spec.flags,
            extraArgs=spec.extraArgs,
            env=spec.env,
            workingDirectory=spec.workingDirectory,
            binary=spec.binary,
            status=status,
            url=url,  # type: ignore[arg-type]
            argv=sp.last_argv if sp is not None else None,
            pid=sp.pid if sp is not None else None,
            engineVersion=engine_version,
            capabilities=capabilities,
            lastRestart=last_restart,
            lastError=sp.last_error if sp is not None else None,
        )

    def _status_for(
        self,
        sp: SupervisedProcess | None,
        readiness: Loading | Ready | None,
    ) -> RuntimeStatus:
        """Map loop state + readiness onto the wire enum.

        The interesting half is `starting` vs `loading` vs `ready`: the
        loop cannot tell those apart — as far as it knows the child is
        simply alive — so the readiness observation supplies the
        distinction. This is the same shape as the component mapping,
        which pairs loop state with a /healthz observation.
        """
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
                await asyncio.gather(
                    *(self._probe_one(s) for s in specs),
                    return_exceptions=True,
                )
                await asyncio.sleep(_READINESS_POLL_SECONDS)
        except asyncio.CancelledError:
            return

    async def _probe_one(self, spec: RuntimeSpec) -> None:
        if spec.name not in self._processes or spec.port is None:
            return
        adapter = adapter_for(spec.engine)
        if adapter is None:
            return
        base = f"http://{spec.host or '127.0.0.1'}:{spec.port}"
        outcome = await adapter.probe_readiness(base)
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


def plan_for(kind: EngineKind, *, version: str | None = None) -> AcquisitionPlan | Unavailable:
    """What we would fetch for this host, or why we cannot.

    Adapter-specific by necessity — asset naming is engine knowledge, the
    same kind as argv construction — so this dispatches rather than
    generalising over an interface that has exactly one implementation.
    Widen it when a second engine ships prebuilt binaries; vLLM will not.
    """
    adapter = adapter_for(kind)
    if not isinstance(adapter, LlamaCppAdapter):
        return Unavailable(
            reason=f"engine {kind.value!r} has no managed-install support in this build"
        )

    if version is None:
        release = adapter.latest_release()
    else:
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
    if release is None:
        return Unavailable(
            reason=(
                "could not reach the upstream release list. Check network access, or "
                "set `binary` on the runtime to a build you already have."
            )
        )
    return adapter.plan_acquisition(detect_host(), release)


def _acquisition_for(kind: EngineKind) -> EngineAcquisition:
    """The acquisition half of an `EngineDescriptor`.

    Never raises and never blocks on the network beyond the release cache:
    this backs `GET /v1/engines`, which the UI polls, and a failed upstream
    check has to leave the panel stale rather than turn it into an error.
    """
    adapter = adapter_for(kind)
    latest: Release | None = None
    checked_at: datetime | None = None
    # Inline isinstance rather than a hoisted flag: mypy narrows on the
    # former and not the latter, and `latest_release` lives on the
    # llama.cpp adapter, not the base one.
    if isinstance(adapter, LlamaCppAdapter):
        latest = adapter.latest_release()
        checked_at = adapter.releases.checked_at
    plan = plan_for(kind)

    detected = detect_host()
    if isinstance(plan, Unavailable):
        return EngineAcquisition(
            installable=False,
            reason=plan.reason,
            detected=detected,
            latestVersion=latest.version if latest else None,
            latestPublishedAt=latest.published_at if latest else None,
            checkedAt=checked_at,
        )
    return EngineAcquisition(
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


def describe_engines() -> list[EngineDescriptor]:
    """What this watchdog knows how to start, and what it found on disk.

    Backs `GET /v1/engines`. Note what `available` does and does not mean:
    it answers "is a binary discoverable on this host", by managed install
    or PATH. A runtime carrying an explicit `binary` bypasses discovery
    entirely and runs happily against an engine reported here as
    unavailable — which reads as a contradiction on a dashboard and is
    worth saying out loud in the `error` text.
    """
    out: list[EngineDescriptor] = []
    for kind, adapter in ADAPTERS.items():
        found = adapter.discover()
        acquisition = _acquisition_for(kind)
        managed = _managed_for(adapter)

        if found is None:
            hint = (
                f"install one with POST /v1/engines/{kind.value}/install"
                if acquisition.installable
                else "set `binary` on a runtime to point at an existing build"
            )
            out.append(
                EngineDescriptor(
                    engine=kind,
                    available=False,
                    error=f"no {adapter.binary_name!r} installed or on PATH — {hint}",
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
                binaryPath=str(found.path),
                version=found.version,
                origin=found.origin,
                flagSchema=adapter.flag_schema(),
                managed=managed,
                acquisition=acquisition,
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
    "EngineInstall",
    "EngineKind",
    "RuntimeSupervisor",
    "close_installers",
    "describe_engines",
    "installer_for",
    "plan_for",
    "validate_spec",
]
