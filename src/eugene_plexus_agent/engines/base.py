"""What an engine adapter has to know.

Two kinds of engine knowledge exist in Eugene Plexus and they live in
different components. How to **start** an engine — argv, working
directory, readiness semantics, which flags are worth exposing — is here,
in the supervisor. How to **talk to** one — the wire protocol — lives in
the inference-driver. Nobody imports anybody; the split is the reason
there is no shared engine library.

An adapter is deliberately the *only* way an engine becomes supported.
`RuntimeSpec` carries intent (this engine, this model, these flags) and
never a command line, so without an adapter there is nothing that knows
how to launch a thing or when it has finished loading.

M4 added a second engine and, with it, the discovery that the readiness
outcomes below had been written in the first engine's mechanism rather
than by what they mean. They are defined by meaning now — see
`NotAnswering`, `Loading` and `interpret_readiness`.
"""

from __future__ import annotations

import abc
import shutil
from dataclasses import dataclass
from pathlib import Path

from .._generated.models import (
    ConfigSchema,
    EngineKind,
    HostAccelerator,
    ManualInstall,
    ModelFormat,
    Origin,
    Policy,
    PythonEngine,
    RuntimeCapabilities,
    RuntimeSpec,
)
from .acquisition import ManagedStore, engine_root


@dataclass(frozen=True)
class DiscoveredBinary:
    """Where an engine's executable came from, and what it says it is."""

    path: Path
    origin: Origin
    version: str | None = None
    #: For a Python-package engine, the environment the console script
    #: belongs to. None for a self-contained binary such as llama-server.
    python: PythonEngine | None = None


class EngineUnavailableError(Exception):
    """No usable binary for this engine on this host.

    Raised by `resolve_binary` and `discover`, and turned into a
    `SpawnPlanError` by the planner — a declared runtime whose engine is
    missing is a crash, not a silently-skipped entry.
    """


@dataclass(frozen=True)
class NotAnswering:
    """We have no readiness information about the engine.

    Defined by what it means, not by how any one engine reports it: *the
    process is not yet serving, and nothing tells us why.* For an engine
    that answers HTTP while it loads (llama-server) this is only the
    moments before its socket is up. For one that does not (vLLM binds
    its port and answers nothing until the model is resident) a bare
    connection failure is turned into `Loading` by the supervisor, which
    holds the process handle and so knows the child is alive — see
    `interpret_readiness`. This outcome therefore survives only where
    genuinely nothing is known.

    `reached` separates "the port refused or timed out" (False) from
    "something answered, just not with a readiness we can vouch for"
    (True). Only the first can be a silent load. A 503 from a listening
    server is information, and must not be read as silence.
    """

    detail: str | None = None
    reached: bool = False


@dataclass(frozen=True)
class Loading:
    """The engine is alive and working, and is not servable yet.

    Its own outcome rather than folded into `NotAnswering` because the
    two need different operator responses: a large model can sit here
    for minutes and that is fine, whereas an endpoint that never answers
    at all is usually wrong argv or a claimed port.

    How this is *observed* is per-engine, and that is most of why
    readiness is per-adapter. llama-server says so itself, answering
    `/health` with 503 and a loading status. vLLM says nothing — for it,
    "the process is alive and nothing answers" *is* this state, because
    there is nothing else it could be, and only the supervisor can say
    so because only the supervisor has the pid.

    `past_budget` is set when a silent load has outrun the adapter's
    `startup_budget_seconds`. The state stays `loading` — a live, silent
    process is still loading as far as anyone can tell — but the
    operator gets the elapsed time and a pointer at the captured output,
    which is how "working on it" stays separable from "wedged" for an
    engine that will not narrate the difference.
    """

    detail: str | None = None
    past_budget: bool = False


@dataclass(frozen=True)
class Ready:
    """Model loaded and serving. The only state the gateway routes to."""

    capabilities: RuntimeCapabilities | None = None
    version: str | None = None


# A sum type, not a hierarchy: there are exactly three answers to "is it
# serving yet", callers match on them, and nothing is shared between them
# worth inheriting.
Readiness = NotAnswering | Loading | Ready


class EngineAdapter(abc.ABC):
    """One engine's start-and-observe knowledge."""

    #: Which `EngineKind` this adapter implements.
    kind: EngineKind

    #: Executable name looked for on PATH when no explicit binary is set.
    binary_name: str

    #: On-disk model formats this engine can load. A property of the
    #: engine, not of this host — it does not change with availability.
    #:
    #: The engine half of a join the UI performs: the library reports
    #: what format each model *is*, this reports what each engine can
    #: *load*, and between them the UI can grey out a launch button and
    #: name the missing engine instead of offering one that fails.
    #:
    #: It lives here because engine knowledge lives here. Putting format
    #: support on the library would give the library a copy of it, and
    #: the copy would be the one that went stale.
    model_formats: tuple[ModelFormat, ...]

    #: Whether the engine answers HTTP while its model loads.
    #:
    #: llama-server does: 503 plus a loading status from the moment its
    #: socket is up. vLLM does not: it binds its port before loading, on
    #: purpose, and does not `listen()` until the model is resident, so
    #: for minutes it is indistinguishable over the network from a dead
    #: process. For an adapter that sets this False, the supervisor
    #: reads "process alive, connection refused" as `Loading` — see
    #: `interpret_readiness`.
    answers_while_loading: bool = True

    #: How long a silent load may run before the supervisor flags it,
    #: in seconds. None means never. Only meaningful when
    #: `answers_while_loading` is False; an engine that narrates its own
    #: load does not need us to guess at its budget. The flagged state is
    #: still `loading` — see `Loading.past_budget`.
    startup_budget_seconds: float | None = None

    #: Whether we install this engine (`managed`: fetch, verify, retain
    #: builds) or only discover what the operator installed (`manual`).
    #: A property of the engine, not of this host: llama.cpp ships
    #: verifiable release assets; vLLM's unit of installation is a Python
    #: environment, which is not something we can fetch and hash.
    install_policy: Policy = Policy.managed

    #: Agent config key holding an install-wide path to this engine's
    #: executable (`vllmBinary`), or None. Exists for engines we never
    #: manage: without it an operator with a perfectly good venv reads
    #: as `available: false` unless they put it on PATH or repeat the
    #: path on every runtime. The *caller* reads the key — config lives
    #: in `AgentState` and an adapter is a stateless singleton — and
    #: passes the value in as `configured`.
    configured_binary_key: str | None = None

    # --- discovery --------------------------------------------------------

    def resolve_binary(
        self, spec: RuntimeSpec, *, configured: str | None = None
    ) -> DiscoveredBinary:
        """Find the executable to launch for this runtime.

        Precedence is deliberate: an explicit `binary` on the runtime
        wins over anything discovered, because an operator who built
        llama.cpp themselves for one model should not have that choice
        silently overridden. Below that, `configured` is the install-wide
        path from the agent's own config, then a managed build, then
        PATH — see `discover`.
        """
        if spec.binary:
            path = Path(spec.binary)
            if not path.is_file():
                raise EngineUnavailableError(
                    f"binary {spec.binary!r} set on runtime {spec.name!r} does not exist"
                )
            return self.describe(path, Origin.configured)

        found = self.discover(configured=configured)
        if found is None:
            raise EngineUnavailableError(self._nothing_found_message())
        return found

    def _nothing_found_message(self) -> str:
        if self.install_policy is Policy.manual:
            return (
                f"no {self.binary_name!r} found for engine {self.kind.value!r} — set "
                f"`{self.configured_binary_key}` in the agent config to the console "
                f"script inside the environment where you installed it, set `binary` "
                f"on the runtime, or put it on PATH. GET /v1/engines carries the "
                f"install command for this host under `acquisition.manualInstall`"
            )
        return (
            f"no {self.binary_name!r} found for engine {self.kind.value!r} — install "
            f"one with POST /v1/engines/{self.kind.value}/install, set `binary` on "
            f"the runtime, or put it on PATH"
        )

    def managed_store(self) -> ManagedStore:
        """Where builds this install fetched for itself live."""
        return ManagedStore(engine_root(), self.kind)

    def discover(self, *, configured: str | None = None) -> DiscoveredBinary | None:
        """Look for a usable binary without a specific runtime in hand.

        Backs `GET /v1/engines`, which the UI reads to decide whether to
        offer the "add a runtime" form at all.

        Precedence: `configured` (the install-wide path from the agent's
        config), then a managed build, then PATH. A configured path that
        does not exist is an error rather than a fallback, for the same
        reason a runtime's missing `binary` is: the operator said where
        the engine is, and quietly using a different one would run
        something they did not choose.

        A managed build beats one on PATH. The operator asked us to
        manage this engine, so a stray `llama-server` that happens to be
        on PATH — an old system package, something another tool
        installed — must not quietly win over the build we fetched and
        verified. An explicit `binary` on a runtime still beats all of
        these; see `resolve_binary`.
        """
        if configured:
            path = Path(configured).expanduser()
            if not path.is_file():
                raise EngineUnavailableError(
                    f"`{self.configured_binary_key or 'configured binary'}` points at "
                    f"{configured}, which does not exist"
                )
            return self.describe(path, Origin.configured)

        managed = self.managed_store().current()
        if managed is not None:
            return DiscoveredBinary(
                path=managed.binary,
                origin=Origin.managed,
                # The recorded build number, not a re-probe: it came off
                # the release we installed, and spawning the binary on
                # every /v1/engines call to re-learn it would be silly.
                version=managed.version,
            )

        which = shutil.which(self.binary_name)
        if which is None:
            return None
        return self.describe(Path(which), Origin.path)

    def describe(self, path: Path, origin: Origin) -> DiscoveredBinary:
        """Turn a path we decided to use into a `DiscoveredBinary`.

        The one place a found executable gets inspected, so an adapter
        that learns more than a version from it (a Python engine reads
        its whole environment) overrides this rather than every caller.
        """
        return DiscoveredBinary(path=path, origin=origin, version=self.probe_version(path))

    def probe_version(self, binary: Path) -> str | None:
        """Ask the binary what it is. None when it won't say.

        Worth having even though nothing depends on it: "which build am I
        actually running" is the first question when a flag that used to
        work stops working after an upgrade.
        """
        return None

    def manual_install(self, host: HostAccelerator) -> ManualInstall | None:
        """How the operator installs this engine themselves, for this host.

        Only a `manual` engine answers. The point is that "we do not
        install this one" stays an answer the operator can act on: a
        refusal that names the exact command is a different product from
        one that says no.
        """
        return None

    # --- launching --------------------------------------------------------

    @abc.abstractmethod
    def build_argv(self, spec: RuntimeSpec, binary: DiscoveredBinary, port: int) -> list[str]:
        """Turn declared intent into a command line.

        `port` is passed in already resolved rather than read off the
        spec, because the agent assigns one when the operator didn't
        pick it — with N runtimes nobody should be handing out port
        numbers by hand.

        The resolved model alias is always passed to the engine
        explicitly, never left to the engine's own default. vLLM's
        default served name is the `--model` argument verbatim, and we
        launch by absolute path — so leaving it unset would publish the
        operator's directory layout as an OpenAI model id.
        """

    def working_directory(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> str | None:
        """Where to run the engine.

        Defaults to the binary's own directory: prebuilt llama.cpp
        releases ship their shared libraries alongside the executable and
        will not start from an unrelated cwd.
        """
        if spec.workingDirectory:
            return spec.workingDirectory
        return str(binary.path.parent)

    def default_env(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> dict[str, str]:
        """Environment this adapter supplies as a *default* for its engine.

        Applied with `setdefault` semantics against the ambient
        environment and then overridden by `RuntimeSpec.env`, so an
        operator beats us at two levels: exporting the variable in the
        shell that starts the agent, or setting it on the runtime through
        the API or the UI. Whatever is injected is logged, because an
        environment variable nobody typed is exactly the kind of thing
        that makes a later bug report unreadable.

        This is for values an engine **cannot start without on the
        detected host**, not for tuning. A default that merely performs
        better is a decision belonging to the operator or to
        `ModelProfile`; a default that is the difference between running
        and not running is ours to supply, because the alternative is a
        working install that refuses to work for a reason only someone
        who reads upstream's source would find. See vLLM's for the two
        that qualify, and `docs/acceptance/m4-vllm-run.md` for the
        tracebacks they replace.
        """
        return {}

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        """A better `lastError` than "exited with code N", or None.

        The supervisor keeps a bounded tail of the engine's own output
        and offers it here on a non-zero exit. An adapter that recognises
        a known failure signature returns a sentence naming the fix;
        anything else returns None and the generic message stands.

        Deliberately NOT a pre-launch refusal. The obvious design was to
        check the host's toolchain before spawning and refuse early, and
        it is wrong: a warm Triton cache runs vLLM with no C compiler on
        the host at all — measured, `CC=/nonexistent` and it served in
        19.9s — so refusing on a missing compiler would reject a launch
        that works, on any host that has run the engine once. Explaining
        a real failure cannot be a false refusal, which is the whole
        reason it reads output after the fact instead of probing before.
        """
        return None

    # --- observing --------------------------------------------------------

    @abc.abstractmethod
    async def probe_readiness(self, base_url: str) -> Readiness:
        """Ask a running engine whether it is serving yet.

        A *network* observation only. It must not try to infer whether
        the process is alive — it cannot — and it must keep a short
        timeout: an engine that refuses connections for minutes turns a
        long timeout into a slow poll, not a better answer. The
        supervisor combines this with what it knows about the process in
        `interpret_readiness`.
        """

    # --- configuring ------------------------------------------------------

    @abc.abstractmethod
    def flag_schema(self) -> ConfigSchema:
        """The **curated** flag surface, as a standard `ConfigSchema`.

        Curated, not complete. `llama-server` alone has hundreds of
        flags and a form exposing all of them is a form nobody can use;
        what belongs here is what a person actually turns. Everything
        omitted stays reachable through `RuntimeSpec.extraArgs`.

        Returning a `ConfigSchema` rather than something engine-shaped is
        what lets the generic config editor render engine flags with no
        engine-specific UI code.
        """

    def validate_flags(self, flags: dict[str, object]) -> list[str]:
        """Return the flag keys this adapter does not recognise.

        Callers reject on a non-empty result. A typo'd flag that silently
        vanishes is far worse than one that errors: the engine starts,
        behaves differently from what the operator asked for, and nothing
        says why.
        """
        known = {f.key for f in self.flag_schema().fields}
        return sorted(k for k in flags if k not in known)


def interpret_readiness(
    adapter: EngineAdapter,
    outcome: Readiness,
    *,
    process_alive: bool,
    elapsed_seconds: float | None,
) -> Readiness:
    """Combine a network probe with what the supervisor knows.

    The one rule M4 added: for an engine that does not answer while
    loading, **"the process is alive and nothing answers" is `Loading`.**
    It is not an inference or a guess — there is no other thing it could
    be — and it needs the process handle, which the supervisor has and a
    probe does not. That is the whole reason lifecycle adapters live in
    the supervisor rather than in the driver, which sees only the
    network and for which vLLM's entire load phase would look like a dead
    engine.

    Everything else passes through untouched: a probe that reached
    something (a 503, a 500) is information, not silence; an engine that
    narrates its own load never needs this; and a process that is not
    alive has nothing to be loading.

    Past the adapter's startup budget the answer is still `Loading`,
    flagged, with the elapsed time — the operator gets evidence, not a
    state machine that guesses "wedged" from a clock.
    """
    if (
        not isinstance(outcome, NotAnswering)
        or outcome.reached
        or adapter.answers_while_loading
        or not process_alive
    ):
        return outcome

    budget = adapter.startup_budget_seconds
    elapsed = f"{elapsed_seconds:.0f}s" if elapsed_seconds is not None else "an unknown time"
    past_budget = budget is not None and elapsed_seconds is not None and elapsed_seconds > budget
    if past_budget:
        assert budget is not None
        detail = (
            f"process alive for {elapsed} and still not answering; {adapter.kind.value}'s "
            f"startup budget is {budget:.0f}s. Still treated as loading — a live, silent "
            f"process cannot be anything else — but check the captured engine output "
            f"for a stall."
        )
    else:
        detail = (
            f"process alive, nothing answering yet ({elapsed}). {adapter.kind.value} "
            f"answers nothing until the model is resident, so this is the load."
        )
    return Loading(detail=detail, past_budget=past_budget)


def default_model_alias(model_path: str) -> str:
    """Filename with its extension stripped.

    Because models are stored as plainly-named files in directories the
    user chose, the obvious name is already the right one — so this is
    what `modelAlias` defaults to, and setting it is an override rather
    than a requirement. Multi-file formats point at a directory, whose
    name is the model name.

    Engine-agnostic on purpose: every adapter passes the resolved alias
    to its engine explicitly, so they had better agree on the default.
    """
    path = Path(model_path)
    if path.suffix.lower() == ".gguf":
        return path.stem
    return path.name or path.stem
