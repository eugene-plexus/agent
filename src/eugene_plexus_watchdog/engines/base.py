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
"""

from __future__ import annotations

import abc
import shutil
from dataclasses import dataclass
from pathlib import Path

from .._generated.models import (
    ConfigSchema,
    EngineKind,
    Origin,
    RuntimeCapabilities,
    RuntimeSpec,
)


@dataclass(frozen=True)
class DiscoveredBinary:
    """Where an engine's executable came from, and what it says it is."""

    path: Path
    origin: Origin
    version: str | None = None


class EngineUnavailableError(Exception):
    """No usable binary for this engine on this host.

    Raised by `resolve_binary`, and turned into a `SpawnPlanError` by the
    planner — a declared runtime whose engine is missing is a crash, not
    a silently-skipped entry.
    """


@dataclass(frozen=True)
class NotAnswering:
    """The engine's endpoint isn't responding at all yet."""

    detail: str | None = None


@dataclass(frozen=True)
class Loading:
    """Answering, but the model is still being read into memory.

    Its own outcome rather than folded into `NotAnswering` because the
    two need different operator responses: a large quant off a slow disk
    can sit here for minutes and that is fine, whereas an endpoint that
    never answers at all is usually wrong argv or a claimed port. A
    generic TCP check cannot tell them apart, which is most of why
    readiness is per-adapter.
    """

    detail: str | None = None


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

    # --- discovery --------------------------------------------------------

    def resolve_binary(self, spec: RuntimeSpec) -> DiscoveredBinary:
        """Find the executable to launch for this runtime.

        Precedence is deliberate: an explicit `binary` on the runtime
        wins over anything discovered, because an operator who built
        llama.cpp themselves for one model should not have that choice
        silently overridden.
        """
        if spec.binary:
            path = Path(spec.binary)
            if not path.is_file():
                raise EngineUnavailableError(
                    f"binary {spec.binary!r} set on runtime {spec.name!r} does not exist"
                )
            return DiscoveredBinary(
                path=path, origin=Origin.configured, version=self.probe_version(path)
            )

        found = self.discover()
        if found is None:
            raise EngineUnavailableError(
                f"no {self.binary_name!r} found for engine {self.kind.value!r} — set "
                f"`binary` on the runtime, or put it on PATH"
            )
        return found

    def discover(self) -> DiscoveredBinary | None:
        """Look for a usable binary without a specific runtime in hand.

        Backs `GET /v1/engines`, which the UI reads to decide whether to
        offer the "add a runtime" form at all. PATH only for now; engine
        acquisition adds the managed location ahead of it.
        """
        which = shutil.which(self.binary_name)
        if which is None:
            return None
        path = Path(which)
        return DiscoveredBinary(path=path, origin=Origin.path, version=self.probe_version(path))

    def probe_version(self, binary: Path) -> str | None:
        """Ask the binary what it is. None when it won't say.

        Worth having even though nothing depends on it: "which build am I
        actually running" is the first question when a flag that used to
        work stops working after an upgrade.
        """
        return None

    # --- launching --------------------------------------------------------

    @abc.abstractmethod
    def build_argv(self, spec: RuntimeSpec, binary: DiscoveredBinary, port: int) -> list[str]:
        """Turn declared intent into a command line.

        `port` is passed in already resolved rather than read off the
        spec, because the watchdog assigns one when the operator didn't
        pick it — with N runtimes nobody should be handing out port
        numbers by hand.
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

    # --- observing --------------------------------------------------------

    @abc.abstractmethod
    async def probe_readiness(self, base_url: str) -> Readiness:
        """Ask a running engine whether it is serving yet."""

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
