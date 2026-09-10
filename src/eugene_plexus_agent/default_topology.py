"""The topology every install has, declared by the agent on its first boot.

An install has exactly one control root, one gateway and one library, on
ports the specs' `servers` entries already declare. That is not a choice
an operator makes; it is what a control plane *is*. The only genuinely
per-install components are inference-drivers, and since M6 the agent
declares those itself, one per runtime. So there was never anything for
a human to construct — yet until now every path to a working install
required constructing it by hand.

What that cost, concretely: the first-run wizard configures components
that already exist and cannot create them, so it would complete against
an empty topology, warn that the gateway was missing, flip
`firstRunComplete` and hand the operator a control plane with nothing in
it. The alternative was a shell script that built the topology over
HTTP. Both are the same bug wearing different clothes - the agent knew
what an install needs and waited to be told anyway.

**Headless is the argument that settles it.** A tailnet install has no
browser at first boot; if the topology can only be built from the UI,
the headless server this project exists to be cannot start itself. So
the agent seeds, and the browser is for changing what it chose.

Four conditions, all required:

  * **Nobody has set this install up** - `firstRunComplete` is false.
    That flag, not the presence of the file, is the signal. `AgentState.
    load()` writes a defaults file when one is missing and `__main__`
    loads a bootstrap state before the app is even built, so "did the
    config file exist" is already false by the time anything could ask,
    and a first boot that crashed before seeding would never seed again.
    It is also the seam every acceptance script already sits on: m6 and
    m7 write `firstRunComplete: true` before starting an agent, so none
    of them change behaviour.
  * **Nothing is declared yet** - an operator who emptied their topology
    after setup has `firstRunComplete` true and keeps their empty
    topology; this only ever adds to nothing.
  * **Unenrolled** - a node that has joined an install gets its topology
    from the install, and a second node must never raise a rival control
    root. See `EUGENE_PLEXUS_AGENT_DEFAULT_TOPOLOGY=0` for the case this
    cannot detect: a fresh node that is *about* to enroll.
  * **Importable** - each component is declared only if its package
    imports from this agent's own interpreter, because that is what the
    agent would spawn it with (`watchdog-venv-is-runtime`). Declaring a
    component that cannot start produces a crash loop and a red
    dashboard, which is a worse first impression than a smaller one.

No config files are written. Each component owns its own file and comes
up on defaults when it is absent, which is what `degraded-mode-required`
already promises; the library's model directories are the operator's to
point at, and guessing them here would be the content-addressed-store
mistake in miniature.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass

from ._generated.models import ComponentEntry, ComponentKind, SpawnConfig
from .state import AgentState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DefaultComponent:
    name: str
    kind: ComponentKind
    port: int
    module: str


# Ports are the specs' `servers` defaults, which are authoritative, and
# are what the UI and every doc assume. An operator who needs different
# ones edits the topology afterwards like any other setting.
DEFAULTS: tuple[DefaultComponent, ...] = (
    DefaultComponent("control", ComponentKind.control, 8083, "eugene_plexus_control"),
    DefaultComponent("gateway", ComponentKind.gateway, 8080, "eugene_plexus_gateway"),
    DefaultComponent("library", ComponentKind.library, 8082, "eugene_plexus_library"),
)


def is_installed(module: str) -> bool:
    """Does this module import from the interpreter that would spawn it?

    `find_spec` rather than `import`: importing a component pulls its
    whole FastAPI app into the agent's process for no reason, and a
    component whose import has side effects would run them here.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A namespace package shadowed by a broken install raises rather
        # than returning None. Not installed, for our purposes.
        return False


def entry_for(component: DefaultComponent, state: AgentState) -> ComponentEntry:
    return ComponentEntry(
        name=component.name,
        kind=component.kind,
        url=f"http://127.0.0.1:{component.port}",  # type: ignore[arg-type]
        spawn=SpawnConfig(configFile=str(state.path.parent / f"{component.name}.yaml")),
        safeMode=False,
    )


def should_seed(state: AgentState, *, enrolled: bool) -> bool:
    """Is this an install nobody has set up yet?

    Deliberately semantic rather than filesystem-based. The obvious test
    - "the config file did not exist" - is unanswerable in practice:
    `AgentState.load()` writes a defaults file when one is missing, and
    `__main__` loads a bootstrap state before `create_app` runs, so the
    file always exists by the time the lifespan looks. It is also wrong
    on its own terms, because a first boot that died before seeding
    would leave a file behind and never seed again.
    """
    if enrolled:
        return False
    if state.get_config("firstRunComplete"):
        return False
    return not state.list_topology_entries()


def seed(state: AgentState) -> list[str]:
    """Declare the default topology. Returns the names declared.

    Idempotent by name, so a partially-seeded install completes rather
    than conflicting. The caller decides *whether* to seed; this decides
    *what*.
    """
    declared: list[str] = []
    for component in DEFAULTS:
        if state.get_topology_entry(component.name) is not None:
            continue
        if not is_installed(component.module):
            log.warning(
                "not declaring %s: %s does not import from this agent's interpreter. "
                "Install it into the agent's environment and declare it with "
                "POST /v1/components, or re-run a fresh install.",
                component.name,
                component.module,
            )
            continue
        state.add_topology_entry(entry_for(component, state))
        declared.append(component.name)
    return declared


__all__ = ["DEFAULTS", "DefaultComponent", "entry_for", "is_installed", "seed", "should_seed"]
