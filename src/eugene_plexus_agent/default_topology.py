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
from pathlib import Path

from . import ports
from ._generated.common_models import ConfigUpdateRequest
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
    # Row 53: what the gateway's platform doors keep (files first), on the
    # root beside control. Loopback: other machines reach it through this
    # agent's proxy, as they reach the library.
    DefaultComponent("store", ComponentKind.store, 8084, "eugene_plexus_store"),
)

#: The store joined the defaults after installs existed (row 53). Written
#: beside agent.yaml the first time this agent declares the store, by
#: `seed` or by `add_store`, so a store the owner later removed or moved to
#: another machine is not declared here again.
STORE_DECLARED_MARKER = ".store-declared"


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


#: Every port this install claims for itself, so a walk off one default
#: cannot land on another. 8079 is the agent, 8081 the inference-driver
#: default the wizard and every companion use.
RESERVED_PORTS: frozenset[int] = frozenset({8079, 8081} | {c.port for c in DEFAULTS})


def entry_for(
    component: DefaultComponent, state: AgentState, *, port: int | None = None
) -> ComponentEntry:
    chosen = component.port if port is None else port
    return ComponentEntry(
        name=component.name,
        kind=component.kind,
        url=f"http://127.0.0.1:{chosen}",  # type: ignore[arg-type]
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


def mark_onboarded(state: AgentState) -> bool:
    """Record that this install is set up. Returns True if it changed.

    **Enrolling IS onboarding**, which is the whole framing of
    `onboarding.py`: one question -- am I the root of a new install or
    joining an existing one? -- answered three ways. A node that
    answered "joining" has answered it, and there is nothing left for a
    first-run wizard to do. Running one would raise a SECOND install on
    a machine that already belongs to one.

    This exists because two notions of "set up" had drifted apart.
    `should_seed` above already treats `enrolled` as decisive and
    refuses to seed a control plane onto a node -- correct, and
    invisible. But `firstRunComplete` was written by exactly one thing,
    the web wizard, so an enrolled node still carried `false`, and the
    UI reads that flag alone: sign in on a worker and it bounces you
    into the wizard. Reported from a real two-machine install.

    Fixing the flag rather than teaching the UI about enrollment is
    deliberate: the flag is what every reader consults, so making it
    true of the actual state fixes readers that do not exist yet.
    """
    if state.get_config("firstRunComplete"):
        return False
    # Through the ordinary patch path rather than poking `_config`, so
    # this write is validated and persisted exactly like the wizard's.
    # `ConfigUpdateRequest` declares no fields and allows extras, so a
    # one-key patch really is a one-key patch.
    state.apply_config_patch(ConfigUpdateRequest(firstRunComplete=True))
    return True


def seed(state: AgentState) -> list[str]:
    """Declare the default topology. Returns the names declared.

    Idempotent by name, so a partially-seeded install completes rather
    than conflicting. The caller decides *whether* to seed; this decides
    *what*.

    **A port something else is holding is not a port this install can
    have** (review §6.1 #7). Seeded onto a taken one, a component exits
    within a second of every spawn with `exited with code 1`, and the
    consequences differ by which port it was: **8083** and the wizard's
    first Continue has already set the agent's passphrase before the
    trust root fails, so the operator is stranded half-initialized;
    **8080** — the commonest occupied port on any development box — and
    the wizard *completes*, the install is silently useless, Home shows
    nothing routable and Try it never appears.

    So the port is probed before it is declared, and the walk skips the
    rest of the install's own defaults. This is not reclamation: nothing
    is killed, because at boot the agent cannot tell its own orphan from
    a server the operator meant to be running. It is the choice the
    wizard already makes for a companion driver's port, applied to the
    three components nobody chose.
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
        # Ports already handed to an earlier component in this same loop
        # count as taken: nothing is listening on them yet.
        taken = {_port_of(entry) for entry in state.list_topology_entries()}
        port = ports.first_free(component.port, reserved=RESERVED_PORTS | taken)
        if port != component.port:
            holder = ports.describe_holder(component.port)
            log.warning(
                "port %d is already in use%s, so %s is declared on %d instead. Change it "
                "from Config in the web UI if you want the usual port back.",
                component.port,
                f" by {holder}" if holder else "",
                component.name,
                port,
            )
        state.add_topology_entry(entry_for(component, state, port=port))
        declared.append(component.name)
        if component.kind == ComponentKind.store:
            _mark_store_declared(state)
    return declared


def _store_marker(state: AgentState) -> Path:
    return Path(state.path).parent / STORE_DECLARED_MARKER


def _mark_store_declared(state: AgentState) -> None:
    try:
        _store_marker(state).write_text("row 53", encoding="utf-8")
    except OSError as exc:
        log.warning("could not record that the store was declared here: %s", exc)


def add_store(state: AgentState) -> str | None:
    """Declare the store on a root that was set up before it existed.

    Every install has one (row 53), on the root, as it has a gateway and a
    library. A root set up before this release has a topology already, so
    `seed` never runs again there; this adds the store once, when:

      * this agent runs the install's active control root;
      * no store is declared here;
      * this agent never declared one before (the marker), so a store the
        owner removed, or moved to another machine, stays where they put it;
      * the package imports from this agent's interpreter.

    Returns the name declared, or None.
    """
    store = next(c for c in DEFAULTS if c.kind == ComponentKind.store)
    entries = state.list_topology_entries()
    if _store_marker(state).exists():
        return None
    if any(e.kind == ComponentKind.store for e in entries):
        _mark_store_declared(state)
        return None
    from .standby import hosts_active_control

    if not hosts_active_control(state):
        return None
    if state.get_topology_entry(store.name) is not None:
        log.warning(
            "not declaring the store: a component named %r is already declared here", store.name
        )
        return None
    if not is_installed(store.module):
        log.warning(
            "not declaring the store: %s does not import from this agent's interpreter. "
            "Update this install to get it.",
            store.module,
        )
        return None
    taken = {_port_of(entry) for entry in entries}
    # A declared port is taken even while nothing listens on it yet: this
    # runs before supervision starts, and a root that moved control behind
    # the machine port has it on 8084, which `first_free` would hand out
    # again because it only skips reserved ports after the preferred one.
    preferred = next(p for p in range(store.port, store.port + 64) if p not in taken)
    port = ports.first_free(preferred, reserved=RESERVED_PORTS | taken)
    state.add_topology_entry(entry_for(store, state, port=port))
    _mark_store_declared(state)
    return store.name


def _port_of(entry: ComponentEntry) -> int:
    """The port out of a declared URL, or 0 when it carries none."""
    try:
        return int(str(entry.url).rstrip("/").rsplit(":", 1)[-1])
    except ValueError:
        return 0


__all__ = [
    "DEFAULTS",
    "RESERVED_PORTS",
    "STORE_DECLARED_MARKER",
    "DefaultComponent",
    "add_store",
    "entry_for",
    "is_installed",
    "seed",
    "should_seed",
]
