"""The companion driver: one inference-driver per runtime, kept by the agent.

M2 named the gap — launching a model declared a runtime and stopped
there, so the model loaded, served, and was unreachable. M4 taught a
driver to *follow* a runtime by name. This is the step that *creates*
one: when a runtime is declared with `autoDriver` (the default), the
agent also declares a component named `<runtime>-driver`, kind
`inference-driver`, whose whole config is

    provider: openai_compat_custom
    runtimeName: <runtime>
    modelId: <alias>

and keeps the two together — the driver is removed with the runtime,
renamed with it, and re-targeted when the alias changes.

Decided over a declared pool (2026-09-10): one driver per backend is the
rule the contract has stated since M0, a runtime is a backend, and a
pool would need allocation state that survives restarts and would give
drivers names that mean a different model every hour. The cost is one
FastAPI process per loaded model.

A companion is recognised by its config file living under the agent's
own `drivers/` directory at the path this module would choose. A
component that merely *shares the name* — an operator's hand-made
`foo-driver` — is not one, and declaring runtime `foo` next to it is a
409 rather than a silent takeover.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import yaml

from ._generated.models import ComponentEntry, ComponentKind, RuntimeSpec, SpawnConfig
from .engines import default_model_alias
from .state import AgentState

log = logging.getLogger(__name__)

COMPANION_SUFFIX = "-driver"
COMPANION_DIR = "drivers"
COMPANION_PROVIDER = "openai_compat_custom"


class ComponentSupervisorLike(Protocol):
    """The three calls this module makes on the component supervisor."""

    def add_and_start(self, entry: ComponentEntry) -> None: ...
    async def remove_and_stop(self, name: str) -> None: ...
    async def restart(self, name: str) -> bool: ...


class CompanionConflict(Exception):
    """A component already holds the companion's name and is not one."""


def companion_name(runtime_name: str) -> str:
    return f"{runtime_name}{COMPANION_SUFFIX}"


def companion_config_path(state: AgentState, component_name: str) -> Path:
    """Beside the agent's own config, under `drivers/`."""
    return state.path.parent / COMPANION_DIR / f"{component_name}.yaml"


def is_companion(entry: ComponentEntry, state: AgentState) -> bool:
    """Ours iff its config file is exactly where we would have put it."""
    if entry.kind is not ComponentKind.inference_driver or entry.spawn is None:
        return False
    try:
        return (
            Path(entry.spawn.configFile).resolve()
            == companion_config_path(state, entry.name).resolve()
        )
    except OSError:
        return False


MANAGED_KEYS = ("provider", "runtimeName", "modelId")
"""The three fields the agent owns in a companion's config file.

**Everything else in that document belongs to the operator** (R2.5).
The driver's own `PATCH /v1/config` writes into the same file -- that is
how a Config tab in the tree saves anything -- so `requestTimeoutSeconds`,
`logLevel` and every field a future provider adds arrive here from a
browser, not from us.
"""


def render_config(*, runtime_name: str, alias: str) -> dict[str, Any]:
    return {
        "provider": COMPANION_PROVIDER,
        "runtimeName": runtime_name,
        "modelId": alias,
    }


def _read_config(path: Path) -> dict[str, Any]:
    """Whatever is on disk, or `{}` — never a reason not to boot.

    A companion config we cannot parse is the operator's to fix through
    the driver's own degraded-mode config surface; refusing to reconcile
    the runtime over it would take the whole install down for one bad
    file (`degraded-mode-required`).
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_config(path: Path, managed: dict[str, Any]) -> bool:
    """Set the three fields we manage; leave the rest of the file alone.

    **True iff a MANAGED key moved**, which is what the caller turns
    into a restart. Not "iff the bytes changed": an operator's edit
    changes the bytes and must not restart their driver a second time,
    and a reconcile that reported `changed` for its own reformatting
    would restart every companion in the install at every boot.

    As found (review §6.2 #13, verification): this rendered the three
    fields and wrote the result over the file, so the boot reconcile --
    which runs for every runtime, and M6 declares one companion per
    runtime -- silently discarded every setting the operator had saved.
    A knob that does not survive the next restart is not a knob, and
    `requestTimeoutSeconds` is precisely the one R2.5 exists to make
    worth turning.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    document = _read_config(path) if path.exists() else {}
    changed = any(document.get(key) != value for key, value in managed.items())
    if not changed and path.exists():
        return False
    document.update(managed)
    path.write_text(
        yaml.safe_dump(document, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )
    return changed


def resolved_alias(spec: RuntimeSpec) -> str:
    return spec.modelAlias or default_model_alias(spec.modelPath)


async def ensure_companion(
    state: AgentState,
    supervisor: ComponentSupervisorLike | None,
    spec: RuntimeSpec,
) -> ComponentEntry:
    """Declare the companion for `spec` if missing; re-target it if not.

    Idempotent, which is what lets boot-time reconcile call it for every
    runtime. Raises `CompanionConflict` when a non-companion component
    already holds the name.
    """
    name = companion_name(spec.name)
    existing = state.get_topology_entry(name)
    if existing is not None and not is_companion(existing, state):
        raise CompanionConflict(
            f"a component named {name!r} already exists and is not a companion driver "
            f"(its config is {existing.spawn.configFile if existing.spawn else 'remote'}, not "
            f"{companion_config_path(state, name)}). Rename the runtime, or set "
            f"autoDriver: false and front it with that driver by hand."
        )

    path = companion_config_path(state, name)
    changed = _write_config(path, render_config(runtime_name=spec.name, alias=resolved_alias(spec)))

    if existing is None:
        entry = ComponentEntry(
            name=name,
            kind=ComponentKind.inference_driver,
            url=f"http://127.0.0.1:{state.allocate_component_port()}",  # type: ignore[arg-type]
            spawn=SpawnConfig(configFile=str(path)),
            safeMode=False,
        )
        entry = state.add_topology_entry(entry)
        log.info("declared companion driver %s for runtime %s on %s", name, spec.name, entry.url)
        if supervisor is not None:
            supervisor.add_and_start(entry)
        return entry

    if changed and supervisor is not None:
        # `modelId` and `runtimeName` are read at driver startup; a
        # re-targeted companion has to restart to serve the new alias.
        log.info("companion driver %s re-targeted; restarting it", name)
        await supervisor.restart(name)
    return existing


async def remove_companion(
    state: AgentState,
    supervisor: ComponentSupervisorLike | None,
    runtime_name: str,
) -> bool:
    """Take the companion down with its runtime. The config file is
    left, like every component's, so re-declaring picks up where it was."""
    name = companion_name(runtime_name)
    existing = state.get_topology_entry(name)
    if existing is None or not is_companion(existing, state):
        return False
    if supervisor is not None:
        await supervisor.remove_and_stop(name)
    state.remove_topology_entry(name)
    log.info("removed companion driver %s with runtime %s", name, runtime_name)
    return True


async def reconcile(
    state: AgentState,
    supervisor: ComponentSupervisorLike | None,
) -> list[str]:
    """At boot: every runtime with `autoDriver` gets its companion.

    An agent upgraded onto an existing install gains companions for its
    runtimes on first start; an operator who deleted one by hand gets it
    back, which is the honest reading of `autoDriver: true`. Conflicts
    are logged, not fatal — a boot must not stop over one runtime.
    """
    created: list[str] = []
    for spec in state.list_runtime_specs():
        if spec.autoDriver is False:
            continue
        before = state.get_topology_entry(companion_name(spec.name))
        try:
            await ensure_companion(state, supervisor, spec)
        except CompanionConflict as e:
            log.error("runtime %s: %s", spec.name, e)
            continue
        if before is None:
            created.append(companion_name(spec.name))
    return created


__all__ = [
    "COMPANION_DIR",
    "COMPANION_PROVIDER",
    "COMPANION_SUFFIX",
    "MANAGED_KEYS",
    "CompanionConflict",
    "companion_config_path",
    "companion_name",
    "ensure_companion",
    "is_companion",
    "reconcile",
    "remove_companion",
    "render_config",
    "resolved_alias",
]
