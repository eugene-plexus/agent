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
from ._private_files import write_private
from .engines import adapter_for, default_model_alias
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


MANAGED_KEYS = (
    "provider",
    "runtimeName",
    "modelId",
    "upstreamModelId",
    "decisionMaxConcurrent",
    # CB4: set from the runtime's `slotPinning` flag, with the engine's
    # `--no-cache-idle-slots`; never one without the other.
    "slotPinning",
)

#: Handed to every companion at spawn, naming `MANAGED_KEYS`, so the driver
#: shows them read-only and refuses a PATCH of them (2026-09-30, settings
#: never lie). Before, all five were ordinary editable fields: an edit
#: stuck until this agent's next boot rewrote it, so the page showed a
#: value that was about to stop being true.
MANAGED_KEYS_ENV = "EUGENE_PLEXUS_DRIVER_MANAGED_KEYS"


def _companion_env() -> dict[str, str]:
    return {MANAGED_KEYS_ENV: ",".join(MANAGED_KEYS)}


"""The six fields the agent owns in a companion's config file.

**Everything else in that document belongs to the operator** (R2.5).
The driver's own `PATCH /v1/config` writes into the same file -- that is
how a Config tab in the tree saves anything -- so `requestTimeoutSeconds`,
`logLevel` and every field a future provider adds arrive here from a
browser, not from us.
"""


def render_config(
    *,
    runtime_name: str,
    alias: str,
    upstream: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The managed document, `upstreamModelId` included even when null.

    Rendered unconditionally rather than only for engines that need it,
    because `_write_config` compares only the keys present in `managed`:
    a key emitted conditionally would linger in the file after a runtime
    changed engine, and the driver would keep translating for a backend
    that no longer speaks the old sentinel.
    """
    document: dict[str, Any] = {
        "provider": COMPANION_PROVIDER,
        "runtimeName": runtime_name,
        "modelId": alias,
        "upstreamModelId": upstream,
        # None unless the engine's overrides say otherwise, rendered
        # anyway for the same clearing rule as upstreamModelId.
        "decisionMaxConcurrent": None,
        "slotPinning": None,
    }
    for key, value in (overrides or {}).items():
        if key not in MANAGED_KEYS:
            raise ValueError(f"engine companion override {key!r} is not a managed key")
        document[key] = value
    return document


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
    """Set the fields we manage (`MANAGED_KEYS`); leave the rest of the file alone.

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

    **Written owner-only and atomically** (2026-09-22). Keeping the
    operator's fields means keeping an `apiKey` they saved on the
    driver's own Config page, so this file is exactly as secret as that
    key; `write_text` created it at the umask's 0644 and truncated it
    in place.
    """
    document = _read_config(path) if path.exists() else {}
    changed = any(document.get(key) != value for key, value in managed.items())
    if not changed and path.exists():
        return False
    document.update(managed)
    write_private(path, yaml.safe_dump(document, sort_keys=True, default_flow_style=False))
    return changed


def _seed_config(path: Path, seeds: dict[str, str]) -> bool:
    """Fill `seeds` into the companion's file where those fields are empty.

    True iff something was written, which is a change the driver reads at
    startup. Never replaces a value: an operator's own, or one the driver
    has since sealed into an envelope, stands. Written the way
    `_write_config` writes, owner-only and atomically, because what is
    seeded here is a key.
    """
    if not seeds:
        return False
    document = _read_config(path) if path.exists() else {}
    missing = {key: value for key, value in seeds.items() if not document.get(key)}
    if not missing:
        return False
    document.update(missing)
    write_private(path, yaml.safe_dump(document, sort_keys=True, default_flow_style=False))
    log.info("companion %s: %s filled from its runtime", path.stem, ", ".join(sorted(missing)))
    return True


def companion_secrets(spec: RuntimeSpec) -> dict[str, str]:
    """The engine's fill-if-empty fields for its companion (Kev's key)."""
    adapter = adapter_for(spec.engine)
    if adapter is None:
        return {}
    return dict(adapter.companion_secrets(spec))


def resolved_alias(spec: RuntimeSpec) -> str:
    return spec.modelAlias or default_model_alias(spec.modelPath)


def companion_overrides(spec: RuntimeSpec) -> dict[str, Any]:
    """The engine's own managed-key overrides — Kev's decision provider
    and concurrency ceiling; nothing for the chat engines."""
    adapter = adapter_for(spec.engine)
    if adapter is None:
        return {}
    return dict(adapter.companion_overrides(spec))


def upstream_model_id(spec: RuntimeSpec) -> str | None:
    """What the companion must send its backend, when that is not the alias.

    Engine knowledge, so the adapter answers: `mlx_lm.server` has no flag
    to serve a chosen name and resolves only upstream's `default_model`
    sentinel, so its companion translates; every other engine is launched
    WITH the alias (`--alias`, `--served-model-name`) and gets None.
    """
    adapter = adapter_for(spec.engine)
    if adapter is None:
        return None
    return adapter.upstream_model_id(spec)


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
    changed = _write_config(
        path,
        render_config(
            runtime_name=spec.name,
            alias=resolved_alias(spec),
            upstream=upstream_model_id(spec),
            overrides=companion_overrides(spec),
        ),
    )
    # After the managed keys: a seeded key is read at startup like them,
    # so filling one is a change that restarts the driver.
    changed = _seed_config(path, companion_secrets(spec)) or changed

    if existing is None:
        entry = ComponentEntry(
            name=name,
            kind=ComponentKind.inference_driver,
            url=f"http://127.0.0.1:{state.allocate_component_port()}",  # type: ignore[arg-type]
            spawn=SpawnConfig(configFile=str(path), env=_companion_env()),
            safeMode=False,
        )
        entry = state.add_topology_entry(entry)
        log.info("declared companion driver %s for runtime %s on %s", name, spec.name, entry.url)
        if supervisor is not None:
            supervisor.add_and_start(entry)
        return entry

    spawn = existing.spawn
    if (
        spawn is not None
        and (spawn.env or {}).get(MANAGED_KEYS_ENV) != _companion_env()[MANAGED_KEYS_ENV]
    ):
        # A companion declared before its managed keys were named: name
        # them, once. The supervisor holds the entry it started from, so
        # it is started again from the new one.
        env = dict(spawn.env or {}) | _companion_env()
        updated = existing.model_copy(update={"spawn": spawn.model_copy(update={"env": env})})
        state.update_topology_entry(name, updated)
        log.info("companion driver %s now names the keys this agent manages", name)
        if supervisor is not None:
            await supervisor.remove_and_stop(name)
            supervisor.add_and_start(updated)
        return updated

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
