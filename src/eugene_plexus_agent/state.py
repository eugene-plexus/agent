"""File-backed state for the agent: UI prefs + topology, one YAML file.

Owns the persistent contents of `agent.yaml`. Two facades read through
this — `ConfigStore` (the standard config trio for UI prefs +
firstRunComplete) and `TopologyStore` (the components list under
`/v1/components`). Both share one lock and one on-disk file so concurrent
PATCH operations from different routes can't corrupt each other.

The on-disk shape:

    firstRunComplete: false
    uiTheme: auto
    uiFontSize: medium
    components:
      - name: gateway
        kind: gateway
        url: http://127.0.0.1:8080
        spawn:
          configFile: ~/.eugene-plexus/gateway/config.yaml
        safeMode: false
      - name: qwen3-30b
        kind: inference-driver
        url: http://127.0.0.1:8081
        spawn:
          configFile: ~/.eugene-plexus/drivers/qwen3-30b.yaml
        safeMode: false
      - ...

Flat config-field naming (uiTheme, uiFontSize) rather than a nested `ui:`
section because the existing ConfigField machinery is flat — `category:
"ui"` provides the UI's grouping hint without forcing the YAML into a
shape the generic config editor doesn't natively understand.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from ._generated.common_models import (
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    ConfigValueType,
)
from ._generated.models import (
    Component,
    ComponentEntry,
    ComponentKind,
    ComponentStatus,
    RuntimeSpec,
)

# Port range the agent assigns engine runtimes from when the operator
# doesn't pick one. Above the component ports (8079-8083) and clear of the
# usual ephemeral range, so an assigned port doesn't collide with an
# outbound socket the OS handed out.
_RUNTIME_PORT_BASE = 8090
_RUNTIME_PORT_SPAN = 100

CONFIG_FIELDS: list[ConfigField] = [
    ConfigField(
        key="firstRunComplete",
        label="First-run setup complete",
        description=(
            "Set to true by the first-run wizard's final Start step. "
            "While false the UI routes operators to /setup; flipping it "
            "back to false re-enters the wizard on the next reload. "
            "Safe to leave alone unless you want to re-do setup."
        ),
        category="setup",
        valueType=ConfigValueType.boolean,
        default=False,
    ),
    ConfigField(
        key="securityMode",
        label="Security mode",
        description=(
            "How the agent handles its master encryption key "
            "between restarts. Set during the wizard, editable later. "
            "'Prompt on startup' keeps the master key in process "
            "memory only (passphrase required every restart). "
            "'OS keyring' stores it in your OS Credential Manager / "
            "Keychain / Secret Service for auto-unlock; more "
            "convenient, weaker boundary."
        ),
        category="security",
        valueType=ConfigValueType.enum,
        default="prompt_on_startup",
        enumValues=["prompt_on_startup", "os_keyring"],
        enumLabels=["Prompt on startup", "OS keyring auto-unlock"],
        requiresRestart=True,
    ),
    ConfigField(
        key="uiTheme",
        label="Theme",
        description="Color theme for the Eugene Plexus UI.",
        category="ui",
        valueType=ConfigValueType.enum,
        default="auto",
        enumValues=["light", "dark", "auto"],
        enumLabels=["Light", "Dark", "Auto (follow system)"],
    ),
    ConfigField(
        key="uiFontSize",
        label="Font size",
        description="Base font size for the Eugene Plexus UI.",
        category="ui",
        valueType=ConfigValueType.enum,
        default="medium",
        enumValues=["small", "medium", "large"],
        enumLabels=["Small", "Medium", "Large"],
    ),
    ConfigField(
        key="vllmBinary",
        label="vLLM binary",
        description=(
            "Install-wide path to the `vllm` console script inside the "
            "virtual environment where you installed vLLM — for example "
            "`/home/you/vllm/.venv/bin/vllm`. vLLM is an engine this agent "
            "drives but does not install, so without this it is found only "
            "if it is on PATH or a `binary` is set on every runtime. Point "
            "at the console script, not at a Python interpreter or a venv "
            "directory: the script's shebang binds its own interpreter, so "
            "nothing needs activating. A `binary` set on an individual "
            "runtime still wins over this. Read at the next spawn; no agent "
            "restart needed."
        ),
        category="engines",
        valueType=ConfigValueType.file_path,
    ),
]
CATEGORY_LABELS: dict[str, str] = {
    "setup": "Setup",
    "security": "Security",
    "ui": "Appearance",
    "engines": "Engines",
}

_CONFIG_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in CONFIG_FIELDS}


def _config_defaults() -> dict[str, Any]:
    return {f.key: f.default for f in CONFIG_FIELDS if f.default is not None}


class AgentState:
    """Threadsafe owner of `agent.yaml`. Single lock, single file write.

    Holds three things:
      * The flat config dict exposed via `/v1/config` (`firstRunComplete`,
        `securityMode`, UI prefs).
      * The component topology under `/v1/components`.
      * (v0.2) An `auth` block (`passphraseHash`, `masterSalt`) which is
        persisted but NOT exposed via the config endpoints — it's
        internal trust-root state read by the login endpoint only.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._config: dict[str, Any] = _config_defaults()
        self._components: dict[str, ComponentEntry] = {}
        # Engine runtimes, persisted under `runtimes:`. Kept separate from
        # `components` because a third-party binary shares none of a
        # component's declarative shape — see the agent spec's
        # components-vs-runtimes table.
        self._runtimes: dict[str, RuntimeSpec] = {}
        # v0.2 auth block. Persisted to disk under `auth:` in agent.yaml.
        # passphraseHash: Argon2id-PHC string (verifiable, not reversible)
        # masterSalt:     base64-encoded 16-byte salt used to derive the
        #                 master key from the passphrase via Argon2id raw.
        self._auth: dict[str, Any] = {}

    @property
    def path(self) -> Path:
        """Where this state lives. Companion driver configs are written
        beside it, under `drivers/`."""
        return self._path

    # ----- lifecycle --------------------------------------------------

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"{self._path} must be a YAML mapping at the root")
                merged = _config_defaults()
                for k, v in raw.items():
                    if k in _CONFIG_FIELDS_BY_KEY:
                        merged[k] = v
                self._config = merged
                comps_raw = raw.get("components") or []
                self._components = {}
                for entry in comps_raw:
                    parsed = ComponentEntry.model_validate(entry)
                    self._components[parsed.name] = parsed
                runtimes_raw = raw.get("runtimes") or []
                self._runtimes = {}
                for entry in runtimes_raw:
                    spec = RuntimeSpec.model_validate(entry)
                    self._runtimes[spec.name] = spec
                self._auth = dict(raw.get("auth") or {})
            else:
                self._config = _config_defaults()
                self._components = {}
                self._runtimes = {}
                self._auth = {}
                self._write_locked()

    # ----- config trio ------------------------------------------------

    def as_config_document(self) -> ConfigDocument:
        with self._lock:
            return ConfigDocument.model_validate(dict(self._config))

    def apply_config_patch(self, request: ConfigUpdateRequest) -> ConfigUpdateResult:
        applied: list[str] = []
        rejected: list[ConfigFieldError] = []

        patch: dict[str, Any] = request.model_dump()

        with self._lock:
            for key, new_value in patch.items():
                field = _CONFIG_FIELDS_BY_KEY.get(key)
                if field is None:
                    rejected.append(ConfigFieldError(key=key, message="unknown field"))
                    continue
                err = _validate(field, new_value)
                if err is not None:
                    rejected.append(ConfigFieldError(key=key, message=err))
                    continue
                if new_value is None and field.default is not None:
                    self._config[key] = field.default
                else:
                    self._config[key] = new_value
                applied.append(key)

            if applied:
                self._write_locked()

            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=False,
                pendingRestart=[],
            )

    def as_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            component="agent",
            fields=list(CONFIG_FIELDS),
            categories=CATEGORY_LABELS,
        )

    def get_config(self, key: str) -> Any:
        with self._lock:
            return self._config.get(key)

    # ----- topology ---------------------------------------------------
    #
    # `list_topology_entries` returns the declarative half — what the
    # operator wrote in `agent.yaml`. The routes layer combines this
    # with live state from the Supervisor (status, pid, lastRestart,
    # lastError) to produce the full `Component` view.

    def list_topology_entries(self) -> list[ComponentEntry]:
        with self._lock:
            return list(self._components.values())

    def get_topology_entry(self, name: str) -> ComponentEntry | None:
        with self._lock:
            return self._components.get(name)

    def add_topology_entry(self, entry: ComponentEntry) -> ComponentEntry:
        with self._lock:
            if entry.name in self._components:
                raise KeyError(f"component {entry.name!r} already exists")
            self._components[entry.name] = entry
            self._write_locked()
            return entry

    def update_topology_entry(self, name: str, entry: ComponentEntry) -> ComponentEntry | None:
        with self._lock:
            if name not in self._components:
                return None
            # Allow rename: replace under the new key, drop the old.
            if entry.name != name:
                if entry.name in self._components:
                    raise KeyError(f"component {entry.name!r} already exists")
                del self._components[name]
            self._components[entry.name] = entry
            self._write_locked()
            return entry

    def remove_topology_entry(self, name: str) -> bool:
        with self._lock:
            if name not in self._components:
                return False
            del self._components[name]
            self._write_locked()
            return True

    # ----- backwards-compatible Component composition -----------------
    #
    # Tests built against the skeleton's `list_components` etc. continue
    # to work — they just see `unreachable` status when no supervisor is
    # injected. Production code (routes layer) goes through the
    # Supervisor for live status.

    def list_components(self) -> list[Component]:
        return [_to_component(e) for e in self.list_topology_entries()]

    def get_component(self, name: str) -> Component | None:
        entry = self.get_topology_entry(name)
        return _to_component(entry) if entry is not None else None

    def add_component(self, entry: ComponentEntry) -> Component:
        return _to_component(self.add_topology_entry(entry))

    def update_component(self, name: str, entry: ComponentEntry) -> Component | None:
        updated = self.update_topology_entry(name, entry)
        return _to_component(updated) if updated is not None else None

    def remove_component(self, name: str) -> bool:
        return self.remove_topology_entry(name)

    # ----- runtimes (engine processes) --------------------------------

    def list_runtime_specs(self) -> list[RuntimeSpec]:
        with self._lock:
            return list(self._runtimes.values())

    def get_runtime_spec(self, name: str) -> RuntimeSpec | None:
        with self._lock:
            return self._runtimes.get(name)

    def add_runtime(self, spec: RuntimeSpec) -> RuntimeSpec:
        with self._lock:
            if spec.name in self._runtimes:
                raise KeyError(f"runtime {spec.name!r} already exists")
            resolved = self._resolve_ports_locked(spec)
            self._runtimes[resolved.name] = resolved
            self._write_locked()
            return resolved

    def update_runtime(self, name: str, spec: RuntimeSpec) -> RuntimeSpec | None:
        with self._lock:
            if name not in self._runtimes:
                return None
            if spec.name != name:
                if spec.name in self._runtimes:
                    raise KeyError(f"runtime {spec.name!r} already exists")
                del self._runtimes[name]
            resolved = self._resolve_ports_locked(spec)
            self._runtimes[resolved.name] = resolved
            self._write_locked()
            return resolved

    def remove_runtime(self, name: str) -> bool:
        with self._lock:
            if name not in self._runtimes:
                return False
            del self._runtimes[name]
            self._write_locked()
            return True

    def _component_ports_locked(self) -> set[int]:
        """Ports the components' URLs claim — companion drivers live in
        the same range as runtimes, so both have to be checked."""
        taken: set[int] = set()
        for entry in self._components.values():
            port = urlparse(str(entry.url)).port
            if port is not None:
                taken.add(port)
        return taken

    def allocate_component_port(self) -> int:
        """A free port for a component the agent declares itself — the
        companion driver. Same range as runtimes, checked against both,
        so a companion never lands on a port a later runtime gets."""
        with self._lock:
            taken = self._component_ports_locked() | {
                other.port for other in self._runtimes.values() if other.port is not None
            }
            for candidate in range(_RUNTIME_PORT_BASE, _RUNTIME_PORT_BASE + _RUNTIME_PORT_SPAN):
                if candidate not in taken:
                    return candidate
        raise ValueError(
            f"no free port in {_RUNTIME_PORT_BASE}-{_RUNTIME_PORT_BASE + _RUNTIME_PORT_SPAN - 1} "
            f"for a companion driver"
        )

    def _resolve_ports_locked(self, spec: RuntimeSpec) -> RuntimeSpec:
        """Assign a port when the operator did not pick one, and reject a
        collision when they did.

        Assignment happens here, at write time, rather than at spawn time
        so the port is *persisted*: an engine's port ends up in a driver's
        config, and a value that changed on every restart would be
        useless there. With N runtimes nobody should be handing out port
        numbers by hand. Component URL ports count as taken too, since
        companion drivers are allocated from the same range.
        """
        taken = {
            other.port
            for name, other in self._runtimes.items()
            if name != spec.name and other.port is not None
        } | self._component_ports_locked()
        if spec.port is not None:
            if spec.port in taken:
                raise ValueError(f"port {spec.port} is already claimed by another runtime")
            return spec

        for candidate in range(_RUNTIME_PORT_BASE, _RUNTIME_PORT_BASE + _RUNTIME_PORT_SPAN):
            if candidate not in taken:
                return spec.model_copy(update={"port": candidate})
        raise ValueError(
            f"no free port in {_RUNTIME_PORT_BASE}-"
            f"{_RUNTIME_PORT_BASE + _RUNTIME_PORT_SPAN - 1} for runtime {spec.name!r}"
        )

    # ----- auth (v0.2) -----------------------------------------------
    #
    # NOT exposed via /v1/config. These are internal trust-root state:
    # the passphrase hash + master-key salt. Login reads them; the
    # wizard sets them; nothing else touches them.

    def has_passphrase(self) -> bool:
        """True once the wizard has set a passphrase."""
        with self._lock:
            return bool(self._auth.get("passphraseHash"))

    def get_passphrase_hash(self) -> str | None:
        with self._lock:
            value = self._auth.get("passphraseHash")
            return value if isinstance(value, str) else None

    def get_master_salt_b64(self) -> str | None:
        with self._lock:
            value = self._auth.get("masterSalt")
            return value if isinstance(value, str) else None

    def set_passphrase(self, *, passphrase_hash: str, master_salt_b64: str) -> None:
        """Persist the wizard's passphrase hash + master-key salt.

        Idempotent — calling again overwrites prior values (used by
        the "change passphrase" flow in v0.3+; v0.2 only sets at
        first run).
        """
        if not passphrase_hash:
            raise ValueError("passphraseHash must not be empty")
        if not master_salt_b64:
            raise ValueError("masterSalt must not be empty")
        with self._lock:
            self._auth = {
                "passphraseHash": passphrase_hash,
                "masterSalt": master_salt_b64,
            }
            self._write_locked()

    # ----- internals --------------------------------------------------

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out: dict[str, Any] = dict(self._config)
        out["components"] = [
            entry.model_dump(exclude_none=True, mode="json") for entry in self._components.values()
        ]
        out["runtimes"] = [
            spec.model_dump(exclude_none=True, mode="json") for spec in self._runtimes.values()
        ]
        if self._auth:
            out["auth"] = dict(self._auth)
        with self._path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, sort_keys=True, default_flow_style=False)


def _to_component(entry: ComponentEntry) -> Component:
    """Compose a Component from a topology entry, with placeholder status.

    Used by the legacy `AgentState.list_components` etc. methods —
    when no Supervisor is available the status is hard-coded to
    `unreachable`. The routes layer overrides this with live state from
    the Supervisor when one is wired up."""
    return Component(
        name=entry.name,
        kind=ComponentKind(entry.kind.value),
        url=entry.url,
        spawn=entry.spawn,
        safeMode=entry.safeMode,
        status=ComponentStatus.unreachable,
    )


def _validate(field: ConfigField, value: Any) -> str | None:
    if value is None:
        return None
    vt = field.valueType
    if vt == ConfigValueType.boolean:
        if not isinstance(value, bool):
            return f"expected boolean, got {type(value).__name__}"
        return None
    if vt == ConfigValueType.enum:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        allowed = field.enumValues or []
        if value not in allowed:
            return f"must be one of {allowed}"
        return None
    if vt == ConfigValueType.file_path or vt == ConfigValueType.string:
        # Existence is deliberately not checked here: the path is read
        # at spawn and at discovery, both of which report a missing file
        # with the path named. Rejecting it at PATCH time would stop an
        # operator from pointing at an environment they are about to
        # create.
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        return None
    return f"unsupported valueType for agent config: {vt}"
