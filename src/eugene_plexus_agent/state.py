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

import contextlib
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from . import model_copies, model_paths, share_credentials
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

log = logging.getLogger(__name__)

#: Where a config file that would not load is kept, so coming up on
#: defaults is never the same thing as losing the operator's topology.
#: One fixed name rather than a timestamped series: the next degraded
#: boot has the same file to preserve, and a directory that fills with
#: `agent.yaml.broken-17897...` is litter nobody reads.
UNREADABLE_SUFFIX = ".unreadable"

#: Signs, in the RAW TEXT of a file that would not parse, that this
#: install held a passphrase. The structured read is the thing that
#: failed, so this is deliberately a text scan.
#:
#: **`firstRunComplete` is deliberately NOT one of these.** It is also
#: the flag that means *skip onboarding*: every multi-host acceptance
#: script since M0 pre-writes it and then sets a passphrase, and an
#: enrolled node carries it with no passphrase of its own by design. Only
#: the auth keys themselves are evidence that a master salt existed, and
#: this predicate exists to refuse minting a second one.
#:
#: Truncation is why the scan finds them: `yaml.safe_dump` sorts keys, so
#: `auth` is written first and a half-finished file keeps it and loses
#: the tail — which is exactly the shape the old non-atomic write left
#: behind.
_AUTH_MARKERS = (
    re.compile(r"^\s+masterSalt\s*:", re.MULTILINE),
    re.compile(r"^\s+passphraseHash\s*:", re.MULTILINE),
)

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
        key="engineBinaryRoots",
        label="Trusted engine directories",
        description=(
            "Allow runtime binaries in these directories, in addition to managed builds, "
            "the engine found on PATH, and vllmBinary. Symlinks must resolve inside a trusted "
            "directory. Changes apply on the next launch."
        ),
        category="engines",
        valueType=ConfigValueType.path_list,
        default=[],
    ),
    ConfigField(
        key="allowUnrestrictedEngineLaunch",
        label="Allow unrestricted engine launch",
        description=(
            "Expert override: allow arbitrary runtime binary paths and raw extraArgs. "
            "These can execute code as the agent's OS user. Plexus credentials are still "
            "removed from the child environment. Changes apply on the next launch."
        ),
        category="engines",
        valueType=ConfigValueType.boolean,
        default=False,
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
    ConfigField(
        key="mlxBinary",
        label="MLX binary",
        description=(
            "Install-wide path to the `mlx_lm.server` console script inside "
            "the virtual environment where you installed mlx-lm — for example "
            "`/Users/you/eugene-mlx/bin/mlx_lm.server`. The dot is part of "
            "the script name, not a file extension. Like vLLM, mlx-lm is an "
            "engine this agent drives but does not install, so without this "
            "it is found only if it is on PATH or a `binary` is set on every "
            "runtime. Point at the console script, not at a Python "
            "interpreter or a venv directory. Apple silicon only: `pip "
            "install mlx-lm` will succeed elsewhere and install nothing that "
            "can run a model, because upstream marks its `mlx` dependency "
            "`platform_system == 'Darwin'`. A `binary` set on an individual "
            "runtime still wins over this. Read at the next spawn; no agent "
            "restart needed."
        ),
        category="engines",
        valueType=ConfigValueType.file_path,
    ),
    ConfigField(
        key="kevPython",
        label="Kev interpreter",
        description=(
            "Install-wide path to the Python interpreter inside the Kev "
            "checkout's own virtual environment — for example "
            "`/home/you/eugene-kev/.venv/bin/python`. Kev has no console "
            "script: the launch is `python -m kev.serve`, so the engine is "
            "an interpreter that can import it, which `uv sync --extra "
            "serve` in the checkout arranges. Deliberately never found on "
            "PATH — a bare `python` is not evidence of a Kev environment — "
            "so without this (or `binary` on the runtime) the engine reads "
            "as not installed. GET /v1/engines carries the pinned checkout "
            "recipe. Read at the next spawn; no agent restart needed."
        ),
        category="engines",
        valueType=ConfigValueType.file_path,
    ),
    ConfigField(
        key="advertiseUrl",
        label="Advertise address",
        description=(
            "The address at which OTHER HOSTS reach this agent — for example "
            "`http://100.64.0.7:8079` on a tailnet. Sent to the control root at "
            "enrollment as this node's URL, and stamped onto every component this "
            "agent spawns (same host, that component's port) so a gateway on "
            "another host can reach a companion driver here. Leave empty on a "
            "single-host install; when empty, enrollment derives it from the "
            "interface this agent used to reach the control root and GET /v1/node "
            "shows what it derived. Setting a host that is not loopback also makes "
            "spawned components bind 0.0.0.0 instead of loopback, because a "
            "component that must be reached from another host cannot bind only "
            "to this one; engines are never widened. Announced to the control "
            "root the moment you change it — no restart and no re-enrollment — "
            "and read again at the next spawn. Set it on the node that runs the "
            "gateway if other nodes' browsers cannot open it: a node whose "
            "registry entry is a loopback address can be reached by nothing. "
            "Give the full address including the port the OUTSIDE uses, which "
            "is not necessarily the port this agent binds — a container "
            "published on a different port is the common case."
        ),
        category="node",
        valueType=ConfigValueType.url,
    ),
    ConfigField(
        key="pathMappings",
        label="Library folder overrides",
        description=(
            "Only where THIS machine mounts a Library folder somewhere other than "
            "the folder's own mounts say. Leave this empty: a Library folder "
            "carries the mounts its nodes reach it by (Library -> Folders), one "
            "for Linux/macOS nodes and one for Windows nodes, and this machine "
            "inherits the one of its kind. Add a row here only for the odd box "
            "-- from `/models` (the folder, spelled exactly as the Library lists "
            "it) to `Z:\\models` (where this machine mounted that share). The "
            "rest of the path is carried over; the most specific rule wins; an "
            "override beats the folder's mount. Applied at every launch, so a "
            "change takes effect at the next start without re-declaring "
            "anything; `Runtime.localPath` shows what was opened. This only says "
            "where the share is; whether this machine also keeps its own copy of "
            "the models it runs is the Model storage setting below. Test checks "
            "the rule against the Library's real files before you save it."
        ),
        category="storage",
        valueType=ConfigValueType.path_mappings,
        default=[],
    ),
    ConfigField(
        key="shareCredentials",
        label="Logins for file servers",
        description=(
            "Only if a Library folder lives on a server that asks this machine "
            "to log in. One row per server -- `192.168.16.252`, not a whole "
            "path -- and it covers every folder on it, because Windows allows "
            "one login per server. Leave it empty if models open fine today. "
            "You will need a row once Eugene starts on its own at boot: before "
            "you sign in there is no such thing as 'the password you saved in "
            "File Explorer', and a server can refuse an anonymous visitor even "
            "when the folder itself has no password on it. Where the folder is "
            "mounted is the Library folder overrides setting above; this is who "
            "this machine says it is when it gets there. Test tries the login "
            "before you save it."
        ),
        category="storage",
        valueType=ConfigValueType.share_credentials,
        default=[],
    ),
    ConfigField(
        key="modelCopyEnabled",
        label="Keep a local copy of the models this machine runs",
        description=(
            "Copy each model this machine runs onto its own disk the first time "
            "it starts, and open the copy from then on. Reading a 25 GB model "
            "over a gigabit share takes about four minutes every single start; "
            "from a local SSD it takes seconds. The first start after you turn "
            "this on is no faster -- that is when the copying happens -- and "
            "every start after it is. Only the models this machine's own "
            "runtimes point at are copied, nothing else, ever: delete a runtime "
            "and its copy goes with it. The folder the models came from is "
            "never written to. Where the share itself is mounted is the Library "
            "folder overrides setting above."
        ),
        category="modelStorage",
        valueType=ConfigValueType.boolean,
        default=False,
    ),
    ConfigField(
        key="modelCopyDir",
        label="Where to keep them",
        description=(
            "A folder on this machine's own disk, on the fastest drive with room "
            "to spare. Copies are named exactly as the model is named, in the "
            "same folder structure, so what is here stays readable and useful "
            "with or without Eugene Plexus. Everything in this folder is ours to "
            "delete; do not point it at a folder that holds anything else."
        ),
        category="modelStorage",
        valueType=ConfigValueType.file_path,
    ),
    ConfigField(
        key="modelCopyMinFreeGb",
        label="Always leave at least this much free (GB)",
        description=(
            "A copy is skipped when making it would leave less than this much "
            "free space on that drive, and the model is read over the network "
            "as usual -- starting a model never fails because of this setting. "
            "If free space drops below this for any other reason, copies are "
            "deleted oldest first until it is back."
        ),
        category="modelStorage",
        valueType=ConfigValueType.integer,
        default=model_copies.DEFAULT_MIN_FREE_GB,
    ),
]
CATEGORY_LABELS: dict[str, str] = {
    "setup": "Setup",
    "security": "Security",
    "ui": "Appearance",
    "engines": "Engines",
    "node": "Node",
    "storage": "Library",
    "modelStorage": "Model storage",
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
        # Why this agent is running on defaults, when it is (R1.5,
        # review §6.1 #6). None on the ordinary path.
        self._degraded_reason: str | None = None
        # Whether the file we could not read carried auth keys. Decides
        # whether first-run setup is offered or refused.
        self._lost_passphrase = False

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

    def load_or_degrade(self) -> str | None:
        """Load, or come up on defaults and return why.

        **`degraded-mode-required`, applied at last to the file that
        rule's own component owns** (review §6.1 #6). `load()` raised
        into an uncaught lifespan, so a half-written `agent.yaml` was an
        install that would not boot and could not be repaired from the
        UI; the only escape was an environment variable a
        scheduled-task user cannot know or set, and the Windows task
        gives up after three restarts.

        Three things happen on a failure, and the pair of the last two
        is what makes coming up on defaults safe:

        * **The in-memory state is reset.** `load()` mutates as it
          parses, so a raise part-way leaves a half-built topology — and
          the next `PATCH /v1/config` would persist that, turning a
          damaged file into a damaged install.
        * **The file is preserved** beside itself, so the first repair
          write cannot be the thing that loses the operator's topology.
        * **Whether the file carried auth keys is remembered**, because
          they went with the rest and `POST /v1/auth/initialize` must
          refuse rather than mint a second master salt.

        Deliberately one method rather than a try/except at each of the
        two call sites: `app.py` and `__main__.py` both load, and a rule
        enforced by remembering to write the same four lines twice is a
        rule that will be half-applied.
        """
        try:
            self.load()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            raw_text = ""
            with contextlib.suppress(OSError):
                raw_text = self._path.read_text(encoding="utf-8", errors="replace")
            with self._lock:
                self._config = _config_defaults()
                self._components = {}
                self._runtimes = {}
                self._auth = {}
                self._degraded_reason = reason
                self._lost_passphrase = any(m.search(raw_text) for m in _AUTH_MARKERS)
            self._preserve_unreadable()
            log.error(
                "%s could not be read (%s); running on defaults with no topology. A copy is "
                "at %s. Repair it from Config in the web UI, or restore that copy and "
                "restart.",
                self._path,
                reason,
                self._path.with_suffix(self._path.suffix + UNREADABLE_SUFFIX),
            )
            return reason
        with self._lock:
            self._degraded_reason = None
            self._lost_passphrase = False
        return None

    def _preserve_unreadable(self) -> None:
        """Keep the file we could not read, beside itself.

        Best-effort and never raises: this runs while explaining a
        failure, and an exception here would turn one fault into two.
        A copy rather than a move, so an operator who fixes the original
        by hand is not surprised to find it gone.
        """
        target = self._path.with_suffix(self._path.suffix + UNREADABLE_SUFFIX)
        try:
            target.write_bytes(self._path.read_bytes())
        except OSError as exc:  # pragma: no cover - defensive
            log.warning("could not preserve %s as %s: %s", self._path, target, exc)

    @property
    def degraded_reason(self) -> str | None:
        """Why this agent is on defaults, or None. Reported by `/healthz`."""
        with self._lock:
            return self._degraded_reason

    def lost_its_passphrase(self) -> bool:
        """Did this install hold a passphrase that we can no longer read?

        True only in the narrow case that is actually reachable: the file
        would not load, and its raw text carries the auth keys. The one
        caller is the refusal in `POST /v1/auth/initialize`, where
        answering *no* here mints a second master salt and orphans
        every secret the first one sealed — every provider API key, and
        the install signing key on every enrolled node — with nothing
        saying that had happened.

        Deliberately narrower than "has this install been set up".
        `firstRunComplete` is not evidence: see `_AUTH_MARKERS`.
        """
        with self._lock:
            return self._lost_passphrase

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
        """Persist the whole file, atomically (review §6.1 #6).

        **A bare `open("w")` truncates before the first byte is
        written**, so anything failing between the truncate and the
        flush — a power cut, a full disk, a process killed by the
        supervisor's escalation deadline — left a zero-length or
        half-written `agent.yaml`, which until R1.5's other half was an
        install that would not boot at all. Temp + `fsync` +
        `os.replace` cannot do that: the target is only ever swapped for
        a file already complete on disk, and `os.replace` is atomic on
        both platforms within one directory.

        **The write frequency is higher than it looks**, which is what
        makes the window reachable rather than theoretical: every config
        PATCH and every companion declaration rewrites this file, and
        M6 declares one companion per runtime.

        `fsync` before the replace, not after: the ordering is what the
        durability depends on, and skipping it would leave the metadata
        rename ahead of the data on a crash.
        """
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

        # Same directory, so `os.replace` is a rename within one volume.
        # A temp file under the system temp dir would make it a copy,
        # which is exactly the non-atomic thing being removed here.
        tmp = self._path.with_suffix(self._path.suffix + f".tmp-{os.getpid()}")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                yaml.safe_dump(out, f, sort_keys=True, default_flow_style=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path)
        except BaseException:
            # Including `KeyboardInterrupt`/`SystemExit`: a half-written
            # temp file left behind is litter, and the reason this
            # method exists is that interruptions happen mid-write.
            tmp.unlink(missing_ok=True)
            raise


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
    if vt == ConfigValueType.path_list:
        if not isinstance(value, list) or any(
            not isinstance(path, str) or not path.strip() for path in value
        ):
            return "expected a list of non-empty directory paths"
        return None
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
    if vt in (ConfigValueType.file_path, ConfigValueType.string, ConfigValueType.url):
        # Existence is deliberately not checked here: the path is read
        # at spawn and at discovery, both of which report a missing file
        # with the path named. Rejecting it at PATCH time would stop an
        # operator from pointing at an environment they are about to
        # create.
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        return None
    if vt == ConfigValueType.integer:
        # `isinstance(True, int)` is True in Python, and a boolean here
        # is a client sending the wrong field rather than a number it
        # meant. Floats are refused rather than truncated for the same
        # reason: 49.5 GB of headroom is a value someone typed, and
        # silently storing 49 is worse than saying no.
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected a whole number, got {type(value).__name__}"
        if value < 0:
            return "must not be negative"
        return None
    if vt == ConfigValueType.path_mappings:
        # Shape only. Existence is checked by `POST /v1/config/test`, for
        # the reason `file_path` gives above: a share about to be mounted
        # is an environment the operator is about to create.
        return model_paths.validate_rules(value)
    if vt == ConfigValueType.share_credentials:
        # Shape only, same argument one step further: whether a server
        # accepts a login is a fact about a machine that may not be
        # switched on yet, and refusing to SAVE it would leave an
        # operator unable to prepare an install before the NAS arrives.
        return share_credentials.validate_entries(value)
    return f"unsupported valueType for agent config: {vt}"
