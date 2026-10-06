"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py` and `topology.py`),
which is editable via `PATCH /v1/config` and the `/v1/components`
endpoints. These settings only control bootstrap: where to find the
state file, which interface to bind, and the safe-mode escape hatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import PrivateAttr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EUGENE_PLEXUS_AGENT_",
        env_file=None,
        case_sensitive=False,
    )

    config_file: Path = Path("agent.yaml")
    """Where the persistent state lives — UI prefs, firstRunComplete, and the
    components topology. Single file by deliberate choice; per the OpenClaw
    lesson, mistakes in one component's config can't wedge the whole Plexus
    because each body component owns its own separate file. This file is
    only the agent's own state."""

    bind_host: str = "127.0.0.1"
    """Network interface to bind. Override to 0.0.0.0 for tailnet exposure."""

    bind_port: int = 8079
    """Port to bind. 8079 unless told otherwise, so the UI ships pointed at a
    known target. A setting rather than a literal from M7, because two agents
    sharing one box for a same-host acceptance run need two ports, and because
    every component this agent spawns is told where its agent is
    (`EUGENE_PLEXUS_<KIND>_AGENT_URL`) from this value rather than from a
    loopback:8079 assumption that was only ever wrong about the port."""

    safe_mode: bool = False
    """If true, skip loading the persistent state file at startup and run on
    built-in defaults — empty topology, default UI prefs. Provides a recovery
    path when agent.yaml itself is malformed. PATCH /v1/config still
    writes to the on-disk file normally."""

    passphrase_file: Path | None = None
    """Where `securityMode: passphrase_file` keeps the passphrase.

    Set by the Linux system install, where the agent runs as its own
    account and has no keyring (see `passphrase_file`). Startup-only
    bootstrap: the agent must know where to look before it is unlocked,
    which is before any config it could read is open."""

    entrypoint_config: Path | None = None
    """Opt-in container HTTPS entry point. Unset, `entrypoint.json` beside
    `agent.yaml` is read where the bundled proxy exists (Settings, Container
    access setup writes it). An invalid file stops the agent; a missing one
    falls back to the direct ports (`entrypoint.resolve`)."""
    entrypoint_binary: str = "caddy"
    entrypoint_confirm_seconds: int = 900
    """How long a configuration applied from Settings waits for an operator
    request through it before going back. Startup-only; tests shorten it."""
    _entrypoint_console_origin: str | None = PrivateAttr(default=None)
    _entrypoint_fallback: str | None = PrivateAttr(default=None)
    """Why the entry point named above is off, when its file did not exist."""
    _entrypoint_named: Path | None = PrivateAttr(default=None)
    """The file this agent reads, kept after a fallback clears `entrypoint_config`."""
    _entrypoint_nodes: bool = PrivateAttr(default=False)
    """The entry point serves the node hostname; otherwise the control root
    keeps its direct port, so enrolled machines need no change."""
    _entrypoint_nodes_origin: str | None = PrivateAttr(default=None)
    _entrypoint_nodes_probe: str | None = PrivateAttr(default=None)
    """Handed to the control root: its nodes name, whether that name answers
    any network for the node paths, and where to read the key it presents."""
    _entrypoint_console_direct: bool = PrivateAttr(default=False)
    """The console stays on this agent's own port; only Workbench is behind the
    entry point, so this agent keeps its direct bind."""
    _entrypoint_seen: bool = PrivateAttr(default=False)
    _entrypoint_prepared: bool = PrivateAttr(default=False)
    _entrypoint_ready: Any = PrivateAttr(default=None)
    """`entrypoint.prepare`'s answer, once per process."""

    @field_validator("passphrase_file", "entrypoint_config", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """A blank variable is an unset one, and says so.

        Unraid passes every template variable, blank ones included, and
        `Path("")` is `.`: a root with the variable left empty logged
        *"passphrase file . could not be read (Is a directory)"* instead of
        naming the variable that was not set (found on a live NAS,
        2026-09-28).
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    ui_dir: Path | None = None
    """Serve the web UI from this directory instead of the installed
    `eugene-plexus-ui` distribution.

    For development, where the assets are `ui/out` in a checkout being
    rebuilt every few seconds. Startup-only bootstrap, which is the
    sanctioned use of an env var — an install has nothing here it would
    want to change while running, and the supported way to have no UI is
    to not install the distribution.

    **Acceptance runs must leave this unset.** A run that points at a
    checkout tests the directory and never the wheel, which is this
    project's recurring failure shape: a check whose subject is not
    where it is looking."""

    default_topology: bool = True
    """On a first boot (no config file yet) and while unenrolled, declare the
    control root, gateway and library this install must have — see
    `default_topology.py` for why that is the agent's job and not a wizard's.

    Set `EUGENE_PLEXUS_AGENT_DEFAULT_TOPOLOGY=0` for the one case the
    conditions cannot detect: a fresh node that is about to *join* an existing
    install, which needs an empty topology because its control root lives
    elsewhere. Startup-only bootstrap, which is the sanctioned use of an env
    var; everything a running install can change stays in the config UI."""


def load_settings() -> Settings:
    return Settings()
