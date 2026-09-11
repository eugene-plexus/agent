"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py` and `topology.py`),
which is editable via `PATCH /v1/config` and the `/v1/components`
endpoints. These settings only control bootstrap: where to find the
state file, which interface to bind, and the safe-mode escape hatch.
"""

from __future__ import annotations

from pathlib import Path

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
