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

    safe_mode: bool = False
    """If true, skip loading the persistent state file at startup and run on
    built-in defaults — empty topology, default UI prefs. Provides a recovery
    path when agent.yaml itself is malformed. PATCH /v1/config still
    writes to the on-disk file normally."""


def load_settings() -> Settings:
    return Settings()
