"""Entrypoint: `python -m eugene_plexus_agent`."""

from __future__ import annotations

import logging

import uvicorn

from .app import create_app
from .console_logging import install_console_capture
from .settings import load_settings
from .state import AgentState


def main() -> None:
    settings = load_settings()

    # Mirror stdout/stderr to a rotating log file FIRST — before anything
    # else writes a line. Captures the agent's own uvicorn output,
    # every supervised child line (which we re-emit through `print()`),
    # and any library-level log calls. The file lives next to
    # agent.yaml so it's discoverable for bug reports without the
    # operator having to fish through env vars or task command flags.
    log_dir = settings.config_file.resolve().parent / "logs"
    log_path = install_console_capture(log_dir=log_dir)
    print(f"agent: console output is mirrored to {log_path}", flush=True)

    # uvicorn configures only its own loggers, so without this the agent's
    # own log calls never reach the console at all - every child component
    # already does this and the supervisor was the one that didn't, which
    # is why "declared the default topology" and "declared N companion
    # driver(s) at boot" were both invisible. force=True overrides any
    # basicConfig uvicorn already applied.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    bootstrap_state = AgentState(settings.config_file)
    if not settings.safe_mode:
        bootstrap_state.load()

    # 8079 unless EUGENE_PLEXUS_AGENT_BIND_PORT says otherwise. A
    # bootstrap setting rather than a config field: it has to be known
    # before there is a config to read, and it is what lets two agents
    # share one box.
    port = settings.bind_port
    log_level = "info"

    app = create_app(settings)
    uvicorn.run(app, host=settings.bind_host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
