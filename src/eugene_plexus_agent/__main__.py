"""Entrypoint: `python -m eugene_plexus_agent`."""

from __future__ import annotations

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

    bootstrap_state = AgentState(settings.config_file)
    if not settings.safe_mode:
        bootstrap_state.load()

    # Agent port is fixed at 8079 in v0.1 so the UI ships with a
    # known target. If a future release makes it configurable the
    # value will move into agent.yaml — the pattern would mirror
    # how the body components handle their own bind ports.
    port = 8079
    log_level = "info"

    app = create_app(settings)
    uvicorn.run(app, host=settings.bind_host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
