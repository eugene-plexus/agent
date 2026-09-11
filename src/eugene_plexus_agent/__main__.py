"""Entrypoint: `python -m eugene_plexus_agent`, or `eugene-plexus-agent`.

Two things happen here that are not "start a server", and both exist
because **the same entry point runs on every machine in an install** —
the control root's host and every worker. A worker cannot be onboarded
from a browser: you cannot reach its web UI until it binds non-loopback,
it binds non-loopback only when it advertises a non-loopback address, and
setting that address is part of what joining does. So the machine has to
ask, or be told on the command line.

    eugene-plexus-agent                 start (asks once, on a fresh boot with a TTY)
    eugene-plexus-agent join --control <url> --token <jwt>

See `onboarding.py` for why a missing TTY means "start a new install"
rather than "wait".
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .app import create_app
from .console_logging import install_console_capture
from .onboarding import JoinRequest, ask, has_tty, is_fresh_boot, run_join
from .settings import Settings, load_settings
from .state import AgentState


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eugene-plexus-agent",
        description=(
            "Per-host node agent: supervises components and engine runtimes, and "
            "joins this machine to an Eugene Plexus install."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    join = sub.add_parser(
        "join",
        help="Enroll this machine as a worker node in an existing install, then exit.",
        description=(
            "Enroll this machine with a control root and write node.yaml, without "
            "starting the agent. The next normal start finds an enrolled node and "
            "does not declare a control plane of its own. This is the scripted "
            "answer to the same question the first-boot prompt asks."
        ),
    )
    join.add_argument("--control", required=True, metavar="URL", help="Control root URL.")
    join.add_argument(
        "--token",
        required=True,
        metavar="JWT",
        help="A join token minted at that control root (Nodes -> Add a node).",
    )
    join.add_argument(
        "--name",
        metavar="NAME",
        help="Name for this node in the install. Defaults to the hostname.",
    )
    join.add_argument(
        "--advertise",
        metavar="URL",
        help=(
            "Address other hosts reach this machine at. Derived from the route to the "
            "control root when omitted, which is right on a mesh VPN and worth setting "
            "by hand when it is not."
        ),
    )
    join.add_argument(
        "--force",
        action="store_true",
        help="Join even though components are already declared here (see the refusal's text).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv if argv is not None else sys.argv[1:])
    settings = load_settings()

    if args.command == "join":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
        raise SystemExit(
            run_join(
                JoinRequest(
                    control_url=args.control,
                    token=args.token,
                    name=args.name,
                    advertise_url=args.advertise,
                    force=args.force,
                ),
                settings,
            )
        )

    _serve(settings)


def build_server(settings: Settings) -> uvicorn.Server:
    """Everything `_serve` does except block on the socket.

    Split out at install-paths §9 step 3 so a Windows service can own the
    run loop. `uvicorn.run()` builds a `Server` and calls `.run()` on it,
    which installs SIGINT/SIGTERM handlers — a service has neither, and
    stops by having its control handler set `should_exit` on the server
    object instead. That is the only reason this function exists, and it
    is why the service still gets the *agent's* own lifespan shutdown
    even where its children get a hard kill (see `process_signals`).
    """
    # **The onboarding question, asked before anything is written.** Only
    # on a boot that would otherwise declare a control plane, and only
    # with a TTY to ask into — a service unit or container has neither a
    # terminal nor anyone watching one, and an agent that blocks at boot
    # waiting for an answer is worse than one that picks the common case.
    if is_fresh_boot(settings) and has_tty():
        request = ask(settings)
        if request is not None:
            code = run_join(request, settings)
            if code != 0:
                raise SystemExit(code)

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
    config = uvicorn.Config(app, host=settings.bind_host, port=port, log_level=log_level)
    return uvicorn.Server(config)


def _serve(settings: Settings) -> None:
    build_server(settings).run()


if __name__ == "__main__":
    main()
