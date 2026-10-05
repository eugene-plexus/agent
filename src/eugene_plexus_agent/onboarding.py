"""Onboarding a machine: one question, asked three ways.

`eugene-plexus-agent` is the same entry point on every host, the control
root's included. On a boot that is genuinely fresh there is exactly one
question — **am I the root of a new install, or joining an existing
one?** — and everything else follows from it. Until M9 the only way to
answer "joining" was to know that
`EUGENE_PLEXUS_AGENT_DEFAULT_TOPOLOGY=0` existed, and **no test had ever
used it**: every multi-host acceptance script pre-wrote
`firstRunComplete: true` with an empty component list, which is the
bypass, written so fluently that nobody noticed it was standing in for a
product feature that did not exist.

Three ways to answer, one decision:

| Path | When |
|---|---|
| interactive prompt | a fresh boot **with a TTY** |
| `eugene-plexus-agent join` | scripted, provisioning, Ansible |
| `EUGENE_PLEXUS_AGENT_DEFAULT_TOPOLOGY=0` | a service unit or container that is a node |

**No TTY means today's behaviour: seed as root.** A service-managed or
containerised start has no stdin and must never block on a question.
Seeding is both the existing behaviour and right for the single-machine
case, which is overwhelmingly the common one; a node started by a service
unit is by definition being provisioned, and provisioning has `join` and
the env var. This is the one place the milestone keeps a silent default,
and it is the safe direction — a spurious control plane on a machine that
meant to be a node is recoverable, whereas an agent that hangs at boot
waiting for a terminal nobody is watching is not.

**Why the web wizard cannot be the only path.** A bootstrap paradox, not
a preference: you cannot reach a worker's web UI from your laptop until
that agent binds non-loopback; it binds non-loopback only when it
advertises a non-loopback address; and setting that address is part of
what joining does. The browser arrives after the thing it would
configure. Same argument that already settled first-boot seeding, and
stronger here, because a worker in another building is the case the whole
multi-host arc exists for.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from . import node_identity
from .default_topology import DEFAULTS, should_seed
from .enrollment import (
    TOKEN_REFUSED,
    EnrollmentError,
    EnrollmentOutcome,
    perform_enrollment,
    resolve_advertise_url,
)
from .settings import Settings
from .state import AgentState

log = logging.getLogger(__name__)

#: `join`'s exit code when the control root refused the token itself —
#: unknown, expired, withdrawn or already used. The installers read it:
#: every other failure can be retried with the same command, this one
#: needs a new token.
EXIT_TOKEN_REFUSED = 3


@dataclass(frozen=True)
class JoinRequest:
    control_url: str
    token: str
    name: str | None = None
    advertise_url: str | None = None
    force: bool = False
    root_key: str | None = None
    owner: tuple[str, str] | None = None
    """A Job Site's owner confirming at this machine: their sign-in name and
    password, read here and sent once, never kept (remote-nodes.md §3.3)."""


def has_tty() -> bool:
    """Both ends, because the question needs an answer as well as a place
    to be printed. A process with stdout attached and stdin closed would
    print a prompt into a log and then read EOF forever."""
    try:
        return sys.stdin is not None and sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):  # pragma: no cover - closed streams
        return False


def is_fresh_boot(settings: Settings) -> bool:
    """Would this boot declare a control plane if nobody said otherwise?

    Asks the same question `should_seed` asks, off the same state, so the
    prompt cannot appear on a boot that would not have seeded — which
    would be a question with no consequence. Identity is read straight
    off disk rather than through the app, because this runs before one
    exists.
    """
    if settings.safe_mode or not settings.default_topology:
        return False
    state = AgentState(settings.config_file)
    try:
        state.load()
    except Exception:  # pragma: no cover - a broken file is not a fresh boot
        return False
    identity = node_identity.NodeIdentityStore(
        settings.config_file.resolve().parent / node_identity.NODE_FILE
    )
    try:
        identity.load()
    except ValueError:
        return False
    return should_seed(state, enrolled=identity.record.enrolled)


def ask(settings: Settings) -> JoinRequest | None:
    """Ask the one question. Returns a `JoinRequest` to join, or None to
    be the root of a new install.

    Anything unexpected — EOF, Ctrl-C, an answer nobody can parse — falls
    through to `None`, because the fallback has to be the behaviour that
    works unattended.
    """
    print()
    print("  Eugene Plexus — first start on this machine.")
    print()
    print("  1) Start a NEW install here. This machine becomes the control root")
    print("     and runs the gateway and model library. Choose this if it is your")
    print("     first (or only) machine.")
    print()
    print("  2) JOIN an existing install as a worker node. This machine will run")
    print("     engines for a control root somewhere else. You will need a join")
    print("     token from that install.")
    print()
    try:
        answer = input("  New install, or join? [1] ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        log.info("no answer given; starting a new install on this machine")
        return None

    if answer not in ("2", "join", "j"):
        return None

    try:
        control_url = input("  Control root URL (e.g. http://100.64.0.1:8083): ").strip()
        token = input("  Join token: ").strip()
        name = input(f"  Name for this node [{_default_name()}]: ").strip()
        advertise = input("  Address other hosts reach this machine at [auto-detect]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        log.warning("join cancelled; starting a new install on this machine instead")
        return None

    if not control_url or not token:
        print()
        print("  A control root URL and a join token are both required. Mint one on")
        print("  the control root: Nodes -> Add a node.")
        print("  Starting a new install here instead; re-run with")
        print("  `eugene-plexus-agent join --control <url> --token <token>` to change that.")
        return None

    return JoinRequest(
        control_url=control_url,
        token=token,
        name=name or None,
        advertise_url=advertise or None,
    )


def run_join(request: JoinRequest, settings: Settings) -> int:
    """Enroll this host, write `node.yaml`, and return a process exit code.

    Runs with no app around it: nothing is spawned, no auth state exists,
    and there are no children to restart. That is the point — the next
    normal start finds an enrolled node, and `should_seed` declines to
    raise a rival control plane because `enrolled` is the condition it
    already tests.
    """
    state = AgentState(settings.config_file)
    try:
        state.load()
    except Exception as exc:
        print(f"error: could not read {settings.config_file}: {exc}", file=sys.stderr)
        return 2

    identity = node_identity.NodeIdentityStore(
        settings.config_file.resolve().parent / node_identity.NODE_FILE
    )
    try:
        identity.load()
    except ValueError as exc:
        print(f"error: could not read the node identity file: {exc}", file=sys.stderr)
        return 2

    declared = sorted(e.name for e in state.list_topology_entries())
    discard: list[str] = []
    if declared and not request.force:
        kept_because = _why_components_are_kept(state, identity, declared)
        if kept_because is not None:
            # Short lines: a console wraps at its own width, mid-word, and
            # "it has a pas / sphrase" is how the first version read.
            print(
                "error: this machine already runs an install of its own\n"
                f"       ({', '.join(declared)}).\n"
                f"       {kept_because}",
                file=sys.stderr,
            )
            print(
                "       Joining would leave that install's control root running beside the\n"
                "       one you are joining. Uninstall it first with the installer\n"
                "       (-Uninstall on Windows, --uninstall on Linux and macOS), then join\n"
                "       again. Pass --force only if these components belong to the install\n"
                "       you are joining.",
                file=sys.stderr,
            )
            return 2
        # Nobody ever set this one up, so it holds nothing a person put
        # there. Discarded only after the enrollment succeeds, so a bad
        # token leaves the machine exactly as it was.
        discard = declared

    try:
        outcome = asyncio.run(_join(request, settings, state, identity))
    except EnrollmentError as exc:
        if exc.slug == TOKEN_REFUSED:
            # Its own exit code, because it is the one failure where running
            # the same command again cannot work: the installer reads it
            # and says to make a new token, never to retry (2026-09-26).
            # The root's own words, wrapped: a URL and a reason on one line
            # ran past 80 columns and a console broke it mid-word.
            said = textwrap.fill(
                f"({exc.detail}).", width=80, initial_indent=" " * 7, subsequent_indent=" " * 8
            )
            print(
                "error: the control root did not accept this join token\n"
                f"{said}\n"
                "       A join token works once, and expires. Make a new one on the\n"
                "       control root's Nodes page (Add a node) and run the command it\n"
                "       gives you. Nothing on this machine changed.",
                file=sys.stderr,
            )
            return EXIT_TOKEN_REFUSED
        print(f"error: {exc.title.lower()}: {exc.detail}", file=sys.stderr)
        return 1

    for name in discard:
        state.remove_topology_entry(name)
    if discard:
        print(
            "removed the control plane this machine had started on its own ("
            + ", ".join(discard)
            + "): nobody had set it up, so it held nothing."
        )

    print(f"joined {request.control_url} as node {outcome.name!r} at epoch {outcome.epoch}.")
    print(f"identity written to {identity.path}.")
    if outcome.advertise_url:
        print(f"other hosts will reach this machine at {outcome.advertise_url}.")
    else:
        print(
            "warning: no address to advertise, so the control root has recorded this node\n"
            "         as unreachable. Set `advertiseUrl` in the agent config, or re-run\n"
            "         with --advertise http://<this-host>:<port>.",
            file=sys.stderr,
        )
    print("start the agent normally; it will not declare a control plane of its own.")
    return 0


_SEEDED = frozenset(component.name for component in DEFAULTS)


def _is_loopback_url(url: str | None) -> bool:
    import ipaddress
    from urllib.parse import urlsplit

    host = urlsplit(url or "").hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _why_components_are_kept(
    state: AgentState,
    identity: node_identity.NodeIdentityStore,
    declared: list[str],
) -> str | None:
    """Why the components declared here must survive a join, or None.

    **A machine that started once without being told to join** (a plain
    install, or the service coming up before anyone ran the join) seeds a
    control plane of its own. Nobody can have put anything into it: the
    console asks for a passphrase before any other page opens, and the
    passphrase is what this reads. So a join discards it instead of
    refusing. Troy, 2026-09-26: "Someone running a join cmd with a valid
    token should be assumed to know what they're doing."

    Anything a person could have made is kept, and the join refuses:
    a passphrase (the wizard ran), an enrollment (this is a node of some
    install already), a component the seed never declares, or a model
    declared to run.
    """
    # Enrollment first: it says the most. A machine joined to a control
    # root on loopback is the control host of an install of its own,
    # which is what the setup wizard makes (Amish_Station, 2026-09-26).
    record = identity.record
    if record.enrolled:
        if _is_loopback_url(record.control_url):
            return "It is that install's control host, with a trust root of its own."
        return f"It is already joined to {record.control_url}."
    if state.has_passphrase():
        return "It has been set up: a passphrase was chosen on it."
    extra = sorted(set(declared) - _SEEDED)
    if extra:
        return "It declares " + ", ".join(extra) + ", which only a person adds."
    if state.list_runtime_specs():
        return "It has models declared to run."
    return None


async def _join(
    request: JoinRequest,
    settings: Settings,
    state: AgentState,
    identity: node_identity.NodeIdentityStore,
) -> EnrollmentOutcome:
    from .engines.devices import detect_devices

    # A Job Site has no address: nothing connects to it, so none is derived.
    advertise = (
        None
        if request.owner is not None
        else await resolve_advertise_url(
            configured=request.advertise_url or state.get_config("advertiseUrl"),
            control_url=request.control_url,
            bind_port=int(settings.bind_port),
        )
    )
    snapshot = await asyncio.to_thread(detect_devices)
    return await perform_enrollment(
        store=identity,
        control_url=request.control_url,
        token=request.token,
        name=request.name,
        advertise_url=advertise,
        devices=[d.model_dump(exclude_none=True, mode="json") for d in snapshot.devices],
        root_key=request.root_key,
        owner=request.owner,
    )


def _default_name() -> str:
    import socket

    return socket.gethostname()


def config_dir(settings: Settings) -> Path:
    return settings.config_file.resolve().parent


__all__ = [
    "EXIT_TOKEN_REFUSED",
    "JoinRequest",
    "ask",
    "config_dir",
    "has_tty",
    "is_fresh_boot",
    "run_join",
]
