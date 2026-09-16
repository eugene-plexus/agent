"""macOS's application firewall, through `socketfilterfw`.

Smaller again than Linux's, and for a structural reason worth stating:
**macOS's built-in firewall filters by application, not by port.** There
is no port to allow, so `FirewallPort.scope` here is always `program`
when a verdict exists at all, and the remedy names the interpreter
rather than 8079.

Its global state is readable without privileges
(`socketfilterfw --getglobalstate`), and so is whether a given binary is
listed (`--getappblocked`). Adding one is not: `--add` needs root, and a
launchd *agent* runs as the person, so the command is printed.

The failure this exists to name is the same one Windows has and Linux
does not: macOS prompts on the desktop the first time a program listens
off loopback, and a launchd agent with nobody watching can be denied
silently — the *"Do you want the application to accept incoming network
connections?"* dialog answered by nobody, or by a Deny somebody clicked
once and forgot.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

from .._generated.models import DefaultInbound, FirewallPort, HostFirewall, Scope, Verdict
from . import FirewallQuery, unsupported

log = logging.getLogger(__name__)


def _euid() -> int:
    """POSIX-only symbol, reached through `getattr` so a Windows mypy run
    can still check this file."""
    getter = getattr(os, "geteuid", None)
    return int(getter()) if getter is not None else 0


_TIMEOUT = 5.0
SOCKETFILTERFW = "/usr/libexec/ApplicationFirewall/socketfilterfw"
PRODUCT = "Application Firewall"


def _run(args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [SOCKETFILTERFW, *args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("socketfilterfw %s failed: %s", args, exc)
        return 127, ""
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def read(query: FirewallQuery) -> HostFirewall:
    if not os.path.exists(SOCKETFILTERFW) and shutil.which("socketfilterfw") is None:
        return unsupported(
            "This machine has no `socketfilterfw`, so nothing here can read the firewall.",
            query,
        )

    code, out = _run(["--getglobalstate"])
    if code != 0:
        return unsupported(
            "The macOS firewall's state could not be read. Check System Settings -> "
            "Network -> Firewall.",
            query,
        )
    enabled = "enabled" in out.lower() and "disabled" not in out.lower()

    if not enabled:
        return HostFirewall(
            supported=True,
            product=PRODUCT,
            enabled=False,
            defaultInbound=DefaultInbound.allow,
            ports=[FirewallPort(port=p, verdict=Verdict.allowed) for p in query.ports],
            detail="The macOS firewall is switched off, so nothing here is blocking Eugene.",
        )

    program = query.program
    verdict = Verdict.unknown
    detail: str | None = None
    if program:
        code, out = _run(["--getappblocked", program])
        low = out.lower()
        if code == 0 and "incoming connections" in low:
            # "…is set to allow incoming connections" / "…block incoming
            # connections". Read the sentence rather than the exit code:
            # a program the firewall has never heard of also exits 0.
            if "allow incoming" in low:
                verdict = Verdict.allowed
            elif "block incoming" in low:
                verdict = Verdict.blocked
        if verdict is Verdict.unknown:
            detail = (
                "The firewall is on and has no entry for Eugene yet. macOS asks the first "
                "time it listens for connections from other devices — if nobody is at this "
                "Mac to answer, the answer is no."
            )

    remedy = rule_command(program) if verdict is not Verdict.allowed else None
    return HostFirewall(
        supported=True,
        product=PRODUCT,
        enabled=True,
        defaultInbound=DefaultInbound.block,
        ports=[
            FirewallPort(
                port=p,
                verdict=verdict,
                scope=Scope.program if verdict is not Verdict.unknown else None,
                remedy=remedy,
            )
            for p in query.ports
        ],
        detail=detail,
    )


def rule_command(program: str | None) -> str:
    """macOS allows a **program**, so this names the interpreter.

    The one place the "scope to ports, not the program" rule cannot be
    followed: there is no port-scoped alternative on this platform.
    Which is also why a macOS install's allow survives a Python upgrade
    no better than a Windows dialog's does — the path changes and the
    entry stops applying.
    """
    target = program or "/path/to/python"
    return (
        f'sudo {SOCKETFILTERFW} --add "{target}" && sudo {SOCKETFILTERFW} --unblockapp "{target}"'
    )


def add_rule(program: str | None) -> tuple[bool, str]:
    """Printed, not run, for the same reason as Linux: it needs root."""
    if _euid() != 0:
        return False, (
            "Allowing Eugene through the macOS firewall needs administrator rights, and "
            f"this agent does not have them. Run this yourself: {rule_command(program)}"
        )
    target = program or ""
    if not target:
        return False, "This agent could not work out which program to allow."
    _run(["--add", target])
    code, out = _run(["--unblockapp", target])
    if code != 0:
        return False, f"The firewall refused: {out.strip()}"
    return True, "Allowed Eugene through the macOS firewall."


def remove_rule(program: str | None) -> tuple[bool, str]:
    if _euid() != 0:
        return False, (
            "Removing the firewall entry needs administrator rights. Run this yourself: "
            f'sudo {SOCKETFILTERFW} --remove "{program or ""}"'
        )
    code, out = _run(["--remove", program or ""])
    if code != 0:
        return False, f"The firewall refused: {out.strip()}"
    return True, "Removed Eugene's macOS firewall entry."
