"""`ufw` and `firewalld`, read as far as a user-mode agent can read them.

Honest about a smaller answer than Windows gives, and the honesty is the
point. **Whether a firewall is enabled is readable without root; whether
our port is allowed usually is not** — `ufw status` and
`firewall-cmd --list-ports` both want root, and a detector that read
their refusal as "no rule found" would report `blocked` on a machine
that is allowing us. So the shape is:

- Neither present, or both inactive: `allowed`. Nothing is in the way,
  and a Linux box with no firewall really does let the connection in.
- One active and readable: the real verdict.
- One active and not readable: `unknown`, with the exact command to run.

An unprivileged read of `/etc/ufw/ufw.conf` gives `ENABLED=yes|no`, and
`systemctl is-active firewalld` answers without privileges. Those two
are what separate "no firewall" from "a firewall we cannot question",
which is the distinction that decides whether the card may say the
person is fine.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from .._generated.models import DefaultInbound, FirewallPort, HostFirewall, Scope, Verdict
from . import FirewallQuery

log = logging.getLogger(__name__)


def _euid() -> int:
    """POSIX-only symbol, reached through `getattr` so a Windows mypy run
    can still check this file."""
    getter = getattr(os, "geteuid", None)
    return int(getter()) if getter is not None else 0


_TIMEOUT = 5.0

UFW_CONF = Path("/etc/ufw/ufw.conf")


def _run(argv: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("%s failed: %s", argv[0], exc)
        return 127, ""
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _ufw_enabled() -> bool | None:
    """`ENABLED=yes` in ufw.conf, or None when we cannot tell.

    The config file rather than `ufw status`, because the file is
    world-readable on every distribution that ships ufw and the command
    is not.
    """
    try:
        text = UFW_CONF.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("ENABLED="):
            return line.split("=", 1)[1].strip().strip('"').lower() in ("yes", "true", "1")
    return None


def _ufw_allowed_ports() -> set[int] | None:
    """Ports ufw allows, or None when the status could not be read."""
    if shutil.which("ufw") is None:
        return None
    code, out = _run(["ufw", "status"])
    if code != 0 or "Status:" not in out:
        return None
    ports: set[int] = set()
    for line in out.splitlines():
        if "ALLOW" not in line.upper():
            continue
        head = line.split()[0] if line.split() else ""
        head = head.split("/")[0]
        if head.isdigit():
            ports.add(int(head))
    return ports


def _firewalld_active() -> bool:
    code, out = _run(["systemctl", "is-active", "firewalld"])
    return code == 0 and out.strip().startswith("active")


def _firewalld_allowed_ports() -> set[int] | None:
    if shutil.which("firewall-cmd") is None:
        return None
    code, out = _run(["firewall-cmd", "--list-ports"])
    if code != 0:
        return None
    ports: set[int] = set()
    for token in out.split():
        head = token.split("/")[0]
        if head.isdigit():
            ports.add(int(head))
    return ports


def read(query: FirewallQuery) -> HostFirewall:
    ufw_on = _ufw_enabled()
    firewalld_on = _firewalld_active()

    if not firewalld_on and ufw_on is not True:
        # Nothing enforcing. The one case on this platform where
        # `allowed` is an honest answer rather than an assumption:
        # `ufw.conf` says disabled or is absent, and firewalld is not
        # running.
        return HostFirewall(
            supported=True,
            product="ufw/firewalld",
            enabled=False,
            defaultInbound=DefaultInbound.allow,
            ports=[FirewallPort(port=p, verdict=Verdict.allowed) for p in query.ports],
            detail="No host firewall is running on this machine.",
        )

    product = "firewalld" if firewalld_on else "ufw"
    allowed = _firewalld_allowed_ports() if firewalld_on else _ufw_allowed_ports()

    if allowed is None:
        # Enabled, and it will not tell us. `unknown`, with the command
        # that answers it -- never `blocked`, which would be an eager
        # refusal about a machine that may well be fine.
        check = "sudo firewall-cmd --list-ports" if firewalld_on else "sudo ufw status"
        return HostFirewall(
            supported=True,
            product=product,
            enabled=True,
            defaultInbound=DefaultInbound.unknown,
            ports=[
                FirewallPort(port=p, verdict=Verdict.unknown, remedy=rule_command(product, (p,)))
                for p in query.ports
            ],
            detail=(
                f"{product} is running and this agent is not root, so it cannot read the "
                f"rules. Run `{check}` to see them."
            ),
        )

    ports = [
        FirewallPort(
            port=p,
            verdict=Verdict.allowed if p in allowed else Verdict.blocked,
            scope=Scope.port if p in allowed else None,
            remedy=None if p in allowed else rule_command(product, (p,)),
        )
        for p in query.ports
    ]
    return HostFirewall(
        supported=True,
        product=product,
        enabled=True,
        defaultInbound=DefaultInbound.block,
        ports=ports,
    )


def rule_command(product: str, ports: tuple[int, ...]) -> str:
    joined = " ".join(f"{p}/tcp" for p in ports)
    if product == "firewalld":
        args = " ".join(f"--add-port={p}/tcp" for p in ports)
        return f"sudo firewall-cmd --permanent {args} && sudo firewall-cmd --reload"
    return " && ".join(f"sudo ufw allow {p}/tcp" for p in ports) or f"sudo ufw allow {joined}"


def add_rule(ports: tuple[int, ...]) -> tuple[bool, str]:
    """Not done for the person, deliberately.

    Adding a rule here needs root, and the two ways to get it are a
    password prompt this agent has no terminal for and a sudoers entry
    granting a web server permanent root. Neither is worth a switch.
    The command is printed instead, which is what a Linux operator
    expects and what `easy-default-expert-override` asks for when the
    easy path is not safely available.
    """
    firewalld_on = _firewalld_active()
    product = "firewalld" if firewalld_on else "ufw"
    if not firewalld_on and _ufw_enabled() is not True:
        return True, "No host firewall is running, so there is nothing to allow."
    if _euid() != 0:
        return False, (
            "Adding a firewall rule needs root, and this agent is not running as root. "
            f"Run this yourself: {rule_command(product, ports)}"
        )
    code, out = _run(_argv_for(product, ports))
    if code != 0:
        return (
            False,
            f"The firewall refused: {out.strip().splitlines()[-1] if out.strip() else code}",
        )
    return True, f"Allowed {', '.join(str(p) for p in ports)} in {product}."


def remove_rule() -> tuple[bool, str]:
    """Nothing is removed automatically, for the reason above."""
    return False, (
        "This agent does not change firewall rules on Linux. Remove the rule you added, "
        "for example `sudo ufw delete allow 8079/tcp`."
    )


def _argv_for(product: str, ports: tuple[int, ...]) -> list[str]:
    if product == "firewalld":
        return ["firewall-cmd", "--permanent", *[f"--add-port={p}/tcp" for p in ports]]
    return ["ufw", "allow", f"{ports[0]}/tcp"]
