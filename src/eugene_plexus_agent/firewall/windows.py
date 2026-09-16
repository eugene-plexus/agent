r"""Windows Defender Firewall, read through `HNetCfg.FwPolicy2`.

COM rather than the `NetSecurity` cmdlets, and the reason is not taste.
Measured unelevated on a real Windows 11 host, 2026-09-15:

| read                                   | cmdlets | COM   |
| -------------------------------------- | ------- | ----- |
| profiles                               | 900 ms  |       |
| connection profiles (network category) | 156 ms  |       |
| every inbound rule + its port filter   | 720 ms* | 38 ms |
| application filters                    | 1078 ms |       |
| **total**                              | ~2.8 s  | 78 ms |

\* and it **failed**: `Get-NetFirewallPortFilter` raised *Access is
denied* unelevated and returned a truncated list on the way out, so a
cmdlet-based detector reports "nothing covers our port" from a partial
read. COM enumerated all 708 rules with no error. The cmdlet numbers
also exclude a PowerShell subprocess launch, which this agent would have
had to pay on every read.

Two shapes this module exists to get right, both measured rather than
reasoned about:

**A program rule is how the common Windows install is already allowed.**
The live worker on the measured host had no rule mentioning 8079 or
8080. It was reachable — the control root was probing it successfully —
through an *inbound allow for the program*, created by Windows' own
*Windows Security Alert* dialog the first time the agent listened off
loopback in an interactive session. A port-only detector would have
reported `blocked` on a machine that demonstrably was not. So a program
rule counts, and `FirewallPort.scope` reports which kind of rule
decided, because the program it names is a **versioned** interpreter
path (`...\pythons\cpython-3.12.14-windows-x86_64-none\python.exe`)
that changes when the interpreter is upgraded — an allow that silently
stops applying. A rule *we* add is bound to ports, for that reason.

**`NotConfigured` means block.** `Get-NetFirewallProfile` reports
`DefaultInboundAction: NotConfigured` on a stock machine; the COM
property returns `NET_FW_ACTION_BLOCK` (0) for the same state. Compare
the cmdlet's string to `"Block"` and a stock, blocking machine reads as
not blocking. This module never sees that string, and `DefaultInbound`
has a third member so the distinction survives into the contract.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

from .._generated.models import DefaultInbound, FirewallPort, HostFirewall, Scope, Verdict
from . import FirewallQuery, unsupported

log = logging.getLogger(__name__)

PRODUCT = "Windows Defender Firewall"

# NET_FW_PROFILE_TYPE2
_PROFILES: tuple[tuple[int, str], ...] = ((1, "Domain"), (2, "Private"), (4, "Public"))
# NET_FW_ACTION: block is 0, allow is 1. The one place the numbers matter.
_ACTION_BLOCK = 0
_ACTION_ALLOW = 1
# NET_FW_RULE_DIRECTION
_DIR_IN = 1
# NET_FW_IP_PROTOCOL
_PROTO_TCP = 6

# The display name every rule this agent adds carries, so `remove_rule`
# takes back exactly what `add_rule` put there and nothing a person
# created by hand.
RULE_NAME = "Eugene Plexus"


@dataclass(frozen=True)
class _Rule:
    """One inbound allow-or-block, reduced to what a verdict needs."""

    name: str
    action: int
    profiles: int
    ports: tuple[int, ...]
    """Empty when the rule does not name specific local TCP ports."""
    any_port: bool
    program: str | None
    protocol: int


def _policy() -> Any:
    """The firewall policy object, or None when COM is unavailable.

    `pywin32` arrives with the Windows `[service]` extra, which both
    installer paths take; an install without it reports `unknown` rather
    than guessing, which is the whole rule of this package.
    """
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return None
    # A FastAPI worker thread has not necessarily initialised COM. This
    # is idempotent and the matching uninitialise is deliberately not
    # called: the thread is pooled and re-used, and tearing COM down
    # under a pooled thread is how you get a hang.
    with contextlib.suppress(Exception):  # already initialised is fine
        pythoncom.CoInitialize()
    return win32com.client.Dispatch("HNetCfg.FwPolicy2")


def _parse_ports(spec: object) -> tuple[tuple[int, ...], bool]:
    """`"8079,8080"` or `"8000-8100"` or `"*"` -> the ports, and any-port.

    Ranges are expanded only when small. A rule saying `1-65535` is a
    rule saying "any", and enumerating it to answer a question about two
    ports would allocate 65k integers to reach the same verdict.
    """
    text = "" if spec is None else str(spec).strip()
    if not text or text == "*":
        return (), True
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                continue
            if hi - lo > 4096:
                return (), True
            out.extend(range(lo, hi + 1))
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return tuple(out), False


def _read_rules(policy: Any) -> list[_Rule]:
    """Every enabled inbound rule, reduced.

    Each rule is read inside its own `try`: the collection contains
    entries whose properties raise on access (a rule whose owning
    package is gone, typically), and one of those must not cost the
    whole enumeration.
    """
    rules: list[_Rule] = []
    for raw in policy.Rules:
        try:
            if not raw.Enabled or int(raw.Direction) != _DIR_IN:
                continue
            ports, any_port = _parse_ports(raw.LocalPorts)
            program = str(raw.ApplicationName) if raw.ApplicationName else None
            rules.append(
                _Rule(
                    name=str(raw.Name or ""),
                    action=int(raw.Action),
                    profiles=int(raw.Profiles),
                    ports=ports,
                    any_port=any_port,
                    program=program,
                    protocol=int(raw.Protocol) if raw.Protocol is not None else -1,
                )
            )
        except Exception:  # pragma: no cover - a rule we cannot read is one we skip
            continue
    return rules


def _covers_profiles(rule_profiles: int, active: int) -> bool:
    """Does this rule apply on every profile currently in force?

    Every one, not any one. A laptop that is Private at home and Public
    at a cafe is the case that makes "any" the wrong answer: a rule
    scoped to Private only would read as `allowed` while the machine sits
    on a Public network refusing everything. `NET_FW_PROFILE2_ALL` is
    0x7fffffff, which covers whatever the active mask holds.
    """
    if active == 0:
        return False
    return (rule_profiles & active) == active


def _matches_program(rule_program: str | None, program: str | None) -> bool:
    if not rule_program or not program:
        return False
    return os.path.normcase(os.path.normpath(rule_program)) == os.path.normcase(
        os.path.normpath(program)
    )


def _verdict_for_port(
    port: int,
    *,
    rules: list[_Rule],
    active: int,
    program: str | None,
    blocking: bool,
) -> FirewallPort:
    """One port's verdict, and the rule that decided it.

    Order matters and is the order Windows itself resolves in: an
    explicit **block** beats an allow. A block naming our program is the
    fingerprint of somebody clicking Cancel on the Security Alert dialog
    in a previous session, which is a real and otherwise invisible cause
    of "it worked on my other PC".
    """

    def _tcp(rule: _Rule) -> bool:
        return rule.protocol in (_PROTO_TCP, -1, 256)

    def _hits(rule: _Rule) -> Scope | None:
        if not _tcp(rule) or not _covers_profiles(rule.profiles, active):
            return None
        if port in rule.ports:
            return Scope.port
        if _matches_program(rule.program, program) and (rule.any_port or port in rule.ports):
            return Scope.program
        return None

    for want_action, verdict in (
        (_ACTION_BLOCK, Verdict.blocked),
        (_ACTION_ALLOW, Verdict.allowed),
    ):
        for rule in rules:
            if rule.action != want_action:
                continue
            scope = _hits(rule)
            if scope is None:
                continue
            return FirewallPort(
                port=port,
                verdict=verdict,
                rule=rule.name or None,
                scope=scope,
                profiles=_profile_names(rule.profiles),
                remedy=_remedy(port) if verdict is Verdict.blocked else None,
            )

    if not blocking:
        # Nothing matched and unmatched inbound traffic is not blocked.
        # Still not `allowed`: "the firewall is off" is one product's
        # answer and says nothing about a third party, which the caller
        # has already folded in before reaching here.
        return FirewallPort(port=port, verdict=Verdict.allowed)
    return FirewallPort(port=port, verdict=Verdict.blocked, remedy=_remedy(port))


def _profile_names(mask: int) -> list[str]:
    return [name for bit, name in _PROFILES if mask & bit]


def rule_command(ports: tuple[int, ...]) -> str:
    """The command that allows these ports, ready to paste.

    **Scoped to ports, not to the program**, which is the lesson of the
    rule already on the measured host: a program rule names an
    interpreter path inside the install directory and stops applying the
    day that interpreter is upgraded, with nothing anywhere saying so.

    `Private,Domain` and not `Public`: a network Windows has classified
    as Public is one the person is being told to treat as hostile, and
    `activeProfiles` on the card is what tells them to reclassify it
    instead.
    """
    joined = ",".join(str(p) for p in ports)
    return (
        f'New-NetFirewallRule -DisplayName "{RULE_NAME}" -Direction Inbound '
        f"-Protocol TCP -LocalPort {joined} -Action Allow -Profile Private,Domain"
    )


def _remedy(port: int) -> str:
    return rule_command((port,))


def _third_party() -> list[str]:
    """Firewalls registered with Security Center that are not Defender.

    Reading only Windows Firewall's own state is the trap this exists
    for: "Windows Firewall is off" with a third-party product enforcing
    reads as `allowed` to a detector that checks one and not the other.
    112 ms on the measured host, through WMI rather than COM.
    """
    try:
        import win32com.client

        locator = win32com.client.Dispatch("WbemScripting.SWbemLocator")
        service = locator.ConnectServer(".", r"root\SecurityCenter2")
        found = []
        for product in service.ExecQuery("SELECT displayName FROM FirewallProduct"):
            name = str(product.displayName or "").strip()
            if name and "windows" not in name.lower():
                found.append(name)
        return found
    except Exception:
        log.debug("could not query SecurityCenter2 for third-party firewalls", exc_info=True)
        return []


def read(query: FirewallQuery) -> HostFirewall:
    """Windows Defender Firewall's answer about `query.ports`."""
    policy = _policy()
    if policy is None:
        return unsupported(
            "This agent cannot read the Windows firewall: `pywin32` is not installed in its "
            "environment. Install it (`pip install eugene-plexus-agent[service]`) or check "
            "Windows Defender Firewall yourself.",
            query,
        )

    active = int(policy.CurrentProfileTypes)
    enabled_any = False
    blocking = False
    for bit, _name in _PROFILES:
        if not (active & bit):
            continue
        if policy.FirewallEnabled(bit):
            enabled_any = True
            if int(policy.DefaultInboundAction(bit)) == _ACTION_BLOCK:
                blocking = True

    third_party = _third_party()
    rules = _read_rules(policy)
    ports = [
        _verdict_for_port(p, rules=rules, active=active, program=query.program, blocking=blocking)
        for p in query.ports
    ]

    detail: str | None = None
    if third_party:
        # Not a refinement of the verdict — a replacement of it. Whatever
        # Defender says, another product is in the path and we cannot ask
        # it anything.
        ports = [
            FirewallPort(
                port=p.port,
                verdict=Verdict.unknown,
                rule=p.rule,
                scope=p.scope,
                profiles=p.profiles,
                remedy=p.remedy,
            )
            for p in ports
        ]
        detail = (
            f"{', '.join(third_party)} is also managing this machine's firewall, and nothing "
            "here can ask it whether Eugene is allowed through. Check it there, or try "
            "opening Eugene from another device."
        )
    elif not enabled_any:
        detail = "The Windows firewall is switched off for the network this machine is on."

    return HostFirewall(
        supported=True,
        product=PRODUCT,
        enabled=enabled_any,
        defaultInbound=DefaultInbound.block if blocking else DefaultInbound.allow,
        activeProfiles=_profile_names(active),
        thirdPartyProducts=third_party or None,
        ports=ports,
        detail=detail,
    )


def elevated() -> bool:
    """Is this process running with administrator rights?

    Decides whether a rule can be added in place or has to go through a
    prompt on the desktop.
    """
    if sys.platform != "win32":  # pragma: no cover - guarded by the caller
        return False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # pragma: no cover - defensive
        return False


# --------------------------------------------------------------------------- #
# changing it
# --------------------------------------------------------------------------- #


def add_rule(ports: tuple[int, ...], *, timeout: float = 30.0) -> tuple[bool, str]:
    """Allow `ports` inbound. Returns `(applied, detail)`.

    **The command run is the command printed.** `rule_command` produces
    one string, and both this and the card use it, so a person who is
    told "run this as administrator" runs exactly what we would have.

    Elevated, it is a subprocess and the answer is known when it
    returns. Unelevated, it is a `Start-Process -Verb RunAs`, which
    raises a UAC prompt on the desktop and returns immediately —
    `applied` is False there, not because it failed but because **we do
    not know yet**, and the caller's job is to re-read the verdict
    rather than to report a success nobody confirmed. An agent running
    as a service has no desktop for the prompt to appear on and does not
    need one: it is already elevated.
    """
    import subprocess

    command = rule_command(ports)
    if elevated():
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"Could not add the firewall rule: {exc}. Run this yourself: {command}"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            reason = tail[-1] if tail else f"exit code {proc.returncode}"
            return False, f"Windows refused the firewall rule: {reason}. Command: {command}"
        return True, f"Added the firewall rule {RULE_NAME!r} for {_ports_phrase(ports)}."

    escaped = command.replace("'", "''")
    launcher = (
        "Start-Process powershell -Verb RunAs -WindowStyle Hidden "
        f"-ArgumentList '-NoProfile','-NonInteractive','-Command','{escaped}'"
    )
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", launcher],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not ask for administrator rights: {exc}. Run this yourself: {command}"
    return False, (
        "Windows is asking for permission to add the firewall rule — say Yes to the prompt on "
        "this PC's desktop, then check again. If no prompt appeared, run this in an "
        f"administrator PowerShell: {command}"
    )


def remove_rule(*, timeout: float = 30.0) -> tuple[bool, str]:
    """Take back the rule this agent added, and only that one.

    Matched by display name, so a rule a person wrote themselves — with
    their own name, their own scope — survives turning the switch off.
    `-ErrorAction SilentlyContinue` because "there was no rule to
    remove" is the ordinary case on the way out, not a failure.
    """
    import subprocess

    command = f'Remove-NetFirewallRule -DisplayName "{RULE_NAME}" -ErrorAction SilentlyContinue'
    if not elevated():
        escaped = command.replace("'", "''")
        launcher = (
            "Start-Process powershell -Verb RunAs -WindowStyle Hidden "
            f"-ArgumentList '-NoProfile','-NonInteractive','-Command','{escaped}'"
        )
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", launcher],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return (
                False,
                f"Could not ask for administrator rights: {exc}. Run this yourself: {command}",
            )
        return False, (
            "Windows is asking for permission to remove the firewall rule — say Yes to the "
            f"prompt on this PC's desktop. Or run this as administrator: {command}"
        )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not remove the firewall rule: {exc}. Run this yourself: {command}"
    return True, f"Removed the firewall rule {RULE_NAME!r}."


def _ports_phrase(ports: tuple[int, ...]) -> str:
    return ", ".join(str(p) for p in ports)
