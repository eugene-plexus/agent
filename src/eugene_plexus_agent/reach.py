"""Can anything else on the network get to this machine?

Three things have to be true, and every surface before this one reported
at most one of them while the symptom of all three is identical —
*connection refused*:

1. **Something is listening** on an address other than loopback.
2. **The node advertises** that address, so the install and the person's
   phone know to use it.
3. **The host firewall lets the connection in** (`firewall/`).

This module owns 1 and 2, and the restart question that falls out of 1.

**Everything here is read, not inferred.** `bound_addresses` reports the
interface each process was actually started on -- the value handed to
`bind()` -- and not what the config file says today. That distinction is
the entire value of the object: the case that matters is precisely the
one where the setting and the running process disagree, and a view that
re-derived itself from the setting could never show it.

---

**§0's measurement, and the thing it broke.** The advertise address the
agent has derived since M7 is the local end of a TCP connection *to the
control root* — which on a tailnet is exactly the interface the root can
reach back on, and which on a **standalone install is `127.0.0.1`**,
because on a standalone install the control root is on loopback. So the
plan's "propose the LAN address the agent already derives" could not
work for the person S5 is for. `proposed_host` is a second derivation
for that case: connect a UDP socket to a public address and read the
local end. No packet is sent — the kernel resolves a route and binds a
local end — so it costs half a millisecond, needs no network, and
answers on a machine that has never enrolled and never will.

**The agent's own socket cannot follow the setting.** A listening socket
is fixed for the life of the process. Supervised components are
respawned when the setting changes and pick up `BIND_HOST` from their
environment, so they are fine; this agent is not, and pretending
otherwise would be the silent failure the whole slice exists to remove.
`restart_required` reports the disagreement and `describe_restart` says
whether this agent can arrange its own restart, because a switch that
stops the agent with nothing to start it again takes the console with
it.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ._generated.models import AgentRestart, BoundAddress, Mechanism
from .node_identity import format_url, is_loopback_host

log = logging.getLogger(__name__)


def _euid() -> int:
    """This process's effective uid, and 0 on Windows, which has none.

    Through `getattr` because mypy on a Windows checkout does not believe
    `os.geteuid` exists -- it is a POSIX-only symbol, and every caller
    here is already inside a platform branch.
    """
    getter = getattr(os, "geteuid", None)
    return int(getter()) if getter is not None else 0


def _uid() -> int:
    getter = getattr(os, "getuid", None)
    return int(getter()) if getter is not None else 0


WINDOWS_SERVICE_NAME = "EugenePlexusAgent"
WINDOWS_TASK_NAME = "EugenePlexusAgent"
SYSTEMD_UNIT = "eugene-plexus-agent"
LAUNCHD_LABEL = "com.eugeneplexus.agent"


# --------------------------------------------------------------------------- #
# where this host is on its own network
# --------------------------------------------------------------------------- #


def proposed_host() -> str | None:
    """This host's address on the network it routes through, or None.

    A UDP socket `connect()`ed to a public address sends nothing: it
    makes the kernel pick a route and bind a local end, which is then
    read straight back off the socket. That is the address another
    device on the same network would use.

    Deliberately not the hostname, for the reason
    `derive_advertise_host` gives: it does not resolve across a tailnet
    unless MagicDNS happens to be on, and an address that works on some
    networks is a bug report waiting on the others.

    None on a machine with no route off itself, which is an honest
    answer and the reason `POST /v1/node/reach` can 400.
    """
    for family, probe in (
        (socket.AF_INET, ("192.0.2.1", 9)),
        (socket.AF_INET6, ("2001:db8::1", 9)),
    ):
        sock = socket.socket(family, socket.SOCK_DGRAM)
        try:
            sock.connect(probe)
            host = sock.getsockname()[0]
        except OSError:
            continue
        finally:
            sock.close()
        if host and not is_loopback_host(str(host)):
            return str(host)
    return None


def proposed_url(bind_port: int) -> str | None:
    """`proposed_host` as the URL this node would advertise."""
    host = proposed_host()
    return format_url(host, int(bind_port)) if host else None


# --------------------------------------------------------------------------- #
# what is actually listening
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Listener:
    """One process of this install, the port it serves, and the interface
    it was actually started on.

    `bind_host` is the value handed to `bind()` -- this agent's own
    uvicorn host, or the `<KIND>_BIND_HOST` the supervisor spawned a
    child with. `None` means no bind host was named and the component's
    own loopback default applied.
    """

    process: str
    port: int
    bind_host: str | None


def bound_addresses(listeners: list[Listener]) -> list[BoundAddress]:
    """What each process is listening on, from the bind rather than a probe.

    **A connect probe was written first and then thrown away**, and the
    measurement is worth keeping. Probing this host's own address to see
    whether a process is reachable there conflates two answers: on
    Windows a connection to a *closed* local port does not come back
    refused, it is dropped, so a probe costs a full timeout per closed
    port (measured: 362 ms for `127.0.0.1:8080` with nothing on it,
    against 6.6 ms for an open one) -- and worse, a connection to the
    host's own LAN address is evaluated by the host firewall, so a probe
    that failed could not say whether the bind was narrow or the
    firewall was shut. `restartRequired` hangs off this answer, and
    telling somebody to restart Eugene when the real problem is a
    firewall rule is the kind of confident wrong advice this slice
    exists to remove.

    The bind value is exact, free, and is the property being asked
    about. The firewall keeps its own object and its own verdict.
    """
    out: list[BoundAddress] = []
    for listener in listeners:
        host = (listener.bind_host or "").strip() or "127.0.0.1"
        out.append(
            BoundAddress(
                process=listener.process,
                host=host,
                port=listener.port,
                reachableOffHost=not is_loopback_host(host),
            )
        )
    return out


def restart_required(*, advertise_url: str | None, agent_bound: BoundAddress | None) -> bool:
    """Does this agent's own socket disagree with what the node advertises?

    True only when the node advertises a non-loopback address and this
    agent is not bound to one. The inverse -- reach turned off while the
    agent is still bound wide -- is deliberately **not** a restart: the
    components have come back on loopback, the address is no longer
    advertised, and forcing a restart to close one socket would cost the
    console for a change nobody outside can observe. The bind is
    reported either way, so an operator who wants it narrowed can see
    that it is not.
    """
    if is_loopback_host(_host_of(advertise_url)):
        return False
    return agent_bound is None or not bool(agent_bound.reachableOffHost)


def _host_of(url: str | None) -> str | None:
    if not url:
        return None
    from urllib.parse import urlparse

    return urlparse(url).hostname


# --------------------------------------------------------------------------- #
# how this agent would come back
# --------------------------------------------------------------------------- #


def describe_restart() -> AgentRestart:
    """What starts this agent, and whether it can ask for a restart.

    Detected from evidence about *this* process rather than from what an
    installer once did, because the two drift: a person can stop the
    service and run the agent in a terminal, and the switch must not then
    stop it believing the SCM will bring it back.
    """
    if sys.platform == "win32":
        return _windows_restart()
    if sys.platform == "darwin":
        return _launchd_restart()
    if sys.platform.startswith("linux"):
        return _systemd_restart()
    return AgentRestart(
        mechanism=Mechanism.unknown,
        canSelfRestart=False,
        detail=f"Nothing here knows how {platform.system() or sys.platform} starts this agent.",
    )


def _windows_restart() -> AgentRestart:
    if _running_as_windows_service():
        return AgentRestart(
            mechanism=Mechanism.service,
            canSelfRestart=True,
            command=f"Restart-Service {WINDOWS_SERVICE_NAME}",
        )
    if _windows_task_runs_this_install():
        return AgentRestart(
            mechanism=Mechanism.logon_task,
            canSelfRestart=True,
            command=(
                f'schtasks /End /TN "{WINDOWS_TASK_NAME}" && '
                f'schtasks /Run /TN "{WINDOWS_TASK_NAME}"'
            ),
            detail="This agent starts when you log in.",
        )
    return AgentRestart(
        mechanism=Mechanism.none,
        canSelfRestart=False,
        command="eugene-plexus-agent",
        detail=(
            "Nothing starts this agent automatically — it is running because somebody ran "
            "it. Stopping it would end the install until it is started again by hand."
        ),
    )


def _running_as_windows_service() -> bool:
    """Is this process the Windows service, rather than a task or a shell?

    **Session 0**, which is where the SCM runs every service and where
    nothing interactive ever runs. `ProcessIdToSessionId` comes straight
    off `kernel32`, so this needs no `pywin32` and works on the
    unelevated install too.

    The first version of this asked `GetConsoleWindow() == 0` -- a
    service has no console, which is true and is the property step 2's
    graceful-shutdown work turns on. It is not a *test* for one: this
    module was smoke-tested from a Git Bash shell, which uses a ConPTY
    with no classic console window, and the answer came back `service`
    on a box whose agent is a logon task. A property every service has
    is not a property only services have.
    """
    try:
        import ctypes
        from ctypes import wintypes

        session = wintypes.DWORD()
        ok = ctypes.windll.kernel32.ProcessIdToSessionId(
            wintypes.DWORD(os.getpid()), ctypes.byref(session)
        )
        return bool(ok) and session.value == 0
    except Exception:  # pragma: no cover - defensive
        return False


def _windows_task_runs_this_install() -> bool:
    r"""Is the logon task the thing that started *this* process?

    **The task existing is not the answer**, and the first version of
    this said it was. The acceptance run found it: a throwaway agent
    started from a shell, in a checkout's own virtualenv, on ports +100,
    reported `logon_task` / `canSelfRestart: true` -- because the live
    install on the same box owns a task by that name. Pressing the
    switch's restart there would have run `schtasks /End` against **the
    operator's real agent**, stopping the live install and starting it
    again while the throwaway kept running. A machine can hold two
    installs; every other part of this codebase already knows that
    (`keyring_store` scopes its entry by install, the acceptance scripts
    clear the ambient environment for the same reason).

    The discriminator is the task's program against this process's
    `sys.prefix`: the installer registers
    `<prefix>\Scripts\eugene-plexus-agent.exe`, so a task whose action
    lives inside the virtualenv this interpreter is running from is this
    install's task, and one that does not is somebody else's.

    Deliberately not `sys.executable`: in a uv-made virtualenv that is
    the *base* interpreter under `pythons\cpython-...`, which is outside
    the prefix and shared between installs.
    """
    try:
        proc = subprocess.run(
            ["schtasks", "/Query", "/TN", WINDOWS_TASK_NAME, "/FO", "LIST", "/V"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    return task_runs_from_prefix(proc.stdout or "", sys.prefix)


def task_runs_from_prefix(query_output: str, prefix: str) -> bool:
    """Does `schtasks /V /FO LIST` output name a program inside `prefix`?

    Split out so it can be tested against real output without a task
    existing, and without this test being the one that has to be right
    about `schtasks`' localised field names -- it looks for the path, not
    for the label in front of it.
    """
    wanted = os.path.normcase(os.path.normpath(prefix))
    for line in query_output.splitlines():
        _, _, value = line.partition(":")
        candidate = (value or line).strip().strip('"')
        if not candidate:
            continue
        # A "Task To Run" line is `<exe> <args>`; the exe is what matters
        # and an installed console script has no spaces in its path.
        head = candidate.split(" --")[0].strip().strip('"')
        try:
            normalised = os.path.normcase(os.path.normpath(head))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if normalised.startswith(wanted + os.sep):
            return True
    return False


def _systemd_restart() -> AgentRestart:
    """systemd sets `INVOCATION_ID` in every unit's environment.

    Present and non-empty is proof this process is a unit; it is not
    proof of *which* unit, so the command names the one the installer
    writes and the detail says so rather than pretending to have looked
    it up.
    """
    if not os.environ.get("INVOCATION_ID"):
        return AgentRestart(
            mechanism=Mechanism.none,
            canSelfRestart=False,
            command="eugene-plexus-agent",
            detail=(
                "This agent is not running under systemd — it is running because somebody "
                "started it."
            ),
        )
    user = "--user " if _euid() != 0 else ""
    return AgentRestart(
        mechanism=Mechanism.systemd,
        canSelfRestart=shutil.which("systemctl") is not None,
        command=f"systemctl {user}restart {SYSTEMD_UNIT}",
    )


def _launchd_restart() -> AgentRestart:
    # The plist existing is not proof this process is the thing it
    # starts -- the same mistake the Windows task test made, and the
    # acceptance run caught. A launchd-started process is reparented to
    # launchd itself, so its parent is pid 1.
    plist = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    if not plist.exists() or os.getppid() != 1:
        return AgentRestart(
            mechanism=Mechanism.none,
            canSelfRestart=False,
            command="eugene-plexus-agent",
            detail=(
                "Nothing starts this agent automatically on this Mac -- there is no launchd "
                "agent for Eugene, or this process was not started by one."
            ),
        )
    uid = _uid()
    return AgentRestart(
        mechanism=Mechanism.launchd,
        canSelfRestart=shutil.which("launchctl") is not None,
        command=f"launchctl kickstart -k gui/{uid}/{LAUNCHD_LABEL}",
    )


def restart_argv(restart: AgentRestart) -> list[str] | None:
    """The argv a detached helper runs to bring this agent back.

    **It asks this agent's own supervisor to restart it** rather than
    spawning a replacement, and that is the whole design. A detached
    copy would not be a child of the service or the task, so the next
    boot would start a second agent onto a port the orphan is holding —
    the stacking failure `ports.py` exists to diagnose, manufactured on
    purpose. Asking the supervisor leaves the process tree exactly as it
    was.

    None when there is no supervisor to ask, which is what
    `canSelfRestart: false` means.
    """
    if not restart.canSelfRestart:
        return None
    if restart.mechanism is Mechanism.service:
        # `sc stop` returns as soon as the SCM has the request, so the
        # start has to wait for the stop to finish; `cmd /c` sequences
        # them in the detached helper rather than here, where this
        # process is about to be the thing that stops.
        return [
            "cmd",
            "/c",
            f"sc stop {WINDOWS_SERVICE_NAME} & "
            f"timeout /t 3 /nobreak >nul & "
            f"sc start {WINDOWS_SERVICE_NAME}",
        ]
    if restart.mechanism is Mechanism.logon_task:
        return [
            "cmd",
            "/c",
            f'schtasks /End /TN "{WINDOWS_TASK_NAME}" & '
            f"timeout /t 3 /nobreak >nul & "
            f'schtasks /Run /TN "{WINDOWS_TASK_NAME}"',
        ]
    if restart.mechanism is Mechanism.systemd:
        user = ["--user"] if _euid() != 0 else []
        return ["systemctl", *user, "restart", SYSTEMD_UNIT]
    if restart.mechanism is Mechanism.launchd:
        return ["launchctl", "kickstart", "-k", f"gui/{_uid()}/{LAUNCHD_LABEL}"]
    return None


def spawn_restart(restart: AgentRestart) -> tuple[bool, str]:
    """Ask the supervisor to restart this agent, from a detached child.

    Detached because the command kills this process: a child in this
    process's group would be killed with it, half way through, leaving
    an install that is stopped and not started. Returns as soon as the
    helper is launched — the caller has a response to send before the
    connection goes away.
    """
    argv = restart_argv(restart)
    if argv is None:
        return False, (
            (restart.detail or "This agent cannot restart itself.")
            + (f" Restart it with: {restart.command}" if restart.command else "")
        )
    null = subprocess.DEVNULL
    try:
        if sys.platform == "win32":
            detached = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(argv, creationflags=detached, stdin=null, stdout=null, stderr=null)
        else:
            subprocess.Popen(argv, start_new_session=True, stdin=null, stdout=null, stderr=null)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, (
            f"Could not ask {restart.mechanism.value} to restart this agent: {exc}. "
            + (f"Restart it with: {restart.command}" if restart.command else "")
        )
    log.warning(
        "restarting this agent through %s so its own socket picks up the new address",
        restart.mechanism.value,
    )
    return True, (
        "Eugene is restarting so this machine's address takes effect. This page will "
        "reconnect in a few seconds."
    )


__all__ = [
    "Listener",
    "bound_addresses",
    "describe_restart",
    "proposed_host",
    "proposed_url",
    "restart_argv",
    "restart_required",
    "spawn_restart",
]
