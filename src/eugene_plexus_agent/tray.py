"""A notification-area icon for turning Eugene off to play a game.

Troy, 2026-09-18: *"many users with RTX graphic cards will want to turn
Eugene off easily to play a video game, and then turn it back on to
infer again."*

## Why this is a separate process, and not a thread in the agent

**Session 0 isolation**, since Vista. A Windows service runs in session
0 and nothing it draws can appear on anybody's desktop, so a tray icon
was never going to live inside the agent once R2.6 made the agent a
service. It has to run in the signed-in person's own session.

Which is a pleasing result rather than an annoyance: the per-user logon
task that R2.6 takes *away* from the agent is exactly the right
mechanism for this. **The logon task does not disappear, it changes
job.** The agent boots without anyone; the icon exists only when
somebody is there to look at it, which is what an icon is for.

## Why it talks to the SCM and not to the agent's API

Stopping and starting a service needs no Eugene credential at all —
`install.ps1` grants the installing account start/stop rights on this
one service with `sc sdset`, so there is no UAC prompt per click and no
token to store. That matters more than it sounds: this is a long-lived
unattended process sitting in a user session, and the alternatives were
an operator session token on disk (the strongest credential this product
issues, held by a process anything in that session can read) or widening
what an `aud: client` key may do, three slices after R2.4 spent itself
narrowing exactly that.

So the menu is deliberately coarse. **Stop Eugene frees the graphics
card by stopping the engines with it**, which is the thing that was
asked for. A finer *"unload the models but keep serving"* would be
better on a machine somebody else is also using, and it needs a
credential this process should not have; see the design doc's open
questions rather than adding one here.

## What it never does

It does not install anything, does not elevate, does not restart the
agent on a schedule, and does not run at all off Windows. It is a
remote control, and a remote control that can brick the television is a
worse remote control.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import webbrowser

log = logging.getLogger(__name__)

SERVICE_NAME = "EugenePlexusAgent"
DEFAULT_PORT = 8079

#: `Shell_NotifyIcon` message ids. `WM_USER + 20` is the conventional
#: private id for a taskbar callback; nothing else in this process uses
#: the window, so there is nothing to collide with.
_WM_TRAYICON = 0x0400 + 20
_ID_OPEN = 1023
_ID_STOP = 1024
_ID_START = 1025
_ID_QUIT = 1026


class ServiceState:
    """The three answers the menu cares about."""

    running = "running"
    stopped = "stopped"
    unknown = "unknown"


def query_state(service_name: str = SERVICE_NAME) -> str:
    """Is the service running? `unknown` when we cannot tell.

    Read-only and unelevated: `QueryServiceStatus` needs
    `SERVICE_QUERY_STATUS`, which everyone has by default. `unknown` is
    a real answer and is shown as one — an icon that claims *stopped*
    for a service it could not read would have somebody clicking Start
    on something already running.
    """
    if sys.platform != "win32":
        return ServiceState.unknown
    try:
        import win32service
        import win32serviceutil

        status = win32serviceutil.QueryServiceStatus(service_name)
    except Exception:
        return ServiceState.unknown
    current = status[1]
    if current == win32service.SERVICE_RUNNING:
        return ServiceState.running
    if current == win32service.SERVICE_STOPPED:
        return ServiceState.stopped
    # START_PENDING / STOP_PENDING / PAUSED: in motion, and the honest
    # thing is to say we do not know rather than to pick an end state.
    return ServiceState.unknown


def _run_sc(*args: str) -> tuple[bool, str]:
    """`sc.exe`, and the reason it is not `win32serviceutil.StopService`.

    Both work. `sc` reports a refusal as an exit code and a sentence we
    can show, where the pywin32 call raises a `pywintypes.error` whose
    string is the least readable thing in this product. The one that
    ends up in front of a person wins.
    """
    try:
        done = subprocess.run(
            ["sc", *args],
            capture_output=True,
            text=True,
            timeout=30,
            # No console flash on a machine somebody is playing a game on.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)
    if done.returncode == 0:
        return True, ""
    text = (done.stdout or done.stderr or "").strip()
    if "5:" in text or "Access is denied" in text:
        # The one failure worth naming, because its fix is in the
        # installer and not in anything the person can click here.
        return False, (
            "Windows would not let this account stop the service. Re-run the "
            "Eugene installer from an elevated PowerShell to grant it."
        )
    return False, text or f"sc {args[0]} failed"


def stop_service(service_name: str = SERVICE_NAME) -> tuple[bool, str]:
    """Stop Eugene, which is what frees the graphics card."""
    return _run_sc("stop", service_name)


def start_service(service_name: str = SERVICE_NAME) -> tuple[bool, str]:
    return _run_sc("start", service_name)


def open_ui(port: int = DEFAULT_PORT) -> None:
    webbrowser.open(f"http://127.0.0.1:{port}/")


def menu_for(state: str) -> list[tuple[int, str, bool]]:
    """`(command id, label, enabled)` for the current state.

    Pure, so the whole of what a person is offered can be asserted
    without a message loop — the icon itself needs a desktop and a
    running service and is the part no test can reach.

    **Both actions are always present and one is greyed**, rather than
    swapping a single item between Stop and Start. A menu whose entries
    move under the cursor is a menu that gets misclicked, and the two
    mistakes here are *turning Eugene off in the middle of an answer*
    and *turning the graphics card back on in the middle of a game*.
    """
    running = state == ServiceState.running
    stopped = state == ServiceState.stopped
    return [
        (_ID_OPEN, "Open Eugene", True),
        (_ID_STOP, "Stop Eugene (frees the graphics card)", running),
        (_ID_START, "Start Eugene", stopped),
        (_ID_QUIT, "Hide this icon", True),
    ]


def tooltip_for(state: str) -> str:
    """What hovering says. Under 64 characters: `Shell_NotifyIcon`'s
    `szTip` is 128 wide and truncates without saying so, and a truncated
    status is a wrong status."""
    if state == ServiceState.running:
        return "Eugene is running"
    if state == ServiceState.stopped:
        return "Eugene is stopped - the graphics card is free"
    return "Eugene: cannot tell (is it installed as a service?)"


def main(argv: list[str] | None = None) -> int:
    """`eugene-plexus-tray`. Returns an exit code; never raises at a person.

    Off Windows this is a sentence rather than a traceback: the module
    is importable everywhere so the tests and CI (Linux) can cover
    everything above, and only the message loop is platform-bound.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    port = DEFAULT_PORT
    for index, value in enumerate(args):
        if value == "--port" and index + 1 < len(args):
            try:
                port = int(args[index + 1])
            except ValueError:
                print(f"--port needs a number, got {args[index + 1]!r}", file=sys.stderr)
                return 2
    if sys.platform != "win32":
        print(
            "The Eugene Plexus tray icon is Windows-only. On Linux and macOS the "
            "agent is a systemd unit or a launchd job; use those.",
            file=sys.stderr,
        )
        return 1
    try:
        from ._tray_window import run_message_loop
    except ImportError:
        print(
            "The tray icon needs pywin32, which is not installed in "
            f"{sys.executable}.\n    pip install 'eugene-plexus-agent[tray]'",
            file=sys.stderr,
        )
        return 1
    return run_message_loop(port=port)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
