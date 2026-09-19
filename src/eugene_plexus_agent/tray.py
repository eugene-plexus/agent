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
import time
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


#: How long `open_and_wait` gives the service to answer before opening
#: the browser anyway. A cold start loads no models -- it is the agent,
#: the control root, the gateway and the library -- so this is generous
#: for the slow case and invisible in the common one.
READY_TIMEOUT_SECONDS = 30.0


def wait_until_answering(port: int, timeout: float = READY_TIMEOUT_SECONDS) -> bool:
    """Poll `/healthz` until the agent answers, or the budget runs out.

    **`sc start` returning success is not the thing to wait for.** It
    means the Service Control Manager accepted the request; the agent
    still has to load its config, recover its key and bring up four
    children. Opening a browser at that moment shows *connection
    refused*, which reads as *Eugene is broken* rather than *Eugene is
    starting* -- and it is the person's first impression after clicking
    a Start menu entry.

    `urllib`, not `httpx`: this is one request at a time in a tiny GUI
    process, and R1.1's whole finding was the cost of building an
    `httpx` client -- 104 ms of certifi parsing -- for exactly this kind
    of one-shot call. A loopback `http://` URL builds no SSL context at
    all.
    """
    import urllib.error
    import urllib.request

    # Deliberately `perf_counter`, not `monotonic`: on the Python both
    # installers provision, Windows `monotonic()` sits on a 15.6 ms grid.
    deadline = time.perf_counter() + timeout
    url = f"http://127.0.0.1:{port}/healthz"
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


def open_and_wait(port: int = DEFAULT_PORT) -> tuple[bool, str]:
    """Make Eugene usable and show it. `(ok, why)`.

    **This is what the Start menu entry does**, and it exists because
    stopping Eugene takes the web UI with it -- so the obvious way back
    (*open the page*) is the one that cannot work. Start it if it is
    stopped, wait for it to answer, then open the browser.

    A person who stopped Eugene to play a game and then clicks
    "Eugene Plexus" has asked for it back. There is no confirmation,
    because there is no other reading of that click.
    """
    state = query_state()
    if state == ServiceState.stopped:
        started, why = start_service()
        if not started:
            return False, why
    if state in (ServiceState.stopped, ServiceState.unknown):
        # `unknown` covers START_PENDING and a service we could not
        # read; waiting costs at most the budget and answers both.
        wait_until_answering(port)
    open_ui(port)
    return True, ""


def claim_single_instance(name: str = "EugenePlexusTray") -> bool:
    """True if this process is the only tray icon in this session.

    **Session-local, deliberately not `Global\\`.** Two people signed in
    to one box each get their own icon, which is right: the icon belongs
    to a desktop, and the service it controls is shared.

    Without this, the Start menu entry -- whose whole job is to bring the
    icon back after somebody hid it -- would put a SECOND icon beside an
    existing one every time it was clicked. The handle is deliberately
    leaked: it must live as long as the process, and the OS releases it
    at exit.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
        kernel32.CreateMutexW(None, False, name)
        return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS
    except Exception:  # pragma: no cover - defensive
        # A mutex we could not take is not a reason to refuse to show an
        # icon. Two icons is a worse outcome than one, and no icon is
        # worse than two.
        return True


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
        # **Named for where it comes back from**, because this is the one
        # entry here that takes something away. Before the Start menu
        # entry existed it was a one-way door: hide the icon with Eugene
        # stopped and the only routes back were services.msc, an elevated
        # Start-Service, or signing out and in. The label is the fix's
        # visible half.
        (_ID_QUIT, "Hide this icon (it is in your Start menu)", True),
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
    # `--open`: start Eugene if it is stopped, wait for it, show it. What
    # the Start menu entry passes. `--no-icon`: do that and exit, for an
    # install that asked for no tray icon but still wants a way in.
    wants_open = "--open" in args
    wants_icon = "--no-icon" not in args
    if sys.platform != "win32":
        print(
            "The Eugene Plexus tray icon is Windows-only. On Linux and macOS the "
            "agent is a systemd unit or a launchd job; use those.",
            file=sys.stderr,
        )
        return 1
    # **The open action runs before the single-instance check**, because
    # an icon already sitting in the tray is exactly the case where
    # somebody clicked the Start menu entry to get Eugene back. Refusing
    # to act because an icon exists would make the entry do nothing in
    # the state it is most useful in.
    if wants_open:
        opened, why = open_and_wait(port)
        if not opened:
            print(why, file=sys.stderr)
            # Not exit 2: Eugene may be fine and only the START failed,
            # and there is nothing here for a person to correct.
            return 1

    if not wants_icon:
        return 0

    # A second icon beside the first is worse than no second process.
    if not claim_single_instance():
        return 0

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
