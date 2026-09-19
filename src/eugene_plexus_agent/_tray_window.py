"""The Win32 half of the tray icon: a hidden window and a message loop.

Split from `tray.py` so everything decidable — what the menu offers,
what the tooltip says, whether the service is running, what `sc` said
when it refused — is importable and testable on the Linux that CI runs
on. What is left here is the part that genuinely needs a desktop, and it
is deliberately the smallest thing that can work.

**`Shell_NotifyIcon` needs a window to send its callbacks to**, so there
is one, and it is never shown. It processes exactly four messages: the
icon's callback, the menu's command, an explicit destroy, and the
taskbar-created broadcast — the last because Explorer restarting is
routine and an icon that vanishes with it looks like the product
crashed.
"""

from __future__ import annotations

import logging

from . import tray

log = logging.getLogger(__name__)

_WM_TRAYICON = tray._WM_TRAYICON
_ID_OPEN = tray._ID_OPEN
_ID_STOP = tray._ID_STOP
_ID_START = tray._ID_START
_ID_QUIT = tray._ID_QUIT

#: How often the icon re-reads the service state, in milliseconds. Slow
#: on purpose: this is a person's own desktop, the state changes when
#: they change it, and `QueryServiceStatus` is a syscall we have no
#: business making four times a second.
_POLL_MS = 5000


def run_message_loop(*, port: int) -> int:  # pragma: no cover - needs a desktop
    import win32api
    import win32con
    import win32gui

    state = {"value": tray.query_state()}
    taskbar_created = win32gui.RegisterWindowMessage("TaskbarCreated")

    def notify(flags: int, message: int) -> None:
        try:
            win32gui.Shell_NotifyIcon(
                message,
                (
                    hwnd,
                    0,
                    flags,
                    _WM_TRAYICON,
                    icon,
                    tray.tooltip_for(state["value"]),
                ),
            )
        except Exception:
            # Explorer is restarting, or the shell is not ready. The
            # taskbar-created broadcast below is what recovers it; this
            # must not take the process down with it.
            log.debug("Shell_NotifyIcon refused", exc_info=True)

    def show_menu() -> None:
        state["value"] = tray.query_state()
        menu = win32gui.CreatePopupMenu()
        for command, label, enabled in tray.menu_for(state["value"]):
            flags = win32con.MF_STRING
            if not enabled:
                flags |= win32con.MF_GRAYED
            win32gui.AppendMenu(menu, flags, command, label)
        x, y = win32gui.GetCursorPos()
        # **`SetForegroundWindow` before, and a null post after.** Both
        # are the documented workaround for a menu that will not dismiss
        # when the person clicks elsewhere -- the one bug every tray icon
        # ships with once.
        win32gui.SetForegroundWindow(hwnd)
        win32gui.TrackPopupMenu(
            menu, win32con.TPM_LEFTALIGN | win32con.TPM_BOTTOMALIGN, x, y, 0, hwnd, None
        )
        win32gui.PostMessage(hwnd, win32con.WM_NULL, 0, 0)
        win32gui.DestroyMenu(menu)

    def on_command(command: int) -> None:
        if command == _ID_OPEN:
            tray.open_ui(port)
            return
        if command == _ID_QUIT:
            win32gui.DestroyWindow(hwnd)
            return
        if command == _ID_STOP:
            ok, why = tray.stop_service()
        elif command == _ID_START:
            ok, why = tray.start_service()
        else:
            return
        state["value"] = tray.query_state()
        notify(win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP, win32gui.NIM_MODIFY)
        if not ok:
            # A balloon rather than a dialog: the person is on their way
            # into a game and a modal box in front of it is worse than
            # the problem.
            win32gui.Shell_NotifyIcon(
                win32gui.NIM_MODIFY,
                (
                    hwnd,
                    0,
                    win32gui.NIF_INFO,
                    _WM_TRAYICON,
                    icon,
                    tray.tooltip_for(state["value"]),
                    200,
                    "Eugene",
                    why[:200],
                    win32gui.NIIF_WARNING,
                ),
            )

    def wndproc(hwnd_: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == _WM_TRAYICON:
            if lparam in (win32con.WM_RBUTTONUP, win32con.WM_LBUTTONUP):
                show_menu()
            return 0
        if msg == win32con.WM_COMMAND:
            on_command(win32api.LOWORD(wparam))
            return 0
        if msg == taskbar_created:
            notify(win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP, win32gui.NIM_ADD)
            return 0
        if msg == win32con.WM_TIMER:
            fresh = tray.query_state()
            if fresh != state["value"]:
                state["value"] = fresh
                notify(win32gui.NIF_TIP, win32gui.NIM_MODIFY)
            return 0
        if msg == win32con.WM_DESTROY:
            win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, (hwnd_, 0))
            win32gui.PostQuitMessage(0)
            return 0
        return int(win32gui.DefWindowProc(hwnd_, msg, wparam, lparam))

    wc = win32gui.WNDCLASS()
    wc.lpszClassName = "EugenePlexusTray"
    wc.lpfnWndProc = wndproc
    win32gui.RegisterClass(wc)
    hwnd = win32gui.CreateWindow(
        wc.lpszClassName, "Eugene Plexus", 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None
    )

    # The application icon, or the shell's generic one. A missing icon
    # file is not a reason to have no tray icon at all.
    icon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)

    notify(win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP, win32gui.NIM_ADD)
    win32gui.SetTimer(hwnd, 1, _POLL_MS, None)
    win32gui.PumpMessages()
    return 0
