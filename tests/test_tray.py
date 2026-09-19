"""The notification-area icon (R2.6).

Everything decidable is here; the message loop is in `_tray_window` and
needs a desktop, which is the line the split was drawn on.

Troy asked for this so an RTX owner can turn Eugene off to play a game
and back on afterwards. The thing that would make it useless is not a
crash — it is an icon that says *stopped* about a service that is
running, or a Stop that fails silently because the account was never
granted the right.
"""

from __future__ import annotations

import subprocess

import pytest

from eugene_plexus_agent import tray


def test_the_menu_greys_the_action_that_cannot_apply() -> None:
    running = dict((label, enabled) for _, label, enabled in tray.menu_for("running"))
    stopped = dict((label, enabled) for _, label, enabled in tray.menu_for("stopped"))

    stop = "Stop Eugene (frees the graphics card)"
    assert running[stop] is True
    assert running["Start Eugene"] is False
    assert stopped[stop] is False
    assert stopped["Start Eugene"] is True


def test_both_actions_are_always_present_so_nothing_moves_under_the_cursor() -> None:
    """A menu whose entries swap position is a menu that gets misclicked.

    The two mistakes available here are turning Eugene off in the middle
    of an answer and turning the graphics card back on in the middle of
    a game, so the layout is fixed and the wrong one is greyed.
    """
    labels = [
        [label for _, label, _ in tray.menu_for(state)]
        for state in ("running", "stopped", "unknown")
    ]
    assert labels[0] == labels[1] == labels[2]


def test_an_unreadable_service_offers_both_rather_than_guessing() -> None:
    """`unknown` is a real answer and is not rounded to `stopped`.

    Rounding it would put Start in front of somebody whose service is
    already running, which is the one click that does nothing and looks
    broken.
    """
    offered = {label: enabled for _, label, enabled in tray.menu_for("unknown")}
    assert offered["Start Eugene"] is False
    assert offered["Stop Eugene (frees the graphics card)"] is False
    assert offered["Open Eugene"] is True


def test_the_tooltip_says_the_graphics_card_is_free() -> None:
    """Which is the whole reason a person clicked Stop."""
    assert "graphics card is free" in tray.tooltip_for("stopped")
    assert "running" in tray.tooltip_for("running")
    assert "cannot tell" in tray.tooltip_for("unknown")


@pytest.mark.parametrize("state", ["running", "stopped", "unknown"])
def test_no_tooltip_is_long_enough_for_the_shell_to_truncate(state: str) -> None:
    """`szTip` is 128 wide and truncates without saying so.

    A truncated status is a wrong status, and the failure is invisible
    on the developer's machine because the strings only get long in
    other languages.
    """
    assert len(tray.tooltip_for(state)) < 64


def test_access_denied_names_the_fix_that_is_not_on_this_menu(monkeypatch) -> None:
    """The one failure whose remedy is in the installer.

    Without the `sc sdset` grant, Stop fails with error 5 and a person
    would reasonably conclude the button is broken. It is not: their
    account was never given the right, and nothing they can click here
    will give it to them.
    """

    class Done:
        returncode = 1
        stdout = "[SC] OpenService FAILED 5:\n\nAccess is denied.\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    ok, why = tray.stop_service()
    assert ok is False
    assert "elevated PowerShell" in why
    assert "installer" in why


def test_a_refusal_we_do_not_recognise_is_passed_through(monkeypatch) -> None:
    """Rather than replaced with a guess."""

    class Done:
        returncode = 1060
        stdout = "[SC] OpenService FAILED 1060: the service does not exist"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    ok, why = tray.stop_service()
    assert ok is False
    assert "1060" in why


def test_stopping_does_not_flash_a_console(monkeypatch) -> None:
    """On a machine somebody is about to play a game on.

    A black window appearing and vanishing when you click a tray item
    reads as *something went wrong*, every time.
    """
    seen: dict[str, object] = {}

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def record(*args, **kwargs):
        seen.update(kwargs)
        return Done()

    monkeypatch.setattr(subprocess, "run", record)
    tray.stop_service()
    assert seen.get("creationflags", 0) == getattr(subprocess, "CREATE_NO_WINDOW", 0)


def test_it_refuses_off_windows_with_a_sentence(capsys) -> None:
    """Importable everywhere so CI covers the rest of this file."""
    import sys

    if sys.platform == "win32":
        pytest.skip("this refusal only fires off Windows")
    assert tray.main([]) == 1
    assert "Windows-only" in capsys.readouterr().err


def test_a_bad_port_is_refused_before_anything_else(capsys) -> None:
    assert tray.main(["--port", "not-a-number"]) == 2
    assert "needs a number" in capsys.readouterr().err
