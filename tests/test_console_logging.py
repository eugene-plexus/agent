"""Tests for the stdout/stderr-to-rotating-file tee.

Verifies the contract pieces that matter:
  - Console side gets the line verbatim (ANSI intact for color).
  - File side gets the line with ANSI stripped.
  - Lines are split at newlines so the RotatingFileHandler can roll
    over on clean boundaries.
  - install is idempotent — repeated calls don't stack handlers.
"""

from __future__ import annotations

import io
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from eugene_plexus_agent.console_logging import (
    _ANSI_SGR_RE,
    _TeeStream,
    install_console_capture,
)


@pytest.fixture(autouse=True)
def _reset_capture_state() -> object:
    """`install_console_capture` mutates process-global state -- a handler
    on a named logger, and `sys.stdout`/`sys.stderr`.

    Without this the tests are order-dependent, and silently so: the
    function returns early when a handler is already attached, so a test
    that ran after the idempotency one got a path where it expected the
    degraded `None` and failed for a reason having nothing to do with its
    subject. Found exactly that way.
    """
    capture = logging.getLogger("eugene_plexus_agent._console_capture")
    saved_out, saved_err = sys.stdout, sys.stderr
    for h in list(capture.handlers):
        capture.removeHandler(h)
        h.close()
    yield
    for h in list(capture.handlers):
        capture.removeHandler(h)
        h.close()
    sys.stdout, sys.stderr = saved_out, saved_err


def test_tee_writes_console_verbatim_and_file_stripped(tmp_path: Path) -> None:
    """Console side keeps the SGR escapes (so terminals stay colored);
    file side has them stripped (so editors/Slack render cleanly)."""
    console = io.StringIO()
    log_path = tmp_path / "captured.log"
    capture = logging.getLogger("test_tee_writes_console_verbatim_and_file_stripped")
    capture.propagate = False
    capture.setLevel(logging.INFO)
    capture.handlers.clear()
    handler = RotatingFileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    capture.addHandler(handler)

    tee = _TeeStream(console, capture)
    tee.write("[orchestrator] \x1b[31mERROR\x1b[0m: something broke\n")
    handler.flush()

    # Console got the line with ANSI codes intact.
    console_text = console.getvalue()
    assert "\x1b[31mERROR\x1b[0m" in console_text
    assert console_text.endswith("\n")

    # File got the same line with ANSI codes stripped.
    file_text = log_path.read_text(encoding="utf-8")
    assert "ERROR" in file_text
    assert "\x1b[" not in file_text
    assert "[orchestrator] ERROR: something broke" in file_text


def test_tee_buffers_partial_lines_until_newline(tmp_path: Path) -> None:
    """A write without a trailing newline should NOT emit to the file
    yet — wait for the next write that completes the line. Otherwise
    multi-write log records would emit as several truncated rows."""
    console = io.StringIO()
    capture = logging.getLogger("test_tee_buffers_partial_lines_until_newline")
    capture.propagate = False
    capture.setLevel(logging.INFO)
    capture.handlers.clear()
    seen: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    capture.addHandler(_Capture())

    tee = _TeeStream(console, capture)
    tee.write("hello ")  # no newline yet
    assert seen == []
    tee.write("world\n")  # completes the line
    assert seen == ["hello world"]


def test_install_is_idempotent_within_a_process(tmp_path: Path) -> None:
    """A second `install_console_capture()` call must not stack a second
    RotatingFileHandler — tests that import the agent package
    multiple times shouldn't end up with N handlers writing N copies."""
    log_dir = tmp_path / "logs"
    first = install_console_capture(log_dir=log_dir)
    second = install_console_capture(log_dir=log_dir)
    assert first == second

    capture = logging.getLogger("eugene_plexus_agent._console_capture")
    rotating = [h for h in capture.handlers if isinstance(h, RotatingFileHandler)]
    assert len(rotating) == 1, (
        f"expected one RotatingFileHandler, got {len(rotating)} — "
        "install_console_capture is stacking handlers on re-entry"
    )


def test_ansi_regex_only_matches_sgr_sequences() -> None:
    """The strip regex must not eat unrelated content that happens to
    look like a `[` after an escape (we have no such content today,
    but the bound is worth pinning)."""
    assert _ANSI_SGR_RE.sub("", "\x1b[31mRED\x1b[0m") == "RED"
    assert _ANSI_SGR_RE.sub("", "\x1b[1;33;42mfancy\x1b[0m") == "fancy"
    # Bracketed text that isn't an SGR escape is left alone.
    assert _ANSI_SGR_RE.sub("", "[orchestrator] msg") == "[orchestrator] msg"


def test_an_unwritable_log_directory_does_not_stop_the_agent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A log file is a convenience, and the supervisor used to die for it.**

    `mkdir` and the handler were both unguarded, so an unwritable `logs/`
    raised `PermissionError` five frames deep inside `logging.handlers`
    and the agent never started. The traceback named neither the
    directory nor the reason, and `degraded-mode-required` -- which this
    project applies to bad config -- had never been applied to the
    agent's own conveniences.

    Found on a real Unraid install: a data directory created by an
    earlier container under one uid, mounted into a new one running as
    `--user 99:100`. `logs/` already existed, so `mkdir(exist_ok=True)`
    succeeded and opening the file inside it did not -- which is why this
    simulates the failure at the handler rather than by chmod-ing a
    directory, a thing Windows would not honour anyway.
    """
    import eugene_plexus_agent.console_logging as cl

    def refuse(*args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cl, "RotatingFileHandler", refuse)

    log_dir = tmp_path / "logs"
    result = cl.install_console_capture(log_dir=log_dir)

    assert result is None, "an unopenable log file must report itself, not raise"
    said = capsys.readouterr().out
    assert "console output only" in said
    assert str(log_dir) in said, "the message has to name the directory to be actionable"

    # And the console still works, which is the whole point of degrading.
    print("still alive")
    assert "still alive" in capsys.readouterr().out
