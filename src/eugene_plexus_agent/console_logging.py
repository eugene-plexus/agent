"""Console capture: mirror agent stdout/stderr to a rotating log file.

The agent process is the single point where every supervised body
component's output flows through (via `supervisor.SupervisedProcess.
_pipe_child_output`), alongside the agent's own uvicorn logs. That
combined stream is by far the most useful diagnostic surface — but
in an interactive terminal it scrolls off, and operators can't share
it with bug reports unless they remembered to redirect output at
task-launch time. Asking operators to edit task commands or set env
vars violates the project's GUI-equality principle: anything an
operator might need belongs in the UI or it happens automatically.

This module installs a same-process tee: every write to stdout/stderr
forwards to the original stream (so the live terminal is unchanged)
AND to a rotating log file. The file copy has ANSI SGR escapes
stripped so it renders cleanly in any text editor or Slack paste.

Defaults are hardcoded and require no operator action:

  - Path: `<config_file dir>/logs/agent.log`
    (i.e. next to `agent.yaml`, wherever the operator chose to put it)
  - 10 MB per file, 5 backups (50 MB max disk per stream)

Tuning these would mean adding fields to the config_store + UI controls
— deferred to v0.3 along with the rest of the logging-config UX.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

# SGR (Select Graphic Rendition) escapes — the `\x1b[<n>;<n>m` family
# used by `supervisor._colorize_alerts` for red/yellow inline coloring.
# Stripped from the file copy so the .log is readable in any editor.
_ANSI_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")


class _TeeStream(io.TextIOBase):
    """Write-through mirror: forwards to a console stream verbatim AND
    to a line-oriented capture logger with ANSI codes stripped.

    Buffers incoming text until each newline so the file handler sees
    one record per line — important because `RotatingFileHandler`
    rolls over on emit() boundaries, not mid-line.
    """

    def __init__(self, console: TextIO, capture: logging.Logger) -> None:
        super().__init__()
        self._console = console
        self._capture = capture
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        # Console: forward immediately, untouched (preserves ANSI color).
        with contextlib.suppress(Exception):
            self._console.write(data)
        # File: split on newlines, emit one record per complete line.
        # Partial trailing text stays in `_buf` until the next newline.
        with self._lock:
            self._buf += data
            while True:
                nl = self._buf.find("\n")
                if nl < 0:
                    break
                line = self._buf[:nl]
                self._buf = self._buf[nl + 1 :]
                self._capture.info(_ANSI_SGR_RE.sub("", line))
        return len(data)

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self._console.flush()
        # File handlers flush per-emit; nothing else to do here. We
        # deliberately do NOT flush partial buffered text — a half-line
        # at exit time is rare and the next write would emit it anyway.

    def isatty(self) -> bool:
        # Tee'd output is not a TTY. Callers using `isatty()` to detect
        # color support get False; the supervisor's own color decision
        # uses NO_COLOR env-var instead and runs before we install
        # the tee, so live-terminal colors are unaffected.
        return False

    def fileno(self) -> int:
        return self._console.fileno()

    def writable(self) -> bool:
        return True


def install_console_capture(
    *,
    log_dir: Path,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> Path:
    """Replace sys.stdout/sys.stderr with tees that mirror to a rotating
    log file under `log_dir`. Returns the resolved log file path.

    Idempotent: a second call within the same process is a no-op so
    test fixtures that import the agent module repeatedly don't
    stack handlers. Call as early as possible in main() — anything
    that writes to stdout before this runs is missed in the file copy.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "agent.log"

    # Dedicated logger — not propagated to root, so unrelated library
    # calls don't accidentally leak into the file copy, and the file
    # copy doesn't double-echo to anything attached to root.
    capture = logging.getLogger("eugene_plexus_agent._console_capture")
    capture.propagate = False
    if any(isinstance(h, RotatingFileHandler) for h in capture.handlers):
        return log_path
    capture.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    # Raw passthrough — incoming lines are already formatted by uvicorn
    # / the supervisor reader / each component's own logger. Adding a
    # timestamp prefix here would double-stamp lines that have one.
    handler.setFormatter(logging.Formatter("%(message)s"))
    capture.addHandler(handler)

    # Use the unwrapped __stdout__ / __stderr__ as the console side, so
    # repeated calls don't recursively wrap a previous tee on stdout.
    # __stdout__/__stderr__ can be None (e.g. a no-console pythonw host);
    # fall back to the current streams so the tee still installs.
    console_out: TextIO = sys.__stdout__ or sys.stdout
    console_err: TextIO = sys.__stderr__ or sys.stderr
    sys.stdout = _TeeStream(console_out, capture)
    sys.stderr = _TeeStream(console_err, capture)
    return log_path
