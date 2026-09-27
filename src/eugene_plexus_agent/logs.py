"""This machine's log, readable from any console (2026-09-27).

Troy: *"Like a Docker environment, I should be able to read all node logs
from the UI."* The agent already writes one combined stream per machine --
its own lines and every child it supervises, each child line prefixed by
the supervisor (`[gateway] `, `[engine: qwen] `) -- to `logs/agent.log`,
rotated 10 MB x 5 (`console_logging`). What was missing is any way to read
it that is not a shell on that machine, and on a Windows service install
the file is readable only by SYSTEM and Administrators.

Four pieces live here:

* **The line format.** Every line is stamped at receipt, in UTC, and
  tagged with its source: `2026-09-27T19:51:54.123Z [engine: qwen] text`
  -- `docker logs -t`. The agent's own lines are tagged `[agent]` and lose
  the local-time `asctime` logging gave them, which the stamp replaces;
  engine output had no time at all. Lines written before this format are
  still read: their source from the prefix, their time from an agent
  `asctime` when there is one, else none.
* **Reading history**: newest last, across the rotated files, filtered by
  source, time and text, scanned backwards so a `tail` of 500 reads the
  end of one file rather than all 60 MB.
* **Following**: a process-wide bus the tee publishes each stamped line to,
  so `GET /v1/logs/stream` can hand lines out as they are written.
* **Masking**: tokens and API keys are replaced on the way out. A line
  that prints one is a defect to fix where it is written; this is the
  insurance for the one nobody has found yet.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# The stamped form: time, the source in brackets, the text.
_STAMPED = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z) \[([^\]]+)\] ?(.*)$")
# The supervisor's own prefix on a child line: `[gateway] ` or `[engine: qwen] `.
_CHILD = re.compile(r"^\[([^\]\n]+)\] (.*)$")
# The asctime the agent's logging format puts first, local time.
_ASCTIME = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) (.*)$")

AGENT_SOURCE = "agent"
UPDATE_SOURCE = "update"
LOG_FILE = "agent.log"
BACKUPS = 5

MAX_TAIL = 5000
DEFAULT_TAIL = 500


@dataclass(frozen=True)
class Line:
    time: datetime | None
    source: str
    text: str


def _now() -> datetime:
    return datetime.now(UTC)


def _format_time(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + (
        f"{moment.microsecond // 1000:03d}Z"
    )


def stamp(line: str, *, now: datetime | None = None) -> str:
    """The file's form of one line of console output."""
    moment = now or _now()
    child = _CHILD.match(line)
    if child is not None:
        source, text = child.group(1), child.group(2)
    else:
        source = AGENT_SOURCE
        own = _ASCTIME.match(line)
        text = own.group(3) if own is not None else line
    return f"{_format_time(moment)} [{source}] {text}"


def parse(line: str) -> Line:
    """One line of the file, stamped or from before stamping."""
    stamped = _STAMPED.match(line)
    if stamped is not None:
        moment = datetime.strptime(stamped.group(1), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
        return Line(moment, stamped.group(2), stamped.group(3))
    child = _CHILD.match(line)
    if child is not None:
        return Line(None, child.group(1), child.group(2))
    own = _ASCTIME.match(line)
    if own is not None:
        local = datetime.strptime(f"{own.group(1)}.{own.group(2)}", "%Y-%m-%d %H:%M:%S.%f")
        return Line(local.astimezone(UTC), AGENT_SOURCE, own.group(3))
    return Line(None, AGENT_SOURCE, line)


# --- masking ----------------------------------------------------------------

_SECRETS = [
    # A JWT: every token this install mints, and most bearer tokens.
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    # Provider keys: Anthropic and OpenAI (sk-...), OpenRouter (sk-or-...),
    # Hugging Face (hf_...), GitHub (ghp_/gho_/ghs_/github_pat_).
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\b(?:gh[posu]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    # Whatever follows a bearer scheme, whatever its shape.
    re.compile(r"(?i)(?<=bearer )[A-Za-z0-9._~+/=-]{12,}"),
]
MASK = "[redacted]"


def redact(text: str) -> str:
    for pattern in _SECRETS:
        text = pattern.sub(MASK, text)
    return text


# --- reading ----------------------------------------------------------------


def files(log_dir: Path) -> list[Path]:
    """The stream's files, newest first: agent.log, then .1 ... .5."""
    names = [LOG_FILE] + [f"{LOG_FILE}.{n}" for n in range(1, BACKUPS + 1)]
    return [log_dir / name for name in names if (log_dir / name).is_file()]


def _lines_backwards(path: Path, block: int = 64 * 1024) -> Iterator[str]:
    """A file's lines, last first, without reading it whole."""
    try:
        handle = path.open("rb")
    except OSError:
        return
    with handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        rest = b""
        while position > 0:
            step = min(block, position)
            position -= step
            handle.seek(position)
            chunk = handle.read(step) + rest
            parts = chunk.split(b"\n")
            rest = parts[0]
            for raw in reversed(parts[1:]):
                if raw:
                    yield raw.decode("utf-8", errors="replace").rstrip("\r")
        if rest:
            yield rest.decode("utf-8", errors="replace").rstrip("\r")


def matches(
    line: Line, *, sources: Sequence[str] | None, contains: str | None, since: datetime | None
) -> bool:
    if sources and line.source not in sources:
        return False
    if contains and contains.lower() not in line.text.lower():
        return False
    return not (since is not None and line.time is not None and line.time < since)


@dataclass
class Page:
    lines: list[Line]
    sources: list[str]
    truncated: bool


def read(
    log_dir: Path,
    *,
    sources: Sequence[str] | None = None,
    contains: str | None = None,
    since: datetime | None = None,
    tail: int = DEFAULT_TAIL,
    update_log: Path | None = None,
) -> Page:
    """The newest `tail` lines that match, oldest first, and every source
    seen on the way. `truncated` says older matching lines exist."""
    tail = max(1, min(tail, MAX_TAIL))
    wanted: list[Line] = []
    seen: set[str] = set()
    truncated = False
    done = False
    for path in files(log_dir):
        for raw in _lines_backwards(path):
            line = parse(raw)
            seen.add(line.source)
            # Newest first, stamped at receipt: the first line older than
            # `since` means every line after it is too, so stop reading.
            if since is not None and line.time is not None and line.time < since:
                done = True
                break
            if not matches(line, sources=sources, contains=contains, since=None):
                continue
            if len(wanted) >= tail:
                truncated = True
                done = True
                break
            wanted.append(line)
        if done:
            break
    if update_log is not None and update_log.is_file():
        seen.add(UPDATE_SOURCE)
        if sources and UPDATE_SOURCE in sources:
            # The updater's own log has no stamps; it is the installer's
            # output, in order, read whole (a few hundred lines at most).
            with contextlib.suppress(OSError):
                text = update_log.read_text(encoding="utf-8", errors="replace")
                extra = [Line(None, UPDATE_SOURCE, t) for t in text.splitlines() if t.strip()]
                extra = [
                    x for x in extra if matches(x, sources=sources, contains=contains, since=None)
                ]
                wanted.extend(reversed(extra[-tail:]))
    wanted.reverse()
    return Page(lines=wanted[-tail:], sources=sorted(seen), truncated=truncated)


# --- following --------------------------------------------------------------


class Bus:
    """Every stamped line the tee writes, handed to whoever is following.

    The tee writes from whatever thread printed, so a line reaches each
    follower's queue through its own event loop. A follower that falls
    behind by more than its queue loses lines rather than growing without
    bound; the stream tells it how many.
    """

    def __init__(self, depth: int = 2000) -> None:
        self._lock = threading.Lock()
        self._followers: list[_Follower] = []
        self._depth = depth

    def publish(self, stamped: str) -> None:
        with self._lock:
            followers = list(self._followers)
        if not followers:
            return
        line = parse(stamped)
        for follower in followers:
            follower.offer(line)

    @contextlib.contextmanager
    def follow(self) -> Iterator[_Follower]:
        follower = _Follower(asyncio.get_running_loop(), self._depth)
        with self._lock:
            self._followers.append(follower)
        try:
            yield follower
        finally:
            with self._lock:
                self._followers.remove(follower)


class _Follower:
    def __init__(self, loop: asyncio.AbstractEventLoop, depth: int) -> None:
        self._loop = loop
        self.queue: asyncio.Queue[Line] = asyncio.Queue(maxsize=depth)
        self.dropped = 0

    def offer(self, line: Line) -> None:
        def put() -> None:
            try:
                self.queue.put_nowait(line)
            except asyncio.QueueFull:
                self.dropped += 1

        with contextlib.suppress(RuntimeError):  # the loop has closed
            self._loop.call_soon_threadsafe(put)


BUS = Bus()
