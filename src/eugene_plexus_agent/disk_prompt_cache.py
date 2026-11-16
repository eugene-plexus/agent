"""The prompt cache on disk (row 15, DI2 in `docs/design/deepinfer-lessons.md`; call DI-C4).

Every engine stop throws away the conversations a llama.cpp engine holds: an idle unload, a
model switch, a profile change, an update or an agent restart. The next turn of each then
rereads its whole prompt. llama-server can save a slot to a file and read it back
(`--slot-save-path`, `POST /slots/{id}?action=save|restore`): the file holds the slot's
tokens, its state and its context checkpoints, so it works for sliding-window and hybrid
models too. This module owns the files; `LlamaCppAdapter` speaks to the engine, and
`RuntimeSupervisor` decides when.

**When.** Before a planned stop or restart of a ready runtime, the agent saves every slot that
ran a conversation and is not answering now; when the runtime is next ready, and before it
is reported ready, it reads the newest files back, at most one per slot. A crash saves
nothing. The engine then finds a conversation's slot by its own longest-common-prefix
choice, so this works whether or not the profile pins slots (CB4), which is off by default.

**A file is read back only by the launch that saved it.** Each runtime has its own folder,
and in it each launch has a folder named by a hash of everything that shapes the state: the
command line (model, context, slots, cache type, projector, drafter, every other flag) with
the identity (size and modification time) of every file it names, the engine program's
identity, and the `LLAMA_ARG_*` variables the engine starts with. The engine is given only
that folder, so a mismatch is a miss and never a load.

**The files are copies of people's chats.** They are kept in this machine's private Eugene
folder (owner-only on Linux and macOS; on Windows the install folder's ACL), only for an
engine that listens on this machine alone (llama-server's slot routes take no key), never
shown as text, deleted `diskPromptCacheDays` after their last use (3, DI-C4), least
recently used first above `diskPromptCacheGib` (20), with their runtime, and all at once
when the setting is turned off or Clear is pressed.

Files are saved as `<name>.part` and renamed to `<name>.bin` only when the engine reports the
save complete, so a save cut short (an agent stopping) is never read back.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import stat
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ENABLED_KEY = "diskPromptCache"
DAYS_KEY = "diskPromptCacheDays"
CAP_KEY = "diskPromptCacheGib"
MIN_FREE_KEY = "diskPromptCacheMinFreeGb"
KEYS = frozenset({ENABLED_KEY, DAYS_KEY, CAP_KEY, MIN_FREE_KEY})
CATEGORY = "promptCache"

#: DI-C4 (Troy, 2026-10-21): on for llama.cpp, kept 3 days, a 20 GiB cap that is a setting.
DEFAULT_ENABLED = True
DEFAULT_DAYS = 3
DEFAULT_CAP_GIB = 20
#: Review S6: the folder is beside agent.yaml and node.yaml, so a save never takes the
#: disk below this much free space.
DEFAULT_MIN_FREE_GB = 10
GIB = 1024**3

FOLDER_NAME = "prompt-cache"
FLAG = "--slot-save-path"
VARIABLE = "LLAMA_ARG_SLOT_SAVE_PATH"
SAVED = ".bin"
PART = ".part"
#: A `.part` this old in a folder an engine uses is a save that never finished.
STALE_PART_SECONDS = 3600.0

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

#: Command-line flags that say where the engine listens or what it is called, not what its
#: state is: left out of the key, so moving a runtime to another port keeps its files.
_NOT_STATE = frozenset({"--host", "--port", "--alias", "-a", FLAG})
_NOT_STATE_VARIABLES = frozenset({"LLAMA_ARG_HOST", "LLAMA_ARG_PORT", "LLAMA_ARG_ALIAS", VARIABLE})
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
_FILE_NAME = re.compile(r"^(\d+)-[0-9a-f]{6}-s(\d+)(?:-t(\d+))?$")
_HOST_FLAGS = ("--host",)
#: Flags with which every save fails: no slot routes, or slot routes that want a key.
_NO_SLOTS = ("--no-slots",)
_SLOTS = ("--slots",)
_KEY_FLAGS = ("--api-key", "--api-key-file")
_KEY_VARIABLES = ("LLAMA_ARG_API_KEY", "LLAMA_ARG_API_KEY_FILE")
_FALSE = frozenset({"0", "false", "off", "no", "disabled"})

NOTE_OFF = "Off in this machine's settings."


# --- the settings ---------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    enabled: bool
    days: int
    cap_bytes: int
    min_free_bytes: int = DEFAULT_MIN_FREE_GB * GIB

    @property
    def max_age_seconds(self) -> float:
        return self.days * 86400.0


def _whole(value: object, default: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return default
    return value


def settings_from_config(get_config: Callable[[str], Any] | None) -> Settings:
    """The three settings, each degrading to its default rather than raising: a bad value
    must never stop an engine from starting (degraded mode)."""
    if get_config is None:
        return Settings(enabled=False, days=DEFAULT_DAYS, cap_bytes=DEFAULT_CAP_GIB * GIB)
    enabled = get_config(ENABLED_KEY)
    return Settings(
        enabled=DEFAULT_ENABLED if enabled is None else enabled is True,
        days=_whole(get_config(DAYS_KEY), DEFAULT_DAYS, 1),
        cap_bytes=_whole(get_config(CAP_KEY), DEFAULT_CAP_GIB, 1) * GIB,
        min_free_bytes=_whole(get_config(MIN_FREE_KEY), DEFAULT_MIN_FREE_GB, 0) * GIB,
    )


# --- where a launch keeps its files ---------------------------------------------------


def runtime_folder(root: Path, runtime: str) -> Path:
    """The runtime's own folder: its name when that is a plain file name, else a hash."""
    if _SAFE_NAME.fullmatch(runtime) and runtime not in (".", ".."):
        return root / runtime
    return root / ("r-" + hashlib.sha256(runtime.encode("utf-8")).hexdigest()[:16])


def _identity(token: str) -> list[object] | None:
    """`[size, mtime_ns, device, file id]` of a file a command-line value names (`PATH`, or
    the value of `--flag=PATH`), or None when it names none. The file id is the inode, or
    the file index on Windows: a file replaced in place under the same size and time is
    still another file."""
    if token.startswith("-"):
        _, equals, value = token.partition("=")
        if not equals:
            return None
        token = value
    if not token or len(token) > 4096:
        return None
    try:
        found = os.stat(token)
    except (OSError, ValueError):
        return None
    if not stat.S_ISREG(found.st_mode):
        return None
    return [found.st_size, found.st_mtime_ns, found.st_dev, found.st_ino]


def launch_key(argv: Sequence[str], env: Mapping[str, str] | None) -> str:
    """The hash that names a launch's folder: what shapes the engine's state, and nothing
    about where it listens. `argv[0]` is the engine program."""
    items = [str(token) for token in argv]
    program = items[0] if items else ""
    kept: list[object] = [["program", program, _identity(program)]]
    skip = False
    for token in items[1:]:
        if skip:
            skip = False
            continue
        name, equals, _ = token.partition("=")
        if name in _NOT_STATE:
            skip = not equals
            continue
        kept.append([token, _identity(token)])
    variables = [
        [key, value, _identity(value)]
        for key, value in sorted((env or {}).items())
        if key.startswith("LLAMA_ARG_") and key not in _NOT_STATE_VARIABLES
    ]
    payload = json.dumps({"argv": kept, "env": variables}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def last_value(argv: Sequence[str], names: Sequence[str]) -> str | None:
    """The value of the last `--flag value` or `--flag=value` among `argv`: the one
    llama-server's parser ends with."""
    found: str | None = None
    items = [str(token) for token in argv]
    for index, token in enumerate(items):
        name, equals, inline = token.partition("=")
        if name not in names:
            continue
        if equals:
            found = inline
        elif index + 1 < len(items):
            found = items[index + 1]
    return found


def _last_switch(argv: Sequence[str], on: Sequence[str], off: Sequence[str]) -> bool | None:
    """Whether the last of a pair of switches (`--slots`/`--no-slots`) is on; None if neither."""
    found: bool | None = None
    for token in argv:
        if token in on:
            found = True
        elif token in off:
            found = False
    return found


def folder_of(argv: Sequence[str]) -> Path | None:
    """The folder a running engine saves to: the last `--slot-save-path` on its command line."""
    found = last_value(argv, (FLAG,))
    return Path(found) if found else None


def listens_on(argv: Sequence[str], env: Mapping[str, str] | None) -> str:
    """Where the engine listens, as llama-server decides it (review B1): the last `--host`
    on its command line, `extraArgs` included, else `LLAMA_ARG_HOST`, else loopback."""
    host = last_value(argv, _HOST_FLAGS)
    if host is None:
        host = (env or {}).get("LLAMA_ARG_HOST") or None
    return host or "127.0.0.1"


def _why_saves_fail(argv: Sequence[str], env: Mapping[str, str] | None) -> str | None:
    """Review S7: a command line on which every save would fail, and why."""
    env = env or {}
    slots = _last_switch(argv, _SLOTS, _NO_SLOTS)
    if slots is None and env.get("LLAMA_ARG_ENDPOINT_SLOTS", "").strip().lower() in _FALSE:
        slots = False
    if slots is False:
        return "Not kept: the slot routes are off (--no-slots), so the engine cannot save them."
    if any(last_value(argv, (flag,)) for flag in _KEY_FLAGS) or any(
        env.get(variable) for variable in _KEY_VARIABLES
    ):
        return (
            "Not kept: the engine was given an API key (--api-key), so Eugene cannot ask it "
            "to save or read them back."
        )
    return None


def is_managed(root: Path | None, folder: Path | None) -> bool:
    """Whether `folder` is a launch folder this agent made (two levels under its root)."""
    if root is None or folder is None:
        return False
    try:
        return Path(os.path.abspath(folder)).parent.parent == Path(os.path.abspath(root))
    except (OSError, ValueError):
        return False


def make_private_dir(path: Path, root: Path) -> None:
    """`path` and every folder from `root` down to it, owner-only on POSIX."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        return
    current = Path(path)
    stop = Path(os.path.abspath(root)).parent
    while Path(os.path.abspath(current)) != stop and current != current.parent:
        with contextlib.suppress(OSError):
            os.chmod(current, 0o700)
        current = current.parent


@dataclass(frozen=True)
class Launch:
    """What a launch does with the prompt cache on disk: `folder` when it saves and reads
    back, else `note` says why it does not."""

    folder: Path | None
    note: str | None


def plan(
    *,
    runtime: str,
    host: str | None,
    argv: list[str],
    env: Mapping[str, str] | None,
    extra_args: Sequence[str] | None,
    root: Path | None,
    settings: Settings,
    search: bool,
    offered: frozenset[str] | None,
) -> tuple[list[str], Launch | None]:
    """The command line with Eugene's `--slot-save-path`, and what it means.

    `argv` ends with the runtime's `extraArgs`, verbatim; the flag goes in front of them,
    so an expert's own value still comes last. None for a search model (no cache).
    """
    if search:
        return argv, None
    by_hand = folder_of(extra_args or []) is not None or bool((env or {}).get(VARIABLE))
    if by_hand:
        return argv, Launch(
            None,
            f"Saving is set by hand ({FLAG} or {VARIABLE}), so Eugene does not save "
            "conversations or read them back.",
        )
    if not settings.enabled:
        return argv, Launch(None, NOTE_OFF)
    if root is None:
        return argv, Launch(None, "This agent has no folder of its own to keep them in.")
    listening = listens_on(argv, env) if argv else (host or "127.0.0.1")
    if listening not in LOOPBACK_HOSTS:
        # llama-server's slot routes take no key: an engine other machines can reach would
        # let them write, read back and erase files in this folder.
        return argv, Launch(
            None,
            f"Not kept: this model listens on {listening}, which other machines can reach, "
            "and saving needs an engine that listens on this machine only.",
        )
    failing = _why_saves_fail(argv, env)
    if failing is not None:
        return argv, Launch(None, failing)
    if offered is not None and FLAG not in offered:
        return argv, Launch(None, f"This llama.cpp build has no {FLAG}, so nothing is saved.")
    folder = runtime_folder(root, runtime) / launch_key(argv, env)
    try:
        make_private_dir(folder, root)
    except OSError as exc:
        # The cause only: the note is read by services and operations tokens, which are
        # not told where this machine keeps its files.
        return argv, Launch(None, f"Its folder could not be made: {exc.strerror or 'refused'}.")
    tail = len(extra_args or [])
    head = argv[: len(argv) - tail] if tail else list(argv)
    return [*head, FLAG, str(folder), *argv[len(head) :]], Launch(folder, None)


# --- what the engine answered -----------------------------------------------------------


@dataclass(frozen=True)
class SlotSave:
    """One slot's save: `filename` as the engine was given it, or `error` in its words."""

    slot: int
    filename: str | None
    tokens: int | None
    bytes: int | None
    error: str | None
    #: Not attempted: `error` is the agent's reason (no room, no time).
    skipped: bool = False
    #: The engine did not answer within the time left.
    timed_out: bool = False
    seconds: float | None = None


@dataclass(frozen=True)
class SlotSaves:
    saves: list[SlotSave]
    #: Why no slot could be saved at all (the slot list did not answer), in its words.
    error: str | None


@dataclass(frozen=True)
class SlotRestore:
    slot: int
    tokens: int | None
    error: str | None
    #: The engine refused the file itself (a 4xx): it is deleted, never tried again.
    refused: bool


# --- the files --------------------------------------------------------------------------


@dataclass(frozen=True)
class Saved:
    path: Path
    slot: int | None
    bytes: int
    used: float
    saved: int
    tokens: int | None = None


def _parse(name: str) -> tuple[int, int | None, int | None]:
    """`(saved at, slot, tokens)` from a file's name."""
    match = _FILE_NAME.match(name)
    if match is None:
        return 0, None, None
    tokens = match.group(3)
    return int(match.group(1)), int(match.group(2)), int(tokens) if tokens else None


def new_name(slot: int) -> str:
    """A file's name without its suffix: when it was saved, a random tail, the slot."""
    return f"{time.time_ns() // 1_000_000}-{secrets.token_hex(3)}-s{slot}"


def part_name(name: str) -> str:
    return name + PART


def _entries(folder: Path) -> Iterable[os.DirEntry[str]]:
    try:
        with os.scandir(folder) as found:
            yield from list(found)
    except OSError:
        return


def saved(folder: Path) -> list[Saved]:
    """The finished files in a launch folder, most recently used first."""
    out: list[Saved] = []
    for entry in _entries(folder):
        if not entry.name.endswith(SAVED) or entry.is_symlink():
            continue
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        at, slot, tokens = _parse(entry.name[: -len(SAVED)])
        out.append(Saved(Path(entry.path), slot, info.st_size, info.st_mtime, at, tokens))
    out.sort(key=lambda s: (s.used, s.saved, s.path.name), reverse=True)
    return out


def finish(folder: Path, name: str, tokens: int | None = None) -> Path | None:
    """The engine finished saving `<name>.part`: make it `<name>-t<tokens>.bin`, owner-only.
    The token count in the name is what the next save's size is estimated from."""
    source = folder / part_name(name)
    target = folder / (name + (f"-t{tokens}" if tokens else "") + SAVED)
    try:
        if os.name != "nt":
            os.chmod(source, 0o600)
        os.replace(source, target)
    except OSError as exc:
        log.warning("prompt cache on disk: could not keep %s: %s", source.name, exc)
        with contextlib.suppress(OSError):
            source.unlink()
        return None
    return target


def discard(path: Path) -> bool:
    """Delete `path`; whether it is gone."""
    try:
        path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def bytes_per_token(folder: Path) -> float | None:
    """The largest size per token of the files this launch saved, or None when no file says
    its token count: what a next save of N tokens is estimated to take (review S6)."""
    rates = [f.bytes / f.tokens for f in saved(folder) if f.tokens]
    return max(rates) if rates else None


def free_bytes(folder: Path) -> int | None:
    try:
        return shutil.disk_usage(folder).free
    except OSError:
        return None


def prune(folder: Path, keep: int) -> int:
    """Keep the `keep` most recently used finished files of a launch folder (the engine reads
    back at most one per slot) and nothing else: every other regular file there is the
    cache's, whatever its name (review S3), and goes. The number deleted."""
    kept = {item.path.name for item in saved(folder)[: max(keep, 0)]}
    gone = 0
    for entry in _entries(folder):
        if entry.name in kept or entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            continue
        if discard(Path(entry.path)) and entry.name.endswith(SAVED):
            gone += 1
    return gone


@dataclass
class _Walk:
    files: list[tuple[Path, Path, int, float]] = field(default_factory=list)
    """(runtime folder, file, bytes, last used), every finished file."""
    parts: list[tuple[Path, Path, float]] = field(default_factory=list)


def _regular(entry: os.DirEntry[str]) -> os.stat_result | None:
    if entry.is_symlink():
        return None
    try:
        info = entry.stat(follow_symlinks=False)
    except OSError:
        return None
    return info if stat.S_ISREG(info.st_mode) else None


def _walk(root: Path) -> _Walk:
    """Every regular file under the root down to the launch folders, whatever its name
    (review S3): a file another local process had the engine save is the cache's too, and
    counts toward the cap and the days. An unfinished save (`.part`) is listed apart."""
    found = _Walk()
    for top in _entries(root):
        if (info := _regular(top)) is not None:
            found.files.append((root, Path(top.path), info.st_size, info.st_mtime))
            continue
        if top.is_symlink() or not top.is_dir(follow_symlinks=False):
            continue
        for launch in _entries(Path(top.path)):
            if (info := _regular(launch)) is not None:
                found.files.append((Path(top.path), Path(launch.path), info.st_size, info.st_mtime))
                continue
            if launch.is_symlink() or not launch.is_dir(follow_symlinks=False):
                continue
            for entry in _entries(Path(launch.path)):
                if (info := _regular(entry)) is None:
                    continue
                path = Path(entry.path)
                if entry.name.endswith(PART):
                    found.parts.append((Path(launch.path), path, info.st_mtime))
                else:
                    found.files.append((Path(top.path), path, info.st_size, info.st_mtime))
    return found


@dataclass(frozen=True)
class SweepResult:
    expired: int = 0
    over_cap: int = 0
    unused: int = 0
    bytes_freed: int = 0


def sweep(
    root: Path | None,
    settings: Settings,
    *,
    declared: Iterable[str] | None = None,
    active: Iterable[Path] = (),
    now: float | None = None,
) -> SweepResult:
    """Apply the limits. Off: every file goes. On: a runtime no longer declared loses its
    folder, a file unused for `days` goes, then the least recently used above the cap, and an
    unfinished save no engine is writing. Folders an engine uses are kept, even empty.

    Never follows a link: a link in the folder is not ours to delete through."""
    if root is None or not root.is_dir() or root.is_symlink():
        return SweepResult()
    now = time.time() if now is None else now
    in_use = {Path(os.path.abspath(path)) for path in active}
    expired = over_cap = unused = freed = 0
    if declared is not None and settings.enabled:
        names = {runtime_folder(root, name).name for name in declared}
        for runtime in _entries(root):
            if runtime.is_symlink() or not runtime.is_dir(follow_symlinks=False):
                continue
            path = Path(runtime.path)
            if runtime.name in names or any(p.parent == path for p in in_use):
                continue
            walked = _walk_one(path)
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                unused += walked[0]
                freed += walked[1]
    walk = _walk(root)
    for launch, part, mtime in walk.parts:
        if (
            not settings.enabled
            or Path(os.path.abspath(launch)) not in in_use
            or now - mtime > STALE_PART_SECONDS
        ):
            discard(part)
    keep: list[tuple[Path, int, float]] = []
    for _, path, size, used in walk.files:
        # Counted only when it is really gone (review): a file the system kept is still
        # there, and still counts toward the cap below.
        if (not settings.enabled or now - used > settings.max_age_seconds) and discard(path):
            expired += 1
            freed += size
            continue
        keep.append((path, size, used))
    total = sum(size for _, size, _ in keep)
    for path, size, _ in sorted(keep, key=lambda item: item[2]):
        if total <= settings.cap_bytes:
            break
        if discard(path):
            total -= size
            over_cap += 1
            freed += size
    _drop_empty(root, in_use)
    if expired or over_cap or unused:
        log.info(
            "prompt cache on disk: deleted %d expired, %d over the %d GiB cap, %d of removed "
            "runtimes (%d bytes)",
            expired,
            over_cap,
            settings.cap_bytes // GIB,
            unused,
            freed,
        )
    return SweepResult(expired=expired, over_cap=over_cap, unused=unused, bytes_freed=freed)


def _walk_one(runtime: Path) -> tuple[int, int]:
    count = size = 0
    for launch in _entries(runtime):
        if launch.is_symlink() or not launch.is_dir(follow_symlinks=False):
            continue
        for entry in _entries(Path(launch.path)):
            if (info := _regular(entry)) is not None and not entry.name.endswith(PART):
                count += 1
                size += info.st_size
    return count, size


def _drop_empty(root: Path, in_use: set[Path]) -> None:
    for runtime in _entries(root):
        if runtime.is_symlink() or not runtime.is_dir(follow_symlinks=False):
            continue
        for launch in _entries(Path(runtime.path)):
            path = Path(os.path.abspath(launch.path))
            if launch.is_symlink() or not launch.is_dir(follow_symlinks=False) or path in in_use:
                continue
            with contextlib.suppress(OSError):
                os.rmdir(launch.path)
        if not any(p.parent == Path(os.path.abspath(runtime.path)) for p in in_use):
            with contextlib.suppress(OSError):
                os.rmdir(runtime.path)


def remove_runtime(root: Path | None, runtime: str) -> None:
    """A runtime was deleted: its saved conversations go with it."""
    if root is None:
        return
    folder = runtime_folder(root, runtime)
    if folder.is_dir() and not folder.is_symlink():
        shutil.rmtree(folder, ignore_errors=True)


@dataclass(frozen=True)
class Cleared:
    conversations: int
    bytes_freed: int
    not_deleted: int
    reason: str | None


def clear(root: Path | None) -> Cleared:
    """Delete every file. Folders stay: a running engine saves into its own."""
    if root is None or not root.is_dir() or root.is_symlink():
        return Cleared(0, 0, 0, None)
    walk = _walk(root)
    count = freed = failed = 0
    reason: str | None = None
    for _, path, size, _ in walk.files:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            failed += 1
            reason = reason or f"{path.name}: {exc.strerror or exc}"
            continue
        count += 1
        freed += size
    for _, path, _ in walk.parts:
        discard(path)
    log.info("prompt cache on disk: cleared %d conversations (%d bytes)", count, freed)
    return Cleared(count, freed, failed, reason)


@dataclass(frozen=True)
class Usage:
    conversations: int
    bytes: int
    oldest: datetime | None
    newest: datetime | None
    by_runtime: dict[str, tuple[int, int]]


def usage(root: Path | None, runtimes: Iterable[str]) -> Usage:
    """What is kept: counts, sizes and dates. `runtimes` names the declared runtimes, so a
    folder is reported under its runtime's name rather than its folder name."""
    if root is None or not root.is_dir() or root.is_symlink():
        return Usage(0, 0, None, None, {})
    names = {runtime_folder(root, name).name: name for name in runtimes}
    walk = _walk(root)
    by: dict[str, tuple[int, int]] = {}
    used = [mtime for _, _, _, mtime in walk.files]
    for folder, _, size, _ in walk.files:
        name = names.get(folder.name, folder.name)
        count, total = by.get(name, (0, 0))
        by[name] = (count + 1, total + size)
    return Usage(
        conversations=len(walk.files),
        bytes=sum(size for _, _, size, _ in walk.files),
        oldest=datetime.fromtimestamp(min(used), UTC) if used else None,
        newest=datetime.fromtimestamp(max(used), UTC) if used else None,
        by_runtime=by,
    )


def folder_usage(folder: Path | None, *, slots: int | None = None) -> tuple[int, int]:
    """`(conversations, bytes)` a launch folder would read back: at most `slots` files."""
    if folder is None:
        return 0, 0
    files = saved(folder)
    if slots is not None:
        files = files[: max(slots, 0)]
    return len(files), sum(f.bytes for f in files)


def restore_order(files: Sequence[Saved], slots: int) -> list[tuple[int, Saved]]:
    """`(slot, file)` for the newest `slots` files: each to the slot it was saved from when
    that slot is free, the rest to the lowest free slots. A conversation whose driver pins
    slots (CB4) then finds its state where its pin points."""
    chosen = list(files[: max(slots, 0)])
    taken: dict[int, Saved] = {}
    rest: list[Saved] = []
    for item in chosen:
        if item.slot is not None and 0 <= item.slot < slots and item.slot not in taken:
            taken[item.slot] = item
        else:
            rest.append(item)
    free = iter(slot for slot in range(slots) if slot not in taken)
    for item in rest:
        slot = next(free, None)
        if slot is None:
            break
        taken[slot] = item
    return sorted(taken.items())
