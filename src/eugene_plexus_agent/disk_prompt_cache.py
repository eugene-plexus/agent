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
KEYS = frozenset({ENABLED_KEY, DAYS_KEY, CAP_KEY})
CATEGORY = "promptCache"

#: DI-C4 (Troy, 2026-10-21): on for llama.cpp, kept 3 days, a 20 GiB cap that is a setting.
DEFAULT_ENABLED = True
DEFAULT_DAYS = 3
DEFAULT_CAP_GIB = 20
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
_FILE_NAME = re.compile(r"^(\d+)-[0-9a-f]{6}-s(\d+)$")

NOTE_OFF = "Off in this machine's settings."


# --- the settings ---------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    enabled: bool
    days: int
    cap_bytes: int

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
    )


# --- where a launch keeps its files ---------------------------------------------------


def runtime_folder(root: Path, runtime: str) -> Path:
    """The runtime's own folder: its name when that is a plain file name, else a hash."""
    if _SAFE_NAME.fullmatch(runtime) and runtime not in (".", ".."):
        return root / runtime
    return root / ("r-" + hashlib.sha256(runtime.encode("utf-8")).hexdigest()[:16])


def _identity(token: str) -> list[object] | None:
    """`[size, mtime_ns]` of a file a command-line value names, or None when it names none."""
    if not token or token.startswith("-") or len(token) > 4096:
        return None
    try:
        found = os.stat(token)
    except (OSError, ValueError):
        return None
    if not stat.S_ISREG(found.st_mode):
        return None
    return [found.st_size, found.st_mtime_ns]


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
    variables = sorted(
        (key, value)
        for key, value in (env or {}).items()
        if key.startswith("LLAMA_ARG_") and key not in _NOT_STATE_VARIABLES
    )
    payload = json.dumps({"argv": kept, "env": variables}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def folder_of(argv: Sequence[str]) -> Path | None:
    """The folder a running engine saves to: the last `--slot-save-path` on its command line."""
    found: str | None = None
    items = [str(token) for token in argv]
    for index, token in enumerate(items):
        name, equals, inline = token.partition("=")
        if name != FLAG:
            continue
        if equals:
            found = inline
        elif index + 1 < len(items):
            found = items[index + 1]
    return Path(found) if found else None


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
    if (host or "127.0.0.1") not in LOOPBACK_HOSTS:
        # llama-server's slot routes take no key: an engine other machines can reach would
        # let them write, read back and erase files in this folder.
        return argv, Launch(
            None,
            f"Not kept: this model listens on {host}, which other machines can reach, and "
            "saving needs an engine that listens on this machine only.",
        )
    if offered is not None and FLAG not in offered:
        return argv, Launch(None, f"This llama.cpp build has no {FLAG}, so nothing is saved.")
    folder = runtime_folder(root, runtime) / launch_key(argv, env)
    try:
        make_private_dir(folder, root)
    except OSError as exc:
        return argv, Launch(None, f"Its folder could not be made: {exc}")
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


def _parse(name: str) -> tuple[int, int | None]:
    match = _FILE_NAME.match(name)
    if match is None:
        return 0, None
    return int(match.group(1)), int(match.group(2))


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
        at, slot = _parse(entry.name[: -len(SAVED)])
        out.append(Saved(Path(entry.path), slot, info.st_size, info.st_mtime, at))
    out.sort(key=lambda s: (s.used, s.saved, s.path.name), reverse=True)
    return out


def finish(folder: Path, name: str) -> Path | None:
    """The engine finished saving `<name>.part`: make it `<name>.bin`, owner-only."""
    source = folder / part_name(name)
    target = folder / (name + SAVED)
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


def discard(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def touch(path: Path) -> None:
    """Mark a file used now: it was read back."""
    with contextlib.suppress(OSError):
        os.utime(path)


def prune(folder: Path, keep: int) -> int:
    """Keep the `keep` most recently used files of a launch folder (the engine reads back at
    most one per slot), and no unfinished save. The number deleted."""
    gone = 0
    for item in saved(folder)[max(keep, 0) :]:
        discard(item.path)
        gone += 1
    for entry in _entries(folder):
        if entry.name.endswith(PART) and not entry.is_symlink():
            discard(Path(entry.path))
    return gone


@dataclass
class _Walk:
    files: list[tuple[Path, Path, int, float]] = field(default_factory=list)
    """(runtime folder, file, bytes, last used), every finished file."""
    parts: list[tuple[Path, Path, float]] = field(default_factory=list)


def _walk(root: Path) -> _Walk:
    found = _Walk()
    for runtime in _entries(root):
        if runtime.is_symlink() or not runtime.is_dir(follow_symlinks=False):
            continue
        for launch in _entries(Path(runtime.path)):
            if launch.is_symlink() or not launch.is_dir(follow_symlinks=False):
                continue
            for entry in _entries(Path(launch.path)):
                if entry.is_symlink():
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                path = Path(entry.path)
                if entry.name.endswith(SAVED):
                    found.files.append((Path(runtime.path), path, info.st_size, info.st_mtime))
                elif entry.name.endswith(PART):
                    found.parts.append((Path(launch.path), path, info.st_mtime))
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
            unused += walked[0]
            freed += walked[1]
            shutil.rmtree(path, ignore_errors=True)
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
        if not settings.enabled or now - used > settings.max_age_seconds:
            discard(path)
            expired += 1
            freed += size
        else:
            keep.append((path, size, used))
    total = sum(size for _, size, _ in keep)
    for path, size, _ in sorted(keep, key=lambda item: item[2]):
        if total <= settings.cap_bytes:
            break
        discard(path)
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
            if entry.name.endswith(SAVED) and not entry.is_symlink():
                count += 1
                with contextlib.suppress(OSError):
                    size += entry.stat(follow_symlinks=False).st_size
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
