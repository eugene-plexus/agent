"""A node's own copy of the models its runtimes point at.

Design: `docs/design/node-local-model-copy.md`. M11 put the model on
another machine and taught this agent where the share is mounted here;
that works, and on a 1 Gbps link it costs **four minutes of every start**
for a 25 GB model, ten if the engine maps the file instead of reading it.
This keeps a local copy so the second start costs seconds.

Three properties hold it to being a copy rather than a store, and all
three fall out of one rule -- **the set is the distinct `modelPath`s of
this node's declared runtimes**:

* **Nothing can be put in it by choosing.** There is no "add" here. A
  model is copied because a runtime on this node points at it, and for
  no other reason, which is what keeps differentiator #3 honest.
* **Nothing has to be evicted on a policy.** Delete a runtime and its
  copy is no longer wanted; two GPUs running two models want two copies;
  four M6 replicas of one model want one, because a set of paths
  de-duplicates itself. No LRU, no TTL, no heuristic to tune.
* **We manage what we made and never touch what you put there.** Every
  path written here is under the operator's chosen copy directory, named
  at the model's own relative path. A Library folder is never written
  to, renamed in, or deleted from.

The disk knob is **free space to leave**, not a cap to use: headroom is
the quantity the volume's owner cares about, and it stays meaningful as
the disk fills for reasons that have nothing to do with us.

Two things this module is careful about, both learned the hard way
elsewhere in this project:

* **A partial copy must never carry the final name.** Bytes land in a
  temp file beside the destination and are renamed after `fsync`, so an
  engine can only ever open a whole file. Same discipline as the
  library's downloader.
* **A launch never fails because of a copy.** Not enough headroom, a
  source that disappeared, a destination that will not take writes --
  every one of them falls back to opening the share exactly as M11 does
  today, with the reason reported on the runtime so a four-minute load
  is explained rather than mysterious.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

from .model_paths import PathRule, _parts, is_windows_shaped, match, resolve_model_path

log = logging.getLogger(__name__)

ENABLED_KEY = "modelCopyEnabled"
DIR_KEY = "modelCopyDir"
MIN_FREE_GB_KEY = "modelCopyMinFreeGb"

DEFAULT_MIN_FREE_GB = 50

#: Suffix for bytes in flight. Deliberately not a name any engine would
#: open, and deliberately in the destination directory rather than a
#: temp dir, so the rename is on one filesystem and therefore atomic.
PARTIAL_SUFFIX = ".ep-partial"

#: Read/write chunk. Large enough that the syscall overhead is noise
#: against a gigabit link, small enough that a headroom breach or a
#: cancel is noticed within a fraction of a second.
CHUNK_BYTES = 4 * 1024 * 1024

#: How often, in chunks, to re-check free space during a copy. A 25 GB
#: transfer takes minutes and something else on the box can fill the
#: disk underneath it.
HEADROOM_CHECK_EVERY = 16

GIB = 1024**3


@dataclass(frozen=True)
class CopySettings:
    """The three per-node config values, already interpreted."""

    enabled: bool
    directory: str | None
    min_free_bytes: int

    @property
    def usable(self) -> bool:
        """Switched on AND told where. A toggle with no directory is the
        state right after someone flips it in a config file by hand; it
        is not an error, it just cannot do anything."""
        return self.enabled and bool(self.directory)


def settings_from_config(get_config: Callable[[str], Any]) -> CopySettings:
    """Read the trio. Anything malformed degrades to "off" rather than
    raising: `degraded-mode-required` applies to a convenience feature
    more strongly than to anything else, not less."""
    enabled = bool(get_config(ENABLED_KEY))
    raw_dir = get_config(DIR_KEY)
    directory = str(raw_dir).strip() if isinstance(raw_dir, str) and raw_dir.strip() else None
    raw_min = get_config(MIN_FREE_GB_KEY)
    try:
        min_free_gb = int(raw_min) if raw_min is not None else DEFAULT_MIN_FREE_GB
    except (TypeError, ValueError):
        log.warning(
            "%s is not a number (%r); using %s GB", MIN_FREE_GB_KEY, raw_min, DEFAULT_MIN_FREE_GB
        )
        min_free_gb = DEFAULT_MIN_FREE_GB
    if min_free_gb < 0:
        min_free_gb = DEFAULT_MIN_FREE_GB
    return CopySettings(
        enabled=enabled,
        directory=os.path.expanduser(directory) if directory else None,
        min_free_bytes=min_free_gb * GIB,
    )


# --- where a copy goes --------------------------------------------------------


def relative_name(declared: str, rules: Sequence[PathRule]) -> str:
    """The model's path relative to its Library folder.

    `<copy dir>/<that>` is the whole naming scheme (§4.4): plain,
    predictable, and useful to a human with a file manager, which is the
    property that distinguishes this from a content-addressed cache.

    The folder is whichever rule matched -- `rule.source` is the folder
    as the Library spells it, and `match` already returns the remainder.
    With no rule (a single-host install, where a copy is pointless
    anyway) the root is stripped and the rest kept, because a drive
    letter or a leading slash is not a name and `D:/models/x.gguf` must
    not escape the copy directory.
    """
    best: tuple[int, tuple[str, ...]] | None = None
    for rule in rules:
        remainder = match(rule, declared)
        if remainder is None:
            continue
        depth = len(_parts(rule.source, windows=is_windows_shaped(rule.source))[0])
        if best is None or depth > best[0]:
            best = (depth, remainder)
    if best is not None:
        return "/".join(best[1])
    pure: PurePath = (
        PureWindowsPath(declared) if is_windows_shaped(declared) else PurePosixPath(declared)
    )
    parts = pure.parts
    # **Drop the anchor, and use the RAW parts.** `_parts` answers with
    # case-folded components first, which would lowercase a filename; and
    # an anchor left in place is not a name but a destination —
    # `os.path.join(copy_dir, "D:\\", ...)` resolves to `D:\` on Windows,
    # so the copy would land outside the directory the operator chose,
    # which is the one promise this makes about where it writes.
    if pure.anchor:
        parts = parts[1:]
    return "/".join(parts)


@dataclass(frozen=True)
class CopyPlan:
    """One model file, where it is now and where its copy belongs."""

    declared: str
    source: str
    destination: str

    @property
    def partial(self) -> str:
        return self.destination + PARTIAL_SUFFIX


def plan_for(
    declared: str,
    rules: Sequence[PathRule],
    settings: CopySettings,
    *,
    expand: Callable[[str], str] = os.path.expanduser,
) -> CopyPlan | None:
    """Where this node would keep its copy of `declared`, or None when
    it would not keep one at all.

    None also covers the case that matters more than it looks: a source
    that is **already inside the copy directory**. Without it, a restart
    after a copy would plan to copy the copy onto itself.
    """
    if not settings.usable:
        return None
    assert settings.directory is not None
    source = resolve_model_path(declared, rules, expand=expand).local_path
    relative = relative_name(declared, rules)
    if not relative:
        return None
    destination = os.path.join(settings.directory, *relative.split("/"))
    if os.path.normcase(os.path.abspath(source)) == os.path.normcase(os.path.abspath(destination)):
        return None
    return CopyPlan(declared=declared, source=source, destination=destination)


def wanted(
    declared_paths: Iterable[str],
    rules: Sequence[PathRule],
    settings: CopySettings,
) -> dict[str, CopyPlan]:
    """The whole set this node wants, keyed by destination.

    Keyed by destination rather than by runtime on purpose: **M6
    replicas are N runtimes over one file**, and this is where that
    stops being a special case -- two runtimes pointing at one model
    produce one entry because they produce one key.
    """
    out: dict[str, CopyPlan] = {}
    for declared in declared_paths:
        plan = plan_for(declared, rules, settings)
        if plan is not None:
            out.setdefault(plan.destination, plan)
    return out


# --- is the copy any good -----------------------------------------------------


@dataclass(frozen=True)
class Stat:
    size: int
    mtime: float


def _stat(path: str) -> Stat | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return Stat(size=st.st_size, mtime=st.st_mtime)


def copy_is_current(plan: CopyPlan, *, mtime_tolerance: float = 2.0) -> bool:
    """Size and mtime against the source (§4.3).

    **The tradeoff is stated rather than hidden:** a GGUF edited in place
    with its mtime preserved would serve stale. A content hash would
    catch it and costs a full read of both copies on every start, which
    is the cost this whole feature exists to remove.

    The mtime tolerance is not slack for the sake of it: SMB and FAT
    round timestamps (FAT to two seconds), so a byte-perfect copy can
    come back with an mtime a second off its source and re-copying 25 GB
    over that would be worse than the problem.
    """
    source = _stat(plan.source)
    destination = _stat(plan.destination)
    if source is None or destination is None:
        return False
    if source.size != destination.size:
        return False
    return abs(source.mtime - destination.mtime) <= mtime_tolerance


# --- headroom -----------------------------------------------------------------


def free_bytes(directory: str) -> int | None:
    """Free space on the volume holding `directory`, walking up to the
    nearest existing parent -- the copy directory may not exist yet, and
    "how much room is there" is answerable before it does."""
    probe = Path(directory)
    while True:
        try:
            return shutil.disk_usage(probe).free
        except OSError:
            if probe.parent == probe:
                return None
            probe = probe.parent


def headroom_shortfall(plan: CopyPlan, settings: CopySettings, size: int) -> int:
    """Bytes by which this copy would eat into the headroom. <= 0 fits.

    Unmeasurable free space answers 0 -- it fits. That is the
    `easy-default-expert-override` corollary: an eager refusal can be
    wrong and an explanation of a real failure cannot, so a disk we
    cannot measure gets to try and fail honestly rather than be refused
    on a guess.
    """
    free = free_bytes(os.path.dirname(plan.destination) or ".")
    if free is None:
        return 0
    return (size + settings.min_free_bytes) - free


# --- doing it -----------------------------------------------------------------


@dataclass
class CopyState:
    """Live progress of one copy, shared with whatever reports it.

    Mutated from the worker thread and read from the event loop. Every
    field is a single machine word under CPython, so reads are coherent
    without a lock; `bytes_per_second` is computed from the pair rather
    than averaged over the whole copy, so a share that has just got
    slower shows it.
    """

    destination: str
    total_bytes: int | None
    bytes_copied: int = 0
    _window: tuple[float, int] = field(default=(0.0, 0), repr=False)
    _rate: float | None = field(default=None, repr=False)

    def note(self, copied: int, now: float) -> None:
        self.bytes_copied = copied
        then, at = self._window
        if then == 0.0:
            self._window = (now, copied)
            return
        if now - then >= 1.0:
            self._rate = (copied - at) / (now - then)
            self._window = (now, copied)

    @property
    def bytes_per_second(self) -> float | None:
        return self._rate


class CopyAborted(RuntimeError):
    """A copy stopped on purpose, with a reason fit to show an operator."""


def copy_file(
    plan: CopyPlan,
    settings: CopySettings,
    state: CopyState,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """Copy source to destination, leaving nothing behind if it fails.

    Blocking; the caller runs it in a thread. Raises `CopyAborted` with
    a plain-language reason, or an OSError for anything the OS refused.

    Headroom is re-checked while copying, not only before: a 25 GB
    transfer takes minutes and something else can fill the disk during
    it. A breach aborts and removes the partial, because the alternative
    is a feature meant to speed up a node being the reason its disk
    filled up.
    """
    os.makedirs(os.path.dirname(plan.destination) or ".", exist_ok=True)
    partial = plan.partial
    copied = 0
    started = time.monotonic()
    try:
        with open(plan.source, "rb") as src, open(partial, "wb") as dst:
            chunks = 0
            while True:
                if should_cancel is not None and should_cancel():
                    raise CopyAborted("the copy was cancelled")
                chunk = src.read(CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)
                copied += len(chunk)
                chunks += 1
                state.note(copied, time.monotonic())
                if chunks % HEADROOM_CHECK_EVERY == 0:
                    remaining = (state.total_bytes or copied) - copied
                    free = free_bytes(os.path.dirname(plan.destination) or ".")
                    if free is not None and free - remaining < settings.min_free_bytes:
                        raise CopyAborted(
                            "stopped copying to keep "
                            f"{settings.min_free_bytes // GIB} GB free on this disk"
                        )
            dst.flush()
            os.fsync(dst.fileno())
    except BaseException:
        _remove_quietly(partial)
        raise
    # mtime carried over so `copy_is_current` can compare it, and because
    # a copy that claims to be newer than its source is a lie about a
    # file the operator may well go looking at in a file manager.
    shutil.copystat(plan.source, partial)
    os.replace(partial, plan.destination)
    state.note(copied, time.monotonic())
    log.info(
        "copied %s to %s (%.1f GB in %.0fs)",
        plan.source,
        plan.destination,
        copied / GIB,
        time.monotonic() - started,
    )


def _remove_quietly(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return True
    except OSError as exc:
        log.debug("could not remove %s: %s", path, exc)
        return False


# --- getting the space back ---------------------------------------------------


@dataclass(frozen=True)
class Skipped:
    path: str
    reason: str
    runtime: str | None = None


@dataclass
class ClearResult:
    deleted: list[str] = field(default_factory=list)
    bytes_freed: int = 0
    skipped: list[Skipped] = field(default_factory=list)


def copies_on_disk(directory: str | None) -> list[str]:
    """Every file under the copy directory, partials excluded.

    Walks what we made rather than deriving it from the runtime list, so
    a copy orphaned by a deleted runtime is still found -- that is the
    case where the two disagree, and the disk is the honest answer.
    """
    if not directory or not os.path.isdir(directory):
        return []
    out: list[str] = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(PARTIAL_SUFFIX):
                continue
            out.append(os.path.join(root, name))
    return sorted(out)


def clear(
    directory: str | None,
    *,
    in_use: dict[str, str] | None = None,
) -> ClearResult:
    """Delete every copy this node holds, reporting what survived.

    **Stops nothing.** `in_use` maps a destination path to the runtime
    holding it, and those are skipped by name -- stopping an engine to
    reclaim disk is a decision with someone's session on the other end
    of it, and a button labelled "clear" must not make it.

    Windows enforces that anyway: an open or mapped file cannot be
    deleted at all, so a running engine's copy is protected by the OS
    rather than by this code. Linux unlinks it happily, the engine keeps
    its inode, and the space comes back when the process exits -- which
    is why the skip list is built from `in_use` on both platforms rather
    than from whether the delete raised.
    """
    result = ClearResult()
    held = {os.path.normcase(os.path.abspath(p)): r for p, r in (in_use or {}).items()}
    for path in copies_on_disk(directory):
        runtime = held.get(os.path.normcase(os.path.abspath(path)))
        if runtime is not None:
            result.skipped.append(
                Skipped(path=path, runtime=runtime, reason=f"runtime {runtime!r} is using it")
            )
            continue
        st = _stat(path)
        try:
            os.remove(path)
        except OSError as exc:
            result.skipped.append(Skipped(path=path, reason=str(exc)))
            continue
        result.deleted.append(path)
        result.bytes_freed += st.size if st is not None else 0
    _prune_empty_dirs(directory)
    return result


def evict_for_headroom(
    directory: str | None,
    settings: CopySettings,
    *,
    in_use: dict[str, str] | None = None,
    keep: Iterable[str] = (),
) -> ClearResult:
    """Give space back, oldest first, until the headroom is restored.

    Only ever runs when free space has fallen below the headroom for a
    reason of its own -- this is not an LRU, and a copy is never evicted
    to make room for another copy. `keep` is what a caller is about to
    need; `in_use` is protected as in `clear`.
    """
    result = ClearResult()
    if not directory:
        return result
    protected = {os.path.normcase(os.path.abspath(p)) for p in keep}
    held = {os.path.normcase(os.path.abspath(p)): r for p, r in (in_use or {}).items()}
    candidates = [
        (os.stat(p).st_mtime, p)
        for p in copies_on_disk(directory)
        if os.path.normcase(os.path.abspath(p)) not in protected and os.path.exists(p)
    ]
    for _mtime, path in sorted(candidates):
        free = free_bytes(directory)
        if free is None or free >= settings.min_free_bytes:
            break
        runtime = held.get(os.path.normcase(os.path.abspath(path)))
        if runtime is not None:
            result.skipped.append(
                Skipped(path=path, runtime=runtime, reason=f"runtime {runtime!r} is using it")
            )
            continue
        st = _stat(path)
        try:
            os.remove(path)
        except OSError as exc:
            result.skipped.append(Skipped(path=path, reason=str(exc)))
            continue
        result.deleted.append(path)
        result.bytes_freed += st.size if st is not None else 0
    _prune_empty_dirs(directory)
    return result


def remove_unwanted(
    directory: str | None,
    keep: Iterable[str],
    *,
    in_use: dict[str, str] | None = None,
) -> ClearResult:
    """Delete copies of models no runtime on this node points at any more.

    This is the whole eviction policy for the ordinary case: a runtime
    deleted or repointed takes its copy with it, with no LRU and no
    timer, because the set was never anything but a function of the
    runtime list.
    """
    wanted_paths = {os.path.normcase(os.path.abspath(p)) for p in keep}
    result = ClearResult()
    held = {os.path.normcase(os.path.abspath(p)): r for p, r in (in_use or {}).items()}
    for path in copies_on_disk(directory):
        key = os.path.normcase(os.path.abspath(path))
        if key in wanted_paths:
            continue
        if key in held:
            result.skipped.append(
                Skipped(path=path, runtime=held[key], reason=f"runtime {held[key]!r} is using it")
            )
            continue
        st = _stat(path)
        try:
            os.remove(path)
        except OSError as exc:
            result.skipped.append(Skipped(path=path, reason=str(exc)))
            continue
        result.deleted.append(path)
        result.bytes_freed += st.size if st is not None else 0
        log.info("removed local copy %s: no runtime points at it any more", path)
    _prune_empty_dirs(directory)
    return result


def _prune_empty_dirs(directory: str | None) -> None:
    """Leave no empty scaffolding behind. The copy directory itself
    stays: the operator made that choice, and removing it would make the
    next copy recreate it for no reason."""
    if not directory or not os.path.isdir(directory):
        return
    for root, dirs, files in os.walk(directory, topdown=False):
        if os.path.abspath(root) == os.path.abspath(directory):
            continue
        if not dirs and not files:
            with_error = False
            try:
                os.rmdir(root)
            except OSError:
                with_error = True
            if with_error:
                log.debug("could not remove empty directory %s", root)


# --- the one seam -------------------------------------------------------------


@dataclass(frozen=True)
class LocalPath:
    """Where this host opens a model, and which rule decided."""

    path: str
    source: str  # same_path | inherited | override | copy

    @property
    def is_copy(self) -> bool:
        return self.source == "copy"


def resolve_local_path(
    declared: str,
    rules: Sequence[PathRule],
    settings: CopySettings,
    *,
    expand: Callable[[str], str] = os.path.expanduser,
    rule_source: Callable[[PathRule | None], str] | None = None,
) -> LocalPath:
    """M11's resolution with the copy looked at first.

    **One seam, used by everything.** The spawn, the runtime view and
    admission all resolve a declared path, and if they disagreed about
    which file is under discussion the symptom would be admission
    refusing a model that is sitting right there. So this is the only
    function any of them calls.

    A copy that exists but is stale is *not* used -- returning it would
    serve yesterday's weights from a path the operator cannot see.
    """
    plan = plan_for(declared, rules, settings, expand=expand)
    if plan is not None and copy_is_current(plan):
        return LocalPath(path=plan.destination, source="copy")
    resolution = resolve_model_path(declared, rules, expand=expand)
    if rule_source is not None:
        return LocalPath(path=resolution.local_path, source=rule_source(resolution.rule))
    return LocalPath(
        path=resolution.local_path,
        source="same_path" if resolution.rule is None else "override",
    )


__all__ = [
    "CHUNK_BYTES",
    "DEFAULT_MIN_FREE_GB",
    "DIR_KEY",
    "ENABLED_KEY",
    "MIN_FREE_GB_KEY",
    "PARTIAL_SUFFIX",
    "ClearResult",
    "CopyAborted",
    "CopyPlan",
    "CopySettings",
    "CopyState",
    "LocalPath",
    "Skipped",
    "clear",
    "copies_on_disk",
    "copy_file",
    "copy_is_current",
    "evict_for_headroom",
    "free_bytes",
    "headroom_shortfall",
    "plan_for",
    "relative_name",
    "remove_unwanted",
    "resolve_local_path",
    "settings_from_config",
    "wanted",
]
