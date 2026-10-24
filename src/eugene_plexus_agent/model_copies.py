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

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
SOURCE_SUFFIX = ".source.json"
SOURCE_CHECK_SECONDS = 5.0

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
    folder_roots: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """Switched on AND told where. A toggle with no directory is the
        state right after someone flips it in a config file by hand; it
        is not an error, it just cannot do anything."""
        return self.enabled and bool(self.directory)


def settings_from_config(
    get_config: Callable[[str], Any], *, folder_roots: Sequence[str] = ()
) -> CopySettings:
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
        folder_roots=tuple(folder_roots),
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
class CopyMember:
    """One file of a model's copy: where it is now and where it belongs."""

    source: str
    destination: str

    @property
    def partial(self) -> str:
        return self.destination + PARTIAL_SUFFIX


@dataclass(frozen=True)
class CopyPlan:
    """One model, where it is now and where its copy belongs.

    `source` and `destination` are the file the runtime opens (a GGUF's
    first shard, a prepared model's provenance file). `members` is every
    file the model is made of, that one first (LS7, agent#12 and #10): a
    split GGUF's shards, or a prepared model's set, each kept at its place
    relative to its Library folder so the relative paths between them still
    hold in the copy. Empty means the one file alone.
    """

    declared: str
    source: str
    destination: str
    members: tuple[CopyMember, ...] = ()
    root: str | None = None

    @property
    def partial(self) -> str:
        return self.destination + PARTIAL_SUFFIX

    @property
    def files(self) -> tuple[CopyMember, ...]:
        return self.members or (CopyMember(self.source, self.destination),)

    @property
    def destinations(self) -> tuple[str, ...]:
        return tuple(member.destination for member in self.files)


#: A split GGUF's shard name (`-00001-of-00004.gguf`), as llama.cpp names them.
_SHARD = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
#: A prepared model's provenance file (LS3).
PREPARED_SUFFIX = ".eugene-prepared.json"
#: How long a model's member list is reused before the share is read again:
#: the runtime list is polled, and a prepared model's list reads JSON and
#: walks folders on what may be a network share.
MEMBERS_SECONDS = 30.0
_members_cache: dict[str, tuple[float, tuple[str, ...] | None]] = {}


def shards_of(path: str) -> tuple[str, ...]:
    """Every shard of a split GGUF, the first first; the file alone when it
    is not one."""
    folder, name = os.path.split(path)
    found = _SHARD.match(name)
    if found is None:
        return (path,)
    total = int(found.group("total"))
    width = len(found.group("index"))
    stem = found.group("stem")
    return tuple(
        os.path.join(folder, f"{stem}-{i:0{width}d}-of-{found.group('total')}.gguf")
        for i in range(1, total + 1)
    )


def _read_members(source: str) -> tuple[str, ...] | None:
    lowered = source.lower()
    if lowered.endswith(PREPARED_SUFFIX):
        return _prepared_members(source)
    if lowered.endswith(".json"):
        # An engine's own configuration is not a model a node copies; it is
        # reached through the provenance file that names it.
        return None
    files = shards_of(source)
    if len(files) > 1 and not all(os.path.isfile(f) for f in files):
        return None
    return files


def _prepared_members(source: str) -> tuple[str, ...] | None:
    """The files a prepared model is made of, from its engine's adapter,
    which alone knows what its configuration names."""
    import json

    from ._generated.models import EngineKind
    from .engines import ADAPTERS

    try:
        raw = json.loads(Path(source).read_text(encoding="utf-8-sig"))
        engine = raw.get("engine") if isinstance(raw, dict) else None
        kind = EngineKind(engine) if isinstance(engine, str) else None
    except (OSError, ValueError):
        return None
    adapter = ADAPTERS.get(kind) if kind is not None else None
    if adapter is None:
        return None
    files = adapter.prepared_files(Path(source))
    return tuple(str(f) for f in files) if files else None


def members_of(source: str) -> tuple[str, ...] | None:
    """Every file the model at `source` is made of, `source` first, or None
    when the set cannot be read (the model is then not copied, and runs
    from where it is)."""
    now = time.perf_counter()
    cached = _members_cache.get(source)
    if cached is not None and now - cached[0] <= MEMBERS_SECONDS:
        return cached[1]
    found = _read_members(source)
    _members_cache[source] = (now, found)
    return found


def plan_for(
    declared: str,
    rules: Sequence[PathRule],
    settings: CopySettings,
    *,
    expand: Callable[[str], str] = os.path.expanduser,
    companions: Sequence[str] = (),
) -> CopyPlan | None:
    """Where this node would keep its copy of `declared`, or None when
    it would not keep one at all.

    `companions` are files beside the model that the declaration also
    loads (SD1: a drafting companion), declared as the model is. They join
    the copy in their own places, so the engine opens them beside the
    model's copy. One that is not there is left out, to fail the launch
    with its own reason rather than block the model's copy.

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
    depth = len(relative.split("/"))
    parents = (PureWindowsPath if is_windows_shaped(source) else PurePosixPath)(source).parents
    if not 0 < depth <= len(parents):
        return None
    root = str(parents[depth - 1])
    layout_root = root
    folders = [
        folder
        for folder in settings.folder_roots
        if match(PathRule(folder, folder), declared) is not None
    ]
    if folders:
        folder = max(
            folders, key=lambda folder: len(_parts(folder, windows=is_windows_shaped(folder))[0])
        )
        root = resolve_model_path(folder, rules, expand=expand).local_path
    try:
        files = _bounded(source + ":members", lambda: members_of(source))
    except (OSError, TimeoutError):
        files = None
    if files is None:
        # Prepared and sharded models must keep the entire saved set. Never
        # treat an unreadable multi-file source as one complete file.
        record = _record(destination)
        saved_members = record.get("members") if record else None
        if isinstance(saved_members, list) and all(isinstance(p, str) for p in saved_members):
            files = tuple(saved_members)
    if files is None:
        return None
    for extra in companions:
        local = resolve_model_path(extra, rules, expand=expand).local_path
        if os.path.isfile(local) and local not in files:
            files = (*files, local)
    if len(files) <= 1:
        return CopyPlan(declared=declared, source=source, destination=destination, root=root)
    # Keep the established copy layout, independently of which Library
    # folder root must answer before a missing member means removal.
    members: list[CopyMember] = []
    for path in files:
        inside = os.path.relpath(path, layout_root)
        if os.path.isabs(inside) or inside == ".." or inside.startswith(".." + os.sep):
            # A file outside the model's Library folder cannot keep its
            # place in the copy: the model runs from where it is.
            return None
        members.append(CopyMember(path, os.path.join(settings.directory, inside)))
    return CopyPlan(
        declared=declared, source=source, destination=destination, members=tuple(members), root=root
    )


def wanted(
    declared_paths: Iterable[str | tuple[str, Sequence[str]]],
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
        path, companions = (declared, ()) if isinstance(declared, str) else declared
        plan = plan_for(path, rules, settings, companions=companions)
        if plan is None:
            continue
        held = out.get(plan.destination)
        if held is None:
            out[plan.destination] = plan
            continue
        # One model, two runtimes, one drafting with a companion: the copy
        # keeps the companion for the runtime that wants it.
        members = {m.destination: m for m in (*held.files, *plan.files)}
        if len(members) > len(held.files):
            out[plan.destination] = CopyPlan(
                declared=held.declared,
                source=held.source,
                destination=held.destination,
                members=tuple(members.values()),
                root=held.root,
            )
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
    """Every file of the copy is current (`member_is_current`)."""
    return check_copy(plan, mtime_tolerance=mtime_tolerance).usable


_pending_checks: dict[str, tuple[float, Future[Any]]] = {}
_checks_lock = threading.Lock()


def _bounded(key: str, read: Callable[[], Any]) -> Any:
    """One daemon per pending path: a stuck mount cannot hold shutdown open."""
    with _checks_lock:
        held = _pending_checks.get(key)
        if held is None or held[1].done():
            future: Future[Any] = Future()

            def run() -> None:
                try:
                    future.set_result(read())
                except BaseException as exc:
                    future.set_exception(exc)

            held = (time.perf_counter(), future)
            _pending_checks[key] = held
            threading.Thread(target=run, daemon=True, name="copy-source-check").start()
    started, future = held
    try:
        return future.result(
            timeout=max(0.0, SOURCE_CHECK_SECONDS - (time.perf_counter() - started))
        )
    finally:
        if future.done():
            with _checks_lock:
                if _pending_checks.get(key) == held:
                    _pending_checks.pop(key, None)


_record_lock = threading.RLock()


def _record(destination: str) -> dict[str, Any] | None:
    with _record_lock:
        return _read_record(destination)


def _read_record(destination: str) -> dict[str, Any] | None:
    try:
        raw = json.loads(Path(destination + SOURCE_SUFFIX).read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except (OSError, ValueError):
        return None


def _save_source(member: CopyMember, source: Stat, members: Sequence[str] = ()) -> None:
    with _record_lock:
        held = _read_record(member.destination)
        if (
            held
            and held.get("source") == member.source
            and held.get("size") == source.size
            and held.get("mtime") == source.mtime
            and held.get("copiedAt")
            and (not members or held.get("members") == list(members))
        ):
            return
        _write_source(member, source, members)


def _write_source(member: CopyMember, source: Stat, members: Sequence[str]) -> None:
    payload: dict[str, Any] = {
        "source": member.source,
        "size": source.size,
        "mtime": source.mtime,
        "copiedAt": datetime.now(UTC).isoformat(),
    }
    if members:
        payload["members"] = list(members)
    path = member.destination + SOURCE_SUFFIX
    partial = path + PARTIAL_SUFFIX
    try:
        with open(partial, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except OSError as exc:
        _remove_quietly(partial)
        log.warning("could not record the source of %s: %s", member.destination, exc)


@dataclass(frozen=True)
class CopyCheck:
    usable: bool
    note: str | None = None
    refusal: str | None = None


def check_copy(plan: CopyPlan, *, mtime_tolerance: float = 2.0) -> CopyCheck:
    root = plan.root or os.path.dirname(plan.source)

    def probe() -> tuple[list[Stat | None], str | None]:
        try:
            sources = [_stat(member.source) for member in plan.files]
        except OSError:
            sources = [None] * len(plan.files)
        if any(source is None for source in sources):
            try:
                with os.scandir(root) as entries:
                    next(entries, None)
            except OSError as exc:
                return sources, str(exc)
        return sources, None

    try:
        sources, cause = _bounded(plan.source + ":stat", probe)
    except TimeoutError:
        sources = [None] * len(plan.files)
        cause = f"the check timed out after {SOURCE_CHECK_SECONDS:g} seconds"
    destinations = [_stat(member.destination) for member in plan.files]
    if cause is not None:
        records = [_record(member.destination) for member in plan.files]
        recorded = all(
            record
            and record.get("source") == member.source
            and isinstance(record.get("size"), int)
            and record.get("mtime") is not None
            and record.get("copiedAt")
            for member, record in zip(plan.files, records, strict=True)
        )
        if recorded and all(
            destination is not None and record is not None and destination.size == record["size"]
            for destination, record in zip(destinations, records, strict=True)
        ):
            return CopyCheck(
                True, f"Started from this machine's copy. {root} cannot be checked: {cause}."
            )
        if not recorded and any(destination is not None for destination in destinations):
            return CopyCheck(
                False,
                refusal=(
                    "was made before Eugene recorded its source, and "
                    f"{root} cannot be checked now: {cause}."
                ),
            )
        return CopyCheck(
            False, refusal=f"cannot be used because {root} cannot be checked: {cause}."
        )
    for member, source, destination in zip(plan.files, sources, destinations, strict=True):
        if source is None or destination is None or source.size != destination.size:
            return CopyCheck(False)
        if abs(source.mtime - destination.mtime) > mtime_tolerance:
            # Existing Ollama hard links have the blob's timestamp.
            try:

                def same_blob(member: CopyMember = member) -> bool:
                    blob = matching_ollama_blob(member.source)
                    return blob is not None and os.path.samefile(blob, member.destination)

                same = _bounded(member.source + ":samefile", same_blob)
            except (OSError, TimeoutError):
                same = False
            if not same:
                return CopyCheck(False)
    for member, source in zip(plan.files, sources, strict=True):
        assert source is not None
        saved_record = _record(member.destination)
        if not saved_record or any(
            saved_record.get(key) != value
            for key, value in (
                ("source", member.source),
                ("size", source.size),
                ("mtime", source.mtime),
            )
        ):
            _save_source(member, source, [m.source for m in plan.files])
    return CopyCheck(True)


def total_size(plan: CopyPlan) -> int:
    """Every file of the model summed, as the share has them; raises
    OSError when one cannot be read."""
    return sum(os.path.getsize(member.source) for member in plan.files)


def member_is_current(member: CopyMember, *, mtime_tolerance: float = 2.0) -> bool:
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
    source = _stat(member.source)
    destination = _stat(member.destination)
    if source is None or destination is None:
        return False
    if source.size != destination.size:
        return False
    if abs(source.mtime - destination.mtime) <= mtime_tolerance:
        return True
    blob = matching_ollama_blob(member.source)
    return blob is not None and os.path.samefile(blob, member.destination)


_ollama_hashes: dict[tuple[str, int, int], str] = {}


def _file_digest(path: str) -> str:
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key not in _ollama_hashes:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            while block := stream.read(CHUNK_BYTES):
                digest.update(block)
        if len(_ollama_hashes) > 128:
            _ollama_hashes.clear()
        _ollama_hashes[key] = digest.hexdigest()
    return _ollama_hashes[key]


def matching_ollama_blob(source: str) -> Path | None:
    """An identical blob in this node's store, proved by SHA-256."""
    from .ollama_store import local_models

    try:
        size = os.path.getsize(source)
        for _, model in local_models():
            for blob in (model.blob, model.projector):
                if blob is None or blob.stat().st_size != size:
                    continue
                expected = blob.name.removeprefix("sha256-")
                if _file_digest(source) == expected and _file_digest(str(blob)) == expected:
                    return blob
    except (OSError, ValueError):
        return None
    return None


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
    # A same-volume blob link uses no second allocation. A failed link still
    # falls through to the normal copy and its live headroom checks.
    if all(
        (blob := matching_ollama_blob(member.source)) is not None
        and os.stat(blob).st_dev == os.stat(_existing_parent(Path(member.destination))).st_dev
        for member in plan.files
    ):
        size = 0
    free = free_bytes(os.path.dirname(plan.destination) or ".")
    if free is None:
        return 0
    return (size + settings.min_free_bytes) - free


def _existing_parent(path: Path) -> Path:
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


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
    files: int = 1
    files_copied: int = 0
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
    """Copy every file of the model, leaving nothing partial behind.

    Blocking; the caller runs it in a thread. Raises `CopyAborted` with
    a plain-language reason, or an OSError for anything the OS refused.

    Headroom is re-checked while copying, not only before: a 25 GB
    transfer takes minutes and something else can fill the disk during
    it. A breach aborts and removes the partial, because the alternative
    is a feature meant to speed up a node being the reason its disk
    filled up.

    A file already current is kept (a copy stopped halfway carries on),
    and counts as copied; the runtime opens the copy only once every file
    is (`copy_is_current`), so a set is never used half made.
    """
    files = plan.files
    state.files = len(files)
    copied = 0
    for member in files:
        if member_is_current(member):
            st = _stat(member.destination)
            copied += st.size if st is not None else 0
            state.files_copied += 1
            state.note(copied, time.perf_counter())
            continue
        copied = _copy_one(member, settings, state, copied, should_cancel)
        state.files_copied += 1
    for member in files:
        source = _stat(member.source)
        if source is not None:
            _save_source(member, source, [m.source for m in files])


def _copy_one(
    member: CopyMember,
    settings: CopySettings,
    state: CopyState,
    copied: int,
    should_cancel: Callable[[], bool] | None,
) -> int:
    """One file, from a temp name renamed after fsync; `copied` is the
    set's running total, returned grown by this file."""
    os.makedirs(os.path.dirname(member.destination) or ".", exist_ok=True)
    partial = member.partial
    blob = matching_ollama_blob(member.source)
    if blob is not None:
        try:
            if should_cancel is not None and should_cancel():
                raise CopyAborted("the copy was cancelled")
            os.link(blob, partial)
            os.replace(partial, member.destination)
            copied += os.path.getsize(member.source)
            state.note(copied, time.perf_counter())
            return copied
        except OSError:
            _remove_quietly(partial)
    started = time.perf_counter()
    first = copied
    try:
        with open(member.source, "rb") as src, open(partial, "wb") as dst:
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
                state.note(copied, time.perf_counter())
                if chunks % HEADROOM_CHECK_EVERY == 0:
                    remaining = (state.total_bytes or copied) - copied
                    free = free_bytes(os.path.dirname(member.destination) or ".")
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
    shutil.copystat(member.source, partial)
    os.replace(partial, member.destination)
    state.note(copied, time.perf_counter())
    log.info(
        "copied %s to %s (%.1f GB in %.0fs)",
        member.source,
        member.destination,
        (copied - first) / GIB,
        time.perf_counter() - started,
    )
    return copied


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
            if name.endswith((PARTIAL_SUFFIX, SOURCE_SUFFIX)):
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
        _remove_quietly(path + SOURCE_SUFFIX)
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
        _remove_quietly(path + SOURCE_SUFFIX)
        result.deleted.append(path)
        result.bytes_freed += st.size if st is not None else 0
    _prune_empty_dirs(directory, keep)
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
        _remove_quietly(path + SOURCE_SUFFIX)
        result.deleted.append(path)
        result.bytes_freed += st.size if st is not None else 0
        log.info("removed local copy %s: no runtime points at it any more", path)
    _prune_empty_dirs(directory, keep)
    return result


def _prune_empty_dirs(directory: str | None, keep: Iterable[str] = ()) -> None:
    """Leave no empty scaffolding behind.

    The copy directory itself stays: the operator made that choice, and
    removing it would make the next copy recreate it for no reason.

    **`keep` is not optional in practice, and the live install proved
    it.** A copy creates its directory and then opens a temp file in it,
    and the reconcile that runs on the declaration list walks the same
    tree. Between those two instants the directory is empty and looks
    like scaffolding -- so the first boot with copying switched on
    deleted the directory out from under the copy, which failed with
    `No such file or directory` on a path it had just created. A
    directory we are about to fill is not empty.
    """
    if not directory or not os.path.isdir(directory):
        return
    protected = {os.path.normcase(os.path.abspath(os.path.dirname(p) or directory)) for p in keep}
    for root, dirs, files in os.walk(directory, topdown=False):
        if os.path.abspath(root) == os.path.abspath(directory):
            continue
        here = os.path.normcase(os.path.abspath(root))
        if any(p == here or p.startswith(here + os.sep) for p in protected):
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
    note: str | None = None
    refusal: str | None = None

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
    check = check_copy(plan) if plan is not None else CopyCheck(False)
    if plan is not None and check.usable:
        return LocalPath(path=plan.destination, source="copy", note=check.note)
    resolution = resolve_model_path(declared, rules, expand=expand)
    if rule_source is not None:
        return LocalPath(
            path=resolution.local_path, source=rule_source(resolution.rule), refusal=check.refusal
        )
    return LocalPath(
        path=resolution.local_path,
        source="same_path" if resolution.rule is None else "override",
        refusal=check.refusal,
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
