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
  to, renamed in, or deleted from. Inside the copy directory, a file is
  Eugene's copy only when the record Eugene writes beside it
  (`<file>.source.json`, written before the copy takes its name) names
  the file it copies (agent#54). Cleanup deletes nothing else, a copy
  never replaces a file without one, and no record is ever written for a
  file Eugene did not copy.

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

#: What the copy folder's help and status say about uninstalling. Remove it
#: when specs#136 lands: until then the uninstaller's purge option (install.sh
#: `--purge-downloads`, install.ps1 `-PurgeDownloads`, the macOS dialog's
#: downloads choice) deletes the whole copy folder, not only Eugene's copies.
UNINSTALL_PURGE_WARNING = (
    "Uninstalling Eugene with the option to delete downloads removes this whole folder, "
    "so keep no other files in it."
)

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


def _within(path: str, folder: str) -> bool:
    try:
        return path != folder and os.path.commonpath([path, folder]) == folder
    except ValueError:  # another drive, or one path relative
        return False


def library_overlap(directory: str, library_roots: Iterable[str]) -> str | None:
    """How a copy directory overlaps a Library folder on this machine, as a
    phrase that follows the directory's name, or None when it does not.

    `library_roots` are the folders as this machine opens them. A copy
    directory that is a Library folder, lies inside one, or holds one is
    refused (agent#54): copies would land in the Library, and the folder
    would mix the person's models with Eugene's copies.
    """
    chosen = os.path.normcase(os.path.abspath(os.path.expanduser(directory)))
    for root in library_roots:
        folder = os.path.normcase(os.path.abspath(os.path.expanduser(root)))
        if chosen == folder:
            return f"is the Library folder {root}"
        if _within(chosen, folder):
            return f"is inside the Library folder {root}"
        if _within(folder, chosen):
            return f"holds the Library folder {root}"
    return None


def library_roots_here(
    settings: CopySettings,
    rules: Sequence[PathRule],
    *,
    expand: Callable[[str], str] = os.path.expanduser,
) -> list[str]:
    """Every Library folder (`settings.folder_roots`) where this machine
    opens it, through the effective rules; a folder this machine has no
    form of (the other OS's shape, no mount here) is left out."""
    windows = os.name == "nt"
    out: list[str] = []
    for folder in settings.folder_roots:
        local = resolve_model_path(folder, rules, expand=expand).local_path
        if is_windows_shaped(local) == windows:
            out.append(local)
    return out


def copy_dir_overlap(
    settings: CopySettings,
    rules: Sequence[PathRule],
    *,
    expand: Callable[[str], str] = os.path.expanduser,
) -> str | None:
    """How the copy folder overlaps a Library folder this machine opens, as
    `library_overlap` says it, or None. While it does, nothing is copied."""
    if not settings.directory:
        return None
    return library_overlap(settings.directory, library_roots_here(settings, rules, expand=expand))


def _inside_copy_dir(path: str, directory: str) -> bool:
    return _within(
        os.path.normcase(os.path.abspath(path)),
        os.path.normcase(os.path.abspath(os.path.expanduser(directory))),
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
    resolution = resolve_model_path(declared, rules, expand=expand)
    source = resolution.local_path
    relative = relative_name(declared, rules)
    if not relative:
        return None
    destination = os.path.join(settings.directory, *relative.split("/"))
    if os.path.normcase(os.path.abspath(source)) == os.path.normcase(os.path.abspath(destination)):
        return None
    if not _inside_copy_dir(destination, settings.directory):
        # A `..` in the declared path (review S5): never write outside the
        # folder the operator chose.
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
    matched_root = [root] if folders or resolution.rule is not None else []
    if library_overlap(
        settings.directory, [*matched_root, *library_roots_here(settings, rules, expand=expand)]
    ):
        # Review S4: a copy folder in, equal to or above a Library folder
        # copies nothing, whatever let the value be saved.
        return None
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
        try:
            inside = os.path.relpath(path, layout_root)
        except ValueError:  # on another drive (review S3)
            return None
        if os.path.isabs(inside) or inside == ".." or inside.startswith(".." + os.sep):
            # A file outside the model's Library folder cannot keep its
            # place in the copy: the model runs from where it is.
            return None
        member_destination = os.path.join(settings.directory, inside)
        if not _inside_copy_dir(member_destination, settings.directory):
            return None
        members.append(CopyMember(path, member_destination))
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


#: The most of a record that is read. A record holds one path, two numbers,
#: a time and the model's member list, so even a prepared model of tens of
#: thousands of files stays far below this; the cap bounds what a file put
#: in the copy folder can make every walk read (agent#54 review).
RECORD_MAX_BYTES = 4 * 1024 * 1024


def _read_record(destination: str) -> dict[str, Any] | None:
    """The record beside `destination`, or None when there is none or it
    is not one: too big, not JSON, nested deeper than the parser goes."""
    try:
        with open(destination + SOURCE_SUFFIX, "rb") as stream:
            data = stream.read(RECORD_MAX_BYTES + 1)
        if len(data) > RECORD_MAX_BYTES:
            return None
        raw = json.loads(data.decode("utf-8"))
        return raw if isinstance(raw, dict) else None
    except (OSError, ValueError, RecursionError):
        return None


def _names_a_source(record: dict[str, Any] | None) -> bool:
    source = record.get("source") if record else None
    return isinstance(source, str) and bool(source.strip())


def _proves(record: dict[str, Any] | None, path: str) -> bool:
    """`record` names a source and the size it records is the size of the
    file at `path` now. Every record is written with the size of the bytes
    about to take the name (`_mark`) or of a copy that matches its source
    (`check_copy`), so a size that differs means the file is not the copy
    the record was written for."""
    if not _names_a_source(record):
        return False
    assert record is not None
    size = record.get("size")
    if not isinstance(size, int) or isinstance(size, bool):
        return False
    try:
        return os.path.getsize(path) == size
    except OSError:
        return False


def is_eugenes_copy(path: str) -> bool:
    """Whether Eugene made the file at `path` (agent#54).

    The one proof: the record Eugene writes beside every file it copies,
    before the file takes its name, is there, names the file it copies,
    and records this file's size. Size, time and name alone prove nothing
    -- a person who copied a model with its timestamp has a file that
    matches all three -- so a file without the record is never deleted,
    replaced or claimed.
    """
    return _proves(_record(path), path)


def _save_source(member: CopyMember, source: Stat, members: Sequence[str] = ()) -> str | None:
    """Write or refresh the record of `member`; the OS's reason when it
    could not be written, else None."""
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
            return None
        return _write_source(member, source, members)


def _write_source(member: CopyMember, source: Stat, members: Sequence[str]) -> str | None:
    payload: dict[str, Any] = {
        "source": member.source,
        "size": source.size,
        "mtime": source.mtime,
        "copiedAt": datetime.now(UTC).isoformat(),
    }
    if members:
        payload["members"] = list(members)
    return _write_record(member.destination, payload)


def _write_record(destination: str, payload: dict[str, Any]) -> str | None:
    """Write `payload` as the record beside `destination`, whole or not at
    all; the OS's reason when it could not, else None."""
    path = destination + SOURCE_SUFFIX
    partial = path + PARTIAL_SUFFIX
    try:
        with open(partial, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except OSError as exc:
        _remove_quietly(partial)
        log.warning("could not record the source of %s: %s", destination, exc)
        return str(exc)
    return None


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
        if not _proves(saved_record, member.destination):
            # A file with no record may be opened, since it is the same
            # file, but is never claimed: Eugene did not copy it, or cannot
            # prove it did (agent#54).
            continue
        assert saved_record is not None
        if any(
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
    members = [m.source for m in files]
    state.files = len(files)
    copied = 0
    for member in files:
        if member_is_current(member):
            st = _stat(member.destination)
            copied += st.size if st is not None else 0
            state.files_copied += 1
            state.note(copied, time.perf_counter())
            continue
        copied = _copy_one(member, settings, state, copied, should_cancel, members)
        state.files_copied += 1
    for member in files:
        # Refresh Eugene's own records (the set may have grown a member);
        # a current file without one stays unclaimed (agent#54). Only while
        # the source still has the copy's size, so the record keeps proving it.
        if not is_eugenes_copy(member.destination):
            continue
        source = _stat(member.source)
        held = _stat(member.destination)
        if source is not None and held is not None and source.size == held.size:
            _save_source(member, source, members)


def _copy_one(
    member: CopyMember,
    settings: CopySettings,
    state: CopyState,
    copied: int,
    should_cancel: Callable[[], bool] | None,
    members: Sequence[str] = (),
) -> int:
    """One file, from a temp name renamed after fsync; `copied` is the
    set's running total, returned grown by this file.

    A file already at the destination that is not Eugene's copy is never
    replaced (agent#54), and the record that proves the copy is Eugene's
    is written before the copy takes its name, so no whole file is ever
    left without it."""
    _refuse_a_file_not_ours(member)
    os.makedirs(os.path.dirname(member.destination) or ".", exist_ok=True)
    partial = member.partial
    blob = matching_ollama_blob(member.source)
    if blob is not None:
        previous: dict[str, Any] | None = None
        marked = False
        try:
            if should_cancel is not None and should_cancel():
                raise CopyAborted("the copy was cancelled")
            os.link(blob, partial)
            previous = _mark(member, members, partial)
            marked = True
            os.replace(partial, member.destination)
            copied += os.path.getsize(member.source)
            state.note(copied, time.perf_counter())
            return copied
        except OSError:
            _remove_quietly(partial)
            if marked:
                _unmark(member, previous)
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
    previous = _mark(member, members, partial)
    try:
        os.replace(partial, member.destination)
    except OSError:
        _remove_quietly(partial)
        _unmark(member, previous)
        raise
    state.note(copied, time.perf_counter())
    log.info(
        "copied %s to %s (%.1f GB in %.0fs)",
        member.source,
        member.destination,
        (copied - first) / GIB,
        time.perf_counter() - started,
    )
    return copied


def _refuse_a_file_not_ours(member: CopyMember, partial: str | None = None) -> None:
    """Abort when a file that is not Eugene's copy is at the destination,
    removing `partial` if given; the file itself is left alone (agent#54)."""
    if os.path.lexists(member.destination) and not is_eugenes_copy(member.destination):
        if partial is not None:
            _remove_quietly(partial)
        raise CopyAborted(
            f"not copied to this machine: {member.destination} is already there and is not "
            "a copy Eugene made, so Eugene leaves it alone"
        )


def _mark(member: CopyMember, members: Sequence[str], partial: str) -> dict[str, Any] | None:
    """Write the record that makes `member` Eugene's copy, while its bytes
    are still under the partial name, and answer the record it replaced.

    The destination is checked again first: a copy takes minutes, and a
    file put there meanwhile is not replaced (review S2). The record's size
    is the partial's, so it proves exactly the bytes about to take the
    name; a source that changed size while it was copied stops the copy.
    On failure the partial is removed and the cause given, since a copy
    with no record could never be cleaned up."""
    _refuse_a_file_not_ours(member, partial)
    previous = _record(member.destination)
    source = _stat(member.source)
    written = _stat(partial)
    if source is None or written is None:
        error: str | None = "its source could not be read"
    elif source.size != written.size:
        error = "the source changed while it was copied"
    else:
        error = _save_source(member, source, members)
    if error is not None:
        _remove_quietly(partial)
        raise CopyAborted(
            f"not copied to this machine: the record that marks {member.destination} as "
            f"Eugene's copy could not be written: {error}"
        )
    return previous


def _unmark(member: CopyMember, previous: dict[str, Any] | None) -> None:
    """Undo `_mark` after the rename failed. A record with no copy could
    later vouch for a file put there, so it goes; the record of a copy
    still in place (a running engine holds it open) is put back, or that
    copy would stop being provably Eugene's."""
    record = member.destination + SOURCE_SUFFIX
    if (
        os.path.lexists(member.destination)
        and previous is not None
        and _write_record(member.destination, previous) is None
    ):
        return
    if not _remove_quietly(record):
        log.warning("could not remove %s after the copy failed", record)


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
    #: Files in the copy directory that are not Eugene's copies, left alone.
    left_alone: int = 0
    #: Eugene's records whose copy was gone, removed (review S1).
    records_removed: list[str] = field(default_factory=list)


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


@dataclass
class _Found:
    copies: list[str] = field(default_factory=list)
    others: int = 0
    #: Destinations whose record is Eugene's but whose file is gone.
    orphans: list[str] = field(default_factory=list)


def _walk(directory: str | None) -> _Found:
    """Eugene's copies under `directory`, how many other files are there,
    and the records left beside nothing.

    Partials and records are Eugene's own working files and neither. A file
    is a copy only by `is_eugenes_copy` (agent#54): the folder is the
    operator's choice and can hold their own models. A record whose file is
    gone could vouch for a file put at that name later, so it is found to
    be removed (review S1).
    """
    found = _Found()
    if not directory or not os.path.isdir(directory):
        return found
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(PARTIAL_SUFFIX):
                continue
            path = os.path.join(root, name)
            if name.endswith(SOURCE_SUFFIX):
                destination = path[: -len(SOURCE_SUFFIX)]
                if not os.path.lexists(destination) and _is_our_record(destination):
                    found.orphans.append(destination)
                continue
            if is_eugenes_copy(path):
                found.copies.append(path)
            else:
                found.others += 1
    found.copies.sort()
    found.orphans.sort()
    return found


def _is_our_record(destination: str) -> bool:
    """The record beside `destination` has the shape Eugene writes."""
    record = _record(destination)
    return _names_a_source(record) and bool(record and record.get("copiedAt"))


def copies_on_disk(directory: str | None) -> list[str]:
    """Every copy Eugene made under the copy directory.

    Walks what we made rather than deriving it from the runtime list, so
    a copy orphaned by a deleted runtime is still found -- that is the
    case where the two disagree, and the disk is the honest answer. Only
    files with Eugene's record are listed: nothing else is ours to delete.
    """
    return _walk(directory).copies


def files_not_copied(directory: str | None) -> int:
    """How many files in the copy directory are not Eugene's copies; the
    setting says so, and cleanup leaves every one of them alone."""
    return _walk(directory).others


def _delete_copy(path: str, result: ClearResult) -> bool:
    """Delete one of Eugene's copies and its record, noting the outcome."""
    if not is_eugenes_copy(path):
        # Its record went or changed since the walk: no longer provably ours.
        result.left_alone += 1
        return False
    st = _stat(path)
    try:
        os.remove(path)
    except OSError as exc:
        result.skipped.append(Skipped(path=path, reason=str(exc)))
        return False
    if not _remove_quietly(path + SOURCE_SUFFIX):
        log.warning(
            "deleted the copy %s, but not its record; the next cleanup removes the record", path
        )
    result.deleted.append(path)
    result.bytes_freed += st.size if st is not None else 0
    return True


def _remove_orphans(found: _Found, protected: Iterable[str], result: ClearResult) -> None:
    """Remove the records whose copy is gone, except for a destination a
    caller is about to fill or a runtime holds: a copy in flight writes its
    record before its file takes the name."""
    spared = {_key(p) for p in protected}
    for destination in found.orphans:
        if _key(destination) in spared or os.path.lexists(destination):
            continue
        record = destination + SOURCE_SUFFIX
        if _remove_quietly(record):
            result.records_removed.append(record)
        else:
            log.warning("could not remove %s, a record whose copy is gone", record)


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
    held = {_key(p): r for p, r in (in_use or {}).items()}
    found = _walk(directory)
    result.left_alone = found.others
    for path in found.copies:
        runtime = held.get(_key(path))
        if runtime is not None:
            result.skipped.append(
                Skipped(path=path, runtime=runtime, reason=f"runtime {runtime!r} is using it")
            )
            continue
        _delete_copy(path, result)
    _remove_orphans(found, in_use or {}, result)
    _prune_empty_dirs(
        directory, keep=list(in_use or {}), removed=[*result.deleted, *result.records_removed]
    )
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
    need; `in_use` is protected as in `clear`. Only Eugene's own copies
    are candidates, however full the disk.
    """
    result = ClearResult()
    if not directory:
        return result
    keep = list(keep)
    protected = {_key(p) for p in keep}
    held = {_key(p): r for p, r in (in_use or {}).items()}
    found = _walk(directory)
    result.left_alone = found.others
    candidates = [
        (os.stat(p).st_mtime, p)
        for p in found.copies
        if _key(p) not in protected and os.path.exists(p)
    ]
    for _mtime, path in sorted(candidates):
        free = free_bytes(directory)
        if free is None or free >= settings.min_free_bytes:
            break
        runtime = held.get(_key(path))
        if runtime is not None:
            result.skipped.append(
                Skipped(path=path, runtime=runtime, reason=f"runtime {runtime!r} is using it")
            )
            continue
        _delete_copy(path, result)
    _remove_orphans(found, [*keep, *(in_use or {})], result)
    _prune_empty_dirs(
        directory,
        keep=[*keep, *(in_use or {})],
        removed=[*result.deleted, *result.records_removed],
    )
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
    runtime list. A file that is not Eugene's copy is never a candidate.
    """
    keep = list(keep)
    wanted_paths = {_key(p) for p in keep}
    result = ClearResult()
    held = {_key(p): r for p, r in (in_use or {}).items()}
    found = _walk(directory)
    result.left_alone = found.others
    for path in found.copies:
        key = _key(path)
        if key in wanted_paths:
            continue
        if key in held:
            result.skipped.append(
                Skipped(path=path, runtime=held[key], reason=f"runtime {held[key]!r} is using it")
            )
            continue
        if _delete_copy(path, result):
            log.info("removed local copy %s: no runtime points at it any more", path)
    _remove_orphans(found, [*keep, *(in_use or {})], result)
    _prune_empty_dirs(
        directory,
        keep=[*keep, *(in_use or {})],
        removed=[*result.deleted, *result.records_removed],
    )
    return result


def _prune_empty_dirs(
    directory: str | None, keep: Iterable[str] = (), removed: Iterable[str] = ()
) -> None:
    """Remove the folders that deleting `removed` left empty, and no other.

    Only a folder that held a copy just deleted is a candidate, walking up
    toward the copy directory while each is empty (agent#54): a folder the
    person made, or one that still holds anything, stays. `os.rmdir`
    removes only an empty folder, so a folder with a file in it is never
    removed. The copy directory itself stays: the operator made that
    choice, and removing it would make the next copy recreate it for no
    reason.

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
    top = _key(directory)
    protected = {_key(os.path.dirname(p) or directory) for p in keep}
    for path in removed:
        here = os.path.dirname(os.path.abspath(path))
        while True:
            key = os.path.normcase(here)
            try:
                inside = key != top and os.path.commonpath([key, top]) == top
            except ValueError:  # another drive
                inside = False
            if not inside or any(p == key or p.startswith(key + os.sep) for p in protected):
                break
            try:
                os.rmdir(here)
            except OSError:
                break  # not empty, or not ours to remove
            here = os.path.dirname(here)


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
    "copy_dir_overlap",
    "copy_file",
    "copy_is_current",
    "evict_for_headroom",
    "files_not_copied",
    "free_bytes",
    "headroom_shortfall",
    "is_eugenes_copy",
    "library_overlap",
    "library_roots_here",
    "plan_for",
    "relative_name",
    "remove_unwanted",
    "resolve_local_path",
    "settings_from_config",
    "wanted",
]
