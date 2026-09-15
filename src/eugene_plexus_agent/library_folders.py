"""The Library's folders, as this node reaches them (2026-09-14).

M11 put a path rule on each node's agent: `pathMappings`, one `{from,
to}` per folder per node, typed by hand. This module keeps that
mechanism and changes who states the rule. The Library's folder record
now carries `mounts` -- where other machines find the folder, one per OS
shape -- and this node **inherits** the mount of its own shape as a rule
`folder.path -> mount`. `pathMappings` is demoted to the overrides: the
one machine that mounts a share somewhere else says so, and nobody else
types anything.

Three things live here:

* **A copy of the folder list, on disk.** The install-wide lookup that
  finds the library from a worker spends the caller's credential and
  nothing else (`install_proxy`, deliberately), so there is no
  background loop -- a spawn has no caller. Instead every request-scoped
  path that already talks to the library (admission, create, update,
  start, the Test button, the check endpoint) refreshes this copy, and a
  spawn reads it. `library_folders.json` beside `agent.yaml`, so an
  agent that restarts with the library down still resolves.
* **The effective rules**: this node's overrides first, then the
  inherited rules for folders no override names. `resolve_model_path`
  gives ties to the first rule listed, so an override for a folder wins
  over the folder's own mount without a second precedence mechanism.
* **The rule itself**: a declared model must be under a Library folder.
  `folder_for` answers it component-wise with the folder's own shape,
  exactly as a mapping's `from` is matched -- a folder is a rule with no
  `to`.

What is deliberately not here: any live read at spawn time, and any
refusal when the list has never been fetched. A worker in its first
minute and a single box whose library has not started both degrade to
M11's behaviour and say so.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ._generated.models import FolderReachSource, LibraryFolderReach, LibraryFolderStatus
from .model_paths import PathRule, _parts, is_windows_shaped, match, resolve_model_path

log = logging.getLogger(__name__)

FOLDERS_FILE = "library_folders.json"

# What the runtime supervisor is handed: a callable returning the
# inherited rules, read live at every plan and compose like the config.
RulesProvider = Callable[[], Sequence[PathRule]]


@dataclass(frozen=True)
class FolderRecord:
    """One `LibraryFolder` as the library reported it."""

    path: str
    mounts: tuple[str, ...] = ()

    def mount_for(self, *, windows: bool) -> str | None:
        """The first mount of this host's shape, or None."""
        for mount in self.mounts:
            if is_windows_shaped(mount) == windows:
                return mount
        return None

    def as_rule(self) -> PathRule:
        """A folder is a rule with no `to`: the same matcher decides
        whether a declared path is under it."""
        return PathRule(source=self.path, target=self.path)


def parse_folders(value: Any) -> list[FolderRecord]:
    """Folder records from the wire or the file, leniently: a bare string
    is a folder with no mounts, junk is skipped with a log line."""
    if not isinstance(value, list):
        return []
    out: list[FolderRecord] = []
    for index, item in enumerate(value):
        if isinstance(item, str) and item.strip():
            out.append(FolderRecord(path=item.strip()))
            continue
        if isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"].strip():
            raw = item.get("mounts")
            mounts = (
                tuple(m.strip() for m in raw if isinstance(m, str) and m.strip())
                if isinstance(raw, list)
                else ()
            )
            out.append(FolderRecord(path=item["path"].strip(), mounts=mounts))
            continue
        log.warning("library folder entry %d is malformed (%r); ignoring it", index, item)
    return out


def identity(source: str) -> tuple[str, ...]:
    """What makes two `from`s the same directory: their comparable parts
    under the convention the string's own shape names."""
    return _parts(source, windows=is_windows_shaped(source))[0]


def host_is_windows() -> bool:
    return os.name == "nt"


class LibraryFolderCache:
    """This node's copy of the library's folder list, persisted."""

    def __init__(self, path: Path, *, windows: bool | None = None) -> None:
        self._path = path
        self._folders: list[FolderRecord] | None = None
        self._fetched_at: float | None = None
        self._library_url: str | None = None
        self._windows = host_is_windows() if windows is None else windows

    # --- state -------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def folders(self) -> list[FolderRecord] | None:
        """The folders, or None when the library has never been read."""
        return None if self._folders is None else list(self._folders)

    @property
    def known(self) -> bool:
        return self._folders is not None

    @property
    def library_url(self) -> str | None:
        return self._library_url

    def age_seconds(self, *, now: float | None = None) -> int | None:
        if self._fetched_at is None:
            return None
        return max(0, int((time.time() if now is None else now) - self._fetched_at))

    def load(self) -> None:
        """Read the copy from disk. A missing or malformed file is "never
        fetched", never an error: the agent must start regardless."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning(
                "could not read %s (%s); treating the folder list as unknown", self._path, exc
            )
            return
        if not isinstance(raw, dict):
            return
        self._folders = parse_folders(raw.get("folders"))
        fetched = raw.get("fetchedAt")
        self._fetched_at = float(fetched) if isinstance(fetched, int | float) else None
        url = raw.get("libraryUrl")
        self._library_url = url if isinstance(url, str) else None

    def update(
        self,
        folders: Sequence[FolderRecord],
        *,
        library_url: str | None,
        now: float | None = None,
    ) -> None:
        """A fresh answer from the library: hold it and write it down."""
        self._folders = list(folders)
        self._fetched_at = time.time() if now is None else now
        self._library_url = library_url
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "folders": [{"path": f.path, "mounts": list(f.mounts)} for f in self._folders],
                "fetchedAt": self._fetched_at,
                "libraryUrl": library_url,
            }
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            # Degraded mode: the copy in memory is what this process
            # needs; the file is for the next one.
            log.warning(
                "could not write %s (%s); the folder list is held in memory only", self._path, exc
            )

    # --- the rules ---------------------------------------------------------

    def inherited_rules(self) -> list[PathRule]:
        """One rule per folder that carries a mount of this host's shape
        and whose mount is not the folder's own path."""
        rules: list[PathRule] = []
        for folder in self._folders or []:
            mount = folder.mount_for(windows=self._windows)
            if mount is None or identity(mount) == identity(folder.path):
                continue
            rules.append(PathRule(source=folder.path, target=mount))
        return rules

    def folder_for(self, declared: str) -> FolderRecord | None:
        """The Library folder `declared` lies under, longest first."""
        best: tuple[int, FolderRecord] | None = None
        for folder in self._folders or []:
            if match(folder.as_rule(), declared) is None:
                continue
            depth = len(identity(folder.path))
            if best is None or depth > best[0]:
                best = (depth, folder)
        return best[1] if best else None


def effective_rules(overrides: Sequence[PathRule], inherited: Sequence[PathRule]) -> list[PathRule]:
    """This node's overrides, then every inherited rule they do not
    shadow. Order is precedence on a tie, and the longest `from` still
    wins across the union -- `resolve_model_path` is unchanged."""
    shadowed = {identity(rule.source) for rule in overrides}
    return [*overrides, *(rule for rule in inherited if identity(rule.source) not in shadowed)]


# --- refreshing the copy ------------------------------------------------------


class FolderSource(Protocol):
    """Anything that can be asked for the library's folders."""

    async def folders(self) -> list[dict[str, Any]] | None: ...


async def refresh(cache: LibraryFolderCache, library: object | None) -> bool:
    """Ask `library` for its folders and hold the answer. False when there
    was no library to ask or it did not answer; the copy stands."""
    if library is None or not hasattr(library, "folders"):
        return False
    try:
        answer = await library.folders()
    except Exception as exc:  # a management-plane call must not fail a launch
        log.info("library folders could not be read (%s); keeping the last copy", exc)
        return False
    if answer is None:
        return False
    cache.update(parse_folders(answer), library_url=getattr(library, "base_url", None))
    return True


# --- the check endpoint -------------------------------------------------------


def check_reach(
    cache: LibraryFolderCache,
    overrides: Sequence[PathRule],
    library_models: Sequence[dict[str, Any]] | None,
    *,
    library_consulted: bool,
    exists: Callable[[str], bool] = os.path.exists,
    isdir: Callable[[str], bool] = os.path.isdir,
    expand: Callable[[str], str] = os.path.expanduser,
    now: float | None = None,
) -> LibraryFolderReach:
    """One row per folder: what this host would open, which rule said so,
    whether it is there, and how many of the library's models under it
    are reachable. Real filesystem calls; run it off the event loop."""
    rules = effective_rules(overrides, cache.inherited_rules())
    override_set = set(overrides)
    rows: list[LibraryFolderStatus] = []
    for folder in cache.folders or []:
        resolution = resolve_model_path(folder.path, rules, expand=expand)
        local = resolution.local_path
        rule = resolution.rule
        if rule is None:
            source = FolderReachSource.same_path
            local = expand(local)
        elif rule in override_set:
            source = FolderReachSource.override
        else:
            source = FolderReachSource.inherited
        present = bool(exists(local))
        is_dir = bool(isdir(local)) if present else None

        under: int | None = None
        reachable: int | None = None
        if library_models is not None:
            under = 0
            reachable = 0
            for model in library_models:
                path = model.get("path")
                if not isinstance(path, str) or match(folder.as_rule(), path) is None:
                    continue
                under += 1
                if exists(resolve_model_path(path, rules, expand=expand).local_path):
                    reachable += 1

        problem: str | None = None
        if not present:
            problem = f"{local} does not exist on this host"
        elif is_dir is False:
            problem = f"{local} is not a directory"
        elif under and reachable is not None and reachable < under:
            problem = (
                f"{under - reachable} of {under} library models under {folder.path} are not "
                f"at their resolved path here"
            )

        rows.append(
            LibraryFolderStatus(
                path=folder.path,
                localPath=local,
                source=source,
                mount=(
                    rule.target
                    if rule is not None and source is FolderReachSource.inherited
                    else None
                ),
                override=(
                    rule.as_mapping()
                    if rule is not None and source is FolderReachSource.override
                    else None
                ),
                exists=present,
                isDirectory=is_dir,
                modelsUnder=under,
                modelsReachable=reachable,
                problem=problem,
            )
        )
    return LibraryFolderReach(
        libraryConsulted=library_consulted,
        folderListAgeSeconds=0 if library_consulted else cache.age_seconds(now=now),
        libraryUrl=cache.library_url,
        folders=rows,
    )


def not_in_library_detail(model_path: str) -> str:
    """The 400's sentence, in one place so the route and the tests agree."""
    return (
        f"{model_path} is not under any Library folder. Add the directory that holds it to the "
        f"Library (Library -> Folders), then scan. A node runs only what the Library catalogues."
    )


__all__ = [
    "FOLDERS_FILE",
    "FolderRecord",
    "FolderSource",
    "LibraryFolderCache",
    "RulesProvider",
    "check_reach",
    "effective_rules",
    "host_is_windows",
    "identity",
    "not_in_library_detail",
    "parse_folders",
    "refresh",
]
