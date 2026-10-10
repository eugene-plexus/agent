"""What an engine read off its own files, as a provenance file records it (LS7).

library-sources-and-engines.md §6.8 (B22 replaced, B26) and §6.10. The
adapter reads its entry file as this node reaches it (`PreparedFacts`);
the Library spells paths its own way, and a provenance file lists the
model's files relative to its own folder. This turns the one into the
other, for both the preparation that just made the model and the
console's *Add a prepared model* (`POST /v1/engines/{engine}/prepared/inspect`).
"""

from __future__ import annotations

import contextlib
import ntpath
import os
import posixpath
from pathlib import Path
from typing import Any

from .engines.base import PreparedFacts
from .model_paths import is_windows_shaped


def library_spelling(local: Path, *, entry_local: Path, entry_declared: str) -> str | None:
    """`local` as the Library spells it: its place relative to the entry's
    folder on this node, applied to the entry as the Library spells it.
    None when the two are not on one drive here, so no relative place
    exists."""
    try:
        relative = os.path.relpath(local, entry_local.parent)
    except ValueError:
        return None
    parts = Path(relative).parts
    if is_windows_shaped(entry_declared):
        return ntpath.normpath(ntpath.join(ntpath.dirname(entry_declared), *parts))
    return posixpath.normpath(posixpath.join(posixpath.dirname(entry_declared), *parts))


def facts_fields(facts: PreparedFacts, *, folder: Path) -> dict[str, Any]:
    """The provenance fields the facts fill, with each file relative to
    `folder` (the one the provenance file is written into) and sized here."""
    out: dict[str, Any] = {}
    for key, value in (
        ("title", facts.title),
        ("architecture", facts.architecture),
        ("quantization", facts.quantization),
        ("contextLength", facts.context_length),
        ("mode", facts.mode),
    ):
        if value is not None:
            out[key] = value
    files = []
    for path, shared in facts.files:
        try:
            relative = Path(os.path.relpath(path, folder)).as_posix()
        except ValueError:  # another drive: not a file of this folder's model
            continue
        item: dict[str, Any] = {"path": relative}
        with contextlib.suppress(OSError):
            item["sizeBytes"] = path.stat().st_size
        if shared:
            item["shared"] = True
        files.append(item)
    if files:
        out["files"] = files
    return out


def draft(
    facts: PreparedFacts, *, engine: str, entry_local: Path, entry_declared: str
) -> dict[str, Any]:
    """A `PreparedProvenance` for *Add a prepared model*: the entry as the
    Library spells it, its files relative to the entry's folder (where the
    provenance file goes unless the person says otherwise), and the source
    model the configuration names."""
    body: dict[str, Any] = {"engine": engine, "entry": entry_declared}
    source: dict[str, Any] = {}
    if facts.source_file is not None:
        spelled = library_spelling(
            facts.source_file, entry_local=entry_local, entry_declared=entry_declared
        )
        if spelled is not None:
            source["path"] = spelled
    if facts.repo_id:
        source["repoId"] = facts.repo_id
    if facts.hub_file:
        source["file"] = facts.hub_file
    if source:
        body["source"] = source
    body.update(facts_fields(facts, folder=entry_local.parent))
    return body
