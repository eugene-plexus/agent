"""What v0.1 saved in `agent.yaml`, converted once.

Old saved data is rewritten in the current format ONCE. The bytes it held are kept
beside it (`agent.yaml.v0.1.bak`, owner-only like the file: it holds the passphrase's
hash), the rewrite carries `stateVersion: 2`, and from then on only the current format
is read: nothing else in the agent knows a v0.1 shape. The release after v0.2 deletes this
module and its one call in `state.py`.

**Flash attention was a checkbox.** It is now On, Off or not set, where not set leaves
llama.cpp's own choice (auto). A runtime saved with the checkbox keeps what it did:
`true` always sent `on`, so it becomes `"on"`; `false` always sent nothing, which left
auto, so the key is dropped, never turned into `"off"`, which would switch flash
attention off in every runtime saved before. The Library converts its saved profiles the
same way, in its own files.

A file already stamped is not looked at again, and a file with no checkbox in it is not
rewritten at all (the stamp arrives with its next ordinary write). A conversion that
cannot finish raises `ConversionFailed`, whose message names the cause and says the file
was left as it was. `AgentState.load_or_degrade` degrades on it as on any file it cannot
read: the agent comes up on defaults, keeps a copy, and says why on `/healthz`. The rewrite
is a temp file and a rename, so a crash leaves the old bytes or the new ones and the next
start begins again from whichever it finds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ._private_files import write_private

#: Written to `agent.yaml` by every write of this release: the format it reads.
STATE_VERSION = 2

#: Appended to a file's name for the copy kept before it was converted.
BACKUP_SUFFIX = ".v0.1.bak"

_MOST_BACKUPS = 100


class ConversionFailed(Exception):
    """Saved data from v0.1 could not be converted; the file is as it was."""


def keep_backup(path: Path, data: bytes) -> Path:
    """Keep `data`, what `path` held before it was converted, beside it.

    A copy that is already there and says the same thing is the one kept (a start
    that stopped after this and began again). One that says something else is never
    written over, because it may be the only copy of what that was: the next free
    name, `.v0.1.bak.1` and so on, holds this one.
    """
    for number in range(_MOST_BACKUPS):
        candidate = path.with_name(path.name + BACKUP_SUFFIX + (f".{number}" if number else ""))
        try:
            if candidate.read_bytes() == data:
                return candidate
        except FileNotFoundError:
            write_private(candidate, data)
            return candidate
    raise OSError(f"{_MOST_BACKUPS} different copies of {path.name} are already kept beside it")


def _failed(path: Path, cause: Exception, backup: Path | None = None) -> ConversionFailed:
    kept = f" The copy from before is {backup}." if backup is not None else ""
    return ConversionFailed(
        f"{path}: what v0.1 saved could not be converted ({type(cause).__name__}: {cause}). "
        f"It is left as it was.{kept} Fix the cause and restart."
    )


def _flash_attention_three_states(flags: dict[str, Any]) -> bool:
    """One runtime's `flags`, in place. Whether anything changed."""
    value = flags.get("flashAttention")
    if value is True:
        flags["flashAttention"] = "on"
        return True
    if value is False:
        del flags["flashAttention"]
        return True
    return False


def convert(path: Path, original: bytes, raw: dict[str, Any]) -> dict[str, Any]:
    """`raw`, what `original` (the bytes of `path`) parsed to, in the current format.

    Returns `raw` itself when there is nothing to convert. Otherwise the converted
    mapping, after `original` is kept and the rewrite, stamped, is in place.
    """
    stamp = raw.get("stateVersion")
    if type(stamp) is int and stamp >= STATE_VERSION:
        return raw
    runtimes = raw.get("runtimes")
    changed = False
    if isinstance(runtimes, list):
        for entry in runtimes:
            if isinstance(entry, dict) and isinstance(entry.get("flags"), dict):
                changed = _flash_attention_three_states(entry["flags"]) or changed
    if not changed:
        return raw
    backup: Path | None = None
    try:
        backup = keep_backup(path, original)
        converted = {**raw, "stateVersion": STATE_VERSION}
        write_private(path, yaml.safe_dump(converted, sort_keys=True, default_flow_style=False))
    except Exception as exc:
        raise _failed(path, exc, backup) from exc
    return converted
