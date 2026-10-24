"""Programs an owner approved for Eugene's own account (row 8, part 8c).

Call R8-EXEC-TRUST, Choice 3. `exec_trust.assess` says `needs_approval`
for a program that one account besides root and Eugene's can change, and
that no group or others can. An owner approves it here (the console's "Run
it", `POST /v1/exec-approvals`), and it runs while the approval holds.

An approval binds the path that runs, that other account's uid, the file's
SHA-256, and every file, folder and link the check walked with its owner
and mode. The first check that finds any of them different drops it, and
the runtime asks again. **It trusts that account**: the hash and the
chain catch a change made after the approval; they are no boundary
against the account that owns the file.

Kept in `exec-approvals.json` in the agent's own state folder, written
owner-only. Approvals are read only while that file and every folder above
it can be changed by root and Eugene's account alone (the same walk as a
program's), so nobody else can approve by writing the file.
"""

from __future__ import annotations

import logging
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from . import exec_trust
from ._generated.models import (
    ExecApproval,
    ExecApprovalList,
    ExecChainItem,
    ExecTrust,
    ExecTrustKind,
)
from ._private_files import write_private

__all__ = [
    "FILE_NAME",
    "ExecApproval",
    "NotApprovable",
    "approve",
    "configure",
    "forget",
    "holds",
    "listed",
    "remove",
    "view",
]

log = logging.getLogger(__name__)

FILE_NAME = "exec-approvals.json"

_folder: Path | None = None
_lock = threading.RLock()


class NotApprovable(Exception):
    """Why a program was not approved, in words a console shows."""


def configure(folder: Path | None) -> None:
    """Keep approvals in FOLDER, the agent's own state folder; None keeps
    none (and approves nothing)."""
    global _folder
    _folder = folder


def _file() -> Path | None:
    return _folder / FILE_NAME if _folder is not None else None


def _kept_safe(path: Path) -> str | None:
    """Why approvals at PATH could have been written by someone else, or
    None when only root and Eugene's account can change the file (or,
    before it exists, its folder)."""
    for item in (path.parent, path):
        spelled = item.as_posix()
        if item == path and not exec_trust.present(spelled):
            break
        verdict = exec_trust.assess_path(spelled)
        if verdict.kind is not exec_trust.Kind.ok:
            return f"{spelled}: {verdict.why()}"
    return None


def _read() -> list[ExecApproval]:
    path = _file()
    if path is None:
        return []
    why = _kept_safe(path)
    if why is not None:
        log.warning("approved programs are not read, so none runs: %s", why)
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as e:
        log.warning("approved programs could not be read, so none runs: %s", e)
        return []
    try:
        return list(ExecApprovalList.model_validate_json(raw).approvals)
    except ValidationError as e:
        log.warning("%s is not a list of approvals, so none runs: %s", path, e)
        return []


def _write(approvals: list[ExecApproval]) -> None:
    path = _file()
    if path is None:
        raise NotApprovable("This agent keeps no approvals.")
    why = _kept_safe(path)
    if why is not None:
        raise NotApprovable(f"No approval is kept, because others could change it: {why}")
    body = ExecApprovalList(approvals=approvals).model_dump_json(indent=2) + "\n"
    write_private(path, body)


def _key(verdict: exec_trust.Verdict) -> str:
    """What an approval is kept by: the path a setting looks the program up
    by, made absolute with its links kept, so the chain approved is the one
    every later check walks (8c review)."""
    return verdict.lookup or verdict.program or verdict.given


def _chain(verdict: exec_trust.Verdict) -> list[tuple[str, int, int]]:
    return [(c.path, c.uid, c.mode) for c in verdict.chain]


def _lapsed(approval: ExecApproval, verdict: exec_trust.Verdict) -> str | None:
    """What differs from what was approved, or None."""
    if approval.accountUid != verdict.account_uid:
        return f"it is {verdict.account}'s now, not {approval.account}'s"
    if approval.sha256 != verdict.sha256:
        return "the file changed"
    if [(c.path, c.uid, c.mode) for c in approval.chain] != _chain(verdict):
        return "a folder, link or file on its way changed"
    return None


def holds(verdict: exec_trust.Verdict) -> bool:
    """An owner's approval of exactly this program holds. One that no
    longer matches is dropped, so the program asks again."""
    if verdict.kind is not exec_trust.Kind.needs_approval or verdict.program is None:
        return False
    with _lock:
        approvals = _read()
        match = next((a for a in approvals if a.path == _key(verdict)), None)
        if match is None:
            return False
        why = _lapsed(match, verdict)
        if why is None:
            return True
        log.warning("the approval to run %s lapsed (%s); it asks again", match.path, why)
        try:
            _write([a for a in approvals if a.id != match.id])
        except (OSError, NotApprovable) as e:
            log.warning("the lapsed approval of %s could not be removed: %s", match.path, e)
        return False


def forget(verdict: exec_trust.Verdict) -> None:
    """Drop any approval of a program the check refused now: putting back
    what made it refused does not bring the approval back (8c review)."""
    key = _key(verdict)
    with _lock:
        approvals = _read()
        kept = [a for a in approvals if a.path != key]
        if len(kept) == len(approvals):
            return
        log.warning("the approval to run %s lapsed (it is refused now); it asks again", key)
        try:
            _write(kept)
        except (OSError, NotApprovable) as e:
            log.warning("the lapsed approval of %s could not be removed: %s", key, e)


def approve(path: str, sha256: str, *, by: str | None) -> ExecApproval:
    """Approve PATH as it is now, when the check says `needs_approval` and
    its SHA-256 is SHA256, the one the owner was shown. Replaces an earlier
    approval of the same program. Raises NotApprovable naming the cause."""
    verdict = exec_trust.assess(path)
    if verdict.kind is exec_trust.Kind.ok:
        raise NotApprovable(
            f"Eugene's own account runs {path} already, so there is nothing to approve."
        )
    if verdict.kind is exec_trust.Kind.refused or verdict.program is None:
        forget(verdict)
        raise NotApprovable(verdict.message() or f"{path} cannot be approved.")
    if verdict.sha256 != sha256:
        raise NotApprovable(
            f"{_key(verdict)} changed since it was shown (its SHA-256 is now "
            f"{verdict.sha256}). Look at it again before approving it."
        )
    assert verdict.account is not None and verdict.account_uid is not None
    approval = ExecApproval(
        id=secrets.token_urlsafe(9),
        path=_key(verdict),
        account=verdict.account,
        accountUid=verdict.account_uid,
        sha256=sha256,
        chain=[ExecChainItem(path=p, uid=u, mode=m) for p, u, m in _chain(verdict)],
        approvedAt=datetime.now(UTC),
        approvedBy=by,
    )
    with _lock:
        kept = [a for a in _read() if a.path != approval.path]
        _write([*kept, approval])
    log.info(
        "an owner (%s) approved %s, which %s can change, for Eugene's own account",
        by,
        approval.path,
        approval.account,
    )
    return approval


def listed() -> list[ExecApproval]:
    with _lock:
        return _read()


def remove(approval_id: str) -> bool:
    """Take an approval back. False when there was none by that id."""
    with _lock:
        approvals = _read()
        kept = [a for a in approvals if a.id != approval_id]
        if len(kept) == len(approvals):
            return False
        _write(kept)
    return True


def view(verdict: exec_trust.Verdict | None) -> ExecTrust | None:
    """`ExecTrust` for a program Eugene's own account did not run."""
    if verdict is None or verdict.kind is exec_trust.Kind.ok:
        return None
    return ExecTrust(
        kind=ExecTrustKind(verdict.kind.value),
        path=_key(verdict),
        account=verdict.account if verdict.kind is exec_trust.Kind.needs_approval else None,
        sha256=verdict.sha256 if verdict.kind is exec_trust.Kind.needs_approval else None,
    )
