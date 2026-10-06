"""Who is who on this machine, for its Job Site (§2.2, §3.2, J27).

A link says *this Eugene person is this OS account here*. This agent makes
one only at the machine: on its loopback link page, from the account that
owns the browser's connection and the person's own sign-in (Windows service
installs, `routes/site_link.py`), or at the join, for the owner (`site
join`). Neither the root nor the site host can make one. The root may ask to
**remove** one (`DELETE /v1/site/links/{subject}`), which only takes access
away.

The file lives in the install's administrator-only place, `site/`, beside
the local-server list (§3.2). The site host's account may read both; each
linked account may read the server list, which its worker runs from; nobody
but SYSTEM and Administrators may write either.

**One link per person, one person per account.** Links are refused for an
account that is never a person's: LocalSystem and the other system
accounts, service and virtual accounts (the site host's among them), root,
and system uids. The same rules as the worker's own (`site-host`
`accounts.py`).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SITE_DIR = "site"
LINKS_FILE = "links.json"
SERVERS_FILE = "servers.yaml"
#: The site host's account on a Windows service install (C1).
SITE_HOST_ACCOUNT = r"NT SERVICE\EugenePlexusApp-site-host"

_SYSTEM_SIDS = frozenset({"S-1-5-18", "S-1-5-19", "S-1-5-20"})
_PROGRAM_PREFIXES = ("S-1-5-80-", "S-1-5-82-", "S-1-5-90-", "S-1-5-96-")
LINUX_UID_MIN = 1000
MACOS_UID_MIN = 500


class LinkError(Exception):
    """A link this machine will not make or keep; the message says why."""


def not_a_person(account: str) -> str | None:
    """Why `account` (a SID, or a uid in decimal) is never a person's."""
    if account.startswith("S-"):
        if account in _SYSTEM_SIDS:
            return "a system account"
        if account.startswith(_PROGRAM_PREFIXES):
            return "a service or virtual account"
        if not account.startswith(("S-1-5-21-", "S-1-12-1-")):
            return "not a person's account"
        return None
    try:
        uid = int(account)
    except ValueError:
        return "not an account this system knows"
    if uid == 0:
        return "root"
    if uid < (MACOS_UID_MIN if sys.platform == "darwin" else LINUX_UID_MIN) or uid == 65534:
        return "a system account"
    return None


@dataclass(frozen=True)
class Link:
    subject: str
    name: str | None
    account: str
    account_name: str
    linked_at: str

    def as_json(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "name": self.name,
            "account": self.account,
            "accountName": self.account_name,
            "linkedAt": self.linked_at,
        }


def site_dir(config_dir: Path) -> Path:
    return config_dir / SITE_DIR


class LinkStore:
    """The links file. Every write replaces it whole, atomically."""

    def __init__(self, config_dir: Path) -> None:
        self.dir = site_dir(config_dir)
        self.path = self.dir / LINKS_FILE
        self._lock = threading.Lock()

    def load(self) -> list[Link]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            log.warning("the links file could not be read (%s); nobody is linked", exc)
            return []
        links: list[Link] = []
        entries = raw.get("links") if isinstance(raw, dict) else None
        for entry in entries if isinstance(entries, list) else []:
            try:
                links.append(
                    Link(
                        str(entry["subject"]),
                        entry.get("name"),
                        str(entry["account"]),
                        str(entry["accountName"]),
                        str(entry["linkedAt"]),
                    )
                )
            except (KeyError, TypeError):
                continue
        return links

    def _write(self, links: list[Link]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = json.dumps(
            {"version": 1, "links": [link.as_json() for link in links]}, indent=2
        ).encode()
        temporary = self.path.with_suffix(".tmp")
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(temporary, 0o644)
        os.replace(temporary, self.path)

    def for_subject(self, subject: str) -> Link | None:
        return next((link for link in self.load() if link.subject == subject), None)

    def for_account(self, account: str) -> Link | None:
        return next((link for link in self.load() if link.account == account), None)

    def add(
        self,
        *,
        subject: str,
        name: str | None,
        account: str,
        account_name: str,
        never: frozenset[str] = frozenset(),
    ) -> Link:
        """Link a person to an account. `never` names further accounts that
        are not a person's here (the site host's, the agent's)."""
        if why := not_a_person(account):
            raise LinkError(f"{account_name} is {why}, so no one can be linked to it.")
        if account in never:
            raise LinkError(f"{account_name} is one of Eugene's own accounts, never a person's.")
        with self._lock:
            links = self.load()
            for link in links:
                if link.subject == subject and link.account == account:
                    return link
                if link.account == account:
                    raise LinkError(
                        f"{account_name} is already linked to {link.name or 'someone else'}. "
                        "They, or this site's owner, remove that link first."
                    )
                if link.subject == subject:
                    raise LinkError(
                        f"You are already linked to {link.account_name} here. Remove that link "
                        "first, from Workbench or on this page while signed in as it."
                    )
            new = Link(subject, name, account, account_name, datetime.now(UTC).isoformat())
            self._write([*links, new])
            return new

    def remove(self, subject: str) -> Link | None:
        with self._lock:
            links = self.load()
            kept = [link for link in links if link.subject != subject]
            if len(kept) == len(links):
                return None
            self._write(kept)
            return next(link for link in links if link.subject == subject)


# --- Windows: who may read what --------------------------------------------------


def _icacls(path: Path, *args: str) -> None:
    out = subprocess.run(
        ["icacls", str(path), *args], capture_output=True, text=True, check=False, timeout=120
    )
    if out.returncode != 0:
        raise OSError(f"icacls {path} {' '.join(args)} failed: {(out.stdout + out.stderr).strip()}")


def protect_windows(config_dir: Path, links: list[Link], *, site_host_exists: bool) -> None:
    """`site\\`: SYSTEM and Administrators write; the site host's account
    reads. The server list: each linked account reads it too, because its
    worker runs from it; nobody else."""
    folder = site_dir(config_dir)
    folder.mkdir(parents=True, exist_ok=True)
    grants = ["*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F"]
    if site_host_exists:
        # The account exists only once its service does (C1).
        grants.append(f"{SITE_HOST_ACCOUNT}:(OI)(CI)RX")
    _icacls(folder, "/inheritance:r", "/grant:r", *grants)
    servers = folder / SERVERS_FILE
    if servers.exists():
        _icacls(servers, "/reset")
        grants = [f"*{link.account}:R" for link in links if link.account.startswith("S-")]
        if grants:
            _icacls(servers, "/grant:r", *grants)


def account_name(sid: str) -> str:
    """`DOMAIN\\name` for a SID, or the SID itself when it does not resolve."""
    if sys.platform != "win32":
        import pwd

        try:
            return pwd.getpwuid(int(sid)).pw_name
        except (KeyError, ValueError):
            return sid
    try:
        import win32security

        name, domain, _ = win32security.LookupAccountSid(
            None, win32security.ConvertStringSidToSid(sid)
        )
        return f"{domain}\\{name}" if domain else str(name)
    except Exception:
        return sid


def account_sid(name: str) -> str:
    """The SID (Windows) or uid (elsewhere) of an account named at the
    machine (`--site-account`)."""
    if sys.platform != "win32":
        import pwd

        try:
            return str(pwd.getpwnam(name).pw_uid)
        except KeyError:
            raise LinkError(f"There is no account named {name} on this machine.") from None
    import win32security

    try:
        sid, _domain, _kind = win32security.LookupAccountName(None, name)
    except Exception:
        raise LinkError(f"There is no account named {name} on this machine.") from None
    return str(win32security.ConvertSidToStringSid(sid))
