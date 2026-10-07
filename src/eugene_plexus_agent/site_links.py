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

**A link carries its person's keys** (J14a, `person-held-keys.md` §4.1):
the public half of a key made in their browser on the loopback page, pinned
here by this agent at the machine, and checked by the site host against
every change that person approves. The root never sees one. At most eight a
person; one whose id is not the SHA-256 of its own public key is dropped.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
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
MAX_KEYS = 8
_KEY_LENGTH = {"Ed25519": 32, "ES256": 65}


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


def key_id(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:32]


def check_key(alg: str, public_b64: str) -> bytes:
    """The raw public key, or `LinkError` saying why it is not one."""
    try:
        raw = base64.b64decode(public_b64, validate=True)
    except (binascii.Error, ValueError):
        raise LinkError("That key could not be read.") from None
    if _KEY_LENGTH.get(alg) != len(raw):
        raise LinkError("That is not an Ed25519 or P-256 public key.")
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        if alg == "Ed25519":
            Ed25519PublicKey.from_public_bytes(raw)
        else:
            ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw)
    except ValueError:
        raise LinkError("That is not a valid public key.") from None
    return raw


def _key(entry: Any) -> dict[str, Any] | None:
    """A pinned key from the file, or None when it is not a sound one."""
    if not isinstance(entry, dict):
        return None
    try:
        raw = check_key(str(entry["alg"]), str(entry["publicKey"]))
    except (KeyError, LinkError):
        return None
    if entry.get("id") != key_id(raw):
        return None
    return {
        "id": entry["id"],
        "alg": entry["alg"],
        "publicKey": entry["publicKey"],
        "label": entry.get("label"),
        "addedAt": str(entry.get("addedAt") or ""),
    }


@dataclass(frozen=True)
class Link:
    subject: str
    name: str | None
    account: str
    account_name: str
    linked_at: str
    keys: tuple[dict[str, Any], ...] = ()

    def as_json(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "subject": self.subject,
            "name": self.name,
            "account": self.account,
            "accountName": self.account_name,
            "linkedAt": self.linked_at,
        }
        if self.keys:
            value["keys"] = [dict(k) for k in self.keys]
        return value


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
                keys = [_key(k) for k in entry.get("keys") or []]
                links.append(
                    Link(
                        str(entry["subject"]),
                        entry.get("name"),
                        str(entry["account"]),
                        str(entry["accountName"]),
                        str(entry["linkedAt"]),
                        tuple(k for k in keys if k is not None)[:MAX_KEYS],
                    )
                )
            except (KeyError, TypeError, AttributeError):
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

    def add_key(self, account: str, alg: str, public_b64: str, label: str | None) -> dict[str, Any]:
        """Pin a key to the person linked to `account`, at the machine. The
        same key again is the key already pinned."""
        raw = check_key(alg, public_b64)
        ident = key_id(raw)
        with self._lock:
            links = self.load()
            link = next((link for link in links if link.account == account), None)
            if link is None:
                raise LinkError("Link this account first, then add a key.")
            for key in link.keys:
                if key["id"] == ident:
                    return dict(key)
            if len(link.keys) >= MAX_KEYS:
                raise LinkError(f"You have {MAX_KEYS} keys here already. Remove one first.")
            key = {
                "id": ident,
                "alg": alg,
                "publicKey": base64.b64encode(raw).decode("ascii"),
                "label": (label or "")[:120] or None,
                "addedAt": datetime.now(UTC).isoformat(),
            }
            updated = Link(
                link.subject,
                link.name,
                link.account,
                link.account_name,
                link.linked_at,
                (*link.keys, key),
            )
            self._write([updated if item is link else item for item in links])
            return key

    def remove_key(self, account: str, ident: str) -> bool:
        with self._lock:
            links = self.load()
            link = next((link for link in links if link.account == account), None)
            if link is None or not any(k["id"] == ident for k in link.keys):
                return False
            kept = tuple(k for k in link.keys if k["id"] != ident)
            updated = Link(
                link.subject, link.name, link.account, link.account_name, link.linked_at, kept
            )
            self._write([updated if item is link else item for item in links])
            return True

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
