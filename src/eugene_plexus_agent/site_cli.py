"""`eugene-plexus-agent site`: this machine as a Job Site, from the machine itself.

    eugene-plexus-agent site join --url URL --token TOKEN --owner NAME --label NAME
        [--root-key KEY] [--password-stdin]
    eugene-plexus-agent site leave
    eugene-plexus-agent site status
    eugene-plexus-agent site audit [--limit N]
    eugene-plexus-agent site server add ID --name NAME --command PATH
        [--arg=A]... [--env K=V]... [--system]
    eugene-plexus-agent site server remove ID

**Adding a local MCP server is the machine administrator's act**
(`remote-nodes.md` §6.2): it names a program on this machine, so it is done
here, elevated (an administrator token on Windows, uid 0 on Linux), and
never from Workbench. It writes `site-servers.yaml` beside `agent.yaml`, in
the install's protected configuration, which the site host's own account
cannot write, with the program's SHA-256: a program that changes after it
is added is not run. Who may use it is then the site owner's policy, set
from Workbench; a new server is off until they turn it on.

**`join` makes this machine a Job Site**, its own enrollment, separate
from the node (J19, J21, J35). Elevated, because it turns on a service. It
asks the running agent to install the site host, then runs the **site
host's own** join with the site's own directory: the site's key is made
there, by the site host, and the agent never holds it (J23). The owner
confirms with their own password, read from the terminal or, with
`--password-stdin`, from the first line of standard input. `leave` tells the
root and forgets the enrollment; the site's policy and audit log stay.

**`--system`** marks a server able to alter the operating system, and adding
one is J9's gate: the administrator proves elevation here, and their consent
is recorded with the program's hash. The host refuses to turn on a `system`
server without that record.

`status` and `audit` read what the host keeps (its policy and its audit
log) from its own directory; they change nothing. The running agent picks up
an added or removed server at its next poll and restarts the host with it.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ._generated.site_host_models import SiteLocalServerList
from .apps import APPS_DIR
from .settings import Settings
from .site_host import HELPER_ID, SYSTEMD_STATE_ROOT, set_wanted

SERVERS_FILE = "site-servers.yaml"
#: How long `join` waits for the running agent to prepare the site host.
PREPARE_SECONDS = 900
MAX_SERVERS = 32
_ID = re.compile(r"^[a-z][a-z0-9-]{0,39}$")


class SiteError(Exception):
    """Refused; the message says why and what to do."""


def elevated() -> bool:
    """An administrator token on Windows; uid 0 elsewhere."""
    if sys.platform == "win32":
        import ctypes

        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:  # pragma: no cover - defensive
            return False
    getter = getattr(os, "geteuid", None)
    return getter is not None and int(getter()) == 0


def add_parser(sub: Any) -> None:
    site = sub.add_parser(
        "site",
        help="This machine's tools for Workbench: status, audit log, local servers.",
        description=(
            "Show what this machine's site host allows and what it was asked, and add or "
            "remove the local MCP servers it may offer. Adding or removing a server needs "
            "an administrator (root)."
        ),
    )
    actions = site.add_subparsers(dest="site_command", required=True)
    joining = actions.add_parser("join", help="Make this machine a job site (elevated).")
    joining.add_argument("--url", required=True, help="The root's address, from the join command.")
    joining.add_argument("--token", required=True, help="The site invitation.")
    joining.add_argument("--owner", required=True, help="How the owner signs in to Eugene.")
    joining.add_argument("--label", required=True, help="This machine's name in the install.")
    joining.add_argument("--root-key", dest="root_key", help="The root's identity key (HTTPS).")
    joining.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the owner's password from the first line of standard input.",
    )
    joining.add_argument("--python", help=argparse.SUPPRESS)
    joining.add_argument("--data-dir", dest="data_dir", help=argparse.SUPPRESS)
    leaving = actions.add_parser("leave", help="This machine stops being a job site (elevated).")
    leaving.add_argument("--python", help=argparse.SUPPRESS)
    leaving.add_argument("--data-dir", dest="data_dir", help=argparse.SUPPRESS)
    actions.add_parser("status", help="What this machine allows, and to whom.")
    audit = actions.add_parser("audit", help="The newest lines of this machine's audit log.")
    audit.add_argument("--limit", type=int, default=50, help="How many lines (1-200).")
    server = actions.add_parser("server", help="Add or remove a local MCP server (elevated).")
    verbs = server.add_subparsers(dest="server_command", required=True)
    add = verbs.add_parser("add", help="Add a local MCP server, off until the site's owner says.")
    add.add_argument("id", help="Its id: lower-case letters, digits and hyphens.")
    add.add_argument("--name", required=True, help="Its name, as people will see it.")
    # Not dest "command": that is the subcommand's, and sharing it once
    # made `site server add --command X` start the whole agent instead.
    add.add_argument(
        "--command", dest="program", required=True, help="The program's absolute path."
    )
    add.add_argument(
        "--arg",
        action="append",
        default=[],
        help="One argument; repeat for more. One that begins with - is given as --arg=-x.",
    )
    add.add_argument("--env", action="append", default=[], help="NAME=VALUE; repeat for more.")
    add.add_argument(
        "--system",
        action="store_true",
        help=(
            "It can alter this machine's operating system. Adding it records your consent, "
            "as this machine's administrator, with the program's hash (J9)."
        ),
    )
    remove = verbs.add_parser("remove", help="Remove a local MCP server.")
    remove.add_argument("id")


def run(args: argparse.Namespace, settings: Settings) -> int:
    config_dir = Path(settings.config_file).resolve().parent
    try:
        if args.site_command == "join":
            print(join(config_dir, args))
        elif args.site_command == "leave":
            print(leave(config_dir, args))
        elif args.site_command == "status":
            print(status(config_dir))
        elif args.site_command == "audit":
            print(audit(config_dir, args.limit))
        elif args.server_command == "add":
            print(
                add_server(
                    config_dir,
                    server_id=args.id,
                    name=args.name,
                    command=args.program,
                    args=list(args.arg),
                    env=list(args.env),
                    system=args.system,
                )
            )
        else:
            print(remove_server(config_dir, args.id))
    except SiteError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


# --- the protected list ---------------------------------------------------------


def read_servers(config_dir: Path) -> list[dict[str, Any]]:
    path = config_dir / SERVERS_FILE
    if not path.exists():
        return []
    try:
        value = SiteLocalServerList.model_validate(yaml.safe_load(path.read_text("utf-8")) or {})
    except (OSError, ValueError, ValidationError) as exc:
        raise SiteError(
            f"{path} could not be read ({type(exc).__name__}). Fix or remove it."
        ) from None
    return [s.model_dump(mode="json", exclude_none=True) for s in value.servers]


def write_servers(config_dir: Path, servers: list[dict[str, Any]]) -> None:
    SiteLocalServerList.model_validate({"servers": servers})
    path = config_dir / SERVERS_FILE
    temporary = path.with_suffix(".tmp")
    data = yaml.safe_dump({"servers": servers}, sort_keys=False, allow_unicode=True).encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _need_elevation() -> None:
    if not elevated():
        raise SiteError(
            "Adding or removing a local server names a program on this machine, so only its "
            "administrator may. Run this again as an administrator (Windows) or as root."
        )


def add_server(
    config_dir: Path,
    *,
    server_id: str,
    name: str,
    command: str,
    args: list[str],
    env: list[str],
    system: bool,
) -> str:
    _need_elevation()
    if not _ID.match(server_id):
        raise SiteError("A server's id is lower-case letters, digits and hyphens, at most 40.")
    if server_id.startswith("files"):
        raise SiteError("Ids beginning 'files' are Eugene's own file server. Choose another.")
    program = Path(command)
    if not program.is_absolute() or not program.is_file():
        raise SiteError(f"{command} is not a program on this machine. Give its absolute path.")
    environment: dict[str, str] = {}
    for item in env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SiteError(f"--env {item!r} is not NAME=VALUE.")
        environment[key] = value
    servers = read_servers(config_dir)
    if any(s["id"] == server_id for s in servers):
        raise SiteError(f"This machine already has a server called {server_id}. Remove it first.")
    if len(servers) >= MAX_SERVERS:
        raise SiteError(f"This machine already has {MAX_SERVERS} local servers. Remove one first.")
    entry: dict[str, Any] = {
        "id": server_id,
        "name": name.strip(),
        "command": str(program),
        "args": args,
        "env": environment,
        "sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "system": system,
        "consentedAt": datetime.now(UTC).isoformat() if system else None,
    }
    try:
        write_servers(config_dir, [*servers, {k: v for k, v in entry.items() if v is not None}])
    except ValidationError as exc:
        first = exc.errors()[0]
        raise SiteError(f"This server is not valid: {first['msg']}.") from None
    lines = [
        f"Added {name} ({server_id}). It is off until this site's owner turns it on in "
        "Workbench (Job sites), and says who may use which of its tools.",
        "Eugene restarts this machine's tools with it within a minute.",
    ]
    if system:
        lines.insert(
            1,
            "Marked as able to change this machine's operating system. Your consent, as its "
            "administrator, is recorded with the program's hash; if the program changes it "
            "will not run.",
        )
    return "\n".join(lines)


def remove_server(config_dir: Path, server_id: str) -> str:
    _need_elevation()
    servers = read_servers(config_dir)
    kept = [s for s in servers if s["id"] != server_id]
    if len(kept) == len(servers):
        raise SiteError(f"This machine has no local server called {server_id}.")
    write_servers(config_dir, kept)
    return f"Removed {server_id}. Eugene restarts this machine's tools without it within a minute."


# --- joining and leaving ----------------------------------------------------------


def _host_data(config_dir: Path) -> Path:
    if sys.platform == "linux" and (SYSTEMD_STATE_ROOT / HELPER_ID).exists():
        return SYSTEMD_STATE_ROOT / HELPER_ID
    return config_dir / APPS_DIR / HELPER_ID / "data"


def _host_python(config_dir: Path) -> Path | None:
    """The site host's interpreter, once the agent has prepared it."""
    versions = config_dir / APPS_DIR / HELPER_ID / "versions"
    found: list[Path] = []
    for venv in versions.glob("*/venv"):
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if python.exists():
            found.append(python)
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def _prepared(config_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    if args.python and args.data_dir:
        return Path(args.python), Path(args.data_dir)
    deadline = time.perf_counter() + PREPARE_SECONDS
    said = False
    while True:
        python, data = _host_python(config_dir), _host_data(config_dir)
        if python is not None and data.exists():
            return python, data
        if time.perf_counter() >= deadline:
            raise SiteError(
                "Eugene did not prepare this machine's job site in time. Check that its "
                "service is running, then run this again."
            )
        if not said:
            print("Preparing this machine's job site. This takes a minute or two the first time.")
            said = True
        time.sleep(2)


def _password(args: argparse.Namespace) -> str:
    if args.password_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    return getpass.getpass(f"{args.owner}, your Eugene password (confirms this machine): ")


def _give_to_owner_of(data: Path) -> None:
    """POSIX: what an administrator made in the site's directory belongs to
    the site host's account, as the directory does."""
    chown = getattr(os, "chown", None)
    if chown is None:
        return
    owner = data.stat()
    for name in ("site.json", "site_key.pem", "root_tls.json"):
        path = data / name
        if path.exists():
            chown(path, owner.st_uid, owner.st_gid)


def join(config_dir: Path, args: argparse.Namespace) -> str:
    _need_elevation()
    password = _password(args)
    if not password:
        raise SiteError("A job site is confirmed by its owner's own password.")
    set_wanted(config_dir, True)
    python, data = _prepared(config_dir, args)
    command = [
        str(python),
        "-m",
        "eugene_plexus_site_host",
        "join",
        "--url",
        args.url,
        "--token",
        args.token,
        "--owner",
        args.owner,
        "--label",
        args.label,
        "--data-dir",
        str(data),
    ]
    if args.root_key:
        command += ["--root-key", args.root_key]
    done = subprocess.run(
        command, input=password + "\n", capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise SiteError((done.stderr or done.stdout).strip() or "The join did not finish.")
    _give_to_owner_of(data)
    return done.stdout.strip()


def leave(config_dir: Path, args: argparse.Namespace) -> str:
    _need_elevation()
    python = Path(args.python) if args.python else _host_python(config_dir)
    data = Path(args.data_dir) if args.data_dir else _host_data(config_dir)
    if python is not None and data.exists():
        done = subprocess.run(
            [str(python), "-m", "eugene_plexus_site_host", "leave", "--data-dir", str(data)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        if done.returncode != 0:
            raise SiteError((done.stderr or done.stdout).strip() or "The site did not leave.")
    set_wanted(config_dir, False)
    return "This machine is no longer a job site. Its folders' list and audit log are kept."


# --- reading what the host keeps ------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except FileNotFoundError:
        return None
    except PermissionError:
        raise SiteError(
            "This account cannot read this machine's tools. Run it as an administrator (root)."
        ) from None
    except (OSError, ValueError) as exc:
        raise SiteError(f"{path} could not be read ({type(exc).__name__}).") from None


def status(config_dir: Path) -> str:
    policy = _read_json(_host_data(config_dir) / "policy.json")
    servers = read_servers(config_dir)
    lines: list[str] = []
    if policy is None:
        lines.append("This machine's tools have nothing registered yet.")
    else:
        lines.append(
            "Eugene's owner may use folders here in dev mode: "
            + ("yes, if Eugene is in dev mode" if policy.get("ownerInDevMode") else "no")
        )
        folders = policy.get("folders") or []
        lines.append(f"Folders ({len(folders)}):")
        for folder in folders:
            people = ", ".join(
                f"{p['subject']} ({'may change files' if p.get('writable') else 'read'})"
                for p in folder.get("people") or []
            )
            lines.append(f"  {folder['name']}: {folder['path']}")
            lines.append(f"    people: {people or 'nobody'}")
    enabled = (policy or {}).get("enabled") or {}
    access = (policy or {}).get("access") or []
    lines.append(f"Local servers ({len(servers)}):")
    for server in servers:
        flag = " [system]" if server.get("system") else ""
        lines.append(
            f"  {server['id']}{flag}: {server['name']} - "
            + ("on" if enabled.get(server["id"]) else "off")
        )
        for entry in access:
            if entry.get("server") == server["id"]:
                tools = ", ".join(t["name"] for t in entry.get("tools") or [])
                lines.append(f"    {entry['subject']}: {tools}")
    return "\n".join(lines)


def audit(config_dir: Path, limit: int) -> str:
    if not 1 <= limit <= 200:
        raise SiteError("--limit is 1 to 200.")
    data = _host_data(config_dir)
    lines: list[str] = []
    for name in ("audit.1.jsonl", "audit.jsonl"):
        path = data / name
        try:
            lines += path.read_text("utf-8").splitlines()
        except FileNotFoundError:
            continue
        except PermissionError:
            raise SiteError(
                "This account cannot read this machine's audit log. Run it as an administrator "
                "(root)."
            ) from None
    out: list[str] = []
    for line in reversed(lines[-limit:]):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        what = entry.get("tool") or entry.get("action") or entry.get("method") or ""
        where = f" on {entry['server']}" if entry.get("server") else ""
        reason = f" - {entry['reason']}" if entry.get("reason") else ""
        out.append(
            f"{entry.get('at', '')}  {entry.get('subject', '')}  {what}{where}  "
            f"{entry.get('decision', '')}{reason}"
        )
    return "\n".join(out) if out else "Nothing has been asked of this machine yet."
