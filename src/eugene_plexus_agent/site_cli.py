"""`eugene-plexus-agent site`: this machine as a Job Site, from the machine itself.

    eugene-plexus-agent site join --url URL --token TOKEN --owner NAME --label NAME
        [--root-key KEY] [--password-stdin]
    eugene-plexus-agent site leave
    eugene-plexus-agent site link --person NAME [--account ACCOUNT] [--password-stdin]
    eugene-plexus-agent site unlink --person NAME
    eugene-plexus-agent site status
    eugene-plexus-agent site audit [--limit N]
    eugene-plexus-agent site server add ID --name NAME --command PATH
        [--arg=A]... [--env K=V]... [--system]
    eugene-plexus-agent site server remove ID
    eugene-plexus-agent site consent [--off]

**Adding a local MCP server is the machine administrator's act**
(`remote-nodes.md` §6.2): it names a program on this machine, so it is done
here, elevated (an administrator token on Windows, uid 0 on Linux), and
never from Workbench. It writes `site/servers.yaml` beside `agent.yaml`, in
the install's protected configuration, which neither the site host nor a
worker can write, with the program's SHA-256: a program that changes after
it is added is not run. Who may use it is then the site owner's policy, set
from Workbench; a new server is off until they turn it on.

**`join` makes this machine a Job Site**, its own enrollment, separate
from the node (J19, J21, J35). Elevated, because it turns on a service. It
asks the running agent to install the site host, then runs the **site
host's own** join with the site's own directory: the site's key is made
there, by the site host, and the agent never holds it (J23). The owner
confirms with their own password, read from the terminal or, with
`--password-stdin`, from the first line of standard input. `leave` tells the
root and forgets the enrollment and the links; the site's policy and audit
log stay.

**`join` also links the owner** to the account that ran it (§2.2): the
elevated administrator's own on Windows, or the console's signed-in person
when SYSTEM runs it, or this account on a per-user install, where no
elevation is needed. `--site-account` names another. **`link` and
`unlink`** are the expert's way to link someone else (elevated): it checks
their Eugene password with the root, through the site's own key, as root
does on Linux (J36). Most people link on the page at the machine instead
(`http://127.0.0.1:<port>/link`, a Windows service install).

**`--system`** marks a server able to alter the operating system, and adding
one is J9's gate: the administrator proves elevation here, and their consent
is recorded with the program's hash. The host refuses to turn on a `system`
server without that record.

**`consent`** is J9's proof for commands (2b.4, J30, J89): an administrator,
elevated here, allows Workbench to run commands on this machine, each as the
person who signed it; `--off` takes that back. It is recorded in the same
protected list, which the site host and every worker read and cannot write.
`join` asks the same question on a service install, once the machine joined
(`--commands` or `--no-commands` answers it ahead). The Windows tray's
*Allow commands* runs it behind a UAC prompt.

`status` and `audit` read what the host keeps (its policy and its audit
log) from its own directory; they change nothing. The running agent picks up
an added or removed server at its next poll and restarts the host with it.
"""

from __future__ import annotations

import argparse
import contextlib
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
from .apps import APPS_DIR, APPS_FILE, AppStore, venv_python
from .settings import Settings
from .site_host import HELPER_ID, servers_path, set_wanted
from .site_links import LinkError, LinkStore, account_name, account_sid

#: Where root keeps a Linux system install's site (install.sh, §2.4).
ROOT_SITE_DATA = Path("/var/lib/eugene-plexus-site")
SITE_HOST_ACCOUNT = r"NT SERVICE\EugenePlexusApp-site-host"
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
    joining.add_argument(
        "--site-account",
        dest="site_account",
        help="The owner's account on this machine, when not the one running this.",
    )
    joining.add_argument(
        "--no-browser",
        dest="no_browser",
        action="store_true",
        help="Print the page where the owner adds their key, without opening it.",
    )
    answer = joining.add_mutually_exclusive_group()
    answer.add_argument(
        "--commands",
        dest="commands",
        action="store_const",
        const=True,
        help="Allow Workbench to run commands here, each signed by its person (J9).",
    )
    answer.add_argument(
        "--no-commands",
        dest="commands",
        action="store_const",
        const=False,
        help="Do not allow commands now; an administrator can allow them later.",
    )
    joining.add_argument("--python", help=argparse.SUPPRESS)
    joining.add_argument("--data-dir", dest="data_dir", help=argparse.SUPPRESS)
    linking = actions.add_parser(
        "link", help="Link a person to their account on this machine (elevated)."
    )
    linking.add_argument("--person", required=True, help="How they sign in to Eugene.")
    linking.add_argument("--account", help="Their account here; the console's person if absent.")
    linking.add_argument("--password-stdin", action="store_true")
    linking.add_argument("--python", help=argparse.SUPPRESS)
    linking.add_argument("--data-dir", dest="data_dir", help=argparse.SUPPRESS)
    unlinking = actions.add_parser("unlink", help="Remove a person's link here (elevated).")
    unlinking.add_argument("--person", required=True, help="How they sign in to Eugene.")
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
    consent_parser = actions.add_parser(
        "consent",
        help="Allow Workbench to run commands on this machine (elevated, J9).",
        description=(
            "As this machine's administrator, allow Workbench to run commands here. Each "
            "command runs as the person who signed it, with their own signature, in their own "
            "workspaces whose rules ask about commands. --off takes it back."
        ),
    )
    consent_parser.add_argument("--off", action="store_true", help="Take the consent back.")
    # The tray's elevated run names the install's configuration: an
    # administrator's own account may not carry the person's variable.
    consent_parser.add_argument("--config-file", dest="config_file", help=argparse.SUPPRESS)


def run(args: argparse.Namespace, settings: Settings) -> int:
    config_dir = Path(settings.config_file).resolve().parent
    try:
        if args.site_command == "join":
            print(join(config_dir, args, port=settings.bind_port))
        elif args.site_command == "leave":
            print(leave(config_dir, args))
        elif args.site_command == "link":
            print(link(config_dir, args))
        elif args.site_command == "unlink":
            print(unlink(config_dir, args.person))
        elif args.site_command == "status":
            print(status(config_dir))
        elif args.site_command == "audit":
            print(audit(config_dir, args.limit))
        elif args.site_command == "consent":
            if args.config_file:
                config_dir = Path(args.config_file).resolve().parent
            print(consent(config_dir, allow=not args.off))
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


def read_list(config_dir: Path) -> dict[str, Any]:
    """The protected list as it is (`SiteLocalServerList`): the servers, and
    the administrator's consent to commands when given."""
    path = servers_path(config_dir)
    if not path.exists():
        return {"servers": []}
    try:
        value = SiteLocalServerList.model_validate(yaml.safe_load(path.read_text("utf-8")) or {})
    except (OSError, ValueError, ValidationError) as exc:
        raise SiteError(
            f"{path} could not be read ({type(exc).__name__}). Fix or remove it."
        ) from None
    return dict(value.model_dump(mode="json", exclude_none=True))


def read_servers(config_dir: Path) -> list[dict[str, Any]]:
    return list(read_list(config_dir).get("servers") or [])


def write_servers(config_dir: Path, servers: list[dict[str, Any]]) -> None:
    """The servers, keeping the consent to commands as it is."""
    write_list(config_dir, {**read_list(config_dir), "servers": servers})


def write_list(config_dir: Path, value: dict[str, Any]) -> None:
    SiteLocalServerList.model_validate(value)
    path = servers_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    data = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _root_owns_the_list(config_dir: Path) -> None:
    """On a Linux system install the list is root's, beside the links
    (`/etc/eugene-plexus/site/servers.yaml`), and this agent's code never
    runs as root: its account owns the prefix it runs from (§2.4, J36)."""
    if sys.platform == "linux" and _system_install(config_dir):
        raise SiteError(
            "On a Linux system install root keeps this machine's local servers, in "
            "/etc/eugene-plexus/site/servers.yaml. Edit it there as root; the site host "
            "and its workers read it when they next start."
        )


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
    _root_owns_the_list(config_dir)
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


def _administrator() -> str | None:
    """Who consented, for display: the account running this."""
    try:
        return getpass.getuser()
    except Exception:
        return None


def consent(config_dir: Path, *, allow: bool) -> str:
    """J9's proof for commands (J30, J89): recorded only by an administrator,
    elevated here, in the protected list."""
    if sys.platform == "linux" and _system_install(config_dir):
        raise SiteError(
            "On a Linux system install root keeps this consent: run the installer again with "
            "--site-commands (or --site-no-commands to take it back)."
        )
    if not elevated():
        raise SiteError(
            "Allowing commands lets programs run on this machine, so only its administrator "
            "may. Run this again as an administrator (Windows) or as root."
        )
    value = read_list(config_dir)
    if allow:
        who = _administrator()
        value["commands"] = {
            "consentedAt": datetime.now(UTC).isoformat(),
            **({"by": who} if who else {}),
        }
    else:
        value.pop("commands", None)
    write_list(config_dir, value)
    if allow:
        return (
            "Commands from Workbench may run on this machine now. Each one runs as the person "
            "who signed it, in their own workspaces whose rules ask about commands. Take it "
            "back with --off, or the machine's owner can from Workbench."
        )
    return "Commands from Workbench no longer run on this machine."


def remove_server(config_dir: Path, server_id: str) -> str:
    _root_owns_the_list(config_dir)
    _need_elevation()
    servers = read_servers(config_dir)
    kept = [s for s in servers if s["id"] != server_id]
    if len(kept) == len(servers):
        raise SiteError(f"This machine has no local server called {server_id}.")
    write_servers(config_dir, kept)
    return f"Removed {server_id}. Eugene restarts this machine's tools without it within a minute."


# --- joining and leaving ----------------------------------------------------------


def _host_data(config_dir: Path) -> Path:
    if sys.platform == "linux" and ROOT_SITE_DATA.exists():
        return ROOT_SITE_DATA
    return config_dir / APPS_DIR / HELPER_ID / "data"


def _system_install(config_dir: Path) -> bool:
    """Whether this is the machine's own install rather than one person's:
    under ProgramData on Windows, under /var/lib on Linux."""
    where = str(config_dir.resolve()).lower()
    if sys.platform == "win32":
        program_data = os.environ.get("PROGRAMDATA", r"C:\ProgramData").lower()
        return where.startswith(program_data)
    return where.startswith("/var/lib/")


def _console_sid() -> str | None:
    """The person signed in at this machine's console, when SYSTEM runs this."""
    try:
        import win32security
        import win32ts

        session = win32ts.WTSGetActiveConsoleSessionId()
        token = win32ts.WTSQueryUserToken(session)
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        return str(win32security.ConvertSidToStringSid(user))
    except Exception:
        return None


def _own_sid() -> str:
    if sys.platform != "win32":
        return str(os.getuid())
    import win32api
    import win32con
    import win32security

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    return str(win32security.ConvertSidToStringSid(user))


def _never() -> frozenset[str]:
    accounts = {"S-1-5-18"}
    with contextlib.suppress(LinkError):
        accounts.add(account_sid(SITE_HOST_ACCOUNT))
    return frozenset(accounts)


def _owner_account(args: argparse.Namespace) -> str:
    """Whose account a link names (§2.2): the one named, else the one
    running this, else the console's person when SYSTEM runs it."""
    named = getattr(args, "site_account", None) or getattr(args, "account", None)
    if named:
        return account_sid(named)
    mine = _own_sid()
    if mine == "S-1-5-18":
        console = _console_sid()
        if console is None:
            raise SiteError(
                "Nobody is signed in at this machine's console. Name the account with "
                "--site-account."
            )
        return console
    if sys.platform != "win32" and mine == "0":
        sudo = os.environ.get("SUDO_UID")
        if not sudo:
            raise SiteError("Name the person's account here with --site-account.")
        return sudo
    return mine


def _host_python(config_dir: Path) -> Path | None:
    """The site host's interpreter, once the agent has installed it: the
    version `apps.yaml` records. An interpreter alone is not an install:
    uv makes a venv's `python` before it installs anything into it, and the
    first Windows run's `site join` ran one with no site host in it."""
    store = AppStore(config_dir / APPS_FILE)
    try:
        store.load()
    except (OSError, ValueError, KeyError, TypeError):
        return None
    record = store.get(HELPER_ID)
    if record is None:
        return None
    python = venv_python(store.version_dir(HELPER_ID, record.version) / "venv")
    return python if python.exists() else None


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


def join(config_dir: Path, args: argparse.Namespace, *, port: int | None = None) -> str:
    if sys.platform == "linux" and _system_install(config_dir):
        raise SiteError(
            "On a Linux system install root makes this machine a job site: run the installer "
            "with --job-site (J36)."
        )
    if _system_install(config_dir):
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
    said = [done.stdout.strip(), _link_owner(config_dir, data, args)]
    said.append(_join_consent(config_dir, getattr(args, "commands", None)))
    if port is not None and not _system_install(config_dir):
        opened = not getattr(args, "no_browser", False)
        said.append(_key_page(f"http://127.0.0.1:{port}/link", opened=opened))
    return "\n".join(line for line in said if line)


#: J30's one question at the join, asked of the administrator running it.
CONSENT_QUESTION = "Allow tools that change this machine's settings or run programs? [y/N] "


def _join_consent(config_dir: Path, answer: bool | None) -> str:
    """J30: consent at the join, by the administrator running it, on an
    install where that run is elevated (a service install). Asked at the
    terminal unless answered ahead; a per-user install is never elevated, so
    it says how an administrator can allow commands later."""
    if not elevated():
        return (
            "Commands from Workbench do not run here until an administrator allows them: "
            "eugene-plexus-agent site consent, run as an administrator."
        )
    if answer is None:
        terminal = getattr(sys.stdin, "isatty", None)
        if not (callable(terminal) and terminal()):
            # Nobody to ask: no consent is given without an answer.
            answer = False
        else:
            try:
                answer = input(CONSENT_QUESTION).strip().lower() in {"y", "yes"}
            except EOFError:
                answer = False
    if not answer:
        later = (
            "Allow commands in Eugene's tray icon"
            if sys.platform == "win32"
            else "eugene-plexus-agent site consent, as root"
        )
        return f"Commands from Workbench do not run here. To allow them later: {later}."
    try:
        return consent(config_dir, allow=True)
    except SiteError as exc:
        return str(exc)


def _key_page(page: str, *, opened: bool) -> str:
    """J14a.2: on a per-user install the owner's key is made where the join
    happened, in their own browser on this machine (J48: no tool runs here
    until it is). Opened for them where a desktop session can show it,
    printed either way."""
    said = f"No tool runs here until you add your own key: open {page} in your browser here."
    if opened and _open_in_browser(page):
        said += " It is opening now."
    return said


def _open_in_browser(page: str) -> bool:
    """Open `page` in this session's browser, never in the terminal: Python's
    `webbrowser` falls back to a text browser that would take it over."""
    try:
        if sys.platform == "win32":
            os.startfile(page)
            return True
        if sys.platform == "darwin":
            command = ["open", page]
        elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            command = ["xdg-open", page]
        else:
            return False
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except OSError:
        return False


def _link_owner(config_dir: Path, data: Path, args: argparse.Namespace) -> str:
    """The owner's link, from the join's own answer and the account that ran it."""
    site = _read_json(data / "site.json") or {}
    owner, name = site.get("owner"), site.get("ownerName")
    if not isinstance(owner, str):
        return "The owner could not be linked: the site's enrollment could not be read."
    try:
        account = _owner_account(args)
        LinkStore(config_dir).add(
            subject=owner,
            name=name if isinstance(name, str) else None,
            account=account,
            account_name=account_name(account),
            never=_never(),
        )
    except (LinkError, SiteError) as exc:
        return f"The owner was not linked to an account here: {exc}"
    return f"{name or 'The owner'}'s calls here run as {account_name(account)}."


def link(config_dir: Path, args: argparse.Namespace) -> str:
    """Link someone at the machine, by their Eugene password (elevated)."""
    if sys.platform == "linux" and _system_install(config_dir):
        raise SiteError(
            "On a Linux system install root links people: run the installer with --site-link (J36)."
        )
    _need_elevation()
    python = Path(args.python) if args.python else _host_python(config_dir)
    data = Path(args.data_dir) if args.data_dir else _host_data(config_dir)
    if python is None or not (data / "site.json").exists():
        raise SiteError("This machine is not a job site.")
    password = (
        sys.stdin.readline().rstrip("\r\n")
        if args.password_stdin
        else getpass.getpass(f"{args.person}, your Eugene password (links you here): ")
    )
    done = subprocess.run(
        [
            str(python),
            "-m",
            "eugene_plexus_site_host",
            "check-person",
            "--name",
            args.person,
            "--data-dir",
            str(data),
        ],
        input=password + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise SiteError((done.stderr or done.stdout).strip() or "The person could not be checked.")
    person = json.loads(done.stdout)
    account = _owner_account(args)
    try:
        LinkStore(config_dir).add(
            subject=person["subject"],
            name=person.get("name"),
            account=account,
            account_name=account_name(account),
            never=_never(),
        )
    except LinkError as exc:
        raise SiteError(str(exc)) from None
    return f"{person.get('name')}'s calls here now run as {account_name(account)}."


def unlink(config_dir: Path, person: str) -> str:
    _need_elevation()
    store = LinkStore(config_dir)
    for entry in store.load():
        if person in (entry.name, entry.subject):
            store.remove(entry.subject)
            return f"{entry.name or entry.subject} is no longer linked to {entry.account_name}."
    raise SiteError(f"Nobody called {person} is linked here.")


def leave(config_dir: Path, args: argparse.Namespace) -> str:
    if _system_install(config_dir):
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
    links = LinkStore(config_dir)
    for entry in links.load():
        links.remove(entry.subject)
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
