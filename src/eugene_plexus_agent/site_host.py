"""This machine's site host, and the workers that run each person's tools
(`docs/design/job-sites-own-enrollment.md`, J19, J21, J23, §2.4, §3.2).

A Job Site is its own enrollment, held by the site host (`eugene-plexus/
site-host`): its key, its root, its owner, its policy. It joins, polls and
answers its root itself. Each person's tools run in a **worker**, as that
person's own OS account. Until the standalone install (J21) the agent on a
node is the machine's privileged starter, and does this and nothing else:

1. **Installs the site host** at a pinned commit and keeps it running: on a
   Windows service install as a bundled app in an account of its own (C1);
   on a per-user install as this agent's own child (J38). On a Linux system
   install root installs and runs it, never this unprivileged agent (§2.4),
   and the agent only reads which site it is.
2. **Keeps the links** between people and accounts, made at the machine
   (`site_links.py`, `routes/site_link.py`), and the read grants that go
   with them: the site host's account reads the links and the server list;
   each linked account reads the server list and the worker program.
3. **Starts each linked person's worker** as that person
   (`site_workers.py`): on Windows while they are signed in (J25), on a
   per-user install for the installing person.
4. **Says which site it hosts** (J32), for display.
5. **Carries a person's approvals** (J14a) from its loopback page to the
   site host's own loopback API (`held`), with the token the site host keeps
   in its data directory. The signature is checked there, not here.

The agent holds none of the site's identity, and the node's enrollment is
unchanged by any of this: the two share nothing but the machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from fastapi import FastAPI

from ._generated.models import AppManifest, AppOrigin
from ._generated.site_host_models import SiteLocalServerList
from ._http import client_for, internal_client
from ._private_files import write_private
from .apps import SITE_HOST_ENTRIES, AppManager, venv_python
from .site_links import (
    SERVERS_FILE,
    SITE_HOST_ACCOUNT,
    LinkStore,
    account_sid,
    protect_windows,
    site_dir,
)
from .site_workers import ChildStarter, WindowsStarter, WorkerProgram

log = logging.getLogger(__name__)

HELPER_ID = "site-host"
#: The id slice 2's host ran under, and the helper before it. Removed when
#: found: the operator-managed node folders retired (J20).
RETIRED_ID = "node-files"
PACKAGE = "eugene-plexus-site-host"
ENTRY = "eugene_plexus_site_host"

SITE_HOST_COMMIT = "4ea7562af8641b67d641e530987b820a1c02f104"
SITE_HOST_SOURCE = f"https://github.com/eugene-plexus/site-host/archive/{SITE_HOST_COMMIT}.tar.gz"
#: A checkout to install instead, for development and the acceptance runs.
SOURCE_OVERRIDE = "EUGENE_PLEXUS_AGENT_SITE_HOST_SOURCE"
#: Written by an administrator at the machine (`site join`, `site leave`).
WANTED_FILE = "site-host.json"
#: Where root keeps a Linux system install's site host (`install.sh`): its
#: port, for the agent to ask which site it is.
ROOT_SITE_FILE = Path("/etc/eugene-plexus/site/host.json")
REPORT_SECONDS = 60.0

Mode = Literal["service", "user", "root"]
#: The site host's approval API's token, in its own data directory (J53).
LOCAL_TOKEN_FILE = "local_token"


class HostUnavailable(Exception):
    """The site host is not running, or did not answer."""


def source() -> tuple[str, str]:
    """Where the host installs from, and the version it is recorded as."""
    override = os.environ.get(SOURCE_OVERRIDE, "").strip()
    if not override:
        return SITE_HOST_SOURCE, SITE_HOST_COMMIT
    root = Path(override)
    digest = hashlib.sha256()
    for path in [*sorted((root / "src").rglob("*.py")), root / "pyproject.toml"]:
        if path.exists():
            digest.update(path.read_bytes())
    return str(root.resolve()), "local-" + digest.hexdigest()[:16]


def wanted(config_dir: Path) -> bool:
    """Whether an administrator turned the site host on here."""
    try:
        value = json.loads((config_dir / WANTED_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        log.warning("%s could not be read (%s); no site host runs", WANTED_FILE, exc)
        return False
    return isinstance(value, dict) and value.get("enabled") is True


def set_wanted(config_dir: Path, enabled: bool) -> None:
    path = config_dir / WANTED_FILE
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"enabled": enabled}), encoding="utf-8")
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def servers_path(config_dir: Path) -> Path:
    return site_dir(config_dir) / SERVERS_FILE


def local_servers(config_dir: Path) -> list[dict[str, Any]]:
    """The servers an administrator added at this machine. A file that does
    not read is no servers, logged: it never stops the site."""
    path = servers_path(config_dir)
    if not path.exists():
        return []
    try:
        value = SiteLocalServerList.model_validate(yaml.safe_load(path.read_text("utf-8")) or {})
    except (OSError, ValueError) as exc:
        log.warning("%s could not be read (%s); no local servers run", path, type(exc).__name__)
        return []
    return [s.model_dump(mode="json", exclude_none=True) for s in value.servers]


def channel_name(config_dir: Path) -> str:
    """The pipe or socket the site host and its workers meet on. One per
    install, so a second install on the machine never shares it."""
    tag = hashlib.sha256(str(config_dir.resolve()).encode()).hexdigest()[:12]
    if sys.platform == "win32":
        return rf"\\.\pipe\eugene-plexus-site-{tag}"
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and len(runtime) < 60:
        return f"{runtime}/eugene-plexus-site-{tag}.sock"
    candidate = site_dir(config_dir) / "channel.sock"
    if len(str(candidate)) < 100:
        return str(candidate)
    return f"/tmp/eugene-plexus-site-{os.getuid()}-{tag}.sock"


def _stamp(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def manifest(environment: dict[str, str]) -> AppManifest:
    where, version = source()
    return AppManifest.model_validate(
        {
            "id": HELPER_ID,
            "name": "Job site",
            "summary": "This machine as a job site: its tools for Workbench. Managed by Eugene.",
            "source": where,
            "version": version,
            "package": PACKAGE,
            "entry": ENTRY,
            "python": "3.12",
            "ui": False,
            "configTrio": False,
            "uses": [],
            "signIn": False,
            "localActions": True,
            "environment": environment,
        }
    )


def _icacls(path: Path, *args: str) -> None:
    out = subprocess.run(
        ["icacls", str(path), *args], capture_output=True, text=True, check=False, timeout=900
    )
    if out.returncode != 0:
        raise OSError(f"icacls {path} failed: {(out.stdout + out.stderr).strip()}")


class SiteHostSupervisor:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self._host = internal_client(timeout=10.0, follow_redirects=False)
        self._root_client: httpx.AsyncClient | None = None
        self._root: str | None = None
        self.error: str | None = None
        self._attempt_version: str | None = None
        self._retry_at = 0.0
        self.site: str | None = None
        """The site this machine hosts, as the host's own health says."""
        self._reported: tuple[str, tuple[str, ...]] | None = None
        self._report_at = 0.0
        self._wake = asyncio.Event()
        self._windows: WindowsStarter | None = None
        self._child: ChildStarter | None = None
        self._granted: tuple[Any, ...] | None = None

    @property
    def manager(self) -> AppManager | None:
        return getattr(self.app.state, "apps", None)

    @property
    def config_dir(self) -> Path:
        return Path(self.app.state.settings.config_file).resolve().parent

    def mode(self) -> Mode | None:
        """How this install hosts a site: as the machine's service, as the
        person who installed it, or not at all because root does (Linux)."""
        manager = self.manager
        if manager is None:
            return None
        if manager.accounts.kind == "windows_service":
            return "service"
        if manager.accounts.kind == "systemd":
            return "root"
        from ._generated.models import InstallMechanism
        from .install_info import mechanism

        # A service install whose accounts are broken (pywin32 missing, an old
        # install.sh's units) is not a per-user install: hosting the site as
        # this agent's own child would run it as LocalSystem or the agent's
        # account. No site until it is repaired, as `accounts.reason` says.
        if mechanism() in (
            InstallMechanism.container,
            InstallMechanism.windows_service,
            InstallMechanism.systemd_system,
        ):
            return None
        return "user"

    # --- the starter's part: links ---------------------------------------------

    def link_store(self) -> LinkStore | None:
        if self.mode() in ("service", "user") and wanted(self.config_dir):
            return LinkStore(self.config_dir)
        return None

    def link_page_offered(self) -> bool:
        """The page is for a Windows service install: there the agent may read
        any connection's account, and nobody else can make a link (J36, J38)."""
        return sys.platform == "win32" and self.mode() == "service" and wanted(self.config_dir)

    def never_linked(self) -> frozenset[str]:
        """Eugene's own accounts, which are never a person's."""
        accounts = {"S-1-5-18"}
        with contextlib.suppress(Exception):
            accounts.add(account_sid(SITE_HOST_ACCOUNT))
        return frozenset(accounts)

    def links_changed(self) -> None:
        self._wake.set()

    def link_page(self) -> str | None:
        if not self.link_page_offered():
            return None
        port = getattr(self.app.state.settings, "bind_port", None) or 8079
        return f"http://127.0.0.1:{port}/link"

    # --- the site host's approval API (J14a, J53) ---------------------------------

    async def held(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """One call to the site host's `/v1/held` API on loopback."""
        manager = self.manager
        record = manager.store.get(HELPER_ID) if manager else None
        if manager is None or record is None or not record.enabled or not record.port:
            raise HostUnavailable("This machine's job site is not running.")
        token_file = manager.store.data_dir(HELPER_ID) / LOCAL_TOKEN_FILE
        try:
            token = (await asyncio.to_thread(token_file.read_text, encoding="utf-8")).strip()
        except OSError:
            raise HostUnavailable("This machine's job site has not started yet.") from None
        try:
            return await self._host.request(
                method,
                f"http://127.0.0.1:{record.port}{path}",
                headers={"Authorization": f"Bearer {token}"},
                **kwargs,
            )
        except httpx.HTTPError:
            raise HostUnavailable("This machine's job site did not answer.") from None

    # --- the site host's launch environment -------------------------------------

    def environment(self, mode: Mode) -> dict[str, str]:
        folder = site_dir(self.config_dir)
        environment = {
            "SITE_HOST_PROTECTED_ROOTS": json.dumps([str(self.config_dir)]),
            "SITE_HOST_LINKS_FILE": str(folder / "links.json"),
            "SITE_HOST_LOCAL_SERVERS_FILE": str(folder / SERVERS_FILE),
            "SITE_HOST_CHANNEL": channel_name(self.config_dir),
            # Not read by the host: a change restarts it, so it reads the new list.
            "SITE_HOST_SERVERS_STAMP": _stamp(folder / SERVERS_FILE),
        }
        if page := self.link_page():
            environment["SITE_HOST_LINK_PAGE"] = page
        return environment

    # --- install and start ------------------------------------------------------

    async def _retire_old(self, manager: AppManager) -> None:
        """We manage what we made: slice 2's host, under its old id, goes."""
        record = manager.store.get(RETIRED_ID)
        if record is not None and record.manifest.entry in SITE_HOST_ENTRIES:
            log.warning("removing %s: the operator-managed node folders retired", RETIRED_ID)
            await manager.uninstall(RETIRED_ID)

    async def reconcile(self, enabled: bool) -> None:
        manager = self.manager
        mode = self.mode()
        if manager is None or mode not in ("service", "user"):
            return
        await self._retire_old(manager)
        record = manager.store.get(HELPER_ID)
        if record is not None and record.manifest.entry not in SITE_HOST_ENTRIES:
            return
        if not enabled:
            if manager.installer.running(HELPER_ID):
                await manager.installer.cancel(HELPER_ID)
            if record is not None and (record.enabled or manager.supervisor.is_running(HELPER_ID)):
                record.enabled = False
                manager.store.put(record)
                await manager.stop(HELPER_ID)
            self.error, self._attempt_version, self.site = None, None, None
            await self._stop_workers()
            return
        assert mode is not None
        await asyncio.to_thread(site_dir(self.config_dir).mkdir, parents=True, exist_ok=True)
        environment = await asyncio.to_thread(self.environment, mode)
        want = await asyncio.to_thread(manifest, environment)
        if record is None or record.version != want.version or record.manifest.entry != ENTRY:
            await self._install(manager, want)
            return
        if dict(record.manifest.environment or {}) != environment:
            # The links page, the channel or the local servers changed: the
            # host learns them only at start.
            record.manifest = record.manifest.model_copy(update={"environment": environment})
            manager.store.put(record)
            if record.enabled:
                await manager.restart(record)
                return
        if not record.enabled:
            record.enabled = True
            manager.store.put(record)
            await manager.start(record)
        elif not manager.supervisor.is_running(HELPER_ID):
            await manager.start(record)

    async def _install(self, manager: AppManager, want: AppManifest) -> None:
        if manager.installer.running(HELPER_ID):
            return
        if self._attempt_version == want.version and time.perf_counter() < self._retry_at:
            progress = manager.installer.snapshot(HELPER_ID)
            self.error = (progress.error or progress.message) if progress else self.error
            return
        uv = manager.uv()
        if uv is None:
            self.error = "Eugene cannot find uv to prepare this job site. Repair its installation."
            return
        # A local credential for the app launcher and nothing else.
        key_file = manager.store.key_file(HELPER_ID)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        if not key_file.exists():
            write_private(key_file, secrets.token_urlsafe(32))
        manager.catalogue[HELPER_ID] = want
        manager.reserve_port(HELPER_ID)
        self._attempt_version, self._retry_at = want.version, time.perf_counter() + 300
        self.error = "Preparing this job site…"

        async def installed(value: AppManifest) -> None:
            await manager.installed_callback(value, AppOrigin.catalogue)
            self.error = None

        manager.installer.start(want, store=manager.store, uv=uv, on_installed=installed)

    # --- the starter's part: workers --------------------------------------------

    def _program(self, mode: Mode) -> WorkerProgram | None:
        """What every worker runs: the site host's own installed program."""
        manager = self.manager
        record = manager.store.get(HELPER_ID) if manager else None
        if manager is None or record is None or not record.enabled:
            return None
        python = venv_python(manager.store.version_dir(HELPER_ID, record.version) / "venv")
        if not python.exists():
            return None
        if mode == "service":
            try:
                host = account_sid(SITE_HOST_ACCOUNT)
            except Exception:
                return None
        else:
            host = _own_account()
        return WorkerProgram(
            python=python,
            channel=channel_name(self.config_dir),
            host=host,
            servers=servers_path(self.config_dir),
            protect=(self.config_dir, manager.store.root),
            shared=mode == "user",
        )

    def _grant_windows(self, links: LinkStore, program: WorkerProgram | None) -> None:
        """The read grants that go with the links: the site host's account
        reads `site\\`; each linked account reads the server list; every
        person may run the worker program, never write it."""
        manager = self.manager
        site_host_exists = True
        try:
            account_sid(SITE_HOST_ACCOUNT)
        except Exception:
            site_host_exists = False
        current = links.load()
        stamp = (
            tuple(link.account for link in current),
            _stamp(servers_path(self.config_dir)),
            site_host_exists,
            str(program.python) if program else None,
        )
        if stamp == self._granted:
            return
        protect_windows(self.config_dir, current, site_host_exists=site_host_exists)
        if program is not None and manager is not None:
            from .app_accounts import base_interpreter

            venv = program.python.parent.parent
            for folder in {venv.parent, base_interpreter(venv).parent}:
                _icacls(folder, "/grant", "*S-1-5-32-545:(OI)(CI)RX")
        self._granted = stamp

    async def _workers(self, mode: Mode) -> None:
        program = self._program(mode)
        if mode == "service" and sys.platform == "win32":
            links = LinkStore(self.config_dir)
            await asyncio.to_thread(self._grant_windows, links, program)
            if self._windows is None:
                self._windows = WindowsStarter(links)
            self._windows.program = program
            await asyncio.to_thread(self._windows.step)
        elif mode == "user":
            if self._child is None:
                self._child = ChildStarter(_own_account())
            self._child.program = program
            await self._child.step()

    async def _stop_workers(self) -> None:
        if self._windows is not None:
            await asyncio.to_thread(self._windows.close)
        if self._child is not None:
            await self._child.stop()

    # --- which site this machine hosts (J32) -----------------------------------

    async def _hosted(self, mode: Mode | None) -> str | None:
        port: int | None = None
        if mode == "root":
            try:
                text = await asyncio.to_thread(ROOT_SITE_FILE.read_text, encoding="utf-8")
                port = int(json.loads(text)["port"])
            except (OSError, ValueError, KeyError, TypeError):
                return None
        else:
            manager = self.manager
            record = manager.store.get(HELPER_ID) if manager else None
            if record is None or not record.enabled or not record.port:
                return None
            port = record.port
        try:
            answer = await self._host.get(f"http://127.0.0.1:{port}/healthz")
            site = answer.json().get("site")
        except (httpx.HTTPError, ValueError, AttributeError):
            return self.site
        return site if isinstance(site, str) and site else None

    async def _report(self) -> None:
        """Tell the root which site this node hosts, when it changed, and
        every ten minutes besides. Display only, so a failure is a debug line."""
        identity = self.app.state.node_identity.record
        if not identity.enrolled or not identity.control_url:
            return
        listed = (self.site,) if self.site else ()
        key = (str(identity.name), listed)
        if key == self._reported and time.perf_counter() < self._report_at:
            return
        root = str(identity.control_url).rstrip("/")
        if root != self._root:
            if self._root_client is not None:
                await self._root_client.aclose()
            self._root_client = client_for(root, timeout=15.0, follow_redirects=False)
            self._root = root
        assert self._root_client is not None
        token = self.app.state.auth_state.trust.agent_token("control")
        try:
            answer = await self._root_client.put(
                f"{root}/v1/nodes/{identity.name}/hosted-sites",
                json={"sites": list(listed)},
                headers={"Authorization": f"Bearer {token}"},
            )
            answer.raise_for_status()
        except httpx.HTTPError as exc:
            log.debug("could not tell the root which site this node hosts: %s", exc)
            return
        self._reported, self._report_at = key, time.perf_counter() + 600

    async def step(self) -> None:
        mode = self.mode()
        enabled = mode == "root" or await asyncio.to_thread(wanted, self.config_dir)
        await self.reconcile(enabled)
        if enabled and mode in ("service", "user"):
            await self._workers(mode)
        self.site = await self._hosted(mode) if enabled else None
        await self._report()

    async def run(self) -> None:
        try:
            while True:
                try:
                    await self.step()
                except (httpx.HTTPError, ValueError, KeyError, OSError, RuntimeError) as exc:
                    log.debug("site host supervision: %s", type(exc).__name__)
                self._wake.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), 5)
        finally:
            await self._stop_workers()
            await self._host.aclose()
            if self._root_client is not None:
                await self._root_client.aclose()


def _own_account() -> str:
    if sys.platform == "win32":
        import win32api
        import win32con
        import win32security

        token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        return str(win32security.ConvertSidToStringSid(user))
    return str(os.getuid())
