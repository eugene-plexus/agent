"""This machine's site host: installed and kept running, never relayed to
(`docs/design/job-sites-own-enrollment.md`, J19, J21, J23).

A Job Site is its own enrollment, held by the site host (`eugene-plexus/
site-host`): its key, its root, its owner, its policy. It joins, polls and
answers its root itself. Until the standalone install (J21), the agent on a
node does three things for it and nothing else:

1. **Installs it** at a pinned commit, as a bundled app in an OS account of
   its own (C1), never inside this privileged process, when an
   administrator turned it on here (`site join`, which writes
   `site-host.json`).
2. **Tells it what only the agent may say**, in its launch environment:
   the install's private directory it must never touch, and the local
   servers a machine administrator added here (`site-servers.yaml`,
   remote-nodes.md §6.2). Never who the site is, nor who owns it: that is
   the site's own enrollment.
3. **Says which site it hosts** (J32), so the console can link the node
   and the site. The site's id comes from the host's own health answer;
   the root records it for display only.

The agent holds none of the site's identity, and the node's enrollment is
unchanged by any of this: the two share nothing but the machine.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI

from ._generated.models import AppManifest, AppOrigin
from ._generated.site_host_models import SiteLocalServerList
from ._http import client_for, internal_client
from ._private_files import write_private
from .apps import SITE_HOST_ENTRIES, AppManager

log = logging.getLogger(__name__)

HELPER_ID = "site-host"
#: The id slice 2's host ran under, and the helper before it. Removed when
#: found: the operator-managed node folders retired (J20).
RETIRED_ID = "node-files"
PACKAGE = "eugene-plexus-site-host"
ENTRY = "eugene_plexus_site_host"

SITE_HOST_COMMIT = "38d7ed8c0185440c6fc9abc6139fc48834d614f2"
SITE_HOST_SOURCE = f"https://github.com/eugene-plexus/site-host/archive/{SITE_HOST_COMMIT}.tar.gz"
#: A checkout to install instead, for development and the acceptance runs.
SOURCE_OVERRIDE = "EUGENE_PLEXUS_AGENT_SITE_HOST_SOURCE"
SERVERS_FILE = "site-servers.yaml"
#: The host's copy, beside its install: readable by its account, not writable.
SERVERS_COPY = "site-servers.json"
#: Written by an administrator at the machine (`site join`, `site leave`).
WANTED_FILE = "site-host.json"
#: Where a Linux system install's apps keep their state (systemd's
#: `StateDirectory=eugene-plexus-apps/%i`).
SYSTEMD_STATE_ROOT = Path("/var/lib/eugene-plexus-apps")
REPORT_SECONDS = 60.0


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


def data_dir(store_data_dir: Path, account_kind: str | None) -> Path:
    """Where the site host keeps the site: its enrollment, key, policy and log."""
    if account_kind == "systemd":
        return SYSTEMD_STATE_ROOT / HELPER_ID
    return store_data_dir


def local_servers(config_dir: Path) -> list[dict[str, Any]]:
    """The servers an administrator added at this machine. A file that does
    not read is no servers, logged: it never stops the site."""
    path = config_dir / SERVERS_FILE
    if not path.exists():
        return []
    try:
        value = SiteLocalServerList.model_validate(yaml.safe_load(path.read_text("utf-8")) or {})
    except (OSError, ValueError) as exc:
        log.warning("%s could not be read (%s); no local servers run", path, type(exc).__name__)
        return []
    return [s.model_dump(mode="json", exclude_none=True) for s in value.servers]


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

    @property
    def manager(self) -> AppManager | None:
        return getattr(self.app.state, "apps", None)

    @property
    def config_dir(self) -> Path:
        return Path(self.app.state.settings.config_file).resolve().parent

    def environment(self, manager: AppManager) -> dict[str, str]:
        return {
            "SITE_HOST_PROTECTED_ROOTS": json.dumps([str(self.config_dir)]),
            **self._servers_copy(manager),
        }

    def _servers_copy(self, manager: AppManager) -> dict[str, str]:
        """The local servers, copied where the host may read and not write
        them, named with their SHA-256. A list may outgrow one variable, and
        an argument may hold braces the launcher would read as a placeholder."""
        data = json.dumps(local_servers(self.config_dir), ensure_ascii=False).encode()
        path = manager.store.app_dir(HELPER_ID) / SERVERS_COPY
        if not path.exists() or path.read_bytes() != data:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            os.chmod(temporary, 0o644)
            os.replace(temporary, path)
        return {
            "SITE_HOST_LOCAL_SERVERS_FILE": str(path),
            "SITE_HOST_LOCAL_SERVERS_SHA256": hashlib.sha256(data).hexdigest(),
        }

    # --- install and start ------------------------------------------------------

    async def _retire_old(self, manager: AppManager) -> None:
        """We manage what we made: slice 2's host, under its old id, goes."""
        record = manager.store.get(RETIRED_ID)
        if record is not None and record.manifest.entry in SITE_HOST_ENTRIES:
            log.warning("removing %s: the operator-managed node folders retired", RETIRED_ID)
            await manager.uninstall(RETIRED_ID)

    async def reconcile(self, enabled: bool) -> None:
        manager = self.manager
        if manager is None or not manager.accounts.available:
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
            return
        environment = await asyncio.to_thread(self.environment, manager)
        want = await asyncio.to_thread(manifest, environment)
        if record is None or record.version != want.version or record.manifest.entry != ENTRY:
            await self._install(manager, want)
            return
        if dict(record.manifest.environment or {}) != environment:
            # The local servers changed: the host learns them only at start.
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

    # --- which site this machine hosts (J32) -----------------------------------

    async def _hosted(self) -> str | None:
        manager = self.manager
        record = manager.store.get(HELPER_ID) if manager else None
        if record is None or not record.enabled or not record.port:
            return None
        try:
            answer = await self._host.get(f"http://127.0.0.1:{record.port}/healthz")
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
        enabled = await asyncio.to_thread(wanted, self.config_dir)
        await self.reconcile(enabled)
        self.site = await self._hosted() if enabled else None
        await self._report()

    async def run(self) -> None:
        try:
            while True:
                try:
                    await self.step()
                except (httpx.HTTPError, ValueError, KeyError, OSError, RuntimeError) as exc:
                    log.debug("site host supervision: %s", type(exc).__name__)
                await asyncio.sleep(5)
        finally:
            await self._host.aclose()
            if self._root_client is not None:
                await self._root_client.aclose()
