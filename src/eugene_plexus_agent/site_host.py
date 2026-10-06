"""This machine's site host: installed, supervised and relayed to (J6, J8).

The host (`eugene-plexus/site-host`, contract `specs/openapi/site-host.yaml`)
is the local MCP host whose policy is final on this machine. The agent does
three things for it and decides nothing about a tool call:

1. **Installs it** at a pinned commit, as a bundled app in an OS account of
   its own (C1), never inside this privileged process (J5, J6f).
2. **Tells it what only the agent may say**, in its launch environment:
   whether this machine is a job site, the owner it pinned at its join
   (J6b), the install's private directory it must never touch, and the
   local servers a machine administrator added here (`site-servers.yaml`,
   §6.2).
3. **Relays the root's queue**: it polls the root, claims an operation,
   checks it is bound to this machine's enrolment and still in time, and
   hands it to the host as it came. The host's answer goes back as it came.

Every connection here is opened by this machine (`remote-nodes.md` §2.2).
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

from . import app_accounts
from ._generated.models import AppManifest, AppOrigin
from ._generated.site_host_models import SiteLocalServerList
from ._http import client_for, internal_client
from ._private_files import write_private
from .apps import NODE_FILES_ENTRIES, AppManager

log = logging.getLogger(__name__)

HELPER_ID = "node-files"
"""Kept from the helper it replaces: the OS account an owner gave folder
permission to is named after it, so a new id would strand every grant."""
PACKAGE = "eugene-plexus-site-host"
ENTRY = "eugene_plexus_site_host"
#: The entries this app id may hold: the host, and the helper it replaces,
#: until the next reconcile installs the host over it.
MANAGED_ENTRIES = NODE_FILES_ENTRIES
PROTOCOL = "mcp-2026-07-28"

SITE_HOST_COMMIT = "38d7ed8c0185440c6fc9abc6139fc48834d614f2"
SITE_HOST_SOURCE = f"https://github.com/eugene-plexus/site-host/archive/{SITE_HOST_COMMIT}.tar.gz"
#: A checkout to install instead, for development and the acceptance runs.
SOURCE_OVERRIDE = "EUGENE_PLEXUS_AGENT_SITE_HOST_SOURCE"
SERVERS_FILE = "site-servers.yaml"
#: The host's copy, beside its install: readable by its account, not writable.
SERVERS_COPY = "site-servers.json"
MAX_RESULT = 70_000


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


def local_servers(config_dir: Path) -> list[dict[str, Any]]:
    """The servers an administrator added at this machine. A file that does
    not read is no servers, logged: it never stops file support."""
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
            "name": "Node file support",
            "summary": "This machine's tools for Workbench. Managed by Eugene.",
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


class SiteHostRelay:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self._root_client: httpx.AsyncClient | None = None
        self._root: str | None = None
        self._host = internal_client(timeout=25.0, follow_redirects=False)
        self._error: str | None = None
        self._attempt_version: str | None = None
        self._retry_at = 0.0
        self._warned_owner: str | None = None

    @property
    def manager(self) -> AppManager | None:
        return getattr(self.app.state, "apps", None)

    @property
    def config_dir(self) -> Path:
        return Path(self.app.state.settings.config_file).resolve().parent

    def environment(self, manager: AppManager) -> dict[str, str] | None:
        """What the host is told at start; None while a site's owner is unknown."""
        identity = self.app.state.node_identity.record
        env = {"SITE_HOST_PROTECTED_ROOTS": json.dumps([str(self.config_dir)])}
        if not getattr(identity, "job_site", None):
            return {**env, "SITE_HOST_MODE": "node"}
        if not identity.site_owner:
            return None
        return {
            **env,
            "SITE_HOST_MODE": "site",
            "SITE_HOST_OWNER": identity.site_owner,
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

    # --- what this machine reports ---------------------------------------------

    def _availability(self) -> dict[str, Any]:
        manager = self.manager
        if manager is None:
            return {"supported": False, "ready": False, "reason": "This node has no app manager."}
        if not manager.accounts.available:
            return {"supported": False, "ready": False, "reason": manager.accounts.reason}
        record = manager.store.get(HELPER_ID)
        if record is not None and record.manifest.entry not in MANAGED_ENTRIES:
            return {
                "supported": False,
                "ready": False,
                "reason": "A custom app uses the reserved node-files name. "
                "Uninstall that custom app in Apps before enabling file support.",
            }
        account = app_accounts.account_name(str(manager.accounts.kind), HELPER_ID)
        if record is None:
            return {
                "supported": True,
                "ready": False,
                "reason": self._error or "File support is disabled.",
                "account": account,
            }
        state, detail, _, _, _ = manager.supervisor.status(HELPER_ID)
        running = record.enabled and state.value == "running" and record.manifest.entry == ENTRY
        if running:
            return {"supported": True, "ready": True, "reason": None, "account": account}
        progress = manager.installer.snapshot(HELPER_ID)
        reason = self._error or detail or (progress.message if progress else None) or "Starting."
        return {"supported": True, "ready": False, "reason": reason[:1024], "account": account}

    async def report(self) -> dict[str, Any]:
        value = self._availability()
        if not value["ready"]:
            return value
        try:
            response = await self._ask("GET", "/v1/report")
            host = response.json()
        except (httpx.HTTPError, ValueError, RuntimeError):
            return {**value, "ready": False, "reason": "This machine's tools are starting."}
        return {
            **value,
            "ready": bool(host.get("ready")),
            "reason": host.get("reason"),
            "protocol": host.get("protocol"),
            "hostVersion": host.get("version"),
            "site": host.get("site"),
        }

    async def _ask(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        manager = self.manager
        record = manager.store.get(HELPER_ID) if manager else None
        token = manager.supervisor.admin_token(HELPER_ID) if manager else None
        if record is None or not record.enabled or not token:
            raise RuntimeError("the site host is not running")
        response = await self._host.request(
            method,
            f"http://127.0.0.1:{record.port}{path}",
            headers={"Authorization": f"Bearer {token}"},
            **kwargs,
        )
        response.raise_for_status()
        return response

    # --- install and start ------------------------------------------------------

    async def reconcile(self, config: dict[str, Any]) -> None:
        manager = self.manager
        if manager is None or not manager.accounts.available:
            return
        record = manager.store.get(HELPER_ID)
        if record is not None and record.manifest.entry not in MANAGED_ENTRIES:
            return
        if not config["enabled"]:
            if manager.installer.running(HELPER_ID):
                await manager.installer.cancel(HELPER_ID)
            if record is not None and (record.enabled or manager.supervisor.is_running(HELPER_ID)):
                record.enabled = False
                manager.store.put(record)
                await manager.stop(HELPER_ID)
            self._error, self._attempt_version = None, None
            return
        environment = await asyncio.to_thread(self.environment, manager)
        if environment is None:
            self._error = "Waiting for the root to name this job site's owner."
            return
        wanted = await asyncio.to_thread(manifest, environment)
        if record is None or record.version != wanted.version or record.manifest.entry != ENTRY:
            await self._install(manager, wanted)
            return
        if dict(record.manifest.environment or {}) != environment:
            # The owner, the local servers or the mode changed: the host
            # learns them only at start, so it restarts with them.
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

    async def _install(self, manager: AppManager, wanted: AppManifest) -> None:
        if manager.installer.running(HELPER_ID):
            return
        if self._attempt_version == wanted.version and time.perf_counter() < self._retry_at:
            progress = manager.installer.snapshot(HELPER_ID)
            self._error = (progress.error or progress.message) if progress else self._error
            return
        uv = manager.uv()
        if uv is None:
            self._error = (
                "Eugene cannot find uv to prepare this machine's tools. Repair its installation."
            )
            return
        # A local credential for the host and nothing else: no inference key.
        key_file = manager.store.key_file(HELPER_ID)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        if not key_file.exists():
            write_private(key_file, secrets.token_urlsafe(32))
        manager.catalogue[HELPER_ID] = wanted
        manager.reserve_port(HELPER_ID)
        self._attempt_version, self._retry_at = wanted.version, time.perf_counter() + 300
        self._error = "Preparing this machine's tools…"

        async def installed(value: AppManifest) -> None:
            await manager.installed_callback(value, AppOrigin.catalogue)
            self._error = None

        manager.installer.start(wanted, store=manager.store, uv=uv, on_installed=installed)

    # --- relay ------------------------------------------------------------------

    def validate(self, command: dict[str, Any], config: dict[str, Any]) -> None:
        """Bound to this machine's enrolment, in time, and of a kind the host
        takes. Whether the call may run is the host's to decide (J8)."""
        identity = self.app.state.node_identity.record
        if (
            not identity.enrolled
            or not config["enabled"]
            or command.get("node") != identity.name
            or command.get("nodeKey") != identity.signing_public_key
            or command.get("enrolledAt") != config["enrolledAt"]
            or command.get("nodeKey") != config["nodeKey"]
            or not isinstance(command.get("expiresAt"), int | float)
            or not time.time() < command["expiresAt"] <= time.time() + 30
        ):
            raise ValueError("This operation does not belong to this enrolled node or has expired.")
        if command.get("kind") not in ("mcp", "manage") or not command.get("subject"):
            raise ValueError("This Eugene sent an operation this machine does not take. Update it.")
        if getattr(identity, "job_site", None):
            return
        # An ordinary node: the root's grants are final (J6d), and must be
        # folders it registered here, as it registered them.
        folders = {f["id"]: f for f in config.get("folders") or []}
        for grant in command.get("grants") or []:
            folder = folders.get(grant.get("folderId"))
            if (
                folder is None
                or any(grant.get(k) != folder[k] for k in ("path", "identity"))
                or (grant.get("writable") and not folder["writable"])
            ):
                raise ValueError("This operation exceeds the node's current folder grant.")

    async def perform(self, command: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        try:
            self.validate(command, config)
        except ValueError as exc:
            return {"status": "failed", "message": str(exc)}
        common = {
            "id": command["id"],
            "expiresAt": command["expiresAt"],
            "subject": command["subject"],
        }
        acting = False
        if command["kind"] == "mcp":
            request = command.get("request") or {}
            acting = request.get("method") == "tools/call"
            path, body = (
                "/v1/mcp",
                {
                    **common,
                    "server": command.get("server"),
                    "request": request,
                    "grants": command.get("grants") or [],
                    "installMode": command.get("installMode") or "production",
                },
            )
        else:
            path, body = (
                "/v1/manage",
                {**common, "action": command.get("action"), "arguments": command.get("arguments")},
            )
        try:
            response = await self._ask("POST", path, json=body)
            if len(response.content) > MAX_RESULT:
                raise ValueError("oversized answer")
            value: dict[str, Any] = response.json()
        except (httpx.HTTPError, ValueError, RuntimeError):
            return {
                "status": "uncertain" if acting else "failed",
                "message": "This machine's tools did not confirm the operation."
                + (" It may have run; check before trying again." if acting else ""),
            }
        answer = {k: v for k, v in value.items() if v is not None}
        if isinstance(answer.get("message"), str):
            answer["message"] = answer["message"][:1024]
        return answer

    def _pin_owner(self, owner: Any) -> None:
        """A job site's owner is pinned once; the root naming someone else
        later changes nothing (rule 2 of §3.3)."""
        identity = self.app.state.node_identity.record
        if not getattr(identity, "job_site", None) or not isinstance(owner, str) or not owner:
            return
        pinned = self.app.state.node_identity.pin_site_owner(owner)
        if pinned != owner and self._warned_owner != owner:
            self._warned_owner = owner
            log.warning(
                "the root named a different owner for this job site; it keeps the one it "
                "pinned at its join"
            )

    async def step(self) -> None:
        identity = self.app.state.node_identity.record
        if not identity.enrolled:
            await self.reconcile({"enabled": False})
            await asyncio.sleep(2)
            return
        root = str(identity.control_url).rstrip("/")
        link = getattr(self.app.state, "root_link", None)
        if getattr(identity, "job_site", None) and link is not None:
            # A job site reaches its root pinned (J7a) and, off its own
            # network, through the system's proxy.
            post = self._through(link)
        else:
            if root != self._root:
                if self._root_client is not None:
                    await self._root_client.aclose()
                self._root_client = client_for(root, timeout=15.0, follow_redirects=False)
                self._root = root
            assert self._root_client is not None
            post = self._direct(self._root_client, root)
        headers = {
            "Authorization": "Bearer " + self.app.state.auth_state.trust.agent_token("control")
        }
        response = await post("/v1/node-helpers/poll", json=await self.report(), headers=headers)
        response.raise_for_status()
        value = response.json()
        config = value["configuration"]
        self._pin_owner(value.get("siteOwner"))
        await self.reconcile(config)
        ident = value.get("operation")
        if ident:
            # Claim immediately before running: the root checks permission again.
            claim = await post(f"/v1/node-helpers/operations/{ident}/claim", headers=headers)
            claim.raise_for_status()
            outcome = await self.perform(claim.json(), config)
            # Never run it again if the result acknowledgement is lost.
            answer = await post(
                f"/v1/node-helpers/operations/{ident}/result", json=outcome, headers=headers
            )
            answer.raise_for_status()

    @staticmethod
    def _direct(client: httpx.AsyncClient, root: str) -> Any:
        async def post(path: str, **kwargs: Any) -> httpx.Response:
            return await client.post(root + path, **kwargs)

        return post

    @staticmethod
    def _through(link: Any) -> Any:
        async def post(path: str, **kwargs: Any) -> httpx.Response:
            response: httpx.Response = await link.request("POST", path, **kwargs)
            return response

        return post

    async def run(self) -> None:
        try:
            while True:
                try:
                    await self.step()
                except (httpx.HTTPError, ValueError, KeyError, OSError, RuntimeError) as exc:
                    # Normal for an older or offline root. Not an attention issue.
                    log.debug("site host relay unavailable: %s", type(exc).__name__)
                    await asyncio.sleep(5)
        finally:
            await self._host.aclose()
            if self._root_client is not None:
                await self._root_client.aclose()
