"""Bundled file capability: one enrolled node, an isolated worker, outbound delivery."""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import importlib.resources
import io
import json
import logging
import secrets
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from . import app_accounts
from ._generated.models import AppManifest, AppOrigin
from ._http import client_for, internal_client
from ._private_files import write_private
from .apps import AppManager

log = logging.getLogger(__name__)
HELPER_ID = "node-files"
PACKAGE = "eugene_plexus_node_helper"


def manifest(manager: AppManager, private_root: Path) -> AppManifest:
    """A dependency-free wheel built exclusively from this agent release's resources."""
    source = importlib.resources.files("eugene_plexus_agent").joinpath("_node_file_helper")
    files = {
        f"{PACKAGE}/{f.name}": f.read_bytes() for f in source.iterdir() if f.name.endswith(".py")
    }
    digest = hashlib.sha256(b"".join(files[k] for k in sorted(files))).hexdigest()[:16]
    version = "0.1.0+" + digest
    directory = manager.store.root / ".bundled"
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / f"{PACKAGE}-{version}-py3-none-any.whl"
    if not wheel.exists():
        dist = f"{PACKAGE}-{version}.dist-info"
        files[f"{dist}/METADATA"] = (
            f"Metadata-Version: 2.1\nName: eugene-plexus-node-helper\n"
            f"Version: {version}\nRequires-Python: >=3.12\n"
        ).encode()
        files[f"{dist}/WHEEL"] = b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        record = io.StringIO(newline="")
        writer = csv.writer(record)
        for name, data in files.items():
            hash_value = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            )
            writer.writerow([name, "sha256=" + hash_value, str(len(data))])
        writer.writerow([f"{dist}/RECORD", "", ""])
        files[f"{dist}/RECORD"] = record.getvalue().encode()
        temporary = wheel.with_suffix(".tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        temporary.replace(wheel)
    return AppManifest.model_validate(
        {
            "id": HELPER_ID,
            "name": "Node file support",
            "summary": "Managed by Eugene's node file capability. Assign folder access in People.",
            "source": str(wheel.resolve()),
            "version": version,
            "package": "eugene-plexus-node-helper",
            "entry": PACKAGE,
            "python": "3.12",
            "ui": False,
            "configTrio": False,
            "uses": [],
            "signIn": False,
            "localActions": True,
            "environment": {
                "NODE_HELPER_PROTECTED_ROOTS": json.dumps([str(private_root.resolve())])
            },
        }
    )


class NodeFileHelper:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self._root_client: httpx.AsyncClient | None = None
        self._root: str | None = None
        self._worker = internal_client(timeout=15.0, follow_redirects=False)
        self._manifest: AppManifest | None = None
        self._error: str | None = None
        self._attempt_version: str | None = None
        self._retry_at = 0.0

    @property
    def manager(self) -> AppManager | None:
        return getattr(self.app.state, "apps", None)

    def report(self) -> dict[str, Any]:
        manager = self.manager
        if manager is None:
            return {
                "supported": False,
                "ready": False,
                "reason": "This node has no app-account manager.",
            }
        if not manager.accounts.available:
            return {"supported": False, "ready": False, "reason": manager.accounts.reason}
        record = manager.store.get(HELPER_ID)
        reason: str | None
        if record is not None and record.manifest.entry != PACKAGE:
            return {
                "supported": False,
                "ready": False,
                "reason": "A custom app uses the reserved node-files name. "
                "Uninstall that custom app in Apps before enabling file support.",
            }
        if record is None:
            reason = self._error or "File support is disabled."
            ready = False
        else:
            state, detail, _, _, _ = manager.supervisor.status(HELPER_ID)
            ready = record.enabled and state.value == "running"
            progress = manager.installer.snapshot(HELPER_ID)
            reason = (
                None
                if ready
                else (
                    self._error
                    or detail
                    or (progress.message if progress else "File support is starting.")
                )
            )
        return {
            "supported": True,
            "ready": ready,
            "reason": reason[:1024] if reason else None,
            "account": app_accounts.account_name(str(manager.accounts.kind), HELPER_ID),
        }

    async def reconcile(self, config: dict[str, Any]) -> None:
        manager = self.manager
        if manager is None or not manager.accounts.available:
            return
        record = manager.store.get(HELPER_ID)
        if not config["enabled"]:
            if record is not None and record.manifest.entry != PACKAGE:
                return
            if manager.installer.running(HELPER_ID):
                await manager.installer.cancel(HELPER_ID)
            if record is not None and (record.enabled or manager.supervisor.is_running(HELPER_ID)):
                record.enabled = False
                manager.store.put(record)
                await manager.stop(HELPER_ID)
            self._error = None
            self._attempt_version = None
            return
        if record is not None and record.manifest.entry != PACKAGE:
            return
        if self._manifest is None:
            self._manifest = await asyncio.to_thread(
                manifest, manager, self.app.state.settings.config_file.resolve().parent
            )
        wanted = self._manifest
        if record is None or record.version != wanted.version:
            if manager.installer.running(HELPER_ID):
                return
            if self._attempt_version == wanted.version and time.perf_counter() < self._retry_at:
                progress = manager.installer.snapshot(HELPER_ID)
                self._error = (progress.error or progress.message) if progress else self._error
                return
            uv = manager.uv()
            if uv is None:
                self._error = (
                    "Eugene cannot find uv to prepare the file helper. "
                    "Repair this node's installation."
                )
                return
            # Only a local worker credential exists; no inference key is issued.
            key_file = manager.store.key_file(HELPER_ID)
            key_file.parent.mkdir(parents=True, exist_ok=True)
            if not key_file.exists():
                write_private(key_file, secrets.token_urlsafe(32))
            manager.catalogue[HELPER_ID] = wanted
            manager.reserve_port(HELPER_ID)
            self._attempt_version, self._retry_at = wanted.version, time.perf_counter() + 300
            self._error = "Preparing the isolated file helper…"

            async def installed(value: AppManifest) -> None:
                await manager.installed_callback(value, AppOrigin.catalogue)
                self._error = None

            manager.installer.start(wanted, store=manager.store, uv=uv, on_installed=installed)
            return
        if not record.enabled:
            record.enabled = True
            manager.store.put(record)
            await manager.start(record)
        elif not manager.supervisor.is_running(HELPER_ID):
            await manager.start(record)

    def validate(self, command: dict[str, Any], config: dict[str, Any]) -> None:
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
        if command.get("tool") == "inspect":
            if command.get("subject") != "operator":
                raise ValueError("Only an operator may register a folder.")
            return
        grant = command.get("folder") or {}
        folder = next((f for f in config["folders"] if f["id"] == grant.get("id")), None)
        if (
            folder is None
            or any(grant.get(k) != folder[k] for k in ("path", "identity"))
            or not command.get("subject")
            or grant.get("subject") != command["subject"]
            or (grant.get("writable") and not folder["writable"])
            or (command.get("tool") == "write_text" and grant.get("writable") is not True)
        ):
            raise ValueError("This operation exceeds the node's current folder grant.")

    async def perform(self, command: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        try:
            self.validate(command, config)
        except ValueError as exc:
            return {"status": "failed", "message": str(exc)}
        manager = self.manager
        record = manager.store.get(HELPER_ID) if manager else None
        token = manager.supervisor.admin_token(HELPER_ID) if manager else None
        if record is None or not record.enabled or not token:
            return {"status": "failed", "message": "The isolated file helper is not running."}
        try:
            response = await self._worker.post(
                f"http://127.0.0.1:{record.port}/execute",
                json=command,
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            if len(response.content) > 70_000:
                raise ValueError("oversized result")
            value: dict[str, Any] = response.json()
            return value
        except (httpx.HTTPError, ValueError):
            return {
                "status": "uncertain" if command.get("tool") == "write_text" else "failed",
                "message": "The isolated file helper did not confirm the operation. "
                "Check the file before retrying.",
            }

    async def step(self) -> None:
        identity = self.app.state.node_identity.record
        if not identity.enrolled:
            await self.reconcile({"enabled": False})
            await asyncio.sleep(2)
            return
        root = str(identity.control_url).rstrip("/")
        if root != self._root:
            if self._root_client is not None:
                await self._root_client.aclose()
            self._root_client = client_for(root, timeout=15.0, follow_redirects=False)
            self._root = root
        assert self._root_client is not None
        headers = {
            "Authorization": "Bearer " + self.app.state.auth_state.trust.agent_token("control")
        }
        response = await self._root_client.post(
            root + "/v1/node-helpers/poll", json=self.report(), headers=headers
        )
        response.raise_for_status()
        value = response.json()
        config = value["configuration"]
        await self.reconcile(config)
        ident = value.get("operation")
        if ident:
            # Claim immediately before execution: the root rechecks all permissions.
            claim = await self._root_client.post(
                root + f"/v1/node-helpers/operations/{ident}/claim", headers=headers
            )
            claim.raise_for_status()
            outcome = await self.perform(claim.json(), config)
            # Never retry execution if the result acknowledgement is lost.
            answer = await self._root_client.post(
                root + f"/v1/node-helpers/operations/{ident}/result", json=outcome, headers=headers
            )
            answer.raise_for_status()

    async def run(self) -> None:
        try:
            while True:
                try:
                    await self.step()
                except (httpx.HTTPError, ValueError, KeyError, OSError, RuntimeError) as exc:
                    # Normal for older/offline roots. Do not create an attention issue.
                    log.debug("node file relay unavailable: %s", type(exc).__name__)
                    await asyncio.sleep(5)
        finally:
            await self._worker.aclose()
            if self._root_client is not None:
                await self._root_client.aclose()
