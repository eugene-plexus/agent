"""Durable, operator-authorized coordination of existing update endpoints.

The console's own agent holds the job, so the initiating session remains
exchangeable through the normal proxy. No new service-token write authority.
Persist before each write; on recovery observe its result, never replay it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request

from ._generated.models import (
    UpdateAllItem,
    UpdateAllItemKind,
    UpdateAllItemState,
    UpdateAllRun,
    UpdateAllRunState,
)
from ._http import internal_client
from ._private_files import write_private
from .dependencies import verify_bearer
from .tokens import TYP_SESSION

ACTIVE = {"pending", "checking", "updating"}
GOOD = {"updated", "current"}
POLL_SECONDS = 3.0
ITEM_TIMEOUT = timedelta(minutes=45)
log = logging.getLogger(__name__)


class UpdateAllError(Exception):
    def __init__(self, detail: str, status: int = 409) -> None:
        super().__init__(detail)
        self.status = status


class UpdateAll:
    def __init__(self, app: FastAPI, path: Path) -> None:
        self.app = app
        self.path = path
        self.run: UpdateAllRun | None = None
        self._token = ""
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._load_error = False
        self._client = internal_client(
            transport=httpx.ASGITransport(app=app), base_url="http://update-all.local"
        )
        if path.is_file():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.run = UpdateAllRun.model_validate(saved["run"])
                self._token = saved.get("token", "")
            except (OSError, ValueError, KeyError, TypeError):
                # A corrupt job must not prevent the agent itself starting,
                # nor authorize replacing an update with an unknown outcome.
                self._load_error = True
                log.error("Could not read saved Update all progress at %s", path)

    def _save(self) -> None:
        assert self.run is not None
        write_private(
            self.path,
            json.dumps({"run": self.run.model_dump(mode="json"), "token": self._token}),
        )

    def _authorize(self) -> None:
        request = Request({"type": "http", "app": self.app, "headers": []})
        try:
            claims = verify_bearer(request, self._token, classes=(TYP_SESSION,))
        except HTTPException as exc:
            raise UpdateAllError(
                "The initiating session expired or was signed out. "
                "Sign in and choose Update all again.",
                401,
            ) from exc
        if claims.act is not None:
            raise UpdateAllError("Start Update all from the console where you signed in.")
        if self.run and datetime.now(UTC) >= self.run.expiresAt:
            raise UpdateAllError("The two-hour update window ended. Choose Update all again.", 401)

    async def _call(
        self, target: str, path: str, method: str = "GET", body: Any = None
    ) -> dict[str, Any]:
        self._authorize()
        record = self.app.state.node_identity.record
        local = target == "agent" or (record and target == f"node:{record.name}")
        url = path if local else f"/api/proxy/{target}{path}"
        try:
            # ASGITransport itself has no network timeout. Bound the whole
            # proxy call, including a channel check's outgoing GitHub reads.
            async with asyncio.timeout(120 if method == "POST" else 15):
                response = await self._client.request(
                    method, url, json=body, headers={"authorization": f"Bearer {self._token}"}
                )
        except (httpx.HTTPError, TimeoutError) as exc:
            raise UpdateAllError("The machine did not answer in time.", 503) from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", "")
                if isinstance(detail, dict):
                    detail = detail.get("detail", detail.get("title", ""))
            except ValueError:
                detail = ""
            raise UpdateAllError(
                str(detail or f"The machine returned HTTP {response.status_code}."),
                response.status_code,
            )
        return dict(response.json())

    def resume(self) -> None:
        if (
            self.run
            and self.run.state.value == "running"
            and (self._task is None or self._task.done())
        ):
            self._task = asyncio.create_task(self._work(), name="update-all")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self._client.aclose()

    async def start(self, token: str) -> UpdateAllRun:
        async with self._lock:
            if self._load_error:
                raise UpdateAllError(
                    "Saved update progress could not be read. Check this agent's log."
                )
            if self.run and self.run.state.value == "running":
                return self.run
            old_token = self._token
            self._token = token
            old_run = self.run
            self.run = None
            try:
                nodes = (await self._call("control", "/v1/nodes"))["nodes"]
                # Enrollment records say 'agent' even for the initial root;
                # role=control is recorded by promotion. Ask who actually
                # hosts the active control process, as Machines does elsewhere.
                limit = asyncio.Semaphore(8)

                async def hosts_control(node: dict[str, Any]) -> str | None:
                    async with limit:
                        try:
                            identity = await self._call(f"node:{node['name']}", "/v1/node")
                        except UpdateAllError as exc:
                            if exc.status == 401:
                                raise
                            return None
                        return str(node["name"]) if identity.get("hostsControl") else None

                roots = [
                    name
                    for name in await asyncio.gather(*(hosts_control(n) for n in nodes))
                    if name
                ]
                if len(roots) != 1:
                    raise UpdateAllError(
                        "The install must have one available control root before updating."
                    )
                root = roots[0]
                names = [root, *sorted({node["name"] for node in nodes} - {root})]
                now = datetime.now(UTC)
                self.run = UpdateAllRun(
                    id=uuid4().hex,
                    root=root,
                    startedAt=now,
                    expiresAt=now + timedelta(hours=2),
                    state=UpdateAllRunState.running,
                    items=[
                        UpdateAllItem(
                            node=name,
                            kind=UpdateAllItemKind.node,
                            state=UpdateAllItemState.pending,
                            detail="Waiting",
                        )
                        for name in names
                    ],
                )
                self._save()
            except Exception:
                self.run, self._token = old_run, old_token
                raise
            self.resume()
            return self.run

    def _set(self, item: UpdateAllItem, state: str, detail: str) -> None:
        item.state = UpdateAllItemState(state)
        item.detail = detail
        self._save()

    def _begin(self, item: UpdateAllItem, target: str, detail: str) -> None:
        item.target = target
        item.startedAt = datetime.now(UTC)
        self._set(item, "updating", detail)

    async def _work(self) -> None:
        assert self.run is not None
        try:
            # Give the listener and the control component time to start after
            # this agent's own update. Every mutation still checks the session.
            await asyncio.sleep(POLL_SECONDS)
            for node in [i for i in self.run.items if i.kind.value == "node"]:
                self._authorize()
                if node.state.value in ACTIVE:
                    try:
                        await self._node(node)
                    except UpdateAllError as exc:
                        if exc.status == 401:
                            raise
                        self._set(node, "failed", str(exc))
                if node.node == self.run.root and node.state.value not in GOOD:
                    self._skip_pending(
                        "Update the control root successfully first, then choose Update all again."
                    )
                    break
                if node.state.value in GOOD:
                    await self._apps(node)
        except asyncio.CancelledError:
            # A normal shutdown preserves the job and its short-lived session.
            raise
        except Exception as exc:
            detail = (
                str(exc)
                if isinstance(exc, UpdateAllError)
                else "Update all stopped unexpectedly. Choose Update all again."
            )
            if not any(i.state.value in ACTIVE for i in self.run.items):
                self.run.items.append(
                    UpdateAllItem(
                        node=self.run.root,
                        kind=UpdateAllItemKind.app,
                        state=UpdateAllItemState.failed,
                        detail=detail,
                    )
                )
            self._skip_pending(detail)
        if any(i.state.value in ACTIVE for i in self.run.items):
            self._skip_pending("This item is no longer available. Check the machine or Apps.")
        self.run.state = (
            UpdateAllRunState.complete
            if all(i.state.value in GOOD for i in self.run.items)
            else UpdateAllRunState.attention
        )
        self.run.finishedAt = datetime.now(UTC)
        self._token = ""
        self._save()

    def _skip_pending(self, detail: str) -> None:
        assert self.run is not None
        for item in self.run.items:
            if item.state.value in ACTIVE:
                item.state = UpdateAllItemState.skipped
                item.detail = detail
        self._save()

    async def _node(self, item: UpdateAllItem) -> None:
        target = f"node:{item.node}"
        if item.state.value != "updating":
            self._set(item, "checking", "Checking this machine's update channel")
            view = await self._call(target, "/v1/node/update/check", "POST", {})
            running = view.get("running")
            if running:
                self._begin(item, running["target"], "Waiting for the update already running")
                item.startedAt = datetime.fromisoformat(running["startedAt"].replace("Z", "+00:00"))
                self._save()
            else:
                if view.get("error") or not view.get("newest"):
                    raise UpdateAllError(
                        view.get("error") or "No verified update target was returned."
                    )
                if not view.get("available"):
                    self._set(item, "current", "No newer update on this machine's channel")
                    return
                if not view["apply"]["possible"]:
                    steps = " ".join(
                        s["text"] + (" " + s["command"] if s.get("command") else "")
                        for s in view["apply"].get("steps", [])
                    )
                    self._set(
                        item,
                        "manual",
                        (
                            view["apply"].get("reason", "Update this machine manually.")
                            + " "
                            + steps
                        ).strip(),
                    )
                    return
                item.components = view["newest"].get("components", {})
                self._begin(
                    item, view["newest"]["ref"], "Updating; waiting for this machine to return"
                )
                try:
                    await self._call(target, "/v1/node/update", "POST", {"target": item.target})
                except UpdateAllError as exc:
                    # A lost reply can mean the installer already stopped the
                    # agent. Observe it; replaying POST could install twice.
                    if exc.status < 500:
                        raise
        await self._wait_node(item)

    def _timed_out(self, item: UpdateAllItem) -> bool:
        assert self.run is not None
        now = datetime.now(UTC)
        return now >= self.run.expiresAt or (
            item.startedAt is not None and now - item.startedAt > ITEM_TIMEOUT
        )

    async def _wait_node(self, item: UpdateAllItem) -> None:
        while not self._timed_out(item):
            try:
                identity = await self._call(f"node:{item.node}", "/v1/node")
                view = identity.get("update", {})
                last = view.get("last") or {}
                recent = (
                    datetime.fromisoformat(last["startedAt"].replace("Z", "+00:00"))
                    if last.get("startedAt")
                    else None
                )
                if (
                    last.get("target") == item.target
                    and recent
                    and item.startedAt
                    and recent >= item.startedAt - timedelta(seconds=2)
                ):
                    if last["outcome"] == "failed":
                        raise UpdateAllError(last.get("detail") or "The node update failed.")
                    if last["outcome"] == "succeeded" and not view.get("running"):
                        stamps = {
                            c["name"]: c.get("commit")
                            for c in identity.get("install", {}).get("components", [])
                            if c.get("state") != "missing"
                        }
                        if item.components and any(
                            stamps.get(k) != v for k, v in item.components.items()
                        ):
                            raise UpdateAllError(
                                "The installer finished, but the installed versions do not match "
                                "its target. Check this machine."
                            )
                        self._set(item, "updated", "Updated and back online")
                        return
            except UpdateAllError as exc:
                if exc.status < 500:
                    raise
            await asyncio.sleep(POLL_SECONDS)
        raise UpdateAllError(
            "This machine did not confirm its update within 45 minutes. "
            "Check it before trying again."
        )

    async def _apps(self, node: UpdateAllItem) -> None:
        assert self.run is not None
        target = f"node:{node.node}"
        # Fetch after the node update: its old catalogue cannot name new apps.
        try:
            catalogue = await self._call(target, "/v1/app-catalogue")
        except UpdateAllError as exc:
            if exc.status == 401:
                raise
            existing = next(
                (
                    i
                    for i in self.run.items
                    if i.node == node.node and i.kind.value == "app" and i.app is None
                ),
                None,
            )
            if existing is None:
                self.run.items.append(
                    UpdateAllItem(
                        node=node.node,
                        kind=UpdateAllItemKind.app,
                        state=UpdateAllItemState.failed,
                        detail=f"Could not check installed apps: {exc}",
                    )
                )
                self._save()
            return
        for entry in catalogue["apps"]:
            if entry.get("origin") != "catalogue" or not entry.get("installedVersion"):
                continue
            manifest = entry["manifest"]
            item = next(
                (i for i in self.run.items if i.node == node.node and i.app == manifest["id"]), None
            )
            if item is None:
                item = UpdateAllItem(
                    node=node.node,
                    kind=UpdateAllItemKind.app,
                    app=manifest["id"],
                    state=UpdateAllItemState.pending,
                    detail="Waiting",
                )
                self.run.items.append(item)
                self._save()
            if item.state.value not in ACTIVE:
                continue
            try:
                await self._app(item, entry, target)
            except UpdateAllError as exc:
                if exc.status == 401:
                    raise
                self._set(item, "failed", str(exc))

    async def _app(self, item: UpdateAllItem, entry: dict[str, Any], target: str) -> None:
        path = f"/v1/apps/{item.app}"
        if item.state.value != "updating":
            if entry["installedVersion"] == entry["manifest"]["version"]:
                self._set(item, "current", "Already at the catalogue version")
                return
            # Observe an independently started install without replacing it.
            try:
                existing = await self._call(target, path + "/install")
            except UpdateAllError as exc:
                if exc.status != 404:
                    raise
                existing = {}
            in_flight = existing.get("state") not in (None, "done", "failed", "cancelled")
            self._begin(item, entry["manifest"]["version"], "Updating installed app")
            if not in_flight:
                try:
                    await self._call(target, path + "/install", "POST", {})
                except UpdateAllError as exc:
                    if exc.status < 500:
                        raise
        while not self._timed_out(item):
            try:
                app = await self._call(target, path)
                try:
                    progress = await self._call(target, path + "/install")
                except UpdateAllError as exc:
                    if exc.status != 404:
                        raise
                    progress = {}
                if progress.get("state") in ("failed", "cancelled"):
                    raise UpdateAllError(progress.get("error") or "The app update did not finish.")
                if app["version"] == item.target and progress.get("state") in (None, "done"):
                    if app.get("status") == "crashed":
                        raise UpdateAllError(
                            "The app updated but could not start. Check it in Apps."
                        )
                    self._set(item, "updated", "Updated to " + str(item.target))
                    return
                if progress.get("state") in (None, "done"):
                    raise UpdateAllError(
                        progress.get("error")
                        or "The app did not reach the requested version. Retry from Apps."
                    )
            except UpdateAllError as exc:
                if exc.status < 500:
                    raise
            await asyncio.sleep(POLL_SECONDS)
        raise UpdateAllError(
            "The app did not confirm its update within 45 minutes. Check it in Apps."
        )


def manager(app: FastAPI) -> UpdateAll:
    found: UpdateAll | None = getattr(app.state, "update_all", None)
    if found is None:
        found = UpdateAll(app, app.state.settings.config_file.parent / "update-all.json")
        app.state.update_all = found
    return found
