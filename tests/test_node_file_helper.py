"""Bundled worker, real file handles, independent scope checks and outbound delivery."""

from __future__ import annotations

import copy
import os
import secrets
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import app_accounts
from eugene_plexus_agent._node_file_helper import folder_io
from eugene_plexus_agent._node_file_helper.__main__ import Handler, Server, Worker
from eugene_plexus_agent.apps import AppStore, load_catalogue, validate_manifest
from eugene_plexus_agent.node_file_helper import HELPER_ID, NodeFileHelper, manifest


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> Worker:
    # Exercises held-handle IO, not OS service isolation (the C1 acceptance covers that).
    monkeypatch.setenv(
        "EUGENE_PLEXUS_APP_ACCOUNT_KIND", "windows_service" if os.name == "nt" else "systemd"
    )
    return Worker([])


def command(root: Path, tool: str = "read_text", **args: Any) -> dict[str, Any]:
    return {
        "id": secrets.token_hex(16),
        "expiresAt": time.time() + 20,
        "subject": "ada",
        "tool": tool,
        "arguments": args or {"path": "note.txt"},
        "folder": {
            "id": "f1",
            "subject": "ada",
            "path": str(root),
            "identity": folder_io.inspect(str(root), []),
            "writable": True,
        },
    }


def test_worker_real_read_write_create_and_replay(worker: Worker, tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("Original", encoding="utf-8")
    read = command(tmp_path)
    result = worker.execute(read)
    assert result["text"] == "Original"
    with pytest.raises(folder_io.FolderError, match="already used"):
        worker.execute(read)
    write = command(
        tmp_path, "write_text", path="note.txt", text="Updated", expectedSha256=result["sha256"]
    )
    worker.execute(write)
    assert path.read_text() == "Updated"
    with pytest.raises(folder_io.FolderError):
        worker.execute({**write, "id": "stale-file-version"})
    worker.execute(command(tmp_path, "write_text", path="new.txt", text="New", expectedSha256=""))
    assert (tmp_path / "new.txt").read_text() == "New"
    worker.execute(
        command(tmp_path, "write_text", path="unicode.txt", text="🚀" * 8192, expectedSha256="")
    )
    assert (tmp_path / "unicode.txt").stat().st_size == 32768


@pytest.mark.parametrize(
    "bad",
    [
        "person",
        "readonly",
        "expired",
        "future",
        "absolute",
        "traversal",
        "extra",
        "shell",
        "identity",
        "large",
    ],
)
def test_worker_refuses_scope_expansion(worker: Worker, tmp_path: Path, bad: str) -> None:
    path = tmp_path / "note.txt"
    path.write_text("Original")
    call = command(tmp_path, "write_text", path="note.txt", text="Wrong", expectedSha256="")
    if bad == "person":
        call["subject"] = "bo"
    if bad == "readonly":
        call["folder"]["writable"] = False
    if bad == "expired":
        call["expiresAt"] = time.time() - 1
    if bad == "future":
        call["expiresAt"] = time.time() + 600
    if bad == "absolute":
        call["arguments"]["path"] = str(path)
    if bad == "traversal":
        call["arguments"]["path"] = "../escaped.txt"
    if bad == "extra":
        call["arguments"]["followLinks"] = True
    if bad == "shell":
        call["tool"] = "exec"
    if bad == "identity":
        call["folder"]["identity"] = "replacement-folder"
    if bad == "large":
        call["arguments"]["text"] = "x" * 8193
    with pytest.raises(folder_io.FolderError):
        worker.execute(call)
    assert path.read_text() == "Original"


def test_protected_root_and_hard_links_are_refused(worker: Worker, tmp_path: Path) -> None:
    secret = tmp_path / "private"
    secret.mkdir()
    (secret / "node.yaml").write_text("secret")
    worker.protected = [secret]
    inspect = {
        "id": "inspect-root",
        "expiresAt": time.time() + 20,
        "subject": "operator",
        "tool": "inspect",
        "arguments": {"path": str(tmp_path)},
    }
    with pytest.raises(folder_io.FolderError):
        worker.execute(inspect)
    shared = tmp_path / "shared"
    shared.mkdir()
    os.link(secret / "node.yaml", shared / "note.txt")
    with pytest.raises(folder_io.FolderError):
        worker.execute(command(shared))


def test_worker_http_requires_local_credential_and_limits_payload(
    worker: Worker, tmp_path: Path
) -> None:
    (tmp_path / "note.txt").write_text("Hello")
    server = Server(("127.0.0.1", 0), Handler)
    server.worker, server.credential = worker, "local-only-worker-token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False
        ) as client:
            assert client.post("/execute", json=command(tmp_path)).status_code == 403
            client.headers["Authorization"] = "Bearer local-only-worker-token"
            assert client.post("/execute", content=b"x" * 65537).status_code == 413
            result = client.post("/execute", json=command(tmp_path))
            assert result.json()["result"]["text"] == "Hello"
            assert result.headers["cache-control"] == "no-store"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_bundle_is_dependency_free_and_has_no_node_credentials(tmp_path: Path) -> None:
    manager = SimpleNamespace(store=AppStore(tmp_path / "apps.yaml"))
    made = manifest(manager, tmp_path / "private")
    validate_manifest(made)
    assert not made.uses and not made.signIn and not made.ui
    with zipfile.ZipFile(made.source) as wheel:
        assert not any("eugene_plexus_agent/" in name for name in wheel.namelist())
        metadata = next(n for n in wheel.namelist() if n.endswith("/METADATA"))
        assert b"Requires-Dist" not in wheel.read(metadata)
        assert len([n for n in wheel.namelist() if n.endswith(".py")]) == 5
    assert manifest(manager, tmp_path / "private").version == made.version
    assert next(m for m in load_catalogue() if m.id == "workbench").localActions is False


async def test_relay_checks_node_and_folder_then_claims_without_reexecuting(
    app: FastAPI, tmp_path: Path
) -> None:
    relay = NodeFileHelper(app)
    identity = SimpleNamespace(
        enrolled=True, name="desktop", signing_public_key="key", control_url="http://root"
    )
    app.state.node_identity = SimpleNamespace(record=identity)
    call = command(tmp_path)
    call.update(node="desktop", nodeKey="key", enrolledAt="enrollment-1")
    config = {
        "node": "desktop",
        "nodeKey": "key",
        "enrolledAt": "enrollment-1",
        "enabled": True,
        "folders": [copy.deepcopy(call["folder"])],
    }
    relay.validate(call, config)
    for field, value in [
        ("node", "other"),
        ("nodeKey", "old-key"),
        ("enrolledAt", "old-enrollment"),
    ]:
        with pytest.raises(ValueError):
            relay.validate({**call, field: value}, config)
    with pytest.raises(ValueError):
        relay.validate({**call, "folder": {**call["folder"], "path": "/outside"}}, config)
    seen: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("poll"):
            return httpx.Response(
                200, json={"configuration": config, "operation": "job1" if len(seen) == 1 else None}
            )
        if request.url.path.endswith("claim"):
            return httpx.Response(200, json=call)
        raise httpx.ReadTimeout("result acknowledgement lost", request=request)

    relay._root = "http://root"
    relay._root_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    app.state.auth_state = SimpleNamespace(
        trust=SimpleNamespace(agent_token=lambda _: "node-service")
    )
    relay.reconcile = AsyncMock()
    relay.perform = AsyncMock(return_value={"status": "done", "result": {"text": "x"}})
    try:
        with pytest.raises(httpx.ReadTimeout):
            await relay.step()
        await relay.step()
        assert relay.perform.await_count == 1
        identity.enrolled = False
        await relay.step()
        relay.reconcile.assert_awaited_with({"enabled": False})
    finally:
        await relay._root_client.aclose()
        await relay._worker.aclose()


async def test_unsupported_helper_never_installs_and_managed_helper_is_not_an_app(
    app: FastAPI, authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = app.state.apps
    manager.accounts = app_accounts.AccountSupport(
        None, "Isolated accounts are not available here."
    )
    install = AsyncMock()
    monkeypatch.setattr(manager.installer, "start", install)
    relay = NodeFileHelper(app)
    try:
        await relay.reconcile({"enabled": True})
        assert relay.report()["supported"] is False
        install.assert_not_called()
        for suffix in ["install", "start", "stop", "restart"]:
            response = authed_client.post(f"/v1/apps/{HELPER_ID}/{suffix}")
            assert response.status_code == 409
            assert "People" in response.text
    finally:
        await relay._worker.aclose()
