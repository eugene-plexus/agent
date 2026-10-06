"""The site host's relay (Job Sites J6, J8): it decides nothing about a tool
call, binds each operation to this machine's enrolment, hands the host what
only the agent may say, and adds local servers only at the machine."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import app_accounts, site_cli, site_host
from eugene_plexus_agent._generated.models import AppOrigin
from eugene_plexus_agent.apps import AppStore, InstalledApp, is_node_files
from eugene_plexus_agent.site_host import HELPER_ID, SiteHostRelay


def identity(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "enrolled": True,
        "name": "desk",
        "signing_public_key": "key",
        "control_url": "http://root",
        "job_site": None,
        "site_owner": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def operation(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": "op-1",
        "expiresAt": time.time() + 20,
        "node": "desk",
        "nodeKey": "key",
        "enrolledAt": "e1",
        "subject": "p-ada",
        "kind": "mcp",
        "server": "files",
        "request": {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        "grants": [
            {
                "folderId": "f1",
                "name": "Shared",
                "path": "/srv/shared",
                "identity": "1:2",
                "writable": False,
            }
        ],
        "installMode": "production",
    }
    value.update(overrides)
    return value


CONFIG = {
    "node": "desk",
    "nodeKey": "key",
    "enrolledAt": "e1",
    "enabled": True,
    "folders": [
        {"id": "f1", "name": "Shared", "path": "/srv/shared", "identity": "1:2", "writable": False}
    ],
}


def test_an_operation_is_bound_to_this_machine_and_in_time(app: FastAPI) -> None:
    relay = SiteHostRelay(app)
    app.state.node_identity = SimpleNamespace(record=identity())
    relay.validate(operation(), CONFIG)
    for field, value in [
        ("node", "other"),
        ("nodeKey", "old-key"),
        ("enrolledAt", "old-enrolment"),
        ("expiresAt", time.time() - 1),
        ("expiresAt", time.time() + 600),
        ("kind", "tool"),
        ("subject", ""),
    ]:
        with pytest.raises(ValueError):
            relay.validate(operation(**{field: value}), CONFIG)


def test_on_a_node_the_grants_must_be_the_folders_the_root_registered(app: FastAPI) -> None:
    relay = SiteHostRelay(app)
    app.state.node_identity = SimpleNamespace(record=identity())
    good = operation()["grants"][0]
    for bad in (
        {**good, "path": "/outside"},
        {**good, "identity": "9:9"},
        {**good, "folderId": "f2"},
        {**good, "writable": True},
    ):
        with pytest.raises(ValueError, match="folder grant"):
            relay.validate(operation(grants=[good, bad]), CONFIG)
    # A job site's own policy decides; its folders are not the root's.
    app.state.node_identity = SimpleNamespace(record=identity(job_site=True, site_owner="p-ada"))
    relay.validate(operation(grants=[{**good, "path": "/outside"}]), CONFIG)


async def test_perform_hands_the_host_the_grants_and_its_answer_back(app: FastAPI) -> None:
    relay = SiteHostRelay(app)
    app.state.node_identity = SimpleNamespace(record=identity())
    sent: list[tuple[str, dict[str, Any]]] = []

    async def ask(method: str, path: str, **kwargs: Any) -> httpx.Response:
        sent.append((path, kwargs["json"]))
        return httpx.Response(200, json={"status": "done", "message": None, "response": {}})

    relay._ask = ask  # type: ignore[method-assign]
    try:
        answer = await relay.perform(operation(), CONFIG)
        assert answer == {"status": "done", "response": {}}
        path, body = sent[0]
        assert path == "/v1/mcp" and body["grants"] == operation()["grants"]
        assert body["server"] == "files" and "nodeKey" not in body

        async def silent(method: str, path: str, **kwargs: Any) -> httpx.Response:
            raise httpx.ReadTimeout("lost")

        relay._ask = silent  # type: ignore[method-assign]
        call = operation(request={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {}})
        assert (await relay.perform(call, CONFIG))["status"] == "uncertain"
        assert (await relay.perform(operation(), CONFIG))["status"] == "failed"
    finally:
        await relay._host.aclose()


async def test_a_claimed_operation_is_never_run_twice(app: FastAPI) -> None:
    relay = SiteHostRelay(app)
    record = identity()
    app.state.node_identity = SimpleNamespace(record=record)
    seen: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("poll"):
            first = len([p for p in seen if p.endswith("poll")]) == 1
            return httpx.Response(
                200, json={"configuration": CONFIG, "operation": "job1" if first else None}
            )
        if request.url.path.endswith("claim"):
            return httpx.Response(200, json=operation())
        raise httpx.ReadTimeout("result acknowledgement lost", request=request)

    relay._root = "http://root"
    relay._root_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    app.state.auth_state = SimpleNamespace(trust=SimpleNamespace(agent_token=lambda _: "service"))
    relay.reconcile = AsyncMock()  # type: ignore[method-assign]
    relay.report = AsyncMock(return_value={"supported": True, "ready": True})  # type: ignore[method-assign]
    relay.perform = AsyncMock(return_value={"status": "done"})  # type: ignore[method-assign]
    try:
        with pytest.raises(httpx.ReadTimeout):
            await relay.step()
        await relay.step()
        assert relay.perform.await_count == 1
        record.enrolled = False
        await relay.step()
        relay.reconcile.assert_awaited_with({"enabled": False})
    finally:
        await relay._root_client.aclose()
        await relay._host.aclose()


def test_a_job_site_keeps_the_owner_it_pinned(app: FastAPI, tmp_path: Path) -> None:
    pinned: list[str] = []
    record = identity(job_site=True, site_owner=None)

    def pin(owner: str) -> str:
        if record.site_owner is None:
            record.site_owner = owner
            pinned.append(owner)
        return str(record.site_owner)

    app.state.node_identity = SimpleNamespace(record=record, pin_site_owner=pin)
    relay = SiteHostRelay(app)
    relay._pin_owner("p-ada")
    relay._pin_owner("p-mallory")
    assert record.site_owner == "p-ada" and pinned == ["p-ada"]


def test_the_host_learns_its_mode_owner_and_servers_from_the_agent(
    app: FastAPI, tmp_path: Path
) -> None:
    relay = SiteHostRelay(app)
    manager: Any = SimpleNamespace(store=AppStore(relay.config_dir / "apps.yaml"))
    app.state.node_identity = SimpleNamespace(record=identity())
    node = relay.environment(manager)
    assert node is not None and node["SITE_HOST_MODE"] == "node"
    assert "SITE_HOST_OWNER" not in node and "SITE_HOST_LOCAL_SERVERS_FILE" not in node
    app.state.node_identity = SimpleNamespace(record=identity(job_site=True))
    assert relay.environment(manager) is None
    app.state.node_identity = SimpleNamespace(record=identity(job_site=True, site_owner="p-ada"))
    # A long list, and braces the launcher would otherwise read as a placeholder.
    servers = [
        {
            "id": f"tool-{n}",
            "name": f"Tool {n}",
            "command": sys.executable,
            "args": ["{config}", "x" * 1500],
            "sha256": "a" * 64,
        }
        for n in range(4)
    ]
    (relay.config_dir / site_host.SERVERS_FILE).write_text(
        yaml.safe_dump({"servers": servers}), encoding="utf-8"
    )
    site = relay.environment(manager)
    assert site is not None and site["SITE_HOST_MODE"] == "site"
    assert site["SITE_HOST_OWNER"] == "p-ada"
    copy = Path(site["SITE_HOST_LOCAL_SERVERS_FILE"])
    data = copy.read_bytes()
    assert hashlib.sha256(data).hexdigest() == site["SITE_HOST_LOCAL_SERVERS_SHA256"]
    assert [s["id"] for s in json.loads(data)] == [s["id"] for s in servers]
    assert copy.parent == manager.store.app_dir(HELPER_ID)
    assert all(
        len(v) <= 2048 and "{" not in v
        for v in site.values()
        if v != site["SITE_HOST_PROTECTED_ROOTS"]
    )
    assert json.loads(site["SITE_HOST_PROTECTED_ROOTS"]) == [str(relay.config_dir)]


async def test_unsupported_never_installs_and_the_host_is_not_an_app(
    app: FastAPI, authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = app.state.apps
    manager.accounts = app_accounts.AccountSupport(
        None, "Isolated accounts are not available here."
    )
    install = AsyncMock()
    monkeypatch.setattr(manager.installer, "start", install)
    relay = SiteHostRelay(app)
    try:
        await relay.reconcile({"enabled": True})
        assert relay._availability()["supported"] is False
        install.assert_not_called()
        for suffix in ["install", "start", "stop", "restart"]:
            response = authed_client.post(f"/v1/apps/{HELPER_ID}/{suffix}")
            assert response.status_code == 409 and "People" in response.text
    finally:
        await relay._host.aclose()
    assert is_node_files(HELPER_ID, site_host.ENTRY)
    assert is_node_files(HELPER_ID, "eugene_plexus_node_helper")
    assert not is_node_files("workbench", site_host.ENTRY)
    record = InstalledApp(
        manifest=site_host.manifest({"SITE_HOST_MODE": "node"}),
        origin=AppOrigin.catalogue,
        port=8300,
        installed_at=datetime.now(UTC),
    )
    manager.store.put(record)
    assert all(v.id != HELPER_ID for v in manager.views())
    assert authed_client.delete(f"/v1/apps/{HELPER_ID}").status_code == 409


# --- the `site` command ------------------------------------------------------------


def program(tmp_path: Path) -> Path:
    path = tmp_path / "server.exe"
    path.write_bytes(b"a program")
    return path


def test_adding_a_server_needs_the_machines_administrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "elevated", lambda: False)
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.add_server(
            tmp_path,
            server_id="notes",
            name="Notes",
            command=str(program(tmp_path)),
            args=[],
            env=[],
            system=False,
        )
    assert not (tmp_path / site_cli.SERVERS_FILE).exists()
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.remove_server(tmp_path, "notes")


def test_a_server_is_written_with_its_programs_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "elevated", lambda: True)
    exe = program(tmp_path)
    said = site_cli.add_server(
        tmp_path,
        server_id="notes",
        name="Notes",
        command=str(exe),
        args=["--root", "{here}"],
        env=["MODE=fast"],
        system=False,
    )
    assert "off until" in said
    saved = yaml.safe_load((tmp_path / site_cli.SERVERS_FILE).read_text(encoding="utf-8"))
    entry = saved["servers"][0]
    assert entry["sha256"] == hashlib.sha256(b"a program").hexdigest()
    assert entry["args"] == ["--root", "{here}"] and entry["env"] == {"MODE": "fast"}
    assert entry["system"] is False and "consentedAt" not in entry
    assert site_host.local_servers(tmp_path)[0]["id"] == "notes"
    for bad, match in (("files-x", "Eugene's own"), ("Notes", "lower-case"), ("notes", "already")):
        with pytest.raises(site_cli.SiteError, match=match):
            site_cli.add_server(
                tmp_path, server_id=bad, name="X", command=str(exe), args=[], env=[], system=False
            )
    with pytest.raises(site_cli.SiteError, match="absolute path"):
        site_cli.add_server(
            tmp_path, server_id="rel", name="X", command="server.exe", args=[], env=[], system=False
        )
    assert "Removed" in site_cli.remove_server(tmp_path, "notes")
    assert site_host.local_servers(tmp_path) == []


def test_a_system_server_records_the_administrators_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "elevated", lambda: True)
    said = site_cli.add_server(
        tmp_path,
        server_id="settings",
        name="Settings",
        command=str(program(tmp_path)),
        args=[],
        env=[],
        system=True,
    )
    assert "consent" in said
    entry = site_host.local_servers(tmp_path)[0]
    assert entry["system"] is True and entry["consentedAt"]


def test_status_and_audit_read_what_the_host_keeps(tmp_path: Path) -> None:
    data = tmp_path / "apps" / HELPER_ID / "data"
    data.mkdir(parents=True)
    (data / "policy.json").write_text(
        json.dumps(
            {
                "version": 1,
                "folders": [
                    {
                        "id": "a" * 32,
                        "name": "Notes",
                        "path": "/srv/notes",
                        "identity": "1:2",
                        "writable": True,
                        "people": [{"subject": "p-bo", "writable": True}],
                    }
                ],
                "access": [],
                "enabled": {},
                "ownerInDevMode": False,
            }
        ),
        encoding="utf-8",
    )
    lines = [
        {
            "at": "2026-10-05T12:00:00Z",
            "subject": "p-bo",
            "kind": "mcp",
            "server": "files",
            "tool": "read_text",
            "decision": "allowed",
        },
        {
            "at": "2026-10-05T12:01:00Z",
            "subject": "operator",
            "kind": "mcp",
            "server": "files",
            "tool": "read_text",
            "decision": "refused",
            "reason": "Dev mode alone opens nothing",
        },
    ]
    (data / "audit.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), "utf-8")
    shown = site_cli.status(tmp_path)
    assert "Notes: /srv/notes" in shown and "p-bo (may change files)" in shown
    log = site_cli.audit(tmp_path, 10).splitlines()
    assert log[0].startswith("2026-10-05T12:01") and "refused" in log[0]
    assert "allowed" in log[1]
