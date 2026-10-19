"""Update all sequencing, restart recovery, uncertain writes and authority."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI

from eugene_plexus_agent import update_all
from eugene_plexus_agent._generated.models import UpdateAllItemState, UpdateAllRunState


class Fleet:
    def __init__(self):
        self.calls = []
        self.manual = set()
        self.offline = set()
        self.node_failures = set()
        self.app_failures = set()
        self.updated = {}
        self.apps = {"root": "old", "worker": "old"}
        self.lost_reply = False
        self.hold = False
        self.app_progress = {}

    async def call(self, target, path, method="GET", body=None):
        self.calls.append((target, path, method))
        name = target.removeprefix("node:")
        if path == "/v1/nodes":
            return {
                "nodes": [{"name": "worker", "role": "node"}, {"name": "root", "role": "control"}]
            }
        if name in self.offline:
            raise update_all.UpdateAllError("Machine offline", 503)
        if path == "/v1/node/update/check":
            return {
                "available": name not in self.updated,
                "newest": {"ref": "new", "components": {"agent": "new-agent"}},
                "apply": {"possible": name not in self.manual, "reason": "Use Docker Force Update"},
            }
        if path == "/v1/node/update" and method == "POST":
            self.updated[name] = {
                "target": body["target"],
                "startedAt": datetime.now(UTC).isoformat(),
                "outcome": "failed" if name in self.node_failures else "succeeded",
                "detail": "Installer failed",
            }
            if self.lost_reply:
                raise update_all.UpdateAllError("Disconnected", 503)
            return self.updated[name]
        if path == "/v1/node":
            return {
                "hostsControl": name == "root",
                "update": {} if self.hold else {"last": self.updated.get(name)},
                "install": {
                    "components": [{"name": "agent", "state": "stamped", "commit": "new-agent"}]
                },
            }
        if path == "/v1/app-catalogue":
            return {
                "apps": [
                    {
                        "origin": "catalogue",
                        "installedVersion": self.apps[name],
                        "manifest": {"id": "workbench", "version": "new-app"},
                    },
                    {
                        "origin": "custom",
                        "installedVersion": "old",
                        "manifest": {"id": "custom", "version": "new"},
                    },
                    {"origin": "catalogue", "manifest": {"id": "absent", "version": "new"}},
                ]
            }
        if path == "/v1/apps/workbench/install":
            if method == "POST":
                self.apps[name] = "new-app"
                self.app_progress[name] = {
                    "state": "failed" if name in self.app_failures else "done",
                    "error": "App restart failed",
                }
                return self.app_progress[name]
            if name not in self.app_progress:
                raise update_all.UpdateAllError("No install", 404)
            return self.app_progress[name]
        if path == "/v1/apps/workbench":
            return {"version": self.apps[name], "status": "running"}
        raise AssertionError((target, path, method))


@pytest.fixture
def fleet(monkeypatch):
    monkeypatch.setattr(update_all, "POLL_SECONDS", 0.001)
    return Fleet()


def manager(tmp_path, monkeypatch, fleet):
    instance = update_all.UpdateAll(FastAPI(), tmp_path / "update-all.json")
    monkeypatch.setattr(instance, "_call", fleet.call)
    monkeypatch.setattr(instance, "_authorize", lambda: None)
    return instance


async def done(instance):
    # A hang guard, not a speed check: the Windows runner is slow at the state writes.
    await asyncio.wait_for(asyncio.shield(instance._task), timeout=10)
    return instance.run


async def test_root_first_then_apps_and_workers(tmp_path, monkeypatch, fleet):
    m = manager(tmp_path, monkeypatch, fleet)
    try:
        run = await m.start("test-session")
        assert await m.start("another-session") is run
        result = await done(m)
        assert result.state.value == "complete"
        writes = [c for c in fleet.calls if c[2] == "POST" and not c[1].endswith("check")]
        assert writes == [
            ("node:root", "/v1/node/update", "POST"),
            ("node:root", "/v1/apps/workbench/install", "POST"),
            ("node:worker", "/v1/node/update", "POST"),
            ("node:worker", "/v1/apps/workbench/install", "POST"),
        ]
        assert len(result.items) == 4  # No absent or custom app installed.
        assert json.loads(m.path.read_text())["token"] == ""
        assert "test-session" not in result.model_dump_json()
    finally:
        await m.close()


@pytest.mark.parametrize("failure", ["manual", "node_failures"])
async def test_root_failure_blocks_workers(tmp_path, monkeypatch, fleet, failure):
    getattr(fleet, failure).add("root")
    m = manager(tmp_path, monkeypatch, fleet)
    try:
        await m.start("test-session")
        result = await done(m)
        assert result.state.value == "attention"
        assert result.items[1].state.value == "skipped"
        assert not any(c[0] == "node:worker" and c[2] == "POST" for c in fleet.calls)
    finally:
        await m.close()


async def test_offline_worker_does_not_hide_root_workbench_success(tmp_path, monkeypatch, fleet):
    fleet.offline.add("worker")
    m = manager(tmp_path, monkeypatch, fleet)
    try:
        await m.start("test-session")
        result = await done(m)
        assert result.state.value == "attention"
        assert fleet.apps["root"] == "new-app"
        assert next(i for i in result.items if i.node == "worker").state.value == "failed"
    finally:
        await m.close()


async def test_lost_write_reply_is_observed_without_replaying(tmp_path, monkeypatch, fleet):
    fleet.lost_reply = True
    m = manager(tmp_path, monkeypatch, fleet)
    try:
        await m.start("test-session")
        assert (await done(m)).state.value == "complete"
        assert len([c for c in fleet.calls if c[1] == "/v1/node/update"]) == 2
    finally:
        await m.close()


async def test_restart_recovers_pending_job_and_does_not_repeat_update(
    tmp_path, monkeypatch, fleet
):
    fleet.hold = True
    m = manager(tmp_path, monkeypatch, fleet)
    await m.start("test-session")
    for _ in range(100):
        if fleet.updated:
            break
        await asyncio.sleep(0.002)
    assert m.run.items[0].state.value == "updating"
    await m.close()
    assert json.loads(m.path.read_text())["token"] == "test-session"
    recovered = manager(tmp_path, monkeypatch, fleet)
    try:
        fleet.hold = False
        recovered.resume()
        assert (await done(recovered)).state.value == "complete"
        assert fleet.calls.count(("node:root", "/v1/node/update", "POST")) == 1
    finally:
        await recovered.close()


async def test_app_restart_failure_is_not_hidden_by_updated_version(tmp_path, monkeypatch, fleet):
    fleet.app_failures.add("root")
    m = manager(tmp_path, monkeypatch, fleet)
    try:
        await m.start("test-session")
        result = await done(m)
        assert result.state.value == "attention"
        assert next(i for i in result.items if i.app == "workbench").state.value == "failed"
        assert fleet.apps["worker"] == "new-app"
    finally:
        await m.close()


async def test_revocation_stops_remaining_work_and_erases_session(tmp_path, monkeypatch, fleet):
    m = manager(tmp_path, monkeypatch, fleet)
    try:
        await m.start("test-session")

        def revoked():
            raise update_all.UpdateAllError("Signed out", 401)

        monkeypatch.setattr(m, "_authorize", revoked)
        assert (await done(m)).state.value == "attention"
        assert not any(c[2] == "POST" for c in fleet.calls)
        assert json.loads(m.path.read_text())["token"] == ""
    finally:
        await m.close()


async def test_stale_success_cannot_finish_new_job(tmp_path, monkeypatch, fleet):
    m = manager(tmp_path, monkeypatch, fleet)
    try:
        await m.start("test-session")
        await done(m)
        item = m.run.items[0]
        item.state = UpdateAllItemState.updating
        item.startedAt = datetime.now(UTC) - timedelta(minutes=46)
        m.run.state = UpdateAllRunState.running
        with pytest.raises(update_all.UpdateAllError, match="45 minutes"):
            await m._wait_node(item)
    finally:
        await m.close()


def test_update_all_routes_require_operator(client):
    assert client.get("/v1/updates/all").status_code == 401
    assert client.post("/v1/updates/all", json={}).status_code == 401


def test_status_contains_no_credential_and_get_does_not_start_work(authed_client):
    response = authed_client.get("/v1/updates/all")
    assert response.status_code == 200
    assert response.json() == {}
    assert authed_client.app.state.update_all._task is None


def test_authority_revalidates_real_session(authed_client):
    m = authed_client.app.state.update_all
    token = authed_client.headers["authorization"].partition(" ")[2]
    m._token = token
    m._authorize()
    m.run = None
    m._token = "not-a-session"
    with pytest.raises(update_all.UpdateAllError, match="session expired"):
        m._authorize()


async def test_offline_root_refuses_before_any_update(tmp_path, monkeypatch, fleet):
    fleet.offline.add("root")
    m = manager(tmp_path, monkeypatch, fleet)
    try:
        with pytest.raises(update_all.UpdateAllError, match="available control root"):
            await m.start("test-session")
        assert not any(c[2] == "POST" for c in fleet.calls)
        assert not m.path.exists()
    finally:
        await m.close()


async def test_missing_installed_component_is_a_failed_update(tmp_path, monkeypatch, fleet):
    m = manager(tmp_path, monkeypatch, fleet)
    original = fleet.call

    async def missing(target, path, method="GET", body=None):
        result = await original(target, path, method, body)
        if path == "/v1/node" and target == "node:root":
            result["install"]["components"] = []
        return result

    monkeypatch.setattr(m, "_call", missing)
    try:
        await m.start("test-session")
        run = await done(m)
        assert run.state.value == "attention"
        assert run.items[0].state.value == "failed"
        assert "do not match" in run.items[0].detail
    finally:
        await m.close()


async def test_old_success_is_ignored_until_the_new_attempt_reports(tmp_path, monkeypatch, fleet):
    m = manager(tmp_path, monkeypatch, fleet)
    try:
        await m.start("test-session")
        await done(m)
        item = m.run.items[0]
        item.state = UpdateAllItemState.updating
        item.startedAt = datetime.now(UTC)
        fleet.updated["root"]["startedAt"] = (item.startedAt - timedelta(hours=1)).isoformat()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(m._wait_node(item), 0.03)
        assert item.state.value == "updating"
        fleet.updated["root"]["startedAt"] = item.startedAt.isoformat()
        await m._wait_node(item)
        assert item.state.value == "updated"
    finally:
        await m.close()


async def test_bad_journal_does_not_start_another_update(tmp_path):
    path = tmp_path / "update-all.json"
    path.write_text("broken")
    m = update_all.UpdateAll(FastAPI(), path)
    try:
        m.resume()
        assert m._task is None
        with pytest.raises(update_all.UpdateAllError, match="could not be read"):
            await m.start("test-session")
        assert path.read_text() == "broken"
    finally:
        await m.close()


def test_actual_signout_revokes_the_job_authority(authed_client):
    m = authed_client.app.state.update_all
    m._token = authed_client.headers["authorization"].partition(" ")[2]
    m._authorize()
    assert authed_client.delete("/v1/auth/sessions/current").status_code == 204
    with pytest.raises(update_all.UpdateAllError, match="signed out"):
        m._authorize()
