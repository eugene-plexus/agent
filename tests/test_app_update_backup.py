"""An update never starts against unprotected data; rollback restores the old data."""

import asyncio
import json
import os
import shutil
import sys

import pytest

from eugene_plexus_agent import apps
from eugene_plexus_agent._generated.models import AppOrigin
from tests.test_apps import _commit, _manager, _manifest, _record


def test_update_refuses_systemd_private_data_without_a_helper(tmp_path):
    from eugene_plexus_agent.app_accounts import AccountSupport

    manager, _, new = installed(tmp_path)
    manager.accounts = AccountSupport("systemd")
    with pytest.raises(apps.AppInstallError, match="private"):
        asyncio.run(manager.installed_callback(new, AppOrigin.custom))
    assert manager.store.get("tiny").version == "v1"


def test_backup_refuses_directory_links(tmp_path):
    manager, data, new = installed(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(outside), str(data / "vector_db"))
    else:
        os.symlink(outside, data / "vector_db", target_is_directory=True)
    try:
        with pytest.raises(apps.AppInstallError, match=r"link|reparse"):
            asyncio.run(manager.installed_callback(new, AppOrigin.custom))
        assert (outside / "keep").read_text() == "keep"
    finally:
        # Remove the link only, so pytest's own cleanup never descends into it.
        if sys.platform == "win32":
            os.rmdir(data / "vector_db")
        else:
            (data / "vector_db").unlink()


def test_failed_restore_stays_stopped_with_backup_for_retry(tmp_path, monkeypatch):
    manager, _data, new = installed(tmp_path)
    asyncio.run(manager.installed_callback(new, AppOrigin.custom))

    def fail(*_args, **_kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr(shutil, "copy2", fail)
    with pytest.raises(apps.AppInstallError, match="disk write failed"):
        asyncio.run(manager.rollback(manager.store.get("tiny")))
    record = manager.store.get("tiny")
    assert not record.enabled
    assert manager.view(record).updateBackup.version == "v1"


def test_reset_route_records_the_request(app, authed_client):
    from tests.test_apps import FakeAppSupervisor

    manager = app.state.apps
    manager.supervisor = FakeAppSupervisor()
    manager.store.put(_record(_manifest(resetOnConnectionChange="RESET_CONFIG_ON_START")))
    reply = authed_client.post("/v1/apps/tiny/restart", json={"resetSettings": True})
    assert reply.status_code == 200, reply.text
    assert (manager.store.data_dir("tiny") / "connection.sha256").read_text() == "reset-requested"


def test_reset_route_refuses_an_app_without_settings(app, authed_client):
    manager = app.state.apps
    manager.store.put(_record())
    reply = authed_client.post("/v1/apps/tiny/restart", json={"resetSettings": True})
    assert reply.status_code == 409, reply.text
    assert "This app has no settings for Eugene to reset." in reply.text


def installed(tmp_path):
    manager = _manager(tmp_path)
    old = _record(_manifest(updateBackup=["webui.db", "vector_db"]))
    manager.store.put(old)
    _commit(manager.store, "tiny", "v1", "2026-10-01T00:00:00Z")
    data = manager.store.data_dir("tiny")
    data.mkdir(parents=True)
    (data / "webui.db").write_text("old chats")
    (data / "given.json").write_text(
        json.dumps({"givenAt": "2026-10-01T00:00:00Z", "webSearchEnabled": False})
    )
    new = _manifest(version="v2", updateBackup=["webui.db", "vector_db"])
    _commit(manager.store, "tiny", "v2", "2026-10-21T00:00:00Z")
    return manager, data, new


def test_update_backs_up_before_first_start(tmp_path):
    manager, data, new = installed(tmp_path)

    async def start(record):
        backup = manager.store.app_dir("tiny") / "backups" / "v1"
        assert (backup / "webui.db").read_text() == "old chats"
        (data / "webui.db").write_text("migrated")

    manager.start = start
    asyncio.run(manager.installed_callback(new, AppOrigin.custom))
    assert manager.view(manager.store.get("tiny")).updateBackup.version == "v1"


def test_rollback_restores_backup(tmp_path):
    manager, data, new = installed(tmp_path)
    asyncio.run(manager.installed_callback(new, AppOrigin.custom))
    (data / "webui.db").write_text("new chats")
    (data / "vector_db").mkdir()
    (data / "vector_db" / "new").write_text("new vectors")
    (data / "given.json").write_text("{}")
    asyncio.run(manager.rollback(manager.store.get("tiny")))
    record = manager.store.get("tiny")
    assert record.version == "v1"
    assert (data / "webui.db").read_text() == "old chats"
    assert not (data / "vector_db").exists()
    assert json.loads((data / "given.json").read_text())["webSearchEnabled"] is False
    assert manager.view(record).updateBackup is None


def test_rollback_removes_sqlite_wal_created_after_the_backup(tmp_path):
    manager, data, new = installed(tmp_path)
    asyncio.run(manager.installed_callback(new, AppOrigin.custom))
    (data / "webui.db-wal").write_text("new transactions")
    (data / "webui.db-shm").write_text("new shared memory")
    asyncio.run(manager.rollback(manager.store.get("tiny")))
    assert not (data / "webui.db-wal").exists()
    assert not (data / "webui.db-shm").exists()


def test_insufficient_space_keeps_current_version(tmp_path, monkeypatch):
    manager, data, new = installed(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100, 100, 0))
    with pytest.raises(apps.AppInstallError, match="backup"):
        asyncio.run(manager.installed_callback(new, AppOrigin.custom))
    assert manager.store.get("tiny").version == "v1"
    assert (data / "webui.db").read_text() == "old chats"
    assert not manager.supervisor.started


@pytest.mark.parametrize(
    "path", ["../outside", "/absolute", "C:drive", "a/../../outside", "a\\..\\outside"]
)
def test_backup_paths_must_stay_inside_data(path):
    with pytest.raises(ValueError, match="updateBackup"):
        apps.validate_manifest(_manifest(updateBackup=[path]))
