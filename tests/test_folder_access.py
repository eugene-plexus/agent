"""Folder permissions: pure refusals, recorded deltas and operator-only routes."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import folder_access as fa

from .conftest import local_service_token


@pytest.mark.parametrize(
    ("path", "extra", "cause"),
    [
        ("D:\\", {}, "drive_root"),
        ("/", {}, "drive_root"),
        ("C:\\Windows", {}, "system"),
        ("/etc", {}, "system"),
        ("/libexec", {}, "system"),
        ("/lib-vendor", {}, "system"),
        ("D:\\ep", {"eugene": ("D:\\ep",)}, "eugene"),
        ("D:\\Projects", {"eugene": ("D:\\Projects\\ep",)}, "eugene"),
        ("C:\\Users\\person\\shared", {"homes": ("C:\\Users\\person",)}, "home"),
        ("/home/person/shared", {}, "home"),
        ("D:\\junction", {"link": True}, "link"),
        ("\\\\server\\share\\folder", {}, "network"),
        ("D:\\mapped", {"network": True}, "network"),
        ("/mnt/disk", {"mount_root": True}, "drive_root"),
        ("/srv/missing", {"is_dir": False}, "not_a_folder"),
        ("D:\\Projects\\acme", {}, None),
        ("/srv/acme", {}, None),
        ("/srv/project\\notes", {}, None),
    ],
)
def test_refusals(path: str, extra: dict, cause: str | None) -> None:
    assert fa.refusal(fa.Facts(path, **extra)) == cause


class MemoryPermissions:
    account = "test account"

    def __init__(self) -> None:
        self.rights: set[str] = {"manual"}
        self.identity = [1, 2, 3]

    def inspect(self, path: str) -> tuple[fa.Facts, list[int]]:
        return fa.Facts(path), self.identity.copy()

    def grant(self, path: str, mode: str, save):
        delta = {"added": mode not in self.rights, "mode": mode}
        save(delta)
        self.rights.add(mode)
        return delta, 0, False

    def revoke(self, path: str, delta: dict) -> bool:
        if delta["added"]:
            self.rights.remove(delta["mode"])
        return True


def service(tmp_path: Path, backend: MemoryPermissions | None = None) -> fa.Access:
    return fa.Access(tmp_path / "folder-access.json", backend or MemoryPermissions())


def test_add_records_and_remove_only_added(tmp_path: Path) -> None:
    backend = MemoryPermissions()
    access = service(tmp_path, backend)
    grant = access.add("workbench", "/srv/acme", "read", "operator")
    assert grant.state == "ready"
    assert grant.addedBy == "operator"
    assert fa.Access(access.file, backend).listing("workbench")[0].id == grant.id
    assert access.remove("workbench", grant.id).state == "removed"
    assert backend.rights == {"manual"}


def test_preexisting_rights_left_alone(tmp_path: Path) -> None:
    backend = MemoryPermissions()
    backend.rights.add("read")
    access = service(tmp_path, backend)
    grant = access.add("workbench", "/srv/acme", "read", "operator")
    assert grant.preexisting
    access.remove("workbench", grant.id)
    assert backend.rights == {"manual", "read"}


def test_identity_change_is_changed_outside(tmp_path: Path) -> None:
    backend = MemoryPermissions()
    access = service(tmp_path, backend)
    grant = access.add("workbench", "/srv/acme", "read", "operator")
    backend.identity[1] += 1
    assert access.remove("workbench", grant.id).state == "changed_outside"
    assert "read" in backend.rights


def test_replaced_folder_is_not_reported_as_an_old_ready_grant(tmp_path: Path) -> None:
    backend = MemoryPermissions()
    access = service(tmp_path, backend)
    old = access.add("workbench", "/srv/acme", "read", "operator")
    backend.identity[1] += 1
    backend.rights.remove("read")
    fresh = access.add("workbench", "/srv/acme", "read", "operator")
    assert fresh.id != old.id
    assert fresh.state == "ready"
    assert access.listing("workbench")[0].state == "changed_outside"


def test_release_only_own_app(tmp_path: Path) -> None:
    access = service(tmp_path)
    grant = access.add("workbench", "/srv/acme", "read", "operator")
    with pytest.raises(KeyError):
        access.release("other", grant.id)
    assert access.listing("workbench")[0].state == "ready"
    assert access.release("workbench", grant.id)[0].id == grant.id


def test_release_unknown_id_never_reports_empty_success(tmp_path: Path) -> None:
    access = service(tmp_path)
    access.add("workbench", "/srv/acme", "read", "operator")
    with pytest.raises(KeyError):
        access.release("workbench", "unknown")


def test_poll_error_is_transient_and_does_not_write_state(tmp_path: Path, monkeypatch) -> None:
    backend = MemoryPermissions()
    access = service(tmp_path, backend)
    access.add("workbench", "/srv/acme", "read", "operator")
    before = access.file.read_bytes()
    original = backend.inspect

    def asleep(_):
        raise OSError("drive asleep")

    monkeypatch.setattr(backend, "inspect", asleep)
    assert "drive asleep" in access.listing("workbench")[0].message
    assert access.file.read_bytes() == before
    monkeypatch.setattr(backend, "inspect", original)
    assert access.listing("workbench")[0].state == "ready"


def test_missing_folder_still_calls_revoke(tmp_path: Path, monkeypatch) -> None:
    backend = MemoryPermissions()
    access = service(tmp_path, backend)
    grant = access.add("workbench", "/srv/acme", "write", "operator")
    monkeypatch.setattr(backend, "inspect", lambda p: (fa.Facts(p, is_dir=False), []))
    assert access.remove("workbench", grant.id).state == "removed"
    assert backend.rights == {"manual"}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux path resolution")
def test_linux_resolves_parent_traversal_before_refusal(tmp_path: Path) -> None:
    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    facts, _ = backend.inspect("/srv/../etc")
    assert fa.refusal(facts) == "system"


def test_linux_waits_past_900_seconds_for_helper(tmp_path: Path, monkeypatch) -> None:
    import json

    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(backend, "inspect", lambda p: (fa.Facts(p), [1, 2, 3]))
    access = fa.Access(tmp_path / "access.json", backend)
    ticks = iter([0, 0, 901, 902, 903, 904])
    monkeypatch.setattr(fa.time, "perf_counter", lambda: next(ticks, 905))
    polls = []

    def waiting(_):
        polls.append(access.listing("workbench")[0].state)
        if len(polls) == 2:
            req = next(backend.ctl.glob("*.req"))
            req.with_suffix(".res").write_text(json.dumps({"code": 0, "output": '{"added":true}'}))

    monkeypatch.setattr(fa.time, "sleep", waiting)
    assert access.add("workbench", "/srv/acme", "read", "operator").state == "ready"
    assert polls == ["applying", "applying"]


@pytest.mark.parametrize(
    "causes",
    [
        "mount: A mounted descendant is not supported.",
        "socket: A special file is inside this folder.; outside: A file has more than one hard link.",
    ],
)
def test_linux_partial_revoke_is_failed_and_retryable(
    tmp_path: Path, monkeypatch, causes: str
) -> None:
    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(backend, "inspect", lambda p: (fa.Facts(p), [1, 2, 3]))
    outcomes = iter([{"removed": False, "message": causes}, {"removed": True}])

    def helper(verb, *args):
        return {"added": True} if verb == "folder-grant" else next(outcomes)

    monkeypatch.setattr(backend, "_ask", helper)
    access = service(tmp_path, backend)
    grant = access.add("workbench", "/srv/acme", "read", "operator")
    removed = access.remove("workbench", grant.id)
    assert removed.state == "failed"
    assert removed.message == (
        f"Some items were not changed: {causes.rstrip('.')}. Try Remove access again."
    )
    assert fa.Access(access.file, backend).listing("workbench")[0] == removed
    assert access.remove("workbench", grant.id).state == "removed"


def test_applying_is_visible_while_propagation_waits(tmp_path: Path) -> None:
    backend = MemoryPermissions()
    started, finish = threading.Event(), threading.Event()
    original = backend.grant

    def waiting(path, mode, save):
        started.set()
        assert finish.wait(5)
        return original(path, mode, save)

    backend.grant = waiting
    access = service(tmp_path, backend)
    thread = None

    def schedule(work):
        nonlocal thread
        thread = threading.Thread(target=work)
        thread.start()

    try:
        grant = access.add("workbench", "/srv/acme", "read", "operator", schedule=schedule)
        assert started.wait(5)
        assert grant.state == "applying"
        assert access.listing("workbench")[0].state == "applying"
    finally:
        finish.set()
        if thread:
            thread.join(5)
    assert access.listing("workbench")[0].state == "ready"


def test_removing_is_visible_without_blocking_reads(tmp_path: Path) -> None:
    backend = MemoryPermissions()
    access = service(tmp_path, backend)
    grant = access.add("workbench", "/srv/acme", "read", "operator")
    started, finish = threading.Event(), threading.Event()
    original = backend.revoke

    def waiting(path, delta):
        started.set()
        finish.wait(2)
        return original(path, delta)

    backend.revoke = waiting
    worker = threading.Thread(target=access.remove, args=("workbench", grant.id))
    worker.start()
    try:
        assert started.wait(2)
        assert access.listing("workbench")[0].state == "removing"
    finally:
        finish.set()
        worker.join(5)


def test_linux_write_waits_for_observed_restart(tmp_path: Path) -> None:
    backend = MemoryPermissions()
    original = backend.grant

    def pending(path, mode, save):
        delta, count, _ = original(path, mode, save)
        return delta, count, True

    backend.grant = pending
    pid = [100]
    access = fa.Access(tmp_path / "folder-access.json", backend, process=lambda: pid[0])
    assert access.add("workbench", "/srv/acme", "write", "operator").restartNeeded
    assert access.listing("workbench")[0].restartNeeded
    pid[0] = 101
    assert not access.listing("workbench")[0].restartNeeded


def test_linux_request_pins_the_identity_seen_before_the_helper(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(backend, "inspect", lambda _: (fa.Facts("/srv/acme"), [10, 20, 30, 40, 50]))
    seen = []

    def helper(verb, ident, mode):
        seen.append(json.loads((backend.ctl / f"{ident}.identity").read_text("utf-8")))
        return {"added": True}

    monkeypatch.setattr(backend, "_ask", helper)
    backend.grant("/srv/acme", "read", lambda _: None)
    assert seen == [[10, 20, 30, 40, 50]]


def test_preexisting_linux_acl_still_uses_the_helpers_restart_observation(
    tmp_path: Path, monkeypatch
) -> None:
    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(backend, "inspect", lambda _: (fa.Facts("/srv/acme"), [1, 2, 3, 4, 5]))
    monkeypatch.setattr(
        backend, "_ask", lambda *_: {"added": False, "restartNeeded": True, "process": 123}
    )
    delta, _, restart = backend.grant("/srv/acme", "write", lambda _: None)
    assert restart
    assert delta["process"] == 123


def test_restart_completion_uses_a_fresh_service_observation(tmp_path: Path) -> None:
    from eugene_plexus_agent import app_accounts
    from eugene_plexus_agent.routes import directories

    runner = SimpleNamespace(
        kind="systemd", state=lambda _: app_accounts.ServiceState("running", 91)
    )
    supervisor = app_accounts.OwnAccountSupervisor(
        runner, ingress=lambda: "", data_dir=lambda _: tmp_path, key_file=lambda _: tmp_path / "key"
    )
    supervisor.status = lambda _: (None, None, None, 7, None)
    assert directories._process(SimpleNamespace(supervisor=supervisor), "workbench") == 91


@pytest.mark.skipif(sys.platform != "linux", reason="Linux metadata handles")
def test_linux_can_identify_a_folder_before_it_has_read_permission(tmp_path: Path) -> None:
    folder = tmp_path / "closed"
    folder.mkdir()
    folder.chmod(0)
    try:
        backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
        facts, identity = backend.inspect(str(folder))
        assert facts.is_dir
        assert len(identity) == 3
    finally:
        folder.chmod(0o700)


def test_add_needs_operator_session(authed_client: TestClient) -> None:
    client = authed_client
    key = client.post("/v1/auth/client-keys", json={"name": "app:workbench@local"}).json()["token"]
    for token in (key, local_service_token(client.app, "gateway")):
        response = client.post(
            "/v1/apps/workbench/folder-access",
            json={"path": "/srv/acme"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code in (401, 403), response.text


def test_routes_refuse_each_path_with_observed_cause(
    authed_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from eugene_plexus_agent.routes import directories

    backend = MemoryPermissions()
    access = service(tmp_path, backend)
    monkeypatch.setattr(directories, "_access", lambda *_: access)
    for facts, cause in [
        (fa.Facts("D:\\"), "drive_root"),
        (fa.Facts("/etc"), "system"),
        (fa.Facts("/srv/ep", eugene=("/srv/ep",)), "eugene"),
        (fa.Facts("/home/test"), "home"),
        (fa.Facts("/srv/link", link=True), "link"),
        (fa.Facts("//host/share"), "network"),
        (fa.Facts("/srv/missing", is_dir=False), "not_a_folder"),
    ]:
        monkeypatch.setattr(backend, "inspect", lambda _, facts=facts: (facts, [1, 2, 3]))
        response = authed_client.post("/v1/apps/workbench/folder-access", json={"path": facts.path})
        assert response.status_code == 400
        assert response.json()["detail"]["refusal"] == cause
        assert fa.MESSAGES[cause] in response.json()["detail"]["detail"]
    assert backend.rights == {"manual"}


def test_release_authenticates_the_installed_apps_own_key(
    authed_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from eugene_plexus_agent.routes import directories

    client = authed_client
    access = service(tmp_path)
    grant = access.add("workbench", "/srv/acme", "read", "operator")
    minted = client.post("/v1/auth/client-keys", json={"name": "app:workbench@this-node"}).json()
    own_key = minted["token"]
    # Key id comes from the authority's list, never a request's claimed name.
    from eugene_plexus_agent import tokens

    claims = client.app.state.auth_state.trust.verify(own_key, classes=(tokens.TYP_CLIENT,))
    manager = SimpleNamespace(
        node_name=lambda: None,
        store=SimpleNamespace(
            get=lambda _: object(), key=lambda _: SimpleNamespace(key_id=claims.jti)
        ),
    )
    monkeypatch.setattr(directories, "_apps", lambda _: manager)
    monkeypatch.setattr(directories, "_access", lambda *_: access)
    for name in ("app:other@this-node", "app:workbench@other-node", "a normal client"):
        token = client.post("/v1/auth/client-keys", json={"name": name}).json()["token"]
        response = client.post(
            "/v1/apps/workbench/folder-access/release",
            json={"path": "/srv/acme"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
    assert (
        client.post(
            "/v1/apps/workbench/folder-access/release", json={"path": "/srv/acme"}
        ).status_code
        == 401
    )
    response = client.post(
        "/v1/apps/workbench/folder-access/release",
        json={"path": "/srv/acme"},
        headers={"Authorization": f"Bearer {own_key}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["grants"][0]["state"] == "ready"
    response = client.post(
        "/v1/apps/workbench/folder-access/release",
        json={"id": grant.id},
        headers={"Authorization": f"Bearer {own_key}"},
    )
    assert response.json()["grants"][0]["state"] == "removed"
    response = client.post(
        "/v1/apps/workbench/folder-access/release",
        json={"id": "missing"},
        headers={"Authorization": f"Bearer {own_key}"},
    )
    assert response.status_code == 404


@pytest.mark.skipif(os.name != "nt", reason="Windows DACLs")
def test_windows_add_then_remove_touches_one_ace(tmp_path: Path) -> None:
    import win32security as ws

    folder = tmp_path / "shared"
    folder.mkdir()
    initial = ws.GetFileSecurity(
        str(folder), ws.DACL_SECURITY_INFORMATION
    ).GetSecurityDescriptorDacl()
    initial.AddAccessAllowedAceEx(ws.ACL_REVISION, 0, 0x80, ws.ConvertStringSidToSid("S-1-1-0"))
    ws.SetNamedSecurityInfo(
        str(folder), ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION, None, None, initial, None
    )
    # A well-known SID; never create an account or change a real host folder.
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    before = ws.GetFileSecurity(str(folder), ws.DACL_SECURITY_INFORMATION)
    saved: list[dict] = []
    delta, _, _ = backend.grant(str(folder), "read", saved.append)
    assert saved and saved[-1] == delta
    assert backend.revoke(str(folder), delta)
    after = ws.GetFileSecurity(str(folder), ws.DACL_SECURITY_INFORMATION)
    assert ws.ConvertSecurityDescriptorToStringSecurityDescriptor(before, 1, 4) == (
        ws.ConvertSecurityDescriptorToStringSecurityDescriptor(after, 1, 4)
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows DACLs")
def test_windows_refuses_a_hardlink_before_changing_its_target(tmp_path: Path) -> None:
    import win32security as ws

    folder = tmp_path / "shared"
    folder.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("Private")
    os.link(target, folder / "linked.txt")
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    before = ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
        ws.GetFileSecurity(str(target), 4), 1, 4
    )
    with pytest.raises(fa.Refused, match="link"):
        backend.grant(str(folder), "read", lambda _: None)
    assert (
        ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
            ws.GetFileSecurity(str(target), 4), 1, 4
        )
        == before
    )
