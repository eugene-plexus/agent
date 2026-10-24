"""Folder permissions: pure refusals, recorded deltas and operator-only routes."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

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
        ("C:\\Windows", {"homes": ("C:\\Windows\\System32\\config\\systemprofile",)}, "system"),
        (
            "C:\\Windows\\System32",
            {
                "eugene": ("C:\\Windows\\ep",),
                "homes": ("C:\\Windows\\System32\\config\\systemprofile",),
            },
            "system",
        ),
        ("/etc", {}, "system"),
        ("/libexec", {}, "system"),
        ("/lib-vendor", {}, "system"),
        ("D:\\ep", {"eugene": ("D:\\ep",)}, "eugene"),
        ("D:\\Projects", {"eugene": ("D:\\Projects\\ep",)}, "eugene"),
        ("C:\\Users\\person\\shared", {"homes": ("C:\\Users\\person",)}, "home"),
        ("C:\\Users\\person", {"eugene": ("C:\\Users\\person\\ep",)}, "home"),
        ("C:\\Users\\person\\ep", {"eugene": ("C:\\Users\\person\\ep",)}, "eugene"),
        ("C:\\Users\\person\\ep\\apps", {"eugene": ("C:\\Users\\person\\ep",)}, "eugene"),
        ("C:\\Users\\person\\AppData\\Local\\Temp\\shared", {}, "home"),
        ("D:\\People\\person\\shared", {"homes": ("D:\\People\\person",)}, "home"),
        ("/home/person/shared", {}, "home"),
        ("/home/runner", {"eugene": ("/home/runner/ep",)}, "home"),
        ("/home", {"eugene": ("/home/runner/ep",)}, "home"),
        ("/home/runner/shared", {"eugene": ("/home/runner/ep",)}, "home"),
        ("/home/runner/ep", {"eugene": ("/home/runner/ep",)}, "eugene"),
        ("/home/runner/ep/apps", {"eugene": ("/home/runner/ep",)}, "eugene"),
        ("/opt", {"eugene": ("/opt/ep",), "homes": ("/opt/service",)}, "system"),
        ("/opt/ep", {"eugene": ("/opt/ep",), "homes": ("/opt/service",)}, "eugene"),
        ("/var/lib/ep/apps", {"eugene": ("/var/lib/ep",)}, "eugene"),
        (
            "/srv/people",
            {"eugene": ("/srv/people/person/ep",), "homes": ("/srv/people/person",)},
            "home",
        ),
        ("D:\\junction", {"link": True}, "link"),
        ("\\\\server\\share\\folder", {}, "network"),
        ("D:\\mapped", {"network": True}, "network"),
        ("/mnt/disk", {"mount_root": True}, "drive_root"),
        ("/srv/missing", {"is_dir": False}, "not_a_folder"),
        # A folder another rule refuses names that rule, not its parent: on
        # Linux, Eugene's prefix and a person's home belong to that account,
        # so every folder inside them also has an "unsafe" parent.
        ("/srv/ep/apps", {"eugene": ("/srv/ep",), "unsafe_parent": "/srv/ep"}, "eugene"),
        ("/home/person/projects", {"unsafe_parent": "/home/person"}, "home"),
        ("/tmp/shared", {"unsafe_parent": "/tmp"}, "system"),
        ("/srv/shared/acme", {"unsafe_parent": "/srv/shared"}, "unsafe_parent"),
        ("D:\\Projects\\acme", {}, None),
        ("/srv/acme", {}, None),
        ("/srv/project\\notes", {}, None),
    ],
)
def test_refusals(path: str, extra: dict, cause: str | None) -> None:
    assert fa.refusal(fa.Facts(path, **extra)) == cause


@pytest.mark.parametrize(
    ("path", "extra", "words"),
    [
        (
            "/srv/ep/apps",
            {"eugene": ("/srv/ep",), "unsafe_parent": "/srv/ep"},
            "This is Eugene's own folder. /srv/ep/apps",
        ),
        (
            "/home/person/projects",
            {"unsafe_parent": "/home/person"},
            "This is inside a person's home. Use a Job Site to work in your own folders. "
            "/home/person/projects",
        ),
        (
            "/srv/shared/acme",
            {"unsafe_parent": "/srv/shared"},
            "/srv/shared can be changed by other accounts.",
        ),
    ],
)
def test_refusal_message_names_the_refused_folder(
    tmp_path: Path, path: str, extra: dict, words: str
) -> None:
    backend = MemoryPermissions()
    backend.inspect = lambda p: (fa.Facts(p, **extra), [])  # type: ignore[method-assign]
    access = service(tmp_path, backend)
    with pytest.raises(fa.Refused) as refused:
        access.add("workbench", path, "read", "operator")
    assert str(refused.value) == words
    assert not access.file.exists()


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


class ExclusivePermissions(MemoryPermissions):
    """A backend whose grants cannot share an item with another grant (Windows)."""

    overlap = False


def test_overlapping_grants_are_refused_when_the_backend_cannot_share(tmp_path: Path) -> None:
    access = service(tmp_path, ExclusivePermissions())
    parent = access.add("workbench", "D:\\Projects", "read", "operator")
    with pytest.raises(ValueError) as inside:
        access.add("workbench", "D:\\Projects\\acme", "write", "operator")
    assert str(inside.value) == (
        "D:\\Projects\\acme is inside D:\\Projects, which Workbench can already open. "
        "Remove access to D:\\Projects first, then give access again."
    )
    with pytest.raises(ValueError, match=r"^D:\\Projects holds D:\\Projects\\acme"):
        fresh = service(tmp_path / "second", ExclusivePermissions())
        fresh.add("workbench", "D:\\Projects\\acme", "read", "operator")
        fresh.add("workbench", "D:\\Projects", "read", "operator")
    # Another app's grant, a sibling and a removed grant do not overlap.
    assert access.add("other", "D:\\Projects\\acme", "write", "operator").state == "ready"
    assert access.add("workbench", "D:\\Projects2", "read", "operator").state == "ready"
    assert access.remove("workbench", parent.id).state == "removed"
    assert access.add("workbench", "D:\\Projects\\acme", "write", "operator").state == "ready"
    # Linux records each grant's own entries and restores the others on removal.
    assert fa.WindowsPermissions.overlap is False
    assert fa.LinuxPermissions.overlap is True


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


@pytest.mark.parametrize("operation", ["add", "lookup", "listing", "remove"])
def test_unreadable_folder_names_cause_and_next_action(
    tmp_path: Path, monkeypatch, operation
) -> None:
    backend = MemoryPermissions()
    access = service(tmp_path, backend)
    grant = access.add("workbench", "/srv/acme", "read", "operator")
    before = access.file.read_bytes()

    def denied(_):
        raise PermissionError(13, "Permission denied", "/srv/acme")

    monkeypatch.setattr(backend, "inspect", denied)
    if operation in ("add", "lookup"):
        with pytest.raises(ValueError, match="Eugene's account cannot open /srv/acme") as exc:
            if operation == "add":
                access.add("workbench", "/srv/acme", "read", "operator")
            else:
                access.lookup("workbench", "/srv/acme")
        message = str(exc.value)
        assert access.file.read_bytes() == before
    else:
        result = (
            access.listing("workbench")[0]
            if operation == "listing"
            else access.remove("workbench", grant.id)
        )
        assert result.state == ("changed_outside" if operation == "listing" else "failed")
        message = result.message
        assert "Eugene's account cannot open /srv/acme" in message
    assert "Check this folder's permissions" in message
    assert "read" in backend.rights


def test_missing_folder_still_calls_revoke(tmp_path: Path, monkeypatch) -> None:
    backend = MemoryPermissions()
    access = service(tmp_path, backend)
    grant = access.add("workbench", "/srv/acme", "write", "operator")
    monkeypatch.setattr(backend, "inspect", lambda p: (fa.Facts(p, is_dir=False), []))
    assert access.remove("workbench", grant.id).state == "removed"
    assert backend.rights == {"manual"}


@pytest.mark.parametrize("probe", ["stat", "lstat", "is_symlink", "exists", "is_dir", "resolve"])
def test_linux_unreadable_folder_uses_root_inspection(tmp_path: Path, monkeypatch, probe) -> None:
    import json

    path = "/srv/overlap/child"
    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(fa, "sys", SimpleNamespace(platform="linux"))
    original = getattr(Path, probe)

    def unreadable(value, *args, **kwargs):
        if value == Path(path):
            raise PermissionError(13, "Permission denied", path)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(Path, probe, unreadable)
    inspected = []

    def helper(verb, ident, mode="read"):
        if verb == "folder-inspect":
            assert (backend.ctl / f"{ident}.path").read_text() == path
            inspected.append(ident)
            return {"facts": {"path": path}, "identity": [10, 20, 30]}
        if verb == "folder-grant":
            assert json.loads((backend.ctl / f"{ident}.identity").read_text()) == [10, 20, 30]
            return {"added": True}
        assert verb == "folder-revoke"
        return {"removed": True}

    monkeypatch.setattr(backend, "_ask", helper)
    access = service(tmp_path, backend)
    grant = access.add("workbench", path, "write", "operator")
    assert grant.state == "ready"
    assert access.listing("workbench")[0].state == "ready"
    assert access.lookup("workbench", path)[0].id == grant.id
    assert access.remove("workbench", grant.id).state == "removed"
    assert len(inspected) == 4  # add, grant, listing and lookup; revoke uses its ledger
    assert not list(backend.ctl.glob("*.path"))


@pytest.mark.parametrize(
    ("extra", "cause"),
    [
        ({"link": True}, "link"),
        ({"is_dir": False}, "not_a_folder"),
        ({"mount_root": True}, "drive_root"),
        ({"network": True}, "network"),
        ({"system": ["/srv/refused"]}, "system"),
        ({"eugene": ["/srv/refused"]}, "eugene"),
        ({"homes": ["/srv/refused"]}, "home"),
    ],
)
def test_linux_preserves_root_inspection_refusals(
    tmp_path: Path, monkeypatch, extra, cause
) -> None:
    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(fa, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        backend,
        "_ask",
        lambda *_: {"facts": {"path": "/srv/refused", **extra}, "identity": []},
    )
    facts, identity = backend.inspect("/srv/refused")
    assert fa.refusal(facts) == cause
    assert identity == []
    assert not list(backend.ctl.glob("*.path"))


def test_linux_inspection_failure_cleans_request(tmp_path: Path, monkeypatch) -> None:
    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(fa, "sys", SimpleNamespace(platform="linux"))

    def unavailable(*_):
        raise OSError("The folder drive is unavailable")

    monkeypatch.setattr(backend, "_ask", unavailable)
    with pytest.raises(OSError, match="folder drive is unavailable"):
        backend.inspect("/srv/project")
    assert not list(backend.ctl.glob("*.path"))


def test_linux_inspection_names_a_helper_that_never_answers(tmp_path: Path, monkeypatch) -> None:
    """An inspection runs under the module lock; a silent helper must not hold it forever."""
    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(fa, "sys", SimpleNamespace(platform="linux"))
    clock = {"now": 0.0, "sleeps": 0}

    def sleep(seconds: float) -> None:
        clock["sleeps"] += 1
        clock["now"] += seconds
        if clock["sleeps"] > 10_000:  # 2000 simulated seconds: the wait never ends
            raise AssertionError("The agent waited for an inspection that never came")

    monkeypatch.setattr(fa.time, "sleep", sleep)
    monkeypatch.setattr(fa.time, "perf_counter", lambda: clock["now"])
    access = fa.Access(tmp_path / "access.json", backend)
    with pytest.raises(fa.InspectionFailed, match="helper did not answer in 30 seconds"):
        access.add("workbench", "/srv/acme", "read", "operator")
    assert clock["now"] < 31
    # The request is withdrawn, so a helper that starts later does not act on it.
    assert not list(backend.ctl.iterdir())
    assert not access.file.exists()


@pytest.mark.parametrize(
    ("extra", "cause"),
    [
        ({"homes": ["/srv/shared/person"]}, "home"),
        ({"system": ["/srv/shared"]}, "system"),
        ({"eugene": ["/srv/shared"]}, "eugene"),
        ({"mount_root": True}, "drive_root"),
        ({"unsafe_parent": "/srv"}, "unsafe_parent"),
    ],
)
@pytest.mark.parametrize("identity_available", [False, True])
def test_linux_removal_uses_saved_grant_when_new_grants_are_refused(
    tmp_path: Path, monkeypatch, extra, cause, identity_available
) -> None:
    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(fa, "sys", SimpleNamespace(platform="linux"))
    policy = {}
    calls = []
    granted = []

    def helper(verb, ident, mode="read"):
        calls.append((verb, ident))
        if verb == "folder-inspect":
            # An older broker (or an unsafe parent) cannot return an identity.
            return {
                "facts": {"path": "/srv/shared", **policy},
                "identity": (
                    []
                    if policy and (not identity_available or cause == "unsafe_parent")
                    else [10, 20, 30]
                ),
            }
        if verb == "folder-grant":
            granted.append(ident)
            return {"added": True}
        assert verb == "folder-revoke"
        assert ident == granted[0]  # The broker selects its root-owned ledger.
        return {"removed": True}

    monkeypatch.setattr(backend, "_ask", helper)
    access = fa.Access(tmp_path / "access.json", backend)
    grant = access.add("workbench", "/srv/shared", "read", "operator")
    assert grant.state == "ready"
    policy.update(extra)
    with pytest.raises(fa.Refused) as exc:
        access.add("workbench", "/srv/shared", "read", "operator")
    assert exc.value.cause == cause
    if cause == "unsafe_parent":
        assert "/srv can be changed by other accounts" in str(exc.value)
    elif identity_available:
        assert access.listing("workbench")[0].state == "ready"
    calls.clear()
    assert access.remove("workbench", grant.id).state == "removed"
    assert calls == [("folder-revoke", granted[0])]


def test_linux_unsafe_parent_listing_can_recover_without_reporting_replacement(
    tmp_path: Path, monkeypatch
) -> None:
    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(fa, "sys", SimpleNamespace(platform="linux"))
    unsafe = False

    def helper(verb, *args):
        if verb == "folder-inspect":
            return {
                "facts": {"path": "/srv/shared", **({"unsafe_parent": "/srv"} if unsafe else {})},
                "identity": [] if unsafe else [10, 20, 30],
            }
        return {"added": True}

    monkeypatch.setattr(backend, "_ask", helper)
    access = fa.Access(tmp_path / "access.json", backend)
    access.add("workbench", "/srv/shared", "read", "operator")
    before = access.file.read_bytes()
    unsafe = True
    row = access.listing("workbench")[0]
    assert row.state == "changed_outside"
    assert "/srv can be changed by other accounts" in row.message
    assert "replaced" not in row.message
    assert access.file.read_bytes() == before
    unsafe = False
    assert access.listing("workbench")[0].state == "ready"


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


@pytest.mark.parametrize("removed", [False, True])
def test_linux_revoke_updates_preexisting_parent_after_reapply(
    tmp_path: Path, monkeypatch, removed: bool
) -> None:
    backend = fa.LinuxPermissions("workbench", tmp_path / "apps", eugene=())
    monkeypatch.setattr(backend, "inspect", lambda p: (fa.Facts(p), [1, 2, 3]))
    record_ids = {}

    def helper(verb, ident, *args):
        if verb == "folder-grant":
            path = (backend.ctl / f"{ident}.path").read_text("utf-8")
            record_ids[path] = ident
            return {"added": path == "/srv/acme/sub"}
        return {
            "removed": removed,
            "message": "" if removed else "socket: A special file is inside this folder.",
            "reapplied": [record_ids["/srv/acme"]],
        }

    monkeypatch.setattr(backend, "_ask", helper)
    access = fa.Access(tmp_path / "access.json", backend)
    parent = access.add("workbench", "/srv/acme", "read", "operator")
    untouched = access.add("workbench", "/srv/other", "read", "operator")
    child = access.add("workbench", "/srv/acme/sub", "write", "operator")
    assert parent.preexisting
    assert untouched.preexisting
    assert not child.preexisting
    assert record_ids[parent.path] != parent.id
    assert access.remove("workbench", child.id).state == ("removed" if removed else "failed")
    refreshed = fa.Access(access.file, backend)
    rows = {row.id: row for row in refreshed.listing("workbench")}
    assert rows[parent.id].preexisting is False
    assert refreshed._read()[parent.id]["delta"]["added"] is True
    assert rows[untouched.id].preexisting is True


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


class AskingPermissions(MemoryPermissions):
    """Removal finds one entry Eugene did not add, until the owner decides it."""

    FOUND: ClassVar[dict] = {
        "kind": "entry",
        "path": "/srv/acme/sub",
        "entry": "group eapp-0123456789ab-files: rwx (set by hand)",
        "fingerprint": {"kind": "entry", "names": ["sub"], "key": "group:9", "value": "rwx"},
    }

    def __init__(self) -> None:
        super().__init__()
        self.present = True
        self.decisions: list[tuple[dict, str]] = []

    def revoke(self, path: str, delta: dict) -> bool:
        if self.present and self.FOUND["fingerprint"] not in delta.get("accepted", []):
            raise fa.UnknownAccess([dict(self.FOUND)])
        return super().revoke(path, delta)

    def decide(self, path: str, delta: dict, fingerprint: dict, action: str) -> None:
        self.decisions.append((fingerprint, action))
        if action == "remove":
            self.present = False
        else:
            delta.setdefault("accepted", []).append(fingerprint)


def test_removal_asks_the_owner_about_an_entry_eugene_did_not_add(tmp_path: Path) -> None:
    """R23-UNKNOWN (Troy, 2026-10-24): prompt; Keep leaves it and says so."""
    backend = AskingPermissions()
    access = service(tmp_path, backend)
    grant = access.add("workbench", "/srv/acme", "write", "operator")
    asked = access.remove("workbench", grant.id)
    assert asked.state == "incomplete"
    assert asked.unknown is not None and len(asked.unknown) == 1
    entry = asked.unknown[0]
    assert (entry.path, entry.kind, entry.decision) == ("/srv/acme/sub", "entry", "pending")
    assert "/srv/acme/sub" in asked.message
    # The private fingerprint never reaches the API.
    assert "fingerprint" not in asked.model_dump_json()
    kept = access.decide("workbench", grant.id, entry.id, "keep")
    assert backend.decisions == [(AskingPermissions.FOUND["fingerprint"], "keep")]
    assert kept.state == "removed"
    assert kept.unknown is not None and [u.decision for u in kept.unknown] == ["kept"]
    assert kept.message == (
        "Eugene's additions were removed. Workbench can still open /srv/acme/sub "
        "through an entry Eugene did not add."
    )
    with pytest.raises(ValueError, match="not waiting for a decision"):
        access.decide("workbench", grant.id, entry.id, "remove")


def test_remove_this_entry_too_deletes_it_and_finishes_the_removal(tmp_path: Path) -> None:
    backend = AskingPermissions()
    access = service(tmp_path, backend)
    grant = access.add("workbench", "/srv/acme", "write", "operator")
    entry = access.remove("workbench", grant.id).unknown[0]  # type: ignore[index]
    with pytest.raises(KeyError):
        access.decide("workbench", grant.id, "no-such-entry", "remove")
    with pytest.raises(KeyError):
        access.decide("other", grant.id, entry.id, "remove")
    done = access.decide("workbench", grant.id, entry.id, "remove")
    assert backend.decisions == [(AskingPermissions.FOUND["fingerprint"], "remove")]
    assert done.state == "removed" and not done.unknown
    assert done.message == "Eugene's additions were removed. Preexisting permissions stay."


def test_decide_route_needs_an_operator_session(
    authed_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from eugene_plexus_agent.routes import directories

    access = service(tmp_path, AskingPermissions())
    monkeypatch.setattr(directories, "_access", lambda *_: access)
    grant = access.add("workbench", "/srv/acme", "write", "operator")
    entry = access.remove("workbench", grant.id).unknown[0]  # type: ignore[index]
    route = f"/v1/apps/workbench/folder-access/{grant.id}/unknown/{entry.id}"
    key = authed_client.post("/v1/auth/client-keys", json={"name": "app:workbench@local"}).json()[
        "token"
    ]
    for token in (key, local_service_token(authed_client.app, "gateway")):
        refused = authed_client.post(
            route, json={"action": "remove"}, headers={"Authorization": f"Bearer {token}"}
        )
        assert refused.status_code in (401, 403), refused.text
    assert access.listing("workbench")[0].state == "incomplete"
    assert authed_client.post(route + "x", json={"action": "keep"}).status_code == 404
    decided = authed_client.post(route, json={"action": "keep"})
    assert decided.status_code == 200, decided.text
    assert decided.json()["state"] == "removed"
    assert decided.json()["unknown"][0]["decision"] == "kept"
    again = authed_client.post(route, json={"action": "remove"})
    assert again.status_code == 409, again.text


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
def test_windows_detected_roots_keep_system_home_and_install_causes(tmp_path: Path) -> None:
    own = Path.home() / "ep"
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(str(own),))
    access = fa.Access(tmp_path / "access.json", backend)
    for path, cause in [
        (os.environ["SYSTEMROOT"], "system"),
        (str(Path.home()), "home"),
        (str(own), "eugene"),
        (str(own / "apps"), "eugene"),
    ]:
        with pytest.raises(fa.Refused) as exc:
            access.add("workbench", path, "read", "operator")
        assert exc.value.cause == cause
    assert not access.file.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows DACLs")
@pytest.mark.usefixtures("outside_windows_homes")
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
@pytest.mark.usefixtures("outside_windows_homes")
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
    with pytest.raises(fa.Refused, match=r"linked\.txt"):
        backend.grant(str(folder), "read", lambda _: None)
    assert (
        ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
            ws.GetFileSecurity(str(target), 4), 1, 4
        )
        == before
    )
