"""An owner approves a program one other account can change (row 8, part 8c).

Call R8-EXEC-TRUST, Choice 3. Run as its own account, the agent runs a
program that one other account can change only once an owner approves it.
The approval binds the path that runs, that account, the file's SHA-256
and every item the check walked, and lapses the first time any differs.
It trusts that account: the hash catches a change, it is not a boundary.

The disk is `test_exec_trust`'s stand-in; the approvals file is real.
"""

from __future__ import annotations

import dataclasses
import json
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import exec_approvals, exec_trust

from .conftest import local_service_token
from .test_exec_trust import Disk, disk, ran, refused  # noqa: F401 -- `disk` is a fixture

TROY_S = "/Users/troy/llama/llama-server"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    exec_approvals.configure(tmp_path)
    # The file is on the real disk while the walk reads the stand-in; the
    # check that the file is Eugene's alone has its own test below.
    monkeypatch.setattr(exec_approvals, "_kept_safe", lambda path: None)
    yield tmp_path / exec_approvals.FILE_NAME
    exec_approvals.configure(None)


def _troy_s(disk: Disk, content: bytes = b"\x7fELF build 1") -> str:  # noqa: F811
    disk.dir("/Users/troy", uid=501).dir("/Users/troy/llama", uid=501)
    disk.file(TROY_S, uid=501, content=content)
    return TROY_S


def _approve(path: str) -> exec_approvals.ExecApproval:
    shown = exec_trust.assess(path)
    assert shown.kind is exec_trust.Kind.needs_approval and shown.sha256
    return exec_approvals.approve(path, shown.sha256, by="operator")


def test_an_approved_program_runs_and_lapses_when_its_bytes_change(disk: Disk, store: Path) -> None:  # noqa: F811
    path = _troy_s(disk)
    refused(path, "only once an owner approves it")
    approval = _approve(path)
    assert (approval.path, approval.account, approval.accountUid) == (path, "troy", 501)
    assert ran(path) == path
    assert json.loads(store.read_text(encoding="utf-8"))["approvals"][0]["id"] == approval.id

    disk.file(path, uid=501, content=b"\x7fELF build 2")  # one byte differs
    refused(path, "only once an owner approves it")
    assert exec_approvals.listed() == []


def test_an_approval_lapses_when_a_folder_on_the_way_changes(disk: Disk, store: Path) -> None:  # noqa: F811
    path = _troy_s(disk)
    _approve(path)
    disk.dir("/Users/troy/llama", uid=501, mode=0o700)  # still troy's alone
    assert exec_trust.assess(path).kind is exec_trust.Kind.needs_approval
    refused(path, "only once an owner approves it")
    assert exec_approvals.listed() == []


def test_an_approval_lapses_when_another_account_holds_it_now(disk: Disk, store: Path) -> None:  # noqa: F811
    path = _troy_s(disk)
    _approve(path)
    for item in ("/Users/troy", "/Users/troy/llama", TROY_S):
        disk.nodes[item]["uid"] = 502
    assert exec_trust.assess(path).account == "alice"
    refused(path, "which belongs to alice")
    assert exec_approvals.listed() == []


def test_each_thing_an_approval_binds_is_compared_on_its_own(disk: Disk, store: Path) -> None:  # noqa: F811
    """The account, the bytes and the chain, each by itself: a verdict
    that differs from the approval in one of them only."""
    path = _troy_s(disk)
    approval = _approve(path)
    now = exec_trust.assess(path)
    assert exec_approvals._lapsed(approval, now) is None
    other = dataclasses.replace(now, account="alice", account_uid=502)
    assert exec_approvals._lapsed(approval, other) == "it is alice's now, not troy's"
    assert exec_approvals._lapsed(approval, dataclasses.replace(now, sha256="0" * 64)) == (
        "the file changed"
    )
    assert exec_approvals._lapsed(approval, dataclasses.replace(now, chain=now.chain[:-1])) == (
        "a folder, link or file on its way changed"
    )


def test_an_approval_does_not_carry_over_to_a_refused_program(disk: Disk, store: Path) -> None:  # noqa: F811
    """Refused once, the approval is gone: its folder opened to anyone and
    closed again, the program asks again (8c review, 5)."""
    path = _troy_s(disk)
    _approve(path)
    disk.dir("/Users/troy/llama", uid=501, mode=0o777)
    refused(path, "will not run", "mode 0777: anyone")
    assert exec_approvals.listed() == []
    disk.dir("/Users/troy/llama", uid=501, mode=0o755)
    refused(path, "only once an owner approves it")


def _linked(disk: Disk) -> str:  # noqa: F811
    """`/Users/troy/current -> /opt/a`, and root's program there."""
    disk.dir("/Users/troy", uid=501).dir("/opt/a").file("/opt/a/llama-server")
    disk.link("/Users/troy/current", "/opt/a")
    named = "/Users/troy/current/llama-server"
    shown = exec_trust.assess(named)
    assert shown.kind is exec_trust.Kind.needs_approval and shown.sha256
    exec_approvals.approve(named, shown.sha256, by="operator")
    assert ran(named) == "/opt/a/llama-server"
    return named


def test_a_refusal_by_the_real_path_drops_the_approval_of_the_name_reaching_it(
    disk: Disk,  # noqa: F811
    store: Path,
) -> None:
    """`/opt/a` opened to anyone and checked by its own path, then closed
    again: the approval of the link that reaches it does not revive (8c
    re-review, 4)."""
    named = _linked(disk)
    disk.dir("/opt/a", mode=0o777)
    refused("/opt/a/llama-server", "will not run", "mode 0777: anyone")
    disk.dir("/opt/a", mode=0o755)
    assert exec_approvals.listed() == []
    refused(named, "only once an owner approves it")


ENV = b"\xcf\xfa\xed\xfe the system's env"


def test_a_refusal_by_one_alias_drops_the_approval_of_another(
    disk: Disk,  # noqa: F811
    store: Path,
) -> None:
    """Alias A -> server is approved; server becomes `env`'s bytes; alias B
    -> server is checked (refused, and the check stops before it returns a
    program); server is put back. A does not run again: the refusal names
    the file it reached, whatever it was looked up by (8c re-review 2, 4)."""
    disk.file("/usr/bin/env", content=ENV)
    disk.dir("/Users/troy", uid=501).dir("/opt/srv").file("/opt/srv/server", content=b"\x7fELF srv")
    disk.link("/Users/troy/a", "/opt/srv/server")
    disk.link("/Users/troy/b", "/opt/srv/server")
    shown = exec_trust.assess("/Users/troy/a")
    assert shown.kind is exec_trust.Kind.needs_approval and shown.sha256
    exec_approvals.approve("/Users/troy/a", shown.sha256, by="operator")
    assert ran("/Users/troy/a") == "/Users/troy/a"
    disk.nodes["/opt/srv/server"]["content"] = ENV
    refused("/Users/troy/b", "is the system's env under another name")
    disk.nodes["/opt/srv/server"]["content"] = b"\x7fELF srv"
    assert exec_approvals.listed() == []
    refused("/Users/troy/a", "only once an owner approves it")


def test_a_refusal_of_what_runs_a_script_drops_the_script_s_approval(
    disk: Disk,  # noqa: F811
    store: Path,
) -> None:
    """troy's script runs root's `/opt/py/python`, which another name
    (`/usr/local/bin/py`) reaches too: refused by that name, the script's
    approval goes, though the interpreter is not the file it approved."""
    disk.file("/usr/bin/env", content=ENV)
    disk.dir("/opt/py").file("/opt/py/python", content=b"\x7fELF python")
    disk.link("/usr/local/bin/py", "/opt/py/python")
    disk.dir("/usr/share/tool")
    # The interpreter is not the last file the chain holds: its argument is.
    disk.dir("/Users/troy", uid=501).file(
        "/Users/troy/tool", uid=501, content=b"#!/opt/py/python -I/usr/share/tool\nprint(1)\n"
    )
    shown = exec_trust.assess("/Users/troy/tool")
    assert shown.kind is exec_trust.Kind.needs_approval and shown.sha256
    exec_approvals.approve("/Users/troy/tool", shown.sha256, by="operator")
    disk.nodes["/opt/py/python"]["content"] = ENV
    refused("/usr/local/bin/py", "is the system's env under another name")
    disk.nodes["/opt/py/python"]["content"] = b"\x7fELF python"
    assert exec_approvals.listed() == []


def test_a_refusal_by_a_hard_link_drops_the_approval_of_the_same_file(
    disk: Disk,  # noqa: F811
    store: Path,
) -> None:
    """The same file by two names that are not links (its device and inode):
    a refusal by one drops the approval of the other."""
    disk.file("/usr/bin/env", content=ENV)
    disk.dir("/Users/troy", uid=501).file("/Users/troy/server", uid=501, ino=77)
    disk.dir("/opt/srv").file("/opt/srv/server", uid=501, ino=77)
    shown = exec_trust.assess("/Users/troy/server")
    assert shown.kind is exec_trust.Kind.needs_approval and shown.sha256
    exec_approvals.approve("/Users/troy/server", shown.sha256, by="operator")
    for name in ("/Users/troy/server", "/opt/srv/server"):
        disk.nodes[name]["content"] = ENV
    refused("/opt/srv/server", "is the system's env under another name")
    for name in ("/Users/troy/server", "/opt/srv/server"):
        disk.nodes[name]["content"] = b"\x7fELF"
    assert exec_approvals.listed() == []


def test_a_refusal_holds_when_the_approvals_cannot_be_written(
    disk: Disk,  # noqa: F811
    store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The store cannot be written when a refusal drops an approval: the
    refusal is held in memory, so the program does not run again when what
    made it refused is put back; the next write that succeeds drops it
    from the file too."""
    named = _linked(disk)
    real_write = exec_approvals.write_private

    def cannot(path: Any, data: Any, **kw: Any) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(exec_approvals, "write_private", cannot)
    disk.dir("/opt/a", mode=0o777)
    refused(named, "will not run", "mode 0777: anyone")
    disk.dir("/opt/a", mode=0o755)
    refused(named, "only once an owner approves it")
    assert json.loads(store.read_text(encoding="utf-8"))["approvals"]  # still in the file
    assert exec_approvals.listed() == []
    monkeypatch.setattr(exec_approvals, "write_private", real_write)
    refused(named, "only once an owner approves it")
    assert json.loads(store.read_text(encoding="utf-8"))["approvals"] == []


def test_a_program_reached_through_a_linked_folder_is_approved_by_the_path_it_is_named_by(
    disk: Disk,  # noqa: F811
    store: Path,
) -> None:
    """`/Users/troy/current/llama-server`, with `current` a link to
    `/opt/real`: troy can point the link elsewhere, so it asks. It is shown,
    approved and kept by the path a setting names, and that path runs
    afterwards; the file that runs is the one the link reaches (8c review,
    6)."""
    disk.dir("/Users/troy", uid=501).dir("/opt/real").file("/opt/real/llama-server")
    disk.link("/Users/troy/current", "/opt/real")
    named = "/Users/troy/current/llama-server"
    shown = exec_trust.assess(named)
    assert shown.kind is exec_trust.Kind.needs_approval and shown.lookup == named
    view = exec_approvals.view(shown)
    assert view is not None and view.path == named
    assert shown.sha256 is not None
    approval = exec_approvals.approve(view.path, shown.sha256, by="operator")
    assert approval.path == named
    assert ran(named) == "/opt/real/llama-server"
    assert [a.id for a in exec_approvals.listed()] == [approval.id]


def test_a_refused_or_changed_or_trusted_program_is_not_approved(disk: Disk, store: Path) -> None:  # noqa: F811
    path = _troy_s(disk)
    with pytest.raises(exec_approvals.NotApprovable, match="changed since it was shown"):
        exec_approvals.approve(path, "0" * 64, by="operator")
    disk.dir("/usr/local/shared", mode=0o775, gid=20).file("/usr/local/shared/llama-server")
    with pytest.raises(exec_approvals.NotApprovable, match="can be changed by others"):
        exec_approvals.approve("/usr/local/shared/llama-server", "0" * 64, by="operator")
    disk.dir(exec_trust_home := "/usr/local/eugene-plexus", uid=400).file(
        f"{exec_trust_home}/llama-server", uid=400
    )
    with pytest.raises(
        exec_approvals.NotApprovable, match="already, so there is nothing to approve"
    ):
        exec_approvals.approve(f"{exec_trust_home}/llama-server", "0" * 64, by="operator")
    assert exec_approvals.listed() == []


def test_approvals_are_not_read_from_a_file_others_could_change(
    disk: Disk,  # noqa: F811
    store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _troy_s(disk)
    _approve(path)
    monkeypatch.setattr(
        exec_approvals, "_kept_safe", lambda p: f"{p} can be changed by others (mode 0666)"
    )
    refused(path, "only once an owner approves it")
    with pytest.raises(exec_approvals.NotApprovable, match="can be changed by others"):
        _approve(path)


def test_the_approvals_file_is_checked_like_a_program(
    disk: Disk,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_kept_safe` itself: the file and every folder above it are root's
    or Eugene's, and nobody else can write them."""
    disk.dir("/usr/local/eugene-plexus", uid=400, mode=0o750)
    disk.file("/usr/local/eugene-plexus/exec-approvals.json", uid=400, mode=0o600)
    assert exec_approvals._kept_safe(Path("/usr/local/eugene-plexus/exec-approvals.json")) is None
    disk.nodes["/usr/local/eugene-plexus/exec-approvals.json"]["mode"] = 0o666
    why = exec_approvals._kept_safe(Path("/usr/local/eugene-plexus/exec-approvals.json"))
    assert why is not None and "can be changed by others" in why
    disk.nodes["/usr/local/eugene-plexus/exec-approvals.json"]["mode"] = 0o600
    disk.nodes["/usr/local/eugene-plexus"]["uid"] = 501
    why = exec_approvals._kept_safe(Path("/usr/local/eugene-plexus/exec-approvals.json"))
    assert why is not None and "belongs to troy" in why


# --- the routes -----------------------------------------------------------------


def _verdict(kind: exec_trust.Kind, sha: str = "ab" * 32) -> exec_trust.Verdict:
    return exec_trust.Verdict(
        kind=kind,
        given=TROY_S,
        program=TROY_S,
        account="troy" if kind is exec_trust.Kind.needs_approval else None,
        account_uid=501 if kind is exec_trust.Kind.needs_approval else None,
        sha256=sha if kind is exec_trust.Kind.needs_approval else None,
        chain=(exec_trust.ChainItem(TROY_S, 501, stat.S_IFREG | 0o755),),
        findings=(
            exec_trust.Finding(
                problem="others" if kind is exec_trust.Kind.refused else "owner",
                path="/Users/troy",
                uid=501,
                text="it is reached through /Users/troy, which can be changed by others "
                "(mode 0777: anyone)"
                if kind is exec_trust.Kind.refused
                else "it is reached through /Users/troy, which belongs to troy",
            ),
        )
        if kind is not exec_trust.Kind.ok
        else (),
    )


@pytest.fixture
def judged(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    now: dict[str, Any] = {"verdict": _verdict(exec_trust.Kind.needs_approval)}
    monkeypatch.setattr(exec_trust, "assess", lambda path: now["verdict"])
    monkeypatch.setattr(exec_approvals, "_kept_safe", lambda path: None)
    return now


def test_the_owner_approves_lists_and_takes_back_through_the_api(
    authed_client: TestClient, judged: dict[str, Any]
) -> None:
    stale = authed_client.post("/v1/exec-approvals", json={"path": TROY_S, "sha256": "cd" * 32})
    assert (
        stale.status_code == 409
        and "changed since it was shown" in stale.json()["detail"]["detail"]
    )

    made = authed_client.post("/v1/exec-approvals", json={"path": TROY_S, "sha256": "ab" * 32})
    assert made.status_code == 201, made.text
    body = made.json()
    assert (body["path"], body["account"], body["accountUid"], body["sha256"]) == (
        TROY_S,
        "troy",
        501,
        "ab" * 32,
    )
    assert body["chain"] == [{"path": TROY_S, "uid": 501, "mode": stat.S_IFREG | 0o755}]
    listed = authed_client.get("/v1/exec-approvals").json()["approvals"]
    assert [a["id"] for a in listed] == [body["id"]]

    assert authed_client.delete(f"/v1/exec-approvals/{body['id']}").status_code == 204
    assert authed_client.get("/v1/exec-approvals").json()["approvals"] == []
    assert authed_client.delete(f"/v1/exec-approvals/{body['id']}").status_code == 404


def test_the_api_refuses_a_refused_program_naming_the_cause(
    authed_client: TestClient, judged: dict[str, Any]
) -> None:
    judged["verdict"] = _verdict(exec_trust.Kind.refused)
    refused_ = authed_client.post("/v1/exec-approvals", json={"path": TROY_S, "sha256": "ab" * 32})
    assert refused_.status_code == 409
    assert "can be changed by others (mode 0777: anyone)" in refused_.json()["detail"]["detail"]
    judged["verdict"] = _verdict(exec_trust.Kind.ok)
    trusted = authed_client.post("/v1/exec-approvals", json={"path": TROY_S, "sha256": "ab" * 32})
    assert trusted.status_code == 409 and "nothing to approve" in trusted.json()["detail"]["detail"]
    assert authed_client.get("/v1/exec-approvals").json()["approvals"] == []


def test_only_an_owner_s_session_approves(
    app: FastAPI, authed_client: TestClient, judged: dict[str, Any]
) -> None:
    """A component's or the gateway's token is not an owner: it can neither
    approve, nor list, nor take back."""
    gateway = {"Authorization": f"Bearer {local_service_token(app, 'gateway')}"}
    library = {"Authorization": f"Bearer {local_service_token(app, 'library')}"}
    body = {"path": TROY_S, "sha256": "ab" * 32}
    for headers in (gateway, library):
        assert (
            authed_client.post("/v1/exec-approvals", json=body, headers=headers).status_code == 401
        )
        assert authed_client.get("/v1/exec-approvals", headers=headers).status_code == 401
    made = authed_client.post("/v1/exec-approvals", json=body).json()
    assert (
        authed_client.delete(f"/v1/exec-approvals/{made['id']}", headers=gateway).status_code == 401
    )
    assert [a["id"] for a in authed_client.get("/v1/exec-approvals").json()["approvals"]] == [
        made["id"]
    ]
