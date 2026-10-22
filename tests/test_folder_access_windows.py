"""Real temporary NTFS objects, including an insertion during the grant."""

import os
from pathlib import Path

import pytest

from eugene_plexus_agent import folder_access as fa

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows DACLs")


def private_acl(root: Path) -> None:
    import win32security as ws

    own = ws.GetFileSecurity(str(root), ws.OWNER_SECURITY_INFORMATION).GetSecurityDescriptorOwner()
    acl = ws.ACL()
    acl.AddAccessAllowedAceEx(ws.ACL_REVISION, 3, 0x1F01FF, own)
    acl.AddAccessAllowedAceEx(ws.ACL_REVISION, 3, 0x1F01FF, ws.ConvertStringSidToSid("S-1-5-18"))
    ws.SetNamedSecurityInfo(
        str(root),
        ws.SE_FILE_OBJECT,
        4 | ws.PROTECTED_DACL_SECURITY_INFORMATION,
        None,
        None,
        acl,
        None,
    )


def test_checks_each_child_and_removes_new_inherited_rights(tmp_path: Path) -> None:
    import win32security as ws

    root = tmp_path / "shared"
    (root / "child").mkdir(parents=True)
    file = root / "child" / "existing.txt"
    file.write_text("Shared")
    own_sid = ws.GetFileSecurity(
        str(root), ws.OWNER_SECURITY_INFORMATION
    ).GetSecurityDescriptorOwner()
    private = ws.ACL()
    private.AddAccessAllowedAceEx(ws.ACL_REVISION, 3, 0x1F01FF, own_sid)
    private.AddAccessAllowedAceEx(
        ws.ACL_REVISION, 3, 0x1F01FF, ws.ConvertStringSidToSid("S-1-5-18")
    )
    ws.SetNamedSecurityInfo(
        str(root),
        ws.SE_FILE_OBJECT,
        4 | ws.PROTECTED_DACL_SECURITY_INFORMATION,
        None,
        None,
        private,
        None,
    )
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    before = ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
        ws.GetFileSecurity(str(file), 4), 1, 4
    )
    delta, count, _ = backend.grant(str(root), "read", lambda _: None)
    assert delta["added"] and count == 0
    assert len(delta["items"]) == 3
    new = root / "new.txt"
    new.write_text("New shared text")
    assert backend.revoke(str(root), delta)
    after = ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
        ws.GetFileSecurity(str(file), 4), 1, 4
    )
    assert after == before
    acl = ws.GetFileSecurity(str(new), 4).GetSecurityDescriptorDacl()
    assert all(
        ws.ConvertSidToStringSid(acl.GetAce(i)[-1]) != "S-1-5-32-545"
        for i in range(acl.GetAceCount())
    )


def test_does_not_propagate_to_a_late_hardlink(tmp_path: Path) -> None:
    import win32security as ws

    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    outside = tmp_path / "outside.txt"
    outside.write_text("Private")
    before = ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
        ws.GetFileSecurity(str(outside), 4), 1, 4
    )
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    late = root / "late.txt"

    def add_link(_):
        if not late.exists():
            os.link(outside, late)

    delta, _, _ = backend.grant(str(root), "read", add_link)
    assert (
        ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
            ws.GetFileSecurity(str(outside), 4), 1, 4
        )
        == before
    )
    late.unlink()
    assert backend.revoke(str(root), delta)


def test_remove_reports_inherited_rights_created_during_removal(
    tmp_path: Path, monkeypatch
) -> None:
    import win32security as ws

    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    access = fa.Access(tmp_path / "folder-access.json", backend)
    grant = access.add("workbench", str(root), "read", "operator")
    assert not grant.preexisting
    original = ws.SetSecurityInfo
    late = root / "late.txt"

    def during_remove(*args):
        if not late.exists():
            late.write_text("Created while access was changing")
        return original(*args)

    monkeypatch.setattr(ws, "SetSecurityInfo", during_remove)
    outcome = access.remove("workbench", grant.id)
    assert outcome.state == "failed"
    assert "during removal" in outcome.message
    monkeypatch.setattr(ws, "SetSecurityInfo", original)
    assert access.remove("workbench", grant.id).state == "removed"
