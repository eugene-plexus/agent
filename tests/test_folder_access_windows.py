"""Real temporary NTFS objects, including an insertion during the grant."""

import os
from pathlib import Path

import pytest

from eugene_plexus_agent import folder_access as fa

pytestmark = [
    pytest.mark.skipif(os.name != "nt", reason="Windows DACLs"),
    pytest.mark.usefixtures("outside_windows_homes"),
]


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


def test_revoke_removes_file_created_during_removal(tmp_path: Path, monkeypatch) -> None:
    import win32security as ws

    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    access = fa.Access(tmp_path / "folder-access.json", backend)
    grant = access.add("workbench", str(root), "read", "operator")
    assert not grant.preexisting
    original = ws.SetKernelObjectSecurity
    late = root / "late.txt"

    def during_remove(*args):
        if not late.exists():
            late.write_text("Created while access was changing")
        return original(*args)

    monkeypatch.setattr(ws, "SetKernelObjectSecurity", during_remove)
    outcome = access.remove("workbench", grant.id)
    assert outcome.state == "removed"
    acl = ws.GetFileSecurity(str(late), 4).GetSecurityDescriptorDacl()
    assert all(
        ws.ConvertSidToStringSid(acl.GetAce(i)[-1]) != backend.account
        for i in range(acl.GetAceCount())
    )
    monkeypatch.setattr(ws, "SetKernelObjectSecurity", original)
    assert access.remove("workbench", grant.id).state == "removed"


@pytest.mark.parametrize("items", [20, 8300])
def test_large_tree_has_bounded_handles_and_two_saves(tmp_path: Path, items: int) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    for i in range(items):
        (root / str(i)).touch()
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    saves = []
    delta, _, _ = backend.grant(str(root), "read", lambda r: saves.append(len(r["items"])))
    assert saves == [items + 1, items + 1]
    assert backend.revoke(str(root), delta)


def test_revoke_after_growth_past_crt_limit(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    for i in range(8300):
        (root / str(i)).touch()
    assert backend.revoke(str(root), delta)


def test_revoke_tolerates_junctions_and_hardlinks(tmp_path: Path) -> None:
    import subprocess

    import win32security as ws

    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    inside = root / "inside"
    inside.write_text("shared")
    outside = tmp_path / "outside"
    outside.mkdir()
    file = outside / "private"
    file.write_text("private")
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    os.link(inside, outside / "also-inside")
    os.link(file, root / "also-private")
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root / "junction"), str(outside)],
        check=True,
        capture_output=True,
    )
    before = ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
        ws.GetFileSecurity(str(file), 4), 1, 4
    )
    assert backend.revoke(str(root), delta)
    assert (
        ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
            ws.GetFileSecurity(str(file), 4), 1, 4
        )
        == before
    )
    for path in (inside, root / "junction"):
        with backend._open(str(path), _single=True, revoke=True) as (handle, _, _):
            acl = backend._acl(ws, handle)
            assert all(
                ws.ConvertSidToStringSid(acl.GetAce(i)[-1]) != backend.account
                for i in range(acl.GetAceCount())
            )
    (root / "junction").rmdir()


def test_revoke_continues_when_file_vanishes_before_write_open(tmp_path: Path, monkeypatch) -> None:
    import contextlib

    import win32security as ws

    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    vanishing = root / "first"
    vanishing.touch()
    tail = root / "last"
    tail.touch()
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    access = fa.Access(tmp_path / "access.json", backend)
    grant = access.add("workbench", str(root), "read", "operator")
    original = backend._open
    observed = []

    @contextlib.contextmanager
    def disappearing(path, **kwargs):
        if path == str(vanishing) and kwargs.get("change"):
            vanishing.unlink()
            raise FileNotFoundError(path)
        with original(path, **kwargs) as result:
            if kwargs.get("change"):
                observed.append(path)
            yield result

    monkeypatch.setattr(backend, "_open", disappearing)
    assert access.remove("workbench", grant.id).state == "removed"
    assert not vanishing.exists()
    assert str(tail) in observed, "Removal must finish the walk before reporting removed"
    for path in (root, tail):
        acl = ws.GetFileSecurity(str(path), 4).GetSecurityDescriptorDacl()
        assert all(
            ws.ConvertSidToStringSid(acl.GetAce(i)[-1]) != backend.account
            for i in range(acl.GetAceCount())
        )


def test_acl_open_shares_all(tmp_path: Path) -> None:
    import win32file

    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    file = root / "held.txt"
    file.write_text("open elsewhere")
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    held = win32file.CreateFile(str(file), 0x80000000, 1, None, 3, 0, None)
    try:
        delta, _, _ = backend.grant(str(root), "read", lambda _: None)
        assert backend.revoke(str(root), delta)
    finally:
        held.Close()


def test_acl_open_failure_names_the_file(tmp_path: Path, monkeypatch) -> None:
    from ctypes import wintypes

    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    file = root / "held.txt"
    file.touch()
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    original = fa.ctypes.WinDLL
    kernel = original("kernel32", use_last_error=True)
    create = kernel.CreateFileW

    def denied(path, *args):
        # Inject the kernel's error code, not an exception with a filename.
        # The real _open() must construct and annotate the resulting error.
        if path == str(file):
            fa.ctypes.set_last_error(5)
            return wintypes.HANDLE(-1).value
        create.argtypes, create.restype = denied.argtypes, denied.restype
        return create(path, *args)

    class Kernel:
        CreateFileW = staticmethod(denied)

        def __getattr__(self, name):
            return getattr(kernel, name)

    monkeypatch.setattr(
        fa.ctypes,
        "WinDLL",
        lambda name, **kw: Kernel() if name == "kernel32" else original(name, **kw),
    )
    with pytest.raises(OSError, match=r"held\.txt"):
        backend.grant(str(root), "read", lambda _: None)


def test_windows_old_and_device_paths_have_correct_refusal(tmp_path: Path) -> None:
    backend = fa.WindowsPermissions("S-1-5-32-545", eugene=(), homes=())
    access = fa.Access(tmp_path / "access.json", backend)
    with pytest.raises(fa.Refused, match="system folders"):
        access.add("workbench", tmp_path.anchor + "Windows.old", "read", "operator")
    with pytest.raises(fa.Refused) as exc:
        access.add("workbench", "\\\\?\\" + str(tmp_path), "read", "operator")
    assert exc.value.cause == "not_a_folder"


USERS = "S-1-5-32-545"


def sid_aces(path: Path, sid: str = USERS) -> list[tuple[int, int]]:
    """(flags, mask) of every ACE that names `sid` on `path`."""
    import win32security as ws

    acl = ws.GetFileSecurity(str(path), 4).GetSecurityDescriptorDacl()
    return [
        (acl.GetAce(i)[0][1], acl.GetAce(i)[1])
        for i in range(acl.GetAceCount())
        if ws.ConvertSidToStringSid(acl.GetAce(i)[-1]) == sid
    ]


def icacls(path: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["icacls", str(path), *args], check=True, capture_output=True)


def owner(path: Path) -> str:
    import win32security as ws

    return str(
        ws.ConvertSidToStringSid(
            ws.GetFileSecurity(
                str(path), ws.OWNER_SECURITY_INFORMATION
            ).GetSecurityDescriptorOwner()
        )
    )


def test_revoke_removes_rights_a_later_protected_folder_made_explicit(tmp_path: Path) -> None:
    """Review blocker: an app protects a folder it made, so Eugene's inherited ACE
    becomes an explicit copy; removal reported success while the copy stayed."""
    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    keep = root / "keep"
    keep.mkdir()
    (keep / "secret.txt").write_text("app data")
    icacls(keep, "/inheritance:d")  # what the app's own account may do to a folder it owns
    assert (3, backend.WRITE) in sid_aces(keep)
    assert backend.revoke(str(root), delta)
    for item in (root, keep, keep / "secret.txt"):
        assert sid_aces(item) == [], item


def test_revoke_names_access_it_cannot_attribute_and_never_reports_success(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    locked = root / "locked"
    locked.mkdir()
    icacls(
        locked,
        "/inheritance:r",
        "/grant:r",
        f"*{owner(locked)}:(OI)(CI)(F)",
        f"*{USERS}:(OI)(CI)(F)",
    )
    with pytest.raises(fa.RemovalIncomplete, match=r"locked.*changed outside Eugene"):
        backend.revoke(str(root), delta)
    # Its origin is unknown, so it stays; the record says what is left.
    assert sid_aces(locked) == [(3, 0x1F01FF)]
    assert sid_aces(root) == []


def test_revoke_keeps_a_by_hand_entry_in_a_folder_protected_before_the_grant(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    closed = root / "closed"
    closed.mkdir()
    icacls(
        closed,
        "/inheritance:r",
        "/grant:r",
        f"*{owner(closed)}:(OI)(CI)(F)",
        f"*{USERS}:(OI)(CI)(RX)",
    )
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, count, _ = backend.grant(str(root), "read", lambda _: None)
    assert count == 1
    assert backend.revoke(str(root), delta)
    assert sid_aces(closed) == [(3, backend.READ)]


def test_windows_refuses_a_grant_inside_or_around_another(tmp_path: Path) -> None:
    """Review: a child grant found the parent's inherited rights and recorded nothing,
    so removing the parent took the child's access while its record said ready."""
    root = tmp_path / "shared"
    (root / "sub").mkdir(parents=True)
    private_acl(root)
    access = fa.Access(tmp_path / "access.json", fa.WindowsPermissions(USERS, eugene=(), homes=()))
    parent = access.add("workbench", str(root), "read", "operator")
    assert parent.state == "ready"
    with pytest.raises(ValueError, match=r"is inside .*shared, which Workbench can already open"):
        access.add("workbench", str(root / "sub"), "write", "operator")
    assert access.remove("workbench", parent.id).state == "removed"
    assert access.add("workbench", str(root / "sub"), "write", "operator").state == "ready"
