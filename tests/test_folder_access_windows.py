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
    # A file with another name is never changed by removal: the owner decides.
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    assert sorted((e["kind"], Path(e["path"]).name) for e in asked.value.entries) == [
        ("hardlink", "also-private"),
        ("hardlink", "inside"),
    ]
    for found in asked.value.entries:
        backend.decide(str(root), delta, found["fingerprint"], "keep")
    assert backend.revoke(str(root), delta)
    assert (
        ws.ConvertSecurityDescriptorToStringSecurityDescriptor(
            ws.GetFileSecurity(str(file), 4), 1, 4
        )
        == before
    )
    # The junction is an item of its own: its ACL is checked, and never followed.
    with backend._open(str(root), names=("junction",), revoke=True) as (handle, _, _):
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
        item = str(Path(path).joinpath(*kwargs.get("names", ())))
        if item == str(vanishing) and kwargs.get("change"):
            vanishing.unlink()
            raise FileNotFoundError(item)
        with original(path, **kwargs) as result:
            if kwargs.get("change"):
                observed.append(item)
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
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    (found,) = asked.value.entries
    assert found["kind"] == "entry" and found["path"].endswith("locked")
    assert "full control" in found["entry"]
    # Its origin is unknown, so it stays until the owner decides (R23-UNKNOWN).
    assert sid_aces(locked) == [(3, 0x1F01FF)]
    assert sid_aces(root) == []
    backend.decide(str(root), delta, found["fingerprint"], "remove")
    assert sid_aces(locked) == []
    assert backend.revoke(str(root), delta)


def test_a_kept_entry_stays_and_no_longer_holds_the_removal(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    extra = root / "extra"
    extra.mkdir()
    icacls(extra, "/grant", f"*{USERS}:(OI)(CI)(F)")
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    (found,) = asked.value.entries
    backend.decide(str(root), delta, found["fingerprint"], "keep")
    assert backend.revoke(str(root), delta)
    assert (3, 0x1F01FF) in sid_aces(extra)


def test_remove_this_entry_too_checks_it_again_first(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    extra = root / "extra"
    extra.mkdir()
    icacls(extra, "/grant", f"*{USERS}:(OI)(CI)(F)")
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    (found,) = asked.value.entries
    icacls(extra, "/remove:g", f"*{USERS}")
    icacls(extra, "/grant", f"*{USERS}:(OI)(CI)(M)")  # changed since it was found
    with pytest.raises(ValueError, match="changed since"):
        backend.decide(str(root), delta, found["fingerprint"], "remove")
    assert sid_aces(extra) == [(3, 0x1301BF)]


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


def null_dacl(path: Path) -> None:
    import win32security as ws

    ws.SetNamedSecurityInfo(
        str(path), ws.SE_FILE_OBJECT, ws.DACL_SECURITY_INFORMATION, None, None, None, None
    )


def dacl(path: Path):
    import win32security as ws

    return ws.GetFileSecurity(str(path), 4).GetSecurityDescriptorDacl()


def test_revoke_repairs_a_null_dacl_the_app_set_after_the_grant(tmp_path: Path) -> None:
    """Re-review blocker: a null DACL gives everyone everything, and removal read
    it as "no entries" and reported success."""
    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    keep = root / "keep"
    keep.mkdir()
    (keep / "secret.txt").write_text("app data")
    null_dacl(keep)
    assert dacl(keep) is None
    assert backend.revoke(str(root), delta)
    # It inherits the folder's entries again, which no longer name the account.
    assert dacl(keep) is not None
    assert sid_aces(keep) == [] and sid_aces(keep / "secret.txt") == []
    assert owner(root) in {
        __import__("win32security").ConvertSidToStringSid(dacl(keep).GetAce(i)[-1])
        for i in range(dacl(keep).GetAceCount())
    }


def test_revoke_keeps_a_null_dacl_that_was_there_before_the_grant(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    (root / "open").mkdir(parents=True)
    private_acl(root)
    null_dacl(root / "open")
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    assert backend.revoke(str(root), delta)
    assert dacl(root / "open") is None


def test_revoke_reports_an_app_owned_item_it_cannot_give_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner keeps WRITE_DAC, so an item the app owns could get its access back."""
    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    made = root / "made.txt"
    made.write_text("app data")
    import win32file

    real_owner = fa.WindowsPermissions._owner

    def owned_by_app(ws, handle):
        found = real_owner(ws, handle)
        path = win32file.GetFinalPathNameByHandle(handle, 0)
        return USERS if path.endswith("made.txt") else found

    def cannot(*_args, **_kwargs):
        raise PermissionError(1307, "This security ID may not be assigned as the owner")

    monkeypatch.setattr(fa.WindowsPermissions, "_owner", staticmethod(owned_by_app))
    monkeypatch.setattr(backend, "_give", cannot)
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    assert [(e["kind"], Path(e["path"]).name) for e in asked.value.entries] == [
        ("owner", "made.txt")
    ]


def elevated() -> bool:
    import ctypes

    return os.name == "nt" and bool(ctypes.windll.shell32.IsUserAnAdmin())


@pytest.mark.skipif(
    not elevated(),
    reason="Giving an item to another owner needs SeRestorePrivilege (CI runs as admin)",
)
def test_revoke_gives_app_owned_items_to_the_folders_owner(tmp_path: Path) -> None:
    import win32security as ws

    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    keep = root / "keep"
    keep.mkdir()
    (keep / "secret.txt").write_text("app data")
    with fa._privileges(ws, "SeRestorePrivilege"):
        for item in (keep, keep / "secret.txt"):
            ws.SetNamedSecurityInfo(
                str(item),
                ws.SE_FILE_OBJECT,
                ws.OWNER_SECURITY_INFORMATION,
                ws.ConvertStringSidToSid(USERS),
                None,
                None,
                None,
            )
    assert owner(keep) == USERS
    assert backend.revoke(str(root), delta)
    assert owner(keep) == owner(root) and owner(keep / "secret.txt") == owner(root)


def test_revoke_keeps_a_by_hand_entry_on_a_folder_protected_after_the_grant(
    tmp_path: Path,
) -> None:
    """Re-review: an explicit by-hand entry for the account on a folder that is
    protected after the grant matched the copy rule and was deleted."""
    root = tmp_path / "shared"
    keep = root / "keep"
    keep.mkdir(parents=True)
    private_acl(root)
    icacls(keep, "/grant", f"*{USERS}:(OI)(CI)(RX)")
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    icacls(keep, "/inheritance:d")
    assert backend.revoke(str(root), delta)
    assert sid_aces(keep) == [(3, backend.READ)]


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


# --- Re-review 2: exemptions follow the file, never its name or a link ------------


@pytest.fixture
def app_owns(monkeypatch: pytest.MonkeyPatch):
    """Make chosen files look owned by the app's account.

    A real transfer of ownership needs privileges the test account lacks, so the
    owner of a file is read from a set of file identities, the way Windows
    names a file: volume serial and file ID, which a rename keeps."""
    import win32file

    owned: set[tuple[int, int]] = set()
    real = fa.WindowsPermissions._owner

    def owner_of(ws, handle):
        path = win32file.GetFinalPathNameByHandle(handle, 0).removeprefix("\\\\?\\")
        info = os.stat(path)
        return USERS if (info.st_dev, info.st_ino) in owned else real(ws, handle)

    monkeypatch.setattr(fa.WindowsPermissions, "_owner", staticmethod(owner_of))

    def own(path: Path) -> None:
        info = os.stat(path)
        owned.add((info.st_dev, info.st_ino))

    return own


def cannot_give(*_args, **_kwargs):
    raise PermissionError(1307, "This security ID may not be assigned as the owner")


def junction(link: Path, target: Path) -> None:
    import subprocess

    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
    )


def shared_root(tmp_path: Path) -> Path:
    root = tmp_path / "shared"
    root.mkdir()
    private_acl(root)
    return root


def test_ownership_exemption_follows_the_file_after_a_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_owns
) -> None:
    """Re-review 2: the app owned `seed` before the grant. After a rename and a new
    app-owned `seed`, removal exempted the new file by its name."""
    root = shared_root(tmp_path)
    (root / "seed").write_text("old")
    app_owns(root / "seed")
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    monkeypatch.setattr(backend, "_give", cannot_give)
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    (root / "seed").rename(root / "moved")
    (root / "seed").write_text("replacement")
    app_owns(root / "seed")
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    assert [(e["kind"], Path(e["path"]).name) for e in asked.value.entries] == [("owner", "seed")]


def test_a_by_hand_entry_stays_with_its_file_after_a_rename(tmp_path: Path) -> None:
    root = shared_root(tmp_path)
    (root / "seed").write_text("x")
    icacls(root / "seed", "/grant", f"*{USERS}:(RX)")
    by_hand = sid_aces(root / "seed")
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    (root / "seed").rename(root / "moved")
    assert backend.revoke(str(root), delta)
    assert sid_aces(root / "moved") == by_hand


def test_a_null_dacl_from_before_the_grant_stays_after_a_rename(tmp_path: Path) -> None:
    root = shared_root(tmp_path)
    (root / "open").write_text("x")
    null_dacl(root / "open")
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    (root / "open").rename(root / "moved")
    assert backend.revoke(str(root), delta)
    assert dacl(root / "moved") is None


def test_keep_is_about_the_file_not_its_name(tmp_path: Path) -> None:
    root = shared_root(tmp_path)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    extra = root / "extra"
    extra.mkdir()
    icacls(extra, "/grant", f"*{USERS}:(OI)(CI)(F)")
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    backend.decide(str(root), delta, asked.value.entries[0]["fingerprint"], "keep")
    extra.rename(root / "renamed")
    assert backend.revoke(str(root), delta)
    assert [e["path"] for e in delta["retained"]] == [str(root / "renamed")]


def test_removal_never_changes_a_file_with_another_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_owns
) -> None:
    """Re-review 2 (blocker): an app-owned file with a hard link outside the folder
    had its owner and ACE changed through both names."""
    root = shared_root(tmp_path)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    made = root / "made.txt"
    made.write_text("app data")
    app_owns(made)
    alias = tmp_path / "alias.txt"
    os.link(made, alias)
    granted = sid_aces(made)
    assert granted  # inherited from the folder
    monkeypatch.setattr(
        backend, "_give", lambda *a, **k: pytest.fail("ownership changed on a shared file")
    )
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    (found,) = asked.value.entries
    assert (found["kind"], Path(found["path"]).name) == ("hardlink", "made.txt")
    assert found["entry"] == fa.HARDLINK
    assert sid_aces(made) == granted and sid_aces(alias) == granted
    # Keep leaves both names as they are; the removal then finishes.
    backend.decide(str(root), delta, found["fingerprint"], "keep")
    assert backend.revoke(str(root), delta)
    assert sid_aces(made) == granted and [e["kind"] for e in delta["retained"]] == ["hardlink"]


def test_remove_on_a_file_with_another_name_lets_the_removal_change_it(
    tmp_path: Path,
) -> None:
    root = shared_root(tmp_path)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    made = root / "made.txt"
    made.write_text("app data")
    os.link(made, tmp_path / "alias.txt")
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    backend.decide(str(root), delta, asked.value.entries[0]["fingerprint"], "remove")
    assert sid_aces(made) == [(16, backend.WRITE)]  # the decision allows; it changes nothing yet
    assert backend.revoke(str(root), delta)
    assert sid_aces(made) == []


def test_a_decision_never_follows_a_junction_put_in_place_of_a_folder(tmp_path: Path) -> None:
    """Re-review 2 (blocker): after the prompt about shared/sub/f, `sub` moved away and
    a junction took its name. The decision followed it and changed an ACE outside."""
    root = shared_root(tmp_path)
    (root / "sub").mkdir()
    made = root / "sub" / "f"
    made.write_text("x")
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    icacls(made, "/inheritance:r", "/grant:r", f"*{owner(made)}:(F)", f"*{USERS}:(F)")
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    (found,) = asked.value.entries
    outside = tmp_path / "outside"
    (root / "sub").rename(outside)
    junction(root / "sub", outside)
    with pytest.raises((ValueError, OSError)):
        backend.decide(str(root), delta, found["fingerprint"], "remove")
    assert sid_aces(outside / "f") == [(0, 0x1F01FF)]


def test_an_ownership_decision_never_follows_a_junction_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = shared_root(tmp_path)
    (root / "sub").mkdir()
    (root / "sub" / "f").write_text("x")
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    info = os.stat(root / "sub" / "f")
    fingerprint = {
        "kind": "owner",
        "names": ["sub", "f"],
        "identity": [info.st_dev, info.st_ino, info.st_birthtime_ns],
        "state": "",
    }
    outside = tmp_path / "outside"
    (root / "sub").rename(outside)
    junction(root / "sub", outside)
    import win32security

    gave: list[str] = []
    real_set = win32security.SetSecurityInfo
    monkeypatch.setattr(
        win32security, "SetSecurityInfo", lambda *a, **k: gave.append("set") or real_set(*a, **k)
    )
    with pytest.raises((ValueError, OSError)):
        backend.decide(str(root), delta, fingerprint, "remove")
    assert gave == []


def test_reopening_below_a_grant_refuses_a_link_in_any_component(tmp_path: Path) -> None:
    root = shared_root(tmp_path)
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "f").write_text("x")
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    with backend._open(str(root), names=("a", "b", "f"), change=True, revoke=True) as (_, facts, _):
        assert not facts.is_dir
    outside = tmp_path / "outside"
    (root / "a" / "b").rename(outside)
    junction(root / "a" / "b", outside)
    with (
        pytest.raises(fa.Refused) as refused,
        backend._open(str(root), names=("a", "b", "f"), change=True, revoke=True),
    ):
        pass
    assert refused.value.cause == "link"


def test_a_stale_null_dacl_decision_is_not_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-review 2: a failed null-DACL repair waited for the owner; a valid manual
    DACL went in; Remove replaced all of it."""
    root = shared_root(tmp_path)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    keep = root / "keep"
    keep.mkdir()
    null_dacl(keep)
    real_inherit = backend._inherit

    def refuses(*_args, **_kwargs):
        raise PermissionError("its DACL is still null")

    monkeypatch.setattr(backend, "_inherit", refuses)
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    (found,) = asked.value.entries
    assert found["kind"] == "null_dacl"
    icacls(keep, "/grant", f"*{USERS}:(OI)(CI)(RX)", f"*{owner(root)}:(OI)(CI)(F)")

    def aces() -> list:
        acl = dacl(keep)
        return [(acl.GetAce(i)[0], acl.GetAce(i)[1]) for i in range(acl.GetAceCount())]

    manual = aces()
    monkeypatch.setattr(backend, "_inherit", real_inherit)
    with pytest.raises(fa.StaleEntry, match="changed since"):
        backend.decide(str(root), delta, found["fingerprint"], "remove")
    assert aces() == manual


def test_a_stale_owner_decision_is_not_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_owns
) -> None:
    root = shared_root(tmp_path)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    made = root / "made.txt"
    made.write_text("app data")
    app_owns(made)
    monkeypatch.setattr(backend, "_give", cannot_give)
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    (found,) = asked.value.entries
    assert found["kind"] == "owner"
    import win32security

    monkeypatch.delattr(backend, "_give")  # the real one decides, with the real check
    changed: list[str] = []
    real_set = win32security.SetSecurityInfo
    monkeypatch.setattr(
        win32security, "SetSecurityInfo", lambda *a, **k: changed.append("set") or real_set(*a, **k)
    )
    # The owner is somebody else by the time the owner decides.
    monkeypatch.setattr(
        fa.WindowsPermissions, "_owner", staticmethod(lambda ws, handle: "S-1-5-32-544")
    )
    with pytest.raises(fa.StaleEntry, match="changed since"):
        backend.decide(str(root), delta, found["fingerprint"], "remove")
    assert changed == []


# --- Re-review 3: the link count and the state are read when Remove is applied ------


def test_remove_on_an_entry_never_changes_a_file_that_gained_another_name(
    tmp_path: Path,
) -> None:
    """Re-review 3 (blocker): an ordinary entry prompt, then an outside hard link,
    then Remove: the ACE went through both names. The link count is read on the
    handle that would change it; such a file is asked about again as a hard link."""
    root = shared_root(tmp_path)
    (root / "seed.txt").write_text("x")
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "read", lambda _: None)
    icacls(root / "seed.txt", "/inheritance:r", "/grant:r", f"*{owner(root)}:(F)", f"*{USERS}:(F)")
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    (found,) = asked.value.entries
    assert found["kind"] == "entry"
    alias = tmp_path / "alias.txt"
    os.link(root / "seed.txt", alias)
    with pytest.raises(fa.StaleEntry, match="another name"):
        backend.decide(str(root), delta, found["fingerprint"], "remove")
    assert sid_aces(root / "seed.txt") == [(0, 0x1F01FF)] == sid_aces(alias)
    with pytest.raises(fa.UnknownAccess) as again:
        backend.revoke(str(root), delta)
    assert [(e["kind"], e["fingerprint"]["links"]) for e in again.value.entries] == [
        ("hardlink", 2)
    ]


def test_a_hard_link_prompt_carries_the_files_state(tmp_path: Path) -> None:
    """Re-review 3: a hard-link prompt had no owner/DACL fingerprint, so an answer
    given after the file changed still let removal change it."""
    root = shared_root(tmp_path)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    made = root / "made.txt"
    made.write_text("app data")
    os.link(made, tmp_path / "alias.txt")
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    (found,) = asked.value.entries
    assert found["fingerprint"]["state"] and found["fingerprint"]["links"] == 2
    icacls(made, "/grant", f"*{USERS}:(RX)")  # changed after the prompt
    with pytest.raises(fa.StaleEntry, match="changed since"):
        backend.decide(str(root), delta, found["fingerprint"], "remove")
    assert not delta.get("allowed")


def test_an_allowed_file_is_asked_again_when_it_gains_another_name(tmp_path: Path) -> None:
    root = shared_root(tmp_path)
    backend = fa.WindowsPermissions(USERS, eugene=(), homes=())
    delta, _, _ = backend.grant(str(root), "write", lambda _: None)
    made = root / "made.txt"
    made.write_text("app data")
    os.link(made, tmp_path / "alias.txt")
    with pytest.raises(fa.UnknownAccess) as asked:
        backend.revoke(str(root), delta)
    backend.decide(str(root), delta, asked.value.entries[0]["fingerprint"], "remove")
    granted = sid_aces(made)
    os.link(made, tmp_path / "second-alias.txt")  # one name more than the owner allowed
    with pytest.raises(fa.UnknownAccess) as again:
        backend.revoke(str(root), delta)
    assert [(e["kind"], e["fingerprint"]["links"]) for e in again.value.entries] == [
        ("hardlink", 3)
    ]
    assert sid_aces(made) == granted
