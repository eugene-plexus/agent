"""Who is who on a machine, for its Job Site (job-sites-own-enrollment.md §2.2,
§3.2): the links file and the rules that keep a link honest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eugene_plexus_agent import site_links
from eugene_plexus_agent.site_links import LinkError, LinkStore, not_a_person

ADA = "S-1-5-21-1-2-3-1001"
BO = "S-1-5-21-1-2-3-1002"


def add(store: LinkStore, subject: str, account: str, **kwargs: object) -> site_links.Link:
    return store.add(
        subject=subject,
        name=kwargs.pop("name", subject),  # type: ignore[arg-type]
        account=account,
        account_name=kwargs.pop("account_name", f"PC\\{account[-4:]}"),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_link_is_stored_and_found_both_ways(tmp_path: Path) -> None:
    store = LinkStore(tmp_path)
    assert store.load() == []
    link = add(store, "p-ada", ADA, name="Ada")
    assert store.path == tmp_path / "site" / "links.json"
    assert store.for_subject("p-ada") == link and store.for_account(ADA) == link
    assert store.for_subject("p-bo") is None and store.for_account(BO) is None
    add(store, "p-bo", BO)
    assert [x.subject for x in LinkStore(tmp_path).load()] == ["p-ada", "p-bo"]
    saved = json.loads(store.path.read_text(encoding="utf-8"))
    assert saved["version"] == 1 and saved["links"][0]["accountName"] == link.account_name
    assert saved["links"][0]["account"] == ADA and saved["links"][0]["name"] == "Ada"


def test_removing_a_link_returns_it_and_keeps_the_others(tmp_path: Path) -> None:
    store = LinkStore(tmp_path)
    add(store, "p-ada", ADA)
    add(store, "p-bo", BO)
    gone = store.remove("p-ada")
    assert gone is not None and gone.account == ADA
    assert [x.subject for x in store.load()] == ["p-bo"]
    assert store.remove("p-ada") is None, "nothing to remove the second time"
    assert [x.subject for x in store.load()] == ["p-bo"]


def test_one_link_per_person(tmp_path: Path) -> None:
    store = LinkStore(tmp_path)
    add(store, "p-ada", ADA, account_name="PC\\ada")
    with pytest.raises(LinkError, match=r"already linked to PC\\ada"):
        add(store, "p-ada", BO)
    assert [x.account for x in store.load()] == [ADA]


def test_one_person_per_account(tmp_path: Path) -> None:
    store = LinkStore(tmp_path)
    add(store, "p-ada", ADA, name="Ada")
    with pytest.raises(LinkError, match="already linked to Ada"):
        add(store, "p-bo", ADA)
    assert [x.subject for x in store.load()] == ["p-ada"]


def test_an_account_linked_to_someone_without_a_name_still_says_so(tmp_path: Path) -> None:
    store = LinkStore(tmp_path)
    store.add(subject="p-ada", name=None, account=ADA, account_name="PC\\ada")
    with pytest.raises(LinkError, match="already linked to someone else"):
        add(store, "p-bo", ADA)


def test_the_same_pair_twice_is_no_change(tmp_path: Path) -> None:
    store = LinkStore(tmp_path)
    first = add(store, "p-ada", ADA)
    before = store.path.read_bytes()
    again = add(store, "p-ada", ADA)
    assert again == first and store.path.read_bytes() == before
    assert len(store.load()) == 1


@pytest.mark.parametrize(
    ("account", "why"),
    [
        ("S-1-5-18", "system account"),  # LocalSystem
        ("S-1-5-19", "system account"),  # Local Service
        ("S-1-5-20", "system account"),  # Network Service
        ("S-1-5-80-1234-5", "service or virtual"),  # NT SERVICE\x
        ("S-1-5-82-1234-5", "service or virtual"),  # IIS app pool
        ("S-1-5-90-0-3", "service or virtual"),  # Window Manager
        ("S-1-5-96-0-3", "service or virtual"),  # Font Driver Host
        ("S-1-5-32-544", "not a person"),  # Administrators (a group)
        ("S-1-1-0", "not a person"),  # Everyone
        ("0", "root"),
        ("65534", "system account"),
        ("999", "system account"),
        ("1", "system account"),
        ("nobody", "not an account"),
    ],
)
def test_accounts_that_are_never_a_persons(tmp_path: Path, account: str, why: str) -> None:
    assert why in (not_a_person(account) or "")
    store = LinkStore(tmp_path)
    with pytest.raises(LinkError, match="no one can be linked"):
        add(store, "p-ada", account)
    assert not store.path.exists(), "a refused link writes nothing"


def test_people_accounts_are_accepted() -> None:
    assert not_a_person(ADA) is None
    assert not_a_person("S-1-12-1-111-222-333-444") is None  # an Entra account
    assert not_a_person("1000") is None


def test_the_macos_floor_is_five_hundred(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(site_links.sys, "platform", "darwin")
    assert not_a_person("501") is None and not_a_person("499") is not None
    monkeypatch.setattr(site_links.sys, "platform", "linux")
    assert not_a_person("501") is not None


def test_the_never_set_names_eugenes_own_accounts(tmp_path: Path) -> None:
    store = LinkStore(tmp_path)
    own = "S-1-5-21-1-2-3-1500"
    with pytest.raises(LinkError, match="Eugene's own accounts"):
        add(store, "p-ada", own, never=frozenset({own}))
    assert not store.path.exists()
    add(store, "p-ada", ADA, never=frozenset({own}))  # others are unaffected


def test_a_write_replaces_the_file_whole_and_leaves_no_temporary(tmp_path: Path) -> None:
    store = LinkStore(tmp_path)
    for n in range(4):
        add(store, f"p-{n}", f"S-1-5-21-1-2-3-{2000 + n}")
    store.remove("p-1")
    assert sorted(p.name for p in store.dir.iterdir()) == ["links.json"]
    assert len(json.loads(store.path.read_text(encoding="utf-8"))["links"]) == 3


def test_a_failed_write_leaves_the_old_file_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LinkStore(tmp_path)
    add(store, "p-ada", ADA)
    before = store.path.read_bytes()

    def broken(source: object, target: object) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(site_links.os, "replace", broken)
    with pytest.raises(OSError, match="disk gone"):
        add(store, "p-bo", BO)
    assert store.path.read_bytes() == before


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"[]", b'"links"', b'{"links": 5}', b"", b"\xff\xfe\x00"],
)
def test_a_file_that_does_not_read_is_no_links(tmp_path: Path, content: bytes) -> None:
    store = LinkStore(tmp_path)
    store.dir.mkdir(parents=True)
    store.path.write_bytes(content)
    assert store.load() == []
    assert store.for_account(ADA) is None
    # And a new link can be made over the wreck.
    add(store, "p-ada", ADA)
    assert [x.subject for x in store.load()] == ["p-ada"]


def test_one_bad_entry_does_not_hide_the_rest(tmp_path: Path) -> None:
    store = LinkStore(tmp_path)
    store.dir.mkdir(parents=True)
    good = {
        "subject": "p-ada",
        "name": "Ada",
        "account": ADA,
        "accountName": "PC\\ada",
        "linkedAt": "2026-10-06T00:00:00+00:00",
    }
    store.path.write_text(json.dumps({"version": 1, "links": [{"subject": "x"}, good, 7]}), "utf-8")
    assert [x.subject for x in store.load()] == ["p-ada"]


def test_protect_windows_grants_what_the_design_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The folder: SYSTEM and Administrators write, the site host reads. The
    server list: each linked account reads it too. Nothing touches the disk
    ACLs here: the icacls call is recorded."""
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(site_links, "_icacls", lambda path, *args: calls.append((path.name, *args)))
    store = LinkStore(tmp_path)
    link = add(store, "p-ada", ADA)
    site_links.protect_windows(tmp_path, [link], site_host_exists=True)
    assert calls == [
        (
            "site",
            "/inheritance:r",
            "/grant:r",
            "*S-1-5-18:(OI)(CI)F",
            "*S-1-5-32-544:(OI)(CI)F",
            f"{site_links.SITE_HOST_ACCOUNT}:(OI)(CI)RX",
        )
    ]
    calls.clear()
    (tmp_path / "site" / "servers.yaml").write_text("servers: []\n", encoding="utf-8")
    site_links.protect_windows(tmp_path, [link], site_host_exists=False)
    assert calls[0][-1] == "*S-1-5-32-544:(OI)(CI)F", "no host account yet, so no grant for it"
    assert calls[1] == ("servers.yaml", "/reset")
    assert calls[2] == ("servers.yaml", "/grant:r", f"*{ADA}:R")


def test_account_name_falls_back_to_the_id_it_was_given() -> None:
    assert site_links.account_name("S-1-5-21-0-0-0-424242") == "S-1-5-21-0-0-0-424242"


def test_an_unknown_account_name_is_a_sentence() -> None:
    with pytest.raises(LinkError, match="no account named"):
        site_links.account_sid("no-such-account-eugene-test")
