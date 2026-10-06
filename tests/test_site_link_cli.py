"""`eugene-plexus-agent site join | link | unlink | leave`: who is linked to which
account on this machine (job-sites-own-enrollment.md §2.2, §3.2, J36, J38).

The site host's own commands (`join`, `check-person`, `leave`) are faked at the
subprocess boundary: what is held here is the agent's side, the owner link and
the rules on who may do what."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eugene_plexus_agent import site_cli, site_host
from eugene_plexus_agent.site_links import LinkError, LinkStore

_REAL_NEVER = site_cli._never  # the autouse fixture replaces it for every test

ADA = "S-1-5-21-1-2-3-1001"
BO = "S-1-5-21-1-2-3-1002"
NAMES = {ADA: "PC\\ada", BO: "PC\\bo", "S-1-5-18": "NT AUTHORITY\\SYSTEM"}


@pytest.fixture(autouse=True)
def machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Who is running this, and what the accounts are called: none of it read
    from the machine the test runs on."""
    monkeypatch.setattr(site_cli, "_own_sid", lambda: ADA)
    monkeypatch.setattr(site_cli, "_console_sid", lambda: None)
    monkeypatch.setattr(site_cli, "account_name", lambda sid: NAMES.get(sid, sid))
    monkeypatch.setattr(site_cli, "_never", lambda: frozenset({"S-1-5-18"}))

    def sid(name: str) -> str:
        for key, value in NAMES.items():
            if value.lower() == name.lower():
                return key
        raise LinkError(f"There is no account named {name} on this machine.")

    monkeypatch.setattr(site_cli, "account_sid", sid)
    monkeypatch.setattr(site_cli, "elevated", lambda: True)


def host(tmp_path: Path, owner: dict[str, str] | None = None) -> tuple[Path, Path]:
    """A prepared site host: its interpreter and its data directory."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    if owner is not None:
        (data / "site.json").write_text(json.dumps(owner), encoding="utf-8")
    return Path(sys.executable), data


def join_args(tmp_path: Path, **changes: Any) -> argparse.Namespace:
    python, data = host(tmp_path)
    values: dict[str, Any] = {
        "url": "https://nodes.example.test",
        "token": "invitation",
        "owner": "ada",
        "label": "desk",
        "root_key": None,
        "password_stdin": True,
        "python": str(python),
        "data_dir": str(data),
        "site_account": None,
    }
    return argparse.Namespace(**{**values, **changes})


def fake_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, owner: dict[str, str] | None = None
) -> list[tuple[list[str], str | None]]:
    """The site host's side: a join that writes `site.json`, a check-person
    that knows one password, a leave."""
    ran: list[tuple[list[str], str | None]] = []
    _, data = host(tmp_path)

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        ran.append((command, kwargs.get("input")))
        verb = command[command.index("eugene_plexus_site_host") + 1]
        if verb == "join":
            (data / "site.json").write_text(
                json.dumps(owner or {"owner": "p-ada", "ownerName": "Ada"}), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, "Joined as ada's job site desk.\n", "")
        if verb == "check-person":
            if kwargs.get("input") != "right\n":
                return subprocess.CompletedProcess(command, 1, "", "That password is wrong.")
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"subject": "p-bo", "name": "Bo"}), ""
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(site_cli.subprocess, "run", run)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(readline=lambda: "right\n"))
    return ran


def links(tmp_path: Path) -> list[tuple[str, str]]:
    return [(x.subject, x.account) for x in LinkStore(tmp_path).load()]


# --- join links the owner -----------------------------------------------------------------


def test_join_links_the_owner_to_the_account_that_ran_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_host(monkeypatch, tmp_path)
    said = site_cli.join(tmp_path, join_args(tmp_path))
    assert "Joined as ada's job site" in said and "Ada's calls here run as PC\\ada" in said
    assert links(tmp_path) == [("p-ada", ADA)]
    (link,) = LinkStore(tmp_path).load()
    assert link.name == "Ada" and link.account_name == "PC\\ada"
    assert site_host.wanted(tmp_path) is True


def test_site_account_names_another_account_for_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_host(monkeypatch, tmp_path)
    said = site_cli.join(tmp_path, join_args(tmp_path, site_account="PC\\bo"))
    assert links(tmp_path) == [("p-ada", BO)] and "run as PC\\bo" in said


def test_a_site_account_that_does_not_exist_leaves_the_join_done_and_the_owner_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_host(monkeypatch, tmp_path)
    said = site_cli.join(tmp_path, join_args(tmp_path, site_account="PC\\nobody"))
    assert "Joined as" in said and "not linked to an account here" in said
    assert "no account named PC\\nobody" in said and links(tmp_path) == []
    assert site_host.wanted(tmp_path) is True


def test_system_running_join_links_the_console_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(site_cli, "_own_sid", lambda: "S-1-5-18")
    monkeypatch.setattr(site_cli, "_console_sid", lambda: BO)
    site_cli.join(tmp_path, join_args(tmp_path))
    assert links(tmp_path) == [("p-ada", BO)], "never SYSTEM, whose calls would run as SYSTEM"


def test_system_with_nobody_at_the_console_links_nobody_and_says_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(site_cli, "_own_sid", lambda: "S-1-5-18")
    said = site_cli.join(tmp_path, join_args(tmp_path))
    assert "Nobody is signed in" in said and "--site-account" in said
    assert links(tmp_path) == []


def test_even_a_named_system_account_is_not_linked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_host(monkeypatch, tmp_path)
    said = site_cli.join(tmp_path, join_args(tmp_path, site_account="NT AUTHORITY\\SYSTEM"))
    assert "system account" in said and links(tmp_path) == []


def test_root_with_sudo_links_the_person_who_ran_sudo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posix = SimpleNamespace(platform="linux", stdin=sys.stdin)
    monkeypatch.setattr(site_cli, "sys", posix)
    monkeypatch.setattr(site_cli, "_own_sid", lambda: "0")
    monkeypatch.setenv("SUDO_UID", "1001")
    assert site_cli._owner_account(argparse.Namespace(site_account=None)) == "1001"
    monkeypatch.delenv("SUDO_UID")
    with pytest.raises(site_cli.SiteError, match="--site-account"):
        site_cli._owner_account(argparse.Namespace(site_account=None))
    monkeypatch.setattr(site_cli, "_own_sid", lambda: "1001")
    assert site_cli._owner_account(argparse.Namespace(site_account=None)) == "1001"


def test_a_site_json_that_cannot_be_read_links_nobody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = fake_host(monkeypatch, tmp_path)
    original = site_cli.subprocess.run

    def join_without_writing(command: list[str], **kwargs: Any) -> Any:
        done = original(command, **kwargs)
        (tmp_path / "data" / "site.json").unlink()
        return done

    monkeypatch.setattr(site_cli.subprocess, "run", join_without_writing)
    said = site_cli.join(tmp_path, join_args(tmp_path))
    assert "could not be linked" in said and links(tmp_path) == [] and ran


# --- who needs to be an administrator ------------------------------------------------------


def test_a_per_users_join_needs_no_elevation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(site_cli, "elevated", lambda: False)
    assert not site_cli._system_install(tmp_path)
    site_cli.join(tmp_path, join_args(tmp_path))
    assert site_host.wanted(tmp_path) is True and links(tmp_path) == [("p-ada", ADA)]
    assert "no longer a job site" in site_cli.leave(tmp_path, join_args(tmp_path))
    assert site_host.wanted(tmp_path) is False


@pytest.mark.skipif(sys.platform != "win32", reason="ProgramData is Windows's")
def test_an_install_under_programdata_is_the_machines_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program_data = tmp_path / "ProgramData"
    monkeypatch.setenv("PROGRAMDATA", str(program_data))
    assert site_cli._system_install(program_data / "EugenePlexus" / "config")
    assert not site_cli._system_install(tmp_path / "LocalAppData" / "EugenePlexus")
    monkeypatch.setattr(site_cli, "elevated", lambda: False)
    fake_args = join_args(tmp_path)
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.join(program_data / "EugenePlexus" / "config", fake_args)
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.leave(program_data / "EugenePlexus" / "config", fake_args)


def test_a_linux_system_install_leaves_join_and_link_to_the_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(site_cli, "sys", SimpleNamespace(platform="linux", stdin=sys.stdin))
    monkeypatch.setattr(site_cli, "_system_install", lambda config_dir: True)
    with pytest.raises(site_cli.SiteError, match="--job-site"):
        site_cli.join(tmp_path, join_args(tmp_path))
    with pytest.raises(site_cli.SiteError, match="--site-link"):
        site_cli.link(tmp_path, join_args(tmp_path, person="Bo", account=None))
    assert ran == [] and site_host.wanted(tmp_path) is False and links(tmp_path) == []


# --- link and unlink ------------------------------------------------------------------------


def link_args(tmp_path: Path, **changes: Any) -> argparse.Namespace:
    values: dict[str, Any] = {"person": "Bo", "account": "PC\\bo", "password_stdin": True}
    python, data = host(tmp_path, {"owner": "p-ada"})
    values.update(python=str(python), data_dir=str(data))
    return argparse.Namespace(**{**values, **changes})


def test_link_checks_the_persons_password_with_the_site_host_and_writes_the_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = fake_host(monkeypatch, tmp_path)
    said = site_cli.link(tmp_path, link_args(tmp_path))
    assert "Bo's calls here now run as PC\\bo" in said
    ((command, given),) = ran
    assert command[command.index("eugene_plexus_site_host") + 1] == "check-person"
    assert command[command.index("--name") + 1] == "Bo"
    assert command[command.index("--data-dir") + 1] == str(tmp_path / "data")
    assert given == "right\n" and "right" not in command
    assert links(tmp_path) == [("p-bo", BO)]


def test_link_without_account_names_the_one_running_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_host(monkeypatch, tmp_path)
    site_cli.link(tmp_path, link_args(tmp_path, account=None))
    assert links(tmp_path) == [("p-bo", ADA)]


def test_a_wrong_password_links_nobody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(readline=lambda: "wrong\n"))
    with pytest.raises(site_cli.SiteError, match="password is wrong"):
        site_cli.link(tmp_path, link_args(tmp_path))
    assert links(tmp_path) == []


def test_link_asks_at_the_terminal_without_password_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = fake_host(monkeypatch, tmp_path)
    asked: list[str] = []
    monkeypatch.setattr(site_cli.getpass, "getpass", lambda prompt: asked.append(prompt) or "right")
    site_cli.link(tmp_path, link_args(tmp_path, password_stdin=False))
    assert asked and "Bo" in asked[0] and ran[0][1] == "right\n"


def test_link_needs_an_administrator_and_a_job_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(site_cli, "elevated", lambda: False)
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.link(tmp_path, link_args(tmp_path))
    monkeypatch.setattr(site_cli, "elevated", lambda: True)
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(site_cli.SiteError, match="not a job site"):
        site_cli.link(tmp_path, link_args(tmp_path, data_dir=str(bare)))
    assert ran == [] and links(tmp_path) == []


def test_link_refuses_what_the_rules_refuse_in_the_rules_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_host(monkeypatch, tmp_path)
    site_cli.link(tmp_path, link_args(tmp_path))
    with pytest.raises(site_cli.SiteError, match="already linked to Bo"):
        monkeypatch.setattr(
            site_cli.subprocess,
            "run",
            lambda command, **k: subprocess.CompletedProcess(
                command, 0, json.dumps({"subject": "p-cy", "name": "Cy"}), ""
            ),
        )
        site_cli.link(tmp_path, link_args(tmp_path, person="Cy"))
    with pytest.raises(site_cli.SiteError, match="system account"):
        site_cli.link(tmp_path, link_args(tmp_path, person="Cy", account="NT AUTHORITY\\SYSTEM"))
    assert links(tmp_path) == [("p-bo", BO)]


def test_unlink_removes_by_name_or_by_subject(tmp_path: Path) -> None:
    store = LinkStore(tmp_path)
    store.add(subject="p-ada", name="Ada", account=ADA, account_name="PC\\ada")
    store.add(subject="p-bo", name="Bo", account=BO, account_name="PC\\bo")
    assert "Bo is no longer linked to PC\\bo" in site_cli.unlink(tmp_path, "Bo")
    assert links(tmp_path) == [("p-ada", ADA)]
    assert "no longer linked" in site_cli.unlink(tmp_path, "p-ada")
    assert links(tmp_path) == []
    with pytest.raises(site_cli.SiteError, match="Nobody called Bo"):
        site_cli.unlink(tmp_path, "Bo")


def test_unlink_needs_an_administrator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    LinkStore(tmp_path).add(subject="p-ada", name="Ada", account=ADA, account_name="PC\\ada")
    monkeypatch.setattr(site_cli, "elevated", lambda: False)
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.unlink(tmp_path, "Ada")
    assert links(tmp_path) == [("p-ada", ADA)]


def test_leaving_removes_every_link_and_turns_the_host_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = fake_host(monkeypatch, tmp_path)
    store = LinkStore(tmp_path)
    store.add(subject="p-ada", name="Ada", account=ADA, account_name="PC\\ada")
    store.add(subject="p-bo", name="Bo", account=BO, account_name="PC\\bo")
    site_host.set_wanted(tmp_path, True)
    said = site_cli.leave(tmp_path, join_args(tmp_path))
    assert "no longer a job site" in said
    assert links(tmp_path) == [] and site_host.wanted(tmp_path) is False
    ((command, _),) = ran
    assert command[command.index("eugene_plexus_site_host") + 1] == "leave"


def test_a_refused_leave_keeps_the_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host(monkeypatch, tmp_path)
    LinkStore(tmp_path).add(subject="p-ada", name="Ada", account=ADA, account_name="PC\\ada")
    monkeypatch.setattr(
        site_cli.subprocess,
        "run",
        lambda command, **k: subprocess.CompletedProcess(command, 1, "", "root unreachable"),
    )
    with pytest.raises(site_cli.SiteError, match="root unreachable"):
        site_cli.leave(tmp_path, join_args(tmp_path))
    assert links(tmp_path) == [("p-ada", ADA)]


# --- the command line -----------------------------------------------------------------------


def parse(*argv: str) -> argparse.Namespace:
    top = argparse.ArgumentParser()
    site_cli.add_parser(top.add_subparsers(dest="command", required=True))
    return top.parse_args(["site", *argv])


def test_the_new_commands_parse() -> None:
    link = parse("link", "--person", "Bo", "--account", "PC\\bo", "--password-stdin")
    assert (link.site_command, link.person, link.account, link.password_stdin) == (
        "link",
        "Bo",
        "PC\\bo",
        True,
    )
    assert parse("unlink", "--person", "Bo").person == "Bo"
    joined = parse(
        "join", "--url", "u", "--token", "t", "--owner", "o", "--label", "l", "--site-account", "x"
    )
    assert joined.site_account == "x"


def test_run_prints_a_refusal_and_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = SimpleNamespace(config_file=tmp_path / "agent.yaml")
    assert site_cli.run(parse("unlink", "--person", "Nobody"), settings) == 2  # type: ignore[arg-type]
    assert "Nobody called Nobody" in capsys.readouterr().err
    LinkStore(tmp_path).add(subject="p-ada", name="Ada", account=ADA, account_name="PC\\ada")
    assert site_cli.run(parse("unlink", "--person", "Ada"), settings) == 0  # type: ignore[arg-type]
    assert os.linesep.join(["Ada is no longer linked to PC\\ada."]) in capsys.readouterr().out


def test_the_real_list_of_eugenes_own_accounts_names_system_and_the_site_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_sid = "S-1-5-80-111"
    monkeypatch.setattr(site_cli, "account_sid", lambda name: host_sid)
    assert _REAL_NEVER() == frozenset({"S-1-5-18", host_sid})

    # A machine where the site host's account does not exist yet still names SYSTEM.
    def missing(name: str) -> str:
        raise LinkError("none")

    monkeypatch.setattr(site_cli, "account_sid", missing)
    assert _REAL_NEVER() == frozenset({"S-1-5-18"})


def test_a_linux_system_install_leaves_the_local_server_list_to_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "sys", SimpleNamespace(platform="linux", stdin=sys.stdin))
    monkeypatch.setattr(site_cli, "_system_install", lambda config_dir: True)
    exe = tmp_path / "server.exe"
    exe.write_bytes(b"MZ")
    with pytest.raises(site_cli.SiteError, match=r"/etc/eugene-plexus/site/servers.yaml"):
        site_cli.add_server(
            tmp_path, server_id="notes", name="N", command=str(exe), args=[], env=[], system=False
        )
    with pytest.raises(site_cli.SiteError, match=r"/etc/eugene-plexus/site/servers.yaml"):
        site_cli.remove_server(tmp_path, "notes")


def test_the_site_host_is_prepared_when_its_install_is_recorded_not_when_its_venv_appears(
    tmp_path: Path,
) -> None:
    """uv makes a venv's interpreter before it installs anything into it; the
    first Windows run's `site join` called that interpreter and found no site
    host. Only the version `apps.yaml` records is an install."""
    from datetime import UTC, datetime

    from eugene_plexus_agent.apps import APPS_FILE, AppOrigin, AppStore, InstalledApp, venv_python

    store = AppStore(tmp_path / APPS_FILE)
    want = site_host.manifest({})
    python = venv_python(store.version_dir(site_cli.HELPER_ID, want.version) / "venv")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    assert site_cli._host_python(tmp_path) is None
    store.put(
        InstalledApp(
            manifest=want, origin=AppOrigin.catalogue, port=8300, installed_at=datetime.now(UTC)
        )
    )
    assert site_cli._host_python(tmp_path) == python
    python.unlink()
    assert site_cli._host_python(tmp_path) is None
