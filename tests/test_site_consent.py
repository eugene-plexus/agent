"""J9's proof for commands (2b.4, J30, J89): an administrator's consent,
recorded only elevated, in the protected list beside the local servers;
asked at the join, given later from the tray or the CLI, and kept when the
list's servers change."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from eugene_plexus_agent import site_cli, site_host, tray


def saved(config_dir: Path) -> dict[str, object]:
    return dict(yaml.safe_load(site_host.servers_path(config_dir).read_text(encoding="utf-8")))


def test_only_an_administrator_allows_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "_system_install", lambda _: False)
    monkeypatch.setattr(site_cli, "elevated", lambda: False)
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.consent(tmp_path, allow=True)
    assert not site_host.servers_path(tmp_path).exists()


def test_consent_is_recorded_kept_by_server_changes_and_taken_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "_system_install", lambda _: False)
    monkeypatch.setattr(site_cli, "elevated", lambda: True)
    said = site_cli.consent(tmp_path, allow=True)
    assert "may run on this machine now" in said
    value = saved(tmp_path)
    assert value["servers"] == [] and isinstance(value["commands"], dict)
    assert value["commands"]["consentedAt"]
    # The site host reads it as the contract says.
    assert site_host.local_servers(tmp_path) == []
    exe = tmp_path / "server.exe"
    exe.write_bytes(b"a program")
    site_cli.add_server(
        tmp_path, server_id="notes", name="Notes", command=str(exe), args=[], env=[], system=False
    )
    assert "commands" in saved(tmp_path) and len(saved(tmp_path)["servers"]) == 1
    site_cli.remove_server(tmp_path, "notes")
    assert "commands" in saved(tmp_path)
    assert "no longer" in site_cli.consent(tmp_path, allow=False)
    assert "commands" not in saved(tmp_path)


@pytest.mark.skipif(sys.platform != "linux", reason="a Linux system install's list is root's")
def test_a_linux_system_install_points_at_the_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "_system_install", lambda _: True)
    with pytest.raises(site_cli.SiteError, match="--site-commands"):
        site_cli.consent(tmp_path, allow=True)


def test_the_join_asks_only_an_administrator_and_records_a_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "_system_install", lambda _: False)
    monkeypatch.setattr(site_cli, "elevated", lambda: False)
    assert "until an administrator allows them" in site_cli._join_consent(tmp_path, None)
    monkeypatch.setattr(site_cli, "elevated", lambda: True)
    # Nobody at a terminal, and no answer given ahead: nothing is allowed.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    assert "do not run here" in site_cli._join_consent(tmp_path, None)
    assert not site_host.servers_path(tmp_path).exists()
    # At a terminal, "y" allows.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert "may run on this machine now" in site_cli._join_consent(tmp_path, None)
    assert "commands" in saved(tmp_path)


def test_answered_ahead_the_join_asks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "_system_install", lambda _: False)
    monkeypatch.setattr(site_cli, "elevated", lambda: True)

    def never(_prompt: str) -> str:
        raise AssertionError("asked although answered")

    monkeypatch.setattr("builtins.input", never)
    assert "do not run here" in site_cli._join_consent(tmp_path, False)
    assert "may run" in site_cli._join_consent(tmp_path, True)


def test_the_tray_offers_commands_only_on_a_job_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = [label for _, label, _ in tray.menu_for("running")]
    assert not any("commands" in label for label in labels)
    offered = [label for _, label, _ in tray.menu_for("running", job_site=True)]
    assert "Allow commands from Workbench..." in offered
    config = tmp_path / "agent.yaml"
    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_CONFIG_FILE", str(config))
    assert tray.job_site_config() is None
    (tmp_path / "site").mkdir()
    assert tray.job_site_config() == str(config)
    program, arguments = tray.consent_command(str(config))
    assert program.endswith("eugene-plexus-agent.exe")
    assert arguments == f'site consent --config-file "{config}"'
