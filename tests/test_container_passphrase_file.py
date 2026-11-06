"""In the container, the agent unlocks from the control root's passphrase file (agent#43).

Found on the Unraid root, 2026-11-06: an update restarted the container, the
control root unlocked itself from `/data/passphrase` (the console worked), and
the agent stayed locked. The image and the Unraid template set the root's
variable alone (`EUGENE_PLEXUS_CONTROL_PASSPHRASE_FILE`), so an agent on
`securityMode: passphrase_file` had nowhere to read: every driver started
without the master key, and an OpenRouter account's saved API key read as
unset until someone signed in again. A person who gave Eugene a passphrase
file must never have to sign in to get service back (Troy, 2026-11-06).

The agent now reads that file at boot when it has none of its own: the same
file the system installs give both. It is the operator's file, so it is read
and never written.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.routes.auth import _operators_file_state
from eugene_plexus_agent.settings import Settings
from tests.conftest import TEST_PASSPHRASE, StubSupervisor

ROOT_VARIABLE = "EUGENE_PLEXUS_CONTROL_PASSPHRASE_FILE"


def _container(
    tmp_path: Path, *, mode: str = "passphrase_file", own: Path | None = None
) -> Settings:
    """The container's agent: its config on /data, the root's variable set, its own unset."""
    (tmp_path / "agent.yaml").write_text(f"securityMode: {mode}\n", encoding="utf-8")
    return Settings(
        config_file=tmp_path / "agent.yaml",
        passphrase_file=own,
        **{ROOT_VARIABLE: str(tmp_path / "passphrase")},
    )


def _app(settings: Settings) -> FastAPI:
    app = create_app(settings=settings)
    app.state.supervisor = StubSupervisor()
    return app


def _initialize(settings: Settings) -> None:
    with TestClient(_app(settings)) as c:
        assert (
            c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
        )


def _unlocked_after_restart(settings: Settings) -> bool:
    app = _app(settings)
    with TestClient(app) as c:
        return (
            bool(c.get("/v1/auth/status").json()["unlocked"])
            and app.state.auth_state.has_master_key()
        )


def test_the_roots_variable_is_read_as_the_container_sets_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE", raising=False)
    monkeypatch.setenv(ROOT_VARIABLE, "/data/passphrase")
    assert Settings().root_passphrase_file == Path("/data/passphrase")
    assert Settings().passphrase_file is None
    monkeypatch.setenv(ROOT_VARIABLE, "  ")  # Unraid passes blank variables
    assert Settings().root_passphrase_file is None


@pytest.mark.parametrize("written", [b"%s", b"%s\n"], ids=["printf", "echo"])
def test_a_restart_unlocks_from_the_roots_file_with_nobody_signing_in(
    tmp_path: Path, written: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _container(tmp_path)
    _initialize(settings)
    root = tmp_path / "passphrase"
    assert not root.exists(), "initialize wrote the operator's file"
    root.write_bytes(written.replace(b"%s", TEST_PASSPHRASE.encode()))
    caplog.set_level(logging.INFO, logger="eugene_plexus_agent.app")
    assert _unlocked_after_restart(settings)
    assert any(
        "recovered from the passphrase file" in r.getMessage() and str(root) in r.getMessage()
        for r in caplog.records
    )


def test_signing_in_never_writes_the_operators_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _container(tmp_path)
    _initialize(settings)
    root = tmp_path / "passphrase"
    caplog.set_level(logging.WARNING, logger="eugene_plexus_agent.passphrase_file")
    with TestClient(_app(settings)) as c:
        assert c.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
    assert not root.exists(), "a sign-in created the operator's file"
    # Settings never lie: a restart reads the operator's file, so it is not "locked after".
    assert not any("next restart" in r.getMessage() for r in caplog.records)
    root.write_bytes(TEST_PASSPHRASE.encode() + b"\n")
    with TestClient(_app(settings)) as c:
        assert c.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
    assert root.read_bytes() == TEST_PASSPHRASE.encode() + b"\n", "a sign-in rewrote it"


def test_a_roots_file_holding_another_passphrase_does_not_unlock(tmp_path: Path) -> None:
    settings = _container(tmp_path)
    _initialize(settings)
    (tmp_path / "passphrase").write_text("somebody else's phrase", encoding="utf-8")
    assert not _unlocked_after_restart(settings)


def test_the_agents_own_file_wins(tmp_path: Path) -> None:
    own = tmp_path / "own-passphrase"
    settings = _container(tmp_path, own=own)
    _initialize(settings)
    assert own.read_text(encoding="utf-8") == TEST_PASSPHRASE
    (tmp_path / "passphrase").write_text("somebody else's phrase", encoding="utf-8")
    assert _unlocked_after_restart(settings)


def test_the_other_modes_do_not_unlock_from_the_roots_file(tmp_path: Path) -> None:
    """The agent's own mode decides, as it does for its own file: `prompt_on_startup`
    was chosen to ask, and a file beside it is not an invitation."""
    settings = _container(tmp_path, mode="prompt_on_startup")
    _initialize(settings)
    (tmp_path / "passphrase").write_text(TEST_PASSPHRASE, encoding="utf-8")
    assert not _unlocked_after_restart(settings)


def test_the_test_button_names_the_operators_file(tmp_path: Path) -> None:
    settings = _container(tmp_path)
    with TestClient(_app(settings)) as c:
        token = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).json()
        c.headers["Authorization"] = f"Bearer {token['sessionToken']}"
        missing = c.post("/v1/config/test").json()
        (tmp_path / "passphrase").write_text(TEST_PASSPHRASE, encoding="utf-8")
        present = c.post("/v1/config/test").json()
    assert missing["ok"] is False and "does not exist" in (missing.get("error") or ""), missing
    said = present.get("summary") or ""
    assert present["ok"] is True and "unlocks itself on restart from" in said, present
    assert "never writes it" in said


def test_a_passphrase_change_says_the_operator_must_update_the_file(tmp_path: Path) -> None:
    root = tmp_path / "passphrase"
    root.write_text("old phrase\n", encoding="utf-8")
    state, why = _operators_file_state(root, "new phrase")
    assert state == "old" and why is not None and str(root) in why and "never" in why
    root.write_text("new phrase\n", encoding="utf-8")
    assert _operators_file_state(root, "new phrase") == ("matches", None)
