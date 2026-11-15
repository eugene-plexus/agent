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


@pytest.mark.parametrize("content", [None, "somebody else's phrase"], ids=["missing", "wrong"])
def test_a_roots_file_that_cannot_unlock_says_the_operator_must_fix_it(
    tmp_path: Path, content: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    """A sign-in never writes the operator's file, so the log must not send them to one."""
    settings = _container(tmp_path)
    _initialize(settings)
    if content is not None:
        (tmp_path / "passphrase").write_text(content, encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="eugene_plexus_agent.passphrase_file")
    assert not _unlocked_after_restart(settings)
    said = [  # what says it stays locked (POSIX also warns of a widely readable file)
        r.getMessage()
        for r in caplog.records
        if str(tmp_path / "passphrase") in r.getMessage() and "locked" in r.getMessage()
    ]
    assert said and all("the agent never writes this file" in s for s in said), said
    assert not any("signs in" in s for s in said), said


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


def test_the_test_button_checks_the_operators_file_as_a_restart_reads_it(tmp_path: Path) -> None:
    """The operator wrote this file, not Eugene, so its existence proves nothing."""
    settings = _container(tmp_path)
    root = tmp_path / "passphrase"
    with TestClient(_app(settings)) as c:
        token = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).json()
        c.headers["Authorization"] = f"Bearer {token['sessionToken']}"
        missing = c.post("/v1/config/test").json()
        root.write_text("", encoding="utf-8")
        empty = c.post("/v1/config/test").json()
        root.write_text("somebody else's phrase", encoding="utf-8")
        wrong = c.post("/v1/config/test").json()
        root.write_text(TEST_PASSPHRASE + "\n", encoding="utf-8")
        right = c.post("/v1/config/test").json()
    for body, fragment in (
        (missing, "does not exist"),
        (empty, "cannot be read as a passphrase"),
        (wrong, "does not hold this install's passphrase"),
    ):
        assert body["ok"] is False and fragment in (body.get("error") or ""), body
    said = right.get("summary") or ""
    assert right["ok"] is True and "unlocks itself on restart from" in said, right
    assert "never writes it" in said


def test_a_passphrase_change_leaves_the_operators_file_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from tests.conftest import StubRuntimeSupervisor

    caplog.set_level(logging.WARNING, logger="eugene_plexus_agent.passphrase_file")
    (tmp_path / "agent.yaml").write_text("securityMode: passphrase_file\n", encoding="utf-8")
    settings = Settings(
        config_file=tmp_path / "agent.yaml",
        default_topology=False,
        **{ROOT_VARIABLE: str(tmp_path / "passphrase")},
    )
    root = tmp_path / "passphrase"
    app = create_app(settings=settings)
    app.state.supervisor = StubSupervisor()
    app.state.runtime_supervisor = StubRuntimeSupervisor()
    new = TEST_PASSPHRASE + "-changed"
    with TestClient(app) as c:
        token = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).json()
        c.headers["Authorization"] = f"Bearer {token['sessionToken']}"
        root.write_bytes(TEST_PASSPHRASE.encode() + b"\n")
        changed = c.post("/v1/auth/passphrase", json={"current": TEST_PASSPHRASE, "new": new})
        assert changed.status_code == 200, changed.text
    assert changed.json()["passphraseFile"] == "old", changed.json()
    assert root.read_bytes() == TEST_PASSPHRASE.encode() + b"\n", "the change wrote the file"
    # Not "nowhere to keep it": there is a file, and it is the operator's to update.
    assert not any("nowhere to keep" in r.getMessage() for r in caplog.records)


def test_an_agent_that_would_ask_beside_a_root_on_the_file_is_warned(
    authed_client: TestClient,
) -> None:
    """A container install set up by the old doc: the root on its file, this agent
    on prompt_on_startup. Its providers would wait for a sign-in after a restart."""
    settings = authed_client.app.state.settings  # type: ignore[attr-defined]
    settings.passphrase_file = None
    settings.root_passphrase_file = Path("/data/passphrase")
    assert (
        authed_client.patch("/v1/config", json={"securityMode": "prompt_on_startup"}).json()[
            "rejected"
        ]
        == []
    )
    body = authed_client.get("/v1/config/schema").json()
    status = next(f for f in body["fields"] if f["key"] == "securityMode")["status"]
    assert status["level"] == "warning" and "Passphrase file auto-unlock" in status["text"]
    assert "/data/passphrase" in status["text"].replace("\\", "/")


def test_a_passphrase_change_says_the_operator_must_update_the_file(tmp_path: Path) -> None:
    root = tmp_path / "passphrase"
    root.write_text("old phrase\n", encoding="utf-8")
    state, why = _operators_file_state(root, "new phrase")
    assert state == "old" and why is not None and str(root) in why and "never" in why
    root.write_text("new phrase\n", encoding="utf-8")
    assert _operators_file_state(root, "new phrase") == ("matches", None)


# --- agent#44: a fresh container install -----------------------------------


def _fresh_container(tmp_path: Path) -> Settings:
    """A container's first start: no agent config yet, the root's variable set."""
    return Settings(
        config_file=tmp_path / "agent.yaml",
        default_topology=False,
        **{ROOT_VARIABLE: str(tmp_path / "passphrase")},
    )


def test_a_fresh_container_tells_the_wizard_it_can_unlock_from_the_operators_file(
    tmp_path: Path,
) -> None:
    """Before agent#44 the flag needed the agent's own file, so the wizard wrote
    prompt_on_startup for the agent and its providers waited for a sign-in."""
    with TestClient(_app(_fresh_container(tmp_path))) as c:
        status = c.get("/v1/auth/status").json()
    assert status["initialized"] is False
    assert status["passphraseFile"] is True, status
    assert status["passphraseFileFromOperator"] is True, status
    # Unauthenticated: the operator's path is not in the answer.
    assert str(tmp_path) not in str(status)


def test_the_wizards_choice_unlocks_a_fresh_container_after_a_restart(tmp_path: Path) -> None:
    """What the wizard does with the flag: initialize, then passphrase_file on the agent.
    The operator's file holds the passphrase; nobody signs in after the restart."""
    settings = _fresh_container(tmp_path)
    root = tmp_path / "passphrase"
    root.write_bytes(TEST_PASSPHRASE.encode() + b"\n")
    with TestClient(_app(settings)) as c:
        token = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).json()
        c.headers["Authorization"] = f"Bearer {token['sessionToken']}"
        patched = c.patch("/v1/config", json={"securityMode": "passphrase_file"}).json()
        assert "securityMode" in patched["applied"], patched
    assert root.read_bytes() == TEST_PASSPHRASE.encode() + b"\n", "setup wrote the file"
    assert _unlocked_after_restart(settings)


def test_the_flag_stays_true_under_another_mode_with_the_operators_file(tmp_path: Path) -> None:
    """The wizard reads it before any mode is set, and a resume reads it after one was."""
    settings = _container(tmp_path, mode="prompt_on_startup")
    with TestClient(_app(settings)) as c:
        status = c.get("/v1/auth/status").json()
    assert status["passphraseFile"] is True and status["passphraseFileFromOperator"] is True


def test_the_agents_own_file_is_not_the_operators(tmp_path: Path) -> None:
    """The Linux system install: Eugene keeps that file, so the wizard says so."""
    settings = _container(tmp_path, own=tmp_path / "own-passphrase")
    with TestClient(_app(settings)) as c:
        status = c.get("/v1/auth/status").json()
    assert status["passphraseFile"] is True
    assert "passphraseFileFromOperator" not in status, status


def _security_mode_status(settings: Settings, root_content: bytes | None) -> dict[str, str]:
    """The agent's Security mode field status, set up as the wizard leaves a container."""
    root = settings.root_passphrase_file
    assert root is not None
    with TestClient(_app(settings)) as c:
        token = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).json()
        c.headers["Authorization"] = f"Bearer {token['sessionToken']}"
        if root_content is not None:
            root.write_bytes(root_content)
        body = c.get("/v1/config/schema").json()
    field = next(f for f in body["fields"] if f["key"] == "securityMode")
    return dict(field["status"])


@pytest.mark.parametrize(
    "content, fragment",
    [
        (None, "does not exist"),
        (b"somebody else's phrase", "does not hold this install's passphrase"),
    ],
    ids=["missing", "wrong"],
)
def test_the_field_never_says_it_unlocks_from_a_file_that_would_not(
    tmp_path: Path, content: bytes | None, fragment: str
) -> None:
    """Settings never lie (review of agent#44): the status read the file's existence
    only, so a file holding another passphrase showed "reads ... to unlock itself"."""
    status = _security_mode_status(_container(tmp_path), content)
    assert status["level"] == "warning", status
    assert fragment in status["text"] and "to unlock itself" not in status["text"], status
    assert "locked" in status["text"], status


def test_the_field_says_it_unlocks_from_a_file_that_would(tmp_path: Path) -> None:
    status = _security_mode_status(_container(tmp_path), TEST_PASSPHRASE.encode() + b"\n")
    assert status["level"] == "info" and "to unlock itself" in status["text"], status


def test_no_message_says_eugene_never_writes_the_operators_file(tmp_path: Path) -> None:
    """After a passphrase change the control root writes the new one there when it can
    (control's passphrase_change.write_file); only this agent never does."""
    from eugene_plexus_agent import passphrase_file

    said = [
        passphrase_file._OPERATORS_FIX,
        _security_mode_status(_container(tmp_path), TEST_PASSPHRASE.encode())["text"],
        _operators_file_state(tmp_path / "passphrase", "another phrase")[1] or "",
    ]
    for text in said:
        assert "Eugene never writes" not in text and "Eugene reads and never" not in text, text
        assert "agent never writes" in text, text
