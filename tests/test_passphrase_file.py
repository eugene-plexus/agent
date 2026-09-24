"""`securityMode: passphrase_file` -- an agent under its own account unlocks itself.

Row 2 of the node.yaml exposure work (2026-09-24): on the Linux system
install the agent runs as `eugene-plexus`, so a program running as the
person cannot read its files or memory -- and that account has no desktop
session, so no Secret Service. The installer writes
`securityMode: passphrase_file` into agent.yaml before the first start and
points `EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE` at a file in the prefix; the
agent writes the passphrase there at initialize and sign-in and reads it
at startup. These tests seed agent.yaml the way the installer does.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import _private_files, passphrase_file
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.settings import Settings
from tests.conftest import TEST_PASSPHRASE, StubSupervisor

POSIX = pytest.mark.skipif(sys.platform == "win32", reason="file mode bits are POSIX only")


def _installed(tmp_path: Path, *, mode: str = "passphrase_file", path: bool = True) -> Settings:
    """agent.yaml as `install.sh` leaves it on a system install: one line."""
    (tmp_path / "agent.yaml").write_text(f"securityMode: {mode}\n", encoding="utf-8")
    return Settings(
        config_file=tmp_path / "agent.yaml",
        passphrase_file=(tmp_path / "passphrase") if path else None,
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


# --- the file ---------------------------------------------------------------


def test_reading_strips_one_trailing_newline_and_nothing_else(tmp_path: Path) -> None:
    f = tmp_path / "p"
    f.write_bytes(b"  spaced phrase \n\n")
    assert passphrase_file.read_passphrase(f) == "  spaced phrase \n"
    f.write_bytes(b"crlf phrase\r\n")
    assert passphrase_file.read_passphrase(f) == "crlf phrase"


@pytest.mark.parametrize(
    "content", [None, b"", b"\n", b"\xff\xfe not utf-8", b"x" * (passphrase_file.MAX_BYTES + 1)]
)
def test_a_file_that_cannot_be_a_passphrase_is_none(tmp_path: Path, content: bytes | None) -> None:
    f = tmp_path / "p"
    if content is not None:
        f.write_bytes(content)
    assert passphrase_file.read_passphrase(f) is None
    assert passphrase_file.read_passphrase(None) is None


def test_storing_writes_once_and_rewrites_only_what_differs(tmp_path: Path) -> None:
    f = tmp_path / "p"
    assert passphrase_file.store_passphrase(f, "first phrase here")
    first = f.stat()
    assert passphrase_file.store_passphrase(f, "first phrase here")
    assert f.stat().st_mtime_ns == first.st_mtime_ns  # same passphrase, no write
    assert passphrase_file.store_passphrase(f, "second phrase here")
    assert f.read_bytes() == b"second phrase here"
    assert not passphrase_file.store_passphrase(None, "anything at all")


@POSIX
def test_the_file_is_owner_read_only(tmp_path: Path) -> None:
    f = tmp_path / "p"
    passphrase_file.store_passphrase(f, "a phrase to keep")
    assert stat.S_IMODE(f.stat().st_mode) == 0o400
    # And a rewrite over a 0400 file still works: it is a rename.
    passphrase_file.store_passphrase(f, "another phrase to keep")
    assert f.read_text() == "another phrase to keep"
    assert stat.S_IMODE(f.stat().st_mode) == 0o400


@POSIX
def test_write_private_narrows_and_never_widens(tmp_path: Path) -> None:
    f = tmp_path / "p"
    _private_files.write_private(f, "x", mode=0o644)
    assert stat.S_IMODE(f.stat().st_mode) == 0o600


# --- the install -------------------------------------------------------------


def test_the_wizard_is_told_before_setup_and_the_file_is_written_at_setup(tmp_path: Path) -> None:
    settings = _installed(tmp_path)
    with TestClient(_app(settings)) as c:
        before = c.get("/v1/auth/status").json()
        assert before["passphraseFile"] is True
        assert before["initialized"] is False
        assert (
            c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
        )
    assert (tmp_path / "passphrase").read_text(encoding="utf-8") == TEST_PASSPHRASE


def test_a_restart_comes_back_unlocked_with_nobody_signing_in(tmp_path: Path) -> None:
    settings = _installed(tmp_path)
    _initialize(settings)
    assert _unlocked_after_restart(settings)


def test_a_file_holding_another_passphrase_does_not_unlock(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verified before derived: a wrong key would open nothing and make
    every sealed value read as corrupt rather than locked."""
    settings = _installed(tmp_path)
    _initialize(settings)
    (tmp_path / "passphrase").unlink()
    (tmp_path / "passphrase").write_text("somebody else's phrase", encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="eugene_plexus_agent.passphrase_file")
    assert not _unlocked_after_restart(settings)
    assert any("is not this install's" in r.getMessage() for r in caplog.records)


def test_signing_in_repairs_a_missing_or_wrong_file(tmp_path: Path) -> None:
    settings = _installed(tmp_path)
    _initialize(settings)
    (tmp_path / "passphrase").unlink()
    assert not _unlocked_after_restart(settings)
    with TestClient(_app(settings)) as c:
        assert c.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
    assert (tmp_path / "passphrase").read_text(encoding="utf-8") == TEST_PASSPHRASE
    assert _unlocked_after_restart(settings)


def test_a_failed_sign_in_writes_nothing(tmp_path: Path) -> None:
    settings = _installed(tmp_path)
    _initialize(settings)
    with TestClient(_app(settings)) as c:
        c.post("/v1/auth/login", json={"passphrase": "not the passphrase at all"})
    assert (tmp_path / "passphrase").read_text(encoding="utf-8") == TEST_PASSPHRASE


def test_the_other_modes_neither_write_nor_trust_the_file(tmp_path: Path) -> None:
    """A file sitting there under prompt_on_startup is not an invitation:
    only the mode the installer chose unlocks from it."""
    settings = _installed(tmp_path, mode="prompt_on_startup")
    _initialize(settings)
    assert not (tmp_path / "passphrase").exists()
    (tmp_path / "passphrase").write_text(TEST_PASSPHRASE, encoding="utf-8")
    assert not _unlocked_after_restart(settings)
    with TestClient(_app(settings)) as c:
        assert c.get("/v1/auth/status").json()["passphraseFile"] is False


def test_the_mode_without_a_path_is_not_reported_as_unlocking(tmp_path: Path) -> None:
    settings = _installed(tmp_path, path=False)
    with TestClient(_app(settings)) as c:
        assert c.get("/v1/auth/status").json()["passphraseFile"] is False


def test_the_test_button_says_what_is_missing(tmp_path: Path) -> None:
    for with_path, written, fragment in (
        (False, False, "needs EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE"),
        (True, False, "does not exist yet"),
        (True, True, "unlocks itself on restart"),
    ):
        root = tmp_path / f"{with_path}-{written}"
        root.mkdir()
        settings = _installed(root, path=with_path)
        with TestClient(_app(settings)) as c:
            token = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).json()
            if with_path and not written:
                (root / "passphrase").unlink()
            c.headers["Authorization"] = f"Bearer {token['sessionToken']}"
            body = c.post("/v1/config/test").json()
        said = (body.get("summary") or "") + (body.get("error") or "")
        assert fragment in said, (with_path, written, body)
        assert body["ok"] is with_path


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_the_file_the_agent_writes_is_its_own(tmp_path: Path) -> None:
    settings = _installed(tmp_path)
    _initialize(settings)
    assert (tmp_path / "passphrase").stat().st_uid == os.getuid()
