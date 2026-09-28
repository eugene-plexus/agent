"""A blank passphrase-file variable is unset, not the current directory."""

from __future__ import annotations

from pathlib import Path

from eugene_plexus_agent.settings import Settings


def test_a_blank_passphrase_file_variable_is_unset(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Unraid passes a blank template variable as an empty string, and
    Path("") is ".", which was reported as a directory, not as unset."""
    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE", "")
    assert Settings().passphrase_file is None
    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE", "   ")
    assert Settings().passphrase_file is None
    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_PASSPHRASE_FILE", "/data/passphrase")
    assert Settings().passphrase_file == Path("/data/passphrase")
