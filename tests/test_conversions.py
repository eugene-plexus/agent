"""What v0.1 saved in `agent.yaml` is converted once, kept as it was, and read in the current
format only.

The fixture is not a look-alike: `fixtures/v0_1/agent.yaml` was written by the agent at the
v0.1.0 pin (d683bf9) from a checkbox-era save, with four llama.cpp runtimes (flash attention
`true`, `false`, `off`, and none) and a passphrase block whose values are placeholders.

Every conversion is held to the same six claims: converted correctly, a backup of the old bytes
kept, the new version recorded, a second start a no-op, a start that stopped midway rerun
cleanly, and a failure that degrades naming its cause.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import conversions
from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest
from eugene_plexus_agent._generated.models import Origin
from eugene_plexus_agent.conversions import (
    BACKUP_SUFFIX,
    STATE_VERSION,
    ConversionFailed,
    convert,
    keep_backup,
)
from eugene_plexus_agent.engines import LlamaCppAdapter
from eugene_plexus_agent.engines.base import DiscoveredBinary
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.state import UNREADABLE_SUFFIX, AgentState

FIXTURE = Path(__file__).parent / "fixtures" / "v0_1" / "agent.yaml"

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="file modes are POSIX")


class Crash(BaseException):
    """The process dying: nothing between the failing write and the next start runs."""


def place(directory: Path, newline: str = "\n") -> Path:
    path = directory / "agent.yaml"
    raw = FIXTURE.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", newline.encode())
    path.write_bytes(raw)
    return path


def backups_of(path: Path) -> list[Path]:
    return sorted(path.parent.glob(path.name + BACKUP_SUFFIX + "*"))


def on_disk(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def runtimes_flags(state: AgentState) -> dict[str, dict[str, Any]]:
    return {spec.name: spec.flags or {} for spec in state.list_runtime_specs()}


def argv_of(state: AgentState, name: str, tmp_path: Path) -> list[str]:
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    binary = DiscoveredBinary(path=exe, origin=Origin.path, version="4589")
    spec = state.get_runtime_spec(name)
    assert spec is not None
    return LlamaCppAdapter().build_argv(spec, binary, port=8090)


def flash(argv: list[str]) -> str | None:
    return argv[argv.index("--flash-attn") + 1] if "--flash-attn" in argv else None


class _Spy:
    """Fails the test if anything is written: what a no-op start must not do."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(conversions, "write_private", self._refuse)

    @staticmethod
    def _refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"a second start wrote a file: {args[:1]}")


# --- converted correctly, backed up, recorded ------------------------------------------


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["unix", "windows"])
def test_the_checkboxes_v0_1_saved_are_converted_and_everything_else_is_kept(
    tmp_path: Path, newline: str
) -> None:
    path = place(tmp_path, newline)
    before = path.read_bytes()
    original = yaml.safe_load(before)
    assert original["runtimes"][0]["flags"]["flashAttention"] is True, (
        "the fixture is the old shape"
    )
    assert original["runtimes"][1]["flags"]["flashAttention"] is False

    state = AgentState(path)
    state.load()

    # Version 2 is the record; every other byte of meaning is as it was, the passphrase block
    # and the order of every list included.
    expected = copy.deepcopy(original)
    expected["runtimes"][0]["flags"]["flashAttention"] = "on"
    del expected["runtimes"][1]["flags"]["flashAttention"]
    expected["stateVersion"] = STATE_VERSION
    assert on_disk(path) == expected
    assert runtimes_flags(state) == {
        "qwen3-8b": {"contextSize": 8192, "gpuLayers": 99, "flashAttention": "on"},
        "qwen3-8b-cpu-only": {"gpuLayers": 0},
        "gemma": {"contextSize": 16384, "flashAttention": "off"},
        "bge": {},
    }
    # The old bytes are kept exactly.
    assert backups_of(path) == [path.with_name(path.name + BACKUP_SUFFIX)]
    assert backups_of(path)[0].read_bytes() == before


@posix_only
def test_the_copy_is_as_private_as_the_file_because_it_holds_the_passphrase_hash(
    tmp_path: Path,
) -> None:
    path = place(tmp_path)
    AgentState(path).load()

    assert (os.stat(backups_of(path)[0]).st_mode & 0o777) == 0o600
    assert (os.stat(path).st_mode & 0o777) == 0o600


def test_a_checkbox_keeps_doing_what_it_did(tmp_path: Path) -> None:
    """`true` always sent `on`. `false` always sent nothing, which left llama.cpp's own
    choice, so it is not `off`: that would switch flash attention off in every runtime
    saved before. Neither is a boolean any more once the file has been loaded."""
    state = AgentState(place(tmp_path))
    state.load()

    assert flash(argv_of(state, "qwen3-8b", tmp_path)) == "on"
    assert flash(argv_of(state, "qwen3-8b-cpu-only", tmp_path)) is None
    assert flash(argv_of(state, "gemma", tmp_path)) == "off"
    assert flash(argv_of(state, "bge", tmp_path)) is None


def test_every_write_of_this_release_carries_the_version(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    state = AgentState(path)
    state.load()
    assert on_disk(path)["stateVersion"] == STATE_VERSION

    state.apply_config_patch(ConfigUpdateRequest(uiFontSize="large"))
    assert on_disk(path)["stateVersion"] == STATE_VERSION
    # And a converted file keeps it through a later write: the converter is not asked again.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    converted = place(elsewhere)
    other = AgentState(converted)
    other.load()
    other.apply_config_patch(ConfigUpdateRequest(uiFontSize="large"))
    assert on_disk(converted)["stateVersion"] == STATE_VERSION
    assert len(backups_of(converted)) == 1


# --- a second start --------------------------------------------------------------------


def test_a_second_start_changes_nothing_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = place(tmp_path)
    AgentState(path).load()
    kept = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    _Spy(monkeypatch)
    second = AgentState(path)
    assert second.load_or_degrade() is None

    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == kept
    assert runtimes_flags(second)["qwen3-8b"]["flashAttention"] == "on"


def test_a_file_with_no_checkbox_in_it_is_not_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "agent.yaml"
    text = (
        "runtimes:\n- engine: llama_cpp\n  flags:\n    flashAttention: 'on'\n"
        "  modelPath: /models/m.gguf\n  name: m\n  port: 8090\n"
    )
    path.write_text(text, encoding="utf-8")

    _Spy(monkeypatch)
    AgentState(path).load()

    assert path.read_text(encoding="utf-8") == text and backups_of(path) == []


def test_a_file_that_has_the_version_is_not_looked_at_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkbox put into a converted file by hand is not converted: only the current
    format is read, and a launch refuses the value (`test_engines.py`)."""
    path = tmp_path / "agent.yaml"
    path.write_text(
        f"stateVersion: {STATE_VERSION}\nruntimes:\n- engine: llama_cpp\n  flags:\n"
        "    flashAttention: true\n  modelPath: /models/m.gguf\n  name: m\n  port: 8090\n",
        encoding="utf-8",
    )
    before = path.read_bytes()

    _Spy(monkeypatch)
    state = AgentState(path)
    state.load()

    assert path.read_bytes() == before and backups_of(path) == []
    assert runtimes_flags(state) == {"m": {"flashAttention": True}}


def test_a_new_install_has_nothing_to_convert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "agent.yaml"
    state = AgentState(path)
    state.load()  # writes its defaults, with the version
    assert backups_of(path) == []

    _Spy(monkeypatch)
    assert convert(path, path.read_bytes(), on_disk(path)) == on_disk(path)


# --- a start that stopped midway -------------------------------------------------------


@pytest.mark.parametrize("stopped", ["before the backup", "after the backup"])
def test_a_start_that_stopped_midway_reruns_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stopped: str
) -> None:
    path = place(tmp_path)
    before = path.read_bytes()
    real = conversions.write_private
    writes: list[Path] = []

    def die_on(number: int) -> Any:
        def write(target: Path, data: Any, **kwargs: Any) -> None:
            writes.append(target)
            if len(writes) == number:
                raise Crash
            real(target, data, **kwargs)

        return write

    # The first write is the copy, the second is the converted file.
    monkeypatch.setattr(
        conversions, "write_private", die_on(1 if stopped == "before the backup" else 2)
    )
    with pytest.raises(Crash):
        AgentState(path).load()
    monkeypatch.undo()

    assert path.read_bytes() == before, "the file is only ever replaced by the finished one"
    assert len(backups_of(path)) == (0 if stopped == "before the backup" else 1)

    state = AgentState(path)
    state.load()

    assert on_disk(path)["stateVersion"] == STATE_VERSION
    assert runtimes_flags(state)["qwen3-8b"]["flashAttention"] == "on"
    assert backups_of(path) == [path.with_name(path.name + BACKUP_SUFFIX)]
    assert backups_of(path)[0].read_bytes() == before


def test_a_copy_that_says_something_else_is_never_written_over(tmp_path: Path) -> None:
    """It may be the only copy of what that was."""
    path = place(tmp_path)
    before = path.read_bytes()
    earlier = path.with_name(path.name + BACKUP_SUFFIX)
    earlier.write_bytes(b"an older agent.yaml, kept by an earlier conversion")

    AgentState(path).load()

    assert earlier.read_bytes() == b"an older agent.yaml, kept by an earlier conversion"
    assert earlier.with_name(earlier.name + ".1").read_bytes() == before
    assert keep_backup(path, before) == earlier.with_name(earlier.name + ".1")


# --- a failure -------------------------------------------------------------------------


def test_a_file_that_cannot_be_converted_comes_up_degraded_and_names_the_cause(
    tmp_path: Path,
) -> None:
    path = place(tmp_path)
    before = path.read_bytes()
    path.with_name(path.name + BACKUP_SUFFIX).mkdir()  # something that is not a file is in the way

    state = AgentState(path)
    reason = state.load_or_degrade()

    assert reason is not None and "ConversionFailed" in reason
    assert str(path) in reason and "left as it was" in reason and BACKUP_SUFFIX in reason
    assert path.read_bytes() == before, "the file is exactly what it was"
    assert path.with_suffix(path.suffix + UNREADABLE_SUFFIX).read_bytes() == before
    assert state.degraded_reason == reason
    assert state.list_runtime_specs() == [] and state.list_topology_entries() == []
    # The passphrase this file held is not lost to the wizard: it would mint a second salt.
    assert state.lost_its_passphrase()


def test_a_write_that_fails_names_its_cause_and_the_copy_that_was_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = place(tmp_path)
    before = path.read_bytes()
    real = conversions.write_private

    def full(target: Path, data: Any, **kwargs: Any) -> None:
        if target == path:
            raise OSError(28, "No space left on device")
        real(target, data, **kwargs)

    monkeypatch.setattr(conversions, "write_private", full)
    with pytest.raises(ConversionFailed) as caught:
        AgentState(path).load()

    assert "No space left on device" in str(caught.value) and "OSError" in str(caught.value)
    assert str(backups_of(path)[0]) in str(caught.value)
    assert path.read_bytes() == before


def test_the_agent_starts_degraded_when_the_conversion_fails_and_says_why_on_healthz(
    app: FastAPI, settings: Settings
) -> None:
    """Through the stubbed supervisors every test app has: this file declares runtimes that
    start at boot, and a test never starts an engine or a driver."""
    path = place(settings.config_file.parent)
    path.with_name(path.name + BACKUP_SUFFIX).mkdir()

    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "degraded"
        detail = body["details"]["configError"]
        assert "could not be converted" in detail and "left as it was" in detail
        assert body["details"]["configFilePreserved"].endswith(UNREADABLE_SUFFIX)


def test_the_agent_converts_the_file_when_it_starts(app: FastAPI, settings: Settings) -> None:
    path = place(settings.config_file.parent)

    with TestClient(app) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        assert app.state.agent_state.degraded_reason is None
        flags = runtimes_flags(app.state.agent_state)
        assert flags["qwen3-8b"]["flashAttention"] == "on"
        assert flags["qwen3-8b-cpu-only"] == {"gpuLayers": 0}

    assert on_disk(path)["stateVersion"] == STATE_VERSION and len(backups_of(path)) == 1
