"""Eugene's own account runs no program anyone else can change (row 8).

The disk is a stand-in, so these run on every platform: each path has an
owner and a mode, and the agent is `_eugeneplexus` (uid 400) or a person.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eugene_plexus_agent import exec_trust
from eugene_plexus_agent._generated.models import EngineKind, RuntimeSpec
from eugene_plexus_agent.engines import EngineUnavailableError, LlamaCppAdapter
from eugene_plexus_agent.engines.vllm import VllmAdapter

NAMES = {0: "root", 400: "_eugeneplexus", 501: "troy"}


def _disk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    me: int,
    entries: dict[str, tuple[int, int]],
    links: dict[str, str] | None = None,
) -> None:
    """ENTRIES: path -> (owner uid, mode bits) for the file and each folder
    above it; LINKS: path -> what it resolves to."""
    table: dict[Path, Any] = {}
    for path, (uid, mode) in entries.items():
        absolute = Path(os.path.abspath(path))
        kind = (
            stat.S_IFREG
            if "." in absolute.name or absolute.name.startswith("llama")
            else stat.S_IFDIR
        )
        table[absolute] = SimpleNamespace(st_uid=uid, st_mode=kind | mode)
    resolved = {Path(os.path.abspath(k)): os.path.abspath(v) for k, v in (links or {}).items()}
    for link in resolved:
        table[link] = SimpleNamespace(st_uid=0, st_mode=stat.S_IFLNK | 0o777)

    def lstat(path: Path) -> Any:
        try:
            return table[Path(path)]
        except KeyError:
            raise FileNotFoundError(2, "No such file or directory") from None

    monkeypatch.setattr(exec_trust, "_euid", lambda: me)
    monkeypatch.setattr(exec_trust, "_account_name", lambda uid: NAMES.get(uid))
    monkeypatch.setattr(exec_trust, "_lstat", lambda p: lstat(Path(p)))
    monkeypatch.setattr(exec_trust, "_realpath", lambda p: resolved.get(Path(p), str(p)))


def _chain(path: str, owners: list[tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """PATH and each folder above it, deepest first, given their owners."""
    here = Path(os.path.abspath(path))
    parts = [here, *here.parents]
    assert len(parts) == len(owners), (parts, owners)
    return {str(p): o for p, o in zip(parts, owners, strict=True)}


ROOT = (0, 0o755)
MINE = (400, 0o755)
TROYS = (501, 0o755)


def test_a_person_s_own_install_is_not_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as a person (troy), not Eugene's account: even another person's
    file in a folder its group can write is not this check's to refuse."""
    path = "/opt/shared/llama-server"
    _disk(monkeypatch, me=501, entries=_chain(path, [(502, 0o755), (502, 0o775), ROOT, ROOT]))
    assert exec_trust.untrusted(path) is None


def test_a_file_of_a_person_s_is_refused_naming_its_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    path = "/Users/troy/llama/llama-server"
    _disk(monkeypatch, me=400, entries=_chain(path, [TROYS, TROYS, TROYS, ROOT, ROOT]))
    reason = exec_trust.untrusted(path)
    assert reason is not None
    assert "belongs to troy" in reason
    assert "Move it, or let Eugene fetch its own build" in reason


def test_a_folder_of_a_person_s_above_it_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    path = "/Users/troy/llama/llama-server"
    _disk(monkeypatch, me=400, entries=_chain(path, [MINE, MINE, TROYS, ROOT, ROOT]))
    reason = exec_trust.untrusted(path)
    assert reason is not None and "which belongs to troy" in reason


def test_a_folder_others_can_write_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    path = "/usr/local/shared/llama-server"
    _disk(monkeypatch, me=400, entries=_chain(path, [ROOT, (0, 0o775), ROOT, ROOT, ROOT]))
    reason = exec_trust.untrusted(path)
    assert reason is not None and "can be changed by others (mode 0775)" in reason


def test_root_s_and_its_own_folders_run(monkeypatch: pytest.MonkeyPatch) -> None:
    path = "/usr/local/eugene-plexus/engines/llama-server"
    _disk(monkeypatch, me=400, entries=_chain(path, [MINE, MINE, MINE, ROOT, ROOT, ROOT]))
    assert exec_trust.untrusted(path) is None


def test_a_link_to_a_person_s_file_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    link = "/usr/local/bin/llama-server"
    target = "/Users/troy/llama-server"
    entries = _chain(target, [TROYS, TROYS, ROOT, ROOT])
    entries.update(
        {str(Path(os.path.abspath(p))): ROOT for p in ("/usr/local/bin", "/usr/local", "/usr")}
    )
    _disk(monkeypatch, me=400, entries=entries, links={link: target})
    reason = exec_trust.untrusted(link)
    assert reason is not None and "belongs to troy" in reason


def test_a_link_to_its_own_file_in_a_person_s_folder_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file is the account's, but the folder it is in is the person's:
    the person can swap it."""
    link = "/usr/local/bin/llama-server"
    target = "/Users/troy/llama-server"
    entries = _chain(target, [MINE, TROYS, ROOT, ROOT])
    entries.update(
        {str(Path(os.path.abspath(p))): ROOT for p in ("/usr/local/bin", "/usr/local", "/usr")}
    )
    _disk(monkeypatch, me=400, entries=entries, links={link: target})
    reason = exec_trust.untrusted(link)
    assert reason is not None and "which belongs to troy" in reason


def test_a_link_in_a_person_s_folder_is_refused_though_it_points_at_root_s_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The person can point the link elsewhere: the folder holding it
    decides."""
    link = "/Users/troy/bin/llama-server"
    target = "/usr/bin/llama-server"
    entries = _chain(target, [ROOT, ROOT, ROOT, ROOT])
    entries.update(
        {
            str(Path(os.path.abspath("/Users/troy/bin"))): TROYS,
            str(Path(os.path.abspath("/Users/troy"))): TROYS,
            str(Path(os.path.abspath("/Users"))): ROOT,
        }
    )
    _disk(monkeypatch, me=400, entries=entries, links={link: target})
    reason = exec_trust.untrusted(link)
    assert reason is not None and "which belongs to troy" in reason


def _spec(**overrides: Any) -> RuntimeSpec:
    base: dict[str, Any] = {
        "name": "m",
        "engine": EngineKind.llama_cpp,
        "modelPath": "/models/m.gguf",
    }
    base.update(overrides)
    return RuntimeSpec.model_validate(base)


def test_a_runtime_s_binary_is_refused_before_it_is_run_even_for_its_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runtime fails naming the path and its owner, and nothing ran:
    not even the `--version` probe."""
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    said = f"{exe} belongs to troy; Eugene's own account runs programs only from folders it or root owns."
    monkeypatch.setattr(exec_trust, "untrusted", lambda path: said)
    ran: list[Path] = []
    adapter = LlamaCppAdapter()
    monkeypatch.setattr(adapter, "probe_version", lambda binary: ran.append(binary) or "1")
    with pytest.raises(EngineUnavailableError, match="belongs to troy"):
        adapter.resolve_binary(_spec(binary=str(exe)))
    assert ran == []


def test_a_configured_engine_binary_is_refused_before_its_environment_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "vllm"
    exe.write_text("#!/usr/bin/python3\n", encoding="utf-8")
    monkeypatch.setattr(exec_trust, "untrusted", lambda path: f"{path} belongs to troy; ...")
    read: list[Path] = []
    from eugene_plexus_agent.engines import vllm

    monkeypatch.setattr(vllm, "inspect_python_environment", lambda path: read.append(path))
    with pytest.raises(EngineUnavailableError, match="belongs to troy"):
        VllmAdapter().discover(configured=str(exe))
    assert read == []
