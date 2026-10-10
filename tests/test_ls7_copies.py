"""A node's copy of a model is every file it is made of (LS7, agent#12 and #10),
and the console knows which drive a copy, or a Strata model, sits on."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from eugene_plexus_agent import drives
from eugene_plexus_agent import model_copies as mc
from eugene_plexus_agent.engines.strata import StrataAdapter, prepared_config, prepared_entry
from eugene_plexus_agent.model_paths import PathRule

FIRST = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
SECOND = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf"
UD = [f"Qwen3.8-Flash-Next-UD-IQ4_XS-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"


@pytest.fixture(autouse=True)
def _fresh_members() -> None:
    mc._members_cache.clear()


def settings(tmp_path: Path) -> mc.CopySettings:
    return mc.CopySettings(enabled=True, directory=str(tmp_path / "copies"), min_free_bytes=0)


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def library(tmp_path: Path) -> tuple[Path, PathRule]:
    """A Library folder as LS5 leaves it: the GGUF's two shards in the
    publisher's folder, and Strata-data beside it with a prepared model."""
    mount = tmp_path / "share"
    repo = mount / REPO
    _write(repo / FIRST, b"first" * 10)
    _write(repo / SECOND, b"second" * 10)
    data = mount / "Strata-data"
    _write(data / "packs" / "iq2_xs" / "dense.bin", b"d" * 30)
    _write(data / "packs" / "iq2_xs" / "expert-profile.bin", b"p" * 3)
    for name in ("vocab.json", "merges.txt", "token_type.json"):
        _write(data / "packs" / "iq2_xs" / "tokenizer" / name, b"{}")
    _write(data / "mtp" / "rt" / "experts.bin", b"m" * 20)
    _write(data / "mtp" / "tensors" / "big.bin", b"t" * 50)  # setup's, not named
    config = {
        "args": [
            "--pack",
            "packs/iq2_xs",
            "--native",
            f"../{REPO}/{FIRST}",
            "--ple-gguf",
            f"../{REPO}/{SECOND}",
            "--expert-profile",
            "packs/iq2_xs/expert-profile.bin",
            "--mtp",
            "mtp/rt",
            "--max-context",
            "131072",
        ],
        "tokenizer": "packs/iq2_xs/tokenizer",
        "parallel": 1,
        "cuda": 13,
        "backend": "cuda",
    }
    (data / "strata-iq2_xs.json").write_text(json.dumps(config), encoding="utf-8")
    (data / "iq2.eugene-prepared.json").write_text(
        json.dumps({"engine": "strata", "entry": "strata-iq2_xs.json", "source": {"file": FIRST}}),
        encoding="utf-8",
    )
    return mount, PathRule(source="/models", target=str(mount))


def test_a_split_gguf_is_copied_with_every_shard(tmp_path: Path) -> None:
    _mount, rule = library(tmp_path)
    plan = mc.plan_for(f"/models/{REPO}/{FIRST}", [rule], settings(tmp_path))
    assert plan is not None
    names = [Path(m.destination).name for m in plan.files]
    assert names == [FIRST, SECOND]
    assert all(m.destination.startswith(str(tmp_path / "copies")) for m in plan.files)


def test_a_gguf_missing_a_shard_is_not_copied_half(tmp_path: Path) -> None:
    mount, rule = library(tmp_path)
    (mount / REPO / SECOND).unlink()
    assert mc.plan_for(f"/models/{REPO}/{FIRST}", [rule], settings(tmp_path)) is None


def test_a_prepared_model_is_copied_whole_and_launches_from_the_copy(tmp_path: Path) -> None:
    _mount, rule = library(tmp_path)
    copies = settings(tmp_path)
    declared = "/models/Strata-data/iq2.eugene-prepared.json"
    plan = mc.plan_for(declared, [rule], copies)
    assert plan is not None
    relative = sorted(
        Path(m.destination).relative_to(tmp_path / "copies").as_posix() for m in plan.files
    )
    assert relative == sorted(
        [
            "Strata-data/iq2.eugene-prepared.json",
            "Strata-data/strata-iq2_xs.json",
            "Strata-data/packs/iq2_xs/dense.bin",
            "Strata-data/packs/iq2_xs/expert-profile.bin",
            "Strata-data/packs/iq2_xs/tokenizer/merges.txt",
            "Strata-data/packs/iq2_xs/tokenizer/token_type.json",
            "Strata-data/packs/iq2_xs/tokenizer/vocab.json",
            "Strata-data/mtp/rt/experts.bin",
            f"{REPO}/{FIRST}",
            f"{REPO}/{SECOND}",
        ]
    )
    # Setup's intermediates the configuration does not name stay where they are.
    assert not any("tensors" in m.destination for m in plan.files)
    assert not mc.copy_is_current(plan)
    state = mc.CopyState(destination=plan.destination, total_bytes=mc.total_size(plan))
    mc.copy_file(plan, copies, state)
    assert (state.files, state.files_copied) == (len(plan.files), len(plan.files))
    assert state.bytes_copied == state.total_bytes
    assert mc.copy_is_current(plan)
    local = mc.resolve_local_path(declared, [rule], copies)
    assert local.is_copy and local.path == plan.destination
    # The copy is whole: its relative paths still hold, so Strata launches it.
    entry = prepared_entry(Path(local.path))
    engine = tmp_path / "engine"
    _write(engine / "engine" / ("strata.exe" if os.name == "nt" else "strata"), b"")
    launched = prepared_config(entry, alias="q", root=engine)
    args = launched["args"]
    assert isinstance(args, list)
    native = Path(str(args[args.index("--native") + 1]))
    assert native.is_relative_to(tmp_path / "copies") and native.name == FIRST


def test_a_copy_stopped_halfway_carries_on_and_is_never_used_half_made(tmp_path: Path) -> None:
    _mount, rule = library(tmp_path)
    copies = settings(tmp_path)
    declared = "/models/Strata-data/iq2.eugene-prepared.json"
    plan = mc.plan_for(declared, [rule], copies)
    assert plan is not None
    first = plan.files[2]
    _write(Path(first.destination), Path(first.source).read_bytes())
    os.utime(first.destination, (time.time(), os.stat(first.source).st_mtime))
    # One file whole is not the model: the share is still what opens.
    assert not mc.resolve_local_path(declared, [rule], copies).is_copy
    copied: list[str] = []
    real = mc._copy_one

    def recording(member, *args):  # type: ignore[no-untyped-def]
        copied.append(member.destination)
        return real(member, *args)

    mc._copy_one = recording  # type: ignore[assignment]
    try:
        state = mc.CopyState(destination=plan.destination, total_bytes=mc.total_size(plan))
        mc.copy_file(plan, copies, state)
    finally:
        mc._copy_one = real  # type: ignore[assignment]
    assert first.destination not in copied  # carried on, not copied again
    assert len(copied) == len(plan.files) - 1
    assert state.files_copied == len(plan.files)


def test_a_file_outside_the_library_folder_keeps_the_model_on_the_share(tmp_path: Path) -> None:
    mount, rule = library(tmp_path)
    elsewhere = _write(tmp_path / "elsewhere" / "pack" / "dense.bin", b"x")
    config_path = mount / "Strata-data" / "strata-iq2_xs.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["args"][1] = str(elsewhere.parent)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan = mc.plan_for("/models/Strata-data/iq2.eugene-prepared.json", [rule], settings(tmp_path))
    assert plan is None


def test_reconciling_keeps_every_file_of_a_wanted_set(tmp_path: Path) -> None:
    _mount, rule = library(tmp_path)
    copies = settings(tmp_path)
    declared = "/models/Strata-data/iq2.eugene-prepared.json"
    plan = mc.plan_for(declared, [rule], copies)
    assert plan is not None
    mc.copy_file(plan, copies, mc.CopyState(destination=plan.destination, total_bytes=None))
    keep = [d for p in mc.wanted([declared], [rule], copies).values() for d in p.destinations]
    removed = mc.remove_unwanted(copies.directory, keep)
    assert removed.deleted == []
    assert mc.copy_is_current(plan)


def test_strata_names_its_set_and_an_unreadable_one_is_none(tmp_path: Path) -> None:
    mount, _rule = library(tmp_path)
    provenance = mount / "Strata-data" / "iq2.eugene-prepared.json"
    files = StrataAdapter().prepared_files(provenance)
    assert files is not None and files[0] == provenance
    (mount / "Strata-data" / "mtp" / "rt" / "experts.bin").unlink()
    (mount / "Strata-data" / "mtp" / "rt").rmdir()
    assert StrataAdapter().prepared_files(provenance) is None


# --- which drive (LS7, Troy reviewing B51) -----------------------------------------


def test_a_share_is_a_network_drive_without_asking_the_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def asked(_path: str) -> str:
        raise AssertionError("a share's kind is its address, not a system question")

    monkeypatch.setattr(drives, "_windows_kind", asked)
    monkeypatch.setattr(drives, "_linux_kind", asked)
    drives._cache.clear()
    assert drives.drive_kind("\\\\nas\\models") == "network"
    assert drives.drive_kind("//nas/models") == "network"


def test_the_copy_folder_says_which_drive_it_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    for kind, level, words in (
        ("ssd", "info", "On an SSD."),
        ("hdd", "warning", "spinning disk"),
        ("network", "warning", "network share"),
    ):
        monkeypatch.setattr(drives, "drive_kind", lambda _p, k=kind: k)
        found = drives.copy_folder_status("D:\\copies")
        assert found is not None and found[0] == level and words in found[1]
    monkeypatch.setattr(drives, "drive_kind", lambda _p: "unknown")
    assert drives.copy_folder_status("D:\\copies") is None


def test_strata_says_so_at_install_and_at_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drives, "drive_kind", lambda _p: "hdd")
    note = drives.strata_install_note("E:\\copies")
    assert note is not None and "spinning disk" in note and "E:\\copies" in note
    warning = drives.strata_run_warning("E:\\models\\x.eugene-prepared.json")
    assert warning is not None and "E:" in warning and "#605" in warning
    assert drives.strata_install_note(None) is None
    monkeypatch.setattr(drives, "drive_kind", lambda _p: "ssd")
    assert drives.strata_install_note("C:\\copies") is None
    assert drives.strata_run_warning("C:\\models\\x.json") is None


def test_strata_names_a_tokenizer_outside_its_pack_and_every_shard(tmp_path: Path) -> None:
    """As a real configuration lays it out (LS5's IQ2_XS: `--pack empty-pack`,
    the tokenizer under `packs/<tag>`), and a three-shard GGUF named by its
    first and last shards only: the middle one is found by the shard rule."""
    data = tmp_path / "lib" / "Strata-data"
    repo = tmp_path / "lib" / "unsloth"
    for name in UD:
        _write(repo / name, b"x")
    _write(data / "empty-pack" / "README", b"empty")
    for name in ("vocab.json", "merges.txt", "token_type.json"):
        _write(data / "packs" / "ud" / "tokenizer" / name, b"{}")
    config = {
        "args": [
            "--pack",
            "empty-pack",
            "--native",
            f"../unsloth/{UD[0]}",
            "--ple-gguf",
            f"../unsloth/{UD[2]}",
            "--max-context",
            "8192",
        ],
        "tokenizer": "packs/ud/tokenizer",
    }
    (data / "strata-ud.json").write_text(json.dumps(config), encoding="utf-8")
    provenance = data / "ud.eugene-prepared.json"
    provenance.write_text(json.dumps({"engine": "strata", "entry": "strata-ud.json"}), "utf-8")
    files = StrataAdapter().prepared_files(provenance)
    assert files is not None
    names = {f.name for f in files}
    assert {"vocab.json", "merges.txt", "token_type.json"} <= names
    assert set(UD) <= names


# --- the supervisor's own reconcile keeps a whole set ---------------------------


async def test_the_supervisor_keeps_every_file_of_a_runtime_it_still_declares(
    tmp_path: Path,
) -> None:
    from typing import Any

    from eugene_plexus_agent._generated.models import RuntimeSpec
    from eugene_plexus_agent.runtimes import RuntimeSupervisor

    _mount, rule = library(tmp_path)
    config: dict[str, Any] = {
        "modelCopyEnabled": True,
        "modelCopyDir": str(tmp_path / "copies"),
        "modelCopyMinFreeGb": 0,
    }
    sup = RuntimeSupervisor(get_config=config.get, inherited_rules=lambda: [rule])
    declared = "/models/Strata-data/iq2.eugene-prepared.json"
    plan = mc.plan_for(declared, [rule], settings(tmp_path))
    assert plan is not None
    mc.copy_file(plan, settings(tmp_path), mc.CopyState(plan.destination, None))
    spec = RuntimeSpec.model_validate({"name": "s", "engine": "strata", "modelPath": declared})
    sup.reconcile_copies([spec])
    assert mc.copy_is_current(plan)
