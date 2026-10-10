"""LS7b: what a prepared model is, read off the engine's own files (B22
replaced, B26), and the oldest engine an entry needs (B30)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eugene_plexus_agent.engines.base import (
    PreparedFacts,
    PreparedInspectError,
    version_tuple,
    with_engine_version,
)
from eugene_plexus_agent.engines.strata import inspect_config, run_mode
from eugene_plexus_agent.engines.strata_models import SUPPORTED_MODELS, supported_here
from eugene_plexus_agent.prepared_facts import draft, facts_fields, library_spelling

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
FIRST = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
SECOND = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf"


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _folder(tmp_path: Path, *extra: str) -> Path:
    """A Library folder as a Strata preparation leaves it; returns the entry."""
    mount = tmp_path / "share"
    _write(mount / REPO / FIRST, b"first" * 10)
    _write(mount / REPO / SECOND, b"second" * 10)
    data = mount / "Strata-data"
    _write(data / "packs" / "iq2_xs" / "dense.bin", b"d" * 30)
    _write(data / "packs" / "iq2_xs" / "expert-profile.bin", b"p" * 3)
    _write(data / "packs" / "iq2_xs" / "tokenizer" / "vocab.json", b"{}")
    _write(data / "mtp" / "rt" / "experts.bin", b"m" * 20)
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
            "65536",
            *extra,
        ],
        "tokenizer": "packs/iq2_xs/tokenizer",
    }
    entry = data / "strata-iq2_xs.json"
    entry.write_text(json.dumps(config), encoding="utf-8")
    return entry


def test_strata_reads_its_configuration_back_into_what_the_library_records(
    tmp_path: Path,
) -> None:
    entry = _folder(tmp_path)
    facts = inspect_config(entry)
    assert facts.source_file == entry.parent.parent / REPO / FIRST
    assert facts.repo_id == REPO
    assert facts.hub_file == f"IQ2_XS/{FIRST}"
    assert facts.title == "Qwen3.8-Flash-Next IQ2_XS"
    assert facts.architecture == "qwen4exp"
    assert facts.quantization == "IQ2_XS"
    assert facts.context_length == 65536
    assert facts.mode == "every expert in RAM"
    names = {p.relative_to(entry.parent).as_posix(): shared for p, shared in facts.files}
    # The entry first; the pack, profile and tokenizer its own; the MTP
    # helper shared; the source model's shards not listed (they are its own).
    assert facts.files[0] == (entry, False)
    assert names == {
        "strata-iq2_xs.json": False,
        "packs/iq2_xs/dense.bin": False,
        "packs/iq2_xs/expert-profile.bin": False,
        "packs/iq2_xs/tokenizer/vocab.json": False,
        "mtp/rt/experts.bin": True,
    }


def test_a_configuration_naming_a_missing_file_says_which(tmp_path: Path) -> None:
    entry = _folder(tmp_path)
    (entry.parent / "mtp" / "rt" / "experts.bin").unlink()
    (entry.parent / "mtp" / "rt").rmdir()
    with pytest.raises(PreparedInspectError, match=r"mtp.+--mtp.+not whole"):
        inspect_config(entry)


def test_a_file_that_is_not_a_strata_configuration_is_refused(tmp_path: Path) -> None:
    other = _write(tmp_path / "notes.json", b'{"hello": 1}')
    with pytest.raises(PreparedInspectError, match="no `args` list"):
        inspect_config(other)
    with pytest.raises(PreparedInspectError, match="not JSON"):
        inspect_config(_write(tmp_path / "x.json", b"{"))


def test_a_source_off_the_list_still_gives_what_the_configuration_says(tmp_path: Path) -> None:
    entry = _folder(tmp_path)
    cfg = json.loads(entry.read_text(encoding="utf-8"))
    renamed = entry.parent.parent / REPO / "mine-00001-of-00002.gguf"
    (entry.parent.parent / REPO / FIRST).rename(renamed)
    cfg["args"][3] = f"../{REPO}/{renamed.name}"
    entry.write_text(json.dumps(cfg), encoding="utf-8")
    facts = inspect_config(entry)
    assert facts.source_file == renamed
    assert facts.title is None and facts.architecture is None and facts.repo_id is None
    assert facts.context_length == 65536


@pytest.mark.parametrize(
    ("extra", "words"),
    [
        ([], "every expert in RAM"),
        (["--resident-budget-gib", "40"], "a RAM budget of 40 GiB of its experts"),
        (["--resident-experts"], "copied into RAM once"),
        (["--mmap-experts"], "read from the SSD as needed"),
    ],
)
def test_the_mode_is_said_in_setups_words(extra: list[str], words: str) -> None:
    assert words in run_mode(["--pack", "p", *extra])


def test_the_draft_spells_paths_as_the_library_does(tmp_path: Path) -> None:
    entry = _folder(tmp_path)
    facts = inspect_config(entry)
    body = draft(
        facts,
        engine="strata",
        entry_local=entry,
        entry_declared="/models/Strata-data/strata-iq2_xs.json",
    )
    assert body["engine"] == "strata"
    assert body["entry"] == "/models/Strata-data/strata-iq2_xs.json"
    assert body["source"] == {
        "path": f"/models/{REPO}/{FIRST}",
        "repoId": REPO,
        "file": f"IQ2_XS/{FIRST}",
    }
    assert body["contextLength"] == 65536
    by_path = {f["path"]: f for f in body["files"]}
    assert by_path["packs/iq2_xs/dense.bin"] == {"path": "packs/iq2_xs/dense.bin", "sizeBytes": 30}
    assert by_path["mtp/rt/experts.bin"]["shared"] is True
    assert "shared" not in by_path["strata-iq2_xs.json"]
    windows = library_spelling(
        facts.source_file or Path(),
        entry_local=entry,
        entry_declared=r"Z:\models\Strata-data\strata-iq2_xs.json",
    )
    assert windows == "Z:\\models\\" + REPO.replace("/", "\\") + "\\" + FIRST


def test_facts_name_files_relative_to_the_provenance_folder(tmp_path: Path) -> None:
    entry = _folder(tmp_path)
    facts = PreparedFacts(files=((entry, False),), mode="m")
    out = facts_fields(facts, folder=entry.parent.parent)
    assert out["files"] == [
        {"path": "Strata-data/strata-iq2_xs.json", "sizeBytes": entry.stat().st_size}
    ]
    assert out["mode"] == "m"
    assert "title" not in out


def test_versions_compare_by_their_numbers() -> None:
    assert version_tuple("v0.1.38") == (0, 1, 38)
    assert version_tuple("0.1.39-rc1") == (0, 1, 39)
    assert version_tuple(None) is None
    assert version_tuple("main") is None


def test_an_entry_needing_a_newer_engine_says_which_one_this_node_has() -> None:
    listed = supported_here(64 * 2**30)
    needs = {m.id: m.preparation.minEngineVersion for m in listed if m.preparation}
    # setup.py MODELS["UD-IQ4_XS"]["engine"] = (0, 1, 38); no other entry names one.
    assert needs["unsloth-UD-IQ4_XS"] == "v0.1.38"
    assert [k for k, v in needs.items() if v] == ["unsloth-UD-IQ4_XS"]
    old = {m.id: m for m in with_engine_version(listed, "v0.1.37")}
    assert old["unsloth-UD-IQ4_XS"].preparation.engineTooOld == "v0.1.37"
    assert old["IQ2_XS"].preparation.engineTooOld is None
    for installed in ("v0.1.38", "v0.1.39", None):
        current = with_engine_version(listed, installed)
        assert all(m.preparation.engineTooOld is None for m in current)
    # The list as published keeps its disk per node (B51) beside the version.
    assert all(m.preparation.diskBytes for m in listed)
    assert len(listed) == len(SUPPORTED_MODELS)


def test_the_route_answers_a_draft_and_refuses_what_it_cannot_read(
    authed_client, tmp_path: Path
) -> None:
    entry = _folder(tmp_path)
    url = "/v1/engines/strata/prepared/inspect"
    response = authed_client.post(url, json={"entry": str(entry)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["entry"] == str(entry)
    assert body["title"] == "Qwen3.8-Flash-Next IQ2_XS"
    assert body["source"]["path"] == str(entry.parent.parent / REPO / FIRST)
    assert {f["path"] for f in body["files"]} >= {"strata-iq2_xs.json", "mtp/rt/experts.bin"}
    bad = _write(tmp_path / "notes.json", b"[]")
    refused = authed_client.post(url, json={"entry": str(bad)})
    assert refused.status_code == 422
    assert "no `args` list" in refused.json()["detail"]["detail"]
    other = authed_client.post("/v1/engines/llama_cpp/prepared/inspect", json={"entry": str(bad)})
    assert other.status_code == 422
    assert "prepares no models" in other.json()["detail"]["detail"]
    assert (
        authed_client.post("/v1/engines/nope/prepared/inspect", json={"entry": "x"}).status_code
        == 404
    )
