"""Row 48b: a complete node copy can start with an unreadable source folder."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent import model_copies as mc
from eugene_plexus_agent.admission import LibraryFitClient, check_admission
from eugene_plexus_agent.library_folders import FolderRecord, LibraryFolderCache, refresh
from eugene_plexus_agent.model_paths import PathRule

from .test_model_copy_lifecycle import MODEL, spec, supervisor_for
from .test_storage_separation import fake_devices


def copied(tmp_path: Path) -> tuple[PathRule, mc.CopyPlan, mc.CopySettings]:
    mount = tmp_path / "mount"
    source = mount / "qwen" / "Qwen3-27B-Q6_K_L.gguf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"weights")
    rule = PathRule(source="/models", target=str(mount))
    settings = mc.CopySettings(True, str(tmp_path / "copies"), 0)
    plan = mc.plan_for(MODEL, [rule], settings)
    assert plan is not None
    mc.copy_file(plan, settings, mc.CopyState(plan.destination, 7))
    return rule, plan, settings


def test_copy_completion_records_source_and_an_old_copy_gets_a_record(tmp_path: Path) -> None:
    rule, plan, settings = copied(tmp_path)
    record = Path(plan.destination + ".source.json")
    raw = json.loads(record.read_text(encoding="utf-8"))
    assert raw["source"] == plan.source and raw["size"] == 7
    assert raw["mtime"] and raw["copiedAt"]
    record.unlink()
    assert mc.resolve_local_path(MODEL, [rule], settings).is_copy
    assert record.is_file()


@pytest.mark.parametrize("blocked", [False, True])
def test_copy_is_used_and_runtime_keeps_note_until_next_checked_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: bool
) -> None:
    rule, plan, settings = copied(tmp_path)
    release = threading.Event()
    stat = mc._stat

    def unavailable(path: str) -> Any:
        if path == plan.source:
            if blocked:
                release.wait(2)
            raise OSError("the network path was not found")
        return stat(path)

    sup = supervisor_for(tmp_path, rule)
    if not blocked:
        Path(rule.target).rename(tmp_path / "offline")
    monkeypatch.setattr(mc, "SOURCE_CHECK_SECONDS", 0.03, raising=False)
    monkeypatch.setattr(mc, "_stat", unavailable)
    try:
        started = time.perf_counter()
        resolved = mc.resolve_local_path(MODEL, [rule], settings)
        assert time.perf_counter() - started < 0.5
        assert resolved.is_copy
        sup.add_and_start(spec())
        note = sup.compose(spec()).localPathNote
        assert note and note.startswith("Started from this machine's copy.")
        assert rule.target in note
        assert ("timed out" if blocked else "cannot be checked") in note
        monkeypatch.setattr(mc, "_stat", stat)
        release.set()
        if not blocked:
            (tmp_path / "offline").rename(rule.target)
        assert sup.compose(spec()).localPathNote == note
        # A new start verifies the source and clears the previous start's note.
        sup._copy_job_for(spec())
        assert sup.compose(spec()).localPathNote is None
    finally:
        release.set()


def test_missing_file_in_readable_folder_and_changed_copy_are_refused(tmp_path: Path) -> None:
    rule, plan, settings = copied(tmp_path)
    Path(plan.source).unlink()
    assert not mc.resolve_local_path(MODEL, [rule], settings).is_copy
    Path(rule.target).rename(tmp_path / "offline")
    Path(plan.destination).write_bytes(b"short")
    assert not mc.resolve_local_path(MODEL, [rule], settings).is_copy


def test_clearing_a_copy_removes_its_record_and_keeps_held_records(tmp_path: Path) -> None:
    _, plan, settings = copied(tmp_path)
    record = Path(plan.destination + ".source.json")
    mc.clear(settings.directory, in_use={plan.destination: "qwen"})
    assert record.is_file()
    mc.clear(settings.directory)
    assert not record.exists()


def test_unmapped_folder_checks_the_library_root_instead_of_the_disk_root(tmp_path: Path) -> None:
    rule, plan, _ = copied(tmp_path)
    source = plan.source
    settings = mc.CopySettings(
        True, str(tmp_path / "unmapped-copies"), 0, folder_roots=(rule.target,)
    )
    same_path = mc.plan_for(source, [], settings)
    assert same_path is not None
    mc.copy_file(same_path, settings, mc.CopyState(same_path.destination, 7))
    Path(rule.target).rename(tmp_path / "offline")
    resolved = mc.resolve_local_path(source, [], settings)
    assert resolved.is_copy and resolved.note and rule.target in resolved.note


def test_concurrent_source_records_are_written_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plan, _ = copied(tmp_path)
    Path(plan.destination + ".source.json").unlink()
    source = mc._stat(plan.source)
    assert source is not None
    replace = mc.os.replace
    writes: list[str] = []

    def recorded(src: str, dst: str) -> None:
        writes.append(dst)
        replace(src, dst)

    monkeypatch.setattr(mc.os, "replace", recorded)
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(lambda _: mc._save_source(plan.files[0], source), range(8)))
    assert len(writes) == 1


def test_unmapped_sharded_copy_keeps_its_layout_and_saved_library_root(tmp_path: Path) -> None:
    mount = tmp_path / "mount"
    mount.mkdir()
    first = mount / "model-00001-of-00002.gguf"
    second = mount / "model-00002-of-00002.gguf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    settings = mc.CopySettings(True, str(tmp_path / "copies"), 0, folder_roots=(str(mount),))
    plan = mc.plan_for(str(first), [], settings)
    assert plan is not None and len(plan.files) == 2
    assert plan.destination == plan.files[0].destination
    mc.copy_file(plan, settings, mc.CopyState(plan.destination, 11))
    mount.rename(tmp_path / "offline")
    mc._members_cache.clear()
    resolved = mc.resolve_local_path(str(first), [], settings)
    assert resolved.is_copy and resolved.note and str(mount) in resolved.note


@pytest.mark.anyio
async def test_copy_without_record_is_refused_with_observed_cause(tmp_path: Path) -> None:
    rule, plan, settings = copied(tmp_path)
    await asyncio.to_thread(Path(plan.destination + ".source.json").unlink, missing_ok=True)
    await asyncio.to_thread(Path(rule.target).rename, tmp_path / "offline")
    answer = await check_admission(
        spec(),
        mappings=[rule],
        copy_settings=settings,
        node_name="Amish_Station",
        snapshot=fake_devices(),
        library=None,
        running=[],
    )
    assert answer.decision.value == "refuse"
    assert "was made before Eugene recorded its source" in answer.reason
    assert "Amish_Station" in answer.reason and rule.target in answer.reason
    assert "cannot be checked" in answer.reason


@pytest.mark.anyio
async def test_safe_mode_folders_keep_saved_file_age_and_mounts(tmp_path: Path) -> None:
    cache = LibraryFolderCache(tmp_path / "library_folders.json")
    cache.update(
        [FolderRecord("/models", ("/mnt/models",))], library_url="http://library", now=1000
    )
    before = cache.path.read_bytes()
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"folders": [], "safeMode": True})
    )
    library = LibraryFitClient("http://library", None, transport=transport)
    try:
        assert not await refresh(cache, library)
    finally:
        if library._own_client is not None:
            await library._own_client.aclose()
    assert cache.path.read_bytes() == before
    assert cache.age_seconds(now=1060) == 60
    assert cache.failure and "The Library is in safe mode" in cache.failure
    assert "keeps the folder list from" in cache.failure
