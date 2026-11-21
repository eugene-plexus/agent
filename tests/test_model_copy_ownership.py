"""agent#54: Eugene removes only the model copies it made.

The copy folder is a setting, and a person can point it at a folder that
already holds their own models. Cleanup used to treat every file there as a
copy, so `remove_unwanted`, `clear` and `evict_for_headroom` deleted the
person's files. A file is Eugene's copy only when the record Eugene writes
beside each copy (`<file>.source.json`, naming the file it copies) is there;
every other file is left alone and counted.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import model_copies as mc
from eugene_plexus_agent.model_paths import PathRule

from .test_library_folders import _install_library

MODEL = "/models/huihui-ai/Huihui-Qwen3-27B-Q6_K_L.gguf"
OTHER = "/models/qwen/Qwen3-1.7B-Q4_K_M.gguf"


def _settings(tmp_path: Path, min_free_gb: int = 0) -> mc.CopySettings:
    return mc.CopySettings(True, str(tmp_path / "copies"), min_free_gb * mc.GIB)


def _share(tmp_path: Path, *declared: str) -> PathRule:
    mount = tmp_path / "mount"
    for path in declared:
        local = mount / path[len("/models/") :]
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"x" * 1024)
    return PathRule(source="/models", target=str(mount))


def _copy(rule: PathRule, conf: mc.CopySettings, declared: str) -> mc.CopyPlan:
    plan = mc.plan_for(declared, [rule], conf)
    assert plan is not None
    mc.copy_file(plan, conf, mc.CopyState(plan.destination, 1024))
    return plan


def _persons_files(tmp_path: Path) -> list[Path]:
    """What a person keeps in the folder they chose: their own models, one in
    a folder of its own, and one at the very path a copy would take."""
    copies = tmp_path / "copies"
    files = [
        copies / "my-model.gguf",
        copies / "mine" / "fine-tune.gguf",
        copies / "huihui-ai" / "notes.txt",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mine")
    return files


# --- the three ways cleanup deletes ---------------------------------------------


def test_remove_unwanted_leaves_a_persons_files_and_removes_eugenes_copy(tmp_path: Path) -> None:
    rule = _share(tmp_path, MODEL, OTHER)
    conf = _settings(tmp_path)
    kept = _copy(rule, conf, MODEL)
    gone = _copy(rule, conf, OTHER)
    mine = _persons_files(tmp_path)

    result = mc.remove_unwanted(conf.directory, [kept.destination])

    assert result.deleted == [gone.destination]
    assert not os.path.exists(gone.destination)
    assert not os.path.exists(gone.destination + mc.SOURCE_SUFFIX)
    assert os.path.exists(kept.destination)
    assert all(path.read_bytes() == b"mine" for path in mine)
    assert result.left_alone == len(mine)


def test_clear_leaves_a_persons_files_and_skips_a_copy_in_use(tmp_path: Path) -> None:
    rule = _share(tmp_path, MODEL, OTHER)
    conf = _settings(tmp_path)
    held = _copy(rule, conf, MODEL)
    gone = _copy(rule, conf, OTHER)
    mine = _persons_files(tmp_path)

    result = mc.clear(conf.directory, in_use={held.destination: "qwen-27b"})

    assert result.deleted == [gone.destination]
    assert [s.runtime for s in result.skipped] == ["qwen-27b"]
    assert os.path.exists(held.destination)
    assert all(path.read_bytes() == b"mine" for path in mine)


def test_eviction_leaves_a_persons_files_however_full_the_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rule = _share(tmp_path, MODEL)
    conf = _settings(tmp_path, min_free_gb=50)
    plan = _copy(rule, mc.CopySettings(True, conf.directory, 0), MODEL)
    mine = _persons_files(tmp_path)
    for path in mine:  # older than the copy, so they would go first
        os.utime(path, (1, 1))
    monkeypatch.setattr(mc, "free_bytes", lambda _d: 1 * mc.GIB)

    result = mc.evict_for_headroom(conf.directory, conf)

    assert result.deleted == [plan.destination]
    assert all(path.read_bytes() == b"mine" for path in mine)


def test_eviction_still_skips_a_copy_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rule = _share(tmp_path, MODEL)
    conf = _settings(tmp_path, min_free_gb=50)
    plan = _copy(rule, mc.CopySettings(True, conf.directory, 0), MODEL)
    monkeypatch.setattr(mc, "free_bytes", lambda _d: 1 * mc.GIB)

    result = mc.evict_for_headroom(conf.directory, conf, in_use={plan.destination: "q"})

    assert result.deleted == [] and [s.runtime for s in result.skipped] == ["q"]
    assert os.path.exists(plan.destination)


def test_only_files_with_a_record_naming_their_source_are_copies(tmp_path: Path) -> None:
    copies = tmp_path / "copies"
    copies.mkdir()
    for name, record in (
        ("proven.gguf", {"source": "/models/proven.gguf"}),
        ("no-record.gguf", None),
        ("empty-source.gguf", {"source": ""}),
        ("not-json.gguf", "not json"),
        ("a-list.gguf", ["source"]),
    ):
        (copies / name).write_bytes(b"x")
        if record is not None:
            text = record if isinstance(record, str) else json.dumps(record)
            (copies / (name + mc.SOURCE_SUFFIX)).write_text(text, encoding="utf-8")
    (copies / ("x.gguf" + mc.PARTIAL_SUFFIX)).write_bytes(b"x")

    assert [Path(p).name for p in mc.copies_on_disk(str(copies))] == ["proven.gguf"]
    assert mc.files_not_copied(str(copies)) == 4


# --- a copy never takes a person's file over ------------------------------------


def test_a_persons_file_at_a_copys_path_is_never_overwritten(tmp_path: Path) -> None:
    rule = _share(tmp_path, MODEL)
    conf = _settings(tmp_path)
    plan = mc.plan_for(MODEL, [rule], conf)
    assert plan is not None
    Path(plan.destination).parent.mkdir(parents=True)
    Path(plan.destination).write_bytes(b"mine")

    with pytest.raises(mc.CopyAborted) as caught:
        mc.copy_file(plan, conf, mc.CopyState(plan.destination, 1024))

    assert plan.destination in str(caught.value)
    assert Path(plan.destination).read_bytes() == b"mine"
    assert not os.path.exists(plan.destination + mc.SOURCE_SUFFIX)
    assert not os.path.exists(plan.partial)


def test_an_identical_file_is_used_but_never_claimed(tmp_path: Path) -> None:
    """A person who copied the model with its timestamp (Explorer keeps it)
    has a file Eugene cannot tell from its own copy by size and time. It may
    be opened, since it is the same file, but it never gets a record."""
    rule = _share(tmp_path, MODEL)
    conf = _settings(tmp_path)
    plan = mc.plan_for(MODEL, [rule], conf)
    assert plan is not None
    Path(plan.destination).parent.mkdir(parents=True)
    shutil.copy2(plan.source, plan.destination)

    assert mc.resolve_local_path(MODEL, [rule], conf).is_copy
    mc.copy_file(plan, conf, mc.CopyState(plan.destination, 1024))
    assert not os.path.exists(plan.destination + mc.SOURCE_SUFFIX)

    mc.remove_unwanted(conf.directory, [])
    mc.clear(conf.directory)
    assert os.path.exists(plan.destination)


def test_each_file_of_a_set_is_marked_before_it_takes_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy stopped between two shards leaves no unmarked whole file."""
    mount = tmp_path / "mount"
    mount.mkdir()
    first = mount / "model-00001-of-00002.gguf"
    second = mount / "model-00002-of-00002.gguf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    conf = mc.CopySettings(True, str(tmp_path / "copies"), 0, folder_roots=(str(mount),))
    mc._members_cache.clear()
    plan = mc.plan_for(str(first), [], conf)
    assert plan is not None and len(plan.files) == 2
    real = mc._copy_one

    def stop_at_the_second(member: mc.CopyMember, *args: Any, **kwargs: Any) -> int:
        if member.source == str(second):
            raise mc.CopyAborted("the copy was cancelled")
        return real(member, *args, **kwargs)

    monkeypatch.setattr(mc, "_copy_one", stop_at_the_second)
    with pytest.raises(mc.CopyAborted):
        mc.copy_file(plan, conf, mc.CopyState(plan.destination, 11))

    assert os.path.exists(plan.files[0].destination + mc.SOURCE_SUFFIX)
    assert mc.copies_on_disk(conf.directory) == [plan.files[0].destination]
    assert mc.files_not_copied(conf.directory) == 0


def test_a_record_that_cannot_be_written_leaves_no_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rule = _share(tmp_path, MODEL)
    conf = _settings(tmp_path)
    plan = mc.plan_for(MODEL, [rule], conf)
    assert plan is not None
    monkeypatch.setattr(mc, "_write_source", lambda *_a: "disk full")

    with pytest.raises(mc.CopyAborted) as caught:
        mc.copy_file(plan, conf, mc.CopyState(plan.destination, 1024))

    assert "disk full" in str(caught.value)
    assert not os.path.exists(plan.destination)
    assert not os.path.exists(plan.partial)


# --- empty folders --------------------------------------------------------------


def test_only_folders_eugene_emptied_are_removed(tmp_path: Path) -> None:
    rule = _share(tmp_path, OTHER)
    conf = _settings(tmp_path)
    gone = _copy(rule, conf, OTHER)
    copies = Path(conf.directory or "")
    persons_empty = copies / "my-empty-folder"
    persons_empty.mkdir()
    shared = copies / "qwen" / "mine.gguf"
    other_copy_folder = Path(gone.destination).parent

    mc.remove_unwanted(conf.directory, [])

    assert not other_copy_folder.exists(), "the folder the copy emptied goes"
    assert persons_empty.is_dir(), "a folder Eugene did not empty stays"
    assert copies.is_dir(), "the copy folder itself stays"

    # A folder that holds a person's file stays when Eugene's copy leaves it.
    _copy(rule, conf, OTHER)
    shared.write_bytes(b"mine")
    result = mc.remove_unwanted(conf.directory, [])
    assert result.deleted == [gone.destination]
    assert shared.read_bytes() == b"mine"


def test_the_copy_folder_stays_when_its_last_copy_goes(tmp_path: Path) -> None:
    copies = tmp_path / "copies"
    copies.mkdir()
    (copies / "m.gguf").write_bytes(b"x")
    (copies / ("m.gguf" + mc.SOURCE_SUFFIX)).write_text('{"source": "/m.gguf"}', "utf-8")

    result = mc.clear(str(copies))

    assert result.deleted == [str(copies / "m.gguf")]
    assert copies.is_dir() and list(copies.iterdir()) == []


def test_the_supervisors_reconcile_and_clear_leave_a_persons_files(tmp_path: Path) -> None:
    """The two calls the supervisor makes, as the review reproduced them."""
    from eugene_plexus_agent.runtimes import RuntimeSupervisor

    rule = _share(tmp_path, MODEL)
    conf = _settings(tmp_path)
    config: dict[str, Any] = {
        "modelCopyEnabled": True,
        "modelCopyDir": conf.directory,
        "modelCopyMinFreeGb": 0,
    }
    sup = RuntimeSupervisor(get_config=config.get, inherited_rules=lambda: [rule])
    plan = _copy(rule, conf, MODEL)
    mine = _persons_files(tmp_path)

    sup.reconcile_copies([])
    result = sup.clear_copies()

    assert not Path(plan.destination).exists()
    assert result.left_alone == len(mine)
    assert all(path.read_bytes() == b"mine" for path in mine)


# --- the setting ----------------------------------------------------------------


def test_overlap_names_the_library_folder_and_how(tmp_path: Path) -> None:
    library = tmp_path / "models"
    roots = [str(library)]
    assert mc.library_overlap(str(library), roots) == f"is the Library folder {library}"
    assert mc.library_overlap(str(library / "copies"), roots) == (
        f"is inside the Library folder {library}"
    )
    assert mc.library_overlap(str(tmp_path), roots) == f"holds the Library folder {library}"
    assert mc.library_overlap(str(tmp_path / "copies"), roots) is None
    assert mc.library_overlap(str(tmp_path / "models-copies"), roots) is None


def _library_at(client: TestClient, tmp_path: Path) -> Path:
    library = tmp_path / "models"
    library.mkdir()
    _install_library(client, [{"path": str(library), "mounts": []}])
    return library


@pytest.mark.parametrize(
    ("where", "how"),
    [("models", "is the"), ("models/copies", "is inside the"), (".", "holds the")],
)
def test_patch_refuses_a_copy_folder_that_overlaps_a_library_folder(
    authed_client: TestClient, tmp_path: Path, where: str, how: str
) -> None:
    library = _library_at(authed_client, tmp_path)
    chosen = os.path.normpath(str(tmp_path / where))

    result = authed_client.patch(
        "/v1/config", json={"modelCopyDir": chosen, "modelCopyEnabled": True}
    ).json()

    rejected = {r["key"]: r["message"] for r in result["rejected"]}
    assert f"{how} Library folder {library}" in rejected["modelCopyDir"]
    assert "modelCopyEnabled" in result["applied"]
    assert authed_client.get("/v1/config").json().get("modelCopyDir") is None


def test_patch_refuses_a_copy_folder_inside_this_machines_mount_of_a_folder(
    authed_client: TestClient, tmp_path: Path
) -> None:
    mount = tmp_path / "mnt" / "models"
    mount.mkdir(parents=True)
    _install_library(authed_client, [{"path": "/library-elsewhere", "mounts": []}])
    result = authed_client.patch(
        "/v1/config",
        json={
            "pathMappings": [{"from": "/library-elsewhere", "to": str(mount)}],
            "modelCopyDir": str(mount / "copies"),
        },
    ).json()

    rejected = {r["key"]: r["message"] for r in result["rejected"]}
    assert f"is inside the Library folder {mount}" in rejected["modelCopyDir"]


def test_patch_takes_a_copy_folder_beside_the_library(
    authed_client: TestClient, tmp_path: Path
) -> None:
    _library_at(authed_client, tmp_path)
    chosen = str(tmp_path / "copies")

    result = authed_client.patch("/v1/config", json={"modelCopyDir": chosen}).json()

    assert result["rejected"] == [] and "modelCopyDir" in result["applied"]


def test_the_setting_counts_the_files_that_are_not_eugenes(
    authed_client: TestClient, tmp_path: Path
) -> None:
    rule = _share(tmp_path, MODEL)
    conf = _settings(tmp_path)
    _copy(rule, conf, MODEL)
    _persons_files(tmp_path)
    authed_client.patch("/v1/config", json={"modelCopyDir": conf.directory})

    body = authed_client.get("/v1/config/schema").json()
    field = next(f for f in body["fields"] if f["key"] == "modelCopyDir")

    assert (
        "3 files here are not marked as Eugene's copies. Eugene leaves them alone."
        in (field["status"]["text"])
    )
    assert "may delete anything" not in field["description"]
