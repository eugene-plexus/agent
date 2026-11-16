"""Row 15 (DI2, call DI-C4): the files of the prompt cache on disk.

Where a launch keeps them, what makes a file belong to one launch only, and the limits:
3 days after the last use, the least recently used first above the cap, all of them when
the setting is off, a runtime's with the runtime. Each test here fails without the code it
names.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from eugene_plexus_agent import disk_prompt_cache as dpc

GIB = 1024**3
ON = dpc.Settings(enabled=True, days=3, cap_bytes=20 * GIB)


def _model(tmp_path: Path, name: str = "m.gguf", body: bytes = b"GGUF-one") -> Path:
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _argv(model: Path, *rest: str) -> list[str]:
    return [
        "/engines/b1/llama-server",
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "18100",
        "--alias",
        "m",
        *rest,
    ]


def _plan(tmp_path: Path, argv: list[str], **kwargs: object) -> tuple[list[str], dpc.Launch | None]:
    options: dict[str, object] = {
        "runtime": "qwen",
        "host": "127.0.0.1",
        "argv": argv,
        "env": {},
        "extra_args": [],
        "root": tmp_path / "prompt-cache",
        "settings": ON,
        "search": False,
        "offered": None,
    }
    options.update(kwargs)
    return dpc.plan(**options)  # type: ignore[arg-type]


def _file(folder: Path, name: str, size: int, age_seconds: float = 0.0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"x" * size)
    when = time.time() - age_seconds
    os.utime(path, (when, when))
    return path


# --- where a launch keeps its files ------------------------------------------------------


def test_a_launch_saves_into_its_own_folder_before_the_extra_arguments(tmp_path: Path) -> None:
    model = _model(tmp_path)
    argv = [*_argv(model), "--threads", "4"]
    planned, launch = _plan(tmp_path, argv, extra_args=["--threads", "4"])
    assert launch is not None and launch.note is None and launch.folder is not None
    root = tmp_path / "prompt-cache"
    assert launch.folder.parent == root / "qwen"
    assert launch.folder.is_dir()
    # The expert's own arguments stay last, so one of theirs still wins.
    assert planned[-2:] == ["--threads", "4"]
    assert planned[-4:-2] == [dpc.FLAG, str(launch.folder)]
    assert dpc.folder_of(planned) == launch.folder
    assert dpc.is_managed(root, launch.folder)
    if os.name != "nt":
        assert (root.stat().st_mode & 0o777) == 0o700
        assert (launch.folder.stat().st_mode & 0o777) == 0o700


def test_where_the_engine_listens_is_not_part_of_the_key(tmp_path: Path) -> None:
    model = _model(tmp_path)
    one = dpc.launch_key(_argv(model), {})
    moved = [*_argv(model)]
    moved[moved.index("--port") + 1] = "18200"
    moved[moved.index("--alias") + 1] = "renamed"
    assert dpc.launch_key(moved, {"LLAMA_ARG_PORT": "1"}) == one


@pytest.mark.parametrize(
    "change",
    ["model", "program", "context", "cache type", "slots", "variable"],
)
def test_anything_that_shapes_the_state_changes_the_key(tmp_path: Path, change: str) -> None:
    model = _model(tmp_path)
    program = _model(tmp_path, "llama-server", b"build one")
    base = _argv(model, "--ctx-size", "8192", "--parallel", "4")
    base[0] = str(program)
    before = dpc.launch_key(base, {})
    after_argv, env = list(base), {}
    if change == "model":
        # The same path, another file: a model replaced in place is another model.
        time.sleep(0.01)
        model.write_bytes(b"GGUF-two-longer")
    elif change == "program":
        program.write_bytes(b"build two, longer")
    elif change == "context":
        after_argv[after_argv.index("--ctx-size") + 1] = "16384"
    elif change == "cache type":
        after_argv += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    elif change == "slots":
        after_argv[after_argv.index("--parallel") + 1] = "2"
    else:
        env = {"LLAMA_ARG_KV_UNIFIED": "1"}
    assert dpc.launch_key(after_argv, env) != before


@pytest.mark.parametrize("spelling", ["--model=", "LLAMA_ARG_MODEL"])
def test_a_model_named_any_way_is_part_of_the_key(tmp_path: Path, spelling: str) -> None:
    model = _model(tmp_path)

    def key() -> str:
        if spelling == "--model=":
            return dpc.launch_key(["/engines/llama-server", f"--model={model}"], {})
        return dpc.launch_key(["/engines/llama-server"], {"LLAMA_ARG_MODEL": str(model)})

    before = key()
    model.write_bytes(b"GGUF-two-longer")
    assert key() != before


def test_a_file_replaced_in_place_with_the_same_size_and_time_is_another_file(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path, body=b"GGUF-one")
    argv = ["/engines/llama-server", "--model", str(model)]
    stamp = model.stat().st_mtime_ns
    before = dpc.launch_key(argv, {})
    other = tmp_path / "other.gguf"
    other.write_bytes(b"GGUF-two")
    os.utime(other, ns=(stamp, stamp))
    os.replace(other, model)
    assert model.stat().st_size == 8 and model.stat().st_mtime_ns == stamp
    assert dpc.launch_key(argv, {}) != before


def test_a_search_model_keeps_no_cache(tmp_path: Path) -> None:
    argv = _argv(_model(tmp_path), "--embeddings")
    assert _plan(tmp_path, argv, search=True) == (argv, None)


def test_off_passes_nothing_and_says_so(tmp_path: Path) -> None:
    argv = _argv(_model(tmp_path))
    planned, launch = _plan(tmp_path, argv, settings=dpc.Settings(False, 3, 20 * GIB))
    assert planned == argv
    assert launch == dpc.Launch(None, dpc.NOTE_OFF)


@pytest.mark.parametrize("where", ["spec", "extraArgs", "extraArgs=", "env"])
def test_an_engine_other_machines_can_reach_saves_nothing(tmp_path: Path, where: str) -> None:
    """Review B1: llama-server takes the last `--host`, and `extraArgs` come last; with no
    `--host` at all it reads LLAMA_ARG_HOST."""
    argv = _argv(_model(tmp_path))
    extra: list[str] = []
    env: dict[str, str] = {}
    if where == "spec":
        argv[argv.index("--host") + 1] = "0.0.0.0"
    elif where == "extraArgs":
        extra = ["--host", "0.0.0.0"]
    elif where == "extraArgs=":
        extra = ["--host=0.0.0.0"]
    else:
        index = argv.index("--host")
        del argv[index : index + 2]
        env = {"LLAMA_ARG_HOST": "0.0.0.0"}
    argv += extra
    planned, launch = _plan(tmp_path, argv, extra_args=extra, env=env)
    assert planned == argv
    assert launch is not None and launch.folder is None
    assert "0.0.0.0" in (launch.note or "")
    assert not (tmp_path / "prompt-cache").exists()


def test_a_later_loopback_host_still_saves(tmp_path: Path) -> None:
    argv = [*_argv(_model(tmp_path)), "--host", "0.0.0.0", "--host=127.0.0.1"]
    _, launch = _plan(tmp_path, argv, extra_args=argv[-3:])
    assert launch is not None and launch.folder is not None


@pytest.mark.parametrize(
    ("extra", "env", "cause"),
    [
        (["--no-slots"], {}, "--no-slots"),
        ([], {"LLAMA_ARG_ENDPOINT_SLOTS": "0"}, "--no-slots"),
        (["--api-key", "secret"], {}, "--api-key"),
        (["--api-key-file=/k"], {}, "--api-key"),
        ([], {"LLAMA_ARG_API_KEY": "secret"}, "--api-key"),
    ],
)
def test_a_command_line_on_which_every_save_fails_keeps_none(
    tmp_path: Path, extra: list[str], env: dict[str, str], cause: str
) -> None:
    """Review S7: no slot routes, or slot routes behind a key, and the runtime says why."""
    argv = [*_argv(_model(tmp_path)), *extra]
    planned, launch = _plan(tmp_path, argv, extra_args=extra, env=env)
    assert planned == argv
    assert launch is not None and launch.folder is None
    assert cause in (launch.note or "")
    assert "secret" not in (launch.note or "")


def test_slots_turned_back_on_still_save(tmp_path: Path) -> None:
    extra = ["--no-slots", "--slots"]
    _, launch = _plan(tmp_path, [*_argv(_model(tmp_path)), *extra], extra_args=extra)
    assert launch is not None and launch.folder is not None


def test_a_folder_that_cannot_be_made_names_the_cause_not_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: Path, root: Path) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(dpc, "make_private_dir", refuse)
    _, launch = _plan(tmp_path, _argv(_model(tmp_path)))
    assert launch is not None and launch.folder is None
    assert "Permission denied" in (launch.note or "")
    assert str(tmp_path) not in (launch.note or "") and "prompt-cache" not in (launch.note or "")


@pytest.mark.parametrize("where", ["extraArgs", "env"])
def test_a_folder_set_by_hand_is_left_to_the_expert(tmp_path: Path, where: str) -> None:
    own = str(tmp_path / "mine")
    extra = [dpc.FLAG, own] if where == "extraArgs" else []
    env = {dpc.VARIABLE: own} if where == "env" else {}
    argv = [*_argv(_model(tmp_path)), *extra]
    planned, launch = _plan(tmp_path, argv, extra_args=extra, env=env)
    assert planned == argv
    assert launch is not None and launch.folder is None
    assert "by hand" in (launch.note or "")


def test_a_build_without_the_flag_is_told(tmp_path: Path) -> None:
    argv = _argv(_model(tmp_path))
    planned, launch = _plan(tmp_path, argv, offered=frozenset({"--model"}))
    assert planned == argv
    assert launch is not None and dpc.FLAG in (launch.note or "")


def test_settings_degrade_to_the_defaults() -> None:
    values = {dpc.ENABLED_KEY: None, dpc.DAYS_KEY: "three", dpc.CAP_KEY: -4}
    found = dpc.settings_from_config(values.get)
    assert found == dpc.Settings(enabled=True, days=3, cap_bytes=20 * GIB)
    assert dpc.settings_from_config({dpc.ENABLED_KEY: False}.get).enabled is False


def test_an_odd_runtime_name_gets_a_hashed_folder(tmp_path: Path) -> None:
    assert dpc.runtime_folder(tmp_path, "qwen-8b").name == "qwen-8b"
    for name in ("..", "a/b", "c:\\x", "spaced name"):
        folder = dpc.runtime_folder(tmp_path, name)
        assert folder.parent == tmp_path and folder.name.startswith("r-")


# --- the files ---------------------------------------------------------------------------


def test_a_save_becomes_a_file_only_when_it_finished(tmp_path: Path) -> None:
    folder = tmp_path / "f"
    name = dpc.new_name(2)
    _file(folder, dpc.part_name(name), 10)
    kept = dpc.finish(folder, name)
    assert kept == folder / f"{name}.bin" and kept.exists()
    assert not (folder / dpc.part_name(name)).exists()
    assert [item.slot for item in dpc.saved(folder)] == [2]
    if os.name != "nt":
        assert (kept.stat().st_mode & 0o777) == 0o600


def test_prune_keeps_the_newest_one_per_slot_and_no_unfinished_save(tmp_path: Path) -> None:
    folder = tmp_path / "f"
    for age in (50, 40, 30, 20, 10):
        _file(folder, f"{dpc.new_name(0)}.bin", 1, age_seconds=age)
        time.sleep(0.002)
    _file(folder, "1-abcdef-s0.part", 1)
    assert dpc.prune(folder, keep=2) == 3
    left = dpc.saved(folder)
    assert len(left) == 2
    assert all(time.time() - item.used < 25 for item in left)
    assert not list(folder.glob("*.part"))


def test_restore_puts_a_file_back_in_the_slot_it_came_from(tmp_path: Path) -> None:
    folder = tmp_path / "f"
    _file(folder, "3-aaaaaa-s1.bin", 1, age_seconds=1)
    _file(folder, "2-bbbbbb-s1.bin", 1, age_seconds=2)
    _file(folder, "1-cccccc-s3.bin", 1, age_seconds=3)
    _file(folder, "0-dddddd-s0.bin", 1, age_seconds=4)
    order = dpc.restore_order(dpc.saved(folder), slots=2)
    # Newest two; slot 1 is its own, and the second slot-1 file takes the free slot 0.
    assert [(slot, item.path.name) for slot, item in order] == [
        (0, "2-bbbbbb-s1.bin"),
        (1, "3-aaaaaa-s1.bin"),
    ]


# --- the limits --------------------------------------------------------------------------


def test_a_file_unused_for_its_days_is_deleted(tmp_path: Path) -> None:
    root = tmp_path / "prompt-cache"
    fresh = _file(root / "qwen" / "k1", "1-aaaaaa-s0.bin", 10, age_seconds=2 * 86400)
    old = _file(root / "qwen" / "k1", "1-bbbbbb-s1.bin", 10, age_seconds=3 * 86400 + 60)
    result = dpc.sweep(root, ON, declared=["qwen"])
    assert fresh.exists() and not old.exists()
    assert result.expired == 1 and result.bytes_freed == 10


def test_above_the_cap_the_least_recently_used_go_first(tmp_path: Path) -> None:
    root = tmp_path / "prompt-cache"
    small = dpc.Settings(enabled=True, days=3, cap_bytes=25)
    oldest = _file(root / "a" / "k", "1-aaaaaa-s0.bin", 10, age_seconds=300)
    middle = _file(root / "b" / "k", "1-bbbbbb-s0.bin", 10, age_seconds=200)
    newest = _file(root / "a" / "k", "1-cccccc-s1.bin", 10, age_seconds=100)
    result = dpc.sweep(root, small, declared=["a", "b"])
    assert not oldest.exists() and middle.exists() and newest.exists()
    assert result.over_cap == 1


def test_off_deletes_every_file_and_keeps_the_folders_in_use(tmp_path: Path) -> None:
    root = tmp_path / "prompt-cache"
    used = root / "qwen" / "k1"
    one = _file(used, "1-aaaaaa-s0.bin", 10)
    two = _file(root / "qwen" / "k2", "1-bbbbbb-s0.bin", 10)
    dpc.sweep(root, dpc.Settings(False, 3, 20 * GIB), declared=["qwen"], active=[used])
    assert not one.exists() and not two.exists()
    assert used.is_dir()
    assert not (root / "qwen" / "k2").exists()


def test_a_runtime_that_is_gone_loses_its_folder(tmp_path: Path) -> None:
    root = tmp_path / "prompt-cache"
    gone = _file(root / "old" / "k", "1-aaaaaa-s0.bin", 10)
    kept = _file(root / "qwen" / "k", "1-bbbbbb-s0.bin", 10)
    result = dpc.sweep(root, ON, declared=["qwen"])
    assert not gone.exists() and not (root / "old").exists() and kept.exists()
    assert result.unused == 1
    dpc.remove_runtime(root, "qwen")
    assert not (root / "qwen").exists()


def test_an_unfinished_save_goes_unless_an_engine_is_writing_it(tmp_path: Path) -> None:
    root = tmp_path / "prompt-cache"
    writing = root / "qwen" / "k1"
    young = _file(writing, "1-aaaaaa-s0.part", 10)
    stale = _file(writing, "1-bbbbbb-s1.part", 10, age_seconds=dpc.STALE_PART_SECONDS + 60)
    orphan = _file(root / "qwen" / "k2", "1-cccccc-s0.part", 10)
    dpc.sweep(root, ON, declared=["qwen"], active=[writing])
    assert young.exists() and not stale.exists() and not orphan.exists()


@pytest.mark.skipif(os.name == "nt", reason="symlinks need a privilege on Windows")
def test_a_link_in_the_folder_is_never_followed(tmp_path: Path) -> None:
    root = tmp_path / "prompt-cache"
    outside = _file(tmp_path / "elsewhere", "1-aaaaaa-s0.bin", 10, age_seconds=30 * 86400)
    (root / "qwen").mkdir(parents=True)
    (root / "qwen" / "k").symlink_to(outside.parent, target_is_directory=True)
    (root / "old").symlink_to(outside.parent, target_is_directory=True)
    dpc.sweep(root, dpc.Settings(False, 3, 0), declared=["qwen"])
    dpc.clear(root)
    assert outside.exists()


def test_clear_deletes_every_file_and_counts_them(tmp_path: Path) -> None:
    root = tmp_path / "prompt-cache"
    _file(root / "qwen" / "k1", "1-aaaaaa-s0.bin", 10)
    _file(root / "qwen" / "k1", "1-aaaaaa-s1.part", 5)
    _file(root / "gemma" / "k2", "1-bbbbbb-s0.bin", 30)
    assert dpc.usage(root, ["qwen", "gemma"]).by_runtime == {"qwen": (1, 10), "gemma": (1, 30)}
    cleared = dpc.clear(root)
    assert (cleared.conversations, cleared.bytes_freed, cleared.not_deleted) == (2, 40, 0)
    assert dpc.usage(root, ["qwen"]).conversations == 0
    assert (root / "qwen" / "k1").is_dir()


# --- review S3: every file in the folder is the cache's ---------------------------------


def test_a_file_saved_under_another_name_counts_and_goes(tmp_path: Path) -> None:
    root = tmp_path / "prompt-cache"
    ours = _file(root / "qwen" / "k1", "1-aaaaaa-s0.bin", 10)
    stray = _file(root / "qwen" / "k1", "anything.dat", 40)
    assert dpc.usage(root, ["qwen"]).bytes == 50
    # Above the cap the oldest goes, whatever its name.
    os.utime(stray, (time.time() - 300, time.time() - 300))
    dpc.sweep(root, dpc.Settings(True, 3, 30), declared=["qwen"])
    assert ours.exists() and not stray.exists()
    stray = _file(root / "qwen" / "k1", "anything.dat", 40, age_seconds=4 * 86400)
    dpc.sweep(root, ON, declared=["qwen"])
    assert not stray.exists()
    stray = _file(root / "qwen" / "k1", "anything.dat", 40)
    cleared = dpc.clear(root)
    assert cleared.conversations == 2 and not stray.exists() and not ours.exists()
    stray = _file(root / "qwen" / "k1", "anything.dat", 40)
    dpc.sweep(root, dpc.Settings(False, 3, 20 * GIB), declared=["qwen"])
    assert not stray.exists()


def test_prune_keeps_only_the_newest_saves(tmp_path: Path) -> None:
    folder = tmp_path / "f"
    ours = _file(folder, "1-aaaaaa-s0.bin", 10)
    stray = _file(folder, "x.dat", 10)
    dpc.prune(folder, keep=4)
    assert ours.exists() and not stray.exists()


def test_a_file_the_system_kept_is_not_counted_as_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "prompt-cache"
    old = _file(root / "qwen" / "k1", "1-aaaaaa-s0.bin", 10, age_seconds=4 * 86400)
    monkeypatch.setattr(dpc, "discard", lambda path: False)
    result = dpc.sweep(root, ON, declared=["qwen"])
    assert old.exists()
    assert (result.expired, result.bytes_freed) == (0, 0)


def test_the_size_of_a_next_save_is_estimated_from_the_last(tmp_path: Path) -> None:
    folder = tmp_path / "f"
    name = dpc.new_name(0)
    _file(folder, dpc.part_name(name), 1000)
    kept = dpc.finish(folder, name, tokens=100)
    assert kept is not None and kept.name.endswith("-s0-t100.bin")
    assert dpc.saved(folder)[0].tokens == 100
    assert dpc.bytes_per_token(folder) == 10.0
