"""A node's own copy of the models its runtimes point at.

Design: `docs/design/node-local-model-copy.md`, §13 step 3. These cover
the copier as a thing in itself -- the set, the naming, the headroom,
the partial, the eviction. Whether anything CALLS it is step 4's
question, and a separate one on purpose: a component test is not a
wiring test, a lesson S7 produced twice in one slice.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_agent import model_copies as mc
from eugene_plexus_agent.model_paths import PathRule

NAS = PathRule(source="/models", target="Z:\\models")

MODEL = "/models/huihui-ai/Huihui-Qwen3-27B-Q6_K_L.gguf"
OTHER = "/models/qwen/Qwen3-1.7B-Q4_K_M.gguf"


def settings(tmp_path: Path, *, enabled: bool = True, min_free_gb: int = 0) -> mc.CopySettings:
    return mc.CopySettings(
        enabled=enabled,
        directory=str(tmp_path / "copies"),
        min_free_bytes=min_free_gb * mc.GIB,
    )


def share(tmp_path: Path, declared: str, size: int = 1024) -> PathRule:
    """Make `declared` real under a fake mount, and return the rule that
    points at it -- the shape of a worker with a Library folder mounted."""
    mount = tmp_path / "mount"
    relative = declared[len("/models/") :]
    path = mount / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return PathRule(source="/models", target=str(mount))


# --- the set ------------------------------------------------------------------


def test_two_runtimes_over_two_models_want_two_copies(tmp_path: Path) -> None:
    rule = share(tmp_path, MODEL)
    share(tmp_path, OTHER)
    plans = mc.wanted([MODEL, OTHER], [rule], settings(tmp_path))
    assert len(plans) == 2


def test_four_replicas_of_one_model_want_one_copy(tmp_path: Path) -> None:
    """The assertion that distinguishes this design from the one first
    proposed (§2.2). M6's headline case is replicas: two `llama-server`
    processes over ONE GGUF, balanced least-busy. Keying a copy to the
    runtime would have copied 25 GB twice."""
    rule = share(tmp_path, MODEL)
    plans = mc.wanted([MODEL, MODEL, MODEL, MODEL], [rule], settings(tmp_path))
    assert len(plans) == 1


def test_nothing_is_wanted_while_the_toggle_is_off(tmp_path: Path) -> None:
    rule = share(tmp_path, MODEL)
    assert mc.wanted([MODEL], [rule], settings(tmp_path, enabled=False)) == {}


def test_a_toggle_with_no_directory_is_off_rather_than_an_error(tmp_path: Path) -> None:
    bare = mc.CopySettings(enabled=True, directory=None, min_free_bytes=0)
    assert bare.usable is False
    assert mc.wanted([MODEL], [share(tmp_path, MODEL)], bare) == {}


def test_a_model_already_inside_the_copy_directory_is_not_copied_onto_itself(
    tmp_path: Path,
) -> None:
    """Without this, the restart after a successful copy plans to copy
    the copy over itself -- and the first thing that copy does is
    truncate its own source."""
    copies = tmp_path / "copies"
    (copies / "x").mkdir(parents=True)
    (copies / "x" / "m.gguf").write_bytes(b"x" * 16)
    rule = PathRule(source="/models", target=str(copies))
    assert mc.plan_for("/models/x/m.gguf", [rule], settings(tmp_path)) is None


# --- naming -------------------------------------------------------------------


def test_a_copy_is_named_at_the_models_own_relative_path(tmp_path: Path) -> None:
    """§4.4: plain and predictable, so what is left after deleting this
    product is a folder of correctly-named GGUFs."""
    plan = mc.plan_for(MODEL, [NAS], settings(tmp_path))
    assert plan is not None
    assert Path(plan.destination) == (
        tmp_path / "copies" / "huihui-ai" / "Huihui-Qwen3-27B-Q6_K_L.gguf"
    )


def test_the_relative_path_is_taken_from_the_folder_not_the_whole_path(tmp_path: Path) -> None:
    assert mc.relative_name(MODEL, [NAS]) == "huihui-ai/Huihui-Qwen3-27B-Q6_K_L.gguf"


def test_a_declared_path_under_no_folder_cannot_escape_the_copy_directory(
    tmp_path: Path,
) -> None:
    """A drive letter is not a name. Without stripping the root,
    `os.path.join(copy_dir, 'D:', ...)` on Windows resolves to D: and
    the copy lands outside the directory the operator chose -- which is
    the one promise this feature makes about where it writes."""
    plan = mc.plan_for("D:\\models\\x.gguf", [], settings(tmp_path))
    assert plan is not None
    destination = Path(plan.destination).resolve()
    assert destination.is_relative_to((tmp_path / "copies").resolve())


# --- staleness ----------------------------------------------------------------


def test_a_copy_matching_size_and_mtime_is_current(tmp_path: Path) -> None:
    rule = share(tmp_path, MODEL)
    plan = mc.plan_for(MODEL, [rule], settings(tmp_path))
    assert plan is not None
    mc.copy_file(plan, settings(tmp_path), mc.CopyState(plan.destination, 1024))
    assert mc.copy_is_current(plan) is True


def test_a_copy_of_a_different_size_is_stale(tmp_path: Path) -> None:
    rule = share(tmp_path, MODEL)
    plan = mc.plan_for(MODEL, [rule], settings(tmp_path))
    assert plan is not None
    mc.copy_file(plan, settings(tmp_path), mc.CopyState(plan.destination, 1024))
    Path(plan.source).write_bytes(b"y" * 2048)
    assert mc.copy_is_current(plan) is False


def test_a_missing_copy_is_not_current(tmp_path: Path) -> None:
    rule = share(tmp_path, MODEL)
    plan = mc.plan_for(MODEL, [rule], settings(tmp_path))
    assert plan is not None
    assert mc.copy_is_current(plan) is False


def test_a_second_of_mtime_drift_is_not_staleness(tmp_path: Path) -> None:
    """SMB and FAT round timestamps. Re-copying 25 GB because a share
    reported a mtime one second off would be worse than the problem."""
    rule = share(tmp_path, MODEL)
    plan = mc.plan_for(MODEL, [rule], settings(tmp_path))
    assert plan is not None
    mc.copy_file(plan, settings(tmp_path), mc.CopyState(plan.destination, 1024))
    st = os.stat(plan.destination)
    os.utime(plan.destination, (st.st_atime, st.st_mtime + 1))
    assert mc.copy_is_current(plan) is True


# --- the copy itself ----------------------------------------------------------


def test_a_partial_copy_never_carries_the_final_name(tmp_path: Path, monkeypatch) -> None:
    """An engine opening a half-written GGUF fails somewhere unhelpful,
    minutes later, in a log nobody is reading."""
    rule = share(tmp_path, MODEL, size=mc.CHUNK_BYTES * 3)
    plan = mc.plan_for(MODEL, [rule], settings(tmp_path))
    assert plan is not None
    seen: list[bool] = []

    real_read = open

    def watched(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        handle = real_read(*args, **kwargs)
        seen.append(os.path.exists(plan.destination))
        return handle

    monkeypatch.setattr("builtins.open", watched)
    mc.copy_file(plan, settings(tmp_path), mc.CopyState(plan.destination, None))
    monkeypatch.undo()

    assert any(seen) is False, "the destination existed before the copy finished"
    assert os.path.exists(plan.destination)
    assert not os.path.exists(plan.partial)


def test_a_failed_copy_leaves_no_partial_behind(tmp_path: Path) -> None:
    rule = share(tmp_path, MODEL, size=mc.CHUNK_BYTES * 2)
    plan = mc.plan_for(MODEL, [rule], settings(tmp_path))
    assert plan is not None
    with pytest.raises(mc.CopyAborted):
        mc.copy_file(
            plan,
            settings(tmp_path),
            mc.CopyState(plan.destination, None),
            should_cancel=lambda: True,
        )
    assert not os.path.exists(plan.partial)
    assert not os.path.exists(plan.destination)


def test_a_headroom_breach_during_the_copy_aborts_and_cleans_up(
    tmp_path: Path, monkeypatch
) -> None:
    """A 25 GB transfer takes minutes and something else can fill the
    disk underneath it. The feature meant to make a node faster must not
    be the reason its disk filled up."""
    rule = share(tmp_path, MODEL, size=mc.CHUNK_BYTES * (mc.HEADROOM_CHECK_EVERY + 2))
    conf = mc.CopySettings(
        enabled=True, directory=str(tmp_path / "copies"), min_free_bytes=100 * mc.GIB
    )
    plan = mc.plan_for(MODEL, [rule], conf)
    assert plan is not None
    monkeypatch.setattr(mc, "free_bytes", lambda _d: 1 * mc.GIB)
    with pytest.raises(mc.CopyAborted) as caught:
        mc.copy_file(plan, conf, mc.CopyState(plan.destination, None))
    assert "free" in str(caught.value)
    assert not os.path.exists(plan.partial)
    assert not os.path.exists(plan.destination)


def test_progress_is_reported_as_the_bytes_move(tmp_path: Path) -> None:
    size = mc.CHUNK_BYTES * 3
    rule = share(tmp_path, MODEL, size=size)
    plan = mc.plan_for(MODEL, [rule], settings(tmp_path))
    assert plan is not None
    state = mc.CopyState(plan.destination, size)
    mc.copy_file(plan, settings(tmp_path), state)
    assert state.bytes_copied == size


def test_a_copy_carries_its_sources_timestamp(tmp_path: Path) -> None:
    """Otherwise `copy_is_current` compares the copy against the moment
    it was made and every restart re-copies the model."""
    rule = share(tmp_path, MODEL)
    plan = mc.plan_for(MODEL, [rule], settings(tmp_path))
    assert plan is not None
    mc.copy_file(plan, settings(tmp_path), mc.CopyState(plan.destination, 1024))
    assert abs(os.stat(plan.destination).st_mtime - os.stat(plan.source).st_mtime) < 1.0


# --- headroom before the copy -------------------------------------------------


def test_a_copy_that_would_eat_the_headroom_is_refused_before_it_starts(
    tmp_path: Path, monkeypatch
) -> None:
    conf = mc.CopySettings(
        enabled=True, directory=str(tmp_path / "copies"), min_free_bytes=50 * mc.GIB
    )
    plan = mc.plan_for(MODEL, [NAS], conf)
    assert plan is not None
    monkeypatch.setattr(mc, "free_bytes", lambda _d: 60 * mc.GIB)
    assert mc.headroom_shortfall(plan, conf, size=20 * mc.GIB) > 0
    assert mc.headroom_shortfall(plan, conf, size=5 * mc.GIB) <= 0


def test_a_disk_whose_free_space_cannot_be_read_is_allowed_to_try(
    tmp_path: Path, monkeypatch
) -> None:
    """`easy-default-expert-override`'s corollary: an eager refusal can
    be wrong and an explanation of a real failure cannot. A disk we
    cannot measure gets to fail honestly rather than be refused on a
    guess about the host."""
    conf = mc.CopySettings(
        enabled=True, directory=str(tmp_path / "copies"), min_free_bytes=50 * mc.GIB
    )
    plan = mc.plan_for(MODEL, [NAS], conf)
    assert plan is not None
    monkeypatch.setattr(mc, "free_bytes", lambda _d: None)
    assert mc.headroom_shortfall(plan, conf, size=900 * mc.GIB) == 0


# --- getting the space back ---------------------------------------------------


def test_a_deleted_runtime_takes_its_copy_with_it(tmp_path: Path) -> None:
    """The whole eviction policy for the ordinary case. No LRU, no
    timer: the set was never anything but a function of the runtime
    list, so a copy nothing points at is simply not wanted."""
    rule = share(tmp_path, MODEL)
    share(tmp_path, OTHER)
    conf = settings(tmp_path)
    for declared in (MODEL, OTHER):
        plan = mc.plan_for(declared, [rule], conf)
        assert plan is not None
        mc.copy_file(plan, conf, mc.CopyState(plan.destination, 1024))
    keep = [p.destination for p in mc.wanted([MODEL], [rule], conf).values()]

    result = mc.remove_unwanted(conf.directory, keep)

    assert len(result.deleted) == 1
    assert "Qwen3-1.7B" in result.deleted[0]
    assert os.path.exists(keep[0])


def test_clear_removes_everything_and_reports_what_it_could_not(tmp_path: Path) -> None:
    rule = share(tmp_path, MODEL)
    share(tmp_path, OTHER)
    conf = settings(tmp_path)
    plans = mc.wanted([MODEL, OTHER], [rule], conf)
    for plan in plans.values():
        mc.copy_file(plan, conf, mc.CopyState(plan.destination, 1024))
    held = next(iter(plans))

    result = mc.clear(conf.directory, in_use={held: "qwen-27b"})

    assert len(result.deleted) == 1
    assert [s.runtime for s in result.skipped] == ["qwen-27b"]
    assert os.path.exists(held), "clear must not stop a runtime to get at its file"
    assert result.bytes_freed == 1024


def test_clear_on_a_node_that_has_never_copied_anything_is_a_success(tmp_path: Path) -> None:
    result = mc.clear(str(tmp_path / "never"))
    assert result.deleted == [] and result.skipped == []


def test_eviction_takes_the_oldest_first_until_the_headroom_is_back(
    tmp_path: Path, monkeypatch
) -> None:
    rule = share(tmp_path, MODEL)
    share(tmp_path, OTHER)
    conf = mc.CopySettings(
        enabled=True, directory=str(tmp_path / "copies"), min_free_bytes=50 * mc.GIB
    )
    plans = mc.wanted([MODEL, OTHER], [rule], conf)
    ordered = []
    for plan in plans.values():
        mc.copy_file(
            plan,
            mc.CopySettings(True, conf.directory, 0),
            mc.CopyState(plan.destination, 1024),
        )
        ordered.append(plan.destination)
    os.utime(ordered[0], (time.time() - 10_000, time.time() - 10_000))

    calls = {"n": 0}

    def free(_d: str) -> int:
        # Below the headroom until one file has gone.
        calls["n"] += 1
        return 10 * mc.GIB if calls["n"] <= 1 else 60 * mc.GIB

    monkeypatch.setattr(mc, "free_bytes", free)
    result = mc.evict_for_headroom(conf.directory, conf)

    assert result.deleted == [ordered[0]], "the oldest copy should go first"
    assert os.path.exists(ordered[1])


def test_eviction_never_touches_a_copy_a_runtime_is_using(tmp_path: Path, monkeypatch) -> None:
    rule = share(tmp_path, MODEL)
    conf = mc.CopySettings(
        enabled=True, directory=str(tmp_path / "copies"), min_free_bytes=50 * mc.GIB
    )
    plan = mc.plan_for(MODEL, [rule], conf)
    assert plan is not None
    mc.copy_file(
        plan, mc.CopySettings(True, conf.directory, 0), mc.CopyState(plan.destination, 1024)
    )
    monkeypatch.setattr(mc, "free_bytes", lambda _d: 1 * mc.GIB)

    result = mc.evict_for_headroom(conf.directory, conf, in_use={plan.destination: "qwen-27b"})

    assert result.deleted == []
    assert [s.runtime for s in result.skipped] == ["qwen-27b"]
    assert os.path.exists(plan.destination)


def test_a_partial_is_never_reported_as_a_copy(tmp_path: Path) -> None:
    """`copies_on_disk` is what Clear and eviction walk. A partial is
    ours, but it is not a copy of anything and reporting one as deleted
    would tell the operator a model went away that never arrived."""
    copies = tmp_path / "copies"
    copies.mkdir()
    (copies / "m.gguf").write_bytes(b"x")
    (copies / ("m.gguf" + mc.PARTIAL_SUFFIX)).write_bytes(b"x")
    assert [Path(p).name for p in mc.copies_on_disk(str(copies))] == ["m.gguf"]


# --- the one seam -------------------------------------------------------------


def test_a_valid_copy_wins_over_the_mount(tmp_path: Path) -> None:
    rule = share(tmp_path, MODEL)
    conf = settings(tmp_path)
    plan = mc.plan_for(MODEL, [rule], conf)
    assert plan is not None
    mc.copy_file(plan, conf, mc.CopyState(plan.destination, 1024))

    resolved = mc.resolve_local_path(MODEL, [rule], conf)

    assert resolved.path == plan.destination
    assert resolved.source == "copy"


def test_a_stale_copy_does_not_win(tmp_path: Path) -> None:
    """Returning it would serve yesterday's weights from a path the
    operator cannot see in the declaration."""
    rule = share(tmp_path, MODEL)
    conf = settings(tmp_path)
    plan = mc.plan_for(MODEL, [rule], conf)
    assert plan is not None
    mc.copy_file(plan, conf, mc.CopyState(plan.destination, 1024))
    Path(plan.source).write_bytes(b"z" * 4096)

    resolved = mc.resolve_local_path(MODEL, [rule], conf)

    assert resolved.path == plan.source
    assert resolved.source != "copy"


def test_with_the_toggle_off_resolution_is_exactly_m11s(tmp_path: Path) -> None:
    """Turning it off reverts by itself, with no state to unwind (§4.2)
    -- including when a copy is still sitting on the disk."""
    rule = share(tmp_path, MODEL)
    on = settings(tmp_path)
    plan = mc.plan_for(MODEL, [rule], on)
    assert plan is not None
    mc.copy_file(plan, on, mc.CopyState(plan.destination, 1024))

    off = settings(tmp_path, enabled=False)
    resolved = mc.resolve_local_path(MODEL, [rule], off)

    assert resolved.path == plan.source
    assert resolved.source != "copy"


def test_settings_survive_a_config_file_someone_edited_by_hand(tmp_path: Path) -> None:
    values: dict[str, Any] = {
        mc.ENABLED_KEY: True,
        mc.DIR_KEY: "  " + str(tmp_path) + "  ",
        mc.MIN_FREE_GB_KEY: "not a number",
    }
    conf = mc.settings_from_config(values.get)
    assert conf.enabled is True
    assert conf.directory == str(tmp_path)
    assert conf.min_free_bytes == mc.DEFAULT_MIN_FREE_GB * mc.GIB


def test_a_directory_a_copy_is_about_to_fill_is_not_pruned(tmp_path: Path) -> None:
    """The live install's first boot with copying on, in one test.

    A copy creates its directory and then opens a temp file in it, and
    the reconcile that runs on the declaration list walks the same tree.
    Between those two instants the directory is empty and looks like
    scaffolding -- so it was deleted out from under the copy, which
    failed with `No such file or directory` on a path it had just
    created. A directory we are about to fill is not empty.
    """
    rule = share(tmp_path, MODEL)
    conf = settings(tmp_path)
    plan = mc.plan_for(MODEL, [rule], conf)
    assert plan is not None
    os.makedirs(os.path.dirname(plan.destination), exist_ok=True)

    mc.remove_unwanted(conf.directory, [plan.destination])

    assert os.path.isdir(os.path.dirname(plan.destination))


def test_an_empty_directory_nothing_wants_is_still_pruned(tmp_path: Path) -> None:
    conf = settings(tmp_path)
    stale = Path(conf.directory or "") / "someone-elses-folder"
    stale.mkdir(parents=True)

    mc.remove_unwanted(conf.directory, [])

    assert not stale.exists()
