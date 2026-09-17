"""Load progress: the bytes, and when there are honestly none.

The subject is `LoadProgressTracker`, whose whole job is to answer "is
this engine reading, or is it hung?" without ever drawing a bar it
cannot back. The negative cases are the load-bearing ones here — a bar
that appears when it should not is worse than no bar, because it looks
authoritative while being frozen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eugene_plexus_agent import process_io
from eugene_plexus_agent.process_io import SOURCE_WINDOWS, LoadProgressTracker


@pytest.fixture
def model(tmp_path: Path) -> Path:
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    return f


def _feed(monkeypatch: pytest.MonkeyPatch, readings: list[int | None]) -> None:
    """Make the platform counter return these values, in order."""
    seq = list(readings)

    def fake(pid: int) -> tuple[int, str] | None:
        value = seq.pop(0)
        return None if value is None else (value, SOURCE_WINDOWS)

    monkeypatch.setattr(process_io, "read_bytes", fake)


# --------------------------------------------------------------------------- #
# advancing bytes
# --------------------------------------------------------------------------- #


def test_advancing_bytes_report_progress_and_a_rate(
    monkeypatch: pytest.MonkeyPatch, model: Path
) -> None:
    _feed(monkeypatch, [0, 100_000_000])
    t = LoadProgressTracker()

    assert t.sample("r", 1, str(model), now=0.0) is None, "one reading cannot be a rate"
    progress = t.sample("r", 1, str(model), now=2.0)

    assert progress is not None
    assert progress.bytes_read == 100_000_000
    assert progress.bytes_per_second == pytest.approx(50_000_000)
    assert progress.total_bytes == 4096
    assert progress.source == SOURCE_WINDOWS


def test_the_denominator_survives_a_path_that_cannot_be_sized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A share that went away is exactly when a load is worth watching,
    so progress is still reported — without a percentage."""
    _feed(monkeypatch, [0, 1_000])
    t = LoadProgressTracker()
    t.sample("r", 1, "//gone/away/model.gguf", now=0.0)

    progress = t.sample("r", 1, "//gone/away/model.gguf", now=1.0)

    assert progress is not None
    assert progress.total_bytes is None
    assert progress.bytes_read == 1_000


# --------------------------------------------------------------------------- #
# the negative cases — each one would otherwise draw a frozen bar
# --------------------------------------------------------------------------- #


def test_a_memory_mapped_load_reports_nothing(monkeypatch: pytest.MonkeyPatch, model: Path) -> None:
    """THE case this is shaped around.

    llama.cpp mmaps by default, and still reads a GGUF's header through
    ordinary reads — so the counter twitches a few megabytes and then
    stops dead for four minutes. A "bytes are non-zero" test would show
    a bar stuck at 0.1%, which is the question being asked, made worse
    by looking like an answer.
    """
    _feed(monkeypatch, [2_000_000, 2_000_000, 2_000_000])
    t = LoadProgressTracker()
    t.sample("r", 1, str(model), now=0.0)

    assert t.sample("r", 1, str(model), now=3.0) is None
    assert t.sample("r", 1, str(model), now=6.0) is None


def test_a_host_with_no_counter_reports_nothing(
    monkeypatch: pytest.MonkeyPatch, model: Path
) -> None:
    _feed(monkeypatch, [None, None])
    t = LoadProgressTracker()
    t.sample("r", 1, str(model), now=0.0)

    assert t.sample("r", 1, str(model), now=1.0) is None


def test_a_runtime_with_no_process_reports_nothing(model: Path) -> None:
    assert LoadProgressTracker().sample("r", None, str(model), now=0.0) is None


# --------------------------------------------------------------------------- #
# restarts
# --------------------------------------------------------------------------- #


def test_a_restart_never_carries_the_previous_run_s_baseline(
    monkeypatch: pytest.MonkeyPatch, model: Path
) -> None:
    """A new process starts its counter at zero. Keeping the old
    samples would compute a huge negative delta — which reads as "not
    advancing" and hides a real load — or, with the readings the other
    way round, a wildly wrong rate."""
    _feed(monkeypatch, [0, 900_000_000, 10_000_000, 60_000_000])
    t = LoadProgressTracker()
    t.sample("r", 1, str(model), now=0.0)
    assert t.sample("r", 1, str(model), now=1.0) is not None

    # pid 2: the restart. First reading of the new process is a baseline.
    assert t.sample("r", 2, str(model), now=2.0) is None
    progress = t.sample("r", 2, str(model), now=3.0)

    assert progress is not None
    assert progress.bytes_read == 60_000_000
    assert progress.bytes_per_second == pytest.approx(50_000_000)


def test_forget_drops_a_runtime_s_samples(monkeypatch: pytest.MonkeyPatch, model: Path) -> None:
    _feed(monkeypatch, [0, 500, 900])
    t = LoadProgressTracker()
    t.sample("r", 1, str(model), now=0.0)
    t.forget("r")

    # Back to needing a baseline, so the next single reading says nothing.
    assert t.sample("r", 1, str(model), now=1.0) is None
    assert t.sample("r", 1, str(model), now=2.0) is not None


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #


def test_the_rate_is_measured_over_a_window_not_the_whole_load(
    monkeypatch: pytest.MonkeyPatch, model: Path
) -> None:
    """A share that has just got slower should say so, rather than
    reporting the average since the load began."""
    _feed(monkeypatch, [0, 1_000_000_000, 1_000_100_000, 1_000_200_000])
    t = LoadProgressTracker()
    t.sample("r", 1, str(model), now=0.0)
    t.sample("r", 1, str(model), now=10.0)
    t.sample("r", 1, str(model), now=20.0)

    progress = t.sample("r", 1, str(model), now=30.0)

    assert progress is not None
    # 100 MB across the last 10s of window, not 1 GB across 30s.
    assert progress.bytes_per_second == pytest.approx(10_000, rel=0.5)


def test_a_long_gap_between_polls_still_yields_a_rate(
    monkeypatch: pytest.MonkeyPatch, model: Path
) -> None:
    """Nothing polls on a guaranteed cadence — the samples come from
    whoever asked. Trimming must never leave fewer than a pair."""
    _feed(monkeypatch, [0, 300_000_000])
    t = LoadProgressTracker()
    t.sample("r", 1, str(model), now=0.0)

    progress = t.sample("r", 1, str(model), now=600.0)

    assert progress is not None
    assert progress.bytes_per_second == pytest.approx(500_000)


# --------------------------------------------------------------------------- #
# the real counter
# --------------------------------------------------------------------------- #


def test_the_platform_counter_reads_this_very_process() -> None:
    """Guards the trap that produced a confident wrong answer while this
    was being measured: `OpenProcess` returns a HANDLE, ctypes defaults a
    restype to `c_int`, and the truncated value makes the call fail while
    a zeroed output struct still looks like a legitimate reading. A
    failure must be None (cannot tell), never 0 (read nothing)."""
    import os

    reading = process_io.read_bytes(os.getpid())

    if sys.platform not in ("win32", "darwin") and not sys.platform.startswith("linux"):
        assert reading is None
        return
    assert reading is not None, "this platform has a counter; it should have answered"
    value, source = reading
    # Python has imported its own standard library by now, so a working
    # counter cannot report zero. Zero here means the call silently failed.
    assert value > 0
    assert source in (
        process_io.SOURCE_WINDOWS,
        process_io.SOURCE_PROC_IO,
        process_io.SOURCE_RUSAGE,
    )


def test_a_dead_pid_is_cannot_tell_rather_than_zero() -> None:
    assert process_io.read_bytes(0x7FFFFFFE) is None
