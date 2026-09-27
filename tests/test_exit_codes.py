"""A process Windows refuses to run says why, not "exited with code 3221225781".

Found 2026-09-26: a friend's freshly reinstalled Windows had no Visual C++
runtime, so `llama-server` died at start with 0xC0000135, and every model
read "crashed" with nothing to say why.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

from eugene_plexus_agent.exit_codes import VC_REDIST_URL, explain_windows_exit
from eugene_plexus_agent.supervisor import SpawnPlan, SupervisedProcess

DLL_NOT_FOUND_UNSIGNED = 3221225781  # what GetExitCodeProcess reports
DLL_NOT_FOUND_SIGNED = -1073741515  # what a signed reader sees


@pytest.mark.parametrize("code", [DLL_NOT_FOUND_UNSIGNED, DLL_NOT_FOUND_SIGNED])
def test_a_missing_dll_names_the_visual_cpp_runtime_and_where_to_get_it(code: int) -> None:
    said = explain_windows_exit(code)
    assert said is not None
    assert "0xC0000135" in said
    assert "Visual C++ Redistributable" in said
    assert VC_REDIST_URL in said
    assert "press start" in said


@pytest.mark.parametrize(
    ("code", "words"),
    [
        (0xC000001D, "lacks an instruction"),
        (0xC0000017, "ran out of memory"),
        (0xC000012D, "ran out of memory"),
        (0xC0000005, "access violation"),
        (0xC0000409, "fatal error"),
        (0xC000007B, "different kind of processor"),
        (0xC0000139, "too old"),
    ],
)
def test_each_status_says_what_happened(code: int, words: str) -> None:
    said = explain_windows_exit(code)
    assert said is not None and words in said
    assert f"0x{code:08X}" in said


def test_a_status_it_does_not_know_is_still_named_by_its_code() -> None:
    assert explain_windows_exit(0xC0000142) == "Windows ended it with status 0xC0000142."


@pytest.mark.parametrize("code", [0, 1, 2, 137, 255])
def test_a_programs_own_exit_code_is_not_a_windows_status(code: int) -> None:
    assert explain_windows_exit(code) is None


class _SilentPlanner:
    """A planner with nothing of its own to say about an exit."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.name = "qwen3-5-4b-q4-k-m"
        self.log_prefix = "[engine: qwen3-5-4b-q4-k-m] "

    def plan(self) -> SpawnPlan:
        return SpawnPlan(argv=self.argv, env=None, cwd=None)

    def on_crash_threshold(self) -> bool:
        return False

    def reset(self) -> None:
        return None

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        return None


def test_the_supervisor_falls_back_to_the_windows_reason() -> None:
    """The wiring: a planner that knows nothing leaves Windows' own status
    to say it, instead of the bare number."""
    sp = SupervisedProcess(_SilentPlanner(["unused"]), logging.getLogger("test"))
    said = sp._explain_exit(DLL_NOT_FOUND_UNSIGNED)
    assert said is not None
    assert "Visual C++ Redistributable" in said


def test_a_planner_that_knows_better_still_wins() -> None:
    class _Knowing(_SilentPlanner):
        def explain_exit(self, return_code: int, output_tail: str) -> str | None:
            return "the planner's own reason"

    sp = SupervisedProcess(_Knowing(["unused"]), logging.getLogger("test"))
    assert sp._explain_exit(DLL_NOT_FOUND_UNSIGNED) == "the planner's own reason"


@pytest.mark.skipif(sys.platform != "win32", reason="an NTSTATUS exit exists only on Windows")
async def test_a_real_process_dying_with_the_status_reaches_last_error(tmp_path: Path) -> None:
    """End to end on Windows: a child that exits exactly as a missing DLL
    makes one exit, and the reason on `last_error` is the Windows one."""
    script = tmp_path / "die.py"
    script.write_text(f"import os\nos._exit({DLL_NOT_FOUND_SIGNED})\n", encoding="utf-8")
    sp = SupervisedProcess(_SilentPlanner([sys.executable, str(script)]), logging.getLogger("test"))
    sp.start()
    try:
        # `last_error`, not the state: the supervisor respawns a crashed
        # child, so `crashed` lasts only until the next attempt starts.
        for _ in range(200):
            if sp.last_error is not None:
                break
            await asyncio.sleep(0.05)
        assert sp.last_error is not None
        assert "Visual C++ Redistributable" in sp.last_error, sp.last_error
        assert "exited with code" not in sp.last_error
    finally:
        await sp.stop()
