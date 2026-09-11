"""Stopping a child on Windows, and saying why one could not start.

Install-paths §9 step 2. The design named three hazards from inherited
assumption; the measurements on 2026-09-11 kept one, reshaped one, and
replaced one:

  * **Kept** — `TerminateProcess` really is the only stop that skips a
    uvicorn child's ASGI lifespan shutdown. A console event does not,
    even with no handler installed, because CPython gives SIGBREAK the
    same default as SIGINT.
  * **Reshaped** — "stale processes stack on one loopback port" is true
    of `http.server` (`allow_reuse_address = True`) and **false** of
    uvicorn, which is what every component runs. So the supervisor's job
    is diagnosis, not reclamation.
  * **New, and it only exists because of the fix** — a graceful request
    can be *ignored*, which `TerminateProcess` could never be. Escalation
    is mandatory.

And one defect nobody had written down: on Windows a requested restart
exits non-zero, so the loop counted every operator restart as a crash.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from eugene_plexus_agent import ports, process_signals
from eugene_plexus_agent.supervisor import ProcessState, SpawnPlan, SupervisedProcess

log = logging.getLogger("test")


class _FakeProcess:
    """A child that exits only when the test says so.

    Unlike the one in test_supervisor.py, `terminate()` here does NOT
    finish the process: the whole subject of these tests is what happens
    between asking and exiting.
    """

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.stdout: asyncio.StreamReader | None = None
        self._exit = asyncio.Event()

    async def wait(self) -> int:
        await self._exit.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.finish(-9)

    def finish(self, returncode: int) -> None:
        self.returncode = returncode
        self._exit.set()


class _Planner:
    def __init__(self, port: int | None = 8080) -> None:
        self._port = port
        self.reset_calls = 0
        self.explained: list[tuple[int, str]] = []

    @property
    def name(self) -> str:
        return "gateway"

    @property
    def log_prefix(self) -> str:
        return "[gateway] "

    def plan(self) -> SpawnPlan:
        return SpawnPlan(argv=["python", "-m", "x"], env={}, port=self._port)

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        self.explained.append((return_code, output_tail))
        return None

    def on_crash_threshold(self) -> bool:
        return False

    def reset(self) -> None:
        self.reset_calls += 1


# ---------------------------------------------------------------------------
# how a child is asked to stop
# ---------------------------------------------------------------------------


def test_posix_asks_with_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_signals.sys, "platform", "linux")
    proc = _FakeProcess()
    assert process_signals.request_stop(proc, name="gateway") == process_signals.StopSignal.sigterm
    assert proc.terminated


def test_windows_asks_with_a_console_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """`CTRL_BREAK_EVENT`, not `CTRL_C_EVENT`: Windows delivers Ctrl-C to
    every process sharing a console or to none, so it cannot be aimed at
    one child without also stopping the agent."""
    monkeypatch.setattr(process_signals.sys, "platform", "win32")
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(process_signals.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    proc = _FakeProcess(pid=1234)
    result = process_signals.request_stop(proc, name="gateway")
    assert result == process_signals.StopSignal.ctrl_break
    assert sent == [(1234, process_signals.CTRL_BREAK_EVENT)]
    assert not proc.terminated


def test_a_console_less_agent_falls_back_and_says_so_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A Windows service has no console, and `GenerateConsoleCtrlEvent`
    fails there with WinError 6 — verified directly, not assumed. The
    fallback has to be loud enough to find and quiet enough to survive
    stopping twelve children.
    """
    monkeypatch.setattr(process_signals.sys, "platform", "win32")

    def no_console(pid: int, sig: int) -> None:
        raise OSError(6, "The handle is invalid")

    monkeypatch.setattr(process_signals.os, "kill", no_console)
    process_signals.reset_console_warning_for_tests()

    with caplog.at_level(logging.WARNING):
        first = process_signals.request_stop(_FakeProcess(), name="gateway")
        second = process_signals.request_stop(_FakeProcess(), name="library")

    assert first == second == process_signals.StopSignal.terminate_process
    warnings = [r for r in caplog.records if "console event" in r.message]
    assert len(warnings) == 1, "the fallback warning must fire once, not once per child"
    assert "Windows service" in warnings[0].getMessage()


def test_a_missing_pid_never_signals_the_agents_own_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`os.kill(0, CTRL_BREAK_EVENT)` signals the caller's own process
    group. Reaching that with a falsy pid would stop the supervisor along
    with the child, so the guard is not decoration."""
    monkeypatch.setattr(process_signals.sys, "platform", "win32")
    monkeypatch.setattr(
        process_signals.os,
        "kill",
        lambda pid, sig: pytest.fail("os.kill must not be reached without a real pid"),
    )
    proc = _FakeProcess(pid=0)
    assert (
        process_signals.request_stop(proc, name="gateway")
        == process_signals.StopSignal.terminate_process
    )
    assert proc.terminated


def test_windows_children_get_their_own_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_signals.sys, "platform", "win32")
    flags = process_signals.spawn_kwargs().get("creationflags", 0)
    assert flags & process_signals.CREATE_NEW_PROCESS_GROUP


def test_posix_spawn_kwargs_still_carry_orphan_prevention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One entry point for spawn kwargs, so neither concern can quietly
    stop being applied. Linux keeps its pdeathsig preexec."""
    monkeypatch.setattr(process_signals.sys, "platform", "linux")
    monkeypatch.setattr(process_signals.orphan_kill.sys, "platform", "linux")
    kwargs = process_signals.spawn_kwargs()
    assert "creationflags" not in kwargs
    assert callable(kwargs.get("preexec_fn"))


# ---------------------------------------------------------------------------
# a stop request that is ignored
# ---------------------------------------------------------------------------


async def test_a_child_that_ignores_the_request_is_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hang the fix introduces, and closes. Measured with a stand-in
    that returned TRUE from a console handler and outlived the event by
    six seconds; `TerminateProcess` had no such failure mode."""
    monkeypatch.setattr("eugene_plexus_agent.supervisor._TERM_TIMEOUT_SECONDS", 0.3)
    proc = _FakeProcess()
    sup = SupervisedProcess(_Planner(), log)
    sup._proc = proc  # type: ignore[assignment]

    await sup.restart()

    assert proc.killed, "a child that never exits must be killed, not waited on forever"


async def test_a_child_that_complies_is_not_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("eugene_plexus_agent.supervisor._TERM_TIMEOUT_SECONDS", 5.0)
    proc = _FakeProcess()
    sup = SupervisedProcess(_Planner(), log)
    sup._proc = proc  # type: ignore[assignment]

    async def comply() -> None:
        await asyncio.sleep(0.05)
        proc.finish(3)

    complier = asyncio.create_task(comply())
    await sup.restart()
    await complier

    assert not proc.killed
    assert proc.returncode == 3


# ---------------------------------------------------------------------------
# the defect: a requested restart counted as a crash
# ---------------------------------------------------------------------------


async def _run_loop_until_spawns(sup: SupervisedProcess, n: int, limit: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + limit
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
        if len(_spawned) >= n:
            return
    raise AssertionError(f"only {len(_spawned)} spawn(s) after {limit}s, wanted {n}")


_spawned: list[_FakeProcess] = []


@pytest.fixture
def loop_spawns(monkeypatch: pytest.MonkeyPatch) -> list[_FakeProcess]:
    """Drives the REAL supervision loop against fake children.

    The accounting under test lives in `_spawn_once`, after the wait —
    so a test that only inspects a flag on `restart()` proves nothing
    about the counter, which is the thing that was wrong.
    """
    _spawned.clear()

    async def fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        proc = _FakeProcess()
        _spawned.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr("eugene_plexus_agent.supervisor._TERM_TIMEOUT_SECONDS", 0.2)
    return _spawned


async def test_a_requested_restart_is_not_a_crash(loop_spawns: list[_FakeProcess]) -> None:
    """**The Windows-only defect nobody had written down.** A console
    event makes the child exit 3 (CPython's KeyboardInterrupt code), and
    the loop counted any non-zero exit as a crash — so every operator
    restart bought a back-off sleep, and enough in a row tripped the
    threshold and dropped the component into safe mode for doing what it
    was asked. POSIX never showed it: SIGTERM gets uvicorn to exit 0.

    Drives the real loop, and asserts the COUNTER — the flag is an
    implementation detail and asserting it would pass against a fix that
    set it and then ignored it.
    """
    sup = SupervisedProcess(_Planner(), log)
    sup.start()
    try:
        await _run_loop_until_spawns(sup, 1)

        for _ in range(3):
            before = len(loop_spawns)
            await sup.restart()
            loop_spawns[before - 1].finish(3)  # what a console-stopped child exits with
            await _run_loop_until_spawns(sup, before + 1)
            assert sup._consecutive_crashes == 0, (
                "a restart the operator asked for was counted as a crash"
            )
            assert sup.state is not ProcessState.crashed
    finally:
        await sup.stop()


async def test_an_unrequested_non_zero_exit_is_still_a_crash(
    loop_spawns: list[_FakeProcess],
) -> None:
    """The half that keeps the fix honest. A child that dies on its own
    must still count, or the crash threshold and the safe-mode fallback
    stop working — and "never counts a crash" would pass the test above
    perfectly."""
    sup = SupervisedProcess(_Planner(), log)
    sup.start()
    try:
        await _run_loop_until_spawns(sup, 1)
        loop_spawns[0].finish(3)  # same exit code, nobody asked for it
        await _run_loop_until_spawns(sup, 2)
        assert sup._consecutive_crashes == 1
    finally:
        await sup.stop()


# ---------------------------------------------------------------------------
# saying who holds the port
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8291): "
        "[winerror 10048] only one usage of each socket address",
        "OSError: [Errno 98] Address already in use",
        "OSError: [Errno 48] Address already in use",
    ],
)
def test_every_platforms_way_of_saying_it_is_recognised(message: str) -> None:
    assert ports.looks_like_address_in_use(message)


def test_an_ordinary_crash_is_not_mistaken_for_a_collision() -> None:
    assert not ports.looks_like_address_in_use("Traceback: KeyError: 'modelId'")
    assert not ports.looks_like_address_in_use("")


def test_a_collision_names_the_port_and_the_holder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ports, "describe_holder", lambda port: "pid 1234 (python.exe)")
    explained = ports.explain_collision(8080, "[winerror 10048] only one usage")
    assert explained is not None
    assert "8080" in explained and "pid 1234 (python.exe)" in explained


def test_a_collision_says_so_when_the_holder_has_already_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "In use" and "was in use a second ago" want different actions, and
    a message that cannot tell them apart sends the operator hunting for
    a process that is not there."""
    monkeypatch.setattr(ports, "describe_holder", lambda port: None)
    explained = ports.explain_collision(8080, "Address already in use")
    assert explained is not None
    assert "restart will probably succeed" in explained


def test_nothing_is_explained_when_nothing_collided() -> None:
    assert ports.explain_collision(8080, "some other failure") is None


def test_the_supervisor_prefers_the_collision_over_the_planners_answer() -> None:
    """The component planner's `explain_exit` returns None on principle,
    so a collision explained there would be explained nowhere. It belongs
    to the loop, which is also the only place that sees both kinds of
    child."""
    planner = _Planner()
    sup = SupervisedProcess(planner, log)
    sup._last_port = 8080
    sup._output_tail.append("[winerror 10048] only one usage of each socket address")
    explained = sup._explain_exit(1)
    assert explained is not None and "8080" in explained
    assert planner.explained == [], "the planner should not have been asked"


def test_describe_holder_finds_a_port_this_process_is_listening_on() -> None:
    """Against a real socket, because the parsing is of real `netstat` /
    `ss` output and a fixture of it would only prove the fixture."""
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        holder = ports.describe_holder(port)
    finally:
        srv.close()
    # Not asserted as non-None: a hardened box may have neither netstat
    # nor ss, and "we could not tell" is a supported answer. What must
    # not happen is a wrong one.
    if holder is not None:
        assert str(port) in holder or "pid" in holder


def test_describe_holder_says_nothing_about_a_free_port() -> None:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()
    assert ports.describe_holder(free) is None


def test_a_collision_with_no_known_port_still_explains_itself() -> None:
    explained = ports.explain_collision(None, "Address already in use")
    assert explained is not None and "already in use" in explained


def test_describe_holder_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """It runs while explaining a crash; one failure must not become two."""

    def boom(argv: Any) -> str:
        raise OSError("no netstat here")

    monkeypatch.setattr(ports, "_run", boom)
    assert ports.describe_holder(8080) is None


# ---------------------------------------------------------------------------
# the accepted degradation, said out loud at boot
# ---------------------------------------------------------------------------


def test_a_console_less_windows_agent_announces_the_hard_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Troy's call, 2026-09-11: Windows ships a real service, and a
    service has no console, so children there are hard-killed. Accepted
    — which is exactly why it has to be announced. Same shape as the
    Vulkan decision: ship it, and badge it permanently."""
    monkeypatch.setattr(process_signals.sys, "platform", "win32")
    monkeypatch.setattr(process_signals, "console_attached", lambda: False)
    graceful, why = process_signals.describe_stop_capability()
    assert graceful is False
    assert "TerminateProcess" in why
    assert "accepted" in why, "a limitation that reads as a fault will be reported as one"


def test_a_windows_agent_with_a_console_says_the_opposite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_signals.sys, "platform", "win32")
    monkeypatch.setattr(process_signals, "console_attached", lambda: True)
    graceful, why = process_signals.describe_stop_capability()
    assert graceful is True
    assert "CTRL_BREAK_EVENT" in why


def test_posix_is_never_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_signals.sys, "platform", "linux")
    graceful, why = process_signals.describe_stop_capability()
    assert graceful is True
    assert "SIGTERM" in why


def test_console_detection_uses_the_probe_that_does_not_lie() -> None:
    """`GetConsoleWindow` returns a null HWND under any ConPTY terminal,
    so it reports "no console" for a process that has one — it returned
    False on this box in BOTH the has-console and no-console cases while
    the work was being done, and sent one probe to a wrong conclusion.
    This asserts the answer, not the mechanism: a test run has a console
    on Windows and the function must say so. On POSIX it is True by
    definition and the assertion is vacuous — stated rather than hidden,
    because a check that passes for a different reason on the CI runner
    than on the box it was written for is the thing this file keeps
    finding elsewhere.
    """
    assert process_signals.console_attached() is True
