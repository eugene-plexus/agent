"""Tests for `Supervisor` and `SupervisedProcess`.

We monkeypatch `asyncio.create_subprocess_exec` to return a controllable
fake process, so the supervision loop runs without actually forking
anything. Verifies the contract pieces that matter:

  - Right command (sys.executable -m <module-for-kind>)
  - Right env vars (config_file, bind_port, safe_mode), correctly
    prefixed per kind
  - Restart-on-exit: the loop respawns after the fake process "exits"
  - Stop: terminates and cleans up the supervision task
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
from typing import Any

import pytest

from eugene_plexus_agent import security
from eugene_plexus_agent._generated.models import (
    ComponentEntry,
    ComponentKind,
    ComponentStatus,
    SpawnConfig,
)
from eugene_plexus_agent.auth_state import AuthState
from eugene_plexus_agent.supervisor import (
    _COMPONENT_SPECS,
    _COMPONENT_STATUS_BY_STATE,
    _HEALTHZ_2XX_LINE,
    ProcessState,
    SpawnPlan,
    SpawnPlanError,
    SupervisedProcess,
    Supervisor,
    _colorize_alerts,
    _ComponentPlanner,
    _format_log_prefix,
)


class _FakeProcess:
    """Stand-in for `asyncio.subprocess.Process`. Tests drive
    `_finish(returncode)` to simulate the child exiting."""

    def __init__(self) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._exit_event = asyncio.Event()
        # Supervisor pipes the child's stdout (with stderr merged in) and
        # spawns a reader task. None here makes the reader a no-op —
        # tests get the supervision-loop behavior without exercising the
        # output prefixing path.
        self.stdout: asyncio.StreamReader | None = None

    async def wait(self) -> int:
        await self._exit_event.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self._finish(0)

    def kill(self) -> None:
        self.killed = True
        self._finish(-9)

    def _finish(self, returncode: int) -> None:
        self.returncode = returncode
        self._exit_event.set()


@pytest.fixture
def driver_entry() -> ComponentEntry:
    return ComponentEntry(
        name="left",
        kind=ComponentKind.inference_driver,
        url="http://127.0.0.1:8081",  # type: ignore[arg-type]
        spawn=SpawnConfig(configFile="/tmp/left/config.yaml"),
        safeMode=False,
    )


async def test_spawn_invokes_correct_command_and_env(
    driver_entry: ComponentEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def fake_create(*args: Any, **kwargs: Any) -> _FakeProcess:
        captured["args"] = args
        captured["env"] = kwargs.get("env")
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    sp = SupervisedProcess.for_component(driver_entry, logging.getLogger("test"))
    sp.start()

    # Give the supervision loop one tick to spawn.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if "args" in captured:
            break

    await sp.stop()

    assert "args" in captured, "subprocess was never invoked"
    assert captured["args"][0] == sys.executable
    assert captured["args"][1] == "-m"
    assert captured["args"][2] == "eugene_plexus_inference_driver"

    env = captured["env"]
    assert env["EUGENE_PLEXUS_DRIVER_CONFIG_FILE"] == "/tmp/left/config.yaml"
    assert env["EUGENE_PLEXUS_DRIVER_BIND_PORT"] == "8081"
    assert env["EUGENE_PLEXUS_DRIVER_SAFE_MODE"] == "0"


async def test_safe_mode_threads_env_var(
    driver_entry: ComponentEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def fake_create(*args: Any, **kwargs: Any) -> _FakeProcess:
        captured["env"] = kwargs.get("env")
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    safe_entry = driver_entry.model_copy(update={"safeMode": True})
    sp = SupervisedProcess.for_component(safe_entry, logging.getLogger("test"))
    sp.start()

    for _ in range(50):
        await asyncio.sleep(0.01)
        if "env" in captured:
            break
    await sp.stop()

    assert captured["env"]["EUGENE_PLEXUS_DRIVER_SAFE_MODE"] == "1"


async def test_clean_exit_triggers_respawn(
    driver_entry: ComponentEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawn_count = 0
    processes: list[_FakeProcess] = []

    async def fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        nonlocal spawn_count
        spawn_count += 1
        proc = _FakeProcess()
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    sp = SupervisedProcess.for_component(driver_entry, logging.getLogger("test"))
    sp.start()

    # Wait for first spawn, then simulate clean exit.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if spawn_count >= 1:
            break
    assert spawn_count == 1
    processes[0]._finish(0)

    # Wait for respawn (the loop sleeps ~0s for clean exits since
    # consecutive_crashes is 0).
    for _ in range(100):
        await asyncio.sleep(0.01)
        if spawn_count >= 2:
            break
    assert spawn_count >= 2, "supervisor did not respawn after clean exit"

    await sp.stop()


async def test_supervisor_stop_all_terminates_each_process(
    driver_entry: ComponentEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    processes: list[_FakeProcess] = []

    async def fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        proc = _FakeProcess()
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    sup = Supervisor(log=logging.getLogger("test"))
    sup.add_and_start(driver_entry)

    for _ in range(50):
        await asyncio.sleep(0.01)
        if processes:
            break

    await sup.stop_all()

    assert processes[0].terminated, "stop_all should terminate every process"


async def test_remote_entry_does_not_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned = False

    async def fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        nonlocal spawned
        spawned = True
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    # A driver on a remote GPU host is the real instance of this case:
    # it belongs next to its engine, so the agent watches it but does
    # not own its lifecycle.
    remote_entry = ComponentEntry(
        name="rtx5090",
        kind=ComponentKind.inference_driver,
        url="http://gpu-box.tailnet:8081",  # type: ignore[arg-type]
        # No spawn block => remote, agent must not try to launch it.
        safeMode=False,
    )
    sup = Supervisor(log=logging.getLogger("test"))
    sup.add_and_start(remote_entry)

    await asyncio.sleep(0.05)
    await sup.stop_all()

    assert spawned is False
    # Status is reported as `unreachable` because no health probe ran.
    status, _, _, pid = sup.status_for("rtx5090", has_spawn=False)
    assert status == ComponentStatus.unreachable
    assert pid is None


# --------------------------------------------------------------------------- #
# v0.2 auth env-var threading
# --------------------------------------------------------------------------- #


async def test_auth_state_threads_signing_key_and_service_token(
    driver_entry: ComponentEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When an AuthState is wired in, every spawned child receives the
    base64'd JWT signing key plus a freshly-issued service token bound
    to the component's kind."""
    captured: dict[str, Any] = {}

    async def fake_create(*_args: Any, **kwargs: Any) -> _FakeProcess:
        captured["env"] = kwargs.get("env")
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    auth = AuthState(signing_key=security.generate_signing_key())
    sp = SupervisedProcess.for_component(driver_entry, logging.getLogger("test"), auth_state=auth)
    sp.start()

    for _ in range(50):
        await asyncio.sleep(0.01)
        if "env" in captured:
            break
    await sp.stop()

    env = captured["env"]
    # Signing key is the shared base64-encoded HMAC key.
    assert base64.b64decode(env["EUGENE_PLEXUS_DRIVER_AUTH_SIGNING_KEY"]) == auth.signing_key
    # Service token must validate against the same signing key with the
    # correct service audience.
    payload = security.decode_token(
        token=env["EUGENE_PLEXUS_DRIVER_SERVICE_TOKEN"],
        signing_key=auth.signing_key,
        expected_audience="service:inference-driver",
    )
    assert payload.sub == "inference-driver"
    # Master key absent because the operator hasn't logged in yet.
    assert "EUGENE_PLEXUS_DRIVER_MASTER_KEY" not in env


async def test_master_key_threaded_after_login(
    driver_entry: ComponentEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once `AuthState.master_key` is populated (i.e. operator has
    logged in), subsequent spawns carry the base64'd master key so
    children can decrypt at-rest secrets."""
    captured: dict[str, Any] = {}

    async def fake_create(*_args: Any, **kwargs: Any) -> _FakeProcess:
        captured["env"] = kwargs.get("env")
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    auth = AuthState(signing_key=security.generate_signing_key())
    auth.set_master_key(b"\x55" * 32)

    sp = SupervisedProcess.for_component(driver_entry, logging.getLogger("test"), auth_state=auth)
    sp.start()

    for _ in range(50):
        await asyncio.sleep(0.01)
        if "env" in captured:
            break
    await sp.stop()

    env = captured["env"]
    assert base64.b64decode(env["EUGENE_PLEXUS_DRIVER_MASTER_KEY"]) == auth.master_key


async def test_every_spawnable_kind_gets_its_own_prefix_and_audience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service-token audience + env-var prefix must follow the kind.

    Driven off `_COMPONENT_SPECS` rather than a hand-written list, so
    adding a kind (library, next) fails here loudly if its spec is
    incomplete instead of surfacing as an unsupported-kind refusal at
    spawn time.
    """
    captured: dict[str, dict[str, str]] = {}

    async def fake_create(*_args: Any, **kwargs: Any) -> _FakeProcess:
        env = kwargs.get("env") or {}
        for kind, spec in _COMPONENT_SPECS.items():
            if f"{spec.env_prefix}_CONFIG_FILE" in env:
                captured[kind.value] = env
                break
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    auth = AuthState(signing_key=security.generate_signing_key())
    procs: list[SupervisedProcess] = []
    for port, kind in enumerate(_COMPONENT_SPECS, start=9000):
        entry = ComponentEntry(
            name=kind.value,
            kind=kind,
            url=f"http://127.0.0.1:{port}",  # type: ignore[arg-type]
            spawn=SpawnConfig(configFile=f"/tmp/{kind.value}/config.yaml"),
            safeMode=False,
        )
        sp = SupervisedProcess.for_component(entry, logging.getLogger("test"), auth_state=auth)
        sp.start()
        procs.append(sp)

    for _ in range(100):
        await asyncio.sleep(0.01)
        if {k.value for k in _COMPONENT_SPECS}.issubset(captured.keys()):
            break

    for sp in procs:
        await sp.stop()

    assert {k.value for k in _COMPONENT_SPECS} == set(captured), (
        "every spawnable kind should have been captured"
    )
    for kind, spec in _COMPONENT_SPECS.items():
        env = captured[kind.value]
        payload = security.decode_token(
            token=env[f"{spec.env_prefix}_SERVICE_TOKEN"],
            signing_key=auth.signing_key,
            expected_audience=f"service:{kind.value}",
        )
        assert payload.sub == kind.value


# --------------------------------------------------------------------------- #
# Child-output filtering / coloring (signal-noise reduction)
# --------------------------------------------------------------------------- #


def test_healthz_filter_matches_2xx_only() -> None:
    """Successful /healthz access logs are the dominant noise source —
    we suppress them. Non-2xx must pass through so a newly-unhealthy
    component is still visible."""
    ok = 'INFO:     127.0.0.1:59471 - "GET /healthz HTTP/1.1" 200 OK\n'
    accepted = 'INFO:     127.0.0.1:59471 - "GET /healthz HTTP/1.1" 202 Accepted\n'
    sad_503 = 'INFO:     127.0.0.1:59471 - "GET /healthz HTTP/1.1" 503 Service Unavailable\n'
    sad_404 = 'INFO:     127.0.0.1:59471 - "GET /healthz HTTP/1.1" 404 Not Found\n'
    unrelated = 'INFO:     127.0.0.1:59471 - "POST /v1/chat HTTP/1.1" 200 OK\n'

    assert _HEALTHZ_2XX_LINE.search(ok)
    assert _HEALTHZ_2XX_LINE.search(accepted)
    assert not _HEALTHZ_2XX_LINE.search(sad_503)
    assert not _HEALTHZ_2XX_LINE.search(sad_404)
    assert not _HEALTHZ_2XX_LINE.search(unrelated)


def test_colorize_alerts_wraps_just_the_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the alert WORD gets wrapped — coloring the whole line makes
    red-on-dark unreadable. Case must be preserved."""
    # Force-enable color regardless of NO_COLOR in the test env.
    monkeypatch.setattr("eugene_plexus_agent.supervisor._USE_COLOR", True)

    err = _colorize_alerts("ERROR: something broke\n")
    assert err.startswith("\x1b[31mERROR\x1b[0m: something broke")

    warn = _colorize_alerts("Warning: dropping cache\n")
    assert warn.startswith("\x1b[33mWarning\x1b[0m: dropping cache")

    # Mid-line, mixed case, both severities in one line.
    multi = _colorize_alerts("warn x; ERROR y\n")
    assert "\x1b[33mwarn\x1b[0m" in multi
    assert "\x1b[31mERROR\x1b[0m" in multi

    # Substring matches don't fire (no \berror\b inside "errored-out").
    no_match = _colorize_alerts("The action errored-out cleanly\n")
    assert "\x1b[" not in no_match


def test_colorize_alerts_respects_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """NO_COLOR is a documented opt-out (https://no-color.org). When set,
    the helper returns the original text unchanged."""
    monkeypatch.setattr("eugene_plexus_agent.supervisor._USE_COLOR", False)
    assert _colorize_alerts("ERROR boom\n") == "ERROR boom\n"


async def test_auto_safe_mode_after_crash_threshold(
    driver_entry: ComponentEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After N consecutive crashes the supervisor falls back to SAFE
    MODE on the next spawn — closes the 'doesn't soft-brick' loop so
    /v1/config stays reachable for repair without operator env-var or
    YAML knowledge."""
    captured_envs: list[dict[str, str]] = []
    processes: list[_FakeProcess] = []

    async def fake_create(*_args: Any, **kwargs: Any) -> _FakeProcess:
        captured_envs.append(kwargs.get("env") or {})
        proc = _FakeProcess()
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    # Cap every asyncio.sleep at 10ms so the supervisor's between-crash
    # backoff (up to 10s) doesn't slow this test by a wall-clock minute.
    original_sleep = asyncio.sleep

    async def _fast_sleep(seconds: float) -> None:
        await original_sleep(min(seconds, 0.01))

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    sp = SupervisedProcess.for_component(driver_entry, logging.getLogger("test"))
    sp.start()

    # Drive 5 consecutive crashes (matches _CRASH_BACKOFF_THRESHOLD).
    for crash_n in range(5):
        for _ in range(200):
            await asyncio.sleep(0.01)
            if len(processes) >= crash_n + 1:
                break
        else:
            pytest.fail(f"spawn {crash_n + 1} never happened")
        processes[crash_n]._finish(1)  # non-zero = crash

    # The 6th spawn should be the auto-safe-mode one.
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(captured_envs) >= 6:
            break

    await sp.stop()

    assert len(captured_envs) >= 6, (
        f"expected >= 6 spawns after 5 crashes; got {len(captured_envs)}"
    )
    # First 5 spawns: normal mode (topology safeMode=False).
    for i in range(5):
        assert captured_envs[i]["EUGENE_PLEXUS_DRIVER_SAFE_MODE"] == "0", (
            f"spawn {i} should have been normal mode, "
            f"got SAFE_MODE={captured_envs[i]['EUGENE_PLEXUS_DRIVER_SAFE_MODE']}"
        )
    # Spawn 6 onward: auto-engaged safe mode.
    assert captured_envs[5]["EUGENE_PLEXUS_DRIVER_SAFE_MODE"] == "1", (
        "spawn after threshold should be SAFE_MODE=1"
    )
    # The loop is still cycling, not given up — and the plan it is now
    # running is the degraded one. `Supervisor.status_for` is what turns
    # that pair into ComponentStatus.safe_mode on the wire.
    assert sp.state == ProcessState.starting
    assert sp.degraded is True


async def test_manual_restart_clears_auto_safe_mode(
    driver_entry: ComponentEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator's `restart()` returns the component to normal-mode
    operation — clears the auto-fallback flag so the next spawn
    follows the topology's safeMode value, not the latched fallback."""
    captured_envs: list[dict[str, str]] = []
    processes: list[_FakeProcess] = []

    async def fake_create(*_args: Any, **kwargs: Any) -> _FakeProcess:
        captured_envs.append(kwargs.get("env") or {})
        proc = _FakeProcess()
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    # Skip the crash dance by driving the planner's escalation directly —
    # that is the same call the supervision loop makes when the threshold
    # trips, so this exercises the real seam rather than poking a flag.
    planner = _ComponentPlanner(driver_entry, logging.getLogger("test"))
    assert planner.on_crash_threshold() is True, "first trip should engage safe mode"
    sp = SupervisedProcess(planner, logging.getLogger("test"))
    sp.start()

    for _ in range(100):
        await asyncio.sleep(0.01)
        if captured_envs:
            break
    assert captured_envs[0]["EUGENE_PLEXUS_DRIVER_SAFE_MODE"] == "1"

    # Manual restart: clears the flag, terminates the proc, supervisor
    # respawns. _FakeProcess.terminate() calls _finish(0) (clean exit),
    # so the supervisor loop resets the crash counter and respawns.
    await sp.restart()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(captured_envs) >= 2:
            break

    await sp.stop()

    assert len(captured_envs) >= 2
    assert captured_envs[1]["EUGENE_PLEXUS_DRIVER_SAFE_MODE"] == "0", (
        "post-restart spawn should be back to normal mode"
    )


def test_log_prefix_disambiguates_renamed_components() -> None:
    """Operators name drivers whatever they like, so a bare `[name]`
    prefix is ambiguous about what kind of thing is talking. When name ==
    the kind's short label the prefix stays `[name]`; otherwise it
    expands to `[label: name]`."""
    # Name matches the kind's label: keep it short, no `[gateway: gateway]`.
    assert _format_log_prefix(ComponentKind.gateway, "gateway") == "[gateway] "

    # Drivers are named after what they serve, so the label differs and
    # the disambiguating prefix kicks in. This is the common case now —
    # there are N drivers and their names carry the real information.
    assert _format_log_prefix(ComponentKind.inference_driver, "qwen3-30b") == (
        "[driver: qwen3-30b] "
    )
    assert _format_log_prefix(ComponentKind.inference_driver, "claude") == "[driver: claude] "

    # A driver an operator named "gateway" still reports as a driver.
    assert _format_log_prefix(ComponentKind.inference_driver, "gateway") == ("[driver: gateway] ")


# --------------------------------------------------------------------------- #
# The planner seam
#
# These are the tests that make the abstraction load-bearing rather than
# decorative: the loop must supervise something that is NOT a Eugene
# Plexus component, since that is the whole reason it was split out.
# --------------------------------------------------------------------------- #


class _FakeEnginePlanner:
    """Stands in for the engine adapters that don't exist yet.

    Deliberately shares nothing with `_ComponentPlanner`: an arbitrary
    argv, a working directory, no env threading, and no recovery — an
    engine that will not start has no safe mode to fall back to.
    """

    def __init__(self, argv: list[str], cwd: str | None = None) -> None:
        self._argv = argv
        self._cwd = cwd
        self.reset_calls = 0
        self.threshold_calls = 0

    @property
    def name(self) -> str:
        return "qwen3-30b"

    @property
    def log_prefix(self) -> str:
        return "[engine: qwen3-30b] "

    def plan(self) -> SpawnPlan:
        return SpawnPlan(argv=list(self._argv), env={"CUDA_VISIBLE_DEVICES": "1"}, cwd=self._cwd)

    def on_crash_threshold(self) -> bool:
        self.threshold_calls += 1
        return False

    def reset(self) -> None:
        self.reset_calls += 1


async def test_loop_spawns_an_arbitrary_argv_from_a_non_component_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine is a foreign binary with flags, not `python -m <module>`.

    The loop must pass the planner's argv through verbatim, honour its
    working directory (prebuilt llama.cpp releases need it to find their
    bundled shared libraries), and use its env as given rather than
    layering component env vars on top.
    """
    captured: dict[str, Any] = {}

    async def fake_create(*args: Any, **kwargs: Any) -> _FakeProcess:
        captured["argv"] = list(args)
        captured["env"] = kwargs.get("env")
        captured["cwd"] = kwargs.get("cwd")
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    argv = [
        "/opt/llama.cpp/llama-server",
        "--model",
        "/models/qwen3-30b-Q4_K_M.gguf",
        "--port",
        "8091",
    ]
    planner = _FakeEnginePlanner(argv, cwd="/opt/llama.cpp")
    sp = SupervisedProcess(planner, logging.getLogger("test"))
    sp.start()

    for _ in range(100):
        await asyncio.sleep(0.01)
        if "argv" in captured:
            break
    await sp.stop()

    assert captured["argv"] == argv, "engine argv must pass through verbatim"
    assert captured["cwd"] == "/opt/llama.cpp"
    # Exactly the planner's env — no component vars leaked in.
    assert captured["env"] == {"CUDA_VISIBLE_DEVICES": "1"}
    assert "EUGENE_PLEXUS_DRIVER_CONFIG_FILE" not in (captured["env"] or {})
    assert sp.name == "qwen3-30b"
    assert sp.degraded is False


async def test_loop_gives_up_when_planner_declines_to_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery policy belongs to the planner. A planner that returns
    False from `on_crash_threshold` must end the loop as `crashed`,
    rather than the loop assuming a safe-mode fallback exists."""

    async def fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        proc = _FakeProcess()
        proc._finish(1)  # non-zero: counts as a crash
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    # Cap every sleep at 10ms so the escalating between-crash backoff
    # (2s, 4s, ... up to 10s) doesn't cost this test half a minute.
    original_sleep = asyncio.sleep

    async def _fast_sleep(seconds: float) -> None:
        await original_sleep(min(seconds, 0.01))

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    planner = _FakeEnginePlanner(["/opt/llama.cpp/llama-server"])
    sp = SupervisedProcess(planner, logging.getLogger("test"))
    sp.start()

    for _ in range(500):
        await original_sleep(0.01)
        if sp.state == ProcessState.crashed and planner.threshold_calls:
            break
    await sp.stop()

    assert planner.threshold_calls == 1, "threshold should be offered exactly once"
    assert sp.state == ProcessState.crashed
    assert sp.last_error == "exited with code 1"


async def test_plan_error_is_a_crash_but_nothing_to_launch_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`plan()` has two non-launch outcomes and they mean opposite things.

    Returning None is a remote entry — nothing to run, no error. Raising
    SpawnPlanError is a declaration this agent cannot build, which
    counts as a crash so the back-off and the operator both hear about it.
    """
    spawned = False

    async def fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        nonlocal spawned
        spawned = True
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    class _NothingToLaunch(_FakeEnginePlanner):
        def plan(self) -> SpawnPlan | None:
            return None

    class _Unbuildable(_FakeEnginePlanner):
        def plan(self) -> SpawnPlan | None:
            raise SpawnPlanError("no adapter for engine 'mlx'")

    quiet = SupervisedProcess(_NothingToLaunch([]), logging.getLogger("test"))
    await quiet._spawn_once()
    assert quiet.state == ProcessState.not_spawnable
    assert quiet.last_error is None
    assert spawned is False

    broken = SupervisedProcess(_Unbuildable([]), logging.getLogger("test"))
    await broken._spawn_once()
    assert broken.state == ProcessState.crashed
    assert broken.last_error == "no adapter for engine 'mlx'"
    assert spawned is False


def test_every_process_state_maps_onto_component_status() -> None:
    """The mapping table is exhaustive by test, not by hope. Adding a
    ProcessState member without a mapping entry would otherwise surface
    as a KeyError from `status_for` at runtime."""
    assert set(_COMPONENT_STATUS_BY_STATE) == set(ProcessState)


async def test_nothing_to_launch_ends_the_loop_instead_of_spinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`plan() -> None` must end supervision, not retry in a tight loop.

    The between-attempt back-off is derived from the crash count, and
    "nothing to launch" is not a crash — so a loop that fell through
    would sleep zero seconds and spin the event loop hot forever.
    """
    sleeps: list[float] = []
    original_sleep = asyncio.sleep

    async def counting_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await original_sleep(min(seconds, 0.01))

    monkeypatch.setattr(asyncio, "sleep", counting_sleep)

    class _NothingToLaunch(_FakeEnginePlanner):
        def plan(self) -> SpawnPlan | None:
            return None

    sp = SupervisedProcess(_NothingToLaunch([]), logging.getLogger("test"))
    sp.start()
    await original_sleep(0.05)

    assert sp.state == ProcessState.not_spawnable
    assert sp._task is not None and sp._task.done(), "loop should have exited"
    assert sleeps == [], "should not have entered the back-off at all"
    await sp.stop()
