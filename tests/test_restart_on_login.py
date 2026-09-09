"""Tests for Phase 7 — restart-on-login signal.

When the operator unlocks the install (initialize on first run, or
login on a fresh process), the agent's AuthState gains the master
key. Children spawned BEFORE that moment ran without MASTER_KEY in
their env — they can read but not decrypt the at-rest envelopes on
disk. Phase 7 closes that gap by asking the supervisor to respawn
every supervised child so they pick up the now-available key.

These tests assert the wiring at the route boundary using the
StubSupervisor, plus the Supervisor.restart_all() primitive itself
against a real fake-process supervisor harness.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent._generated.models import (
    ComponentEntry,
    ComponentKind,
    SpawnConfig,
)
from eugene_plexus_agent.auth_state import AuthState
from eugene_plexus_agent.supervisor import Supervisor
from tests.conftest import TEST_PASSPHRASE, StubSupervisor

# --------------------------------------------------------------------------- #
# Route wiring — initialize / login trigger supervisor.restart_all
# --------------------------------------------------------------------------- #


def test_initialize_signals_restart_all(
    client: TestClient, stub_supervisor: StubSupervisor
) -> None:
    """First-run wizard: after the passphrase is set and the master
    key derives, the supervisor must be told to respawn so any
    already-spawned children (rare on first run, but possible) pick
    up MASTER_KEY."""
    resp = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    assert resp.status_code == 200, resp.text
    assert ("restart_all", "") in stub_supervisor.calls


def test_login_signals_restart_all_only_on_first_unlock(
    client: TestClient, stub_supervisor: StubSupervisor
) -> None:
    """First successful login of a process run → restart_all. Subsequent
    logins in the same process run → no restart (children already have
    the key). Otherwise repeated logins would cause spurious churn."""
    # Set the passphrase.
    init = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    assert init.status_code == 200
    # The initialize already populated master_key + triggered restart_all.
    # Clear the recorded calls so we measure only what login does.
    stub_supervisor.calls.clear()

    # First login *within this same process* — master_key is already set
    # (from initialize), so the route SHOULD NOT call restart_all again.
    login1 = client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    assert login1.status_code == 200
    assert ("restart_all", "") not in stub_supervisor.calls

    # Second login — same story, no restart.
    login2 = client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    assert login2.status_code == 200
    assert ("restart_all", "") not in stub_supervisor.calls


def test_login_after_fresh_process_signals_restart_all(
    settings: object, stub_supervisor: StubSupervisor
) -> None:
    """The scenario that actually matters in production: process starts
    with a configured passphrase on disk (so children spawn at lifespan),
    but no master_key in AuthState yet. Operator's first login of THIS
    process run must trigger restart_all."""
    from fastapi.testclient import TestClient as Client

    from eugene_plexus_agent.app import create_app

    # First app run: set the passphrase. This populates the disk state.
    first_app = create_app(settings=settings)  # type: ignore[arg-type]
    first_app.state.supervisor = stub_supervisor
    with Client(first_app) as c:
        c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})

    # Fresh process: same on-disk passphrase, but AuthState.master_key
    # is None because the new process hasn't unlocked yet.
    fresh_app = create_app(settings=settings)  # type: ignore[arg-type]
    fresh_supervisor = StubSupervisor()
    fresh_app.state.supervisor = fresh_supervisor
    with Client(fresh_app) as fresh:
        # Children may have been spawned in the lifespan with no master_key.
        login = fresh.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
        assert login.status_code == 200, login.text
        assert ("restart_all", "") in fresh_supervisor.calls


def test_failed_login_does_not_signal_restart_all(
    client: TestClient, stub_supervisor: StubSupervisor
) -> None:
    """Wrong-passphrase responses must not churn the supervised
    children. Only a SUCCESSFUL first unlock should signal."""
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    stub_supervisor.calls.clear()

    resp = client.post("/v1/auth/login", json={"passphrase": "not-the-passphrase"})
    assert resp.status_code == 401
    assert ("restart_all", "") not in stub_supervisor.calls


# --------------------------------------------------------------------------- #
# restart_all primitive — operates against a real Supervisor + fake procs
# --------------------------------------------------------------------------- #


class _FakeProcess:
    """Same shape as the fake used in test_supervisor.py. Repeated
    here rather than imported because pytest's test-collection has
    cross-file import friction with private fixtures."""

    def __init__(self) -> None:
        self.pid = 1234
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._exit_event = asyncio.Event()
        # Supervisor pipes the child's stdout and spawns a reader task;
        # None makes the reader a no-op (see test_supervisor.py's fake).
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
def supervisor_with_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Supervisor, list[_FakeProcess]]]:
    spawned: list[_FakeProcess] = []

    async def fake_create(*_args: Any, **_kwargs: Any) -> _FakeProcess:
        proc = _FakeProcess()
        spawned.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    sup = Supervisor(log=logging.getLogger("test"))
    yield sup, spawned


async def _wait_for(condition: Any, max_iterations: int = 100) -> None:
    for _ in range(max_iterations):
        if condition():
            return
        await asyncio.sleep(0.01)


async def test_restart_all_terminates_each_supervised_process(
    supervisor_with_processes: tuple[Supervisor, list[_FakeProcess]],
) -> None:
    sup, spawned = supervisor_with_processes
    entries = [
        ComponentEntry(
            name=name,
            kind=ComponentKind.inference_driver,
            url=f"http://127.0.0.1:{port}",  # type: ignore[arg-type]
            spawn=SpawnConfig(configFile=f"/tmp/{name}/config.yaml"),
            safeMode=False,
        )
        for name, port in [("left", 8081), ("right", 8082)]
    ]
    for entry in entries:
        sup.add_and_start(entry)

    await _wait_for(lambda: len(spawned) >= 2)

    restarted = await sup.restart_all()
    assert set(restarted) == {"left", "right"}
    # The original two children must have been signaled to terminate;
    # the supervision loop will produce respawns asynchronously.
    assert spawned[0].terminated
    assert spawned[1].terminated

    await sup.stop_all()


async def test_restart_all_no_op_when_no_processes_running() -> None:
    """Login can fire before topology is configured — restart_all must
    handle the empty case cleanly without surprising the caller."""
    sup = Supervisor(log=logging.getLogger("test"))
    assert await sup.restart_all() == []


async def test_restart_all_survives_one_failing_restart(
    supervisor_with_processes: tuple[Supervisor, list[_FakeProcess]],
) -> None:
    """If one supervised process's restart raises, restart_all must
    still attempt the others and never bubble the failure up to the
    auth route (which would otherwise turn a recoverable hiccup into
    a 500 on login)."""
    sup, spawned = supervisor_with_processes
    for name, port in [("good", 8081), ("bad", 8082)]:
        entry = ComponentEntry(
            name=name,
            kind=ComponentKind.inference_driver,
            url=f"http://127.0.0.1:{port}",  # type: ignore[arg-type]
            spawn=SpawnConfig(configFile=f"/tmp/{name}/config.yaml"),
            safeMode=False,
        )
        sup.add_and_start(entry)

    await _wait_for(lambda: len(spawned) >= 2)

    async def boom() -> None:
        raise RuntimeError("simulated restart failure")

    # Replace just the "bad" process's restart() with one that raises.
    # The internal dict access here is test-only; the public API is the
    # restart_all() return value plus the side-effect on spawned[*].
    sup._processes["bad"].restart = boom  # type: ignore[method-assign]

    restarted = await sup.restart_all()
    # restart_all returns the list of names it ATTEMPTED, not just
    # the ones that succeeded. Both should be reported.
    assert set(restarted) == {"good", "bad"}
    # The good process was signaled normally.
    assert spawned[0].terminated

    await sup.stop_all()


# --------------------------------------------------------------------------- #
# Master key threading after restart (sanity)
# --------------------------------------------------------------------------- #


async def test_respawn_after_master_key_set_includes_master_key_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end mechanics: spawn a child with no master_key, set
    master_key on AuthState, restart_all, observe that the respawn's
    env DOES contain MASTER_KEY."""
    captured_envs: list[dict[str, str]] = []

    async def fake_create(*_args: Any, **kwargs: Any) -> _FakeProcess:
        env = kwargs.get("env") or {}
        captured_envs.append(dict(env))
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    from eugene_plexus_agent import security

    auth = AuthState(signing_key=security.generate_signing_key())
    sup = Supervisor(log=logging.getLogger("test"), auth_state=auth)

    entry = ComponentEntry(
        name="left",
        kind=ComponentKind.inference_driver,
        url="http://127.0.0.1:8081",  # type: ignore[arg-type]
        spawn=SpawnConfig(configFile="/tmp/left/config.yaml"),
        safeMode=False,
    )
    sup.add_and_start(entry)
    await _wait_for(lambda: len(captured_envs) >= 1)
    # First spawn: no master_key.
    assert "EUGENE_PLEXUS_DRIVER_MASTER_KEY" not in captured_envs[0]

    # Operator logs in: master_key becomes available.
    auth.set_master_key(b"\x55" * 32)
    await sup.restart_all()

    # Wait for the respawn to land.
    await _wait_for(lambda: len(captured_envs) >= 2)
    # Second spawn: master_key threaded.
    assert "EUGENE_PLEXUS_DRIVER_MASTER_KEY" in captured_envs[1]

    await sup.stop_all()
