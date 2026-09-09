"""Pytest fixtures shared across the agent test suite.

Tests run with a no-op stub supervisor injected on `app.state.supervisor`
so the routes layer's spawn / restart / stop calls become observable
no-ops instead of trying to fork real Python processes for the body
components. End-to-end smoke tests against real children belong in a
separate harness, not the unit suite.

v0.2 adds auth-protected routes. Tests get an `authed_client` fixture
that initializes a passphrase, captures the resulting session token,
and attaches it to every request. Tests that exercise the auth surface
itself (login flow, rate limiting, token validation) use the bare
`client` fixture so they control auth themselves.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent._generated.models import (
    ComponentEntry,
    ComponentStatus,
    Runtime,
    RuntimeSpec,
    RuntimeStatus,
)
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.runtimes import RuntimeSupervisor
from eugene_plexus_agent.settings import Settings

TEST_PASSPHRASE = "correct horse battery staple"


class StubSupervisor:
    """No-op stand-in for `Supervisor` used in unit tests.

    Records every call so tests can assert which lifecycle methods the
    routes layer invoked, without ever actually spawning a child."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.restart_all_returns: list[str] = []

    def add_and_start(self, entry: ComponentEntry) -> None:
        self.calls.append(("add_and_start", entry.name))

    async def remove_and_stop(self, name: str) -> None:
        self.calls.append(("remove_and_stop", name))

    async def restart(self, name: str) -> bool:
        self.calls.append(("restart", name))
        return True

    async def restart_all(self) -> list[str]:
        """Records the call so the restart-on-login tests can assert
        the auth route triggered it. Returns an empty list by default
        — tests that need a non-empty result set `restart_all_returns`."""
        self.calls.append(("restart_all", ""))
        return list(self.restart_all_returns)

    async def start_health_loop(self, _get_components: Any) -> None:
        self.calls.append(("start_health_loop", ""))

    async def stop_all(self) -> None:
        self.calls.append(("stop_all", ""))

    def status_for(
        self, _name: str, *, has_spawn: bool
    ) -> tuple[ComponentStatus, str | None, datetime | None, int | None]:
        # Mirrors the real supervisor's "no info available" answer so
        # tests see deterministic placeholder status.
        return ComponentStatus.unreachable, None, None, None


class StubRuntimeSupervisor(RuntimeSupervisor):
    """`RuntimeSupervisor` that records lifecycle calls but never spawns.

    Subclasses the real thing rather than reimplementing it so route
    tests still exercise the genuine `compose()` — url building, alias
    defaulting, the declaration/observation pairing. Only the two methods
    that would fork `llama-server` are replaced.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []
        self.started: set[str] = set()

    def add_and_start(self, spec: RuntimeSpec) -> None:
        self.calls.append(("add_and_start", spec.name))
        if spec.autoStart is not False:
            self.started.add(spec.name)

    async def remove_and_stop(self, name: str) -> None:
        self.calls.append(("remove_and_stop", name))
        self.started.discard(name)

    async def restart(self, name: str) -> bool:
        self.calls.append(("restart", name))
        return name in self.started

    async def stop_one(self, name: str) -> None:
        self.calls.append(("stop_one", name))
        self.started.discard(name)

    async def stop_all(self) -> None:
        self.calls.append(("stop_all", ""))
        self.started.clear()

    async def start_readiness_loop(self, _get_specs: Any) -> None:
        self.calls.append(("start_readiness_loop", ""))

    def is_running(self, name: str) -> bool:
        return name in self.started

    def compose(self, spec: RuntimeSpec) -> Runtime:
        runtime = super().compose(spec)
        # No real process exists, so the parent reports `stopped`. Report
        # `starting` for anything this stub was asked to start, which is
        # what a just-created runtime actually looks like.
        if spec.name in self.started:
            return runtime.model_copy(update={"status": RuntimeStatus.starting})
        return runtime


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(config_file=tmp_path / "agent.yaml")


@pytest.fixture
def stub_supervisor() -> StubSupervisor:
    return StubSupervisor()


@pytest.fixture
def stub_runtime_supervisor() -> StubRuntimeSupervisor:
    return StubRuntimeSupervisor()


@pytest.fixture
def app(
    settings: Settings,
    stub_supervisor: StubSupervisor,
    stub_runtime_supervisor: StubRuntimeSupervisor,
) -> FastAPI:
    app = create_app(settings=settings)
    app.state.supervisor = stub_supervisor
    app.state.runtime_supervisor = stub_runtime_supervisor
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Bare TestClient — no auth headers attached. Use for testing
    the auth surface itself (login, rate limiting, etc.)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def authed_client(app: FastAPI) -> Iterator[TestClient]:
    """TestClient with a pre-initialized passphrase and the resulting
    session token attached as the default Authorization header.
    Use this for testing any v0.2-protected route."""
    with TestClient(app) as c:
        resp = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
        assert resp.status_code == 200, f"initialize failed: {resp.status_code} {resp.text}"
        token = resp.json()["sessionToken"]
        c.headers["Authorization"] = f"Bearer {token}"
        yield c
