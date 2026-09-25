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

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import tokens
from eugene_plexus_agent._generated.models import (
    ComponentEntry,
    ComponentStatus,
    ComputeDevice,
    ComputeDeviceKind,
    Runtime,
    RuntimeSpec,
    RuntimeStatus,
)
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.auth_state import AuthState
from eugene_plexus_agent.engines.devices import DeviceSnapshot
from eugene_plexus_agent.node_identity import NodeIdentityStore
from eugene_plexus_agent.runtimes import RuntimeSupervisor
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.trust import BUNDLE_FILE, NodeTrust

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
            # Mirrors the real supervisor: a start clears why it was stopped.
            self._stop_reasons.pop(spec.name, None)

    async def remove_and_stop(self, name: str) -> None:
        self.calls.append(("remove_and_stop", name))
        self.started.discard(name)

    async def restart(self, name: str) -> bool:
        self.calls.append(("restart", name))
        return name in self.started

    async def stop_one(self, name: str, *, reason: Any = None) -> None:
        self.calls.append(("stop_one", name))
        self.started.discard(name)
        if reason is not None:
            self._stop_reasons[name] = reason

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
            return runtime.model_copy(update={"status": RuntimeStatus.starting, "stopReason": None})
        return runtime


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # `default_topology=False` for the same reason the supervisor is a stub:
    # a unit test declares the topology it means to exercise, and every
    # tmp_path install is technically a first boot. The seeding behaviour
    # itself is opted into by tests/test_default_topology.py.
    return Settings(config_file=tmp_path / "agent.yaml", default_topology=False)


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
    # Admission measures every launch against live devices. Unit tests
    # get a fixed, generous card and no library, so a create is
    # deterministic and never shells out to nvidia-smi — and CI on a
    # GPU-less Linux runner behaves like this box.
    app.state.device_detector = lambda: fake_devices()
    app.state.library_fit_client = None
    # M11: admission refuses a model that is not on this host. Route
    # tests declare paths like `/models/q.gguf` that exist nowhere, so
    # the existence check is a seam here too -- everything is present
    # unless a test says otherwise.
    app.state.model_exists = lambda path: True
    return app


def fake_devices(
    *, free: int = 24 * 1024**3, total: int = 32 * 1024**3, count: int = 1
) -> DeviceSnapshot:
    """A device snapshot tests can shape: `count` CUDA cards of `total`
    bytes with `free` available, plus a CPU carrying host memory."""
    devices = [
        ComputeDevice(
            kind=ComputeDeviceKind.cuda,
            index=i,
            name=f"Fake GPU {i}",
            memoryTotalBytes=total,
            memoryFreeBytes=free,
        )
        for i in range(count)
    ]
    devices.append(
        ComputeDevice(
            kind=ComputeDeviceKind.cpu,
            index=0,
            name="Fake CPU",
            memoryTotalBytes=64 * 1024**3,
            memoryFreeBytes=40 * 1024**3,
        )
    )
    return DeviceSnapshot(
        devices=tuple(devices),
        warnings=(),
        ram_total_bytes=64 * 1024**3,
        ram_available_bytes=40 * 1024**3,
        detected_at=datetime.now(UTC),
    )


# --------------------------------------------------------------------------- #
# Trust: this node's keys, a standalone authority, and a fake control root
# (per-node token keys, 2026-09-25)
# --------------------------------------------------------------------------- #


def standalone_trust(directory: Path) -> NodeTrust:
    """A node that has joined nothing: its own authority, as `node:local`."""
    directory.mkdir(parents=True, exist_ok=True)
    store = NodeIdentityStore(directory / "node.yaml")
    store.ensure_keypair()
    trust = NodeTrust(store, directory / BUNDLE_FILE)
    trust.load()
    return trust


def standalone_auth(directory: Path, *, master_key: bytes | None = None) -> AuthState:
    return AuthState(trust=standalone_trust(directory), master_key=master_key)


@dataclass
class FakeRoot:
    """A control root, as far as one agent can tell: an identity key that
    signs bundles, and a token key that signs sessions and its own tokens."""

    identity: Ed25519PrivateKey = field(default_factory=tokens.generate_private_key)
    token: tokens.Signer = field(
        default_factory=lambda: tokens.Signer(key=tokens.generate_private_key(), issuer="control")
    )
    members: dict[str, tokens.TrustKey] = field(default_factory=dict)
    version: int = 1
    epoch: int = 1

    @property
    def public(self) -> str:
        return tokens.public_b64(self.identity)

    def register(self, name: str, key: Ed25519PublicKey, grants: tuple[str, ...] = ()) -> None:
        self.members[name] = tokens.TrustKey(
            kid=tokens.thumbprint(key),
            issuer=f"node:{name}",
            public=key,
            grants=frozenset({"node", *grants}),
        )

    def bundle(self, *, revoked: tuple[tuple[str, int], ...] = ()) -> tokens.TrustBundle:
        self.version += 1
        return tokens.build_bundle(
            authority=self.identity,
            version=self.version,
            epoch=self.epoch,
            keys=[self.token.trust_key(["authority"]), *self.members.values()],
            revoked_sessions=revoked,
        )

    def session(self, *aud: str, ttl: int = 3600, **extra: Any) -> str:
        token, _ = self.token.mint(
            typ=tokens.TYP_SESSION, sub="operator", aud=list(aud), ttl_seconds=ttl, extra=extra
        )
        return token

    def service(self, aud: str, *, sub: str = "control", ttl: int = 300) -> str:
        token, _ = self.token.mint(typ=tokens.TYP_SERVICE, sub=sub, aud=[aud], ttl_seconds=ttl)
        return token


def enroll_store(
    store: NodeIdentityStore,
    root: FakeRoot,
    name: str,
    *,
    control_url: str = "http://control.invalid:8083",
    grants: tuple[str, ...] = (),
) -> tokens.TrustBundle:
    """Make `store` enrolled with `root`, and keep the bundle beside it."""
    record = store.ensure_keypair()
    assert record.token_private_key is not None
    key = tokens.load_private(record.token_private_key).public_key()
    root.register(name, key, grants)
    bundle = root.bundle()
    tokens.write_bundle_file(store.path.parent / BUNDLE_FILE, bundle)
    store.record_enrollment(
        name=name,
        control_url=control_url,
        epoch=root.epoch,
        control_public_key=root.public,
        recovery_public_key=None,
        advertise_url=None,
    )
    return bundle


def enroll_app(app: FastAPI, root: FakeRoot, name: str, **kwargs: Any) -> tokens.TrustBundle:
    """Enroll a running test app's node with `root` and reload its trust."""
    bundle = enroll_store(app.state.node_identity, root, name, **kwargs)
    app.state.auth_state.trust.load()
    return bundle


def local_service_token(app: FastAPI, sub: str) -> str:
    """What this agent would hand a child of kind `sub` at spawn."""
    trust = app.state.auth_state.trust
    token, _ = trust.mint_service(sub=sub, audience=trust.recipient)
    return str(token)


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


# Every test runs with the ambient EUGENE_PLEXUS_* environment cleared.
#
# **Not hygiene -- two tests really do fail without it**, and they fail
# on a developer machine while CI stays green, because CI has no install
# and a developer's machine does. `install.ps1` sets
# `EUGENE_PLEXUS_AGENT_CONFIG_FILE` in the USER environment on purpose
# (a logon task inherits it), so every shell on that account carries it,
# and `Settings` reads the same prefix the installer writes. The
# symptoms were a bind-host assertion reading 0.0.0.0 and a planner
# assertion finding EUGENE_PLEXUS_* in a child's env -- both of them
# asserting about the developer's install rather than about the code.
#
# Same root cause as the acceptance scripts, which clear this explicitly
# for the same reason; see `scripts/context-honesty-acceptance.sh` in
# the specs repo for what it costs when it is missed.
# `GET /v1/node` reads the host firewall and asks what would restart this
# agent. Both answer differently on every developer machine and again in
# CI -- the library's `recommend()` test read a live GPU and failed
# whenever one was busy, and this is the same shape. Pinned to a
# no-firewall host with nothing supervising the agent; a test about
# reach overrides them.
@pytest.fixture(autouse=True)
def _pin_host_environment_probes(app: FastAPI) -> None:
    from eugene_plexus_agent._generated.models import (
        AgentRestart,
        FirewallPort,
        HostFirewall,
        Mechanism,
        Verdict,
    )

    app.state.firewall_reader = lambda query: HostFirewall(
        supported=True,
        product="Test firewall",
        enabled=False,
        ports=[FirewallPort(port=p, verdict=Verdict.allowed) for p in query.ports],
    )
    app.state.restart_describer = lambda: AgentRestart(
        mechanism=Mechanism.none, canSelfRestart=False, command="eugene-plexus-agent"
    )


@pytest.fixture(autouse=True)
def _isolate_ambient_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [k for k in os.environ if k.startswith("EUGENE_PLEXUS_")]:
        monkeypatch.delenv(key, raising=False)


# Process-wide HTTP clients (`_http.shared_internal_client`) outlive a
# test by construction -- that is the point of them, since building one
# parses certifi's PEM bundle at ~104 ms of synchronous CPU on the event
# loop. The cost in a test session is that a test which installs a
# `MockTransport` would leave it in front of every later test, which is
# how eight tests failed the first time this landed. Every test starts
# with an empty registry. Dropped rather than closed: a test owns
# whatever it installed, and nothing here holds a real socket.
@pytest.fixture(autouse=True)
def _isolate_shared_http_clients() -> Iterator[None]:
    from eugene_plexus_agent import _http

    _http.reset_shared()
    yield
    _http.reset_shared()


# `GET /v1/auth/status` probes the OS keyring once per process. In a test
# that is the developer's real Credential Manager or a CI runner's absent
# Secret Service - neither is the subject. Every test starts with the
# probe memoised to False; a test about the probe itself calls
# `keyring_store.reset_probe_cache()` after installing its fake.
@pytest.fixture(autouse=True)
def _memoise_keyring_probe() -> Iterator[None]:
    from eugene_plexus_agent import keyring_store

    keyring_store._probe_result = False
    keyring_store._probe_done = True
    yield
    keyring_store.reset_probe_cache()
