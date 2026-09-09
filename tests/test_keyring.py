"""Tests for Phase 8 — OS keyring auto-unlock.

The keyring library auto-selects an OS backend (WinVault / Keychain /
Secret Service). Tests can't depend on a real backend being present
and unlocked on CI, so they monkeypatch `keyring.get_password` /
`set_password` / `delete_password` with an in-memory dict that
behaves like an empty-then-populated store. The wrapper module's
graceful-degradation paths are exercised separately by raising
`KeyringError` from the patched functions.

Three integration scenarios matter and are covered:

  * `securityMode == os_keyring` on a fresh process with a stored
    master key → lifespan auto-unlocks; AuthState.master_key is
    populated before any child spawns.
  * Login under `securityMode == os_keyring` persists the derived
    master key for next time.
  * Patching `securityMode` away from `os_keyring` deletes the
    stored key — the install promised a stronger boundary, can't
    leave the auto-unlock secret behind.
"""

from __future__ import annotations

import base64
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import keyring
import keyring.errors
import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_agent import keyring_store, security
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.auth_state import AuthState
from eugene_plexus_agent.settings import Settings
from tests.conftest import TEST_PASSPHRASE, StubSupervisor

# --------------------------------------------------------------------------- #
# In-memory keyring backend
# --------------------------------------------------------------------------- #


class _FakeKeyring:
    """Dict-backed stand-in for the OS keyring. Each test gets a
    fresh one so cross-test state can't leak."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.raise_on: set[str] = set()

    def get(self, service: str, username: str) -> str | None:
        if "get" in self.raise_on:
            raise keyring.errors.KeyringError("simulated backend failure")
        return self.store.get((service, username))

    def set(self, service: str, username: str, password: str) -> None:
        if "set" in self.raise_on:
            raise keyring.errors.KeyringError("simulated backend failure")
        self.store[(service, username)] = password

    def delete(self, service: str, username: str) -> None:
        if "delete" in self.raise_on:
            raise keyring.errors.KeyringError("simulated backend failure")
        try:
            del self.store[(service, username)]
        except KeyError as e:
            raise keyring.errors.PasswordDeleteError(f"no value at {service!r}:{username!r}") from e


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> _FakeKeyring:
    """Patch the keyring module with the fake. The wrapper imports
    `keyring` at module level so we patch the top-level functions."""
    fake = _FakeKeyring()
    monkeypatch.setattr(keyring, "get_password", fake.get)
    monkeypatch.setattr(keyring, "set_password", fake.set)
    monkeypatch.setattr(keyring, "delete_password", fake.delete)
    return fake


# --------------------------------------------------------------------------- #
# Primitive wrapper behavior
# --------------------------------------------------------------------------- #


def test_keyring_store_round_trip(fake_keyring: _FakeKeyring) -> None:
    key = secrets.token_bytes(32)
    assert keyring_store.set_master_key(key) is True
    assert keyring_store.get_master_key() == key
    assert keyring_store.delete_master_key() is True
    assert keyring_store.get_master_key() is None


def test_keyring_store_returns_none_when_empty(fake_keyring: _FakeKeyring) -> None:
    assert keyring_store.get_master_key() is None


def test_keyring_store_set_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        keyring_store.set_master_key(b"\x00" * 16)


def test_keyring_store_get_handles_backend_failure(fake_keyring: _FakeKeyring) -> None:
    fake_keyring.raise_on.add("get")
    # Must not raise; must return None.
    assert keyring_store.get_master_key() is None


def test_keyring_store_set_handles_backend_failure(fake_keyring: _FakeKeyring) -> None:
    fake_keyring.raise_on.add("set")
    # Must not raise; must return False so callers can react.
    assert keyring_store.set_master_key(secrets.token_bytes(32)) is False


def test_keyring_store_delete_returns_false_when_empty(
    fake_keyring: _FakeKeyring,
) -> None:
    """Deleting a non-existent entry is a no-op from our perspective —
    not an error. (The wizard's "switching modes" flow calls delete
    blind without checking presence first.)"""
    assert keyring_store.delete_master_key() is False


def test_keyring_store_get_handles_garbage_b64(fake_keyring: _FakeKeyring) -> None:
    """If something corrupts the keyring entry (or a different program
    writes there), `get_master_key` must return None — never raise,
    never return a half-decoded value."""
    fake_keyring.store[(keyring_store.SERVICE, keyring_store.USERNAME)] = (
        "this-is-not-base64-padding-or-anything"
    )
    assert keyring_store.get_master_key() is None


def test_keyring_store_get_handles_wrong_length(fake_keyring: _FakeKeyring) -> None:
    """An entry that base64-decodes but isn't 32 bytes — discard."""
    fake_keyring.store[(keyring_store.SERVICE, keyring_store.USERNAME)] = base64.b64encode(
        b"\x00" * 16
    ).decode("ascii")
    assert keyring_store.get_master_key() is None


# --------------------------------------------------------------------------- #
# Lifespan auto-unlock
# --------------------------------------------------------------------------- #


def _seed_install(tmp_path: Path, *, security_mode: str) -> tuple[Settings, bytes]:
    """Build a agent.yaml with a passphrase set + the operator's
    chosen securityMode. Returns (settings, derived_master_key)."""
    settings = Settings(config_file=tmp_path / "agent.yaml")
    app1 = create_app(settings=settings)
    app1.state.supervisor = StubSupervisor()
    with TestClient(app1) as c:
        resp = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
        assert resp.status_code == 200
        c.headers["Authorization"] = f"Bearer {resp.json()['sessionToken']}"
        if security_mode != "prompt_on_startup":
            patch = c.patch("/v1/config", json={"securityMode": security_mode})
            assert patch.status_code == 200
    salt_b64 = yaml.safe_load((tmp_path / "agent.yaml").read_text())["auth"]["masterSalt"]
    salt = base64.b64decode(salt_b64)
    return settings, security.derive_master_key(TEST_PASSPHRASE, salt)


@pytest.fixture
def isolated_keyring(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeKeyring]:
    """Same as `fake_keyring` but yielded as a context — used by the
    multi-app flows that need the same isolated keyring backing
    across multiple TestClient lifespans."""
    fake = _FakeKeyring()
    monkeypatch.setattr(keyring, "get_password", fake.get)
    monkeypatch.setattr(keyring, "set_password", fake.set)
    monkeypatch.setattr(keyring, "delete_password", fake.delete)
    yield fake


def test_lifespan_auto_unlock_when_key_present(
    tmp_path: Path, isolated_keyring: _FakeKeyring
) -> None:
    """End-to-end: initialize under os_keyring mode → restart →
    master key auto-recovers from keyring at lifespan time, before
    children spawn."""
    settings, expected_key = _seed_install(tmp_path, security_mode="os_keyring")
    # First app: initialize() + PATCH securityMode persisted the key.
    assert isolated_keyring.store, "initialize should have stored master key"
    stored_b64 = next(iter(isolated_keyring.store.values()))
    assert base64.b64decode(stored_b64) == expected_key

    # Fresh app, same config file, same (isolated) keyring backing.
    fresh_app = create_app(settings=settings)
    fresh_app.state.supervisor = StubSupervisor()
    with TestClient(fresh_app):
        # Lifespan ran. AuthState.master_key should be populated
        # without any login round-trip.
        auth: AuthState = fresh_app.state.auth_state
        assert auth.has_master_key()
        assert auth.master_key == expected_key


def test_lifespan_does_not_auto_unlock_when_mode_is_prompt(
    tmp_path: Path, isolated_keyring: _FakeKeyring
) -> None:
    """Even if a stranger somehow left a key in the keyring under
    our service name, `prompt_on_startup` mode must NOT trust it.
    Defense against an attacker pre-seeding a known key into the
    OS store before the operator finishes the wizard."""
    settings, _ = _seed_install(tmp_path, security_mode="prompt_on_startup")
    # Plant a key in the keyring as if it had been left over.
    isolated_keyring.store[(keyring_store.SERVICE, keyring_store.USERNAME)] = base64.b64encode(
        b"\x77" * 32
    ).decode("ascii")

    fresh_app = create_app(settings=settings)
    fresh_app.state.supervisor = StubSupervisor()
    with TestClient(fresh_app):
        auth: AuthState = fresh_app.state.auth_state
        # The keyring entry was ignored because mode is prompt_on_startup.
        assert not auth.has_master_key()


def test_lifespan_no_auto_unlock_when_no_passphrase_set(
    tmp_path: Path, isolated_keyring: _FakeKeyring
) -> None:
    """Pre-init install (no passphrase yet). The keyring check should
    be skipped entirely — there's no install to unlock."""
    settings = Settings(config_file=tmp_path / "agent.yaml")
    # Pre-seed a key as if from an earlier install (operator wiped
    # agent.yaml but forgot to clear the keyring).
    isolated_keyring.store[(keyring_store.SERVICE, keyring_store.USERNAME)] = base64.b64encode(
        b"\x33" * 32
    ).decode("ascii")

    fresh_app = create_app(settings=settings)
    fresh_app.state.supervisor = StubSupervisor()
    with TestClient(fresh_app):
        auth: AuthState = fresh_app.state.auth_state
        # No passphrase on disk → no install to unlock → no auto-recover.
        assert not auth.has_master_key()


# --------------------------------------------------------------------------- #
# Login / initialize persist the master key when in keyring mode
# --------------------------------------------------------------------------- #


def test_initialize_persists_master_key_when_in_keyring_mode(
    tmp_path: Path, isolated_keyring: _FakeKeyring
) -> None:
    """First-run flow: wizard's initialize call should write the master
    key to the OS keyring IF the operator opted into os_keyring mode
    before initializing. The default mode is prompt_on_startup, so
    this test forces the mode via direct AgentState manipulation."""
    settings = Settings(config_file=tmp_path / "agent.yaml")
    app = create_app(settings=settings)
    app.state.supervisor = StubSupervisor()
    # Pre-set securityMode in the on-disk YAML so initialize sees it.
    # Direct dict manipulation rather than going through PATCH is fine —
    # the wizard's actual order is set-mode-then-initialize in some
    # designs and initialize-then-set-mode in others.
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.safe_dump({"securityMode": "os_keyring"}), encoding="utf-8")

    with TestClient(app) as c:
        resp = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
        assert resp.status_code == 200, resp.text
    # Master key persisted to (fake) keyring.
    assert isolated_keyring.store, "master key should be in keyring after initialize"


def test_login_persists_master_key_when_in_keyring_mode(
    tmp_path: Path, isolated_keyring: _FakeKeyring
) -> None:
    """Initialize under prompt mode, switch to keyring mode, then log
    in fresh → the login flow's persist step must populate keyring."""
    settings, expected = _seed_install(tmp_path, security_mode="prompt_on_startup")
    assert not isolated_keyring.store

    # Same operator now flips into keyring mode. Don't persist on flip
    # (operator hasn't been asked to re-auth yet); next login does it.
    config_path = tmp_path / "agent.yaml"
    on_disk = yaml.safe_load(config_path.read_text())
    on_disk["securityMode"] = "os_keyring"
    config_path.write_text(yaml.safe_dump(on_disk), encoding="utf-8")
    assert not isolated_keyring.store  # config patch on disk doesn't trigger save

    # Fresh process → login.
    fresh_app = create_app(settings=settings)
    fresh_app.state.supervisor = StubSupervisor()
    with TestClient(fresh_app) as c:
        resp = c.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
        assert resp.status_code == 200, resp.text
    assert isolated_keyring.store, "login should have persisted master key"
    stored_b64 = next(iter(isolated_keyring.store.values()))
    assert base64.b64decode(stored_b64) == expected


# --------------------------------------------------------------------------- #
# Switching securityMode away from os_keyring clears the keyring
# --------------------------------------------------------------------------- #


def test_patch_to_prompt_mode_deletes_keyring_entry(
    tmp_path: Path, isolated_keyring: _FakeKeyring
) -> None:
    """Operator opted into auto-unlock, then changed their mind. The
    PATCH must remove the stored secret so the next restart actually
    requires a passphrase prompt."""
    settings, _ = _seed_install(tmp_path, security_mode="os_keyring")
    assert isolated_keyring.store, "precondition: keyring populated"

    app = create_app(settings=settings)
    app.state.supervisor = StubSupervisor()
    with TestClient(app) as c:
        resp = c.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
        c.headers["Authorization"] = f"Bearer {resp.json()['sessionToken']}"
        patch = c.patch("/v1/config", json={"securityMode": "prompt_on_startup"})
        assert patch.status_code == 200
    assert not isolated_keyring.store, (
        "switching away from os_keyring must clear the stored auto-unlock key"
    )


def test_patch_to_keyring_mode_with_active_session_persists_key(
    tmp_path: Path, isolated_keyring: _FakeKeyring
) -> None:
    """The inverse: prompt → keyring with an already-unlocked session
    should persist the in-memory master key immediately so the next
    restart auto-recovers."""
    settings, expected = _seed_install(tmp_path, security_mode="prompt_on_startup")
    assert not isolated_keyring.store

    fresh_app = create_app(settings=settings)
    fresh_app.state.supervisor = StubSupervisor()
    with TestClient(fresh_app) as c:
        login = c.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
        c.headers["Authorization"] = f"Bearer {login.json()['sessionToken']}"
        patch = c.patch("/v1/config", json={"securityMode": "os_keyring"})
        assert patch.status_code == 200
    assert isolated_keyring.store, (
        "switching to os_keyring with an active session should persist the key"
    )
    stored_b64 = next(iter(isolated_keyring.store.values()))
    assert base64.b64decode(stored_b64) == expected


# --------------------------------------------------------------------------- #
# Failure path: backend write fails — login still completes
# --------------------------------------------------------------------------- #


def test_login_succeeds_even_when_keyring_write_fails(
    tmp_path: Path, isolated_keyring: _FakeKeyring
) -> None:
    """A failing OS keyring (locked Secret Service, etc.) must NOT
    block login. The operator can still use the install; they just
    don't get auto-recovery."""
    settings, _ = _seed_install(tmp_path, security_mode="os_keyring")
    isolated_keyring.store.clear()  # reset; treat as fresh install
    isolated_keyring.raise_on.add("set")

    fresh_app = create_app(settings=settings)
    fresh_app.state.supervisor = StubSupervisor()
    with TestClient(fresh_app) as c:
        resp = c.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
        assert resp.status_code == 200, resp.text
        # Authenticated session is fully usable.
        c.headers["Authorization"] = f"Bearer {resp.json()['sessionToken']}"
        assert c.get("/v1/config").status_code == 200


# --------------------------------------------------------------------------- #
# Mode-change without an active session
# --------------------------------------------------------------------------- #


def _login_via_loopback(tmp_path: Path, *, security_mode: str) -> tuple[Any, TestClient, bytes]:
    """Spin up an app + TestClient already authed under the given
    security mode. Returns (app, client, master_key)."""
    settings, expected = _seed_install(tmp_path, security_mode=security_mode)
    app = create_app(settings=settings)
    app.state.supervisor = StubSupervisor()
    client = TestClient(app)
    client.__enter__()
    resp = client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    client.headers["Authorization"] = f"Bearer {resp.json()['sessionToken']}"
    return app, client, expected


def test_mode_changes_no_op_when_mode_did_not_change(
    tmp_path: Path, isolated_keyring: _FakeKeyring
) -> None:
    """Patching securityMode to its current value must not trigger
    any keyring side-effect. Defends against an over-eager UI that
    re-sends the full config on every save."""
    _app, client, _ = _login_via_loopback(tmp_path, security_mode="os_keyring")
    isolated_keyring.store.clear()  # ensure we observe new writes only
    try:
        resp = client.patch("/v1/config", json={"securityMode": "os_keyring"})
        assert resp.status_code == 200
        # No write — set was never called because prior == new.
        assert not isolated_keyring.store
    finally:
        client.__exit__(None, None, None)
