"""S0 of the hobbyist UX plan: the keyring entry is scoped per install, the availability probe is measured, and the
status endpoint reports both.

Why scoping: two installs share one machine more often than a single
slot assumed - the live worker and a `.dev-install`, the live worker and
every acceptance run. With `os_keyring` the desktop default, the second
install's wizard would have overwritten the first's stored key and the
first would have come back locked on its next start.
"""

from __future__ import annotations

import base64
import secrets

import keyring
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import keyring_store

from .conftest import TEST_PASSPHRASE
from .test_keyring import _FakeKeyring


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> _FakeKeyring:
    """The dict-keyed fake from `test_keyring.py`, installed here as well -
    a fixture imported by name is a redefinition to ruff (F811), so the
    five lines are repeated rather than re-exported."""
    fake = _FakeKeyring()
    monkeypatch.setattr(keyring, "get_password", fake.get)
    monkeypatch.setattr(keyring, "set_password", fake.set)
    monkeypatch.setattr(keyring, "delete_password", fake.delete)
    return fake


SALT_A = base64.b64encode(b"A" * 16).decode("ascii")
SALT_B = base64.b64encode(b"B" * 16).decode("ascii")


def _b64(key: bytes) -> str:
    return base64.b64encode(key).decode("ascii")


def test_two_installs_on_one_machine_do_not_share_an_entry(fake_keyring: _FakeKeyring) -> None:
    a = keyring_store.install_id_for(SALT_A)
    b = keyring_store.install_id_for(SALT_B)
    assert a != b
    assert len(a) == 12

    key_a, key_b = secrets.token_bytes(32), secrets.token_bytes(32)
    assert keyring_store.set_master_key(key_a, a) is True
    assert keyring_store.set_master_key(key_b, b) is True
    assert keyring_store.get_master_key(a) == key_a
    assert keyring_store.get_master_key(b) == key_b

    # Leaving os_keyring on one install must not touch the other's key.
    assert keyring_store.delete_master_key(a) is True
    assert keyring_store.get_master_key(a) is None
    assert keyring_store.get_master_key(b) == key_b


def test_the_install_id_is_stable_across_restarts() -> None:
    assert keyring_store.install_id_for(SALT_A) == keyring_store.install_id_for(SALT_A)


def test_the_probe_is_a_round_trip_and_leaves_nothing_behind(fake_keyring: _FakeKeyring) -> None:
    keyring_store.reset_probe_cache()
    assert keyring_store.probe_sync() is True
    assert fake_keyring.store == {}
    # Memoised: the answer is measured once per process. A backend that
    # breaks later does not flip it, and a desktop keyring's first-contact
    # dialog is shown once, not on every page load.
    fake_keyring.raise_on.add("set")
    assert keyring_store.probe_sync() is True


def test_the_probe_says_no_when_the_backend_refuses_writes(fake_keyring: _FakeKeyring) -> None:
    keyring_store.reset_probe_cache()
    fake_keyring.raise_on.add("set")
    assert keyring_store.probe_sync() is False


def test_a_refused_probe_does_not_read_as_a_failed_unlock(
    fake_keyring: _FakeKeyring, caplog: pytest.LogCaptureFixture
) -> None:
    """The probe once logged the master-key write's *"master key NOT
    persisted"*, which reads as a failed unlock when only the check ran.
    One line now, naming the check and the backend's reason, and no
    warning at all: nothing went wrong that anyone must act on."""
    keyring_store.reset_probe_cache()
    fake_keyring.raise_on.add("set")
    with caplog.at_level("DEBUG", logger=keyring_store.__name__):
        assert keyring_store.probe_sync() is False
    text = caplog.text
    assert "master key" not in text
    assert "keyring check" in text
    assert "nothing was unlocked" in text
    assert "refused" in text
    assert "simulated backend failure" in text  # the backend's own reason
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
    assert len(caplog.records) == 1


def test_the_probe_says_no_when_a_write_does_not_read_back(fake_keyring: _FakeKeyring) -> None:
    """A backend that accepts writes and returns nothing is the `fail`
    backend wearing a hat; a read-back is what makes this a measurement."""
    keyring_store.reset_probe_cache()
    fake_keyring.raise_on.add("get")
    assert keyring_store.probe_sync() is False
    # Nothing left behind even on the failing path.
    assert fake_keyring.store == {}


def test_status_reports_unlocked_and_keyring_availability(
    client: TestClient, fake_keyring: _FakeKeyring
) -> None:
    """What the wizard reads before it has a token, and what the Issues
    list will read after."""
    keyring_store.reset_probe_cache()
    fresh = client.get("/v1/auth/status")
    assert fresh.status_code == 200
    assert fresh.json() == {
        "initialized": False,
        "unlocked": False,
        "keyringAvailable": True,
        "passphraseFile": False,
    }

    assert (
        client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
    )
    after = client.get("/v1/auth/status").json()
    assert after["initialized"] is True
    assert after["unlocked"] is True
    assert after["keyringAvailable"] is True


def test_status_says_no_keyring_where_there_is_none(client: TestClient) -> None:
    """The conftest memoises the probe to False - the headless case - so
    the wizard defaults to the passphrase prompt and says why."""
    assert client.get("/v1/auth/status").json()["keyringAvailable"] is False
