"""Row 5 (5b): an agent that keeps its own passphrase takes a new one.

`specs/docs/design/rows/row-05.md` §3. The agent keeps its master key, so
every secret it sealed (share passwords, the children's provider keys)
still opens, and wraps that key under the new passphrase: `agent.yaml`
gains `kekSalt` and `wrappedMasterKey` (R5-RESEAL, Troy 2026-10-21: re-wrap
only). The root sends the change over TLS, or on loopback; a machine that
was off takes it at its next sign-in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import security, share_credentials
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.settings import Settings

from .conftest import TEST_PASSPHRASE, FakeRoot, enroll_app

NEW = "a-new-and-longer-passphrase"
NODE = "gpu-box"


def _auth_block(settings: Settings) -> dict[str, Any]:
    return dict(yaml.safe_load(settings.config_file.read_text(encoding="utf-8"))["auth"])


def _fresh(settings: Settings) -> FastAPI:
    """The same agent, started again on the same files."""
    from .conftest import StubRuntimeSupervisor, StubSupervisor

    again = create_app(settings=settings)
    again.state.supervisor = StubSupervisor()
    again.state.runtime_supervisor = StubRuntimeSupervisor()
    return again


def test_wrapped_key_keeps_share_password_open(
    authed_client: TestClient, settings: Settings
) -> None:
    c = authed_client
    saved = c.patch(
        "/v1/config",
        json={"shareCredentials": [{"host": "nas", "username": "u", "password": "hunter2"}]},
    )
    assert saved.status_code == 200 and saved.json()["rejected"] == [], saved.text
    key = c.app.state.auth_state.master_key  # type: ignore[attr-defined]
    salt_before = _auth_block(settings)["masterSalt"]

    changed = c.post("/v1/auth/passphrase", json={"current": TEST_PASSPHRASE, "new": NEW})
    assert changed.status_code == 200, changed.text
    assert changed.json()["machines"] == [] and changed.json()["passphraseFile"] == "none"
    assert TEST_PASSPHRASE not in changed.text and NEW not in changed.text

    block = _auth_block(settings)
    assert block["masterSalt"] == salt_before, "the master key, and so the keyring id, stay"
    assert block["kekSalt"] and block["wrappedMasterKey"]
    raw = settings.config_file.read_text(encoding="utf-8")
    assert NEW not in raw and TEST_PASSPHRASE not in raw

    with TestClient(_fresh(settings)) as again:
        assert again.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE}).status_code == 401
        assert again.post("/v1/auth/login", json={"passphrase": NEW}).status_code == 200
        state = again.app.state  # type: ignore[attr-defined]
        assert state.auth_state.master_key == key
        stored = state.agent_state.get_config("shareCredentials")
        opened = share_credentials.unseal_entries(stored, state.auth_state.master_key)
        assert opened[0]["password"] == "hunter2"

        # And a second change wraps the same key again.
        again.headers["Authorization"] = (
            "Bearer "
            + again.post("/v1/auth/login", json={"passphrase": NEW}).json()["sessionToken"]
        )
        assert (
            again.post("/v1/auth/passphrase", json={"current": NEW, "new": NEW + "-2"}).status_code
            == 200
        )
    with TestClient(_fresh(settings)) as third:
        assert third.post("/v1/auth/login", json={"passphrase": NEW + "-2"}).status_code == 200
        assert third.app.state.auth_state.master_key == key  # type: ignore[attr-defined]


def test_the_change_needs_the_current_passphrase_and_a_long_new_one(
    authed_client: TestClient,
) -> None:
    c = authed_client
    wrong = c.post("/v1/auth/passphrase", json={"current": "not-it-at-all", "new": NEW})
    assert wrong.status_code == 401 and "not-it-at-all" not in wrong.text
    short = c.post("/v1/auth/passphrase", json={"current": TEST_PASSPHRASE, "new": "xq7-tiny"})
    assert short.status_code == 422 and "xq7-tiny" not in short.text
    anonymous = c.post(
        "/v1/auth/passphrase",
        json={"current": TEST_PASSPHRASE, "new": NEW},
        headers={"Authorization": ""},
    )
    assert anonymous.status_code == 401


def _enrolled_with_own(c: TestClient) -> FakeRoot:
    assert c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
    root = FakeRoot()
    enroll_app(c.app, root, NODE)  # type: ignore[arg-type]
    return root


def test_the_root_sends_the_new_passphrase_over_tls_only(
    client: TestClient, settings: Settings
) -> None:
    c = client
    root = _enrolled_with_own(c)
    key = c.app.state.auth_state.master_key  # type: ignore[attr-defined]
    control = {"Authorization": f"Bearer {root.service(f'node:{NODE}')}"}
    body = {"current": TEST_PASSPHRASE, "new": NEW}

    # Not over TLS and not from this machine: nothing changes.
    plain = c.post("http://testserver/v1/node/passphrase", json=body, headers=control)
    assert plain.status_code == 403, plain.text
    # An operator's session is not the root.
    session = {"Authorization": f"Bearer {root.session(f'node:{NODE}', 'control')}"}
    assert (
        c.post("https://testserver/v1/node/passphrase", json=body, headers=session).status_code
        == 401
    )

    taken = c.post("https://testserver/v1/node/passphrase", json=body, headers=control)
    assert taken.status_code == 204, taken.text
    state = c.app.state.agent_state  # type: ignore[attr-defined]
    assert security.verify_passphrase(NEW, state.get_passphrase_hash())
    assert c.app.state.auth_state.master_key == key  # type: ignore[attr-defined]
    assert state.master_key_for(NEW) == key

    # A replay carries a `current` that no longer verifies.
    again = c.post("https://testserver/v1/node/passphrase", json=body, headers=control)
    assert again.status_code == 409
    assert again.json()["detail"]["type"].endswith("passphrase-differs")

    identity = c.get("/v1/node", headers=session).json()
    assert identity["ownPassphrase"] is True


def test_a_machine_with_no_passphrase_of_its_own_says_so(client: TestClient) -> None:
    root = FakeRoot()
    enroll_app(client.app, root, NODE)  # type: ignore[arg-type]
    control = {"Authorization": f"Bearer {root.service(f'node:{NODE}')}"}
    answer = client.post(
        "https://testserver/v1/node/passphrase",
        json={"current": TEST_PASSPHRASE, "new": NEW},
        headers=control,
    )
    assert answer.status_code == 409
    assert answer.json()["detail"]["type"].endswith("no-own-passphrase")
    session = {"Authorization": f"Bearer {root.session(f'node:{NODE}', 'control')}"}
    assert client.get("/v1/node", headers=session).json()["ownPassphrase"] is False


def _root_that_knows(c: TestClient, root: FakeRoot, passphrase: str) -> list[dict[str, Any]]:
    """The root's sign-in, for this machine: a session for `passphrase` only."""
    seen: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        seen.append(body)
        if request.url.path.endswith("/v1/auth/login") and body.get("passphrase") == passphrase:
            return httpx.Response(
                200,
                json={
                    "sessionToken": root.session(f"node:{NODE}", "control"),
                    "expiresAt": "2099-01-01T00:00:00+00:00",
                    "operatorName": "operator",
                },
            )
        return httpx.Response(401, json={"detail": {"title": "Wrong passphrase"}})

    c.app.state.control_transport = httpx.MockTransport(handle)  # type: ignore[attr-defined]
    return seen


def test_a_machine_that_was_off_takes_the_new_passphrase_at_its_next_sign_in(
    client: TestClient, settings: Settings
) -> None:
    c = client
    root = _enrolled_with_own(c)
    key = c.app.state.auth_state.master_key  # type: ignore[attr-defined]
    _root_that_knows(c, root, NEW)

    # Unlocked: the root accepts what this machine refuses, so it is the
    # new passphrase, and the key is wrapped under it at once.
    signed_in = c.post("/v1/auth/login", json={"passphrase": NEW})
    assert signed_in.status_code == 200, signed_in.text
    state = c.app.state.agent_state  # type: ignore[attr-defined]
    assert security.verify_passphrase(NEW, state.get_passphrase_hash())
    assert state.master_key_for(NEW) == key


def test_a_locked_machine_asks_once_for_the_old_passphrase(
    client: TestClient, settings: Settings
) -> None:
    c = client
    root = _enrolled_with_own(c)
    key = c.app.state.auth_state.master_key  # type: ignore[attr-defined]
    c.app.state.auth_state.master_key = None  # type: ignore[attr-defined]
    _root_that_knows(c, root, NEW)

    asked = c.post("/v1/auth/login", json={"passphrase": NEW})
    assert asked.status_code == 409, asked.text
    problem = asked.json()["detail"]
    assert problem["type"].endswith("old-passphrase-needed")
    assert problem["detail"] == (
        "This machine was off when the passphrase changed. Enter the old passphrase once to finish."
    )
    wrong = c.post(
        "/v1/auth/login", json={"passphrase": NEW, "previousPassphrase": "not-the-old-one"}
    )
    assert wrong.status_code == 401
    done = c.post("/v1/auth/login", json={"passphrase": NEW, "previousPassphrase": TEST_PASSPHRASE})
    assert done.status_code == 200, done.text
    state = c.app.state.agent_state  # type: ignore[attr-defined]
    assert c.app.state.auth_state.master_key == key  # type: ignore[attr-defined]
    assert state.master_key_for(NEW) == key
    assert not security.verify_passphrase(TEST_PASSPHRASE, state.get_passphrase_hash())


def test_a_session_that_does_not_verify_changes_nothing(client: TestClient) -> None:
    """The root's 200 is believed only for a session this machine's trust
    bundle verifies: a forged answer must not rewrap the key."""
    c = client
    _enrolled_with_own(c)
    stranger = FakeRoot()
    _root_that_knows(c, stranger, NEW)
    before = c.app.state.agent_state.get_passphrase_hash()  # type: ignore[attr-defined]
    answer = c.post("/v1/auth/login", json={"passphrase": NEW})
    assert answer.status_code in (401, 502), answer.text
    assert c.app.state.agent_state.get_passphrase_hash() == before  # type: ignore[attr-defined]


def test_the_passphrase_file_is_rewritten(tmp_path: Path) -> None:
    from .conftest import StubRuntimeSupervisor, StubSupervisor

    secret = tmp_path / "passphrase"
    settings = Settings(
        config_file=tmp_path / "agent.yaml", default_topology=False, passphrase_file=secret
    )
    app = create_app(settings=settings)
    app.state.supervisor = StubSupervisor()
    app.state.runtime_supervisor = StubRuntimeSupervisor()
    with TestClient(app) as c:
        token = c.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).json()
        c.headers["Authorization"] = f"Bearer {token['sessionToken']}"
        assert c.patch("/v1/config", json={"securityMode": "passphrase_file"}).status_code == 200
        assert c.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
        assert secret.read_text(encoding="utf-8") == TEST_PASSPHRASE
        changed = c.post("/v1/auth/passphrase", json={"current": TEST_PASSPHRASE, "new": NEW})
        assert changed.json()["passphraseFile"] == "matches"
        assert secret.read_text(encoding="utf-8") == NEW
        key = app.state.auth_state.master_key
    restarted = create_app(settings=settings)
    restarted.state.supervisor = StubSupervisor()
    restarted.state.runtime_supervisor = StubRuntimeSupervisor()
    with TestClient(restarted):
        assert restarted.state.auth_state.master_key == key


@pytest.mark.parametrize("bad", ["", "not-base64!", "AAAA.AAAA"])
def test_a_damaged_wrap_opens_nothing(bad: str) -> None:
    with pytest.raises(ValueError):
        security.unwrap_master_key(bad, NEW, "c2FsdHNhbHRzYWx0c2FsdA==")
