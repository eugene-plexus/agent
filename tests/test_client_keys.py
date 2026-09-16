"""Client keys: mint, list, revoke, and what the audience keeps out.

Hobbyist UX S4 (design `specs/docs/design/hobbyist-ux.md` §7 S4,
decision #7). §0.8 measured the gap these close: the only bearer a
person could paste into Continue or Open WebUI was the operator session
token, and it can do everything and dies in a fortnight.

The load-bearing assertions here are the negative ones. A client key
that opened any operator surface would be a worse credential than the
session token it replaces, so the tests that matter most are the ones
that hand a client token to endpoints on this agent and watch them
refuse -- including through `require_operator_or_service`, which is the
dependency a *new* endpoint is most likely to reach for.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import client_keys, security
from eugene_plexus_agent.auth_state import AuthState
from eugene_plexus_agent.client_keys import ClientKeyRecord, ClientKeyStore

# --------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------- #


def _record(
    key_id: str = "abc",
    *,
    name: str = "Continue",
    created: float = 1000.0,
    expires: float = 9_000_000_000.0,
    revoked: float | None = None,
) -> ClientKeyRecord:
    return ClientKeyRecord(
        id=key_id,
        name=name,
        tail="Xk9Q0z",
        created_at=created,
        expires_at=expires,
        revoked_at=revoked,
    )


def test_store_round_trips_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / client_keys.KEYS_FILE
    store = ClientKeyStore(path)
    store.add(_record("one", name="Continue"))
    store.add(_record("two", name="phone"))

    reopened = ClientKeyStore(path)
    reopened.load()
    assert [r.id for r in reopened.records()] == ["two", "one"] or [
        r.id for r in reopened.records()
    ] == ["one", "two"]
    assert {r.name for r in reopened.records()} == {"Continue", "phone"}


def test_the_file_never_contains_a_token(tmp_path: Path) -> None:
    """The one property worth asserting about the file on disk.

    A record file that accidentally kept the token would turn a
    convenience into a credential store, and nothing else in this module
    would look different.
    """
    path = tmp_path / client_keys.KEYS_FILE
    store = ClientKeyStore(path)
    key = security.generate_signing_key()
    token, exp = security.issue_client_token(signing_key=key, key_id="one", name="Continue")
    store.add(
        ClientKeyRecord(
            id="one",
            name="Continue",
            tail=client_keys.tail_of(token),
            created_at=time.time(),
            expires_at=float(exp),
        )
    )
    written = path.read_text(encoding="utf-8")
    assert token not in written
    # The tail is short enough that finding it proves nothing about the
    # token, and long enough to identify the record.
    assert client_keys.tail_of(token) in written
    assert len(client_keys.tail_of(token)) == client_keys.TAIL_LENGTH


def test_revoke_is_idempotent_and_keeps_the_first_timestamp(tmp_path: Path) -> None:
    store = ClientKeyStore(tmp_path / client_keys.KEYS_FILE)
    store.add(_record("one"))
    first = store.revoke("one", now=500.0)
    second = store.revoke("one", now=900.0)
    assert first is not None and second is not None
    assert first.revoked_at == 500.0
    assert second.revoked_at == 500.0, "a repeated revoke must not re-stamp when it happened"


def test_revision_moves_on_a_revoke_and_not_on_a_mint(tmp_path: Path) -> None:
    store = ClientKeyStore(tmp_path / client_keys.KEYS_FILE)
    _, before = store.revoked()
    store.add(_record("one"))
    _, after_mint = store.revoked()
    assert after_mint == before, "the gateway cannot see a mint, so the revision must not move"
    store.revoke("one")
    _, after_revoke = store.revoked()
    assert after_revoke == before + 1


def test_an_expired_key_leaves_the_revoked_set_and_the_list(tmp_path: Path) -> None:
    """A revocation list that only grows is a leak.

    An expired token is refused by its own `exp` with no list consulted,
    so keeping its id here buys nothing.
    """
    path = tmp_path / client_keys.KEYS_FILE
    store = ClientKeyStore(path)
    store.add(_record("old", expires=100.0, revoked=50.0))
    store.add(_record("new", expires=9_000_000_000.0, revoked=60.0))
    ids, _ = store.revoked(now=200.0)
    assert ids == ["new"]
    assert [r.id for r in store.records(now=200.0)] == ["new"]


def test_an_unreadable_record_is_dropped_and_the_rest_load(tmp_path: Path) -> None:
    """`degraded-mode-required`, applied to the agent's own files."""
    path = tmp_path / client_keys.KEYS_FILE
    path.write_text(
        json.dumps(
            {
                "revision": 3,
                "keys": [
                    {
                        "id": "good",
                        "name": "ok",
                        "tail": "abcdef",
                        "createdAt": 1,
                        "expiresAt": 9e9,
                    },
                    {"id": "bad", "name": "no expiry"},
                    "not even an object",
                ],
            }
        ),
        encoding="utf-8",
    )
    store = ClientKeyStore(path)
    store.load()
    assert [r.id for r in store.records()] == ["good"]
    _, revision = store.revoked()
    assert revision == 3, "the revision survives a partial read; a reader must not see it go back"


def test_a_corrupt_file_is_a_warning_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / client_keys.KEYS_FILE
    path.write_text("this is not json", encoding="utf-8")
    store = ClientKeyStore(path)
    store.load()
    assert store.records() == []


def test_a_missing_file_writes_nothing(tmp_path: Path) -> None:
    """Every install has no client keys until someone mints one; that is
    not a reason to create a file."""
    path = tmp_path / client_keys.KEYS_FILE
    ClientKeyStore(path).load()
    assert not path.exists()


def test_new_key_ids_are_random(tmp_path: Path) -> None:
    ids = {client_keys.new_key_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) == 32 for i in ids)


# --------------------------------------------------------------------- #
# The token
# --------------------------------------------------------------------- #


def test_a_client_token_carries_the_audience_and_the_id() -> None:
    key = security.generate_signing_key()
    token, exp = security.issue_client_token(
        signing_key=key, key_id="deadbeef", name="Open WebUI", ttl_seconds=3600
    )
    payload = security.decode_token(
        token=token, signing_key=key, expected_audience=security.AUDIENCE_CLIENT
    )
    assert payload.aud == "client"
    assert payload.jti == "deadbeef"
    assert payload.sub == "Open WebUI"
    assert exp - payload.iat == 3600


def test_an_operator_session_still_carries_no_jti() -> None:
    """`jti` is not in the `require` list, and this is why.

    Every token minted before 2026-09-15 has none -- including the
    session the operator is holding while the agent is upgraded under
    them. Requiring it would log the whole install out.
    """
    key = security.generate_signing_key()
    token, _ = security.issue_operator_token(signing_key=key)
    payload = security.decode_token(
        token=token, signing_key=key, expected_audience=security.AUDIENCE_OPERATOR
    )
    assert payload.jti is None


# --------------------------------------------------------------------- #
# The endpoints
# --------------------------------------------------------------------- #


def _mint(client: TestClient, name: str = "Continue", **body: object) -> dict:
    resp = client.post("/v1/auth/client-keys", json={"name": name, **body})
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_mint_returns_the_token_once_and_the_list_never_does(authed_client: TestClient) -> None:
    made = _mint(authed_client, "Continue on the laptop")
    token = made["token"]
    assert token and token.count(".") == 2

    listed = authed_client.get("/v1/auth/client-keys")
    assert listed.status_code == 200
    keys = listed.json()["keys"]
    assert len(keys) == 1
    assert keys[0]["name"] == "Continue on the laptop"
    assert keys[0]["id"] == made["key"]["id"]
    assert "token" not in keys[0]
    assert token not in listed.text


def test_the_minted_token_verifies_with_the_installs_signing_key(
    app: FastAPI, authed_client: TestClient
) -> None:
    made = _mint(authed_client)
    auth: AuthState = app.state.auth_state
    payload = security.decode_token(
        token=made["token"],
        signing_key=auth.signing_key,
        expected_audience=security.AUDIENCE_CLIENT,
    )
    assert payload.jti == made["key"]["id"]


def test_the_default_life_is_a_year(authed_client: TestClient) -> None:
    made = _mint(authed_client)
    created = made["key"]["createdAt"]
    expires = made["key"]["expiresAt"]
    # Parsed loosely: what matters is the span, not the serialization.
    from datetime import datetime

    days = (datetime.fromisoformat(expires) - datetime.fromisoformat(created)).days
    assert 364 <= days <= 366


def test_an_explicit_lifetime_is_honoured(authed_client: TestClient) -> None:
    made = _mint(authed_client, ttlDays=7)
    from datetime import datetime

    days = (
        datetime.fromisoformat(made["key"]["expiresAt"])
        - datetime.fromisoformat(made["key"]["createdAt"])
    ).days
    assert days == 7


@pytest.mark.parametrize("ttl", [0, 3651])
def test_a_lifetime_out_of_range_is_refused(authed_client: TestClient, ttl: int) -> None:
    resp = authed_client.post("/v1/auth/client-keys", json={"name": "x", "ttlDays": ttl})
    assert resp.status_code == 422


def test_a_blank_name_is_refused_with_a_sentence(authed_client: TestClient) -> None:
    resp = authed_client.post("/v1/auth/client-keys", json={"name": "   "})
    assert resp.status_code == 422
    assert "name" in resp.text.lower()


def test_revoke_puts_the_id_in_the_set_and_bumps_the_revision(authed_client: TestClient) -> None:
    made = _mint(authed_client)
    key_id = made["key"]["id"]
    before = authed_client.get("/v1/auth/client-keys/revoked").json()
    assert before["ids"] == []

    assert authed_client.delete(f"/v1/auth/client-keys/{key_id}").status_code == 204
    after = authed_client.get("/v1/auth/client-keys/revoked").json()
    assert after["ids"] == [key_id]
    assert after["revision"] > before["revision"]

    listed = authed_client.get("/v1/auth/client-keys").json()["keys"]
    assert listed[0]["revokedAt"] is not None or "revokedAt" in listed[0]


def test_revoking_twice_is_still_204(authed_client: TestClient) -> None:
    key_id = _mint(authed_client)["key"]["id"]
    assert authed_client.delete(f"/v1/auth/client-keys/{key_id}").status_code == 204
    assert authed_client.delete(f"/v1/auth/client-keys/{key_id}").status_code == 204


def test_revoking_an_unknown_id_says_where_records_live(authed_client: TestClient) -> None:
    resp = authed_client.delete("/v1/auth/client-keys/nope")
    assert resp.status_code == 404
    assert "gateway" in resp.text, "the 404 must name where the records actually are"


def test_the_records_survive_a_restart(app: FastAPI, settings, tmp_path: Path) -> None:
    """The store is wired from `settings.config_file`'s directory, and a
    second process finds what the first minted."""
    with TestClient(app) as c:
        resp = c.post("/v1/auth/initialize", json={"passphrase": "correct horse battery staple"})
        c.headers["Authorization"] = f"Bearer {resp.json()['sessionToken']}"
        made = _mint(c, "phone")
    written = settings.config_file.resolve().parent / client_keys.KEYS_FILE
    assert written.exists()
    reopened = ClientKeyStore(written)
    reopened.load()
    assert [r.name for r in reopened.records()] == ["phone"]
    assert made["token"] not in written.read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
# What the audience keeps out -- the assertions that matter most
# --------------------------------------------------------------------- #


def _client_token(app: FastAPI) -> str:
    auth: AuthState = app.state.auth_state
    token, _ = security.issue_client_token(
        signing_key=auth.signing_key, key_id="probe", name="probe"
    )
    return token


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/v1/components"),
        ("get", "/v1/runtimes"),
        ("get", "/v1/config"),
        ("get", "/v1/node"),
        ("get", "/v1/auth/client-keys"),
        ("get", "/v1/auth/client-keys/revoked"),
        ("get", "/v1/directories"),
    ],
)
def test_a_client_key_opens_nothing_on_the_agent(
    app: FastAPI, authed_client: TestClient, method: str, path: str
) -> None:
    """The whole point of a third audience.

    `/v1/components` and `/v1/node` go through
    `require_operator_or_service`, which accepts *any* `service:*` --
    and `client` is deliberately not one, so these refuse without the
    dependency having been told about client keys at all.
    """
    token = _client_token(app)
    resp = getattr(authed_client, method)(path, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401, f"{path} accepted a client key: {resp.status_code}"


def test_a_client_key_cannot_mint_another(app: FastAPI, authed_client: TestClient) -> None:
    """Otherwise a leaked key is a key factory."""
    token = _client_token(app)
    resp = authed_client.post(
        "/v1/auth/client-keys",
        json={"name": "escalation"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


def test_the_gateways_service_token_may_read_the_revoked_set(
    app: FastAPI, authed_client: TestClient
) -> None:
    auth: AuthState = app.state.auth_state
    gateway = security.issue_service_token(signing_key=auth.signing_key, kind="gateway")
    resp = authed_client.get(
        "/v1/auth/client-keys/revoked", headers={"Authorization": f"Bearer {gateway}"}
    )
    assert resp.status_code == 200


@pytest.mark.parametrize("kind", ["library", "inference-driver", "control"])
def test_another_components_service_token_may_not(
    app: FastAPI, authed_client: TestClient, kind: str
) -> None:
    """Narrowed to `service:gateway` exactly, the way starting and
    stopping a runtime already is. A leaked driver token learns nothing
    about which keys an operator turned off."""
    auth: AuthState = app.state.auth_state
    other = security.issue_service_token(signing_key=auth.signing_key, kind=kind)
    resp = authed_client.get(
        "/v1/auth/client-keys/revoked", headers={"Authorization": f"Bearer {other}"}
    )
    assert resp.status_code == 401


def test_a_service_token_may_not_list_the_keys(app: FastAPI, authed_client: TestClient) -> None:
    auth: AuthState = app.state.auth_state
    gateway = security.issue_service_token(signing_key=auth.signing_key, kind="gateway")
    resp = authed_client.get("/v1/auth/client-keys", headers={"Authorization": f"Bearer {gateway}"})
    assert resp.status_code == 401
