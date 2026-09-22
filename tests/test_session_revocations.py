"""Signing out has to reach the proxy, and has to survive a restart.

The UI signs out with `DELETE /v1/auth/sessions/current`, which put the
token in a set in this process's memory that only the agent's own auth
dependencies consulted. Two gaps followed, both found by reading:

* The gateway, the library, the drivers and the control root verify
  tokens themselves and never saw that set, so a signed-out token kept
  working through `/api/proxy/<target>/...` -- the path every browser
  uses -- for the rest of its 14 days.
* A restart forgot every sign-out, and on an enrolled node the signing
  key survives a restart, so the token was simply valid again.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import security, session_revocations
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.auth_state import AuthState
from eugene_plexus_agent.session_revocations import REVOKED_SESSIONS_FILE, RevokedSessions
from eugene_plexus_agent.settings import Settings

from .conftest import TEST_PASSPHRASE, StubRuntimeSupervisor, StubSupervisor, fake_devices
from .test_ui_proxy import declare, record_transport


def _signed_in(client: TestClient) -> str:
    init = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    assert init.status_code == 200, init.text
    return str(init.json()["sessionToken"])


def _sign_out(client: TestClient, token: str) -> None:
    response = client.delete(
        "/v1/auth/sessions/current", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 204, response.text


def _with_upstream(app: FastAPI) -> list[httpx.Request]:
    sent: list[httpx.Request] = []
    app.state.ui_proxy_client = httpx.AsyncClient(transport=record_transport(sent))
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")
    return sent


def _revoked(response: httpx.Response) -> bool:
    return response.status_code == 401 and "revoked" in response.json()["detail"]["title"].lower()


# -- the proxy ---------------------------------------------------------------


def test_a_signed_out_token_is_refused_by_the_proxy_before_anything_is_forwarded(
    app: FastAPI, client: TestClient
) -> None:
    token = _signed_in(client)
    sent = _with_upstream(app)
    bearer = {"Authorization": f"Bearer {token}"}

    assert client.get("/api/proxy/gateway/v1/config", headers=bearer).status_code == 200
    assert len(sent) == 1, "the control case: before signing out, the proxy forwards"

    _sign_out(client, token)
    after = client.get("/api/proxy/gateway/v1/config", headers=bearer)
    assert _revoked(after), f"{after.status_code} {after.text}"
    assert len(sent) == 1, "the signed-out token reached the gateway"


def test_the_same_token_as_an_api_key_header_is_refused_too(
    app: FastAPI, client: TestClient
) -> None:
    """The gateway's Anthropic door reads `x-api-key` before
    `Authorization`, and accepts an operator token there. Screening only
    one header would leave the other as the way round."""
    token = _signed_in(client)
    sent = _with_upstream(app)
    _sign_out(client, token)

    after = client.post("/api/proxy/gateway/v1/messages", headers={"x-api-key": token}, json={})
    assert _revoked(after)
    assert sent == []


def test_a_different_token_is_still_forwarded(app: FastAPI, client: TestClient) -> None:
    """Refusing the one that signed out, not every one: a fresh sign-in
    on the same browser has to work at once."""
    old = _signed_in(client)
    sent = _with_upstream(app)
    _sign_out(client, old)
    fresh = client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE}).json()
    response = client.get(
        "/api/proxy/gateway/v1/config",
        headers={"Authorization": f"Bearer {fresh['sessionToken']}"},
    )
    assert response.status_code == 200
    assert len(sent) == 1


# -- a restart ---------------------------------------------------------------


def _node_app(settings: Settings, signing_key: bytes) -> FastAPI:
    """An agent whose signing key survives a restart, as an enrolled node's
    does -- the case where a forgotten sign-out means a live token."""
    app = create_app(settings=settings)
    app.state.supervisor = StubSupervisor()
    app.state.runtime_supervisor = StubRuntimeSupervisor()
    app.state.device_detector = lambda: fake_devices()
    app.state.library_fit_client = None
    app.state.model_exists = lambda path: True
    app.state.auth_state = AuthState(signing_key=signing_key)
    return app


@pytest.fixture
def install_key() -> bytes:
    return security.generate_signing_key()


@pytest.fixture
def restarted(settings: Settings, install_key: bytes) -> Iterator[tuple[TestClient, str]]:
    """Sign in and out on one agent, then start a second over the same files."""
    with TestClient(_node_app(settings, install_key)) as first:
        token = _signed_in(first)
        _sign_out(first, token)
    app = _node_app(settings, install_key)
    with TestClient(app) as second:
        yield second, token


def test_a_sign_out_survives_a_restart_on_the_agents_own_routes(
    restarted: tuple[TestClient, str],
) -> None:
    client, token = restarted
    response = client.get("/v1/config", headers={"Authorization": f"Bearer {token}"})
    assert _revoked(response), f"{response.status_code}: the restart forgot the sign-out"


def test_a_sign_out_survives_a_restart_through_the_proxy(
    restarted: tuple[TestClient, str],
) -> None:
    client, token = restarted
    sent = _with_upstream(client.app)  # type: ignore[arg-type]
    response = client.get(
        "/api/proxy/gateway/v1/config", headers={"Authorization": f"Bearer {token}"}
    )
    assert _revoked(response)
    assert sent == []


def test_the_file_never_holds_the_token(
    restarted: tuple[TestClient, str], settings: Settings
) -> None:
    """It holds a hash: the file is a list of what must be refused, and a
    list of usable tokens would be a worse thing to leave on disk than the
    problem it solves."""
    _client, token = restarted
    text = (settings.config_file.parent / REVOKED_SESSIONS_FILE).read_text(encoding="utf-8")
    assert token not in text
    assert session_revocations.session_id(token) in text


# -- the file ----------------------------------------------------------------


def test_an_entry_is_pruned_once_its_token_could_no_longer_verify(tmp_path: Path) -> None:
    """Kept for `exp` plus the clock-skew leeway, because that is how long
    a decoder here would still accept the token; after that the token is
    refused as expired and the entry is only litter."""
    path = tmp_path / REVOKED_SESSIONS_FILE
    now = time.time()
    leeway = security.CLOCK_SKEW_LEEWAY_SECONDS
    store = RevokedSessions()
    store.bind(path)
    store.revoke("long-gone", expires_at=int(now - leeway - 60))
    store.revoke("in-the-leeway", expires_at=int(now - leeway + 60))
    store.revoke("live", expires_at=int(now + 3600))

    on_disk = {entry["id"] for entry in json.loads(path.read_text(encoding="utf-8"))["sessions"]}
    assert session_revocations.session_id("long-gone") not in on_disk
    assert session_revocations.session_id("in-the-leeway") in on_disk
    assert session_revocations.session_id("live") in on_disk

    reloaded = RevokedSessions()
    reloaded.bind(path)
    assert reloaded.is_revoked("live")
    assert reloaded.is_revoked("in-the-leeway")
    assert not reloaded.is_revoked("long-gone")


def test_an_expired_entry_already_on_disk_is_pruned_when_the_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / REVOKED_SESSIONS_FILE
    stale = session_revocations.session_id("stale")
    live = session_revocations.session_id("live")
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": [
                    {"id": stale, "until": time.time() - 10},
                    {"id": live, "until": time.time() + 3600},
                ],
            }
        ),
        encoding="utf-8",
    )
    store = RevokedSessions()
    store.bind(path)
    on_disk = {entry["id"] for entry in json.loads(path.read_text(encoding="utf-8"))["sessions"]}
    assert on_disk == {live}


def test_an_unreadable_file_does_not_stop_the_agent(settings: Settings, install_key: bytes) -> None:
    """`degraded-mode-required`: a damaged list of sign-outs costs the
    sign-outs it held, and says so, and nothing else."""
    path = settings.config_file.parent / REVOKED_SESSIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    with TestClient(_node_app(settings, install_key)) as client:
        assert client.get("/healthz").status_code == 200
        token = _signed_in(client)
        _sign_out(client, token)
        assert _revoked(client.get("/v1/config", headers={"Authorization": f"Bearer {token}"}))
    assert json.loads(path.read_text(encoding="utf-8"))["sessions"], "not rewritten"
    assert (path.parent / (REVOKED_SESSIONS_FILE + ".unreadable")).exists(), "no copy kept"
