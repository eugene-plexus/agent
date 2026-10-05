"""Signing in with Eugene (C2): the agent's half.

The root is the provider; this agent forwards `/oidc/*` to it, registers
an app that signs people in as a client, and tells that app where to
sign in. The whole flow runs in `specs/scripts/c2-sign-in-acceptance.py`;
pinned here is what it cannot isolate: which headers cross, and what an
install or an uninstall does when the root refuses.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from eugene_plexus_agent import app_launcher, apps
from eugene_plexus_agent.routes import apps as apps_routes
from eugene_plexus_agent.routes import oidc_forward

from .conftest import FakeRoot, enroll_app
from .test_apps import _created, _fake_python, _manifest, _record

# --------------------------------------------------------------------------- #
# the forward
# --------------------------------------------------------------------------- #


def test_an_unenrolled_machine_has_no_eugene_to_sign_in_with(client: TestClient) -> None:
    answer = client.get("/oidc/.well-known/openid-configuration")
    assert answer.status_code == 503
    assert answer.json()["error"] == "temporarily_unavailable"


def test_the_forward_sets_what_only_it_knows_and_drops_the_callers_copies(
    app: FastAPI, client: TestClient
) -> None:
    enroll_app(app, FakeRoot(), "node-a")
    seen: list[httpx.Request] = []

    def root(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            302,
            headers={
                "location": "http://127.0.0.1:9/cb?code=x",
                "cache-control": "no-store",
                "set-cookie": "root=1",
                "content-security-policy": "frame-ancestors 'none'",
            },
        )

    app.state.oidc_forward_client = httpx.AsyncClient(transport=httpx.MockTransport(root))
    answer = client.post(
        "/oidc/authorize?x=1",
        content=b"request=r&password=p",
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "authorization": "Basic Y2xpZW50OnNlY3JldA==",
            "cookie": "session=the-consoles",
            oidc_forward.FORWARDED_HOST_HEADER: "evil.example",
            oidc_forward.FORWARDED_FOR_HEADER: "10.9.9.9",
            oidc_forward.NODE_TOKEN_HEADER: "forged",
        },
        follow_redirects=False,
    )
    assert answer.status_code == 302
    assert answer.headers["location"] == "http://127.0.0.1:9/cb?code=x"
    assert answer.headers["cache-control"] == "no-store"
    assert "set-cookie" not in answer.headers

    sent = seen[0]
    assert str(sent.url) == "http://control.invalid:8083/oidc/authorize?x=1"
    assert sent.content == b"request=r&password=p"
    assert sent.headers["authorization"] == "Basic Y2xpZW50OnNlY3JldA=="
    assert "cookie" not in sent.headers
    assert sent.headers[oidc_forward.FORWARDED_HOST_HEADER] == "testserver"
    assert sent.headers[oidc_forward.FORWARDED_FOR_HEADER] == "testclient"
    token = sent.headers[oidc_forward.NODE_TOKEN_HEADER]
    assert token != "forged" and token.count(".") == 2


def test_a_root_that_does_not_answer_is_named(app: FastAPI, client: TestClient) -> None:
    enroll_app(app, FakeRoot(), "node-a")

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    app.state.oidc_forward_client = httpx.AsyncClient(transport=httpx.MockTransport(down))
    answer = client.get("/oidc/jwks")
    assert answer.status_code == 503
    assert "control.invalid:8083" in answer.json()["error_description"]


# --------------------------------------------------------------------------- #
# install and uninstall
# --------------------------------------------------------------------------- #


class FakeRegistry:
    enrolled = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None, Any]] = []
        self.refuse: HTTPException | None = None

    async def forward(
        self, method: str, path: str, *, authorization: str | None = None, body: Any = None
    ) -> Any:
        self.calls.append((method, path, authorization, body))
        if self.refuse is not None:
            raise self.refuse
        if method == "POST":
            return {"client": {"clientId": "c-made"}, "clientSecret": "the-secret-shown-once"}
        return None


@pytest.fixture
def installing(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> tuple[apps.AppManager, FakeRegistry]:
    manager: apps.AppManager = app.state.apps
    fake = FakeRegistry()
    monkeypatch.setattr(apps_routes, "registry", lambda request: fake)
    monkeypatch.setattr(apps, "find_uv", lambda configured=None: Path("uv"))

    async def mint(request: Any, body: Any, *, authorization: str | None) -> Any:
        return _created(name=body.name)

    monkeypatch.setattr(apps_routes, "mint_client_key", mint)
    monkeypatch.setattr(
        manager.installer,
        "start",
        lambda manifest, **_kw: apps._Progress(
            app=manifest.id, version=manifest.version
        ).snapshot(),
    )
    return manager, fake


def test_an_app_that_does_not_sign_in_gets_an_empty_secret_and_no_client(
    authed_client: TestClient, installing: tuple[apps.AppManager, FakeRegistry]
) -> None:
    manager, fake = installing
    manager.catalogue = {"tiny": _manifest()}
    assert authed_client.post("/v1/apps/tiny/install").status_code == 202
    assert fake.calls == []
    # systemd will not start a unit whose credential file is missing.
    assert manager.store.oidc_secret_file("tiny").read_text() == ""
    assert manager.store.oidc_client("tiny") is None


def test_an_app_that_signs_in_is_registered_once_with_the_callers_credential(
    authed_client: TestClient, installing: tuple[apps.AppManager, FakeRegistry]
) -> None:
    manager, fake = installing
    manager.catalogue = {"tiny": _manifest(signIn=True, signInCallbackPath="/auth/back")}
    assert authed_client.post("/v1/apps/tiny/install").status_code == 202

    [(method, path, authorization, body)] = fake.calls
    assert (method, path) == ("POST", "/v1/oidc/clients")
    assert authorization == authed_client.headers["Authorization"]
    port = manager.reserve_port("tiny")
    assert body["redirectUris"] == [
        f"http://127.0.0.1:{port}/auth/back",
        f"http://localhost:{port}/auth/back",
    ]
    assert body["owner"].startswith("app:tiny@")
    assert manager.store.oidc_secret_file("tiny").read_text() == "the-secret-shown-once"
    assert manager.store.oidc_client("tiny") == "c-made"

    # A retry after a failed install keeps the client rather than making
    # a second one nobody would ever remove.
    assert authed_client.post("/v1/apps/tiny/install").status_code == 202
    assert len(fake.calls) == 1


def test_the_port_a_callback_names_is_the_port_the_app_gets(tmp_path: Path) -> None:
    from eugene_plexus_agent._generated.models import AppOrigin

    from .test_apps import _manager

    manager = _manager(tmp_path)
    reserved = manager.reserve_port("tiny")
    assert manager.reserve_port("tiny") == reserved
    assert manager.allocate_port() != reserved
    asyncio.run(manager.installed_callback(_manifest(), AppOrigin.catalogue))
    installed = manager.store.get("tiny")
    assert installed is not None and installed.port == reserved


def _moved_app(manager: apps.AppManager) -> Path:
    """An app that signs in, registered at a direct-port address, now on HTTPS."""
    manager.store.put(_record(_manifest(signIn=True)))
    manager.store.put_oidc_client("tiny", "c-old")
    apps_routes.write_private(manager.store.oidc_secret_file("tiny"), "old-secret")
    saved = manager.store.data_dir("tiny") / "chat-data"
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text("keep my chats")
    manager._public_origin = lambda app_id: "https://workbench.home.arpa:8443"
    return saved


HTTPS_CALLBACK = ["https://workbench.home.arpa:8443/oidc/callback"]


def test_a_restart_moves_the_return_address_and_keeps_the_client(
    authed_client: TestClient,
    installing: tuple[apps.AppManager, FakeRegistry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-10-05: the client follows the app's address; it is not replaced,
    so its sign-ins carry on. (Before, a Restart deleted and re-made it.)"""
    manager, fake = installing
    saved = _moved_app(manager)

    async def restart(_record):
        return None

    monkeypatch.setattr(manager, "restart", restart)
    assert authed_client.post("/v1/apps/tiny/restart").status_code == 200
    assert fake.calls == [
        (
            "PUT",
            "/v1/oidc/clients/c-old/redirect-uris",
            authed_client.headers["Authorization"],
            {"redirectUris": HTTPS_CALLBACK},
        )
    ]
    assert manager.store.oidc_client("tiny") == "c-old"
    assert manager.store.oidc_secret_file("tiny").read_text() == "old-secret"
    assert saved.read_text() == "keep my chats"
    assert authed_client.post("/v1/apps/tiny/restart").status_code == 200
    assert len(fake.calls) == 1
    # Returning to direct ports moves it back the same way.
    manager._public_origin = lambda app_id: None
    assert authed_client.post("/v1/apps/tiny/restart").status_code == 200
    assert len(fake.calls) == 2 and fake.calls[-1][0] == "PUT"
    assert all(uri.startswith("http://") for uri in fake.calls[-1][3]["redirectUris"])


def test_a_restart_registers_again_only_a_client_the_root_lost(
    authed_client: TestClient,
    installing: tuple[apps.AppManager, FakeRegistry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, fake = installing
    _moved_app(manager)

    async def restart(_record):
        return None

    monkeypatch.setattr(manager, "restart", restart)
    lost = HTTPException(404, detail="no such app")
    real = fake.forward

    async def forward(method: str, path: str, **kw: Any) -> Any:
        if method == "PUT":
            fake.calls.append((method, path, kw.get("authorization"), kw.get("body")))
            raise lost
        return await real(method, path, **kw)

    monkeypatch.setattr(fake, "forward", forward)
    assert authed_client.post("/v1/apps/tiny/restart").status_code == 200
    assert [call[:2] for call in fake.calls] == [
        ("PUT", "/v1/oidc/clients/c-old/redirect-uris"),
        ("DELETE", "/v1/oidc/clients/c-old"),
        ("POST", "/v1/oidc/clients"),
    ]
    assert fake.calls[-1][3]["redirectUris"] == HTTPS_CALLBACK
    assert manager.store.oidc_client("tiny") == "c-made"
    # Any other refusal is the operator's to see, and changes nothing.
    fake.calls.clear()
    lost = HTTPException(403, detail="not this machine's")
    manager._public_origin = lambda app_id: "https://elsewhere.home.arpa:8443"
    assert authed_client.post("/v1/apps/tiny/restart").status_code == 403
    assert [call[0] for call in fake.calls] == ["PUT"]
    assert manager.store.oidc_client("tiny") == "c-made"


async def test_at_boot_the_agent_moves_it_with_its_own_token(
    authed_client: TestClient,
    app: FastAPI,
    installing: tuple[apps.AppManager, FakeRegistry],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Troy, 2026-10-05: Eugene updates Workbench's return address itself at
    startup -- no operator, no Restart, the same client and secret."""
    from eugene_plexus_agent import sign_in_refresh

    manager, fake = installing
    saved = _moved_app(manager)
    app.state.client_key_registry = fake
    monkeypatch.setattr(sign_in_refresh, "_FIRST_PAUSE", 0.01)
    attempts = {"n": 0}
    real = fake.forward

    async def forward(method: str, path: str, **kw: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            fake.calls.append((method, path, kw.get("authorization"), kw.get("body")))
            raise HTTPException(503, detail="the control root is starting")
        return await real(method, path, **kw)

    monkeypatch.setattr(fake, "forward", forward)
    assert "moving this app's sign-in address" in (manager.view(manager.store.get("tiny")).detail)
    with caplog.at_level("INFO"):
        await asyncio.wait_for(sign_in_refresh.run(app), 10)
    assert [call[:3] for call in fake.calls] == [
        ("PUT", "/v1/oidc/clients/c-old/redirect-uris", None),
        ("PUT", "/v1/oidc/clients/c-old/redirect-uris", None),
    ]
    assert fake.calls[-1][3] == {"redirectUris": HTTPS_CALLBACK}
    assert manager.store.oidc_client("tiny") == "c-old"
    assert manager.store.oidc_secret_file("tiny").read_text() == "old-secret"
    assert manager.sign_in_registration_current(manager.store.get("tiny").manifest)
    assert not (manager.view(manager.store.get("tiny")).detail or "")
    assert saved.read_text() == "keep my chats"
    assert "moved tiny's sign-in address" in caplog.text
    # Current now: the next boot asks nothing.
    await asyncio.wait_for(sign_in_refresh.run(app), 10)
    assert len(fake.calls) == 2


async def test_at_boot_a_client_it_may_not_move_is_left_for_the_operator(
    authed_client: TestClient,
    app: FastAPI,
    installing: tuple[apps.AppManager, FakeRegistry],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from eugene_plexus_agent import sign_in_refresh

    manager, fake = installing
    _moved_app(manager)
    app.state.client_key_registry = fake
    monkeypatch.setattr(sign_in_refresh, "_FIRST_PAUSE", 0.01)
    fake.refuse = HTTPException(403, detail="Not this machine's to change")
    with caplog.at_level("WARNING"):
        await asyncio.wait_for(sign_in_refresh.run(app), 10)
    assert [call[0] for call in fake.calls] == ["PUT"]
    assert "Restart it in Apps" in caplog.text
    assert not manager.sign_in_registration_current(manager.store.get("tiny").manifest)


async def test_at_boot_nothing_is_asked_of_an_unenrolled_machine(
    authed_client: TestClient, app: FastAPI, installing: tuple[apps.AppManager, FakeRegistry]
) -> None:
    from eugene_plexus_agent import sign_in_refresh

    manager, fake = installing
    _moved_app(manager)
    fake.enrolled = False
    app.state.client_key_registry = fake
    await asyncio.wait_for(sign_in_refresh.run(app), 10)
    assert fake.calls == []


def test_an_uninstall_whose_client_removal_fails_removes_nothing_else(
    authed_client: TestClient,
    installing: tuple[apps.AppManager, FakeRegistry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, fake = installing

    async def revoke(request: Any, key_id: str, *, authorization: str | None) -> None:
        return None

    monkeypatch.setattr(apps_routes, "revoke_client_key_at_authority", revoke)
    manager.store.put(_record(_manifest(signIn=True)))
    manager.store.put_oidc_client("tiny", "c-made")
    apps_routes.write_private(manager.store.oidc_secret_file("tiny"), "the-secret")

    fake.refuse = HTTPException(503, detail="root down")
    assert authed_client.delete("/v1/apps/tiny").status_code == 503
    assert manager.store.get("tiny") is not None
    assert manager.store.oidc_client("tiny") == "c-made"

    # A client the root has never heard of is as removed as it gets.
    fake.refuse = HTTPException(404, detail="no such app")
    assert authed_client.delete("/v1/apps/tiny").status_code == 204
    assert fake.calls[-1][:2] == ("DELETE", "/v1/oidc/clients/c-made")
    assert manager.store.get("tiny") is None and manager.store.oidc_client("tiny") is None


# --------------------------------------------------------------------------- #
# what the app is told
# --------------------------------------------------------------------------- #


def _planned(tmp_path: Path, *, sign_in: bool, client_id: str | None) -> dict[str, str]:
    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    record = _record(_manifest(signIn=sign_in))
    _fake_python(store, record)
    if client_id:
        store.put_oidc_client(record.id, client_id)
    planner = apps._AppPlanner(
        record,
        store=store,
        gateway_url=lambda: None,
        bind_host=lambda: None,
        oidc_issuer=lambda: "http://10.0.0.5:8079/oidc",
    )
    return planner.plan().env


def test_an_app_that_signs_in_is_told_where_and_as_which_client(tmp_path: Path) -> None:
    env = _planned(tmp_path, sign_in=True, client_id="c-made")
    assert env["EUGENE_PLEXUS_APP_OIDC_ISSUER"] == "http://10.0.0.5:8079/oidc"
    assert env["EUGENE_PLEXUS_APP_OIDC_CLIENT_ID"] == "c-made"
    assert env["EUGENE_PLEXUS_APP_OIDC_SECRET_FILE"].endswith("oidc_secret")
    # The secret itself never rides in an environment.
    assert "the-secret" not in json.dumps(env)


@pytest.mark.parametrize("sign_in,client_id", [(False, "c-made"), (True, None)])
def test_no_client_means_no_sign_in_variables(
    tmp_path: Path, sign_in: bool, client_id: str | None
) -> None:
    env = _planned(tmp_path, sign_in=sign_in, client_id=client_id)
    assert not any(k.startswith("EUGENE_PLEXUS_APP_OIDC_") for k in env)


def test_the_launcher_points_at_the_secret_systemd_hands_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    creds = tmp_path / "creds"
    creds.mkdir()
    spec_file = tmp_path / "launch.json"

    def env_for(variables: dict[str, str]) -> dict[str, str]:
        spec_file.write_text(json.dumps({"app": "tiny", "argv": ["x"], "env": variables}))
        return app_launcher.child_environment(app_launcher.load_spec(str(spec_file)))

    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(creds))
    signing_in = env_for({"EUGENE_PLEXUS_APP_OIDC_CLIENT_ID": "c-made"})
    assert signing_in["EUGENE_PLEXUS_APP_OIDC_SECRET_FILE"] == str(creds / "oidc_secret")
    assert "EUGENE_PLEXUS_APP_OIDC_SECRET_FILE" not in env_for({})
