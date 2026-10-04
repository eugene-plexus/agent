"""Security boundaries for the opt-in container HTTPS entry point."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request
from pydantic import ValidationError

from eugene_plexus_agent.entrypoint import (
    CLIENT_HEADER,
    TOKEN_HEADER,
    EntryConfig,
    EntryPoint,
    ProxyMetadata,
    caddy_config,
)

from .test_apps import _fake_python, _manager, _manifest, _record


def config(**changes):
    return EntryConfig.model_validate(
        {
            "internal_ca": True,
            "console": {"origin": "https://eugene.home.arpa:8443", "networks": ["192.168.1.0/24"]},
            "workbench": {"origin": "https://workbench.home.arpa:8443", "networks": ["0.0.0.0/0"]},
            **changes,
        }
    )


@pytest.mark.parametrize(
    "origin",
    [
        "http://workbench.home.arpa:8443",
        "https://*.home.arpa:8443",
        "https://127.0.0.1:8443",
        "https://user@workbench.home.arpa:8443",
        "https://workbench.home.arpa:8443/path",
        "https://workbench.home.arpa:8443?other=x",
        "https://workbench.home.arpa:8443#x",
        "https://workbench.home.arpa:9000",
        "https://eugene.home.arpa:8443",
        "https://workbench.home.arpa:8443\n",
        "https://workbench.home.arpa:0",
    ],
)
def test_origins_are_explicit_distinct_https_hostnames(origin):
    with pytest.raises(ValidationError):
        config(workbench={"origin": origin, "networks": ["0.0.0.0/0"]})


def test_invalid_or_missing_policy_cannot_enable_public_admin():
    for changes in (
        {"console": {"origin": "https://eugene.home.arpa:8443", "networks": ["0.0.0.0/0"]}},
        {"console": {"origin": "https://eugene.home.arpa:8443", "networks": []}},
        {"internal_ca": False},
        {"upstream": "http://attacker.example"},
        {"console": {"origin": "https://eugene.home.arpa:8443", "networks": ["192.168.1.7/24"]}},
    ):
        with pytest.raises(ValidationError):
            config(**changes)


@pytest.mark.parametrize(
    "peer,secret,expected",
    [
        ("127.0.0.1", "real-secret", 200),
        ("::1", "real-secret", 200),
        ("127.0.0.1", "fake", 403),
        ("198.51.100.2", "real-secret", 403),
    ],
)
async def test_proxy_metadata_needs_both_loopback_and_this_process_secret(peer, secret, expected):
    app = FastAPI()

    @app.get("/")
    def inspect(request: Request):
        return {
            "peer": request.client.host,
            "scheme": request.url.scheme,
            "headers": dict(request.headers),
        }

    app.add_middleware(ProxyMetadata, config=config(), token="real-secret")
    transport = httpx.ASGITransport(app=app, client=(peer, 123))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        reply = await client.get(
            "/",
            headers={
                TOKEN_HEADER: secret,
                CLIENT_HEADER: "203.0.113.4",
                "Forwarded": "for=evil",
                "X-Forwarded-Proto": "http",
                "X-Eugene-Plexus-Peer": "198.51.100.1",
            },
        )
        assert reply.status_code == expected
        if expected == 200:
            body = reply.json()
            assert body["peer"] == "203.0.113.4" and body["scheme"] == "https"
            assert not any(
                k.startswith(("x-eugene-", "x-forwarded-", "forwarded")) for k in body["headers"]
            )
        ordinary = await client.get("/", headers={"X-Forwarded-For": "203.0.113.4"})
        assert ordinary.json()["peer"] == peer and ordinary.json()["scheme"] == "http"
        duplicate = await client.get(
            "/",
            headers=[
                (TOKEN_HEADER, "real-secret"),
                (TOKEN_HEADER, "fake"),
                (CLIENT_HEADER, "1.2.3.4"),
            ],
        )
        assert duplicate.status_code == 403


def test_upstreams_are_loopback_and_missing_workbench_never_falls_back(tmp_path):
    document = caddy_config(config(), tmp_path, "secret", {"agent": 8079})
    routes = document["apps"]["http"]["servers"]["entry"]["routes"]
    wb = next(r for r in routes if r.get("match", [{}])[0].get("host") == ["workbench.home.arpa"])
    assert wb["handle"][0]["status_code"] == 503
    for route in routes:
        for handler in route.get("handle", []):
            for upstream in handler.get("upstreams", []):
                assert upstream["dial"] == "127.0.0.1:8079"
    assert document["admin"]["listen"].startswith("unix/")


def test_remote_components_are_never_ingress_targets(tmp_path):
    from eugene_plexus_agent._generated.models import ComponentEntry
    from eugene_plexus_agent.state import AgentState

    state = AgentState(tmp_path / "agent.yaml")
    state.add_topology_entry(
        ComponentEntry(name="elsewhere", kind="gateway", url="http://remote:9000")
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            agent_state=state,
            apps=None,
            settings=SimpleNamespace(bind_port=8079),
        )
    )
    assert EntryPoint(app, config(), tmp_path, "secret").ports()["gateway"] is None


def test_public_app_port_cannot_be_reused_during_a_proxy_reload(tmp_path: Path):
    manager = _manager(tmp_path)
    manager._public_origin = lambda app_id: (
        "https://workbench.home.arpa:8443" if app_id == "workbench" else None
    )
    record = _record(_manifest(id="workbench", signIn=True, ui=True))
    manager.store.put(record)
    original = manager.reserve_port("workbench")
    assert original == record.port
    manager.store.remove("workbench")
    assert manager.allocate_port() != original
    assert manager.reserve_port("workbench") == original
    assert manager.sign_in_redirects("workbench", "/oidc/callback") == [
        "https://workbench.home.arpa:8443/oidc/callback",
    ]


def test_only_workbench_receives_its_public_origin_and_local_oidc_transport(tmp_path: Path):
    manager = _manager(tmp_path)
    manager._public_origin = lambda app_id: (
        "https://workbench.home.arpa:8443" if app_id == "workbench" else None
    )
    manager._oidc_backchannel = "http://127.0.0.1:8079/oidc"
    record = _record(_manifest(id="workbench", signIn=True, ui=True))
    _fake_python(manager.store, record)
    asyncio.run(manager.start(record))
    env = manager.supervisor.planners[record.id].base_plan().env
    assert env["EUGENE_PLEXUS_APP_PUBLIC_ORIGIN"] == "https://workbench.home.arpa:8443"
    assert env["EUGENE_PLEXUS_APP_OIDC_BACKCHANNEL"] == "http://127.0.0.1:8079/oidc"
    assert manager.ui_url(record) == "https://workbench.home.arpa:8443/"
    manager._private_apps = True
    other = _record(_manifest(id="other-app", ui=True))
    assert manager.ui_url(other) is None
    assert "no published address" in manager.view(other).detail


@pytest.mark.parametrize("reload_fails", [False, True])
async def test_replacing_workbench_closes_its_route_before_stopping_the_app(tmp_path, reload_fails):
    manager = _manager(tmp_path)
    record = _record(_manifest(id="workbench", ui=True))
    _fake_python(manager.store, record)
    manager.store.put(record)
    await manager.start(record)
    app = SimpleNamespace(
        state=SimpleNamespace(
            apps=manager,
            settings=SimpleNamespace(bind_port=8079),
            agent_state=SimpleNamespace(list_components=lambda: []),
        )
    )
    entry = EntryPoint(app, config(), tmp_path, "secret")
    process = SimpleNamespace(
        returncode=None, terminate=lambda: events.append("proxy stopped"), wait=AsyncMock()
    )
    entry.process = process
    events = []

    def reload(request):
        assert manager.supervisor.is_running("workbench")
        assert "127.0.0.1:8190" not in request.content.decode()
        events.append("route removed")
        return httpx.Response(500 if reload_fails else 200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(reload)) as admin:
        entry._admin = admin
        await manager.stop("workbench")
        assert not manager.supervisor.is_running("workbench")
        assert entry.ports()["workbench"] is None
        assert events == (["route removed", "proxy stopped"] if reload_fails else ["route removed"])
        await manager.start(record)
        assert entry.ports()["workbench"] == record.port


async def test_old_or_misconfigured_workbench_never_inherits_public_route(tmp_path):
    manager = _manager(tmp_path)
    record = _record(_manifest(id="workbench", ui=True))
    _fake_python(manager.store, record)
    manager.store.put(record)
    await manager.start(record)
    app = SimpleNamespace(
        state=SimpleNamespace(
            apps=manager,
            settings=SimpleNamespace(bind_port=8079),
            agent_state=SimpleNamespace(list_components=lambda: []),
        )
    )
    entry = EntryPoint(app, config(), tmp_path, "secret")
    entry.process = SimpleNamespace(returncode=None)
    published = []

    def reload(request):
        published.append("127.0.0.1:8190" in request.content.decode())
        return httpx.Response(200)

    reports = iter(
        [
            {"status": "ok"},
            {"originIsolation": 1, "publicOrigin": "https://wrong.example"},
            {"originIsolation": 1, "publicOrigin": config().workbench.origin},
            {"status": "ok"},
        ]
    )
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(reload)) as admin,
        httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=json.dumps(next(reports)))
            )
        ) as backend,
    ):
        for _ in range(4):
            await entry._sync(admin, backend)
    assert published == [False, True, False]


def automatic(**overrides):
    base = {
        "acme": {"email": "operator@example.org", "accept_terms": True},
        "console": {"origin": "https://eugene.example.org", "networks": ["192.168.1.0/24"]},
        "workbench": {"origin": "https://workbench.example.org", "networks": ["0.0.0.0/0"]},
    }
    return EntryConfig.model_validate({**base, **overrides})


@pytest.mark.parametrize(
    "changes",
    [
        {"internal_ca": True},
        {"acme": {"email": "operator@example.org", "accept_terms": False}},
        {"acme": {"email": "no-address", "accept_terms": True}},
        {"console": {"origin": "https://eugene.home.arpa", "networks": ["192.168.1.0/24"]}},
        {
            "console": {
                "origin": "https://eugene.example.org:8443",
                "networks": ["192.168.1.0/24"],
            },
            "workbench": {
                "origin": "https://workbench.example.org:8443",
                "networks": ["0.0.0.0/0"],
            },
        },
        {"proxy": {"addresses": ["172.30.0.2"], "transport": "https"}},
    ],
)
def test_automatic_certificates_need_explicit_terms_public_names_and_port_443(changes):
    with pytest.raises(ValidationError):
        automatic(**changes)


@pytest.mark.parametrize(
    "addresses", [["0.0.0.0/0"], ["172.30.0.0/24"], ["::/0"], ["0.0.0.0"], ["224.0.0.1"], []]
)
def test_proxy_trust_cannot_cover_a_network(addresses):
    with pytest.raises(ValidationError):
        config(internal_ca=False, proxy={"addresses": addresses})


def test_certificate_modes_keep_destinations_private_and_acme_has_no_http_listener(tmp_path):
    document = caddy_config(automatic(), tmp_path, "secret", {"agent": 8079})
    issuer = document["apps"]["tls"]["automation"]["policies"][0]["issuers"][0]
    assert issuer["ca"] == "https://acme-v02.api.letsencrypt.org/directory"
    assert issuer["challenges"] == {
        "http": {"disabled": True},
        "tls-alpn": {"alternate_port": 8443},
    }
    proxy = config(
        internal_ca=False,
        proxy={"addresses": ["172.30.0.2"]},
        trusted_ca="/data/organisation-ca.pem",
    )
    document = caddy_config(proxy, tmp_path, "secret", {"agent": 8079})
    server = document["apps"]["http"]["servers"]["entry"]
    assert server["automatic_https"] == {"disable": True}
    assert "tls" not in document["apps"] and "tls_connection_policies" not in server
    assert server["trusted_proxies_strict"] == 1
    assert server["trusted_proxies"]["ranges"] == ["172.30.0.2/32"]


def test_setup_preview_requires_an_operator(client):
    reply = client.post("/v1/entrypoint/preview", json={"configuration": {}})
    assert reply.status_code in (401, 403)


def test_setup_preview_is_non_mutating_and_explains_unverified_reachability(
    authed_client, app, monkeypatch
):
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("preview spawned a process")
    )
    reply = authed_client.post(
        "/v1/entrypoint/preview",
        json={"configuration": automatic().model_dump(mode="json", exclude_none=True)},
    )
    assert reply.status_code == 200, reply.text
    assert app.state.entrypoint_config is None
    assert "not been tested" in reply.json()["instructions"][0]
    assert any("port 80 is not required" in step for step in reply.json()["instructions"])
    bad = authed_client.post(
        "/v1/entrypoint/preview", json={"configuration": {"proxy": {"addresses": ["0.0.0.0/0"]}}}
    )
    assert bad.status_code == 400 and "individual proxy IPs" in bad.text


def test_setup_preview_preserves_linux_certificate_paths_on_any_os(authed_client):
    body = config().model_dump(mode="json", exclude_none=True)
    body.update(
        internal_ca=False, certificate="/data/tls/chain.pem", private_key="/data/tls/key.pem"
    )
    reply = authed_client.post("/v1/entrypoint/preview", json={"configuration": body})
    assert reply.status_code == 200, reply.text
    assert reply.json()["configuration"]["certificate"] == "/data/tls/chain.pem"
