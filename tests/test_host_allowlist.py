"""DNS rebinding: a request whose Host is a name this agent was not given.

A web page on `evil.example.com` can answer its own DNS with this
machine's address a moment after it loads, and from then on the browser
treats the agent as that page's own origin: same-origin reads of every
API response, and -- the sharp end -- `POST /v1/auth/initialize`, which
is first-come until first run is done. Nothing checked the Host header.

The allowed matrix is deliberately generous (every way a person really
opens this on a home network or a tailnet), and the configured field is
the expert's override. These go through `TestClient` with a `Host`
header, which reaches the ASGI scope exactly as a browser's does.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from .conftest import TEST_PASSPHRASE
from .test_forwarded_peer import _ui_app
from .test_ui_proxy import declare, record_transport

ALLOWED = [
    "127.0.0.1",
    "127.0.0.1:8079",
    "192.168.16.252:8279",
    "[::1]",
    "[::1]:8079",
    "[fe80::1]:8079",
    "localhost",
    "localhost:8079",
    "LOCALHOST",
    "localhost.",
    "app.localhost",
    "tower",
    "Tower:8079",
    "nas.local",
    "box.lan",
    "printer.home.arpa",
    "svc.internal",
    "box.tail1234.ts.net",
    "box.tail1234.ts.net:8079",
]

BLOCKED = [
    "evil.example.com",
    "evil.example.com:8079",
    # A rebinding service's names embed an address and are still names.
    "127.0.0.1.nip.io",
    "localhost.evil.com",
    "local.evil.com",
    "ts.net.evil.com",
]


def _host(client: TestClient, host: str, path: str = "/healthz", method: str = "GET", **kw: Any):
    return client.request(method, path, headers={"host": host, **kw.pop("headers", {})}, **kw)


@pytest.mark.parametrize("host", ALLOWED)
def test_the_ways_a_person_really_opens_it_are_allowed(client: TestClient, host: str) -> None:
    assert _host(client, host).status_code == 200


@pytest.mark.parametrize("host", BLOCKED)
def test_a_name_the_agent_was_not_given_is_refused(client: TestClient, host: str) -> None:
    response = _host(client, host)
    assert response.status_code == 403, f"{host!r} was answered"


def test_a_rebinding_page_cannot_claim_a_fresh_install(client: TestClient) -> None:
    """The sharp end: first run is first-come, and the attacker's page
    would otherwise come first."""
    refused = _host(
        client,
        "evil.example.com",
        "/v1/auth/initialize",
        method="POST",
        json={"passphrase": "attacker's passphrase"},
    )
    assert refused.status_code == 403
    status = _host(client, "127.0.0.1:8079", "/v1/auth/status").json()
    assert status["initialized"] is False, "the refused request set a passphrase anyway"


def test_the_api_refusal_is_a_problem_naming_the_host_and_the_fix(client: TestClient) -> None:
    response = _host(client, "evil.example.com:8079", "/v1/config")
    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/json")
    problem = response.json()["detail"]
    assert problem["status"] == 403
    assert problem["component"] == "agent"
    assert "evil.example.com" in problem["detail"]
    assert "IP address" in problem["detail"]
    assert "Allowed host names" in problem["detail"]


def test_the_page_refusal_is_a_page_naming_the_host_and_the_fix(tmp_path: Any) -> None:
    with TestClient(_ui_app(tmp_path, with_index=True)) as client:
        allowed = _host(client, "127.0.0.1:8079", "/")
        refused = _host(client, "evil.example.com", "/")
    assert allowed.status_code == 200
    assert refused.status_code == 403
    assert refused.headers["content-type"].startswith("text/html")
    assert "evil.example.com" in refused.text
    assert "localhost" in refused.text
    assert "Allowed host names" in refused.text


def test_the_proxy_forwards_nothing_for_a_refused_name(app: FastAPI, client: TestClient) -> None:
    sent: list[httpx.Request] = []
    app.state.ui_proxy_client = httpx.AsyncClient(transport=record_transport(sent))
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")
    assert _host(client, "evil.example.com", "/api/proxy/gateway/v1/models").status_code == 403
    assert sent == []
    assert _host(client, "127.0.0.1", "/api/proxy/gateway/v1/models").status_code == 200
    assert len(sent) == 1


# -- the expert's override, read per request ---------------------------------


def test_a_configured_name_is_allowed_from_the_next_request(authed_client: TestClient) -> None:
    assert _host(authed_client, "llm.example.com").status_code == 403
    patched = authed_client.patch("/v1/config", json={"allowedHosts": "llm.example.com"})
    assert patched.status_code == 200
    assert patched.json()["applied"] == ["allowedHosts"], patched.text
    assert _host(authed_client, "llm.example.com").status_code == 200
    assert _host(authed_client, "LLM.example.com:443").status_code == 200
    assert _host(authed_client, "other.example.com").status_code == 403


def test_several_names_and_forgiving_spellings(authed_client: TestClient) -> None:
    """Commas or spaces, and a pasted URL means its host: the field is a
    comma-separated string because the config trio has no list-of-names
    type, so it forgives what a person is likely to type."""
    value = "https://llm.example.com/ , chat.example.org:8443  third.example.net"
    patched = authed_client.patch("/v1/config", json={"allowedHosts": value})
    assert patched.json()["applied"] == ["allowedHosts"], patched.text
    for host in ("llm.example.com", "chat.example.org", "third.example.net"):
        assert _host(authed_client, host).status_code == 200, host


def test_a_star_allows_any_name(authed_client: TestClient) -> None:
    assert _host(authed_client, "evil.example.com").status_code == 403
    patched = authed_client.patch("/v1/config", json={"allowedHosts": "*"})
    assert patched.json()["applied"] == ["allowedHosts"], patched.text
    assert _host(authed_client, "evil.example.com").status_code == 200


def test_the_advertise_address_host_is_allowed(authed_client: TestClient) -> None:
    """A node another node reaches by name: that name is the one it was
    told to advertise, so it is by definition one of its own."""
    assert _host(authed_client, "gpu.example.org").status_code == 403
    patched = authed_client.patch(
        "/v1/config", json={"advertiseUrl": "http://gpu.example.org:8079"}
    )
    assert patched.status_code == 200
    assert _host(authed_client, "gpu.example.org:8079").status_code == 200


def test_the_field_says_why_it_exists(authed_client: TestClient) -> None:
    fields = {f["key"]: f for f in authed_client.get("/v1/config/schema").json()["fields"]}
    description = fields["allowedHosts"]["description"]
    assert "DNS rebinding" in description
    assert "reverse proxy" in description


# -- the pieces --------------------------------------------------------------


def test_this_machines_own_names_are_allowed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    from eugene_plexus_agent import host_allowlist

    # The lifespan started the real lookup; let it land first, or it can
    # overwrite the fake one below.
    host_allowlist.start_learning_fqdn().join(timeout=30)
    monkeypatch.setattr(host_allowlist, "_fqdn", None)
    monkeypatch.setattr(socket, "gethostname", lambda: "Box.Corp.Example")
    monkeypatch.setattr(socket, "getfqdn", lambda *a: "box.ad.corp.example")
    host_allowlist.learn_fqdn()
    assert _host(client, "box.corp.example").status_code == 200
    assert _host(client, "BOX.AD.CORP.EXAMPLE:8079").status_code == 200
    assert _host(client, "other.corp.example").status_code == 403


async def test_a_request_with_no_host_header_is_allowed() -> None:
    """HTTP/1.0 tools on this machine send none, and a browser always does,
    so refusing it would break the one and protect against nothing."""
    from eugene_plexus_agent.host_allowlist import HostAllowlistMiddleware

    reached: list[bool] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        reached.append(True)

    middleware = HostAllowlistMiddleware(inner, policy=lambda: None)
    await middleware({"type": "http", "path": "/healthz", "headers": []}, None, None)
    assert reached == [True]


def test_the_refusal_page_escapes_the_name_it_repeats() -> None:
    """The Host header is the attacker's to write, and the page says it back."""
    from eugene_plexus_agent.host_allowlist import refusal_page

    page = refusal_page("<script>alert(1)</script>.example.com")
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


@pytest.mark.parametrize(
    ("header", "host"),
    [
        ("Example.COM:8079", "example.com"),
        ("[::1]:8079", "::1"),
        ("[::1]", "::1"),
        ("::1", "::1"),
        ("localhost.", "localhost"),
        ("tower", "tower"),
    ],
)
def test_the_host_is_read_without_its_port_or_brackets(header: str, host: str) -> None:
    from eugene_plexus_agent.host_allowlist import host_of

    assert host_of(header) == host


def test_login_and_setup_still_work_through_an_allowed_name(client: TestClient) -> None:
    init = _host(
        client,
        "tower:8079",
        "/v1/auth/initialize",
        method="POST",
        json={"passphrase": TEST_PASSPHRASE},
    )
    assert init.status_code == 200
