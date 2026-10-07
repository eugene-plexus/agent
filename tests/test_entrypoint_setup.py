"""One HTTPS port applied from Settings, kept only when it works (2026-10-05).

After moving his NAS onto one HTTPS port by hand -- a downloaded file, a
container variable, port mappings, a Workbench restart and a hand-edited
node.yaml -- Troy: *no weekend LLM enthusiast is going to do all of this.*
These pin what replaced it: Settings writes the file and restarts the agent;
a configuration nobody signs in through goes back by itself; the control
root keeps its direct port for enrolled machines; and every refusal says
which check failed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from eugene_plexus_agent import entrypoint, entrypoint_setup
from eugene_plexus_agent.app import shared_child_env
from eugene_plexus_agent.entrypoint import (
    CLIENT_HEADER,
    TOKEN_HEADER,
    EntryConfig,
    EntryPointConfigError,
    ProxyMetadata,
    caddy_config,
    prepare,
)
from eugene_plexus_agent.node_identity import NodeIdentityStore
from eugene_plexus_agent.routes import entrypoint as entrypoint_routes
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.state import AgentState


def proxy_config(**changes: Any) -> EntryConfig:
    return EntryConfig.model_validate(
        {
            "listen_port": 8088,
            "proxy": {"addresses": ["172.18.0.4"], "transport": "http"},
            "console": {"origin": "https://eugene.example.org", "networks": ["192.168.16.0/24"]},
            "workbench": {"origin": "https://workbench.example.org", "networks": ["0.0.0.0/0"]},
            **changes,
        }
    )


def as_text(config: EntryConfig) -> str:
    return config.model_dump_json(exclude_none=True)


def _settings(tmp_path: Path, **extra: Any) -> Settings:
    return Settings(config_file=tmp_path / "agent.yaml", default_topology=False, **extra)


@pytest.fixture
def can_run(monkeypatch: pytest.MonkeyPatch) -> list[EntryConfig]:
    """This machine can run the entry point, and the proxy accepts what it is shown."""
    checked: list[EntryConfig] = []
    monkeypatch.setattr(entrypoint_setup, "unavailable_reason", lambda settings: None)
    monkeypatch.setattr(
        entrypoint,
        "validate_with_proxy",
        lambda settings, config, directory: checked.append(config),
    )
    return checked


# --------------------------------------------------------------------------- #
# which file, and going back
# --------------------------------------------------------------------------- #


def test_the_file_beside_agent_yaml_turns_it_on_with_no_variable(tmp_path, can_run):
    settings = _settings(tmp_path)
    entrypoint_setup.default_path(settings).write_text(as_text(proxy_config()), encoding="utf-8")
    assert prepare(settings) == proxy_config()
    assert settings.entrypoint_config == tmp_path / "entrypoint.json"
    assert can_run == [proxy_config()]
    assert settings._entrypoint_console_origin == "https://eugene.example.org"
    assert settings._entrypoint_nodes is False


def test_the_file_is_not_read_where_the_proxy_cannot_run(tmp_path, monkeypatch):
    monkeypatch.setattr(entrypoint_setup, "unavailable_reason", lambda settings: "not here")
    settings = _settings(tmp_path)
    entrypoint_setup.default_path(settings).write_text(as_text(proxy_config()), encoding="utf-8")
    assert prepare(settings) is None and settings.entrypoint_config is None
    assert settings._entrypoint_fallback is None


def test_a_file_just_applied_that_will_not_start_goes_back(tmp_path, monkeypatch, can_run):
    settings = _settings(tmp_path)
    path = entrypoint_setup.default_path(settings)
    entrypoint_setup.save_applied(settings, as_text(proxy_config()))

    def refuse(settings: Any, config: Any, directory: Any) -> None:
        raise ValueError("the bundled proxy refused it: bad certificate")

    monkeypatch.setattr(entrypoint, "validate_with_proxy", refuse)
    assert prepare(settings) is None
    assert not path.exists()
    assert path.with_name("entrypoint.json.reverted").read_text() == as_text(proxy_config())
    assert entrypoint_setup.pending(settings) is None
    reason = entrypoint_setup.reverted(settings)
    assert reason and "did not start" in reason and "bad certificate" in reason


def test_a_file_that_will_not_start_and_was_not_just_applied_still_stops(
    tmp_path, monkeypatch, can_run
):
    settings = _settings(tmp_path)
    entrypoint_setup.default_path(settings).write_text(as_text(proxy_config()), encoding="utf-8")

    def refuse(settings: Any, config: Any, directory: Any) -> None:
        raise ValueError("the bundled proxy refused it")

    monkeypatch.setattr(entrypoint, "validate_with_proxy", refuse)
    with pytest.raises(EntryPointConfigError, match="the bundled proxy refused it"):
        prepare(settings)
    assert entrypoint_setup.default_path(settings).is_file()


def test_going_back_puts_the_previous_file_back(tmp_path):
    settings = _settings(tmp_path)
    path = entrypoint_setup.default_path(settings)
    path.write_text("previous", encoding="utf-8")
    entrypoint_setup.save_applied(settings, "first")
    # Applying again before confirming keeps the original to go back to.
    entrypoint_setup.save_applied(settings, "second")
    entrypoint_setup.revert(settings, "nobody signed in")
    assert path.read_text() == "previous"
    assert path.with_name("entrypoint.json.reverted").read_text() == "second"
    assert entrypoint_setup.reverted(settings) == "nobody signed in"
    # The next apply clears the old reason.
    entrypoint_setup.save_applied(settings, "third")
    assert entrypoint_setup.reverted(settings) is None


def test_turning_off_keeps_the_file_and_says_so_under_the_variable(tmp_path, can_run):
    named = tmp_path / "elsewhere.json"
    named.write_text(as_text(proxy_config()), encoding="utf-8")
    settings = _settings(tmp_path, entrypoint_config=named)
    entrypoint_setup.turn_off(settings)
    assert not named.exists() and (tmp_path / "elsewhere.json.disabled").is_file()
    assert prepare(settings) is None
    assert "turned off from Settings" in (settings._entrypoint_fallback or "")


# --------------------------------------------------------------------------- #
# on approval
# --------------------------------------------------------------------------- #


def _approving(tmp_path: Path, deadline: datetime) -> tuple[Settings, Any]:
    settings = _settings(tmp_path, entrypoint_confirm_seconds=900)
    path = entrypoint_setup.default_path(settings)
    path.write_text("previous", encoding="utf-8")
    entrypoint_setup.save_applied(settings, as_text(proxy_config()))
    app = SimpleNamespace(
        state=SimpleNamespace(
            settings=settings,
            entrypoint_config=proxy_config(),
            entrypoint_confirm_by=deadline,
            uvicorn_server=SimpleNamespace(should_exit=False),
        )
    )
    return settings, app


async def test_nobody_signing_in_in_time_goes_back_and_restarts(tmp_path):
    settings, app = _approving(tmp_path, datetime.now(UTC) - timedelta(seconds=1))
    await entrypoint_setup.watch_approval(app)
    await asyncio.sleep(0.05)
    assert entrypoint_setup.default_path(settings).read_text() == "previous"
    assert "https://eugene.example.org within 15 minutes" in (
        entrypoint_setup.reverted(settings) or ""
    )
    assert app.state.restart_requested and app.state.uvicorn_server.should_exit


async def test_a_confirmed_configuration_stays(tmp_path):
    settings, app = _approving(tmp_path, datetime.now(UTC) + timedelta(seconds=0.2))
    entrypoint_setup.confirm(app)
    await entrypoint_setup.watch_approval(app)
    assert entrypoint_setup.pending(settings) is None
    assert entrypoint_setup.default_path(settings).read_text() == as_text(proxy_config())
    assert not getattr(app.state, "restart_requested", False)


async def test_proxy_metadata_marks_requests_that_came_through_it():
    app = FastAPI()

    @app.get("/")
    def inspect(request: Request) -> dict[str, Any]:
        return {"via": getattr(request.state, "via_entrypoint", False)}

    app.add_middleware(ProxyMetadata, config=proxy_config(), token="real-secret")
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 123))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        through = await client.get(
            "/", headers={TOKEN_HEADER: "real-secret", CLIENT_HEADER: "192.168.16.20"}
        )
        assert through.json() == {"via": True}
        assert (await client.get("/")).json() == {"via": False}


def test_an_operator_session_through_the_entry_point_confirms_it(
    authed_client: TestClient, app: FastAPI, settings: Settings
):
    from eugene_plexus_agent.dependencies import _SESSION, verify_bearer

    entrypoint_setup.save_applied(settings, as_text(proxy_config()))
    app.state.entrypoint_confirm_by = datetime.now(UTC) + timedelta(minutes=5)
    token = authed_client.headers["Authorization"].split(" ", 1)[1]
    direct = SimpleNamespace(app=app, state=SimpleNamespace())
    verify_bearer(direct, token, classes=_SESSION)  # type: ignore[arg-type]
    assert entrypoint_setup.pending(settings) is not None
    through = SimpleNamespace(app=app, state=SimpleNamespace(via_entrypoint=True))
    verify_bearer(through, token, classes=_SESSION)  # type: ignore[arg-type]
    assert entrypoint_setup.pending(settings) is None
    assert app.state.entrypoint_confirm_by is None


# --------------------------------------------------------------------------- #
# the routes
# --------------------------------------------------------------------------- #


@pytest.fixture
def restarts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    asked: list[str] = []
    monkeypatch.setattr(
        entrypoint_setup, "schedule_restart", lambda app, why, **kw: asked.append(why)
    )
    return asked


def test_status_says_where_the_file_goes_and_why_it_cannot_run(
    authed_client: TestClient, settings: Settings, monkeypatch
):
    monkeypatch.setattr(entrypoint_setup, "unavailable_reason", lambda s: "no proxy here")
    body = authed_client.get("/v1/entrypoint").json()
    assert body == {
        "available": False,
        "unavailableReason": "no proxy here",
        "path": str(entrypoint_setup.default_path(settings)),
        "active": False,
    }
    refused = authed_client.post(
        "/v1/entrypoint/apply", json={"configuration": as_json(proxy_config())}
    )
    assert refused.status_code == 409 and "no proxy here" in refused.text


def as_json(config: EntryConfig) -> dict[str, Any]:
    return json.loads(as_text(config))


def test_apply_checks_saves_on_approval_and_restarts(
    authed_client: TestClient, settings: Settings, monkeypatch, restarts
):
    monkeypatch.setattr(entrypoint_setup, "unavailable_reason", lambda s: None)
    checked: list[EntryConfig] = []
    monkeypatch.setattr(entrypoint_routes, "validate_with_proxy", lambda s, c, d: checked.append(c))
    bad = authed_client.post(
        "/v1/entrypoint/apply",
        json={
            "configuration": {**as_json(proxy_config()), "proxy": {"addresses": ["172.18.0.0/16"]}}
        },
    )
    assert bad.status_code == 400 and "individual proxy IPs" in bad.text
    assert not entrypoint_setup.default_path(settings).exists() and restarts == []

    reply = authed_client.post(
        "/v1/entrypoint/apply", json={"configuration": as_json(proxy_config())}
    )
    assert reply.status_code == 202, reply.text
    body = reply.json()
    assert body["restarting"] is True and body["active"] is True
    assert body["publicUrls"]["consoleUrl"].rstrip("/") == "https://eugene.example.org"
    confirm_by = datetime.fromisoformat(body["confirmBy"])
    assert timedelta(minutes=14) < confirm_by - datetime.now(UTC) <= timedelta(minutes=15)
    saved = json.loads(entrypoint_setup.default_path(settings).read_text())
    assert EntryConfig.model_validate(saved) == proxy_config()
    assert checked == [proxy_config()]
    assert entrypoint_setup.pending(settings) == {
        "appliedAt": entrypoint_setup.pending(settings)["appliedAt"],  # type: ignore[index]
        "previous": None,
    }
    assert restarts == ["Settings applied an HTTPS setup"]


def test_apply_refuses_what_the_proxy_refuses(
    authed_client: TestClient, settings: Settings, monkeypatch, restarts
):
    monkeypatch.setattr(entrypoint_setup, "unavailable_reason", lambda s: None)

    def refuse(s: Any, c: Any, d: Any) -> None:
        raise ValueError("the bundled proxy refused it: no such file")

    monkeypatch.setattr(entrypoint_routes, "validate_with_proxy", refuse)
    reply = authed_client.post(
        "/v1/entrypoint/apply", json={"configuration": as_json(proxy_config())}
    )
    assert reply.status_code == 400 and "no such file" in reply.text
    assert not entrypoint_setup.default_path(settings).exists() and restarts == []


def test_turning_off_from_settings(
    authed_client: TestClient, app: FastAPI, settings: Settings, restarts
):
    assert authed_client.delete("/v1/entrypoint").status_code == 409
    path = entrypoint_setup.default_path(settings)
    path.write_text(as_text(proxy_config()), encoding="utf-8")
    app.state.entrypoint_config = proxy_config()
    try:
        reply = authed_client.delete("/v1/entrypoint")
    finally:
        app.state.entrypoint_config = None
    assert reply.status_code == 202 and reply.json()["restarting"] is True
    assert not path.exists() and path.with_name("entrypoint.json.disabled").is_file()
    assert restarts == ["Settings turned the HTTPS setup off"]


# --------------------------------------------------------------------------- #
# enrolled machines keep the control root's port
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("nodes", [False, True])
def test_the_control_root_keeps_its_direct_port_unless_nodes_are_behind_it(
    tmp_path: Path, nodes: bool
):
    settings = _settings(tmp_path, entrypoint_config=tmp_path / "entrypoint.json")
    settings._entrypoint_console_origin = "https://eugene.example.org"
    settings._entrypoint_nodes = nodes
    state = AgentState(settings.config_file)
    state.load()
    identity = NodeIdentityStore(tmp_path / "node.yaml")
    identity.load()
    control = shared_child_env(settings, state, identity, "control")
    assert control["AGENT_PUBLIC_ORIGIN"] == "https://eugene.example.org"
    assert (control.get("BIND_HOST") == "127.0.0.1") is nodes
    for kind in ("gateway", "library", "inference-driver", None):
        assert shared_child_env(settings, state, identity, kind)["BIND_HOST"] == "127.0.0.1"


@pytest.mark.parametrize("hosts_control", [False, True])
def test_an_enrolled_workers_drivers_keep_their_direct_bind(
    tmp_path: Path, hosts_control: bool
) -> None:
    """agent#9: the root's gateway dials a worker's inference-drivers and
    tool-drivers at its advertised address. Behind loopback, every model
    and search account on the worker left routing. The control host's
    own drivers, reached by its own gateway, still go behind it."""
    from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest
    from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind

    settings = _settings(tmp_path, entrypoint_config=tmp_path / "entrypoint.json")
    settings._entrypoint_console_origin = "https://worker.example.org"
    state = AgentState(settings.config_file)
    state.load()
    state.apply_config_patch(
        ConfigUpdateRequest.model_validate({"advertiseUrl": "http://192.168.16.20:8079"})
    )
    if hosts_control:
        state.add_topology_entry(
            ComponentEntry(name="control", kind=ComponentKind.control, url="http://127.0.0.1:8083")
        )
    identity = NodeIdentityStore(tmp_path / "node.yaml")
    identity.load()
    identity.record_enrollment(
        name="worker",
        control_url="http://192.168.16.252:8283",
        epoch=1,
        control_public_key="x",
        recovery_public_key=None,
        advertise_url=None,
    )
    for kind in ("inference-driver", "tool-driver"):
        expected = "127.0.0.1" if hosts_control else "0.0.0.0"
        assert shared_child_env(settings, state, identity, kind)["BIND_HOST"] == expected, kind
    for kind in ("gateway", "library", None):
        assert shared_child_env(settings, state, identity, kind)["BIND_HOST"] == "127.0.0.1"


def test_the_supervisor_tells_shared_env_which_kind_it_is_starting() -> None:
    """Without the kind, the control root went loopback with everything else
    (found by sabotage: the env rule was tested, its caller was not)."""
    import logging

    from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind, SpawnConfig
    from eugene_plexus_agent.supervisor import _ComponentPlanner

    asked: list[str | None] = []

    def shared(kind: str | None = None) -> dict[str, str]:
        asked.append(kind)
        return {} if kind == "control" else {"BIND_HOST": "127.0.0.1"}

    control = ComponentEntry(
        name="control",
        kind=ComponentKind.control,
        url="http://127.0.0.1:8083",  # type: ignore[arg-type]
        spawn=SpawnConfig(configFile="/tmp/control.yaml"),
    )
    plan = _ComponentPlanner(control, logging.getLogger("t"), None, shared).plan()
    assert asked == ["control"]
    assert plan.env.get("EUGENE_PLEXUS_CONTROL_BIND_HOST") != "127.0.0.1"


# --------------------------------------------------------------------------- #
# refusals that say which check failed (agent #7)
# --------------------------------------------------------------------------- #


def _bodies(document: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    routes = document["apps"]["http"]["servers"]["entry"]["routes"]
    return [
        (route, handler)
        for route in routes
        for handler in route["handle"]
        if handler["handler"] == "static_response" and handler["status_code"] in (403, 421)
    ]


def test_each_proxy_refusal_names_its_check_and_what_eugene_saw(tmp_path):
    refusals = _bodies(caddy_config(proxy_config(), tmp_path, "secret", {"agent": 8079}))
    texts = [handler["body"] for _, handler in refusals]
    assert len(texts) == len(set(texts)) == 6
    assert "came from {http.request.remote.host}, which is not a proxy Eugene trusts" in texts[0]
    assert "did not say who the visitor is" in texts[1]
    assert "did not mark it as HTTPS" in texts[2] and "Cloudflare" in texts[2]
    assert "came through Cloudflare" in texts[3] and "{http.vars.client_ip}" in texts[3]
    assert refusals[3][0]["match"][0]["header"] == {"Cf-Ray": ["*"]}
    assert "does not answer {http.vars.client_ip}" in texts[4]
    assert refusals[5][1]["status_code"] == 421 and "{http.request.host}" in texts[5]
    for _, handler in refusals:
        assert handler["headers"]["Content-Type"] == ["text/plain; charset=utf-8"]
        assert handler["headers"]["X-Content-Type-Options"] == ["nosniff"]
        # Never what Eugene trusts or allows: only what the caller sent or is.
        for secret in ("172.18.0.4", "192.168.16.0/24", "eugene.example.org"):
            assert secret not in handler["body"]


def test_direct_refusals_name_the_connecting_address(tmp_path):
    direct = EntryConfig.model_validate(
        {
            "internal_ca": True,
            "console": {"origin": "https://eugene.home.arpa:8443", "networks": ["192.168.1.0/24"]},
            "workbench": {"origin": "https://workbench.home.arpa:8443", "networks": ["0.0.0.0/0"]},
        }
    )
    texts = [h["body"] for _, h in _bodies(caddy_config(direct, tmp_path, "s", {"agent": 8079}))]
    assert len(texts) == 3
    assert "does not answer {http.request.remote.host}" in texts[1]


# --------------------------------------------------------------------------- #
# the console from any network, only when the owner says so (2026-10-05)
# --------------------------------------------------------------------------- #


def test_a_public_console_needs_the_owner_to_say_so(tmp_path):
    public = {"origin": "https://eugene.example.org", "networks": ["0.0.0.0/0", "::/0"]}
    with pytest.raises(ValueError, match="public_console"):
        proxy_config(console=public)
    allowed = proxy_config(console=public, public_console=True)
    document = caddy_config(allowed, tmp_path, "secret", {"agent": 8079})
    routes = document["apps"]["http"]["servers"]["entry"]["routes"]
    console = next(
        r
        for r in routes
        if r.get("match", [{}])[0].get("host") == ["eugene.example.org"]
        and "path" not in r["match"][0]
    )
    assert console["match"][0]["client_ip"]["ranges"] == ["0.0.0.0/0", "::/0"]
    # It opens the console and nothing else: node connections stay restricted.
    nodes = {"origin": "https://nodes.example.org", "networks": ["0.0.0.0/0"]}
    with pytest.raises(ValueError, match="node connections require specific"):
        proxy_config(console=public, public_console=True, nodes=nodes)


def test_the_preview_says_what_a_public_console_risks(authed_client: TestClient):
    body = as_json(
        proxy_config(
            console={"origin": "https://eugene.example.org", "networks": ["0.0.0.0/0"]},
            public_console=True,
        )
    )
    reply = authed_client.post("/v1/entrypoint/preview", json={"configuration": body})
    assert reply.status_code == 200, reply.text
    assert reply.json()["configuration"]["public_console"] is True
    assert any(
        "anyone who can reach it can try to sign in" in step
        for step in reply.json()["instructions"]
    )
    refused = authed_client.post(
        "/v1/entrypoint/preview",
        json={"configuration": {**body, "public_console": False}},
    )
    assert refused.status_code == 400 and "public_console" in refused.text


# --------------------------------------------------------------------------- #
# the console on its own port, Workbench behind the proxy (2026-10-05)
# --------------------------------------------------------------------------- #


def direct_config(**changes: Any) -> EntryConfig:
    return proxy_config(
        console_direct=True,
        console={"origin": "https://eugene.example.org", "networks": []},
        **changes,
    )


def test_a_console_on_its_own_port_needs_no_networks_and_cannot_be_public(tmp_path):
    assert direct_config().console_direct is True
    with pytest.raises(ValueError, match="needs at least one source network"):
        proxy_config(console={"origin": "https://eugene.example.org", "networks": []})
    with pytest.raises(ValueError, match="cannot be opened to any network"):
        direct_config(public_console=True)
    with pytest.raises(ValueError, match="needs at least one source network"):
        direct_config(workbench={"origin": "https://workbench.example.org", "networks": []})


def test_its_name_serves_only_workbench_sign_in(tmp_path):
    document = caddy_config(direct_config(), tmp_path, "secret", {"agent": 8079})
    routes = document["apps"]["http"]["servers"]["entry"]["routes"]
    on_console = [
        r for r in routes if r.get("match", [{}])[0].get("host") == ["eugene.example.org"]
    ]
    sign_in, refusal = on_console
    assert sign_in["match"][0]["path"] == ["/oidc/*"]
    assert sign_in["match"][0]["client_ip"]["ranges"] == ["0.0.0.0/0"]  # Workbench's networks
    assert sign_in["handle"][-1]["upstreams"] == [{"dial": "127.0.0.1:8079"}]
    assert refusal["handle"][0]["status_code"] == 403
    assert "stays on its own port" in refusal["handle"][0]["body"]
    assert refusal.get("terminal") is True


def test_any_operator_session_confirms_a_console_that_did_not_move(
    authed_client: TestClient, app: FastAPI, settings: Settings
):
    from eugene_plexus_agent.dependencies import _SESSION, verify_bearer

    entrypoint_setup.save_applied(settings, as_text(direct_config()))
    app.state.entrypoint_config = direct_config()
    app.state.entrypoint_confirm_by = datetime.now(UTC) + timedelta(minutes=5)
    try:
        token = authed_client.headers["Authorization"].split(" ", 1)[1]
        direct = SimpleNamespace(app=app, state=SimpleNamespace())
        verify_bearer(direct, token, classes=_SESSION)  # type: ignore[arg-type]
        assert entrypoint_setup.pending(settings) is None
    finally:
        app.state.entrypoint_config = None


async def test_nobody_signing_in_to_a_console_on_its_own_port_says_so_plainly(tmp_path):
    settings, app = _approving(tmp_path, datetime.now(UTC) - timedelta(seconds=1))
    app.state.entrypoint_config = direct_config()
    await entrypoint_setup.watch_approval(app)
    await asyncio.sleep(0.05)
    reason = entrypoint_setup.reverted(settings) or ""
    assert "nobody signed in to the console within 15 minutes" in reason


def test_the_preview_says_the_console_stays_where_it_is(authed_client: TestClient):
    reply = authed_client.post(
        "/v1/entrypoint/preview", json={"configuration": as_json(direct_config())}
    )
    assert reply.status_code == 200, reply.text
    steps = reply.json()["instructions"]
    assert any("eugene.example.org (sign-in for Workbench only)" in step for step in steps)
    assert any("The console stays on its own port" in step for step in steps)
    assert any("this page comes back in a few seconds" in step for step in steps)
