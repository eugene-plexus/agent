"""A Job Site on the agent's side (`specs/docs/design/remote-nodes.md` §3.1-§3.2).

The entry point's public node route (J3), a site's identity on disk, what a
site refuses to run (J4), how it joins (§3.3 rule 1), and what the control
root is told about its nodes name.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from eugene_plexus_agent import enrollment, node_work, tokens
from eugene_plexus_agent.entrypoint import (
    ENTRY_HEADER,
    PUBLIC_NODE_PATHS,
    PUBLIC_NODES,
    EntryConfig,
    caddy_config,
)
from eugene_plexus_agent.node_identity import NodeIdentityStore


def config(**changes: Any) -> EntryConfig:
    return EntryConfig.model_validate(
        {
            "internal_ca": True,
            "console": {"origin": "https://eugene.home.arpa:8443", "networks": ["192.168.1.0/24"]},
            "workbench": {"origin": "https://workbench.home.arpa:8443", "networks": ["0.0.0.0/0"]},
            "nodes": {"origin": "https://nodes.home.arpa:8443", "networks": ["192.168.1.0/24"]},
            **changes,
        }
    )


def _routes(entry: EntryConfig, tmp: Path) -> list[dict[str, Any]]:
    document = caddy_config(entry, tmp, "secret", {"agent": 8079, "control": 8083})
    routes: list[dict[str, Any]] = document["apps"]["http"]["servers"]["entry"]["routes"]
    return routes


def _for(routes: list[dict[str, Any]], host: str) -> list[dict[str, Any]]:
    return [r for r in routes if any(host in m.get("host", []) for m in r.get("match", []))]


# --------------------------------------------------------------------------- J3


def test_the_public_route_carries_the_node_paths_only_and_marks_them(tmp_path: Path) -> None:
    routes = _for(_routes(config(public_nodes=True), tmp_path), "nodes.home.arpa")
    lan, public, refusal = routes[0], routes[1], routes[2]
    # The name's own networks keep the whole control API, unmarked.
    assert lan["match"][0]["remote_ip"]["ranges"] == ["192.168.1.0/24"]
    assert "path" not in lan["match"][0]
    proxy = lan["handle"][1]["headers"]["request"]
    assert ENTRY_HEADER in proxy["delete"] and ENTRY_HEADER not in proxy["set"]
    # Any network: exactly these methods and paths, marked for the root.
    assert {m["method"][0]: m["path"] for m in public["match"]} == PUBLIC_NODE_PATHS
    assert all("remote_ip" not in m and "client_ip" not in m for m in public["match"])
    assert public["handle"][1]["headers"]["request"]["set"][ENTRY_HEADER] == [PUBLIC_NODES]
    assert public["handle"][1]["upstreams"] == [{"dial": "127.0.0.1:8083"}]
    # Everything else on the name: one sentence naming the console.
    assert refusal["handle"][0]["status_code"] == 403
    assert "machines only" in refusal["handle"][0]["body"]
    assert "https://eugene.home.arpa:8443" in refusal["handle"][0]["body"]
    assert sorted(p for paths in PUBLIC_NODE_PATHS.values() for p in paths) == sorted(
        [
            "/v1/nodes/enroll",
            "/v1/trust/bundle",
            "/v1/trust/tls",
            "/v1/node-helpers/poll",
            "/v1/node-helpers/operations/*/claim",
            "/v1/node-helpers/operations/*/result",
        ]
    )


def test_without_public_nodes_the_name_answers_its_networks_only(tmp_path: Path) -> None:
    routes = _for(_routes(config(), tmp_path), "nodes.home.arpa")
    assert all("remote_ip" in r["match"][0] or r["handle"][0].get("status_code") for r in routes)
    assert not any(
        ENTRY_HEADER in h.get("headers", {}).get("request", {}).get("set", {})
        for r in routes
        for h in r["handle"]
    )


def test_every_other_route_strips_the_mark_a_caller_sends(tmp_path: Path) -> None:
    for route in _routes(config(public_nodes=True), tmp_path):
        for handler in route.get("handle", []):
            request = handler.get("headers", {}).get("request")
            if request and request.get("set", {}).get(ENTRY_HEADER) != [PUBLIC_NODES]:
                assert ENTRY_HEADER in request["delete"]


@pytest.mark.parametrize(
    "changes",
    [
        {"public_nodes": True, "nodes": None},
        {"nodes": {"origin": "https://nodes.home.arpa:8443", "networks": ["0.0.0.0/0"]}},
        {"workbench": {"origin": "https://203.0.113.5:8443", "networks": ["0.0.0.0/0"]}},
        {"nodes": {"origin": "https://127.0.0.1:8443", "networks": ["192.168.1.0/24"]}},
    ],
)
def test_public_nodes_needs_a_name_and_only_the_nodes_name_may_be_an_address(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        config(**changes)


def test_the_nodes_name_may_be_a_bare_address(tmp_path: Path) -> None:
    entry = config(
        public_nodes=True,
        nodes={"origin": "https://203.0.113.5:8443", "networks": ["192.168.1.0/24"]},
    )
    assert entry.nodes is not None and entry.nodes.is_address
    assert entry.public_urls()["nodesUrl"] == "https://203.0.113.5:8443"
    assert entry.public_urls()["publicNodes"] is True
    assert _for(_routes(entry, tmp_path), "203.0.113.5")


def test_automatic_certificates_refuse_an_address() -> None:
    with pytest.raises(ValidationError, match="names, not addresses"):
        EntryConfig.model_validate(
            {
                "acme": {"email": "a@example.com", "accept_terms": True},
                "console": {"origin": "https://eugene.example.com", "networks": ["10.0.0.0/8"]},
                "workbench": {"origin": "https://wb.example.com", "networks": ["0.0.0.0/0"]},
                "nodes": {"origin": "https://203.0.113.5", "networks": ["10.0.0.0/8"]},
            }
        )


def test_the_root_is_told_its_nodes_name_and_where_to_read_its_key(tmp_path: Path) -> None:
    from eugene_plexus_agent import entrypoint
    from eugene_plexus_agent.app import shared_child_env
    from eugene_plexus_agent.settings import Settings
    from eugene_plexus_agent.state import AgentState

    file = tmp_path / "entrypoint.json"
    file.write_text(json.dumps(config(public_nodes=True).model_dump(mode="json")))
    settings = Settings(config_file=tmp_path / "agent.yaml", entrypoint_config=file)
    entrypoint.resolve(settings)
    settings._entrypoint_prepared = False
    loaded = entrypoint.resolve(settings)
    assert loaded is not None
    settings._entrypoint_ready = loaded
    settings._entrypoint_prepared = True
    # What `prepare` records, without running the proxy to validate.
    settings._entrypoint_nodes = True
    settings._entrypoint_nodes_origin = loaded.nodes.origin  # type: ignore[union-attr]
    settings._entrypoint_nodes_public = True
    settings._entrypoint_nodes_probe = f"127.0.0.1:{loaded.listen_port}"
    identity = NodeIdentityStore(tmp_path / "node.yaml")
    env = shared_child_env(settings, AgentState(tmp_path / "agent.yaml"), identity, "control")
    assert env["NODES_ORIGIN"] == "https://nodes.home.arpa:8443"
    assert env["NODES_PUBLIC"] == "true" and env["NODES_PROBE"] == "127.0.0.1:8443"
    other = shared_child_env(settings, AgentState(tmp_path / "agent.yaml"), identity, "gateway")
    assert "NODES_ORIGIN" not in other


# --------------------------------------------------------------------------- J4


def _site(tmp: Path, *, job_site: bool) -> SimpleNamespace:
    store = NodeIdentityStore(tmp / "node.yaml")
    store.ensure_keypair()
    store.record_enrollment(
        name="desk",
        control_url="https://nodes.home.arpa:8443",
        epoch=1,
        control_public_key="key",
        recovery_public_key=None,
        advertise_url=None,
        job_site=job_site,
    )
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(node_identity=store)))


def test_a_site_says_so_in_node_yaml_and_forgets_it_on_leaving(tmp_path: Path) -> None:
    store = _site(tmp_path, job_site=True).app.state.node_identity
    assert "jobSite: true" in (tmp_path / "node.yaml").read_text()
    again = NodeIdentityStore(tmp_path / "node.yaml")
    again.load()
    assert again.record.job_site is True
    store.unenroll()
    assert "jobSite" not in (tmp_path / "node.yaml").read_text()
    ordinary = _site(tmp_path / "other", job_site=False).app.state.node_identity
    assert ordinary.record.job_site is None
    assert "jobSite" not in (tmp_path / "other" / "node.yaml").read_text()


def test_a_site_starts_nothing(tmp_path: Path) -> None:
    request = _site(tmp_path, job_site=True)
    with pytest.raises(HTTPException) as refused:
        node_work.refuse_on_job_site(request)  # type: ignore[arg-type]
    assert refused.value.status_code == 409 and "job site" in str(refused.value.detail)

    async def launch() -> None:
        async with node_work.launch_guard(request):  # type: ignore[arg-type]
            pytest.fail("a job site launched something")

    with pytest.raises(HTTPException):
        asyncio.run(launch())
    node_work.refuse_on_job_site(_site(tmp_path / "o", job_site=False))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- joining


def test_a_site_joins_with_its_owner_and_no_address(tmp_path: Path) -> None:
    identity = tokens.generate_private_key()
    seen: dict[str, Any] = {}

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.update(body)
        node_key = tokens.load_public(body["tokenPublicKey"])
        bundle = tokens.build_bundle(
            authority=identity,
            version=1,
            epoch=1,
            keys=[
                tokens.TrustKey(
                    kid=tokens.thumbprint(node_key),
                    issuer="node:desk",
                    public=node_key,
                    grants=frozenset({"files"}),
                )
            ],
        )
        return httpx.Response(
            201,
            json={
                "name": "desk",
                "epoch": 1,
                "controlPublicKey": tokens.public_b64(identity),
                "trustBundle": {"jws": bundle.jws},
                "grants": ["files"],
            },
        )

    store = NodeIdentityStore(tmp_path / "node.yaml")
    outcome = asyncio.run(
        enrollment.perform_enrollment(
            store=store,
            control_url="http://192.168.1.10:8083",
            token="join",
            name="desk",
            advertise_url="http://192.168.1.20:8079",
            transport=httpx.MockTransport(answer),
            root_key=tokens.public_b64(identity),
            owner=("ada", "her own password"),
        )
    )
    assert outcome.name == "desk"
    assert seen["owner"] == {"name": "ada", "password": "her own password"}
    assert "url" not in seen
    assert store.record.job_site is True and store.record.advertise_url is None
    assert "her own password" not in (tmp_path / "node.yaml").read_text()


def test_a_join_answered_by_another_root_records_nothing(tmp_path: Path) -> None:
    pinned = tokens.generate_private_key()
    other = tokens.generate_private_key()

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={
                "name": "desk",
                "epoch": 1,
                "controlPublicKey": tokens.public_b64(other),
                "trustBundle": {"jws": "x"},
            },
        )

    store = NodeIdentityStore(tmp_path / "node.yaml")
    with pytest.raises(enrollment.EnrollmentError, match="not the root the join command names"):
        asyncio.run(
            enrollment.perform_enrollment(
                store=store,
                control_url="http://192.168.1.10:8083",
                token="join",
                name="desk",
                advertise_url=None,
                transport=httpx.MockTransport(answer),
                root_key=tokens.public_b64(pinned),
                owner=("ada", "pw"),
            )
        )
    assert not store.record.enrolled


def test_over_https_a_site_needs_the_root_key(tmp_path: Path) -> None:
    store = NodeIdentityStore(tmp_path / "node.yaml")
    with pytest.raises(enrollment.EnrollmentError, match=r"root-key|identity key"):
        asyncio.run(
            enrollment.perform_enrollment(
                store=store,
                control_url="https://nodes.example.test:8443",
                token="join",
                name="desk",
                advertise_url=None,
                owner=("ada", "pw"),
            )
        )


# --------------------------------------------------------------------------- what a site does


def test_the_helper_honours_the_systems_proxy_for_a_root_off_this_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proxy fix (§3.1): a forcing network blocked the helper, whose client
    turned the proxy off by hand. The rule is now the address's own."""
    from eugene_plexus_agent import node_file_helper

    made: list[dict[str, Any]] = []

    class Stop(Exception):
        pass

    class Client:
        async def post(self, *args: Any, **kwargs: Any) -> Any:
            raise Stop

        async def aclose(self) -> None:
            pass

    def client_for(url: str, **kwargs: Any) -> Client:
        made.append({"url": url, **kwargs})
        return Client()

    monkeypatch.setattr(node_file_helper, "client_for", client_for)
    for root in ("https://root.example.com", "http://192.168.1.10:8083"):
        app = SimpleNamespace(
            state=SimpleNamespace(
                node_identity=SimpleNamespace(
                    record=SimpleNamespace(enrolled=True, control_url=root, job_site=None)
                ),
                auth_state=SimpleNamespace(trust=SimpleNamespace(agent_token=lambda _: "t")),
            )
        )
        relay = node_file_helper.NodeFileHelper(app)  # type: ignore[arg-type]
        with pytest.raises(Stop):
            asyncio.run(relay.step())
    assert [m["url"] for m in made] == ["https://root.example.com", "http://192.168.1.10:8083"]
    assert all("trust_env" not in m for m in made), made


def test_a_site_never_asks_for_run_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_agent import run_worker

    asked: list[object] = []
    ticks = {"n": 0}

    async def sleep(_seconds: float) -> None:
        ticks["n"] += 1
        if ticks["n"] > 3:
            raise asyncio.CancelledError

    async def library_client_for(context: object) -> None:
        asked.append(context)

    monkeypatch.setattr(run_worker.asyncio, "sleep", sleep)
    monkeypatch.setattr(run_worker.actions, "library_client_for", library_client_for)
    for job_site, expected in ((True, 0), (False, 3)):
        asked.clear()
        ticks["n"] = 0
        request = _site(tmp_path / str(job_site), job_site=job_site)
        worker = run_worker.RunWorker(request.app)  # type: ignore[arg-type]
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(worker.run_forever())
        assert len(asked) == expected
