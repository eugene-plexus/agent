"""A worker's browser is a console for the install, not for one host.

Reported from the live two-machine install: clicking Gateway on the
worker returned a 503 with a JSON envelope pasted into the message. The
worker declares no gateway, correctly -- `should_seed` refuses to seed a
control plane onto an enrolled node -- so every screen but Config and
the playground was unreachable from there.

The hop goes to the **owning node's agent**, not to the component, and
`install_proxy`'s docstring carries why: a containerised control root
reports its gateway as `http://127.0.0.1:8080/`, which is true on that
host, useless everywhere else, and unfixable by rewriting because the
NAS publishes that port as 8280 and nothing inside the container knows
it. One address per node is the thing the install already maintains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import install_proxy
from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind

# `httpx.Response(json=...)` reads itself eagerly, so a mock built that
# way arrives already consumed and `aiter_raw()` refuses it. The proxy's
# own suite hit this first and explains it at length.
from .test_ui_proxy import byte_stream

CONTROL_URL = "http://ctl.invalid:8083"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


@dataclass
class FakeRecord:
    enrolled: bool = True
    control_url: str | None = CONTROL_URL
    name: str | None = "worker-1"


class FakeIdentity:
    def __init__(self, record: FakeRecord) -> None:
        self.record = record


def enroll(app: FastAPI, **kwargs: Any) -> FakeRecord:
    record = FakeRecord(**kwargs)
    app.state.node_identity = FakeIdentity(record)
    return record


def declare(app: FastAPI, name: str, kind: str, url: str) -> None:
    app.state.agent_state.add_topology_entry(
        ComponentEntry(name=name, kind=ComponentKind(kind), url=url)  # type: ignore[arg-type]
    )


def control_transport(
    recorder: list[httpx.Request],
    *,
    components: list[dict[str, Any]] | None = None,
    nodes: list[dict[str, Any]] | None = None,
    status_code: int = 200,
    fail: bool = False,
) -> httpx.MockTransport:
    """A stand-in control root that records what was asked of it."""

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.append(request)
        if fail:
            raise httpx.ConnectError("connection refused", request=request)
        if status_code >= 400:
            return httpx.Response(
                status_code,
                json={"detail": {"title": "Locked", "detail": "the trust root is sealed"}},
            )
        if request.url.path == "/v1/components":
            body: dict[str, Any] = {
                "components": components
                if components is not None
                else [{"node": "root", "name": "gateway", "kind": "gateway"}]
            }
        else:
            body = {
                "nodes": nodes
                if nodes is not None
                else [{"name": "root", "url": "http://root.invalid:8279/"}]
            }
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


class Upstream:
    """Whatever the proxy actually sent, and to where."""

    def __init__(self, app: FastAPI) -> None:
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=byte_stream(b'{"ok": true}'),
            )

        app.state.ui_proxy_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


@pytest.fixture
def upstream(app: FastAPI) -> Upstream:
    return Upstream(app)


@pytest.fixture
def asked(app: FastAPI) -> list[httpx.Request]:
    """The default install: a gateway on node `root`, reachable."""
    recorder: list[httpx.Request] = []
    app.state.control_transport = control_transport(recorder)
    enroll(app)
    return recorder


def detail(response: httpx.Response) -> str:
    return str(response.json()["detail"]["detail"])


# --------------------------------------------------------------------------
# the hop
# --------------------------------------------------------------------------


def test_a_singleton_elsewhere_is_forwarded_to_the_node_that_has_it(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    """The whole feature in one assertion. The request goes to the
    owning node's *agent proxy*, not to the component's own URL, which
    is the address that cannot survive leaving its host."""
    assert client.get("/api/proxy/gateway/v1/models").status_code == 200
    assert str(upstream.last.url) == "http://root.invalid:8279/api/proxy/gateway/v1/models"


def test_the_query_string_survives_the_hop(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    client.get("/api/proxy/gateway/v1/metrics/requests?limit=5&status=served&status=failed")
    assert upstream.last.url.query.decode() == "limit=5&status=served&status=failed"


def test_a_driver_on_another_node_hops_too(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    """Not only the three singletons. A driver is resolved by name in
    the install-wide view exactly as it is locally, so a worker can open
    another worker's driver config."""
    app.state.control_transport = control_transport(
        [],
        components=[{"node": "other", "name": "llama-1", "kind": "inference-driver"}],
        nodes=[{"name": "other", "url": "http://other.invalid:8079/"}],
    )
    enroll(app)

    assert client.get("/api/proxy/llama-1/v1/info").status_code == 200
    assert str(upstream.last.url) == "http://other.invalid:8079/api/proxy/llama-1/v1/info"


def test_a_local_component_is_never_looked_up(
    app: FastAPI, client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    """Local first, always. A control root is enrolled too, and asking
    the install where its own gateway is would put an HTTP call in the
    path of every request the console makes."""
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")

    assert client.get("/api/proxy/gateway/v1/models").status_code == 200
    assert str(upstream.last.url) == "http://gw.invalid:8080/v1/models"
    assert asked == []


# --------------------------------------------------------------------------
# the credential
# --------------------------------------------------------------------------


def test_the_lookup_spends_the_callers_own_token(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    """An enrolled node holds the install's signing key, so the browser's
    token is one the root accepts. The proxy minting one of its own
    would take back the property the port to the agent was built for."""
    client.get("/api/proxy/gateway/v1/models", headers={"Authorization": "Bearer operators-token"})

    assert [str(r.url.path) for r in asked] == ["/v1/components", "/v1/nodes"]
    assert {r.headers.get("authorization") for r in asked} == {"Bearer operators-token"}


def test_the_token_also_reaches_the_far_component(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    client.get("/api/proxy/gateway/v1/models", headers={"Authorization": "Bearer operators-token"})
    assert upstream.last.headers["authorization"] == "Bearer operators-token"


# --------------------------------------------------------------------------
# one hop, never two
# --------------------------------------------------------------------------


def test_the_hop_is_marked_with_this_nodes_name(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    client.get("/api/proxy/gateway/v1/models")
    assert upstream.last.headers[install_proxy.HOP_HEADER] == "worker-1"


def test_a_forwarded_request_is_never_forwarded_again(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    """Sabotage of the loop guard: without it this request would be sent
    straight back out to `root`, which would send it back here."""
    response = client.get(
        "/api/proxy/gateway/v1/models", headers={install_proxy.HOP_HEADER: "root"}
    )

    assert response.status_code == 503
    assert "'root' forwarded this request here" in detail(response)
    assert upstream.requests == []
    assert asked == []


def test_the_hop_marker_never_reaches_a_component(
    app: FastAPI, client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    """The receiving end of a hop resolves locally and forwards. The
    marker is ours and a component has no use for it."""
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")

    client.get("/api/proxy/gateway/v1/models", headers={install_proxy.HOP_HEADER: "worker-9"})

    assert install_proxy.HOP_HEADER not in upstream.last.headers


def test_a_component_the_registry_puts_here_is_not_chased_in_a_circle(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    """The registry and this node's topology disagree. Hopping to
    ourselves would terminate -- the marker would stop the second pass --
    but only after a pointless round trip, and the error would name the
    wrong problem."""
    app.state.control_transport = control_transport(
        [],
        components=[{"node": "worker-1", "name": "gateway", "kind": "gateway"}],
        nodes=[{"name": "worker-1", "url": "http://worker.invalid:8079/"}],
    )
    enroll(app)

    response = client.get("/api/proxy/gateway/v1/models")
    assert response.status_code == 503
    assert "runs on this node" in detail(response)
    assert upstream.requests == []


# --------------------------------------------------------------------------
# the four ways it can fail, each with its own next action
# --------------------------------------------------------------------------


def test_a_loopback_registry_entry_is_explained_not_dialled(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    """THE FAILURE THE LIVE INSTALL ACTUALLY HAD. A containerised
    control root derives `127.0.0.1:8079` as its own address, so the
    registry hands out an address that resolves, connects to nothing on
    the asking machine, and would otherwise be reported as `gateway did
    not answer` -- pointing at the wrong host entirely."""
    app.state.control_transport = control_transport(
        [],
        nodes=[{"name": "root", "url": "http://127.0.0.1:8079/"}],
    )
    enroll(app)

    response = client.get("/api/proxy/gateway/v1/models")
    assert response.status_code == 503
    assert "advertises http://127.0.0.1:8079/" in detail(response)
    assert "`advertiseUrl`" in detail(response)
    assert upstream.requests == []


def test_an_unreachable_control_root_says_so(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    app.state.control_transport = control_transport([], fail=True)
    enroll(app)

    response = client.get("/api/proxy/gateway/v1/models")
    assert response.status_code == 503
    assert CONTROL_URL in detail(response)


def test_a_sealed_control_root_relays_what_it_said(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    """A locked trust root answers 503 `Locked`, and that sentence is
    the operator's next action. Flattening it into "could not resolve"
    is the mistake `/nodes` already had to unlearn."""
    app.state.control_transport = control_transport([], status_code=503)
    enroll(app)

    response = client.get("/api/proxy/gateway/v1/models")
    assert response.status_code == 503
    assert "the trust root is sealed" in detail(response)


def test_a_component_nothing_runs_is_not_blamed_on_the_hop(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    app.state.control_transport = control_transport([], components=[])
    enroll(app)

    response = client.get("/api/proxy/library/v1/models")
    assert response.status_code == 503
    assert "anywhere in this install" in detail(response)


def test_an_unenrolled_node_has_nowhere_to_look(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    """A single-host install that has not declared a gateway. There is
    no install-wide view to consult, and saying "it lives on the control
    host" -- which the old message did unconditionally -- would be
    advice about an install this operator does not have."""
    enroll(app, enrolled=False, control_url=None)

    response = client.get("/api/proxy/gateway/v1/models")
    assert response.status_code == 503
    assert "not enrolled with a control root" in detail(response)


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------


def test_the_install_is_read_once_for_many_panels(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    """`/v1/components` polls every node in the install. A dashboard
    opening six panels must not become six polls."""
    for _ in range(4):
        client.get("/api/proxy/gateway/v1/models")

    assert len(upstream.requests) == 4
    assert len(asked) == 2


def test_a_node_that_did_not_answer_is_re_read_next_time(
    app: FastAPI, client: TestClient, asked: list[httpx.Request]
) -> None:
    """A cached address that turns out to be dead must not be repeated
    for the whole TTL: a node that has just moved is exactly when the
    registry is worth re-reading."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app.state.ui_proxy_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert client.get("/api/proxy/gateway/v1/models").status_code == 502
    assert client.get("/api/proxy/gateway/v1/models").status_code == 502
    assert len(asked) == 4


def test_a_refused_credential_is_not_remembered_for_the_next_caller(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    """The proxy is unauthenticated by design -- it is the path the login
    request itself travels -- so an anonymous request can reach this
    lookup and be refused. Caching that refusal would answer the
    signed-in operator with someone else's 401 for the whole negative
    TTL."""
    asked: list[httpx.Request] = []
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request)
        seen.append(request.headers.get("authorization"))
        if request.headers.get("authorization") is None:
            return httpx.Response(401, json={"detail": "Not authenticated"})
        body = (
            {"components": [{"node": "root", "name": "gateway", "kind": "gateway"}]}
            if request.url.path == "/v1/components"
            else {"nodes": [{"name": "root", "url": "http://root.invalid:8279/"}]}
        )
        return httpx.Response(200, json=body)

    app.state.control_transport = httpx.MockTransport(handler)
    enroll(app)

    assert client.get("/api/proxy/gateway/v1/models").status_code == 503
    response = client.get("/api/proxy/gateway/v1/models", headers={"Authorization": "Bearer t"})
    assert response.status_code == 200


def test_an_unreachable_upstream_names_the_node_and_not_just_a_url(
    app: FastAPI, client: TestClient, asked: list[httpx.Request]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app.state.ui_proxy_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    response = client.get("/api/proxy/gateway/v1/models")
    assert response.status_code == 502
    assert "node 'root'" in detail(response)


def test_the_lookup_is_not_repeated_per_target(
    client: TestClient, upstream: Upstream, app: FastAPI, asked: list[httpx.Request]
) -> None:
    """One snapshot answers for every target in it, which is what makes
    a page of six panels cost one read rather than one per panel."""
    app.state.control_transport = control_transport(
        asked,
        components=[
            {"node": "root", "name": "gateway", "kind": "gateway"},
            {"node": "root", "name": "library", "kind": "library"},
        ],
    )

    client.get("/api/proxy/gateway/v1/models")
    client.get("/api/proxy/library/v1/models")

    assert len(asked) == 2
    assert [str(r.url) for r in upstream.requests] == [
        "http://root.invalid:8279/api/proxy/gateway/v1/models",
        "http://root.invalid:8279/api/proxy/library/v1/models",
    ]


# --------------------------------------------------------------------------
# resolution details that are easy to get subtly wrong
# --------------------------------------------------------------------------


def test_a_kind_wins_over_a_driver_that_shares_its_name(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    """`resolve_local` looks singletons up by KIND, so the install-wide
    lookup has to as well -- otherwise a driver an operator named
    `gateway` would answer for the install's gateway on a worker and
    not on the control host."""
    app.state.control_transport = control_transport(
        [],
        components=[
            {"node": "other", "name": "gateway", "kind": "inference-driver"},
            {"node": "root", "name": "front-door", "kind": "gateway"},
        ],
        nodes=[
            {"name": "other", "url": "http://other.invalid:8079/"},
            {"name": "root", "url": "http://root.invalid:8279/"},
        ],
    )
    enroll(app)

    client.get("/api/proxy/gateway/v1/models")
    assert upstream.last.url.host == "root.invalid"


def test_a_malformed_answer_is_a_miss_rather_than_a_crash(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    """Degraded mode reaches the console too: a root answering something
    unexpected must not turn a page into a 500."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"content-type": "text/plain"})

    app.state.control_transport = httpx.MockTransport(handler)
    enroll(app)

    assert client.get("/api/proxy/gateway/v1/models").status_code == 503


def test_the_body_of_a_post_survives_the_hop(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    client.post("/api/proxy/gateway/v1/chat/completions", json={"model": "m", "messages": []})
    assert json.loads(upstream.last.content)["model"] == "m"
