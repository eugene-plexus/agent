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

import contextlib
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import install_proxy, tokens
from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind

# `httpx.Response(json=...)` reads itself eagerly, so a mock built that
# way arrives already consumed and `aiter_raw()` refuses it. The proxy's
# own suite hit this first and explains it at length.
from .conftest import FakeRoot, enroll_app, local_service_token
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


def test_the_lookup_spends_this_nodes_own_token_never_the_callers(
    app: FastAPI, client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    """Per-node token keys (2026-09-25): a caller's credential is
    addressed to this machine and good nowhere else, so forwarding it to
    the root would be refused -- and a proxy that forwarded whatever it
    was handed is the confused deputy the per-node keys exist to remove.
    The lookup is this agent's business, so it spends this agent's own
    token, addressed to the root alone."""
    client.get("/api/proxy/gateway/v1/models", headers={"Authorization": "Bearer operators-token"})

    assert [str(r.url.path) for r in asked] == ["/v1/components", "/v1/nodes"]
    trust = app.state.auth_state.trust
    for r in asked:
        _, _, token = r.headers["authorization"].partition(" ")
        assert token != "operators-token"
        claims = tokens.verify(
            token, bundle=trust.bundle, recipient="control", classes=(tokens.TYP_SERVICE,)
        )
        assert claims.sub == "agent"
        assert claims.aud == ("control",)


def test_a_credential_the_far_node_would_not_accept_is_stripped(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    """Nothing this node cannot vouch for leaves the machine: a string it
    cannot verify is not forwarded on the chance that someone else can."""
    client.get("/api/proxy/gateway/v1/models", headers={"Authorization": "Bearer operators-token"})
    assert "authorization" not in upstream.last.headers


@dataclass
class Exchanges:
    root: FakeRoot
    requests: list[httpx.Request]


@pytest.fixture
def exchanges(app: FastAPI, client: TestClient) -> Exchanges:
    """This node enrolled for real as `worker-1`, beside a node `root`
    that runs the gateway, and a control root that answers lookups and
    exchanges a session the way RFC 8693 says."""
    root = FakeRoot()
    root.register("root", tokens.generate_private_key().public_key())
    root.register("other", tokens.generate_private_key().public_key())
    enroll_app(app, root, "worker-1", control_url=CONTROL_URL)
    asked: list[httpx.Request] = []
    # The gateway on `root`, a driver on `other`: two far machines.
    lookups = control_transport(
        asked,
        components=[
            {"node": "root", "name": "gateway", "kind": "gateway"},
            {"node": "other", "name": "llama-1", "kind": "inference-driver"},
        ],
        nodes=[
            {"name": "root", "url": "http://root.invalid:8279/"},
            {"name": "other", "url": "http://other.invalid:8079/"},
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/auth/token":
            return lookups.handler(request)  # type: ignore[attr-defined, no-any-return]
        asked.append(request)
        body = json.loads(request.content)
        exchanged = root.session(body["audience"], ttl=300, act={"sub": "node:worker-1"})
        return httpx.Response(
            200,
            json={
                "accessToken": exchanged,
                "issuedTokenType": "urn:ietf:params:oauth:token-type:access_token",
                "tokenType": "Bearer",
                "expiresAt": "2099-01-01T00:00:00Z",
            },
        )

    app.state.control_transport = httpx.MockTransport(handler)
    return Exchanges(root=root, requests=asked)


def test_a_session_for_this_console_is_exchanged_for_the_far_node(
    app: FastAPI, client: TestClient, upstream: Upstream, exchanges: Exchanges
) -> None:
    """The operator signed in here, so the session names this machine
    and the root. The far node gets a short token the root issued for it
    alone, and it is asked for once, not on every request."""
    session = exchanges.root.session("node:worker-1", "control")

    for _ in range(2):
        response = client.get(
            "/api/proxy/gateway/v1/models", headers={"Authorization": f"Bearer {session}"}
        )
        assert response.status_code == 200

    exchanged = [r for r in exchanges.requests if r.url.path == "/v1/auth/token"]
    assert len(exchanged) == 1
    body = json.loads(exchanged[0].content)
    assert body == {"subjectToken": session, "audience": "node:root"}
    # The exchange is this agent acting, so it carries this agent's token.
    _, _, actor = exchanged[0].headers["authorization"].partition(" ")
    trust = app.state.auth_state.trust
    assert (
        tokens.verify(
            actor, bundle=trust.bundle, recipient="control", classes=(tokens.TYP_SERVICE,)
        ).sub
        == "agent"
    )
    _, _, sent = upstream.last.headers["authorization"].partition(" ")
    assert sent != session
    claims = tokens.verify(
        sent, bundle=trust.bundle, recipient="node:root", classes=(tokens.TYP_SESSION,)
    )
    assert claims.aud == ("node:root",)


def test_one_session_is_exchanged_once_per_far_machine(
    app: FastAPI, client: TestClient, upstream: Upstream, exchanges: Exchanges
) -> None:
    """The cache is keyed by the session AND the destination: a token the
    root issued for `root` must never be handed to `other`, which would
    refuse it -- or, worse, to a machine that did not."""
    session = exchanges.root.session("node:worker-1", "control")
    headers = {"Authorization": f"Bearer {session}"}

    assert client.get("/api/proxy/gateway/v1/models", headers=headers).status_code == 200
    to_root = upstream.last.headers["authorization"]
    assert client.get("/api/proxy/llama-1/v1/info", headers=headers).status_code == 200
    to_other = upstream.last.headers["authorization"]

    exchanged = [
        json.loads(r.content)["audience"]
        for r in exchanges.requests
        if r.url.path == "/v1/auth/token"
    ]
    assert exchanged == ["node:root", "node:other"]
    assert to_root != to_other
    trust = app.state.auth_state.trust
    for header, recipient in ((to_root, "node:root"), (to_other, "node:other")):
        token = header.removeprefix("Bearer ")
        assert tokens.verify(
            token, bundle=trust.bundle, recipient=recipient, classes=(tokens.TYP_SESSION,)
        ).aud == (recipient,)


def test_an_api_key_header_is_translated_like_a_bearer(
    app: FastAPI, client: TestClient, upstream: Upstream, exchanges: Exchanges
) -> None:
    """Claude Code sends its credential as `x-api-key`. It leaves this
    machine under the same rule as `Authorization`: exchanged when it is
    this console's session, stripped when this node cannot vouch for it."""
    session = exchanges.root.session("node:worker-1", "control")
    client.get("/api/proxy/gateway/v1/models", headers={"x-api-key": session})
    sent = upstream.last.headers["x-api-key"]
    assert sent != session
    trust = app.state.auth_state.trust
    assert tokens.verify(
        sent, bundle=trust.bundle, recipient="node:root", classes=(tokens.TYP_SESSION,)
    ).aud == ("node:root",)

    client.get("/api/proxy/gateway/v1/models", headers={"x-api-key": "not-a-token"})
    assert "x-api-key" not in upstream.last.headers


def test_a_token_already_addressed_to_the_far_node_passes_unchanged(
    client: TestClient, upstream: Upstream, exchanges: Exchanges
) -> None:
    already = exchanges.root.session("node:root")
    client.get("/api/proxy/gateway/v1/models", headers={"Authorization": f"Bearer {already}"})
    assert upstream.last.headers["authorization"] == f"Bearer {already}"
    assert [r for r in exchanges.requests if r.url.path == "/v1/auth/token"] == []


def test_a_local_service_token_is_reminted_only_within_this_nodes_grants(
    app: FastAPI, client: TestClient, upstream: Upstream, exchanges: Exchanges
) -> None:
    """A child's token is good on this machine alone. Leaving it, it is
    re-minted by this node's key for the far node -- but only as far as
    the root's grants for this node reach: a plain node speaks for its
    agent across machines and for nothing else."""
    driver = local_service_token(app, "inference-driver")
    client.get("/api/proxy/gateway/v1/models", headers={"Authorization": f"Bearer {driver}"})
    assert "authorization" not in upstream.last.headers

    agent = local_service_token(app, "agent")
    client.get("/api/proxy/gateway/v1/models", headers={"Authorization": f"Bearer {agent}"})
    _, _, sent = upstream.last.headers["authorization"].partition(" ")
    assert sent != agent
    claims = tokens.verify(
        sent,
        bundle=app.state.auth_state.trust.bundle,
        recipient="node:root",
        classes=(tokens.TYP_SERVICE,),
    )
    assert (claims.iss, claims.sub, claims.aud) == ("node:worker-1", "agent", ("node:root",))


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


async def test_partial_topology_arrives_after_a_node_times_out() -> None:
    """Real socket: mock transports do not enforce HTTP read deadlines."""

    class RootHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/v1/components":
                time.sleep(5.2)  # control's default five-second fan-out plus transit
                body = {
                    "components": [{"node": "root", "name": "gateway", "kind": "gateway"}],
                    "unreachableNodes": ["worker"],
                }
            else:
                body = {"nodes": [{"name": "root", "url": "http://root.invalid:8279/"}]}
            self.send_response(200)
            self.end_headers()
            with contextlib.suppress(OSError):
                self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args: Any) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), RootHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            owner = await install_proxy.InstallTopology().owner_of(
                "gateway",
                control_url=f"http://127.0.0.1:{server.server_port}",
                authorization=None,
                transport=None,
            )
            assert owner.name == "root"
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_empty_timeout_names_the_failure(
    app: FastAPI,
    client: TestClient,
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    app.state.control_transport = httpx.MockTransport(fail)
    enroll(app)
    response = client.get("/api/proxy/gateway/v1/models")
    assert response.status_code == 503
    assert "ReadTimeout" in detail(response)


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


def test_a_refused_credential_is_not_remembered(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    """The lookup spends this node's own token, so a 401 means the root
    does not know this node's key yet -- an enrollment a moment old, or a
    bundle the pull has not caught up with. Both mend themselves;
    remembering the refusal would keep every console on this machine
    broken for the whole negative TTL after the cause had gone."""
    asked: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request)
        if len(asked) <= 2:
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
    assert client.get("/api/proxy/gateway/v1/models").status_code == 200


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


# --------------------------------------------------------------------------
# node:<name> -- another node's agent, not a component on it
# --------------------------------------------------------------------------


def test_a_node_target_reaches_that_nodes_agent_api_directly(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    """No /api/proxy prefix on the far side: this is the agent's own API,
    for the surfaces that are per-agent by nature (a runtime's stop, an
    engine install). The install-wide inference screen acts on a runtime
    that lives elsewhere through exactly this."""
    assert client.post("/api/proxy/node:root/v1/runtimes/llama-1/stop").status_code == 200
    assert str(upstream.last.url) == "http://root.invalid:8279/v1/runtimes/llama-1/stop"


def test_our_own_name_is_the_local_agent_without_a_lookup(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    """The screen names every node the same way; the local one must not
    cost a round trip to the control root and back to ourselves."""
    assert client.get("/api/proxy/node:worker-1/v1/engines").status_code == 200
    assert str(upstream.last.url) == "http://127.0.0.1:8079/v1/engines"
    assert asked == []


def test_an_unknown_node_is_named_in_the_refusal(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    response = client.get("/api/proxy/node:ghost/v1/engines")
    assert response.status_code == 503
    assert "No node named 'ghost'" in detail(response)
    assert upstream.requests == []


def test_a_loopback_node_is_explained_rather_than_dialled_here_too(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    app.state.control_transport = control_transport(
        [], nodes=[{"name": "root", "url": "http://127.0.0.1:8079/"}]
    )
    enroll(app)

    response = client.get("/api/proxy/node:root/v1/engines")
    assert response.status_code == 503
    assert "`advertiseUrl`" in detail(response)
    assert upstream.requests == []


def test_a_forwarded_node_request_is_not_forwarded_again(
    client: TestClient, upstream: Upstream, asked: list[httpx.Request]
) -> None:
    response = client.get(
        "/api/proxy/node:root/v1/engines", headers={install_proxy.HOP_HEADER: "root"}
    )
    assert response.status_code == 503
    assert upstream.requests == []
    assert asked == []


def test_an_unenrolled_node_knows_no_other_nodes(
    app: FastAPI, client: TestClient, upstream: Upstream
) -> None:
    enroll(app, enrolled=False, control_url=None, name=None)
    response = client.get("/api/proxy/node:anything/v1/engines")
    assert response.status_code == 503
    assert "not enrolled" in detail(response)


# --------------------------------------------------------------------------
# what a registry entry may be (R2.4, review §6.2 #15)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/",
        "http://[fe80::1]:8079/",
        "http://0.0.0.0:8079/",
        "file:///etc/shadow",
        # **The sabotage pass added this one.** Removing the scheme
        # check entirely changed no result, because `file:///...` has no
        # host and was refused by the next branch anyway -- so the four
        # cases above could not tell whether the scheme was looked at.
        # A scheme this agent does not speak, with a perfectly ordinary
        # host, is the case that can.
        "gopher://10.0.0.1:8079/",
    ],
)
def test_the_callers_token_is_not_spent_on_an_address_that_is_not_a_node(
    app: FastAPI, client: TestClient, upstream: Upstream, url: str
) -> None:
    """The hop spends the **operator's own bearer**, by design -- and
    dials a base URL a *node* chose. Until R2.4 the control root took
    any URL a correctly signed announcement named, so a worker with a
    leaked signing key could point every console in the install at a
    cloud instance-metadata service and read the operator's token off it.

    The root refuses those addresses now. This is the second half, and
    it is not redundant: a registry written before that fix still holds
    whatever it was told, and this agent is the process that would spend
    the credential. Checked where it is used, not only where it is
    recorded.
    """
    app.state.control_transport = control_transport([], nodes=[{"name": "root", "url": url}])
    enroll(app)

    response = client.get(
        "/api/proxy/gateway/v1/models",
        headers={"Authorization": "Bearer operators-token"},
    )
    assert response.status_code == 503
    assert upstream.requests == [], "the operator's token was sent to a non-node address"
    assert "cannot be a node" in detail(response).lower()
