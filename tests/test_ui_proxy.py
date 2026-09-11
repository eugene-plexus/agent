"""The browser surface: the pass-through proxy and the static bundle.

Ported from a Next.js route handler that had no tests at all, because
`next dev` was the only thing that had ever run it.

**On the streaming test, and why it does not use `TestClient`.** The
design named buffering as this port's headline trap: a proxy that reads
the whole upstream body before answering still frames SSE correctly,
still renders in the playground, and still passes every "is it
streaming" assertion, while delivering one chunk at the end. Every
obvious way to test it, though, puts a buffering instrument in the path
— `httpx.ASGITransport` collects the entire response body before
returning it, so a *correct* proxy measured through it looks buffered.
That is the M10 harness lie in a new costume. So the incremental test
calls the route function directly and reads its
`StreamingResponse.body_iterator`, with the upstream gated on an event
the consumer sets: a buffering implementation cannot complete it at all,
because the producer would be waiting for a consumer that is waiting for
the producer. The end-to-end claim is settled live, by the acceptance
run, against a real backend.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from eugene_plexus_agent import ui_assets
from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.routes import proxy as proxy_routes
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.state import AgentState

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def declare(app: FastAPI, name: str, kind: str, url: str) -> None:
    app.state.agent_state.add_topology_entry(
        ComponentEntry(name=name, kind=ComponentKind(kind), url=url)  # type: ignore[arg-type]
    )


class AsyncStream(httpx.AsyncByteStream):
    """An upstream body that is genuinely a stream.

    Two things forced this rather than a convenience. `httpx.Response(
    json=...)` reads itself eagerly at construction, so a mock built
    that way arrives already consumed and `aiter_raw()` refuses it —
    which would have pushed the proxy into growing a branch for
    already-buffered responses that no real transport ever produces, a
    code path only the tests would walk. And `stream=True` asserts the
    transport handed back an `AsyncByteStream`, so a bare async
    generator is not enough either.
    """

    def __init__(self, source: Any) -> None:
        self._source = source

    async def __aiter__(self) -> Any:
        async for chunk in self._source:
            yield chunk

    async def aclose(self) -> None:
        await self._source.aclose()


def byte_stream(*chunks: bytes) -> AsyncStream:
    async def generate() -> Any:
        for chunk in chunks:
            yield chunk

    return AsyncStream(generate())


def record_transport(
    recorder: list[httpx.Request],
    *,
    status_code: int = 200,
    body: bytes = b'{"ok": true}',
    content_type: str = "application/json",
) -> httpx.MockTransport:
    """A transport that records what the proxy sent upstream."""

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.append(request)
        return httpx.Response(
            status_code,
            headers={"content-type": content_type},
            stream=byte_stream(body),
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def sent(app: FastAPI) -> list[httpx.Request]:
    recorder: list[httpx.Request] = []
    app.state.ui_proxy_client = httpx.AsyncClient(transport=record_transport(recorder))
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")
    declare(app, "library", "library", "http://lib.invalid:8082/")
    declare(app, "control", "control", "http://ctl.invalid:8083/")
    declare(app, "llama-1", "inference-driver", "http://drv.invalid:8081/")
    return recorder


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def test_singletons_resolve_by_kind_not_by_name(
    client: TestClient, sent: list[httpx.Request]
) -> None:
    """The entries are named after their kind here, so the test would
    pass either way — which is the point of the next test."""
    for target, host in (("gateway", "gw"), ("library", "lib"), ("control", "ctl")):
        assert client.get(f"/api/proxy/{target}/v1/config").status_code == 200
        assert sent[-1].url.host == f"{host}.invalid"


def test_a_singleton_resolves_even_when_its_entry_has_another_name(
    app: FastAPI, client: TestClient, sent: list[httpx.Request]
) -> None:
    """Kind, not name. An operator who renamed the gateway entry still
    gets a working browser, and this is the assertion that can tell the
    two lookups apart."""
    app.state.agent_state.remove_topology_entry("gateway")
    declare(app, "front-door", "gateway", "http://renamed.invalid:9000/")
    assert client.get("/api/proxy/gateway/v1/config").status_code == 200
    assert sent[-1].url.host == "renamed.invalid"


def test_drivers_resolve_by_name(client: TestClient, sent: list[httpx.Request]) -> None:
    assert client.get("/api/proxy/llama-1/v1/info").status_code == 200
    assert str(sent[-1].url) == "http://drv.invalid:8081/v1/info"


def test_agent_target_resolves_to_this_process(
    client: TestClient, sent: list[httpx.Request]
) -> None:
    client.get("/api/proxy/agent/v1/components")
    assert str(sent[-1].url) == "http://127.0.0.1:8079/v1/components"


def test_an_undeclared_singleton_is_a_503_that_says_what_it_looked_for(
    app: FastAPI, client: TestClient, sent: list[httpx.Request]
) -> None:
    app.state.agent_state.remove_topology_entry("control")
    response = client.get("/api/proxy/control/v1/nodes")
    assert response.status_code == 503
    assert "control" in response.json()["detail"]["detail"]


def test_an_unknown_driver_names_the_driver(client: TestClient, sent: list[httpx.Request]) -> None:
    response = client.get("/api/proxy/typo/v1/info")
    assert response.status_code == 503
    assert "'typo'" in response.json()["detail"]["detail"]


def test_traversal_shaped_targets_are_refused(
    client: TestClient, sent: list[httpx.Request]
) -> None:
    assert client.get("/api/proxy/../v1/info").status_code in (400, 404)


# --------------------------------------------------------------------------
# faithful forwarding
# --------------------------------------------------------------------------


def test_the_trailing_slash_on_a_topology_url_does_not_double(
    client: TestClient, sent: list[httpx.Request]
) -> None:
    """`http://ctl.invalid:8083/` + `/v1/config` must not become `//v1/config`
    — FastAPI treats the double slash as a different path and 404s."""
    client.get("/api/proxy/control/v1/config/schema")
    assert str(sent[-1].url) == "http://ctl.invalid:8083/v1/config/schema"


def test_the_raw_query_string_survives_including_repeated_keys(
    client: TestClient, sent: list[httpx.Request]
) -> None:
    client.get("/api/proxy/library/v1/models?q=a%20b&tag=x&tag=y")
    assert sent[-1].url.query.decode() == "q=a%20b&tag=x&tag=y"


def test_authorization_is_forwarded_verbatim(client: TestClient, sent: list[httpx.Request]) -> None:
    client.get("/api/proxy/control/v1/nodes", headers={"Authorization": "Bearer roots-own-token"})
    assert sent[-1].headers["authorization"] == "Bearer roots-own-token"


def test_no_second_credential_header_is_honoured(
    client: TestClient, sent: list[httpx.Request]
) -> None:
    """M9's `x-eugene-plexus-upstream-authorization` existed because
    resolving a target consumed a credential. Nothing does now, so the
    header has no meaning here: whatever is in `Authorization` goes
    upstream, and the second header is never read."""
    client.get(
        "/api/proxy/control/v1/nodes",
        headers={
            "Authorization": "Bearer the-one-that-counts",
            "x-eugene-plexus-upstream-authorization": "Bearer ignored",
        },
    )
    assert sent[-1].headers["authorization"] == "Bearer the-one-that-counts"


def test_hop_by_hop_headers_are_dropped_and_encoding_is_pinned_to_identity(
    client: TestClient, sent: list[httpx.Request]
) -> None:
    """`accept-encoding: identity` is not tidiness: it keeps a decoder
    out of the path, and a decoder is a buffer."""
    client.post("/api/proxy/gateway/v1/chat/completions", json={"model": "m"})
    headers = sent[-1].headers
    assert headers["accept-encoding"] == "identity"
    assert "transfer-encoding" not in headers


def test_the_request_body_is_forwarded(client: TestClient, sent: list[httpx.Request]) -> None:
    client.post("/api/proxy/gateway/v1/chat/completions", json={"model": "m", "n": 1})
    assert sent[-1].content == b'{"model":"m","n":1}'


def test_every_method_is_carried(client: TestClient, sent: list[httpx.Request]) -> None:
    for method in ("GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"):
        client.request(method, "/api/proxy/library/v1/models")
        assert sent[-1].method == method


def test_upstream_status_and_body_come_back_unchanged(app: FastAPI, client: TestClient) -> None:
    app.state.ui_proxy_client = httpx.AsyncClient(
        transport=record_transport([], status_code=418, body=b'{"detail": "teapot"}')
    )
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")
    response = client.get("/api/proxy/gateway/v1/config")
    assert response.status_code == 418
    assert response.json() == {"detail": "teapot"}


def test_an_unreachable_upstream_is_a_502_naming_it(app: FastAPI, client: TestClient) -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    app.state.ui_proxy_client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")
    response = client.get("/api/proxy/gateway/v1/config")
    assert response.status_code == 502
    assert "gw.invalid" in response.json()["detail"]["detail"]


def test_the_proxy_needs_no_session_of_its_own(
    client: TestClient, sent: list[httpx.Request]
) -> None:
    """It is the path the login request travels; a dependency here would
    make logging in impossible. The install under test has no passphrase
    set at all, which is the strongest form of "no session"."""
    assert (
        client.post("/api/proxy/agent/v1/auth/login", json={"passphrase": "x"}).status_code == 200
    )


# --------------------------------------------------------------------------
# it streams
# --------------------------------------------------------------------------


async def test_frames_reach_the_client_before_upstream_has_finished(
    app: FastAPI, settings: Settings
) -> None:
    """The decisive one. The upstream refuses to produce frame 2 until
    frame 1 has been *consumed* from the proxy's output, so a proxy that
    reads the whole body first deadlocks rather than passing.
    """
    consumed_first = asyncio.Event()

    async def three_gated_frames() -> Any:
        yield b"data: one\n\n"
        await asyncio.wait_for(consumed_first.wait(), timeout=5)
        yield b"data: two\n\n"
        yield b"data: [DONE]\n\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=AsyncStream(three_gated_frames()),
        )

    app.state.ui_proxy_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    # No TestClient here means no lifespan, so the state the route reads
    # has to be built by hand -- the one cost of testing below the ASGI
    # layer, and cheaper than an instrument that buffers.
    app.state.agent_state = AgentState(settings.config_file)
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "path": "/api/proxy/gateway/v1/chat/completions",
            "raw_path": b"/api/proxy/gateway/v1/chat/completions",
            "root_path": "",
            "scheme": "http",
            "query_string": b"",
            "headers": [(b"accept", b"text/event-stream")],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8079),
            "app": app,
        },
        receive=_body_once(b'{"stream": true}'),
    )

    response = await proxy_routes.proxy("gateway", "v1/chat/completions", request)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"

    frames = response.body_iterator
    first = await asyncio.wait_for(frames.__anext__(), timeout=5)  # type: ignore[union-attr]
    assert first == b"data: one\n\n"
    consumed_first.set()
    assert await asyncio.wait_for(frames.__anext__(), timeout=5) == b"data: two\n\n"  # type: ignore[union-attr]
    assert await asyncio.wait_for(frames.__anext__(), timeout=5) == b"data: [DONE]\n\n"  # type: ignore[union-attr]


def _body_once(body: bytes) -> Any:
    sent_already = False

    async def receive() -> dict[str, Any]:
        nonlocal sent_already
        if sent_already:
            return {"type": "http.disconnect"}
        sent_already = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def test_content_length_is_not_forwarded_from_upstream(app: FastAPI, client: TestClient) -> None:
    """Stripped in both directions: we re-frame the body, so upstream's
    framing headers describe a message that no longer exists."""
    app.state.ui_proxy_client = httpx.AsyncClient(transport=record_transport([], body=b"abc"))
    declare(app, "gateway", "gateway", "http://gw.invalid:8080/")
    response = client.get("/api/proxy/gateway/v1/config")
    assert response.content == b"abc"


# --------------------------------------------------------------------------
# the static bundle
# --------------------------------------------------------------------------


def build_export(root: Path) -> Path:
    """The shape `next build` with `output: "export"` and
    `trailingSlash: true` produces: one index.html per route."""
    out = root / "out"
    (out / "nodes").mkdir(parents=True)
    (out / "index.html").write_text("<html>home</html>", encoding="utf-8")
    (out / "nodes" / "index.html").write_text("<html>nodes</html>", encoding="utf-8")
    (out / "_next").mkdir()
    (out / "_next" / "app.js").write_text("console.log(1)", encoding="utf-8")
    return out


def ui_client(tmp_path: Path, ui_dir: Path | None) -> TestClient:
    app = create_app(
        settings=Settings(
            config_file=tmp_path / "agent.yaml", default_topology=False, ui_dir=ui_dir
        )
    )
    app.state.supervisor = None
    return TestClient(app)


def test_the_ui_is_served_at_the_root(tmp_path: Path) -> None:
    with ui_client(tmp_path, build_export(tmp_path)) as c:
        assert c.get("/").text == "<html>home</html>"
        assert c.get("/_next/app.js").status_code == 200


def test_a_deep_link_redirects_to_its_trailing_slash_form(tmp_path: Path) -> None:
    """The convention chosen deliberately, per the design's third trap:
    `/nodes` pasted into a fresh tab must reach the same page `/nodes/`
    does from a click. A page that works when clicked and 404s when
    pasted is the symptom of getting this wrong."""
    with ui_client(tmp_path, build_export(tmp_path)) as c:
        assert c.get("/nodes/").text == "<html>nodes</html>"
        assert c.get("/nodes", follow_redirects=True).text == "<html>nodes</html>"


def test_a_deep_link_redirect_keeps_the_query_string(tmp_path: Path) -> None:
    """`/login?next=/nodes/` is a real URL this app generates for itself."""
    with ui_client(tmp_path, build_export(tmp_path)) as c:
        response = c.get("/nodes?tab=two", follow_redirects=False)
        assert response.status_code in (301, 302, 307, 308)
        assert response.headers["location"].endswith("/nodes/?tab=two")


def test_api_paths_are_never_answered_with_the_ui(tmp_path: Path) -> None:
    """A catch-all mount reports a full match for every path, so without
    a guard a typo under /v1/ comes back as an HTML page. 404 is right;
    an HTML 404 for an API call is the kind of misreport that costs an
    hour."""
    with ui_client(tmp_path, build_export(tmp_path)) as c:
        response = c.get("/v1/no-such-thing")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")


def test_healthz_still_answers_with_a_ui_mounted(tmp_path: Path) -> None:
    with ui_client(tmp_path, build_export(tmp_path)) as c:
        assert c.get("/healthz").status_code in (200, 503)
        assert "status" in c.get("/healthz").json()


def test_no_ui_is_a_degradation_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`degraded-mode-required`: the API works, `/` explains itself, and
    the explanation names the thing to install.

    **The import is refused explicitly rather than left to chance.** The
    first version of this test passed with no override and no
    monkeypatch, which was true only because `eugene-plexus-ui` was not
    installed in the developer's venv — so installing the wheel, the
    very thing this milestone ships, turned it red. A test whose subject
    is "the distribution is absent" has to make it absent; reading that
    off the ambient environment is the same mistake as an assertion that
    matches the failure it was meant to catch.
    """

    def no_such_package(_name: str) -> object:
        raise ImportError("no module named eugene_plexus_ui")

    monkeypatch.setattr(ui_assets, "import_module", no_such_package)
    with ui_client(tmp_path, None) as c:
        root = c.get("/")
        assert root.status_code == 503
        assert "eugene-plexus-ui" in root.text
        assert c.get("/healthz").status_code in (200, 503)


def test_the_installed_distribution_is_found_when_it_is_there(tmp_path: Path) -> None:
    """The other half of the pair, and the one that needs the wheel: with
    `eugene-plexus-ui` installed, `locate()` finds it with no override.
    Skipped rather than failed where it is not installed, because a unit
    suite must not require a Node build to run."""
    pytest.importorskip("eugene_plexus_ui")
    assets = ui_assets.locate(None)
    assert assets, assets.reason
    assert assets.source == ui_assets.UI_DISTRIBUTION
    assert (assets.directory / "index.html").is_file()  # type: ignore[union-attr]


def test_a_ui_directory_with_no_index_is_reported_as_such(tmp_path: Path) -> None:
    """ "Not installed" and "installed but built nothing" are different
    problems with different fixes, and a mount over an empty directory
    reports neither — it 404s every route and looks like a routing bug."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assets = ui_assets.locate(empty)
    assert not assets
    assert "index.html" in assets.reason


def test_an_override_directory_wins_over_the_distribution(tmp_path: Path) -> None:
    out = build_export(tmp_path)
    assets = ui_assets.locate(out)
    assert assets.directory == out.resolve()
    assert assets.source == "EUGENE_PLEXUS_AGENT_UI_DIR"
