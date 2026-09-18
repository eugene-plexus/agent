"""R1.2: the limiter's key, the witness's evidence, and the frame headers.

Roadmap `docs/design/release-roadmap.md` §2.2. Findings: review §6.1 #1
and §6.3 #32.

**Every check here is written to fail against the code as it was.** The
ones that matter run the request through the *actual*
`ProxyHeadersMiddleware` that uvicorn installs, configured from the
*actual* `uvicorn.Config` this package's entrypoint builds — because the
defect lives in that pair and nowhere a bare `TestClient` can see it. A
check that drove only the FastAPI app would have been green throughout,
which is this project's recurring shape (a pure helper asserted instead
of its caller) and the reason §1 of the roadmap says write the
reproduction first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from eugene_plexus_agent import off_host, peer
from eugene_plexus_agent.__main__ import build_server
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.routes import proxy as proxy_routes
from eugene_plexus_agent.settings import Settings

from .conftest import TEST_PASSPHRASE

WRONG = "not the passphrase"


def served(app: FastAPI, settings: Settings) -> ProxyHeadersMiddleware:
    """`app`, wrapped the way this package's own entrypoint serves it.

    Not `ProxyHeadersMiddleware(app)` with a hand-written argument: the
    subject of half this file is *what the entrypoint passes*, so the
    value comes from the `uvicorn.Config` the entrypoint built. Wire it
    by hand and the check tests the test.
    """
    config = build_server(settings, unattended=True).config
    assert config.proxy_headers, "uvicorn installs the middleware; this assumes it"
    return ProxyHeadersMiddleware(app, trusted_hosts=config.forwarded_allow_ips)


# -- the entrypoint ---------------------------------------------------------


def test_the_entrypoint_trusts_no_forwarding_header(settings: Settings) -> None:
    """uvicorn's default is `forwarded_allow_ips="127.0.0.1"`, and our
    proxy's peer is always 127.0.0.1 -- so the default trusts every
    caller in the world with one hop of indirection. This install has no
    reverse proxy in front of it that we configured; there is nothing to
    trust."""
    assert build_server(settings, unattended=True).config.forwarded_allow_ips == []


# -- the limiter's key ------------------------------------------------------


def _login(client: TestClient, passphrase: str, **headers: str) -> int:
    return client.post(
        "/v1/auth/login", json={"passphrase": passphrase}, headers=headers or None
    ).status_code


def _initialized(app: FastAPI, settings: Settings, peer_host: str) -> TestClient:
    client = TestClient(served(app, settings), client=(peer_host, 43210))
    client.__enter__()
    assert (
        client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
    )
    return client


def test_a_supplied_forwarded_header_does_not_buy_a_fresh_bucket(
    app: FastAPI, settings: Settings
) -> None:
    """The finding, reproduced end to end.

    Five wrong passphrases, each claiming a different origin, then a
    sixth. Before this slice uvicorn rewrote the peer from
    `X-Forwarded-For` (the peer being loopback, which through our own
    proxy it always is), so those were five buckets of one and the
    limiter counted to one forever.
    """
    client = _initialized(app, settings, "127.0.0.1")
    try:
        for i in range(5):
            assert _login(client, WRONG, **{"x-forwarded-for": f"203.0.113.{i}"}) == 401
        assert _login(client, WRONG, **{"x-forwarded-for": "203.0.113.99"}) == 429
    finally:
        client.__exit__(None, None, None)


def test_our_own_header_does_separate_the_buckets(app: FastAPI, settings: Settings) -> None:
    """The half that needs no attacker.

    With every proxied caller keyed `127.0.0.1`, one person mistyping a
    passphrase five times locked every browser out of the install for a
    minute -- on the one screen where a person who has just chosen a
    passphrase is most likely to mistype it.
    """
    client = _initialized(app, settings, "127.0.0.1")
    try:
        for _ in range(5):
            assert _login(client, WRONG, **{peer.PEER_HEADER: "192.168.1.50"}) == 401
        assert _login(client, WRONG, **{peer.PEER_HEADER: "192.168.1.50"}) == 429
        # A different device: still 401, not locked out by the first
        # one's mistakes.
        assert _login(client, WRONG, **{peer.PEER_HEADER: "192.168.1.51"}) == 401
        # And the right passphrase from it still works.
        assert (
            client.post(
                "/v1/auth/login",
                json={"passphrase": TEST_PASSPHRASE},
                headers={peer.PEER_HEADER: "192.168.1.51"},
            ).status_code
            == 200
        )
    finally:
        client.__exit__(None, None, None)


def test_an_off_host_caller_cannot_name_its_own_bucket(app: FastAPI, settings: Settings) -> None:
    """Off-host, the TCP peer is the evidence and the header is noise.

    That is what makes a forged `PEER_HEADER` worthless: it is read only
    where it cannot have come from anyone but us.
    """
    client = _initialized(app, settings, "198.51.100.7")
    try:
        for i in range(5):
            assert _login(client, WRONG, **{peer.PEER_HEADER: f"10.0.0.{i}"}) == 401
        assert _login(client, WRONG, **{peer.PEER_HEADER: "10.0.0.99"}) == 429
    finally:
        client.__exit__(None, None, None)


# -- the proxy's own forwarding ---------------------------------------------


def _headers_through_the_proxy(
    app: FastAPI, sent: dict[str, str], peer_host: str | None
) -> dict[str, str]:
    """What `routes/proxy.py` would put on the wire for `sent`.

    `peer_host=None` is a request whose transport reported no client at
    all, which ASGI permits and which is the one case the stripped set
    guards on its own.
    """

    class _Req:
        def __init__(self) -> None:
            self.headers = Headers(sent)
            self.client = type("C", (), {"host": peer_host})() if peer_host else None
            self.app = app

    return proxy_routes._request_headers(_Req(), proxy_routes.Route(base="http://127.0.0.1:1/"))  # type: ignore[arg-type]


def test_the_proxy_strips_every_forwarding_header(app: FastAPI) -> None:
    """No proxy of ours forwards them verbatim any more. Each one is
    written by whoever sent it, and this install has no reverse proxy in
    front of it that we configured."""
    sent = dict.fromkeys(peer.FORWARDING_HEADERS, "203.0.113.9")
    sent["accept"] = "application/json"
    out = {key.lower() for key in _headers_through_the_proxy(app, sent, "192.168.1.50")}
    assert not (out & peer.FORWARDING_HEADERS)
    assert "accept" in out


def test_the_proxy_says_who_it_is_forwarding_for(app: FastAPI) -> None:
    out = _headers_through_the_proxy(app, {}, "192.168.1.50")
    assert out[peer.PEER_HEADER] == "192.168.1.50"


def test_a_caller_cannot_launder_our_header_through_the_proxy(app: FastAPI) -> None:
    """`PEER_HEADER` is trustworthy for the same reason `HOP_HEADER` is:
    a caller's copy never survives the hop. If it did, the fix would
    have moved the forgery one header to the left."""
    out = _headers_through_the_proxy(app, {peer.PEER_HEADER: "10.9.9.9"}, "192.168.1.50")
    assert out[peer.PEER_HEADER] == "192.168.1.50"


def test_a_request_with_no_peer_cannot_launder_our_header(app: FastAPI) -> None:
    """The case the strip alone covers, and the reason it is not
    redundant with the overwrite one line below it.

    `_request_headers` sets `PEER_HEADER` from the peer it saw, which
    overwrites a supplied one -- but only when there *is* a peer.
    `scope["client"]` is `None` on some transports, and there the
    overwrite does not happen and the caller's own value would go
    straight through. Found by the sabotage pass: removing
    `PEER_HEADER` from the stripped set escaped every other check in
    this file, which meant the guard was untested rather than
    unnecessary.
    """
    out = _headers_through_the_proxy(app, {peer.PEER_HEADER: "10.9.9.9"}, None)
    assert peer.PEER_HEADER not in {key.lower() for key in out}


def test_the_proxy_carries_the_original_peer_through_a_second_hop(app: FastAPI) -> None:
    """A request that already came through one of our proxies keeps the
    address that proxy saw, rather than being relabelled `127.0.0.1` by
    the next one."""
    out = _headers_through_the_proxy(app, {peer.PEER_HEADER: "192.168.1.50"}, "127.0.0.1")
    assert out[peer.PEER_HEADER] == "192.168.1.50"


# -- the witness ------------------------------------------------------------


def _witnessed(settings: Settings, peer_host: str, **headers: str) -> str | None:
    inner = FastAPI()

    @inner.get("/")
    async def root() -> PlainTextResponse:
        return PlainTextResponse("ok")

    witness = off_host.OffHostWitness()
    off_host.install(inner, witness)
    with TestClient(served(inner, settings), client=(peer_host, 4444)) as client:
        client.get("/", headers=headers or None)
    return witness.address


def test_a_local_caller_cannot_forge_the_reach_proof(settings: Settings) -> None:
    """`lastReachedFrom` is the only evidence on the Reach card that
    another device ever got in -- see `off_host.py`, whose whole argument
    is that the field is evidence rather than configuration. Evidence
    its own subject can write is not evidence."""
    assert _witnessed(settings, "127.0.0.1", **{"x-forwarded-for": "203.0.113.9"}) is None


def test_a_real_off_host_caller_is_still_seen(settings: Settings) -> None:
    assert _witnessed(settings, "192.168.1.20") == "192.168.1.20"


def test_a_browser_reaching_a_component_through_the_proxy_is_seen(settings: Settings) -> None:
    """The proxied hop is loopback, so without our header every phone
    that opened the UI would be invisible to the card that exists to
    notice it."""
    assert _witnessed(settings, "127.0.0.1", **{peer.PEER_HEADER: "192.168.1.20"}) == "192.168.1.20"


# -- the rule itself --------------------------------------------------------


@pytest.mark.parametrize(
    ("tcp", "supplied", "expected"),
    [
        ("127.0.0.1", "192.168.1.9", "192.168.1.9"),
        ("::1", "192.168.1.9", "192.168.1.9"),
        ("::ffff:127.0.0.1", "192.168.1.9", "192.168.1.9"),
        # Off-host: the peer is the truth.
        ("198.51.100.7", "192.168.1.9", "198.51.100.7"),
        # Not an address: not evidence, and an unbounded supply of
        # rate-limit bucket keys for anyone who wants one.
        ("127.0.0.1", "not-an-address", "127.0.0.1"),
        ("127.0.0.1", "", "127.0.0.1"),
        ("127.0.0.1", None, "127.0.0.1"),
        # A test transport's placeholder is unknown, not local.
        ("testclient", "192.168.1.9", "testclient"),
    ],
)
def test_the_header_is_read_only_where_it_can_only_be_ours(
    tcp: str, supplied: str | None, expected: str
) -> None:
    assert peer.peer_of(tcp, supplied) == expected


def test_the_stripped_set_covers_the_forwarding_headers_and_our_own() -> None:
    """The header is trustworthy *because* it is stripped. An edit that
    adds a forwarding header to `peer` and forgets the proxy would
    re-open the finding in a way none of the behavioural checks above
    would name."""
    stripped = proxy_routes._STRIPPED_REQUEST_HEADERS
    assert stripped >= peer.FORWARDING_HEADERS
    assert peer.PEER_HEADER in stripped


# -- frame headers (review §6.3 #32) ----------------------------------------


def _ui_app(tmp_path: Path, *, with_index: bool) -> FastAPI:
    ui = tmp_path / "ui"
    ui.mkdir()
    if with_index:
        (ui / "index.html").write_text("<!doctype html><title>x</title>", encoding="utf-8")
    return create_app(
        Settings(config_file=tmp_path / "agent.yaml", default_topology=False, ui_dir=ui)
    )


def test_the_ui_refuses_to_be_framed(tmp_path: Path) -> None:
    """Defence in depth: the console is same-origin with an
    unauthenticated proxy to every component in the install, so a page
    that can frame it is a page that can watch an operator use it."""
    with TestClient(_ui_app(tmp_path, with_index=True)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_the_no_ui_page_refuses_to_be_framed_too(tmp_path: Path) -> None:
    """The degraded page is served at `/` by the same process, and a
    header only some responses carry is a header an attacker asks for
    the other of."""
    with TestClient(_ui_app(tmp_path, with_index=False)) as client:
        response = client.get("/")
    assert response.status_code == 503
    assert response.headers["x-frame-options"] == "DENY"


def test_the_api_still_answers(client: TestClient) -> None:
    """The headers are about the console. A guard that broke `/healthz`
    would take the whole install down on the next poll."""
    assert client.get("/healthz").status_code == 200
