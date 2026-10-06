"""`/link`: a person links their Eugene sign-in to their OS account, at the machine
(job-sites-own-enrollment.md §2.2, §3.2), and the root's removal of a link.

What is faked, and so left to an elevated acceptance run: the operating
system's answer to "which account owns this loopback connection"
(`_socket_owner`, GetExtendedTcpTable and the process token). Everything the
page decides from that answer, and the whole sign-in against the root (the code
exchange, the JWKS, the ID token's signature and claims), is real."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import app_accounts, site_host
from eugene_plexus_agent.routes import site_link
from eugene_plexus_agent.site_links import LinkStore

from .conftest import FakeRoot, enroll_app

ADA = "S-1-5-21-1-2-3-1001"
BO = "S-1-5-21-1-2-3-1002"
PORT = 8079
CLIENT = ("127.0.0.1", 50123)
BASE = f"http://127.0.0.1:{PORT}"
ISSUER = f"{BASE}/oidc"
ROOT = "http://root.invalid:8083"

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the link page serves Windows")


class StubSite:
    """What the page asks of the supervisor, with the answers a test sets."""

    def __init__(self, store: LinkStore) -> None:
        self.store = store
        self.offered = True
        self.changed = 0
        self.never: frozenset[str] = frozenset()
        self._mode = "service"

    def link_page_offered(self) -> bool:
        return self.offered

    def link_store(self) -> LinkStore | None:
        return self.store if self.offered else None

    def never_linked(self) -> frozenset[str]:
        return self.never

    def links_changed(self) -> None:
        self.changed += 1

    def mode(self) -> str:
        return self._mode


class Marked:
    """An ASGI wrapper marking each request as one that came through the entry point."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] in ("http", "websocket"):
            scope["state"] = {**scope.get("state", {}), "via_entrypoint": True}
        await self.app(scope, receive, send)


@pytest.fixture
def site(app: FastAPI, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> StubSite:
    stub = StubSite(LinkStore(tmp_path))
    app.state.site_host = stub
    monkeypatch.setattr(site_link, "account_name", lambda sid: f"PC\\{sid[-4:]}")
    return stub


class Owner:
    """Who the operating system says is at the other end."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sid = ADA
        self.asked: list[tuple[int, int]] = []
        self.fail: str | None = None

        def socket_owner(client_port: int, server_port: int) -> str:
            self.asked.append((client_port, server_port))
            if self.fail:
                raise site_link.NotHere(self.fail)
            return self.sid

        monkeypatch.setattr(site_link, "_socket_owner", socket_owner)


@pytest.fixture
def owner(monkeypatch: pytest.MonkeyPatch) -> Owner:
    return Owner(monkeypatch)


@pytest.fixture
def browser(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, base_url=BASE, client=CLIENT) as client:
        yield client


def pages(app: FastAPI) -> site_link.LinkPages:
    return app.state.site_link_pages  # type: ignore[no-any-return]


def csrf_of(page: str) -> str:
    found = re.search(r"name=csrf value='([^']+)'", page)
    assert found, page
    return found.group(1)


def post(client: TestClient, path: str, csrf: str | None) -> httpx.Response:
    return client.post(path, data={} if csrf is None else {"csrf": csrf})


# --- who may be served ------------------------------------------------------------------


@windows_only
def test_the_page_is_not_there_unless_a_service_install_offers_it(
    browser: TestClient, site: StubSite, owner: Owner
) -> None:
    site.offered = False
    assert browser.get("/link").status_code == 404
    assert browser.get("/link/start").status_code == 404
    assert browser.get("/link/callback?state=x&code=y").status_code == 404
    assert post(browser, "/link/confirm", "x").status_code == 404
    assert post(browser, "/link/remove", "x").status_code == 404
    assert owner.asked == [], "an unoffered page does not even look at the connection"
    assert site.store.load() == []


@windows_only
def test_no_supervisor_at_all_is_no_page(app: FastAPI, browser: TestClient, owner: Owner) -> None:
    app.state.site_host = None
    assert browser.get("/link").status_code == 404


@windows_only
@pytest.mark.parametrize(
    "header",
    [
        "x-eugene-plexus-peer",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-real-ip",
        "forwarded",
        "x-eugene-plexus-forwarded-for",
        "x-eugene-plexus-forwarded-host",
    ],
)
def test_a_forwarding_header_means_it_is_not_the_machine_itself(
    browser: TestClient, site: StubSite, owner: Owner, header: str
) -> None:
    refused = browser.get("/link", headers={header: "127.0.0.1"})
    assert refused.status_code == 403 and "on the machine itself" in refused.text
    assert "set-cookie" not in refused.headers
    assert owner.asked == [], "refused before the connection table is read"


@windows_only
@pytest.mark.parametrize(
    ("client", "base"),
    [
        (("192.168.1.9", 50123), BASE),  # another machine, straight to the port
        (("127.0.0.2", 50123), BASE),  # not the loopback address
        (
            ("127.0.0.1", 50123),
            "http://192.168.1.5:8079",
        ),  # an address of the machine, not loopback
        (("127.0.0.1", 50123), f"https://127.0.0.1:{PORT}"),  # not plain loopback http
    ],
)
def test_only_a_loopback_connection_to_the_agent_is_served(
    app: FastAPI, site: StubSite, owner: Owner, client: tuple[str, int], base: str
) -> None:
    refused = TestClient(app, base_url=base, client=client).get("/link")  # no lifespan
    assert refused.status_code == 403 and "on the machine itself" in refused.text
    assert owner.asked == []


@windows_only
def test_a_request_through_the_entry_point_is_not_served(
    app: FastAPI, site: StubSite, owner: Owner
) -> None:
    through = TestClient(Marked(app), base_url=BASE, client=CLIENT)  # type: ignore[arg-type]
    refused = through.get("/link")
    assert refused.status_code == 403 and "on the machine itself" in refused.text
    assert owner.asked == []


@windows_only
@pytest.mark.parametrize("account", ["S-1-5-18", "S-1-5-80-5", "S-1-5-32-544", "0"])
def test_a_system_or_service_account_at_the_other_end_is_refused(
    browser: TestClient, site: StubSite, owner: Owner, account: str
) -> None:
    owner.sid = account
    refused = browser.get("/link")
    assert refused.status_code == 403 and "no one can be linked" in refused.text
    assert "set-cookie" not in refused.headers
    assert pages_empty(browser)


def pages_empty(browser: TestClient) -> bool:
    state = getattr(browser.app.state, "site_link_pages", None)  # type: ignore[attr-defined]
    return state is None or not state.attempts


@windows_only
def test_a_connection_the_system_cannot_place_is_refused_with_its_reason(
    browser: TestClient, site: StubSite, owner: Owner
) -> None:
    owner.fail = "The program that opened this page could not be found."
    refused = browser.get("/link")
    assert refused.status_code == 403 and "could not be found" in refused.text


@pytest.mark.skipif(sys.platform == "win32", reason="the refusal is for other systems")
def test_elsewhere_the_page_says_it_is_for_windows(
    browser: TestClient, site: StubSite, owner: Owner
) -> None:
    refused = browser.get("/link")
    assert refused.status_code == 403 and "Windows" in refused.text and owner.asked == []


# --- the page, and the start of an attempt ----------------------------------------------


@windows_only
def test_the_page_sets_a_cookie_and_names_the_account(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner
) -> None:
    answer = browser.get("/link")
    assert answer.status_code == 200
    assert "PC\\1001" in answer.text and "/link/start" in answer.text
    cookie = answer.headers["set-cookie"]
    assert f"{site_link.COOKIE}=" in cookie and "HttpOnly" in cookie
    assert "SameSite=strict" in cookie and "Path=/link" in cookie
    assert owner.asked == [(CLIENT[1], PORT)], "the socket looked up is this very connection"
    (attempt,) = pages(app).attempts.values()
    assert attempt.account == ADA
    assert "no-store" in answer.headers["cache-control"]
    assert answer.headers["x-frame-options"] == "DENY"


@windows_only
def test_a_linked_account_is_shown_its_link_and_a_way_to_remove_it(
    browser: TestClient, site: StubSite, owner: Owner
) -> None:
    site.store.add(subject="p-ada", name="Ada", account=ADA, account_name="PC\\ada")
    answer = browser.get("/link")
    assert "Ada" in answer.text and "/link/remove" in answer.text
    assert csrf_of(answer.text)


@windows_only
def test_starting_without_a_cookie_goes_back_to_the_page(
    browser: TestClient, site: StubSite, owner: Owner
) -> None:
    answer = browser.get("/link/start", follow_redirects=False)
    assert answer.status_code == 303 and answer.headers["location"] == "/link"


@windows_only
def test_starting_sends_the_browser_to_eugenes_sign_in_with_everything_bound(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner
) -> None:
    browser.get("/link")
    (attempt,) = pages(app).attempts.values()
    answer = browser.get("/link/start", follow_redirects=False)
    assert answer.status_code == 303
    target = urlsplit(answer.headers["location"])
    assert target.path == "/oidc/authorize" and not target.netloc, "the agent's own /oidc forward"
    query = {k: v[0] for k, v in parse_qs(target.query).items()}
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(attempt.verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert query == {
        "response_type": "code",
        "client_id": "eugene-site-link",
        "redirect_uri": f"http://127.0.0.1:{PORT}/link/callback",
        "scope": "openid profile",
        "state": attempt.state,
        "nonce": attempt.nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    assert len({attempt.state, attempt.nonce, attempt.verifier, attempt.csrf}) == 4


# --- the sign-in, against a fake root ---------------------------------------------------


class Root:
    """The control root's OIDC provider, as far as this page asks it."""

    def __init__(self) -> None:
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.requests: list[httpx.Request] = []
        self.token_status = 200
        self.down = False
        self.claims: dict[str, Any] = {}
        self.sign_with: Any = None
        self.header: dict[str, Any] = {"kid": "k1"}
        self.algorithm = "RS256"
        self.kid_listed = "k1"
        self.nonce = ""

    def jwks(self) -> dict[str, Any]:
        public = jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key(), as_dict=True)
        return {"keys": [{**public, "kid": self.kid_listed, "alg": "RS256", "use": "sig"}]}

    def id_token(self) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": ISSUER,
            "sub": "p-ada",
            "aud": "eugene-site-link",
            "exp": now + 300,
            "iat": now,
            "nonce": self.nonce,
            "eugene_role": "member",
            "name": "Ada Lovelace",
        }
        claims.update(self.claims)
        claims = {k: v for k, v in claims.items() if v is not None}
        key = self.sign_with or self.key
        if self.algorithm == "RS256":
            material: Any = key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        else:
            material = "secret" * 8
        return jwt.encode(claims, material, algorithm=self.algorithm, headers=self.header)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down:
            raise httpx.ConnectError("down", request=request)
        if request.url.path == "/oidc/token":
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": "invalid_grant"})
            return httpx.Response(200, json={"id_token": self.id_token(), "access_token": "x"})
        if request.url.path == "/oidc/jwks":
            return httpx.Response(200, json=self.jwks())
        return httpx.Response(404)


@pytest.fixture
def root(app: FastAPI) -> Root:
    fake = Root()
    app.state.node_identity = SimpleNamespace(
        record=SimpleNamespace(enrolled=True, control_url=ROOT, name="amish")
    )
    app.state.auth_state = SimpleNamespace(
        trust=SimpleNamespace(agent_token=lambda audience: f"agent-token-for-{audience}")
    )
    app.state.site_link_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    return fake


def begin(app: FastAPI, client: TestClient, root: Root) -> site_link.Attempt:
    assert client.get("/link").status_code == 200
    attempt = list(pages(app).attempts.values())[-1]
    root.nonce = attempt.nonce
    return attempt


def callback(client: TestClient, attempt: site_link.Attempt, **query: str) -> httpx.Response:
    params = {"state": attempt.state, "code": "the-code", **query}
    return client.get("/link/callback", params={k: v for k, v in params.items() if v != ""})


@windows_only
def test_the_happy_path_links_the_person_to_the_account(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    attempt = begin(app, browser, root)
    answer = callback(browser, attempt)
    assert answer.status_code == 200 and "Confirm the link" in answer.text
    assert "Ada Lovelace" in answer.text and "PC\\1001" in answer.text
    assert site.store.load() == [], "nothing is written until the person confirms"

    token_call, jwks_call = root.requests
    form = parse_qs(token_call.content.decode())
    assert form["grant_type"] == ["authorization_code"] and form["code"] == ["the-code"]
    assert form["client_id"] == ["eugene-site-link"]
    assert form["code_verifier"] == [attempt.verifier]
    assert form["redirect_uri"] == [f"http://127.0.0.1:{PORT}/link/callback"]
    assert str(token_call.url) == f"{ROOT}/oidc/token" and jwks_call.url.path == "/oidc/jwks"
    sent = token_call.headers
    assert sent["x-eugene-plexus-forwarded-host"] == f"127.0.0.1:{PORT}"
    assert sent["x-eugene-plexus-forwarded-proto"] == "http"
    assert sent["x-eugene-plexus-node-token"] == "agent-token-for-control"

    done = post(browser, "/link/confirm", csrf_of(answer.text))
    assert done.status_code == 200 and "Linked" in done.text
    (link,) = site.store.load()
    assert (link.subject, link.name, link.account, link.account_name) == (
        "p-ada",
        "Ada Lovelace",
        ADA,
        "PC\\1001",
    )
    assert site.changed == 1, "the supervisor is told, so the worker starts"
    assert not pages(app).attempts, "the attempt is spent"
    again = post(browser, "/link/confirm", csrf_of(answer.text))
    assert again.status_code == 403 and len(site.store.load()) == 1


@windows_only
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda r: r.claims.update(nonce="not-the-one"), id="wrong nonce"),
        pytest.param(lambda r: r.claims.update(nonce=None), id="no nonce"),
        pytest.param(lambda r: r.claims.update(aud="workbench"), id="wrong audience"),
        pytest.param(lambda r: r.claims.update(iss="http://evil.invalid/oidc"), id="wrong issuer"),
        pytest.param(lambda r: r.claims.update(eugene_role="operator"), id="the owner's role"),
        pytest.param(lambda r: r.claims.update(eugene_role=None), id="no role"),
        pytest.param(lambda r: r.claims.update(sub="operator"), id="the owner's subject"),
        pytest.param(lambda r: r.claims.update(exp=int(time.time()) - 3600), id="expired"),
        pytest.param(lambda r: r.claims.update(sub=None), id="no subject"),
        pytest.param(lambda r: setattr(r, "sign_with", r.other), id="signed by another key"),
        pytest.param(
            lambda r: setattr(r, "kid_listed", "k2"), id="a key id the root does not list"
        ),
        pytest.param(
            lambda r: (setattr(r, "algorithm", "HS256"), r.header.update(alg="HS256")),
            id="a symmetric algorithm",
        ),
    ],
)
def test_an_id_token_that_does_not_check_out_links_nobody(
    app: FastAPI,
    browser: TestClient,
    site: StubSite,
    owner: Owner,
    root: Root,
    change: Callable[[Root], object],
) -> None:
    attempt = begin(app, browser, root)
    change(root)
    answer = callback(browser, attempt)
    assert answer.status_code == 403 and "Confirm the link" not in answer.text
    # And the confirmation, with the page's own CSRF token, still writes nothing.
    refused = post(browser, "/link/confirm", attempt.csrf)
    assert refused.status_code == 403
    assert site.store.load() == [] and site.changed == 0


@windows_only
def test_the_owner_is_told_to_sign_in_as_themselves(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    attempt = begin(app, browser, root)
    root.claims.update(eugene_role="operator")
    answer = callback(browser, attempt)
    assert "owner is not a person" in answer.text


@windows_only
@pytest.mark.parametrize(
    ("what", "message"),
    [
        ({"state": "someone-elses"}, "not the one this page started"),
        ({"error": "access_denied"}, "did not finish"),
        ({"code": ""}, "did not finish"),
    ],
)
def test_a_callback_that_is_not_this_attempts_is_refused(
    app: FastAPI,
    browser: TestClient,
    site: StubSite,
    owner: Owner,
    root: Root,
    what: dict[str, str],
    message: str,
) -> None:
    attempt = begin(app, browser, root)
    answer = callback(browser, attempt, **what)
    assert answer.status_code == 403 and message in answer.text
    assert root.requests == [], "no code is traded for a callback that is not ours"


@windows_only
def test_a_callback_without_a_started_attempt_is_refused(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    answer = browser.get("/link/callback", params={"state": "x", "code": "y"})
    assert answer.status_code == 403 and "not started on this page" in answer.text
    assert root.requests == []


@windows_only
def test_a_different_account_at_the_callback_ends_the_attempt(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    attempt = begin(app, browser, root)
    owner.sid = BO  # the same browser cookie, another account's process
    answer = callback(browser, attempt)
    assert answer.status_code == 403 and "not started on this page by this account" in answer.text
    assert not pages(app).attempts, "the attempt ended"
    owner.sid = ADA  # and going back to the first account does not revive it
    assert callback(browser, attempt).status_code == 403
    assert root.requests == []
    assert post(browser, "/link/confirm", attempt.csrf).status_code == 403
    assert site.store.load() == []


@windows_only
def test_the_root_not_accepting_the_code_or_not_answering_is_a_sentence(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    attempt = begin(app, browser, root)
    root.token_status = 400
    assert "did not accept" in callback(browser, attempt).text
    root.token_status = 200
    root.down = True
    assert "did not answer" in callback(browser, attempt).text
    app.state.node_identity.record.enrolled = False
    assert "has not joined an install" in callback(browser, attempt).text


@windows_only
def test_a_stale_attempt_is_swept(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    attempt = begin(app, browser, root)
    next(iter(pages(app).attempts.values())).started -= site_link.ATTEMPT_SECONDS + 1
    assert callback(browser, attempt).status_code == 403
    assert not pages(app).attempts


# --- confirm ----------------------------------------------------------------------------


def signed_in(
    app: FastAPI, browser: TestClient, root: Root
) -> tuple[site_link.Attempt, httpx.Response]:
    attempt = begin(app, browser, root)
    answer = callback(browser, attempt)
    assert answer.status_code == 200, answer.text
    return attempt, answer


@windows_only
def test_confirming_needs_the_pages_own_csrf_token(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    attempt, _ = signed_in(app, browser, root)
    for wrong in (None, "", "guess", attempt.csrf + "x", attempt.state):
        refused = post(browser, "/link/confirm", wrong)
        assert refused.status_code == 403 and "not from this page" in refused.text
    assert site.store.load() == [] and site.changed == 0
    assert post(browser, "/link/confirm", attempt.csrf).status_code == 200


@windows_only
def test_confirming_before_signing_in_to_eugene_links_nobody(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    attempt = begin(app, browser, root)  # no callback yet: no person
    assert post(browser, "/link/confirm", attempt.csrf).status_code == 403
    assert site.store.load() == []


@windows_only
def test_another_browsers_csrf_token_does_not_confirm_this_attempt(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    _, _ = signed_in(app, browser, root)
    second = TestClient(app, base_url=BASE, client=CLIENT)
    other = begin(app, second, root)
    assert post(browser, "/link/confirm", other.csrf).status_code == 403
    assert site.store.load() == []


@windows_only
def test_confirming_from_another_account_is_refused(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    attempt, _ = signed_in(app, browser, root)
    owner.sid = BO
    assert post(browser, "/link/confirm", attempt.csrf).status_code == 403
    assert site.store.load() == []


@windows_only
def test_a_conflicting_link_shows_the_conflict_and_changes_nothing(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    site.store.add(subject="p-bo", name="Bo", account=ADA, account_name="PC\\1001")
    attempt, _ = signed_in(app, browser, root)
    answer = post(browser, "/link/confirm", attempt.csrf)
    assert answer.status_code == 403 and "already linked to Bo" in answer.text
    assert [x.subject for x in site.store.load()] == ["p-bo"] and site.changed == 0

    # The same person already linked to another account of theirs.
    site.store.remove("p-bo")
    site.store.add(subject="p-ada", name="Ada", account=BO, account_name="PC\\1002")
    attempt, _ = signed_in(app, browser, root)
    answer = post(browser, "/link/confirm", attempt.csrf)
    assert "already linked to PC\\1002" in answer.text
    assert [x.account for x in site.store.load()] == [BO]


@windows_only
def test_an_account_the_machine_keeps_for_eugene_is_refused_at_confirm(
    app: FastAPI, browser: TestClient, site: StubSite, owner: Owner, root: Root
) -> None:
    site.never = frozenset({ADA})
    attempt, _ = signed_in(app, browser, root)
    answer = post(browser, "/link/confirm", attempt.csrf)
    assert answer.status_code == 403 and "own accounts" in answer.text
    assert site.store.load() == []


# --- remove -----------------------------------------------------------------------------


def two_links(site: StubSite) -> None:
    site.store.add(subject="p-ada", name="Ada", account=ADA, account_name="PC\\1001")
    site.store.add(subject="p-bo", name="Bo", account=BO, account_name="PC\\1002")


@windows_only
def test_removing_removes_only_the_connections_own_accounts_link(
    browser: TestClient, site: StubSite, owner: Owner
) -> None:
    two_links(site)
    page = browser.get("/link")
    assert post(browser, "/link/remove", None).status_code == 403
    assert post(browser, "/link/remove", "guess").status_code == 403
    assert len(site.store.load()) == 2
    done = post(browser, "/link/remove", csrf_of(page.text))
    assert done.status_code == 200 and "Link removed" in done.text
    assert [x.subject for x in site.store.load()] == ["p-bo"] and site.changed == 1


@windows_only
def test_removing_from_another_account_with_this_cookie_removes_nothing(
    browser: TestClient, site: StubSite, owner: Owner
) -> None:
    two_links(site)
    page = browser.get("/link")
    owner.sid = BO
    refused = post(browser, "/link/remove", csrf_of(page.text))
    assert refused.status_code == 403
    assert len(site.store.load()) == 2 and site.changed == 0


@windows_only
def test_removing_with_no_link_is_harmless(
    browser: TestClient, site: StubSite, owner: Owner
) -> None:
    two_links(site)
    site.store.remove("p-ada")
    page = browser.get("/link")
    attempt = next(iter(pages_of(browser).attempts.values()))
    done = post(browser, "/link/remove", attempt.csrf)
    assert done.status_code == 200 and page.status_code == 200
    assert [x.subject for x in site.store.load()] == ["p-bo"] and site.changed == 0


def pages_of(browser: TestClient) -> site_link.LinkPages:
    return browser.app.state.site_link_pages  # type: ignore[attr-defined,no-any-return]


# --- DELETE /v1/site/links/{subject}: only the root takes access away ---------------------


@pytest.fixture
def enrolled(app: FastAPI) -> FakeRoot:
    root = FakeRoot()
    enroll_app(app, root, "gpu-box")
    return root


def test_the_root_removes_a_link(
    app: FastAPI, client: TestClient, enrolled: FakeRoot, site: StubSite
) -> None:
    two_links(site)
    gone = client.delete(
        "/v1/site/links/p-ada",
        headers={"Authorization": f"Bearer {enrolled.service('node:gpu-box')}"},
    )
    assert gone.status_code == 204
    assert [x.subject for x in site.store.load()] == ["p-bo"] and site.changed == 1


def test_no_one_but_the_root_may_remove_a_link(
    app: FastAPI, client: TestClient, enrolled: FakeRoot, site: StubSite
) -> None:
    from .conftest import local_service_token

    two_links(site)
    refused: list[str | None] = [
        None,
        enrolled.session("node:gpu-box"),  # the operator's own session
        local_service_token(app, "control"),  # this node's key signing as control
        local_service_token(app, "gateway"),
        enrolled.service("node:other-box"),  # addressed to another node
    ]
    for token in refused:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        assert client.delete("/v1/site/links/p-ada", headers=headers).status_code == 401
    assert len(site.store.load()) == 2 and site.changed == 0


def test_a_machine_linked_by_root_says_where_to_remove_it(
    app: FastAPI, client: TestClient, enrolled: FakeRoot, site: StubSite
) -> None:
    site._mode = "root"
    two_links(site)
    answer = client.delete(
        "/v1/site/links/p-ada",
        headers={"Authorization": f"Bearer {enrolled.service('node:gpu-box')}"},
    )
    assert answer.status_code == 409 and "--site-unlink" in answer.text
    assert answer.headers["content-type"].startswith("application/problem+json")
    assert len(site.store.load()) == 2 and site.changed == 0


def test_removing_a_link_that_is_not_there_is_still_nothing_to_do(
    app: FastAPI, client: TestClient, enrolled: FakeRoot, site: StubSite
) -> None:
    two_links(site)
    headers = {"Authorization": f"Bearer {enrolled.service('node:gpu-box')}"}
    assert client.delete("/v1/site/links/p-nobody", headers=headers).status_code == 204
    assert len(site.store.load()) == 2 and site.changed == 0
    site.offered = False  # not a job site: no store, still no error
    assert client.delete("/v1/site/links/p-ada", headers=headers).status_code == 204
    app.state.site_host = None
    assert client.delete("/v1/site/links/p-ada", headers=headers).status_code == 204


# --- the supervisor's side of the page, with the real thing ---------------------------------


@windows_only
def test_the_page_is_offered_only_to_a_wanted_service_install(
    app: FastAPI, client: TestClient
) -> None:
    supervisor = site_host.SiteHostSupervisor(app)
    app.state.settings.bind_port = 8079
    manager = app.state.apps
    manager.accounts = app_accounts.AccountSupport("windows_service", None)
    assert supervisor.mode() == "service" and not supervisor.link_page_offered()
    assert supervisor.link_store() is None and supervisor.link_page() is None
    site_host.set_wanted(supervisor.config_dir, True)
    assert supervisor.link_page_offered()
    assert supervisor.link_page() == "http://127.0.0.1:8079/link"
    assert supervisor.environment("service")["SITE_HOST_LINK_PAGE"] == "http://127.0.0.1:8079/link"
    store = supervisor.link_store()
    assert store is not None and store.path == supervisor.config_dir / "site" / "links.json"
    # A per-user install links at the machine instead (J38): no page, but a store.
    manager.accounts = app_accounts.AccountSupport(None, "no accounts")
    assert supervisor.mode() in ("user", None) and not supervisor.link_page_offered()
    if supervisor.mode() == "user":
        assert supervisor.link_store() is not None
    # Linux system installs are root's: observed, never linked by this agent.
    manager.accounts = app_accounts.AccountSupport("systemd", None)
    assert supervisor.mode() == "root"
    assert supervisor.link_store() is None and not supervisor.link_page_offered()
    supervisor.links_changed()
    assert supervisor._wake.is_set()
    assert "S-1-5-18" in supervisor.never_linked()


def test_the_response_bodies_are_json_parseable_problems() -> None:
    answer = site_link.json_problem(409, "T", "D")
    assert json.loads(bytes(answer.body)) == {
        "type": "about:blank",
        "title": "T",
        "status": 409,
        "detail": "D",
    }
