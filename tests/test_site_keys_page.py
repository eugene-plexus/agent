"""J14a: a person's own key, made and pinned on the loopback link page, and the
changes the site host holds, approved there (`person-held-keys.md` §4.1, §12).

As in `test_site_link_page.py`, the operating system's answer to "which
account owns this connection" is faked; everything the page decides from it
is real. The site host's `/v1/held` API is a stub that records what it was
sent: what it does with a signature is the site host's own tests'."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent.routes import site_link
from eugene_plexus_agent.site_host import HostUnavailable
from eugene_plexus_agent.site_links import MAX_KEYS, LinkStore, key_id

from .test_site_link_page import ADA, BASE, BO, CLIENT, Owner, StubSite, windows_only

KEY_ID = re.compile(r"data-keys='([^']*)'")


@pytest.fixture
def owner(monkeypatch: pytest.MonkeyPatch) -> Owner:
    return Owner(monkeypatch)


@pytest.fixture
def browser(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, base_url=BASE, client=CLIENT) as client:
        yield client


class HeldSite(StubSite):
    """A supervisor whose site host answers `/v1/held` as a test says."""

    def __init__(self, store: LinkStore) -> None:
        super().__init__(store)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.status = 200
        self.body: Any = {"subject": "p-ada", "keys": [], "items": []}
        self.down = False

    async def held(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        self.calls.append((method, path, kwargs))
        if self.down:
            raise HostUnavailable("This machine's job site is not running.")
        if self.status == 204:
            return httpx.Response(204)
        return httpx.Response(self.status, json=self.body)


@pytest.fixture
def held_site(app: FastAPI, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> HeldSite:
    stub = HeldSite(LinkStore(tmp_path))
    app.state.site_host = stub
    monkeypatch.setattr(site_link, "account_name", lambda sid: f"PC\\{sid[-4:]}")
    return stub


def ed25519() -> tuple[str, str]:
    raw = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()[:32]


def p256() -> tuple[str, str]:
    raw = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    )
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()[:32]


def linked(site: StubSite) -> None:
    site.store.add(subject="p-ada", name="Ada", account=ADA, account_name="PC\\1001")
    site.store.add(subject="p-bo", name="Bo", account=BO, account_name="PC\\1002")


def csrf(page: str) -> str:
    found = re.search(r"data-csrf='([^']+)'", page)
    assert found, page
    return found.group(1)


def send(client: TestClient, path: str, token: str | None, body: Any) -> httpx.Response:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Eugene-Csrf"] = token
    return client.post(path, content=json.dumps(body), headers=headers)


# --- the link page's key section ----------------------------------------------------------


@windows_only
def test_a_linked_person_is_offered_a_key_and_the_page_may_run_its_script(
    browser: TestClient, held_site: HeldSite, owner: Owner
) -> None:
    linked(held_site)
    page = browser.get("/link")
    assert "id=site-key" in page.text and "/link/site-keys.js" in page.text
    policy = page.headers["content-security-policy"]
    assert "script-src 'self'" in policy and "connect-src 'self'" in policy
    assert "default-src 'none'" in policy and "frame-ancestors 'none'" in policy
    assert KEY_ID.search(page.text).group(1) == "[]"  # type: ignore[union-attr]
    script = browser.get("/link/site-keys.js")
    assert script.status_code == 200 and script.headers["content-type"].startswith(
        "text/javascript"
    )
    assert script.headers["x-content-type-options"] == "nosniff"
    # The private half is made non-extractable: WebCrypto keeps it from every script.
    assert script.text.count("false,") >= 2 and 'exportKey("raw", pair.publicKey)' in script.text


@windows_only
def test_an_unlinked_account_runs_no_script(
    browser: TestClient, held_site: HeldSite, owner: Owner
) -> None:
    page = browser.get("/link")
    assert "site-keys.js" not in page.text
    assert "script-src" not in page.headers["content-security-policy"]


@windows_only
@pytest.mark.parametrize(("make", "alg"), [(ed25519, "Ed25519"), (p256, "ES256")])
def test_a_key_made_here_is_pinned_to_this_accounts_link(
    browser: TestClient, held_site: HeldSite, owner: Owner, make: Any, alg: str
) -> None:
    linked(held_site)
    token = csrf(browser.get("/link").text)
    public, ident = make()
    browser.headers["user-agent"] = "Mozilla/5.0 Chrome/154.0 Safari/537.36"
    answer = send(browser, "/link/key", token, {"alg": alg, "publicKey": public})
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"id": ident}
    link = held_site.store.for_account(ADA)
    assert link is not None and [k["id"] for k in link.keys] == [ident]
    assert link.keys[0]["alg"] == alg and link.keys[0]["label"].startswith("Chrome on ")
    assert held_site.store.for_account(BO).keys == ()  # type: ignore[union-attr]
    assert held_site.changed == 1
    # The same key again is the key already pinned.
    assert send(browser, "/link/key", token, {"alg": alg, "publicKey": public}).json() == {
        "id": ident
    }
    assert len(held_site.store.for_account(ADA).keys) == 1  # type: ignore[union-attr]
    # The page now lists it.
    assert ident in KEY_ID.search(browser.get("/link").text).group(1)  # type: ignore[union-attr]


@windows_only
def test_pinning_needs_this_pages_token_json_and_a_real_key(
    browser: TestClient, held_site: HeldSite, owner: Owner
) -> None:
    linked(held_site)
    token = csrf(browser.get("/link").text)
    public, _ = ed25519()
    good = {"alg": "Ed25519", "publicKey": public}
    assert send(browser, "/link/key", None, good).status_code == 403
    assert send(browser, "/link/key", "guess", good).status_code == 403
    form = browser.post("/link/key", data=good, headers={"X-Eugene-Csrf": token})
    assert form.status_code == 403
    # A form can send JSON as text/plain; only JSON sent as JSON is taken.
    plain = browser.post(
        "/link/key",
        content=json.dumps(good),
        headers={"X-Eugene-Csrf": token, "Content-Type": "text/plain"},
    )
    assert plain.status_code == 403
    for bad in (
        {"alg": "RS256", "publicKey": public},
        {"alg": "Ed25519", "publicKey": "not base64!"},
        {"alg": "ES256", "publicKey": public},  # 32 bytes is not a P-256 point
        {"alg": "ES256", "publicKey": base64.b64encode(b"\x04" + b"\x01" * 64).decode()},
    ):
        assert send(browser, "/link/key", token, bad).status_code == 403, bad
    assert held_site.store.for_account(ADA).keys == ()  # type: ignore[union-attr]


@windows_only
def test_another_account_or_an_unlinked_one_pins_nothing(
    browser: TestClient, held_site: HeldSite, owner: Owner
) -> None:
    held_site.store.add(subject="p-ada", name="Ada", account=ADA, account_name="PC\\1001")
    token = csrf(browser.get("/link").text)
    public, _ = ed25519()
    owner.sid = BO  # someone else at the other end, with ada's cookie and token
    refused = send(browser, "/link/key", token, {"alg": "Ed25519", "publicKey": public})
    assert refused.status_code == 403 and "not linked" in refused.json()["detail"]
    assert held_site.store.for_account(ADA).keys == ()  # type: ignore[union-attr]


@windows_only
def test_a_person_keeps_at_most_eight_keys_and_removes_their_own(
    browser: TestClient, held_site: HeldSite, owner: Owner
) -> None:
    linked(held_site)
    made = [ed25519() for _ in range(MAX_KEYS + 1)]
    for public, _ in made[:MAX_KEYS]:
        held_site.store.add_key(ADA, "Ed25519", public, "x")
    bo_public, bo_id = made[-1]
    held_site.store.add_key(BO, "Ed25519", bo_public, "bo's")
    page = browser.get("/link")
    token = csrf(page.text)
    refused = send(browser, "/link/key", token, {"alg": "Ed25519", "publicKey": ed25519()[0]})
    assert refused.status_code == 403 and "8 keys" in refused.json()["detail"]
    first = made[0][1]
    gone = browser.post(
        "/link/key/remove", data={"csrf": token, "id": first}, follow_redirects=False
    )
    assert gone.status_code == 303
    assert first not in [k["id"] for k in held_site.store.for_account(ADA).keys]  # type: ignore[union-attr]
    # Bo's key is not ada's to remove.
    browser.post("/link/key/remove", data={"csrf": token, "id": bo_id})
    assert [k["id"] for k in held_site.store.for_account(BO).keys] == [bo_id]  # type: ignore[union-attr]
    stale = browser.post("/link/key/remove", data={"csrf": "guess", "id": made[1][1]})
    assert stale.status_code == 403


# --- approvals -------------------------------------------------------------------------------


@windows_only
def test_the_approval_page_is_for_a_linked_account(
    browser: TestClient, held_site: HeldSite, owner: Owner
) -> None:
    refused = browser.get("/link/approve")
    assert refused.status_code == 403 and "not linked" in refused.text
    linked(held_site)
    public, ident = ed25519()
    held_site.store.add_key(ADA, "Ed25519", public, None)
    page = browser.get("/link/approve")
    assert page.status_code == 200 and "id=site-approve" in page.text
    assert json.loads(KEY_ID.search(page.text).group(1).replace("&quot;", '"')) == [ident]  # type: ignore[union-attr]
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert f"{site_link.COOKIE}=" in page.headers["set-cookie"]


@windows_only
def test_the_list_is_asked_for_the_connections_own_person(
    browser: TestClient, held_site: HeldSite, owner: Owner
) -> None:
    linked(held_site)
    browser.get("/link/approve")
    key = "a" * 32
    listed = browser.get(f"/link/approve/items?key={key}&subject=p-bo")
    assert listed.status_code == 200 and listed.headers["cache-control"] == "no-store"
    (call,) = held_site.calls
    assert call == ("GET", "/v1/held", {"params": {"subject": "p-ada", "key": key}})
    assert browser.get("/link/approve/items?key=nothex").status_code == 403
    held_site.status = 404
    assert browser.get(f"/link/approve/items?key={key}").status_code == 404
    held_site.down = True
    down = browser.get("/link/approve/items")
    assert down.status_code == 503 and "not running" in down.json()["detail"]


@windows_only
def test_an_approval_is_carried_for_the_connections_person_with_this_pages_token(
    browser: TestClient, held_site: HeldSite, owner: Owner
) -> None:
    linked(held_site)
    token = csrf(browser.get("/link/approve").text)
    approval = {"envelope": "{}", "key": "b" * 32, "signature": "c2ln", "subject": "p-bo"}
    held_site.body = {"status": "done", "result": {}}
    assert send(browser, "/link/approve/items/abc123", None, approval).status_code == 403
    assert held_site.calls == []
    done = send(browser, "/link/approve/items/abc123", token, approval)
    assert done.status_code == 200 and done.json() == {"status": "done", "result": {}}
    (call,) = held_site.calls
    assert call == (
        "POST",
        "/v1/held/abc123/approve",
        {"json": {"subject": "p-ada", "envelope": "{}", "key": "b" * 32, "signature": "c2ln"}},
    )
    assert send(browser, "/link/approve/items/ABC", token, approval).status_code == 403
    held_site.status = 404
    assert send(browser, "/link/approve/items/abc123", token, approval).status_code == 404


@windows_only
def test_turning_down_is_carried_too_and_another_account_cannot(
    browser: TestClient, held_site: HeldSite, owner: Owner
) -> None:
    linked(held_site)
    token = csrf(browser.get("/link/approve").text)
    held_site.status = 204
    done = send(browser, "/link/approve/items/abc123/reject", token, {})
    assert done.status_code == 200
    assert held_site.calls[-1] == ("POST", "/v1/held/abc123/reject", {"json": {"subject": "p-ada"}})
    owner.sid = BO
    refused = send(browser, "/link/approve/items/abc123/reject", token, {})
    assert refused.status_code == 403 and len(held_site.calls) == 1


# --- the links file keeps keys ---------------------------------------------------------------


def test_keys_survive_other_links_and_a_bad_one_is_dropped(tmp_path: Any) -> None:
    store = LinkStore(tmp_path)
    store.add(subject="p-ada", name="Ada", account=ADA, account_name="PC\\ada")
    public, ident = ed25519()
    store.add_key(ADA, "Ed25519", public, "Chrome on PC")
    store.add(subject="p-bo", name="Bo", account=BO, account_name="PC\\bo")
    assert [k["id"] for k in store.for_account(ADA).keys] == [ident]  # type: ignore[union-attr]
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    raw["links"][0]["keys"].append({**raw["links"][0]["keys"][0], "id": "0" * 32})
    store.path.write_text(json.dumps(raw), encoding="utf-8")
    assert [k["id"] for k in store.for_account(ADA).keys] == [ident]  # type: ignore[union-attr]
    assert key_id(base64.b64decode(public)) == ident
    assert store.remove_key(ADA, ident) is True and store.remove_key(ADA, ident) is False
