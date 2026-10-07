"""J14a.3: a passkey from Workbench, paired with a code shown on the loopback
page (`person-held-keys.md` §4.2, §12.5).

This page shows the code the site host makes, lists the owner's passkeys and
removes one. The site host's `/v1/passkeys` API is a stub that records what it
was sent; what it does with a code and a MAC is the site host's own tests'."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent.routes import site_link
from eugene_plexus_agent.site_host import HostUnavailable
from eugene_plexus_agent.site_links import LinkStore

from .test_site_keys_page import ME, SOMEONE_ELSE
from .test_site_link_page import BASE, CLIENT, Owner, StubSite

PASSKEY = {
    "id": "c" * 32,
    "credentialId": "Y3JlZA",
    "alg": -7,
    "rpId": "workbench.example",
    "label": "Chrome on laptop",
    "addedAt": "2026-10-07T12:00:00+00:00",
}


class PasskeySite(StubSite):
    """A supervisor whose site host answers each call as a test says."""

    def __init__(self, store: LinkStore) -> None:
        super().__init__(store)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.answers: dict[tuple[str, str], tuple[int, Any]] = {
            ("GET", "/v1/passkeys"): (
                200,
                {"subject": "p-ada", "passkeys": [PASSKEY], "codeExpiresAt": None},
            ),
            ("POST", "/v1/passkeys/code"): (
                200,
                {"subject": "p-ada", "code": "ABCDE-FGHJK", "expiresAt": "2026-10-07T12:10:00Z"},
            ),
        }
        self.down = False

    async def held(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        self.calls.append((method, path, kwargs))
        if self.down:
            raise HostUnavailable("This machine's job site is not running.")
        status, body = self.answers.get((method, path), (204, None))
        if body is None:
            return httpx.Response(status)
        return httpx.Response(status, json=body)


@pytest.fixture
def owner(monkeypatch: pytest.MonkeyPatch) -> Owner:
    return Owner(monkeypatch)


@pytest.fixture
def browser(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, base_url=BASE, client=CLIENT) as client:
        yield client


@pytest.fixture
def site(app: FastAPI, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, owner: Owner) -> PasskeySite:
    """A per-user install (every platform), its owner linked to this account."""
    stub = PasskeySite(LinkStore(tmp_path))
    stub._mode = "user"
    owner.sid = owner.own = ME
    stub.store.add(subject="p-ada", name="Ada", account=ME, account_name="PC\\ada")
    app.state.site_host = stub
    monkeypatch.setattr(site_link, "account_name", lambda sid: f"PC\\{sid[-4:]}")
    return stub


def csrf(page: str) -> str:
    found = re.search(r"(?:name=csrf value|data-csrf)='([^']+)'", page)
    assert found, page
    return found.group(1)


def test_the_page_lists_the_owners_passkeys_and_offers_a_code(
    browser: TestClient, site: PasskeySite
) -> None:
    page = browser.get("/link")
    assert page.status_code == 200, page.text
    assert "A passkey from Workbench" in page.text and "workbench.example" in page.text
    assert "cccc cccc cccc cccc" in page.text and "/link/passkey/code" in page.text
    assert site.calls[-1] == ("GET", "/v1/passkeys", {"params": {"subject": "p-ada"}})


def test_a_code_is_shown_here_for_the_connections_own_person(
    browser: TestClient, site: PasskeySite
) -> None:
    token = csrf(browser.get("/link").text)
    shown = browser.post("/link/passkey/code", data={"csrf": token})
    assert shown.status_code == 200, shown.text
    assert "ABCDE-FGHJK" in shown.text and "never sees it" in shown.text
    assert site.calls[-1] == ("POST", "/v1/passkeys/code", {"json": {"subject": "p-ada"}})
    assert shown.headers["cache-control"].startswith("no-store")


def test_a_code_needs_this_pages_token(browser: TestClient, site: PasskeySite) -> None:
    browser.get("/link")
    for data in ({}, {"csrf": "guess"}):
        refused = browser.post("/link/passkey/code", data=data)
        assert refused.status_code == 403 and "not from this page" in refused.text
    assert ("POST", "/v1/passkeys/code") not in [c[:2] for c in site.calls]


def test_another_account_gets_no_code(browser: TestClient, site: PasskeySite, owner: Owner) -> None:
    token = csrf(browser.get("/link").text)
    owner.sid = SOMEONE_ELSE
    refused = browser.post("/link/passkey/code", data={"csrf": token})
    assert refused.status_code == 403
    assert ("POST", "/v1/passkeys/code") not in [c[:2] for c in site.calls]


def test_the_site_hosts_refusal_says_only_the_owner_pairs(
    browser: TestClient, site: PasskeySite
) -> None:
    site.answers[("POST", "/v1/passkeys/code")] = (404, {})
    token = csrf(browser.get("/link").text)
    refused = browser.post("/link/passkey/code", data={"csrf": token})
    assert refused.status_code == 403 and "Only this machine&#x27;s owner" in refused.text


def test_a_passkey_is_removed_here_with_this_pages_token(
    browser: TestClient, site: PasskeySite
) -> None:
    token = csrf(browser.get("/link").text)
    assert (
        browser.post(
            "/link/passkey/remove", data={"csrf": "guess", "id": PASSKEY["id"]}
        ).status_code
        == 403
    )
    bad = browser.post("/link/passkey/remove", data={"csrf": token, "id": "../x"})
    assert bad.status_code == 403
    gone = browser.post(
        "/link/passkey/remove", data={"csrf": token, "id": PASSKEY["id"]}, follow_redirects=False
    )
    assert gone.status_code == 303 and gone.headers["location"] == "/link"
    assert site.calls[-1] == (
        "DELETE",
        f"/v1/passkeys/{PASSKEY['id']}",
        {"params": {"subject": "p-ada"}},
    )


def test_the_page_still_works_when_the_site_host_cannot_say(
    browser: TestClient, site: PasskeySite
) -> None:
    site.down = True
    page = browser.get("/link")
    assert page.status_code == 200 and "id=site-key" in page.text
    assert "A passkey from Workbench" not in page.text
    token = csrf(page.text)
    refused = browser.post("/link/passkey/code", data={"csrf": token})
    assert refused.status_code == 503 and "not running" in refused.text
