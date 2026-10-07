"""`/link`: a person links their Eugene sign-in to their OS account, at the machine.

`docs/design/job-sites-own-enrollment.md` §2.2, §3.2 (J27, J37). Served on a
Windows service install that hosts a Job Site, where this agent runs as
LocalSystem and is the machine's privileged starter.

**Two proofs, each from where only it can come:**

- **the account** is the one that owns the connecting socket. Only a
  loopback TCP connection is served, made straight to this agent: nothing
  relayed, nothing through the entry point, no forwarding header. The
  agent looks the connection up in the operating system's table, finds the
  process at its far end, and reads that process's token. It is never a
  name the request sends, and it is read again on every request of the
  attempt: a different account ends it.
- **the person** signs in to Eugene, through this agent's own `/oidc`, as the
  built-in loopback client `eugene-site-link` (PKCE; `state` and `nonce`
  bound to a cookie this page set). The callback trades the code and checks
  the ID token itself: RS256 against the root's JWKS, issuer, audience,
  nonce, expiry, and a person rather than Eugene's owner.

Then a confirmation names both, and only its POST, carrying that page's
CSRF token, writes the link (`site_links.py`). One link per person and one
person per account. A person may remove their own link here too.

A root cannot make a link: it needs someone at the machine signed in as that
account. Nor can the site host, which cannot write the links file.

**Keys and approvals** (J14a, `person-held-keys.md`). A linked person makes
their own key on this page: the browser generates it, will not hand its
private half to any script, and keeps it; this agent pins its public half to
their link (`/link/key`). On `/link/approve` the page lists what the site
host holds for them, in the site host's own words, and signs exactly the
text the site host gave for each change they approve; the site host checks
that signature against the pinned key. This agent only carries it: the
signature, not this agent, is the authority. Every request here is checked
from the connection as above, and every write carries the page's CSRF token.

**On a per-user install** (J14a.2: Windows, Linux, macOS) the agent runs as
the one person it serves, linked at `site join` (J38). There this page makes
and pins keys and approves held changes, and links nobody: only the account
this agent runs as is served, read from the connection the same way
(`loopback_peer.py`), and any other account on the machine is refused.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import platform
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx
import jwt
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .. import loopback_peer
from .._http import client_for
from ..dependencies import require_control
from ..site_host import HostUnavailable
from ..site_links import LinkError, LinkStore, account_name, not_a_person
from .oidc_forward import (
    FORWARDED_FOR_HEADER,
    FORWARDED_HOST_HEADER,
    FORWARDED_PROTO_HEADER,
    NODE_TOKEN_HEADER,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["sites"], include_in_schema=False)

CLIENT_ID = "eugene-site-link"
COOKIE = "ep_site_link"
ATTEMPT_SECONDS = 600
MAX_ATTEMPTS = 64
_FORWARDING = (
    "x-eugene-plexus-peer",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-real-ip",
    "forwarded",
    FORWARDED_FOR_HEADER,
    FORWARDED_HOST_HEADER,
)


class NotHere(Exception):
    """This request may not link anyone; the message says why, for the page."""


@dataclass
class Attempt:
    account: str
    state: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    verifier: str = field(default_factory=lambda: secrets.token_urlsafe(48))
    csrf: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    started: float = field(default_factory=time.perf_counter)
    person: str | None = None
    name: str | None = None


# --- who is at the other end of this connection -------------------------------------


def _account(request: Request) -> str:
    """The OS account at the other end of this request's connection, or a
    refusal saying why it is not one this page serves."""
    scope = request.scope
    client, server = scope.get("client"), scope.get("server")
    if (
        not client
        or not server
        or client[0] != "127.0.0.1"
        or server[0] != "127.0.0.1"
        or not client[1]
        or scope.get("scheme") != "http"
        or (scope.get("state") or {}).get("via_entrypoint")
        or any(name in request.headers for name in _FORWARDING)
    ):
        raise NotHere(
            "Open this page on the machine itself, at http://127.0.0.1 and this agent's port. "
            "It cannot be reached from anywhere else, or through another address."
        )
    try:
        account = loopback_peer.peer_account(int(client[1]), int(server[1]))
    except loopback_peer.PeerUnknown as exc:
        raise NotHere(str(exc)) from None
    if _per_user(request):
        own = loopback_peer.own_account()
        if account != own:
            raise NotHere(
                f"This page is for {account_name(own)} only: Eugene on this machine runs as "
                "that account and serves no one else here."
            )
    if why := not_a_person(account):
        raise NotHere(f"This page was opened by {why}, which no one can be linked to.")
    return account


# --- the pages ---------------------------------------------------------------------


def _page(title: str, body: str, status: int = 200, *, script: bool = False) -> HTMLResponse:
    scripts = "script-src 'self'; connect-src 'self'; " if script else ""
    return HTMLResponse(
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font:16px/1.5 system-ui,sans-serif;max-width:36rem;margin:3rem auto;padding:0 1rem;"
        "background:#f7f7f5;color:#1d1d1b}button{font:inherit;padding:.5rem 1rem}"
        ".held{border:1px solid #8884;border-radius:4px;padding:0 1rem 1rem;margin:1rem 0}"
        "@media(prefers-color-scheme:dark){body{background:#16181d;color:#e6e6e3}}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>{body}</body></html>",
        status_code=status,
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; "
            f"{scripts}form-action 'self'; frame-ancestors 'none'",
            "Referrer-Policy": "no-referrer",
        },
    )


def _refused(message: str, status: int = 403) -> HTMLResponse:
    return _page("This page cannot link you", f"<p>{html.escape(message)}</p>", status)


def _form(action: str, label: str, csrf: str) -> str:
    return (
        f"<form method=post action='{html.escape(action)}'>"
        f"<input type=hidden name=csrf value='{html.escape(csrf)}'>"
        f"<button type=submit>{html.escape(label)}</button></form>"
    )


class LinkPages:
    """The page's state: one attempt per browser cookie, ten minutes each."""

    def __init__(self) -> None:
        self.attempts: dict[str, Attempt] = {}

    def _sweep(self) -> None:
        now = time.perf_counter()
        for key, attempt in list(self.attempts.items()):
            if now - attempt.started > ATTEMPT_SECONDS:
                del self.attempts[key]

    def begin(self, account: str) -> tuple[str, Attempt]:
        self._sweep()
        while len(self.attempts) >= MAX_ATTEMPTS:
            self.attempts.pop(next(iter(self.attempts)))
        key = secrets.token_urlsafe(32)
        attempt = Attempt(account)
        self.attempts[key] = attempt
        return key, attempt

    def find(self, request: Request, account: str) -> tuple[str, Attempt] | None:
        self._sweep()
        key = request.cookies.get(COOKIE)
        attempt = self.attempts.get(key or "")
        if key is None or attempt is None:
            return None
        if attempt.account != account:
            # A different account at the other end ends the attempt.
            del self.attempts[key]
            return None
        return key, attempt


def _pages(request: Request) -> LinkPages:
    pages = getattr(request.app.state, "site_link_pages", None)
    if pages is None:
        pages = LinkPages()
        request.app.state.site_link_pages = pages
    return pages


def _store(request: Request) -> LinkStore | None:
    supervisor = getattr(request.app.state, "site_host", None)
    return supervisor.link_store() if supervisor is not None else None


def _issuer(request: Request) -> str:
    """The issuer this agent's `/oidc` forward presents for this request
    (`oidc_forward.py`): the console's origin behind the entry point, else
    the address the request arrived at."""
    entry = getattr(request.app.state, "entrypoint_config", None)
    if entry:
        return f"https://{urlsplit(entry.console.origin).netloc}/oidc"
    return f"http://{request.headers.get('host') or request.url.netloc}/oidc"


def _redirect_uri(request: Request) -> str:
    return f"http://127.0.0.1:{request.scope['server'][1]}/link/callback"


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


async def _root(request: Request, method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Ask the root's OIDC provider, as this agent's own `/oidc` forward would
    for this request: the same issuer, and this agent's token to the root."""
    identity = getattr(request.app.state, "node_identity", None)
    if identity is None or not identity.record.enrolled or not identity.record.control_url:
        raise NotHere("This machine has not joined an install, so there is no Eugene here.")
    root = str(identity.record.control_url).rstrip("/")
    issuer = urlsplit(_issuer(request))
    headers = dict(kwargs.pop("headers", {}))
    headers[FORWARDED_HOST_HEADER] = issuer.netloc
    headers[FORWARDED_PROTO_HEADER] = issuer.scheme
    headers[FORWARDED_FOR_HEADER] = "127.0.0.1"
    headers[NODE_TOKEN_HEADER] = request.app.state.auth_state.trust.agent_token("control")
    client = getattr(request.app.state, "site_link_client", None)
    if client is None or client.is_closed:
        client = client_for(root, timeout=20.0, follow_redirects=False, trust_env=False)
        request.app.state.site_link_client = client
    try:
        return await client.request(method, f"{root}{path}", headers=headers, **kwargs)
    except httpx.HTTPError:
        raise NotHere("The control root did not answer. Try again once it is back.") from None


async def _person(request: Request, attempt: Attempt, code: str) -> tuple[str, str]:
    """Trade the code and check the ID token; the person's id and name."""
    answer = await _root(
        request,
        "POST",
        "/oidc/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": _redirect_uri(request),
            "client_id": CLIENT_ID,
            "code_verifier": attempt.verifier,
        },
    )
    if answer.status_code != 200:
        raise NotHere("Eugene did not accept that sign-in. Start again.")
    try:
        id_token = str(answer.json()["id_token"])
        keys = (await _root(request, "GET", "/oidc/jwks")).json()
        kid = jwt.get_unverified_header(id_token).get("kid")
        jwk = next(k for k in keys.get("keys", []) if k.get("kid") == kid)
        key = jwt.PyJWK(jwk, algorithm="RS256")
        claims = jwt.decode(
            id_token,
            key.key,
            algorithms=["RS256"],
            audience=CLIENT_ID,
            issuer=_issuer(request),
            options={"require": ["iss", "sub", "aud", "exp", "iat"]},
        )
    except (KeyError, ValueError, StopIteration, jwt.PyJWTError):
        raise NotHere("Eugene's answer could not be checked. Start again.") from None
    if claims.get("nonce") != attempt.nonce:
        raise NotHere("This sign-in was not the one this page started. Start again.")
    if claims.get("eugene_role") != "member" or claims.get("sub") == "operator":
        raise NotHere(
            "Eugene's owner is not a person on a job site. Sign in as yourself, with your own "
            "name and password."
        )
    name = str(claims.get("name") or claims.get("preferred_username") or claims["sub"])
    return str(claims["sub"]), name


def _available(request: Request) -> bool:
    """Keys and approvals: a Windows service install, or a per-user install
    where this agent's own account is the one person served (J14a.2)."""
    supervisor = getattr(request.app.state, "site_host", None)
    return bool(supervisor is not None and supervisor.key_page_offered())


def _linking(request: Request) -> bool:
    """Linking people: a Windows service install only (J36, J38)."""
    supervisor = getattr(request.app.state, "site_host", None)
    return bool(supervisor is not None and supervisor.link_page_offered())


def _per_user(request: Request) -> bool:
    supervisor = getattr(request.app.state, "site_host", None)
    return bool(supervisor is not None and supervisor.mode() == "user")


@router.get("/link")
async def link_page(request: Request) -> Response:
    if not _available(request):
        return Response(status_code=404)
    try:
        account = _account(request)
    except NotHere as exc:
        return _refused(str(exc))
    store = _store(request)
    if store is None:
        return Response(status_code=404)
    name = account_name(account)
    link = store.for_account(account)
    key, attempt = _pages(request).begin(account)
    if _per_user(request):
        # Linked at `site join`, and nobody else is served here (J38).
        if link is None:
            return _refused(
                f"This job site is not linked to {name}. Run `site join` again from this "
                "account to make it yours."
            )
        body = (
            f"<p>This job site is <b>{html.escape(link.name or link.subject)}</b>'s, and its "
            f"calls run as your account here, <b>{html.escape(name)}</b>.</p>"
            + _keys_section(link.keys, attempt.csrf)
        )
    elif link is not None:
        body = (
            f"<p>Your account here, <b>{html.escape(name)}</b>, is linked to "
            f"<b>{html.escape(link.name or link.subject)}</b> in Eugene. Their calls to this job "
            "site run as this account, while it is signed in.</p>"
            + _keys_section(link.keys, attempt.csrf)
            + "<h2>This link</h2>"
            + _form("/link/remove", "Remove this link", attempt.csrf)
        )
    else:
        body = (
            f"<p>You are signed in to this machine as <b>{html.escape(name)}</b>.</p>"
            "<p>Sign in to Eugene to link your Eugene name to this account. Your calls to this "
            "job site will then run as this account, with its permissions, while it is signed "
            "in.</p><p><a href='/link/start'>Sign in to link</a></p>"
        )
    title = "Your key" if _per_user(request) else "Link your account"
    page = _page(title, body, script=link is not None)
    page.set_cookie(
        COOKIE, key, httponly=True, samesite="strict", max_age=ATTEMPT_SECONDS, path="/link"
    )
    return page


@router.get("/link/start")
async def link_start(request: Request) -> Response:
    if not _linking(request):
        return Response(status_code=404)
    try:
        account = _account(request)
    except NotHere as exc:
        return _refused(str(exc))
    found = _pages(request).find(request, account)
    if found is None:
        return RedirectResponse("/link", status_code=303)
    _key, attempt = found
    query = urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": _redirect_uri(request),
            "scope": "openid profile",
            "state": attempt.state,
            "nonce": attempt.nonce,
            "code_challenge": _challenge(attempt.verifier),
            "code_challenge_method": "S256",
        }
    )
    return RedirectResponse(f"/oidc/authorize?{query}", status_code=303)


@router.get("/link/callback")
async def link_callback(request: Request) -> Response:
    if not _linking(request):
        return Response(status_code=404)
    try:
        account = _account(request)
        found = _pages(request).find(request, account)
        if found is None:
            raise NotHere("This sign-in was not started on this page by this account. Start again.")
        _key, attempt = found
        if request.query_params.get("state") != attempt.state:
            raise NotHere("This sign-in was not the one this page started. Start again.")
        if request.query_params.get("error"):
            raise NotHere("The sign-in did not finish. Start again.")
        code = request.query_params.get("code")
        if not code:
            raise NotHere("The sign-in did not finish. Start again.")
        attempt.person, attempt.name = await _person(request, attempt, code)
    except NotHere as exc:
        return _refused(str(exc))
    body = (
        f"<p>Link Eugene person <b>{html.escape(attempt.name or '')}</b> to this machine's "
        f"account <b>{html.escape(account_name(account))}</b>?</p>"
        "<p>Calls they make to this job site will run as this account, with everything it can "
        "reach, while it is signed in. Only link yourself.</p>"
        + _form("/link/confirm", "Link", attempt.csrf)
    )
    return _page("Confirm the link", body)


async def _posted_csrf(request: Request) -> str | None:
    try:
        form = await request.form()
    except Exception:
        return None
    value = form.get("csrf")
    return value if isinstance(value, str) else None


@router.post("/link/confirm")
async def link_confirm(request: Request) -> Response:
    if not _linking(request):
        return Response(status_code=404)
    try:
        account = _account(request)
        found = _pages(request).find(request, account)
        if found is None:
            raise NotHere("This link was not started on this page by this account. Start again.")
        key, attempt = found
        csrf = await _posted_csrf(request)
        if csrf is None or not secrets.compare_digest(csrf, attempt.csrf) or attempt.person is None:
            raise NotHere("This confirmation is not from this page. Start again.")
        store = _store(request)
        supervisor = request.app.state.site_host
        if store is None:
            raise NotHere("This machine is not a job site any more.")
        try:
            store.add(
                subject=attempt.person,
                name=attempt.name,
                account=account,
                account_name=account_name(account),
                never=supervisor.never_linked(),
            )
        except LinkError as exc:
            raise NotHere(str(exc)) from None
        _pages(request).attempts.pop(key, None)
        supervisor.links_changed()
    except NotHere as exc:
        return _refused(str(exc))
    return _page(
        "Linked",
        f"<p>Done. <b>{html.escape(attempt.name or '')}</b>'s calls to this job site now run as "
        f"<b>{html.escape(account_name(account))}</b> while it is signed in. You can close this "
        "page.</p>",
    )


@router.post("/link/remove")
async def link_remove(request: Request) -> Response:
    if not _linking(request):
        return Response(status_code=404)
    try:
        account = _account(request)
        found = _pages(request).find(request, account)
        csrf = await _posted_csrf(request)
        if found is None or csrf is None or not secrets.compare_digest(csrf, found[1].csrf):
            raise NotHere("This request is not from this page. Open it again.")
        store = _store(request)
        if store is None:
            raise NotHere("This machine is not a job site any more.")
        link = store.for_account(account)
        if link is not None:
            store.remove(link.subject)
            request.app.state.site_host.links_changed()
        _pages(request).attempts.pop(found[0], None)
    except NotHere as exc:
        return _refused(str(exc))
    return _page("Link removed", "<p>This account is no longer linked to anyone here.</p>")


# --- keys and approvals (J14a) -----------------------------------------------------------

KEY_SCRIPT = "/link/site-keys.js"
_ALGS = {"Ed25519", "ES256"}
_HELD_ID = re.compile(r"^[a-z0-9]{1,32}$")
_KEY_ID = re.compile(r"^[a-f0-9]{32}$")


def _fingerprint(key_id: str) -> str:
    return " ".join(key_id[i : i + 4] for i in range(0, 16, 4))


def _keys_section(keys: tuple[dict[str, Any], ...], csrf: str) -> str:
    rows = "".join(
        f"<li><b>{html.escape(_fingerprint(str(k['id'])))}</b>"
        f" {html.escape(str(k.get('label') or ''))}, added {html.escape(str(k['addedAt'])[:10])} "
        f"<form method=post action='/link/key/remove' style='display:inline'>"
        f"<input type=hidden name=csrf value='{html.escape(csrf)}'>"
        f"<input type=hidden name=id value='{html.escape(str(k['id']))}'>"
        "<button type=submit>Remove</button></form></li>"
        for k in keys
    )
    pinned = html.escape(json.dumps([str(k["id"]) for k in keys]))
    return (
        "<h2>Your key</h2>"
        "<p>Changes to what this machine allows wait until you approve them here, with your own "
        "key. The key is made in this browser and never leaves this machine. Eugene never sees "
        "it.</p>"
        + (f"<ul>{rows}</ul>" if rows else "")
        + f"<div id=site-key data-csrf='{html.escape(csrf)}' data-keys='{pinned}'>"
        "<p data-state>Checking this browser…</p>"
        "<button type=button data-make hidden>Make a key in this browser</button></div>"
        "<p><a href='/link/approve'>Changes waiting for your approval</a></p>"
        f"<script src='{KEY_SCRIPT}'></script>"
    )


def _attempt(request: Request, account: str) -> tuple[str, Attempt, bool]:
    """This browser's attempt, or a new one: the key, it, and whether it is new."""
    found = _pages(request).find(request, account)
    if found is not None:
        return found[0], found[1], False
    key, attempt = _pages(request).begin(account)
    return key, attempt, True


def _linked(request: Request) -> tuple[str, Any]:
    """The account at the other end and its link, or `NotHere` saying why."""
    account = _account(request)
    store = _store(request)
    if store is None:
        raise NotHere("This machine is not a job site any more.")
    link = store.for_account(account)
    if link is None:
        if _per_user(request):
            raise NotHere(
                f"This job site is not linked to {account_name(account)}. Run `site join` again "
                "from this account to make it yours."
            )
        raise NotHere(
            f"{account_name(account)} is not linked to anyone here. Link it on the link page first."
        )
    return account, link


def _json(value: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(value, status_code=status, headers={"Cache-Control": "no-store"})


def _json_refused(message: str, status: int = 403) -> JSONResponse:
    return _json({"detail": message}, status)


async def _csrf_json(request: Request, account: str) -> tuple[str, dict[str, Any]]:
    """The attempt's key and the posted JSON, if the page's CSRF token came with it."""
    found = _pages(request).find(request, account)
    given = request.headers.get("x-eugene-csrf") or ""
    if found is None or not secrets.compare_digest(given, found[1].csrf):
        raise NotHere("This request is not from this page. Open it again.")
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise NotHere("This request is not from this page. Open it again.")
    try:
        value = await request.json()
    except ValueError:
        value = None
    if not isinstance(value, dict):
        raise NotHere("This request could not be read.")
    return found[0], value


def _browser(request: Request) -> str:
    agent = request.headers.get("user-agent", "")
    for marker, name in (("Edg/", "Edge"), ("Firefox/", "Firefox"), ("Chrome/", "Chrome")):
        if marker in agent:
            return name
    return "A browser"


@router.get(KEY_SCRIPT)
async def site_keys_script(request: Request) -> Response:
    if not _available(request):
        return Response(status_code=404)
    from importlib.resources import files

    script = (files("eugene_plexus_agent") / "static_site_keys.js").read_text(encoding="utf-8")
    return Response(
        script,
        media_type="text/javascript",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.post("/link/key")
async def link_key(request: Request) -> Response:
    """Pin the public half of a key this browser just made, to this account's link."""
    if not _available(request):
        return Response(status_code=404)
    try:
        account, _link = _linked(request)
        _key, value = await _csrf_json(request, account)
        alg, public = value.get("alg"), value.get("publicKey")
        if alg not in _ALGS or not isinstance(public, str) or len(public) > 128:
            raise NotHere("That is not a key this page made.")
        store = _store(request)
        assert store is not None
        label = f"{_browser(request)} on {platform.node() or 'this machine'}"
        try:
            pinned = store.add_key(account, alg, public, label)
        except LinkError as exc:
            raise NotHere(str(exc)) from None
        request.app.state.site_host.links_changed()
    except NotHere as exc:
        return _json_refused(str(exc))
    return _json({"id": pinned["id"]})


@router.post("/link/key/remove")
async def link_key_remove(request: Request) -> Response:
    """A key goes, at the machine (J45). What it approved stays."""
    if not _available(request):
        return Response(status_code=404)
    try:
        account, _link = _linked(request)
        found = _pages(request).find(request, account)
        try:
            form = await request.form()
        except Exception:
            form = None
        csrf = form.get("csrf") if form is not None else None
        ident = form.get("id") if form is not None else None
        if (
            found is None
            or not isinstance(csrf, str)
            or not secrets.compare_digest(csrf, found[1].csrf)
        ):
            raise NotHere("This request is not from this page. Open it again.")
        if not isinstance(ident, str) or not _KEY_ID.fullmatch(ident):
            raise NotHere("That key is not one of yours here.")
        store = _store(request)
        assert store is not None
        if store.remove_key(account, ident):
            request.app.state.site_host.links_changed()
    except NotHere as exc:
        return _refused(str(exc))
    return RedirectResponse("/link", status_code=303)


@router.get("/link/approve")
async def approve_page(request: Request) -> Response:
    if not _available(request):
        return Response(status_code=404)
    try:
        account, link = _linked(request)
    except NotHere as exc:
        return _refused(str(exc))
    key, attempt, new = _attempt(request, account)
    pinned = html.escape(json.dumps([str(k["id"]) for k in link.keys]))
    body = (
        f"<p>Changes to this job site for <b>{html.escape(link.name or link.subject)}</b> "
        "wait here until you approve them with your key. Nothing changes until you do.</p>"
        f"<div id=site-approve data-csrf='{html.escape(attempt.csrf)}' data-keys='{pinned}'>"
        "<p data-state>Checking this browser…</p><div data-items></div></div>"
        "<p><a href='/link'>Your link and keys</a></p>"
        f"<script src='{KEY_SCRIPT}'></script>"
    )
    page = _page("Approve changes", body, script=True)
    if new:
        page.set_cookie(
            COOKIE, key, httponly=True, samesite="strict", max_age=ATTEMPT_SECONDS, path="/link"
        )
    return page


@router.get("/link/approve/items")
async def approve_items(request: Request) -> Response:
    if not _available(request):
        return Response(status_code=404)
    try:
        account, link = _linked(request)
        if _pages(request).find(request, account) is None:
            raise NotHere("Open the approval page again.")
        key = request.query_params.get("key")
        if key is not None and not _KEY_ID.fullmatch(key):
            raise NotHere("That key is not one of yours here.")
        params = {"subject": link.subject, **({"key": key} if key else {})}
        answer = await request.app.state.site_host.held("GET", "/v1/held", params=params)
    except NotHere as exc:
        return _json_refused(str(exc))
    except HostUnavailable as exc:
        return _json_refused(str(exc), 503)
    if answer.status_code == 404:
        return _json_refused("That key is not one of yours here. Make one on the link page.", 404)
    if answer.status_code != 200:
        return _json_refused("This job site did not answer. Try again in a moment.", 503)
    return _json(answer.json())


@router.post("/link/approve/items/{ident}")
async def approve_item(request: Request, ident: str) -> Response:
    if not _available(request):
        return Response(status_code=404)
    try:
        if not _HELD_ID.fullmatch(ident):
            raise NotHere("Nothing is waiting under that name.")
        account, link = _linked(request)
        _key, value = await _csrf_json(request, account)
        body = {
            "subject": link.subject,
            "envelope": value.get("envelope"),
            "key": value.get("key"),
            "signature": value.get("signature"),
        }
        answer = await request.app.state.site_host.held(
            "POST", f"/v1/held/{ident}/approve", json=body
        )
    except NotHere as exc:
        return _json_refused(str(exc))
    except HostUnavailable as exc:
        return _json_refused(str(exc), 503)
    if answer.status_code == 404:
        return _json_refused("That is no longer waiting. Open the page again.", 404)
    if answer.status_code == 422:
        return _json_refused("That approval could not be read. Open the page again.", 422)
    if answer.status_code != 200:
        return _json_refused("This job site did not answer. Try again in a moment.", 503)
    return _json(answer.json())


@router.post("/link/approve/items/{ident}/reject")
async def reject_item(request: Request, ident: str) -> Response:
    if not _available(request):
        return Response(status_code=404)
    try:
        if not _HELD_ID.fullmatch(ident):
            raise NotHere("Nothing is waiting under that name.")
        account, link = _linked(request)
        await _csrf_json(request, account)
        answer = await request.app.state.site_host.held(
            "POST", f"/v1/held/{ident}/reject", json={"subject": link.subject}
        )
    except NotHere as exc:
        return _json_refused(str(exc))
    except HostUnavailable as exc:
        return _json_refused(str(exc), 503)
    if answer.status_code == 404:
        return _json_refused("That is no longer waiting. Open the page again.", 404)
    if answer.status_code != 204:
        return _json_refused("This job site did not answer. Try again in a moment.", 503)
    return _json({"status": "done"})


def json_problem(status: int, title: str, detail: str) -> Response:
    return Response(
        json.dumps({"type": "about:blank", "title": title, "status": status, "detail": detail}),
        status_code=status,
        media_type="application/problem+json",
    )


api = APIRouter(tags=["sites"])


@api.delete("/v1/site/links/{subject}", status_code=204, dependencies=[Depends(require_control)])
async def remove_site_link(request: Request, subject: str) -> Response:
    """The root removes a person's link here, when they or their site's owner
    did so in Workbench. Removing a link only takes access away; the root can
    never make one (§3.2)."""
    supervisor = getattr(request.app.state, "site_host", None)
    if supervisor is not None and supervisor.mode() == "root":
        return json_problem(
            409,
            "Linked at the machine",
            "This machine links people with the elevated one-liner, as root (J36). Remove the "
            "link there: run the installer again with --site-unlink.",
        )
    if supervisor is None:
        return Response(status_code=204)
    store = supervisor.link_store()
    if store is not None and store.remove(subject) is not None:
        supervisor.links_changed()
    return Response(status_code=204)
