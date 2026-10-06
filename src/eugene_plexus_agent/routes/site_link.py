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
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import secrets
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx
import jwt
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from .._http import client_for
from ..dependencies import require_control
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


def _socket_owner(client_port: int, server_port: int) -> str:
    """The SID of the process that owns the loopback connection from
    `client_port` to this agent's `server_port` (Windows)."""
    import ctypes
    from ctypes import wintypes

    import win32api
    import win32con
    import win32security

    iphlpapi = ctypes.WinDLL("iphlpapi")

    class Row(ctypes.Structure):
        _fields_ = [
            ("state", wintypes.DWORD),
            ("local_addr", wintypes.DWORD),
            ("local_port", wintypes.DWORD),
            ("remote_addr", wintypes.DWORD),
            ("remote_port", wintypes.DWORD),
            ("pid", wintypes.DWORD),
        ]

    size = wintypes.DWORD(0)
    # AF_INET (2), TCP_TABLE_OWNER_PID_CONNECTIONS (4).
    iphlpapi.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 4, 0)
    buffer = ctypes.create_string_buffer(size.value + 4096)
    size = wintypes.DWORD(len(buffer))
    if iphlpapi.GetExtendedTcpTable(buffer, ctypes.byref(size), False, 2, 4, 0) != 0:
        raise NotHere("This machine's connection table could not be read.")
    count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD))[0]
    rows = ctypes.cast(
        ctypes.addressof(buffer) + ctypes.sizeof(wintypes.DWORD), ctypes.POINTER(Row * count)
    ).contents
    loopback = int.from_bytes(bytes([127, 0, 0, 1]), "little")

    def port(value: int) -> int:
        return ((value & 0xFF) << 8) | ((value >> 8) & 0xFF)

    owners = {
        row.pid
        for row in rows
        if row.local_addr == loopback
        and row.remote_addr == loopback
        and port(row.local_port) == client_port
        and port(row.remote_port) == server_port
    }
    if len(owners) != 1:
        raise NotHere("The program that opened this page could not be found.")
    pid = owners.pop()
    try:
        process = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        token = win32security.OpenProcessToken(process, win32con.TOKEN_QUERY)
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        return str(win32security.ConvertSidToStringSid(user))
    except Exception:
        raise NotHere(
            "The account of the program that opened this page could not be read."
        ) from None


def _account(request: Request) -> str:
    """The OS account at the other end of this request's connection, or a
    refusal saying why it is not one this page serves."""
    if sys.platform != "win32":
        raise NotHere("This page runs on a Windows service install.")
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
    account = _socket_owner(int(client[1]), int(server[1]))
    if why := not_a_person(account):
        raise NotHere(f"This page was opened by {why}, which no one can be linked to.")
    return account


# --- the pages ---------------------------------------------------------------------


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font:16px/1.5 system-ui,sans-serif;max-width:36rem;margin:3rem auto;padding:0 1rem;"
        "background:#f7f7f5;color:#1d1d1b}button{font:inherit;padding:.5rem 1rem}"
        "@media(prefers-color-scheme:dark){body{background:#16181d;color:#e6e6e3}}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>{body}</body></html>",
        status_code=status,
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; "
            "form-action 'self'; frame-ancestors 'none'",
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
    supervisor = getattr(request.app.state, "site_host", None)
    return bool(supervisor is not None and supervisor.link_page_offered())


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
    if link is not None:
        body = (
            f"<p>Your account here, <b>{html.escape(name)}</b>, is linked to "
            f"<b>{html.escape(link.name or link.subject)}</b> in Eugene. Their calls to this job "
            "site run as this account, while it is signed in.</p>"
            + _form("/link/remove", "Remove this link", attempt.csrf)
        )
    else:
        body = (
            f"<p>You are signed in to this machine as <b>{html.escape(name)}</b>.</p>"
            "<p>Sign in to Eugene to link your Eugene name to this account. Your calls to this "
            "job site will then run as this account, with its permissions, while it is signed "
            "in.</p><p><a href='/link/start'>Sign in to link</a></p>"
        )
    page = _page("Link your account", body)
    page.set_cookie(
        COOKIE, key, httponly=True, samesite="strict", max_age=ATTEMPT_SECONDS, path="/link"
    )
    return page


@router.get("/link/start")
async def link_start(request: Request) -> Response:
    if not _available(request):
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
    if not _available(request):
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
    if not _available(request):
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
    if not _available(request):
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
