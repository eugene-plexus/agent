"""Answer only to names this agent could really be opened by: DNS rebinding.

A web page on `evil.example.com` can answer its own DNS with this
machine's address a few seconds after it loads (a short TTL, then a
second answer), and from then on the browser treats this agent as that
page's origin. Same-origin means it can read every response -- the
config trio, the topology, the unauthenticated proxy to every component
-- and, the sharp end, `POST /v1/auth/initialize`, which is first-come
until the wizard has run: the page could set the passphrase of an
install whose owner has not opened it yet. Every route here answered
whatever `Host` it was sent.

**The one thing a rebinding page cannot change is the name in its own
address bar**, so that is what is checked. A browser always sends it as
`Host`, and a page cannot set that header on a fetch.

**The allowed set is generous on purpose** (`easy-default-expert-
override`). It is every way a person really opens this on a home
network or a tailnet, so nobody following our own instructions meets
the refusal:

* any IP literal -- the install instructions and the Reach card both
  hand out an address, and an attacker's page cannot be *at* one;
* `localhost` and anything under `.localhost`, which browsers resolve
  to loopback themselves;
* a dotless name (`tower`, `amish-station`): LAN resolution, and a
  public page cannot own one;
* names under `.local` (mDNS), `.lan`, `.home.arpa` (RFC 8375),
  `.internal` and `.ts.net` (every tailnet's MagicDNS);
* this machine's own name and FQDN, and the host of this node's
  advertise address -- the name other nodes reach it by is its own by
  definition;
* whatever the operator lists in `allowedHosts`, which is where a
  reverse proxy or a custom domain goes. `*` turns the check off.

A missing `Host` is allowed: HTTP/1.0 tools on this machine send none,
a browser always does, so refusing it protects against nothing.

**Pure ASGI, never `BaseHTTPMiddleware`**, for the reason `off_host.py`
gives: this agent proxies token streams and a buffering middleware
would hold them. This one reads the scope and either answers or passes
the request through untouched.

**The field is read on every request** -- a lock and a dict lookup --
so saving it on Config takes effect on the next page load, with no
restart. The FQDN is the one slow input (`getfqdn` can wait on DNS), so
it is learned once, in a thread, at startup; until it arrives the
machine's plain host name stands in.
"""

from __future__ import annotations

import contextlib
import html
import ipaddress
import json
import logging
import re
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from . import node_identity
from ._generated.common_models import Problem

# The same list the UI mount uses to decide what is API, so the two
# cannot disagree about which refusal a path gets.
from .ui_assets import _API_PREFIXES

log = logging.getLogger(__name__)

__all__ = [
    "ALLOWED_SUFFIXES",
    "HostAllowlistMiddleware",
    "HostPolicy",
    "host_of",
    "install",
    "is_allowed",
    "learn_fqdn",
    "parse_allowed_hosts",
    "policy_for",
    "refusal_page",
    "start_learning_fqdn",
]

ALLOWED_SUFFIXES = (".localhost", ".local", ".lan", ".home.arpa", ".internal", ".ts.net")
"""Names under these are private by construction: no public page can hold one."""

_ANY = "*"
_SEPARATORS = re.compile(r"[,\s]+")

# Learned once, off the event loop: `socket.getfqdn()` is a reverse
# lookup that can wait on DNS for seconds on a box whose resolver is
# broken, and nothing about it changes while the process runs.
_fqdn: str | None = None
_learner: threading.Thread | None = None
_learner_lock = threading.Lock()


@dataclass(frozen=True)
class HostPolicy:
    """What this request may be addressed to, beyond the fixed rules."""

    names: frozenset[str] = field(default_factory=frozenset)
    allow_any: bool = False


def host_of(value: str) -> str:
    """The host a `Host` header names: no port, no brackets, lower case.

    A trailing dot is the same name (`localhost.` is `localhost`), and
    leaving it on would let `evil.example.com.` walk round an entry for
    `evil.example.com` -- or, the other way, refuse a name the operator
    listed.
    """
    value = value.strip()
    if value.startswith("["):
        end = value.find("]")
        host = value[1:end] if end != -1 else value[1:]
    elif value.count(":") == 1:
        host = value.split(":", 1)[0]
    else:
        # A bare name, or an IPv6 literal someone sent without brackets.
        host = value
    return host.rstrip(".").lower()


def parse_allowed_hosts(value: Any) -> tuple[frozenset[str], bool]:
    """The operator's `allowedHosts`: names, and whether `*` was among them.

    A comma- or space-separated string, because the config trio has no
    list-of-names type and a new `ConfigValueType` member is a contract
    change. Forgiving about what a person pastes: a URL means its host,
    a port is dropped, case does not matter.
    """
    if not isinstance(value, str):
        return frozenset(), False
    names: set[str] = set()
    allow_any = False
    for raw in _SEPARATORS.split(value):
        entry = raw.strip()
        if not entry:
            continue
        if entry == _ANY:
            allow_any = True
            continue
        if "://" in entry:
            entry = urlsplit(entry).netloc or entry
        entry = entry.split("/", 1)[0]
        host = host_of(entry)
        if host:
            names.add(host)
    return frozenset(names), allow_any


def learn_fqdn() -> None:
    """Remember this machine's FQDN. Blocking; see `start_learning_fqdn`."""
    global _fqdn
    try:
        _fqdn = host_of(socket.getfqdn())
    except OSError as exc:  # pragma: no cover - defensive
        log.debug("could not learn this machine's FQDN: %s", exc)


def start_learning_fqdn() -> threading.Thread:
    """Learn the FQDN in a daemon thread, once per process.

    A thread rather than `asyncio.to_thread`: nothing awaits the answer,
    and a lookup stuck on a broken resolver must not hold up the
    executor's shutdown when the agent stops.
    """
    global _learner
    with _learner_lock:
        if _learner is None:
            _learner = threading.Thread(target=learn_fqdn, name="learn-fqdn", daemon=True)
            _learner.start()
        return _learner


def _machine_names() -> set[str]:
    names: set[str] = set()
    with contextlib.suppress(OSError):
        names.add(host_of(socket.gethostname()))
    if _fqdn:
        names.add(_fqdn)
    return names


def is_allowed(host: str, policy: HostPolicy | None) -> bool:
    """May a request addressed to `host` (already `host_of`'d) be answered?"""
    if not host:
        return True
    if policy is not None and policy.allow_any:
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if host == "localhost" or "." not in host or host.endswith(ALLOWED_SUFFIXES):
        return True
    if policy is not None and host in policy.names:
        return True
    return host in _machine_names()


def _message(host: str) -> str:
    return (
        f"Eugene Plexus was opened as {host!r}, a name this machine has not been told is its "
        "own, so it did not answer. Open it by its IP address (for example "
        "http://192.168.1.20:8079) or as http://localhost:8079 on this machine. If you use this "
        "name on purpose -- through a reverse proxy or your own domain -- add it under Config "
        "-> Agent -> Allowed host names. Names are checked because a web page can point a name "
        "it controls at this machine and then use Eugene as if it were Eugene's own page."
    )


def refusal_page(host: str) -> str:
    """The page a browser shows. The name is the attacker's to write, so it
    is escaped: this must not become a reflected-script page."""
    shown = html.escape(host)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eugene Plexus: address not recognised</title>
<style>
  body {{ font: 15px/1.6 system-ui, sans-serif; margin: 0; padding: 3rem 1.5rem;
          background: #0f1115; color: #e6e8ee; }}
  main {{ max-width: 34rem; margin: 0 auto; }}
  h1 {{ font-size: 1.25rem; margin: 0 0 1rem; }}
  p {{ color: #aeb4c2; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
  .why {{ color: #7f8797; font-size: 13px; }}
</style>
</head>
<body>
<main>
  <h1>This address is not one Eugene Plexus recognises.</h1>
  <p>You opened it as <code>{shown}</code>. Open it by its IP address instead (for example
  <code>http://192.168.1.20:8079</code>), or as <code>http://localhost:8079</code> on this
  machine.</p>
  <p>If you use this name on purpose, through a reverse proxy or your own domain, add it
  under <strong>Config &rarr; Agent &rarr; Allowed host names</strong>.</p>
  <p class="why">Names are checked because a web page can point a name it controls at this
  machine and then use Eugene as if it were Eugene's own page.</p>
</main>
</body>
</html>
"""


def _header(scope: dict[str, Any], name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return bytes(value).decode("latin-1")
    return None


class HostAllowlistMiddleware:
    """Refuse a request addressed to a name this agent was not given."""

    def __init__(self, app: Any, *, policy: Callable[[], HostPolicy | None]) -> None:
        self.app = app
        self.policy = policy

    def _policy(self) -> HostPolicy | None:
        """The configured part, or only the fixed rules if it cannot be read.

        This runs on every request the agent serves, so it must not be
        able to raise: a config that fails to read here would otherwise
        be an agent that answers nothing, which is the one outcome
        `degraded-mode-required` forbids. The fixed rules still admit
        every IP address and `localhost`, so the browser that can repair
        the config can still reach it.
        """
        try:
            return self.policy()
        except Exception:
            log.warning(
                "could not read allowedHosts; applying the built-in names only", exc_info=True
            )
            return None

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        kind = scope.get("type")
        if kind in ("http", "websocket"):
            raw = _header(scope, b"host")
            if raw is not None:
                host = host_of(raw)
                if not is_allowed(host, self._policy()):
                    log.warning(
                        "refused a request addressed to %r (%s %s): not a name this agent "
                        "answers to; add it to allowedHosts if it should be",
                        host,
                        scope.get("method", kind),
                        scope.get("path", ""),
                    )
                    if kind == "websocket":
                        await send({"type": "websocket.close", "code": 1008})
                        return
                    await _refuse(scope, send, host)
                    return
        await self.app(scope, receive, send)


async def _refuse(scope: dict[str, Any], send: Any, host: str) -> None:
    path = str(scope.get("path", ""))
    if path.startswith(_API_PREFIXES):
        # The same envelope every other refusal here has -- FastAPI's
        # `{"detail": <Problem>}` -- so the UI's one unwrapper reads it.
        problem = Problem(
            type="https://github.com/eugene-plexus/agent#unrecognised-host-name",
            title="Unrecognised host name",
            status=403,
            detail=_message(host),
            component="agent",
        ).model_dump(exclude_none=True)
        body = json.dumps({"detail": problem}).encode("utf-8")
        content_type = b"application/json"
    else:
        body = refusal_page(host).encode("utf-8")
        content_type = b"text/html; charset=utf-8"
    await send(
        {
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", content_type),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def policy_for(app: Any) -> HostPolicy | None:
    """This request's policy, from the agent's live config and identity.

    None before the lifespan has built the state, which leaves only the
    fixed rules -- the same set a fresh install with nothing configured
    gets.
    """
    state = getattr(app.state, "agent_state", None)
    if state is None:
        return None
    names, allow_any = parse_allowed_hosts(state.get_config("allowedHosts"))
    identity = getattr(app.state, "node_identity", None)
    record = getattr(identity, "record", None)
    advertise = node_identity.effective_advertise_url(
        state.get_config("advertiseUrl"), getattr(record, "advertise_url", None)
    )
    advertise_host = node_identity.advertise_host(advertise)
    if advertise_host:
        names = names | {host_of(advertise_host)}
    return HostPolicy(names=names, allow_any=allow_any)


def install(app: Any) -> None:
    """Wrap `app` so every request is checked against its own names."""
    app.add_middleware(HostAllowlistMiddleware, policy=lambda: policy_for(app))
