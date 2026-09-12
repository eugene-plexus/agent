"""Resolving a proxy target that lives on another node.

**The browser on a worker could reach nothing but that worker.** Every
UI screen but Config and the playground talks to `gateway`, `library` or
`control`, and an enrolled node declares none of them -- `should_seed`
refuses to seed a control plane onto a node, correctly. So signing in on
a worker gave a console that 503'd on Gateway, Library, Metrics,
Discover and Nodes alike, with a message explaining that the component
lives on the control host and no way to get there. Reported from the
live two-machine install, on the Gateway tab.

## Address nodes, not components

The obvious fix -- ask the control root where the gateway is and go
there -- **does not work, and measuring why decided this design.**
Control's `/v1/components` reports each component's `url` as its owning
agent sees it, which for a containerised control root is
`http://127.0.0.1:8080/`. That address is perfectly correct *on that
host* and useless anywhere else, and no amount of rewriting fixes it:
the UnRAID install publishes the container's 8080 as 8280, and nothing
inside the container knows that mapping exists.

So the hop goes to the **owning node's agent**, at the address that node
told the control root it can be reached at, and that agent resolves the
component against its own topology -- where loopback is true again. One
address per node instead of one per component, and it is the one address
the install already maintains: `Node.url`, announced by each node and
re-announced whenever it changes.

That is also why a node's own `advertiseUrl` is load-bearing here and
not only for routing: a node whose registry entry says `127.0.0.1` can
be reached by nothing, and this module says so in those words rather
than failing with a connection error to a loopback port on the wrong
machine.

## Three properties, each deliberate

**It spends the caller's credential, not one of its own.** The lookup
carries the request's own `Authorization` header to the control root.
An enrolled node holds the install's signing key, so the operator token
the browser got from *this* agent is one the root accepts -- that is
what makes a worker's console a console for the install. Keeping the
proxy credential-free is the property `routes/proxy.py` was rebuilt to
have; spending a minted service token here would have quietly taken it
back.

**One hop, never two.** The resolved node is where the component is, so
a second hop can only be a resolution loop. `HOP_HEADER` marks a request
that has already been forwarded, and the receiving agent then resolves
locally or fails.

**The lookup is cached, because `/v1/components` fans out to every
node.** A dashboard opening six panels must not become six polls of the
whole install. Failures are cached too, for less time: control being
unreachable is exactly when a page would otherwise stall on six
consecutive connect timeouts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .node_identity import advertise_host, is_loopback_host

log = logging.getLogger(__name__)

# Marks a request this agent has already forwarded on behalf of another
# node. Its presence means "resolve locally or fail" -- see the module
# docstring on why a second hop can only be a loop.
HOP_HEADER = "x-eugene-plexus-proxy-hop"

# The lookup must be brisk. A component page that has to wait out the
# proxy's own ten-second connect budget before it can say "the control
# root is unreachable" is worse than one that says it in two.
_LOOKUP_TIMEOUT = httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0)

_HIT_TTL_SECONDS = 30.0
# Shorter, so an install that has just come back does not keep answering
# from a snapshot taken while it was down.
_MISS_TTL_SECONDS = 5.0

_SINGLETON_KINDS = ("gateway", "library", "control")


class InstallLookupError(Exception):
    """Why the install-wide lookup could not answer.

    Carries operator-facing prose rather than a code: every caller of
    this module puts the text straight into a `Problem`, and a failure
    here is always something the operator can act on -- start the root,
    set an advertise address, declare the component.
    """


@dataclass(frozen=True)
class RemoteNode:
    """A node that owns the target, and how to reach its agent."""

    name: str
    agent_url: str


@dataclass
class _Snapshot:
    """One install-wide read, with the clock that expires it."""

    expires_at: float
    # Target key -> node name. Singletons are keyed by kind, drivers by
    # name, which is exactly how `resolve_target` keys them locally.
    owners: dict[str, str] = field(default_factory=dict)
    node_urls: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    # A refusal of the *caller's* credential says nothing about the
    # install, so it must not be remembered on anyone else's behalf: an
    # unauthenticated probe would otherwise poison the answer for the
    # signed-in operator for the whole negative TTL.
    cacheable: bool = True


class InstallTopology:
    """A short-lived cache of who-runs-what, read from the control root.

    Lives on `app.state` so a test can pre-seed it, and so the cache is
    per-process rather than per-request.
    """

    def __init__(self) -> None:
        self._snapshot: _Snapshot | None = None
        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        self._snapshot = None

    async def owner_of(
        self,
        target: str,
        *,
        control_url: str,
        authorization: str | None,
        transport: Any = None,
    ) -> RemoteNode:
        """Which node runs `target`, and the URL of that node's agent.

        Raises `InstallLookupError` carrying prose the operator can act
        on -- each of the four failures below is a different action.
        """
        snapshot = await self._load(
            control_url=control_url, authorization=authorization, transport=transport
        )
        if snapshot.error is not None:
            raise InstallLookupError(snapshot.error)

        node = snapshot.owners.get(target)
        if node is None:
            raise InstallLookupError(
                f"Nothing called {target!r} is running anywhere in this install. The "
                "control root knows of none, so this is not a question of which node you "
                "are browsing from."
            )

        url = snapshot.node_urls.get(node)
        if not url:
            raise InstallLookupError(
                f"{target!r} runs on node {node!r}, but that node has no address in the "
                "install's registry, so nothing here can reach it."
            )
        if is_loopback_host(advertise_host(url)):
            raise InstallLookupError(
                f"{target!r} runs on node {node!r}, which advertises {url} -- a loopback "
                "address, reachable only from that host. Set `advertiseUrl` in that "
                "node's agent config to the address other hosts use for it (the same one "
                "you type in the browser); it re-announces itself immediately."
            )
        return RemoteNode(name=node, agent_url=url)

    # -- internals ---------------------------------------------------

    async def _load(
        self, *, control_url: str, authorization: str | None, transport: Any
    ) -> _Snapshot:
        cached = self._snapshot
        if cached is not None and cached.expires_at > time.monotonic():
            return cached

        async with self._lock:
            # Re-checked under the lock: several panels of one page
            # arrive together, and the point of the lock is that they
            # share one read rather than starting four.
            cached = self._snapshot
            if cached is not None and cached.expires_at > time.monotonic():
                return cached
            snapshot = await self._read(
                control_url=control_url, authorization=authorization, transport=transport
            )
            snapshot.expires_at = time.monotonic() + (
                _MISS_TTL_SECONDS if snapshot.error is not None else _HIT_TTL_SECONDS
            )
            if snapshot.cacheable:
                self._snapshot = snapshot
            return snapshot

    async def _read(
        self, *, control_url: str, authorization: str | None, transport: Any
    ) -> _Snapshot:
        base = control_url.rstrip("/")
        headers = {"accept": "application/json"}
        if authorization:
            headers["authorization"] = authorization
        try:
            async with httpx.AsyncClient(timeout=_LOOKUP_TIMEOUT, transport=transport) as client:
                components, nodes = await asyncio.gather(
                    client.get(f"{base}/v1/components", headers=headers),
                    client.get(f"{base}/v1/nodes", headers=headers),
                )
        except httpx.HTTPError as exc:
            return _Snapshot(
                expires_at=0.0,
                error=(
                    f"The control root at {control_url} did not answer, so this node cannot "
                    f"find out where the rest of the install is: {exc}"
                ),
            )

        for response in (components, nodes):
            if response.status_code >= 400:
                return _Snapshot(
                    expires_at=0.0,
                    error=(
                        f"The control root at {control_url} answered "
                        f"{response.status_code} for {response.request.url.path}: "
                        f"{_detail(response)}"
                    ),
                    cacheable=response.status_code not in (401, 403),
                )

        return _Snapshot(
            expires_at=0.0,
            owners=_owners(components),
            node_urls=_node_urls(nodes),
        )


def _owners(response: httpx.Response) -> dict[str, str]:
    """Target key -> node name, keyed the way the local resolver keys.

    A kind wins over a name, which is the precedence `resolve_target`
    applies locally: `gateway` means the install's gateway even if some
    node also has a driver entry called `gateway`.
    """
    owners: dict[str, str] = {}
    for entry in _items(response, "components"):
        node = entry.get("node")
        name = entry.get("name")
        if not isinstance(node, str):
            continue
        if isinstance(name, str) and name not in owners:
            owners[name] = node
    for entry in _items(response, "components"):
        node = entry.get("node")
        kind = entry.get("kind")
        if isinstance(node, str) and isinstance(kind, str) and kind in _SINGLETON_KINDS:
            owners[kind] = node
    return owners


def _node_urls(response: httpx.Response) -> dict[str, str]:
    urls: dict[str, str] = {}
    for entry in _items(response, "nodes"):
        name = entry.get("name")
        url = entry.get("url")
        if isinstance(name, str) and isinstance(url, str):
            urls[name] = url
    return urls


def _items(response: httpx.Response, key: str) -> list[dict[str, Any]]:
    try:
        body = response.json()
    except ValueError:
        return []
    if not isinstance(body, dict):
        return []
    values = body.get(key)
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def _detail(response: httpx.Response) -> str:
    """The operator-facing half of a `Problem`, or a short excerpt.

    A raw JSON envelope pasted into an error message is how the report
    that started this work read: the reader has to parse a document to
    find one sentence.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            return str(detail.get("detail") or detail.get("title") or detail)
        if detail:
            return str(detail)
    return str(body)[:200]
