"""Same-origin pass-through from the browser to every component.

Ported from `ui/src/app/api/proxy/[target]/[...path]/route.ts`, which
was 147 lines of Next.js route handler and the only dynamic route in
that application. Removing it is what lets the UI ship as a static
export inside a Python wheel — see `install-paths-and-distribution.md`
§3 in the specs repo.

**The port is smaller than the original, not larger**, and the reason
is the whole argument for moving it. The Next handler's hardest step
was resolving a target: to learn where `control` (or a driver) lives it
had to call the agent's bearer-protected `GET /v1/components` over
HTTP, which meant it needed a credential *of its own* just to look
something up. Here the topology is `request.app.state.agent_state` and
the lookup is a list scan.

**That dissolves M9's two-credential hack.** The wizard has to mint a
join token at the control root (which wants the *root's* token) while
resolving where the root is (which wanted the *agent's*), and between
initializing and enrolling those are genuinely two different keys. One
`Authorization` header cannot be both, so the UI grew
`x-eugene-plexus-upstream-authorization` to carry the second. Nothing
consumes a credential to resolve any more, so the wizard sends the
root's token in `Authorization` like any other caller. That header is
not deprecated here; it is not implemented here.

## Three properties worth stating, because each is load-bearing

**Unauthenticated, deliberately.** This is the path the login request
itself travels, so a dependency on a session token would make logging
in impossible. It confers no authority: `Authorization` is forwarded
verbatim, every component behind it enforces its own auth, and the set
of reachable addresses is exactly this agent's declared topology. It is
the same trust boundary the Next server had, one process to the left.

**It streams.** `client.send(stream=True)` plus `StreamingResponse`,
with `accept-encoding: identity` on the way up so no decoder sits in
the path. Buffering here would silently undo M10 — the playground would
still render, still frame SSE correctly, and still pass every "is it
streaming" check, while delivering one chunk at the end. That is the
exact shape of the gap M10 was built to close, and a one-line
convenience would have reintroduced it.

**Targets resolve by kind, not by configuration.** `gateway`,
`library` and `control` are looked up by `ComponentKind` because an
install has exactly one of each; anything else is an inference-driver
**by name**. The Next handler had `GATEWAY_URL` as an env var with a
loopback default, and that is one more place a component's URL can be
written down and disagree with what the agent actually spawned — the
OpenClaw trap the driver path had already avoided. The expert override
is not gone, it moved: edit the topology entry's URL, which is the
place that was always authoritative.

**And a target that is not on this node is forwarded to the node that
has it**, so a worker's browser is a console for the install rather
than for one host. That resolution lives in `install_proxy.py`, which
carries the argument for hopping node-to-node instead of
component-to-component; here it is only ever the second thing tried,
after the local topology and never instead of it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from .. import install_proxy
from .._generated.common_models import Problem
from ..node_identity import local_agent_url
from ..settings import Settings
from ..state import AgentState

log = logging.getLogger(__name__)

router = APIRouter(tags=["ui"])

PROXY_PREFIX = "/api/proxy"

# Targets resolved from the topology by kind, because an install has
# exactly one of each and a second place recording the URL is a second
# place it can be wrong.
_SINGLETON_KINDS = {"gateway": "gateway", "library": "library", "control": "control"}

# `node:<name>` addresses another node's AGENT rather than a component
# on it -- for the surfaces that are per-agent by nature (a runtime's
# start/stop, which engines a host has, an engine install). `agent` is
# still the local one; this is how the install-wide inference screen
# acts on a runtime that lives somewhere else.
NODE_PREFIX = "node:"

# Hop-by-hop headers, plus the two that describe a body we re-frame
# ourselves. `accept-encoding` is dropped and replaced rather than
# forwarded: see the module docstring on why no decoder may sit in this
# path.
_STRIPPED_REQUEST_HEADERS = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailer",
        "upgrade",
        "content-length",
        "transfer-encoding",
        "accept-encoding",
        # Ours, read below and re-set deliberately when we forward. A
        # component has no use for it and should never see it.
        install_proxy.HOP_HEADER,
    }
)

_STRIPPED_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "content-encoding",
        "content-length",
    }
)

_METHODS = ["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS", "HEAD"]

# No read timeout. A first token can be twenty seconds away behind an
# engine that is still loading, and an SSE stream is idle between tokens
# by definition; a read deadline here would cut a working generation off
# mid-answer and report it as an upstream failure. Connect and pool
# deadlines stay, because those really are failures.
_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)


def _problem(status_code: int, title: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/agent#{title.replace(' ', '-').lower()}",
            title=title,
            status=status_code,
            detail=detail,
            component="agent",
        ).model_dump(exclude_none=True),
    )


def is_valid_target_name(value: str) -> bool:
    """Sanity gate on operator-supplied driver names.

    Path traversal is what this refuses; the resolver below only ever
    returns an address already in the topology, so this is a cheap
    short-circuit on obvious garbage rather than the boundary itself.
    """
    return bool(value) and "/" not in value and ".." not in value


@dataclass(frozen=True)
class Route:
    """Where a target's request goes.

    `prefix` is empty for a component on this host and
    `/api/proxy/<target>` for one reached through another node's agent —
    the hop described in `install_proxy`. Same shape either way, so the
    forwarding code below has one path and cannot treat the remote case
    as an afterthought.
    """

    base: str
    prefix: str = ""
    node: str | None = None


def resolve_local(request: Request, target: str) -> str | None:
    """The base URL for a target declared on this host, or None."""
    settings: Settings = request.app.state.settings
    if target == "agent":
        return local_agent_url(settings.bind_host, int(settings.bind_port))

    state: AgentState = request.app.state.agent_state
    entries = state.list_topology_entries()

    kind = _SINGLETON_KINDS.get(target)
    if kind is not None:
        for entry in entries:
            if str(entry.kind) == kind and str(entry.url):
                return str(entry.url)
        return None

    for entry in entries:
        if str(entry.kind) == "inference-driver" and entry.name == target and str(entry.url):
            return str(entry.url)
    return None


def install_topology(request: Request) -> install_proxy.InstallTopology:
    """The shared install-wide lookup cache, created on first use.

    Injectable the same way the upstream client is, so a test can seed a
    resolution without a control root.
    """
    cache: install_proxy.InstallTopology | None = getattr(
        request.app.state, "install_topology", None
    )
    if cache is None:
        cache = install_proxy.InstallTopology()
        request.app.state.install_topology = cache
    return cache


async def resolve_target(request: Request, target: str) -> Route:
    """Where this target lives: on this host, on another node, or nowhere.

    A 503 rather than a fallback address, which is the lesson the Next
    version had already learned: a default that also fails tells the
    operator nothing about which of the two things is missing. The same
    reasoning is why each failure below says a different sentence — the
    operator's next action is different in each, and one generic "not in
    topology" was the report that started this work.
    """
    if target.startswith(NODE_PREFIX):
        return await _resolve_node(request, target[len(NODE_PREFIX) :])

    local = resolve_local(request, target)
    if local is not None:
        return Route(base=local)

    kind = _SINGLETON_KINDS.get(target)
    article = f"component of kind {kind!r}" if kind else f"inference-driver named {target!r}"

    # Already forwarded once. A second hop can only be a resolution
    # loop, so this ends here rather than going back out.
    hop = request.headers.get(install_proxy.HOP_HEADER)
    if hop:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Target not on this node",
            f"Node {hop!r} forwarded this request here because the install's registry says "
            f"the {article} runs on this node, but nothing of that description is declared "
            "here. The registry and this node's topology disagree.",
        )

    identity = getattr(request.app.state, "node_identity", None)
    record = identity.record if identity is not None else None
    if record is None or not record.enrolled or not record.control_url:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Target not in topology",
            f"No {article} is declared on this node, and this node is not enrolled with a "
            "control root, so there is nowhere else to look. Declare one on the Config "
            "page, or join this machine to an existing install.",
        )

    try:
        remote = await install_topology(request).owner_of(
            target,
            control_url=str(record.control_url),
            authorization=request.headers.get("authorization"),
            transport=getattr(request.app.state, "control_transport", None),
        )
    except install_proxy.InstallLookupError as exc:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Target not on this node",
            f"No {article} runs on this node, so it was looked for in the rest of the "
            f"install. {exc}",
        ) from exc

    if record.name and remote.name == record.name:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Target not on this node",
            f"The install's registry says the {article} runs on this node "
            f"({remote.name!r}), but nothing of that description is declared here. Check "
            "the Config page: the registry is reporting a component this agent is not "
            "running.",
        )

    return Route(base=remote.agent_url, prefix=f"{PROXY_PREFIX}/{target}", node=remote.name)


async def _resolve_node(request: Request, name: str) -> Route:
    """`node:<name>`: that node's agent, or this one when the name is ours.

    No `/api/proxy` prefix on the far side -- this reaches the agent's
    own API, so there is no second resolution and no way to loop. The
    hop marker is still set by `_request_headers` and still refused here
    on arrival, which costs nothing and keeps one rule.
    """
    settings: Settings = request.app.state.settings
    identity = getattr(request.app.state, "node_identity", None)
    record = identity.record if identity is not None else None

    if record is not None and record.name and record.name == name:
        return Route(base=local_agent_url(settings.bind_host, int(settings.bind_port)))

    hop = request.headers.get(install_proxy.HOP_HEADER)
    if hop:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Target not on this node",
            f"Node {hop!r} forwarded a request for node {name!r} here, but this node is "
            f"{record.name if record and record.name else 'not enrolled'!r}. The registry "
            "and this node's identity disagree.",
        )

    if record is None or not record.enrolled or not record.control_url:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Target not in topology",
            f"This node is not enrolled with a control root, so it knows of no node named "
            f"{name!r} -- or any other. Use the `agent` target for this host.",
        )

    try:
        url = await install_topology(request).agent_url_of(
            name,
            control_url=str(record.control_url),
            authorization=request.headers.get("authorization"),
            transport=getattr(request.app.state, "control_transport", None),
        )
    except install_proxy.InstallLookupError as exc:
        raise _problem(status.HTTP_503_SERVICE_UNAVAILABLE, "Node unreachable", str(exc)) from exc
    return Route(base=url, node=name)


def get_client(request: Request) -> httpx.AsyncClient:
    """The shared upstream client, created on first use.

    Injectable the way the supervisors are: a test that sets
    `app.state.ui_proxy_client` to a client over `httpx.MockTransport`
    gets a proxy with no sockets in it.
    """
    client: httpx.AsyncClient | None = getattr(request.app.state, "ui_proxy_client", None)
    if client is None:
        client = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=False)
        request.app.state.ui_proxy_client = client
    return client


def _upstream_url(base: str, path: str, query: str) -> httpx.URL:
    # Topology URLs arrive with a trailing slash (`http://127.0.0.1:8081/`).
    # Naive joining gives `http://127.0.0.1:8083//v1/config`, and FastAPI
    # treats the double slash as a different path and 404s. Normalise once.
    url = httpx.URL(base.rstrip("/") + "/" + path.lstrip("/"))
    if query:
        # The raw query string, not a parsed mapping: repeated keys and
        # the caller's own encoding both survive that way, and neither
        # survives a round trip through a dict.
        url = url.copy_with(query=query.encode("utf-8"))
    return url


def _request_headers(request: Request, route: Route) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _STRIPPED_REQUEST_HEADERS
    }
    headers["accept-encoding"] = "identity"
    if route.node is not None:
        # Set only on the hop to another node's agent, and it carries
        # this node's name so the receiver's refusal can say who sent it.
        identity = getattr(request.app.state, "node_identity", None)
        record = identity.record if identity is not None else None
        headers[install_proxy.HOP_HEADER] = (record.name if record else None) or "an enrolled node"
    return headers


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in _STRIPPED_RESPONSE_HEADERS
    }


@router.api_route(PROXY_PREFIX + "/{target}/{path:path}", methods=_METHODS, include_in_schema=False)
async def proxy(target: str, path: str, request: Request) -> StreamingResponse:
    if not is_valid_target_name(target):
        raise _problem(
            status.HTTP_400_BAD_REQUEST,
            "Invalid target",
            f"{target!r} is not a usable proxy target name.",
        )

    route = await resolve_target(request, target)
    url = _upstream_url(route.base, route.prefix + "/" + path.lstrip("/"), request.url.query)

    # The request body is buffered and the response is not, which is the
    # asymmetry that matters: uploads here are small JSON documents,
    # while the response can be a token stream that must not be held.
    body = b"" if request.method in ("GET", "HEAD") else await request.body()

    client = get_client(request)
    upstream_request = client.build_request(
        request.method, url, headers=_request_headers(request, route), content=body
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        log.debug("proxy to %s failed: %s", url, exc)
        if route.node is not None:
            # The lookup said this node is reachable and it was not, so
            # the cached answer is stale; the next request re-reads
            # rather than repeating a known-bad address for 30 seconds.
            install_topology(request).invalidate()
            raise _problem(
                status.HTTP_502_BAD_GATEWAY,
                "Upstream unreachable",
                f"The install's {target} runs on node {route.node!r}, whose agent at "
                f"{route.base} did not answer: {exc}",
            ) from exc
        raise _problem(
            status.HTTP_502_BAD_GATEWAY,
            "Upstream unreachable",
            f"{target} at {url} did not answer: {exc}",
        ) from exc

    async def body_stream() -> AsyncIterator[bytes]:
        try:
            # aiter_raw, not aiter_bytes: we asked upstream for identity
            # encoding, so there is nothing to decode, and the decoder is
            # the thing that would re-buffer an SSE body.
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        body_stream(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream),
    )
