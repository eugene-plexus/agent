"""Who a request actually came from, when it reached us through our own proxy.

**The browser never talks to a component directly.** It loads the UI from
this agent and every call it makes goes to `/api/proxy/<target>/...`,
which this process forwards over loopback. So by the time a login
handler asks `request.client.host`, the answer is `127.0.0.1` for every
caller on earth — the one address it can never usefully be.

Two separate defects come out of that, and only one of them needs an
attacker.

**Every proxied login shares one bucket.** The limiter in
`routes/auth.py` is per source, five failures in sixty seconds. With
every proxied caller keyed `127.0.0.1`, five mistyped passphrases from
one person lock *every* browser out of the install for a minute — on the
one screen where a person who has just set a passphrase is most likely
to mistype it. Nobody has to do anything wrong for this to happen.

**And the key was attacker-supplied.** uvicorn ships
`ProxyHeadersMiddleware` on by default, trusting `X-Forwarded-For` from
`127.0.0.1` — which our proxy's peer always is. The proxy forwarded the
caller's headers verbatim, so anyone who could reach the login page
could pick their own limiter bucket, a fresh one per attempt, and the
limiter counted to one forever. The same lever forged
`NodeReach.lastReachedFrom`, which is the *only* evidence on the Reach
card that another device ever got in — see `off_host.py`, whose whole
argument is that this field is evidence rather than configuration.

## The shape of the fix, and why it is a header we own

The forwarding headers are stripped on the way through the proxy
(`FORWARDING_HEADERS`, in `routes/proxy.py`'s stripped set), uvicorn is
told to trust none of them (`forwarded_allow_ips=[]` in every
entrypoint), and the proxy sets `PEER_HEADER` itself from the peer it
can actually see.

`PEER_HEADER` is trustworthy for exactly the reason
`install_proxy.HOP_HEADER` is: **it is in the stripped set**, so a
caller's copy never survives the hop, and the only value a component
ever reads is one this process just wrote. That is also why it is read
**only when the immediate peer is loopback**. Off-host, the TCP peer is
better evidence than any header, and a request that crossed a network to
get here did not come through our proxy.

It confers no authority — it picks a rate-limit bucket and names a
witness. Nothing is authorised by it.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable

__all__ = ["FORWARDING_HEADERS", "PEER_HEADER", "header_of", "is_loopback", "peer_of"]

# Ours, set by the proxy on the way through and stripped on the way in,
# so a caller cannot supply one. Carries the address the proxy itself
# saw, which is the thing the forwarding headers claim to carry and do
# not.
PEER_HEADER = "x-eugene-plexus-peer"

# The headers that claim to say where a request came from, every one of
# which is written by whoever sent it. Stripped rather than sanitised:
# this install has no reverse proxy in front of it that we configured,
# so any of these arriving is either a caller's invention or a
# deployment we know nothing about. An operator who really does front
# this with nginx has `advertiseUrl` and the component's own bind host;
# they do not have a way to make us believe a header, on purpose.
FORWARDING_HEADERS = frozenset(
    {
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-prefix",
        "x-forwarded-proto",
        "x-real-ip",
    }
)


def _parsed(host: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """`host` as an address, or None if it is not one.

    A value that is not an address is not evidence of anything, and it
    is a fine rate-limit bucket key for an attacker who wants an
    unbounded supply of them. `strip("[]")` because an IPv6 peer arrives
    bracketed from some transports and bare from others.
    """
    if not host:
        return None
    try:
        return ipaddress.ip_address(host.strip().strip("[]"))
    except ValueError:
        return None


def is_loopback(host: str | None) -> bool:
    """Is this address this machine?

    `::1` and IPv4-mapped loopback are, which a string comparison to
    `"127.0.0.1"` would miss. Something that is not an address at all is
    not loopback: a test transport's `"testclient"` placeholder and a
    Unix socket peer are both *unknown*, and treating unknown as local
    is how a header comes to be trusted by a caller we cannot see.
    """
    parsed = _parsed(host)
    if parsed is None:
        return False
    if parsed.is_loopback:
        return True
    mapped = getattr(parsed, "ipv4_mapped", None)
    return mapped is not None and mapped.is_loopback


def peer_of(peer: str | None, supplied: str | None) -> str | None:
    """The address this request came from, as well as we can know it.

    `peer` is the TCP peer (`request.client.host`, `scope["client"][0]`);
    `supplied` is our own `PEER_HEADER` if one arrived.

    The header wins **only** when the peer is loopback, which is exactly
    the case where the peer is our own proxy and therefore says nothing.
    Off-host the peer is the truth and the header is ignored, which is
    what makes a forged one worthless from anywhere it could be forged.
    A header that is not an address is discarded rather than used.
    """
    if is_loopback(peer):
        forwarded = _parsed(supplied)
        if forwarded is not None:
            return str(forwarded)
    return peer


def header_of(raw_headers: Iterable[tuple[bytes, bytes]]) -> str | None:
    """`PEER_HEADER` out of a raw ASGI header list.

    For the middleware in `off_host.py`, which runs on the scope and has
    no `Request` to ask. Case-insensitive because ASGI does not promise
    lowercase, however reliably servers deliver it.
    """
    wanted = PEER_HEADER.encode("latin-1")
    for key, value in raw_headers:
        if key.lower() == wanted:
            return value.decode("latin-1")
    return None
