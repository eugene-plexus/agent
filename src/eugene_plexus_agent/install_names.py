"""Names the owner lists for this node's certificate, at the root (LT2).

`specs/docs/design/tls-for-clients.md` §4: a node's leaf holds its own
address, `<name>.local`, and the names the owner listed for it. The
container's entry point answers for names the owner chose in Settings; when
they are applied, this lists the ones the install CA may sign at the root,
with the owner's own session, beside any listed already.
"""

from __future__ import annotations

import ipaddress
import logging
import re

import httpx

from ._http import internal_client

log = logging.getLogger(__name__)

_SUFFIXES = ("local", "home.arpa", "internal", "lan", "localhost")
#: The install CA's permitted ranges, as control's `install_ca` has them.
_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "fc00::/7",
        "fe80::/10",
        "::1/128",
    )
)
_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def signable(name: str) -> bool:
    """Whether the install CA may sign `name`: the root's rule
    (`install_ca.permitted`), so a name it would refuse is never sent and
    the rest are listed."""
    name = name.strip().rstrip(".").lower()
    try:
        address = ipaddress.ip_address(name.strip("[]"))
    except ValueError:
        if not name or len(name) > 253 or not all(_LABEL.match(p) for p in name.split(".")):
            return False
        return any(name == s or name.endswith("." + s) for s in _SUFFIXES)
    return any(address in network for network in _NETWORKS)


async def add(control_url: str, node: str, names: list[str], authorization: str) -> None:
    """Never raises: a root that is down or refuses leaves the proxy on its own CA."""
    wanted = [n.lower() for n in names if signable(n)]
    if not wanted:
        return
    base = control_url.rstrip("/") + f"/v1/nodes/{node}"
    headers = {"Authorization": authorization}
    try:
        async with internal_client(timeout=15.0) as client:
            current = await client.get(base, headers=headers)
            current.raise_for_status()
            listed = [str(n) for n in current.json().get("certificateNames") or []]
            merged = list(dict.fromkeys([*listed, *wanted]))
            if merged == listed:
                return
            answer = await client.put(
                base + "/certificate-names", json={"names": merged}, headers=headers
            )
            answer.raise_for_status()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning(
            "could not list %s for this machine's certificate at the root (%s); the HTTPS "
            "entry point keeps its own certificate authority",
            ", ".join(wanted),
            exc,
        )
        return
    log.info("listed %s for this machine's certificate at the root", ", ".join(wanted))
