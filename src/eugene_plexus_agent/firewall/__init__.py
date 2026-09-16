r"""What the host firewall says about the ports this install publishes.

The third of the three things that have to be true before another device
can reach this machine (`reach.py` holds the other two), and the only one
outside this process. Its whole job is to turn "connection refused" into
a sentence naming the thing that refused it.

**Every answer is evidence or `unknown`; nothing is ever inferred up to
`allowed`.** A firewall we cannot read, a third-party product we can see
but not query, a platform with no detector — all `unknown`, with a
sentence saying which. That is `easy-default-expert-override`'s
corollary applied to a detector: an eager refusal can be wrong, and an
explanation of a real failure cannot, so the card must be able to say
"we could not confirm" rather than inventing a verdict in either
direction.

Three things were measured on a real Windows host before this was
written, and each of them changed the code:

1. **The rule allowing the live install is bound to the PROGRAM, not to
   a port**, and Windows' own *Windows Security Alert* dialog created it
   — naming
   `%LOCALAPPDATA%\EugenePlexus\pythons\cpython-3.12.14-...\python.exe`.
   A detector that looked only at port rules would have called that
   machine `blocked` while the control root was reaching it every
   fifteen seconds. So program rules count, and `scope` says which kind
   decided, because a program rule naming a versioned interpreter path
   stops covering us the day that interpreter is upgraded.
2. **`Get-NetFirewallPortFilter` fails with *Access is denied* when it
   is not elevated**, and returns a *partial* list on the way out. A
   detector built on the cmdlets would report "no rule covers our port"
   from a truncated read. The COM interface has no such problem
   unelevated — it enumerated all 708 rules — so the cmdlets are not
   used at all.
3. **COM is thirty times faster**: 78 ms for profiles plus every rule,
   against ~2.8 s for the three cmdlets, before a PowerShell subprocess
   has even started. The design budgeted "a few hundred milliseconds"
   and worried about caching; at 78 ms the read happens with the node
   view and the worry goes away.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from .._generated.models import FirewallPort, HostFirewall, Verdict

log = logging.getLogger(__name__)

__all__ = ["FirewallQuery", "read_firewall", "unsupported"]


@dataclass(frozen=True)
class FirewallQuery:
    """What the caller wants a verdict about.

    `ports` are the TCP ports this install publishes. `program` is the
    executable those ports are served from — this agent's interpreter —
    because on Windows the rule that matters is very often bound to it
    rather than to them.
    """

    ports: tuple[int, ...]
    program: str | None = None


def read_firewall(query: FirewallQuery) -> HostFirewall:
    """The host firewall's answer.

    Never raises. A detector that throws while explaining why something
    is unreachable turns one failure into two, and the caller is a
    `GET /v1/node` that has other things to say.
    """
    try:
        if sys.platform == "win32":
            from . import windows

            return windows.read(query)
        if sys.platform == "darwin":
            from . import macos

            return macos.read(query)
        if sys.platform.startswith("linux"):
            from . import linux

            return linux.read(query)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("firewall detection failed", exc_info=True)
        return unsupported(f"The firewall could not be read: {exc}", query)

    return unsupported(
        f"There is no firewall detector for {sys.platform!r}, so nothing here can say "
        "whether other machines are allowed in. Try it from another device.",
        query,
    )


def unsupported(detail: str, query: FirewallQuery) -> HostFirewall:
    """`supported: false`, with every port `unknown` and the reason said.

    Shared by every platform module so a detector that cannot answer
    produces the same shape as one that can, rather than a blank object
    a card would render as "fine".
    """
    return HostFirewall(
        supported=False,
        ports=[FirewallPort(port=p, verdict=Verdict.unknown) for p in query.ports],
        detail=detail,
    )
