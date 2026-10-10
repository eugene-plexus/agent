"""This install's own firewall rule follows its listeners (Troy, 2026-10-10).

The rule *Eugene Plexus* is bound to ports, not to a program (a program rule
names a versioned interpreter path and silently stops applying when it is
upgraded: `firewall/windows.py`). It was written with the ports listening
the day the person allowed it. Every runtime added since gets a driver on a
new port, and each one stayed blocked from the network with nothing saying
so: Strata's driver on Amish_Station's port 8093, which the NAS's gateway
could not reach.

So once the person has allowed Eugene through (the rule exists), the agent
keeps the rule's ports to exactly what this install listens on off
loopback, itself. Never a rule the person did not allow: with no rule there
is nothing to follow, and the reach card still asks. Windows only, and only
elevated (a service install always is): macOS's rule names the program,
which covers every port, and on Linux this agent has no root, so the reach
card's command stays the way.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from .reach import Listener

log = logging.getLogger(__name__)

#: How often the rule is compared with what listens (a read is ~80 ms).
INTERVAL_SECONDS = 30.0

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class Firewall(Protocol):
    def elevated(self) -> bool: ...
    def rule_ports(self) -> tuple[int, ...] | None: ...
    def set_rule_ports(self, ports: tuple[int, ...]) -> tuple[bool, str]: ...


def wanted_ports(listeners: Iterable[Listener]) -> tuple[int, ...]:
    """The ports this install listens on off loopback. No bind host named
    is a component's own loopback default."""
    return tuple(
        sorted(
            {
                listener.port
                for listener in listeners
                if listener.bind_host and listener.bind_host.strip("[]") not in _LOOPBACK
            }
        )
    )


def follow_once(
    listeners: Iterable[Listener],
    *,
    firewall: Firewall | None = None,
    platform: str = sys.platform,
) -> str | None:
    """The rule set to what listens, when it differs; what was done, or None."""
    if firewall is None:
        if platform != "win32":
            return None
        from .firewall import windows

        firewall = windows
    assert firewall is not None
    if not firewall.elevated():
        return None
    have = firewall.rule_ports()
    if have is None:
        return None
    want = wanted_ports(listeners)
    if not want or set(want) == set(have):
        return None
    ok, detail = firewall.set_rule_ports(want)
    if ok:
        log.info("firewall rule follows this install's listeners: %s", detail)
    else:
        log.warning("could not bring the firewall rule up to date: %s", detail)
    return detail


async def run(listeners: Callable[[], list[Listener]], *, firewall: Any = None) -> None:
    """Follow for as long as the agent runs."""
    while True:
        try:
            await asyncio.to_thread(follow_once, listeners(), firewall=firewall)
        except asyncio.CancelledError:
            raise
        except Exception:  # a firewall that cannot be read is no reason to stop
            log.debug("firewall follow failed", exc_info=True)
        await asyncio.sleep(INTERVAL_SECONDS)
