"""Leaving an install, and noticing that the install has removed this machine.

agent#21 (2026-10-20). A machine the install removes while it is still
running keeps its old enrollment. Its sign-in still goes to the root, and
the root, which no longer lists it, addresses the session to `control`
only, so this agent refuses that session everywhere -- including at
`POST /v1/node/unenroll`, the one route that would let it leave. It was
stuck until someone edited its files.

Troy's call (2026-10-20) was both of the issue's options:

* **B, the trust pull.** Every minute this agent takes the root's signed
  bundle. When that bundle no longer lists this machine's own token key,
  the install has removed it, and the machine leaves by itself: it
  becomes standalone, and its own passphrase signs in again.
* **A, the sign-in and a leave by passphrase.** A forwarded sign-in whose
  session does not name this machine is refused with the reason, after
  the same check runs at once. A machine whose root is gone, or that was
  removed, can leave with its own passphrase and no session.

**Why acting on the bundle is safe.** The bundle is signed by the
authority key this machine pinned when it joined, and it is no older than
the one held (rollback and the epoch fence both hold first). Leaving only
makes this machine stop trusting the install; it cannot make the install
trust it (`POST /v1/node/unenroll`'s own argument). Two guards keep a
broken root from sending every machine away at once: a bundle that lists
no key of the root itself is not taken as evidence (the root's
fail-closed empty list), and a machine that runs the install's active
control root never leaves this way.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from . import standby, tokens
from ._http import internal_client
from .machine_trust import pull as pull_machine_list
from .node_identity import FencedError
from .trust import BundleRollback

log = logging.getLogger(__name__)

#: How long one ask of the root waits for its bundle.
PULL_TIMEOUT_SECONDS = 10.0


class Heard(enum.Enum):
    """What one ask of the root found out about this machine."""

    LISTED = "listed"
    """The root's bundle lists this machine's key."""
    LEFT = "left"
    """The root's bundle does not list it, so this machine has left."""
    NO_ANSWER = "no-answer"
    """No bundle this machine could take: the root did not answer, or
    answered with one that is not newer or not signed by its authority."""
    NOT_ENROLLED = "not-enrolled"
    """This machine has joined nothing (or left already)."""


@dataclass(frozen=True)
class Left:
    """The install this machine has just left."""

    name: str
    control_url: str


def removed_from(bundle: tokens.TrustBundle, *, own_kid: str) -> bool:
    """Whether `bundle` says the install no longer lists this machine.

    True only for a bundle that lists the root's own key and not this
    machine's: a bundle with no root key is the root failing closed, not
    a statement about this machine."""
    if own_kid in bundle.keys:
        return False
    return any(
        key.issuer == tokens.ISSUER_CONTROL and tokens.GRANT_AUTHORITY in key.grants
        for key in bundle.keys.values()
    )


def _lock(app: Any) -> asyncio.Lock:
    lock = getattr(app.state, "leave_lock", None)
    if lock is None:
        lock = app.state.leave_lock = asyncio.Lock()
    return lock


async def restart_children(app: Any, *, why: str) -> None:
    """Every supervised component reads the bundle's path and the pinned
    authority from its environment at spawn, so a change of authority
    reaches them only through a respawn. A newer bundle from the same
    authority does not need one: they reload the file. Engines are not
    touched: they have no auth."""
    supervisor = getattr(app.state, "supervisor", None)
    if supervisor is None:
        return
    try:
        restarted = await supervisor.restart_all()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("could not restart supervised components after %s: %s", why, exc)
        return
    if restarted:
        log.info(
            "%s; restarting %d supervised component(s) so they trust the new authority: %s",
            why,
            len(restarted),
            ", ".join(restarted),
        )


async def leave(app: Any, *, notified: bool, why: str) -> Left | None:
    """Leave the install this machine joined; None if it has joined none.

    The one way out, for the route and for the trust pull alike: forget
    the root's bundle, the epoch and the root, become this machine's own
    authority again (which ends every session the root signed), forget
    the machine list, and restart the components."""
    async with _lock(app):
        store = app.state.node_identity
        record = store.record
        if not record.enrolled:
            return None
        left = Left(name=str(record.name), control_url=str(record.control_url))
        trust = app.state.auth_state.trust
        store.unenroll()
        trust.forget()
        trust.become_standalone()
        app.state.machine_trust.forget()
        log.warning(
            "left the install at %s (was %r): %s; this node is its own authority again%s",
            left.control_url,
            left.name,
            why,
            "" if notified else "; THE CONTROL ROOT WAS NOT TOLD",
        )
        await restart_children(app, why="left the install and became its own authority")
    # A standby it still ran has no grant now, since it has joined nothing.
    await standby.reconcile(app)
    return left


async def leave_if_removed(app: Any, bundle: tokens.TrustBundle) -> Left | None:
    """Leave when `bundle`, just taken from the root, no longer lists this
    machine (B). Never on the root's own machine."""
    trust = app.state.auth_state.trust
    if not trust.enrolled or not removed_from(bundle, own_kid=trust.signer().kid):
        return None
    if standby.hosts_active_control(app.state.agent_state):
        log.error(
            "the control root's bundle %d does not list this machine's key, but this machine "
            "runs the install's control root, so it does not leave",
            bundle.version,
        )
        return None
    return await leave(
        app,
        notified=True,
        why=f"the install removed this machine (its trust bundle {bundle.version} no longer "
        "lists this machine's key)",
    )


async def pull_once(app: Any, client: httpx.AsyncClient) -> Heard:
    """Ask the root for its bundle once, take it, and act on it.

    Takes a newer bundle, brings the standby in line with it, leaves when
    it no longer lists this machine, and otherwise pulls the machine list
    on the same minute (LT1). Never raises."""
    trust = app.state.auth_state.trust
    record = app.state.node_identity.record
    # Before asking: a bundle the root pushed once, and that could not be
    # written then, reaches the file even while the root this node joined
    # does not answer (after a promotion).
    trust.save_pending()
    if not (trust.enrolled and record.control_url):
        return Heard.NOT_ENROLLED
    control_url = str(record.control_url).rstrip("/")
    heard = Heard.NO_ANSWER
    try:
        response = await client.get(f"{control_url}/v1/trust/bundle")
        if response.status_code == 200:
            held = trust.bundle
            taken = trust.accept(str(response.json()["jws"]))
            if held is None or taken.version != held.version:
                log.info("pulled trust bundle %d", taken.version)
            await standby.reconcile(app)
            if await leave_if_removed(app, taken) is not None:
                return Heard.LEFT
            heard = Heard.LISTED
        # The machine list rides the same minute (LT1). Public keys and
        # addresses, signed; no credential is sent.
        await pull_machine_list(app.state.machine_trust, client, record.control_url)
        return heard
    except BundleRollback:
        return Heard.NO_ANSWER
    except (tokens.BundleError, FencedError) as exc:
        log.warning("refused the control root's trust bundle: %s", exc)
        return Heard.NO_ANSWER
    except Exception as exc:
        log.debug("could not pull the trust bundle: %s", exc)
        return Heard.NO_ANSWER


async def ask_now(app: Any) -> Heard:
    """`pull_once` outside the minute's loop, with a client of its own."""
    transport = getattr(app.state, "control_transport", None)
    async with internal_client(timeout=PULL_TIMEOUT_SECONDS, transport=transport) as client:
        return await pull_once(app, client)
