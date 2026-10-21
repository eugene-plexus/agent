"""Leaving an install, and noticing that the install has removed this machine.

agent#21 (2026-10-20). A machine the install removes while it is still
running keeps its old enrollment. Its sign-in still goes to the root, and
the root, which no longer lists it, addresses the session to `control`
only, so this agent refuses that session everywhere -- including at
`POST /v1/node/unenroll`, the one route that would let it leave. It was
stuck until someone edited its files.

Troy's call (2026-10-20) was both of the issue's options:

* **B, the trust pull.** Every minute this agent takes the root's signed
  bundle and machine list. When neither lists this machine any more, the
  install has removed it, and the machine leaves by itself: it becomes
  standalone, and its own passphrase signs in again.
* **A, the sign-in and a leave by passphrase.** A forwarded sign-in whose
  session does not name this machine is refused with the reason, after
  the same check runs at once. A machine whose root is gone, or that was
  removed, can leave with its own passphrase and no session.

**Why acting on the root's word is safe.** The bundle and the list are
signed by the authority key this machine pinned when it joined, and
neither may be older than the one held (rollback and the epoch fence
hold first). Leaving only makes this machine stop trusting the install;
it cannot make the install trust it (`POST /v1/node/unenroll`'s own
argument). The guards, from the review of this change:

* the bundle must list a key of the root itself (a root whose own key is
  unreadable fails closed with an empty list);
* the root's signed machine list, taken on the same pull, must not name
  this machine either. A root leaves a key out of the bundle for faults
  too (a duplicated or unreadable key) while it still has the machine
  registered; the list names every registered machine;
* this machine held a bundle before the pull, so a lost bundle file
  followed by an old one is not enough;
* a machine that runs the install's active control root never leaves
  this way;
* a machine with no passphrase of its own leaves, and its first-run
  wizard then accepts a passphrase only from the machine itself
  (agent#22, Troy 2026-10-21): it still listens where other machines
  reached it, and the first person on the network to set one would own it;
* the leave happens only if the enrollment is still the one the pull
  started with, so a re-join in between is not undone.
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
    """The root answered with a bundle, and this machine has not left."""
    LEFT = "left"
    """This machine has left the install."""
    NO_ANSWER = "no-answer"
    """The root did not answer, or answered with an error status."""
    UNTRUSTED = "untrusted"
    """The root answered with something this machine will not take: not a
    bundle, not signed by its authority, older than the one held, or from
    a fenced epoch."""
    NOT_ENROLLED = "not-enrolled"
    """This machine has joined nothing (or left already)."""


@dataclass(frozen=True)
class Left:
    """The install this machine has just left."""

    name: str
    control_url: str


def removed_from(bundle: tokens.TrustBundle, *, own_kid: str) -> bool:
    """Whether `bundle` no longer lists this machine's key.

    True only for a bundle that lists the root's own key and not this
    machine's: a bundle with no root key is the root failing closed, not
    a statement about this machine."""
    if own_kid in bundle.keys:
        return False
    return any(
        key.issuer == tokens.ISSUER_CONTROL and tokens.GRANT_AUTHORITY in key.grants
        for key in bundle.keys.values()
    )


def lists_machine(claims: dict[str, Any] | None, name: str | None) -> bool:
    """Whether a signed machine list names `name`."""
    return any(
        isinstance(node, dict) and node.get("name") == name
        for node in (claims or {}).get("nodes") or []
    )


def _lock(app: Any) -> asyncio.Lock:
    lock = getattr(app.state, "leave_lock", None)
    if lock is None:
        lock = app.state.leave_lock = asyncio.Lock()
    return lock


def _enrollment(record: Any) -> tuple[Any, ...]:
    """Which enrollment this is: the root, the name and when it was made."""
    return (record.control_public_key, record.name, record.enrolled_at)


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


async def leave(
    app: Any, *, notified: bool, why: str, expect: tuple[Any, ...] | None = None
) -> Left | None:
    """Leave the install this machine joined; None if it has joined none,
    or, with `expect`, if its enrollment is no longer that one.

    The one way out, for the route and for the trust pull alike: forget
    the root's bundle, the epoch and the root, become this machine's own
    authority again (which ends every session the root signed), forget
    the machine list, and restart the components."""
    async with _lock(app):
        store = app.state.node_identity
        record = store.record
        if not record.enrolled:
            return None
        if expect is not None and _enrollment(record) != expect:
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


async def _leave_if_removed(
    app: Any, bundle: tokens.TrustBundle, *, list_taken: bool, expect: tuple[Any, ...]
) -> Left | None:
    """Leave when `bundle` and the machine list, both just taken from the
    root, no longer list this machine (B). See the module's guards."""
    trust = app.state.auth_state.trust
    if not trust.enrolled or not removed_from(bundle, own_kid=trust.signer().kid):
        return None
    name = app.state.node_identity.record.name
    why_not: str | None = None
    if not list_taken:
        why_not = "no machine list could be taken from the root to confirm it"
    elif lists_machine(app.state.machine_trust.claims, name):
        why_not = "the root's machine list still names it, so the root still has it registered"
    elif standby.hosts_active_control(app.state.agent_state):
        why_not = "this machine runs the install's control root"
    if why_not is not None:
        if getattr(app.state, "removed_said", None) != (bundle.version, why_not):
            app.state.removed_said = (bundle.version, why_not)
            log.error(
                "the control root's trust bundle %d does not list this machine's key, but it "
                "does not leave the install: %s",
                bundle.version,
                why_not,
            )
        return None
    state = app.state.agent_state
    if not state.has_passphrase():
        # Before leaving, so no moment exists where the wizard answers anyone.
        state.require_setup_here()
    return await leave(
        app,
        notified=True,
        why=f"the install removed this machine (trust bundle {bundle.version} and the "
        "machine list no longer list it)",
        expect=expect,
    )


async def pull_once(app: Any, client: httpx.AsyncClient) -> Heard:
    """Ask the root for its bundle once, take it, and act on it.

    Takes a newer bundle, brings the standby in line with it, pulls the
    machine list on the same minute (LT1), and leaves when both no longer
    list this machine. Never raises."""
    trust = app.state.auth_state.trust
    record = app.state.node_identity.record
    # Before asking: a bundle the root pushed once, and that could not be
    # written then, reaches the file even while the root this node joined
    # does not answer (after a promotion).
    trust.save_pending()
    if not (trust.enrolled and record.control_url):
        return Heard.NOT_ENROLLED
    expect = _enrollment(record)
    control_url = str(record.control_url).rstrip("/")
    heard = Heard.NO_ANSWER
    taken: tokens.TrustBundle | None = None
    held: tokens.TrustBundle | None = None
    try:
        response = await client.get(f"{control_url}/v1/trust/bundle")
        if response.status_code == 200:
            try:
                jws = str(response.json()["jws"])
            except (ValueError, KeyError, TypeError) as exc:
                log.warning("the control root answered its bundle with no bundle: %s", exc)
                return Heard.UNTRUSTED
            held = trust.bundle
            taken = trust.accept(jws)
            if held is None or taken.version != held.version:
                log.info("pulled trust bundle %d", taken.version)
            await standby.reconcile(app)
            heard = Heard.LISTED
        # The machine list rides the same minute (LT1). Public keys and
        # addresses, signed; no credential is sent.
        machines = app.state.machine_trust
        list_before = machines.taken_at
        await pull_machine_list(machines, client, record.control_url)
        list_taken = machines.taken_at is not None and machines.taken_at != list_before
    except BundleRollback as exc:
        log.debug("the control root's bundle is older than the one held: %s", exc)
        return Heard.UNTRUSTED
    except (tokens.BundleError, FencedError) as exc:
        log.warning("refused the control root's trust bundle: %s", exc)
        return Heard.UNTRUSTED
    except Exception as exc:
        log.debug("could not pull the trust bundle: %s", exc)
        return Heard.NO_ANSWER
    if taken is not None and held is not None:
        await _leave_if_removed(app, taken, list_taken=list_taken, expect=expect)
    if not trust.enrolled:
        # This pull, or another one at the same moment, left.
        return Heard.LEFT
    return heard


async def ask_now(app: Any) -> Heard:
    """`pull_once` outside the minute's loop, with a client of its own."""
    transport = getattr(app.state, "control_transport", None)
    async with internal_client(timeout=PULL_TIMEOUT_SECONDS, transport=transport) as client:
        return await pull_once(app, client)
