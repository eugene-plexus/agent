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
import time
from dataclasses import dataclass
from typing import Any

import httpx

from . import standby, tokens
from ._http import internal_client, refused_pin
from .machine_trust import LIST_PATH, MachineListError, _unpinned
from .machine_trust import pull as pull_machine_list
from .node_identity import FencedError
from .trust import BundleRollback

log = logging.getLogger(__name__)

#: How long one ask of the root waits for its bundle.
PULL_TIMEOUT_SECONDS = 10.0

#: Row 4: after this long without an answer from its root, a node asks the
#: machine that held the standby grant whether it is the root now.
ASK_STANDBY_AFTER_SECONDS = 120.0

#: The clock the silence is measured on; a seam for tests.
_clock = time.perf_counter


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
        state = app.state.agent_state
        if not state.has_passphrase():
            # Every way out (agent#22): before forgetting, so no moment exists
            # where the wizard answers the network. A failed write stops here.
            state.require_setup_here()
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
    return await leave(
        app,
        notified=True,
        why=f"the install removed this machine (trust bundle {bundle.version} and the "
        "machine list no longer list it)",
        expect=expect,
    )


async def bundle_response(
    client: httpx.AsyncClient, control_url: str, machines: Any
) -> httpx.Response:
    """`GET /v1/trust/bundle` at the root, through a change of the root's
    machine key that this machine missed (row 3).

    A machine that was off for the whole change holds a machine list with
    only the root's old key, so its pinned call to the root is refused
    before a byte is sent, the bundle's included, and the machine list is
    never reached. So, on that refusal, it fetches the root's signed machine
    list first: `machine_trust.pull` takes it in plain HTTP when the pinned
    call is refused, and only if it verifies against the root's identity key,
    which never changes. With the list taken, the bundle is asked for again,
    over TLS to the root's new key. A list that does not verify changes
    nothing, and the refusal stands."""
    url = f"{control_url}/v1/trust/bundle"
    try:
        return await client.get(url)
    except httpx.HTTPError as exc:
        if refused_pin(exc) is None:
            raise
        before = machines.claims
        await pull_machine_list(machines, client, control_url)
        if machines.claims is before:
            raise
        log.info("took the root's new machine key from its signed machine list")
        return await client.get(url)


async def pull_once(app: Any, client: httpx.AsyncClient) -> Heard:
    """Ask the root for its bundle once, take it, and act on it.

    Takes a newer bundle, brings the standby in line with it, pulls the
    machine list on the same minute (LT1), and leaves when both no longer
    list this machine. Never raises.

    Row 4: a root that has not answered for `ASK_STANDBY_AFTER_SECONDS`
    may have been replaced while this node could not hear it, so the
    machine that held the standby grant is asked too (`ask_standby`)."""
    app.state.root_obsolete = False
    heard = await _pull_once(app, client)
    # Row 4 review: a root that answers only for an epoch this node has left
    # behind is as silent as one that does not answer.
    silent = heard is Heard.NO_ANSWER or (
        heard is Heard.UNTRUSTED and getattr(app.state, "root_obsolete", False)
    )
    if silent:
        since = getattr(app.state, "root_silent_since", None)
        now = _clock()
        if since is None:
            app.state.root_silent_since = now
            record = app.state.node_identity.record
            log.warning(
                "cannot reach the root at %s: %s; this node keeps trying",
                record.control_url,
                getattr(app.state, "root_silent_cause", None) or "no answer",
            )
        elif now - since >= ASK_STANDBY_AFTER_SECONDS:
            await ask_standby(app, client)
    else:
        app.state.root_silent_since = None
    if heard is not Heard.LISTED:
        # Row 4 re-review: a copy promoted here, and restarted before any
        # push, is found on the minute even while the root this node names
        # does not answer; and whether the root here is locked stays known.
        await standby.reconcile(app)
    return heard


def standby_holder(bundle: tokens.TrustBundle | None) -> str | None:
    """The node whose key holds the `standby` grant in `bundle`, if any."""
    for key in bundle.keys.values() if bundle is not None else ():
        if tokens.GRANT_STANDBY in key.grants and key.issuer.startswith("node:"):
            return key.issuer.removeprefix("node:")
    return None


async def ask_standby(app: Any, client: httpx.AsyncClient) -> bool:
    """Row 4's fallback: ask the machine that held the standby grant for its
    root's machine list, through that machine's port and pinned to its key.

    `GET https://<its door>/control/v1/trust/machines`. Its port routes
    `/control` only to a control root it runs as the root, so a copy that is
    still a standby answers nothing. The list is judged by the same rules as
    any other (`MachineTrust.accept`): signed by the root this node pinned,
    newer, and at an epoch above the one its address came with, or nothing
    moves. Returns whether this node moved."""
    trust = app.state.auth_state.trust
    holder = standby_holder(trust.bundle) or getattr(app.state, "standby_holder", None)
    machines = app.state.machine_trust
    if holder is None or holder == app.state.node_identity.record.name:
        return False
    door = next(
        (
            node
            for node in (machines.claims or {}).get("nodes") or []
            if isinstance(node, dict) and node.get("name") == holder
        ),
        None,
    )
    machine_url = door.get("machineUrl") if door else None
    if not isinstance(machine_url, str) or not machine_url.startswith("https://"):
        log.debug("the standby's machine %s has no machine port on record to ask", holder)
        return False
    url = machine_url.rstrip("/") + "/control" + LIST_PATH
    try:
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            if refused_pin(exc) is None:
                raise
            # Row 4 review: that machine changed its key while this node could
            # not hear. Its root's list is public, signed by the identity this
            # node pinned and versioned, so it is fetched once without the
            # key and taken only if it verifies (as `machine_trust.pull` does
            # for the root's own address).
            log.warning(
                "%s, which held the standby grant, presented a key this node does not hold "
                "for it; checking its root's signed machine list",
                holder,
            )
            response = await _unpinned(machine_url.replace("https://", "http://", 1) + LIST_PATH)
        if response.status_code != 200:
            log.info(
                "asked %s, which held the standby grant, for the root: it answered %d",
                holder,
                response.status_code,
            )
            return False
        moved = bool(machines.accept(str(response.json()["jws"])))
        if moved:
            log.warning(
                "this node could not reach its root, asked %s, which held the standby "
                "grant, and follows the root it named",
                holder,
            )
        return moved
    except MachineListError as exc:
        log.warning("refused the machine list %s sent: %s", holder, exc)
    except (httpx.HTTPError, ValueError, KeyError, TypeError, FencedError) as exc:
        log.info("asked %s, which held the standby grant, for the root: %s", holder, exc)
    return False


async def after_root_moved(app: Any) -> None:
    """Row 4: what a node does once it follows a promoted root: the standby
    it runs is brought in line (its own copy may be that root), it announces
    its address and machine key to the new root, which then records them,
    and it pulls the bundle there. Never raises."""
    from . import enrollment, machine_port, node_identity

    app.state.root_silent_since = None
    try:
        await standby.reconcile(app)
        store = app.state.node_identity
        advertise = node_identity.effective_advertise_url(
            app.state.agent_state.get_config("advertiseUrl"), store.record.advertise_url
        )
        if advertise:
            machines = app.state.machine_trust
            await enrollment.announce_address(
                store=store,
                url=advertise,
                transport=getattr(app.state, "control_transport", None),
                machine=machine_port.announced(app) or {},
                old_root=lambda: machines.root_has_list is False,
            )
        await ask_now(app)
    except Exception:  # pragma: no cover - defensive
        log.exception("could not finish following the root to its new address")


async def _pull_once(app: Any, client: httpx.AsyncClient) -> Heard:
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
        response = await bundle_response(client, control_url, app.state.machine_trust)
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
            # Row 4: who to ask if this root goes quiet. A promotion takes
            # the grant, so the last bundle that named a holder is kept.
            app.state.standby_holder = standby_holder(taken) or getattr(
                app.state, "standby_holder", None
            )
            await standby.reconcile(app)
            heard = Heard.LISTED
        # The machine list rides the same minute (LT1). Public keys and
        # addresses, signed; no credential is sent.
        machines = app.state.machine_trust
        list_before = machines.taken_at
        await pull_machine_list(machines, client, record.control_url)
        list_taken = machines.taken_at is not None and machines.taken_at != list_before
        if list_taken and standby.unmarked_root(app):
            # Row 4: a list that names this machine the root marks the copy
            # it runs promoted (`standby._named_root`).
            await standby.reconcile(app)
    except BundleRollback as exc:
        log.debug("the control root's bundle is older than the one held: %s", exc)
        app.state.root_obsolete = True
        app.state.root_silent_cause = f"it answers for an older root ({exc})"
        return Heard.UNTRUSTED
    except FencedError as exc:
        log.warning("refused the control root's trust bundle: %s", exc)
        app.state.root_obsolete = True
        app.state.root_silent_cause = f"it answers for an older root ({exc})"
        return Heard.UNTRUSTED
    except tokens.BundleError as exc:
        log.warning("refused the control root's trust bundle: %s", exc)
        return Heard.UNTRUSTED
    except Exception as exc:
        log.debug("could not pull the trust bundle: %s", exc)
        app.state.root_silent_cause = _cause(exc)
        return Heard.NO_ANSWER
    if heard is Heard.NO_ANSWER:
        app.state.root_silent_cause = f"it answered {response.status_code}"
    if taken is not None and held is not None:
        await _leave_if_removed(app, taken, list_taken=list_taken, expect=expect)
    if not trust.enrolled:
        # This pull, or another one at the same moment, left.
        return Heard.LEFT
    return heard


def _cause(exc: BaseException) -> str:
    """A refusal in a few words: `connection refused`, `timed out`."""
    if isinstance(exc, httpx.TimeoutException):
        return "timed out"
    text = str(exc).strip()
    return text[:200] if text else type(exc).__name__


async def ask_now(app: Any) -> Heard:
    """`pull_once` outside the minute's loop, with a client of its own."""
    transport = getattr(app.state, "control_transport", None)
    async with internal_client(timeout=PULL_TIMEOUT_SECONDS, transport=transport) as client:
        return await pull_once(app, client)
