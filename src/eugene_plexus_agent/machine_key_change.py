"""Changing this machine's TLS key with no stop in traffic (row 3).

`specs/docs/design/rows/row-03.md`. Two pins across an overlap:

1. **Start** (`POST /v1/node/machine-key`). The node must be enrolled, its
   machine port listening, and the root's signed list must say `keyChange`
   (a root that takes `tlsNextPin`). A P-256 key is made and kept in
   `node.yaml` as `tlsNext*`, with the change's state, in one write.
2. **On the root only,** the door file control reads (`machine_door.json`,
   `EUGENE_PLEXUS_CONTROL_MACHINE_DOOR_FILE`) becomes `[current, next]`,
   before the announcement: a list must never name a key for the root's
   own node that its `controlPins` do not, or other machines ignore the
   root's own entry. Join codes keep the current key until the switch.
3. **Announce both** (`tlsNextPin`, under this node's signature).
4. **Wait** until the list this node holds names both, and the root's
   `GET /v1/nodes/{own}` says no reachable machine holds an older one
   (`machineKeyChange.waitingOn`), or 3 minutes have passed since this
   node's list first named both. A root that does not answer is waited
   for: the old key keeps serving, and the change goes on once a root
   answers (a promoted standby holds both pins too: they are replicated).
5. **Switch.** The machine port loads the new key into its live context
   (new handshakes get it; open sessions finish on the old one), then
   `node.yaml` swaps the keys in one write, which deletes the old private
   key. The client doors ask for a new leaf at once. On the root the door
   file becomes `[new, old]`.
6. **Drop.** Announce the new key alone; once the root records it, the
   root's door file becomes `[new]`. Until a root records the drop, the
   list keeps both, so nobody is refused in between.

**A machine that was offline for the whole change** still holds a list
with only the old key. For another node's key that is harmless: it reaches
the root as before and takes the new list there. For the root's own key,
its pinned call to the root is refused, so it fetches the root's signed
machine list once in plain HTTP and takes it only if it verifies against
the root's identity key, which never changes (`machine_trust.pull`,
`leaving.pull_once`). No person is needed.

A change that is stopped before the switch, or that the root refuses,
deletes the next key and announces the current key alone. Nothing here
ever runs on a schedule: keys change when an operator asks (R3-SCHEDULE).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from . import enrollment
from ._http import internal_client, refused_pin
from ._private_files import write_private
from .node_identity import IdentityRecord, NodeIdentityStore, machine_pin

log = logging.getLogger(__name__)

#: One step of the change at a time.
TICK_SECONDS = 15.0
#: How long a reachable machine that holds an older list is waited for. Covers
#: a locked root, which polls no machine, and agents older than row 3, which
#: report no list version.
WAIT_BOUND_SECONDS = 180.0
#: Control reads the door file again at most once a second.
DOOR_SETTLE_SECONDS = 1.5
#: Beside `node.yaml`; control is handed its path.
DOOR_FILE = "machine_door.json"

WAITING, SWITCHED, DONE, STOPPED, FAILED = "waiting", "switched", "done", "stopped", "failed"
RUNNING = frozenset({WAITING, SWITCHED})


class KeyChangeRefused(Exception):
    """A start or a stop that cannot happen now; the route answers with it."""

    def __init__(self, status: int, slug: str, title: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.slug, self.title, self.detail = status, slug, title, detail


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _current_pin(record: IdentityRecord) -> str | None:
    return machine_pin(record.tls_certificate) if record.tls_certificate else None


def announced_next_pin(record: IdentityRecord) -> str | None:
    """The next key this node announces, or None. Only once the change is
    announcing: on the root the door file goes first (`step`)."""
    change = record.machine_key_change or {}
    if (
        change.get("state") == WAITING
        and change.get("announcing")
        and record.tls_next_certificate
        and isinstance(change.get("nextPin"), str)
    ):
        return str(change["nextPin"])
    return None


def machine_fields(record: IdentityRecord, machine_url: str, pin: str) -> dict[str, str]:
    """`machineUrl` and `tlsPin` for an announcement, and `tlsNextPin` while
    the change announces one (`machine_port.announced`)."""
    fields = {"machineUrl": machine_url, "tlsPin": pin}
    next_pin = announced_next_pin(record)
    if next_pin is not None and next_pin != pin:
        fields["tlsNextPin"] = next_pin
    return fields


def door_pins(record: IdentityRecord) -> list[str]:
    """The keys the machine port in front of the root may present, the one
    it presents first. A key the root may still list for this node stays
    until the root has recorded it gone."""
    current = _current_pin(record)
    if current is None:
        return []
    change = record.machine_key_change or {}
    state = change.get("state")
    other: Any = None
    if state == WAITING:
        other = change.get("nextPin")
    elif state == SWITCHED and not change.get("dropAnnounced"):
        other = change.get("previousPin")
    elif state in (STOPPED, FAILED) and not change.get("settled"):
        other = change.get("nextPin")
    if isinstance(other, str) and other and other != current:
        return [current, other]
    return [current]


def status(record: IdentityRecord) -> dict[str, Any] | None:
    """`MachinePort.keyChange`."""
    change = record.machine_key_change
    if not change or change.get("state") not in (*RUNNING, DONE, STOPPED, FAILED):
        return None
    return {
        "state": change["state"],
        "nextPin": change.get("nextPin") if change["state"] in RUNNING else None,
        "startedAt": change.get("startedAt"),
        "finishedAt": change.get("finishedAt"),
        "error": change.get("error"),
    }


@dataclass
class Wiring:
    """What a change needs of the agent, so a test can hand in its own."""

    store: NodeIdentityStore
    trust: Any
    """`machine_trust.MachineTrust`: the list this node holds."""
    door: Callable[[], Any]
    """The `machine_port.MachinePort`, or None when it did not start."""
    fields: Callable[[], dict[str, str] | None]
    """`machineUrl`, `tlsPin` and, while announcing, `tlsNextPin`."""
    advertise: Callable[[], str | None]
    """This node's agent address, as announced."""
    hosts_control: Callable[[], bool]
    token: Callable[[], str]
    """This node's own token for reads at the root."""
    door_file: Path
    machine_url: Callable[[], str | None] = lambda: None
    transport: Callable[[], Any] = lambda: None
    """The test seam for calls to the root (`app.state.control_transport`)."""
    pull: Callable[[], Awaitable[None]] | None = None
    """Take the root's machine list now."""
    on_switched: Callable[[], None] | None = None


class MachineKeyChange:
    """The change, one step at a time, for the agent's life."""

    def __init__(self, wiring: Wiring) -> None:
        self.w = wiring
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._door_written: str | None = None
        #: The last step's failure when `node.yaml` could not hold it either
        #: (a full disk): reported until a step gets past it.
        self._fault: str | None = None

    # ---- what GET /v1/node says --------------------------------------------

    def status(self) -> dict[str, Any] | None:
        held = status(self.w.store.record)
        if held is not None and self._fault is not None:
            held["error"] = self._fault
        return held

    # ---- the root's door file ------------------------------------------------

    def write_door(self) -> None:
        """On the machine that runs the root, the machine port's keys for
        control (`door_pins`); nothing elsewhere. Rewritten only when it
        changes. Owner-only, beside `node.yaml`: whoever can write it can
        already read the machine key itself."""
        if not self.w.hosts_control() or self.w.door() is None:
            return
        pins = door_pins(self.w.store.record)
        if not pins:
            return
        rendered = json.dumps({"pins": pins, "url": self.w.machine_url()}, sort_keys=True)
        if rendered == self._door_written:
            return
        try:
            write_private(self.w.door_file, rendered)
        except OSError as exc:
            log.warning("could not write the machine door file %s: %s", self.w.door_file, exc)
            return
        self._door_written = rendered
        log.info("control is told the machine port presents %s", ", ".join(pins))

    # ---- start and stop --------------------------------------------------------

    async def start(self) -> dict[str, Any]:
        async with self._lock:
            store = self.w.store
            record = store.record
            if not record.enrolled:
                raise KeyChangeRefused(
                    409,
                    "not-enrolled",
                    "Not part of an install",
                    "This machine has joined no install, so no other machine holds its key.",
                )
            door = self.w.door()
            if door is None or not door.status.listening:
                why = getattr(getattr(door, "status", None), "error", None)
                raise KeyChangeRefused(
                    409,
                    "machine-port-down",
                    "Machine port not listening",
                    "This machine's machine port is not listening"
                    + (f" ({why})" if why else "")
                    + ", so it presents no key to change.",
                )
            if (self.w.trust.claims or {}).get("keyChange") is not True:
                raise KeyChangeRefused(
                    409,
                    "root-too-old",
                    "The root cannot change machine keys",
                    "The root runs a version that does not take a next machine key. Update the "
                    "machine that runs the root first.",
                )
            change = record.machine_key_change or {}
            if change.get("state") in RUNNING:
                raise KeyChangeRefused(
                    409,
                    "already-changing",
                    "Machine key already changing",
                    f"This machine's key change started at {change.get('startedAt')} is still "
                    "running.",
                )
            if self.w.fields() is None:
                raise KeyChangeRefused(
                    409,
                    "no-machine-url",
                    "No machine address",
                    "This machine advertises no address, so no other machine reaches it over its "
                    "machine port.",
                )
            store.begin_machine_key_change(
                {
                    "state": WAITING,
                    "startedAt": _now(),
                    "finishedAt": None,
                    "error": None,
                    "previousPin": door.status.pin,
                }
            )
            self.write_door()
            log.warning(
                "changing this machine's key: the next key is %s; the machine port presents %s "
                "until every machine holds a list that names both",
                store.record.machine_key_change and store.record.machine_key_change["nextPin"],
                door.status.pin,
            )
        self._wake.set()
        return self.status() or {}

    async def stop(self) -> dict[str, Any]:
        async with self._lock:
            store = self.w.store
            change = store.record.machine_key_change or {}
            state = change.get("state")
            if state not in RUNNING:
                raise KeyChangeRefused(
                    404,
                    "no-key-change",
                    "No machine key change",
                    "No machine key change is running on this machine.",
                )
            if state == SWITCHED:
                raise KeyChangeRefused(
                    409,
                    "already-switched",
                    "Machine key already switched",
                    "This machine presents its new key and the old one is deleted, so the change "
                    "cannot be stopped; it finishes by itself.",
                )
            store.drop_next_machine_key(
                state=STOPPED,
                finishedAt=_now(),
                error=None,
                announcing=False,
                settled=False,
            )
            log.warning("stopped this machine's key change; the next key is deleted")
            await self._settle()
        return self.status() or {}

    # ---- the loop --------------------------------------------------------------

    async def run(self) -> None:
        while True:
            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception:  # never let the loop die; the machine keeps its key
                log.exception("the machine key change step failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), TICK_SECONDS)
            self._wake.clear()

    async def step(self) -> None:
        async with self._lock:
            self._fault = None
            try:
                self.write_door()
                record = self.w.store.record
                change = record.machine_key_change or {}
                if not record.enrolled:
                    return
                state = change.get("state")
                if state == WAITING:
                    await self._wait(record, change)
                elif state == SWITCHED:
                    await self._drop()
                elif state in (STOPPED, FAILED) and not change.get("settled"):
                    await self._settle()
            except OSError as exc:
                # node.yaml could not be written: nothing moved past what it
                # holds, and the cause is shown until a step gets past it.
                self._fault = f"could not write {self.w.store.path.name} ({exc})"
                log.warning("the machine key change is held: %s", self._fault)

    # ---- the steps -------------------------------------------------------------

    def _update(self, **fields: Any) -> None:
        self.w.store.update_machine_key_change(**fields)

    async def _announce(self) -> enrollment.AnnounceOutcome:
        fields, url = self.w.fields(), self.w.advertise()
        if fields is None or url is None:
            return enrollment.AnnounceOutcome(
                False, False, url, "this machine advertises no address", retry=True
            )
        return await enrollment.announce_address(
            store=self.w.store, url=url, transport=self.w.transport(), machine=fields
        )

    def _listed(self, current: str, next_pin: str) -> bool:
        """The list this node holds names both keys for it (and, on the
        root, in `controlPins`)."""
        both = tuple(sorted({current, next_pin}))
        own = self.w.trust.own_record()
        if own is None or own[1] != both:
            return False
        if self.w.hosts_control():
            control = set((self.w.trust.claims or {}).get("controlPins") or ())
            return set(both) <= control
        return True

    async def _observe(self) -> tuple[dict[str, Any] | None, str | None]:
        """The root's record of this node, or why there is none."""
        record = self.w.store.record
        url = f"{str(record.control_url).rstrip('/')}/v1/nodes/{record.name}"
        try:
            token = self.w.token()
            async with internal_client(timeout=10.0, transport=self.w.transport()) as client:
                response = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as exc:
            pin = refused_pin(exc)
            if pin is not None:
                return None, f"the root presented a key this machine does not hold ({pin})"
            return None, str(exc) or type(exc).__name__
        except Exception as exc:  # a token this node cannot make
            return None, str(exc)
        if response.status_code != 200:
            return None, f"{response.status_code}: {enrollment.problem_detail(response)}"
        try:
            body = response.json()
        except ValueError:
            return None, "the root answered with no JSON"
        return (body, None) if isinstance(body, dict) else (None, "the root answered no record")

    async def _wait(self, record: IdentityRecord, change: dict[str, Any]) -> None:
        current, next_pin = _current_pin(record), change.get("nextPin")
        if (
            current is None
            or not isinstance(next_pin, str)
            or not record.tls_next_certificate
            or not record.tls_next_private_key
        ):
            self.w.store.drop_next_machine_key(
                state=FAILED, finishedAt=_now(), error="the next key was lost", settled=False
            )
            return
        if not change.get("announcing"):
            if self.w.hosts_control():
                # The door file already names both; control reads it again
                # at most once a second.
                await asyncio.sleep(DOOR_SETTLE_SECONDS)
            self._update(announcing=True)
        if not change.get("announced"):
            outcome = await self._announce()
            if not outcome.announced:
                if outcome.retry:
                    self._update(error=f"waiting for the root ({outcome.detail})")
                    return
                log.warning("the root refused this machine's next key: %s", outcome.detail)
                self.w.store.drop_next_machine_key(
                    state=FAILED,
                    finishedAt=_now(),
                    error=f"the root refused it ({outcome.detail})",
                    announcing=False,
                    settled=False,
                )
                await self._settle()
                return
            self._update(announced=True, error=None)
            if self.w.pull is not None:
                # The list now, not at the minute: this node waits for it.
                await self.w.pull()
        if not self._listed(current, next_pin):
            if self.w.pull is not None:
                await self.w.pull()
            if not self._listed(current, next_pin):
                self._update(error="waiting for the root's machine list to name both keys")
                return
        listed_at = change.get("listedAt")
        if not isinstance(listed_at, int | float):
            listed_at = time.time()
            self._update(listedAt=listed_at)
        observed, why = await self._observe()
        if observed is None:
            # The root does not answer: no switch. The old key keeps serving.
            self._update(error=f"waiting for the root ({why})")
            return
        if sorted(observed.get("tlsPins") or ()) != sorted({current, next_pin}):
            self._update(announced=False, error="waiting for the root to record both keys")
            return
        seen = observed.get("machineKeyChange") or {}
        not_ready = [str(n) for n in seen.get("notReady") or ()]
        if not_ready and self.w.hosts_control():
            # Review blocker 2: an agent older than row 3 that misses the
            # root's key change never takes the new one. No bound here: the
            # old key keeps serving until they are updated or removed.
            self._update(
                error=f"waiting: {', '.join(not_ready)} "
                + ("runs" if len(not_ready) == 1 else "run")
                + " a version that cannot take this machine's new key if "
                + ("it misses" if len(not_ready) == 1 else "they miss")
                + " the change; update "
                + ("it" if len(not_ready) == 1 else "them")
                + ", or remove a machine that is gone"
            )
            return
        waiting = [str(n) for n in seen.get("waitingOn") or ()]
        if waiting and time.time() - float(listed_at) < WAIT_BOUND_SECONDS:
            self._update(error=None)
            return
        if waiting:
            log.warning(
                "switching this machine's key with %s still holding an older list (3 minutes "
                "passed); they take the new key from the root's list",
                ", ".join(waiting),
            )
        await self._switch(current, next_pin)

    async def _switch(self, current: str, next_pin: str) -> None:
        record = self.w.store.record
        door = self.w.door()
        if door is None:
            self._update(error="the machine port is not running")
            return
        try:
            door.reload(
                str(record.tls_next_certificate), str(record.tls_next_private_key), next_pin
            )
        except Exception as exc:  # the old key stays loaded
            log.error("the next machine key did not load, so nothing switched: %s", exc)
            self._update(error=f"the next key did not load ({exc})")
            return
        try:
            self.w.store.switch_machine_key(
                state=SWITCHED,
                previousPin=current,
                switchedAt=_now(),
                announcing=False,
                error=None,
            )
        except Exception as exc:
            # `node.yaml` still names the old key, and so does this process
            # (it holds a record only once it is written): the port presents
            # the old key again, or a restart would present a key the root's
            # list has dropped.
            door.reload(str(record.tls_certificate), str(record.tls_private_key), current)
            self._fault = f"nothing switched: could not write {self.w.store.path.name} ({exc})"
            log.error("could not keep the new machine key, so nothing switched: %s", exc)
            return
        self.write_door()
        log.warning(
            "this machine's machine port presents its new key %s; the old key %s is deleted",
            next_pin,
            current,
        )
        if self.w.on_switched is not None:
            try:
                self.w.on_switched()
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("after the machine key switch: %s", exc)
        await self._drop()

    async def _drop(self) -> None:
        change = self.w.store.record.machine_key_change or {}
        if not change.get("dropAnnounced"):
            outcome = await self._announce()
            if not outcome.announced:
                self._update(error=f"waiting for the root to drop the old key ({outcome.detail})")
                return
            self._update(dropAnnounced=True, error=None)
            # The root recorded the new key alone: the door file follows.
            self.write_door()
        # Review blocker 4: done only once a list the root signed, verified
        # here, no longer names the old key. A root restarted sealed takes
        # the drop but signs nothing, and keeps serving both keys.
        if self.w.pull is not None:
            await self.w.pull()
        if self._dropped():
            self._update(state=DONE, finishedAt=_now(), error=None, settled=True)
            log.info("machine key change done: the root's signed list names the new key alone")
            return
        if await self._root_locked():
            self._update(
                error="waiting for the root to be unlocked: its machine list still names "
                "the old key"
            )
        else:
            self._update(error="waiting for the root's machine list to drop the old key")

    def _dropped(self) -> bool:
        """The list this node holds names its current key alone (and, on
        the root, in `controlPins`)."""
        current = _current_pin(self.w.store.record)
        own = self.w.trust.own_record()
        if current is None or own is None or own[1] != (current,):
            return False
        if self.w.hosts_control():
            return list((self.w.trust.claims or {}).get("controlPins") or ()) == [current]
        return True

    async def _root_locked(self) -> bool:
        """The root says it is sealed (`GET /v1/auth/status`)."""
        record = self.w.store.record
        url = f"{str(record.control_url).rstrip('/')}/v1/auth/status"
        try:
            async with internal_client(timeout=10.0, transport=self.w.transport()) as client:
                response = await client.get(url)
            return response.status_code == 200 and response.json().get("unlocked") is False
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def _settle(self) -> None:
        """After a stop or a refusal: the current key alone, at the root and
        then in the door file."""
        outcome = await self._announce()
        if outcome.announced:
            self._update(settled=True)
            self.write_door()


def from_app(app: Any) -> MachineKeyChange:
    """The change, wired to a running agent."""
    from . import machine_port, node_identity, standby
    from .machine_trust import pull

    settings = app.state.settings
    state = app.state.agent_state
    store: NodeIdentityStore = app.state.node_identity

    def advertise() -> str | None:
        return node_identity.effective_advertise_url(
            state.get_config("advertiseUrl"), store.record.advertise_url
        )

    async def pull_now() -> None:
        record = store.record
        if not record.control_url:
            return
        transport = getattr(app.state, "control_transport", None)
        async with internal_client(timeout=10.0, transport=transport) as client:
            await pull(app.state.machine_trust, client, str(record.control_url))

    def switched() -> None:
        # The client doors' leaf is for the old key: ask for one now
        # (`client_tls.wanted` names the change).
        tls = getattr(app.state, "client_tls", None)
        if tls is not None and (reason := tls.wanted()) is not None:
            asyncio.get_running_loop().create_task(tls.request(reason))

    return MachineKeyChange(
        Wiring(
            store=store,
            trust=app.state.machine_trust,
            door=lambda: getattr(app.state, "machine_port", None),
            fields=lambda: machine_port.announced(app),
            advertise=advertise,
            hosts_control=lambda: standby.hosts_active_control(state),
            token=lambda: app.state.auth_state.trust.agent_token("control"),
            door_file=settings.config_file.resolve().parent / DOOR_FILE,
            machine_url=lambda: machine_port.machine_url(app),
            transport=lambda: getattr(app.state, "control_transport", None),
            pull=pull_now,
            on_switched=switched,
        )
    )


__all__ = [
    "DONE",
    "DOOR_FILE",
    "FAILED",
    "RUNNING",
    "STOPPED",
    "SWITCHED",
    "WAITING",
    "KeyChangeRefused",
    "MachineKeyChange",
    "Wiring",
    "announced_next_pin",
    "door_pins",
    "from_app",
    "machine_fields",
    "status",
]
