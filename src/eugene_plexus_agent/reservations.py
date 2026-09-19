"""Memory this node has promised to a launch that has not taken it yet.

Admission reads **live free memory** off the device snapshot, which is
the right number and the wrong one: it is what the card reports now, and
a runtime that was admitted three seconds ago has not read a byte of its
weights. So two launches in quick succession both measure against the
same free memory and both are told `fits` -- the second one for memory
the first already spent. The node-local copy widened that window from
seconds to minutes by adding `copying`, during which no process exists
at all (review §6.2 #19, roadmap R3 item 2).

`check_admission` already took `running`, which reads as though it
accounts for the other runtimes on the device. It does not: `_blockers`
turns it into an advisory *list of what else is there* and nothing is
subtracted from anything. The missing piece is not arithmetic, it is a
record of intent -- so this is a ledger, not a formula.

**Three rules, and none of them is in the finding.**

* **A dry run must not reserve.** `POST /v1/runtimes/admission` is a
  question the Library's launch preview asks on every keystroke of the
  context box; if asking reserved, the preview would refuse the launch
  it is previewing. The dry run, a create and a start share one
  `check_admission` call site, so they are told apart at the routes or
  not at all -- which is why nothing here is called from
  `admission.py`: that module is handed the entries and never writes.

* **`copying` needs a reservation before any process exists.** It is the
  longest part of a first launch on a remote mount (measured: 266 s for
  23.8 GB over SMB) and the one state in which there is nothing to
  observe at all.

* **An abandoned launch must not strand memory forever.** A reservation
  is released when the runtime is next observed to be past its start --
  `ready` holds real memory that the snapshot now counts, and a stopped
  or crashed one holds none -- and by a TTL when it is never observed
  again, which is the case a supervisor restart or a vanished topology
  entry produces. Without the TTL the failure mode of this module is a
  card that can never be launched on until the agent is restarted, which
  is worse than the defect it fixes.

**Deliberately in memory only.** A reservation describes a launch this
process started and is meaningless to the next one: a restarted agent
has no in-flight launches, and persisting them would hold a card across
a reboot for runtimes that died with the process. The ledger is empty on
boot and that is correct rather than a gap.

**Over-counting during `loading` is accepted and is the safe
direction.** Halfway through reading the weights the card already shows
part of the model gone, and the full reservation is still counted -- so
the total is high by up to the model's own size for the length of one
load. The error shrinks to nothing as the load finishes, it refuses
rather than admits, and `?force=true` is the override. The alternative
needs a per-engine load-progress number that S7 established does not
exist.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# Long enough that a real launch never hits it: the longest observed
# first start on this project's own install is a 266 s copy of 23.8 GB
# over SMB plus a four-minute load, and a slower link or a bigger model
# scales both. Short enough that an abandoned launch frees the card
# inside a coffee break. Only reached when the runtime is never observed
# again, because an observation releases it immediately.
DEFAULT_TTL_SECONDS = 1800.0


@dataclass(frozen=True)
class Reservation:
    """One launch's claim on a device, as admission measured it."""

    runtime: str
    device_index: int | None
    """The device admission chose. `None` means it could not place the
    launch -- no accelerator, or a pin this host cannot resolve -- and
    such a reservation is counted against **every** device, because the
    memory is spent somewhere and refusing is the recoverable error."""
    size_bytes: int
    at: float
    """`perf_counter()` at the moment of reserving. A duration, so never
    `monotonic()`: on the Python both installers provision that is
    `GetTickCount64` on Windows, a 15.6 ms grid."""


class ReservationLedger:
    """The promises this agent has made, one per runtime name.

    Not thread-safe and not asked to be: every caller is a route handler
    on the one event loop.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._held: dict[str, Reservation] = {}

    def reserve(self, runtime: str, *, device_index: int | None, size_bytes: int) -> None:
        """Record what a launch of `runtime` is about to take.

        Replaces rather than adds: a runtime has one launch in flight,
        and a restart that re-measures must not stack a second claim on
        top of the first.
        """
        if size_bytes <= 0:
            # Nothing was measured -- an `unknown` fit, or a file that
            # could not be sized. A zero-byte promise is not a promise,
            # and recording one would only make `reservedBytes` say 0
            # where absent is the honest answer.
            self._held.pop(runtime, None)
            return
        self._held[runtime] = Reservation(
            runtime=runtime,
            device_index=device_index,
            size_bytes=size_bytes,
            at=self._clock(),
        )

    def release(self, runtime: str) -> None:
        """Hand a promise back: the launch failed, was stopped, or the
        runtime was deleted."""
        self._held.pop(runtime, None)

    def reconcile(self, pending: Iterable[str]) -> None:
        """Keep only the runtimes still on their way up.

        `pending` is the set observed in `copying`, `starting` or
        `loading`. Anything else either holds its memory for real -- in
        which case the device snapshot counts it and counting both would
        refuse a third launch that fits -- or holds none.
        """
        keep = set(pending)
        for name in [n for n in self._held if n not in keep]:
            del self._held[name]

    def entries(self) -> list[Reservation]:
        """The live promises, expired ones dropped on the way out."""
        self._expire()
        return list(self._held.values())

    def held_bytes(self, *, device_index: int | None, exclude: str | None) -> int:
        """What is promised on one device, ignoring one runtime's own
        claim -- a restart re-measures the runtime that already holds
        one, and counting it against itself refuses every restart."""
        return held_bytes(self.entries(), device_index=device_index, exclude=exclude)

    def _expire(self) -> None:
        cutoff = self._clock() - self._ttl
        for name in [n for n, r in self._held.items() if r.at <= cutoff]:
            del self._held[name]


def held_bytes(
    reservations: Iterable[Reservation],
    *,
    device_index: int | None,
    exclude: str | None,
) -> int:
    """The pure half, so admission can sum a list it was handed without
    reaching for the ledger it came from."""
    total = 0
    for reservation in reservations:
        if exclude is not None and reservation.runtime == exclude:
            continue
        if reservation.device_index is not None and reservation.device_index != device_index:
            continue
        total += reservation.size_bytes
    return total


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "Reservation",
    "ReservationLedger",
    "held_bytes",
]
