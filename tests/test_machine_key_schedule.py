"""Part 3b: a machine key changes once a year by itself, and on request.

Troy's calls R3-SCHEDULE and R3-SCHEDULE-2 (`rows/row-03.md`). A machine key
is self-signed and pinned by its SPKI hash, so it never expires by itself;
Eugene changes it when it is older than `machineKeyChangeDays` (365 by
default, 0 for only on request). The change is row 3's: two pins across an
overlap, with nothing refused. The machine port and the root are the same
real port and fake root as row 3's tests.

Every time here is deterministic: the change's clock is injected, and the
spreads are checked on fixed pins, never on random keys.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_agent import machine_key_change, node_identity
from eugene_plexus_agent.machine_key_change import (
    CATCH_UP_WINDOW,
    catch_up_offset,
    due_after,
)

from .test_machine_key_change import Node

UTC = datetime.UTC
DAY = datetime.timedelta(days=1)

#: Fixed pins, the form of a real one: base64url SHA-256, 43 characters.
PINS = [
    base64.urlsafe_b64encode(hashlib.sha256(f"key-{i}".encode()).digest()).rstrip(b"=").decode()
    for i in range(24)
]


def _set_made(node: Node, made: datetime.datetime) -> None:
    """The machine key as one made at `made`: the same key (the same pin),
    its certificate dated then."""
    record = node.store.record
    aged = node_identity.machine_certificate(record.tls_private_key or "", made=made)
    assert node_identity.machine_pin(aged) == node.old_pin
    node.store._write_locked(replace(record, tls_certificate=aged))


def _due_ago(node: Node, ago: datetime.timedelta, days: int = 365) -> datetime.datetime:
    """Date the key so that it fell due `ago` before now. Returns that time."""
    now = datetime.datetime.now(UTC).replace(microsecond=0)
    spread = due_after(now, node.old_pin, days) - now
    _set_made(node, now - spread - ago)
    due = node.change.schedule()["keyChangeDueAt"]
    assert due is not None and abs(due - (now - ago)) < datetime.timedelta(seconds=2)
    return due


def _clock(node: Node, at: datetime.datetime) -> None:
    node.change.w.now = lambda: at


@pytest.fixture
def lifetime() -> dict[str, int]:
    return {"days": machine_key_change.DEFAULT_KEY_DAYS}


def _wire(node: Node, lifetime: dict[str, int]) -> None:
    node.change.w.lifetime_days = lambda: lifetime["days"]


# --------------------------------------------------------------------------- #
# The day a key falls due, and the catch-up after an update
# --------------------------------------------------------------------------- #


def test_the_machines_of_one_install_do_not_change_on_the_same_day() -> None:
    """Troy's call: a yearly timer per machine, spread by a random offset.
    Keys made the same day fall due on different days, each within a month
    after its year; one key's day does not move."""
    made = datetime.datetime(2026, 3, 1, tzinfo=UTC)
    dues = [due_after(made, pin, 365) for pin in PINS[:12]]
    for due in dues:
        assert made + datetime.timedelta(days=365) <= due
        assert due < made + datetime.timedelta(days=365 + 365 / 12)
    assert [due_after(made, pin, 365) for pin in PINS[:12]] == dues
    assert len({d.date() for d in dues}) >= 10, "twelve keys of one day fall due on 10 days"
    assert due_after(made, PINS[0], 30) - made < datetime.timedelta(days=32.5)


def test_overdue_keys_are_caught_up_over_a_week_not_at_once() -> None:
    """The review: an install updated with every key 400 days old must not
    change every machine in one tick. Each overdue key starts within a week,
    at an hour drawn from its pin."""
    offsets = [catch_up_offset(pin) for pin in PINS]
    assert all(datetime.timedelta(0) <= o < CATCH_UP_WINDOW for o in offsets)
    assert offsets == [catch_up_offset(pin) for pin in PINS]
    assert len({int(o / datetime.timedelta(hours=1)) for o in offsets}) >= 20
    assert sum(1 for o in offsets if o < DAY) <= 6, "a quarter at most starts on day one"


async def test_a_key_overdue_when_first_seen_starts_within_its_catch_up_window(
    tmp_path: Path, lifetime: dict[str, int]
) -> None:
    async with Node(tmp_path) as node:
        _wire(node, lifetime)
        _due_ago(node, datetime.timedelta(days=35))  # an install just updated
        now = datetime.datetime.now(UTC)
        _clock(node, now)
        await node.change.step()
        assert node.change.status() is None, "not at the first check"
        start = node.change.schedule()["keyChangeNextAttemptAt"]
        expected = now + catch_up_offset(node.old_pin)
        assert start is not None and abs(start - expected) < datetime.timedelta(seconds=2)
        assert node.change.schedule()["keyChangeBlocked"] is None
        _clock(node, start - datetime.timedelta(minutes=1))
        await node.change.step()
        assert node.change.status() is None
        _clock(node, start + datetime.timedelta(seconds=1))
        await node.change.step()
        status = node.change.status()
        assert status is not None and status["reason"] == "scheduled"


async def test_a_key_older_than_a_year_changes_by_itself(
    tmp_path: Path, lifetime: dict[str, int]
) -> None:
    """The yearly case: the agent was running when the key fell due, so it
    starts at the next check."""
    async with Node(tmp_path) as node:
        _wire(node, lifetime)
        due = _due_ago(node, datetime.timedelta(minutes=1))
        assert node.change.schedule()["keyChangeNextAttemptAt"] == due
        await node.change.step()
        status = node.change.status()
        assert status is not None and status["state"] == "waiting"
        assert status["reason"] == "scheduled"
        assert node.change.schedule()["keyChangeNextAttemptAt"] is None, "under way"
        next_pin = status["nextPin"]
        node.control.waiting_on = []
        for _ in range(3):
            await node.change.step()
        assert node.state() == "done"
        assert await node.presented() == next_pin
        # The new key is new: due again a year from now, not at once.
        due = node.change.schedule()["keyChangeDueAt"]
        assert due is not None and due > datetime.datetime.now(UTC) + 364 * DAY
        await node.change.step()
        assert node.state() == "done"
        assert node.control.pins == [next_pin]


async def test_a_young_key_is_left_alone(tmp_path: Path, lifetime: dict[str, int]) -> None:
    async with Node(tmp_path) as node:
        _wire(node, lifetime)
        _set_made(node, datetime.datetime.now(UTC) - 300 * DAY)
        await node.change.step()
        assert node.change.status() is None
        assert node.control.announcements == []
        schedule = node.change.schedule()
        assert abs((datetime.datetime.now(UTC) - schedule["keyMadeAt"]).days - 300) <= 1
        assert schedule["keyChangeNextAttemptAt"] == schedule["keyChangeDueAt"]


async def test_zero_days_means_only_on_request(tmp_path: Path, lifetime: dict[str, int]) -> None:
    async with Node(tmp_path) as node:
        lifetime["days"] = 0
        _wire(node, lifetime)
        _set_made(node, datetime.datetime.now(UTC) - 4000 * DAY)
        schedule = node.change.schedule()
        assert schedule["keyChangeDueAt"] is None and schedule["keyChangeNextAttemptAt"] is None
        await node.change.step()
        assert node.change.status() is None
        # On request it still changes, and says so.
        started = await node.change.start()
        assert started["reason"] == "requested"


# --------------------------------------------------------------------------- #
# A start that cannot happen: said once, shown, and tried again later
# --------------------------------------------------------------------------- #


async def test_a_scheduled_start_that_cannot_happen_says_why_and_when_it_tries_again(
    tmp_path: Path, lifetime: dict[str, int], caplog: pytest.LogCaptureFixture
) -> None:
    async with Node(tmp_path) as node:
        _wire(node, lifetime)
        _due_ago(node, datetime.timedelta(minutes=1))
        claims: dict[str, Any] = dict(node.trust.claims or {})
        claims.pop("keyChange")
        node.trust.claims = claims  # a root older than row 3
        now = datetime.datetime.now(UTC)
        _clock(node, now)
        await node.change.step()
        await node.change.step()
        assert node.change.status() is None
        said = [r for r in caplog.records if "cannot start" in r.getMessage()]
        assert len(said) == 1, "said once, not every step"
        schedule = node.change.schedule()
        assert "Update the machine that runs the root" in str(schedule["keyChangeBlocked"])
        assert schedule["keyChangeNextAttemptAt"] == now + datetime.timedelta(hours=1)
        node.trust.claims = {**claims, "keyChange": True}  # the root is updated
        await node.change.step()
        assert node.change.status() is None, "not before the hour has passed"
        _clock(node, now + datetime.timedelta(hours=1, seconds=1))
        await node.change.step()
        status = node.change.status()
        assert status is not None and status["reason"] == "scheduled"
        assert node.change.schedule()["keyChangeBlocked"] is None


async def test_a_key_that_cannot_be_written_is_throttled_and_shown(
    tmp_path: Path,
    lifetime: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The review: a full disk at the start of a scheduled change retried every
    step with a log line each time, and showed nothing."""
    async with Node(tmp_path) as node:
        _wire(node, lifetime)
        _due_ago(node, datetime.timedelta(minutes=1))
        real = node_identity.write_private

        def full(path: Path, text: str) -> None:
            if path.name == "node.yaml":
                raise OSError(28, "No space left on device")
            real(path, text)

        monkeypatch.setattr(node_identity, "write_private", full)
        now = datetime.datetime.now(UTC)
        _clock(node, now)
        for _ in range(3):
            await node.change.step()
        said = [r for r in caplog.records if "cannot start" in r.getMessage()]
        assert len(said) == 1
        schedule = node.change.schedule()
        assert "No space left on device" in str(schedule["keyChangeBlocked"])
        assert schedule["keyChangeNextAttemptAt"] == now + datetime.timedelta(hours=1)
        assert node.store.record.tls_next_private_key is None
        monkeypatch.setattr(node_identity, "write_private", real)
        _clock(node, now + datetime.timedelta(hours=1, seconds=1))
        await node.change.step()
        assert (node.change.status() or {}).get("state") == "waiting"


async def test_a_stopped_scheduled_change_says_it_starts_again_a_day_later(
    tmp_path: Path, lifetime: dict[str, int]
) -> None:
    async with Node(tmp_path) as node:
        _wire(node, lifetime)
        _due_ago(node, datetime.timedelta(minutes=1))
        await node.change.step()
        assert node.state() == "waiting"
        stopped = await node.change.stop()
        finished = datetime.datetime.fromisoformat(stopped["finishedAt"])
        await node.change.step()
        assert node.state() == "stopped"
        assert node.change.schedule()["keyChangeNextAttemptAt"] == finished + DAY
        _clock(node, finished + DAY + datetime.timedelta(seconds=1))
        await node.change.step()
        assert node.state() == "waiting"
        assert (node.change.status() or {}).get("reason") == "scheduled"


# --------------------------------------------------------------------------- #
# What GET /v1/node says, and the setting
# --------------------------------------------------------------------------- #


def test_get_node_says_when_the_key_changes_and_why_it_could_not(
    authed_client: Any, tmp_path: Path
) -> None:
    """What the console shows under each machine (`MachinePort.keyMadeAt`,
    `keyChangeDueAt`, `keyChangeNextAttemptAt`, `keyChangeBlocked`), read from
    the running change."""
    from types import SimpleNamespace

    from eugene_plexus_agent.machine_key_change import MachineKeyChange, Wiring

    app = authed_client.app
    store = app.state.node_identity
    record = store.ensure_keypair()
    made = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    aged = node_identity.machine_certificate(record.tls_private_key or "", made=made)
    store._write_locked(replace(record, tls_certificate=aged))
    pin = node_identity.machine_pin(aged)
    app.state.machine_port = SimpleNamespace(
        status=SimpleNamespace(port=18083, pin=pin, listening=True, error=None)
    )
    change = MachineKeyChange(
        Wiring(
            store=store,
            trust=app.state.machine_trust,
            door=lambda: app.state.machine_port,
            fields=lambda: None,
            advertise=lambda: None,
            hosts_control=lambda: False,
            token=lambda: "t",
            door_file=tmp_path / machine_key_change.DOOR_FILE,
            lifetime_days=lambda: 30,
            now=lambda: made + 5 * DAY,
        )
    )
    app.state.machine_key_change = change
    try:
        machine = authed_client.get("/v1/node").json()["machine"]
    finally:
        app.state.machine_port = None
        app.state.machine_key_change = None
    assert datetime.datetime.fromisoformat(machine["keyMadeAt"]) == made
    due = datetime.datetime.fromisoformat(machine["keyChangeDueAt"])
    assert due == due_after(made, pin, 30)
    assert datetime.datetime.fromisoformat(machine["keyChangeNextAttemptAt"]) == due
    assert machine.get("keyChangeBlocked") is None


def test_the_setting_is_in_the_agent_s_config(authed_client: Any) -> None:
    schema = authed_client.get("/v1/config/schema").json()
    fields = {f["key"]: f for f in schema["fields"]}
    field = fields[machine_key_change.KEY_DAYS_KEY]
    assert field["default"] == 365 and field["minimum"] == 0
    changed = authed_client.patch("/v1/config", json={machine_key_change.KEY_DAYS_KEY: 30})
    assert changed.status_code == 200, changed.text
    assert authed_client.get("/v1/config").json()[machine_key_change.KEY_DAYS_KEY] == 30
