"""Part 3b: a machine key changes once a year by itself, and on request.

Troy's calls R3-SCHEDULE and R3-SCHEDULE-2 (`rows/row-03.md`). A machine key
is self-signed and pinned by its SPKI hash, so it never expires by itself;
Eugene changes it when it is older than `machineKeyChangeDays` (365 by
default, 0 for only on request). The change is row 3's: two pins across an
overlap, with nothing refused. The machine port and the root are the same
real port and fake root as row 3's tests.
"""

from __future__ import annotations

import datetime
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_agent import machine_key_change, node_identity

from .test_machine_key_change import Node

UTC = datetime.UTC


def _age_key(node: Node, days: float) -> None:
    """The machine key as one made `days` ago: the same key (the same pin),
    its certificate dated then."""
    record = node.store.record
    made = datetime.datetime.now(UTC) - datetime.timedelta(days=days)
    aged = node_identity.machine_certificate(record.tls_private_key or "", made=made)
    assert node_identity.machine_pin(aged) == node.old_pin
    node.store._write_locked(replace(record, tls_certificate=aged))


def _clock(node: Node, days: float) -> None:
    """Move the change's clock `days` ahead."""
    later = datetime.datetime.now(UTC) + datetime.timedelta(days=days)
    node.change.w.now = lambda: later


@pytest.fixture
def lifetime() -> dict[str, int]:
    return {"days": machine_key_change.DEFAULT_KEY_DAYS}


def _wire(node: Node, lifetime: dict[str, int]) -> None:
    node.change.w.lifetime_days = lambda: lifetime["days"]


async def test_a_key_older_than_a_year_changes_by_itself(
    tmp_path: Path, lifetime: dict[str, int]
) -> None:
    async with Node(tmp_path) as node:
        _wire(node, lifetime)
        _age_key(node, 400)
        schedule = node.change.schedule()
        assert schedule["keyChangeDueAt"] is not None
        assert schedule["keyChangeDueAt"] < datetime.datetime.now(UTC)

        await node.change.step()  # due: it starts by itself
        status = node.change.status()
        assert status is not None and status["state"] == "waiting"
        assert status["reason"] == "scheduled"
        next_pin = status["nextPin"]
        node.control.waiting_on = []
        for _ in range(3):
            await node.change.step()
        assert node.state() == "done"
        assert await node.presented() == next_pin
        # The new key is new: due again a year from now, not at once.
        due = node.change.schedule()["keyChangeDueAt"]
        assert due is not None and due > datetime.datetime.now(UTC) + datetime.timedelta(days=364)
        await node.change.step()
        assert node.state() == "done"
        assert node.control.pins == [next_pin]


async def test_a_young_key_is_left_alone(tmp_path: Path, lifetime: dict[str, int]) -> None:
    async with Node(tmp_path) as node:
        _wire(node, lifetime)
        _age_key(node, 300)
        await node.change.step()
        assert node.change.status() is None
        assert node.control.announcements == []
        made = node.change.schedule()["keyMadeAt"]
        assert made is not None
        assert abs((datetime.datetime.now(UTC) - made).days - 300) <= 1


async def test_zero_days_means_only_on_request(tmp_path: Path, lifetime: dict[str, int]) -> None:
    async with Node(tmp_path) as node:
        lifetime["days"] = 0
        _wire(node, lifetime)
        _age_key(node, 4000)
        assert node.change.schedule()["keyChangeDueAt"] is None
        await node.change.step()
        assert node.change.status() is None
        # On request it still changes, and says so.
        started = await node.change.start()
        assert started["reason"] == "requested"


async def test_a_scheduled_start_that_cannot_happen_is_tried_again_an_hour_later(
    tmp_path: Path, lifetime: dict[str, int], caplog: pytest.LogCaptureFixture
) -> None:
    async with Node(tmp_path) as node:
        _wire(node, lifetime)
        _age_key(node, 400)
        claims: dict[str, Any] = dict(node.trust.claims or {})
        claims.pop("keyChange")
        node.trust.claims = claims  # a root older than row 3
        await node.change.step()
        await node.change.step()
        assert node.change.status() is None
        said = [r for r in caplog.records if "cannot start" in r.getMessage()]
        assert len(said) == 1, "said once, not every step"
        assert "Update the machine that runs the root" in said[0].getMessage()
        node.trust.claims = {**claims, "keyChange": True}  # the root is updated
        await node.change.step()
        assert node.change.status() is None, "not before an hour has passed"
        _clock(node, 1.05 / 24)
        await node.change.step()
        status = node.change.status()
        assert status is not None and status["reason"] == "scheduled"


async def test_a_stopped_scheduled_change_starts_again_a_day_later(
    tmp_path: Path, lifetime: dict[str, int]
) -> None:
    async with Node(tmp_path) as node:
        _wire(node, lifetime)
        _age_key(node, 400)
        await node.change.step()
        assert node.state() == "waiting"
        await node.change.stop()
        await node.change.step()
        assert node.state() == "stopped"
        _clock(node, 1.01)
        await node.change.step()
        assert node.state() == "waiting"
        assert (node.change.status() or {}).get("reason") == "scheduled"


def test_get_node_says_when_the_key_was_made_and_when_it_changes(
    authed_client: Any, tmp_path: Path
) -> None:
    """What the console shows under each machine (`MachinePort.keyMadeAt`,
    `keyChangeDueAt`), read from the running change."""
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
    app.state.machine_key_change = MachineKeyChange(
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
        )
    )
    try:
        machine = authed_client.get("/v1/node").json()["machine"]
    finally:
        app.state.machine_port = None
        app.state.machine_key_change = None
    assert datetime.datetime.fromisoformat(machine["keyMadeAt"]) == made
    due = datetime.datetime.fromisoformat(machine["keyChangeDueAt"])
    # 30 days, plus this key's spread: up to a twelfth of them.
    assert made + datetime.timedelta(days=30) <= due < made + datetime.timedelta(days=32.5)


def test_the_setting_is_in_the_agent_s_config(authed_client: Any) -> None:
    schema = authed_client.get("/v1/config/schema").json()
    fields = {f["key"]: f for f in schema["fields"]}
    field = fields[machine_key_change.KEY_DAYS_KEY]
    assert field["default"] == 365 and field["minimum"] == 0
    changed = authed_client.patch("/v1/config", json={machine_key_change.KEY_DAYS_KEY: 30})
    assert changed.status_code == 200, changed.text
    assert authed_client.get("/v1/config").json()[machine_key_change.KEY_DAYS_KEY] == 30


def test_the_machines_of_one_install_do_not_change_on_the_same_day() -> None:
    """Troy's call: a yearly timer per machine, spread by a random offset.
    Keys made the same day are due on different days, each within a month
    after its year, and one key's day does not move."""
    from eugene_plexus_agent.machine_key_change import change_due_at

    made = datetime.datetime(2026, 3, 1, tzinfo=UTC)
    dues = set()
    for _ in range(12):
        key, _cert = node_identity.generate_machine_key()
        certificate = node_identity.machine_certificate(key, made=made)
        due = change_due_at(certificate, 365)
        assert due is not None
        assert made + datetime.timedelta(days=365) <= due
        assert due < made + datetime.timedelta(days=365 + 365 / 12)
        assert change_due_at(certificate, 365) == due, "the same key keeps its day"
        dues.add(due.date())
    assert len(dues) >= 8, f"twelve keys made one day fall due on {len(dues)} days"
    assert change_due_at(certificate, 0) is None
