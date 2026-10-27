"""LT1's adversarial review (2026-10-20): each finding, held by a test.

The design and the review's findings are in
`specs/docs/design/tls-between-machines.md` §7.3.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent import _http, enrollment, machine_port, machine_trust
from eugene_plexus_agent.node_identity import NodeIdentityStore

from .conftest import FakeRoot, enroll_store

PIN_R = "R" * 43
PIN_W = "W" * 43


@pytest.fixture(autouse=True)
def _no_table() -> Any:
    yield
    _http.set_machine_table(None)


def _store(tmp_path: Path) -> NodeIdentityStore:
    store = NodeIdentityStore(tmp_path / "node.yaml")
    store.load()
    store.ensure_keypair()
    return store


def test_a_node_cannot_add_its_key_to_the_root_s_door_or_another_machine_s() -> None:
    claims = {
        "version": 3,
        "controlPins": [PIN_R],
        "nodes": [
            # The root's own machine: the same door and key as the root's.
            {"name": "root", "url": "http://10.0.0.5:8079", "machineUrl": "https://10.0.0.5:8083",
             "pins": [PIN_R]},
            # A stolen node key used to claim the root's door with another key.
            {"name": "thief", "url": "http://10.0.0.9:8079", "machineUrl": "https://10.0.0.5:8083",
             "pins": [PIN_W]},
            # Two nodes on one door: neither is trusted there.
            {"name": "a", "url": "http://10.0.0.7:8079", "machineUrl": "https://10.0.0.7:8083",
             "pins": ["A" * 43]},
            {"name": "b", "url": "http://10.0.0.8:8079", "machineUrl": "https://10.0.0.7:8083",
             "pins": ["B" * 43]},
        ],
    }  # fmt: skip
    table = machine_trust.build_table(
        claims, own_name="me", control_url="http://10.0.0.5:8083", token_pins=["T" * 43]
    )
    assert table["machines"] == {"10.0.0.5:8083": [PIN_R]}, "no key was added to the root's door"
    assert table["routes"]["http://10.0.0.5:8079"] == "https://10.0.0.5:8083/agent"
    # Untrusted is refused (`false`), never plain (agent#38 review): `null`
    # would have read as an older version, reached in clear in `automatic`.
    assert table["routes"]["http://10.0.0.9:8079"] is False
    assert table["routes"]["http://10.0.0.7:8079"] is False
    assert table["routes"]["http://10.0.0.8:8079"] is False
    parsed = _http.parse_machine_table(table)
    assert parsed.invalid == {
        "http://10.0.0.9:8079",
        "http://10.0.0.7:8079",
        "http://10.0.0.8:8079",
    }


def test_the_join_token_s_key_gives_way_to_the_signed_list() -> None:
    before = machine_trust.build_table(
        None, own_name="me", control_url="http://10.0.0.5:8083", token_pins=["T" * 43]
    )
    assert before["machines"] == {"10.0.0.5:8083": ["T" * 43]}
    after = machine_trust.build_table(
        {"version": 1, "controlPins": [PIN_R], "nodes": []},
        own_name="me",
        control_url="http://10.0.0.5:8083",
        token_pins=["T" * 43],
    )
    assert after["machines"] == {"10.0.0.5:8083": [PIN_R]}


_ROUTES = machine_port.Routes(
    control=lambda: "http://127.0.0.1:1",
    driver=lambda name: "http://127.0.0.1:2",
    agent_app=None,
    required=lambda: True,
)


@pytest.mark.parametrize(
    "raw",
    [
        b"/v1/sites/../nodes/enroll",
        b"/v1/sites/%2e%2e/nodes/enroll",
        b"/v1/sites/%2E%2E/nodes/enroll",
        b"/v1/sites/./x",
        b"/v1/sites\\..\\nodes",
    ],
)
async def test_a_path_with_dot_segments_is_refused_by_both_branches(raw: bytes) -> None:
    calls: list[str] = []

    async def forward(scope: Any, receive: Any, send: Any, base: str, path: str) -> None:
        calls.append(path)

    peers = machine_port._Peers()
    peers.add(5555, ("192.168.1.50", 40000))
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "path": raw.decode("latin-1"),
        "raw_path": raw,
        "client": ("127.0.0.1", 5555),
    }
    for app in (
        machine_port.PlainApp(_ROUTES, peers, forward),  # type: ignore[arg-type]
        machine_port.MachineApp(_ROUTES, peers, forward),  # type: ignore[arg-type]
    ):
        sent.clear()
        await app(dict(scope), None, send)
        assert sent[0]["status"] == 400, raw
    assert not calls


async def test_the_raw_path_is_what_is_forwarded() -> None:
    calls: list[str] = []

    async def forward(scope: Any, receive: Any, send: Any, base: str, path: str) -> None:
        calls.append(path)

    app = machine_port.MachineApp(_ROUTES, machine_port._Peers(), forward)  # type: ignore[arg-type]
    scope = {
        "type": "http",
        "path": "/drivers/llm/v1/models/org/name",
        "raw_path": b"/drivers/llm/v1/models/org%2Fname",
        "client": ("127.0.0.1", 1),
    }
    await app(scope, None, None)
    assert calls == ["/v1/models/org%2Fname"]


def _jwt_with_typ(typ: str) -> str:
    head = base64.urlsafe_b64encode(json.dumps({"alg": "EdDSA", "typ": typ}).encode())
    return head.decode().rstrip("=") + ".e30.sig"


async def test_required_never_sends_an_install_token_in_plain_to_another_machine() -> None:
    _http.set_machine_table(
        _http.parse_machine_table({"required": True, "local": ["http://10.0.0.6:8079"]})
    )
    session = _jwt_with_typ("ep-session+jwt")
    async with _http.internal_client(timeout=2) as client:
        with pytest.raises(_http.MachineRefused, match="token"):
            await client.get(
                "http://10.0.0.99:8079/v1/node", headers={"Authorization": f"Bearer {session}"}
            )
    table = _http.machine_table()
    # A model server's own key is not this install's token: not this rule's business.
    other = httpx.Request(
        "GET",
        "http://10.0.0.99:11434/v1/models",
        headers={"Authorization": f"Bearer {_jwt_with_typ('JWT')}"},
    )
    assert _http._route(other, table) is table
    # This machine under its own LAN address is not a crossing.
    own = httpx.Request(
        "GET", "http://10.0.0.6:8079/v1/node", headers={"Authorization": f"Bearer {session}"}
    )
    assert _http._route(own, table) is table


async def test_a_401_never_makes_the_node_announce_less(tmp_path: Path) -> None:
    """agent#38 review: a 401 (or the 404 that once marked an older root) is
    what a party in the middle answers too; the node never signs its address
    alone because of one."""
    root = FakeRoot()
    store = _store(tmp_path)
    enroll_store(store, root, "me", control_url="http://control.invalid:8083")
    bodies: list[dict[str, Any]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(401, json={"detail": "Signature rejected"})

    # The attack: the node has no key for the root yet (upgraded in place),
    # so its list request goes in plain HTTP, and a party in the middle
    # answers it 404 ("an older root"), unsigned.
    trust = machine_trust.MachineTrust(tmp_path, store)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(404))
    ) as client:
        await machine_trust.pull(trust, client, "http://control.invalid:8083")
    assert trust.root_has_list is False
    # Then a 401 to the announcement. The agent's loop once handed the 404
    # to `announce_address` (`old_root`), which then signed the address
    # alone; there is no such hint now, and nothing says less.
    import inspect

    hint = (
        {"old_root": lambda: trust.root_has_list is False}
        if "old_root" in inspect.signature(enrollment.announce_address).parameters
        else {}
    )
    machine = {"machineUrl": "https://10.0.0.6:8083", "tlsPin": "M" * 43}
    outcome = await enrollment.announce_address(
        store=store,
        url="http://10.0.0.6:8079",
        transport=httpx.MockTransport(answer),
        machine=machine,
        **hint,
    )
    assert not outcome.announced
    assert [b.get("tlsPin") for b in bodies] == ["M" * 43], "it announced its address alone"


def test_a_record_the_root_holds_otherwise_is_announced_again() -> None:
    fields = {"machineUrl": "https://10.0.0.6:8083", "tlsPin": "M" * 43}
    differs = enrollment._recorded_differs
    assert not differs(fields, None), "no list held is no evidence"
    assert not differs(fields, ("https://10.0.0.6:8083", ("M" * 43,)))
    assert differs(fields, (None, ())), "the root cleared the key"
    assert differs(fields, ("https://10.0.0.6:8083", ("X" * 43,)))
