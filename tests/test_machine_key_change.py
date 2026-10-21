"""Row 3: this machine's TLS key changes across an overlap of two pins.

The machine port runs for real on loopback; the root is a MockTransport that
records announcements, signs machine lists with a real root identity key, and
answers its record of this node (`machineKeyChange.waitingOn`).

* No switch while the root sees a machine holding an older list, nor while
  the root does not answer; after the switch a new handshake gets the new key
  with no restart, an open session keeps the old one, and the next
  announcement carries one key.
* On the root, the door file control reads leads the announcement of both
  keys and follows the drop.
* A stop before the switch, and a refusal, delete the next key.
* A machine that was off while the root's key changed takes the new key by
  itself, from the root's signed list, verified against the root's identity.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import ssl
from pathlib import Path
from typing import Any

import httpx
import jwt
import nacl.signing
import pytest
import yaml

from eugene_plexus_agent import (
    _http,
    leaving,
    machine_key_change,
    machine_port,
    machine_trust,
    node_identity,
)
from eugene_plexus_agent.machine_key_change import KeyChangeRefused, MachineKeyChange, Wiring
from eugene_plexus_agent.node_identity import NodeIdentityStore

from .conftest import FakeRoot, enroll_store

CONTROL_URL = "http://control.invalid:8083"
AGENT_URL = "http://10.0.0.6:8079"
NAME = "gpu-box"


@pytest.fixture(autouse=True)
def _no_table() -> Any:
    yield
    _http.set_machine_table(None)


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(machine_key_change, "DOOR_SETTLE_SECONDS", 0.0)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _presented_pin(port: int) -> str:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with (
        socket.create_connection(("127.0.0.1", port), timeout=10) as raw,
        context.wrap_socket(raw) as tls,
    ):
        der = tls.getpeercert(True)
    assert der is not None
    return _http.spki_pin(der)


async def _hello(scope: dict[str, Any], receive: Any, send: Any) -> None:
    if scope["type"] != "http":
        return
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"hello"})


class FakeControl:
    """The root, as far as a changing node can tell."""

    def __init__(self, root: FakeRoot, store: NodeIdentityStore, machine_url: str) -> None:
        self.root, self.store, self.machine_url = root, store, machine_url
        self.pins: list[str] = []
        self.next_pin: str | None = None
        self.control_pins: list[str] = ["R" * 43]
        self.version = 0
        self._listed: Any = None
        self.waiting_on: list[str] = ["nas"]
        self.not_ready: list[str] = []
        #: A root restarted sealed: it records announcements, and serves the
        #: last list it signed, because it cannot sign another.
        self.locked = False
        self._last_jws: str | None = None
        self.announcements: list[dict[str, Any]] = []
        self.doors_at_announcement: list[Any] = []
        self.door_file: Path | None = None
        self.refuse: tuple[int, str] | None = None
        self.down = False

    def signed_list(self) -> str:
        if self.locked and self._last_jws is not None:
            return self._last_jws
        control_pins = self.control_pins
        if self.door_file is not None and self.door_file.exists():
            # As control does: the root's keys from the door file.
            control_pins = json.loads(self.door_file.read_text(encoding="utf-8"))["pins"]
        content = {
            "required": False,
            "keyChange": True,
            "controlPins": sorted(control_pins),
            "nodes": [
                {
                    "name": NAME,
                    "url": AGENT_URL,
                    "machineUrl": self.machine_url,
                    "pins": sorted(self.pins),
                }
            ],
        }
        if content != self._listed:
            self.version += 1
            self._listed = content
        self._last_jws = jwt.encode(
            {"iat": 1, "version": self.version, **content},
            self.root.identity,
            algorithm="EdDSA",
            headers={"typ": machine_trust.TYP_MACHINES},
        )
        return self._last_jws

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if request.method == "PATCH" and path == f"/v1/nodes/{NAME}":
            body = json.loads(request.content)
            fields = {k: body[k] for k in ("machineUrl", "tlsPin", "tlsNextPin") if k in body}
            message = node_identity.address_message(
                name=NAME, sequence=body["sequence"], url=body["url"], machine=fields
            )
            verify = nacl.signing.VerifyKey(
                base64.b64decode(self.store.record.signing_public_key or "")
            )
            verify.verify(message, base64.b64decode(body["signature"]))
            self.announcements.append(body)
            if self.door_file is not None:
                self.doors_at_announcement.append(
                    json.loads(self.door_file.read_text(encoding="utf-8"))["pins"]
                )
            if self.refuse is not None:
                return httpx.Response(self.refuse[0], json={"detail": self.refuse[1]})
            pins = sorted({body["tlsPin"], *([body["tlsNextPin"]] if "tlsNextPin" in body else [])})
            changed = pins != self.pins
            self.pins, self.next_pin = pins, body.get("tlsNextPin")
            return httpx.Response(
                200,
                json={
                    "name": NAME,
                    "url": body["url"],
                    "sequence": body["sequence"],
                    "changed": changed,
                },
            )
        if request.method == "GET" and path == machine_trust.LIST_PATH:
            return httpx.Response(200, json={"jws": self.signed_list()})
        if request.method == "GET" and path == "/v1/auth/status":
            return httpx.Response(200, json={"initialized": True, "unlocked": not self.locked})
        if request.method == "GET" and path == f"/v1/nodes/{NAME}":
            change = None
            if self.next_pin is not None:
                change = {
                    "nextPin": self.next_pin,
                    "listVersion": self.version,
                    "waitingOn": list(self.waiting_on),
                    "unreachable": [],
                    "notReady": list(self.not_ready),
                }
            return httpx.Response(
                200, json={"name": NAME, "tlsPins": self.pins, "machineKeyChange": change}
            )
        return httpx.Response(404, json={"detail": f"no {request.method} {path} here"})


class Node:
    """A node with a running machine port, a held list, and its change."""

    def __init__(self, tmp_path: Path, *, hosts_control: bool = False) -> None:
        self.tmp_path = tmp_path
        self.hosts_control = hosts_control

    async def __aenter__(self) -> Node:
        tmp_path = self.tmp_path
        self.root = FakeRoot()
        self.store = NodeIdentityStore(tmp_path / "node.yaml")
        self.store.load()
        enroll_store(self.store, self.root, NAME, control_url=CONTROL_URL)
        record = self.store.record
        self.old_pin = node_identity.machine_pin(record.tls_certificate or "")
        self.port = _free_port()
        self.machine_url = f"https://127.0.0.1:{self.port}"
        self.door = machine_port.MachinePort(
            routes=machine_port.Routes(
                control=lambda: None,
                driver=lambda n: None,
                agent_app=_hello,
                required=lambda: False,
            ),
            context=machine_port.server_context(
                record.tls_certificate or "", record.tls_private_key or "", tmp_path
            ),
            bind_host="127.0.0.1",
            port=self.port,
            pin=self.old_pin,
            directory=tmp_path,
        )
        await self.door.start()
        assert self.door.status.listening
        self.control = FakeControl(self.root, self.store, self.machine_url)
        self.control.pins = [self.old_pin]
        if self.hosts_control:
            self.control.control_pins = [self.old_pin]
            self.control.door_file = tmp_path / machine_key_change.DOOR_FILE
        self.transport = httpx.MockTransport(self.control.handle)
        self.trust = machine_trust.MachineTrust(tmp_path, self.store)
        self.trust.accept(self.control.signed_list())

        async def pull() -> None:
            async with _http.internal_client(timeout=5, transport=self.transport) as client:
                await machine_trust.pull(self.trust, client, CONTROL_URL)

        self.change = MachineKeyChange(
            Wiring(
                store=self.store,
                trust=self.trust,
                door=lambda: self.door,
                fields=lambda: machine_key_change.machine_fields(
                    self.store.record, self.machine_url, self.door.status.pin
                ),
                advertise=lambda: AGENT_URL,
                hosts_control=lambda: self.hosts_control,
                token=lambda: "a-token",
                door_file=tmp_path / machine_key_change.DOOR_FILE,
                machine_url=lambda: self.machine_url,
                transport=lambda: self.transport,
                pull=pull,
            )
        )
        self.change.write_door()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.door.stop()

    async def presented(self) -> str:
        return await asyncio.to_thread(_presented_pin, self.port)

    def state(self) -> str | None:
        return (self.store.record.machine_key_change or {}).get("state")

    def door_pins(self) -> list[str]:
        path = self.tmp_path / machine_key_change.DOOR_FILE
        return list(json.loads(path.read_text(encoding="utf-8"))["pins"])


async def test_switch_waits_for_coverage_then_presents_new_key(tmp_path: Path) -> None:
    async with Node(tmp_path) as node:
        await node.change.start()
        next_pin = node.store.record.machine_key_change["nextPin"]  # type: ignore[index]
        assert next_pin != node.old_pin

        # Both keys announced, under this node's signature.
        await node.change.step()
        both = node.control.announcements[-1]
        assert both["tlsPin"] == node.old_pin and both["tlsNextPin"] == next_pin
        assert node.trust.own_record() == (
            node.machine_url,
            tuple(sorted([node.old_pin, next_pin])),
        )

        # A machine the root reached holds an older list: no switch.
        for _ in range(2):
            await node.change.step()
        assert node.state() == "waiting"
        assert await node.presented() == node.old_pin

        # A session opened before the switch, pinned to the old key alone.
        hostport = f"127.0.0.1:{node.port}"
        _http.set_machine_table(_http.parse_machine_table({"machines": {hostport: [node.old_pin]}}))
        async with _http.internal_client(timeout=10) as before:
            assert (await before.get(f"https://{hostport}/agent/hello")).text == "hello"

            node.control.waiting_on = []
            await node.change.step()
            # The step took the root's list, which rebuilt this process's
            # table; the test's own pin for the session goes back in.
            _http.set_machine_table(
                _http.parse_machine_table({"machines": {hostport: [node.old_pin]}})
            )

            # A new handshake gets the new key, with no restart.
            assert await node.presented() == next_pin
            assert node.door.status.pin == next_pin
            # The open session finishes on the key it began with.
            assert (await before.get(f"https://{hostport}/agent/hello")).text == "hello"
        async with _http.internal_client(timeout=10) as after:
            with pytest.raises(httpx.ConnectError):
                await after.get(f"https://{hostport}/agent/hello")

        # The next announcement carries one key: the new one.
        dropped = node.control.announcements[-1]
        assert dropped["tlsPin"] == next_pin and "tlsNextPin" not in dropped
        assert node.state() == "done"
        assert node.control.pins == [next_pin]

        # The old private key is gone from node.yaml; the new one is the key.
        on_disk = yaml.safe_load((tmp_path / "node.yaml").read_text(encoding="utf-8"))
        assert "tlsNextPrivateKey" not in on_disk
        assert node_identity.machine_pin(on_disk["tlsCertificate"]) == next_pin
        status = machine_key_change.status(node.store.record)
        assert status is not None and status["state"] == "done" and status["finishedAt"]


async def test_a_root_that_does_not_answer_is_waited_for(tmp_path: Path) -> None:
    async with Node(tmp_path) as node:
        await node.change.start()
        await node.change.step()
        node.control.waiting_on = []
        node.control.down = True
        # Long past the 3-minute bound: still no switch with no root.
        node.store.update_machine_key_change(listedAt=1.0)
        await node.change.step()
        assert node.state() == "waiting"
        assert await node.presented() == node.old_pin
        change = node.store.record.machine_key_change or {}
        assert "waiting for the root" in str(change.get("error"))
        # The root answers again: the change goes on.
        node.control.down = False
        await node.change.step()
        assert node.state() == "done"


async def test_three_minutes_after_the_list_named_both_it_switches(tmp_path: Path) -> None:
    async with Node(tmp_path) as node:
        await node.change.start()
        await node.change.step()
        await node.change.step()
        assert node.state() == "waiting"
        node.store.update_machine_key_change(listedAt=1.0)
        await node.change.step()
        assert node.state() == "done"


async def test_on_the_root_the_door_file_leads_and_follows(tmp_path: Path) -> None:
    async with Node(tmp_path, hosts_control=True) as node:
        assert node.door_pins() == [node.old_pin]
        await node.change.start()
        next_pin = node.store.record.machine_key_change["nextPin"]  # type: ignore[index]
        # Before any announcement: control is told both keys first.
        assert node.door_pins() == [node.old_pin, next_pin]
        assert not node.control.announcements
        await node.change.step()
        assert node.control.doors_at_announcement[-1] == [node.old_pin, next_pin]
        # The list names both for this node and in controlPins.
        node.control.control_pins = [node.old_pin, next_pin]
        node.control.waiting_on = []
        await node.change.step()
        assert node.state() == "done"
        # The drop was announced while the door still had both keys...
        assert node.control.doors_at_announcement[-1] == [next_pin, node.old_pin]
        # ...and the old key left the door file only after the root took it.
        assert node.door_pins() == [next_pin]


async def test_a_stop_before_the_switch_deletes_the_next_key(tmp_path: Path) -> None:
    async with Node(tmp_path, hosts_control=True) as node:
        await node.change.start()
        await node.change.step()
        stopped = await node.change.stop()
        assert stopped["state"] == "stopped" and stopped["nextPin"] is None
        record = node.store.record
        assert record.tls_next_private_key is None and record.tls_next_certificate is None
        assert node.control.pins == [node.old_pin]
        assert node.door_pins() == [node.old_pin]
        assert await node.presented() == node.old_pin
        with pytest.raises(KeyChangeRefused) as caught:
            await node.change.stop()
        assert caught.value.status == 404


async def test_no_stop_after_the_switch(tmp_path: Path) -> None:
    async with Node(tmp_path) as node:
        await node.change.start()
        await node.change.step()
        next_pin = node.store.record.machine_key_change["nextPin"]  # type: ignore[index]
        # The switch happens, and the root does not take the drop yet (503:
        # a wait, not a refusal). Its list keeps both keys meanwhile.
        node.control.waiting_on = []
        node.control.refuse = (503, "busy")
        await node.change.step()
        assert node.state() == "switched"
        assert await node.presented() == next_pin
        assert node.control.pins == sorted([node.old_pin, next_pin])
        with pytest.raises(KeyChangeRefused) as caught:
            await node.change.stop()
        assert caught.value.status == 409
        node.control.refuse = None
        await node.change.step()
        assert node.state() == "done"
        assert node.control.pins == [next_pin]


async def test_a_refused_next_key_is_deleted_and_says_why(tmp_path: Path) -> None:
    async with Node(tmp_path) as node:
        await node.change.start()
        node.control.refuse = (400, "Machine port cannot be a node's")
        await node.change.step()
        status = machine_key_change.status(node.store.record)
        assert status is not None and status["state"] == "failed"
        assert "the root refused it (Machine port cannot be a node's)" in str(status["error"])
        assert node.store.record.tls_next_private_key is None
        assert await node.presented() == node.old_pin


# --------------------------------------------------------------------------- #
# The adversarial review's blockers (2026-10-21), each by fault injection
# --------------------------------------------------------------------------- #


async def test_a_switch_node_yaml_cannot_keep_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocker 1: node.yaml cannot be written at the switch (a full disk).
    Nothing switches, in memory or on the port, and the status says why; once
    it can be written the change goes on, and node.yaml, the port and the
    root agree on one key."""
    async with Node(tmp_path) as node:
        await node.change.start()
        await node.change.step()
        next_pin = node.store.record.machine_key_change["nextPin"]  # type: ignore[index]
        node.control.waiting_on = []
        real = node_identity.write_private

        def full(path: Path, text: str) -> None:
            if path.name == "node.yaml":
                raise OSError(28, "No space left on device")
            real(path, text)

        monkeypatch.setattr(node_identity, "write_private", full)
        await node.change.step()
        assert node_identity.machine_pin(node.store.record.tls_certificate or "") == node.old_pin
        assert node.state() == "waiting"
        assert await node.presented() == node.old_pin
        status = node.change.status()
        assert status is not None and "No space left on device" in str(status["error"])
        monkeypatch.setattr(node_identity, "write_private", real)
        await node.change.step()
        assert node.state() == "done"
        on_disk = yaml.safe_load((tmp_path / "node.yaml").read_text(encoding="utf-8"))
        assert node_identity.machine_pin(on_disk["tlsCertificate"]) == next_pin
        assert await node.presented() == next_pin
        assert node.control.pins == [next_pin]


async def test_the_root_s_own_change_waits_for_agents_that_cannot_recover(
    tmp_path: Path,
) -> None:
    """Blocker 2: an agent older than row 3 that misses the root's key change
    never takes the new key. The root's change does not switch while the
    root names one, past the 3-minute bound too; another machine's change
    is not held up by it."""
    async with Node(tmp_path, hosts_control=True) as node:
        await node.change.start()
        await node.change.step()
        node.control.waiting_on = []
        node.control.not_ready = ["old-nas"]
        node.store.update_machine_key_change(listedAt=1.0)
        await node.change.step()
        assert node.state() == "waiting"
        assert await node.presented() == node.old_pin
        error = str((node.change.status() or {}).get("error"))
        assert "old-nas" in error and "update" in error
        node.control.not_ready = []
        await node.change.step()
        assert node.state() == "done"
    async with Node(tmp_path / "worker") as worker:
        await worker.change.start()
        await worker.change.step()
        worker.control.waiting_on = []
        worker.control.not_ready = ["old-nas"]
        await worker.change.step()
        assert worker.state() == "done"


async def test_the_root_announces_both_keys_only_once_its_door_file_names_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second review's should-fix: if the door file cannot be written,
    the root's node entry would name both keys while controlPins names the
    old one, and every other machine would ignore the root's own entry. So
    nothing is announced until control is told both keys."""
    async with Node(tmp_path, hosts_control=True) as node:
        real = machine_key_change.write_private

        def full(path: Path, text: str) -> None:
            if path.name == machine_key_change.DOOR_FILE:
                raise OSError(28, "No space left on device")
            real(path, text)

        monkeypatch.setattr(machine_key_change, "write_private", full)
        await node.change.start()
        next_pin = node.store.record.machine_key_change["nextPin"]  # type: ignore[index]
        await node.change.step()
        assert node.control.announcements == []
        error = str((node.change.status() or {}).get("error"))
        assert "door file" in error and "No space left on device" in error
        assert node.door_pins() == [node.old_pin]
        monkeypatch.setattr(machine_key_change, "write_private", real)
        await node.change.step()
        assert node.control.announcements[-1]["tlsNextPin"] == next_pin
        assert node.control.doors_at_announcement[-1] == [node.old_pin, next_pin]


def _list(root: FakeRoot, version: int, pins: list[str]) -> str:
    return jwt.encode(
        {"iat": 1, "version": version, "required": False, "controlPins": pins, "nodes": []},
        root.identity,
        algorithm="EdDSA",
        headers={"typ": machine_trust.TYP_MACHINES},
    )


def test_a_list_is_acknowledged_only_once_its_table_is_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocker 3: the version this node reports (which the root reads to stop
    waiting for it) is the one its children can read: not while the table
    file cannot be written, and not before their one-second cache passes. The
    same list taken again writes the table it could not."""
    root = FakeRoot()
    store = NodeIdentityStore(tmp_path / "node.yaml")
    store.load()
    enroll_store(store, root, NAME, control_url="http://10.0.0.5:8083")
    trust = machine_trust.MachineTrust(tmp_path, store)
    trust.load()
    monkeypatch.setattr(machine_trust, "CHILD_CACHE_SECONDS", 0.0)
    trust.accept(_list(root, 1, ["R" * 43]))
    assert trust.version == 1
    real = machine_trust.write_private

    def failing(path: Path, text: str) -> None:
        if path.name == machine_trust.PINS_FILE:
            raise OSError(28, "No space left on device")
        real(path, text)

    monkeypatch.setattr(machine_trust, "write_private", failing)
    trust.accept(_list(root, 2, ["S" * 43]))
    assert trust.version == 1, "the children still read list 1"
    monkeypatch.setattr(machine_trust, "write_private", real)
    monkeypatch.setattr(machine_trust, "CHILD_CACHE_SECONDS", 60.0)
    trust.accept(_list(root, 2, ["S" * 43]))
    on_disk = json.loads((tmp_path / machine_trust.PINS_FILE).read_text(encoding="utf-8"))
    assert on_disk["version"] == 2
    # Written, but within the children's cache: still the previous one.
    assert trust.version == 1
    monkeypatch.setattr(machine_trust, "CHILD_CACHE_SECONDS", 0.0)
    assert trust.version == 2


async def test_done_only_once_a_signed_list_drops_the_old_key(tmp_path: Path) -> None:
    """Blocker 4: a root restarted sealed takes the drop but cannot sign a
    list, and keeps serving both keys, so the old key is still accepted. The
    change is not done until a verified list drops it, and says what it
    waits for."""
    async with Node(tmp_path) as node:
        await node.change.start()
        await node.change.step()
        next_pin = node.store.record.machine_key_change["nextPin"]  # type: ignore[index]
        node.control.signed_list()  # the overlap, as published
        node.control.locked = True
        node.control.waiting_on = []
        await node.change.step()
        assert node.control.pins == [next_pin]  # the root took the drop...
        assert node.state() == "switched"  # ...and the list still has both
        assert "unlocked" in str((node.change.status() or {}).get("error"))
        node.control.locked = False
        await node.change.step()
        assert node.state() == "done"
        assert node.trust.own_record() == (node.machine_url, (next_pin,))


async def test_a_change_needs_a_root_that_takes_a_next_key(tmp_path: Path) -> None:
    async with Node(tmp_path) as node:
        claims = dict(node.trust.claims or {})
        claims.pop("keyChange")
        node.trust.claims = claims
        with pytest.raises(KeyChangeRefused) as caught:
            await node.change.start()
        assert caught.value.slug == "root-too-old"
        assert node.store.record.tls_next_private_key is None


def test_the_routes_are_an_operator_s_and_say_why_not(authed_client: Any) -> None:
    """No machine port in a test app: the start says so; with no session,
    neither route answers."""
    started = authed_client.post("/v1/node/machine-key")
    assert started.status_code == 409, started.text
    assert "machine port" in started.json()["detail"]["detail"]
    for method in ("POST", "DELETE"):
        refused = authed_client.request(
            method, "/v1/node/machine-key", headers={"Authorization": ""}
        )
        assert refused.status_code == 401, refused.text


# --------------------------------------------------------------------------- #
# A machine that was off while the root's key changed
# --------------------------------------------------------------------------- #


class _RootApp:
    """Control behind the root's machine port: its signed list and bundle."""

    def __init__(self) -> None:
        self.list_jws = ""
        self.bundle_jws = ""
        self.paths: list[str] = []

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        self.paths.append(scope["path"])
        body = {"jws": self.list_jws if scope["path"] == "/v1/trust/machines" else self.bundle_jws}
        payload = json.dumps(body).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": payload})


async def _serve(app: Any) -> tuple[Any, Any, int]:
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, lifespan="off", log_level="warning")
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server, sock, int(sock.getsockname()[1])


async def test_a_machine_off_for_the_root_s_key_change_takes_the_new_key_by_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator's required check (row 3 §5): stopped before the change
    and started after the old pin left, it holds only the root's old key."""
    # The machine port here stands in for the root's, at a loopback address;
    # the table treats it as another machine.
    monkeypatch.setattr(machine_trust, "is_loopback_host", lambda host: False)
    plain_calls: list[str] = []
    plain_app = machine_port.PlainApp.__call__

    async def counted(self: Any, scope: dict[str, Any], receive: Any, send: Any) -> None:
        plain_calls.append(str(scope.get("path")))
        await plain_app(self, scope, receive, send)

    monkeypatch.setattr(machine_port.PlainApp, "__call__", counted)
    root = FakeRoot()
    control = _RootApp()
    server, sock, control_port = await _serve(control)
    root_store = NodeIdentityStore(tmp_path / "root" / "node.yaml")
    root_store.load()
    root_store.ensure_keypair()
    root_record = root_store.record
    new_root_pin = node_identity.machine_pin(root_record.tls_certificate or "")
    port = _free_port()
    door = machine_port.MachinePort(
        routes=machine_port.Routes(
            control=lambda: f"http://127.0.0.1:{control_port}",
            driver=lambda n: None,
            agent_app=_hello,
            required=lambda: True,
        ),
        context=machine_port.server_context(
            root_record.tls_certificate or "", root_record.tls_private_key or "", tmp_path / "root"
        ),
        bind_host="127.0.0.1",
        port=port,
        pin=new_root_pin,
        directory=tmp_path / "root",
    )
    await door.start()
    try:
        control_url = f"http://127.0.0.1:{port}"
        store = NodeIdentityStore(tmp_path / "worker" / "node.yaml")
        store.load()
        enroll_store(store, root, NAME, control_url=control_url)
        trust = machine_trust.MachineTrust(tmp_path / "worker", store)

        def listed(version: int, pins: list[str], signer: Any = root.identity) -> str:
            return jwt.encode(
                {"iat": 1, "version": version, "required": True, "keyChange": True,
                 "controlPins": pins, "nodes": []},
                signer,
                algorithm="EdDSA",
                headers={"typ": machine_trust.TYP_MACHINES},
            )  # fmt: skip

        # What it held when it stopped: the root's old key.
        trust.accept(listed(3, ["O" * 43]))
        assert _http.machine_table().machines[f"127.0.0.1:{port}"] == frozenset({"O" * 43})
        control.bundle_jws = "a-bundle"

        # A list that does not verify against the root's identity changes
        # nothing, and the refusal stands.
        control.list_jws = listed(9, [new_root_pin], signer=FakeRoot().identity)
        async with _http.internal_client(timeout=10) as client:
            with pytest.raises(httpx.ConnectError):
                await leaving.bundle_response(client, control_url, trust)
        assert trust.claims is not None and trust.claims["version"] == 3
        assert "/v1/trust/bundle" not in control.paths

        # The root's real list: taken in plain HTTP, verified, and the bundle
        # then comes over TLS to the root's new key.
        control.list_jws = listed(9, [new_root_pin])
        async with _http.internal_client(timeout=10) as client:
            answer = await leaving.bundle_response(client, control_url, trust)
        assert answer.status_code == 200 and answer.json() == {"jws": "a-bundle"}
        assert trust.claims is not None and trust.claims["version"] == 9
        assert _http.machine_table().machines[f"127.0.0.1:{port}"] == frozenset({new_root_pin})
        assert control.paths[-1] == "/v1/trust/bundle"
        # Over TLS throughout, checking no key for the list: nothing plain
        # crossed between the machines.
        assert plain_calls == []
    finally:
        await door.stop()
        await server.shutdown(sockets=[sock])
        sock.close()
