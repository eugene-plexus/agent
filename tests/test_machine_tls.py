"""TLS between machines (LT1), on the agent: the machine key, the machine
port's two branches, the signed machine list and the table it builds, joining
with the root's key from the join token, and the signed announcement.

The machine port runs for real on loopback here: a TLS front with the
first-byte switch, its inner listeners, and a pinned client on the other side.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
from pathlib import Path
from typing import Any

import httpx
import jwt
import nacl.exceptions
import nacl.signing
import pytest
import yaml

from eugene_plexus_agent import _http, enrollment, machine_port, machine_trust, node_identity
from eugene_plexus_agent.node_identity import NodeIdentityStore

from .conftest import FakeRoot, enroll_store


@pytest.fixture(autouse=True)
def _no_table() -> Any:
    yield
    _http.set_machine_table(None)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _store(tmp_path: Path) -> NodeIdentityStore:
    store = NodeIdentityStore(tmp_path / "node.yaml")
    store.load()
    store.ensure_keypair()
    return store


# --------------------------------------------------------------------------- #
# The machine key
# --------------------------------------------------------------------------- #


def test_a_node_makes_a_machine_key_once_and_keeps_it(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.record
    assert record.tls_private_key and "PRIVATE KEY" in record.tls_private_key
    assert record.tls_certificate and "CERTIFICATE" in record.tls_certificate
    pin = node_identity.machine_pin(record.tls_certificate)
    assert len(pin) == 43
    again = NodeIdentityStore(tmp_path / "node.yaml")
    again.load()
    again.ensure_keypair()
    assert node_identity.machine_pin(again.record.tls_certificate or "") == pin
    on_disk = yaml.safe_load((tmp_path / "node.yaml").read_text(encoding="utf-8"))
    assert on_disk["tlsPrivateKey"] == record.tls_private_key


# --------------------------------------------------------------------------- #
# The machine port, running
# --------------------------------------------------------------------------- #


class _Echo:
    """An ASGI app that answers with what it was given: path, peer, headers."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        record = {
            "path": scope["path"],
            "client": list(scope["client"]) if scope.get("client") else None,
            "headers": {k.decode(): v.decode() for k, v in scope["headers"]},
            "body": body.decode(),
        }
        self.seen.append(record)
        payload = json.dumps(record).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": payload})


async def _serve_plain(app: Any) -> tuple[Any, int]:
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, lifespan="off", log_level="warning", forwarded_allow_ips=[])
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return (server, sock), int(sock.getsockname()[1])


class _Door:
    def __init__(self, tmp_path: Path, *, required: bool = False, control: bool = True) -> None:
        self.tmp_path = tmp_path
        self.required = required
        self.with_control = control
        self.agent = _Echo()
        self.control = _Echo()
        self.driver = _Echo()

    async def __aenter__(self) -> _Door:
        store = _store(self.tmp_path)
        record = store.record
        self.pin = node_identity.machine_pin(record.tls_certificate or "")
        self._control_server, control_port = await _serve_plain(self.control)
        self._driver_server, driver_port = await _serve_plain(self.driver)
        routes = machine_port.Routes(
            control=lambda: f"http://127.0.0.1:{control_port}" if self.with_control else None,
            driver=lambda name: f"http://127.0.0.1:{driver_port}" if name == "llm" else None,
            agent_app=self.agent,
            required=lambda: self.required,
        )
        self.port = _free_port()
        context = machine_port.server_context(
            record.tls_certificate or "", record.tls_private_key or "", self.tmp_path
        )
        self.door = machine_port.MachinePort(
            routes=routes, context=context, bind_host="127.0.0.1", port=self.port, pin=self.pin
        )
        await self.door.start()
        assert self.door.status.listening
        return self

    def pin_table(self, pins: list[str] | None = None) -> None:
        hostport = f"127.0.0.1:{self.port}"
        _http.set_machine_table(
            _http.parse_machine_table(
                {
                    "machines": {hostport: pins or [self.pin]},
                    "routes": {"http://10.9.9.9:8079": f"https://{hostport}/agent"},
                    "names": {hostport: "worker-1"},
                }
            )
        )

    async def __aexit__(self, *exc: object) -> None:
        await self.door.stop()
        for server, sock in (self._control_server, self._driver_server):
            await server.shutdown(sockets=[sock])
            sock.close()


async def test_the_agent_route_is_served_in_process_with_the_real_peer(tmp_path: Path) -> None:
    async with _Door(tmp_path) as door:
        door.pin_table()
        async with _http.internal_client(timeout=10) as client:
            answer = await client.get(
                "http://10.9.9.9:8079/v1/components?all=1",
                headers={
                    "x-forwarded-for": "6.6.6.6",
                    "x-eugene-plexus-peer": "6.6.6.6",
                    "authorization": "Bearer abc",
                },
            )
        assert answer.status_code == 200, answer.text
        seen = answer.json()
        assert seen["path"] == "/v1/components"
        # The caller's socket, never the pump's loopback leg and never a header.
        assert seen["client"][0] == "127.0.0.1"
        assert seen["headers"]["x-eugene-plexus-peer"] == "127.0.0.1"
        assert "x-forwarded-for" not in seen["headers"]
        assert seen["headers"]["authorization"] == "Bearer abc"


async def test_control_and_drivers_are_forwarded_and_streamed(tmp_path: Path) -> None:
    async with _Door(tmp_path) as door:
        door.pin_table()
        base = f"https://127.0.0.1:{door.port}"
        body = b"x" * (3 * 1024 * 1024)
        async with _http.internal_client(timeout=20) as client:
            control = await client.post(base + "/control/v1/nodes/enroll", content=body)
            driver = await client.get(base + "/drivers/llm/v1/info")
            missing = await client.get(base + "/drivers/other/v1/info")
            nowhere = await client.get(base + "/elsewhere")
        assert control.json()["path"] == "/v1/nodes/enroll"
        assert len(control.json()["body"]) == len(body)
        assert control.json()["headers"]["x-eugene-plexus-peer"] == "127.0.0.1"
        assert driver.json()["path"] == "/v1/info"
        assert missing.status_code == 404 and "other" in missing.text
        assert nowhere.status_code == 404


async def test_plain_http_reaches_control_while_it_is_allowed(tmp_path: Path) -> None:
    async with _Door(tmp_path) as door:
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            answer = await client.get(f"http://127.0.0.1:{door.port}/v1/nodes")
        assert answer.status_code == 200
        assert answer.json()["path"] == "/v1/nodes"
        assert door.control.seen and not door.agent.seen


async def test_plain_http_from_another_machine_is_refused_once_required(tmp_path: Path) -> None:
    peers = machine_port._Peers()
    forward_calls: list[str] = []

    async def forward(scope: Any, receive: Any, send: Any, base: str, path: str) -> None:
        forward_calls.append(path)

    routes = machine_port.Routes(
        control=lambda: "http://127.0.0.1:1",
        driver=lambda name: None,
        agent_app=None,
        required=lambda: True,
    )
    app = machine_port.PlainApp(routes, peers, forward)  # type: ignore[arg-type]
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    peers.add(5555, ("192.168.1.50", 40000))
    scope = {"type": "http", "path": "/v1/trust/bundle", "client": ("127.0.0.1", 5555)}
    await app(scope, None, send)
    assert sent[0]["status"] == 403 and not forward_calls
    # A Job Site keeps the door it joined at (design §2).
    peers.add(5556, ("192.168.1.51", 40001))
    await app({"type": "http", "path": "/v1/sites/poll", "client": ("127.0.0.1", 5556)}, None, send)
    assert forward_calls == ["/v1/sites/poll"]
    forward_calls.clear()
    # This machine's own callers keep their plain door to control.
    sent.clear()
    await app({"type": "http", "path": "/v1/nodes", "client": ("127.0.0.1", 6666)}, None, send)
    assert forward_calls == ["/v1/nodes"]


async def test_a_machine_port_with_another_key_is_refused_before_a_byte(tmp_path: Path) -> None:
    async with _Door(tmp_path) as door:
        _, other_cert = node_identity.generate_machine_key()
        door.pin_table([node_identity.machine_pin(other_cert)])
        async with _http.internal_client(timeout=10) as client:
            with pytest.raises(httpx.ConnectError) as caught:
                await client.get("http://10.9.9.9:8079/v1/node", headers={"authorization": "B x"})
        assert _http.refused_pin(caught.value) == door.pin
        assert "worker-1's machine key doesn't match" in str(caught.value)
        await asyncio.sleep(0.1)
        assert not door.agent.seen


async def test_a_port_another_process_holds_is_reported_and_retried(tmp_path: Path) -> None:
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = int(holder.getsockname()[1])
    store = _store(tmp_path)
    record = store.record
    door = machine_port.MachinePort(
        routes=machine_port.Routes(
            control=lambda: None, driver=lambda n: None, agent_app=_Echo(), required=lambda: False
        ),
        context=machine_port.server_context(
            record.tls_certificate or "", record.tls_private_key or "", tmp_path
        ),
        bind_host="127.0.0.1",
        port=port,
        pin="x",
    )
    try:
        await door.start()
        assert not door.status.listening
        assert door.status.error and str(port) in door.status.error
    finally:
        await door.stop()
        holder.close()


async def test_a_machine_port_still_binding_is_announced_all_the_same(tmp_path: Path) -> None:
    """agent#38: the root must know this machine takes TLS before its port
    listens. Announced without its door (8083 held a moment longer at a
    restart, say), the root recorded it as an older version, and in
    `automatic` every machine sent it tokens in plain HTTP until it was."""
    from types import SimpleNamespace

    from eugene_plexus_agent.settings import Settings
    from eugene_plexus_agent.state import AgentState

    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = int(holder.getsockname()[1])
    store = _store(tmp_path)
    record = store.record
    pin = node_identity.machine_pin(record.tls_certificate or "")
    door = machine_port.MachinePort(
        routes=machine_port.Routes(
            control=lambda: None, driver=lambda n: None, agent_app=_Echo(), required=lambda: False
        ),
        context=machine_port.server_context(
            record.tls_certificate or "", record.tls_private_key or "", tmp_path
        ),
        bind_host="127.0.0.1",
        port=port,
        pin=pin,
    )
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    app = SimpleNamespace(
        state=SimpleNamespace(
            machine_port=door,
            agent_state=state,
            node_identity=store,
            settings=Settings(config_file=tmp_path / "agent.yaml", bind_port=8079),
        )
    )
    try:
        await door.start()
        assert not door.status.listening
        # Without an address others could use, no door is announced.
        assert machine_port.announced(app) is None  # type: ignore[arg-type]
        from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest

        state.apply_config_patch(
            ConfigUpdateRequest.model_validate({"advertiseUrl": "http://10.0.0.6:8079"})
        )
        fields = machine_port.announced(app)  # type: ignore[arg-type]
        assert fields is not None
        # agent#38 review: and why it does not answer, for the root's card.
        held = fields.pop("machineError")
        assert held.startswith(f"port {port} is held by"), held
        assert fields == {"machineUrl": f"https://10.0.0.6:{port}", "tlsPin": pin}
        assert machine_port.machine_url(app) is None, "what listens is still reported as it is"
        # The address an enrolment is about to register wins over the one held.
        fields = machine_port.announced(app, "http://10.0.0.7:8079")  # type: ignore[arg-type]
        assert fields is not None
        assert fields["machineUrl"] == f"https://10.0.0.7:{port}" and fields["tlsPin"] == pin
        # And a node that opens no machine port announces none.
        app.state.machine_port = None
        assert machine_port.announced(app) is None  # type: ignore[arg-type]
    finally:
        await door.stop()
        holder.close()


async def test_no_token_goes_in_plain_to_a_machine_the_table_does_not_name_yet() -> None:
    """agent#38, in the agent's own copy of `_http` (its console hops, Update
    all and Library calls reach other machines' agents): in `automatic`, a
    machine missing from this process's table is not sent a token in plain
    HTTP; one the table says runs an older version still is."""

    def token() -> str:
        def part(value: dict[str, str]) -> str:
            return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

        return f"{part({'alg': 'EdDSA', 'typ': 'ep-session+jwt'})}.{part({'sub': 'p'})}.c2ln"

    sent: list[str] = []

    def plain(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, json={})

    _http.set_machine_table(
        _http.parse_machine_table(
            {"routes": {"http://10.9.9.8:8079": None}, "local": ["http://10.9.9.2:8079"]}
        )
    )
    headers = {"Authorization": f"Bearer {token()}"}
    async with _http.internal_client(timeout=2) as client:
        transport = client._transport
        assert isinstance(transport, _http.MachineTransport)
        transport._plain = httpx.MockTransport(plain)  # type: ignore[assignment]
        with pytest.raises(_http.MachineRefused) as caught:
            await client.get("http://10.9.9.7:8079/api/proxy/library/v1/folders", headers=headers)
        assert "does not know yet whether 10.9.9.7 takes TLS" in str(caught.value)
        await client.get("http://10.9.9.8:8079/v1/node", headers=headers)
        await client.get("http://10.9.9.2:8079/v1/node", headers=headers)
        await client.get("http://10.9.9.7:8079/healthz")
    assert sent == [
        "http://10.9.9.8:8079/v1/node",
        "http://10.9.9.2:8079/v1/node",
        "http://10.9.9.7:8079/healthz",
    ]


# --------------------------------------------------------------------------- #
# Control leaves the machine port on the root
# --------------------------------------------------------------------------- #


def test_control_moves_off_the_machine_port_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_agent._generated.models import ComponentEntry, SpawnConfig
    from eugene_plexus_agent.state import AgentState

    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    state.add_topology_entry(
        ComponentEntry(
            name="control",
            kind="control",
            url="http://127.0.0.1:8083",  # type: ignore[arg-type]
            spawn=SpawnConfig(configFile="control.yaml"),
        )
    )
    from eugene_plexus_agent import ports

    # Held by another process: control keeps it, and nothing moves.
    monkeypatch.setattr(ports, "is_free", lambda port, **_: port != 8083)
    assert machine_port.move_control_behind(state, 8083) is None
    assert str(state.get_topology_entry("control").url).rstrip("/") == "http://127.0.0.1:8083"
    monkeypatch.setattr(ports, "is_free", lambda port, **_: True)
    moved = machine_port.move_control_behind(state, 8083)
    assert moved is not None and moved > 8083
    assert str(state.get_topology_entry("control").url).rstrip("/") == f"http://127.0.0.1:{moved}"
    assert machine_port.move_control_behind(state, 8083) is None


# --------------------------------------------------------------------------- #
# The signed machine list and the table
# --------------------------------------------------------------------------- #


def _signed(root: FakeRoot, version: int, **claims: Any) -> str:
    payload = {"iat": 1, "version": version, "required": False, "controlPins": [], "nodes": []}
    payload.update(claims)
    return jwt.encode(
        payload, root.identity, algorithm="EdDSA", headers={"typ": machine_trust.TYP_MACHINES}
    )


PIN_W = "W" * 43
PIN_R = "R" * 43


def test_the_list_builds_the_table_for_this_node_and_its_children(tmp_path: Path) -> None:
    root = FakeRoot()
    store = _store(tmp_path)
    enroll_store(store, root, "me", control_url="http://10.0.0.5:8083")
    trust = machine_trust.MachineTrust(tmp_path, store)
    trust.load()
    changed = trust.accept(
        _signed(
            root,
            1,
            controlPins=[PIN_R],
            nodes=[
                {
                    "name": "me",
                    "url": "http://10.0.0.6:8079",
                    "machineUrl": "https://10.0.0.6:8083",
                    "pins": ["M" * 43],
                },
                {
                    "name": "gpu",
                    "url": "http://10.0.0.7:8079",
                    "machineUrl": "https://10.0.0.7:8083",
                    "pins": [PIN_W],
                },
                {"name": "old", "url": "http://10.0.0.8:8079", "machineUrl": None, "pins": []},
            ],
        )
    )
    assert changed
    table = json.loads((tmp_path / machine_trust.PINS_FILE).read_text(encoding="utf-8"))
    assert table["machines"] == {"10.0.0.5:8083": [PIN_R], "10.0.0.7:8083": [PIN_W]}
    assert table["routes"] == {
        "http://10.0.0.5:8083": "https://10.0.0.5:8083/control",
        "http://10.0.0.7:8079": "https://10.0.0.7:8083/agent",
        "http://10.0.0.8:8079": None,
    }
    # In this process too.
    assert (
        _http.machine_route("http://10.0.0.7:8079/v1/node") == "https://10.0.0.7:8083/agent/v1/node"
    )
    assert _http.machine_route("http://10.0.0.5:8083/v1/trust/bundle") == (
        "https://10.0.0.5:8083/control/v1/trust/bundle"
    )


def test_a_list_not_signed_by_this_node_s_root_or_older_is_refused(tmp_path: Path) -> None:
    root, stranger = FakeRoot(), FakeRoot()
    store = _store(tmp_path)
    enroll_store(store, root, "me", control_url="http://10.0.0.5:8083")
    trust = machine_trust.MachineTrust(tmp_path, store)
    with pytest.raises(machine_trust.MachineListError, match="not signed"):
        trust.accept(_signed(stranger, 1, controlPins=[PIN_W]))
    trust.accept(_signed(root, 5, controlPins=[PIN_R]))
    with pytest.raises(machine_trust.MachineListError, match="older"):
        trust.accept(_signed(root, 4, controlPins=[PIN_W]))
    assert _http.machine_table().machines["10.0.0.5:8083"] == frozenset({PIN_R})


def test_the_list_kept_on_disk_is_used_after_a_restart(tmp_path: Path) -> None:
    root = FakeRoot()
    store = _store(tmp_path)
    enroll_store(store, root, "me", control_url="http://10.0.0.5:8083")
    machine_trust.MachineTrust(tmp_path, store).accept(_signed(root, 2, controlPins=[PIN_R]))
    _http.set_machine_table(None)
    again = machine_trust.MachineTrust(tmp_path, store)
    again.load()
    assert again.claims is not None and again.claims["version"] == 2
    assert _http.machine_pinned("https://10.0.0.5:8083")


def test_required_changing_is_reported(tmp_path: Path) -> None:
    root = FakeRoot()
    store = _store(tmp_path)
    enroll_store(store, root, "me", control_url="http://10.0.0.5:8083")
    trust = machine_trust.MachineTrust(tmp_path, store)
    changes: list[bool] = []
    trust.on_required_changed = changes.append
    trust.accept(_signed(root, 1))
    trust.accept(_signed(root, 2, required=True))
    trust.accept(_signed(root, 3, required=True))
    assert changes == [True]


# --------------------------------------------------------------------------- #
# Joining with the root's key, and the signed announcement
# --------------------------------------------------------------------------- #


def test_a_join_token_gives_up_the_root_s_key() -> None:
    secret = "abcDEF_-123"
    assert enrollment.split_join_token(f"{secret}.{PIN_R}") == (secret, PIN_R)
    assert enrollment.split_join_token(secret) == (secret, None)
    # A pin cut short when the command was copied is an error, never a plain join.
    with pytest.raises(ValueError, match="damaged"):
        enrollment.split_join_token(f"{secret}.short")
    url, transport = enrollment.pinned_enrolment("http://10.0.0.5:8083", PIN_R)
    assert url == "https://10.0.0.5:8083/control/v1/nodes/enroll"
    assert isinstance(transport, _http.MachineTransport)
    url, transport = enrollment.pinned_enrolment("https://nodes.example:8443", PIN_R)
    assert url == "https://nodes.example:8443/v1/nodes/enroll" and transport is None


async def test_enrolment_to_a_machine_with_another_key_sends_no_token(tmp_path: Path) -> None:
    """Something on the network answering for the root, with its own key:
    the join token never leaves this machine."""
    async with _Door(tmp_path / "impostor") as impostor:
        store = _store(tmp_path)
        with pytest.raises(enrollment.EnrollmentError) as caught:
            await enrollment.perform_enrollment(
                store=store,
                control_url=f"http://127.0.0.1:{impostor.port}",
                token=f"the-secret.{PIN_R}",
                name="me",
                advertise_url=None,
            )
        assert caught.value.slug == "control-root-key-refused"
        assert not impostor.control.seen and not impostor.agent.seen
        assert not store.record.enrolled


async def test_enrolment_to_the_root_s_key_goes_over_tls(tmp_path: Path) -> None:
    async with _Door(tmp_path / "root") as root_door:
        store = _store(tmp_path)
        with pytest.raises(enrollment.EnrollmentError):
            # The echo is no control root; what matters is where it went.
            await enrollment.perform_enrollment(
                store=store,
                control_url=f"http://127.0.0.1:{root_door.port}",
                token=f"the-secret.{root_door.pin}",
                name="me",
                advertise_url=None,
                machine={"machineUrl": "https://10.0.0.6:8083", "tlsPin": "M" * 43},
            )
        (request,) = root_door.control.seen
        assert request["path"] == "/v1/nodes/enroll"
        sent = json.loads(request["body"])
        assert sent["token"] == "the-secret"
        assert sent["machineUrl"] == "https://10.0.0.6:8083" and sent["tlsPin"] == "M" * 43


async def test_the_announcement_signs_the_machine_port_and_never_says_less(
    tmp_path: Path,
) -> None:
    """The machine port is under the node's signature, and a refusal never
    makes the node announce its address alone (agent#38 review: a party in
    the middle answers 401 too, and the address alone used to clear the key
    at the root)."""
    root = FakeRoot()
    store = _store(tmp_path)
    enroll_store(store, root, "me", control_url="http://control.invalid:8083")
    bodies: list[dict[str, Any]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "machineUrl" in body:
            return httpx.Response(401, json={"detail": "Signature rejected"})
        return httpx.Response(200, json={"changed": True})

    machine = {"machineUrl": "https://10.0.0.6:8083", "tlsPin": "M" * 43}
    outcome = await enrollment.announce_address(
        store=store,
        url="http://10.0.0.6:8079",
        transport=httpx.MockTransport(answer),
        machine=machine,
    )
    assert not outcome.announced
    (first,) = bodies
    verify_key = nacl.signing.VerifyKey(base64.b64decode(store.record.signing_public_key or ""))

    def verifies(body: dict[str, Any], fields: dict[str, str]) -> bool:
        message = node_identity.address_message(
            name="me", sequence=body["sequence"], url=body["url"], machine=fields
        )
        try:
            verify_key.verify(message, base64.b64decode(body["signature"]))
        except nacl.exceptions.BadSignatureError:
            return False
        return True

    assert first["tlsPin"] == "M" * 43 and verifies(first, machine)
    assert not verifies(first, {}), "a signature over the address alone would let a key be added"


# --------------------------------------------------------------------------- #
# agent#38, the second model's review
# --------------------------------------------------------------------------- #


def _door_app(tmp_path: Path, store: NodeIdentityStore, door: Any = None) -> Any:
    from types import SimpleNamespace

    from eugene_plexus_agent.settings import Settings
    from eugene_plexus_agent.state import AgentState

    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    settings = Settings(
        config_file=tmp_path / "agent.yaml", bind_port=8079, machine_port_enabled=True
    )
    return SimpleNamespace(
        state=SimpleNamespace(
            machine_port=door, agent_state=state, node_identity=store, settings=settings
        )
    )


async def test_a_machine_key_that_does_not_load_still_announces_its_door(tmp_path: Path) -> None:
    """Review 7b: a key that does not load opened no door, and the node was
    announced with none, so the root took it for an older version. It
    announces the door its certificate names, and says why it does not
    answer."""
    store = _store(tmp_path)
    record = store.record
    pin = node_identity.machine_pin(record.tls_certificate or "")
    # Another key under the same certificate: OpenSSL refuses the pair.
    other, _ = node_identity.generate_machine_key()
    import dataclasses

    store._record = dataclasses.replace(record, tls_private_key=other)
    app = _door_app(tmp_path, store)
    assert await machine_port.start(app) is None
    assert app.state.machine_port is None, "nothing takes the failure for a running port"
    fields = machine_port.announced(app, "http://10.0.0.6:8079")
    assert fields is not None
    assert (fields["machineUrl"], fields["tlsPin"]) == ("https://10.0.0.6:8083", pin)
    assert fields["machineError"].startswith("its machine key does not load")


def test_an_ipv6_address_announces_its_door_and_why_it_does_not_answer(tmp_path: Path) -> None:
    """Review 7b: an IPv6 advertise address named no door (the port listens
    on IPv4), so the node was taken for an older version. It announces the
    door, and that the port does not answer there."""
    from types import SimpleNamespace

    store = _store(tmp_path)
    door = SimpleNamespace(status=machine_port.Status(port=8083, pin="P" * 43, listening=True))
    app = _door_app(tmp_path, store, door)
    fields = machine_port.announced(app, "http://[fd00::6]:8079")
    assert fields is not None
    assert fields["machineUrl"] == "https://[fd00::6]:8083"
    assert "IPv4 only" in fields["machineError"]
    plain = machine_port.announced(app, "http://10.0.0.6:8079")
    assert plain == {"machineUrl": "https://10.0.0.6:8083", "tlsPin": "P" * 43}
    assert machine_port.machine_url(app) is None or ":" not in str(machine_port.machine_url(app))


def test_the_console_s_hops_go_by_their_route_alone() -> None:
    """Review 1b: the console's proxy and the forwarded sign-in carry what a
    person sent (a passphrase, a cookie), not only tokens; their clients are
    machine-hop clients, which send nothing in plain HTTP to a machine the
    table does not name."""
    from types import SimpleNamespace

    from eugene_plexus_agent.routes import oidc_forward, proxy

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    for client in (
        proxy.get_client(request),  # type: ignore[arg-type]
        oidc_forward._client(request, "http://root.lan:8083"),  # type: ignore[arg-type]
    ):
        transport = client._transport
        assert isinstance(transport, _http.MachineTransport) and transport.hop


async def test_the_site_host_s_report_to_the_root_takes_the_machine_transport() -> None:
    """Review 1a: a root reached as `root.lan` (a dotted name, so not
    "internal") got the egress client, and the report's Bearer went out with
    no machine routing, under `required` too."""
    from types import SimpleNamespace

    from eugene_plexus_agent.site_host import SiteHostSupervisor

    app = SimpleNamespace(
        state=SimpleNamespace(
            node_identity=SimpleNamespace(
                record=SimpleNamespace(
                    enrolled=True, control_url="http://root.lan:8083", name="amish"
                )
            ),
            auth_state=SimpleNamespace(trust=SimpleNamespace(agent_token=lambda audience: "t")),
        )
    )
    _http.set_machine_table(
        _http.parse_machine_table({"routes": {"http://root.lan:8083": None}, "required": True})
    )
    supervisor = SiteHostSupervisor(app)  # type: ignore[arg-type]
    try:
        supervisor.site = None
        await supervisor._report()  # required, and no key for the root: refused, nothing sent
        assert supervisor._root_client is not None
        transport = supervisor._root_client._transport
        assert isinstance(transport, _http.MachineTransport) and transport.hop
    finally:
        if supervisor._root_client is not None:
            await supervisor._root_client.aclose()
        await supervisor._host.aclose()


async def test_a_hop_with_a_passphrase_goes_nowhere_unknown() -> None:
    """Review 1b, behaviour: a sign-in through the console's proxy to a
    machine the table does not name yet is not sent, though it carries no
    token; to an older machine it still goes."""
    sent: list[str] = []

    def plain(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, json={})

    _http.set_machine_table(_http.parse_machine_table({"routes": {"http://10.9.9.8:8079": None}}))
    async with _http.internal_client(timeout=2, machine_hop=True) as client:
        transport = client._transport
        assert isinstance(transport, _http.MachineTransport)
        transport._plain = httpx.MockTransport(plain)  # type: ignore[assignment]
        with pytest.raises(_http.MachineRefused):
            await client.post("http://10.9.9.7:8079/v1/auth/login", json={"passphrase": "p"})
        await client.post("http://10.9.9.8:8079/v1/auth/login", json={"passphrase": "p"})
    assert sent == ["http://10.9.9.8:8079/v1/auth/login"]
