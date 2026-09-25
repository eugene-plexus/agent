"""This host in an install: identity, enrollment, the trust bundle, and
what each of them changes about the rest of the agent.

The control root is a fake behind an httpx `MockTransport`, and a real
listening socket, because the advertise address is derived from the
local end of a TCP connection to it and a fake that skipped that step
would agree with whatever the code guessed. The fake signs real bundles
and real sessions with keys of its own (`FakeRoot`), so every token that
verifies here verified against a bundle, the way it will in production
(per-node token keys, 2026-09-25).
"""

from __future__ import annotations

import base64
import json
import logging
import socket
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
import nacl.exceptions
import nacl.signing
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import enrollment, node_identity, tokens
from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind, SpawnConfig
from eugene_plexus_agent.app import create_app, shared_child_env
from eugene_plexus_agent.node_identity import NodeIdentityStore
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.supervisor import _ComponentPlanner

from .conftest import (
    TEST_PASSPHRASE,
    FakeRoot,
    StubRuntimeSupervisor,
    StubSupervisor,
    fake_devices,
    local_service_token,
)

JOIN_TOKEN = "join-token-minted-for-this-test"


def _verify_ed25519(public: str, message: bytes, signature: str) -> bool:
    try:
        nacl.signing.VerifyKey(base64.b64decode(public, validate=True)).verify(
            message, base64.b64decode(signature, validate=True)
        )
    except (nacl.exceptions.CryptoError, ValueError):
        return False
    return True


class FakeControl:
    """A control root behind a MockTransport, and a real listening socket
    so the agent's advertise-host derivation has something to connect to.

    It keeps a registry of each node's token key and grants and signs
    trust bundles over it exactly as the real root does, so what this
    agent accepts is what a real bundle would have let it accept.
    """

    def __init__(self) -> None:
        self.root = FakeRoot()
        # Readable plaintext, not key-shaped base64 next to a key-shaped
        # name — gitleaks flags the latter.
        self.recovery_public = base64.b64encode(
            b"recovery-recipient-public-key-32".ljust(32, b"0")[:32]
        ).decode()
        self.refuse: tuple[int, dict[str, Any]] | None = None
        self.corrupt: str | None = None
        self.enroll_requests: list[dict[str, Any]] = []
        self.nodes: dict[str, dict[str, Any]] = {}
        self.address_requests: list[dict[str, Any]] = []
        self.revoked: list[str] = []
        self.refuse_revoke: int | None = None
        self.logins: list[dict[str, Any]] = []
        self.sign_outs: list[str] = []
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._listener.getsockname()[1]}"

    @property
    def public(self) -> str:
        return self.root.public

    def close(self) -> None:
        self._listener.close()

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/nodes/enroll":
                return self._enroll(request)
            if path == "/v1/auth/login":
                return self._login(request)
            if path == "/v1/auth/sessions/current":
                self.sign_outs.append(request.headers.get("authorization", ""))
                return httpx.Response(204)
            if path == "/v1/trust/bundle":
                return httpx.Response(200, json={"jws": self.root.bundle().jws})
            if path.startswith("/v1/nodes/") and request.method == "PATCH":
                return self._announce(path.rsplit("/", 1)[-1], request)
            if path.startswith("/v1/nodes/") and request.method == "DELETE":
                return self._revoke(path.rsplit("/", 1)[-1], request)
            return httpx.Response(404, json={"detail": "no such route on the fake control"})

        return httpx.MockTransport(handle)

    def _enroll(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.enroll_requests.append(body)
        if self.refuse is not None:
            code, payload = self.refuse
            return httpx.Response(code, json={"detail": payload})
        name = body["name"]
        self.nodes[name] = {
            "url": body.get("url"),
            "signingPublicKey": body.get("signingPublicKey"),
            "advertiseSequence": 0,
        }
        self.root.register(name, tokens.load_public(body["tokenPublicKey"]))
        bundle = self.root.bundle()
        payload: dict[str, Any] = {
            "name": name,
            "epoch": self.root.epoch,
            "trustBundle": {"jws": bundle.jws},
            "controlPublicKey": self.public,
            "recoveryPublicKey": self.recovery_public,
        }
        if self.corrupt == "no-bundle":
            payload.pop("trustBundle")
        if self.corrupt == "foreign-bundle":
            payload["trustBundle"] = {"jws": FakeRoot().bundle().jws}
        if self.corrupt == "not-listed":
            self.root.members.pop(name)
            payload["trustBundle"] = {"jws": self.root.bundle().jws}
        if self.corrupt == "misnamed":
            # This node's own key, listed as some other machine: trusting
            # it would let this node's tokens speak as that one.
            key = self.root.members.pop(name).public
            self.root.register("someone-else", key)
            payload["trustBundle"] = {"jws": self.root.bundle().jws}
        return httpx.Response(201, json=payload)

    def _login(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        header = request.headers.get("authorization", "")
        self.logins.append({"passphrase": body.get("passphrase"), "authorization": header})
        if body.get("passphrase") != TEST_PASSPHRASE:
            return httpx.Response(401, json={"detail": {"title": "Wrong passphrase"}})
        aud = ["control"]
        try:
            actor = tokens.verify(
                header.removeprefix("Bearer "),
                bundle=self.root.bundle(),
                recipient="control",
                classes=[tokens.TYP_SERVICE],
            )
            if actor.sub == "agent" and actor.issuer_node:
                aud = [f"node:{actor.issuer_node}", "control"]
        except tokens.TokenError:
            pass
        token = self.root.session(*aud)
        return httpx.Response(
            200,
            json={"sessionToken": token, "expiresAt": "2099-01-01T00:00:00+00:00"},
        )

    def _announce(self, name: str, request: httpx.Request) -> httpx.Response:
        """The real verification: the key enrollment recorded, over the
        **raw body's** url."""
        body = json.loads(request.content)
        self.address_requests.append({"name": name, **body})
        record = self.nodes.get(name)
        if record is None:
            return httpx.Response(404, json={"detail": "no such node"})
        public = record.get("signingPublicKey")
        if not public:
            return httpx.Response(401, json={"detail": "no signing key; re-enroll"})
        message = node_identity.address_message(
            name=name, sequence=int(body["sequence"]), url=body["url"]
        )
        if not _verify_ed25519(public, message, body["signature"]):
            return httpx.Response(401, json={"detail": "signature rejected"})
        if body["url"] == record["url"]:
            return httpx.Response(
                200,
                json={
                    "name": name,
                    "url": record["url"],
                    "sequence": record["advertiseSequence"],
                    "changed": False,
                },
            )
        if int(body["sequence"]) <= int(record["advertiseSequence"]):
            return httpx.Response(409, json={"detail": "stale announcement"})
        record["url"] = body["url"]
        record["advertiseSequence"] = int(body["sequence"])
        return httpx.Response(
            200,
            json={
                "name": name,
                "url": record["url"],
                "sequence": record["advertiseSequence"],
                "changed": True,
            },
        )

    def _revoke(self, name: str, request: httpx.Request) -> httpx.Response:
        if self.refuse_revoke is not None:
            return httpx.Response(self.refuse_revoke, json={"detail": "refused"})
        if not request.headers.get("authorization"):
            return httpx.Response(401, json={"detail": "operator token required"})
        if name not in self.nodes:
            return httpx.Response(404, json={"detail": "no such node"})
        self.nodes.pop(name)
        self.root.members.pop(name, None)
        self.revoked.append(name)
        return httpx.Response(202, json={"reason": "revocation", "version": self.root.version})


@pytest.fixture
def control() -> Iterator[FakeControl]:
    fake = FakeControl()
    yield fake
    fake.close()


def _enroll(client: TestClient, control: FakeControl, **body: Any) -> httpx.Response:
    """Enroll, and on success sign in again, through the root.

    The session that asked for the enrollment was signed by this agent as
    its own authority and stops verifying the moment it trusts the
    install's bundle instead; that is the designed price of enrollment.
    Signing in again forwards to the fake root, so every test past this
    point is also an assertion that a root-minted session verifies here.
    """
    client.app.state.control_transport = control.transport()  # type: ignore[attr-defined]
    payload: dict[str, Any] = {"controlUrl": control.url, "token": JOIN_TOKEN, "name": "gpu-box"}
    payload.update(body)
    response = client.post("/v1/node/enroll", json=payload)
    if response.status_code == 200:
        signed_in = client.post(
            "/v1/auth/login",
            json={"passphrase": TEST_PASSPHRASE},
            headers={"Authorization": ""},
        )
        assert signed_in.status_code == 200, signed_in.text
        client.headers["Authorization"] = f"Bearer {signed_in.json()['sessionToken']}"
    return response


def _auth(client: TestClient) -> Any:
    return client.app.state.auth_state  # type: ignore[attr-defined]


def _restarts(client: TestClient) -> int:
    """Restarts since the fixture set up: `initialize` restarts children
    once itself (so they pick up the master key), and that one is not
    what these tests are counting."""
    stub: StubSupervisor = client.app.state.supervisor  # type: ignore[attr-defined]
    return sum(1 for call in stub.calls if call[0] == "restart_all") - 1


# --------------------------------------------------------------------------- #
# GET /v1/node, unenrolled
# --------------------------------------------------------------------------- #


def test_node_reports_devices_and_unenrolled(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/node")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["enrolled"] is False
    for absent in ("controlUrl", "name", "epoch", "trustBundleVersion", "controlPublicKey"):
        assert absent not in body
    assert body["tokenPublicKey"], "a standalone node has a token key of its own"
    assert body["os"] in ("windows", "linux", "macos")
    assert body["arch"] in ("x64", "arm64")
    kinds = [d["kind"] for d in body["devices"]]
    assert kinds == ["cuda", "cpu"]
    assert body["devices"][0]["memoryFreeBytes"] == 24 * 1024**3
    assert body["agentVersion"]


def test_node_carries_this_hosts_own_clock(authed_client: TestClient) -> None:
    """Bracketed the way a caller is told to bracket it."""
    before = datetime.now(UTC)
    response = authed_client.get("/v1/node")
    after = datetime.now(UTC)
    assert response.status_code == 200, response.text
    reported = response.json()["time"]
    assert reported.endswith("Z") or "+" in reported[10:] or reported[10:].count("-") > 0
    parsed = datetime.fromisoformat(reported)
    assert parsed.tzinfo is not None
    assert before <= parsed <= after


def test_the_clock_is_read_per_request_not_at_startup(authed_client: TestClient) -> None:
    first = datetime.fromisoformat(authed_client.get("/v1/node").json()["time"])
    time.sleep(0.01)
    second = datetime.fromisoformat(authed_client.get("/v1/node").json()["time"])
    assert second > first


def test_node_is_readable_with_a_childs_local_token(client: TestClient) -> None:
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    token = local_service_token(client.app, "gateway")  # type: ignore[arg-type]
    response = client.get("/v1/node", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_node_requires_auth(client: TestClient) -> None:
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    assert client.get("/v1/node").status_code == 401


# --------------------------------------------------------------------------- #
# Enrollment
# --------------------------------------------------------------------------- #


def test_enrollment_sends_public_keys_only_and_keeps_the_bundle(
    authed_client: TestClient, control: FakeControl, settings: Settings
) -> None:
    """The whole exchange, from the agent's side.

    Public keys go out — sealing, identity and the token key — and a
    bundle that lists this node's token key comes back and is kept. No
    private key crosses the wire in either direction (2026-09-25).
    """
    response = _enroll(authed_client, control)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["enrolled"] is True
    assert body["name"] == "gpu-box"
    assert body["epoch"] == 1
    assert body["controlPublicKey"] == control.public
    assert body["trustBundleVersion"] >= 1
    assert urlparse(body["controlUrl"]).port == urlparse(control.url).port
    advertised = urlparse(body["advertiseUrl"])
    assert (advertised.hostname, advertised.port) == ("127.0.0.1", settings.bind_port)
    assert "privateKey" not in json.dumps(body)

    sent = control.enroll_requests[-1]
    assert sent["token"] == JOIN_TOKEN
    assert sent["name"] == "gpu-box"
    for key in ("publicKey", "signingPublicKey", "tokenPublicKey"):
        assert len(base64.b64decode(sent[key], validate=True)) == 32
    assert sent["tokenPublicKey"] == body["tokenPublicKey"]
    assert sent["url"] == f"http://127.0.0.1:{settings.bind_port}"
    assert [d["kind"] for d in sent["devices"]] == ["cuda", "cpu"]
    assert "rivate" not in json.dumps(sent), "no private key goes to the root"
    assert _restarts(authed_client) == 1

    node_file = settings.config_file.parent / "node.yaml"
    text = node_file.read_text(encoding="utf-8")
    assert "tokenPrivateKey:" in text and "signingKey:" not in text
    assert JOIN_TOKEN not in text
    kept = json.loads((settings.config_file.parent / "trust_bundle.json").read_text("utf-8"))
    assert tokens.parse_bundle(kept["jws"], authority=control.public).version >= 1


def test_a_root_minted_session_verifies_and_a_self_minted_one_no_longer_does(
    authed_client: TestClient, control: FakeControl
) -> None:
    """Enrolling changes the authority: the session this agent minted as
    its own authority is dead, and the root's is good."""
    stale = dict(authed_client.headers)
    assert _enroll(authed_client, control).status_code == 200
    assert authed_client.get("/v1/node", headers=stale).status_code == 401
    assert authed_client.get("/v1/node").status_code == 200
    # And a session the root addressed to some other console is not ours.
    elsewhere = control.root.session("node:laptop", "control")
    assert (
        authed_client.get("/v1/node", headers={"Authorization": f"Bearer {elsewhere}"}).status_code
        == 401
    )


def test_enrolling_twice_is_a_409_that_names_the_way_out(
    authed_client: TestClient, control: FakeControl
) -> None:
    assert _enroll(authed_client, control).status_code == 200
    again = _enroll(authed_client, control)
    assert again.status_code == 409
    assert "gpu-box" in again.text and "/v1/node/unenroll" in again.text
    assert len(control.enroll_requests) == 1, "a second enrollment must not reach the root"


def test_a_root_that_refuses_the_token_is_a_502_and_nothing_is_recorded(
    authed_client: TestClient, control: FakeControl
) -> None:
    control.refuse = (401, {"title": "Join token rejected", "detail": "join token is unknown"})
    response = _enroll(authed_client, control)
    assert response.status_code == 502
    assert "401" in response.text and "join token is unknown" in response.text
    assert authed_client.get("/v1/node").json()["enrolled"] is False
    assert _restarts(authed_client) == 0


@pytest.mark.parametrize(
    ("corruption", "words"),
    [
        ("no-bundle", "trustBundle"),
        ("foreign-bundle", "does not verify"),
        ("not-listed", "does not list this node"),
        ("misnamed", "as node:gpu-box"),
    ],
)
def test_an_enrollment_whose_bundle_cannot_be_trusted_records_nothing(
    authed_client: TestClient, control: FakeControl, corruption: str, words: str
) -> None:
    """A root whose bundle is missing, signed by someone else, or silent
    about this node would leave an enrolled node that trusts nothing."""
    control.corrupt = corruption
    response = _enroll(authed_client, control)
    assert response.status_code == 502, response.text
    assert words in response.text
    assert authed_client.get("/v1/node").json()["enrolled"] is False


def test_an_unreachable_root_is_a_502(authed_client: TestClient) -> None:
    def refuse_everything(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    authed_client.app.state.control_transport = httpx.MockTransport(refuse_everything)  # type: ignore[attr-defined]
    response = authed_client.post(
        "/v1/node/enroll", json={"controlUrl": "http://127.0.0.1:1", "token": JOIN_TOKEN}
    )
    assert response.status_code == 502
    assert "still unenrolled" in response.text


def test_a_configured_advertise_url_wins_over_derivation(
    authed_client: TestClient, control: FakeControl
) -> None:
    patched = authed_client.patch("/v1/config", json={"advertiseUrl": "http://100.64.0.7:8079"})
    assert patched.status_code == 200, patched.text
    assert _enroll(authed_client, control).status_code == 200
    assert control.enroll_requests[-1]["url"] == "http://100.64.0.7:8079"
    assert authed_client.get("/v1/node").json()["advertiseUrl"].rstrip("/") == (
        "http://100.64.0.7:8079"
    )


def test_the_name_defaults_to_the_hostname(authed_client: TestClient, control: FakeControl) -> None:
    response = _enroll(authed_client, control, name=None)
    assert response.status_code == 200, response.text
    assert control.enroll_requests[-1]["name"] == socket.gethostname()
    assert response.json()["name"] == socket.gethostname()


def test_identity_and_the_bundle_survive_a_restart(
    authed_client: TestClient,
    control: FakeControl,
    settings: Settings,
) -> None:
    """A second process over the same directory comes up enrolled,
    trusting the kept bundle from its first request."""
    assert _enroll(authed_client, control).status_code == 200

    reborn = create_app(settings=settings)
    reborn.state.supervisor = StubSupervisor()
    reborn.state.runtime_supervisor = StubRuntimeSupervisor()
    reborn.state.device_detector = lambda: fake_devices()
    reborn.state.library_fit_client = None
    with TestClient(reborn) as client:
        session = control.root.session("node:gpu-box", "control")
        response = client.get("/v1/node", headers={"Authorization": f"Bearer {session}"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["enrolled"] is True
        assert body["name"] == "gpu-box"
        assert body["controlPublicKey"] == control.public


def test_runtimes_report_the_node_once_enrolled(
    authed_client: TestClient, control: FakeControl
) -> None:
    spec = {
        "name": "qwen",
        "engine": "llama_cpp",
        "modelPath": "/models/qwen.gguf",
        "autoStart": False,
    }
    created = authed_client.post("/v1/runtimes", json=spec)
    assert created.status_code == 201, created.text
    assert created.json()["node"] is None
    assert _enroll(authed_client, control).status_code == 200
    listed = authed_client.get("/v1/runtimes").json()["runtimes"]
    assert [r["node"] for r in listed] == ["gpu-box"]


# --------------------------------------------------------------------------- #
# The trust bundle: push, pull, fencing
# --------------------------------------------------------------------------- #


def _push(client: TestClient, bundle: tokens.TrustBundle) -> httpx.Response:
    return client.post(
        "/v1/node/trust-bundle", json={"jws": bundle.jws}, headers={"Authorization": ""}
    )


def test_a_newer_bundle_is_taken_and_nothing_restarts(
    authed_client: TestClient, control: FakeControl, settings: Settings
) -> None:
    """A revocation elsewhere in the install is a new bundle here, and it
    takes effect without a restart: children reload the file."""
    assert _enroll(authed_client, control).status_code == 200
    control.root.register("attic", tokens.generate_private_key().public_key())
    newer = control.root.bundle()
    response = _push(authed_client, newer)
    assert response.status_code == 200, response.text
    assert response.json()["trustBundleVersion"] == newer.version
    assert _restarts(authed_client) == 1, "only enrollment restarted anything"
    kept = json.loads((settings.config_file.parent / "trust_bundle.json").read_text("utf-8"))
    assert kept["jws"] == newer.jws


def test_a_bundle_signed_by_another_root_is_401_and_changes_nothing(
    authed_client: TestClient, control: FakeControl
) -> None:
    assert _enroll(authed_client, control).status_code == 200
    held = _auth(authed_client).trust.bundle
    response = _push(authed_client, FakeRoot().bundle())
    assert response.status_code == 401
    assert "not signed by the control root" in response.text
    assert _auth(authed_client).trust.bundle == held


def test_an_older_bundle_is_refused_as_a_rollback(
    authed_client: TestClient, control: FakeControl
) -> None:
    """A captured bundle replayed after a revocation would bring a revoked
    key back; the version refuses it."""
    assert _enroll(authed_client, control).status_code == 200
    old = control.root.bundle()
    newer = control.root.bundle()
    assert _push(authed_client, newer).status_code == 200
    stale = _push(authed_client, old)
    assert stale.status_code == 409
    assert "below" in stale.text
    assert authed_client.get("/v1/node").json()["trustBundleVersion"] == newer.version


def test_a_lower_epoch_is_fenced_with_409(authed_client: TestClient, control: FakeControl) -> None:
    """M5 §6, on the side that does the fencing: a superseded root is
    refused by this node alone, with no election."""
    assert _enroll(authed_client, control).status_code == 200
    control.root.epoch = 3
    announced = _push(authed_client, control.root.bundle())
    assert announced.status_code == 200, announced.text
    assert announced.json()["epoch"] == 3

    control.root.epoch = 2
    stale = _push(authed_client, control.root.bundle())
    assert stale.status_code == 409
    assert "epoch" in stale.text
    assert authed_client.get("/v1/node").json()["epoch"] == 3


def test_a_lost_bundle_still_fences_a_lower_epoch(
    authed_client: TestClient, control: FakeControl, settings: Settings
) -> None:
    """The kept bundle is the usual fence; the epoch in `node.yaml` is the
    one that survives losing it. A superseded root cannot re-seat itself
    on a node that deleted, or never wrote, its bundle file."""
    assert _enroll(authed_client, control).status_code == 200
    control.root.epoch = 3
    assert _push(authed_client, control.root.bundle()).status_code == 200
    (settings.config_file.parent / "trust_bundle.json").unlink()
    _auth(authed_client).trust.forget()

    control.root.epoch = 2
    stale = _push(authed_client, control.root.bundle())
    assert stale.status_code == 409, stale.text
    assert _auth(authed_client).trust.bundle is None


def test_an_unenrolled_agent_refuses_a_bundle(
    authed_client: TestClient, control: FakeControl
) -> None:
    response = _push(authed_client, control.root.bundle())
    assert response.status_code == 401
    assert "not enrolled" in response.text.lower()


def test_revoking_a_key_in_the_bundle_ends_its_tokens_here(
    authed_client: TestClient, control: FakeControl
) -> None:
    """A gateway on another machine, granted and then revoked: its token
    works until this node takes the bundle without it, and not after."""
    assert _enroll(authed_client, control).status_code == 200
    nas = tokens.Signer(key=tokens.generate_private_key(), issuer="node:nas")
    control.root.register("nas", nas.key.public_key(), ("gateway",))
    assert _push(authed_client, control.root.bundle()).status_code == 200
    token, _ = nas.mint(
        typ=tokens.TYP_SERVICE, sub="gateway", aud=["node:gpu-box"], ttl_seconds=600
    )
    headers = {"Authorization": f"Bearer {token}"}
    assert authed_client.get("/v1/runtimes", headers=headers).status_code == 200

    control.root.members.pop("nas")
    assert _push(authed_client, control.root.bundle()).status_code == 200
    assert authed_client.get("/v1/runtimes", headers=headers).status_code == 401


def test_another_machines_agent_reads_nothing_here(
    authed_client: TestClient, control: FakeControl
) -> None:
    """D5: this agent's reads take its own children, the root, and a
    gateway -- not another machine's agent. A leaked worker key signs
    `sub: agent` for every machine in the install, and buys none of them."""
    assert _enroll(authed_client, control).status_code == 200
    attic = tokens.Signer(key=tokens.generate_private_key(), issuer="node:attic")
    control.root.register("attic", attic.key.public_key())
    assert _push(authed_client, control.root.bundle()).status_code == 200
    token, _ = attic.mint(
        typ=tokens.TYP_SERVICE, sub="agent", aud=["node:gpu-box"], ttl_seconds=600
    )
    headers = {"Authorization": f"Bearer {token}"}
    for path in ("/v1/runtimes", "/v1/components", "/v1/node"):
        assert authed_client.get(path, headers=headers).status_code == 401, path


def test_only_this_agents_children_may_ask_it_for_a_token(
    authed_client: TestClient, control: FakeControl
) -> None:
    """`POST /v1/auth/service-token` signs with this node's key, so it is
    for this machine's own components: not another machine's agent, not
    a granted gateway elsewhere, not even the operator."""
    assert _enroll(authed_client, control).status_code == 200
    nas = tokens.Signer(key=tokens.generate_private_key(), issuer="node:nas")
    control.root.register("nas", nas.key.public_key(), ("gateway",))
    assert _push(authed_client, control.root.bundle()).status_code == 200
    body = {"audience": "node:nas"}
    for sub in ("agent", "gateway"):
        remote, _ = nas.mint(typ=tokens.TYP_SERVICE, sub=sub, aud=["node:gpu-box"], ttl_seconds=600)
        refused = authed_client.post(
            "/v1/auth/service-token", json=body, headers={"Authorization": f"Bearer {remote}"}
        )
        assert refused.status_code == 401, (sub, refused.text)
    assert authed_client.post("/v1/auth/service-token", json=body).status_code == 401
    child = local_service_token(authed_client.app, "agent")  # type: ignore[arg-type]
    granted = authed_client.post(
        "/v1/auth/service-token", json=body, headers={"Authorization": f"Bearer {child}"}
    )
    assert granted.status_code == 200, granted.text


# --------------------------------------------------------------------------- #
# Who may declare a runtime
# --------------------------------------------------------------------------- #


def test_only_the_control_roots_own_token_may_declare_a_runtime(
    authed_client: TestClient, control: FakeControl
) -> None:
    """The root forwards declarations with a token signed by its own key
    and addressed to this node. A child's local token cannot declare, and
    neither can a gateway's: a leaked worker key runs nothing here."""
    assert _enroll(authed_client, control).status_code == 200
    spec = {"name": "qwen", "engine": "llama_cpp", "modelPath": "/m.gguf", "autoStart": False}

    for refused_token in (
        local_service_token(authed_client.app, "library"),  # type: ignore[arg-type]
        local_service_token(authed_client.app, "gateway"),  # type: ignore[arg-type]
        # This node's own key can sign `sub: control`; only the root's
        # key is the control root. A stolen node key declares nothing.
        local_service_token(authed_client.app, "control"),  # type: ignore[arg-type]
        control.root.service("node:attic"),
    ):
        refused = authed_client.post(
            "/v1/runtimes", json=spec, headers={"Authorization": f"Bearer {refused_token}"}
        )
        assert refused.status_code == 401, refused.text

    root_token = control.root.service("node:gpu-box")
    accepted = authed_client.post(
        "/v1/runtimes", json=spec, headers={"Authorization": f"Bearer {root_token}"}
    )
    assert accepted.status_code == 201, accepted.text
    denied = authed_client.delete(
        "/v1/runtimes/qwen", headers={"Authorization": f"Bearer {root_token}"}
    )
    assert denied.status_code == 401, "deleting it is still the operator's"


# --------------------------------------------------------------------------- #
# The advertise address, on components and on children
# --------------------------------------------------------------------------- #


def _driver_entry(name: str, port: int, *, spawn: bool = True) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "kind": "inference-driver",
        "url": f"http://127.0.0.1:{port}",
    }
    if spawn:
        entry["spawn"] = {"configFile": f"/tmp/{name}.yaml"}
    return entry


def test_components_carry_an_advertise_url_only_when_the_node_advertises(
    authed_client: TestClient,
) -> None:
    created = authed_client.post("/v1/components", json=_driver_entry("d1", 8091))
    assert created.status_code == 201, created.text
    assert created.json()["advertiseUrl"] is None, "a single-host install has nothing to advertise"

    assert (
        authed_client.patch(
            "/v1/config", json={"advertiseUrl": "http://100.64.0.7:8079"}
        ).status_code
        == 200
    )
    spawned = authed_client.get("/v1/components/d1").json()
    assert urlparse(spawned["url"]).hostname == "127.0.0.1", "url keeps its one meaning"
    assert (urlparse(spawned["advertiseUrl"]).hostname, urlparse(spawned["advertiseUrl"]).port) == (
        "100.64.0.7",
        8091,
    )

    remote = authed_client.post("/v1/components", json=_driver_entry("far", 8092, spawn=False))
    assert remote.status_code == 201
    assert remote.json()["advertiseUrl"] is None, (
        "a remote entry is already where the operator said"
    )

    listed = {c["name"]: c for c in authed_client.get("/v1/components").json()["components"]}
    assert listed["d1"]["advertiseUrl"] and listed["far"]["advertiseUrl"] is None


def test_children_are_told_their_agents_url_and_a_bind_host_when_advertising(
    settings: Settings, tmp_path: Any
) -> None:
    from eugene_plexus_agent.state import AgentState

    state = AgentState(settings.config_file)
    state.load()
    identity = NodeIdentityStore(tmp_path / "node.yaml")
    identity.load()

    # Loopback everywhere: the agent's URL carries its port; nothing widens.
    env = shared_child_env(
        Settings(config_file=settings.config_file, bind_port=8084), state, identity
    )
    assert env == {"AGENT_URL": "http://127.0.0.1:8084"}

    # A wildcard bind still resolves children to loopback.
    wildcard = Settings(config_file=settings.config_file, bind_host="0.0.0.0", bind_port=8079)
    assert shared_child_env(wildcard, state, identity)["AGENT_URL"] == "http://127.0.0.1:8079"

    # A specific interface has to be reached there.
    pinned = Settings(config_file=settings.config_file, bind_host="100.64.0.7", bind_port=8079)
    assert shared_child_env(pinned, state, identity)["AGENT_URL"] == "http://100.64.0.7:8079"

    # Advertising a non-loopback address widens the children's bind.
    from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest

    state.apply_config_patch(
        ConfigUpdateRequest.model_validate({"advertiseUrl": "http://100.64.0.7:8079"})
    )
    assert shared_child_env(settings, state, identity)["BIND_HOST"] == "0.0.0.0"

    # Advertising loopback — the same-box pair — does not.
    state.apply_config_patch(
        ConfigUpdateRequest.model_validate({"advertiseUrl": "http://127.0.0.1:8084"})
    )
    assert "BIND_HOST" not in shared_child_env(settings, state, identity)


def test_the_planner_prefixes_shared_env_per_kind_and_the_operator_wins() -> None:
    shared = {"AGENT_URL": "http://127.0.0.1:8084", "BIND_HOST": "0.0.0.0"}
    driver = ComponentEntry(
        name="d1",
        kind=ComponentKind.inference_driver,
        url="http://127.0.0.1:8091",  # type: ignore[arg-type]
        spawn=SpawnConfig(configFile="/tmp/d1.yaml"),
    )
    plan = _ComponentPlanner(driver, logging.getLogger("t"), None, lambda: shared).plan()
    assert plan is not None
    assert plan.env["EUGENE_PLEXUS_DRIVER_AGENT_URL"] == "http://127.0.0.1:8084"
    assert plan.env["EUGENE_PLEXUS_DRIVER_BIND_HOST"] == "0.0.0.0"
    assert plan.env["EUGENE_PLEXUS_DRIVER_BIND_PORT"] == "8091"

    # The trust root gets the bootstrap values too — it has to be reachable
    # by standbys and remote agents — and still no auth trio.
    control = ComponentEntry(
        name="control",
        kind=ComponentKind.control,
        url="http://127.0.0.1:8083",  # type: ignore[arg-type]
        spawn=SpawnConfig(configFile="/tmp/control.yaml"),
    )
    plan = _ComponentPlanner(control, logging.getLogger("t"), None, lambda: shared).plan()
    assert plan is not None
    assert plan.env["EUGENE_PLEXUS_CONTROL_BIND_HOST"] == "0.0.0.0"
    assert not any("TRUST" in k or "SERVICE_TOKEN" in k for k in plan.env)

    # An explicit per-component value from the operator wins.
    pinned = ComponentEntry(
        name="d2",
        kind=ComponentKind.inference_driver,
        url="http://127.0.0.1:8092",  # type: ignore[arg-type]
        spawn=SpawnConfig(
            configFile="/tmp/d2.yaml", env={"EUGENE_PLEXUS_DRIVER_BIND_HOST": "10.0.0.5"}
        ),
    )
    plan = _ComponentPlanner(pinned, logging.getLogger("t"), None, lambda: shared).plan()
    assert plan is not None
    assert plan.env["EUGENE_PLEXUS_DRIVER_BIND_HOST"] == "10.0.0.5"


def test_derive_advertise_host_reads_the_local_end_of_the_socket(control: FakeControl) -> None:
    assert node_identity.derive_advertise_host(control.url) == "127.0.0.1"
    assert node_identity.derive_advertise_host("http://127.0.0.1:1", timeout=0.5) is None
    assert node_identity.derive_advertise_host("not a url") is None


def test_address_helpers() -> None:
    assert node_identity.local_agent_url("127.0.0.1", 8079) == "http://127.0.0.1:8079"
    assert node_identity.local_agent_url("0.0.0.0", 8084) == "http://127.0.0.1:8084"
    assert node_identity.local_agent_url("::", 8084) == "http://127.0.0.1:8084"
    assert node_identity.local_agent_url("100.64.0.7", 8079) == "http://100.64.0.7:8079"
    assert node_identity.format_url("fd7a::1", 8079) == "http://[fd7a::1]:8079"
    assert node_identity.is_loopback_host("127.0.0.1")
    assert node_identity.is_loopback_host("localhost")
    assert node_identity.is_loopback_host("::1")
    assert node_identity.is_loopback_host(None)
    assert not node_identity.is_loopback_host("100.64.0.7")
    assert not node_identity.is_loopback_host("gpu-box.tailnet")
    assert node_identity.effective_advertise_url("http://a:1/", "http://b:2") == "http://a:1"
    assert node_identity.effective_advertise_url("  ", "http://b:2/") == "http://b:2"
    assert node_identity.effective_advertise_url(None, None) is None


# --------------------------------------------------------------------------- #
# Leaving an install (M9)
# --------------------------------------------------------------------------- #


def test_unenroll_stops_trusting_the_install_and_tells_the_root(
    authed_client: TestClient, control: FakeControl
) -> None:
    """The inverse of enrolling. The assertion that matters is the last
    one: a session the root signs no longer verifies here."""
    assert _enroll(authed_client, control).status_code == 200
    session = control.root.session("node:gpu-box", "control")

    response = authed_client.post("/v1/node/unenroll")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["controlNotified"] is True
    assert body["previousName"] == "gpu-box"
    assert body["identity"]["enrolled"] is False
    assert control.revoked == ["gpu-box"]
    assert (
        authed_client.get("/v1/node", headers={"Authorization": f"Bearer {session}"}).status_code
        == 401
    )
    # Its own authority again, not the old root's: a token the old root
    # signs for a standalone machine is refused, and a sign-in here works.
    as_standalone = control.root.session("node:local")
    assert (
        authed_client.get(
            "/v1/node", headers={"Authorization": f"Bearer {as_standalone}"}
        ).status_code
        == 401
    )
    signed_in = authed_client.post(
        "/v1/auth/login", json={"passphrase": TEST_PASSPHRASE}, headers={"Authorization": ""}
    )
    assert signed_in.status_code == 200, signed_in.text
    own = {"Authorization": f"Bearer {signed_in.json()['sessionToken']}"}
    assert authed_client.get("/v1/node", headers=own).status_code == 200


def test_unenroll_keeps_this_nodes_own_keypairs(
    authed_client: TestClient, control: FakeControl
) -> None:
    """They are the *host's* identity, not the install's. The same node
    re-joining anywhere is still the same node, and a departing install
    holding only public halves can read nothing with them."""
    _enroll(authed_client, control)
    store: NodeIdentityStore = authed_client.app.state.node_identity  # type: ignore[attr-defined]
    before = (
        store.record.public_key,
        store.record.signing_public_key,
        store.record.token_private_key,
    )
    assert all(before)

    authed_client.post("/v1/node/unenroll")
    after = store.record
    assert (after.public_key, after.signing_public_key, after.token_private_key) == before
    # And everything that belonged to the install is gone.
    assert after.epoch is None
    assert after.control_url is None
    assert after.control_public_key is None
    assert after.advertise_sequence == 0


def test_unenroll_proceeds_when_the_control_root_is_unreachable(
    authed_client: TestClient, control: FakeControl
) -> None:
    """`degraded-mode-required`, applied to a trust operation: an operator
    detaching a node from a dead install is exactly the case where
    refusing is useless. The node leaves and *says* the root was not
    told, because the install still trusts a key nobody discarded."""
    _enroll(authed_client, control)

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("control root is gone")

    authed_client.app.state.control_transport = httpx.MockTransport(dead)  # type: ignore[attr-defined]
    response = authed_client.post("/v1/node/unenroll")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["controlNotified"] is False
    assert "DELETE /v1/nodes/gpu-box" in body["detail"]
    store: NodeIdentityStore = authed_client.app.state.node_identity  # type: ignore[attr-defined]
    assert store.record.enrolled is False


def test_unenroll_can_skip_telling_the_root(
    authed_client: TestClient, control: FakeControl
) -> None:
    """The expert override: an operator who already knows the root is gone
    should not wait out a timeout to detach."""
    _enroll(authed_client, control)
    response = authed_client.post("/v1/node/unenroll", json={"notifyControl": False})
    assert response.status_code == 200, response.text
    assert response.json()["controlNotified"] is False
    assert control.revoked == []


def test_unenroll_restarts_children_and_is_409_when_not_enrolled(
    authed_client: TestClient, control: FakeControl
) -> None:
    """Children read the authority they trust from their environment, so
    a change of authority reaches them only through a respawn - the same
    reason enrollment restarts them."""
    assert authed_client.post("/v1/node/unenroll").status_code == 409

    _enroll(authed_client, control)
    before = _restarts(authed_client)
    assert authed_client.post("/v1/node/unenroll").status_code == 200
    assert _restarts(authed_client) == before + 1


# --------------------------------------------------------------------------- #
# Saying where this host is (M9) - the defect that was live for four
# milestones: announced once at enrollment, never again.
# --------------------------------------------------------------------------- #


def test_enrollment_sends_the_nodes_signing_public_key(
    authed_client: TestClient, control: FakeControl
) -> None:
    """Without it the root has nothing to verify a re-advertisement
    against, and this node can never tell it that it moved."""
    _enroll(authed_client, control)
    store: NodeIdentityStore = authed_client.app.state.node_identity  # type: ignore[attr-defined]
    assert control.enroll_requests[-1]["signingPublicKey"]
    assert control.enroll_requests[-1]["signingPublicKey"] == store.record.signing_public_key
    # The two keys are different keys. One seals, one signs; neither can
    # do the other's job.
    assert store.record.signing_public_key != store.record.public_key


@pytest.mark.asyncio
async def test_a_node_that_moved_announces_its_new_address(
    authed_client: TestClient, control: FakeControl
) -> None:
    """The whole point. `announce_address` is what the lifespan runs on
    every boot; here it runs directly so the assertion is about the
    exchange rather than about task scheduling."""
    _enroll(authed_client, control)
    store: NodeIdentityStore = authed_client.app.state.node_identity  # type: ignore[attr-defined]
    # Whatever enrollment derived — the point is that it is not the
    # address this host is about to be found at.
    assert control.nodes["gpu-box"]["url"] != "http://10.0.0.9:8079"

    outcome = await enrollment.announce_address(
        store=store, url="http://10.0.0.9:8079", transport=control.transport()
    )
    assert outcome.announced and outcome.changed
    assert control.nodes["gpu-box"]["url"] == "http://10.0.0.9:8079"
    assert store.record.advertise_sequence == 1


@pytest.mark.asyncio
async def test_an_unchanged_address_is_announced_and_changes_nothing(
    authed_client: TestClient, control: FakeControl
) -> None:
    """The common case is a restart. It must be free at the root - every
    reboot of every node appending a log entry would be a log that grows
    with uptime rather than with events."""
    _enroll(authed_client, control)
    store: NodeIdentityStore = authed_client.app.state.node_identity  # type: ignore[attr-defined]
    recorded = control.nodes["gpu-box"]["url"]

    outcome = await enrollment.announce_address(
        store=store, url=recorded, transport=control.transport()
    )
    assert outcome.announced and outcome.changed is False


@pytest.mark.asyncio
async def test_the_sequence_advances_even_when_an_announcement_fails(
    authed_client: TestClient, control: FakeControl
) -> None:
    """Claimed before the call and never rolled back. A gap costs nothing
    - the root only needs the next number to be higher - while re-using
    one after a timeout that actually succeeded would look exactly like a
    replay and be refused forever."""
    _enroll(authed_client, control)
    store: NodeIdentityStore = authed_client.app.state.node_identity  # type: ignore[attr-defined]

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("root is gone")

    outcome = await enrollment.announce_address(
        store=store, url="http://10.0.0.9:8079", transport=httpx.MockTransport(dead)
    )
    assert outcome.announced is False
    assert store.record.advertise_sequence == 1

    outcome = await enrollment.announce_address(
        store=store, url="http://10.0.0.9:8079", transport=control.transport()
    )
    assert outcome.announced and outcome.changed
    assert store.record.advertise_sequence == 2


@pytest.mark.asyncio
async def test_announcing_never_raises_when_the_root_refuses(
    authed_client: TestClient, control: FakeControl
) -> None:
    """This runs at boot. Throwing here would take supervision down over
    a management-plane problem, which is the inversion
    `degraded-mode-required` exists to prevent."""
    _enroll(authed_client, control)
    store: NodeIdentityStore = authed_client.app.state.node_identity  # type: ignore[attr-defined]
    control.nodes["gpu-box"]["signingPublicKey"] = None

    outcome = await enrollment.announce_address(
        store=store, url="http://10.0.0.9:8079", transport=control.transport()
    )
    assert outcome.announced is False
    assert outcome.detail


@pytest.mark.asyncio
async def test_a_configured_advertise_url_beats_a_derived_one(control: FakeControl) -> None:
    """The expert override, and it wins even when it is wrong - an
    override that cannot be wrong is not one."""
    resolved = await enrollment.resolve_advertise_url(
        configured="http://tailnet-name:8079",
        control_url=control.url,
        bind_port=8079,
        persisted="http://stale:8079",
    )
    assert resolved == "http://tailnet-name:8079"


@pytest.mark.asyncio
async def test_an_unconfigured_advertise_url_is_re_derived_not_read_back(
    control: FakeControl,
) -> None:
    """The line that actually fixes the defect. A host that rebooted onto
    a new address has a *stale* persisted value, so resolving from it
    first would announce the old one - confidently, and forever."""
    resolved = await enrollment.resolve_advertise_url(
        configured=None,
        control_url=control.url,
        bind_port=8079,
        persisted="http://192.0.2.99:8079",
    )
    assert resolved is not None
    assert urlparse(resolved).hostname == "127.0.0.1"
    assert resolved != "http://192.0.2.99:8079"


@pytest.mark.asyncio
async def test_a_persisted_address_is_the_fallback_when_the_root_is_unreachable() -> None:
    """Better than nothing on a boot where the route cannot be measured:
    the last known address is more likely right than no address at all."""
    resolved = await enrollment.resolve_advertise_url(
        configured=None,
        control_url="http://127.0.0.1:1",
        bind_port=8079,
        persisted="http://192.0.2.99:8079",
    )
    assert resolved == "http://192.0.2.99:8079"


def test_a_joined_node_with_no_passphrase_signs_in_through_the_root(
    authed_client: TestClient, control: FakeControl
) -> None:
    """A worker has no passphrase of its own, and since 2026-09-25 it is a
    console anyway: the sign-in forwards to the root, with this agent's
    own token as the actor, and comes back addressed to this machine.
    Until then it answered "log in at the control root instead"."""
    _enroll(authed_client, control)
    state = authed_client.app.state.agent_state  # type: ignore[attr-defined]
    state._auth.pop("passphraseHash", None)

    wrong = authed_client.post("/v1/auth/login", json={"passphrase": "not the passphrase"})
    assert wrong.status_code == 401, wrong.text

    response = authed_client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    assert response.status_code == 200, response.text
    session = response.json()["sessionToken"]
    claims = _auth(authed_client).trust.verify(session, classes=(tokens.TYP_SESSION,))
    assert claims.aud == ("node:gpu-box", "control")
    actor = control.logins[-1]["authorization"].removeprefix("Bearer ")
    actor_claims = tokens.verify(
        actor, bundle=control.root.bundle(), recipient="control", classes=[tokens.TYP_SERVICE]
    )
    assert (actor_claims.sub, actor_claims.iss) == ("agent", "node:gpu-box")


def test_a_forwarded_sign_in_that_fails_counts_against_the_limit(
    authed_client: TestClient, control: FakeControl
) -> None:
    """Every forwarded attempt reaches the root from this one agent, so
    the root cannot tell browsers apart: the limit is kept here, per
    source, or it is not kept at all."""
    _enroll(authed_client, control)
    state = authed_client.app.state.agent_state  # type: ignore[attr-defined]
    state._auth.pop("passphraseHash", None)
    for _ in range(5):
        wrong = authed_client.post("/v1/auth/login", json={"passphrase": "not the passphrase"})
        assert wrong.status_code == 401, wrong.text
    limited = authed_client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    assert limited.status_code == 429, limited.text


def test_a_sign_in_while_the_root_is_down_says_so(
    authed_client: TestClient, control: FakeControl
) -> None:
    _enroll(authed_client, control)

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("control root is gone")

    authed_client.app.state.control_transport = httpx.MockTransport(dead)  # type: ignore[attr-defined]
    response = authed_client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    assert response.status_code == 503, response.text
    assert "control root" in response.json()["detail"]["detail"]
    # The session already held keeps working: nothing here needed the root.
    assert authed_client.get("/v1/node").status_code == 200


def test_a_sign_out_is_forwarded_to_the_root(
    authed_client: TestClient, control: FakeControl
) -> None:
    """Install-wide sign-out: the root replicates it into the bundle."""
    _enroll(authed_client, control)
    session = authed_client.headers["Authorization"]
    assert authed_client.delete("/v1/auth/sessions/current").status_code == 204
    assert control.sign_outs == [session]
    assert authed_client.get("/v1/node").status_code == 401


# --------------------------------------------------------------------------- #
# What the agent itself binds
# --------------------------------------------------------------------------- #


def _bind_host_of(tmp_path: Any, **settings_kwargs: Any) -> str:
    """The interface `build_server` actually hands uvicorn.

    Asserted off `server.config.host` rather than off the resolver,
    because the resolver was never the part that was wrong: the rule
    existed in `shared_child_env` and this module's docstring, and the
    socket was opened somewhere that had not heard about it.
    """
    from eugene_plexus_agent.__main__ import build_server

    settings = Settings(
        config_file=tmp_path / "agent.yaml",
        bind_port=8179,
        default_topology=False,
        **settings_kwargs,
    )
    return str(build_server(settings, unattended=True).config.host)


def test_agent_binds_loopback_when_it_advertises_nothing(tmp_path: Any) -> None:
    """The single-machine default, which is nearly every install.

    This is the test that makes the next one mean something: a fix that
    widened unconditionally would put four services plus the agent on
    every interface of every laptop that ever ran this.
    """
    assert _bind_host_of(tmp_path) == "127.0.0.1"


def test_agent_binds_wide_when_its_node_advertises_a_lan_address(tmp_path: Any) -> None:
    """The bug, from 2026-09-11: a joined node advertised an address it
    did not bind.

    Enrollment is outbound, so `201 Created` says the node reached the
    root and nothing about the return path. The node came up on
    loopback, told the root to call back on its LAN address, and looked
    healthy from every direction except that one -- union topology, idle
    unload and start-on-demand would all have failed silently. The
    installers widen the bind themselves now; this covers
    `eugene-plexus-agent join` run by hand, which is the path
    `docs/deployment/tailnet.md` documents.
    """
    NodeIdentityStore(tmp_path / "node.yaml").record_advertise_url("http://192.168.16.75:8079")
    assert _bind_host_of(tmp_path) == "0.0.0.0"


def test_agent_binds_wide_from_the_config_field_too(tmp_path: Any) -> None:
    """`advertiseUrl` in agent.yaml, not node.yaml -- machine A's case.

    The control host sets the address before its first start and is not
    enrolled at that moment, so a rule keyed on enrollment would miss
    exactly the machine `tailnet.md` calls its most important
    instruction. Keyed on the advertised address instead, which is the
    same thing `shared_child_env` keys on and the same precedence
    `GET /v1/node` reports: the config field wins over node.yaml.
    """
    (tmp_path / "agent.yaml").write_text("advertiseUrl: http://100.64.0.1:8079\n", encoding="utf-8")
    assert _bind_host_of(tmp_path) == "0.0.0.0"


def test_an_explicit_bind_host_wins_even_when_it_will_not_work(tmp_path: Any) -> None:
    """`easy-default-expert-override`, and the half of it that has teeth.

    An operator who names the interface gets the interface -- including
    `127.0.0.1` on a node that advertises a LAN address, which is the
    broken install this whole change exists to prevent. The rule is that
    the topmost override wins even with a value that will fail, so the
    default must be distinguishable from someone typing the default:
    pydantic's `model_fields_set` carries that, and `bind_host` is
    absent from it when nothing supplied one.
    """
    NodeIdentityStore(tmp_path / "node.yaml").record_advertise_url("http://192.168.16.75:8079")
    assert _bind_host_of(tmp_path, bind_host="127.0.0.1") == "127.0.0.1"
    assert _bind_host_of(tmp_path, bind_host="192.168.16.75") == "192.168.16.75"
