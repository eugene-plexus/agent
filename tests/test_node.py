"""This host in an install: identity, enrollment, the signed re-key, and
what each of them changes about the rest of the agent.

The control root is a fake behind an httpx `MockTransport`, and a real
listening socket, because the advertise address is derived from the
local end of a TCP connection to it and a fake that skipped that step
would agree with whatever the code guessed. What is under test is the
agent's half of an exchange the control repo's suite has exercised from
its side against fake agents since M5.
"""

from __future__ import annotations

import base64
import json
import logging
import socket
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import enrollment, node_identity, security
from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind, SpawnConfig
from eugene_plexus_agent.app import create_app, shared_child_env
from eugene_plexus_agent.node_identity import (
    NodeIdentityStore,
    generate_control_identity_for_tests,
    rekey_message,
    sign_rekey_message,
)
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.supervisor import _ComponentPlanner

from .conftest import TEST_PASSPHRASE, StubRuntimeSupervisor, StubSupervisor, fake_devices

JOIN_TOKEN = "join-token-minted-for-this-test"


class FakeControl:
    """A control root: one enrollment endpoint behind a MockTransport, and
    a real listening socket so the agent's advertise-host derivation has
    something to connect to.

    Holds a real Ed25519 identity so tests can sign re-keys the way the
    control root does, and an install signing key to hand out.
    """

    def __init__(self) -> None:
        self.private, self.public = generate_control_identity_for_tests()
        self.signing_key = security.generate_signing_key()
        # Readable plaintext, not key-shaped base64 next to a key-shaped
        # name — gitleaks flags the latter.
        self.recovery_public = base64.b64encode(
            b"recovery-recipient-public-key-32".ljust(32, b"0")[:32]
        ).decode()
        self.epoch = 1
        self.signing_key_id = "1"
        self.refuse: tuple[int, dict[str, Any]] | None = None
        self.enroll_requests: list[dict[str, Any]] = []
        # The node registry, as far as a fake needs one: the signing
        # public key it was given, and the address/sequence high-water
        # mark it enforces exactly as the real root does.
        self.nodes: dict[str, dict[str, Any]] = {}
        self.address_requests: list[dict[str, Any]] = []
        self.revoked: list[str] = []
        self.refuse_revoke: int | None = None
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._listener.getsockname()[1]}"

    @property
    def signing_key_b64(self) -> str:
        return base64.b64encode(self.signing_key).decode("ascii")

    def close(self) -> None:
        self._listener.close()

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/nodes/enroll":
                return self._enroll(request)
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
        # Enrolling replaces the record, sequence included — the real
        # root's `enrollNode` does, and a rebuilt host whose counter
        # restarted could never re-advertise otherwise.
        self.nodes[body["name"]] = {
            "url": body.get("url"),
            "signingPublicKey": body.get("signingPublicKey"),
            "advertiseSequence": 0,
        }
        return httpx.Response(
            201,
            json={
                "name": body["name"],
                "epoch": self.epoch,
                "signingKey": self.signing_key_b64,
                "signingKeyId": self.signing_key_id,
                "controlPublicKey": self.public,
                "recoveryPublicKey": self.recovery_public,
            },
        )

    def _announce(self, name: str, request: httpx.Request) -> httpx.Response:
        """The real verification, not a rubber stamp.

        Signature checked against the key enrollment recorded, over the
        **raw body's** url — which is the whole trap: verify a parsed URL
        and every announcement 401s on a trailing slash.
        """
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
        if not node_identity.verify_rekey_signature(
            control_public_key=public, message=message, signature=body["signature"]
        ):
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
        self.revoked.append(name)
        return httpx.Response(202, json={"reason": "revocation", "signingKeyId": "2"})

    def rekey(
        self,
        *,
        signing_key: bytes,
        key_id: str,
        epoch: int,
        sign_with: str | None = None,
    ) -> dict[str, Any]:
        """A re-key body as the control root would send it."""
        key_b64 = base64.b64encode(signing_key).decode("ascii")
        message = rekey_message(signing_key=key_b64, signing_key_id=key_id, epoch=epoch)
        return {
            "signingKey": key_b64,
            "signingKeyId": key_id,
            "epoch": epoch,
            "signature": sign_rekey_message(
                control_private_key=sign_with or self.private, message=message
            ),
        }


@pytest.fixture
def control() -> Iterator[FakeControl]:
    fake = FakeControl()
    yield fake
    fake.close()


def _enroll(client: TestClient, control: FakeControl, **body: Any) -> httpx.Response:
    """Enroll, and on success re-authenticate the client with a token
    minted from THE INSTALL'S key — the way the control root would mint
    one. The operator session that requested the enrollment was signed
    with this agent's old random key and stops verifying the moment the
    install key is adopted; that is the designed price of enrollment, the
    same one a rotation charges, and every test past this point is also
    an assertion that a control-minted token verifies here."""
    client.app.state.control_transport = control.transport()  # type: ignore[attr-defined]
    payload: dict[str, Any] = {"controlUrl": control.url, "token": JOIN_TOKEN, "name": "gpu-box"}
    payload.update(body)
    response = client.post("/v1/node/enroll", json=payload)
    if response.status_code == 200:
        token, _ = security.issue_operator_token(signing_key=control.signing_key)
        client.headers["Authorization"] = f"Bearer {token}"
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
# GET /v1/node, unenrolled — unchanged from M6
# --------------------------------------------------------------------------- #


def test_node_reports_devices_and_unenrolled(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/node")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["enrolled"] is False
    for absent in ("controlUrl", "name", "epoch", "signingKeyId", "controlPublicKey"):
        assert absent not in body
    assert body["os"] in ("windows", "linux", "macos")
    assert body["arch"] in ("x64", "arm64")
    kinds = [d["kind"] for d in body["devices"]]
    assert kinds == ["cuda", "cpu"]
    assert body["devices"][0]["memoryFreeBytes"] == 24 * 1024**3
    assert body["agentVersion"]


def test_node_is_readable_with_a_service_token(client: TestClient) -> None:
    client.post("/v1/auth/initialize", json={"passphrase": "pw"})
    signing_key = client.app.state.auth_state.signing_key  # type: ignore[attr-defined]
    token = security.issue_service_token(signing_key=signing_key, kind="control")
    response = client.get("/v1/node", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_node_requires_auth(client: TestClient) -> None:
    client.post("/v1/auth/initialize", json={"passphrase": "pw"})
    assert client.get("/v1/node").status_code == 401


# --------------------------------------------------------------------------- #
# Enrollment
# --------------------------------------------------------------------------- #


def test_enrollment_presents_the_public_key_and_url_and_adopts_the_install_key(
    authed_client: TestClient, control: FakeControl, settings: Settings
) -> None:
    """The whole exchange, from the agent's side.

    The request carries the *public* key only, the inventory, and where
    other hosts reach this agent — derived, here, from the real socket to
    the fake control. The response is recorded, the install's signing key
    replaces this agent's random one, and every supervised component is
    restarted so it picks the key up.
    """
    before = _auth(authed_client).signing_key
    response = _enroll(authed_client, control)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["enrolled"] is True
    assert body["name"] == "gpu-box"
    assert body["epoch"] == 1
    assert body["signingKeyId"] == "1"
    assert body["controlPublicKey"] == control.public
    assert urlparse(body["controlUrl"]).port == urlparse(control.url).port
    advertised = urlparse(body["advertiseUrl"])
    assert (advertised.hostname, advertised.port) == ("127.0.0.1", settings.bind_port)
    assert "privateKey" not in json.dumps(body)

    sent = control.enroll_requests[-1]
    assert sent["token"] == JOIN_TOKEN
    assert sent["name"] == "gpu-box"
    assert len(base64.b64decode(sent["publicKey"], validate=True)) == 32
    assert sent["url"] == f"http://127.0.0.1:{settings.bind_port}"
    assert [d["kind"] for d in sent["devices"]] == ["cuda", "cpu"]
    assert sent["os"] and sent["arch"] and sent["agentVersion"]
    assert "privateKey" not in sent

    after = _auth(authed_client).signing_key
    assert after == control.signing_key and after != before
    assert _restarts(authed_client) == 1

    # On disk, beside agent.yaml, with the private half — and without the
    # join token, which is spent.
    node_file = settings.config_file.parent / "node.yaml"
    assert node_file.exists()
    text = node_file.read_text(encoding="utf-8")
    assert "privateKey:" in text and control.signing_key_b64 in text
    assert JOIN_TOKEN not in text


def test_enrolling_twice_is_a_409_that_names_the_way_out(
    authed_client: TestClient, control: FakeControl
) -> None:
    stale = dict(authed_client.headers)
    assert _enroll(authed_client, control).status_code == 200
    # The session that asked for the enrollment is dead: the key it was
    # signed with is gone. Designed, documented, and the same price a
    # rotation charges at the control root.
    assert authed_client.get("/v1/node", headers=stale).status_code == 401
    again = _enroll(authed_client, control)
    assert again.status_code == 409
    # The way out is an operation now, not a filesystem instruction. Before
    # M9 this message told the operator to delete node.yaml by hand.
    assert "gpu-box" in again.text and "/v1/node/unenroll" in again.text
    assert len(control.enroll_requests) == 1, "a second enrollment must not reach the root"


def test_a_root_that_refuses_the_token_is_a_502_and_nothing_is_recorded(
    authed_client: TestClient, control: FakeControl
) -> None:
    before = _auth(authed_client).signing_key
    control.refuse = (401, {"title": "Join token rejected", "detail": "join token is unknown"})
    response = _enroll(authed_client, control)
    assert response.status_code == 502
    assert "401" in response.text and "join token is unknown" in response.text
    assert authed_client.get("/v1/node").json()["enrolled"] is False
    assert _auth(authed_client).signing_key == before
    assert _restarts(authed_client) == 0


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
    assert patched.json()["applied"] == ["advertiseUrl"]

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


def test_a_malformed_enrollment_is_a_502_and_nothing_is_recorded(
    authed_client: TestClient, control: FakeControl
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"name": "gpu-box", "epoch": 1, "signingKey": "short"})

    authed_client.app.state.control_transport = httpx.MockTransport(handle)  # type: ignore[attr-defined]
    response = authed_client.post(
        "/v1/node/enroll", json={"controlUrl": control.url, "token": JOIN_TOKEN, "name": "gpu-box"}
    )
    assert response.status_code == 502
    assert "signingKey" in response.text
    assert authed_client.get("/v1/node").json()["enrolled"] is False


def test_identity_and_the_install_key_survive_a_restart(
    authed_client: TestClient,
    control: FakeControl,
    settings: Settings,
) -> None:
    """The property M5 §8 asked for — a restart is not a re-key — on the
    agent's side. A second process over the same directory comes up
    enrolled, verifying tokens with the install's key from its first
    request, which is what a token minted at the control root needs."""
    assert _enroll(authed_client, control).status_code == 200

    reborn = create_app(settings=settings)
    reborn.state.supervisor = StubSupervisor()
    reborn.state.runtime_supervisor = StubRuntimeSupervisor()
    reborn.state.device_detector = lambda: fake_devices()
    reborn.state.library_fit_client = None
    with TestClient(reborn) as client:
        assert reborn.state.auth_state.signing_key == control.signing_key
        # A token the control root would mint — signed with the install key
        # this agent never generated — verifies here.
        token, _ = security.issue_operator_token(signing_key=control.signing_key)
        response = client.get("/v1/node", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["enrolled"] is True
        assert body["name"] == "gpu-box"
        assert body["epoch"] == 1
        assert body["controlPublicKey"] == control.public


def test_runtimes_report_the_node_once_enrolled(
    authed_client: TestClient, control: FakeControl
) -> None:
    """`Runtime.node` is filled from the agent's identity, never declared,
    and absent until there is an identity to fill it from."""
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
# Re-key and epoch fencing
# --------------------------------------------------------------------------- #


def test_a_signed_rekey_is_adopted_and_children_restart(
    authed_client: TestClient, control: FakeControl, settings: Settings
) -> None:
    assert _enroll(authed_client, control).status_code == 200
    new_key = security.generate_signing_key()

    # No bearer on the request: the signature is the credential.
    response = authed_client.post(
        "/v1/node/rekey",
        json=control.rekey(signing_key=new_key, key_id="2", epoch=1),
        headers={"Authorization": ""},
    )
    assert response.status_code == 200, response.text
    assert response.json()["signingKeyId"] == "2"
    assert _auth(authed_client).signing_key == new_key
    assert _restarts(authed_client) == 2, "enrollment and the re-key each restart the children"
    text = (settings.config_file.parent / "node.yaml").read_text(encoding="utf-8")
    assert base64.b64encode(new_key).decode() in text
    assert control.signing_key_b64 not in text


def test_a_rekey_with_a_bad_signature_is_401_and_changes_nothing(
    authed_client: TestClient, control: FakeControl
) -> None:
    assert _enroll(authed_client, control).status_code == 200
    impostor_private, _ = generate_control_identity_for_tests()
    new_key = security.generate_signing_key()

    forged = control.rekey(signing_key=new_key, key_id="2", epoch=1, sign_with=impostor_private)
    response = authed_client.post("/v1/node/rekey", json=forged)
    assert response.status_code == 401
    assert "not signed by the control root" in response.text
    assert _auth(authed_client).signing_key == control.signing_key

    garbage = dict(forged, signature="bm90LWEtc2lnbmF0dXJl")
    assert authed_client.post("/v1/node/rekey", json=garbage).status_code == 401
    assert _restarts(authed_client) == 1


def test_a_lower_epoch_is_fenced_with_409(authed_client: TestClient, control: FakeControl) -> None:
    """M5 §6, on the side that does the fencing: a superseded root
    presenting a lower epoch is refused by this node alone, with no
    election and no agreement with any other agent."""
    assert _enroll(authed_client, control).status_code == 200
    key = control.signing_key

    # An announcement: same key, higher epoch. Recorded, nothing restarted.
    announced = authed_client.post(
        "/v1/node/rekey", json=control.rekey(signing_key=key, key_id="1", epoch=3)
    )
    assert announced.status_code == 200, announced.text
    assert announced.json()["epoch"] == 3
    assert _restarts(authed_client) == 1

    # The old root, back at epoch 2 with a genuine signature: fenced.
    stale = authed_client.post(
        "/v1/node/rekey", json=control.rekey(signing_key=key, key_id="1", epoch=2)
    )
    assert stale.status_code == 409
    assert "already acknowledged epoch 3" in stale.text
    assert authed_client.get("/v1/node").json()["epoch"] == 3

    # And a rotation it tries to push from there is fenced too, key or no key.
    stale_key = authed_client.post(
        "/v1/node/rekey",
        json=control.rekey(signing_key=security.generate_signing_key(), key_id="9", epoch=0),
    )
    assert stale_key.status_code == 409
    assert _auth(authed_client).signing_key == key


def test_a_replayed_generation_at_the_same_epoch_is_409(
    authed_client: TestClient, control: FakeControl
) -> None:
    assert _enroll(authed_client, control).status_code == 200
    second = security.generate_signing_key()
    assert (
        authed_client.post(
            "/v1/node/rekey", json=control.rekey(signing_key=second, key_id="2", epoch=1)
        ).status_code
        == 200
    )
    replay = authed_client.post(
        "/v1/node/rekey", json=control.rekey(signing_key=control.signing_key, key_id="1", epoch=1)
    )
    assert replay.status_code == 409
    assert "replayed rotation" in replay.text
    assert _auth(authed_client).signing_key == second

    # Re-sending the current generation is idempotent, not a replay.
    again = authed_client.post(
        "/v1/node/rekey", json=control.rekey(signing_key=second, key_id="2", epoch=1)
    )
    assert again.status_code == 200
    assert _restarts(authed_client) == 2


def test_an_unenrolled_agent_refuses_a_rekey(
    authed_client: TestClient, control: FakeControl
) -> None:
    response = authed_client.post(
        "/v1/node/rekey",
        json=control.rekey(signing_key=security.generate_signing_key(), key_id="2", epoch=1),
    )
    assert response.status_code == 401
    assert "not enrolled" in response.text.lower()


def test_the_canonical_message_is_the_contracted_form() -> None:
    """Three fields, sorted keys, no whitespace — the sentence in
    `RekeyRequest.signature`, byte for byte. A fourth field or a space
    anywhere and every rotation in the install fails to verify."""
    message = rekey_message(signing_key="a2V5", signing_key_id="7", epoch=42)
    assert message == b'{"epoch":42,"signingKey":"a2V5","signingKeyId":"7"}'
    private, public = generate_control_identity_for_tests()
    signature = sign_rekey_message(control_private_key=private, message=message)
    assert node_identity.verify_rekey_signature(
        control_public_key=public, message=message, signature=signature
    )
    assert not node_identity.verify_rekey_signature(
        control_public_key=public, message=message + b" ", signature=signature
    )
    assert not node_identity.verify_rekey_signature(
        control_public_key=public, message=message, signature="not base64!!"
    )


# --------------------------------------------------------------------------- #
# service:control may declare a runtime
# --------------------------------------------------------------------------- #


def test_the_control_roots_service_token_may_declare_a_runtime(client: TestClient) -> None:
    """The control root forwards declarations to the node that will run
    them with `service:control`; through M6 the agent refused. Checked
    exactly — a library's token still cannot."""
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    signing_key = client.app.state.auth_state.signing_key  # type: ignore[attr-defined]
    spec = {"name": "qwen", "engine": "llama_cpp", "modelPath": "/m.gguf", "autoStart": False}

    library = security.issue_service_token(signing_key=signing_key, kind="library")
    refused = client.post("/v1/runtimes", json=spec, headers={"Authorization": f"Bearer {library}"})
    assert refused.status_code == 401
    assert "service:control" in refused.text

    control = security.issue_service_token(signing_key=signing_key, kind="control")
    accepted = client.post(
        "/v1/runtimes", json=spec, headers={"Authorization": f"Bearer {control}"}
    )
    assert accepted.status_code == 201, accepted.text

    # Deleting it is still the operator's.
    denied = client.delete("/v1/runtimes/qwen", headers={"Authorization": f"Bearer {control}"})
    assert denied.status_code == 401


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
    assert "EUGENE_PLEXUS_CONTROL_AUTH_SIGNING_KEY" not in plan.env

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


def test_unenroll_discards_the_installs_key_and_tells_the_root(
    authed_client: TestClient, control: FakeControl
) -> None:
    """The inverse of enrolling, and the reason it is safe to run here.

    Revocation exists because a node that still *holds* the signing key
    can still authenticate. Un-enrolling discards it - the opposite
    direction - so a node cannot escape revocation this way, only disarm
    itself. The assertion that matters is the last one: the install's key
    no longer verifies here.
    """
    assert _enroll(authed_client, control).status_code == 200
    assert _auth(authed_client).signing_key == control.signing_key

    response = authed_client.post("/v1/node/unenroll")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["controlNotified"] is True
    assert body["previousName"] == "gpu-box"
    assert body["identity"]["enrolled"] is False
    assert control.revoked == ["gpu-box"]
    assert _auth(authed_client).signing_key != control.signing_key


def test_unenroll_keeps_this_nodes_own_keypairs(
    authed_client: TestClient, control: FakeControl
) -> None:
    """They are the *host's* identity, not the install's. The same node
    re-joining anywhere is still the same node, and a departing install
    holding only public halves can read nothing with them."""
    _enroll(authed_client, control)
    store: NodeIdentityStore = authed_client.app.state.node_identity  # type: ignore[attr-defined]
    before = (store.record.public_key, store.record.signing_public_key)
    assert all(before)

    authed_client.post("/v1/node/unenroll")
    after = store.record
    assert (after.public_key, after.signing_public_key) == before
    # And everything that belonged to the install is gone.
    assert after.signing_key is None
    assert after.signing_key_id is None
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
    """Children hold the install's signing key in their environment, so a
    key this agent just threw away reaches them only through a respawn -
    the same reason enrollment restarts them."""
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


def test_a_joined_node_is_told_to_log_in_at_the_control_root(
    authed_client: TestClient, control: FakeControl
) -> None:
    """A worker has no passphrase of its own and never will.

    It verifies tokens with the install's signing key, so an operator
    session minted at the control root already works there. The default
    message told it to run `POST /v1/auth/initialize`, which on a machine
    that has joined an install is advice to raise a second one. Found by
    M9's acceptance run -- the first thing that ever tried to log in at a
    joined node.
    """
    _enroll(authed_client, control)
    # A node onboarded by `eugene-plexus-agent join` has an identity and
    # no passphrase; simulate that half by clearing the passphrase the
    # fixture set.
    state = authed_client.app.state.agent_state  # type: ignore[attr-defined]
    state._auth.pop("passphraseHash", None)

    response = authed_client.post("/v1/auth/login", json={"passphrase": "anything"})
    assert response.status_code == 503, response.text
    detail = response.json()["detail"]["detail"]
    assert "control root" in detail
    assert "first-run setup" in detail or "setup" in detail
