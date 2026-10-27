"""`NodeTrust` on its own: what this node signs, and for whom.

The routes that use it are tested where they live; these pin the rules
a route cannot see, because the verifier on the far side would refuse
the token anyway and a route test would pass either way.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import tokens
from eugene_plexus_agent import trust as trust_module
from eugene_plexus_agent.node_identity import FencedError, NodeIdentityStore
from eugene_plexus_agent.trust import BUNDLE_FILE, MintRefused, NodeTrust

from .conftest import (
    TEST_PASSPHRASE,
    FakeRoot,
    enroll_app,
    enroll_store,
    local_service_token,
    standalone_trust,
)


def _enrolled(tmp_path: Path, *, grants: tuple[str, ...] = ()) -> tuple[NodeTrust, FakeRoot]:
    root = FakeRoot()
    store = NodeIdentityStore(tmp_path / "node.yaml")
    enroll_store(store, root, "gpu-box", grants=grants)
    trust = NodeTrust(store, tmp_path / BUNDLE_FILE)
    trust.load()
    return trust, root


@pytest.fixture
def _root_over_machine_tls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The root is reached over machine TLS pinned to its key, so only the
    root can answer a pull, and the limit is kept (row 46). Without this
    the test root's address is plain HTTP across a network."""
    monkeypatch.setattr(trust_module, "channel_authenticated", lambda url: True, raising=False)


def _claims_of(token: str) -> dict[str, object]:
    import jwt

    return dict(jwt.decode(token, options={"verify_signature": False}))


def test_a_token_for_another_machine_never_outlives_the_hour_a_verifier_allows(
    tmp_path: Path,
) -> None:
    """The verifier refuses a longer one, so without the cap a caller
    asking for two hours would get a token that fails everywhere -- and a
    leaked one would be a leaked two hours."""
    trust, _ = _enrolled(tmp_path)
    token, exp = trust.mint_service(sub="agent", audience="node:attic", ttl_seconds=7200)
    claims = _claims_of(token)
    assert exp - int(claims["iat"]) == tokens.MAX_REMOTE_SERVICE_SECONDS  # type: ignore[call-overload]


def test_a_token_for_this_machine_keeps_the_lifetime_asked_for(tmp_path: Path) -> None:
    trust, _ = _enrolled(tmp_path)
    token, exp = trust.mint_service(sub="gateway", audience="node:gpu-box", ttl_seconds=7200)
    assert exp - int(_claims_of(token)["iat"]) == 7200  # type: ignore[call-overload]


def test_a_plain_node_speaks_across_machines_only_as_its_agent(tmp_path: Path) -> None:
    trust, _ = _enrolled(tmp_path)
    trust.mint_service(sub="agent", audience="node:attic")
    for sub in ("gateway", "library", "inference-driver", "control"):
        with pytest.raises(MintRefused):
            trust.mint_service(sub=sub, audience="node:attic")


def test_the_gateway_grant_lets_this_node_send_a_gateway_token(tmp_path: Path) -> None:
    trust, _ = _enrolled(tmp_path, grants=("gateway",))
    token, _ = trust.mint_service(sub="gateway", audience="control")
    assert _claims_of(token)["aud"] == ["control"]
    with pytest.raises(MintRefused):
        trust.mint_service(sub="library", audience="control")


def test_an_enrolled_node_mints_no_session_and_no_client_key(tmp_path: Path) -> None:
    trust, _ = _enrolled(tmp_path)
    with pytest.raises(MintRefused):
        trust.mint_local_session()
    with pytest.raises(MintRefused):
        trust.mint_local_client(key_id="k", name="app", ttl_seconds=60)


def test_a_standalone_node_is_its_own_authority_and_nobody_elses(tmp_path: Path) -> None:
    trust = standalone_trust(tmp_path)
    assert trust.recipient == "node:local"
    session, _ = trust.mint_local_session()
    assert trust.verify(session, classes=(tokens.TYP_SESSION,)).aud == ("node:local",)
    # A root this node never joined signs nothing it accepts.
    foreign = FakeRoot().session("node:local")
    with pytest.raises(tokens.TokenError):
        trust.verify(foreign, classes=(tokens.TYP_SESSION,))


def test_the_recorded_epoch_fences_even_with_no_bundle_held(tmp_path: Path) -> None:
    root = FakeRoot(epoch=3)
    store = NodeIdentityStore(tmp_path / "node.yaml")
    enroll_store(store, root, "gpu-box")
    (tmp_path / BUNDLE_FILE).unlink()
    trust = NodeTrust(store, tmp_path / BUNDLE_FILE)
    trust.load()
    assert trust.bundle is None
    root.epoch = 2
    with pytest.raises(FencedError):
        trust.accept(root.bundle().jws)
    assert trust.bundle is None


# --------------------------------------------------------------------------- #
# How long since this node heard from the root
# --------------------------------------------------------------------------- #


def _heard(tmp_path: Path, word_ago: float, pulled_ago: float | None = None) -> None:
    """The kept record of the root's last word (row 46), as of `word_ago`.

    The bundle file's time is set back with it, which the agent before the
    row 46 review read as the time last heard; it reads it no more."""
    now = time.time()
    bundle = tmp_path / BUNDLE_FILE
    if bundle.exists():
        os.utime(bundle, (now - word_ago, now - word_ago))
    (tmp_path / getattr(trust_module, "HEARD_FILE", "trust_heard.json")).write_text(
        json.dumps(
            {
                "wordAt": now - word_ago,
                "pulledAt": None if pulled_ago is None else now - pulled_ago,
            }
        ),
        encoding="utf-8",
    )


def test_the_age_is_since_the_bundle_was_taken_not_since_it_was_signed(
    tmp_path: Path, _root_over_machine_tls: None
) -> None:
    """A pull returns the same signed bundle until something changes, so
    the bundle's own `iat` is days old on a quiet install that heard from
    its root a minute ago. The age is when this node last took one."""
    trust, root = _enrolled(tmp_path)
    # Heard fifteen minutes ago, as far as the record says; taking a
    # bundle now is what must bring the age back to zero.
    _heard(tmp_path, 900)
    trust.load()
    assert (trust.heard_age_seconds() or 0) >= 895
    old = tokens.build_bundle(
        authority=root.identity,
        version=root.version + 1,
        epoch=root.epoch,
        keys=[root.token.trust_key(["authority"]), *root.members.values()],
        now=int(time.time()) - 5 * 86400,
    )
    # The minute's pull, from the root over its machine TLS: word now.
    trust.accept(old.jws, pulled=True)
    age = trust.heard_age_seconds()
    assert age is not None and age < 5
    assert trust.heard_age_seconds(now=time.time() + 700) in (699, 700, 701)


def test_the_age_survives_a_restart_from_its_own_record(
    tmp_path: Path, _root_over_machine_tls: None
) -> None:
    _enrolled(tmp_path)
    _heard(tmp_path, 900)
    kept = tmp_path / BUNDLE_FILE
    again = NodeTrust(NodeIdentityStore(tmp_path / "node.yaml"), kept)
    again._identity.load()
    again.load()
    age = again.heard_age_seconds()
    assert age is not None and 895 <= age <= 905


def test_with_no_record_the_age_is_the_bundles_signature_never_the_files_time(
    tmp_path: Path, _root_over_machine_tls: None
) -> None:
    """The bundle file is rewritten whenever any bundle is taken, a replay
    included, so its time says nothing about the root (row 46 review)."""
    trust, root = _enrolled(tmp_path)
    signed = tokens.build_bundle(
        authority=root.identity,
        version=root.version + 1,
        epoch=root.epoch,
        keys=[root.token.trust_key(["authority"]), *root.members.values()],
        now=int(time.time()) - 30 * 3600,
    )
    tokens.write_bundle_file(tmp_path / BUNDLE_FILE, signed)  # written just now
    trust.load()
    assert (trust.word_age_seconds() or 0) >= 30 * 3600 - 5


def test_a_standalone_node_has_no_root_to_hear_from(tmp_path: Path) -> None:
    assert standalone_trust(tmp_path).heard_age_seconds() is None


# --------------------------------------------------------------------------- #
# Keeping the bundle on disk
# --------------------------------------------------------------------------- #


def _kept_version(tmp_path: Path, root: FakeRoot) -> int:
    document = json.loads((tmp_path / BUNDLE_FILE).read_text(encoding="utf-8"))
    return tokens.parse_bundle(str(document["jws"]), authority=root.public).version


def _held_open(monkeypatch: pytest.MonkeyPatch, times: int) -> list[int]:
    """The bundle file refused `times` times, as Windows refuses to replace
    a file another process has open. Returns the versions each write tried."""
    real = tokens.write_bundle_file
    tried: list[int] = []

    def write(path: Path, bundle: tokens.TrustBundle) -> None:
        tried.append(bundle.version)
        if len(tried) <= times:
            raise PermissionError(13, "Access is denied")
        real(path, bundle)

    monkeypatch.setattr(tokens, "write_bundle_file", write)
    monkeypatch.setattr(trust_module, "SAVE_RETRY_DELAYS_SECONDS", (0.0, 0.0))
    return tried


def test_a_bundle_file_held_open_for_a_moment_is_written_anyway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every component this agent runs reads the bundle file, and Windows
    refuses to replace a file while anyone has it open. A moment's hold is
    waited out, not taken as a failure."""
    trust, root = _enrolled(tmp_path)
    tried = _held_open(monkeypatch, times=2)
    taken = trust.accept(root.bundle().jws)
    assert _kept_version(tmp_path, root) == taken.version
    assert tried == [taken.version] * 3


def test_a_bundle_the_file_refused_is_written_at_the_next_chance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A promoted root pushes its bundle once, and this node's pulls still go
    to the root that stopped, so no later pull rewrites the file. The bundle
    is held, the failure is said, and the file is written when it can be."""
    trust, root = _enrolled(tmp_path)
    before = _kept_version(tmp_path, root)
    _held_open(monkeypatch, times=3)
    with caplog.at_level(logging.WARNING, logger=trust_module.log.name):
        taken = trust.accept(root.bundle().jws)
    assert trust.bundle is not None and trust.bundle.version == taken.version
    assert _kept_version(tmp_path, root) == before, "the file was refused every time"
    assert f"could not keep trust bundle {taken.version}" in caplog.text
    assert "Access is denied" in caplog.text
    assert trust.save_pending() is True
    assert _kept_version(tmp_path, root) == taken.version


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to replace an open file")
def test_windows_refuses_to_replace_the_open_bundle_file_and_the_agent_writes_it_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real refusal, not a stand-in: a component reading the file while a
    pushed bundle arrives (found by standby-acceptance.py on 2026-10-08)."""
    trust, root = _enrolled(tmp_path)
    before = _kept_version(tmp_path, root)
    monkeypatch.setattr(trust_module, "SAVE_RETRY_DELAYS_SECONDS", (0.0,))
    with open(tmp_path / BUNDLE_FILE, encoding="utf-8"):
        taken = trust.accept(root.bundle().jws)
    assert trust.bundle is not None and trust.bundle.version == taken.version
    assert _kept_version(tmp_path, root) == before
    assert trust.save_pending() is True
    assert _kept_version(tmp_path, root) == taken.version


def test_the_trust_pull_writes_a_held_bundle_while_the_root_does_not_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a promotion the root this node joined is gone, so every pull
    fails; the bundle the new root pushed still reaches the file."""
    from eugene_plexus_agent import app as app_module

    trust, root = _enrolled(tmp_path)
    _held_open(monkeypatch, times=3)
    taken = trust.accept(root.bundle().jws)
    assert _kept_version(tmp_path, root) != taken.version

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("the old root is stopped", request=request)

    app = SimpleNamespace(
        state=SimpleNamespace(
            auth_state=SimpleNamespace(trust=trust),
            node_identity=trust._identity,
            control_transport=httpx.MockTransport(refuse),
        )
    )
    monkeypatch.setattr(app_module, "TRUST_PULL_INTERVAL_SECONDS", 0.01)

    async def run() -> None:
        task = asyncio.create_task(app_module._pull_trust_bundle(app))  # type: ignore[arg-type]
        try:
            for _ in range(200):
                if _kept_version(tmp_path, root) == taken.version:
                    return
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert _kept_version(tmp_path, root) == taken.version


# --------------------------------------------------------------------------- #
# A limit on frozen trust (row 46)
# --------------------------------------------------------------------------- #

_DAY = 24 * 3600
_REFUSED_25_HOURS = (
    "This machine has not heard from the root for 25 hours, so it takes no changes "
    "from other machines. Check its network to the root."
)


@pytest.fixture
def _any_model_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    import eugene_plexus_agent.admission as admission_module

    monkeypatch.setattr(admission_module, "model_size_bytes", lambda path: 1024**3)


def _install(client: TestClient) -> tuple[FakeRoot, tokens.Signer]:
    """This agent joined as `gpu-box`; `attic` is another member, joined with
    the gateway grant, whose gateway may start and stop runtimes here."""
    initialized = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    assert initialized.status_code == 200, initialized.text
    root = FakeRoot()
    attic = tokens.Signer(key=tokens.generate_private_key(), issuer="node:attic")
    root.register("attic", attic.key.public_key(), ("gateway",))
    enroll_app(_app_of(client), root, "gpu-box")
    return root, attic


def _app_of(client: TestClient) -> FastAPI:
    app = client.app
    assert isinstance(app, FastAPI)
    return app


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _from(signer: tokens.Signer, sub: str = "gateway") -> dict[str, str]:
    token, _ = signer.mint(typ=tokens.TYP_SERVICE, sub=sub, aud=["node:gpu-box"], ttl_seconds=300)
    return _bearer(token)


def _silent_for(client: TestClient, seconds: float) -> None:
    """The root last heard from `seconds` ago, as the record says."""
    trust = _app_of(client).state.auth_state.trust
    _heard(trust.bundle_path.parent, seconds, seconds)
    trust.load()


def _declare(client: TestClient, root: FakeRoot) -> None:
    declared = client.post(
        "/v1/runtimes",
        json={"name": "big", "engine": "llama_cpp", "modelPath": "/models/big.gguf"},
        headers=_bearer(root.session("node:gpu-box")),
    )
    assert declared.status_code == 201, declared.text


@pytest.mark.usefixtures("_any_model_fits")
@pytest.mark.usefixtures("_root_over_machine_tls")
def test_a_stale_bundle_refuses_another_machines_mutation(client: TestClient) -> None:
    """The residual risk TOK-RES named: a node cut off from the root trusts its
    last bundle forever, so a machine removed since keeps working there. Past
    the limit, another machine's key changes nothing here, and says why."""
    root, attic = _install(client)
    _declare(client, root)
    other = _from(attic)
    # In time, the gateway on another machine stops and starts a runtime here.
    assert client.post("/v1/runtimes/big/stop", headers=other).status_code == 202
    assert client.post("/v1/runtimes/big/start", headers=other).status_code == 202

    _silent_for(client, 25 * 3600)
    for path in ("/v1/runtimes/big/stop", "/v1/runtimes/big/start"):
        refused = client.post(path, headers=other)
        assert refused.status_code == 403, (path, refused.text)
        problem = refused.json()["detail"]
        assert problem["title"] == "Root not heard"
        assert problem["detail"] == _REFUSED_25_HOURS
    # The same machine's reads still answer.
    assert client.get("/v1/runtimes", headers=other).status_code == 200

    # The limit is a setting, read at every request: 0 is none, and 48 hours
    # is more than 25.
    session = _bearer(root.session("node:gpu-box"))
    for hours, status in ((0, 202), (48, 202), (24, 403)):
        patched = client.patch(
            "/v1/config", json={"maxTrustBundleAgeHours": hours}, headers=session
        )
        assert patched.json()["applied"] == ["maxTrustBundleAgeHours"], patched.text
        assert client.post("/v1/runtimes/big/stop", headers=other).status_code == status, hours

    # Word from the root lifts it: the minute's pull, even of the same bundle.
    trust = _app_of(client).state.auth_state.trust
    trust.accept(trust.bundle.jws, pulled=True)
    assert client.post("/v1/runtimes/big/stop", headers=other).status_code == 202


@pytest.mark.usefixtures("_any_model_fits")
@pytest.mark.usefixtures("_root_over_machine_tls")
def test_inference_and_the_roots_tokens_still_pass(client: TestClient) -> None:
    """A dead root never stops answers (M5): past the limit the root's own
    tokens, this machine's own tokens for its children, every read and the
    questions on an answer's path all still pass."""
    root, attic = _install(client)
    _silent_for(client, 25 * 3600)

    # The root's control token declares; its session stops.
    declared = client.post(
        "/v1/runtimes",
        json={"name": "big", "engine": "llama_cpp", "modelPath": "/models/big.gguf"},
        headers=_bearer(root.service("node:gpu-box")),
    )
    assert declared.status_code == 201, declared.text
    session = _bearer(root.session("node:gpu-box"))
    assert client.post("/v1/runtimes/big/stop", headers=session).status_code == 202
    # This machine's own gateway wakes and unloads it.
    own = _bearer(local_service_token(_app_of(client), "gateway"))
    assert client.post("/v1/runtimes/big/start", headers=own).status_code == 202
    assert client.post("/v1/runtimes/big/stop", headers=own).status_code == 202

    # Another machine still reads, and still asks the questions an answer needs.
    other = _from(attic)
    assert client.get("/v1/runtimes", headers=other).status_code == 200
    # The runtime's companion driver is in the topology now; the stub
    # supervisor runs none of it.
    _app_of(client).state.supervisor.is_supervised = lambda _name: False
    assert client.get("/v1/node", headers=other).status_code == 200
    admission = client.post(
        "/v1/runtimes/admission",
        json={"name": "q", "engine": "llama_cpp", "modelPath": "/models/q.gguf"},
        headers=other,
    )
    assert admission.status_code == 200, admission.text
    _root_answers_admission(client)
    asked = client.post(
        "/v1/auth/client-keys/admission",
        json={"keyId": "k1", "action": "acquire", "requestId": "r1", "model": "m"},
        headers=other,
    )
    assert asked.status_code == 200, asked.text
    # And still cannot change anything.
    refused = client.post("/v1/runtimes/big/start", headers=other)
    assert refused.status_code == 403, refused.text


def _root_answers_admission(client: TestClient) -> None:
    """Client-key admission is forwarded to the root; here it answers."""
    from eugene_plexus_agent.client_key_registry import ClientKeyRegistry

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"keyId": "k1", "keyName": "app"})

    app = _app_of(client)
    registry = ClientKeyRegistry(app)
    registry.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app.state.client_key_registry = registry


@pytest.mark.usefixtures("_root_over_machine_tls")
def test_a_route_nobody_named_is_refused_as_a_change(client: TestClient) -> None:
    """A route added without a line in either list does not slip through."""
    from fastapi import Depends

    from eugene_plexus_agent.dependencies import require_operator_or_gateway

    app = _app_of(client)

    @app.post("/v1/row46-new-door", dependencies=[Depends(require_operator_or_gateway)])
    async def new_door() -> dict[str, bool]:
        return {"done": True}

    # Ahead of the proxy and the console's own pages, which answer the rest.
    routes = app.router.routes
    routes.insert(0, routes.pop())
    _, attic = _install(client)
    assert client.post("/v1/row46-new-door", headers=_from(attic)).status_code == 200
    _silent_for(client, 25 * 3600)
    refused = client.post("/v1/row46-new-door", headers=_from(attic))
    assert refused.status_code == 403, refused.text


def _every_route(app: FastAPI) -> set[tuple[str, str]]:
    """(method, path template) of every route, through included routers
    whether FastAPI flattens them (older) or keeps them (0.16x)."""
    from fastapi.routing import APIWebSocketRoute

    found: set[tuple[str, str]] = set()

    def walk(routes: list[Any], prefix: str) -> None:
        for route in routes:
            included = getattr(route, "original_router", None)
            if included is not None:
                walk(included.routes, prefix + (route.include_context.prefix or ""))
                continue
            if isinstance(route, APIWebSocketRoute):
                found.add(("WEBSOCKET", prefix + route.path))
            for method in getattr(route, "methods", None) or ():
                found.add((method, prefix + route.path))

    walk(app.router.routes, "")
    return found


def test_every_route_that_changes_something_is_named_for_the_freeze(app: FastAPI) -> None:
    """Every route on a method other than GET, HEAD and OPTIONS is in exactly
    one of the two lists, and neither list names a route that is gone. A new
    route fails here until someone says whether it changes something."""
    from eugene_plexus_agent.dependencies import (
        FROZEN_TRUST_CHANGES,
        FROZEN_TRUST_QUESTIONS,
        READ_METHODS,
    )

    found = {(m, p) for m, p in _every_route(app) if m not in READ_METHODS}
    named = FROZEN_TRUST_CHANGES | FROZEN_TRUST_QUESTIONS
    assert not found - named, (
        "name each in dependencies.FROZEN_TRUST_CHANGES or FROZEN_TRUST_QUESTIONS: "
        f"{sorted(found - named)}"
    )
    assert not FROZEN_TRUST_CHANGES & FROZEN_TRUST_QUESTIONS
    assert not named - found, f"named but not served: {sorted(named - found)}"
    # The design's list (row-46.md section 2), each a change.
    for change in (
        ("POST", "/v1/runtimes/{name}/start"),
        ("POST", "/v1/runtimes/{name}/stop"),
        ("POST", "/v1/runtimes"),
        ("PATCH", "/v1/runtimes/{name}"),
        ("POST", "/v1/components"),
        ("PATCH", "/v1/components/{name}"),
        ("DELETE", "/v1/components/{name}"),
        ("PATCH", "/v1/config"),
        ("POST", "/v1/engines/{engine}/install"),
    ):
        assert change in FROZEN_TRUST_CHANGES, change


@pytest.mark.usefixtures("_root_over_machine_tls")
def test_a_pushed_copy_of_the_held_bundle_is_not_word_from_the_root(tmp_path: Path) -> None:
    """The push door takes no bearer: anybody who can reach this machine can
    post a bundle they saw once. Only a newer one counts, from when the root
    signed it; a bundle that does not verify changes nothing."""
    trust, root = _enrolled(tmp_path)
    _heard(tmp_path, 25 * 3600)
    trust.load()
    held = trust.bundle
    assert held is not None

    trust.accept(held.jws)
    assert (trust.heard_age_seconds() or 0) >= 25 * 3600 - 5
    assert trust.frozen_age(24) is not None
    # And not across a restart either: the record was not touched.
    trust.load()
    assert trust.frozen_age(24) is not None

    forged = tokens.build_bundle(
        authority=tokens.generate_private_key(),
        version=held.version + 5,
        epoch=held.epoch,
        keys=list(held.keys.values()),
    )
    with pytest.raises(tokens.BundleError):
        trust.accept(forged.jws)
    assert trust.frozen_age(24) is not None

    def signed(version: int, seconds_ago: int) -> str:
        return tokens.build_bundle(
            authority=root.identity,
            version=version,
            epoch=root.epoch,
            keys=[root.token.trust_key(["authority"]), *root.members.values()],
            now=int(time.time()) - seconds_ago,
        ).jws

    # Newer, but signed before this node last heard: still silent that long.
    trust.accept(signed(held.version + 1, 30 * 3600))
    assert (trust.heard_age_seconds() or 0) >= 25 * 3600 - 5
    # Newer, signed an hour ago: the root was there an hour ago.
    trust.accept(signed(held.version + 2, 3600))
    age = trust.heard_age_seconds()
    assert age is not None and 3595 <= age <= 3605
    assert trust.frozen_age(24) is None
    # A pull is word now, whatever it holds.
    trust.accept(signed(held.version + 2, 3600), pulled=True)
    assert (trust.heard_age_seconds() or 0) < 5


@pytest.mark.usefixtures("_root_over_machine_tls")
def test_a_clock_set_back_does_not_shorten_the_silence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wall clock can be moved; this process's own count cannot. A pull
    after the clock moved back still counts as word."""
    trust, _ = _enrolled(tmp_path)
    held = trust.bundle
    assert held is not None
    trust.accept(held.jws, pulled=True)
    start_wall, start_perf = time.time(), time.perf_counter()
    clock = {"wall": start_wall, "perf": start_perf}
    monkeypatch.setattr(
        trust_module,
        "time",
        SimpleNamespace(
            time=lambda: clock["wall"],
            perf_counter=lambda: clock["perf"],
            sleep=time.sleep,
        ),
    )
    # A day and an hour pass, and the wall clock is set back two days.
    clock["perf"] = start_perf + _DAY + 3600
    clock["wall"] = start_wall - 2 * _DAY
    assert (trust.heard_age_seconds() or 0) >= _DAY + 3600 - 5
    assert trust.frozen_age(24) is not None
    trust.accept(held.jws, pulled=True)
    assert trust.frozen_age(24) is None
    assert (trust.heard_age_seconds() or 0) < 5


def test_the_limit_is_the_setting_and_a_bad_value_is_the_default(tmp_path: Path) -> None:
    import yaml

    from eugene_plexus_agent.state import CONFIG_FIELDS, AgentState
    from eugene_plexus_agent.trust import (
        DEFAULT_MAX_AGE_HOURS,
        MAX_AGE_HOURS_KEY,
        MOST_MAX_AGE_HOURS,
        max_age_hours,
    )

    field = next(f for f in CONFIG_FIELDS if f.key == MAX_AGE_HOURS_KEY)
    assert field.default == DEFAULT_MAX_AGE_HOURS == 24
    assert (field.minimum, field.maximum) == (0, MOST_MAX_AGE_HOURS)
    for bad in (-1, MOST_MAX_AGE_HOURS + 1, "a day", 1.5, True, None):
        assert max_age_hours(bad) == 24, bad
    assert max_age_hours(0) == 0 and max_age_hours(6) == 6

    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({MAX_AGE_HOURS_KEY: "a day"}), encoding="utf-8")
    state = AgentState(path)
    state.load()
    assert state.as_config_document().model_dump()[MAX_AGE_HOURS_KEY] == 24


@pytest.mark.usefixtures("_root_over_machine_tls")
def test_the_agent_says_the_limit_and_whether_it_refuses_now(client: TestClient) -> None:
    """Settings never lie: the limit in effect, on `GET /v1/node` and on the
    setting itself, and whether it is refusing changes now."""
    root, _ = _install(client)
    session = _bearer(root.session("node:gpu-box"))

    def said() -> tuple[dict[str, Any], dict[str, Any]]:
        node = client.get("/v1/node", headers=session).json()
        schema = client.get("/v1/config/schema", headers=session).json()
        field = next(f for f in schema["fields"] if f["key"] == "maxTrustBundleAgeHours")
        return node, field

    node, field = said()
    assert node["trustBundleMaxAgeSeconds"] == 24 * 3600
    assert node["trustBundleFrozen"] is False
    assert field["default"] == 24 and field.get("status") is None

    _silent_for(client, 25 * 3600)
    node, field = said()
    assert node["trustBundleFrozen"] is True
    assert field["status"] == {"level": "warning", "text": _REFUSED_25_HOURS}

    client.patch("/v1/config", json={"maxTrustBundleAgeHours": 0}, headers=session)
    node, field = said()
    assert node["trustBundleMaxAgeSeconds"] == 0
    assert node["trustBundleFrozen"] is False
    assert field.get("status") is None


def test_a_machine_that_joined_nothing_has_no_limit_to_report(authed_client: TestClient) -> None:
    node = authed_client.get("/v1/node").json()
    assert "trustBundleFrozen" not in node and "trustBundleMaxAgeSeconds" not in node
    schema = authed_client.get("/v1/config/schema").json()
    field = next(f for f in schema["fields"] if f["key"] == "maxTrustBundleAgeHours")
    assert field["status"]["level"] == "info"


# --------------------------------------------------------------------------- #
# Row 46 review: a replay, a plain HTTP pull, and the browser proxy
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("_any_model_fits", "_root_over_machine_tls")
def test_a_replayed_bundle_then_a_restart_still_refuses(client: TestClient) -> None:
    """Review blocker 1: past the limit, the held bundle posted again to the
    push door (which takes no bearer) and the agent restarted. The bundle
    file is written again; the record of the root's last word is not."""
    root, attic = _install(client)
    _declare(client, root)
    _silent_for(client, 25 * 3600)
    other = _from(attic)
    assert client.post("/v1/runtimes/big/stop", headers=other).status_code == 403
    trust = _app_of(client).state.auth_state.trust
    replayed = client.post("/v1/node/trust-bundle", json={"jws": trust.bundle.jws})
    assert replayed.status_code == 200, replayed.text
    assert client.post("/v1/runtimes/big/stop", headers=other).status_code == 403
    # What the agent reads at its next start.
    trust.load()
    refused = client.post("/v1/runtimes/big/stop", headers=other)
    assert refused.status_code == 403, refused.text
    assert refused.json()["detail"]["detail"] == _REFUSED_25_HOURS


def test_a_pull_in_plain_http_is_not_word_from_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review blocker 2: a party in the middle of a plain HTTP pull answers
    with the bundle it saw last. It cannot unfreeze anything: in plain HTTP
    the limit is not kept, and when the root is reached over machine TLS
    again, the silence counts from the last real word."""
    trust, _ = _enrolled(tmp_path)
    _heard(tmp_path, 25 * 3600, 25 * 3600)
    trust.load()
    held = trust.bundle
    assert held is not None
    tls = {"to the root": False}
    monkeypatch.setattr(trust_module, "channel_authenticated", lambda url: tls["to the root"])

    trust.accept(held.jws, pulled=True)
    assert (trust.word_age_seconds() or 0) >= 25 * 3600 - 5
    # The root's address answered, which is all plain HTTP can say.
    assert (trust.heard_age_seconds() or 0) < 5
    # And the limit is not kept: a live root whose bundle does not change
    # would freeze its machines after a day.
    assert trust.limit_in_effect(24) == 0
    assert trust.frozen_age(24) is None

    tls["to the root"] = True
    assert trust.frozen_age(24) is not None
    trust.load()
    assert trust.frozen_age(24) is not None
    # The root itself, over its machine TLS: word now.
    trust.accept(held.jws, pulled=True)
    assert trust.frozen_age(24) is None
    assert (trust.word_age_seconds() or 0) < 5


@pytest.mark.usefixtures("_any_model_fits")
def test_without_an_authenticated_channel_the_limit_is_not_kept_and_says_so(
    client: TestClient,
) -> None:
    """The test root is plain HTTP across a network: the limit is not kept,
    nothing is refused, and the agent says why rather than a limit it does
    not keep."""
    root, attic = _install(client)
    _declare(client, root)
    _silent_for(client, 25 * 3600)
    assert client.post("/v1/runtimes/big/stop", headers=_from(attic)).status_code == 202
    # The runtime's companion driver is in the topology; the stub runs none.
    _app_of(client).state.supervisor.is_supervised = lambda _name: False
    session = _bearer(root.session("node:gpu-box"))
    node = client.get("/v1/node", headers=session).json()
    assert node["trustBundleFrozen"] is False
    assert node["trustBundleMaxAgeSeconds"] == 0
    schema = client.get("/v1/config/schema", headers=session).json()
    field = next(f for f in schema["fields"] if f["key"] == "maxTrustBundleAgeHours")
    assert field["status"] == {"level": "warning", "text": trust_module.NOT_ENFORCED}
    assert field["status"]["text"] == (
        "Not enforced: this machine reaches the root without encryption."
    )


def test_which_channels_only_the_root_can_answer() -> None:
    from eugene_plexus_agent import _http
    from eugene_plexus_agent.trust import channel_authenticated

    assert channel_authenticated("http://127.0.0.1:8283")
    assert channel_authenticated("http://localhost:8283/")
    assert channel_authenticated("http://[::1]:8283")
    assert not channel_authenticated("http://10.0.0.5:8283")
    assert not channel_authenticated("not a url")
    try:
        _http.set_machine_table(
            _http.parse_machine_table(
                {
                    "machines": {"10.0.0.5:9443": ["pin"]},
                    "routes": {"http://10.0.0.5:8283": "https://10.0.0.5:9443/control"},
                }
            )
        )
        assert channel_authenticated("http://10.0.0.5:8283")
        assert channel_authenticated("https://10.0.0.5:9443/control")
        assert not channel_authenticated("http://10.0.0.6:8283")
        assert not channel_authenticated("https://10.0.0.6:9443/control")
    finally:
        _http.set_machine_table(None)


def _components(client: TestClient) -> list[httpx.Request]:
    """A Library and an inference-driver on this machine, behind a stub."""
    from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind

    app = _app_of(client)
    for name, kind in (("library", "library"), ("llama", "inference-driver")):
        app.state.agent_state.add_topology_entry(
            ComponentEntry(name=name, kind=ComponentKind(kind), url="http://127.0.0.1:9/")  # type: ignore[arg-type]
        )
    seen: list[httpx.Request] = []

    class Body(httpx.AsyncByteStream):
        """The proxy streams what it gets back; a mock body must be a stream."""

        def __init__(self, data: bytes) -> None:
            self.data = data

        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield self.data

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "PUT":
            return httpx.Response(204, stream=Body(b""))
        return httpx.Response(
            200, headers={"content-type": "application/json"}, stream=Body(b'{"ok": true}')
        )

    app.state.ui_proxy_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return seen


@pytest.mark.usefixtures("_root_over_machine_tls")
def test_a_change_through_the_proxy_from_another_machine_is_refused_past_the_limit(
    client: TestClient,
) -> None:
    """Review should-fix 3: a component checks its token itself and knows
    nothing of the root's silence, so a write through this agent's proxy
    is refused here; answers and reads go on."""
    root, attic = _install(client)
    seen = _components(client)
    report = "/api/proxy/library/v1/nodes/attic/ollama"
    as_agent = _from(attic, "agent")
    assert client.put(report, json={}, headers=as_agent).status_code == 204

    _silent_for(client, 25 * 3600)
    reached = len(seen)
    refused = client.put(report, json={}, headers=as_agent)
    assert refused.status_code == 403, refused.text
    assert refused.json()["detail"]["title"] == "Root not heard"
    assert refused.json()["detail"]["detail"] == _REFUSED_25_HOURS
    for method, path in (
        ("POST", "/api/proxy/llama/v1/row46-new-door"),
        ("PATCH", "/api/proxy/llama/v1/config"),
        ("POST", "/api/proxy/library/v1/downloads"),
    ):
        answer = client.request(method, path, json={}, headers=_from(attic))
        assert answer.status_code == 403, (method, path, answer.text)
    assert len(seen) == reached, "a refused change reached the component"

    # Answers, questions and reads go on, from the same machine.
    assert (
        client.post("/api/proxy/llama/v1/generate", json={}, headers=_from(attic)).status_code
        == 200
    )
    assert (
        client.post("/api/proxy/library/v1/eligibility", json={}, headers=as_agent).status_code
        == 200
    )
    assert client.get("/api/proxy/library/v1/models", headers=as_agent).status_code == 200
    # The root's own session still changes it.
    session = _bearer(root.session("node:gpu-box"))
    assert client.put(report, json={}, headers=session).status_code == 204


def test_a_proxied_request_to_a_kind_nobody_named_is_a_change() -> None:
    from eugene_plexus_agent.dependencies import proxied_change

    assert proxied_change("inference-driver", "POST", "/v1/generate/stream") is False
    assert proxied_change("gateway", "POST", "/v1/models/qwen:countTokens") is False
    assert proxied_change("gateway", "POST", "/v1/models/a/b:countTokens") is True
    assert proxied_change("library", "POST", "/v1/keep-jobs/j1/files/state") is False
    assert proxied_change("library", "PUT", "/v1/keep-jobs/j1/files") is True
    assert proxied_change("store", "POST", "/v1/files") is True
    assert proxied_change(None, "POST", "/v1/generate") is True
    assert proxied_change("something-new", "POST", "/v1/generate") is True
    assert proxied_change("store", "GET", "/v1/files") is False
