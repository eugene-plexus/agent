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
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from eugene_plexus_agent import tokens
from eugene_plexus_agent import trust as trust_module
from eugene_plexus_agent.node_identity import FencedError, NodeIdentityStore
from eugene_plexus_agent.trust import BUNDLE_FILE, MintRefused, NodeTrust

from .conftest import FakeRoot, enroll_store, standalone_trust


def _enrolled(tmp_path: Path, *, grants: tuple[str, ...] = ()) -> tuple[NodeTrust, FakeRoot]:
    root = FakeRoot()
    store = NodeIdentityStore(tmp_path / "node.yaml")
    enroll_store(store, root, "gpu-box", grants=grants)
    trust = NodeTrust(store, tmp_path / BUNDLE_FILE)
    trust.load()
    return trust, root


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


def test_the_age_is_since_the_bundle_was_taken_not_since_it_was_signed(tmp_path: Path) -> None:
    """A pull returns the same signed bundle until something changes, so
    the bundle's own `iat` is days old on a quiet install that heard from
    its root a minute ago. The age is when this node last took one."""
    import os
    import time

    trust, root = _enrolled(tmp_path)
    # Heard fifteen minutes ago, as far as the kept file says; taking a
    # bundle now is what must bring the age back to zero.
    then = time.time() - 900
    os.utime(tmp_path / BUNDLE_FILE, (then, then))
    trust.load()
    assert (trust.heard_age_seconds() or 0) >= 895
    old = tokens.build_bundle(
        authority=root.identity,
        version=root.version + 1,
        epoch=root.epoch,
        keys=[root.token.trust_key(["authority"]), *root.members.values()],
        now=int(time.time()) - 5 * 86400,
    )
    trust.accept(old.jws)
    age = trust.heard_age_seconds()
    assert age is not None and age < 5
    assert trust.heard_age_seconds(now=time.time() + 700) in (699, 700, 701)


def test_the_age_survives_a_restart_from_the_kept_file(tmp_path: Path) -> None:
    import os
    import time

    _enrolled(tmp_path)
    kept = tmp_path / BUNDLE_FILE
    then = time.time() - 900
    os.utime(kept, (then, then))
    again = NodeTrust(NodeIdentityStore(tmp_path / "node.yaml"), kept)
    again._identity.load()
    again.load()
    age = again.heard_age_seconds()
    assert age is not None and 895 <= age <= 905


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
