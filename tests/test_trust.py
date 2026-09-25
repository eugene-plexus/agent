"""`NodeTrust` on its own: what this node signs, and for whom.

The routes that use it are tested where they live; these pin the rules
a route cannot see, because the verifier on the far side would refuse
the token anyway and a route test would pass either way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eugene_plexus_agent import tokens
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
