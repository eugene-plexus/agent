"""R7: algorithm selection comes from trusted key material, never token headers."""

import base64
import logging
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eugene_plexus_agent import security


def _pair():
    private = Ed25519PrivateKey.generate()
    return (
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ),
    )


def _claims(aud="operator"):
    now = int(time.time())
    return {"sub": "operator", "aud": aud, "iat": now, "exp": now + 60}


def test_public_key_verifies_eddsa():
    private, public = _pair()
    token = jwt.encode(_claims(), private, algorithm="EdDSA")
    assert security.decode_token(token=token, signing_key=public).aud == "operator"


def test_legacy_hs256_remains_available_only_with_legacy_key():
    legacy = b"L" * 32
    token = jwt.encode(_claims(), legacy, algorithm="HS256")
    assert security.decode_token(token=token, signing_key=legacy).aud == "operator"
    _, public = _pair()
    with pytest.raises(jwt.InvalidTokenError):
        security.decode_token(token=token, signing_key=public)


def test_public_material_cannot_be_used_as_hmac_secret():
    _, public = _pair()
    raw_public = serialization.load_pem_public_key(public).public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    token = jwt.encode(_claims(), raw_public, algorithm="HS256")
    with pytest.raises(jwt.InvalidTokenError):
        security.decode_token(token=token, signing_key=public)


def test_wrong_public_key_and_unsigned_tokens_are_refused():
    private, _ = _pair()
    _, other = _pair()
    for token in (
        jwt.encode(_claims(), private, algorithm="EdDSA"),
        jwt.encode(_claims(), key="", algorithm="none"),
    ):
        with pytest.raises(jwt.InvalidTokenError):
            security.decode_token(token=token, signing_key=other)


def test_new_keys_mint_eddsa_and_public_half_cannot_mint():
    private = security.generate_signing_key()
    token, _ = security.issue_operator_token(signing_key=private)
    assert jwt.get_unverified_header(token)["alg"] == "EdDSA"
    public = security.verification_key(private)
    assert public.startswith(b"-----BEGIN PUBLIC KEY-----")
    assert security.decode_token(token=token, signing_key=public).aud == "operator"
    with pytest.raises(ValueError):
        security.issue_operator_token(signing_key=public)


@pytest.mark.parametrize(
    "kind,prefix", [("gateway", "GATEWAY"), ("library", "LIBRARY"), ("inference-driver", "DRIVER")]
)
@pytest.mark.parametrize("legacy", [False, True])
def test_children_receive_only_their_verification_material(kind, prefix, legacy):
    from eugene_plexus_agent._generated.models import ComponentEntry
    from eugene_plexus_agent.auth_state import AuthState
    from eugene_plexus_agent.supervisor import _ComponentPlanner

    key = b"L" * 32 if legacy else security.generate_signing_key()
    entry = ComponentEntry.model_validate(
        {
            "name": "test",
            "kind": kind,
            "url": "http://127.0.0.1:8181",
            "spawn": {"configFile": "config.yaml"},
        }
    )
    plan = _ComponentPlanner(entry, logging.getLogger("test"), AuthState(signing_key=key)).plan()
    assert plan is not None
    suffix = "AUTH_SIGNING_KEY" if legacy else "AUTH_VERIFY_KEY"
    received = base64.b64decode(plan.env[f"EUGENE_PLEXUS_{prefix}_{suffix}"])
    assert received == security.verification_key(key)
    if not legacy:
        assert f"EUGENE_PLEXUS_{prefix}_AUTH_SIGNING_KEY" not in plan.env
        assert base64.b64encode(key).decode() not in plan.env.values()
        with pytest.raises(ValueError):
            security.issue_operator_token(signing_key=received)
