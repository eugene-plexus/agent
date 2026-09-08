"""Auth surface tests: initialize, login, logout, token enforcement.

Uses the bare `client` fixture (not `authed_client`) so each test
exercises a known starting state and drives the auth flow itself.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from eugene_plexus_watchdog import security

from .conftest import TEST_PASSPHRASE

# --------------------------------------------------------------------------- #
# Pre-init: only /healthz and /v1/auth/initialize work
# --------------------------------------------------------------------------- #


def test_protected_routes_locked_before_initialize(client: TestClient) -> None:
    """Until a passphrase is set, every endpoint except /healthz and
    /v1/auth/initialize requires a session token nobody has yet."""
    # Public routes still work.
    assert client.get("/healthz").status_code == 200
    # Protected routes return 401 (no token) — Setup-required 503 is
    # surfaced only through the dependency check on initialize-gated
    # endpoints; the bare auth dependency rejects with 401.
    for r in [
        client.get("/v1/config"),
        client.get("/v1/components"),
        client.patch("/v1/config", json={"uiTheme": "dark"}),
    ]:
        assert r.status_code == 401, f"expected 401 from {r.request.url}, got {r.status_code}"


def test_login_returns_503_when_uninitialized(client: TestClient) -> None:
    """Login can't succeed if no passphrase has ever been set."""
    response = client.post("/v1/auth/login", json={"passphrase": "anything"})
    assert response.status_code == 503
    body = response.json()
    assert "Setup required" in body["detail"]["title"]


# --------------------------------------------------------------------------- #
# Initialize
# --------------------------------------------------------------------------- #


def test_initialize_sets_passphrase_and_returns_session_token(client: TestClient) -> None:
    response = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    assert response.status_code == 200
    body = response.json()
    assert body["sessionToken"]
    assert body["expiresAt"]
    # The token should work against a protected endpoint.
    client.headers["Authorization"] = f"Bearer {body['sessionToken']}"
    config = client.get("/v1/config")
    assert config.status_code == 200


def test_initialize_refuses_when_already_initialized(client: TestClient) -> None:
    assert (
        client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
    )
    second = client.post("/v1/auth/initialize", json={"passphrase": "another-passphrase"})
    assert second.status_code == 409
    assert "Already initialized" in second.json()["detail"]["title"]


def test_initialize_rejects_empty_passphrase(client: TestClient) -> None:
    response = client.post("/v1/auth/initialize", json={"passphrase": ""})
    # FastAPI's pydantic min_length=1 returns 422.
    assert response.status_code == 422


def test_initialize_persists_passphrase_for_subsequent_login(client: TestClient) -> None:
    """The passphrase hash + master-key salt must round-trip to disk so
    a watchdog restart (modelled here by a second login on the same
    state) verifies against the same hash."""
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    # Don't reuse the initialize-issued token; do a fresh login.
    client.headers.pop("Authorization", None)
    login = client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    assert login.status_code == 200
    assert login.json()["sessionToken"]


# --------------------------------------------------------------------------- #
# Login
# --------------------------------------------------------------------------- #


def test_login_with_correct_passphrase_succeeds(client: TestClient) -> None:
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    client.headers.pop("Authorization", None)
    response = client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    assert response.status_code == 200
    assert response.json()["sessionToken"]


def test_login_with_wrong_passphrase_is_401(client: TestClient) -> None:
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    response = client.post("/v1/auth/login", json={"passphrase": "WRONG"})
    assert response.status_code == 401
    assert "Wrong passphrase" in response.json()["detail"]["title"]


def test_login_rate_limits_after_five_failures(client: TestClient) -> None:
    """Repeated wrong-passphrase attempts from the same source get
    locked out. The legitimate operator with the right passphrase
    still has to wait through the window."""
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    client.headers.pop("Authorization", None)
    for _ in range(5):
        r = client.post("/v1/auth/login", json={"passphrase": "WRONG"})
        assert r.status_code == 401
    blocked = client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    assert blocked.status_code == 429


# --------------------------------------------------------------------------- #
# Logout / token revocation
# --------------------------------------------------------------------------- #


def test_logout_revokes_current_session_token(client: TestClient) -> None:
    init = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    token = init.json()["sessionToken"]
    client.headers["Authorization"] = f"Bearer {token}"

    # Token works before logout.
    assert client.get("/v1/config").status_code == 200

    # Logout uses the same token to authorize revoking itself.
    response = client.delete("/v1/auth/sessions/current")
    assert response.status_code == 204

    # Same token must now be rejected.
    after = client.get("/v1/config")
    assert after.status_code == 401
    assert "revoked" in after.json()["detail"]["title"].lower()


def test_logout_without_token_is_401(client: TestClient) -> None:
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    client.headers.pop("Authorization", None)
    response = client.delete("/v1/auth/sessions/current")
    assert response.status_code == 401


# --------------------------------------------------------------------------- #
# Token validation (signature + audience + expiry)
# --------------------------------------------------------------------------- #


def test_bogus_token_rejected(client: TestClient) -> None:
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    client.headers["Authorization"] = "Bearer not-a-real-jwt"
    assert client.get("/v1/config").status_code == 401


def test_token_signed_with_wrong_key_rejected(client: TestClient) -> None:
    """A JWT signed with a key different from the watchdog's
    in-memory signing key must fail signature verification."""
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    bogus, _ = security.issue_operator_token(signing_key=b"\x00" * 32)
    client.headers["Authorization"] = f"Bearer {bogus}"
    assert client.get("/v1/config").status_code == 401


def test_service_token_rejected_for_operator_routes(client: TestClient) -> None:
    """A valid service token must NOT grant access to operator-audience
    routes. Defends against a leaked component-internal token being
    used against the UI surface."""
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    # The signing key lives on the running app's auth_state — pull it
    # off so the forged token IS validly signed but audience-wrong.
    signing_key = client.app.state.auth_state.signing_key  # type: ignore[attr-defined]
    svc = security.issue_service_token(signing_key=signing_key, kind="gateway")
    client.headers["Authorization"] = f"Bearer {svc}"
    assert client.get("/v1/config").status_code == 401


# --------------------------------------------------------------------------- #
# Master key derivation determinism
# --------------------------------------------------------------------------- #


def test_master_key_recovered_on_login(client: TestClient) -> None:
    """The master key set at initialize must equal the master key
    derived again at login (deterministic KDF). Tests reach into
    app.state to confirm — production code shouldn't, but this is the
    invariant that lets at-rest encryption survive process restarts."""
    init = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    assert init.status_code == 200
    master_after_init = client.app.state.auth_state.master_key  # type: ignore[attr-defined]
    assert master_after_init is not None
    assert len(master_after_init) == 32

    # Pretend the process restarted: wipe in-memory master_key. The
    # persisted passphrase hash + salt must let login re-derive the
    # exact same bytes.
    client.app.state.auth_state.master_key = None  # type: ignore[attr-defined]
    client.headers.pop("Authorization", None)
    login = client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    assert login.status_code == 200
    master_after_login = client.app.state.auth_state.master_key  # type: ignore[attr-defined]
    assert master_after_login == master_after_init


# --------------------------------------------------------------------------- #
# Envelope round-trip with the derived master key
# --------------------------------------------------------------------------- #


def test_envelope_round_trip_uses_derived_master_key(client: TestClient) -> None:
    """End-to-end: passphrase → master key → secretbox envelope →
    decrypt. This is the exact path child components will use for
    at-rest apiKey storage in subsequent phases of v0.2 security."""
    client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    master_key = client.app.state.auth_state.master_key  # type: ignore[attr-defined]

    envelope = security.seal("sk-test-secret-12345", master_key)
    assert envelope.alg == "secretbox-xsalsa20poly1305"
    decrypted = security.open_envelope(envelope, master_key)
    assert decrypted == "sk-test-secret-12345"


def test_argon2id_kdf_is_deterministic_for_same_inputs() -> None:
    """The same passphrase + salt MUST produce the same 32-byte key
    every time. This is what lets login recover the master key
    without storing it in plaintext."""
    salt = security.generate_master_key_salt()
    k1 = security.derive_master_key("hunter2", salt)
    k2 = security.derive_master_key("hunter2", salt)
    k3 = security.derive_master_key("hunter3", salt)
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 32


# --------------------------------------------------------------------------- #
# Schema includes securityMode field
# --------------------------------------------------------------------------- #


def test_config_schema_exposes_security_mode(authed_client: TestClient) -> None:
    """The wizard reads this to render the security screen's radio."""
    schema = authed_client.get("/v1/config/schema").json()
    field = next(f for f in schema["fields"] if f["key"] == "securityMode")
    assert field["valueType"] == "enum"
    assert set(field["enumValues"]) == {"prompt_on_startup", "os_keyring"}
    assert field["category"] == "security"
