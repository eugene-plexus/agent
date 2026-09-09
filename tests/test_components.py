"""Tests for the topology endpoints under /v1/components.

Skeleton-level — verifies the wire shape and persistence semantics.
Lifecycle behavior (actual subprocess spawning, status reporting, real
restart) lands when the supervisor is implemented.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from eugene_plexus_agent import security


def _service_token(client: TestClient) -> str:
    """Mint a validly-signed service-audience token for the running app.

    Peers (the gateway, the drivers) call /v1/components with a
    token like this to auto-resolve topology. Signed with the app's live
    signing key so signature verification passes; only the audience marks
    it as a service rather than operator token.
    """
    signing_key = client.app.state.auth_state.signing_key  # type: ignore[attr-defined]
    return security.issue_service_token(signing_key=signing_key, kind="gateway")


def _gateway_entry() -> dict[str, object]:
    return {
        "name": "gateway",
        "kind": "gateway",
        "url": "http://127.0.0.1:8080",
        "spawn": {"configFile": "/tmp/orch/config.yaml"},
        "safeMode": False,
    }


def _left_driver_entry() -> dict[str, object]:
    return {
        "name": "left",
        "kind": "inference-driver",
        "url": "http://127.0.0.1:8081",
        "spawn": {"configFile": "/tmp/drivers/left/config.yaml"},
        "safeMode": False,
    }


def test_list_components_starts_empty(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/components")
    assert response.status_code == 200
    assert response.json() == {"components": []}


def test_create_then_list_includes_component(authed_client: TestClient) -> None:
    response = authed_client.post("/v1/components", json=_gateway_entry())
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "gateway"
    # Stub supervisor reports `unreachable` since it never spawns; real
    # supervisor would transition through starting -> running.
    assert body["status"] == "unreachable"

    listing = authed_client.get("/v1/components").json()
    names = [c["name"] for c in listing["components"]]
    assert names == ["gateway"]


def test_create_rejects_duplicate_name(authed_client: TestClient) -> None:
    authed_client.post("/v1/components", json=_gateway_entry())
    response = authed_client.post("/v1/components", json=_gateway_entry())
    assert response.status_code == 409


def test_get_component_404_when_missing(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/components/nonexistent")
    assert response.status_code == 404


def test_patch_component_updates_safe_mode(authed_client: TestClient) -> None:
    authed_client.post("/v1/components", json=_gateway_entry())

    updated = _gateway_entry()
    updated["safeMode"] = True
    response = authed_client.patch("/v1/components/gateway", json=updated)
    assert response.status_code == 200
    assert response.json()["safeMode"] is True


def test_patch_component_rejects_rename_collision(authed_client: TestClient) -> None:
    authed_client.post("/v1/components", json=_gateway_entry())
    authed_client.post("/v1/components", json=_left_driver_entry())

    rename_to_existing = _gateway_entry()
    rename_to_existing["name"] = "left"
    response = authed_client.patch("/v1/components/gateway", json=rename_to_existing)
    assert response.status_code == 409


def test_delete_component_removes_it(authed_client: TestClient) -> None:
    authed_client.post("/v1/components", json=_gateway_entry())
    response = authed_client.delete("/v1/components/gateway")
    assert response.status_code == 204

    listing = authed_client.get("/v1/components").json()
    assert listing["components"] == []


def test_delete_component_404_when_missing(authed_client: TestClient) -> None:
    response = authed_client.delete("/v1/components/nonexistent")
    assert response.status_code == 404


def test_restart_component_404_when_missing(authed_client: TestClient) -> None:
    response = authed_client.post("/v1/components/nonexistent/restart")
    assert response.status_code == 404


def test_restart_component_returns_202_and_calls_supervisor(
    authed_client: TestClient, stub_supervisor: object
) -> None:
    authed_client.post("/v1/components", json=_gateway_entry())
    response = authed_client.post("/v1/components/gateway/restart")
    assert response.status_code == 202
    body = response.json()
    assert body["scheduled"] is True
    # The route delegated to the supervisor's restart method.
    assert ("restart", "gateway") in stub_supervisor.calls  # type: ignore[attr-defined]


def test_restart_component_409_for_remote(authed_client: TestClient) -> None:
    """Per the spec: the agent cannot restart something it doesn't own."""
    remote_entry = {
        "name": "rtx5090",
        "kind": "inference-driver",
        "url": "http://gpu-box.tailnet:8081",
        # No spawn block => remote.
    }
    authed_client.post("/v1/components", json=remote_entry)
    response = authed_client.post("/v1/components/rtx5090/restart")
    assert response.status_code == 409


def test_topology_persists_across_state_reloads(
    authed_client: TestClient,
    settings: object,  # imported from conftest fixture
) -> None:
    """Adding a component must round-trip through the YAML file so a
    process restart picks it up. Drives this end-to-end by using a real
    on-disk path from the fixture."""
    authed_client.post("/v1/components", json=_gateway_entry())
    listing_before = authed_client.get("/v1/components").json()
    assert len(listing_before["components"]) == 1

    # Reload state from disk via a fresh app on the same config_file.
    # Inject another stub supervisor so the fresh lifespan doesn't try
    # to spawn the orchestrator package for real.
    from fastapi.testclient import TestClient as FreshClient

    from eugene_plexus_agent.app import create_app

    from .conftest import StubSupervisor

    fresh_app = create_app(settings=settings)  # type: ignore[arg-type]
    fresh_app.state.supervisor = StubSupervisor()
    with FreshClient(fresh_app) as fresh:
        # Fresh process; need to log in to talk to protected routes.
        # The passphrase + master-key salt persisted across the restart,
        # so this re-derives the same master key and issues a new
        # session token signed by the fresh process's new signing key.
        from .conftest import TEST_PASSPHRASE

        login = fresh.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
        assert login.status_code == 200, f"login failed: {login.text}"
        fresh.headers["Authorization"] = f"Bearer {login.json()['sessionToken']}"

        response = fresh.get("/v1/components")
        assert response.status_code == 200
        assert response.json() == listing_before


# --------------------------------------------------------------------------- #
# v0.2.1: service tokens may READ topology but never mutate it.
# Fixes the auth mismatch where peer auto-resolve silently no-op'd because
# the whole router was operator-only. (project_agent_components_auth_mismatch)
# --------------------------------------------------------------------------- #


def test_service_token_can_read_components_list(authed_client: TestClient) -> None:
    authed_client.post("/v1/components", json=_gateway_entry())
    # Swap the operator session for a peer's service token.
    authed_client.headers["Authorization"] = f"Bearer {_service_token(authed_client)}"
    response = authed_client.get("/v1/components")
    assert response.status_code == 200
    assert [c["name"] for c in response.json()["components"]] == ["gateway"]


def test_service_token_can_read_single_component(authed_client: TestClient) -> None:
    authed_client.post("/v1/components", json=_gateway_entry())
    authed_client.headers["Authorization"] = f"Bearer {_service_token(authed_client)}"
    response = authed_client.get("/v1/components/gateway")
    assert response.status_code == 200
    assert response.json()["name"] == "gateway"


def test_service_token_cannot_create_component(authed_client: TestClient) -> None:
    authed_client.headers["Authorization"] = f"Bearer {_service_token(authed_client)}"
    response = authed_client.post("/v1/components", json=_gateway_entry())
    assert response.status_code == 401


def test_service_token_cannot_patch_component(authed_client: TestClient) -> None:
    authed_client.post("/v1/components", json=_gateway_entry())
    authed_client.headers["Authorization"] = f"Bearer {_service_token(authed_client)}"
    updated = _gateway_entry()
    updated["safeMode"] = True
    response = authed_client.patch("/v1/components/gateway", json=updated)
    assert response.status_code == 401


def test_service_token_cannot_delete_component(authed_client: TestClient) -> None:
    authed_client.post("/v1/components", json=_gateway_entry())
    authed_client.headers["Authorization"] = f"Bearer {_service_token(authed_client)}"
    response = authed_client.delete("/v1/components/gateway")
    assert response.status_code == 401


def test_service_token_cannot_restart_component(authed_client: TestClient) -> None:
    authed_client.post("/v1/components", json=_gateway_entry())
    authed_client.headers["Authorization"] = f"Bearer {_service_token(authed_client)}"
    response = authed_client.post("/v1/components/gateway/restart")
    assert response.status_code == 401


def test_unauthenticated_read_still_rejected(authed_client: TestClient) -> None:
    """Loosening to operator-OR-service is not the same as public —
    a request with no token at all is still 401 on the read endpoints."""
    authed_client.headers.pop("Authorization", None)
    assert authed_client.get("/v1/components").status_code == 401
