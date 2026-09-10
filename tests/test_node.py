"""GET /v1/node — the unenrolled shape, with real devices.

Enrollment is still M5 debt; what this asserts is that the control
root's node poll finally gets an answer with a device inventory in it.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from eugene_plexus_agent import security


def test_node_reports_devices_and_unenrolled(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/node")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["enrolled"] is False
    assert "controlUrl" not in body
    assert body["os"] in ("windows", "linux", "macos")
    assert body["arch"] in ("x64", "arm64")
    kinds = [d["kind"] for d in body["devices"]]
    # The conftest detector: one CUDA card and the CPU, CPU last.
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
