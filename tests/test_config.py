"""Tests for the standard config trio (UI prefs + firstRunComplete)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_config_schema_lists_expected_fields(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "agent"
    keys = {f["key"] for f in body["fields"]}
    assert keys == {"firstRunComplete", "securityMode", "uiTheme", "uiFontSize"}


def test_get_config_returns_defaults_on_first_run(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert body["firstRunComplete"] is False
    assert body["uiTheme"] == "auto"
    assert body["uiFontSize"] == "medium"


def test_patch_config_applies_valid_change(authed_client: TestClient) -> None:
    response = authed_client.patch("/v1/config", json={"uiTheme": "dark"})
    assert response.status_code == 200
    body = response.json()
    assert "uiTheme" in body["applied"]
    assert body["rejected"] == []
    # Agent config never requires restart in v0.1 — UI prefs can apply live.
    assert body["requiresRestart"] is False

    follow = authed_client.get("/v1/config")
    assert follow.json()["uiTheme"] == "dark"


def test_patch_config_rejects_invalid_enum(authed_client: TestClient) -> None:
    response = authed_client.patch("/v1/config", json={"uiTheme": "neon"})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert any(r["key"] == "uiTheme" for r in body["rejected"])


def test_patch_config_rejects_unknown_field(authed_client: TestClient) -> None:
    response = authed_client.patch("/v1/config", json={"madeUpField": 42})
    body = response.json()
    assert any(r["key"] == "madeUpField" and "unknown" in r["message"] for r in body["rejected"])


def test_first_run_complete_flips_through_patch(authed_client: TestClient) -> None:
    response = authed_client.patch("/v1/config", json={"firstRunComplete": True})
    assert response.status_code == 200
    assert "firstRunComplete" in response.json()["applied"]

    follow = authed_client.get("/v1/config")
    assert follow.json()["firstRunComplete"] is True
