"""Tests for the standard config trio (UI prefs + firstRunComplete)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from eugene_plexus_agent.state import AgentState


def test_get_config_schema_lists_expected_fields(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "agent"
    keys = {f["key"] for f in body["fields"]}
    assert keys == {
        "firstRunComplete",
        "securityMode",
        "maxTrustBundleAgeHours",
        "vllmBinary",
        "mlxBinary",
        "kevPython",
        "strataServer",
        "engineBinaryRoots",
        "allowUnrestrictedEngineLaunch",
        "advertiseUrl",
        "machineUrl",
        "machineKeyChangeDays",
        "clientCertificateFile",
        "clientKeyFile",
        "allowedHosts",
        "pathMappings",
        "shareCredentials",
        "modelCopyEnabled",
        "modelCopyDir",
        "hostHistoryEnabled",
        "hostHistorySampleSeconds",
        "hostHistoryRawHours",
        "hostHistoryMinuteDays",
        "hostHistoryHourDays",
        "modelCopyMinFreeGb",
        "uvBinary",
        "allowCustomApps",
        "updateChecks",
        "updateChannel",
        "reportOllama",
    }
    # Checking is on by default and installs nothing; the channel is a
    # binary choice whose default is releases (Troy, 2026-09-29).
    checks = next(f for f in body["fields"] if f["key"] == "updateChecks")
    assert checks["category"] == "updates" and checks["default"] is True
    channel = next(f for f in body["fields"] if f["key"] == "updateChannel")
    assert channel["enumValues"] == ["edge", "releases"] and channel["default"] == "releases"
    assert body["categories"]["updates"] == "Updates"
    # Apps have their own category; running code that is not in the
    # catalogue is an expert switch and off until someone turns it on.
    custom = next(f for f in body["fields"] if f["key"] == "allowCustomApps")
    assert custom["category"] == "apps" and custom["default"] is False
    assert body["categories"]["apps"] == "Apps"
    # The copy trio is its own category, next to but not inside the
    # Library folder overrides: "where the share is mounted here" and
    # "does this machine keep its own copy" are different questions with
    # the same subject, and each field's text points at the other
    # (`cross-link-related-settings`).
    copy_enabled = next(f for f in body["fields"] if f["key"] == "modelCopyEnabled")
    assert copy_enabled["valueType"] == "boolean"
    assert copy_enabled["category"] == "modelStorage"
    assert body["categories"]["modelStorage"] == "Model storage"
    # `vllmBinary` is a plain config field on purpose — the generic editor
    # renders a `file_path` with no engine-specific UI code.
    vllm_binary = next(f for f in body["fields"] if f["key"] == "vllmBinary")
    assert vllm_binary["valueType"] == "file_path"
    assert vllm_binary["category"] == "engines"
    assert "engines" in body["categories"]
    # `advertiseUrl` (M7) is a `url` under its own category, so the editor
    # groups where-other-hosts-reach-me apart from engines and appearance.
    advertise = next(f for f in body["fields"] if f["key"] == "advertiseUrl")
    assert advertise["valueType"] == "url"
    assert advertise["category"] == "node"
    assert "node" in body["categories"]


def test_get_config_returns_defaults_on_first_run(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert body["firstRunComplete"] is False
    assert body["securityMode"] == "prompt_on_startup"


def test_patch_config_applies_valid_change(authed_client: TestClient) -> None:
    response = authed_client.patch("/v1/config", json={"modelCopyMinFreeGb": 60})
    assert response.status_code == 200
    body = response.json()
    assert "modelCopyMinFreeGb" in body["applied"]
    assert body["rejected"] == []
    # Nothing here is read only at start, so a save applies live.
    assert body["requiresRestart"] is False

    follow = authed_client.get("/v1/config")
    assert follow.json()["modelCopyMinFreeGb"] == 60


def test_patch_config_rejects_invalid_enum(authed_client: TestClient) -> None:
    response = authed_client.patch("/v1/config", json={"securityMode": "neon"})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert any(r["key"] == "securityMode" for r in body["rejected"])


def test_ui_theme_and_font_size_are_retired(authed_client: TestClient, tmp_path: Path) -> None:
    """Nothing read either (the console keeps its look in the browser), so a
    setting that offered them showed a choice with no effect. Gone from the
    schema and the document; a PATCH is refused as retired, not as unknown; a
    file an older build wrote, holding them, still loads."""
    schema = authed_client.get("/v1/config/schema").json()
    assert not {"uiTheme", "uiFontSize"} & {f["key"] for f in schema["fields"]}
    assert "ui" not in schema["categories"], "no field is left in Appearance"
    document = authed_client.get("/v1/config").json()
    assert "uiTheme" not in document and "uiFontSize" not in document

    for key, value in (("uiTheme", "dark"), ("uiFontSize", "large")):
        body = authed_client.patch("/v1/config", json={key: value}).json()
        assert body["applied"] == []
        [refused] = body["rejected"]
        assert refused["key"] == key
        assert "retired" in refused["message"], refused
        assert "unknown" not in refused["message"]

    path = tmp_path / "old-agent.yaml"
    path.write_text(
        "firstRunComplete: true\nuiTheme: dark\nuiFontSize: large\ncomponents: []\nruntimes: []\n",
        encoding="utf-8",
        newline="\n",
    )
    state = AgentState(path)
    state.load()
    loaded = state.as_config_document().model_dump()
    assert loaded["firstRunComplete"] is True
    assert "uiTheme" not in loaded and "uiFontSize" not in loaded


def test_patch_config_rejects_unknown_field(authed_client: TestClient) -> None:
    response = authed_client.patch("/v1/config", json={"madeUpField": 42})
    body = response.json()
    assert any(r["key"] == "madeUpField" and "unknown" in r["message"] for r in body["rejected"])


def test_vllm_binary_round_trips_and_is_not_checked_for_existence(
    authed_client: TestClient,
) -> None:
    """Existence is checked at discovery and at spawn, where a missing
    file is reported with the path named. Rejecting it here would stop an
    operator from pointing at an environment they are about to create."""
    response = authed_client.patch("/v1/config", json={"vllmBinary": "/opt/vllm/.venv/bin/vllm"})
    assert response.status_code == 200
    assert "vllmBinary" in response.json()["applied"]
    assert response.json()["requiresRestart"] is False
    assert authed_client.get("/v1/config").json()["vllmBinary"] == "/opt/vllm/.venv/bin/vllm"

    wrong_type = authed_client.patch("/v1/config", json={"vllmBinary": 42})
    assert any(r["key"] == "vllmBinary" for r in wrong_type.json()["rejected"])

    cleared = authed_client.patch("/v1/config", json={"vllmBinary": None})
    assert "vllmBinary" in cleared.json()["applied"]
    assert authed_client.get("/v1/config").json().get("vllmBinary") is None


def test_first_run_complete_flips_through_patch(authed_client: TestClient) -> None:
    response = authed_client.patch("/v1/config", json={"firstRunComplete": True})
    assert response.status_code == 200
    assert "firstRunComplete" in response.json()["applied"]

    follow = authed_client.get("/v1/config")
    assert follow.json()["firstRunComplete"] is True


def test_an_integer_field_can_actually_be_saved(authed_client: TestClient) -> None:
    """Found by the model-copy acceptance run, not by a unit test.

    `_validate` had no `integer` branch, so the first integer field this
    agent ever declared -- `modelCopyMinFreeGb` -- fell through to
    "unsupported valueType" and every PATCH of it was refused. The
    refusal is reported inside a 200 (`ConfigUpdateResult.errors`), so
    nothing that checked the status code could see it, and the value
    could only ever be set by editing agent.yaml by hand: the setting was
    invisible to the UI that exists to expose it.
    """
    response = authed_client.patch("/v1/config", json={"modelCopyMinFreeGb": 120})
    assert response.status_code == 200
    body = response.json()
    assert body["rejected"] == [], body["rejected"]
    assert "modelCopyMinFreeGb" in body["applied"]
    assert authed_client.get("/v1/config").json()["modelCopyMinFreeGb"] == 120


def test_an_integer_field_refuses_what_is_not_a_whole_number(authed_client: TestClient) -> None:
    for bad_value in (True, 12.5, "120"):
        body = authed_client.patch("/v1/config", json={"modelCopyMinFreeGb": bad_value}).json()
        assert [r["key"] for r in body["rejected"]] == ["modelCopyMinFreeGb"], bad_value
        assert body["applied"] == [], bad_value
    negative = authed_client.patch("/v1/config", json={"modelCopyMinFreeGb": -1}).json()
    assert negative["rejected"][0]["message"] == "must not be negative"
