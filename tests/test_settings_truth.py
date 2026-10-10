"""Settings never lie (Troy, 2026-09-29: fundamental).

No widget anywhere may show a value other than the one in effect. Found on
Troy's worker: `updateChannel` had never been saved, the agent followed its
alpha.5 install (releases), and Settings showed Edge.

What is pinned here, for the agent's own config trio:

- **`updateChannel` is a binary choice with a default** -- `releases`, or
  what the environment names (the `:edge` image sets `edge`). The default
  is reported as the value in effect and never written to `agent.yaml`.
- **No install changes channel because of that.** One that never saved a
  channel keeps the one it followed: a container's from its image, at load;
  a native install's at its first update check that reads the releases.
  Until then it is `pending`, and says so rather than showing a default it
  is not using.
- **A null in the file is the default**, as a PATCH of null is.
- **The schema says what an unset value does on this machine**: the
  engine found on PATH, the derived advertise address, a security mode
  that cannot work here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest
from eugene_plexus_agent.state import (
    DEFAULT_UPDATE_CHANNEL_VARIABLE,
    AgentState,
)


def _state(tmp_path: Path, content: dict[str, Any] | None) -> AgentState:
    path = tmp_path / "agent.yaml"
    if content is not None:
        path.write_text(yaml.safe_dump(content), encoding="utf-8")
    state = AgentState(path)
    state.load()
    return state


def _on_disk(tmp_path: Path) -> dict[str, Any]:
    return yaml.safe_load((tmp_path / "agent.yaml").read_text(encoding="utf-8")) or {}


def _field(state: AgentState, key: str):  # type: ignore[no-untyped-def]
    return next(f for f in state.as_config_schema().fields if f.key == key)


# --------------------------------------------------------------------------- #
# The update channel
# --------------------------------------------------------------------------- #


def test_a_new_install_follows_the_default_and_never_writes_it(tmp_path: Path) -> None:
    state = _state(tmp_path, None)
    assert state.as_config_document().model_dump()["updateChannel"] == "releases"
    assert "updateChannel" not in _on_disk(tmp_path)
    field = _field(state, "updateChannel")
    assert field.default == "releases" and field.defaultSource is None
    assert field.unsetMeans is None


def test_the_environment_names_the_default_and_the_schema_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DEFAULT_UPDATE_CHANNEL_VARIABLE, "edge")
    monkeypatch.setenv("EUGENE_PLEXUS_CONTAINER_IMAGE", "ghcr.io/eugene-plexus/control-plane:edge")
    state = _state(tmp_path, None)
    assert state.as_config_document().model_dump()["updateChannel"] == "edge"
    field = _field(state, "updateChannel")
    assert field.default == "edge"
    assert field.defaultSource and DEFAULT_UPDATE_CHANNEL_VARIABLE in field.defaultSource
    assert "control-plane:edge" in field.defaultSource
    # A reset goes back to it, and still writes nothing.
    state.apply_config_patch(ConfigUpdateRequest.model_validate({"updateChannel": "releases"}))
    assert _on_disk(tmp_path)["updateChannel"] == "releases"
    state.apply_config_patch(ConfigUpdateRequest.model_validate({"updateChannel": None}))
    assert "updateChannel" not in _on_disk(tmp_path)
    assert state.as_config_document().model_dump()["updateChannel"] == "edge"


def test_a_saved_channel_is_kept(tmp_path: Path) -> None:
    state = _state(tmp_path, {"updateChannel": "edge"})
    assert state.as_config_document().model_dump()["updateChannel"] == "edge"


def test_a_file_with_no_channel_follows_the_default(tmp_path: Path) -> None:
    state = _state(tmp_path, {"firstRunComplete": True})
    assert state.as_config_document().model_dump()["updateChannel"] == "releases"
    assert "updateChannel" not in _on_disk(tmp_path)
    assert _field(state, "updateChannel").unsetMeans is None


def test_a_container_follows_the_default_its_environment_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EUGENE_PLEXUS_CONTAINER_IMAGE", "ghcr.io/eugene-plexus/control-plane:edge")
    monkeypatch.setenv(DEFAULT_UPDATE_CHANNEL_VARIABLE, "edge")
    state = _state(tmp_path, {"firstRunComplete": True})
    assert "updateChannel" not in _on_disk(tmp_path)
    assert state.as_config_document().model_dump()["updateChannel"] == "edge"


def test_a_channel_that_is_not_one_is_ignored_for_the_default(tmp_path: Path) -> None:
    state = _state(tmp_path, {"updateChannel": "beta"})
    assert state.as_config_document().model_dump()["updateChannel"] == "releases"


# --------------------------------------------------------------------------- #
# Nulls and defaults
# --------------------------------------------------------------------------- #


def test_a_null_in_the_file_is_the_default(tmp_path: Path) -> None:
    state = _state(tmp_path, {"updateChecks": None, "modelCopyMinFreeGb": None})
    doc = state.as_config_document().model_dump()
    assert doc["updateChecks"] is True and doc["modelCopyMinFreeGb"] == 50


def test_security_mode_needs_no_restart(tmp_path: Path) -> None:
    """The keyring entry is written or removed at PATCH time and the mode is
    read at the next start: nothing in the running process changes."""
    assert _field(_state(tmp_path, None), "securityMode").requiresRestart is False


# --------------------------------------------------------------------------- #
# The live schema
# --------------------------------------------------------------------------- #


def _schema_field(client: TestClient, key: str) -> dict[str, Any]:
    body = client.get("/v1/config/schema").json()
    return next(f for f in body["fields"] if f["key"] == key)


def test_an_unset_engine_path_says_what_path_finds(
    authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_agent.routes import config as config_routes

    found = {"vllm": "/opt/vllm/bin/vllm", "mlx_lm.server": None}
    monkeypatch.setattr(config_routes.shutil, "which", lambda name: found.get(name))
    vllm = _schema_field(authed_client, "vllmBinary")
    assert vllm["unsetResolvesTo"] == "/opt/vllm/bin/vllm"
    assert "/opt/vllm/bin/vllm" in vllm["unsetMeans"]
    mlx = _schema_field(authed_client, "mlxBinary")
    assert mlx.get("unsetResolvesTo") is None and "no `mlx_lm.server`" in mlx["unsetMeans"]


def test_passphrase_file_with_nowhere_to_keep_it_warns(authed_client: TestClient) -> None:
    authed_client.app.state.settings.passphrase_file = None  # type: ignore[attr-defined]
    saved = authed_client.patch("/v1/config", json={"securityMode": "passphrase_file"})
    assert saved.json()["rejected"] == []
    status = _schema_field(authed_client, "securityMode")["status"]
    assert status["level"] == "warning" and "every start" in status["text"]


def test_copying_on_with_no_folder_warns(authed_client: TestClient) -> None:
    authed_client.patch("/v1/config", json={"modelCopyEnabled": True})
    field = _schema_field(authed_client, "modelCopyDir")
    assert field["status"]["level"] == "warning" and "nothing is copied" in field["unsetMeans"]


def test_an_unset_advertise_address_says_it_is_not_set(authed_client: TestClient) -> None:
    field = _schema_field(authed_client, "advertiseUrl")
    assert field["unsetMeans"]
