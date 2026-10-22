"""Row 34 model selection and settings actually seeded into an app."""

import json

import pytest

from eugene_plexus_agent import app_launcher


def model(ident, locality="local", ready=1):
    return {
        "id": ident,
        "x_eugene_plexus": {"surfaces": ["image"], "locality": locality, "ready_backends": ready},
    }


def spec(tmp_path):
    key = tmp_path / "key"
    key.write_text("secret-key")
    return {
        "argv": ["unused"],
        "dataDir": str(tmp_path),
        "keyFile": str(key),
        "values": {"gatewayUrl": "http://unused.invalid"},
        "environment": {
            "MODEL": "{imageModel}",
            "ENABLE_IMAGE_GENERATION": "{imageModelSet}",
            "ENABLE_WEB_SEARCH": "true",
        },
        "resetOnConnectionChange": "RESET_CONFIG_ON_START",
    }


@pytest.mark.parametrize(
    "listing, chosen",
    [
        ([model("cloud", "external"), model("local")], "local"),
        ([model("stopped", ready=0), model("running")], "running"),
        ([model("cloud", "external"), model("stopped", ready=0)], "stopped"),
        ([model("cloud", "external")], "cloud"),
    ],
)
def test_image_model_placeholder(tmp_path, monkeypatch, listing, chosen):
    monkeypatch.setattr(app_launcher, "ask_gateway_for_models", lambda *_: (listing, None))
    _, env, _ = app_launcher.render_start(spec(tmp_path), {})
    assert env["MODEL"] == chosen
    assert env["ENABLE_IMAGE_GENERATION"] == "true"


def test_no_image_model_disables_images(tmp_path, monkeypatch):
    monkeypatch.setattr(app_launcher, "ask_gateway_for_models", lambda *_: ([], None))
    _, env, notes = app_launcher.render_start(spec(tmp_path), {"MODEL": "old"})
    assert "MODEL" not in env
    assert env["ENABLE_IMAGE_GENERATION"] == "false"
    assert any("image model" in note for note in notes)


def test_reset_settings_sets_variable_once(tmp_path, monkeypatch):
    listing = [model("first")]
    monkeypatch.setattr(app_launcher, "ask_gateway_for_models", lambda *_: (listing, None))
    start = spec(tmp_path)
    app_launcher.render_start(start, {})
    given = json.loads((tmp_path / "given.json").read_text())
    assert given["imageModel"] == "first"
    assert "secret-key" not in json.dumps(given)
    listing[:] = [model("second")]
    _, env, _ = app_launcher.render_start(start, {})
    assert "RESET_CONFIG_ON_START" not in env
    assert json.loads((tmp_path / "given.json").read_text()) == given
    (tmp_path / app_launcher.CONNECTION_FILE).write_text("reset-requested")
    _, env, _ = app_launcher.render_start(start, {})
    assert env["RESET_CONFIG_ON_START"] == "true"
    assert json.loads((tmp_path / "given.json").read_text())["imageModel"] == "second"
    _, env, _ = app_launcher.render_start(start, {})
    assert "RESET_CONFIG_ON_START" not in env
