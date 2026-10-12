"""What an app is told about the install's own models (MR6, specs
docs/design/workbench-memory-and-documents.md).

`{embeddingModel}` and `{rerankModel}` are answered at every start by asking
the gateway, with the app's own key, which models it lists for that key. The
checks are about what a plausible wrong implementation would get wrong:

* **never an outside model**: a model is the install's own when the gateway's
  listing says where it runs (`x_eugene_plexus.locality`). Not by `owned_by`:
  the real gateway fills that with the driver's provider key
  (`openai_compat_custom` for every runtime of ours) and says `local` only
  when no driver names a provider or two different ones do, so the contract's
  "`owned_by` is `local`" matches no local model and calls a model served by
  two outside accounts local;
* the first such model that lists the surface, the search model and the
  sorter chosen independently;
* a gateway that is down, slow, refusing, redirecting or wrong leaves the
  variable unset, and the start's notes name what was observed, never the key;
* one request per start, only for a manifest that names a model, never through
  the user's proxy and never to an address the gateway did not give;
* a change of model is not a change of connection.

The gateway here is a fake on a port of its own (`fake_gateway.py`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_agent import app_launcher, apps
from eugene_plexus_agent._generated.models import AppOrigin

from .fake_gateway import FakeGateway, listed
from .test_app_accounts import _free_port, _Ingress
from .test_app_start import EVERY, KEY, _fake_package, _manifest, _planner, _spec

BOTH = {"EMBED": "{embeddingModel}", "RERANK": "{rerankModel}"}
CHAT = "Qwen3-0.6B-Q4_K_M"
OUTSIDE = "openrouter/qwen/qwen3-embedding-8b"
UNSET_EMBED = (
    "{embeddingModel} has no value at this start: no local search model is running "
    "(the gateway lists none for this app's key)."
)
UNSET_RERANK = (
    "{rerankModel} has no value at this start: no local result sorter is running "
    "(the gateway lists none for this app's key)."
)


@pytest.fixture
def gateway() -> Iterator[FakeGateway]:
    fake = FakeGateway(key=KEY)
    try:
        yield fake
    finally:
        fake.close()


def _spec_for(tmp_path: Path, url: str | None, **overrides: Any) -> dict[str, Any]:
    """A start that names both models in its environment, asking the gateway at `url`."""
    values = {"bindHost": "127.0.0.1", "port": "8190"}
    if url is not None:
        values["gatewayUrl"] = url
    spec = _spec(tmp_path, args=[], environment=dict(BOTH), values=values)
    spec.update(overrides)
    return spec


def _start(
    spec: dict[str, Any], inherited: dict[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    _argv, env, notes = app_launcher.render_start(spec, dict(inherited or {}))
    return env, notes


# --------------------------------------------------------------------------- #
# which model
# --------------------------------------------------------------------------- #


def test_the_first_local_model_that_lists_the_surface_is_the_answer(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.models = [
        listed(CHAT, "chat", "completion"),
        listed("embed-a", "embeddings"),
        listed("embed-b", "embeddings"),
        listed("rerank-a", "rerank"),
        listed("rerank-b", "rerank"),
    ]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    # Not the chat model that came first, and not the second of either kind.
    assert (env["EMBED"], env["RERANK"]) == ("embed-a", "rerank-a")
    assert notes == []
    # One request, with the app's own key, for the models that key may use.
    assert gateway.requests == [{"path": "/v1/models", "authorization": f"Bearer {KEY}"}]


def test_a_local_model_is_known_by_where_it_runs_not_by_who_owns_it(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """The real gateway lists a model of a local runtime `owned_by` its driver's
    provider key, never `local` (probed against its app on 2026-10-12): the
    contract's "`owned_by` is `local`" matched none of them, and a start would
    have left both variables unset on every install."""
    gateway.models = [
        listed("embed-a", "embeddings", owned_by="openai_compat_custom", locality="local"),
        listed("rerank-a", "rerank", owned_by="strata_local", locality="local"),
    ]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert (env["EMBED"], env["RERANK"]) == ("embed-a", "rerank-a") and notes == []


@pytest.mark.parametrize(
    ("locality", "owned_by"),
    [
        pytest.param("external", "openrouter", id="a-cloud-account"),
        pytest.param("external", "local", id="two-outside-accounts-which-the-gateway-calls-local"),
        pytest.param("unknown", "openai_compat_custom", id="an-endpoint-nobody-said-was-local"),
        pytest.param(None, "local", id="a-gateway-that-does-not-say-where-models-run"),
    ],
)
def test_a_model_not_known_to_run_here_is_never_chosen(
    tmp_path: Path, gateway: FakeGateway, locality: str | None, owned_by: str
) -> None:
    gateway.models = [
        listed(OUTSIDE, "embeddings", locality=locality, owned_by=owned_by),
        listed(OUTSIDE + "-sorter", "rerank", locality=locality, owned_by=owned_by),
    ]
    env, notes = _start(
        _spec_for(tmp_path, gateway.url), {"EMBED": "inherited", "RERANK": "inherited"}
    )
    # Not chosen, and not an inherited value either.
    assert "EMBED" not in env and "RERANK" not in env
    said = " ".join(notes)
    assert OUTSIDE in said and OUTSIDE + "-sorter" in said
    assert "no local search model is running" in said and "no local result sorter" in said
    assert "the documents would leave the install" in said


def test_an_outside_model_listed_first_is_passed_over(tmp_path: Path, gateway: FakeGateway) -> None:
    gateway.models = [
        listed(OUTSIDE, "embeddings", locality="external", owned_by="openrouter"),
        listed("embed-local", "embeddings"),
    ]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert env["EMBED"] == "embed-local" and "RERANK" not in env
    # Only the sorter is missing, and it is the only thing said.
    assert notes == [
        UNSET_RERANK,
        "RERANK is not set at this start: there is no value for {rerankModel}.",
    ]


def test_an_id_that_cannot_ride_in_an_environment_variable_is_passed_over(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.models = [
        listed("two\nlines", "embeddings"),
        listed("nul\x00inside", "embeddings"),
        listed("x" * 300, "embeddings"),
        listed("", "embeddings"),
        listed("good", "embeddings"),
    ]
    env, _notes = _start(_spec_for(tmp_path, gateway.url))
    assert env["EMBED"] == "good"


def test_a_listing_with_junk_in_it_is_read_for_what_it_holds(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """A start is never the thing that crashes: a model that is not an object, or
    has no `x_eugene_plexus`, or whose surfaces are not a list, is no answer."""
    local = {"surfaces": ["embeddings"], "locality": "local"}
    gateway.models = [
        "junk",  # type: ignore[list-item]
        None,  # type: ignore[list-item]
        {"id": 5, "x_eugene_plexus": local},
        {"id": "no-info"},
        {"id": "info-is-text", "x_eugene_plexus": "text"},
        {"id": "surfaces-are-text", "x_eugene_plexus": {**local, "surfaces": "embeddings"}},
        listed("good", "embeddings"),
    ]
    env, _notes = _start(_spec_for(tmp_path, gateway.url))
    assert env["EMBED"] == "good"


def test_the_search_model_and_the_sorter_are_chosen_independently(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.models = [listed("embed-a", "embeddings")]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert env["EMBED"] == "embed-a" and "RERANK" not in env
    assert UNSET_RERANK in notes and UNSET_EMBED not in notes

    gateway.models = [listed("rerank-a", "rerank")]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert env["RERANK"] == "rerank-a" and "EMBED" not in env
    assert UNSET_EMBED in notes and UNSET_RERANK not in notes


def test_no_local_model_leaves_the_variables_unset_and_says_so(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.models = [listed(CHAT, "chat")]
    env, notes = _start(_spec_for(tmp_path, gateway.url), {"EMBED": "inherited"})
    assert "EMBED" not in env and "RERANK" not in env
    # Why, per model, and then which variable is left out, as for any placeholder.
    assert notes == [
        UNSET_EMBED,
        UNSET_RERANK,
        "EMBED is not set at this start: there is no value for {embeddingModel}.",
        "RERANK is not set at this start: there is no value for {rerankModel}.",
    ]


def test_a_gateway_that_lists_nothing_is_no_local_model(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.models = []
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert "EMBED" not in env and UNSET_EMBED in notes and UNSET_RERANK in notes


def test_a_model_written_into_the_spec_is_not_taken_on_its_word(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """The agent writes the spec before the app starts; the model that ran
    then may not run now. Only the gateway's answer at the start counts."""
    spec = _spec_for(tmp_path, gateway.url)
    spec["values"].update({"embeddingModel": "from-the-spec", "rerankModel": "this-too"})
    env, notes = _start(spec)
    assert "EMBED" not in env and "RERANK" not in env
    assert "from-the-spec" not in json.dumps([env, notes])


# --------------------------------------------------------------------------- #
# when the gateway cannot say
# --------------------------------------------------------------------------- #


def test_a_gateway_that_is_down_is_named_as_it_was_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows takes about two seconds to refuse a connection to a port nothing
    # holds; a machine slower than that must still be told "refused", not "slow".
    monkeypatch.setattr(app_launcher, "MODELS_TIMEOUT_SECONDS", 30.0)
    url = f"http://127.0.0.1:{_free_port()}"  # nothing listens here
    env, notes = _start(_spec_for(tmp_path, url), {"EMBED": "inherited"})
    assert "EMBED" not in env and "RERANK" not in env
    first = notes[0]
    assert first.startswith(
        "{embeddingModel} has no value at this start: the launcher could not ask which "
        f"search model runs, because the gateway at {url} could not be reached ("
    )
    assert "refused" in first
    assert any(n.startswith("{rerankModel} has no value") for n in notes)
    assert "EMBED is not set at this start: there is no value for {embeddingModel}." in notes


def test_a_gateway_that_does_not_answer_in_time_is_named(
    tmp_path: Path, gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_launcher, "MODELS_TIMEOUT_SECONDS", 0.3)
    gateway.models = [listed("embed-a", "embeddings")]
    gateway.mode, gateway.delay = "slow", 30.0
    began = time.perf_counter()
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert time.perf_counter() - began < 5
    assert "EMBED" not in env
    assert (
        "{embeddingModel} has no value at this start: the launcher could not ask which search "
        f"model runs, because the gateway at {gateway.url} did not answer within 0.3 seconds."
    ) in notes


def test_the_whole_wait_is_bounded_not_each_read(
    tmp_path: Path, gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "A few seconds in all": an answer that keeps arriving, a little at a time,
    never trips a socket's timeout and would hold the start for as long as it likes."""
    monkeypatch.setattr(app_launcher, "MODELS_TIMEOUT_SECONDS", 0.5)
    gateway.models = [listed(f"embed-{n}", "embeddings") for n in range(8)]
    gateway.mode, gateway.chunk, gateway.interval = "drip", 16, 0.05
    began = time.perf_counter()
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert time.perf_counter() - began < 3
    assert "EMBED" not in env
    assert any("did not answer within 0.5 seconds" in n for n in notes)


def test_a_key_the_gateway_refuses_is_named_with_the_gateways_own_words(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.key = "some-other-key"
    gateway.models = [listed("embed-a", "embeddings")]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert "EMBED" not in env
    assert notes[0] == (
        "{embeddingModel} has no value at this start: the launcher could not ask which search "
        f"model runs, because the gateway at {gateway.url} answered HTTP 401 and did not "
        "accept this app's key; it said: Bearer token rejected: the signature does not verify."
    )
    assert KEY not in " ".join(notes)


@pytest.mark.parametrize(
    ("status", "meaning"),
    [
        (
            403,
            "answered HTTP 403 and would not list its models for this app's key; it said: slow down",
        ),
        (429, "answered HTTP 429; it said: slow down"),
        (503, "answered HTTP 503, which is a failure on its side; it said: slow down"),
    ],
)
def test_other_refusals_are_named_by_their_code(
    tmp_path: Path, gateway: FakeGateway, status: int, meaning: str
) -> None:
    gateway.mode, gateway.status, gateway.said = "problem", status, "slow down"
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert "EMBED" not in env and f"because the gateway at {gateway.url} {meaning}." in notes[0]


def test_an_address_that_is_not_the_gateways_is_named(tmp_path: Path, gateway: FakeGateway) -> None:
    env, notes = _start(_spec_for(tmp_path, gateway.url + "/not-the-gateway"))
    assert "EMBED" not in env
    assert (
        "answered HTTP 404 and has no list of models at that address (is it the gateway's?)"
        in (notes[0])
    )
    assert "nothing is served at /not-the-gateway/v1/models" in notes[0]


def test_an_answer_that_is_not_a_list_of_models_is_named(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.mode = "garbage"
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert "EMBED" not in env
    assert "answered with something that is not a list of models" in notes[0]


def test_a_gateway_that_hangs_up_is_named(tmp_path: Path, gateway: FakeGateway) -> None:
    gateway.mode = "hangup"
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert "EMBED" not in env
    assert f"because the gateway at {gateway.url} could not be read (" in notes[0]
    assert "closed connection without response" in notes[0]


def test_a_server_that_does_not_speak_http_is_named(tmp_path: Path, gateway: FakeGateway) -> None:
    gateway.mode = "not-http"
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert "EMBED" not in env
    assert f"because the gateway at {gateway.url} could not be read (" in notes[0]


def test_a_list_too_long_to_read_is_named(
    tmp_path: Path, gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_launcher, "MODELS_MAX_BYTES", 200)
    gateway.models = [listed(f"embed-{n}", "embeddings") for n in range(8)]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert "EMBED" not in env
    assert "answered with a list of models too long to read" in notes[0]


@pytest.mark.parametrize(
    ("body", "said"),
    [
        (b'{"detail": {"title": "Key revoked", "detail": "Make a new one."}}', "Make a new one."),
        (b'{"title": "Key revoked", "detail": "Make a new one."}', "Make a new one."),
        (b'{"error": {"message": "no such model"}}', "no such model"),
        (b'{"detail": "Not authenticated"}', "Not authenticated"),
        (b'{"title": "Key revoked"}', "Key revoked"),
        (b'{"detail": "a  long\\n  line"}', "a long line"),
        (b"[]", ""),
        (b'{"detail": [1, 2]}', ""),
        (b"<html>", ""),
        (b"", ""),
    ],
)
def test_what_a_refusal_said_is_read_in_every_shape_the_gateway_uses(
    body: bytes, said: str
) -> None:
    assert app_launcher._said(body) == said


def test_what_a_refusal_said_is_short() -> None:
    assert len(app_launcher._said(json.dumps({"detail": "word " * 500}).encode())) == 240


@pytest.mark.parametrize("address", ["gateway.example", "file:///etc/passwd", "ftp://host/"])
def test_an_address_that_is_not_http_is_refused_and_nothing_is_opened(
    tmp_path: Path, address: str
) -> None:
    env, notes = _start(_spec_for(tmp_path, address))
    assert "EMBED" not in env
    assert "the gateway's address does not start with http:// or https://" in notes[0]


def test_the_key_is_not_sent_where_a_redirect_points(tmp_path: Path, gateway: FakeGateway) -> None:
    elsewhere = FakeGateway([listed("embed-a", "embeddings")])  # accepts any key
    try:
        gateway.mode, gateway.location = "redirect", elsewhere.url + "/v1/models"
        env, notes = _start(_spec_for(tmp_path, gateway.url))
        assert elsewhere.requests == []
        assert "EMBED" not in env
        assert "answered HTTP 302 and sent the launcher elsewhere" in notes[0]
    finally:
        elsewhere.close()


def test_a_gateway_that_echoes_the_key_does_not_put_it_in_the_notes(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.mode = "echo"
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert "EMBED" not in env
    said = " ".join(notes)
    assert "answered HTTP 500" in said and "could not handle 'Bearer [this app's key]'" in said
    assert KEY not in said


def test_without_a_key_the_gateway_is_not_asked(tmp_path: Path, gateway: FakeGateway) -> None:
    gateway.models = [listed("embed-a", "embeddings")]
    spec = _spec_for(tmp_path, gateway.url, keyFile=str(tmp_path / "no-such-key"))
    env, notes = _start(spec)
    assert gateway.requests == [] and "EMBED" not in env
    assert (
        "this app has no key to ask the gateway with (its key file is missing or empty)"
        in (notes[0])
    )
    # An empty file is no key either.
    (tmp_path / "empty-key").write_text("\n", encoding="utf-8")
    _env, notes = _start(_spec_for(tmp_path, gateway.url, keyFile=str(tmp_path / "empty-key")))
    assert gateway.requests == [] and "no key to ask" in notes[0]


def test_without_an_address_the_gateway_is_not_asked(tmp_path: Path, gateway: FakeGateway) -> None:
    gateway.models = [listed("embed-a", "embeddings")]
    env, notes = _start(_spec_for(tmp_path, None))
    assert gateway.requests == [] and "EMBED" not in env
    assert "this start was given no address for the gateway" in notes[0]


# --------------------------------------------------------------------------- #
# how it asks
# --------------------------------------------------------------------------- #


def test_the_gateway_is_asked_once_per_start_and_only_when_a_model_is_named(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.models = [listed("embed-a", "embeddings"), listed("rerank-a", "rerank")]
    values = {
        "bindHost": "127.0.0.1",
        "port": "8190",
        "gatewayUrl": gateway.url,
        "appUrl": "http://127.0.0.1:8190",
        "oidcIssuer": "http://127.0.0.1:8079/oidc",
        "oidcClientId": "c-made",
    }
    # Every other placeholder, and no model: nothing is asked.
    app_launcher.render_start(
        _spec(tmp_path, args=["--port", "{port}"], environment=dict(EVERY), values=values), {}
    )
    assert gateway.requests == []

    # One model, named in an argument: asked.
    argv, _env, _notes = app_launcher.render_start(
        _spec(tmp_path, args=["--embedding-model", "{embeddingModel}"], values=values), {}
    )
    assert argv[-2:] == ["--embedding-model", "embed-a"] and len(gateway.requests) == 1

    # Both, in the environment: one request answers the two.
    gateway.requests.clear()
    _start(_spec_for(tmp_path, gateway.url))
    assert len(gateway.requests) == 1

    # The next start asks again: a model that came or went is met then.
    _start(_spec_for(tmp_path, gateway.url))
    assert len(gateway.requests) == 2


def test_a_model_that_changes_between_starts_is_met_at_the_next_one(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    spec = _spec_for(tmp_path, gateway.url)
    gateway.models = [listed("embed-a", "embeddings")]
    assert _start(spec)[0]["EMBED"] == "embed-a"
    gateway.models = [listed("embed-b", "embeddings"), listed("embed-a", "embeddings")]
    assert _start(spec)[0]["EMBED"] == "embed-b"
    gateway.models = []
    assert "EMBED" not in _start(spec)[0]


def test_a_start_waits_a_few_seconds_at_most() -> None:
    """A start does not hold the app back for a convenience: a few seconds, in all."""
    assert 1 <= app_launcher.MODELS_TIMEOUT_SECONDS <= 10


def test_a_gateway_address_with_a_path_is_asked_at_that_path(tmp_path: Path) -> None:
    """The address the agent hands an app can be its own proxy's: a path, and
    perhaps a trailing slash."""
    fake = FakeGateway([listed("embed-a", "embeddings")], key=KEY, prefix="/api/proxy/gateway")
    try:
        for url in (fake.url, fake.url + "/"):
            env, notes = _start(_spec_for(tmp_path, url))
            assert env["EMBED"] == "embed-a" and UNSET_RERANK in notes
        assert {r["path"] for r in fake.requests} == {"/api/proxy/gateway/v1/models"}
    finally:
        fake.close()


def test_the_gateway_is_never_asked_through_the_users_proxy(
    tmp_path: Path, gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proxy would be handed the app's key, and cannot reach this install's
    own gateway anyway."""
    dead = f"http://127.0.0.1:{_free_port()}"
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.setenv(name, dead)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    gateway.models = [listed("embed-a", "embeddings")]
    env, _notes = _start(_spec_for(tmp_path, gateway.url))
    assert env["EMBED"] == "embed-a" and len(gateway.requests) == 1


# --------------------------------------------------------------------------- #
# a change of model is not a change of connection
# --------------------------------------------------------------------------- #


def test_a_change_of_model_is_not_a_change_of_connection(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    spec = _spec_for(
        tmp_path,
        gateway.url,
        environment={"EMBED": "{embeddingModel}"},
        resetOnConnectionChange="RESET_CONFIG_ON_START",
    )
    reset = "RESET_CONFIG_ON_START"
    gateway.models = [listed("embed-a", "embeddings")]
    env, notes = _start(spec)
    assert env["EMBED"] == "embed-a" and reset not in env and notes == []

    # A different model comes first: the app is handed it, and nothing is reset.
    gateway.models = [listed("embed-b", "embeddings"), listed("embed-a", "embeddings")]
    env, notes = _start(spec)
    assert env["EMBED"] == "embed-b" and reset not in env and notes == []

    # None runs: the variable is left unset, and still nothing is reset.
    gateway.models = []
    env, notes = _start(spec)
    assert "EMBED" not in env and reset not in env
    assert not any("changed since the last start" in n for n in notes)

    # What was kept is the gateway's address and the app's key and nothing else.
    kept = (tmp_path / "data" / app_launcher.CONNECTION_FILE).read_text(encoding="utf-8")
    assert kept.strip() == hashlib.sha256(f"{gateway.url}\n{KEY}".encode()).hexdigest()

    # A rotated key is still a change of connection, and still resets.
    (tmp_path / "client_key").write_text("a-new-key", encoding="utf-8")
    gateway.key = "a-new-key"
    gateway.models = [listed("embed-a", "embeddings")]
    env, notes = _start(spec)
    assert env[reset] == "true" and env["EMBED"] == "embed-a"
    assert any("changed since the last start" in n for n in notes)


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #


def test_the_two_model_placeholders_are_placeholders() -> None:
    assert set(app_launcher.MODEL_SURFACES) <= app_launcher.PLACEHOLDERS
    assert app_launcher.MODEL_SURFACES == {"embeddingModel": "embeddings", "rerankModel": "rerank"}
    # Neither is a secret: a model's id may ride in an argument.
    assert not set(app_launcher.MODEL_SURFACES) & app_launcher.SECRET_PLACEHOLDERS
    apps.validate_manifest(
        _manifest(
            environment={"EMBED": "{embeddingModel}", "RERANK": "{rerankModel}"},
            args=["--embedding-model", "{embeddingModel}", "--sorter", "{rerankModel}"],
        )
    )
    # And a refusal of a placeholder that is not one lists them.
    with pytest.raises(ValueError, match=r"\{embeddingModel\}.*\{rerankModel\}"):
        apps.validate_manifest(_manifest(environment={"X": "{nope}"}))


# --------------------------------------------------------------------------- #
# the catalogue's Open WebUI, and the two ways an app starts
# --------------------------------------------------------------------------- #


def _open_webui(tmp_path: Path, url: str) -> apps._AppPlanner:
    (webui,) = [m for m in apps.load_catalogue() if m.id == "open-webui"]
    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    store.data_dir("open-webui").mkdir(parents=True)
    store.key_file("open-webui").write_text(KEY, encoding="utf-8")
    store.oidc_secret_file("open-webui").write_text("the-oidc-secret", encoding="utf-8")
    store.put_oidc_client("open-webui", "c-webui")
    record = apps.InstalledApp(
        manifest=webui, origin=AppOrigin.catalogue, port=8191, installed_at=datetime.now(UTC)
    )
    return _planner(store, record, gateway=url)


def test_open_webui_is_seeded_with_document_search_through_eugene() -> None:
    (webui,) = [m for m in apps.load_catalogue() if m.id == "open-webui"]
    env = webui.environment or {}
    assert {k: v for k, v in env.items() if k.startswith(("RAG_", "ENABLE_RAG_"))} == {
        "RAG_EMBEDDING_ENGINE": "openai",
        "RAG_EMBEDDING_MODEL": "{embeddingModel}",
        "RAG_RERANKING_ENGINE": "external",
        "RAG_RERANKING_MODEL": "{rerankModel}",
        "RAG_EXTERNAL_RERANKER_URL": "{gatewayUrl}/v1/rerank",
        "RAG_EXTERNAL_RERANKER_API_KEY": "{clientKey}",
        "ENABLE_RAG_HYBRID_SEARCH": "true",
    }


def test_the_agent_writes_no_model_into_the_spec(tmp_path: Path, gateway: FakeGateway) -> None:
    launch = _open_webui(tmp_path, gateway.url).launch()
    assert "embeddingModel" not in launch["values"] and "rerankModel" not in launch["values"]
    assert launch["values"]["gatewayUrl"] == gateway.url
    assert launch["environment"]["RAG_EMBEDDING_MODEL"] == "{embeddingModel}"
    assert launch["environment"]["RAG_RERANKING_MODEL"] == "{rerankModel}"


def test_open_webui_is_given_every_search_variable_when_both_models_run(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.models = [
        listed(CHAT, "chat"),
        listed("Qwen3-Embedding-0.6B-Q8_0", "embeddings"),
        listed("bge-reranker-v2-m3-Q8_0", "rerank"),
    ]
    plan = _open_webui(tmp_path, gateway.url).plan()
    env = plan.env
    assert env["RAG_EMBEDDING_ENGINE"] == "openai"
    assert env["RAG_EMBEDDING_MODEL"] == "Qwen3-Embedding-0.6B-Q8_0"
    assert env["RAG_RERANKING_ENGINE"] == "external"
    assert env["RAG_RERANKING_MODEL"] == "bge-reranker-v2-m3-Q8_0"
    assert env["RAG_EXTERNAL_RERANKER_URL"] == f"{gateway.url}/v1/rerank"
    assert env["RAG_EXTERNAL_RERANKER_API_KEY"] == KEY
    assert env["ENABLE_RAG_HYBRID_SEARCH"] == "true"
    # What the app already had is as it was, and the key rode no argument.
    assert env["OPENAI_API_BASE_URL"] == f"{gateway.url}/v1" and env["OPENAI_API_KEY"] == KEY
    assert not any(KEY in a for a in plan.argv)
    assert len(gateway.requests) == 1


def test_open_webui_leaves_the_two_model_variables_unset_when_none_runs(
    tmp_path: Path, gateway: FakeGateway, caplog: pytest.LogCaptureFixture
) -> None:
    gateway.models = [listed(CHAT, "chat"), listed(OUTSIDE, "embeddings", locality="external")]
    with caplog.at_level(logging.WARNING):
        env = _open_webui(tmp_path, gateway.url).plan().env
    assert "RAG_EMBEDDING_MODEL" not in env and "RAG_RERANKING_MODEL" not in env
    # The rest of its search is set up all the same: it falls back to words and
    # its own order of results, and its admin can still name a model.
    assert env["RAG_EMBEDDING_ENGINE"] == "openai" and env["RAG_RERANKING_ENGINE"] == "external"
    assert env["RAG_EXTERNAL_RERANKER_URL"] == f"{gateway.url}/v1/rerank"
    assert env["RAG_EXTERNAL_RERANKER_API_KEY"] == KEY and env["ENABLE_RAG_HYBRID_SEARCH"] == "true"
    # The agent's log says which, and why.
    assert "no local search model is running" in caplog.text
    assert OUTSIDE in caplog.text and "no local result sorter is running" in caplog.text
    assert "RAG_EMBEDDING_MODEL is not set at this start" in caplog.text
    assert KEY not in caplog.text


def test_the_agents_own_supervisor_says_why_a_gateway_would_not_answer(
    tmp_path: Path, gateway: FakeGateway, caplog: pytest.LogCaptureFixture
) -> None:
    gateway.key = "some-other-key"
    with caplog.at_level(logging.DEBUG):
        env = _open_webui(tmp_path, gateway.url).plan().env
    assert "RAG_EMBEDDING_MODEL" not in env
    assert "answered HTTP 401 and did not accept this app's key" in caplog.text
    assert KEY not in caplog.text


def test_the_launcher_hands_the_app_its_models_and_says_why_when_it_cannot(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """The own-account path end to end: the launcher asks, starts the app with
    the answer, and the reason reaches the Logs page through its forwarding."""
    packages = _fake_package(tmp_path)
    ingress = _Ingress()
    data = tmp_path / "data"
    data.mkdir()
    key = tmp_path / "client_key"
    key.write_text(KEY, encoding="utf-8")
    spec: dict[str, Any] = {
        "app": "fake",
        "argv": apps.entry_argv(Path(sys.executable), _manifest()),
        "args": [],
        "environment": {"DATA": "{dataDir}", **BOTH},
        "values": {"gatewayUrl": gateway.url},
        "env": {"PYTHONPATH": str(packages)},
        "dataDir": str(data),
        "keyFile": str(key),
        "ingress": ingress.url,
    }
    gateway.models = [listed("embed-a", "embeddings"), listed("rerank-a", "rerank")]
    app_launcher._STOP.clear()
    try:
        assert app_launcher.run(spec) == 0
        record = json.loads((data / "record.json").read_text(encoding="utf-8"))
        assert record["env"]["EMBED"] == "embed-a" and record["env"]["RERANK"] == "rerank-a"

        # The gateway stops accepting the app's key: nothing is set, and it says so.
        gateway.key = "rotated"
        assert app_launcher.run(spec) == 0
        record = json.loads((data / "record.json").read_text(encoding="utf-8"))
        assert record["env"]["EMBED"] is None and record["env"]["RERANK"] is None
        deadline = time.perf_counter() + 10
        while not any("answered HTTP 401" in line for line in ingress.lines):
            assert time.perf_counter() < deadline, ingress.lines
            time.sleep(0.1)
        said = [line for line in ingress.lines if "{embeddingModel} has no value" in line]
        assert said and said[0].startswith("[launcher] ")
        assert not any(KEY in line for line in ingress.lines)
    finally:
        ingress.server.shutdown()
