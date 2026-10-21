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
import threading
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
# a model that is listed and stopped
# --------------------------------------------------------------------------- #


def test_a_running_model_is_chosen_before_a_stopped_one(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """The gateway keeps listing a model whose runtime was stopped, with no
    backend ready: the one that is running is the better answer, wherever it is."""
    gateway.models = [
        listed("embed-stopped", "embeddings", ready=0),
        listed("embed-running", "embeddings", ready=1),
        listed("rerank-stopped", "rerank", ready=0),
        listed("rerank-running", "rerank", ready=2),
    ]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert (env["EMBED"], env["RERANK"]) == ("embed-running", "rerank-running")
    assert notes == []


def test_a_stopped_model_is_chosen_when_it_is_all_there_is_and_the_notes_say_so(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """An app that keeps what its first start saw is better off with the install's
    model than with its own default, so a model that is listed is the answer even
    when nothing runs it now. The first listed, and the note says it is not running."""
    gateway.models = [
        listed("embed-a", "embeddings", ready=0),
        listed("embed-b", "embeddings", ready=0),
        listed("rerank-a", "rerank"),
    ]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert env["EMBED"] == "embed-a" and env["RERANK"] == "rerank-a"
    # One line, about the model that is not running, and nothing about the one that is.
    (said,) = notes
    assert said.startswith("{embeddingModel} is embed-a at this start, ")
    assert "the gateway lists it but it is not running now" in said
    assert "no other local search model is" in said
    assert "rather than its own default" in said and "under Backends" in said


def test_a_stopped_model_is_said_for_each_of_the_two_on_its_own(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.models = [listed("embed-a", "embeddings"), listed("rerank-a", "rerank", ready=0)]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert (env["EMBED"], env["RERANK"]) == ("embed-a", "rerank-a")
    (said,) = notes
    assert said.startswith("{rerankModel} is rerank-a at this start, ")
    assert "no other local result sorter is" in said


@pytest.mark.parametrize(
    "ready",
    [
        pytest.param(None, id="a-gateway-older-than-the-field"),
        pytest.param("lots", id="text"),
        pytest.param(True, id="a-boolean"),
    ],
)
def test_a_model_that_does_not_say_it_is_stopped_is_taken_as_running(
    tmp_path: Path, gateway: FakeGateway, ready: Any
) -> None:
    gateway.models = [listed("embed-a", "embeddings", ready=ready)]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert env["EMBED"] == "embed-a"
    assert not any("not running now" in n for n in notes)


def test_an_outside_model_that_runs_does_not_displace_a_local_one_that_does_not(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    gateway.models = [
        listed(OUTSIDE, "embeddings", locality="external", owned_by="openrouter", ready=3),
        listed("embed-local", "embeddings", ready=0),
    ]
    env, notes = _start(_spec_for(tmp_path, gateway.url))
    assert env["EMBED"] == "embed-local"
    assert any("not running now" in n for n in notes)


# --------------------------------------------------------------------------- #
# when the gateway cannot say
# --------------------------------------------------------------------------- #


def test_a_gateway_that_never_comes_up_is_named_as_it_was_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows takes about two seconds to say a connection to a port nothing holds
    # is refused, so the time given has to be longer than that to see it.
    monkeypatch.setattr(app_launcher, "MODELS_TIMEOUT_SECONDS", 2.5)
    monkeypatch.setattr(app_launcher, "MODELS_RETRY_SECONDS", 0.1)
    url = f"http://127.0.0.1:{_free_port()}"  # nothing listens here
    began = time.perf_counter()
    env, notes = _start(_spec_for(tmp_path, url), {"EMBED": "inherited"})
    assert time.perf_counter() - began < 6
    assert "EMBED" not in env and "RERANK" not in env
    first = notes[0]
    assert first.startswith(
        "{embeddingModel} has no value at this start: the launcher could not ask which "
        f"search model runs, because the gateway at {url} refused the connection ("
    )
    assert "refused" in first and first.endswith(
        "; it had not come up after 2.5 seconds of trying."
    )
    assert any(n.startswith("{rerankModel} has no value") for n in notes)
    assert "EMBED is not set at this start: there is no value for {embeddingModel}." in notes


def test_a_gateway_that_is_not_up_yet_is_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent has only just started and its gateway is a moment behind. An app
    that keeps what its first start saw must not be given the less for it."""
    monkeypatch.setattr(app_launcher, "MODELS_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(app_launcher, "MODELS_RETRY_SECONDS", 0.2)
    port = _free_port()
    up: list[FakeGateway] = []

    def come_up() -> None:
        up.append(FakeGateway([listed("embed-a", "embeddings")], key=KEY, port=port))

    # Windows retries a refused connection by itself for about two seconds and
    # would reach a gateway that came up inside them without any help from
    # here, so this one comes up after.
    timer = threading.Timer(2.6, come_up)
    timer.start()
    began = time.perf_counter()
    try:
        env, notes = _start(_spec_for(tmp_path, f"http://127.0.0.1:{port}"))
    finally:
        timer.cancel()
        waited = time.perf_counter() - began
        for fake in up:
            fake.close()
    assert env.get("EMBED") == "embed-a", notes
    assert 2.5 <= waited < 8
    # Only the try that got through reached it: a refused one reaches nobody.
    assert len(up) == 1 and len(up[0].requests) == 1


@pytest.mark.parametrize("mode", ["problem", "garbage", "hangup", "wrong-key"])
def test_only_a_refused_connection_is_tried_again(
    tmp_path: Path, gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A gateway that is there and says no, or something else, has answered: asking
    again would only be asking a thing that said no."""
    monkeypatch.setattr(app_launcher, "MODELS_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(app_launcher, "MODELS_RETRY_SECONDS", 0.05)
    gateway.models = [listed("embed-a", "embeddings")]
    if mode == "wrong-key":
        gateway.key = "some-other-key"
    else:
        gateway.mode, gateway.status = mode, 503
    env, _notes = _start(_spec_for(tmp_path, gateway.url))
    assert "EMBED" not in env and len(gateway.requests) == 1


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
    spec = _reset_spec(tmp_path, gateway.url)
    gateway.models = [listed("embed-a", "embeddings")]
    env, notes = _start(spec)
    assert env["EMBED"] == "embed-a" and RESET not in env and notes == []

    # A different model comes first: the app is handed it, and nothing is reset.
    gateway.models = [listed("embed-b", "embeddings"), listed("embed-a", "embeddings")]
    env, notes = _start(spec)
    assert env["EMBED"] == "embed-b" and RESET not in env and notes == []

    # None runs: the variable is left unset, and still nothing is reset.
    gateway.models = []
    env, notes = _start(spec)
    assert "EMBED" not in env and RESET not in env
    assert not any(CHANGED in n for n in notes)

    # The key is not in what was kept.
    kept = (tmp_path / "data" / app_launcher.CONNECTION_FILE).read_text(encoding="utf-8")
    assert KEY not in kept and gateway.url not in kept

    # A rotated key is still a change, and still resets.
    (tmp_path / "client_key").write_text("a-new-key", encoding="utf-8")
    gateway.key = "a-new-key"
    gateway.models = [listed("embed-a", "embeddings")]
    env, notes = _start(spec)
    assert env[RESET] == "true" and env["EMBED"] == "embed-a"
    assert any(CHANGED in n for n in notes)


RESET = "RESET_CONFIG_ON_START"
#: Part of the line that says why a start reset.
CHANGED = "changed since the last start"
#: What Eugene gives the app, as a manifest writes it: unfilled.
GIVEN = {"EMBED": "{embeddingModel}", "BASE": "{gatewayUrl}/v1", "SETTING": "one"}


def _reset_spec(
    tmp_path: Path, url: str, environment: dict[str, str] | None = None
) -> dict[str, Any]:
    return _spec_for(
        tmp_path,
        url,
        environment=dict(GIVEN if environment is None else environment),
        resetOnConnectionChange=RESET,
    )


def test_a_changed_setting_resets_once_and_says_what_changed(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """Open WebUI keeps what its first start saw and ignores its environment after,
    so a setting Eugene corrects reaches an install that already started only
    through a reset, which clears what its admin set: once, and said."""
    gateway.models = [listed("embed-a", "embeddings")]
    env, notes = _start(_reset_spec(tmp_path, gateway.url))
    assert RESET not in env and notes == []

    corrected = dict(GIVEN, SETTING="two")
    env, notes = _start(_reset_spec(tmp_path, gateway.url, corrected))
    assert env[RESET] == "true"
    (said,) = notes
    assert CHANGED in said and f"{RESET}=true" in said
    assert "the gateway's address, the app's key, or the settings Eugene gives it" in said

    # Once: the same settings again are not a change.
    env, notes = _start(_reset_spec(tmp_path, gateway.url, corrected))
    assert RESET not in env and notes == []


@pytest.mark.parametrize(
    "corrected",
    [
        pytest.param({**GIVEN, "SETTING": "two"}, id="a-value-corrected"),
        pytest.param({**GIVEN, "ADDED": "x"}, id="a-setting-added"),
        pytest.param({k: v for k, v in GIVEN.items() if k != "SETTING"}, id="a-setting-removed"),
        pytest.param({**GIVEN, "EMBED": "{rerankModel}"}, id="a-placeholder-changed"),
    ],
)
def test_any_change_to_the_settings_as_written_is_a_change(
    tmp_path: Path, gateway: FakeGateway, corrected: dict[str, str]
) -> None:
    _start(_reset_spec(tmp_path, gateway.url))
    env, _notes = _start(_reset_spec(tmp_path, gateway.url, corrected))
    assert env[RESET] == "true"


def test_the_same_settings_in_another_order_are_not_a_change(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """The digest holds them in a stable order: a manifest that lists the same
    variables differently has changed nothing Eugene gives the app."""
    gateway.models = [listed("embed-a", "embeddings")]
    _start(_reset_spec(tmp_path, gateway.url))
    shuffled = dict(reversed(list(GIVEN.items())))
    assert list(shuffled) != list(GIVEN)
    env, notes = _start(_reset_spec(tmp_path, gateway.url, shuffled))
    assert RESET not in env and notes == []


def test_the_same_settings_are_not_a_change(tmp_path: Path, gateway: FakeGateway) -> None:
    gateway.models = [listed("embed-a", "embeddings")]
    for _ in range(3):
        env, notes = _start(_reset_spec(tmp_path, gateway.url))
        assert RESET not in env and notes == []


def test_a_digest_from_before_the_settings_joined_it_counts_as_a_change(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """A release before this kept the gateway's address and the app's key alone. An
    install that started under it has Open WebUI's old defaults in its database, and
    the first start after is the one that must correct them."""
    gateway.models = [listed("embed-a", "embeddings")]
    (tmp_path / "data").mkdir()
    old = hashlib.sha256(f"{gateway.url}\n{KEY}".encode()).hexdigest()
    (tmp_path / "data" / app_launcher.CONNECTION_FILE).write_text(old, encoding="utf-8")
    env, notes = _start(_reset_spec(tmp_path, gateway.url))
    assert env[RESET] == "true" and any(CHANGED in n for n in notes)
    # And once.
    env, notes = _start(_reset_spec(tmp_path, gateway.url))
    assert RESET not in env and notes == []


def test_a_change_of_address_or_key_still_resets(tmp_path: Path, gateway: FakeGateway) -> None:
    _start(_reset_spec(tmp_path, gateway.url))
    other = FakeGateway(key=KEY)
    try:
        env, _notes = _start(_reset_spec(tmp_path, other.url))
        assert env[RESET] == "true"
    finally:
        other.close()


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #


def test_the_two_model_placeholders_are_placeholders() -> None:
    assert set(app_launcher.MODEL_SURFACES) <= app_launcher.PLACEHOLDERS
    assert app_launcher.MODEL_SURFACES == {
        "embeddingModel": "embeddings",
        "rerankModel": "rerank",
        "imageModel": "image",
    }
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
    search = (
        "RAG_EMBEDDING_ENGINE",
        "RAG_EMBEDDING_MODEL",
        "RAG_RERANKING_ENGINE",
        "RAG_RERANKING_MODEL",
        "RAG_EXTERNAL_RERANKER_URL",
        "RAG_EXTERNAL_RERANKER_API_KEY",
        "ENABLE_RAG_HYBRID_SEARCH",
    )
    assert {k: env.get(k) for k in search} == {
        "RAG_EMBEDDING_ENGINE": "openai",
        "RAG_EMBEDDING_MODEL": "{embeddingModel}",
        "RAG_RERANKING_ENGINE": "external",
        "RAG_RERANKING_MODEL": "{rerankModel}",
        "RAG_EXTERNAL_RERANKER_URL": "{gatewayUrl}/v1/rerank",
        "RAG_EXTERNAL_RERANKER_API_KEY": "{clientKey}",
        "ENABLE_RAG_HYBRID_SEARCH": "true",
    }


def test_open_webui_sends_its_embeddings_in_batches_one_request_at_a_time(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """Open WebUI's defaults are one passage to a request, every request of a file
    together, and the first 429 fails the whole file (config.py lines 1019 and 1025;
    retrieval/utils.py lines 1165-1181 and 929-955). An app's key has limits, and a
    10-passage file was served 2 of 10 (MR6's acceptance). 32 passages to a request,
    one request at a time, is about 32 times fewer requests and none together."""
    (webui,) = [m for m in apps.load_catalogue() if m.id == "open-webui"]
    env = webui.environment or {}
    assert env.get("RAG_EMBEDDING_BATCH_SIZE") == "32"
    assert env.get("RAG_EMBEDDING_CONCURRENT_REQUESTS") == "1"
    # Rendered as it is started, and read by Open WebUI as integers: a batch of
    # several, and a limit of one at a time (0, its default, is no limit).
    rendered = _open_webui(tmp_path, gateway.url).plan().env
    assert rendered["RAG_EMBEDDING_BATCH_SIZE"] == "32"
    assert rendered["RAG_EMBEDDING_CONCURRENT_REQUESTS"] == "1"


#: Every address Open WebUI 0.11.4's config.py defaults to OpenAI's after resetting
#: `OPENAI_API_BASE_URL` to https://api.openai.com/v1 (lines 362-367): its own, and
#: the five features that read it after (lines 1098, 1505, 1535, 1580, 1621).
OPENAI_ADDRESSES = (
    "OPENAI_API_BASE_URL",
    "RAG_OPENAI_API_BASE_URL",
    "IMAGES_OPENAI_API_BASE_URL",
    "IMAGES_EDIT_OPENAI_API_BASE_URL",
    "AUDIO_STT_OPENAI_API_BASE_URL",
    "AUDIO_TTS_OPENAI_API_BASE_URL",
)
OPENAI_KEYS = tuple(a.replace("BASE_URL", "KEY") for a in OPENAI_ADDRESSES)


def test_open_webui_names_the_gateway_for_every_address_it_would_default_to_openai() -> None:
    (webui,) = [m for m in apps.load_catalogue() if m.id == "open-webui"]
    env = webui.environment or {}
    assert {name: env.get(name) for name in OPENAI_ADDRESSES} == dict.fromkeys(
        OPENAI_ADDRESSES, "{gatewayUrl}/v1"
    )
    assert {name: env.get(name) for name in OPENAI_KEYS} == dict.fromkeys(
        OPENAI_KEYS, "{clientKey}"
    )


def test_nothing_open_webui_is_started_with_is_openais_address(
    tmp_path: Path, gateway: FakeGateway
) -> None:
    """Since C4 an uploaded document's text was sent to https://api.openai.com/v1/embeddings
    (refused there, for want of a key): Open WebUI resets `OPENAI_API_BASE_URL` to that after
    building its connections, and an address it is not given defaults to the reset one. Rendered
    as it is started, no value of the entry may be OpenAI's, and every OpenAI address is the
    gateway's, whichever models run."""
    for run, models in enumerate(
        ([], [listed("embed-a", "embeddings"), listed("rerank-a", "rerank")])
    ):
        gateway.models = models
        (tmp_path / f"run{run}").mkdir()
        plan = _open_webui(tmp_path / f"run{run}", gateway.url).plan()
        (webui,) = [m for m in apps.load_catalogue() if m.id == "open-webui"]
        rendered = {k: plan.env[k] for k in (webui.environment or {}) if k in plan.env}
        assert rendered, "the entry rendered nothing"
        assert not [k for k, v in rendered.items() if "api.openai.com" in v]
        assert not any("api.openai.com" in a for a in plan.argv)
        addresses = {
            k: v
            for k, v in rendered.items()
            if k == "OPENAI_API_BASE_URL" or k.endswith("_OPENAI_API_BASE_URL")
        }
        assert set(addresses) == set(OPENAI_ADDRESSES)
        assert set(addresses.values()) == {f"{gateway.url}/v1"}
        keys = {
            k: v
            for k, v in rendered.items()
            if k == "OPENAI_API_KEY" or k.endswith("_OPENAI_API_KEY")
        }
        assert set(keys) == set(OPENAI_KEYS) and set(keys.values()) == {KEY}


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
