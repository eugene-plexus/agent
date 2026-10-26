"""Standalone limits use the same atomic registry and survive restart."""

import json
from dataclasses import replace

import pytest

from eugene_plexus_agent import _private_files
from eugene_plexus_agent.client_admission import AdmissionRefusal, validate_limits
from eugene_plexus_agent.client_keys import ClientKeyStore
from tests.test_client_keys import _record


def test_standalone_restart_and_failed_writes(tmp_path, monkeypatch):
    path = tmp_path / "keys.json"
    store = ClientKeyStore(path)
    store.add(replace(_record(), limits={"maxConcurrentRequests": 1, "requestsPerMinute": 2}))
    store.admit(key_id="abc", action="acquire", request_id="one", model="m")
    store = ClientKeyStore(path)
    store.load()
    with pytest.raises(AdmissionRefusal) as caught:
        store.admit(key_id="abc", action="acquire", request_id="two", model="m")
    assert caught.value.status == 429
    before = path.read_bytes()
    with monkeypatch.context() as patch:

        def fail(*args):
            raise OSError("full")

        # The replace lives in the shared private-file writer since 2026-09-22.
        patch.setattr(_private_files.os, "replace", fail)
        with pytest.raises(OSError):
            store.admit(key_id="abc", action="release", request_id="one", model="m")
        assert path.read_bytes() == before
    with pytest.raises(AdmissionRefusal):
        store.admit(key_id="abc", action="acquire", request_id="two", model="m")
    store.admit(key_id="abc", action="release", request_id="one", model="m")
    store.admit(key_id="abc", action="acquire", request_id="two", model="m")
    store.admit(key_id="abc", action="release", request_id="two", model="m")
    with pytest.raises(AdmissionRefusal) as caught:
        store.admit(key_id="abc", action="acquire", request_id="three", model="m")
    assert caught.value.status == 429


def test_legacy_remains_explicitly_unrestricted_until_edited(tmp_path):
    store = ClientKeyStore(tmp_path / "keys.json")
    store.add(_record())
    assert store.records()[0].limits is None
    for i in range(5):
        store.admit(key_id="abc", action="acquire", request_id=str(i), model="m")
    store.set_limits("abc", {"allowedModels": []})
    with pytest.raises(AdmissionRefusal) as caught:
        store.admit(key_id="abc", action="acquire", request_id="blocked", model="m")
    assert caught.value.status == 403
    with pytest.raises(AdmissionRefusal) as caught:
        store.admit(key_id="abc", action="renew", request_id="0", model="m")
    assert caught.value.status == 409


def test_local_only_survives_restart_and_changes_invalidate_existing_lease(tmp_path):
    path = tmp_path / "keys.json"
    store = ClientKeyStore(path)
    store.add(replace(_record(), limits={"localOnly": True}))
    store = ClientKeyStore(path)
    store.load()
    result = store.admit(key_id="abc", action="acquire", request_id="one", model="m")
    assert result["limits"]["localOnly"] is True
    store.set_limits("abc", {"localOnly": False})
    with pytest.raises(AdmissionRefusal) as caught:
        store.admit(key_id="abc", action="renew", request_id="one", model="m")
    assert caught.value.status == 409


def test_standalone_http_limits_and_operator_edit(authed_client):
    c = authed_client
    made = c.post("/v1/auth/client-keys", json={"name": "test"}).json()
    assert made["key"]["limits"]["maxConcurrentRequests"] == 2
    key = made["key"]["id"]
    path = f"/v1/auth/client-keys/{key}/limits"
    assert c.put(path, json={"limits": {"allowedModels": []}}).status_code == 200
    payload = {"action": "acquire", "keyId": key, "requestId": "one", "model": "m"}
    assert c.post("/v1/auth/client-keys/admission", json=payload).status_code == 403
    headers = {"Authorization": "Bearer " + made["token"]}
    assert (
        c.post("/v1/auth/client-keys/admission", json=payload, headers=headers).status_code == 401
    )
    assert c.put(path, json={"limits": {}}, headers=headers).status_code == 401
    proof = {**payload, "personToken": "a-person-proof"}
    assert c.post("/v1/auth/client-keys/admission", json=proof).status_code == 503
    assert (
        c.post(
            "/v1/auth/client-keys", json={"name": "bad", "limits": {"allowedModels": [" "]}}
        ).status_code
        == 422
    )


def test_an_account_pattern_admits_its_models_and_nothing_else(tmp_path):
    """P1: `openrouter/*` scopes a key to one provider account; `*` is the
    only wildcard and crosses `/`."""
    store = ClientKeyStore(tmp_path / "keys.json")
    store.add(replace(_record(), limits={"allowedModels": ["openrouter/*", "qwen3-8b"]}))
    store.admit(key_id="abc", action="acquire", request_id="a", model="openrouter/anthropic/x")
    store.admit(key_id="abc", action="acquire", request_id="b", model="qwen3-8b")
    for i, model in enumerate(("work/anthropic/x", "qwen3-8b-instruct", "openrouter")):
        with pytest.raises(AdmissionRefusal) as caught:
            store.admit(key_id="abc", action="acquire", request_id=f"no{i}", model=model)
        assert caught.value.status == 403, model


def test_only_star_is_special() -> None:
    from eugene_plexus_agent.client_admission import model_permitted

    assert model_permitted(["a?b"], "a?b") and not model_permitted(["a?b"], "axb")
    assert model_permitted(["*:free"], "openrouter/respan/span-01-lite:free")
    assert not model_permitted([], "anything")


def test_a_keys_tool_scope_survives_a_restart_and_reaches_the_gateway(tmp_path):
    """P8: `allowedTools` rides on the admission answer the gateway reads.
    Absent (every key minted before P8) means every tool; a list narrows."""
    path = tmp_path / "keys.json"
    store = ClientKeyStore(path)
    store.add(replace(_record(), limits={"allowedTools": []}))
    store = ClientKeyStore(path)
    store.load()
    result = store.admit(key_id="abc", action="acquire", request_id="one", model="m")
    assert result["limits"]["allowedTools"] == []
    store.set_limits("abc", {"allowedModels": None})
    result = store.admit(key_id="abc", action="acquire", request_id="two", model="m")
    assert "allowedTools" not in result["limits"]


def test_a_bad_tool_scope_is_a_client_error(authed_client):
    for tools in ([" "], ["web_search", "web_search"], "web_search", ["x" * 65]):
        response = authed_client.post(
            "/v1/auth/client-keys", json={"name": "bad", "limits": {"allowedTools": tools}}
        )
        assert response.status_code == 422, (tools, response.text)
    made = authed_client.post(
        "/v1/auth/client-keys", json={"name": "ok", "limits": {"allowedTools": ["web_search"]}}
    )
    assert made.status_code in (200, 201), made.text
    assert made.json()["key"]["limits"]["allowedTools"] == ["web_search"]


def test_write_logs_rides_on_the_admission_answer_and_is_off_by_default(tmp_path):
    """C1: `writeLogs` is what the log ingress asks the authority about.
    Absent (every key minted before C1) means a key may not send."""
    path = tmp_path / "keys.json"
    store = ClientKeyStore(path)
    store.add(replace(_record(), limits={"writeLogs": True}))
    store = ClientKeyStore(path)
    store.load()
    result = store.admit(key_id="abc", action="check", request_id="one", model=None)
    assert result["limits"]["writeLogs"] is True
    store.set_limits("abc", {"allowedModels": None})
    result = store.admit(key_id="abc", action="check", request_id="two", model=None)
    assert "writeLogs" not in result["limits"]


def test_write_logs_must_be_a_boolean():
    # Over HTTP the generated model reads it first, as it does localOnly;
    # this guards what arrives any other way (a replicated or loaded record).
    with pytest.raises(ValueError):
        validate_limits({"writeLogs": 1})
    assert validate_limits({"writeLogs": False}) == validate_limits({})


def test_a_search_is_admitted_with_no_model_under_the_keys_limits(tmp_path):
    """Row 34: a web search has no model, and still counts against the key."""
    store = ClientKeyStore(tmp_path / "keys.json")
    store.add(
        replace(
            _record(),
            limits={
                "maxConcurrentRequests": 1,
                "requestsPerMinute": 2,
                "allowedModels": ["chat"],
            },
        )
    )
    ask = lambda action, request, **kw: store.admit(  # noqa: E731
        key_id="abc", action=action, request_id=request, model=None, purpose="search", **kw
    )
    assert ask("acquire", "s1")["leaseSeconds"] > 0
    with pytest.raises(AdmissionRefusal) as caught:
        ask("acquire", "s2")
    assert caught.value.status == 429
    ask("release", "s1")
    ask("acquire", "s2")
    ask("release", "s2")
    with pytest.raises(AdmissionRefusal) as caught:
        ask("acquire", "s3")
    assert caught.value.status == 429
    with pytest.raises(AdmissionRefusal) as caught:
        store.admit(key_id="abc", action="acquire", request_id="s4", model="chat", purpose="search")
    assert caught.value.status == 422


def test_an_enrolled_agent_passes_on_the_roots_queue_band(app, client):
    """Row 70: the root answers a verified person's `queueBand`. An agent
    whose models predate it fails the whole forward (`extra='forbid'`);
    this one passes it on to the gateway, for row 11's queue."""
    import httpx

    from tests.test_acting_for_operator import _enrolled, _root

    root = _enrolled(app)
    answer = {
        "keyId": "k1",
        "keyName": "app:workbench@gpu-box",
        "personId": "p1",
        "appClientId": "c1",
        "queueBand": "background",
    }
    seen = _root(app, lambda request: httpx.Response(200, json=answer))
    session = root.session("node:gpu-box", "control")
    forwarded = client.post(
        "/v1/auth/client-keys/admission",
        json={"action": "check", "keyId": "k1", "requestId": "r1", "personToken": "proof"},
        headers={"Authorization": f"Bearer {session}"},
    )
    assert forwarded.status_code == 200, forwarded.text
    assert forwarded.json()["queueBand"] == "background"
    assert seen[0].url.path == "/v1/auth/client-keys/admission"


def test_an_enrolled_agent_passes_on_why_the_root_refused(app, client):
    """Row 11: the gateway keeps a request refused for `concurrency` in its
    queue and refuses one refused for `rate`, so the forward passes the root's
    `X-Eugene-Refusal-Reason` on beside `Retry-After`."""
    import httpx

    from tests.test_acting_for_operator import _enrolled, _root

    root = _enrolled(app)
    reasons = iter(["concurrency", "rate"])
    _root(
        app,
        lambda request: httpx.Response(
            429,
            json={"detail": "This key's shared concurrent request limit is reached."},
            headers={"Retry-After": "3", "X-Eugene-Refusal-Reason": next(reasons)},
        ),
    )
    session = root.session("node:gpu-box", "control")
    for expected in ("concurrency", "rate"):
        refused = client.post(
            "/v1/auth/client-keys/admission",
            json={"action": "acquire", "keyId": "k1", "requestId": expected, "model": "m"},
            headers={"Authorization": f"Bearer {session}"},
        )
        assert refused.status_code == 429
        assert refused.headers["Retry-After"] == "3"
        assert refused.headers["X-Eugene-Refusal-Reason"] == expected


def test_a_standalone_agent_names_why_it_refused(authed_client):
    c = authed_client
    made = c.post(
        "/v1/auth/client-keys",
        json={"name": "busy", "limits": {"maxConcurrentRequests": 1, "requestsPerMinute": 2}},
    ).json()
    key = made["key"]["id"]

    def call(action, request_id):
        return c.post(
            "/v1/auth/client-keys/admission",
            json={"action": action, "keyId": key, "requestId": request_id, "model": "m"},
        )

    assert call("acquire", "one").status_code == 200
    busy = call("acquire", "two")
    assert busy.status_code == 429 and busy.headers["X-Eugene-Refusal-Reason"] == "concurrency"
    assert call("release", "one").status_code == 200
    assert call("acquire", "two").status_code == 200
    assert call("release", "two").status_code == 200
    used = call("acquire", "three")
    assert used.status_code == 429 and used.headers["X-Eugene-Refusal-Reason"] == "rate"


def test_a_standalone_band_change_does_not_cut_a_renewal(tmp_path):
    store = ClientKeyStore(tmp_path / "keys.json")
    store.add(replace(_record(), limits={"maxConcurrentRequests": 2, "requestsPerMinute": 60}))
    store.admit(key_id="abc", action="acquire", request_id="one", model="m")
    store.set_limits(
        "abc",
        {
            "maxConcurrentRequests": 2,
            "requestsPerMinute": 60,
            "band": "background",
            "maxQueuedRequests": 0,
        },
    )
    renewed = store.admit(key_id="abc", action="renew", request_id="one", model="m")
    assert renewed["limits"]["band"] == "background"
    assert renewed["limits"]["maxQueuedRequests"] == 0
    with pytest.raises(ValueError):
        validate_limits({"band": "urgent"})
    with pytest.raises(ValueError):
        validate_limits({"maxQueuedRequests": 65})


def test_a_standalone_check_with_a_model_applies_allowed_models(tmp_path):
    store = ClientKeyStore(tmp_path / "keys.json")
    store.add(replace(_record(), limits={"allowedModels": ["allowed"]}))
    store.admit(key_id="abc", action="check", request_id="c1", model="allowed")
    with pytest.raises(AdmissionRefusal) as caught:
        store.admit(key_id="abc", action="check", request_id="c2", model="excluded")
    assert caught.value.status == 403
    store.admit(key_id="abc", action="check", request_id="c3", model=None)


def test_an_enrolled_agent_forwards_only_the_limits_it_was_given(app, client):
    """Row 11 gave `ClientKeyLimits` a band and a queue bound. Forwarded with
    every default filled in, a key made through a newer agent was refused by a
    root a release behind (`extra='forbid'`: Workbench's install stopped at
    its key). What the caller set is what goes, and the root fills the rest."""
    import httpx

    from tests.test_acting_for_operator import _created, _enrolled, _root

    root = _enrolled(app)
    seen = _root(app, lambda request: httpx.Response(201, json=_created("App")))
    session = {"Authorization": f"Bearer {root.session('node:gpu-box', 'control')}"}
    made = client.post(
        "/v1/auth/client-keys",
        json={"name": "App", "limits": {"maxConcurrentRequests": 4}},
        headers=session,
    )
    assert made.status_code == 201, made.text
    sent = json.loads(seen[-1].content)
    assert sent == {"name": "App", "limits": {"maxConcurrentRequests": 4}}
    _root(app, lambda request: httpx.Response(200, json=_created("App")["key"]))
    edited = client.put(
        "/v1/auth/client-keys/k1/limits",
        json={"limits": {"band": "background"}},
        headers=session,
    )
    assert edited.status_code == 200, edited.text
