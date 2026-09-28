"""Standalone limits use the same atomic registry and survive restart."""

from dataclasses import replace

import pytest

from eugene_plexus_agent import _private_files
from eugene_plexus_agent.client_admission import AdmissionRefusal
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
