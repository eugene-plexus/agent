"""Row 65: local manifests, a worker report and inode-sharing model copies."""

import hashlib
import json
import os
import struct

import pytest

from eugene_plexus_agent import model_copies as mc


@pytest.fixture
def ollama(tmp_path, monkeypatch):
    store = tmp_path / "ollama"

    def text(value):
        return struct.pack("<Q", len(value)) + value

    data = (
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + text(b"general.architecture")
        + struct.pack("<I", 8)
        + text(b"llama")
    )
    digest = hashlib.sha256(data).hexdigest()
    blob = store / "blobs" / ("sha256-" + digest)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(data)
    manifest = store / "manifests" / "registry.ollama.ai" / "library" / "llama3" / "8b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "runner": "llamacpp",
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": "sha256:" + digest,
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("OLLAMA_MODELS", str(store))
    return store, blob, digest


def test_report_body_from_store(ollama):
    from eugene_plexus_agent import ollama_store

    store, blob, digest = ollama
    report = ollama_store.report_body()
    assert report["readAt"]
    assert report["stores"][0]["path"] == str(store)
    model = report["stores"][0]["models"][0]
    assert model["tags"] == ["llama3:8b"]
    assert model["digest"] == "sha256:" + digest
    assert model["sizeBytes"] == blob.stat().st_size
    assert model["architecture"] == "llama"


def test_node_copy_links_ollama_blob(ollama, tmp_path):
    _, blob, _ = ollama
    source = tmp_path / "library" / "model.gguf"
    source.parent.mkdir()
    source.write_bytes(blob.read_bytes())
    os.utime(source, (1000, 1000))
    destination = tmp_path / "copies" / "model.gguf"
    plan = mc.CopyPlan(str(source), str(source), str(destination))
    settings = mc.CopySettings(True, str(destination.parent), 0)
    mc.copy_file(plan, settings, mc.CopyState(str(destination), source.stat().st_size))
    assert os.path.samefile(destination, blob)
    assert mc.copy_is_current(plan)
    source.unlink()
    assert destination.read_bytes() == blob.read_bytes()


def test_kept_model_uses_original_served_name():
    from eugene_plexus_agent.run_worker import runtime_spec

    model = {
        "path": "/models/ollama/llama3-8b/llama3-8b.gguf",
        "name": "llama3:8b",
        "keptFrom": {"servedName": "llama3:8b"},
    }
    profile = {"id": "tuned", "name": "Tuned", "engine": "llama_cpp"}
    assert runtime_spec(model, profile, start=True).modelAlias == "llama3:8b"


@pytest.mark.asyncio
async def test_send_verifies_assignment_before_reading_store(ollama):
    from fastapi import HTTPException

    from eugene_plexus_agent._generated.models import OllamaSendRequest
    from eugene_plexus_agent.routes.ollama import send

    class Library:
        async def operation_request(self, method, path, **kwargs):
            assert method == "GET"
            return {"node": "another", "state": "running", "digest": "sha256:" + ollama[2]}

    with pytest.raises(HTTPException) as refused:
        await send(Library(), "worker", OllamaSendRequest(job="j", digest="sha256:" + ollama[2]))
    assert refused.value.status_code == 403


@pytest.mark.asyncio
async def test_send_resumes_only_reserved_roles(ollama):
    from eugene_plexus_agent._generated.models import OllamaSendRequest
    from eugene_plexus_agent.routes.ollama import send

    _, blob, digest = ollama
    received = bytearray(blob.read_bytes()[:7])
    calls = []

    class Library:
        async def operation_request(self, method, path, **kwargs):
            calls.append((method, path, kwargs))
            if method == "GET":
                return {"node": "worker", "state": "running", "digest": "sha256:" + digest}
            if method == "PUT":
                assert kwargs["params"] == {"role": "weights", "offset": 7}
                received.extend(kwargs["content"])
            elif method == "POST":
                assert kwargs["json"] == {"role": "weights"}
            return {"sizeBytes": None, "sha256": None, "receivedBytes": len(received)}

    result = await send(Library(), "worker", OllamaSendRequest(job="j", digest="sha256:" + digest))
    assert bytes(received) == blob.read_bytes()
    assert result.bytesSent == blob.stat().st_size - 7
    assert calls[-1][1].endswith("/files/complete")


def test_send_route_refuses_paths_and_urls(authed_client):
    response = authed_client.post(
        "/v1/ollama/send",
        json={
            "job": "j",
            "digest": "sha256:" + "a" * 64,
            "path": "C:/secret",
            "url": "https://elsewhere/upload",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled,local,expected", [(True, False, 1), (False, False, 0), (True, True, 0)]
)
async def test_reporting_default_change_detection_and_switch(
    ollama, monkeypatch, enabled, local, expected
):
    import asyncio
    from types import SimpleNamespace

    from eugene_plexus_agent import ollama_report

    calls = []

    class Library:
        async def operation_request(self, method, path, **kwargs):
            calls.append((method, path, kwargs))

    async def library_client(context):
        return Library()

    ticks = 0

    async def pause(seconds):
        nonlocal ticks
        assert seconds == 60
        ticks += 1
        if ticks == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(ollama_report, "library_client_for", library_client)
    monkeypatch.setattr(ollama_report.asyncio, "sleep", pause)
    state = SimpleNamespace(
        get_config=lambda key: enabled,
        list_topology_entries=lambda: [SimpleNamespace(kind="library")] if local else [],
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            agent_state=state,
            node_identity=SimpleNamespace(record=SimpleNamespace(name="worker", enrolled=True)),
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await ollama_report.run(app)
    assert len(calls) == expected
