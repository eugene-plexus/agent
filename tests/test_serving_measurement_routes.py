import asyncio
import json
import time
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

from eugene_plexus_agent._generated.models import Runtime
from eugene_plexus_agent.routes import serving_measurements as routes


def test_measurement_doors_require_an_operator(client):
    assert client.get("/v1/serving-measurements").status_code == 401
    assert (
        client.post("/v1/serving-measurements/preflight", json={"runtime": "r"}).status_code == 401
    )
    assert client.post("/v1/serving-measurements", json={"runtime": "r"}).status_code == 401
    assert client.post("/v1/serving-measurements/no/cancel").status_code == 401


def test_missing_runtime_and_busy_jobs_name_the_cause(authed_client, app):
    result = authed_client.post("/v1/serving-measurements", json={"runtime": "missing"})
    assert result.status_code == 404
    assert "missing" in result.text
    app.state.benchmarks = SimpleNamespace(active=True)
    try:
        response = authed_client.post("/v1/serving-measurements", json={"runtime": "missing"})
        assert response.status_code == 409
        assert "benchmark" in response.text.lower()
    finally:
        del app.state.benchmarks


def test_serving_job_excludes_other_measurements(authed_client, app):
    app.state.serving_measurements = SimpleNamespace(active=True)
    try:
        response = authed_client.post(
            "/v1/benchmarks/preflight",
            json={
                "modelId": "m",
                "profileId": "p",
                "profileName": "p",
                "runtime": {
                    "name": "r",
                    "engine": "llama_cpp",
                    "modelPath": "/m.gguf",
                    "flags": {"contextSize": 4096},
                },
            },
        )
        assert any("serving measurement" in p.lower() for p in response.json()["problems"])
    finally:
        del app.state.serving_measurements


@pytest.fixture
def measurement(app, monkeypatch):
    runtime = Runtime(
        name="r", engine="llama_cpp", modelPath="/m.gguf", modelAlias="m", status="ready"
    )
    monkeypatch.setattr(routes, "runtime_for", lambda *a: runtime)
    monkeypatch.setattr(routes, "node_name", lambda *a: "tower")
    calls = []
    replicas = []

    async def address(self, target):
        return "http://fixture/" + target, {
            "Authorization": self.request.headers.get("authorization")
        }

    async def handle(request):
        calls.append(request)
        if request.url.path.endswith("/copy/stop"):
            replicas[0]["eligible"] = False
        if request.url.path.endswith("/v1/admin/routing"):
            return httpx.Response(
                200, json={"slots": [{"model": "m", "tiers": [{"backends": replicas}]}]}
            )
        return httpx.Response(200, json={})

    async def level(self, people, seconds, workload, progress):
        self.started = time.perf_counter()
        self.partial = self.summarize(people).model_copy(
            update={"requests": 2, "ttftP50Ms": 20.0, "tokensPerSecondEachP50": 50.0}
        )
        return self.partial

    monkeypatch.setattr(routes.Doors, "address", address)
    monkeypatch.setattr(routes.Sweep, "level", level)
    app.state.serving_transport = httpx.MockTransport(handle)
    return runtime, replicas, calls


def wait_finished(client, job_id):
    # Yield to the TestClient's loop, without a process or listener.
    for _ in range(100):
        job = next(
            j
            for j in client.get("/v1/serving-measurements").json()["measurements"]
            if j["id"] == job_id
        )
        if job["state"] != "running" and not job["keyLive"]:
            return job
        time.sleep(0.01)
    pytest.fail("measurement did not finish")


def test_the_key_is_local_only_and_one_model_and_revoked_at_finish(authed_client, app, measurement):
    response = authed_client.post("/v1/serving-measurements", json={"runtime": "r", "maxPeople": 1})
    assert response.status_code == 202, response.text
    job = wait_finished(authed_client, response.json()["id"])
    keys = authed_client.get("/v1/auth/client-keys").json()["keys"]
    key = next(k for k in keys if k["name"].startswith("Measurement "))
    assert key["limits"]["localOnly"]
    assert key["limits"]["allowedModels"] == ["m"]
    assert key["limits"]["maxConcurrentRequests"] == 64
    assert key["limits"]["requestsPerMinute"] == 10000
    assert (
        datetime.fromisoformat(key["expiresAt"].replace("Z", "+00:00"))
        - datetime.fromisoformat(key["createdAt"].replace("Z", "+00:00"))
    ).total_seconds() == 86400
    assert key["revokedAt"] is not None
    assert job["state"] == "completed"
    stored = app.state.serving_measurements.path.read_text()
    assert "Bearer " not in stored and '"token"' not in stored


def test_preflight_and_start_recheck_replica_consent_and_restart_only_stopped(
    authed_client, measurement
):
    _, replicas, calls = measurement
    replicas.append(
        {"node": "studio", "runtime": "copy", "runtime_status": "ready", "eligible": True}
    )
    check = authed_client.post("/v1/serving-measurements/preflight", json={"runtime": "r"}).json()
    assert check["otherReplicas"] == [{"node": "studio", "runtime": "copy", "model": "m"}]
    assert not any("stop" in str(c.url) for c in calls)
    assert authed_client.post("/v1/serving-measurements", json={"runtime": "r"}).status_code == 409
    body = {
        "runtime": "r",
        "maxPeople": 1,
        "stopReplicas": check["otherReplicas"]
        + [{"node": "studio", "runtime": "already-stopped", "model": "m"}],
    }
    result = authed_client.post("/v1/serving-measurements", json=body)
    assert result.status_code == 202, result.text
    job = wait_finished(authed_client, result.json()["id"])
    actions = [str(c.url) for c in calls if c.method == "POST"]
    assert actions == [
        "http://fixture/node:studio/v1/runtimes/copy/stop",
        "http://fixture/node:studio/v1/runtimes/copy/start",
    ]
    assert job["restarts"][0]["state"] == "restarted"
    assert json.loads(next(c for c in calls if c.method == "POST").content) == {
        "reason": "measurement"
    }


def test_mint_failure_and_revoke_retry_keep_the_observed_cause(
    authed_client, app, measurement, monkeypatch
):
    from fastapi import HTTPException

    async def refuse(*a, **kw):
        raise HTTPException(503, "control is sealed")

    real_mint = routes.mint_client_key
    monkeypatch.setattr(routes, "mint_client_key", refuse)
    response = authed_client.post("/v1/serving-measurements", json={"runtime": "r"})
    assert response.status_code == 503 and "control is sealed" in response.text
    monkeypatch.setattr(routes, "mint_client_key", real_mint)
    real_revoke = routes.revoke_client_key_at_authority
    monkeypatch.setattr(routes, "revoke_client_key_at_authority", refuse)
    result = authed_client.post("/v1/serving-measurements", json={"runtime": "r", "maxPeople": 1})
    job_id = result.json()["id"]
    time.sleep(0.01)
    failed_revoke = authed_client.post(f"/v1/serving-measurements/{job_id}/cancel").json()
    assert failed_revoke["keyLive"] and "control is sealed" in failed_revoke["detail"]
    monkeypatch.setattr(routes, "revoke_client_key_at_authority", real_revoke)
    final = authed_client.post(f"/v1/serving-measurements/{job_id}/cancel").json()
    assert not final["keyLive"] and final["state"] == "completed"


def test_a_wildcard_alias_cannot_mint_a_key_for_other_models(authed_client, measurement):
    runtime, _, _ = measurement
    runtime.modelAlias = "model*"
    result = authed_client.post("/v1/serving-measurements", json={"runtime": "r"})
    assert result.status_code == 422 and "wildcard" in result.text


def test_cancel_aborts_the_active_level_and_keeps_partial_results(
    authed_client, measurement, monkeypatch
):
    async def blocked(self, people, *args):
        self.started = time.perf_counter()
        try:
            await asyncio.Event().wait()
        finally:
            self.partial = self.summarize(people)

    monkeypatch.setattr(routes.Sweep, "level", blocked)
    response = authed_client.post("/v1/serving-measurements", json={"runtime": "r"})
    job_id = response.json()["id"]
    start = time.perf_counter()
    cancelled = authed_client.post(f"/v1/serving-measurements/{job_id}/cancel").json()
    assert time.perf_counter() - start < 5
    assert cancelled["state"] == "cancelled" and not cancelled["keyLive"]
    assert len(cancelled["levels"]) == 1


@pytest.mark.asyncio
async def test_engine_counters_and_absent_observations_stay_distinct():
    async def handle(request):
        return httpx.Response(
            200,
            text='vllm:request_success_total{model="m",reason="stop"} 3\nvllm:request_success_total{model="m",reason="length"} 4\n',
        )

    runtime = Runtime(name="r", engine="vllm", modelPath="/m", status="ready", url="http://fixture")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        assert await routes.engine_counter(client, runtime) == 7
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404))
    ) as client:
        assert await routes.engine_counter(client, runtime) is None


def test_a_ready_local_copy_is_named_even_when_the_gateway_view_is_stale(
    authed_client, app, measurement, monkeypatch
):
    # The gateway refreshes its view on a timer, so right after a copy
    # restarts it can still show the copy as not ready. The agent knows its own
    # runtimes, so it must name the copy and stop it.
    _, replicas, _calls = measurement
    replicas.append(
        {"node": "tower", "runtime": "copy", "runtime_status": "starting", "eligible": False}
    )
    spec = SimpleNamespace(name="copy")
    monkeypatch.setattr(app.state.agent_state, "list_runtime_specs", lambda: [spec])
    monkeypatch.setattr(
        routes,
        "_compose",
        lambda s, sup: Runtime(
            name="copy", engine="llama_cpp", modelPath="/m.gguf", modelAlias="m", status="ready"
        ),
    )
    check = authed_client.post("/v1/serving-measurements/preflight", json={"runtime": "r"}).json()
    assert check["otherReplicas"] == [{"node": "tower", "runtime": "copy", "model": "m"}]
