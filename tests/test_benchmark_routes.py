import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from eugene_plexus_agent import security


def request_body():
    return {
        "modelId": "m",
        "profileId": "p",
        "profileName": "test",
        "runtime": {
            "name": "test",
            "engine": "llama_cpp",
            "modelPath": "/models/q.gguf",
            "flags": {"contextSize": 4096},
        },
    }


def test_reads_and_writes_require_operator(client, app):
    assert client.get("/v1/benchmarks").status_code == 401
    token = security.issue_service_token(
        signing_key=app.state.auth_state.signing_key, kind="gateway"
    )
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/v1/benchmarks", headers=headers).status_code == 401
    assert client.post("/v1/benchmarks", json=request_body(), headers=headers).status_code == 401


def test_busy_runtime_refuses_without_stopping(authed_client, stub_runtime_supervisor):
    spec = request_body()["runtime"]
    assert authed_client.post("/v1/runtimes", json=spec).status_code == 201
    before = list(stub_runtime_supervisor.calls)
    response = authed_client.post("/v1/benchmarks", json=request_body())
    assert response.status_code == 409 and "Stop" in response.text
    assert stub_runtime_supervisor.calls == before


@pytest.mark.parametrize(
    "verb,path",
    [
        ("post", "/v1/runtimes"),
        ("patch", "/v1/runtimes/test"),
        ("post", "/v1/runtimes/test/start"),
        ("post", "/v1/runtimes/test/restart"),
        ("post", "/v1/benchmarks"),
    ],
)
def test_active_benchmark_excludes_launch_mutations(authed_client, app, verb, path):
    app.state.benchmarks = SimpleNamespace(active=True)
    try:
        response = getattr(authed_client, verb)(
            path, json=request_body()["runtime"] if "runtimes" in path else request_body()
        )
        assert response.status_code == 409
    finally:
        del app.state.benchmarks


def test_validation_precedes_binary_probe(authed_client, monkeypatch):
    from eugene_plexus_agent.routes import benchmarks

    monkeypatch.setattr(benchmarks, "prepare_binary", lambda *a: pytest.fail("must not probe"))
    body = request_body()
    body["runtime"]["extraArgs"] = ["--anything"]
    response = authed_client.post("/v1/benchmarks", json=body)
    assert response.status_code == 422 and "extraArgs" in response.text
    assert authed_client.get("/v1/benchmarks").json() == {"benchmarks": []}
    assert authed_client.post("/v1/benchmarks/missing/cancel").status_code == 404


@pytest.mark.parametrize("benchmark_first", [True, False])
def test_launch_race_is_serialized_in_both_directions(
    authed_client, app, tmp_path, monkeypatch, benchmark_first
):
    import httpx

    from eugene_plexus_agent.routes import benchmarks, runtimes

    entered, release = None, None
    original_admission = runtimes._admission_for

    async def slow_admission(*args):
        entered.set()
        await release.wait()
        return await original_admission(*args)

    monkeypatch.setattr(benchmarks, "_admission_for", slow_admission)
    monkeypatch.setattr(runtimes, "_admission_for", slow_admission)
    monkeypatch.setattr(
        benchmarks, "prepare_binary", lambda *a: (Path(sys.executable), "fixture", "fixture")
    )
    original = benchmarks.benchmark_args
    monkeypatch.setattr(
        benchmarks,
        "benchmark_args",
        lambda body, binary, model, help_text=None: (
            ([sys.executable, "-c", "import time; time.sleep(60)"], [0, 1984, 3968])
            if help_text == "fixture"
            else original(body, binary, model, help_text)
        ),
    )
    model = tmp_path / "m.gguf"
    model.write_bytes(b"fixture")
    body = request_body()
    body["runtime"]["modelPath"] = str(model)

    async def scenario():
        nonlocal entered, release
        entered, release = asyncio.Event(), asyncio.Event()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers=authed_client.headers,
        ) as client:

            async def benchmark():
                return await client.post("/v1/benchmarks", json=body)

            async def runtime():
                return await client.post("/v1/runtimes", json=body["runtime"])

            first = asyncio.create_task(benchmark() if benchmark_first else runtime())
            await asyncio.wait_for(entered.wait(), 3)
            second = asyncio.create_task(runtime() if benchmark_first else benchmark())
            await asyncio.sleep(0.05)
            assert not second.done(), "second launch escaped the admission lock"
            release.set()
            first_result, second_result = await asyncio.gather(first, second)
            assert first_result.status_code == (202 if benchmark_first else 201), first_result.text
            assert second_result.status_code == 409, second_result.text
            if benchmark_first:
                job_id = first_result.json()["id"]
                cancel = await client.post(f"/v1/benchmarks/{job_id}/cancel")
                assert cancel.json()["state"] == "cancelled"
                assert (await client.post("/v1/runtimes", json=body["runtime"])).status_code == 201

    authed_client.portal.call(scenario)


def test_gateway_wake_cannot_bypass_benchmark(authed_client, app):
    spec = request_body()["runtime"] | {"autoStart": False}
    assert authed_client.post("/v1/runtimes", json=spec).status_code == 201
    app.state.benchmarks = SimpleNamespace(active=True)
    token = security.issue_service_token(
        signing_key=app.state.auth_state.signing_key, kind="gateway"
    )
    try:
        response = authed_client.post(
            "/v1/runtimes/test/start", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 409
        assert authed_client.post("/v1/runtimes/test/stop").status_code == 202
    finally:
        del app.state.benchmarks
