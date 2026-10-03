import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from .conftest import local_service_token


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
    token = local_service_token(app, "gateway")
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/v1/benchmarks", headers=headers).status_code == 401
    assert client.post("/v1/benchmarks", json=request_body(), headers=headers).status_code == 401


def test_busy_runtime_refuses_without_stopping(authed_client, stub_runtime_supervisor):
    # Nothing agreed to, so nothing stopped: the same 409 R6.1 gave, now
    # naming the model and saying how to agree (2026-09-30, Troy).
    spec = request_body()["runtime"]
    assert authed_client.post("/v1/runtimes", json=spec).status_code == 201
    before = list(stub_runtime_supervisor.calls)
    response = authed_client.post("/v1/benchmarks", json=request_body())
    assert response.status_code == 409
    assert "test" in response.json()["detail"] and "Agree" in response.json()["detail"]
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

    original_measure = runtimes._measure_launch

    async def slow_measure(*args):
        entered.set()
        await release.wait()
        return await original_measure(*args)

    monkeypatch.setattr(benchmarks, "_admission_for", slow_admission)
    monkeypatch.setattr(runtimes, "_admission_for", slow_admission)
    # A launch measures through `_measure_launch` since it began carrying
    # `engine_places` to the reservation (2026-10-03); hold it there too.
    monkeypatch.setattr(runtimes, "_measure_launch", slow_measure)
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


def _startable(monkeypatch, tmp_path):
    """A benchmark request whose job starts and then idles until cancelled."""
    from eugene_plexus_agent.routes import benchmarks

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
    return body


def _declare(client, tmp_path, name):
    path = tmp_path / f"{name}.gguf"
    path.write_bytes(b"fixture")
    spec = {"name": name, "engine": "llama_cpp", "modelPath": str(path)}
    assert client.post("/v1/runtimes", json=spec).status_code == 201


def test_preflight_lists_what_is_running_and_changes_nothing(
    authed_client, stub_runtime_supervisor, monkeypatch, tmp_path
):
    body = _startable(monkeypatch, tmp_path)
    _declare(authed_client, tmp_path, "busy")
    before = list(stub_runtime_supervisor.calls)
    answer = authed_client.post("/v1/benchmarks/preflight", json=body).json()
    assert answer["runningRuntimes"] == ["busy"]
    assert answer["problems"] == []
    assert stub_runtime_supervisor.calls == before
    assert authed_client.get("/v1/benchmarks").json() == {"benchmarks": []}


def test_agreed_stop_is_measurement_and_restarts_after(
    authed_client, stub_runtime_supervisor, monkeypatch, tmp_path
):
    body = _startable(monkeypatch, tmp_path)
    _declare(authed_client, tmp_path, "busy")
    body["stopRuntimes"] = ["busy"]
    started = authed_client.post("/v1/benchmarks", json=body)
    assert started.status_code == 202, started.text
    assert ("stop_one", "busy") in stub_runtime_supervisor.calls
    runtime = authed_client.get("/v1/runtimes/busy").json()
    assert runtime["status"] == "stopped" and runtime["stopReason"] == "measurement"
    assert [(r["name"], r["state"]) for r in started.json()["restarts"]] == [("busy", "pending")]
    # A start while the benchmark runs is refused; the restart is the job's.
    assert authed_client.post("/v1/runtimes/busy/start").status_code == 409
    job = authed_client.post(f"/v1/benchmarks/{started.json()['id']}/cancel").json()
    assert job["state"] == "cancelled"
    assert [r["state"] for r in job["restarts"]] == ["restarted"]
    assert stub_runtime_supervisor.is_running("busy")
    stored = authed_client.get("/v1/benchmarks").json()["benchmarks"][0]
    assert stored["restarts"][0]["state"] == "restarted"


def test_a_model_started_after_the_question_is_never_stopped(
    authed_client, stub_runtime_supervisor, monkeypatch, tmp_path
):
    body = _startable(monkeypatch, tmp_path)
    _declare(authed_client, tmp_path, "busy")
    _declare(authed_client, tmp_path, "late")
    body["stopRuntimes"] = ["busy"]
    before = list(stub_runtime_supervisor.calls)
    response = authed_client.post("/v1/benchmarks", json=body)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "late" in detail and "busy" not in detail
    assert stub_runtime_supervisor.calls == before


def test_restart_after_off_leaves_them_stopped(
    authed_client, stub_runtime_supervisor, monkeypatch, tmp_path
):
    body = _startable(monkeypatch, tmp_path)
    _declare(authed_client, tmp_path, "busy")
    body |= {"stopRuntimes": ["busy"], "restartAfter": False}
    started = authed_client.post("/v1/benchmarks", json=body).json()
    job = authed_client.post(f"/v1/benchmarks/{started['id']}/cancel").json()
    assert [r["state"] for r in job["restarts"]] == ["skipped"]
    assert not stub_runtime_supervisor.is_running("busy")


def test_refusal_after_stopping_puts_them_back_at_once(
    authed_client, stub_runtime_supervisor, monkeypatch, tmp_path
):
    from eugene_plexus_agent._generated.models import AdmissionDecision
    from eugene_plexus_agent.routes import benchmarks

    body = _startable(monkeypatch, tmp_path)
    _declare(authed_client, tmp_path, "busy")
    body["stopRuntimes"] = ["busy"]

    async def refuse(*args):
        return SimpleNamespace(decision=AdmissionDecision.refuse, reason="too big for this card")

    monkeypatch.setattr(benchmarks, "_admission_for", refuse)
    response = authed_client.post("/v1/benchmarks", json=body)
    assert response.status_code == 422 and "too big" in response.text
    calls = [c for c in stub_runtime_supervisor.calls if c[1] == "busy"]
    assert calls[-2:] == [("stop_one", "busy"), ("add_and_start", "busy")]
    assert stub_runtime_supervisor.is_running("busy")
    assert authed_client.get("/v1/benchmarks").json() == {"benchmarks": []}


def test_gateway_wake_cannot_bypass_benchmark(authed_client, app):
    spec = request_body()["runtime"] | {"autoStart": False}
    assert authed_client.post("/v1/runtimes", json=spec).status_code == 201
    app.state.benchmarks = SimpleNamespace(active=True)
    token = local_service_token(app, "gateway")
    try:
        response = authed_client.post(
            "/v1/runtimes/test/start", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 409
        assert authed_client.post("/v1/runtimes/test/stop").status_code == 202
    finally:
        del app.state.benchmarks
