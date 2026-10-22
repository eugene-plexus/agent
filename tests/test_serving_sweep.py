import asyncio
import json
import random
import time
from datetime import UTC, datetime

import httpx
import pytest

from eugene_plexus_agent._generated.models import (
    BenchmarkState,
    ServingMeasurementRequest,
    ServingProvenance,
)
from eugene_plexus_agent.serving_sweep import (
    Observation,
    ServingMeasurements,
    Session,
    Sweep,
    burst_size,
    effective_request,
    knee_for,
    pause_seconds,
    people_levels,
    percentile,
    provenance_key,
)


def provenance(build="b1"):
    return ServingProvenance(
        engine="llama_cpp",
        engineBuild=build,
        binaryHash=None,
        launchArgv=[],
        modelPath="m.gguf",
        modelSizeBytes=3,
        modelModifiedAt=None,
        gpus=[],
        driver=None,
        cpu="fixture",
        ramBytes=1000,
        agentCommit=None,
        gatewayCommit=None,
    )


def test_pauses_hit_the_published_points():
    rng = random.Random(12)
    samples = [pause_seconds(rng) for _ in range(200_000)]
    for q, value, tolerance in ((0.5, 2, 0.05), (0.9, 24, 0.1), (0.99, 684, 0.15)):
        assert percentile(samples, q) == pytest.approx(value, rel=tolerance)
    assert max(samples) <= 1800


def test_burst_sizes_have_median_7_and_p90_51():
    rng = random.Random(21)
    values = [burst_size(rng) for _ in range(200_000)]
    assert percentile(values, 0.5) == pytest.approx(7, abs=1)
    assert percentile(values, 0.9) == pytest.approx(51, abs=2)
    assert max(values) == 64


def test_96_percent_of_turns_reuse_90_percent():
    session = Session(random.Random(4), "person", "agents", 262144)
    reused = []
    for _ in range(10000):
        session.advance("answer " * 317)
        reused.append(session.reuse)
    assert sum(r >= 0.9 for r in reused) / len(reused) == pytest.approx(0.96, abs=0.012)
    assert session.system == Session(random.Random(5), "other", "agents", 262144).system


def test_defaults_resolve_and_full_reports_the_window_in_effect():
    body = effective_request(ServingMeasurementRequest(runtime="r"), slots=2)
    assert (body.maxPeople, body.windowSeconds, body.workload) == (8, 300, "agents")
    body = effective_request(
        ServingMeasurementRequest(runtime="r", full=True, windowSeconds=10), slots=100
    )
    assert (body.maxPeople, body.windowSeconds) == (256, 3600)
    assert people_levels(7) == [1, 2, 4, 7]


class Tokens(httpx.AsyncByteStream):
    def __init__(self, owner, body):
        self.owner, self.body = owner, body

    async def __aiter__(self):
        self.owner.active += 1
        try:
            deadline = time.perf_counter()
            for _ in range(8):
                deadline += 0.02 * max(1, self.owner.active / 2)
                await asyncio.sleep(max(0, deadline - time.perf_counter()))
                yield b'data: {"choices":[{"delta":{"content":"word "}}]}\n\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":8,"prompt_tokens_details":{"cached_tokens":95}}}\n\n'
            yield b"data: [DONE]\n\n"
        finally:
            self.owner.active -= 1


class Backend:
    active = 0
    runtime = "r"

    async def __call__(self, request):
        body = json.loads(request.content)
        assert body["stream_options"]["include_usage"]
        assert request.headers["authorization"] == "Bearer measurement-secret"
        return httpx.Response(
            200,
            headers={
                "X-Eugene-Plexus-Runtime": self.runtime,
                "X-Eugene-Plexus-Node": "tower",
                "X-Eugene-Plexus-Waited-Ms": "3",
            },
            stream=Tokens(self, body),
        )


@pytest.mark.asyncio
async def test_a_two_slot_engine_knees_at_two_people():
    async with httpx.AsyncClient(transport=httpx.MockTransport(Backend())) as client:
        sweep = Sweep(
            client,
            "http://fixture",
            "measurement-secret",
            "r",
            "tower",
            "m",
            8192,
            random.Random(0),
        )
        levels = [await sweep.level(n, 0.35, "streams") for n in (1, 2, 4)]
    assert knee_for(levels, 1000, 2 / 3).people == 2
    assert knee_for(levels, 1000, 2 / 3).found
    assert levels[0].tokensPerSecondEachP50 == pytest.approx(50, rel=0.1)
    assert levels[-1].streamsPeak == 4
    assert levels[0].outputTokens == levels[0].requests * 8
    assert levels[0].cachedTokens == levels[0].requests * 95
    assert levels[0].waitedP50Ms == 3
    assert levels[0].foreignRequests is None


@pytest.mark.asyncio
async def test_an_answer_from_another_runtime_stops_the_run():
    backend = Backend()
    backend.runtime = "other"
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
        sweep = Sweep(
            client,
            "http://fixture",
            "measurement-secret",
            "r",
            "tower",
            "m",
            8192,
            random.Random(0),
        )
        with pytest.raises(ValueError, match=r"other.*tower.*r.*tower"):
            await sweep.level(1, 0.1, "streams")
    assert sweep.partial.errors == 1


@pytest.mark.parametrize("metadata", ["headers", "envelope", "runtime-header", "node-header"])
@pytest.mark.parametrize("counted", [True, False])
@pytest.mark.asyncio
async def test_the_same_runtime_on_another_machine_stops_the_run(metadata, counted):
    async def handle(request):
        headers, routing = {}, {}
        for field, value in (("runtime", "r"), ("node", "studio")):
            if metadata == "headers" or metadata == f"{field}-header":
                headers[f"x-eugene-plexus-{field}"] = value
            else:
                routing[field] = value
        frame = {"choices": [{"delta": {"content": "word"}}], "x_eugene_plexus": routing}
        return httpx.Response(
            200, headers=headers, text="data: " + json.dumps(frame) + "\n\ndata: [DONE]\n\n"
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        sweep = Sweep(client, "http://fixture", "secret", "r", "tower", "m", 8192, random.Random(0))
        with pytest.raises(ValueError, match=r"studio.*tower.*run stopped"):
            if counted:
                await sweep.level(1, 0.01, "streams")
            else:
                await sweep.turn(Session(random.Random(0), "warmup", "agents", 8192), counted=False)
    assert sweep.fatal is not None
    if counted:
        assert sweep.partial.errors == 1


@pytest.mark.parametrize("missing", ["runtime", "node", "runtime and node"])
@pytest.mark.asyncio
async def test_an_answer_with_missing_provenance_stops_the_run(missing):
    async def handle(request):
        routing = {"runtime": "r", "node": "tower"}
        for field in missing.split(" and "):
            del routing[field]
        frame = {"choices": [{"delta": {"content": "word"}}], "x_eugene_plexus": routing}
        return httpx.Response(200, text="data: " + json.dumps(frame) + "\n\ndata: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        sweep = Sweep(client, "http://fixture", "secret", "r", "tower", "m", 8192, random.Random(0))
        with pytest.raises(ValueError, match=r"did not report.*run stopped"):
            await sweep.level(1, 0.01, "streams")
    assert sweep.fatal is not None
    assert sweep.partial.errors == 1


def test_a_changed_engine_build_makes_the_record_stale(tmp_path):
    store = ServingMeasurements(tmp_path / "runs.json")
    job = store.new(
        effective_request(ServingMeasurementRequest(runtime="r"), 2), "tower", "m", provenance()
    )
    job.state = BenchmarkState.completed
    store.save()
    loaded = ServingMeasurements(store.path)
    record = loaded.list({"r": provenance("b2")})[0]
    assert not record.current
    assert "b1" in record.staleBecause[0] and "b2" in record.staleBecause[0]
    assert job.provenanceKey == provenance_key(provenance())
    assert (
        provenance_key(provenance())
        == __import__("hashlib")
        .sha256(
            json.dumps(
                provenance().model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        )
        .hexdigest()
    )


def test_restart_fails_active_run_and_keeps_key_metadata_without_secret(tmp_path):
    store = ServingMeasurements(tmp_path / "runs.json")
    job = store.new(
        effective_request(ServingMeasurementRequest(runtime="r"), 1), "tower", "m", provenance()
    )
    store.attach_key(job.id, "key-id", datetime.now(UTC))
    store.save()
    loaded = ServingMeasurements(store.path)
    assert loaded.jobs[0].state == "failed"
    assert "agent restarted" in loaded.jobs[0].detail.lower()
    assert loaded.keys[job.id]["id"] == "key-id"
    assert "token" not in store.path.read_text()


def test_retains_latest_completed_per_runtime_beyond_twenty(tmp_path):
    store = ServingMeasurements(tmp_path / "runs.json")
    first = store.new(
        effective_request(ServingMeasurementRequest(runtime="old"), 1), "tower", "m", provenance()
    )
    first.state = BenchmarkState.completed
    for _ in range(25):
        store.new(
            effective_request(ServingMeasurementRequest(runtime="r"), 1), "tower", "m", provenance()
        ).state = BenchmarkState.failed
    store.save()
    assert len(store.jobs) == 21
    assert first in store.jobs


@pytest.mark.asyncio
async def test_warm_start_is_uncounted_and_precedes_the_level_window():
    class WarmSweep(Sweep):
        async def turn(self, session, *, counted=True):
            if not counted:
                await asyncio.sleep(0.12)
                return "warm answer"
            self.observations.append(Observation(output=8))
            return "answer"

    async with httpx.AsyncClient() as client:
        sweep = WarmSweep(
            client, "http://fixture", "secret", "r", "tower", "m", 8192, random.Random(0)
        )
        result = await sweep.level(1, 0.08, "agents")
    assert result.requests >= 1
    assert result.durationSeconds < 0.12


@pytest.mark.asyncio
async def test_tokenless_success_is_an_error_and_does_not_invent_speed():
    async def handle(request):
        return httpx.Response(
            200,
            text='data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}\n\ndata: [DONE]\n\n',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        sweep = Sweep(client, "http://fixture", "secret", "r", "tower", "m", 8192, random.Random(0))
        result = await sweep.level(1, 0.001, "streams")
    assert result.errors == result.requests
    assert result.tokensPerSecondEachP50 is None
    assert "no content" in next(iter(result.errorCauses)).lower()


def test_an_unusable_baseline_does_not_claim_one_person_capacity():
    sweep = Sweep(None, "http://fixture", "secret", "r", "tower", "m", 8192, random.Random(0))
    sweep.started = time.perf_counter()
    level = sweep.summarize(1).model_copy(
        update={"requests": 1, "ttftP50Ms": 2000.0, "tokensPerSecondEachP50": 50.0}
    )
    assert knee_for([level], 1000, 2 / 3) is None


@pytest.mark.asyncio
async def test_a_restart_preserves_the_inflight_level_checkpoint(tmp_path):
    entered = asyncio.Event()

    class CheckpointSweep(Sweep):
        async def level(self, people, seconds, workload, progress=None):
            self.started = time.perf_counter()
            self.observations = [Observation(output=8, prompt=100)]
            progress(0.5)
            entered.set()
            await asyncio.Event().wait()

    store = ServingMeasurements(tmp_path / "runs.json")
    job = store.new(
        effective_request(ServingMeasurementRequest(runtime="r"), 1), "tower", "m", provenance()
    )

    async def noop(*args):
        pass

    async with httpx.AsyncClient() as client:
        sweep = CheckpointSweep(
            client, "http://fixture", "secret", "r", "tower", "m", 8192, random.Random(0)
        )
        task = asyncio.create_task(store.run(job, sweep, noop, noop))
        try:
            await entered.wait()
            loaded = ServingMeasurements(store.path)
            assert loaded.jobs[0].levels[0].outputTokens == 8
            assert loaded.jobs[0].state == "failed"
        finally:
            task.cancel()
            await task


@pytest.mark.asyncio
async def test_a_record_write_failure_still_revokes_the_measurement_key(tmp_path, monkeypatch):
    class FailedSweep:
        partial = None
        notes = frozenset()

        async def level(self, *args):
            raise ValueError("engine refused")

    store = ServingMeasurements(tmp_path / "runs.json")
    job = store.new(
        effective_request(ServingMeasurementRequest(runtime="r"), 1), "tower", "m", provenance()
    )
    job.keyLive = True

    def disk_full():
        raise OSError("disk full")

    async def cleanup(record):
        record.keyLive = False

    async def noop(*args):
        pass

    monkeypatch.setattr(store, "save", disk_full)
    await store.run(job, FailedSweep(), cleanup, noop)
    assert not job.keyLive and "disk full" in job.detail
