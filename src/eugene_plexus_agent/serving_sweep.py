"""Seeded serving load and durable, node-owned measurement jobs.

Only the gateway receives generated work. Credentials and session text stay in
memory; the store contains observations and key ids needed for later revocation.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import httpx

from ._generated.models import (
    BenchmarkState,
    ServingDrafting,
    ServingKnee,
    ServingLevel,
    ServingMeasurement,
    ServingMeasurementRequest,
    ServingProvenance,
    ServingWorkload,
)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    return ordered[lo] + (ordered[math.ceil(position)] - ordered[lo]) * (position - lo)


def pause_seconds(rng: random.Random) -> float:
    """Inverse CDF, linear in log time between the published quantiles."""
    u = rng.random()
    points = ((0.0, 0.05), (0.5, 2.0), (0.9, 24.0), (0.99, 684.0), (1.0, 1800.0))
    for (p, value), (next_p, next_value) in pairwise(points):
        if u <= next_p:
            return math.exp(math.log(value) + (u - p) / (next_p - p) * math.log(next_value / value))
    return 1800.0


def burst_size(rng: random.Random) -> int:
    return min(64, max(1, round(rng.lognormvariate(math.log(7), math.log(51 / 7) / 1.28155))))


def people_levels(maximum: int) -> list[int]:
    values = [1]
    while values[-1] < maximum:
        values.append(min(maximum, values[-1] * 2))
    return values


def effective_request(body: ServingMeasurementRequest, slots: int) -> ServingMeasurementRequest:
    workload = body.workload or ServingWorkload.agents
    return body.model_copy(
        update={
            "workload": ServingWorkload(workload),
            "maxPeople": body.maxPeople or min(256, 4 * slots),
            "windowSeconds": 3600
            if body.full
            else body.windowSeconds or (120 if workload == "streams" else 300),
            "full": bool(body.full),
            "seed": body.seed if body.seed is not None else 0,
            "ttftThresholdMs": body.ttftThresholdMs or 1000,
            "speedRatioThreshold": body.speedRatioThreshold or 2 / 3,
            "stopReplicas": body.stopReplicas or [],
        }
    )


def provenance_key(provenance: ServingProvenance) -> str:
    value = json.dumps(
        provenance.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def knee_for(levels: list[ServingLevel], ttft: float, ratio: float) -> ServingKnee | None:
    usable = [
        level
        for level in levels
        if level.tokensPerSecondEachP50 is not None
        and level.ttftP50Ms is not None
        and level.requests > level.errors
    ]
    if not usable or usable[0].people != 1 or (usable[0].ttftP50Ms or 0) > ttft:
        return None
    baseline = usable[0].tokensPerSecondEachP50
    assert baseline is not None
    previous = usable[0]
    found = False
    for level in usable:
        assert level.ttftP50Ms is not None and level.tokensPerSecondEachP50 is not None
        if level.ttftP50Ms > ttft or level.tokensPerSecondEachP50 < baseline * ratio:
            found = True
            break
        previous = level
    assert previous.tokensPerSecondEachP50 is not None and previous.ttftP50Ms is not None
    return ServingKnee(
        found=found,
        people=previous.people,
        tokensPerSecondEach=previous.tokensPerSecondEachP50,
        ttftP50Ms=previous.ttftP50Ms,
    )


_CORPUS = (
    files("eugene_plexus_agent").joinpath("data/evaluation_text.txt").read_text(encoding="utf-8")
)


def text_tokens(count: int, offset: int = 0) -> str:
    """Corpus text at four characters/token; actual engine usage is recorded.

    This is a workload size estimate, never an output-token measurement. Different
    tokenizers necessarily make different prompt sizes from the same bundled text.
    """
    size = max(0, count) * 4
    repeats = (size + offset) // len(_CORPUS) + 1
    return (_CORPUS * repeats)[offset : offset + size]


class Session:
    def __init__(
        self, rng: random.Random, identity: str, workload: str, context: int, *, warm: bool = True
    ) -> None:
        self.rng, self.identity, self.workload = rng, identity, workload
        self.context = context
        self.system = text_tokens(2000)
        self.seed = f"\nSession {identity}.\n"
        self.history = self.system + self.seed
        self.reuse = 1.0
        if workload == "agents" and warm:
            turns = max(1, round(rng.lognormvariate(math.log(107), math.log(354 / 107) / 1.28155)))
            turn = rng.randrange(turns)
            self.history += text_tokens(min(turn * 949, max(0, int(context * 0.8) - 2000)))
        self.advance("")

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.history[len(self.system) :]},
        ]

    def advance(self, answer: str) -> None:
        old = len(self.history)
        if self.workload != "agents":
            self.history = (
                self.system + self.seed + text_tokens(60 if self.workload == "chat" else 128)
            )
            self.reuse = 1.0
            return
        addition = text_tokens(
            max(1, round(self.rng.lognormvariate(math.log(632), 0.5))), self.rng.randrange(1000)
        )
        retained = old
        if self.rng.random() < 0.04:
            retained = int(old * self.rng.uniform(0.7, 0.89))
            retained = max(len(self.system + self.seed), retained)
            self.history = self.history[:retained]
        if len(self.history + answer + addition) / 4 > 0.9 * self.context:
            self.history = self.system + self.seed
            retained = len(self.history)
        self.reuse = retained / max(1, old)
        self.history += answer + addition

    def output_limit(self) -> int:
        value = (
            round(self.rng.lognormvariate(math.log(317), 0.5))
            if self.workload == "agents"
            else 400
            if self.workload == "chat"
            else 128
        )
        return max(1, min(value, int(self.context * 0.9) - len(self.history) // 4))


@dataclass
class Observation:
    ttft: float | None = None
    times: list[float] = field(default_factory=list)
    output: int | None = None
    prompt: int | None = None
    cached: int | None = None
    waited: float | None = None
    error: str | None = None
    answer: str = ""
    reuse: float | None = None


class Sweep:
    def __init__(
        self,
        client: httpx.AsyncClient,
        gateway: str,
        token: str,
        runtime: str,
        node: str,
        model: str,
        context: int,
        rng: random.Random,
    ) -> None:
        self.client, self.gateway, self.token = client, gateway.rstrip("/"), token
        self.runtime, self.node, self.model = runtime, node, model
        self.context, self.rng = context, rng
        self.partial: ServingLevel | None = None
        self.observations: list[Observation] = []
        self.inflight = self.peak = 0
        self.area = 0.0
        self.changed = 0.0
        self.started = 0.0
        self.lag = 0.0
        self.notes: set[str] = set()
        self.fatal: str | None = None
        self.warmups = 0

    def transition(self, delta: int) -> None:
        now = time.perf_counter()
        self.area += self.inflight * (now - self.changed)
        self.changed = now
        self.inflight += delta
        self.peak = max(self.peak, self.inflight)

    def check_runtime(self, runtime: str | None, node: str | None) -> None:
        if (runtime is not None and runtime != self.runtime) or (
            node is not None and node != self.node
        ):
            self.fatal = (
                f"A request was answered by {runtime or 'an unknown runtime'} "
                f"on {node or 'an unknown machine'}, "
                f"not {self.runtime} on {self.node}. The run stopped."
            )
            raise ValueError(self.fatal)

    async def turn(self, session: Session, *, counted: bool = True) -> str:
        observation = Observation(reuse=session.reuse if session.workload == "agents" else None)
        if counted:
            self.observations.append(observation)
            self.transition(1)
        else:
            self.warmups += 1
        start = time.perf_counter()
        try:
            async with self.client.stream(
                "POST",
                self.gateway + "/v1/chat/completions",
                headers={"Authorization": "Bearer " + self.token},
                json={
                    "model": self.model,
                    "messages": session.messages,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": session.output_limit(),
                    "seed": self.rng.randrange(2**31),
                    "prompt_cache_key": session.identity,
                },
            ) as response:
                if response.status_code != 200:
                    raw = (await response.aread()).decode(errors="replace")
                    raise ValueError(f"Gateway returned {response.status_code}: {raw[:600]}")
                runtime = response.headers.get("X-Eugene-Plexus-Runtime")
                node = response.headers.get("X-Eugene-Plexus-Node")
                self.check_runtime(runtime, node)
                if response.headers.get("X-Eugene-Plexus-Waited-Ms"):
                    observation.waited = float(response.headers["X-Eugene-Plexus-Waited-Ms"])
                done = False
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    frame = json.loads(data)
                    if frame.get("error"):
                        raise ValueError(f"Gateway stream error: {frame['error']}")
                    routing = frame.get("x_eugene_plexus") or {}
                    if "runtime" in routing or "node" in routing:
                        runtime = routing.get("runtime", runtime)
                        node = routing.get("node", node)
                        self.check_runtime(runtime, node)
                    if routing.get("waited_ms") is not None and observation.waited is None:
                        observation.waited = float(routing["waited_ms"])
                    if usage := frame.get("usage"):
                        observation.output = usage.get("completion_tokens")
                        observation.prompt = usage.get("prompt_tokens")
                        details = usage.get("prompt_tokens_details") or {}
                        observation.cached = details.get("cached_tokens")
                    for choice in frame.get("choices", []):
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            now = time.perf_counter()
                            if observation.ttft is None:
                                observation.ttft = (now - start) * 1000
                            observation.times.append(now)
                            observation.answer += content
                if not done:
                    raise ValueError("Gateway closed the stream before [DONE].")
                if observation.ttft is None:
                    raise ValueError("The engine returned no content tokens.")
                missing = [
                    field for field, value in (("runtime", runtime), ("node", node)) if not value
                ]
                if missing:
                    self.fatal = (
                        f"The gateway did not report the {' and '.join(missing)} "
                        "that answered. The run stopped."
                    )
                    raise ValueError(self.fatal)
                if observation.output is None or observation.prompt is None:
                    self.notes.add("The engine did not report complete token usage.")
                if observation.cached is None:
                    self.notes.add("The engine did not report cached token usage.")
                if observation.waited is None:
                    self.notes.add("The gateway did not report wait time.")
        except asyncio.CancelledError:
            observation.error = "The request was cancelled."
            raise
        except (ValueError, httpx.HTTPError) as exc:
            observation.error = str(exc) or type(exc).__name__
            if self.fatal or not counted:
                raise
        finally:
            if counted:
                self.transition(-1)
        return observation.answer

    async def level(
        self,
        people: int,
        seconds: float,
        workload: str,
        progress: Callable[[float], None] | None = None,
    ) -> ServingLevel:
        self.observations = []
        self.inflight = self.peak = 0
        self.area = self.lag = 0.0
        self.fatal = None
        self.warmups = 0
        self.started = self.changed = time.perf_counter()
        deadline = self.started + seconds
        tasks: list[asyncio.Task[None]] = []

        async def person(
            identity: str, *, sub: bool = False, session: Session | None = None
        ) -> None:
            rng = random.Random(self.rng.randrange(2**63))
            session = session or Session(rng, identity, workload, self.context, warm=not sub)
            turns = rng.randint(3, 8) if sub else None
            while time.perf_counter() < deadline and not self.fatal:
                answer = await self.turn(session)
                session.advance(answer)
                if self.fatal:
                    return
                if turns is not None:
                    turns -= 1
                    if not turns:
                        return
                if workload == "agents" and not sub and rng.random() < 0.05:
                    children = [
                        asyncio.create_task(person(f"{identity}/{rng.randrange(2**63)}", sub=True))
                        for _ in range(burst_size(rng))
                    ]
                    try:
                        await asyncio.gather(*children)
                    finally:
                        for child in children:
                            child.cancel()
                        await asyncio.gather(*children, return_exceptions=True)
                pause = (
                    pause_seconds(rng) if workload == "agents" else 40 if workload == "chat" else 0
                )
                await asyncio.sleep(min(pause, max(0, deadline - time.perf_counter())))

        async def monitor() -> None:
            while True:
                before = time.perf_counter()
                await asyncio.sleep(0.05)
                self.lag = max(self.lag, (time.perf_counter() - before - 0.05) * 1000)
                if progress:
                    progress(min(1.0, (time.perf_counter() - self.started) / seconds))

        watcher: asyncio.Task[None] | None = None
        try:
            sessions = [
                Session(
                    random.Random(self.rng.randrange(2**63)),
                    f"{people}/{n}/{self.rng.randrange(2**63)}",
                    workload,
                    self.context,
                )
                for n in range(people)
            ]
            if workload == "agents":

                async def warm(session: Session) -> None:
                    session.advance(await self.turn(session, counted=False))

                tasks = [asyncio.create_task(warm(session)) for session in sessions]
                await asyncio.gather(*tasks)
            self.started = self.changed = time.perf_counter()
            deadline = self.started + seconds
            watcher = asyncio.create_task(monitor())
            tasks = [
                asyncio.create_task(person(session.identity, session=session))
                for session in sessions
            ]
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if watcher:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
            self.partial = self.summarize(people)
        return self.partial

    def summarize(self, people: int) -> ServingLevel:
        duration = time.perf_counter() - self.started
        observations = self.observations
        ttft = [o.ttft for o in observations if o.ttft is not None]
        intervals = [
            (b - a) * 1000 for o in observations for a, b in zip(o.times, o.times[1:], strict=False)
        ]
        speeds = [
            (o.output - 1) / (o.times[-1] - o.times[0])
            for o in observations
            if o.output is not None and o.output > 1 and len(o.times) > 1
        ]
        waited = [o.waited for o in observations if o.waited is not None]
        reuse = [o.reuse for o in observations if o.reuse is not None]
        output = sum(o.output or 0 for o in observations)
        errors = Counter(o.error for o in observations if o.error)
        return ServingLevel(
            people=people,
            streamsMean=self.area / max(duration, 0.000001),
            streamsPeak=self.peak,
            requests=len(observations),
            errors=sum(errors.values()),
            errorCauses=dict(errors),
            ttftP50Ms=percentile(ttft, 0.5),
            ttftP90Ms=percentile(ttft, 0.9),
            itlP50Ms=percentile(intervals, 0.5),
            itlP90Ms=percentile(intervals, 0.9),
            tokensPerSecondEachP50=percentile(speeds, 0.5),
            tokensPerSecondEachP10=percentile(speeds, 0.1),
            tokensPerSecondTotal=output / max(duration, 0.000001)
            if observations and all(o.output is not None for o in observations)
            else None,
            promptTokens=sum(o.prompt or 0 for o in observations),
            cachedTokens=sum(o.cached or 0 for o in observations),
            waitedP50Ms=percentile(waited, 0.5),
            foreignRequests=None,
            watts=None,
            outputTokens=output,
            durationSeconds=duration,
            promptReuse=sum(reuse) / len(reuse) if reuse else None,
            loopLagMaxMs=self.lag,
            loopLagged=self.lag > 50,
        )


class ServingMeasurements:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.jobs: list[ServingMeasurement] = []
        self.keys: dict[str, dict[str, str]] = {}
        self.task: asyncio.Task[None] | None = None
        self.cleanup: Callable[[ServingMeasurement], Awaitable[None]] | None = None
        self.load_error: str | None = None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.jobs = [ServingMeasurement.model_validate(j) for j in data["measurements"]]
            self.keys = data.get("keys", {})
        except FileNotFoundError:
            pass
        except (ValueError, KeyError, OSError) as exc:
            self.load_error = f"Saved serving measurements could not be read: {exc}"
        for job in self.jobs:
            if job.state == BenchmarkState.running:
                job.state = BenchmarkState.failed
                job.finishedAt = datetime.now(UTC)
                job.detail = "The agent restarted during this measurement. The run failed."
                for restart in job.restarts:
                    if restart.state == "pending":
                        restart.state = "failed"  # type: ignore[assignment]
                        restart.detail = (
                            "The agent restarted before the replica restart was confirmed."
                        )
        if self.jobs:
            self.save()

    @property
    def active(self) -> bool:
        return self.task is not None and not self.task.done()

    def new(
        self, body: ServingMeasurementRequest, node: str, model: str, provenance: ServingProvenance
    ) -> ServingMeasurement:
        job = ServingMeasurement(
            id=uuid4().hex,
            node=node,
            runtime=body.runtime,
            model=model,
            state=BenchmarkState.running,
            request=body,
            startedAt=datetime.now(UTC),
            finishedAt=None,
            progress=0,
            detail="Preparing measurement.",
            levels=[],
            knee=None,
            provenance=provenance,
            provenanceKey=provenance_key(provenance),
            current=True,
            staleBecause=[],
            keyLive=False,
            restarts=[],
            drafting=ServingDrafting(type=None, model=None, acceptance=None),
        )
        self.jobs.append(job)
        return job

    def attach_key(self, job_id: str, key_id: str, expires: datetime) -> None:
        self.keys[job_id] = {"id": key_id, "expiresAt": expires.isoformat()}
        next(j for j in self.jobs if j.id == job_id).keyLive = True

    def save(self) -> None:
        keep = {j.id for j in self.jobs[-20:]}
        completed: dict[str, str] = {}
        for job in self.jobs:
            if job.state == BenchmarkState.completed:
                completed[job.runtime] = job.id
            if job.keyLive:
                keep.add(job.id)
        keep.update(completed.values())
        self.jobs = [j for j in self.jobs if j.id in keep]
        self.keys = {k: v for k, v in self.keys.items() if k in keep}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"measurements": [j.model_dump(mode="json") for j in self.jobs], "keys": self.keys},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def list(self, present: dict[str, ServingProvenance]) -> list[ServingMeasurement]:
        for job in self.jobs:
            current = present.get(job.runtime)
            job.current = current is not None and provenance_key(current) == job.provenanceKey
            job.staleBecause = []
            if current is None:
                job.staleBecause = [f"The runtime {job.runtime} is no longer present."]
            elif not job.current:
                for key, old in job.provenance.model_dump(mode="json").items():
                    new = current.model_dump(mode="json")[key]
                    if old != new:
                        label = "engine build" if key == "engineBuild" else key
                        job.staleBecause.append(f"The {label} changed: {old} → {new}.")
        return list(reversed(self.jobs))

    async def run(
        self,
        job: ServingMeasurement,
        sweep: Sweep,
        cleanup: Callable[[ServingMeasurement], Awaitable[None]],
        observe: Callable[[ServingLevel | None], Awaitable[None]],
    ) -> None:
        self.cleanup = cleanup
        levels = people_levels(job.request.maxPeople or 1)

        def keep(level: ServingLevel) -> None:
            job.levels = [old for old in job.levels if old.people != level.people] + [level]

        last_checkpoint = 0.0
        final, detail = BenchmarkState.failed, "The measurement stopped unexpectedly."
        try:
            for index, people in enumerate(levels):
                await observe(None)

                def progress(fraction: float, index: int = index, people: int = people) -> None:
                    nonlocal last_checkpoint
                    job.progress = (index + fraction) / len(levels)
                    job.detail = (
                        f"{people} people: {fraction * (job.request.windowSeconds or 0):.0f} "
                        f"of {job.request.windowSeconds} seconds."
                    )
                    if time.perf_counter() - last_checkpoint >= 1:
                        keep(sweep.summarize(people))
                        self.save()
                        last_checkpoint = time.perf_counter()

                level = await sweep.level(
                    people, job.request.windowSeconds or 300, str(job.request.workload), progress
                )
                await observe(level)
                keep(level)
                job.knee = knee_for(
                    job.levels,
                    job.request.ttftThresholdMs or 1000,
                    job.request.speedRatioThreshold or 2 / 3,
                )
                self.save()
                if level.requests and level.errors / level.requests >= 0.05:
                    causes = "; ".join(level.errorCauses)
                    raise ValueError(
                        f"Stopped at {people} people: {level.errors / level.requests:.0%} "
                        f"of requests failed ({causes})."
                    )
                if level.ttftP50Ms is not None and level.ttftP50Ms > 60000:
                    raise ValueError(
                        f"Stopped at {people} people: first token took "
                        f"{level.ttftP50Ms / 1000:.1f} seconds."
                    )
                if job.knee and job.knee.found:
                    break
            final = BenchmarkState.completed
            job.progress = 1
            detail = "Measurement completed. " + " ".join(sorted(sweep.notes))
        except asyncio.CancelledError:
            final = BenchmarkState.cancelled
            detail = "Measurement cancelled. Partial results were kept."
        except Exception as exc:
            final = BenchmarkState.failed
            detail = str(exc) or type(exc).__name__
        finally:
            if sweep.partial is not None:
                keep(sweep.partial)
            # The job ends only after the stopped replicas are started again,
            # so a finished job never shows a restart still pending.
            if job.restarts:
                job.detail = "Starting the stopped replicas again."

            def save_final() -> None:
                try:
                    self.save()
                except OSError as exc:
                    cause = f" The measurement record could not be saved: {exc}"
                    if cause not in job.detail:
                        job.detail += cause

            save_final()
            cleanup_error = ""
            try:
                await cleanup(job)
            except Exception as exc:
                cleanup_error = f" Cleanup failed: {exc}"
            finally:
                job.state = final
                job.detail = detail + cleanup_error
                job.finishedAt = datetime.now(UTC)
                save_final()

    async def close(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            await self.task
