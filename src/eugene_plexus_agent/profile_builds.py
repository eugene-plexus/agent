"""Profile builds: measure settings for one model on this node, write nothing.

`docs/design/profile-builder.md` is the design; §0 is why every rule below
exists. In one paragraph: llama.cpp's own `--fit` already places a model
well, MoE expert offload included, so a build never searches placement.
It chooses what fit does not — the context, the cache precision (only
within the accuracy level, whose quality cost it measures on THIS model,
because that cost varied ninefold between two models) and the memory margin —
asks fit where each candidate would go, measures each with llama-bench
using exactly that placement, and confirms the recommended one in
llama-server. The operator saves the result as a profile; the build never
writes one.

Two instrument facts from §0 are load-bearing here and each has a test:

* llama-bench's `-fitc` does NOT change where it places a model — its
  context is its own test's — so placement is computed with
  `llama-fit-params -c N` and passed to llama-bench explicitly (M3).
* The first request after a load prefills at a quarter of the steady
  rate; llama-bench's own warm-up run absorbs it, so it stays on (M3).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import re
import shutil
import socket
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from . import orphan_kill
from ._generated.models import (
    BenchmarkState,
    BuildCandidate,
    CacheQuality,
    CacheType,
    MeasurementRestart,
    MeasurementRestartState,
    ProfileBuild,
    ProfileBuildAccuracy,
    ProfileBuildPhase,
    ProfileBuildRequest,
)
from ._http import shared_internal_client
from .engines.llama_cpp import (
    CACHE_TYPE_KEY,
    FLASH_ATTENTION_KEY,
    flash_attention_argv,
    flash_attention_choice,
)

log = logging.getLogger(__name__)

TIMEOUT = 1800
MAX_HISTORY = 20
MAX_OUTPUT = 4 * 1024 * 1024

# The quality step (§3): four 4,096-token chunks, KL divergence and
# same-top-token against the f16 cache. At least two chunks are needed,
# which is llama-perplexity's own minimum at that context.
QUALITY_CONTEXT = 4096
QUALITY_CHUNKS = 4
MIN_EVALUATION_TOKENS = 2 * QUALITY_CONTEXT
# High was 96.5% until the bundled text proved harder than wikitext: the
# MoE model's 8-bit cache scored 96.32% ± 0.21 on it against 97.29% there
# (Troy lowered it, 2026-09-30). Low's other lever, a smaller file, is the
# page's (moe-aware-fit call C).
THRESHOLDS: dict[ProfileBuildAccuracy, float] = {
    ProfileBuildAccuracy.high: 96.0,
    ProfileBuildAccuracy.medium: 92.0,
    ProfileBuildAccuracy.low: 88.0,
}
# What each level may consider, most precise first. Max never measures.
CANDIDATE_TYPES: dict[ProfileBuildAccuracy, tuple[CacheType, ...]] = {
    ProfileBuildAccuracy.max: (CacheType.f16,),
    ProfileBuildAccuracy.high: (CacheType.f16, CacheType.q8_0),
    ProfileBuildAccuracy.medium: (CacheType.f16, CacheType.q8_0, CacheType.q4_0),
    ProfileBuildAccuracy.low: (CacheType.f16, CacheType.q8_0, CacheType.q4_0),
}
PRECISION_ORDER = (CacheType.f16, CacheType.q8_0, CacheType.q4_0)

# Contexts tried, capped at the model's trained context (§4).
LADDER = (4096, 8192, 16384, 32768, 65536, 131072)
MAX_CONTEXT = 262144
# Every candidate's second decode point is at this one depth, which the
# smallest rung can reach. **Candidates are compared at the same depth or
# the comparison measures depth, not settings**: the first CPU acceptance
# run (2026-09-30) measured each at half its own context, so 4k and 8k
# looked faster than 40k only for being read shallower, on a node where
# all five placed identically and a longer context costs nothing until
# it is filled. Depth-within-a-candidate is the R6.1 benchmark's job.
COMMON_DEPTH = 2048
PROMPT_TOKENS, GEN_TOKENS, REPETITIONS = 512, 128, 3
# The default choice: the longest context keeping this share of the
# fastest decode speed measured at the common depth (§5).
BALANCE = 0.8
# Two speeds this close are the same speed: llama-bench's own spread on
# decode was 0.2-1% in §0, and without a tolerance noise picks the winner.
SPEED_TOLERANCE = 0.03

# The quality baseline stores every scored token's distribution as 16-bit
# values. Bounded by the largest vocabulary in common use rather than read
# off the file, which the agent does not parse.
VOCAB_BOUND = 262144
BASELINE_BYTES = QUALITY_CHUNKS * (QUALITY_CONTEXT // 2) * VOCAB_BOUND * 2
DISK_NEEDED_BYTES = 2 * BASELINE_BYTES
# A custom text shorter than this almost certainly yields < 8,192 tokens.
# Only the preflight uses it; the build reports the real count.
LIKELY_TOO_SHORT_BYTES = 32 * 1024

# Placement arguments fit may print, and how llama-bench spells a list.
_FIT_VALUE_FLAGS = {
    "-c": "context",
    "--ctx-size": "context",
    "-ngl": "-ngl",
    "--n-gpu-layers": "-ngl",
    "-ot": "-ot",
    "--override-tensor": "-ot",
    "-ts": "-ts",
    "--tensor-split": "-ts",
    "-sm": "-sm",
    "--split-mode": "-sm",
    "-dev": "-dev",
    "--device": "-dev",
    "-mg": "-mg",
    "--main-gpu": "-mg",
    "-ncmoe": "-ncmoe",
    "--n-cpu-moe": "-ncmoe",
}
# llama-bench separates sweep values with commas, so a list inside one
# value uses its own separator: `;` between override patterns, `/` between
# devices and proportions.
_BENCH_LIST_SEPARATOR = {"-ot": ";", "-ts": "/", "-dev": "/"}

# Profile settings that are placement or loading inputs and so travel to
# every tool alike. `gpuLayers` is deliberately absent: a build leaves
# layers to fit, so a profile's own number is not measured.
_PASSTHROUGH_INT = {
    "threads": "--threads",
    "batchSize": "--batch-size",
    "ubatchSize": "--ubatch-size",
    "mainGpu": "--main-gpu",
}


class BuildError(ValueError):
    """A build cannot continue; the message is for the operator."""


# --------------------------------------------------------------------------- #
# Pure pieces
# --------------------------------------------------------------------------- #


def split_arguments(line: str) -> list[str]:
    """Split fit's printed arguments without treating backslashes as escapes.

    `-ot "blk\\.41\\.ffn_(gate|up).*=CPU,…"` is a regular expression full
    of backslashes; POSIX shell splitting would eat them.
    """
    out: list[str] = []
    current: list[str] = []
    quoted = False
    started = False
    for ch in line.strip():
        if ch == '"':
            quoted = not quoted
            started = True
        elif ch.isspace() and not quoted:
            if started:
                out.append("".join(current))
                current, started = [], False
        else:
            current.append(ch)
            started = True
    if quoted:
        raise BuildError("llama.cpp's fit printed an unterminated quoted argument.")
    if started:
        out.append("".join(current))
    return out


def parse_fit(stdout: str) -> tuple[int, list[str]]:
    """The context and placement fit printed on its last stdout line."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise BuildError("llama.cpp's fit printed no arguments.")
    tokens = split_arguments(lines[-1])
    context: int | None = None
    placement: list[str] = []
    i = 0
    while i < len(tokens):
        flag = tokens[i]
        role = _FIT_VALUE_FLAGS.get(flag)
        if role is None or i + 1 >= len(tokens):
            raise BuildError(f"llama.cpp's fit chose {flag!r}, which a build cannot carry.")
        value = tokens[i + 1]
        if role == "context":
            context = int(value)
        else:
            placement += [role, value]
        i += 2
    if context is None:
        raise BuildError("llama.cpp's fit printed no context size.")
    return context, placement


def bench_placement(placement: Sequence[str]) -> list[str]:
    """Fit's placement in llama-bench's spelling (lists re-separated)."""
    out: list[str] = []
    for flag, value in zip(placement[::2], placement[1::2], strict=True):
        separator = _BENCH_LIST_SEPARATOR.get(flag)
        out += [flag, value.replace(",", separator) if separator else value]
    return out


def quality_from_output(text: str, cache: CacheType, threshold: float) -> CacheQuality:
    """Read llama-perplexity's `--kl-divergence` report."""
    same = re.search(r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)\s*%", text)
    kld = re.search(r"Mean\s+KLD:\s*(-?[\d.]+)\s*±\s*([\d.]+)", text)
    chunks = re.search(r"computing over (\d+) chunks, n_ctx=(\d+)", text)
    if not same or not kld or not chunks:
        raise BuildError("llama-perplexity did not report a KL-divergence comparison.")
    percent, error = float(same.group(1)), float(same.group(2))
    n_chunks, n_ctx = int(chunks.group(1)), int(chunks.group(2))
    if not all(math.isfinite(v) for v in (percent, error)):
        raise BuildError("llama-perplexity reported a non-finite quality figure.")
    return CacheQuality(
        cacheType=cache,
        sameTopTokenPercent=percent,
        standardError=error,
        meanKld=max(0.0, float(kld.group(1))),
        tokensScored=n_chunks * (n_ctx // 2),
        passes=percent - error >= threshold,
    )


def too_short(text: str) -> tuple[int, int] | None:
    """(needed, found) when llama-perplexity refused a short text."""
    need = re.search(r"need at least (\d+) tokens", text)
    have = re.search(r"tokenizes to only (\d+) tokens", text)
    if need:
        return int(need.group(1)), int(have.group(1)) if have else 0
    return None


def contexts_for(trained: int | None) -> list[int]:
    """The ladder, capped at what the model was trained for."""
    cap = min(trained, MAX_CONTEXT) if trained else LADDER[-1]
    rungs = [c for c in LADDER if c <= cap]
    if trained and cap > (rungs[-1] if rungs else 0):
        rungs.append(cap)
    return rungs or [cap]


def deep_depth(context: int) -> int:
    return min(context // 2, COMMON_DEPTH)


def _speed(candidate: BuildCandidate) -> float:
    return candidate.deepDecodeTokensPerSecond or 0.0


def mark_frontier(candidates: list[BuildCandidate]) -> None:
    """On the frontier unless another is at least as long and not slower.

    "Not slower" allows SPEED_TOLERANCE, so on a node where every context
    places alike (all on the CPU, or everything fits) the longest context
    dominates the rest instead of noise choosing among equals. And at the
    same context and speed the more precise cache dominates the less
    precise one: the third GPU acceptance run kept f16 and q8_0 at 8k both,
    48.4 against 48.8 tok/s, where the smaller cache bought nothing but a
    quality cost.
    """
    measured = [c for c in candidates if c.deepDecodeTokensPerSecond]
    for c in candidates:
        c.onFrontier = False
    for c in measured:
        tolerance = SPEED_TOLERANCE * _speed(c)
        dominated = any(
            o is not c
            and o.contextSize >= c.contextSize
            and _speed(o) >= _speed(c) - tolerance
            and (
                o.contextSize > c.contextSize
                or _speed(o) > _speed(c) + tolerance
                or _precision(o) > _precision(c)
            )
            for o in measured
        )
        c.onFrontier = not dominated


def _precision(candidate: BuildCandidate) -> int:
    """Higher is more precise: f16 over q8_0 over q4_0."""
    return len(PRECISION_ORDER) - PRECISION_ORDER.index(CacheType(candidate.cacheType))


def recommend(candidates: Sequence[BuildCandidate]) -> int | None:
    """The longest frontier context keeping BALANCE of the fastest speed."""
    frontier = [
        (i, c) for i, c in enumerate(candidates) if c.onFrontier and c.deepDecodeTokensPerSecond
    ]
    if not frontier:
        return None
    fastest = max(c.deepDecodeTokensPerSecond or 0 for _, c in frontier)
    keeping = [
        (i, c) for i, c in frontier if (c.deepDecodeTokensPerSecond or 0) >= BALANCE * fastest
    ]
    return max(keeping, key=lambda pair: pair[1].contextSize)[0]


def passthrough_args(flags: dict[str, Any], *, bench: bool) -> list[str]:
    """Profile settings every tool must see alike."""
    out: list[str] = []
    for key, option in _PASSTHROUGH_INT.items():
        value = flags.get(key)
        if value is not None:
            out += [option, str(value)]
    if flags.get("devices"):
        devices = ",".join(part.strip() for part in str(flags["devices"]).split(","))
        out += ["--device", devices.replace(",", "/") if bench else devices]
    if flags.get("splitMode"):
        out += ["--split-mode", str(flags["splitMode"])]
    if flags.get("tensorSplit"):
        split = ",".join(part.strip() for part in str(flags["tensorSplit"]).split(","))
        out += ["--tensor-split", split.replace(",", "/") if bench else split]
    if flags.get("noMmap") or flags.get("mlock"):
        mode = (
            "mlock"
            if flags.get("noMmap") and flags.get("mlock")
            else ("none" if flags.get("noMmap") else "mmap+mlock")
        )
        out += ["--load-mode", mode]
    return out


def cache_args(cache: CacheType, flash_attention: str | None) -> list[str]:
    args = ["--cache-type-k", cache.value, "--cache-type-v", cache.value]
    # The launch's own rule (agent#6): a quantised cache needs it on, and an
    # f16 trial follows the profile, so the build measures what will run.
    return args + flash_attention_argv(
        {CACHE_TYPE_KEY: cache.value, FLASH_ATTENTION_KEY: flash_attention}
    )


def validate_request(body: ProfileBuildRequest) -> dict[str, Any]:
    """The profile flags a build can carry, or a BuildError saying why not."""
    spec = body.runtime
    if spec.engine.value != "llama_cpp":
        raise BuildError(
            "Settings builds use llama.cpp's own tools and are only available for "
            f"llama_cpp runtimes; this profile declares {spec.engine.value!r}."
        )
    if spec.extraArgs:
        raise BuildError(
            "A settings build cannot reproduce raw extraArgs. Use a profile with supported fields."
        )
    flags = {k: v for k, v in (spec.flags or {}).items() if v is not None}
    parallel = flags.get("parallelSlots", 1)
    if type(parallel) is not int or parallel != 1:
        raise BuildError("A settings build measures one conversation at a time. Use one slot.")
    return flags


# --------------------------------------------------------------------------- #
# The plan a route hands the runner
# --------------------------------------------------------------------------- #


@dataclass
class Tools:
    server: Path
    bench: Path
    fit: Path
    perplexity: Path
    version: str | None


@dataclass
class BuildPlan:
    tools: Tools
    model: Path
    work: Path
    text: Path
    flags: dict[str, Any]
    contexts: list[int]
    margin: int | None
    env: dict[str, str]
    detect: Callable[[], Any] | None = None
    stopped: list[str] = field(default_factory=list)


AfterJob = Callable[[], Awaitable[list[MeasurementRestart]]]

INTERRUPTED_RESTART = (
    "The agent restarted during the build. A model set to start automatically "
    "came back with it; start any other from its page."
)


def bundled_text() -> Path:
    return Path(str(files("eugene_plexus_agent").joinpath("data/evaluation_text.txt")))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


class ProfileBuilds:
    """One build at a time on this node, with a persisted history."""

    def __init__(self, path: Path, *, timeout: float = TIMEOUT) -> None:
        self.path = path
        self.timeout = timeout
        self.jobs: list[ProfileBuild] = []
        self.task: asyncio.Task[None] | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.stopping = False
        self.shutting_down = False
        self.after: AfterJob | None = None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.jobs = [ProfileBuild.model_validate(j) for j in raw][-MAX_HISTORY:]
        except FileNotFoundError:
            pass
        except (ValueError, OSError):
            log.exception("Cannot read profile build history %s; starting empty", path)
        interrupted = False
        for job in self.jobs:
            if job.state == BenchmarkState.running:
                job.state = BenchmarkState.failed
                job.phase = ProfileBuildPhase.finished
                job.detail = "The agent restarted before the build finished. Start a new build."
                job.finishedAt = datetime.now(UTC)
                for restart in job.restarts:
                    if restart.state == MeasurementRestartState.pending:
                        restart.state = MeasurementRestartState.skipped
                        restart.detail = INTERRUPTED_RESTART
                interrupted = True
        if interrupted:
            with contextlib.suppress(OSError):
                self._save()

    @property
    def active(self) -> bool:
        return self.task is not None and not self.task.done()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps([j.model_dump(mode="json") for j in self.jobs]), encoding="utf-8"
        )
        temporary.replace(self.path)

    def start(
        self,
        body: ProfileBuildRequest,
        plan: BuildPlan,
        *,
        node: str,
        restarts: list[MeasurementRestart],
        evaluation: dict[str, Any],
        after: AfterJob | None,
    ) -> ProfileBuild:
        if self.active:
            raise BuildError("A settings build is already running on this node.")
        stat = plan.model.stat()
        job = ProfileBuild.model_validate(
            {
                "id": uuid4().hex,
                "node": node,
                "modelId": body.modelId,
                "profileId": body.profileId,
                "runtime": body.runtime.model_dump(mode="json"),
                "accuracy": body.accuracy.value,
                "memoryMarginMiB": body.memoryMarginMiB,
                "evaluation": evaluation,
                "state": "running",
                "phase": "quality"
                if body.accuracy is not ProfileBuildAccuracy.max
                else "candidates",
                "startedAt": datetime.now(UTC),
                "progress": 0,
                "detail": "Starting…",
                "quality": [],
                "allowedCacheTypes": [CacheType.f16.value],
                "candidates": [],
                "recommended": None,
                "engineVersion": plan.tools.version,
                "localPath": str(plan.model),
                "modelSizeBytes": stat.st_size,
                "hardware": {},
                "restarts": [r.model_dump(mode="json") for r in restarts],
            }
        )
        previous = self.jobs
        self.jobs = [*self.jobs, job][-MAX_HISTORY:]
        try:
            self._save()
        except OSError:
            self.jobs = previous
            raise
        self.stopping = False
        self.after = after
        self.task = asyncio.create_task(self._run(job, body, plan))
        return job

    async def cancel(self, job_id: str) -> ProfileBuild | None:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if job is not None and job.state == BenchmarkState.running:
            self.stopping = True
            await self._terminate()
            if self.task is not None:
                await asyncio.shield(self.task)
        return job

    async def close(self) -> None:
        self.shutting_down = True
        if self.active:
            await self.cancel(self.jobs[-1].id)

    async def _terminate(self) -> None:
        proc = self.process
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()

    # -- children ------------------------------------------------------------

    async def _child(
        self,
        argv: list[str],
        plan: BuildPlan,
        *,
        on_stderr: Callable[[str], None] | None = None,
    ) -> tuple[int, str, str]:
        """Run one tool to completion; its output, bounded."""
        if self.stopping:
            raise asyncio.CancelledError
        self.process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=plan.env,
            cwd=str(Path(argv[0]).parent),
            limit=MAX_OUTPUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            **orphan_kill.kwargs_for_platform(),
        )
        win_job = orphan_kill.windows_job()
        if win_job is not None:
            win_job.assign(self.process.pid)
        out: list[str] = []
        err: list[str] = []
        size = 0

        async def read(stream: asyncio.StreamReader, sink: list[str], hook) -> None:  # type: ignore[no-untyped-def]
            nonlocal size
            while line := await stream.readline():
                size += len(line)
                if size > MAX_OUTPUT:
                    raise BuildError(
                        f"{Path(argv[0]).name} printed more output than a build keeps."
                    )
                text = line.decode("utf-8", errors="replace")
                sink.append(text)
                if hook is not None:
                    hook(text.strip())

        assert self.process.stdout is not None and self.process.stderr is not None
        try:
            await asyncio.gather(
                read(self.process.stdout, out, None),
                read(self.process.stderr, err, on_stderr),
            )
            code = await self.process.wait()
        finally:
            if self.process.returncode is None:
                await self._terminate()
            self.process = None
        if self.stopping:
            raise asyncio.CancelledError
        return code, "".join(out), "".join(err)

    # -- phases --------------------------------------------------------------

    async def _quality(self, job: ProfileBuild, body: ProfileBuildRequest, plan: BuildPlan) -> None:
        threshold = THRESHOLDS[body.accuracy]
        lower = [c for c in CANDIDATE_TYPES[body.accuracy] if c is not CacheType.f16]
        # The job's own folder, removed when it ends; the route creates it
        # only for a custom text, so the baseline cannot assume it exists.
        plan.work.mkdir(parents=True, exist_ok=True)
        base = plan.work / "baseline.kld"
        common = [
            str(plan.tools.perplexity),
            "--model",
            str(plan.model),
            "--file",
            str(plan.text),
            "--ctx-size",
            str(QUALITY_CONTEXT),
            "--chunks",
            str(QUALITY_CHUNKS),
            *(["--fit-target", str(plan.margin)] if plan.margin is not None else []),
            *passthrough_args(plan.flags, bench=False),
        ]
        steps = 1 + len(lower)
        job.detail = "Measuring the full-precision baseline on the evaluation text…"
        self._save()
        code, out, err = await self._child(
            [*common, *cache_args(CacheType.f16, "on"), "--kl-divergence-base", str(base)], plan
        )
        if short := too_short(out + err):
            need, have = short
            job.evaluation.tokens = have
            raise BuildError(
                f"The evaluation text gives {have:,} tokens for this model; "
                f"the quality check needs at least {need:,}. Use a longer text."
            )
        if code != 0 or not base.is_file():
            raise BuildError(f"The quality baseline failed (exit {code}). {_tail(err)}")
        job.progress = 1 / steps * 0.3
        allowed = [CacheType.f16]
        for i, cache in enumerate(lower, start=2):
            job.detail = f"Measuring what the {cache.value} cache changes…"
            self._save()
            code, out, err = await self._child(
                [
                    *common,
                    *cache_args(cache, "on"),
                    "--kl-divergence-base",
                    str(base),
                    "--kl-divergence",
                ],
                plan,
            )
            if code != 0:
                raise BuildError(
                    f"The {cache.value} quality run failed (exit {code}). {_tail(err)}"
                )
            quality = quality_from_output(out + err, cache, threshold)
            job.quality.append(quality)
            if quality.passes:
                allowed.append(cache)
            job.progress = i / steps * 0.3
            self._save()
        with contextlib.suppress(OSError):
            base.unlink()
        job.allowedCacheTypes = allowed

    async def _candidates(self, job: ProfileBuild, plan: BuildPlan) -> None:
        allowed = [CacheType(c) for c in (job.allowedCacheTypes or [CacheType.f16])]
        allowed.sort(key=PRECISION_ORDER.index)
        job.phase = ProfileBuildPhase.candidates
        found: list[BuildCandidate] = []
        for n, context in enumerate(plan.contexts, start=1):
            job.detail = f"Asking llama.cpp where a {context:,}-token context would go…"
            placements: dict[CacheType, list[str]] = {}
            for cache in allowed:
                argv = [
                    str(plan.tools.fit),
                    "--model",
                    str(plan.model),
                    "--ctx-size",
                    str(context),
                    *cache_args(cache, flash_attention_choice(plan.flags)),
                    *(["--fit-target", str(plan.margin)] if plan.margin is not None else []),
                    *passthrough_args(plan.flags, bench=False),
                ]
                code, out, err = await self._child(argv, plan)
                if code != 0:
                    log.info(
                        "fit could not place %d tokens with %s: %s", context, cache, _tail(err)
                    )
                    continue
                try:
                    fitted, placement = parse_fit(out)
                except BuildError as exc:
                    log.info("fit output for %d tokens unreadable: %s", context, exc)
                    continue
                if fitted < context:
                    # Fit lowered the context to place the model: this
                    # context cannot be had here, only a shorter one.
                    continue
                placements[cache] = placement
            kept = _distinct_placements(placements)
            for cache in kept:
                found.append(
                    BuildCandidate(
                        contextSize=context,
                        cacheType=cache,
                        placement=placements[cache],
                        onFrontier=False,
                    )
                )
            job.candidates = list(found)
            job.progress = 0.3 + 0.1 * n / len(plan.contexts)
            self._save()
        if not found:
            raise BuildError(
                "llama.cpp's fit could not place this model at any context on this node, "
                "even with part of it in system memory."
            )

    async def _measure(self, job: ProfileBuild, plan: BuildPlan) -> None:
        job.phase = ProfileBuildPhase.measuring
        total = len(job.candidates)
        for n, candidate in enumerate(job.candidates, start=1):
            depth = deep_depth(candidate.contextSize)
            job.detail = (
                f"Measuring {candidate.contextSize:,} tokens with the "
                f"{candidate.cacheType.value} cache ({n} of {total})…"
            )
            self._save()
            model = str(plan.model)
            argv = [
                str(plan.tools.bench),
                "--model",
                model,
                "--n-prompt",
                str(PROMPT_TOKENS),
                "--n-gen",
                str(GEN_TOKENS),
                "--n-depth",
                f"0,{depth}",
                "--repetitions",
                str(REPETITIONS),
                "--output",
                "jsonl",
                *cache_args(candidate.cacheType, flash_attention_choice(plan.flags)),
                *bench_placement(candidate.placement),
                *passthrough_args(plan.flags, bench=True),
            ]
            code, out, err = await self._child(argv, plan)
            if code != 0:
                candidate.detail = (
                    f"llama-bench failed on this candidate (exit {code}). {_tail(err)}"
                )
                self._save()
                continue
            try:
                self._read_bench(candidate, out, depth, job)
            except BuildError as exc:
                candidate.detail = str(exc)
            job.progress = 0.4 + 0.45 * n / total
            self._save()
        mark_frontier(job.candidates)
        job.recommended = recommend(job.candidates)
        if job.recommended is None:
            raise BuildError("No candidate could be measured on this node.")

    def _read_bench(
        self, candidate: BuildCandidate, out: str, depth: int, job: ProfileBuild
    ) -> None:
        seen: dict[tuple[int, int, int], float] = {}
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            key = (raw.get("n_prompt"), raw.get("n_gen"), raw.get("n_depth"))
            rate = raw.get("avg_ts")
            if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
                raise BuildError("llama-bench returned an invalid speed.")
            seen[key] = float(rate)
            job.hardware = {
                k: str(raw[k])
                for k in ("cpu_info", "gpu_info", "backends", "build_commit", "build_number")
                if k in raw
            }
        want = {
            "decode": (0, GEN_TOKENS, 0),
            "deep": (0, GEN_TOKENS, depth),
            "prefill": (PROMPT_TOKENS, 0, 0),
        }
        if missing := [name for name, key in want.items() if key not in seen]:
            raise BuildError("llama-bench did not report " + ", ".join(missing) + ".")
        candidate.decodeTokensPerSecond = seen[want["decode"]]
        candidate.deepDepth = depth
        candidate.deepDecodeTokensPerSecond = seen[want["deep"]]
        candidate.prefillTokensPerSecond = seen[want["prefill"]]

    async def _confirm(self, job: ProfileBuild, plan: BuildPlan) -> None:
        job.phase = ProfileBuildPhase.confirming
        order = [job.recommended] if job.recommended is not None else []
        order += [
            i
            for i, c in sorted(
                enumerate(job.candidates), key=lambda p: -(p[1].deepDecodeTokensPerSecond or 0)
            )
            if c.onFrontier and i not in order
        ]
        for index in order[:2]:
            candidate = job.candidates[index]
            job.detail = (
                f"Loading {candidate.contextSize:,} tokens in llama-server to confirm it serves…"
            )
            self._save()
            ok, detail, used = await self._serve_once(candidate, plan)
            candidate.confirmed = ok
            candidate.graphicsMemoryBytes = used
            candidate.detail = detail
            if ok:
                job.recommended = index
                self._save()
                return
            candidate.onFrontier = False
            self._save()
        job.recommended = None
        raise BuildError(
            "llama-server could not load the measured settings on this node; "
            "each candidate's reason is kept with it."
        )

    async def _serve_once(
        self, candidate: BuildCandidate, plan: BuildPlan
    ) -> tuple[bool, str, int | None]:
        before = await asyncio.to_thread(_accelerator_free, plan.detect)
        port = free_port()
        argv = [
            str(plan.tools.server),
            "--model",
            str(plan.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(candidate.contextSize),
            "--parallel",
            "1",
            *cache_args(candidate.cacheType, flash_attention_choice(plan.flags)),
            *(["--fit-target", str(plan.margin)] if plan.margin is not None else []),
            *passthrough_args(plan.flags, bench=False),
        ]
        self.process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=plan.env,
            cwd=str(plan.tools.server.parent),
            limit=MAX_OUTPUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            **orphan_kill.kwargs_for_platform(),
        )
        win_job = orphan_kill.windows_job()
        if win_job is not None:
            win_job.assign(self.process.pid)
        tail: list[str] = []

        async def drain() -> None:
            assert self.process is not None and self.process.stderr is not None
            while line := await self.process.stderr.readline():
                tail.append(line.decode("utf-8", errors="replace"))
                del tail[:-40]

        reader = asyncio.create_task(drain())
        try:
            base = f"http://127.0.0.1:{port}"
            client = shared_internal_client("profile-builds", timeout=10)
            deadline = time.perf_counter() + 600
            while True:
                if self.stopping:
                    raise asyncio.CancelledError
                if self.process.returncode is not None:
                    return False, f"llama-server exited while loading. {_tail(''.join(tail))}", None
                try:
                    health = await client.get(f"{base}/health")
                    if health.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() > deadline:
                    return False, "llama-server did not finish loading within 10 minutes.", None
                await asyncio.sleep(1)
            used = _used_since(before, await asyncio.to_thread(_accelerator_free, plan.detect))
            reply = await client.post(
                f"{base}/completion", json={"prompt": "Hello", "n_predict": 8}, timeout=120
            )
            if reply.status_code != 200:
                return False, f"llama-server answered {reply.status_code} to a short request.", used
            return True, "Loaded and answered in llama-server.", used
        finally:
            await self._terminate()
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            self.process = None

    # -- the whole run ---------------------------------------------------------

    async def _run(self, job: ProfileBuild, body: ProfileBuildRequest, plan: BuildPlan) -> None:
        try:
            async with asyncio.timeout(self.timeout):
                if body.accuracy is not ProfileBuildAccuracy.max:
                    job.phase = ProfileBuildPhase.quality
                    await self._quality(job, body, plan)
                await self._candidates(job, plan)
                await self._measure(job, plan)
                await self._confirm(job, plan)
            job.state = BenchmarkState.completed
            job.progress = 1
            chosen = job.candidates[job.recommended] if job.recommended is not None else None
            job.detail = (
                f"Measured {sum(1 for c in job.candidates if c.decodeTokensPerSecond)} settings. "
                + (
                    f"Suggested: {chosen.contextSize:,} tokens, {chosen.cacheType.value} cache."
                    if chosen
                    else ""
                )
            ).strip()
        except TimeoutError:
            job.state = BenchmarkState.failed
            job.detail = "The build reached its 30-minute limit. What it measured is kept."
        except BuildError as exc:
            job.state = BenchmarkState.failed
            job.detail = str(exc)
        except (OSError, ValueError) as exc:
            job.state = BenchmarkState.failed
            job.detail = str(exc)
        except asyncio.CancelledError:
            self.stopping = True
        except Exception:
            log.exception("Unexpected profile build failure")
            job.state = BenchmarkState.failed
            job.detail = "The build failed unexpectedly; check this node's agent log."
        finally:
            await self._terminate()
            self.process = None
            if self.stopping:
                job.state = BenchmarkState.cancelled
                job.detail = "Build cancelled. What it measured is kept."
            job.phase = ProfileBuildPhase.finished
            job.finishedAt = datetime.now(UTC)
            with contextlib.suppress(OSError):
                shutil.rmtree(plan.work, ignore_errors=True)
            try:
                self._save()
            except OSError:
                log.exception("Could not save the finished profile build")
            await self._restart_after(job)

    async def _restart_after(self, job: ProfileBuild) -> None:
        after, self.after = self.after, None
        if after is None:
            return
        if self.shutting_down:
            for restart in job.restarts:
                if restart.state == MeasurementRestartState.pending:
                    restart.state = MeasurementRestartState.skipped
                    restart.detail = INTERRUPTED_RESTART
        else:
            try:
                job.restarts = await after()
            except Exception:
                log.exception("Restarting what a profile build stopped failed")
        with contextlib.suppress(OSError):
            self._save()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _distinct_placements(placements: dict[CacheType, list[str]]) -> list[CacheType]:
    """Keep the most precise cache, plus any smaller one that places differently.

    Two caches placed identically run at the same speed (M4, 4k), so the
    less precise one would only cost quality. A smaller cache that moves
    weights back onto the GPU (M4, 64k) is a real alternative.
    """
    ordered = sorted(placements, key=PRECISION_ORDER.index)
    kept: list[CacheType] = []
    for cache in ordered:
        if not any(placements[k] == placements[cache] for k in kept):
            kept.append(cache)
    return kept


def _tail(text: str, limit: int = 400) -> str:
    text = text.strip()
    return text[-limit:] if text else ""


def _accelerator_free(detect: Callable[[], Any] | None) -> list[int | None]:
    if detect is None:
        return []
    try:
        snapshot = detect()
        return [d.memoryFreeBytes for d in snapshot.accelerators() if not d.sharedMemory]
    except Exception:
        log.debug("could not read device memory for a build", exc_info=True)
        return []


def _used_since(before: list[int | None], after: list[int | None]) -> int | None:
    if not before or len(before) != len(after):
        return None
    total = 0
    for b, a in zip(before, after, strict=True):
        if b is None or a is None:
            return None
        total += max(0, b - a)
    return total


def evaluation_record(text: str | None) -> dict[str, Any]:
    """Which text the quality step used; a custom one is identified, never stored."""
    if text is None:
        return {"source": "bundled", "sha256": None, "tokens": None}
    return {
        "source": "custom",
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "tokens": None,
    }
