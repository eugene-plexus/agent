"""Owned, bounded llama-bench jobs. No runtime or Library state is mutated."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import orphan_kill
from ._generated.models import Benchmark, BenchmarkPoint, BenchmarkRequest, BenchmarkState
from .child_env import child_environment

log = logging.getLogger(__name__)
TIMEOUT = 900
MAX_HISTORY = 20
MAX_OUTPUT = 1024 * 1024
_PROGRESS = re.compile(r"benchmark (\d+)/(\d+): (?:depth|prompt|generation) run (\d+)/(\d+)")
_INTEGER_FLAGS = {
    "gpuLayers": ("--n-gpu-layers", 0),
    "threads": ("--threads", 1),
    "batchSize": ("--batch-size", 1),
    "ubatchSize": ("--ubatch-size", 1),
    "mainGpu": ("--main-gpu", 0),
}
_ENV_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
}


def benchmark_args(
    body: BenchmarkRequest, binary: Path, model: str, help_text: str | None = None
) -> tuple[list[str], list[int]]:
    spec = body.runtime
    flags = {key: value for key, value in (spec.flags or {}).items() if value is not None}
    if spec.engine.value != "llama_cpp":
        raise ValueError("Benchmark currently supports llama.cpp profiles only.")
    if spec.extraArgs:
        raise ValueError(
            "Benchmark cannot reproduce raw extraArgs. Use a profile with supported fields."
        )
    if set(spec.env or {}) - _ENV_KEYS:
        raise ValueError("Benchmark supports only GPU visibility environment settings.")
    known = set(_INTEGER_FLAGS) | {
        "contextSize",
        "parallelSlots",
        "flashAttention",
        "continuousBatching",
        "tensorSplit",
        "noMmap",
        "mlock",
    }
    if set(flags) - known:
        raise ValueError("Unsupported benchmark settings: " + ", ".join(sorted(set(flags) - known)))
    context = flags.get("contextSize")
    if type(context) is not int or not 256 <= context <= 262144:
        raise ValueError(
            "Set a context size between 256 and 262144 in this profile before benchmarking."
        )
    if type(flags.get("parallelSlots", 1)) is not int or flags.get("parallelSlots", 1) != 1:
        raise ValueError("llama-bench measures one sequence. Use a profile with one parallel slot.")
    for key in ("flashAttention", "continuousBatching", "noMmap", "mlock"):
        if key in flags and type(flags[key]) is not bool:
            raise ValueError(f"{key} must be a boolean.")
    tokens = body.tokens or 128
    if tokens >= context:
        raise ValueError("Generated tokens must be fewer than the profile's context size.")
    depths = sorted({0, (context - tokens) // 2, context - tokens})
    if "," in model:
        raise ValueError(
            "llama-bench treats commas in model paths as a sweep; rename this file to benchmark it."
        )
    argv = [
        str(binary),
        "--model",
        model,
        "--n-prompt",
        "0",
        "--n-gen",
        str(tokens),
        "--n-depth",
        ",".join(map(str, depths)),
        "--repetitions",
        str(body.repetitions or 3),
        "--output",
        "jsonl",
        "--progress",
    ]
    required = {"--n-depth", "--output", "--progress"}
    for key, (option, minimum) in _INTEGER_FLAGS.items():
        if key not in flags:
            continue
        value = flags[key]
        if type(value) is not int or not minimum <= value <= 1000000:
            raise ValueError(f"{key} must be an integer at least {minimum} (not a sweep).")
        argv += [option, str(value)]
        required.add(option)
    if "tensorSplit" in flags:
        value = flags["tensorSplit"]
        if not isinstance(value, str) or not re.fullmatch(
            r"\s*\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*\s*", value
        ):
            raise ValueError("tensorSplit must contain comma-separated nonnegative proportions.")
        argv += ["--tensor-split", "/".join(part.strip() for part in value.split(","))]
        required.add("--tensor-split")
    # The runtime adapter emits presence-only booleans: false leaves the
    # engine default alone. Do the same here instead of forcing a different mode.
    if flags.get("flashAttention"):
        argv += ["--flash-attn", "on"]
        required.add("--flash-attn")
    if flags.get("noMmap") or flags.get("mlock"):
        mode = (
            "mlock"
            if flags.get("noMmap") and flags.get("mlock")
            else ("none" if flags.get("noMmap") else "mmap+mlock")
        )
        argv += ["--load-mode", mode]
        required.add("--load-mode")
    if help_text is not None:
        supported = set(re.findall(r"--[a-z][a-z0-9-]*", help_text))
        if missing := required - supported:
            raise ValueError(
                "Installed llama-bench lacks " + ", ".join(sorted(missing)) + "; update llama.cpp."
            )
    return argv, depths


def parse_point(
    raw: dict[str, Any], depths: list[int], tokens: int, repetitions: int
) -> BenchmarkPoint:
    if (
        any(type(raw.get(key)) is not int for key in ("n_prompt", "n_gen", "n_depth"))
        or raw.get("n_prompt") != 0
        or raw.get("n_gen") != tokens
        or raw.get("n_depth") not in depths
    ):
        raise ValueError("Benchmark output does not match the requested decode sweep.")
    samples = raw.get("samples_ts")
    if not isinstance(samples, list) or len(samples) != repetitions:
        raise ValueError("Benchmark returned the wrong number of samples.")
    mean, deviation = raw.get("avg_ts"), raw.get("stddev_ts")
    values = [mean, deviation, *samples]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Benchmark returned nonfinite or missing timing values.")
    assert mean is not None and deviation is not None
    if mean <= 0 or deviation < 0 or any(v <= 0 for v in samples):
        raise ValueError("Benchmark returned invalid timing values.")
    return BenchmarkPoint.model_validate(
        {
            "depth": raw["n_depth"],
            "tokensPerSecond": mean,
            "standardDeviation": deviation,
            "samples": samples,
        }
    )


class Benchmarks:
    def __init__(self, path: Path, *, timeout: float = TIMEOUT) -> None:
        self.path = path
        self.timeout = timeout
        self.jobs: list[Benchmark] = []
        self.task: asyncio.Task[None] | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.stopping = False
        try:
            self.jobs = [
                Benchmark.model_validate(j) for j in json.loads(path.read_text(encoding="utf-8"))
            ][-MAX_HISTORY:]
        except FileNotFoundError:
            pass
        except (ValueError, OSError):
            log.exception("Cannot read benchmark history %s; starting with an empty history", path)
        interrupted = False
        for job in self.jobs:
            if job.state == BenchmarkState.running:
                job.state = BenchmarkState.failed
                job.detail = "Agent restarted before the benchmark finished. Start a new benchmark."
                job.finishedAt = datetime.now(UTC)
                interrupted = True
        if interrupted:
            try:
                self._save()
            except OSError:
                log.exception("Could not persist interrupted benchmark history")

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
        body: BenchmarkRequest,
        *,
        node: str,
        binary: Path,
        version: str | None,
        model: Path,
        argv: list[str],
        depths: list[int],
    ) -> Benchmark:
        if self.active:
            raise ValueError("A benchmark is already running on this node.")
        stat = model.stat()
        job = Benchmark.model_validate(
            {
                "id": uuid4().hex,
                "request": body.model_dump(mode="json"),
                "node": node,
                "state": "running",
                "startedAt": datetime.now(UTC),
                "progress": 0,
                "detail": "Loading model for the context-depth sweep…",
                "depths": depths,
                "points": [],
                "binary": str(binary),
                "engineVersion": version,
                "localPath": str(model),
                "modelSizeBytes": stat.st_size,
                "modelModifiedAt": datetime.fromtimestamp(stat.st_mtime, UTC),
                "command": argv,
                "hardware": {},
            }
        )
        previous = self.jobs
        self.jobs = ([*self.jobs, job])[-MAX_HISTORY:]
        try:
            self._save()  # An unwritable history refuses before any process exists.
        except OSError:
            self.jobs = previous
            raise
        self.stopping = False
        self.task = asyncio.create_task(self._run(job, argv))
        return job

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

    async def cancel(self, job_id: str) -> Benchmark | None:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if job is not None and job.state == BenchmarkState.running:
            self.stopping = True
            await self._terminate()
            if self.task is not None:
                await asyncio.shield(self.task)
        return job

    async def close(self) -> None:
        if self.active:
            await self.cancel(self.jobs[-1].id)

    async def _run(self, job: Benchmark, argv: list[str]) -> None:
        readers: list[asyncio.Task[None]] = []
        tail = ""
        output_bytes = 0
        depths = [d.root for d in job.depths]

        async def read(stream: asyncio.StreamReader, *, results: bool) -> None:
            nonlocal tail, output_bytes
            while line := await stream.readline():
                output_bytes += len(line)
                if output_bytes > MAX_OUTPUT:
                    raise ValueError("Benchmark output exceeded its limit.")
                text = line.decode("utf-8", errors="replace").strip()
                if results and text:
                    raw = json.loads(text)
                    if not isinstance(raw, dict):
                        raise ValueError("Benchmark output was not a result object.")
                    point = parse_point(
                        raw, depths, job.request.tokens or 128, job.request.repetitions or 3
                    )
                    if any(p.depth == point.depth for p in job.points):
                        raise ValueError("Benchmark returned a duplicate depth.")
                    job.points.append(point)
                    job.points.sort(key=lambda p: p.depth)
                    job.hardware = {
                        key: str(raw[key])
                        for key in (
                            "cpu_info",
                            "gpu_info",
                            "backends",
                            "build_commit",
                            "build_number",
                        )
                        if key in raw
                    }
                    job.progress = min(0.99, len(job.points) / len(depths))
                    self._save()
                elif not results:
                    tail = (tail + "\n" + text)[-2000:]
                    match = _PROGRESS.search(text)
                    if match:
                        test, total, repetition, repetitions = map(int, match.groups())
                        if total != len(depths) or repetitions != (job.request.repetitions or 3):
                            raise ValueError("Benchmark progress does not match this sweep.")
                        if 1 <= test <= total and 1 <= repetition <= repetitions:
                            job.progress = max(
                                job.progress,
                                min(0.99, ((test - 1) + (repetition - 1) / repetitions) / total),
                            )
                            job.detail = (
                                f"Depth {depths[test - 1]} tokens · "
                                f"repetition {repetition}/{repetitions}"
                            )

        try:
            if self.stopping:
                return
            env = child_environment()
            env.update(job.request.runtime.env or {})
            self.process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(Path(argv[0]).parent),
                limit=MAX_OUTPUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                **orphan_kill.kwargs_for_platform(),
            )
            win_job = orphan_kill.windows_job()
            if win_job is not None:
                win_job.assign(self.process.pid)
            if self.stopping:
                await self._terminate()
            assert self.process.stdout is not None and self.process.stderr is not None
            readers = [
                asyncio.create_task(read(self.process.stdout, results=True)),
                asyncio.create_task(read(self.process.stderr, results=False)),
            ]
            async with asyncio.timeout(self.timeout):
                await asyncio.gather(*readers)
                code = await self.process.wait()
            if self.stopping:
                return
            if code != 0:
                raise ValueError(f"llama-bench exited with code {code}. {tail.strip()}")
            if {p.depth for p in job.points} != set(depths):
                raise ValueError("Benchmark ended without results for every requested depth.")
            job.state = BenchmarkState.completed
            job.progress = 1
            job.detail = (
                "Context-depth sweep complete. Tokenization, sampling "
                "and parallel serving are not measured."
            )
        except TimeoutError:
            job.state = BenchmarkState.failed
            job.detail = "Benchmark reached the 15-minute time limit. Try a shorter context."
        except (OSError, ValueError) as exc:
            job.state = BenchmarkState.failed
            job.detail = str(exc)
        except asyncio.CancelledError:
            self.stopping = True
        except Exception:
            log.exception("Unexpected benchmark failure")
            job.state = BenchmarkState.failed
            job.detail = "Benchmark failed unexpectedly; check this node's agent log."
        finally:
            await self._terminate()
            for reader in readers:
                if not reader.done():
                    reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            self.process = None
            if self.stopping:
                job.state = BenchmarkState.cancelled
                job.detail = "Benchmark cancelled. Partial results are retained."
            job.finishedAt = datetime.now(UTC)
            try:
                self._save()
            except OSError:
                log.exception("Could not save completed benchmark")
