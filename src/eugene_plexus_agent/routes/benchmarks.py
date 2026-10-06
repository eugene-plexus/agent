"""Operator-owned profile benchmarks, using the runtime's policy and path seams."""

import asyncio
import os
import socket
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from .._generated.models import (
    AdmissionDecision,
    Benchmark,
    BenchmarkList,
    BenchmarkRequest,
    EngineKind,
    MeasurementPreflight,
)
from ..benchmarks import Benchmarks, benchmark_args
from ..child_env import child_environment
from ..dependencies import require_operator_session
from ..engines.base import EngineUnavailableError
from ..engines.llama_cpp import LlamaCppAdapter
from ..measurement_node import (
    active_measurement,
    pending,
    refusal_for,
    restart_stopped,
    running_runtimes,
    stop_agreed,
    unlisted,
)
from ..model_copies import resolve_local_path, settings_from_config
from ..node_work import launch_lock
from ..runtimes import _configured_binary, validate_spec
from .runtimes import (
    _admission_for,
    effective_rules_for,
    refresh_library_folders,
    require_library_folder,
)

router = APIRouter(tags=["benchmarks"], dependencies=[Depends(require_operator_session)])


def manager_for(request: Request) -> Benchmarks:
    manager = getattr(request.app.state, "benchmarks", None)
    if manager is None:
        manager = Benchmarks(
            request.app.state.settings.config_file.resolve().parent / "benchmarks.json"
        )
        request.app.state.benchmarks = manager
    return manager


def prepare_binary(body: BenchmarkRequest, get_config):  # type: ignore[no-untyped-def]
    adapter = LlamaCppAdapter()
    found = adapter.resolve_binary(body.runtime, configured=_configured_binary(adapter, get_config))
    server = found.path.resolve()
    binary = server.with_name("llama-bench.exe" if os.name == "nt" else "llama-bench")
    if not binary.is_file() or binary.resolve().parent != server.parent:
        raise ValueError(
            "llama-bench is missing beside the selected llama-server. "
            "Install a complete llama.cpp build."
        )
    probe = subprocess.run(
        [str(binary), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        env=child_environment(),
        cwd=binary.parent,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if probe.returncode:
        raise ValueError(
            "Installed llama-bench could not show its supported options. "
            "Check the engine installation."
        )
    return binary, found.version, probe.stdout + probe.stderr


@router.get("/v1/benchmarks", response_model=BenchmarkList)
async def list_benchmarks(request: Request) -> BenchmarkList:
    return BenchmarkList(benchmarks=list(reversed(manager_for(request).jobs)))


def _engine_refusal(body: BenchmarkRequest) -> str | None:
    spec = body.runtime
    if spec.engine is EngineKind.llama_cpp:
        return None
    # The instrument is llama-bench, which measures GGUF through
    # llama.cpp and nothing else. Before this check, an MLX or
    # vLLM spec fell into `prepare_binary`'s unconditional
    # LlamaCppAdapter and the refusal read "no llama-server on
    # PATH" — a true sentence about the wrong subject.
    return (
        f"Benchmarking uses llama.cpp's own llama-bench and is only "
        f"available for llama_cpp runtimes; this spec declares "
        f"{spec.engine.value!r}. Measure a {spec.engine.value} model by "
        f"running it and reading the gateway's per-request metrics."
    )


async def _prepare(body: BenchmarkRequest, request: Request):  # type: ignore[no-untyped-def]
    """Everything a start checks that does not depend on free memory.

    Raises ValueError (and friends) with the sentence to show. Memory is
    checked separately, after any agreed stop, because a model the
    operator agreed to stop is still holding its memory until it is.
    """
    state = request.app.state.agent_state
    spec = body.runtime
    # Reject incompatible settings and launch-policy violations BEFORE probing a binary.
    benchmark_args(body, Path("llama-bench"), spec.modelPath)
    if reason := validate_spec(spec, state.get_config):
        raise ValueError(reason)
    await refresh_library_folders(request)
    require_library_folder(request, spec.modelPath)
    local = await asyncio.to_thread(
        resolve_local_path,
        spec.modelPath,
        effective_rules_for(request),
        settings_from_config(state.get_config),
    )
    model = Path(local.path)
    if not await asyncio.to_thread(model.is_file):
        raise ValueError(
            "The model file is not available on this node. Check its Library folder mapping."
        )
    binary, version, help_text = await asyncio.to_thread(prepare_binary, body, state.get_config)
    argv, depths = benchmark_args(body, binary, str(model), help_text)
    return model, binary, version, argv, depths


def _node_name(request: Request) -> str:
    identity = getattr(request.app.state, "node_identity", None)
    return identity.record.name if identity and identity.record.enrolled else socket.gethostname()


_PREPARE_ERRORS = (ValueError, EngineUnavailableError, OSError, subprocess.TimeoutExpired)


@router.post("/v1/benchmarks/preflight", response_model=MeasurementPreflight)
async def preflight_benchmark(body: BenchmarkRequest, request: Request) -> MeasurementPreflight:
    problems: list[str] = []
    if kind := active_measurement(request.app):
        problems.append(f"{kind} is already running on this node.")
    if reason := _engine_refusal(body):
        problems.append(reason)
    else:
        try:
            await _prepare(body, request)
        except HTTPException as exc:
            problems.append(str(exc.detail))
        except _PREPARE_ERRORS as exc:
            problems.append(str(exc))
    running = running_runtimes(request)
    if not running and not problems:
        # Only meaningful with nothing running: with models loaded, free
        # memory says nothing about the node once they are stopped.
        admission = await _admission_for(request, body.runtime)
        if admission.decision == AdmissionDecision.refuse:
            problems.append(admission.reason or "Memory admission refused this model.")
    return MeasurementPreflight(
        runningRuntimes=running,
        problems=problems,
        estimateSeconds=None,
        detail="Memory is checked when the benchmark starts, after any agreed stop."
        if running
        else "",
    )


@router.post("/v1/benchmarks", response_model=Benchmark, status_code=202)
async def start_benchmark(body: BenchmarkRequest, request: Request) -> Benchmark:
    async with launch_lock(request):
        manager = manager_for(request)
        if kind := active_measurement(request.app):
            raise HTTPException(409, f"{kind} is already running on this node.")
        running = running_runtimes(request)
        if missing := unlisted(running, body.stopRuntimes or []):
            raise refusal_for(missing, body.stopRuntimes or [])
        if reason := _engine_refusal(body):
            raise HTTPException(422, reason)
        try:
            model, binary, version, argv, depths = await _prepare(body, request)
        except _PREPARE_ERRORS as exc:
            raise HTTPException(422, str(exc)) from exc
        # Past every check that does not need the memory: now stop what the
        # operator agreed to, and only then ask whether the model fits.
        stopped = await stop_agreed(request, running)
        try:
            admission = await _admission_for(request, body.runtime)
            if admission.decision == AdmissionDecision.refuse:
                raise HTTPException(422, admission.reason)
            return manager.start(
                body,
                node=_node_name(request),
                binary=binary,
                version=version,
                model=model,
                argv=argv,
                depths=depths,
                restarts=pending(stopped),
                after=lambda: restart_stopped(
                    request, stopped, enabled=body.restartAfter is not False
                ),
            )
        except Exception as exc:
            # The job never started: put back what was stopped for it, now.
            await restart_stopped(request, stopped, enabled=True, hold_lock=False)
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, _PREPARE_ERRORS):
                raise HTTPException(422, str(exc)) from exc
            raise


@router.post("/v1/benchmarks/{job_id}/cancel", response_model=Benchmark)
async def cancel_benchmark(job_id: str, request: Request) -> Benchmark:
    job = await manager_for(request).cancel(job_id)
    if job is None:
        raise HTTPException(404, "Benchmark not found on this node.")
    return job
