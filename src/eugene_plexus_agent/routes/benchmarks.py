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
    RuntimeStatus,
)
from ..benchmarks import Benchmarks, benchmark_args
from ..child_env import child_environment
from ..dependencies import require_operator_session
from ..engines.base import EngineUnavailableError
from ..engines.llama_cpp import LlamaCppAdapter
from ..model_copies import resolve_local_path, settings_from_config
from ..node_work import launch_lock
from ..runtimes import _configured_binary, validate_spec
from .runtimes import (
    _admission_for,
    _compose,
    _supervisor,
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
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
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


@router.post("/v1/benchmarks", response_model=Benchmark, status_code=202)
async def start_benchmark(body: BenchmarkRequest, request: Request) -> Benchmark:
    async with launch_lock(request):
        manager = manager_for(request)
        if manager.active:
            raise HTTPException(409, "A benchmark is already running on this node.")
        state = request.app.state.agent_state
        busy = [
            s.name
            for s in state.list_runtime_specs()
            if _compose(s, _supervisor(request)).status != RuntimeStatus.stopped
        ]
        if busy:
            raise HTTPException(
                409, "Stop this node's runtimes before benchmarking: " + ", ".join(busy)
            )
        spec = body.runtime
        # Reject incompatible settings and launch-policy violations BEFORE probing a binary.
        try:
            benchmark_args(body, Path("llama-bench"), spec.modelPath)
            if reason := validate_spec(spec, state.get_config):
                raise ValueError(reason)
            await refresh_library_folders(request)
            require_library_folder(request, spec.modelPath)
            admission = await _admission_for(request, spec)
            if admission.decision == AdmissionDecision.refuse:
                raise HTTPException(422, admission.reason)
            local = await asyncio.to_thread(
                resolve_local_path,
                spec.modelPath,
                effective_rules_for(request),
                settings_from_config(state.get_config),
            )
            model = Path(local.path)
            if not await asyncio.to_thread(model.is_file):
                raise ValueError(
                    "The model file is not available on this node. "
                    "Check its Library folder mapping."
                )
            binary, version, help_text = await asyncio.to_thread(
                prepare_binary, body, state.get_config
            )
            argv, depths = benchmark_args(body, binary, str(model), help_text)
            identity = getattr(request.app.state, "node_identity", None)
            node = (
                identity.record.name
                if identity and identity.record.enrolled
                else socket.gethostname()
            )
            return manager.start(
                body,
                node=node,
                binary=binary,
                version=version,
                model=model,
                argv=argv,
                depths=depths,
            )
        except (ValueError, EngineUnavailableError, OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(422, str(exc)) from exc


@router.post("/v1/benchmarks/{job_id}/cancel", response_model=Benchmark)
async def cancel_benchmark(job_id: str, request: Request) -> Benchmark:
    job = await manager_for(request).cancel(job_id)
    if job is None:
        raise HTTPException(404, "Benchmark not found on this node.")
    return job
