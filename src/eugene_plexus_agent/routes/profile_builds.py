"""Operator-owned profile builds, through the runtime's policy and path seams.

The design is `docs/design/profile-builder.md`; the job itself is
`profile_builds.py`. This module checks everything a start depends on,
asks nothing of memory until the agreed models are stopped, and hands the
runner a plan — the same shape as the benchmark routes, on purpose.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request

from .._generated.models import (
    MeasurementPreflight,
    ProfileBuild,
    ProfileBuildAccuracy,
    ProfileBuildList,
    ProfileBuildRequest,
)
from ..child_env import child_environment
from ..dependencies import require_operator_session
from ..engines.base import EngineUnavailableError
from ..engines.devices import detect_devices
from ..engines.llama_cpp import LlamaCppAdapter
from ..gguf_context import trained_context
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
from ..profile_builds import (
    CANDIDATE_TYPES,
    DISK_NEEDED_BYTES,
    LIKELY_TOO_SHORT_BYTES,
    MIN_EVALUATION_TOKENS,
    BuildError,
    BuildPlan,
    ProfileBuilds,
    Tools,
    bundled_text,
    contexts_for,
    evaluation_record,
    validate_request,
)
from ..runtimes import _configured_binary, validate_spec
from .benchmarks import _node_name
from .runtimes import (
    effective_rules_for,
    library_client_for,
    refresh_library_folders,
    require_library_folder,
)

router = APIRouter(tags=["benchmarks"], dependencies=[Depends(require_operator_session)])

# The options each tool must list in its own help for a build to use it.
REQUIRED_OPTIONS = {
    "llama-bench": {
        "--n-depth",
        "--output",
        "--cache-type-k",
        "--cache-type-v",
        "--override-tensor",
        "--n-gpu-layers",
        "--flash-attn",
    },
    "llama-fit-params": {"--fit-target", "--ctx-size", "--cache-type-k"},
    "llama-perplexity": {"--kl-divergence-base", "--kl-divergence", "--chunks", "--cache-type-k"},
}
_ERRORS = (BuildError, ValueError, EngineUnavailableError, OSError, subprocess.TimeoutExpired)


def manager_for(request: Request) -> ProfileBuilds:
    manager = getattr(request.app.state, "profile_builds", None)
    if manager is None:
        root = request.app.state.settings.config_file.resolve().parent
        manager = ProfileBuilds(root / "profile_builds.json")
        request.app.state.profile_builds = manager
    return manager


def _work_root(request: Request) -> Path:
    return Path(request.app.state.settings.config_file).resolve().parent / "profile-builds"


def prepare_tools(body: ProfileBuildRequest, get_config) -> Tools:  # type: ignore[no-untyped-def]
    """llama-server as discovery selects it, and its three siblings, checked."""
    adapter = LlamaCppAdapter()
    found = adapter.resolve_binary(body.runtime, configured=_configured_binary(adapter, get_config))
    server = found.path.resolve()
    suffix = ".exe" if os.name == "nt" else ""
    paths: dict[str, Path] = {}
    for tool, required in REQUIRED_OPTIONS.items():
        path = server.with_name(tool + suffix)
        if not path.is_file() or path.resolve().parent != server.parent:
            raise BuildError(
                f"{tool} is missing beside the selected llama-server "
                f"({found.version or 'unknown build'}). Install a complete llama.cpp build."
            )
        probe = subprocess.run(
            [str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            env=child_environment(),
            cwd=path.parent,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        listed = probe.stdout + probe.stderr
        if missing := sorted(opt for opt in required if opt not in listed):
            raise BuildError(
                f"The installed {tool} ({found.version or 'unknown build'}) lacks "
                + ", ".join(missing)
                + "; update llama.cpp."
            )
        paths[tool] = path
    return Tools(
        server=server,
        bench=paths["llama-bench"],
        fit=paths["llama-fit-params"],
        perplexity=paths["llama-perplexity"],
        version=found.version,
    )


async def _trained_context(request: Request, model_id: str, model: Path) -> int | None:
    """What the model was trained for: the library's reading, else the file's header."""
    library = await library_client_for(request)
    if library is not None:
        for entry in await library.list_models() or []:
            length = entry.get("contextLength")
            if entry.get("id") == model_id and isinstance(length, int) and length > 0:
                return length
    return await asyncio.to_thread(trained_context, model)


async def _prepare(body: ProfileBuildRequest, request: Request) -> tuple[BuildPlan, Tools]:
    """Everything a start checks that does not depend on free memory."""
    state = request.app.state.agent_state
    spec = body.runtime
    flags = validate_request(body)
    if reason := validate_spec(spec, state.get_config):
        raise BuildError(reason)
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
        raise BuildError(
            "The model file is not available on this node. Check its Library folder mapping."
        )
    tools = await asyncio.to_thread(prepare_tools, body, state.get_config)
    work = _work_root(request) / uuid4().hex
    if body.accuracy is not ProfileBuildAccuracy.max:
        work.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(work.parent).free
        if free < DISK_NEEDED_BYTES:
            raise BuildError(
                f"The quality check needs about {DISK_NEEDED_BYTES / 2**30:.1f} GiB free for a "
                f"temporary file on the drive holding {work.parent}; it has "
                f"{free / 2**30:.1f} GiB. Free some space or choose Max accuracy."
            )
    env = child_environment()
    env.update(spec.env or {})
    detector = getattr(request.app.state, "device_detector", None) or detect_devices
    plan = BuildPlan(
        tools=tools,
        model=model,
        work=work,
        text=bundled_text(),
        flags=flags,
        contexts=contexts_for(await _trained_context(request, body.modelId, model)),
        margin=body.memoryMarginMiB,
        env=env,
        detect=detector,
    )
    return plan, tools


def _estimate(body: ProfileBuildRequest, plan: BuildPlan) -> tuple[int, str]:
    """Roughly how long, from the file's size and where it is read from."""
    size = plan.model.stat().st_size
    shared = str(plan.model).startswith(("\\\\", "//"))
    rate = 100e6 if shared else 1.5e9
    load = size / rate
    quality_runs = (
        len(CANDIDATE_TYPES[body.accuracy]) if body.accuracy is not ProfileBuildAccuracy.max else 0
    )
    measure_runs = len(plan.contexts)
    seconds = int(quality_runs * (load + 40) + measure_runs * (load + 25) + (load + 30))
    where = "a network share (about 100 MB/s)" if shared else "a local disk"
    detail = (
        f"{quality_runs} quality run(s), about {measure_runs} measurement(s) and one "
        f"confirmation, each loading {size / 1e9:.1f} GB from {where}."
    )
    if shared:
        detail += " A local copy of the model on this node would make each load much faster."
    return seconds, detail


@router.get("/v1/profile-builds", response_model=ProfileBuildList)
async def list_profile_builds(request: Request) -> ProfileBuildList:
    return ProfileBuildList(builds=list(reversed(manager_for(request).jobs)))


@router.post("/v1/profile-builds/preflight", response_model=MeasurementPreflight)
async def preflight_profile_build(
    body: ProfileBuildRequest, request: Request
) -> MeasurementPreflight:
    problems: list[str] = []
    estimate: int | None = None
    detail = ""
    if kind := active_measurement(request.app):
        problems.append(f"{kind} is already running on this node.")
    if body.evaluationText is not None and len(body.evaluationText.encode("utf-8")) < (
        LIKELY_TOO_SHORT_BYTES
    ):
        problems.append(
            "This text is probably too short for the quality check: it needs at least "
            f"{MIN_EVALUATION_TOKENS:,} tokens (roughly 32 KB of English). "
            "The build reports the exact count."
        )
    try:
        plan, _ = await _prepare(body, request)
        estimate, detail = _estimate(body, plan)
    except HTTPException as exc:
        problems.append(str(exc.detail))
    except _ERRORS as exc:
        problems.append(str(exc))
    # No memory admission here or at start, deliberately. Admission refuses
    # a launch predicted to exceed free graphics memory, but llama.cpp's fit
    # places such a model partly in system memory, and that placement is
    # exactly what a build measures (design §0 M6: the MoE model admission
    # scores `split` decodes at 46 tok/s on 8 GB). Fit reports the model it
    # cannot place at any context, and the build says so.
    running = running_runtimes(request)
    if running:
        detail = (detail + " The models listed are stopped while it runs.").strip()
    return MeasurementPreflight(
        runningRuntimes=running, problems=problems, estimateSeconds=estimate, detail=detail
    )


@router.post("/v1/profile-builds", response_model=ProfileBuild, status_code=202)
async def start_profile_build(body: ProfileBuildRequest, request: Request) -> ProfileBuild:
    async with launch_lock(request):
        manager = manager_for(request)
        if kind := active_measurement(request.app):
            raise HTTPException(409, f"{kind} is already running on this node.")
        running = running_runtimes(request)
        if missing := unlisted(running, body.stopRuntimes or []):
            raise refusal_for(missing, body.stopRuntimes or [])
        try:
            plan, _ = await _prepare(body, request)
        except _ERRORS as exc:
            raise HTTPException(422, str(exc)) from exc
        if body.evaluationText is not None:
            plan.work.mkdir(parents=True, exist_ok=True)
            plan.text = plan.work / "evaluation.txt"
            plan.text.write_text(body.evaluationText, encoding="utf-8")
        stopped = await stop_agreed(request, running)
        try:
            return manager.start(
                body,
                plan,
                node=_node_name(request),
                restarts=pending(stopped),
                evaluation=evaluation_record(body.evaluationText),
                after=lambda: restart_stopped(
                    request, stopped, enabled=body.restartAfter is not False
                ),
            )
        except Exception as exc:
            await restart_stopped(request, stopped, enabled=True, hold_lock=False)
            shutil.rmtree(plan.work, ignore_errors=True)
            if isinstance(exc, _ERRORS):
                raise HTTPException(422, str(exc)) from exc
            raise


@router.post("/v1/profile-builds/{job_id}/cancel", response_model=ProfileBuild)
async def cancel_profile_build(job_id: str, request: Request) -> ProfileBuild:
    job = await manager_for(request).cancel(job_id)
    if job is None:
        raise HTTPException(404, "Build not found on this node.")
    return job
