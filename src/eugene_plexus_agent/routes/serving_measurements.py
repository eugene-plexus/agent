"""Measure an already running runtime through the install's existing doors."""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from .. import host_history
from .._generated.models import (
    AllowedModel,
    BenchmarkState,
    ClientKeyCreateRequest,
    ClientKeyLimits,
    HostMetric,
    MeasurementRestartState,
    Runtime,
    ServingDrafting,
    ServingGpu,
    ServingLevel,
    ServingMeasurement,
    ServingMeasurementList,
    ServingMeasurementPreflight,
    ServingMeasurementRequest,
    ServingMeasurementRestart,
    ServingProvenance,
    ServingRuntime,
)
from .._http import internal_client
from ..dependencies import require_operator_session
from ..measurement_node import active_measurement, restart_stopped, stop_agreed
from ..node_work import launch_lock
from ..serving_sweep import ServingMeasurements, Sweep, effective_request, people_levels
from .auth import mint_client_key, revoke_client_key_at_authority
from .node import _devices
from .proxy import _translate, resolve_target
from .runtimes import _compose, _supervisor

router = APIRouter(tags=["runtimes"], dependencies=[Depends(require_operator_session)])


def manager_for(request: Request) -> ServingMeasurements:
    manager = getattr(request.app.state, "serving_measurements", None)
    if manager is None:
        manager = ServingMeasurements(
            request.app.state.settings.config_file.resolve().parent / "serving-measurements.json"
        )
        request.app.state.serving_measurements = manager
    return manager


def node_name(request: Request) -> str:
    identity = getattr(request.app.state, "node_identity", None)
    return identity.record.name if identity and identity.record.enrolled else socket.gethostname()


def runtime_for(request: Request, name: str) -> Runtime:
    spec = request.app.state.agent_state.get_runtime_spec(name)
    if spec is None:
        raise HTTPException(404, f"No runtime named {name} is declared on {node_name(request)}.")
    return _compose(spec, _supervisor(request))


def ready(runtime: Runtime, node: str) -> None:
    if runtime.status != "ready":
        raise HTTPException(
            422,
            f"{runtime.modelAlias or runtime.name} on {node} is "
            f"{runtime.status}. Start it, then measure.",
        )


def _identity(path: str) -> tuple[int | None, datetime | None]:
    try:
        stat = Path(path).stat()
        return stat.st_size, datetime.fromtimestamp(stat.st_mtime, UTC)
    except OSError:
        return None, None


def _binary_hash(argv: list[str]) -> str | None:
    if not argv:
        return None
    try:
        with Path(argv[0]).open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        return None


def _commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and re.fullmatch("[0-9a-f]{40}", value) else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _redacted(argv: list[str]) -> list[str]:
    result: list[str] = []
    secret_next = False
    for value in argv:
        secret = bool(re.search(r"(?i)(api[-_]?key|token|password|secret)", value.split("=")[0]))
        result.append(
            "[redacted]"
            if secret_next
            else value.split("=")[0] + "=[redacted]"
            if secret and "=" in value
            else value
        )
        secret_next = secret and "=" not in value
    return result


async def provenance_for(
    request: Request, runtime: Runtime, gateway_commit: str | None
) -> ServingProvenance:
    snapshot = await _devices(request)
    argv = runtime.argv or []
    size, modified = await asyncio.to_thread(
        _identity, runtime.openedPath or runtime.localPath or runtime.modelPath
    )
    binary_hash = await asyncio.to_thread(_binary_hash, argv)
    cpu = snapshot.cpu()
    return ServingProvenance(
        engine=str(runtime.engine),
        engineBuild=runtime.engineVersion,
        binaryHash=binary_hash,
        launchArgv=_redacted(argv),
        modelPath=runtime.openedPath or runtime.localPath or runtime.modelPath,
        modelSizeBytes=size,
        modelModifiedAt=modified,
        gpus=[
            ServingGpu(name=d.name or str(d.kind), uuid=None, memoryBytes=d.memoryTotalBytes)
            for d in snapshot.accelerators()
        ],
        driver=None,
        cpu=cpu.name if cpu else None,
        ramBytes=snapshot.ram_total_bytes,
        agentCommit=await asyncio.to_thread(_commit),
        gatewayCommit=gateway_commit,
    )


class Doors:
    def __init__(self, request: Request) -> None:
        self.request = request
        self.client = internal_client(
            timeout=httpx.Timeout(120, connect=5),
            limits=httpx.Limits(max_connections=320, max_keepalive_connections=320),
            transport=getattr(request.app.state, "serving_transport", None),
        )
        self.gateway = ""

    async def address(self, target: str) -> tuple[str, dict[str, str]]:
        route = await resolve_target(self.request, target)
        credential = self.request.headers.get("authorization", "")
        if route.recipient and credential:
            translated = await _translate(
                self.request, credential.removeprefix("Bearer "), route.recipient
            )
            if not translated:
                raise HTTPException(403, f"The caller's credential cannot act on {target}.")
            credential = "Bearer " + translated
        return route.base.rstrip("/") + route.prefix, {"Authorization": credential}

    async def call(self, target: str, method: str, path: str, body: Any = None) -> dict[str, Any]:
        address, headers = await self.address(target)
        response = await self.client.request(
            method, address + path, json=body, headers=headers, timeout=10
        )
        if response.status_code >= 400:
            raise HTTPException(
                503, f"{target} returned {response.status_code}: {response.text[:600]}"
            )
        return response.json() if response.content else {}

    async def routing(self) -> dict[str, Any]:
        address, _ = await self.address("gateway")
        self.gateway = address
        return await self.call("gateway", "GET", "/v1/admin/routing")

    async def gateway_commit(self) -> str | None:
        try:
            health = await self.call("gateway", "GET", "/healthz")
            return health.get("commit") or health.get("gitCommit")
        except (httpx.HTTPError, HTTPException):
            return None


async def inspect(
    body: ServingMeasurementRequest, request: Request, doors: Doors
) -> tuple[Runtime | None, ServingMeasurementRequest, ServingMeasurementPreflight]:
    problems: list[str] = []
    runtime = None
    if kind := active_measurement(request.app):
        problems.append(f"{kind} is already running on this node.")
    try:
        runtime = runtime_for(request, body.runtime)
        ready(runtime, node_name(request))
    except HTTPException as exc:
        problems.append(str(exc.detail))
    caps = runtime.capabilities if runtime else None
    effective = effective_request(body, caps.parallelSlots if caps and caps.parallelSlots else 1)
    replicas: dict[tuple[str, str], ServingRuntime] = {}
    if runtime:
        routing = await doors.routing()
        alias = runtime.modelAlias or runtime.name
        for slot in routing.get("slots", []):
            if slot.get("model") != alias:
                continue
            for tier in slot.get("tiers", []):
                for backend in tier.get("backends", []):
                    name, node = backend.get("runtime"), backend.get("node")
                    if (
                        name
                        and node
                        and backend.get("eligible")
                        and backend.get("runtime_status") == "ready"
                        and (node, name) != (node_name(request), runtime.name)
                    ):
                        replicas[node, name] = ServingRuntime(node=node, runtime=name, model=alias)
    others: list[ServingRuntime] = []
    for spec in request.app.state.agent_state.list_runtime_specs():
        other = _compose(spec, _supervisor(request))
        if spec.name != body.runtime and other.status != "stopped":
            others.append(
                ServingRuntime(
                    node=node_name(request), runtime=spec.name, model=other.modelAlias or other.name
                )
            )
    return (
        runtime,
        effective,
        ServingMeasurementPreflight(
            otherReplicas=list(replicas.values()),
            otherRuntimes=others,
            problems=problems,
            estimateSeconds=len(people_levels(effective.maxPeople or 1))
            * (effective.windowSeconds or 300),
        ),
    )


async def revoke(request: Request, manager: ServingMeasurements, job: ServingMeasurement) -> None:
    if not job.keyLive:
        return
    key = manager.keys.get(job.id)
    if key is None:
        job.detail += " The measurement key id is missing; revocation could not be confirmed."
        return
    if datetime.fromisoformat(key["expiresAt"]) <= datetime.now(UTC):
        job.keyLive = False
        return
    try:
        await revoke_client_key_at_authority(
            request, key["id"], authorization=request.headers.get("authorization")
        )
        job.keyLive = False
    except (HTTPException, httpx.HTTPError) as exc:
        cause = exc.detail if isinstance(exc, HTTPException) else str(exc)
        job.detail += f" The measurement key could not be revoked: {cause}"


@router.get("/v1/serving-measurements", response_model=ServingMeasurementList)
async def list_measurements(request: Request, runtime: str | None = None) -> ServingMeasurementList:
    manager = manager_for(request)
    if manager.load_error:
        raise HTTPException(503, manager.load_error)
    doors = Doors(request)
    present: dict[str, ServingProvenance] = {}
    try:
        commit = await doors.gateway_commit() if manager.jobs else None
        for name in {j.runtime for j in manager.jobs}:
            try:
                current = runtime_for(request, name)
            except HTTPException:
                continue
            present[name] = await provenance_for(request, current, commit)
        jobs = manager.list(present)
        return ServingMeasurementList(
            measurements=[j for j in jobs if runtime is None or j.runtime == runtime]
        )
    finally:
        await doors.client.aclose()


@router.post("/v1/serving-measurements/preflight", response_model=ServingMeasurementPreflight)
async def preflight(
    body: ServingMeasurementRequest, request: Request
) -> ServingMeasurementPreflight:
    doors = Doors(request)
    try:
        _, _, result = await inspect(body, request, doors)
        return result
    except httpx.HTTPError as exc:
        raise HTTPException(503, f"The gateway could not be read: {exc}") from exc
    finally:
        await doors.client.aclose()


@router.post("/v1/serving-measurements", response_model=ServingMeasurement, status_code=202)
async def start(body: ServingMeasurementRequest, request: Request) -> ServingMeasurement:
    async with launch_lock(request):
        manager = manager_for(request)
        if manager.load_error:
            raise HTTPException(503, manager.load_error)
        if kind := active_measurement(request.app):
            raise HTTPException(409, f"{kind} is already running on this node.")
        runtime = runtime_for(request, body.runtime)
        node = node_name(request)
        ready(runtime, node)
        doors = Doors(request)
        stopped: list[ServingRuntime] = []
        job: ServingMeasurement | None = None
        try:
            _, effective, check = await inspect(body, request, doors)
            if check.problems:
                raise HTTPException(422, " ".join(check.problems))
            alias = runtime.modelAlias or runtime.name
            if "*" in alias:
                raise HTTPException(
                    422,
                    "The model alias contains a wildcard. Give it an exact "
                    "name before measuring so the key can allow only this model.",
                )
            agreed = {(r.node, r.runtime, r.model) for r in effective.stopReplicas or []}
            if any(
                r.model != alias or (r.node, r.runtime) == (node, runtime.name)
                for r in effective.stopReplicas or []
            ):
                raise HTTPException(422, "Only other replicas of this model may be stopped.")
            missing = [r for r in check.otherReplicas if (r.node, r.runtime, r.model) not in agreed]
            if missing:
                raise HTTPException(
                    409,
                    "Agree to stopping these other copies before measuring: "
                    + ", ".join(f"{r.runtime} on {r.node}" for r in missing),
                )
            provenance = await provenance_for(request, runtime, await doors.gateway_commit())
            try:
                created = await mint_client_key(
                    request,
                    ClientKeyCreateRequest(
                        name=f"Measurement {runtime.name[:29]} {datetime.now(UTC):%Y-%m-%d}",
                        ttlDays=1,
                        limits=ClientKeyLimits(
                            localOnly=True,
                            allowedModels=[AllowedModel(alias)],
                            maxConcurrentRequests=64,
                            requestsPerMinute=10000,
                        ),
                    ),
                    authorization=request.headers.get("authorization"),
                )
            except HTTPException as exc:
                raise HTTPException(
                    503, f"Eugene could not make a measurement key: {exc.detail}"
                ) from exc
            job = manager.new(effective, node, alias, provenance)
            manager.attach_key(job.id, created.key.id, created.key.expiresAt)
            speculation = runtime.speculation
            job.drafting = ServingDrafting(
                type=", ".join(speculation.types) if speculation else "off",
                model=speculation.draftModelPath if speculation else None,
                acceptance=(
                    speculation.acceptedTokens / speculation.draftedTokens
                    if speculation
                    and speculation.draftedTokens
                    and speculation.acceptedTokens is not None
                    else None
                ),
            )
            manager.save()
            for replica in check.otherReplicas:
                if replica.node == node:
                    await stop_agreed(request, [replica.runtime])
                else:
                    await doors.call(
                        f"node:{replica.node}",
                        "POST",
                        f"/v1/runtimes/{quote(replica.runtime, safe='')}/stop",
                        {"reason": "measurement"},
                    )
                stopped.append(replica)
                job.restarts.append(
                    ServingMeasurementRestart(
                        node=replica.node,
                        runtime=replica.runtime,
                        state=MeasurementRestartState.pending,
                        detail=None,
                    )
                )
                manager.save()
            # A stopped replica can remain in the gateway's cached view until
            # its next refresh. Wait for that observation before sending load.
            deadline = time.perf_counter() + 10
            while stopped:
                _, _, settled = await inspect(effective, request, doors)
                remaining = settled.otherReplicas
                if not remaining:
                    break
                if any((r.node, r.runtime, r.model) not in agreed for r in remaining):
                    raise HTTPException(
                        409,
                        "Another copy became ready while preparing the run. "
                        "Review the other copies again.",
                    )
                if time.perf_counter() >= deadline:
                    raise HTTPException(
                        503,
                        "The gateway still reports these stopped copies ready: "
                        + ", ".join(r.runtime for r in remaining),
                    )
                await asyncio.sleep(0.1)
            caps = runtime.capabilities
            context = caps.contextLength if caps and caps.contextLength else 32768
            sweep = Sweep(
                doors.client,
                doors.gateway,
                created.token,
                runtime.name,
                node,
                alias,
                context,
                random.Random(effective.seed),
            )
            before: float | None = None
            began = 0.0

            async def observe(level: ServingLevel | None) -> None:
                nonlocal before, began
                if level is None:
                    before = await engine_counter(doors.client, runtime)
                    began = time.time()
                else:
                    after = await engine_counter(doors.client, runtime)
                    if before is not None and after is not None and after >= before:
                        level.foreignRequests = max(
                            0,
                            int(after - before) - level.requests - sweep.warmups,
                        )
                    level.watts = await watts(request, began, time.time())

            async def cleanup(record: ServingMeasurement) -> None:
                try:
                    await revoke(request, manager, record)
                    await restart_replicas(request, doors, record)
                finally:
                    await doors.client.aclose()

            manager.task = asyncio.create_task(manager.run(job, sweep, cleanup, observe))
            return job
        except BaseException as exc:
            if job:
                job.state = BenchmarkState.failed
                job.finishedAt = datetime.now(UTC)
                job.detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
                await revoke(request, manager, job)
                await restart_replicas(request, doors, job, hold_lock=False)
                manager.save()
            await doors.client.aclose()
            if isinstance(exc, httpx.HTTPError):
                raise HTTPException(503, f"The gateway could not be read: {exc}") from exc
            raise


async def restart_replicas(
    request: Request, doors: Doors, job: ServingMeasurement, *, hold_lock: bool = True
) -> None:
    for restart in job.restarts:
        if restart.state != "pending":
            continue
        try:
            if restart.node == job.node:
                result = (
                    await restart_stopped(
                        request, [restart.runtime], enabled=True, hold_lock=hold_lock
                    )
                )[0]
                restart.state, restart.detail = result.state, result.detail
            else:
                await doors.call(
                    f"node:{restart.node}",
                    "POST",
                    f"/v1/runtimes/{quote(restart.runtime, safe='')}/start",
                )
                restart.state = MeasurementRestartState.restarted
        except Exception as exc:
            restart.state = MeasurementRestartState.failed
            restart.detail = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)


async def engine_counter(client: httpx.AsyncClient, runtime: Runtime) -> float | None:
    if not runtime.url:
        return None
    try:
        response = await client.get(str(runtime.url).rstrip("/") + "/metrics", timeout=2)
        if response.status_code != 200:
            return None
        totals: dict[str, float] = {}
        for line in response.text.splitlines():
            match = re.match(
                r"^(llamacpp:requests_total|vllm:request_success_total)(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
                line,
            )
            if match:
                totals[match[1]] = totals.get(match[1], 0) + float(match[2])
        return next(iter(totals.values()), None)
    except (httpx.HTTPError, ValueError):
        return None


async def watts(request: Request, start: float, end: float) -> float | None:
    store = getattr(request.app.state, "host_history", None)
    if store is None:
        return None
    try:
        history = await asyncio.to_thread(
            store.query,
            now=end,
            start=start,
            end=end,
            settings=host_history.read_settings(request.app.state.agent_state.get_config),
            metrics=[HostMetric.gpuPower],
        )
        means = [
            sum(values)
            for values in zip(*(s.mean for s in history.series), strict=False)
            if values and all(v is not None for v in values)
        ]
        return sum(means) / len(means) if means else None
    except (OSError, RuntimeError, ValueError):
        return None


@router.post("/v1/serving-measurements/{job_id}/cancel", response_model=ServingMeasurement)
async def cancel(job_id: str, request: Request) -> ServingMeasurement:
    manager = manager_for(request)
    job = next((j for j in manager.jobs if j.id == job_id), None)
    if job is None:
        raise HTTPException(404, "Measurement not found on this node.")
    if job.state == BenchmarkState.running and manager.task and not manager.task.done():
        manager.task.cancel()
        # Abort streams immediately; key revocation is attempted by cleanup.
        await asyncio.wait({manager.task}, timeout=4)
    else:
        await revoke(request, manager, job)
        manager.save()
    return job
