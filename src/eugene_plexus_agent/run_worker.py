"""Execute Library-owned run operations without a browser session.

Each iteration claims one durable stage, reconciles its side effect, then
checkpoints. A lost checkpoint is safe: profiles and runtime declarations have
stable identities, and engine installers expose their current state. Library
leases fence stale workers; the existing node launch guard still owns admission.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException

from ._generated.models import RuntimeSpec
from .admission import LibraryFitClient
from .engines.devices import detect_devices
from .node_work import launch_guard
from .routes import runtimes as actions
from .runtime_context import NodeContext
from .runtimes import describe_engines, installer_for

log = logging.getLogger(__name__)


def runtime_spec(model: dict[str, Any], profile: dict[str, Any], *, start: bool) -> RuntimeSpec:
    base = re.sub("[^a-z0-9]+", "-", model["name"].lower()).strip("-") or "model"
    suffix = (
        "" if profile.get("default") else "-" + re.sub("[^a-z0-9]+", "-", profile["name"].lower())
    )
    return RuntimeSpec.model_validate(
        {
            "name": (base + suffix)[:60].rstrip("-"),
            "engine": profile["engine"],
            "modelPath": model["path"],
            "flags": profile.get("flags") or {},
            "extraArgs": profile.get("extraArgs") or [],
            "env": profile.get("env") or {},
            "autoStart": start,
            "profile": {"id": profile["id"], "name": profile["name"]},
        }
    )


class NodeActions:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def engines(self) -> list[dict[str, Any]]:
        found = await asyncio.to_thread(
            describe_engines, get_config=self.app.state.agent_state.get_config
        )
        return [e.model_dump(mode="json") for e in found]

    async def install(
        self, engine: str, *, previous: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        found = next((e for e in await self.engines() if e["engine"] == engine), None)
        if found and found.get("available"):
            return None
        installer = installer_for(actions._engine_kind(engine))
        snapshot = installer.snapshot() if installer else None
        if snapshot is not None:
            record = snapshot.model_dump(mode="json")
            if record["state"] in {"failed", "cancelled"}:
                if previous is not None:
                    raise ValueError(
                        record.get("error") or record.get("message") or "Engine installation failed"
                    )
                # A new operator-approved run may retry a previous failed install.
            elif record["state"] == "done":
                raise ValueError("The engine was installed but is still unavailable on this node")
            else:
                return record
        try:
            result = await actions.install_engine(engine)
            return result.model_dump(mode="json")
        except HTTPException as exc:
            if exc.status_code != 409:
                raise
            return (await actions.get_engine_install(engine)).model_dump(mode="json")

    async def context_size(
        self, library: LibraryFitClient, model: dict[str, Any], engine: str
    ) -> int | None:
        spec = RuntimeSpec.model_validate(
            {
                "name": "context-probe",
                "engine": engine,
                "modelPath": model["path"],
                "autoStart": False,
            }
        )
        try:
            answer = await actions._admission_for(NodeContext(self.app), spec)
            maximum = answer.maxContextLength
            own = model.get("contextLength")
            if maximum and maximum > 0 and (own is None or maximum < own):
                return maximum
            if str(answer.fit) in {"fits", "unknown"} or engine != "llama_cpp":
                return None
            detector = getattr(self.app.state, "device_detector", None) or detect_devices
            snapshot = await asyncio.to_thread(detector)
            # The fit request must describe THIS node, including unified memory.
            devices = [d for d in snapshot.devices if d.kind.value != "cpu"]
            cpu = next((d for d in snapshot.devices if d.kind.value == "cpu"), None)
            if not snapshot.devices:
                return None
            params = {
                "vramBytes": sum(
                    d.memoryFreeBytes
                    if d.memoryFreeBytes is not None
                    else (d.memoryTotalBytes or 0)
                    for d in devices
                ),
                "ramBytes": (
                    cpu.memoryFreeBytes if cpu.memoryFreeBytes is not None else cpu.memoryTotalBytes
                )
                if cpu
                else 0,
                "gpuCount": max(1, len(devices)),
                "unifiedMemory": "true"
                if any(d.kind.value == "metal" or d.sharedMemory for d in devices)
                else "false",
            }
            fit = await library.operation_request(
                "GET", f"/v1/models/{quote(model['id'], safe='')}/fit", params=params
            )
            longest = fit.get("maxContextExpertsInRam")
            if (
                (fit.get("fit") or {}).get("offload") == "experts"
                and isinstance(longest, int)
                and longest > 0
            ):
                return min(longest, own) if own is not None else longest
        except (httpx.HTTPError, HTTPException, ValueError):
            log.info("context suggestion unavailable; launch admission will check engine defaults")
        return None

    async def launch(
        self, model: dict[str, Any], profile: dict[str, Any], *, start: bool
    ) -> dict[str, Any]:
        context = NodeContext(self.app)
        spec = runtime_spec(model, profile, start=start)
        async with launch_guard(context):
            state = self.app.state.agent_state
            existing = state.get_runtime_spec(spec.name)
            if existing is None:
                result = await actions.declare_runtime(context, spec)
            else:
                for key in ("engine", "modelPath", "flags", "extraArgs", "env"):
                    if getattr(existing, key) != getattr(spec, key):
                        raise ValueError(
                            f"Runtime {spec.name!r} already exists with different {key}; "
                            "rename or edit it on Inference"
                        )
                if existing.autoDriver is not False:
                    # A crash can land between persisting a runtime and creating
                    # its companion. Complete that declaration before starting it.
                    await actions.ensure_companion(
                        state, actions._component_supervisor(context), existing
                    )
                if start:
                    await actions.start_declared_runtime(context, existing.name)
                result = actions._compose(existing, actions._supervisor(context))
        return result.model_dump(mode="json")

    async def runtime(self, name: str) -> dict[str, Any]:
        spec = self.app.state.agent_state.get_runtime_spec(name)
        if spec is None:
            raise ValueError(f"Runtime {name!r} was removed")
        return actions._compose(spec, actions._supervisor(NodeContext(self.app))).model_dump(
            mode="json"
        )


class RunWorker:
    def __init__(self, app: FastAPI, *, node_actions: NodeActions | None = None) -> None:
        self.app = app
        self.actions = node_actions or NodeActions(app)

    async def tick(self, library: LibraryFitClient, *, node: str | None) -> None:
        params = {"node": node} if node else {}
        listing = await library.operation_request(
            "GET", "/v1/run-operations/assigned", params=params
        )
        for brief in listing.get("operations", [])[:128]:
            base = f"/v1/run-operations/{quote(brief['id'], safe='')}"
            try:
                job = await library.operation_request("POST", base + "/claim", params=params)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {403, 404, 409}:
                    continue
                raise
            try:
                patch = await self.advance(library, base, job, params)
            except asyncio.CancelledError:
                raise  # lease expires; restart reconciles the last side effect
            except httpx.HTTPError:
                continue  # transport uncertainty is not a failed operation
            except Exception as exc:
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                if isinstance(detail, dict):
                    detail = detail.get("detail") or detail.get("title") or str(detail)
                patch = {
                    "step": "failed",
                    "error": str(detail),
                    "failedStep": {
                        "checking": "check",
                        "awaiting-install": "install",
                        "installing": "install",
                        "settings": "settings",
                        "launching": "launch",
                        "loading": "load",
                    }.get(job["step"], "launch"),
                }
            try:
                await library.operation_request(
                    "POST",
                    base + "/checkpoint",
                    params=params,
                    json={"lease": job["lease"], **patch},
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 409:
                    raise

    async def advance(
        self, library: LibraryFitClient, base: str, job: dict[str, Any], params: dict[str, str]
    ) -> dict[str, Any]:
        step, model, engine = job["step"], job["model"], job["engine"]
        if step == "checking":
            engines = [
                e
                for e in await self.actions.engines()
                if model["format"] in (e.get("modelFormats") or [])
            ]
            if (model.get("safetensors") or {}).get("mlxQuantization") is not None:
                engines = [e for e in engines if e["engine"] == "mlx"]
            selected = next((e for e in engines if e.get("available")), None)
            if selected:
                return {"step": "settings", "engine": selected["engine"]}
            selected = next(
                (e for e in engines if (e.get("acquisition") or {}).get("installable")), None
            )
            if selected is None:
                reasons = "; ".join(
                    " ".join(
                        filter(
                            None,
                            [
                                (e.get("acquisition") or {}).get("reason") or e["engine"],
                                ((e.get("acquisition") or {}).get("manualInstall") or {}).get(
                                    "command"
                                ),
                                ((e.get("acquisition") or {}).get("manualInstall") or {}).get(
                                    "docsUrl"
                                ),
                            ],
                        )
                    )
                    for e in engines
                )
                raise ValueError(
                    f"No installed or installable engine can load {model['format']} "
                    f"on this node. {reasons}"
                )
            return {"step": "awaiting-install", "engine": selected["engine"]}
        if step == "awaiting-install":
            return {
                "step": "installing"
                if job["answer"] == "install"
                else "settings"
                if job["answer"] == "skip"
                else step
            }
        if step == "installing":
            install = await self.actions.install(engine, previous=job.get("install"))
            return {"step": "settings" if install is None else "installing", "install": install}
        if step == "settings":
            context = await self.actions.context_size(library, model, engine)
            await library.operation_request(
                "POST",
                base + "/profile",
                params=params,
                json={"lease": job["lease"], "engine": engine, "contextSize": context},
            )
            # Skip still declares a stopped runtime; it never installs or starts.
            return {"step": "launching"}
        if step == "launching":
            result = await self.actions.launch(model, job["profile"], start=job["answer"] != "skip")
            return {
                "step": "skipped"
                if job["answer"] == "skip"
                else "ready"
                if result["status"] == "ready"
                else "loading",
                "runtime": result["name"],
                "runtimeStatus": result["status"],
                "loadingSince": int(time.time() * 1000),
            }
        if step == "loading":
            result = await self.actions.runtime(job["runtime"])
            if result["status"] == "crashed":
                raise ValueError(result.get("lastError") or "Engine stopped unexpectedly")
            if time.time() * 1000 - job.get("loadingSince", job["startedAt"]) > 3_600_000:
                raise ValueError(
                    "Model is still loading after one hour. Inference shows the engine's status."
                )
            return {
                "step": "ready" if result["status"] == "ready" else "loading",
                "runtimeStatus": result["status"],
            }
        raise ValueError(f"Unsupported run stage {step}")

    async def run_forever(self) -> None:
        while True:
            await asyncio.sleep(2)
            try:
                identity = getattr(self.app.state, "node_identity", None)
                node = (
                    identity.record.name
                    if identity is not None and identity.record.enrolled
                    else None
                )
                library = await actions.library_client_for(NodeContext(self.app))
                if library is not None:
                    await self.tick(library, node=node)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug("run operations unavailable: %s", exc)
