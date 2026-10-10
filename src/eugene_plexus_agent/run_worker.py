"""Execute Library-owned run operations without a browser session.

Each iteration claims one durable stage, reconciles its side effect, then
checkpoints. A lost checkpoint is safe: profiles and runtime declarations have
stable identities, and engine installers expose their current state. Library
leases fence stale workers; the existing node launch guard still owns admission.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException

from ._generated.models import EngineKind, RuntimeSpec
from .admission import LibraryFitClient
from .engines import adapter_for
from .engines.acquisition import engine_root
from .engines.devices import detect_devices
from .model_paths import join_local, resolve_model_path
from .node_work import launch_guard
from .preparation import PreparationError, PreparationJobs, PreparationResult, Progress, Recipe
from .preparation_send import CHUNK_BYTES, REQUEST_TIMEOUT, Sender, detail_of
from .routes import runtimes as actions
from .runtime_context import NodeContext
from .runtimes import _configured_binary, describe_engines, installer_for

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


_RUNNABLE = ("runs", "may_run")


def takes_context_size(engine: str) -> bool:
    """Whether the engine's launch flags have `contextSize`, the one a Run
    writes into the profile it makes."""
    from .engines import ADAPTERS

    try:
        adapter = ADAPTERS.get(EngineKind(engine))
    except ValueError:
        return False
    if adapter is None:
        return False
    return any(field.key == "contextSize" for field in adapter.flag_schema().fields)


def could_have_here(engine: dict[str, Any]) -> bool:
    """The console's `offeredOnThisNode`, for an engine not installed: Eugene
    installs it itself, or it is installable, or the agent wrote an install
    command for this hardware. MLX on Windows is none of these."""
    acquisition = engine.get("acquisition") or {}
    return (
        acquisition.get("policy") != "manual"
        or bool(acquisition.get("installable"))
        or bool((acquisition.get("manualInstall") or {}).get("command"))
    )


def eligibility_engine(engine: dict[str, Any]) -> dict[str, Any]:
    """An engine as `POST /v1/eligibility` takes it."""
    return {
        "engine": engine["engine"],
        "available": bool(engine.get("available")),
        "installable": not engine.get("available") and could_have_here(engine),
        "experimental": bool(engine.get("experimental")),
        "accepts": engine.get("accepts") or [],
    }


async def judge(
    library: LibraryFitClient, model: dict[str, Any], engines: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every engine's verdict on `model`, best first, from the library: the
    one judge (library-sources-and-engines.md, LS1)."""
    answer = await library.operation_request(
        "POST",
        "/v1/eligibility",
        json={"models": [model["id"]], "engines": [eligibility_engine(e) for e in engines]},
    )
    found = (answer or {}).get("models") or []
    if not found:
        raise ValueError(f"The library no longer knows {model.get('name') or model['id']}.")
    verdicts: list[dict[str, Any]] = found[0]["engines"]
    return verdicts


def _why_none(model: dict[str, Any], verdicts: list[dict[str, Any]], by_name: dict) -> str:
    parts = []
    for v in verdicts:
        acquisition = by_name.get(v["engine"], {}).get("acquisition") or {}
        manual = acquisition.get("manualInstall") or {}
        if v["verdict"] in _RUNNABLE:
            how = acquisition.get("reason") or "is not installed here"
            words = [f"{v['engine']} {how}", manual.get("command"), manual.get("docsUrl")]
            parts.append(" ".join(filter(None, words)))
        elif v["verdict"] == "after_preparation":
            # Asked for, never implied (LS5, B54): name the action.
            parts.append(
                f"{v['engine']} can run it after preparing it: choose Prepare on the model's "
                "Library page"
            )
        else:
            parts.append(f"{v['engine']} {v['reason']}")
    name = model.get("name") or model["format"]
    return f"No installed or installable engine can run {name} on this node. " + "; ".join(parts)


def preparation_folder(engine: str, library_folder: str) -> Path:
    """This node's folder standing for one Library folder while `engine`
    prepares a model in it (LS10, B101): under the engines' own folder, and
    named for the Library folder as the library spells it."""
    key = hashlib.sha256(library_folder.encode("utf-8")).hexdigest()[:16]
    return engine_root() / "preparing" / engine / key


def _prepare_route(
    model: dict[str, Any], verdicts: list[dict[str, Any]], by_name: dict, engine: str
) -> dict[str, Any]:
    """Where a run that asked for a preparation goes from checking (LS5): the
    engine must prepare this model, and be here or installable here."""
    name = model.get("name") or model["format"]
    verdict = next((v for v in verdicts if v["engine"] == engine), None)
    if verdict is None:
        raise ValueError(f"This node has no engine {engine} to prepare {name}.")
    if verdict["verdict"] != "after_preparation":
        raise ValueError(
            f"{engine} does not prepare {name}: {verdict.get('reason') or verdict['verdict']}"
        )
    if verdict["available"]:
        return {"step": "preparing", "engine": engine}
    acquisition = by_name.get(engine, {}).get("acquisition") or {}
    if acquisition.get("installable"):
        return {"step": "awaiting-install", "engine": engine}
    how = acquisition.get("reason") or "is not installed here and cannot be installed by Eugene"
    raise ValueError(f"{engine} would prepare {name}, but {how}.")


def _discard(paths: tuple[Path, ...]) -> None:
    """What a listed preparation sent, gone from this node (B107); what
    cannot be removed now is left for the next preparation to replace."""
    for path in paths:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except OSError as exc:
            log.info("left %s on this node: %s", path, exc)


class NodeActions:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        # On the app, so uninstalling an engine can see a preparation using it.
        jobs = getattr(app.state, "preparations", None)
        if not isinstance(jobs, PreparationJobs):
            jobs = app.state.preparations = PreparationJobs()
        self.preparations: PreparationJobs = jobs
        #: Each finished preparation's files on their way to the library.
        self.senders: dict[str, Sender] = {}

    async def prepare(self, job: dict[str, Any]) -> Progress:
        """The operation's preparation: planned and started (or waiting for
        another) on first sight, its progress after (LS5)."""
        progress = self.preparations.poll(job["id"])
        if progress is not None:
            return progress
        wanted = job["intent"]["preparation"]
        kind = actions._engine_kind(wanted["engine"])
        adapter = adapter_for(kind)
        if adapter is None:
            raise ValueError(f"This agent has no engine {wanted['engine']}")
        model = job["model"]
        context = NodeContext(self.app)
        await actions.refresh_library_folders(context)
        rules = actions.effective_rules_for(context)
        root = model.get("root")
        if not root:
            raise ValueError(f"The library did not say which Library folder holds {model['name']}.")
        folder = resolve_model_path(root, rules).local_path
        local = resolve_model_path(model["path"], rules).local_path
        if not await asyncio.to_thread(Path(folder).is_dir):
            # Said as what it is, not as a missing model file further down.
            where = root if folder == root else f"{root} (here {folder})"
            raise ValueError(
                f"This node cannot reach the Library folder {where}: give the folder a "
                "mount for this node under Library, Folders, then prepare it again."
            )
        state = self.app.state.agent_state
        binary = await asyncio.to_thread(
            adapter.discover, configured=_configured_binary(adapter, state.get_config)
        )
        if binary is None:
            raise ValueError(f"{wanted['engine']} is not installed on this node.")

        def plan() -> Recipe:
            return adapter.plan_preparation(
                binary=binary,
                folder=Path(folder),
                work=preparation_folder(kind.value, root),
                model=Path(local),
                source_path=model["path"],
                context=wanted.get("contextSize"),
            )

        try:
            recipe = await asyncio.to_thread(plan)
        except PreparationError as exc:
            raise ValueError(str(exc)) from exc
        return self.preparations.start(job["id"], recipe, engine=kind.value)

    def prepared_body(self, job: dict[str, Any], result: PreparationResult) -> dict[str, Any]:
        """`POST /v1/run-operations/{id}/prepared`: the entry as the library
        spells it (its place in this node's folder for the preparation is
        its place in the Library folder, LS10), and its provenance."""
        root: str = job["model"]["root"]
        if result.root is None:
            raise ValueError("The preparation did not say which folder stands for the Library's.")
        try:
            inside = Path(os.path.relpath(result.entry, result.root))
        except ValueError as exc:  # another drive
            raise ValueError(f"{result.entry} is not under {result.root}") from exc
        if inside.parts[:1] == ("..",) or inside.is_absolute():
            raise ValueError(f"{result.entry} is not under {result.root}")
        entry = join_local(root, inside.parts)
        return {
            "name": result.name,
            "provenance": {
                "engine": job["intent"]["preparation"]["engine"],
                "entry": entry,
                "recipe": result.recipe,
                "recipeVersion": result.recipe_version,
                "source": result.source,
                # The library writes the provenance beside the entry, so the
                # files' places relative to the entry's folder are theirs.
                **result.facts,
            },
        }

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
        if not takes_context_size(engine):
            # vLLM's context is its own flag (`maxModelLen`), and a profile
            # carrying `contextSize` is refused at launch: it then takes the
            # model's own, which admission measures by its share (LS6).
            return None
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
        # A preparation whose operation was cancelled (or ended) stops (B53).
        self.actions.preparations.cancel_except(
            {o["id"] for o in listing.get("operations", []) if o.get("step") == "preparing"}
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
                        "preparing": "prepare",
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
            # The library judges (LS1); its verdicts come best first:
            # installed before not, runs before may_run, then preference.
            engines = await self.actions.engines()
            by_name = {e["engine"]: e for e in engines}
            verdicts = await judge(library, model, engines)
            preparation = (job.get("intent") or {}).get("preparation")
            if preparation:
                return _prepare_route(model, verdicts, by_name, preparation["engine"])
            selected = next(
                (v for v in verdicts if v["available"] and v["verdict"] in _RUNNABLE), None
            )
            if selected:
                return {"step": "settings", "engine": selected["engine"]}
            selected = next(
                (
                    v
                    for v in verdicts
                    if not v["available"]
                    and v["verdict"] in _RUNNABLE
                    and (by_name[v["engine"]].get("acquisition") or {}).get("installable")
                ),
                None,
            )
            if selected is None:
                raise ValueError(_why_none(model, verdicts, by_name))
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
            after = "preparing" if (job.get("intent") or {}).get("preparation") else "settings"
            return {"step": after if install is None else "installing", "install": install}
        if step == "preparing":
            return await self._prepare(library, base, job, params)
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

    async def _prepare(
        self, library: LibraryFitClient, base: str, job: dict[str, Any], params: dict[str, str]
    ) -> dict[str, Any]:
        """The `preparing` step (LS5): start or follow the node's job; once it
        is done the library lists the prepared model and the run goes on
        with it."""
        if job.get("preparedFrom") is not None:
            # Listed already; only the checkpoint after it was lost.
            self.actions.preparations.forget(job["id"])
            return {"step": "settings"}
        progress = await self.actions.prepare(job)
        status = progress.snapshot()
        if not progress.finished:
            return {"step": "preparing", "preparation": status}
        if progress.state != "done" or progress.result is None:
            where = await self._send_log(library, base, job, params, progress)
            self.actions.preparations.forget(job["id"])
            error = progress.error or "The preparation stopped without saying why."
            raise ValueError(error + (f" (its whole output: {where})" if where else ""))
        # LS10: the library writes the files into the Library folder.
        sender = self.actions.senders.get(job["id"])
        if sender is None:
            sender = self.actions.senders[job["id"]] = Sender(progress.result)
        try:
            sent = await sender.send(library, base, params, job["lease"])
        except ValueError:
            self.actions.senders.pop(job["id"], None)
            self.actions.preparations.forget(job["id"])
            raise
        if not sent:
            return {"step": "preparing", "preparation": sender.status(status)}
        body = self.actions.prepared_body(job, progress.result)
        try:
            await library.operation_request(
                "POST", base + "/prepared", params=params, json={**body, "lease": job["lease"]}
            )
        except httpx.HTTPStatusError as exc:
            if not 400 <= exc.response.status_code < 500:
                raise
            # The library's own words: the file it would not write, and why.
            try:
                detail = exc.response.json().get("detail")
            except ValueError:
                detail = None
            raise ValueError(
                f"The library could not list the prepared model: {detail or exc}"
            ) from exc
        self.actions.senders.pop(job["id"], None)
        self.actions.preparations.forget(job["id"])
        await asyncio.to_thread(_discard, progress.result.discard)
        return {"step": "settings", "preparation": status}

    async def _send_log(
        self,
        library: LibraryFitClient,
        base: str,
        job: dict[str, Any],
        params: dict[str, str],
        progress: Progress,
    ) -> str | None:
        """A failed preparation's log, sent to the library (B106), so the
        failure names a file the person can open beside their models: where
        it is then, as this node reaches the Library folder; this node's own
        copy when it could not be sent."""
        local, name = progress.log_file, progress.log_name
        if local is None or name is None or not local.is_file():
            return None
        try:
            data = await asyncio.to_thread(local.read_bytes)
            data = data[-CHUNK_BYTES:]  # its end says why it stopped
            lease = job["lease"]
            await library.operation_request(
                "PUT",
                base + "/files",
                params={**params, "path": name, "offset": "0", "lease": lease},
                content=data,
                headers={"content-type": "application/octet-stream"},
                timeout=REQUEST_TIMEOUT,
            )
            await library.operation_request(
                "POST",
                base + "/files/complete",
                params=params,
                json={
                    "lease": lease,
                    "path": name,
                    "sizeBytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
                timeout=REQUEST_TIMEOUT,
            )
        except httpx.HTTPStatusError as exc:
            log.warning("could not send the preparation's log: %s", detail_of(exc))
            return str(local)
        except httpx.HTTPError as exc:
            log.warning("could not send the preparation's log: %s", exc)
            return str(local)
        root = job["model"]["root"]
        rules = actions.effective_rules_for(NodeContext(self.app))
        folder = resolve_model_path(root, rules).local_path
        return join_local(folder, tuple(name.split("/")))

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
