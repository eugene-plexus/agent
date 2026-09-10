"""VRAM-aware admission: would this runtime fit where it is pointed?

Refuse, never queue (decided 2026-09-10). A queue is a promise about
when memory frees that the control plane cannot keep; the on-demand path
— `RuntimeSpec.startOnDemand`, driven by the gateway — is how a model
that does not fit right now gets loaded when it is actually asked for.

A refusal is a statement of arithmetic with an override, not a judgement
about the model: required bytes, free bytes, the device, the basis, and
which runtimes hold the memory, and `?force=true` on the launch if the
operator knows better. The alternative — spawn and let CUDA say no —
costs a minute of weight reading and an out-of-memory in a log, which
is the behaviour this replaces.

Two inputs, both measured on the host that will spawn:

* **Free memory per device**, live, from `engines/devices.py`. The
  budget is the *largest single* target device's free memory, because
  llama.cpp's default split is by layers and a model that fits across
  two cards and on neither is `split`, not `fits`.
* **Required bytes**, from the library's fit computation when a
  `library` component is in this agent's topology (the same arithmetic
  the discovery screen shows, at the context this spec asks for), and
  from the model file's size plus a fixed allowance otherwise. `basis`
  says which.

`unknown` never refuses. A verdict computed from a budget that could
not be measured is worse than no verdict, and three of the detection
paths are unverified on real hardware.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from ._generated.models import (
    Admission,
    AdmissionBasis,
    AdmissionBlocker,
    AdmissionDecision,
    AdmissionFit,
    ComputeDevice,
    ComputeDeviceKind,
    EngineKind,
    RuntimeSpec,
    RuntimeStatus,
)
from .engines.devices import DeviceSnapshot

log = logging.getLogger(__name__)

# Without the library's metadata the only number the agent has is the
# file's size; weights plus a tenth is the allowance for KV cache and
# compute buffers at a modest context. Labelled `file_size` so nobody
# mistakes it for arithmetic.
FILE_SIZE_ALLOWANCE = 0.10

# llama.cpp: `--n-gpu-layers` at or above this is "everything", which is
# what every profile that means full offload writes.
FULL_OFFLOAD_LAYERS = 999

# The library answers from memory; anything slower is the library being
# down, and admission should fall back rather than wait on it.
_LIBRARY_TIMEOUT_SECONDS = 5.0

_PIN_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")

_RUNNING = frozenset({RuntimeStatus.starting, RuntimeStatus.loading, RuntimeStatus.ready})


@dataclass(frozen=True)
class LibraryFit:
    """What the library said a model needs."""

    required_bytes: int
    verdict: str
    context_length: int | None


class FitSource(Protocol):
    """Anything that can answer "what does this model need" — the real
    library client, or a fake in tests."""

    async def fit(
        self,
        model_path: str,
        *,
        context_length: int | None,
        vram_bytes: int | None,
        ram_bytes: int | None,
    ) -> LibraryFit | None: ...


class LibraryFitClient:
    """Ask the library component for a model's fit.

    Two calls: `GET /v1/models?path=` to turn the operator's path into
    the library's id — normalization happens library-side, which is the
    point of that parameter — then `GET /v1/models/{id}/fit` with the
    budget this agent measured passed as overrides, so the verdict is
    about *this* device rather than whatever the library detected.
    """

    def __init__(self, base_url: str, service_token: str | None) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {service_token}"} if service_token else {}

    async def fit(
        self,
        model_path: str,
        *,
        context_length: int | None,
        vram_bytes: int | None,
        ram_bytes: int | None,
    ) -> LibraryFit | None:
        try:
            async with httpx.AsyncClient(timeout=_LIBRARY_TIMEOUT_SECONDS) as client:
                lookup = await client.get(
                    f"{self._base}/v1/models",
                    params={"path": model_path},
                    headers=self._headers,
                )
                if lookup.status_code >= 400:
                    log.info("library lookup for %s returned %d", model_path, lookup.status_code)
                    return None
                models = lookup.json().get("models") or []
                if not models:
                    log.info("library does not know %s; falling back to file size", model_path)
                    return None
                model = models[0]
                params: dict[str, Any] = {}
                if context_length is not None:
                    params["contextLength"] = context_length
                elif isinstance(model.get("contextLength"), int):
                    # The spec left context to the engine, which takes
                    # the model's own. Ask about that, not the library's
                    # guidance default.
                    params["contextLength"] = model["contextLength"]
                if vram_bytes is not None:
                    params["vramBytes"] = vram_bytes
                if ram_bytes is not None:
                    params["ramBytes"] = ram_bytes
                response = await client.get(
                    f"{self._base}/v1/models/{quote(str(model['id']), safe='')}/fit",
                    params=params,
                    headers=self._headers,
                )
                if response.status_code >= 400:
                    log.info("library fit for %s returned %d", model_path, response.status_code)
                    return None
                fit = response.json().get("fit") or {}
                required = fit.get("requiredBytes")
                verdict = fit.get("verdict")
                if not isinstance(required, int) or not isinstance(verdict, str):
                    return None
                context = fit.get("contextLength")
                return LibraryFit(
                    required_bytes=required,
                    verdict=verdict,
                    context_length=context if isinstance(context, int) else None,
                )
        except (httpx.HTTPError, ValueError) as e:
            log.info("library unreachable for fit (%s); falling back to file size", e)
            return None


@dataclass(frozen=True)
class RunningRuntime:
    """One other runtime on this host, as admission sees it."""

    spec: RuntimeSpec
    status: RuntimeStatus


# --- pure helpers -----------------------------------------------------------


def pinned_indices(env: dict[str, str] | None) -> set[int] | None:
    """Device ordinals a spec's `env` pins it to, or None for "all".

    Understands the comma-separated integer form. A UUID or `MIG-`
    selector is legitimate for the engine but not something the agent
    can match against `ComputeDevice.index`, so it reads as "all" with
    the caller told nothing narrower could be inferred.
    """
    if not env:
        return None
    for var in _PIN_ENV_VARS:
        raw = env.get(var)
        if raw is None:
            continue
        indices: set[int] = set()
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            if not part.lstrip("-").isdigit():
                return None
            indices.add(int(part))
        return indices
    return None


def target_devices(spec: RuntimeSpec, snapshot: DeviceSnapshot) -> list[ComputeDevice]:
    """The devices a launch of `spec` would land on.

    Accelerators when there are any, narrowed by the spec's pin; the CPU
    when the host has no accelerator, because a llama.cpp launch on such
    a box runs on the CPU and host memory is its budget.
    """
    accelerators = snapshot.accelerators()
    if not accelerators:
        cpu = snapshot.cpu()
        return [cpu] if cpu is not None else []
    pinned = pinned_indices(spec.env)
    if pinned is None:
        return accelerators
    chosen = [d for d in accelerators if d.index is not None and d.index in pinned]
    # A pin to a device this host does not have is an engine error at
    # spawn; admission measures against everything rather than nothing.
    return chosen or accelerators


def wants_full_offload(spec: RuntimeSpec) -> bool:
    """Whether the spec asks for the whole model on the accelerator.

    llama.cpp with `gpuLayers` unset or at/above 999 is full offload;
    below is the operator choosing partial offload knowingly, which is
    what turns `tight` and `split` from refusals into admits. vLLM has no
    partial offload, so it is always full.
    """
    if spec.engine is not EngineKind.llama_cpp:
        return True
    flags = spec.flags or {}
    layers = flags.get("gpuLayers")
    if layers is None:
        return True
    try:
        return int(layers) >= FULL_OFFLOAD_LAYERS
    except (TypeError, ValueError):
        return True


def model_size_bytes(model_path: str) -> int | None:
    """Bytes on disk: one file, or every file under a directory."""
    path = Path(model_path)
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return None
    return None


def local_verdict(
    required: int,
    *,
    free: int | None,
    total: int | None,
    ram_available: int | None,
) -> AdmissionFit:
    """The library's four words, computed here when the library is not."""
    if free is None or total is None:
        return AdmissionFit.unknown
    if required <= free:
        return AdmissionFit.fits
    if required <= total:
        return AdmissionFit.tight
    if ram_available is not None and required <= total + ram_available:
        return AdmissionFit.split
    if ram_available is None:
        # Cannot rule out host memory taking the spill; say so rather
        # than call it impossible.
        return AdmissionFit.split
    return AdmissionFit.no


def decide(fit: AdmissionFit, *, full_offload: bool) -> AdmissionDecision:
    """The design's table, as code."""
    if fit is AdmissionFit.fits or fit is AdmissionFit.unknown:
        return AdmissionDecision.admit
    if fit is AdmissionFit.no:
        return AdmissionDecision.refuse
    # tight or split: refuse a full-offload launch, admit partial offload.
    return AdmissionDecision.refuse if full_offload else AdmissionDecision.admit


def _library_verdict(word: str) -> AdmissionFit:
    try:
        return AdmissionFit(word)
    except ValueError:
        return AdmissionFit.unknown


def _blockers(
    spec: RuntimeSpec,
    targets: list[ComputeDevice],
    snapshot: DeviceSnapshot,
    running: list[RunningRuntime],
) -> list[AdmissionBlocker]:
    """Runtimes holding memory on the target device(s), evictable first."""
    target_indices = {d.index for d in targets if d.index is not None}
    out: list[AdmissionBlocker] = []
    for other in running:
        if other.spec.name == spec.name or other.status not in _RUNNING:
            continue
        theirs = target_devices(other.spec, snapshot)
        their_indices = {d.index for d in theirs if d.index is not None}
        if target_indices and their_indices and not (target_indices & their_indices):
            continue
        timeout = other.spec.idleUnloadSeconds
        out.append(
            AdmissionBlocker(
                name=other.spec.name,
                status=other.status,
                idleUnloadSeconds=timeout if timeout else None,
                evictable=bool(timeout),
            )
        )
    out.sort(key=lambda b: (not b.evictable, b.name))
    return out


def _gib(count: int | None) -> str:
    if count is None:
        return "unknown"
    return f"{count / (1024**3):.1f} GiB"


# --- the decision -----------------------------------------------------------


async def check_admission(
    spec: RuntimeSpec,
    *,
    snapshot: DeviceSnapshot,
    library: FitSource | None,
    running: list[RunningRuntime],
    size_of: Callable[[str], int | None] | None = None,
) -> Admission:
    """Measure `spec` against the device it targets, right now.

    `size_of` is the on-disk sizer, injectable so tests need not create
    a 200 GiB file to describe one — which Windows would zero-fill.
    """
    sizer = size_of or model_size_bytes
    targets = target_devices(spec, snapshot)
    warnings = list(snapshot.warnings)
    flags = spec.flags or {}
    context = flags.get("contextSize")
    context_length = (
        int(context) if isinstance(context, int | float | str) and str(context).isdigit() else None
    )

    if not targets:
        return Admission(
            decision=AdmissionDecision.admit,
            fit=AdmissionFit.unknown,
            basis=AdmissionBasis.file_size,
            contextLength=context_length,
            blockers=[],
            reason=(
                f"admit on faith: no compute device could be detected on this host, so "
                f"{spec.modelPath} was not measured against anything."
            ),
            warning="; ".join(warnings) or "no device detected",
        )

    # Largest single device by free memory — the card the engine would
    # actually have to fit on.
    device = max(targets, key=lambda d: (d.memoryFreeBytes or -1, -(d.index or 0)))
    free = device.memoryFreeBytes
    total = device.memoryTotalBytes
    ram_available = snapshot.ram_available_bytes if device.kind is not ComputeDeviceKind.cpu else 0
    blockers = _blockers(spec, targets, snapshot, running)
    full_offload = wants_full_offload(spec)

    # Required bytes: the library when it answers, the file otherwise.
    required: int | None = None
    basis = AdmissionBasis.file_size
    fit = AdmissionFit.unknown
    if library is not None:
        answer = await library.fit(
            spec.modelPath,
            context_length=context_length,
            vram_bytes=free,
            ram_bytes=ram_available if device.kind is not ComputeDeviceKind.cpu else None,
        )
        if answer is not None:
            required = answer.required_bytes
            basis = AdmissionBasis.metadata
            fit = _library_verdict(answer.verdict) if free is not None else AdmissionFit.unknown
            if answer.context_length is not None:
                context_length = answer.context_length
    if required is None:
        size = sizer(spec.modelPath)
        if size is not None:
            required = int(size * (1 + FILE_SIZE_ALLOWANCE))
            fit = local_verdict(required, free=free, total=total, ram_available=ram_available)
        else:
            warnings.append(f"{spec.modelPath} could not be sized on disk")

    decision = decide(fit, full_offload=full_offload)
    where = f"device {device.index} ({device.name or device.kind.value})"
    held = (
        "Held by: "
        + ", ".join(
            f"{b.name} ({b.status.value if b.status else 'running'}"
            + (", evictable" if b.evictable else "")
            + ")"
            for b in blockers
        )
        + "."
        if blockers
        else "Nothing else of ours holds memory on it."
    )
    ctx_text = f" at {context_length} context" if context_length else ""
    basis_text = "library metadata" if basis is AdmissionBasis.metadata else "file size plus 10%"

    if fit is AdmissionFit.unknown:
        reason = (
            f"admit on faith: {spec.modelPath} could not be fully measured against {where} "
            f"(free {_gib(free)} of {_gib(total)}; required {_gib(required)} by {basis_text}). "
            f"{held}"
        )
        warning: str | None = "; ".join(warnings) or "budget or size could not be measured"
    elif decision is AdmissionDecision.admit:
        reason = (
            f"admit: {spec.modelPath} needs about {_gib(required)}{ctx_text} ({basis_text}) and "
            f"{where} has {_gib(free)} free of {_gib(total)}; verdict {fit.value}"
            + (
                " with partial offload requested, so the spill is the operator's choice"
                if not full_offload and fit is not AdmissionFit.fits
                else ""
            )
            + f". {held}"
        )
        warning = "; ".join(warnings) or None
    else:
        fix = (
            "Stop one of them, lower contextSize, set gpuLayers below full for partial offload, "
            "or pass ?force=true to launch anyway."
            if blockers
            else "Lower contextSize, set gpuLayers below full for partial offload, pick a smaller "
            "quant, or pass ?force=true to launch anyway."
        )
        reason = (
            f"refuse: {spec.modelPath} needs about {_gib(required)}{ctx_text} ({basis_text}) but "
            f"{where} has {_gib(free)} free of {_gib(total)}; verdict {fit.value}. {held} {fix}"
        )
        warning = "; ".join(warnings) or None

    return Admission(
        decision=decision,
        fit=fit,
        basis=basis,
        requiredBytes=required,
        freeBytes=free,
        totalBytes=total,
        device=device,
        contextLength=context_length,
        blockers=blockers,
        reason=reason,
        warning=warning,
    )


__all__ = [
    "FILE_SIZE_ALLOWANCE",
    "FULL_OFFLOAD_LAYERS",
    "FitSource",
    "LibraryFit",
    "LibraryFitClient",
    "RunningRuntime",
    "check_admission",
    "decide",
    "local_verdict",
    "model_size_bytes",
    "pinned_indices",
    "target_devices",
    "wants_full_offload",
]
