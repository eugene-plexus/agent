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
  from the model file's size plus an estimated KV cache at that same
  context otherwise. `basis` says which.
* **Memory already promised**, from `reservations.py`. Free memory is a
  live reading and a launch spends it over the minutes it takes to copy
  and load a file, so what is left after the launches already under way
  is the budget rather than what the card reports (review §6.2 #19).

`unknown` never refuses. A verdict computed from a budget that could
not be measured is worse than no verdict, and three of the detection
paths are unverified on real hardware.

Since M11 the question before memory is answered here too: **is the
model on this host at all?** `modelPath` is the library's spelling and
the library may be on another machine, so it is resolved through this
node's `pathMappings` (`model_paths.py`) and the result is reported as
`location`. A path that names nothing here is a `refuse` with `fit:
unknown` -- the rule above is about a budget that could not be measured,
and a file that is not there is a measurement. Until this existed a
launch of such a path was admitted on faith, given a companion driver,
and crashed at spawn.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from . import model_copies
from ._generated.models import (
    Admission,
    AdmissionBasis,
    AdmissionBlocker,
    AdmissionDecision,
    AdmissionFit,
    ComputeDevice,
    ComputeDeviceKind,
    EngineKind,
    ModelLocation,
    RuntimeSpec,
    RuntimeStatus,
)
from ._http import internal_client, shared_internal_client
from .engines.devices import DeviceSnapshot
from .model_paths import PathRule, resolve_model_path
from .reservations import Reservation, held_bytes

log = logging.getLogger(__name__)

# Without the library's metadata the only number the agent has is the
# file's size. Labelled `file_size` so nobody mistakes it for
# arithmetic -- but it is no longer a flat fraction, and that mattered.
#
# **A flat tenth was context-blind** (review §6.2 #19): an 8B Q4 asked
# for at 128k needs about 17 GB once its KV cache is counted and was
# admitted against 5.5 GB. A KV cache is linear in context by
# construction -- one entry per token, per attention layer -- so a
# constant makes the discovery screen's context control change its own
# label, change the number echoed back, and change no verdict.
#
# The three constants below are the library's own fallback, duplicated
# rather than shared: components share schemas, not code. Keep them in
# step with `fit.py`'s `ESTIMATED_KV_FRACTION`,
# `ESTIMATED_KV_BASELINE_CONTEXT` and `DEFAULT_OVERHEAD_BYTES` -- two
# estimates of the same quantity that disagree is this project's
# signature defect, and the library's copy carries the reasoning.
ESTIMATED_KV_FRACTION = 0.15
ESTIMATED_KV_BASELINE_CONTEXT = 8192
OVERHEAD_BYTES = 1024**3

# What the estimate assumes when the spec leaves `contextSize` to the
# engine, which is what every profile that has not been edited does. Not
# the model's trained context: current files declare 262144 and almost
# nothing holds that, so assuming it would refuse everything. The number
# is reported on `Admission.contextLength`, because an assumption the
# caller cannot see is one they cannot argue with.
ASSUMED_CONTEXT_LENGTH = 8192

# llama.cpp: `--n-gpu-layers` at or above this is "everything". 99 is the
# idiom — no model this project has met has 99 layers, `-ngl 99` is what
# every profile here writes, and upstream's own examples use it. The
# first live run had 999 here and admitted a 36 GiB launch as "partial
# offload the operator chose" because the spec said 99.
FULL_OFFLOAD_LAYERS = 99

# The library answers from memory; anything slower is the library being
# down, and admission should fall back rather than wait on it.
_LIBRARY_TIMEOUT_SECONDS = 5.0

_PIN_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")

# Statuses in which a runtime holds memory on its device, or is on its
# way to holding it. `copying` is here since 2026-09-19: no process
# exists during it, which is why it was missed, and on a remote mount it
# is the longest part of a first launch (measured: 266 s for 23.8 GB
# over SMB). A runtime the operator would have to stop to make room is
# one this list has to name.
_HOLDING = frozenset(
    {
        RuntimeStatus.copying,
        RuntimeStatus.starting,
        RuntimeStatus.loading,
        RuntimeStatus.ready,
    }
)

# The subset that has not taken its memory yet, so a reservation for it
# still stands. `ready` is deliberately absent: the device snapshot
# counts a loaded model, and counting both would refuse a third launch
# that fits.
PENDING_STATUSES = frozenset({RuntimeStatus.copying, RuntimeStatus.starting, RuntimeStatus.loading})

# The existence check, as a module attribute so a test can describe a
# file that is not there the same way `model_size_bytes` lets it
# describe one that is 200 GiB.
path_exists: Callable[[str], bool] = os.path.exists


@dataclass(frozen=True)
class LibraryFit:
    """What the library said a model needs -- and weighs."""

    required_bytes: int
    verdict: str
    context_length: int | None
    max_context_length: int | None = None
    """The largest contextSize at which the weights and KV fit the budget
    the agent handed the library; 0 when the weights alone do not fit;
    None when the library did not say."""
    # What the library's scan recorded for the model: the whole model,
    # and the one file a launch line names. What `location` compares a
    # mapped file against (M11).
    size_bytes: int | None = None
    weights_size_bytes: int | None = None


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

    def __init__(self, base_url: str, service_token: str | None, *, transport: Any = None) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {service_token}"} if service_token else {}
        # Injectable so a test can stand in for the library without a
        # socket; production leaves it None.
        self._transport = transport
        self._own_client: httpx.AsyncClient | None = None

    def _client(self) -> httpx.AsyncClient:
        """The client these three reads share.

        An admission makes up to three calls to the library and used to
        build a client for each -- three certifi parses, ~104 ms of
        synchronous CPU apiece on the event loop, and the UI calls
        admission every time the new-profile form opens. In production
        the client is process-wide, keyed by the library's URL, so a
        second admission pays nothing at all. With an injected transport
        (tests) it is per-instance, because a shared one would outlive
        the fixture that owns the transport.
        """
        if self._transport is None:
            return shared_internal_client(f"library:{self._base}", timeout=_LIBRARY_TIMEOUT_SECONDS)
        if self._own_client is None:
            self._own_client = internal_client(
                timeout=_LIBRARY_TIMEOUT_SECONDS, transport=self._transport
            )
        return self._own_client

    @property
    def base_url(self) -> str:
        return self._base

    async def list_models(self) -> list[dict[str, Any]] | None:
        """Every model the library knows, raw -- for the mapping check
        behind `POST /v1/config/test`. None when it could not answer."""
        try:
            client = self._client()
            response = await client.get(f"{self._base}/v1/models", headers=self._headers)
            if response.status_code >= 400:
                log.info("library model list returned %d", response.status_code)
                return None
            models = response.json().get("models")
            return models if isinstance(models, list) else None
        except (httpx.HTTPError, ValueError) as e:
            log.info("library unreachable for its model list (%s)", e)
            return None

    async def folders(self) -> list[dict[str, Any]] | None:
        """The library's folders with their mounts, raw -- what this node
        inherits its path rules from (2026-09-14). None when it could not
        answer, and the node keeps its last copy."""
        try:
            client = self._client()
            response = await client.get(f"{self._base}/v1/folders", headers=self._headers)
            if response.status_code >= 400:
                log.info("library folder list returned %d", response.status_code)
                return None
            folders = response.json().get("folders")
            return folders if isinstance(folders, list) else None
        except (httpx.HTTPError, ValueError) as e:
            log.info("library unreachable for its folder list (%s)", e)
            return None

    async def fit(
        self,
        model_path: str,
        *,
        context_length: int | None,
        vram_bytes: int | None,
        ram_bytes: int | None,
    ) -> LibraryFit | None:
        try:
            client = self._client()
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
            max_context = response.json().get("maxContextLength")
            size = model.get("sizeBytes")
            weights = next(
                (
                    f.get("sizeBytes")
                    for f in model.get("files") or []
                    if isinstance(f, dict) and f.get("role") == "weights"
                ),
                size if model.get("fileCount") == 1 else None,
            )
            return LibraryFit(
                required_bytes=required,
                verdict=verdict,
                context_length=context if isinstance(context, int) else None,
                max_context_length=max_context if isinstance(max_context, int) else None,
                size_bytes=size if isinstance(size, int) else None,
                weights_size_bytes=weights if isinstance(weights, int) else None,
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

    llama.cpp with `gpuLayers` unset, negative (`-1` is "all" upstream),
    or at/above 99 is full offload; below is the operator choosing
    partial offload knowingly, which is what turns `tight` and `split`
    from refusals into admits. vLLM has no partial offload, so it is
    always full.
    """
    if spec.engine is not EngineKind.llama_cpp:
        return True
    flags = spec.flags or {}
    layers = flags.get("gpuLayers")
    if layers is None:
        return True
    try:
        count = int(layers)
    except (TypeError, ValueError):
        return True
    return count < 0 or count >= FULL_OFFLOAD_LAYERS


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


def file_size_requirement(size_bytes: int, context_length: int) -> int:
    """What a launch needs, when the file's size is all we know.

    Weights, plus a KV cache estimated as a fraction of them **per
    `ESTIMATED_KV_BASELINE_CONTEXT` tokens**, plus a flat overhead for
    compute buffers and the device context. The middle term is the whole
    of the fix: it is the one that moves when the operator moves the
    context control, and it was a constant.

    It is still an estimate and `basis: file_size` still says so. It is
    now an estimate of the right *shape* -- wrong by a factor, not by a
    factor that grows with the number being turned.

    `context_length` is required rather than defaulted. The caller has
    to settle on a number it can also report, and a second default here
    would be a place for the two to disagree that no check could see --
    the sabotage pass found exactly that and this is the answer to it.
    """
    kv = size_bytes * ESTIMATED_KV_FRACTION * (context_length / ESTIMATED_KV_BASELINE_CONTEXT)
    return int(size_bytes + kv) + OVERHEAD_BYTES


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
        if other.spec.name == spec.name or other.status not in _HOLDING:
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
    reservations: Sequence[Reservation] = (),
    size_of: Callable[[str], int | None] | None = None,
    mappings: Sequence[PathRule] = (),
    node_name: str | None = None,
    exists: Callable[[str], bool] | None = None,
    copy_settings: model_copies.CopySettings | None = None,
) -> Admission:
    """Measure `spec` against the device it targets, right now.

    `size_of` is the on-disk sizer, injectable so tests need not create
    a 200 GiB file to describe one — which Windows would zero-fill.
    `exists` is its sibling for the file being there at all. `mappings`
    are this node's `pathMappings`, applied to `modelPath` before
    anything on disk is asked about it (M11); `node_name` is for the
    refusal's prose.
    """
    sizer = size_of or model_size_bytes
    is_there = exists or path_exists
    targets = target_devices(spec, snapshot)
    warnings = list(snapshot.warnings)
    flags = spec.flags or {}
    context = flags.get("contextSize")
    context_length = (
        int(context) if isinstance(context, int | float | str) and str(context).isdigit() else None
    )
    max_context_length: int | None = None

    # Where the model is on this host, before any question about memory.
    # The stat runs off the event loop: a dead network share blocks for
    # as long as the OS takes to give up.
    location = await asyncio.to_thread(_locate, spec, mappings, is_there, sizer, copy_settings)
    if not location.exists:
        return _refuse_missing(
            spec, location, node_name=node_name, context_length=context_length, warnings=warnings
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
    # actually have to fit on — MINUS what this node has already promised
    # to launches that have not taken their memory yet. Free memory is a
    # live reading, so the card a second launch picks has to be the one
    # with room left after the first, not the one that still looks empty.
    def _spare(device: ComputeDevice) -> int:
        free = device.memoryFreeBytes
        if free is None:
            return -1
        return max(0, free - held_bytes(reservations, device_index=device.index, exclude=spec.name))

    device = max(targets, key=lambda d: (_spare(d), -(d.index or 0)))
    free = device.memoryFreeBytes
    total = device.memoryTotalBytes
    # This runtime's own reservation is never counted against it: a
    # restart re-measures a runtime that already holds one, and counting
    # it would refuse every restart.
    reserved = held_bytes(reservations, device_index=device.index, exclude=spec.name)
    # The budget the verdict is computed against. `freeBytes` on the wire
    # stays the card's own reading, because reporting the reduced number
    # there would be a claim about the card that is not true.
    budget = max(0, free - reserved) if free is not None else None
    # `ram_bytes` means "system RAM a partial offload can spill into,
    # BESIDE the device pool". On a discrete GPU that is real; on Apple
    # unified memory the "VRAM" pool and host RAM are the same silicon,
    # so passing both double-counts the machine — the exact "all host
    # RAM read as free VRAM" defect the MLX slice exists to avoid. A
    # metal device therefore gets no spillover term: its budget IS the
    # unified pool (and today `memoryFreeBytes` is absent there, so the
    # verdict is honestly `unknown` until a real Mac measures it).
    spillover_device = device.kind not in (ComputeDeviceKind.cpu, ComputeDeviceKind.metal)
    ram_available = snapshot.ram_available_bytes if spillover_device else 0
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
            vram_bytes=budget,
            ram_bytes=ram_available if spillover_device else None,
        )
        if answer is not None:
            required = answer.required_bytes
            basis = AdmissionBasis.metadata
            fit = _library_verdict(answer.verdict) if budget is not None else AdmissionFit.unknown
            if answer.context_length is not None:
                context_length = answer.context_length
            max_context_length = answer.max_context_length
            location, mismatch = _compare_sizes(location, answer)
            if mismatch is not None:
                warnings.append(mismatch)
    if required is None:
        # Already measured once, at the local path, when the location
        # was resolved.
        size = location.sizeBytes
        if size is not None:
            # The context the cache was sized for is reported, assumption
            # included: an assumption the caller cannot see is one they
            # cannot argue with, and this one decides the verdict.
            context_length = context_length or ASSUMED_CONTEXT_LENGTH
            required = file_size_requirement(size, context_length)
            fit = local_verdict(required, free=budget, total=total, ram_available=ram_available)
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
    # Slots divide the context they do not multiply the memory --
    # measured, see `per_request_context`. Said here because the runtime
    # will read back the divided number and the contract calls that
    # "clamped", which is the wrong diagnosis with the wrong remedy.
    slots_flag = flags.get("parallelSlots")
    slot_note = slot_division_note(
        context_length,
        int(slots_flag)
        if isinstance(slots_flag, int | float | str) and str(slots_flag).isdigit()
        else None,
    )
    slot_text = f" {slot_note[0].upper()}{slot_note[1:]}." if slot_note else ""
    # The number a refusal hands back. Discover scores at the library's
    # guidance context and a profile that leaves contextSize to the engine
    # is scored at the model's own, so the two screens disagreed about
    # context, never about the model -- and the refusal said "lower
    # contextSize" with no number to lower it to.
    fits_up_to = (
        f" It fits up to {max_context_length} context on this device." if max_context_length else ""
    )
    basis_text = (
        "library metadata"
        if basis is AdmissionBasis.metadata
        else "file size plus an estimated KV cache"
    )
    # Named separately from `held`, which lists runtimes. This is memory
    # nothing is holding yet and everything about the verdict turns on
    # it, so a card that reads 24 GiB free and refuses a 20 GiB model has
    # to say why in the same sentence as the numbers.
    reserved_text = (
        f" {_gib(reserved)} of it is reserved for a launch already under way, "
        f"leaving {_gib(budget)}."
        if reserved
        else ""
    )

    if fit is AdmissionFit.unknown:
        reason = (
            f"admit on faith: {spec.modelPath} could not be fully measured against {where} "
            f"(free {_gib(free)} of {_gib(total)}; required {_gib(required)} by {basis_text})."
            f"{reserved_text} "
            f"{held}"
        )
        warning: str | None = "; ".join(warnings) or "budget or size could not be measured"
    elif decision is AdmissionDecision.admit:
        reason = (
            f"admit: {spec.modelPath} needs about {_gib(required)}{ctx_text} ({basis_text}) and "
            f"{where} has {_gib(free)} free of {_gib(total)};{reserved_text} verdict {fit.value}"
            + (
                " with partial offload requested, so the spill is the operator's choice"
                if not full_offload and fit is not AdmissionFit.fits
                else ""
            )
            + f". {held}{slot_text}"
        )
        warning = "; ".join(warnings) or None
    else:
        lower = (
            f"set contextSize to {max_context_length} or below"
            if max_context_length
            else "lower contextSize"
        )
        # The remedy for a reservation is time, and it is the only one
        # on this list the operator does not have to do anything for --
        # so it goes first, or they go looking for memory they are about
        # to be given back.
        waiting = "Wait for the launch already under way, " if reserved else ""
        rest = (
            f"stop one of them, {lower}, set gpuLayers below full for partial offload, "
            "or pass ?force=true to launch anyway."
            if blockers
            else f"{lower}, set gpuLayers below full for partial offload, "
            "pick a smaller quant, or pass ?force=true to launch anyway."
        )
        fix = waiting + rest if waiting else f"{rest[0].upper()}{rest[1:]}"
        reason = (
            f"refuse: {spec.modelPath} needs about {_gib(required)}{ctx_text} ({basis_text}) but "
            f"{where} has {_gib(free)} free of {_gib(total)};{reserved_text} "
            f"verdict {fit.value}.{fits_up_to} "
            f"{held}{slot_text} {fix}"
        )
        warning = "; ".join(warnings) or None

    return Admission(
        decision=decision,
        fit=fit,
        basis=basis,
        requiredBytes=required,
        freeBytes=free,
        reservedBytes=reserved or None,
        totalBytes=total,
        device=device,
        contextLength=context_length,
        maxContextLength=max_context_length,
        blockers=blockers,
        location=location,
        reason=reason,
        warning=warning,
    )


def per_request_context(context_length: int | None, parallel_slots: int | None) -> int | None:
    """The window ONE request gets, which is not what was configured.

    **Measured against llama-server b11001** (0.4.1-dev, f266648fa) with
    a real model on this box: `-c 32768 --parallel 4` logs
    `llama_context: n_ctx = 32768` and `n_ctx_slot = 8192`, four slots of
    it, `kv_unified = 'false'`. `-c` is the TOTAL KV budget and
    `--parallel` divides it.

    That measurement is the whole of roadmap R1.3's half of review §6.3
    #36, which was a contradiction inside this repo: the config copy
    said `-c` was per-slot and that memory multiplied with slots, while
    `fit.py` and `admission.py` computed the opposite. The arithmetic was
    right. The copy is fixed, and this is the number it now needs.

    Why it matters past the copy: `/props` exposes only
    `default_generation_settings.n_ctx`, the per-slot value (`props.n_ctx`
    was absent on b11001), so a runtime launched at 32768 with 4 slots
    reports `capabilities.contextLength: 8192` — and the contract's word
    for a number that differs from the request is *clamped*, which sends
    an operator looking for memory they are not short of.
    """
    if context_length is None:
        return None
    slots = parallel_slots or 1
    return max(context_length // slots, 1) if slots > 1 else context_length


def slot_division_note(context_length: int | None, parallel_slots: int | None) -> str | None:
    """One sentence, only when the division actually happens.

    `None` at one slot on purpose. The default is one slot, so a
    sentence about division on every launch would appear on almost all
    of them and train people to skip the reason — which is the surface
    that has to carry a refusal's arithmetic.
    """
    if context_length is None or not parallel_slots or parallel_slots < 2:
        return None
    window = per_request_context(context_length, parallel_slots)
    assert window is not None
    return (
        f"{parallel_slots} slots divide that context, so each request gets "
        f"{window:,} tokens; the memory is the same either way"
    )


def _locate(
    spec: RuntimeSpec,
    mappings: Sequence[PathRule],
    is_there: Callable[[str], bool],
    sizer: Callable[[str], int | None],
    copy_settings: model_copies.CopySettings | None = None,
) -> ModelLocation:
    """Resolve `modelPath` the way a spawn would, and stat it.

    **The same seam as the launch, and that is the point.** If admission
    resolved a path the spawn would not, the two would disagree about
    which file is under discussion -- and the shape it would take is
    admission refusing a model that is sitting on this node's own disk,
    because it looked for it on a share that is down.
    """
    resolution = resolve_model_path(spec.modelPath, mappings)
    local = resolution.local_path
    if copy_settings is not None:
        local = model_copies.resolve_local_path(spec.modelPath, mappings, copy_settings).path
    present = bool(is_there(local))
    return ModelLocation(
        path=spec.modelPath,
        localPath=local,
        exists=present,
        mapping=resolution.rule.as_mapping() if resolution.rule is not None else None,
        sizeBytes=sizer(local) if present else None,
    )


def _compare_sizes(location: ModelLocation, answer: LibraryFit) -> tuple[ModelLocation, str | None]:
    """Attach what the library says the file weighs, and whether the
    file here agrees. A disagreement is a warning, not a refusal: a
    stale library scan is likelier than a mapping that points at a
    look-alike, and the engine will say if the file is broken."""
    if location.sizeBytes is None:
        return location, None
    expected = (
        answer.weights_size_bytes if os.path.isfile(location.localPath) else answer.size_bytes
    )
    if expected is None:
        return location, None
    matches = expected == location.sizeBytes
    located = location.model_copy(
        update={"librarySizeBytes": expected, "sizeMatchesLibrary": matches}
    )
    if matches:
        return located, None
    return located, (
        f"{location.localPath} is {location.sizeBytes} bytes here but the library lists "
        f"{expected} for {location.path}; the mapping may point at a different file, or the "
        f"library's scan is stale"
    )


def _refuse_missing(
    spec: RuntimeSpec,
    location: ModelLocation,
    *,
    node_name: str | None,
    context_length: int | None,
    warnings: list[str],
) -> Admission:
    """The one launch failure the dry run can predict with certainty."""
    where = f"on {node_name}" if node_name else "on this host"
    tab = f"Library -> {node_name or 'this machine'} -> Folders"
    if location.mapping is None:
        fix = (
            f"Nothing exists at {location.localPath}. If these files live on another machine "
            f"-- the Library's own folder, say -- mount that share here and say where: on the "
            f"Library folder's mounts, so every node of this kind inherits it, or as this "
            f"node's own override: {tab}."
        )
    else:
        fix = (
            f"The rule {location.mapping.from_} -> {location.mapping.to} applied and "
            f"nothing exists at {location.localPath}. Check that the share is mounted at "
            f"{location.mapping.to}, or fix the folder's mount or this node's override: {tab}."
        )
    return Admission(
        decision=AdmissionDecision.refuse,
        fit=AdmissionFit.unknown,
        basis=AdmissionBasis.file_size,
        contextLength=context_length,
        blockers=[],
        location=location,
        reason=(
            f"refuse: {spec.modelPath} is not {where}. {fix} Or pass ?force=true to launch anyway."
        ),
        warning="; ".join(warnings) or None,
    )


__all__ = [
    "ASSUMED_CONTEXT_LENGTH",
    "ESTIMATED_KV_FRACTION",
    "FULL_OFFLOAD_LAYERS",
    "FitSource",
    "LibraryFit",
    "LibraryFitClient",
    "RunningRuntime",
    "check_admission",
    "decide",
    "file_size_requirement",
    "local_verdict",
    "model_size_bytes",
    "path_exists",
    "pinned_indices",
    "target_devices",
    "wants_full_offload",
]
