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
  budget is one card's free memory, or, for a launch the engine spreads
  across several cards (`place`), what those cards have free between
  them. Until 2026-09-27 it was always the largest single card, on the
  reasoning that a model which fits across two cards and on neither is
  `split` -- which is wrong: llama.cpp puts whole layers on each card,
  so that model runs entirely in GPU memory, and `split` made admission
  refuse it.
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
import re
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
    EngineFitModel,
    EngineKind,
    FitModelKind,
    ModelLocation,
    RuntimeSpec,
    RuntimeStatus,
)
from ._http import internal_client, shared_internal_client
from .engines.devices import DeviceSnapshot
from .enrollment import problem_detail
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
        unified_memory: bool = False,
        gpu_count: int = 1,
        fit_model: str | None = None,
        gpu_memory_utilization: float | None = None,
        vram_total_bytes: int | None = None,
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
        self.last_error: str | None = None
        """Why the last folder read returned None, for the node's copy to
        keep and the Folders page to show."""

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

    async def operation_request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Authenticated, fresh-token transport for assigned durable run work."""
        response = await self._client().request(
            method, f"{self._base}{path}", headers=self._headers, **kwargs
        )
        response.raise_for_status()
        return response.json() if response.content else None

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
        self.last_error = None
        try:
            client = self._client()
            response = await client.get(f"{self._base}/v1/folders", headers=self._headers)
            if response.status_code >= 400:
                self.last_error = (
                    f"The Library at {self._base} answered {response.status_code}: "
                    f"{problem_detail(response)}"
                )
                log.warning("library folder list: %s", self.last_error)
                return None
            folders = response.json().get("folders")
            if isinstance(folders, list):
                return folders
            self.last_error = f"The Library at {self._base} answered without a folder list."
            return None
        except (httpx.HTTPError, ValueError) as e:
            # A timeout's text is the empty string; its type is the reason.
            self.last_error = (
                f"The Library at {self._base} could not be reached: {e or type(e).__name__}"
            )
            log.warning("library folder list: %s", self.last_error)
            return None

    async def fit(
        self,
        model_path: str,
        *,
        context_length: int | None,
        vram_bytes: int | None,
        ram_bytes: int | None,
        unified_memory: bool = False,
        gpu_count: int = 1,
        fit_model: str | None = None,
        gpu_memory_utilization: float | None = None,
        vram_total_bytes: int | None = None,
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
            if unified_memory:
                params["unifiedMemory"] = "true"
            if gpu_count > 1:
                params["gpuCount"] = gpu_count
            # The engine's own fit model (LS6); absent, the library's
            # arithmetic is `spill`, llama.cpp's.
            if fit_model is not None:
                params["fitModel"] = fit_model
            if gpu_memory_utilization is not None:
                params["gpuMemoryUtilization"] = gpu_memory_utilization
            if vram_total_bytes is not None:
                params["vramTotalBytes"] = vram_total_bytes
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


# Which kind of card each pin variable names. A pin reaches its own
# backend only: CUDA_VISIBLE_DEVICES hides CUDA devices and says nothing
# to Vulkan, which is why a pinned runtime on a `+vulkan` build is also
# given an empty GGML_VK_VISIBLE_DEVICES (`LlamaCppAdapter.default_env`).
_PIN_KINDS = {
    "CUDA_VISIBLE_DEVICES": ComputeDeviceKind.cuda,
    "HIP_VISIBLE_DEVICES": ComputeDeviceKind.rocm,
    "ROCR_VISIBLE_DEVICES": ComputeDeviceKind.rocm,
}

# `CUDA0`, `Vulkan1`, `ROCm0`: llama.cpp's own device names.
_DEVICE_NAME_RE = re.compile(r"^(?P<backend>[A-Za-z]+)(?P<index>\d+)$")


def pin_kind(env: dict[str, str] | None) -> ComputeDeviceKind | None:
    """The kind of card the spec's pin variable names, if it pins."""
    for var in _PIN_ENV_VARS:
        if env and env.get(var) is not None:
            return _PIN_KINDS.get(var)
    return None


def explicit_devices(
    spec: RuntimeSpec, accelerators: Sequence[ComputeDevice]
) -> tuple[list[ComputeDevice], list[str]] | None:
    """The runtime's own `devices` list (llama.cpp's `--device`), mapped.

    `CUDA<n>` is the n-th CUDA card. A Vulkan name cannot be mapped:
    Vulkan numbers every GPU in the machine, the NVIDIA ones included,
    in an order no tool here reads, so `Vulkan1` might be the integrated
    GPU or the second card. Those come back unmapped, and the launch is
    admitted without a memory check rather than measured against a
    guess. This is the expert's path (an integrated GPU as overflow,
    measured once on one rig and not judged: see the two-card record),
    and `unknown` never refuses. `None` means the runtime names no list.
    """
    if spec.engine is not EngineKind.llama_cpp:
        return None
    raw = (spec.flags or {}).get("devices")
    if not isinstance(raw, str) or not raw.strip():
        return None
    mapped: list[ComputeDevice] = []
    unmapped: list[str] = []
    for name in (part.strip() for part in raw.split(",") if part.strip()):
        if name.lower() == "none":
            continue
        match = _DEVICE_NAME_RE.match(name)
        backend = match.group("backend").lower() if match else ""
        kind = {"cuda": ComputeDeviceKind.cuda, "rocm": ComputeDeviceKind.rocm}.get(backend)
        device = (
            next(
                (
                    d
                    for d in accelerators
                    if d.kind is kind and d.index == int(match.group("index"))
                ),
                None,
            )
            if match and kind is not None
            else None
        )
        if device is None:
            unmapped.append(name)
        elif device not in mapped:
            mapped.append(device)
    return mapped, unmapped


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
    explicit = explicit_devices(spec, accelerators)
    if explicit is not None:
        mapped, unmapped = explicit
        if mapped:
            return mapped
        if not unmapped:
            # `--device none`: the processor only.
            cpu = snapshot.cpu()
            return [cpu] if cpu is not None else accelerators
        return accelerators
    pinned = pinned_indices(spec.env)
    if pinned is None:
        return accelerators
    kind = pin_kind(spec.env)
    chosen = [
        d
        for d in accelerators
        if d.index is not None and d.index in pinned and (kind is None or d.kind is kind)
    ]
    # A pin to a device this host does not have is an engine error at
    # spawn; admission measures against everything rather than nothing.
    return chosen or accelerators


# --- which cards a launch uses --------------------------------------------


@dataclass(frozen=True)
class Placement:
    """Where one launch lands: one card, or every card it spreads across.

    **Added 2026-09-27.** Admission used to take the largest single card
    and score against it, and its own docstring said a model that fits
    across two cards and on neither is `split`. That is not what split
    means: llama.cpp puts whole layers on each card, so a model spread
    across two 5090s runs entirely in GPU memory, and `split` (spilling
    into system RAM) made a full-offload launch refuse a model the engine
    would have served.
    """

    main: ComputeDevice
    devices: tuple[ComputeDevice, ...]
    shares: tuple[float, ...]
    """Each card's fraction of the model, in `devices` order."""
    free: int | None
    total: int | None
    reserved: int
    budget: int | None
    """What the verdict is computed against. For one card, its free
    memory less what is reserved on it. For several, the most the model
    can be while every card's share still fits in what that card has
    left, which is the sum when the shares follow free memory."""

    @property
    def spread(self) -> bool:
        return len(self.devices) > 1


def _int_flag(value: object, default: int) -> int:
    try:
        return int(value) if value is not None else default  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default


def spread_devices(spec: RuntimeSpec, targets: list[ComputeDevice]) -> list[ComputeDevice]:
    """The cards one launch of `spec` spreads the model across, if several.

    * **llama.cpp** splits across every visible card unless `splitMode`
      is `none` (then it uses one, `mainGpu`). Visible means what
      `target_devices` already narrowed by `CUDA_VISIBLE_DEVICES` or
      `HIP_VISIBLE_DEVICES`, so a runtime pinned to one card is one card.
    * **vLLM** uses `tensorParallelSize` x `pipelineParallelSize` cards,
      one by default.

    Cards of one kind only: the build computes on one backend, and the
    device list is ordered so the first accelerator is the one it uses.
    Empty means one card.
    """
    accelerators = [d for d in targets if d.kind is not ComputeDeviceKind.cpu]
    if len(accelerators) < 2:
        return []
    kind = accelerators[0].kind
    same = [d for d in accelerators if d.kind is kind]
    flags = spec.flags or {}
    if spec.engine is EngineKind.llama_cpp:
        if str(flags.get("splitMode") or "").strip().lower() == "none":
            return []
        # **Every card in the list, whatever its kind** (2026-09-27). The
        # device list names exactly what the build chosen for this
        # machine uses, and the `+vulkan` build uses an AMD or Intel card
        # beside the NVIDIA one through its second backend.
        return accelerators
    if spec.engine is EngineKind.vllm:
        count = _int_flag(flags.get("tensorParallelSize"), 1) * _int_flag(
            flags.get("pipelineParallelSize"), 1
        )
        return same[:count] if count > 1 and len(same) > 1 else []
    return []


def split_shares(
    spec: RuntimeSpec,
    devices: Sequence[ComputeDevice],
    spare: Callable[[ComputeDevice], int | None],
) -> list[float]:
    """Each card's fraction of the model, in order.

    `tensorSplit` when the runtime sets it, as llama.cpp reads it: one
    proportion per card in order, a missing one being zero. Otherwise
    each card's share of the memory left free, which is llama.cpp's own
    default (it splits by free memory). vLLM's tensor parallelism shards
    evenly.
    """
    count = len(devices)
    if spec.engine is EngineKind.vllm:
        return [1.0 / count] * count
    raw = (spec.flags or {}).get("tensorSplit")
    if isinstance(raw, str) and raw.strip():
        try:
            values = [float(part) for part in raw.split(",") if part.strip()]
        except ValueError:
            values = []
        values = (values + [0.0] * count)[:count]
        total = sum(v for v in values if v > 0)
        if total > 0:
            return [max(v, 0.0) / total for v in values]
    spares = [max(spare(d) or 0, 0) for d in devices]
    total_spare = sum(spares)
    if total_spare <= 0:
        return [1.0 / count] * count
    return [s / total_spare for s in spares]


def place(
    spec: RuntimeSpec,
    targets: list[ComputeDevice],
    reservations: Sequence[Reservation] = (),
) -> Placement:
    """Which card, or cards, this launch measures against."""

    def held(device: ComputeDevice) -> int:
        return held_bytes(
            reservations,
            device_index=device.index,
            exclude=spec.name,
            device_kind=device.kind.value,
        )

    def spare(device: ComputeDevice) -> int | None:
        free = device.memoryFreeBytes
        return None if free is None else max(0, free - held(device))

    spread = spread_devices(spec, targets)
    if not spread and explicit_devices(spec, targets) not in (None, ([], [])):
        mapped, unmapped_names = explicit_devices(spec, targets) or ([], [])
        if unmapped_names:
            # A device named that no reading here covers: nothing honest
            # to measure against, so the budget is unknown.
            device = mapped[0] if mapped else targets[0]
            return Placement(
                main=device,
                devices=(device,),
                shares=(1.0,),
                free=device.memoryFreeBytes,
                total=device.memoryTotalBytes,
                reserved=held(device),
                budget=None,
            )
    if not spread:
        # Largest single device by free memory -- the card the engine
        # would have to fit on -- MINUS what this node has already
        # promised to launches that have not taken their memory yet.
        device = max(
            targets, key=lambda d: (s if (s := spare(d)) is not None else -1, -(d.index or 0))
        )
        reserved = held(device)
        free = device.memoryFreeBytes
        return Placement(
            main=device,
            devices=(device,),
            shares=(1.0,),
            free=free,
            total=device.memoryTotalBytes,
            reserved=reserved,
            budget=max(0, free - reserved) if free is not None else None,
        )

    explicit = explicit_devices(spec, targets)
    unmapped = bool(explicit and explicit[1])
    shares = split_shares(spec, spread, spare)
    kept = [(d, p) for d, p in zip(spread, shares, strict=True) if p > 0]
    devices = tuple(d for d, _ in kept)
    fractions = tuple(p for _, p in kept)
    spares = [spare(d) for d in devices]
    budget: int | None
    if unmapped or any(s is None for s in spares):
        budget = None
    else:
        # The card that fills first bounds the whole model.
        budget = int(min((s or 0) / p for s, p in zip(spares, fractions, strict=True)))
    frees = [d.memoryFreeBytes for d in devices]
    totals = [d.memoryTotalBytes for d in devices]
    main_index = (spec.flags or {}).get("mainGpu")
    main = next(
        (d for d in devices if main_index is not None and d.index == _int_flag(main_index, -1)),
        devices[0],
    )
    return Placement(
        main=main,
        devices=devices,
        shares=fractions,
        free=None if any(f is None for f in frees) else sum(f or 0 for f in frees),
        total=None if any(t is None for t in totals) else sum(t or 0 for t in totals),
        reserved=sum(held(d) for d in devices),
        budget=budget,
    )


def admission_split(
    spec: RuntimeSpec,
    admission: Admission,
    reservations: Sequence[Reservation] = (),
    *,
    size: int | None = None,
) -> list[tuple[str, int, int]]:
    """A split launch's promise, divided the way its weights will be.

    From the answer the route already has, the cards and their free
    memory, less what the ledger holds on each: the same numbers the
    verdict divided the model by. A first version divided by free memory
    alone, and a test of two equal cards, one spoken for, caught it
    promising half of a new launch to the card that had no room.

    `size` is the promise being divided, when it is not what admission
    measured (`reservation_bytes`).
    """
    devices = admission.devices or []
    required = size if size is not None else (admission.requiredBytes or 0)
    if len(devices) < 2 or required <= 0:
        return []

    def spare(device: ComputeDevice) -> int | None:
        free = device.memoryFreeBytes
        if free is None:
            return None
        return max(
            0,
            free
            - held_bytes(
                reservations,
                device_index=device.index,
                exclude=spec.name,
                device_kind=device.kind.value,
            ),
        )

    shares = split_shares(spec, devices, spare)
    return [
        (d.kind.value, d.index, int(required * p))
        for d, p in zip(devices, shares, strict=True)
        if d.index is not None and p > 0
    ]


# llama.cpp's `--fit-target` default: the memory its fit leaves free on
# every device (`-fitt`, "default: 1024" MiB, b11375's help).
DEFAULT_FIT_MARGIN_BYTES = 1024 * 1024**2

_CONTEXT_ARGS = ("-c", "--ctx-size")


def reservation_bytes(spec: RuntimeSpec, admission: Admission, *, engine_places: bool) -> int:
    """What the ledger promises for a launch admission just measured.

    **What the engine will take, which is what admission measured except
    in one case** (drift audit 2026-10-03, settings never lie): a
    llama.cpp launch that leaves `contextSize` unset on a build whose fit
    is on. llama.cpp does not run that at the context admission assumed;
    `common/fit.cpp` (b11375) starts at the model's trained context and
    shrinks it only until every card is full to `--fit-target`. So:

    * sized at the model's own context (the library answered, `metadata`):
      that, or the cards' room less the margin when that is smaller;
    * sized at an ASSUMED context (`file_size`, `ASSUMED_CONTEXT_LENGTH`):
      the cards' room less the margin -- what the fit fills for any model
      whose trained context is larger than the assumption, which is every
      current one. The ledger counted 8,192 tokens of a card the engine
      had filled.

    The room is what admission reported free less what other launches
    already hold. Over-counting for a model with a tiny trained context
    lasts one load and refuses rather than admits, the direction the
    ledger already accepts (`reservations.py`).
    """
    required = admission.requiredBytes or 0
    if required <= 0 or admission.freeBytes is None:
        return required
    if not _engine_sizes_context(spec, engine_places=engine_places):
        return required
    cards = len(admission.devices) if admission.devices else 1
    margin = _fit_margin_bytes(spec) * cards
    room = max(admission.freeBytes - (admission.reservedBytes or 0) - margin, 0)
    if admission.basis is AdmissionBasis.metadata:
        return min(required, room)
    return room


def _engine_sizes_context(spec: RuntimeSpec, *, engine_places: bool) -> bool:
    """Whether llama.cpp, not the profile, decides this launch's context."""
    if spec.engine is not EngineKind.llama_cpp or not engine_places or fit_disabled(spec):
        return False
    if (spec.flags or {}).get("contextSize") is not None:
        return False
    args = [a.strip() for a in (spec.extraArgs or [])]
    return not any(a in _CONTEXT_ARGS or a.startswith("--ctx-size=") for a in args)


def _fit_margin_bytes(spec: RuntimeSpec) -> int:
    """`memoryMargin` (MiB, `--fit-target`), or llama.cpp's own default."""
    margin = (spec.flags or {}).get("memoryMargin")
    if isinstance(margin, int) and not isinstance(margin, bool) and margin >= 0:
        return margin * 1024**2
    return DEFAULT_FIT_MARGIN_BYTES


def wants_full_offload(spec: RuntimeSpec, *, engine_places: bool = False) -> bool:
    """Whether the spec asks for the whole model on the accelerator.

    llama.cpp with `gpuLayers` negative (`-1` is "all" upstream) or at or
    above 99 is full offload; below is the operator choosing partial
    offload knowingly, which is what turns `tight` and `split` from
    refusals into admits. vLLM has no partial offload, so it is always
    full.

    **Unset depends on the build** (2026-09-30, moe-aware-fit call A).
    `engine_places` is the caller saying this spec's llama-server lists
    `--fit` in its own help, which every build the installers ship does,
    on by default. There an unset `gpuLayers` asks llama.cpp to place the
    model, experts or layers in host memory as needed -- a partial launch
    the engine arranges, not a demand for the whole model on the card. It
    was read as full offload, which refused every `split` a profile build
    saves (it leaves `gpuLayers` unset on purpose) on exactly the cards
    where the build had just measured it running. A spec passing
    `--fit off` gets the old reading back.
    """
    if spec.engine is not EngineKind.llama_cpp:
        return True
    flags = spec.flags or {}
    layers = flags.get("gpuLayers")
    if layers is None:
        return not (engine_places and not fit_disabled(spec))
    try:
        count = int(layers)
    except (TypeError, ValueError):
        return True
    return count < 0 or count >= FULL_OFFLOAD_LAYERS


def fit_disabled(spec: RuntimeSpec) -> bool:
    """Whether the spec's raw arguments switch llama.cpp's own fit off."""
    args = [a.strip().lower() for a in (spec.extraArgs or [])]
    for i, arg in enumerate(args):
        if arg in ("--fit=off", "-fit=off"):
            return True
        if arg in ("--fit", "-fit") and i + 1 < len(args) and args[i + 1] == "off":
            return True
    return False


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


# --- each engine's own fit model (LS6) --------------------------------------


def engine_fit_model(kind: EngineKind) -> EngineFitModel | None:
    """What the engine's adapter declares (`EngineDescriptor.fit`); None: it
    has no fit model, and its fit is not estimated."""
    from .engines import ADAPTERS  # the adapters import the supervisor

    adapter = ADAPTERS.get(kind)
    return adapter.fit_model if adapter is not None else None


def engine_name(kind: EngineKind) -> str:
    return _ENGINE_NAMES.get(kind, kind.value)


_ENGINE_NAMES = {
    EngineKind.llama_cpp: "llama.cpp",
    EngineKind.vllm: "vLLM",
    EngineKind.mlx: "MLX",
    EngineKind.kev: "Kev",
    EngineKind.strata: "Strata",
}


def share_of(spec: RuntimeSpec, declared: EngineFitModel) -> float | None:
    """A share-taking engine's share of each card: the launch's own
    (`gpuMemoryUtilization`), else the engine's default."""
    asked = (spec.flags or {}).get("gpuMemoryUtilization")
    if isinstance(asked, int | float) and not isinstance(asked, bool) and 0 < asked <= 1:
        return float(asked)
    return declared.gpuMemoryUtilization


#: Strata's own words mapped to admission's (`FitVerdict` in common.yaml).
_STRATA_FIT = {
    "fits": AdmissionFit.fits,
    "tight": AdmissionFit.tight,
    "split": AdmissionFit.split,
    "no": AdmissionFit.no,
    "unknown": AdmissionFit.unknown,
}


def strata_admission(
    spec: RuntimeSpec,
    *,
    snapshot: DeviceSnapshot,
    location: ModelLocation,
    reservations: Sequence[Reservation],
    running: list[RunningRuntime],
    warnings: list[str],
) -> Admission:
    """Strata by its setup's own table for this model on this node (LS6).

    Its setup's verdict (`strata_models.setup_fit`) on this node's total RAM
    and the card it would use: *fits* and its RAM-budget mode admit, its
    low-RAM mode (`split`) admits, *tight* (the system would page its
    experts) and *does not fit* refuse, as setup stops by default. Then what
    is free now: in the mode that copies every expert into RAM, less RAM free
    than the experts is `tight`, since the system would page them.

    Strata fills its card's free memory with its expert cache, so the ledger
    holds that card's free memory while it loads (`requiredBytes`): a second
    launch on the card in that window is refused, not starved.
    """
    from .engines.strata import choice_of_runtime, strata_card
    from .engines.strata_models import experts_in_ram_bytes, setup_fit

    card = strata_card(snapshot)
    choice = choice_of_runtime(Path(location.localPath or spec.modelPath))
    held = (
        held_bytes(
            reservations,
            device_index=card.index,
            exclude=spec.name,
            device_kind=card.kind.value,
        )
        if card is not None
        else 0
    )
    free = card.memoryFreeBytes if card is not None else None
    room = max(0, free - held) if free is not None else None
    blockers = _blockers(spec, [card], snapshot, running) if card is not None else []
    where = (
        f"device {card.index} ({card.name or card.kind.value})" if card is not None else "no card"
    )

    def answer(
        decision: AdmissionDecision, fit: AdmissionFit, reason: str, warning: str | None = None
    ) -> Admission:
        return Admission(
            decision=decision,
            fit=fit,
            basis=AdmissionBasis.engine_table,
            # Nothing measured, nothing held: an unknown never refuses.
            requiredBytes=room if fit is not AdmissionFit.unknown else None,
            freeBytes=free,
            reservedBytes=held or None,
            totalBytes=card.memoryTotalBytes if card is not None else None,
            device=card,
            blockers=blockers,
            location=location,
            reason=reason,
            warning=warning if warning is not None else ("; ".join(warnings) or None),
        )

    if choice is None:
        return answer(
            AdmissionDecision.admit,
            AdmissionFit.unknown,
            f"admit on faith: {spec.modelPath} names no model on Strata's list (made "
            "outside Eugene, or its source is not known), so its setup's table cannot say "
            "what it needs.",
            "; ".join(warnings) or "Strata's fit is not estimated for this model",
        )
    ram_total = snapshot.ram_total_bytes
    if ram_total is None:
        return answer(
            AdmissionDecision.admit,
            AdmissionFit.unknown,
            f"admit on faith: this machine's RAM could not be read, and Strata's setup "
            f"sizes {choice.model} by it.",
            "; ".join(warnings) or "RAM could not be read",
        )
    vram_gib = (card.memoryTotalBytes or 0) / 2**30 if card is not None else None
    verdict = setup_fit(choice, ram_total / 2**30, vram_gib)
    fit = _STRATA_FIT[verdict.verdict.value]
    words = verdict.words
    available = snapshot.ram_available_bytes
    experts = experts_in_ram_bytes(choice)
    if (
        fit is AdmissionFit.fits
        and verdict.mode == "ram"
        and available is not None
        and available < experts
    ):
        fit = AdmissionFit.tight
        words = (
            f"Strata copies its {choice.arena_gb:g} GB of experts into RAM, and "
            f"{_gib(available)} of RAM is free now: the system would page them"
        )
    if fit is AdmissionFit.unknown:
        return answer(
            AdmissionDecision.admit,
            fit,
            f"admit on faith: {words}.",
            "; ".join(warnings) or "Strata's fit is not estimated here",
        )
    fills = (
        f" It fills the free memory of {where} with its expert cache: "
        f"{_gib(room)} held while it loads."
        if room is not None
        else ""
    )
    if fit in (AdmissionFit.fits, AdmissionFit.split):
        return answer(
            AdmissionDecision.admit,
            fit,
            f"admit: {choice.model} on Strata, by its setup's table: {words}.{fills}",
        )
    fix = (
        "Close what holds RAM, pick a smaller size from Strata's list, or pass "
        "?force=true to start anyway."
    )
    return answer(
        AdmissionDecision.refuse,
        fit,
        f"refuse: {choice.model} on Strata, by its setup's table: {words}. {fix}",
    )


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
    folders_unread: str | None = None,
    engine_places: bool = False,
) -> Admission:
    """Measure `spec` against the device it targets, right now.

    `size_of` is the on-disk sizer, injectable so tests need not create
    a 200 GiB file to describe one — which Windows would zero-fill.
    `exists` is its sibling for the file being there at all. `mappings`
    are this node's `pathMappings`, applied to `modelPath` before
    anything on disk is asked about it (M11); `node_name` is for the
    refusal's prose. `folders_unread` is why this node could not read the
    Library's folders on this request, when it could not: then no folder's
    mount may have applied, and the refusal has to say so.
    `engine_places` is whether this spec's llama-server places a model by
    itself (`wants_full_offload`).
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
            spec,
            location,
            node_name=node_name,
            context_length=context_length,
            warnings=warnings,
            folders_unread=folders_unread,
        )

    # Each engine by its own fit model (LS6, Troy's L11): never another
    # engine's arithmetic in its place.
    declared = engine_fit_model(spec.engine)
    if spec.engine is EngineKind.strata:
        return strata_admission(
            spec,
            snapshot=snapshot,
            location=location,
            reservations=reservations,
            running=running,
            warnings=warnings,
        )
    if declared is None:
        return Admission(
            decision=AdmissionDecision.admit,
            fit=AdmissionFit.unknown,
            basis=AdmissionBasis.file_size,
            contextLength=context_length,
            blockers=[],
            location=location,
            reason=(
                f"admit on faith: {engine_name(spec.engine)} has no memory estimate in Eugene "
                f"yet, so {spec.modelPath} was not measured; never another engine's "
                "arithmetic in its place."
            ),
            warning="; ".join(warnings) or f"{engine_name(spec.engine)}'s fit is not estimated",
        )
    shares = declared.kind is FitModelKind.reserved_share
    if shares:
        # vLLM's context is its own flag, `maxModelLen`; unset, it takes the
        # model's own, which the library then measures.
        from .engines.vllm import model_length

        context_length = model_length(flags)

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

    # The card, or cards, the engine would actually have to fit on,
    # MINUS what this node has already promised to launches that have not
    # taken their memory yet. Free memory is a live reading, so the card a
    # second launch picks has to be the one with room left after the
    # first, not the one that still looks empty. This runtime's own
    # reservation is never counted against it: a restart re-measures a
    # runtime that already holds one, and counting it would refuse every
    # restart. `freeBytes` on the wire stays the cards' own reading,
    # because reporting the reduced number there would be a claim about
    # the card that is not true.
    placement = place(spec, targets, reservations)
    device = placement.main
    free = placement.free
    total = placement.total
    reserved = placement.reserved
    budget = placement.budget
    cards = len(placement.devices)
    # `ram_bytes` means "system RAM a partial offload can spill into,
    # BESIDE the device pool". On a discrete GPU that is real; on Apple
    # unified memory the "VRAM" pool and host RAM are the same silicon,
    # so passing both double-counts the machine — the exact "all host
    # RAM read as free VRAM" defect the MLX slice exists to avoid. A
    # metal device therefore gets no spillover term: its budget IS the
    # unified pool (and today `memoryFreeBytes` is absent there, so the
    # verdict is honestly `unknown` until a real Mac measures it).
    #
    # **Every shared-memory device is that case** (2026-09-27): an
    # integrated GPU computes out of the same RAM, and so does a GB10.
    # Before `sharedMemory` existed only a Mac was known to be one.
    unified = device.kind is ComputeDeviceKind.metal or bool(device.sharedMemory)
    spillover_device = device.kind is not ComputeDeviceKind.cpu and not unified
    ram_available = snapshot.ram_available_bytes if spillover_device else 0
    blockers = _blockers(spec, targets, snapshot, running)
    full_offload = wants_full_offload(spec, engine_places=engine_places)
    engine_spills = not full_offload and (spec.flags or {}).get("gpuLayers") is None

    # Required bytes: the library when it answers, the file otherwise.
    required: int | None = None
    basis = AdmissionBasis.file_size
    fit = AdmissionFit.unknown
    utilization = share_of(spec, declared) if shares else None
    if library is not None:
        answer = await library.fit(
            spec.modelPath,
            context_length=context_length,
            vram_bytes=budget,
            ram_bytes=ram_available if spillover_device else None,
            unified_memory=unified,
            gpu_count=cards,
            fit_model=declared.kind.value if shares else None,
            gpu_memory_utilization=utilization,
            vram_total_bytes=total if shares else None,
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
            # One allowance per card: each holds its own compute buffers.
            required = file_size_requirement(size, context_length) + OVERHEAD_BYTES * (cards - 1)
            fit = local_verdict(required, free=budget, total=total, ram_available=ram_available)
        else:
            warnings.append(f"{spec.modelPath} could not be sized on disk")

    decision = decide(fit, full_offload=full_offload)
    # A share-taking engine takes its whole share when it starts, whatever
    # the model needs inside it, so that is what it holds on the cards and
    # what the ledger promises (LS6).
    share_text = ""
    if shares and utilization is not None and total:
        taken = int(total * utilization)
        share_text = (
            f" {engine_name(spec.engine)} takes {utilization:.0%} of the cards' total memory "
            f"({_gib(taken)}) when it starts and refuses to start with less free."
        )
        if required is not None and basis is AdmissionBasis.metadata:
            required = max(required, taken)
    where = (
        f"{cards} cards ("
        + ", ".join(f"device {d.index} {d.name or d.kind.value}" for d in placement.devices)
        + ")"
        if placement.spread
        else f"device {device.index} ({device.name or device.kind.value})"
    )
    has = "have" if placement.spread else "has"
    between = " between them" if placement.spread else ""
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
        f" It fits up to {max_context_length} context on "
        + ("these cards." if placement.spread else "this device.")
        if max_context_length
        else ""
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
            f"{where} {has} {_gib(free)} free of {_gib(total)}{between};{reserved_text} "
            f"verdict {fit.value}"
            + (
                " with GPU layers left to llama.cpp, which places what does not fit "
                "in system memory itself"
                if engine_spills and fit is not AdmissionFit.fits
                else " with partial offload requested, so the spill is the operator's choice"
                if not full_offload and fit is not AdmissionFit.fits
                else ""
            )
            + f".{share_text} {held}{slot_text}"
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
        if shares:
            # Nothing moves to system memory: the remedies are its context,
            # its share, and what else holds the cards.
            lower = (
                f"set maxModelLen to {max_context_length} or below"
                if max_context_length
                else "lower maxModelLen"
            )
            rest = (
                f"stop one of them, {lower}, change gpuMemoryUtilization, "
                "or pass ?force=true to launch anyway."
                if blockers
                else f"{lower}, change gpuMemoryUtilization, pick a smaller model, "
                "or pass ?force=true to launch anyway."
            )
        else:
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
            f"{where} {has} {_gib(free)} free of {_gib(total)}{between};{reserved_text} "
            f"verdict {fit.value}.{share_text}{fits_up_to} "
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
        devices=list(placement.devices) if placement.spread else None,
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

    `None` at one slot on purpose, and `None` unset because nothing is
    divided there either: llama-server's automatic slots (`-np -1`, four
    of them) share one unified pool, so a request alone can use the whole
    context (`tools/server/server.cpp` at b11375). This said *the default
    is one slot*, which was the profile field's claim and never
    llama-server's (drift audit 2026-10-03); the arithmetic was right for
    the wrong reason. A sentence on every launch that left slots alone
    would appear on almost all of them and train people to skip the
    reason — which is the surface that has to carry a refusal's arithmetic.
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
    folders_unread: str | None = None,
) -> Admission:
    """The one launch failure the dry run can predict with certainty."""
    where = f"on {node_name}" if node_name else "on this host"
    tab = f"Library -> {node_name or 'this machine'} -> Folders"
    if location.mapping is None and folders_unread:
        # Not "set a mount": the operator may well have set one. This node
        # could not read the folders, so none of their mounts applied and
        # the Library's own spelling was tried (2026-09-27, the live
        # install, whose control host was registered at 127.0.0.1).
        fix = (
            f"Nothing exists at {location.localPath}, and no Library folder's mount was "
            f"applied, because {node_name or 'this machine'} could not read the Library's "
            f"folders. {folders_unread}"
        )
    elif location.mapping is None:
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
