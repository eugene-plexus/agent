"""Callable launch checks for stable card identity and temporary refits (row 43).

The saved declaration is never changed. Observations live in their own file;
the expected set survives a reduced launch so a returned card is used next time.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from ._generated.models import (
    Admission,
    AdmissionDecision,
    AdmissionFit,
    Change,
    ComputeDevice,
    ComputeDeviceKind,
    ConfigField,
    ConfigValueType,
    EngineKind,
    RuntimeMissingDevice,
    RuntimeSpec,
    RuntimeStartedOnFewerCards,
)
from ._private_files import write_private
from .admission import main_device, place, spread_devices, target_devices
from .engines.devices import DeviceSnapshot

log = logging.getLogger(__name__)
Measure = Callable[[RuntimeSpec], Awaitable[Admission]]


def missing_card_field() -> ConfigField:
    return ConfigField(
        key="missingCard",
        label="If a card is missing",
        category="memory",
        requiresRestart=True,
        valueType=ConfigValueType.enum,
        default="start-on-cards-left",
        enumValues=["start-on-cards-left", "do-not-start"],
        enumLabels=["Start on the cards left", "Do not start"],
        description="Start on the cards left with what fits, or wait for the missing card. "
        "Any launch changes are reported. Your saved profile stays the same.",
    )


@dataclass
class CardLaunch:
    spec: RuntimeSpec
    expected: list[ComputeDevice]
    devices: list[ComputeDevice]
    missing: list[RuntimeMissingDevice] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    verified: bool = True

    @property
    def report(self) -> RuntimeStartedOnFewerCards | None:
        return (
            RuntimeStartedOnFewerCards(
                missing=self.missing, changes=[Change(c) for c in self.changes]
            )
            if self.missing
            else None
        )


def select_cards(
    spec: RuntimeSpec, snapshot: DeviceSnapshot, *, expected: list[ComputeDevice] | None = None
) -> CardLaunch:
    """Resolve index pins once, then match UUIDs even after the bus renumbers.

    A device without a readable UUID keeps the index path. An unreadable UUID
    on a present card is not evidence that it disappeared.
    """
    if spec.engine not in (EngineKind.llama_cpp, EngineKind.vllm):
        return CardLaunch(spec, [], [])
    raw_pin = (spec.env or {}).get("CUDA_VISIBLE_DEVICES")
    if raw_pin is not None and (not raw_pin.strip() or raw_pin.strip() == "-1"):
        return CardLaunch(spec, [], [])
    present = snapshot.accelerators()
    if (
        expected
        and any(d.kind is ComputeDeviceKind.cuda for d in expected)
        and snapshot.cuda_inventory_known is False
    ):
        log.warning(
            "The card check could not read NVIDIA's inventory (nvidia-smi did not "
            "answer with readable rows); starting without claiming a card is missing."
        )
        return CardLaunch(spec, expected, [], verified=False)
    targets = target_devices(spec, snapshot)
    targets = [d for d in targets if d.kind is not ComputeDeviceKind.cpu]
    if not expected:
        if raw_pin is not None and any(
            not any(
                (d.uuid and d.uuid.startswith(token.strip())) or str(d.index) == token.strip()
                for d in present
                if d.kind is ComputeDeviceKind.cuda
            )
            for token in raw_pin.split(",")
        ):
            log.warning(
                "CUDA selection %r could not be matched by UUID; leaving this "
                "untested selector to the engine.",
                raw_pin,
            )
            return CardLaunch(spec, [], [], verified=False)
        spread = spread_devices(spec, targets)
        if spread:
            expected = spread
        elif targets:
            expected = [place(spec, targets, ()).main]
        else:
            expected = []
    chosen: list[ComputeDevice] = []
    missing: list[RuntimeMissingDevice] = []
    for old in expected:
        found = next((d for d in present if old.uuid and d.uuid == old.uuid), None)
        if found is None and (not old.uuid or any(d.uuid is None for d in present)):
            found = next((d for d in present if d.kind is old.kind and d.index == old.index), None)
        if found is not None:
            chosen.append(found)
        elif old.uuid:
            missing.append(
                RuntimeMissingDevice(
                    uuid=old.uuid, name=old.name or old.uuid, lastIndex=old.index or 0
                )
            )
    flags = dict(spec.flags or {})
    if missing and flags.get("missingCard") == "do-not-start":
        names = ", ".join(f"{d.name} (was #{d.lastIndex})" for d in missing)
        raise ValueError(f"{names} is missing; this runtime is set not to start without it.")
    if missing and not chosen:
        kinds = {d.kind for d in expected}
        chosen = [d for d in present if d.kind in kinds]
    launch = CardLaunch(spec, list(expected), chosen, missing)
    if not missing and not any(d.uuid for d in chosen):
        return launch
    if not chosen:
        # Unknown inventory stays untested. A measured absence can still run on
        # the CPU for llama.cpp; vLLM will explain its unsupported configuration.
        if not missing:
            return launch
        if spec.engine is EngineKind.vllm:
            raise ValueError("No cards are left for vLLM; all expected cards are missing.")
        flags.update(devices="none", gpuLayers=0)
        launch.changes.append("All model layers run in system memory.")
    env = dict(spec.env or {})
    cuda = [d for d in chosen if d.kind is ComputeDeviceKind.cuda]
    if cuda:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(d.uuid or str(d.index) for d in cuda)
    elif missing and any(d.kind is ComputeDeviceKind.cuda for d in expected):
        env["CUDA_VISIBLE_DEVICES"] = ""
    raw_devices = flags.get("devices")
    if isinstance(raw_devices, str) and cuda:
        tokens = [t.strip() for t in raw_devices.split(",")]
        other = [t for t in tokens if not t.lower().startswith("cuda")]
        flags["devices"] = ",".join([*(f"CUDA{i}" for i in range(len(cuda))), *other])
    if "mainGpu" in flags and cuda:
        old_main = main_device(spec, expected)
        new_main = next(
            (i for i, d in enumerate(cuda) if old_main and d.uuid == old_main.uuid), None
        )
        if new_main is None:
            new_main = max(range(len(cuda)), key=lambda i: cuda[i].memoryTotalBytes or 0)
        flags["mainGpu"] = new_main
        if missing:
            launch.changes.append(f"Main card is {cuda[new_main].name}.")
    if missing and isinstance(flags.get("tensorSplit"), str):
        try:
            weights = [float(v) for v in flags["tensorSplit"].split(",")]
            remaining = [
                weights[i] if i < len(weights) else 0
                for i, old in enumerate(expected)
                if any(d.uuid == old.uuid for d in chosen)
            ]
            total = sum(remaining)
            if total > 0:
                flags["tensorSplit"] = ",".join(f"{v / total:g}" for v in remaining)
            else:
                flags.pop("tensorSplit", None)
        except ValueError:
            flags.pop("tensorSplit", None)
        launch.changes.append("The model split uses the cards left.")
    if missing and spec.engine is EngineKind.vllm:
        from .engines.vllm import refit_parallelism

        flags, changes = refit_parallelism(flags, len(chosen), spec.modelPath)
        launch.changes.extend(changes)
        count = int(str(flags.get("tensorParallelSize", 1))) * int(
            str(flags.get("pipelineParallelSize", 1))
        )
        launch.devices = chosen[:count]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(d.uuid or str(d.index) for d in launch.devices)
    launch.spec = spec.model_copy(update={"flags": flags, "env": env or None})
    return launch


async def refit_on_cards_left(launch: CardLaunch, measure: Measure) -> tuple[CardLaunch, Admission]:
    """Measure the cards left, apply a suggested context or RAM spill, and verify it."""
    answer = await measure(launch.spec)
    if not launch.verified and (answer.location is None or answer.location.exists):
        # A CPU reading is not the budget of an unreadable accelerator. Keep
        # genuine file-location failures, but never refuse this untested launch.
        reason = (
            "The requested cards could not be matched to a readable inventory; "
            "their memory budget is unknown. Start with the saved selection."
        )
        return launch, answer.model_copy(
            update={
                "decision": AdmissionDecision.admit,
                "fit": AdmissionFit.unknown,
                "device": None,
                "devices": None,
                "freeBytes": None,
                "totalBytes": None,
                "reservedBytes": None,
                "requiredBytes": None,
                "reason": reason,
                "warning": reason,
            }
        )
    if not launch.missing or answer.fit in (AdmissionFit.fits, AdmissionFit.unknown):
        return launch, answer
    if answer.fit is AdmissionFit.no:
        raise ValueError(f"Cannot start on the cards left. {answer.reason}")
    flags = dict(launch.spec.flags or {})
    key = "maxModelLen" if launch.spec.engine is EngineKind.vllm else "contextSize"
    maximum = answer.maxContextLength
    if maximum and (not flags.get(key) or maximum < int(flags[key])):
        old = flags.get(key) or answer.contextLength
        flags[key] = maximum
        launch.changes.append(f"Context {old} → {maximum}.")
    elif launch.spec.engine is EngineKind.llama_cpp:
        # llama.cpp's automatic placement is already used by admission for new
        # runtimes. Remove the full-offload demand so it can spill into RAM.
        flags.pop("gpuLayers", None)
        launch.changes.append("Model layers that do not fit on the cards run in system memory.")
    launch.spec = launch.spec.model_copy(update={"flags": flags})
    answer = await measure(launch.spec)
    if answer.fit is AdmissionFit.no or answer.decision.value == "refuse":
        raise ValueError(f"Cannot start on the cards left. {answer.reason}")
    return launch, answer


def _signature(spec: RuntimeSpec) -> str:
    # A context change or a credential rotation is not a new card selection.
    # Keep the binding until the person changes the device selectors themselves.
    env = {
        key: (spec.env or {}).get(key)
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    }
    flags = {
        key: (spec.flags or {}).get(key)
        for key in (
            "devices",
            "mainGpu",
            "tensorSplit",
            "splitMode",
            "tensorParallelSize",
            "pipelineParallelSize",
        )
    }
    return json.dumps([spec.engine.value, env, flags], sort_keys=True)


class DeviceMemory:
    """Observed cards, keyed by declaration, separately from config and profiles."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._records: dict[str, dict] = {}
        if path is not None:
            try:
                self._records = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(self._records, dict):
                    self._records = {}
            except (OSError, ValueError) as exc:
                log.debug("No saved device observations: %s", exc)

    def expected(self, spec: RuntimeSpec) -> list[ComputeDevice]:
        record = self._records.get(spec.name, {})
        if not isinstance(record, dict):
            return []
        if record.get("signature") != _signature(spec):
            return []
        try:
            return [ComputeDevice.model_validate(d) for d in record.get("expected") or []]
        except (TypeError, ValueError):
            return []

    def record(self, spec: RuntimeSpec, launch: CardLaunch) -> None:
        if not launch.verified:
            return
        self._records[spec.name] = {
            "signature": _signature(spec),
            "expected": [d.model_dump(mode="json") for d in launch.expected],
            "last": [d.uuid for d in launch.devices if d.uuid],
            "report": launch.report.model_dump(mode="json") if launch.report else None,
        }
        if self.path is not None:
            try:
                write_private(self.path, json.dumps(self._records))
            except OSError as exc:
                log.warning("Could not save device observations to %s: %s", self.path, exc)

    def observed(self, spec: RuntimeSpec) -> tuple[list[str], RuntimeStartedOnFewerCards | None]:
        record = self._records.get(spec.name, {})
        if not isinstance(record, dict):
            return [], None
        report = record.get("report")
        last = record.get("last")
        last = last if isinstance(last, list) and all(isinstance(d, str) for d in last) else []
        try:
            return last, RuntimeStartedOnFewerCards.model_validate(report) if report else None
        except ValueError:
            return [], None
