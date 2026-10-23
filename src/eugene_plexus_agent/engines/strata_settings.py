"""Strata v0.1.39's ordered cards, layer boundaries and answer projection."""

from __future__ import annotations

import itertools
import math
from pathlib import Path, PureWindowsPath

from .._generated.models import ComputeDevice, ComputeDeviceKind, StrataEffectiveLaunch
from ..gguf_context import drafting_facts
from .devices import DeviceSnapshot

VECTOR = Path(
    "data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
)
PROJECTION_ARGS = {
    "--control-vector-scaled": 1,
    "--control-vector-layer-range": 2,
    "--cvec-mode": 1,
    "--cvec-dir": 1,
}


def gpu_selection(value: object) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("gpus must be a nonempty ordered list of card ordinals")
    # The shared GUI's ordered list editor supplies text rows; the durable
    # preparation contract supplies integers. Both resolve to inventory ordinals.
    value = [int(i) if isinstance(i, str) and i.isascii() and i.isdigit() else i for i in value]
    if any(type(i) is not int or i < 0 for i in value):
        raise ValueError("gpus must contain nonnegative whole card ordinals")
    if len(set(value)) != len(value):
        raise ValueError("gpus contains a duplicate card ordinal")
    return list(value)


def selected_cards(snapshot: DeviceSnapshot, value: object = None) -> list[ComputeDevice]:
    """This recipe installs CUDA 13. No substitute for an explicitly named card."""
    cards = [d for d in snapshot.devices if d.kind is ComputeDeviceKind.cuda and not d.sharedMemory]
    if value is None:
        if not cards:
            cause = "; ".join(snapshot.warnings) or "no discrete CUDA card is present"
            raise ValueError(f"Strata's CUDA 13 engine has no compatible card here: {cause}")
        return [min(cards, key=lambda d: (-round((d.memoryTotalBytes or 0) / 2**30), d.index or 0))]
    indices = gpu_selection(value)
    out = []
    for index in indices:
        found = next((d for d in cards if d.index == index), None)
        if found is None:
            observed = [d for d in snapshot.devices if d.index == index]
            cause = (
                ", ".join(d.kind.value for d in observed) or "not present in this node's inventory"
            )
            raise ValueError(
                f"Strata card {index} is {cause}; this engine requires a discrete CUDA card"
            )
        out.append(found)
    return out


def layer_count(model: Path) -> int:
    facts = drafting_facts(model)
    # Every family supported by this pinned engine is Qwen3.8-Flash-Next (48
    # layers). Prefer the file's own count, including adopted configurations.
    return facts.block_count if facts is not None and facts.block_count else 48


def layer_split(value: object, count: int, layers: int = 48) -> str | None:
    if value is None or value == "auto":
        return "auto" if count > 1 else None
    if count < 2:
        raise ValueError("an explicit layer boundary requires several selected cards")
    items = value if isinstance(value, list) else value.split(",") if isinstance(value, str) else []
    if not items or any(type(i) not in (str, int) or not str(i).strip().isdigit() for i in items):
        raise ValueError("layer boundaries must be rising whole numbers, or auto")
    boundaries = [int(i) for i in items]
    if len(boundaries) != count - 1:
        raise ValueError(f"layer split needs one boundary per later card ({count - 1} boundaries)")
    if (
        boundaries[0] < 2
        or boundaries[-1] >= layers
        or any(b <= a for a, b in itertools.pairwise(boundaries))
    ):
        raise ValueError(
            f"layer boundaries must rise from at least 2 and stay below the model's {layers} layers"
        )
    return ",".join(map(str, boundaries))


def projection_mode(value: object) -> str:
    if value not in ("off", "on"):
        raise ValueError("speedProjection must be off or on")
    return str(value)


def vector_value(value: str, *, root: Path, cwd: Path) -> str:
    """Bundled vectors come from this node's install, never a copied engine path."""
    name, separator, scale = value.rpartition(":")
    try:
        if not separator or not math.isfinite(float(scale)):
            raise ValueError
    except ValueError as exc:
        raise ValueError("--control-vector-scaled needs a vector path and finite scale") from exc
    path = root / VECTOR if PureWindowsPath(name).name == VECTOR.name else Path(name).expanduser()
    if not path.is_absolute():
        path = cwd / path
    if not path.is_file():
        raise ValueError(f"speed projection vector is missing: {path}")
    return f"{path.resolve()}:{scale}"


def apply_projection(cfg: dict[str, object], mode: object, root: Path) -> None:
    if mode is None:
        return
    requested = projection_mode(mode)
    original = cfg.get("args")
    if not isinstance(original, list) or not all(isinstance(a, str) for a in original):
        raise ValueError("args must be a list of strings")
    args = []
    i = 0
    while i < len(original):
        flag = original[i]
        if flag in PROJECTION_ARGS:
            i += 1 + PROJECTION_ARGS[flag]
        else:
            args.append(flag)
            i += 1
    if requested == "on":
        args += [
            "--control-vector-scaled",
            vector_value(f"{VECTOR}:1.0", root=root, cwd=root),
            "--control-vector-layer-range",
            "4",
            "44",
            "--cvec-mode",
            "project",
            "--cvec-dir",
            "per-layer",
        ]
    cfg["args"] = args
    sampling = cfg.get("sampling") or {}
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be an object")
    cfg["sampling"] = {**sampling, "experimental_speed_projection": requested == "on"}


def effective_launch(cfg: dict[str, object]) -> StrataEffectiveLaunch:
    """Only the applied configuration; no profile or preparation defaults."""
    gpu = cfg.get("gpu")
    gpus = gpu_selection([gpu] if type(gpu) is int else gpu)
    args = cfg.get("args")
    if not isinstance(args, list):
        raise ValueError("launch has no native arguments")
    sampling = cfg.get("sampling") or {}
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be an object")
    on = False
    if "--control-vector-scaled" in args:
        value = str(args[args.index("--control-vector-scaled") + 1])
        on = float(value.rsplit(":", 1)[1]) != 0
    return StrataEffectiveLaunch.model_validate(
        {
            "speedProjection": "on"
            if on and sampling.get("experimental_speed_projection") is not False
            else "off",
            "gpus": gpus,
            "layerSplit": layer_split(cfg.get("layer_split"), len(gpus)),
        }
    )
