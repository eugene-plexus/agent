"""Optional hardware facts. Unsupported reads stay null, including partial links."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from ._generated.models import ComputeDevice, ComputeDeviceKind
from .engines import gpu_probe


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def sysfs_health(path: Path) -> dict[str, Any]:
    """AMD Linux device directory, also callable by later fit/placement rows."""

    def integer(name: str) -> int | None:
        value = _text(path / name)
        return int(value) if value and value.isdigit() else None

    def generation(name: str) -> int | None:
        value = _text(path / name)
        match = re.match(r"([\d.]+)\s+GT/s", value or "")
        return (
            {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}.get(float(match[1]))
            if match
            else None
        )

    link = dict(
        generation=generation("current_link_speed"),
        width=integer("current_link_width"),
        maxGeneration=generation("max_link_speed"),
        maxWidth=integer("max_link_width"),
    )
    out: dict[str, Any] = {
        "uuid": _text(path / "unique_id"),
        "pciBusId": path.resolve().name
        if re.fullmatch(r"[\da-fA-F]{4}:[\da-fA-F]{2}:[\da-fA-F]{2}\.\d", path.resolve().name)
        else None,
        "pcieLink": link if any(v is not None for v in link.values()) else None,
    }
    hwmons = sorted((path / "hwmon").glob("hwmon*"))
    for key, filename in (
        ("powerLimitWatts", "power1_cap"),
        ("powerDefaultLimitWatts", "power1_cap_default"),
    ):
        raw = next((_text(h / filename) for h in hwmons if (h / filename).is_file()), None)
        out[key] = int(raw) / 1_000_000 if raw and raw.isdigit() else None
    return out


def enrich_devices(
    devices: list[ComputeDevice], adapters: list[gpu_probe.Adapter]
) -> list[ComputeDevice]:
    from .host_probes import Nvml

    nvml = Nvml() if any(d.kind is ComputeDeviceKind.cuda for d in devices) else None
    try:
        cuda = [d for d in devices if d.kind is ComputeDeviceKind.cuda]
        remaining = list(adapters)
        result = []
        for device in devices:
            facts: dict[str, Any] = {}
            if device.kind is ComputeDeviceKind.cuda and nvml is not None:
                facts = nvml.health(device.index or 0, [d.index or 0 for d in cuda])
            elif sys.platform.startswith("linux"):
                adapter = next((a for a in remaining if a.name == device.name), None)
                if adapter is not None:
                    remaining.remove(adapter)
                    if adapter.vendor == gpu_probe.AMD and adapter.key:
                        facts = sysfs_health(Path(adapter.key))
            # Validate the nested structures rather than leaving raw dictionaries
            # in a model_copy update (which skips Pydantic validation).
            result.append(ComputeDevice.model_validate({**device.model_dump(), **facts}))
        return result
    finally:
        if nvml is not None:
            nvml.close()
