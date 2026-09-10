"""Per-device memory on this host, live.

Two consumers, one reader. `GET /v1/node` reports what this host can
compute on — the `Node.devices` inventory the control root aggregates —
and admission (`admission.py`) measures a launch against the free
memory of the device it targets. Both need the number from the host
that will spawn the engine, which is this one, and both need it live:
free memory moves every second, and a verdict from a cached reading is
a verdict about a moment that has passed.

This is a second copy of a small thing the library also does
(`hardware.py` there). Deliberate: components share schemas, not code,
and the agent has to answer with no library present. The two can
disagree by a few MiB across a second; admission uses this one because
this is the host that spawns.

Only the NVIDIA path has met real hardware (the RTX 5090 this was
written against). ROCm, Intel and Apple unified memory are written from
their tools' documented output and are **unverified**; each failure
appends a warning rather than defaulting silently, because a fit verdict
computed from a wrong budget is worse than no verdict — and admission
never refuses on `unknown`.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from .._generated.models import ComputeDevice, ComputeDeviceKind

log = logging.getLogger(__name__)

MIB = 1024 * 1024

# Vendor CLIs are quick when present and hang when the driver stack is
# wedged. Admission runs behind a POST an operator is waiting on.
_PROBE_TIMEOUT_SECONDS = 10.0

# Apple silicon: the GPU addresses host RAM, capped by the wired limit.
# 75% is the default `iogpu.wired_limit_pct`; the library uses the same
# figure, and both say so in a warning.
APPLE_WIRED_LIMIT_FRACTION = 0.75

Runner = Callable[[list[str]], str | None]
MemoryReader = Callable[[], tuple[int | None, int | None]]


@dataclass(frozen=True)
class DeviceSnapshot:
    """What was detected, when, and what could not be."""

    devices: tuple[ComputeDevice, ...]
    warnings: tuple[str, ...]
    ram_total_bytes: int | None
    ram_available_bytes: int | None
    detected_at: datetime

    def accelerators(self) -> list[ComputeDevice]:
        """Everything that is not the CPU."""
        return [d for d in self.devices if d.kind is not ComputeDeviceKind.cpu]

    def cpu(self) -> ComputeDevice | None:
        return next((d for d in self.devices if d.kind is ComputeDeviceKind.cpu), None)


def _run(argv: list[str]) -> str | None:
    """Run a vendor probe; its stdout, or None if it did not work."""
    if shutil.which(argv[0]) is None:
        return None
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("device probe %r failed: %s", argv[0], e)
        return None
    if proc.returncode != 0:
        log.debug("device probe %r exited %d: %s", argv[0], proc.returncode, proc.stderr[:200])
        return None
    return proc.stdout or ""


# --- host memory ----------------------------------------------------------


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_memory() -> tuple[int | None, int | None]:
    if sys.platform != "win32":  # pragma: no cover - unreachable on Windows
        return None, None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError) as exc:
        log.warning("GlobalMemoryStatusEx failed (%s)", exc)
        return None, None
    if not ok:
        return None, None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _linux_memory() -> tuple[int | None, int | None]:
    """`MemAvailable`, not `MemFree` — the kernel's own estimate of what a
    new allocation can have without swapping."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    values[key] = int(parts[0]) * 1024
    except OSError as exc:
        log.warning("/proc/meminfo unreadable (%s)", exc)
        return None, None
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if available is None and total is not None:
        available = values.get("MemFree", 0) + values.get("Cached", 0) + values.get("Buffers", 0)
    return total, available


def _macos_memory(run: Runner) -> tuple[int | None, int | None]:
    total: int | None = None
    out = run(["sysctl", "-n", "hw.memsize"])
    if out and out.strip().isdigit():
        total = int(out.strip())
    available: int | None = None
    stats = run(["vm_stat"])
    if stats:
        page_size = 4096
        header = re.search(r"page size of (\d+) bytes", stats)
        if header:
            page_size = int(header.group(1))
        pages = dict(re.findall(r"^(.*?):\s+(\d+)\.?$", stats, flags=re.MULTILINE))
        counted = [
            int(pages[k])
            for k in ("Pages free", "Pages inactive", "Pages speculative")
            if k in pages
        ]
        if counted:
            available = sum(counted) * page_size
    return total, available


def host_memory(run: Runner = _run) -> tuple[int | None, int | None]:
    """`(total, available)` in bytes, either possibly None."""
    if sys.platform == "win32":
        return _windows_memory()
    if sys.platform == "darwin":
        return _macos_memory(run)
    return _linux_memory()


# --- accelerators -----------------------------------------------------------


def _nvidia(run: Runner, warnings: list[str]) -> list[ComputeDevice]:
    """Verified against a real RTX 5090 on Windows. `nounits` keeps the
    CSV numeric; the memory columns are MiB, nvidia-smi's own unit."""
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if out is None:
        return []
    devices: list[ComputeDevice] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 4:
            warnings.append(f"could not parse an nvidia-smi row: {line.strip()!r}")
            continue
        try:
            index = int(fields[0])
            total = int(float(fields[2])) * MIB
            free = int(float(fields[3])) * MIB
        except ValueError:
            warnings.append(f"could not parse an nvidia-smi row: {line.strip()!r}")
            continue
        devices.append(
            ComputeDevice(
                kind=ComputeDeviceKind.cuda,
                index=index,
                name=fields[1],
                memoryTotalBytes=total,
                memoryFreeBytes=free,
            )
        )
    return devices


def _amd(run: Runner, warnings: list[str]) -> list[ComputeDevice]:
    """UNVERIFIED — no AMD hardware on the dev box. Parses the CSV form
    of `rocm-smi --showmeminfo vram`, whose column names have moved
    between ROCm releases; a parse failure is a warning, not a size."""
    out = run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if out is None:
        return []
    lines = [line for line in out.splitlines() if line.strip()]
    if len(lines) < 2:
        warnings.append("rocm-smi returned no VRAM rows; AMD device memory is unknown")
        return []
    header = [h.strip().lower() for h in lines[0].split(",")]
    total_col = next((i for i, h in enumerate(header) if "total" in h and "vram" in h), None)
    used_col = next((i for i, h in enumerate(header) if "used" in h and "vram" in h), None)
    if total_col is None:
        warnings.append(
            f"rocm-smi output has no recognisable VRAM total column (saw {header}); "
            "AMD device memory is unknown"
        )
        return []
    devices: list[ComputeDevice] = []
    for index, line in enumerate(lines[1:]):
        fields = [f.strip() for f in line.split(",")]
        if len(fields) <= total_col:
            continue
        try:
            total = int(fields[total_col])
            used = int(fields[used_col]) if used_col is not None and len(fields) > used_col else 0
        except ValueError:
            warnings.append(f"could not parse a rocm-smi row: {line.strip()!r}")
            continue
        devices.append(
            ComputeDevice(
                kind=ComputeDeviceKind.rocm,
                index=index,
                name=fields[0] or f"AMD GPU {index}",
                memoryTotalBytes=total,
                memoryFreeBytes=max(total - used, 0),
            )
        )
    if devices:
        warnings.append("ROCm device detection is untested against real hardware")
    return devices


def _intel(run: Runner, warnings: list[str]) -> list[ComputeDevice]:
    """UNVERIFIED — `xpu-smi discovery` names devices but reports no free
    memory in a stable form, so these devices carry no memory numbers
    and admission treats them as `unknown`."""
    out = run(["xpu-smi", "discovery", "--dump", "1,2"])
    if out is None:
        return []
    devices: list[ComputeDevice] = []
    for index, line in enumerate(out.splitlines()[1:]):
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        devices.append(
            ComputeDevice(
                kind=ComputeDeviceKind.xpu,
                index=int(fields[0]),
                name=fields[1] or f"Intel GPU {index}",
            )
        )
    if devices:
        warnings.append(
            "xpu-smi does not report device memory in a stable form; Intel device memory is "
            "unknown and admission cannot refuse on it. Untested against real hardware."
        )
    return devices


def _apple(ram_total: int | None, warnings: list[str]) -> list[ComputeDevice]:
    """UNVERIFIED — from Apple's documented behaviour. No separate VRAM
    pool: the GPU addresses host RAM up to the wired limit. Reporting
    zero here would tell a 96 GB Mac it has no GPU."""
    if ram_total is None:
        warnings.append("could not read total memory, so the unified-memory budget is unknown")
        return []
    warnings.append(
        f"unified memory: reporting {APPLE_WIRED_LIMIT_FRACTION:.0%} of RAM as the GPU budget "
        "(the default iogpu.wired_limit_pct). Untested against real hardware."
    )
    return [
        ComputeDevice(
            kind=ComputeDeviceKind.metal,
            index=0,
            name=platform.processor() or "Apple silicon",
            memoryTotalBytes=int(ram_total * APPLE_WIRED_LIMIT_FRACTION),
            # Free is genuinely unknowable without the wired-page count;
            # left absent so admission reports `unknown` rather than
            # inventing a number.
        )
    ]


def detect_devices(*, run: Runner = _run, memory: MemoryReader | None = None) -> DeviceSnapshot:
    """Every device this host can compute on, with live memory.

    The CPU is always last, carrying host memory, so a box with no
    accelerator still has a budget for a CPU-only launch and a
    `Node.devices` that is not empty.
    """
    warnings: list[str] = []
    ram_total, ram_available = memory() if memory is not None else host_memory(run)

    devices: list[ComputeDevice] = []
    devices += _nvidia(run, warnings)
    devices += _amd(run, warnings)
    devices += _intel(run, warnings)
    if (
        not devices
        and sys.platform == "darwin"
        and platform.machine().lower() in ("arm64", "aarch64")
    ):
        devices += _apple(ram_total, warnings)

    devices.append(
        ComputeDevice(
            kind=ComputeDeviceKind.cpu,
            index=0,
            name=platform.processor() or platform.machine() or "CPU",
            memoryTotalBytes=ram_total,
            memoryFreeBytes=ram_available,
        )
    )
    if ram_total is None:
        warnings.append("host memory could not be read")

    return DeviceSnapshot(
        devices=tuple(devices),
        warnings=tuple(warnings),
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        detected_at=datetime.now(UTC),
    )


__all__ = ["MIB", "DeviceSnapshot", "detect_devices", "host_memory"]
