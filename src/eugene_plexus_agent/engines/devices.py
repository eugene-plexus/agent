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
written against), and Apple silicon a virtual Mac: GitHub's macOS
runners, where the budget is Metal's own working-set figure (A4,
2026-09-30). ROCm and Intel are written from their tools' documented
output and are **unverified**; each failure
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
from ..child_env import child_environment
from . import gpu_probe
from . import host as host_mod

log = logging.getLogger(__name__)

MIB = 1024 * 1024

# Vendor CLIs are quick when present and hang when the driver stack is
# wedged. Admission runs behind a POST an operator is waiting on.
_PROBE_TIMEOUT_SECONDS = 10.0

# Apple silicon: the GPU addresses host RAM, up to the working set Metal
# allows, which `gpu_probe.metal_device()` reads from Metal itself. This
# fraction is only the fallback when Metal cannot be asked. It was 0.75,
# from documentation; Metal reports two thirds on every runner measured
# (A4, 2026-09-30), and a budget that is too high says "fits" about a
# model Metal will not hold. The library carries the same figure.
APPLE_WIRED_LIMIT_FRACTION = 2 / 3

Runner = Callable[[list[str]], str | None]
MemoryReader = Callable[[], tuple[int | None, int | None]]
AdapterLister = Callable[[], list[gpu_probe.Adapter]]


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


#: Reads this node's devices: an engine's fit table needs its RAM and card
#: (LS6). Called at most once per engine listing.
DevicesReader = Callable[[], DeviceSnapshot]


def _run(argv: list[str]) -> str | None:
    """Run a vendor probe; its stdout, or None if it did not work.

    **The resolved path, not the bare name** -- R2.3's finding in
    `host._run`, which this copy never got (found 2026-09-27).
    `CreateProcess` searches System32 before PATH, so on Windows a bare
    `nvidia-smi` ran the System32 copy whatever `which` had found, and
    the device list and the engine picker could be reading two different
    programs.
    """
    exe = shutil.which(argv[0])
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, *argv[1:]],
            capture_output=True,
            env=child_environment(),
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


# What `nvidia-smi` prints in a memory column it will not fill: `[N/A]` in
# CSV, `Not Supported` in its table. GB10 (DGX Spark) has no memory of its
# own to report, which is the case this is for.
_NVIDIA_UNREPORTED = {"[n/a]", "n/a", "[not supported]", "not supported"}


def _nvidia(
    run: Runner,
    warnings: list[str],
    ram_total: int | None = None,
    ram_available: int | None = None,
) -> list[ComputeDevice]:
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
        if fields[0].isdigit() and {fields[2].lower(), fields[3].lower()} <= _NVIDIA_UNREPORTED:
            # **Unified memory, NVIDIA's kind** (GB10 / DGX Spark). Until
            # 2026-09-27 this row was "could not parse" and the machine
            # read "no GPU". UNVERIFIED: written from NVIDIA's documented
            # behaviour, no GB10 has reported here.
            warnings.append(
                f"{fields[1]} reports no memory of its own, so it is read as computing "
                "out of host memory (NVIDIA's unified memory). Untested against real "
                "hardware."
            )
            devices.append(
                ComputeDevice(
                    kind=ComputeDeviceKind.cuda,
                    index=int(fields[0]),
                    name=fields[1],
                    memoryTotalBytes=ram_total,
                    memoryFreeBytes=ram_available,
                    sharedMemory=True,
                )
            )
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


def _apple_chip(run: Runner = _run) -> str | None:
    """`Apple M2 Pro`, where `platform.processor()` says only `arm`."""
    out = run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return (out or "").strip() or None


def _cpu_name() -> str:
    if sys.platform == "darwin":
        chip = _apple_chip()
        if chip:
            return chip
    return platform.processor() or platform.machine() or "CPU"


def _apple(
    ram_total: int | None,
    warnings: list[str],
    metal: Callable[[], gpu_probe.MetalDevice | None] | None = None,
) -> list[ComputeDevice]:
    """No separate VRAM pool: the GPU addresses host RAM up to the working
    set Metal allows. Reporting zero here would tell a 96 GB Mac it has no
    GPU.

    The budget is Metal's own `recommendedMaxWorkingSetSize`, read on this
    host (A4, 2026-09-30: it matched MLX's figure on GitHub's macOS
    runners). Only when Metal cannot be asked is it a fraction of RAM, and
    then a warning says so."""
    device = (metal or gpu_probe.metal_device)()
    if device is not None:
        budget = device.working_set_bytes
        name = device.name or _apple_chip() or "Apple silicon"
    elif ram_total is None:
        warnings.append("could not read total memory, so the unified-memory budget is unknown")
        return []
    else:
        budget = int(ram_total * APPLE_WIRED_LIMIT_FRACTION)
        name = _apple_chip() or "Apple silicon"
        warnings.append(
            f"unified memory: Metal could not be asked for its working-set limit, so "
            f"{APPLE_WIRED_LIMIT_FRACTION:.0%} of RAM is reported as the GPU budget"
        )
    return [
        ComputeDevice(
            kind=ComputeDeviceKind.metal,
            index=0,
            name=name,
            memoryTotalBytes=budget,
            # Free is genuinely unknowable without the wired-page count;
            # left absent so admission reports `unknown` rather than
            # inventing a number.
            sharedMemory=True,
        )
    ]


_FAMILY_KIND = {
    "vulkan": ComputeDeviceKind.vulkan,
    "rocm": ComputeDeviceKind.rocm,
    "sycl": ComputeDeviceKind.xpu,
}


def _os_devices(
    warnings: list[str],
    ram_available: int | None,
    lister: AdapterLister,
) -> list[ComputeDevice]:
    """The GPUs no vendor tool answered for, from the operating system.

    **The half R2.3 did not build** (2026-09-27). R2.3 taught the engine
    picker to see an Intel or AMD card on Windows and give it the Vulkan
    build, and this list went on asking only vendor tools. So an Intel
    Arc mini PC ran the Vulkan build and read "no GPU", and every fit,
    admission and starter pick on it was scored against system RAM.

    `gpu_probe.family` is the decision the engine picker makes too, so
    the cards listed here are the ones the build that was chosen uses.
    """
    os_name, arch = host_mod.platform_names()
    try:
        found = lister()
    except gpu_probe.GpuProbeError as exc:
        warnings.append(f"could not list this machine's GPUs from the operating system: {exc}")
        return []
    if not found:
        return []
    os_kind = host_mod._detect_os()
    has_intel = any(a.vendor == gpu_probe.INTEL for a in found)
    chosen = gpu_probe.family(
        os_name,
        arch,
        found,
        vulkan_loader=gpu_probe.vulkan_loader_present(os_name),
        rocm=host_mod.rocm_installed(os_kind),
        sycl=has_intel and host_mod.sycl_sees_gpu(),
    )
    warnings.extend(chosen.notes)
    kind = _FAMILY_KIND.get(chosen.accelerator)
    if kind is None:
        return []
    devices: list[ComputeDevice] = []
    for index, adapter in enumerate(chosen.adapters):
        total, free = adapter.budget(ram_available)
        if total is None:
            warnings.append(
                f"{adapter.name} does not report its memory on this platform, so every fit "
                "against it is unknown rather than a guess."
            )
        elif free is None:
            warnings.append(
                f"how much of {adapter.name}'s memory is in use could not be read, so its "
                "free memory is unknown and admission cannot refuse on it."
            )
        devices.append(
            ComputeDevice(
                kind=kind,
                index=index,
                name=adapter.name,
                memoryTotalBytes=total,
                memoryFreeBytes=free,
                sharedMemory=True if adapter.integrated else None,
            )
        )
    return devices


def _left_out(found: list[gpu_probe.Adapter], os_name: str, vulkan_loader: bool) -> list[str]:
    """A sentence for each discrete GPU beside NVIDIA that nothing will use.

    On Windows with the Vulkan loader, a discrete AMD or Intel card beside
    an NVIDIA one is not left out: the build is the CUDA one with the
    Vulkan backend added (`_beside_nvidia`). What is left is named, with
    why, because a card the product skips without a word is a card its
    owner concludes is broken. Integrated GPUs beside a card are left out
    silently: that is every desktop with a graphics card.
    """
    if gpu_probe.combinable_with_cuda(os_name, "x64") and not vulkan_loader:
        why = (
            "the Vulkan backend that would add it needs the Vulkan loader "
            "(vulkan-1.dll), which its graphics driver installs"
        )
    elif os_name == "linux":
        why = (
            "on Linux the CUDA and Vulkan builds are compiled separately and their core "
            "libraries differ, so they cannot be combined into one that uses both"
        )
    else:
        why = "no build published for this machine carries both backends"
    return [
        f"{adapter.name} is also here, and the CUDA build this machine uses cannot compute "
        f"on it: {why}."
        for adapter in found
        if not adapter.integrated and adapter.vendor not in (gpu_probe.NVIDIA, gpu_probe.OTHER)
    ]


def _beside_nvidia(
    found: list[gpu_probe.Adapter], ram_available: int | None, warnings: list[str]
) -> list[ComputeDevice]:
    """A discrete AMD or Intel card beside NVIDIA, as the combined build sees it.

    **Added 2026-09-27** for a second vendor's card. `gpu_probe.beside_nvidia`
    is the rule and the engine picker calls it too, so a card is listed
    exactly when the build chosen for this machine carries the Vulkan
    backend that reaches it. Named ones it does not reach are said.
    """
    os_name, arch = host_mod.platform_names()
    loader = gpu_probe.vulkan_loader_present(os_name)
    extra = gpu_probe.beside_nvidia(os_name, arch, found, vulkan_loader=loader)
    warnings.extend(_left_out([a for a in found if a not in extra], os_name, loader))
    devices: list[ComputeDevice] = []
    for index, adapter in enumerate(extra):
        total, free = adapter.budget(ram_available)
        if free is None:
            warnings.append(
                f"how much of {adapter.name}'s memory is in use could not be read, so its "
                "free memory is unknown and admission cannot refuse on it."
            )
        devices.append(
            ComputeDevice(
                kind=ComputeDeviceKind.vulkan,
                index=index,
                name=adapter.name,
                memoryTotalBytes=total,
                memoryFreeBytes=free,
            )
        )
    return devices


def _os_adapters_quietly(lister: AdapterLister) -> list[gpu_probe.Adapter]:
    try:
        return lister()
    except gpu_probe.GpuProbeError as exc:
        log.debug("could not list GPUs from the operating system: %s", exc)
        return []


def detect_devices(
    *,
    run: Runner = _run,
    memory: MemoryReader | None = None,
    os_adapters: AdapterLister | None = None,
) -> DeviceSnapshot:
    """Every device this host can compute on, with live memory.

    The CPU is always last, carrying host memory, so a box with no
    accelerator still has a budget for a CPU-only launch and a
    `Node.devices` that is not empty.

    `os_adapters` replaces the operating system's GPU list, for tests.
    """
    warnings: list[str] = []
    ram_total, ram_available = memory() if memory is not None else host_memory(run)
    lister = os_adapters or gpu_probe.adapters

    devices: list[ComputeDevice] = []
    devices += _nvidia(run, warnings, ram_total, ram_available)
    if devices:
        # **Beside NVIDIA, only what the build reaches.** `rocm-smi` and
        # `xpu-smi` used to be read here too, so a Linux machine with an
        # NVIDIA card and a ROCm one listed both, and a launch would have
        # been scored across a card the CUDA build cannot use.
        if sys.platform != "darwin":
            devices += _beside_nvidia(_os_adapters_quietly(lister), ram_available, warnings)
    else:
        devices += _amd(run, warnings)
        devices += _intel(run, warnings)
    if (
        not devices
        and sys.platform == "darwin"
        and platform.machine().lower() in ("arm64", "aarch64")
    ):
        devices += _apple(ram_total, warnings)
    elif not devices and sys.platform != "darwin":
        devices += _os_devices(warnings, ram_available, lister)

    devices.append(
        ComputeDevice(
            kind=ComputeDeviceKind.cpu,
            index=0,
            name=_cpu_name(),
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
