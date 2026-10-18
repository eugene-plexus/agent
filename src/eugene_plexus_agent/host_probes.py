"""What this host's hardware is doing now, read cheaply enough to do every 10 s.

Host history's sampler (`host_history.py`; specs
`docs/design/host-history.md`) calls `HostProbe.sample` on a worker thread.
Every reading is a `Reading`; a series this host cannot read is still one,
with `value=None` and `unavailable` saying why, so a screen says "not
reported" and the reason instead of drawing a zero.

**Where each figure comes from, vendor-neutral first:**

* **NVIDIA, both platforms: NVML** (`nvml.dll`, `libnvidia-ml.so.1`), which
  every NVIDIA driver installs, loaded with `ctypes`: memory, utilization,
  temperature, power, and on Linux each process's memory. Never a spawned
  `nvidia-smi` per sample.
* **Windows, every vendor: PDH**, one query kept open for the life of the
  probe: `GPU Adapter Memory` (what each card holds), `GPU Engine`
  (utilization, the busiest engine's sum over processes, as Task Manager
  shows it) and `GPU Process Memory` (each process's share, which is how
  Eugene's share is measured on Windows, NVIDIA included).
* **Linux amdgpu: sysfs** (`mem_info_*`, `gpu_busy_percent`, `hwmon`) and
  each engine process's DRM `fdinfo`. UNVERIFIED on real hardware, like
  `gpu_probe.linux_adapters`: written from the kernel's documented ABI.
* **CPU, RAM, swap and disks: psutil**, already a dependency.

Measured on the development box (RTX 5090 beside an AMD integrated GPU,
2026-10-18): one full sample took 1.4 ms of wall time and 1.6 ms of CPU.

**Eugene's share is per process, not from the admission ledger**: the
ledger holds only reservations for launches that have not taken their
memory yet, and a ready runtime is not in it.

Which card is which: the device list is `GET /v1/node`'s (so a series'
`deviceIndex` is the index there), refreshed every ten minutes; each card
is matched to the operating system's adapter by name, in order where two
share one. Two identical cards on one Windows machine could therefore have
their Eugene's-share figures swapped; nothing else depends on the match.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import ctypes
import dataclasses
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from ._generated.models import ComputeDevice, ComputeDeviceKind, HostMetric, HostUnit
from .engines import gpu_probe
from .engines.devices import DeviceSnapshot

log = logging.getLogger(__name__)

#: How often the device list (which spawns `nvidia-smi` once) is re-read.
INVENTORY_SECONDS = 600.0
#: How long one disk's free-space read may take before it is reported as
#: not answering. A dead network share blocks for as long as the OS allows.
DISK_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class Reading:
    """One series' value in one sample."""

    key: str
    metric: HostMetric
    unit: HostUnit
    label: str
    value: float | None
    unavailable: str | None = None
    device_kind: ComputeDeviceKind | None = None
    device_index: int | None = None
    device_name: str | None = None
    volume: str | None = None
    holds: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeCounts:
    """This agent's runtimes as the sampler needs them."""

    running: int
    loading: int
    #: The engine processes this agent launched; their descendants count too.
    pids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Folder:
    """A folder whose volume's free space is recorded, and what it holds."""

    role: str
    path: str


# --------------------------------------------------------------------------- #
# NVML
# --------------------------------------------------------------------------- #


class _NvmlMemory(ctypes.Structure):
    _fields_ = (
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    )


class _NvmlMemoryV2(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    )


class _NvmlUtilization(ctypes.Structure):
    _fields_ = (("gpu", ctypes.c_uint), ("memory", ctypes.c_uint))


class _NvmlProcess(ctypes.Structure):
    _fields_ = (
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    )


_NVML_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF
_NVML_TEMPERATURE_GPU = 0


@dataclass
class NvmlReading:
    memory_used: int | None = None
    memory_total: int | None = None
    utilization: float | None = None
    temperature: float | None = None
    power: float | None = None
    errors: dict[str, str] = field(default_factory=dict)


class Nvml:
    """NVIDIA's management library, loaded once, or the reason it was not."""

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        self.error: str | None = None
        self._handles: dict[int, ctypes.c_void_p] = {}
        self._load()

    def _candidates(self) -> list[str]:
        if sys.platform == "win32":
            root = os.environ.get("SYSTEMROOT", r"C:\Windows")
            return [
                os.path.join(root, "System32", "nvml.dll"),
                os.path.join(
                    os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                    "NVIDIA Corporation",
                    "NVSMI",
                    "nvml.dll",
                ),
            ]
        if sys.platform == "darwin":
            return []
        return ["libnvidia-ml.so.1", "libnvidia-ml.so"]

    def _load(self) -> None:
        candidates = self._candidates()
        if not candidates:
            self.error = "NVIDIA's management library does not exist on macOS"
            return
        last: str | None = None
        for candidate in candidates:
            try:
                lib = ctypes.CDLL(candidate)
            except OSError as exc:
                last = str(exc)
                continue
            try:
                init = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
                status = init()
            except AttributeError as exc:
                last = str(exc)
                continue
            if status != 0:
                self.error = f"NVML did not start: {self._describe(lib, status)}"
                return
            self._lib = lib
            self.error = None
            return
        self.error = f"NVIDIA's management library (NVML) could not be loaded: {last}"

    @staticmethod
    def _describe(lib: ctypes.CDLL, status: int) -> str:
        try:
            lib.nvmlErrorString.restype = ctypes.c_char_p
            text = lib.nvmlErrorString(status)
            return f"{text.decode(errors='replace')} ({status})" if text else f"error {status}"
        except (AttributeError, OSError):  # pragma: no cover - every NVML has it
            return f"error {status}"

    @property
    def loaded(self) -> bool:
        return self._lib is not None

    def _handle(self, index: int) -> ctypes.c_void_p | str:
        lib = self._lib
        assert lib is not None
        if index in self._handles:
            return self._handles[index]
        handle = ctypes.c_void_p()
        getter = getattr(lib, "nvmlDeviceGetHandleByIndex_v2", None) or (
            lib.nvmlDeviceGetHandleByIndex
        )
        status = getter(ctypes.c_uint(index), ctypes.byref(handle))
        if status != 0:
            return f"NVML has no device {index}: {self._describe(lib, status)}"
        self._handles[index] = handle
        return handle

    def read(self, index: int) -> NvmlReading:
        out = NvmlReading()
        lib = self._lib
        if lib is None:
            reason = self.error or "NVML is not loaded"
            out.errors = dict.fromkeys(("memory", "utilization", "temperature", "power"), reason)
            return out
        handle = self._handle(index)
        if isinstance(handle, str):
            out.errors = dict.fromkeys(("memory", "utilization", "temperature", "power"), handle)
            return out
        # v2 leaves out what the driver reserves, so total and used agree
        # with `nvidia-smi` and with `GET /v1/node`; v1 counts it as used
        # (0.4 GB on the 5090).
        v2 = getattr(lib, "nvmlDeviceGetMemoryInfo_v2", None)
        memory_v2 = _NvmlMemoryV2(version=ctypes.sizeof(_NvmlMemoryV2) | (2 << 24))
        status = v2(handle, ctypes.byref(memory_v2)) if v2 is not None else -1
        if status == 0:
            out.memory_used, out.memory_total = int(memory_v2.used), int(memory_v2.total)
        else:
            memory = _NvmlMemory()
            status = lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
            if status == 0:
                out.memory_used, out.memory_total = int(memory.used), int(memory.total)
            else:
                out.errors["memory"] = f"NVML: {self._describe(lib, status)}"
        rates = _NvmlUtilization()
        status = lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(rates))
        if status == 0:
            out.utilization = float(rates.gpu)
        else:
            out.errors["utilization"] = f"NVML: {self._describe(lib, status)}"
        temperature = ctypes.c_uint()
        status = lib.nvmlDeviceGetTemperature(
            handle, _NVML_TEMPERATURE_GPU, ctypes.byref(temperature)
        )
        if status == 0:
            out.temperature = float(temperature.value)
        else:
            out.errors["temperature"] = f"NVML: {self._describe(lib, status)}"
        milliwatts = ctypes.c_uint()
        status = lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(milliwatts))
        if status == 0:
            out.power = milliwatts.value / 1000.0
        else:
            out.errors["power"] = f"NVML: {self._describe(lib, status)}"
        return out

    def process_memory(self, index: int) -> dict[int, int] | str:
        """`{pid: bytes}` for every process on the card, or why not."""
        lib = self._lib
        if lib is None:
            return self.error or "NVML is not loaded"
        handle = self._handle(index)
        if isinstance(handle, str):
            return handle
        found: dict[int, int] = {}
        for name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
        ):
            getter = getattr(lib, name, None)
            if getter is None:
                return f"this driver's NVML has no {name}"
            buffer = (_NvmlProcess * 128)()
            count = ctypes.c_uint(128)
            status = getter(handle, ctypes.byref(count), buffer)
            if status != 0:
                return f"NVML: {self._describe(lib, status)}"
            for entry in buffer[: count.value]:
                if entry.usedGpuMemory == _NVML_NOT_AVAILABLE:
                    return "this driver does not report each process's memory"
                pid = int(entry.pid)
                found[pid] = max(found.get(pid, 0), int(entry.usedGpuMemory))
        return found

    def close(self) -> None:
        if self._lib is not None:
            with contextlib.suppress(AttributeError, OSError):
                self._lib.nvmlShutdown()
            self._lib = None


# --------------------------------------------------------------------------- #
# Windows: PDH, one query for the life of the probe
# --------------------------------------------------------------------------- #

_PDH_FMT_DOUBLE = 0x00000200
_PDH_FMT_LARGE = 0x00000400


class _PdhDouble(ctypes.Structure):
    _fields_ = (("CStatus", ctypes.c_ulong), ("value", ctypes.c_double))


class _PdhLarge(ctypes.Structure):
    _fields_ = (("CStatus", ctypes.c_ulong), ("value", ctypes.c_longlong))


class _PdhDoubleItem(ctypes.Structure):
    _fields_ = (("szName", ctypes.c_wchar_p), ("FmtValue", _PdhDouble))


class _PdhLargeItem(ctypes.Structure):
    _fields_ = (("szName", ctypes.c_wchar_p), ("FmtValue", _PdhLarge))


ENGINE = "\\GPU Engine(*)\\Utilization Percentage"
PROCESS_DEDICATED = "\\GPU Process Memory(*)\\Dedicated Usage"
PROCESS_SHARED = "\\GPU Process Memory(*)\\Shared Usage"
ADAPTER_DEDICATED = "\\GPU Adapter Memory(*)\\Dedicated Usage"
ADAPTER_SHARED = "\\GPU Adapter Memory(*)\\Shared Usage"

_ENGINE_NAME = re.compile(r"pid_(\d+)_(luid_0x[0-9a-f]+_0x[0-9a-f]+)_phys_(\d+)_eng_(\d+)_")
_PROCESS_NAME = re.compile(r"pid_(\d+)_(luid_0x[0-9a-f]+_0x[0-9a-f]+)_phys_")
_ADAPTER_NAME = re.compile(r"(luid_0x[0-9a-f]+_0x[0-9a-f]+)_phys_")


@dataclass
class WindowsGpuCounters:
    """One collection of the PDH GPU counters, keyed by adapter LUID."""

    utilization: dict[str, float] = field(default_factory=dict)
    adapter_dedicated: dict[str, int] = field(default_factory=dict)
    adapter_shared: dict[str, int] = field(default_factory=dict)
    #: `{luid: {pid: (dedicated, shared)}}`
    processes: dict[str, dict[int, tuple[int, int]]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def parse_engine_utilization(values: dict[str, float]) -> dict[str, float]:
    """Per adapter: each engine's sum over processes, then the busiest engine.

    The figure Task Manager shows for a GPU. Clamped to 100, because the
    per-process figures are sampled separately and can sum slightly past it.
    """
    engines: dict[tuple[str, str, str], float] = {}
    for name, value in values.items():
        found = _ENGINE_NAME.match(name.lower())
        if found is None:
            continue
        key = (found.group(2), found.group(3), found.group(4))
        engines[key] = engines.get(key, 0.0) + value
    out: dict[str, float] = {}
    for (luid, _phys, _engine), total in engines.items():
        out[luid] = min(100.0, max(out.get(luid, 0.0), total))
    return out


def parse_process_memory(
    dedicated: dict[str, int], shared: dict[str, int]
) -> dict[str, dict[int, tuple[int, int]]]:
    out: dict[str, dict[int, tuple[int, int]]] = {}
    for source, slot in ((dedicated, 0), (shared, 1)):
        for name, value in source.items():
            found = _PROCESS_NAME.match(name.lower())
            if found is None:
                continue
            pid, luid = int(found.group(1)), found.group(2)
            pair = list(out.setdefault(luid, {}).get(pid, (0, 0)))
            pair[slot] += int(value)
            out[luid][pid] = (pair[0], pair[1])
    return out


def parse_adapter_memory(values: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, value in values.items():
        found = _ADAPTER_NAME.match(name.lower())
        if found is not None:
            out[found.group(1)] = out.get(found.group(1), 0) + int(value)
    return out


class WindowsPdh:
    """The GPU counters, one open query. Rate counters (engine utilization)
    read the change since the previous collection, which is why the query
    outlives a sample."""

    def __init__(self) -> None:
        self._pdh: ctypes.WinDLL | None = None  # type: ignore[name-defined,unused-ignore]
        self._query = ctypes.c_void_p()
        self._handles: dict[str, ctypes.c_void_p] = {}
        self.missing: dict[str, str] = {}
        self.error: str | None = None
        if sys.platform != "win32":
            self.error = "PDH exists only on Windows"
            return
        try:
            pdh = ctypes.WinDLL("pdh.dll")  # type: ignore[attr-defined,unused-ignore]
        except OSError as exc:  # pragma: no cover - pdh.dll ships with Windows
            self.error = f"pdh.dll could not be loaded: {exc}"
            return
        status = pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query))
        if status != 0:
            self.error = f"PdhOpenQueryW failed with 0x{status & 0xFFFFFFFF:08X}"
            return
        self._pdh = pdh
        for path in (ENGINE, PROCESS_DEDICATED, PROCESS_SHARED, ADAPTER_DEDICATED, ADAPTER_SHARED):
            handle = ctypes.c_void_p()
            status = pdh.PdhAddEnglishCounterW(self._query, path, None, ctypes.byref(handle))
            if status != 0:
                self.missing[path] = (
                    f"Windows has no {path.split(chr(92))[1]} counter here "
                    f"(0x{status & 0xFFFFFFFF:08X})"
                )
                continue
            self._handles[path] = handle
        # The first collection only primes the rate counters.
        pdh.PdhCollectQueryData(self._query)

    def _array(self, path: str, fmt: int, item: type[ctypes.Structure]) -> dict[str, float]:
        pdh = self._pdh
        handle = self._handles.get(path)
        if pdh is None or handle is None:
            return {}
        size = ctypes.c_ulong(0)
        count = ctypes.c_ulong(0)
        pdh.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(size), ctypes.byref(count), None)
        if size.value == 0:
            return {}
        buffer = ctypes.create_string_buffer(size.value)
        if pdh.PdhGetFormattedCounterArrayW(
            handle, fmt, ctypes.byref(size), ctypes.byref(count), buffer
        ):
            return {}
        items = ctypes.cast(buffer, ctypes.POINTER(item))
        out: dict[str, float] = {}
        for index in range(count.value):
            entry = items[index]
            if entry.FmtValue.CStatus > 1 or not entry.szName:
                continue
            out[entry.szName] = entry.FmtValue.value
        return out

    def collect(self) -> WindowsGpuCounters:
        out = WindowsGpuCounters()
        pdh = self._pdh
        if pdh is None:
            reason = self.error or "PDH is not open"
            out.errors = dict.fromkeys(("engine", "process", "adapter"), reason)
            return out
        status = pdh.PdhCollectQueryData(self._query)
        if status != 0:
            reason = f"PdhCollectQueryData failed with 0x{status & 0xFFFFFFFF:08X}"
            out.errors = dict.fromkeys(("engine", "process", "adapter"), reason)
            return out
        if ENGINE in self.missing:
            out.errors["engine"] = self.missing[ENGINE]
        else:
            out.utilization = parse_engine_utilization(
                self._array(ENGINE, _PDH_FMT_DOUBLE, _PdhDoubleItem)
            )
        if PROCESS_DEDICATED in self.missing:
            out.errors["process"] = self.missing[PROCESS_DEDICATED]
        else:
            out.processes = parse_process_memory(
                {
                    k: int(v)
                    for k, v in self._array(
                        PROCESS_DEDICATED, _PDH_FMT_LARGE, _PdhLargeItem
                    ).items()
                },
                {
                    k: int(v)
                    for k, v in self._array(PROCESS_SHARED, _PDH_FMT_LARGE, _PdhLargeItem).items()
                },
            )
        if ADAPTER_DEDICATED in self.missing:
            out.errors["adapter"] = self.missing[ADAPTER_DEDICATED]
        else:
            out.adapter_dedicated = parse_adapter_memory(
                {
                    k: int(v)
                    for k, v in self._array(
                        ADAPTER_DEDICATED, _PDH_FMT_LARGE, _PdhLargeItem
                    ).items()
                }
            )
            out.adapter_shared = parse_adapter_memory(
                {
                    k: int(v)
                    for k, v in self._array(ADAPTER_SHARED, _PDH_FMT_LARGE, _PdhLargeItem).items()
                }
            )
        return out

    def close(self) -> None:
        if self._pdh is not None:
            self._pdh.PdhCloseQuery(self._query)
            self._pdh = None


# --------------------------------------------------------------------------- #
# Linux amdgpu: sysfs and fdinfo
# --------------------------------------------------------------------------- #


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return int(text) if text.lstrip("-").isdigit() else None


def amdgpu_live(device: Path) -> tuple[float | None, float | None, float | None]:
    """`(utilization %, temperature °C, power W)` from sysfs, each or None."""
    busy = _read_int(device / "gpu_busy_percent")
    temperature: float | None = None
    power: float | None = None
    for hwmon in sorted((device / "hwmon").glob("hwmon*")):
        milli = _read_int(hwmon / "temp1_input")
        if milli is not None and temperature is None:
            temperature = milli / 1000.0
        micro = _read_int(hwmon / "power1_average")
        if micro is None:
            micro = _read_int(hwmon / "power1_input")
        if micro is not None and power is None:
            power = micro / 1_000_000.0
    return (float(busy) if busy is not None else None), temperature, power


_KIB = 1024
_FDINFO_UNITS = {"": 1, "B": 1, "KiB": _KIB, "MiB": _KIB**2, "GiB": _KIB**3}


def parse_fdinfo(text: str) -> tuple[str | None, str | None, str | None, int, int]:
    """`(driver, pci slot, client id, vram bytes, gtt bytes)` of one DRM fd."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()

    def amount(*names: str) -> int:
        for name in names:
            raw = fields.get(name)
            if raw is None:
                continue
            number, _, unit = raw.partition(" ")
            if number.isdigit():
                return int(number) * _FDINFO_UNITS.get(unit.strip(), 1)
        return 0

    return (
        fields.get("drm-driver"),
        fields.get("drm-pdev"),
        fields.get("drm-client-id"),
        amount("drm-resident-vram", "drm-memory-vram"),
        amount("drm-resident-gtt", "drm-memory-gtt"),
    )


def amdgpu_process_memory(
    pids: Iterable[int], slot: str, *, shared: bool, proc: Path = Path("/proc")
) -> int:
    """What these processes hold on the amdgpu at `slot`, from DRM fdinfo.

    One DRM client can be open on several fds; counted once per client.
    """
    clients: dict[str, int] = {}
    for pid in pids:
        directory = proc / str(pid) / "fdinfo"
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                text = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "drm-driver" not in text:
                continue
            driver, pdev, client, vram, gtt = parse_fdinfo(text)
            if driver != "amdgpu" or pdev != slot:
                continue
            clients[client or f"{pid}:{entry.name}"] = vram + (gtt if shared else 0)
    return sum(clients.values())


# --------------------------------------------------------------------------- #
# Disks
# --------------------------------------------------------------------------- #


def volume_of(path: str) -> str:
    """The drive or mount point `path` is on: `D:\\`, `\\\\server\\share`, `/mnt/models`."""
    absolute = os.path.abspath(os.path.expanduser(path))
    if sys.platform == "win32":
        drive = os.path.splitdrive(absolute)[0]
        return drive + "\\" if drive.endswith(":") else drive or absolute
    current = Path(absolute)
    while not current.exists() and current != current.parent:
        current = current.parent
    with contextlib.suppress(OSError):
        current = current.resolve()
    while not os.path.ismount(current) and current != current.parent:
        current = current.parent
    return str(current)


def nearest_existing(path: str) -> str:
    current = Path(os.path.abspath(os.path.expanduser(path)))
    while not current.exists() and current != current.parent:
        current = current.parent
    return str(current)


# --------------------------------------------------------------------------- #
# The probe
# --------------------------------------------------------------------------- #


@dataclass
class _Gpu:
    device: ComputeDevice
    adapter: gpu_probe.Adapter | None


def match_adapters(
    devices: Sequence[ComputeDevice], adapters: Sequence[gpu_probe.Adapter]
) -> list[_Gpu]:
    """Each accelerator with the operating system's adapter of the same
    name, in order where two share a name."""
    remaining = list(adapters)
    out: list[_Gpu] = []
    for device in devices:
        if device.kind is ComputeDeviceKind.cpu:
            continue
        wanted = (device.name or "").strip().lower()
        found = next((a for a in remaining if a.name.strip().lower() == wanted), None)
        if found is not None:
            remaining.remove(found)
        out.append(_Gpu(device=device, adapter=found))
    return out


_MACOS_GPU = "macOS gives GPU utilization, temperature and power only to privileged tools"
_WINDOWS_NO_SENSORS = (
    "Windows publishes no temperature or power for this card; only its vendor's own tools read them"
)
_NO_MATCH = "this card could not be matched to the operating system's GPU counters"


class HostProbe:
    """Reads one sample. Not thread-safe: the sampler calls it from one
    worker thread at a time."""

    def __init__(
        self,
        *,
        devices: Callable[[], DeviceSnapshot],
        adapters: Callable[[], list[gpu_probe.Adapter]] | None = None,
        runtimes: Callable[[], RuntimeCounts],
        folders: Callable[[], list[Folder]],
        nvml: Callable[[], Nvml] = Nvml,
        windows: Callable[[], WindowsPdh] = WindowsPdh,
        platform: str = sys.platform,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._devices = devices
        self._adapters = adapters or _quiet_adapters
        self._runtimes = runtimes
        self._folders = folders
        self._nvml_factory = nvml
        self._windows_factory = windows
        self._platform = platform
        self._clock = clock
        self._nvml: Nvml | None = None
        self._pdh: WindowsPdh | None = None
        self._gpus: list[_Gpu] = []
        self._inventory_at: float | None = None
        self._inventory_error: str | None = None
        self._disk_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="host-disk"
        )
        self._disk_pending: dict[str, concurrent.futures.Future[tuple[int, int]]] = {}
        psutil.cpu_percent(None)  # primes the delta the first sample reads

    # -- inventory ---------------------------------------------------------

    def _refresh_inventory(self) -> None:
        now = self._clock()
        if self._inventory_at is not None and now - self._inventory_at < INVENTORY_SECONDS:
            return
        self._inventory_at = now
        try:
            snapshot = self._devices()
        except Exception as exc:  # a device read is never a reason to stop sampling
            self._inventory_error = f"the device list could not be read: {exc}"
            log.warning("host history: %s", self._inventory_error)
            return
        self._inventory_error = None
        try:
            adapters = self._adapters()
        except Exception as exc:
            log.debug("host history: adapters unreadable: %s", exc)
            adapters = []
        self._gpus = match_adapters(snapshot.devices, adapters)
        if any(g.device.kind is ComputeDeviceKind.cuda for g in self._gpus) and self._nvml is None:
            self._nvml = self._nvml_factory()
        if self._platform == "win32" and self._gpus and self._pdh is None:
            self._pdh = self._windows_factory()

    # -- sample ------------------------------------------------------------

    def sample(self) -> list[Reading]:
        self._refresh_inventory()
        counts = self._runtimes()
        pids = _process_trees(counts.pids)
        readings: list[Reading] = []
        memory = psutil.virtual_memory()
        ram_available = int(memory.available)
        counters = self._pdh.collect() if self._pdh is not None else None
        for gpu in self._gpus:
            readings += self._gpu(gpu, pids, ram_available, counters)
        readings += self._host(memory)
        readings += self._disks()
        readings += [
            Reading("runtimesRunning", HostMetric.runtimesRunning, HostUnit.count,
                    "Runtimes running", float(counts.running)),
            Reading("runtimesLoading", HostMetric.runtimesLoading, HostUnit.count,
                    "Runtimes loading", float(counts.loading)),
        ]  # fmt: skip
        return readings

    def _gpu(
        self,
        gpu: _Gpu,
        pids: set[int],
        ram_available: int,
        counters: WindowsGpuCounters | None,
    ) -> list[Reading]:
        device = gpu.device
        name = device.name or f"GPU {device.index}"
        prefix = f"gpu:{device.kind.value}:{device.index}"
        values: dict[HostMetric, float | None] = {}
        reasons: dict[HostMetric, str] = {}

        def put(metric: HostMetric, value: float | None, reason: str | None) -> None:
            values[metric] = value
            if value is None:
                reasons[metric] = reason or "not read on this host"

        total = device.memoryTotalBytes
        used: float | None = None
        used_reason: str | None = None
        adapter = gpu.adapter
        luid = adapter.key if adapter is not None and self._platform == "win32" else None
        shared = bool(device.sharedMemory)

        if device.kind is ComputeDeviceKind.cuda and self._nvml is not None:
            read = self._nvml.read(device.index or 0)
            if read.memory_total is not None and not shared:
                total = read.memory_total
                used = float(read.memory_used or 0)
            elif shared:
                used = _shared_used(total, ram_available)
            else:
                used_reason = read.errors.get("memory")
            put(HostMetric.gpuUtilization, read.utilization, read.errors.get("utilization"))
            put(HostMetric.gpuTemperature, read.temperature, read.errors.get("temperature"))
            put(HostMetric.gpuPower, read.power, read.errors.get("power"))
        elif self._platform == "win32":
            if counters is None or luid is None:
                used_reason = _NO_MATCH if luid is None else "PDH is not open"
            elif adapter is not None and "adapter" not in counters.errors:
                live = dataclasses.replace(
                    adapter,
                    dedicated_used_bytes=counters.adapter_dedicated.get(luid),
                    shared_used_bytes=counters.adapter_shared.get(luid),
                )
                budget_total, free = live.budget(ram_available)
                total = budget_total if budget_total is not None else total
                if total is not None and free is not None:
                    used = float(max(0, total - free))
                else:
                    used_reason = "Windows did not report this card's memory use"
            else:
                used_reason = counters.errors.get("adapter")
            if counters is not None and luid is not None and "engine" not in counters.errors:
                put(HostMetric.gpuUtilization, counters.utilization.get(luid, 0.0), None)
            else:
                put(HostMetric.gpuUtilization, None,
                    (counters.errors.get("engine") if counters else None) or _NO_MATCH)  # fmt: skip
            put(HostMetric.gpuTemperature, None, _WINDOWS_NO_SENSORS)
            put(HostMetric.gpuPower, None, _WINDOWS_NO_SENSORS)
        elif self._platform == "darwin":
            used = _shared_used(total, ram_available)
            for metric in (
                HostMetric.gpuUtilization,
                HostMetric.gpuTemperature,
                HostMetric.gpuPower,
            ):
                put(metric, None, _MACOS_GPU)
        else:
            path = Path(adapter.key) if adapter is not None and adapter.key else None
            if path is not None and adapter is not None and adapter.vendor == gpu_probe.AMD:
                fresh = next((a for a in gpu_probe.linux_adapters() if a.key == adapter.key), None)
                budget_total, free = (fresh or adapter).budget(ram_available)
                total = budget_total if budget_total is not None else total
                if total is not None and free is not None:
                    used = float(max(0, total - free))
                else:
                    used_reason = "amdgpu did not report this card's memory use"
                busy, temperature, power = amdgpu_live(path)
                put(HostMetric.gpuUtilization, busy, "amdgpu publishes no gpu_busy_percent here")
                put(HostMetric.gpuTemperature, temperature, "amdgpu publishes no temperature here")
                put(HostMetric.gpuPower, power, "amdgpu publishes no power reading here")
            else:
                reason = (
                    "the Linux driver for this card publishes no figures this agent reads"
                    if path is not None
                    else _NO_MATCH
                )
                used_reason = reason
                for metric in (
                    HostMetric.gpuUtilization,
                    HostMetric.gpuTemperature,
                    HostMetric.gpuPower,
                ):
                    put(metric, None, reason)

        put(HostMetric.gpuMemoryUsed, used, used_reason)
        put(HostMetric.gpuMemoryTotal, float(total) if total is not None else None,
            "this card does not report its memory size")  # fmt: skip
        eugene, eugene_reason = self._eugene_share(gpu, pids, counters, luid, shared)
        put(HostMetric.gpuMemoryEugene, eugene, eugene_reason)

        labels = {
            HostMetric.gpuMemoryUsed: (HostUnit.bytes, f"{name} memory used"),
            HostMetric.gpuMemoryTotal: (HostUnit.bytes, f"{name} memory"),
            HostMetric.gpuMemoryEugene: (HostUnit.bytes, f"{name} memory held by Eugene"),
            HostMetric.gpuUtilization: (HostUnit.percent, f"{name} utilization"),
            HostMetric.gpuTemperature: (HostUnit.celsius, f"{name} temperature"),
            HostMetric.gpuPower: (HostUnit.watts, f"{name} power"),
        }
        return [
            Reading(
                key=f"{prefix}:{metric.value}",
                metric=metric,
                unit=unit,
                label=label,
                value=values.get(metric),
                unavailable=reasons.get(metric),
                device_kind=device.kind,
                device_index=device.index,
                device_name=device.name,
            )
            for metric, (unit, label) in labels.items()
        ]

    def _eugene_share(
        self,
        gpu: _Gpu,
        pids: set[int],
        counters: WindowsGpuCounters | None,
        luid: str | None,
        shared: bool,
    ) -> tuple[float | None, str | None]:
        if self._platform == "win32":
            if luid is None:
                return None, _NO_MATCH
            if counters is None or "process" in counters.errors:
                return None, (counters.errors.get("process") if counters else "PDH is not open")
            per = counters.processes.get(luid, {})
            held = sum(d + (s if shared else 0) for pid, (d, s) in per.items() if pid in pids)
            return float(held), None
        if self._platform == "darwin":
            return None, "macOS does not say how much GPU memory each process holds"
        device = gpu.device
        if device.kind is ComputeDeviceKind.cuda and self._nvml is not None:
            found = self._nvml.process_memory(device.index or 0)
            if isinstance(found, str):
                return None, found
            return float(sum(v for pid, v in found.items() if pid in pids)), None
        adapter = gpu.adapter
        if adapter is not None and adapter.vendor == gpu_probe.AMD and adapter.key:
            slot = Path(adapter.key).resolve().name
            return float(amdgpu_process_memory(pids, slot, shared=shared)), None
        return None, "this card's driver does not say how much memory each process holds"

    def _host(self, memory: Any) -> list[Reading]:
        total = float(memory.total)
        used = total - float(memory.available)
        try:
            swap = psutil.swap_memory()
            swap_used: float | None = float(swap.used)
            swap_total: float | None = float(swap.total)
            swap_reason = None
        except (OSError, RuntimeError) as exc:
            swap_used = swap_total = None
            swap_reason = f"swap could not be read: {exc}"
        return [
            Reading("cpuUtilization", HostMetric.cpuUtilization, HostUnit.percent, "CPU",
                    float(psutil.cpu_percent(None))),
            Reading("ramUsed", HostMetric.ramUsed, HostUnit.bytes, "RAM used", used),
            Reading("ramTotal", HostMetric.ramTotal, HostUnit.bytes, "RAM", total),
            Reading("swapUsed", HostMetric.swapUsed, HostUnit.bytes, "Swap used", swap_used,
                    swap_reason),
            Reading("swapTotal", HostMetric.swapTotal, HostUnit.bytes, "Swap", swap_total,
                    swap_reason),
        ]  # fmt: skip

    def _disks(self) -> list[Reading]:
        volumes: dict[str, list[Folder]] = {}
        for folder in self._folders():
            if not folder.path:
                continue
            try:
                volume = volume_of(folder.path)
            except OSError:
                continue
            volumes.setdefault(volume, []).append(folder)
        out: list[Reading] = []
        for volume, folders in sorted(volumes.items()):
            holds = tuple(dict.fromkeys(f"{f.role}: {f.path}" for f in folders))
            free, total, reason = self._usage(volume, folders[0].path)
            for metric, value, label in (
                (HostMetric.diskFree, free, f"Free on {volume}"),
                (HostMetric.diskTotal, total, f"Size of {volume}"),
            ):
                out.append(
                    Reading(
                        key=f"disk:{volume}:{metric.value}",
                        metric=metric,
                        unit=HostUnit.bytes,
                        label=label,
                        value=value,
                        unavailable=reason if value is None else None,
                        volume=volume,
                        holds=holds,
                    )
                )
        return out

    def _usage(self, volume: str, path: str) -> tuple[float | None, float | None, str | None]:
        pending = self._disk_pending.get(volume)
        if pending is not None and not pending.done():
            return None, None, f"{volume} has not answered for {DISK_TIMEOUT_SECONDS:.0f} s or more"
        future = self._disk_pool.submit(_disk_usage, path)
        self._disk_pending[volume] = future
        try:
            free, total = future.result(timeout=DISK_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            return None, None, f"{volume} did not answer within {DISK_TIMEOUT_SECONDS:.0f} s"
        except OSError as exc:
            return None, None, f"{path} could not be read: {exc}"
        return float(free), float(total), None

    def close(self) -> None:
        if self._nvml is not None:
            self._nvml.close()
        if self._pdh is not None:
            self._pdh.close()
        self._disk_pool.shutdown(wait=False, cancel_futures=True)


def _disk_usage(path: str) -> tuple[int, int]:
    usage = psutil.disk_usage(nearest_existing(path))
    return int(usage.free), int(usage.total)


def _shared_used(total: int | None, ram_available: int) -> float | None:
    """A unified-memory GPU's use: what of its allowance RAM no longer has."""
    if total is None:
        return None
    return float(max(0, total - min(total, ram_available)))


def _quiet_adapters() -> list[gpu_probe.Adapter]:
    try:
        return gpu_probe.adapters()
    except gpu_probe.GpuProbeError as exc:
        log.debug("host history: no OS adapter list: %s", exc)
        return []


def _process_trees(roots: Iterable[int]) -> set[int]:
    """The engine processes and every descendant, by pid."""
    out: set[int] = set()
    for pid in roots:
        out.add(pid)
        with contextlib.suppress(psutil.Error):
            out.update(child.pid for child in psutil.Process(pid).children(recursive=True))
    return out


__all__ = [
    "Folder",
    "HostProbe",
    "Nvml",
    "Reading",
    "RuntimeCounts",
    "WindowsPdh",
    "amdgpu_process_memory",
    "match_adapters",
    "parse_engine_utilization",
    "parse_fdinfo",
    "parse_process_memory",
    "volume_of",
]
