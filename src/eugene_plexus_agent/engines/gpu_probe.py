"""Every GPU this machine has, asked of the operating system.

**Why this exists (2026-09-27).** An Intel Arc mini PC read "no GPU" on
alpha.3. The engine picker had already chosen the Vulkan build for it,
from `Win32_VideoController`'s names, but the device list asked only
`nvidia-smi`, `rocm-smi` and `xpu-smi` -- none of which a Windows machine
with an Arc or a Radeon has unless its owner installed a vendor SDK. So
the card that build was serving was invisible, and every fit on that
machine was scored against system RAM. The same held for every AMD card
on Windows, and on Linux for every AMD card without ROCm.

The operating system already knows every adapter, its vendor, whether
it is integrated, and how much memory it has and is using. That is one
mechanism for every vendor, and it needs nothing installed:

* **Windows: DXCore**, for the adapters (Windows 10 2004 and later), and
  the **`GPU Adapter Memory` performance counters** through PDH for how
  much of each is in use, system-wide. DXCore's own memory budget is the
  calling process's view, so it cannot say what other programs hold.
  Measured on the development box (an RTX 5090 beside an AMD integrated
  GPU): free memory from these two agreed with `nvidia-smi` to 18 MiB of
  32 GiB, and the AMD adapter reported itself integrated, with a 2 GiB
  carve-out and 75 GiB of shared memory.
* **Linux: sysfs**, `/sys/class/drm/card*/device`. amdgpu publishes VRAM
  and GTT totals and usage there. i915 and xe publish no memory figures
  this code knows, so an Intel card's size is reported as unknown.
  **The Linux half is unverified on real hardware**: it is written from
  the kernel's documented sysfs ABI.

**This file is copied, not shared.** The agent and the library each keep
an identical copy, because components share schemas and not code. It
imports nothing but the standard library for that reason, and a change
to one copy is a change to both.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import dataclasses
import logging
import os
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

NVIDIA = "nvidia"
AMD = "amd"
INTEL = "intel"
QUALCOMM = "qualcomm"
OTHER = "other"

# PCI vendor ids, plus the ACPI id Qualcomm's Adreno reports on Windows
# on ARM ("QCOM" as four bytes), which is not a PCI id at all.
_VENDOR_IDS = {
    0x10DE: NVIDIA,
    0x1002: AMD,
    0x1022: AMD,
    0x8086: INTEL,
    0x5143: QUALCOMM,
    0x4D4F4351: QUALCOMM,
}
_VENDOR_NAMES = {NVIDIA: "NVIDIA", AMD: "AMD", INTEL: "Intel", QUALCOMM: "Qualcomm"}


class GpuProbeError(Exception):
    """The operating system could not be asked. Never means *no GPU*."""


@dataclass(frozen=True)
class Adapter:
    """One hardware display adapter, as the operating system reports it."""

    name: str
    vendor: str
    integrated: bool
    dedicated_bytes: int | None
    """Memory of its own: VRAM on a card, the boot-time carve-out on an
    integrated GPU. None when the platform will not say."""
    shared_bytes: int | None
    """Host RAM the operating system lets it address."""
    dedicated_used_bytes: int | None = None
    """System-wide, every process included. None when unreadable."""
    shared_used_bytes: int | None = None

    def budget(self, ram_available: int | None) -> tuple[int | None, int | None]:
        """`(total, free)` as an engine on this adapter would have them.

        A card has its VRAM. An integrated GPU has its carve-out plus the
        shared allowance; what is left of the allowance is also capped by
        the RAM actually available, because the allowance is a ceiling
        the operating system will let it reach, not memory set aside.
        """
        if not self.integrated:
            if self.dedicated_bytes is None:
                return None, None
            if self.dedicated_used_bytes is None:
                return self.dedicated_bytes, None
            return self.dedicated_bytes, max(0, self.dedicated_bytes - self.dedicated_used_bytes)
        dedicated = self.dedicated_bytes or 0
        shared = self.shared_bytes or 0
        total = dedicated + shared
        if total == 0:
            return None, None
        if self.dedicated_used_bytes is None or self.shared_used_bytes is None:
            return total, None
        shared_left = max(0, shared - self.shared_used_bytes)
        if ram_available is not None:
            shared_left = min(shared_left, ram_available)
        return total, max(0, dedicated - self.dedicated_used_bytes) + shared_left


@dataclass(frozen=True)
class Family:
    """Which build serves this machine's GPUs, when no vendor tool answered.

    `accelerator` is `"vulkan"`, `"rocm"`, `"sycl"` or `"none"`, spelled
    as the agent's `HostAccelerator.accelerator`. `adapters` are the ones
    that build would compute on, in the order the operating system lists
    them. `notes` name every GPU left out and why, because a card the
    product skips without saying so reads as a product that is slow.
    """

    accelerator: str
    adapters: tuple[Adapter, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


def vulkan_build_published(os_name: str, arch: str) -> bool:
    """Whether upstream publishes a Vulkan llama.cpp for this platform.

    Read off the b11211 release (2026-09-27): `win-vulkan-x64`,
    `ubuntu-vulkan-x64` and `ubuntu-vulkan-arm64`. There is no
    `win-vulkan-arm64`, which is what a Snapdragon X laptop would need.
    """
    if os_name == "windows":
        return arch == "x64"
    return os_name == "linux" and arch in ("x64", "arm64")


def combinable_with_cuda(os_name: str, arch: str) -> bool:
    """Whether the Vulkan backend can be added to the CUDA build here.

    Windows x64 only. Upstream's Windows CUDA and Vulkan builds share
    byte-identical core libraries and the backends are plug-ins; the
    Vulkan one adds exactly `ggml-vulkan.dll` (measured on b11211). The
    Linux builds are compiled separately, and even `libggml-base.so`
    differs between them, so adding one's backend to the other would be
    an ABI gamble.
    """
    return os_name == "windows" and arch == "x64"


def beside_nvidia(
    os_name: str, arch: str, adapters: Sequence[Adapter], *, vulkan_loader: bool
) -> list[Adapter]:
    """The discrete AMD or Intel cards a Vulkan backend beside CUDA would add.

    What llama.cpp does with the combined build and nothing pinned: every
    discrete GPU through whichever backend reaches it, an NVIDIA card
    once (it skips the Vulkan copy of a card CUDA already has, by PCI
    id), and no integrated GPU while there is a discrete one. So a
    discrete Radeon or Arc beside a 5090 joins the split, and the
    integrated GPU in the same machine does not.
    """
    if not vulkan_loader or not combinable_with_cuda(os_name, arch):
        return []
    return [a for a in adapters if not a.integrated and a.vendor in (AMD, INTEL)]


def vulkan_selection(adapters: Sequence[Adapter]) -> list[Adapter]:
    """The adapters llama.cpp's Vulkan backend uses when nothing is pinned.

    Every discrete GPU, or, when there is none, the first integrated one:
    upstream's own default in `ggml_vk_instance_init`. Mirrored rather
    than improved on, because a device list that disagrees with the
    engine is a fit scored against a card the engine does not use. On
    the development box that is the difference between a 5090 and an
    integrated Radeon whose 77 GiB shared allowance would otherwise look
    like the biggest card in the machine.
    """
    discrete = [a for a in adapters if not a.integrated]
    return discrete if discrete else list(adapters[:1])


def family(
    os_name: str,
    arch: str,
    adapters: Sequence[Adapter],
    *,
    vulkan_loader: bool,
    rocm: bool = False,
    sycl: bool = False,
) -> Family:
    """Decide the build for a machine no vendor tool answered on.

    `rocm` means AMD's SDK is installed without its tool answering (the
    HIP SDK on Windows, `/opt/rocm` on Linux); `sycl` means `sycl-ls` saw
    an Intel GPU. The caller has already tried NVIDIA's tool, so an
    NVIDIA adapter here is one without it (the Mesa driver on Linux)
    and Vulkan is the build that can reach it.
    """
    gpus = [a for a in adapters if a.vendor != OTHER]
    if not gpus:
        return Family("none")

    if rocm:
        amd = [a for a in gpus if a.vendor == AMD]
        if amd:
            return Family("rocm", tuple(vulkan_selection(amd)))
    if sycl:
        intel = [a for a in gpus if a.vendor == INTEL]
        if intel:
            return Family("sycl", tuple(vulkan_selection(intel)))

    notes: list[str] = []
    if not vulkan_build_published(os_name, arch):
        for adapter in gpus:
            if adapter.vendor == QUALCOMM:
                notes.append(
                    f"{adapter.name} is here, and llama.cpp publishes no Vulkan build for "
                    "Windows on ARM. Its Adreno build (win-opencl-adreno-arm64) is tuned "
                    "for Q4_0 models and has not been run by this project, so models run "
                    "on the CPU build, which is tuned for ARM. Choose the Adreno build "
                    "under the engine's other builds to try it."
                )
            else:
                notes.append(
                    f"{adapter.name} is here, and llama.cpp publishes no Vulkan build for "
                    f"{os_name} on {arch}, so models run on the CPU."
                )
        return Family("none", (), tuple(notes))

    if not vulkan_loader:
        names = ", ".join(a.name for a in gpus)
        package = (
            "your graphics driver" if os_name == "windows" else "libvulkan1 and mesa-vulkan-drivers"
        )
        return Family(
            "none",
            (),
            (
                f"{names} is here, but the Vulkan loader is not installed, so the Vulkan "
                f"build could not use it and models run on the CPU. Install {package}, "
                "then reinstall llama.cpp.",
            ),
        )

    chosen = vulkan_selection(gpus)
    # An integrated GPU beside a card is left out without a note: that is
    # every desktop with a graphics card in it, and saying so on every
    # read would be noise. A second integrated GPU is the case worth a
    # sentence, because nothing else would explain it.
    if all(a.integrated for a in gpus):
        for adapter in gpus[1:]:
            notes.append(
                f"{adapter.name} is also here; llama.cpp's Vulkan build uses one integrated "
                f"GPU when there is no discrete card, and that is {chosen[0].name}."
            )
    return Family("vulkan", tuple(chosen), tuple(notes))


def vulkan_loader_present(os_name: str) -> bool:
    """Whether the Vulkan loader the Vulkan build links against is here.

    Every current GPU driver on Windows installs `vulkan-1.dll` into
    System32. On Linux it is the `libvulkan1` package, which a desktop
    install has and a server install may not.
    """
    if os_name == "windows":
        root = os.environ.get("SYSTEMROOT") or r"C:\Windows"
        return (Path(root) / "System32" / "vulkan-1.dll").is_file()
    for candidate in (
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
        "/usr/lib64/libvulkan.so.1",
        "/usr/lib/libvulkan.so.1",
    ):
        if Path(candidate).is_file():
            return True
    return ctypes.util.find_library("vulkan") is not None


def adapters(os_name: str | None = None) -> list[Adapter]:
    """Every hardware GPU on this machine. Raises `GpuProbeError` when the
    operating system could not be asked; an empty list means none."""
    name = os_name or _os_name()
    if name == "windows":
        return windows_adapters()
    if name == "linux":
        return linux_adapters()
    return []


def _os_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


# --------------------------------------------------------------------------- #
# Windows: DXCore for the adapters, PDH for what they are using
# --------------------------------------------------------------------------- #


class _GUID(ctypes.Structure):
    _fields_ = (
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    )

    @classmethod
    def of(cls, text: str) -> _GUID:
        value = uuid.UUID(text)
        guid = cls()
        guid.Data1, guid.Data2, guid.Data3 = value.fields[0], value.fields[1], value.fields[2]
        for index, byte in enumerate(value.bytes[8:]):
            guid.Data4[index] = byte
        return guid


# From dxcore_interface.h.
_IID_ADAPTER_FACTORY = "78ee5945-c36e-4b13-a669-005dd11c0f06"
_IID_ADAPTER_LIST = "526c7776-40e9-459b-b711-f32ad76dfc28"
_IID_ADAPTER = "f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e"
_ATTRIBUTE_CORE_COMPUTE = "248e2800-a793-4724-abaa-23a6de1be090"

# DXCoreAdapterProperty.
_INSTANCE_LUID = 0
_DRIVER_DESCRIPTION = 2
_HARDWARE_ID = 3
_DEDICATED_ADAPTER_MEMORY = 7
_SHARED_SYSTEM_MEMORY = 9
_IS_HARDWARE = 11
_IS_INTEGRATED = 12

_PDH_FMT_LARGE = 0x00000400


def _method(obj: ctypes.c_void_p, index: int, restype: Any, *argtypes: Any) -> Callable[..., Any]:
    """Call slot `index` of a COM object's vtable."""
    if sys.platform != "win32":  # pragma: no cover - narrows `ctypes.WINFUNCTYPE` for type checkers
        raise GpuProbeError("COM exists only on Windows")
    table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    function = prototype(table[index])
    return lambda *args: function(obj, *args)


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        with contextlib.suppress(Exception):
            _method(obj, 2, ctypes.c_ulong)()


def windows_adapters() -> list[Adapter]:
    """Hardware compute adapters from DXCore, with system-wide usage."""
    if sys.platform != "win32":  # pragma: no cover - the import-time guard for type checkers
        raise GpuProbeError("DXCore exists only on Windows")
    try:
        dxcore = ctypes.WinDLL("dxcore.dll")
    except OSError as exc:
        raise GpuProbeError(
            f"DXCore is not on this machine ({exc}); it arrived in Windows 10 version 2004"
        ) from exc

    hresult = ctypes.c_long
    factory = ctypes.c_void_p()
    status = dxcore.DXCoreCreateAdapterFactory(
        ctypes.byref(_GUID.of(_IID_ADAPTER_FACTORY)), ctypes.byref(factory)
    )
    if status != 0:
        raise GpuProbeError(f"DXCoreCreateAdapterFactory failed with 0x{status & 0xFFFFFFFF:08X}")
    listing = ctypes.c_void_p()
    found: list[tuple[Adapter, str]] = []
    try:
        create_list = _method(
            factory,
            3,
            hresult,
            ctypes.c_uint32,
            ctypes.POINTER(_GUID),
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )
        attribute = _GUID.of(_ATTRIBUTE_CORE_COMPUTE)
        status = create_list(
            1,
            ctypes.byref(attribute),
            ctypes.byref(_GUID.of(_IID_ADAPTER_LIST)),
            ctypes.byref(listing),
        )
        if status != 0:
            raise GpuProbeError(f"DXCore could not list adapters: 0x{status & 0xFFFFFFFF:08X}")
        count = _method(listing, 4, ctypes.c_uint32)()
        get_adapter = _method(
            listing,
            3,
            hresult,
            ctypes.c_uint32,
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )
        for index in range(count):
            adapter = ctypes.c_void_p()
            if get_adapter(index, ctypes.byref(_GUID.of(_IID_ADAPTER)), ctypes.byref(adapter)):
                continue
            try:
                read = _describe(adapter)
                if read is not None:
                    found.append(read)
            finally:
                _release(adapter)
    finally:
        _release(listing)
        _release(factory)

    usage = _windows_usage()
    out: list[Adapter] = []
    for described, luid in found:
        dedicated_used, shared_used = usage.get(luid, (None, None))
        out.append(
            dataclasses.replace(
                described, dedicated_used_bytes=dedicated_used, shared_used_bytes=shared_used
            )
        )
    return out


def _describe(adapter: ctypes.c_void_p) -> tuple[Adapter, str] | None:
    """One DXCore adapter, or None for a software renderer."""
    hresult = ctypes.c_long
    get_size = _method(adapter, 7, hresult, ctypes.c_uint32, ctypes.POINTER(ctypes.c_size_t))
    get = _method(adapter, 6, hresult, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p)

    def raw(prop: int) -> bytes | None:
        size = ctypes.c_size_t()
        if get_size(prop, ctypes.byref(size)) != 0 or size.value == 0:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if get(prop, size.value, buffer) != 0:
            return None
        return buffer.raw

    hardware = raw(_IS_HARDWARE)
    if not hardware or not hardware[0]:
        # The Microsoft Basic Render Driver: a CPU pretending to be a GPU.
        return None
    hardware_id = raw(_HARDWARE_ID)
    luid = raw(_INSTANCE_LUID)
    description = raw(_DRIVER_DESCRIPTION)
    dedicated = raw(_DEDICATED_ADAPTER_MEMORY)
    shared = raw(_SHARED_SYSTEM_MEMORY)
    integrated = raw(_IS_INTEGRATED)
    vendor_id = int.from_bytes(hardware_id[0:4], "little") if hardware_id else None
    vendor = _VENDOR_IDS.get(vendor_id, OTHER) if vendor_id is not None else OTHER
    name = (
        description.split(b"\x00", 1)[0].decode("utf-8", "replace").strip() if description else ""
    )
    key = ""
    if luid and len(luid) >= 8:
        low = int.from_bytes(luid[0:4], "little")
        high = int.from_bytes(luid[4:8], "little")
        key = f"luid_0x{high:08x}_0x{low:08x}"
    return (
        Adapter(
            name=name or f"{_VENDOR_NAMES.get(vendor, 'Unknown')} GPU",
            vendor=vendor,
            integrated=bool(integrated and integrated[0]),
            dedicated_bytes=int.from_bytes(dedicated[0:8], "little") if dedicated else None,
            shared_bytes=int.from_bytes(shared[0:8], "little") if shared else None,
        ),
        key,
    )


class _PdhCounterValue(ctypes.Structure):
    _fields_ = (("CStatus", ctypes.c_ulong), ("largeValue", ctypes.c_longlong))


class _PdhCounterItem(ctypes.Structure):
    _fields_ = (("szName", ctypes.c_wchar_p), ("FmtValue", _PdhCounterValue))


def _windows_usage() -> dict[str, tuple[int | None, int | None]]:
    """`{luid key: (dedicated used, shared used)}`, system-wide.

    Empty when the counters are not there, which leaves free memory
    unknown rather than equal to the total: a fit against a total is a
    promise that the rest of the machine holds nothing.
    """
    if sys.platform != "win32":  # pragma: no cover
        return {}
    try:
        pdh = ctypes.WinDLL("pdh.dll")
    except OSError as exc:  # pragma: no cover - pdh.dll ships with Windows
        log.debug("pdh.dll unavailable: %s", exc)
        return {}
    query = ctypes.c_void_p()
    if pdh.PdhOpenQueryW(None, None, ctypes.byref(query)) != 0:
        return {}
    try:
        handles: dict[str, ctypes.c_void_p] = {}
        for counter in ("Dedicated Usage", "Shared Usage"):
            handle = ctypes.c_void_p()
            path = f"\\GPU Adapter Memory(*)\\{counter}"
            status = pdh.PdhAddEnglishCounterW(query, path, None, ctypes.byref(handle))
            if status != 0:
                log.debug("PDH counter %s unavailable: 0x%08X", path, status & 0xFFFFFFFF)
                return {}
            handles[counter] = handle
        if pdh.PdhCollectQueryData(query) != 0:
            return {}
        read: dict[str, dict[str, int]] = {}
        for counter, handle in handles.items():
            size = ctypes.c_ulong(0)
            count = ctypes.c_ulong(0)
            pdh.PdhGetFormattedCounterArrayW(
                handle, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None
            )
            if size.value == 0:
                continue
            buffer = ctypes.create_string_buffer(size.value)
            if pdh.PdhGetFormattedCounterArrayW(
                handle, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buffer
            ):
                continue
            items = ctypes.cast(buffer, ctypes.POINTER(_PdhCounterItem))
            totals = read.setdefault(counter, {})
            for index in range(count.value):
                item = items[index]
                if item.FmtValue.CStatus > 1 or not item.szName:
                    continue
                # `luid_0x00000000_0x0000D550_phys_0`: an adapter linked
                # from several physical GPUs has one instance per GPU.
                key = item.szName.lower().split("_phys", 1)[0]
                totals[key] = totals.get(key, 0) + int(item.FmtValue.largeValue)
    finally:
        pdh.PdhCloseQuery(query)
    dedicated = read.get("Dedicated Usage", {})
    shared = read.get("Shared Usage", {})
    return {key: (dedicated.get(key), shared.get(key)) for key in set(dedicated) | set(shared)}


# --------------------------------------------------------------------------- #
# Linux: sysfs
# --------------------------------------------------------------------------- #

_DISPLAY_CLASS = 0x03
_INTEL_INTEGRATED_SLOT = "0000:00:02.0"
_APU_CARVE_OUT_CEILING = 2 * 1024**3


def linux_adapters(root: Path | str = "/sys/class/drm") -> list[Adapter]:
    """Display controllers under `/sys/class/drm`, with amdgpu's memory.

    UNVERIFIED on real hardware (the development box's Linux is WSL2,
    which has no DRM devices). Written from the kernel's sysfs ABI:
    `vendor`, `class`, and amdgpu's `mem_info_vram_*` / `mem_info_gtt_*`.

    Two heuristics, each named. An Intel GPU at PCI `0000:00:02.0` is
    the integrated one (that slot has been Intel's iGPU for over a
    decade; a discrete Arc sits behind a bridge). An AMD GPU whose VRAM
    is under 2 GiB while its GTT is larger is an APU: its "VRAM" is a
    boot-time carve-out of system memory. A Strix Halo with a 96 GB
    carve-out reads as a card with 96 GB, which is how its memory
    behaves once reserved.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    out: list[Adapter] = []
    # A connector (`card0-DP-1`) matches the glob too. Its `device` is the
    # DRM card rather than the PCI device, so it has no `vendor` file and
    # is skipped by the read below; a name filter beside it could never
    # change an answer.
    for card in sorted(base.glob("card[0-9]*")):
        device = card / "device"
        vendor_id = _hex(device / "vendor")
        device_class = _hex(device / "class")
        if vendor_id is None or device_class is None or device_class >> 16 != _DISPLAY_CLASS:
            continue
        vendor = _VENDOR_IDS.get(vendor_id, OTHER)
        slot = _slot(device)
        name = (
            _read(device / "product_name") or f"{_VENDOR_NAMES.get(vendor, 'Unknown')} GPU ({slot})"
        )
        vram = _int(device / "mem_info_vram_total")
        vram_used = _int(device / "mem_info_vram_used")
        gtt = _int(device / "mem_info_gtt_total")
        gtt_used = _int(device / "mem_info_gtt_used")
        if vendor == AMD and vram is not None:
            apu = vram < _APU_CARVE_OUT_CEILING and gtt is not None and gtt > vram
            out.append(
                Adapter(
                    name=name,
                    vendor=vendor,
                    integrated=apu,
                    dedicated_bytes=vram,
                    shared_bytes=gtt if apu else None,
                    dedicated_used_bytes=vram_used,
                    shared_used_bytes=gtt_used if apu else None,
                )
            )
            continue
        out.append(
            Adapter(
                name=name,
                vendor=vendor,
                integrated=vendor == INTEL and slot == _INTEL_INTEGRATED_SLOT,
                dedicated_bytes=None,
                shared_bytes=None,
            )
        )
    return out


def _slot(device: Path) -> str:
    """The PCI address, `0000:00:02.0`: `PCI_SLOT_NAME` in the device's
    `uevent`, else the name of the directory `device` links to."""
    for line in (_read(device / "uevent") or "").splitlines():
        key, _, value = line.partition("=")
        if key == "PCI_SLOT_NAME" and value:
            return value.strip()
    return device.resolve().name


def _read(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return text or None


def _hex(path: Path) -> int | None:
    text = _read(path)
    try:
        return int(text, 16) if text else None
    except ValueError:
        return None


def _int(path: Path) -> int | None:
    text = _read(path)
    return int(text) if text and text.isdigit() else None


__all__ = [
    "AMD",
    "INTEL",
    "NVIDIA",
    "OTHER",
    "QUALCOMM",
    "Adapter",
    "Family",
    "GpuProbeError",
    "adapters",
    "beside_nvidia",
    "combinable_with_cuda",
    "family",
    "linux_adapters",
    "vulkan_build_published",
    "vulkan_loader_present",
    "vulkan_selection",
    "windows_adapters",
]
