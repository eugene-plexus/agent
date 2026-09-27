"""An Intel Arc mini PC read "no GPU" on alpha.3 (2026-09-26).

R2.3 taught the engine picker to see an Intel or AMD card on Windows and
give it the Vulkan build, and the device list went on asking only
`nvidia-smi`, `rocm-smi` and `xpu-smi`. So the Arc ran the Vulkan build
and every fit, admission and starter pick on that machine was scored
against system RAM. The same read found four more machines the product
left out: every AMD card on Windows (the same "no GPU"), a Snapdragon X
laptop (asked for a `win-vulkan-arm64` upstream does not publish), an
AMD card on Linux without ROCm (a CPU build, with `ubuntu-vulkan-x64`
published), and every Intel laptop on Linux (a SYCL build it needs a
oneAPI runtime to load, chosen because the `i915` driver directory
exists).

The operating system already lists every adapter, its vendor, whether
it is integrated and how much memory it has and uses: DXCore and the
GPU performance counters on Windows, sysfs on Linux. `gpu_probe` asks
it, and `gpu_probe.family` is the one decision both the engine picker
and the device list make, so they cannot disagree about which card a
model is scored against.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import eugene_plexus_agent.admission as admission_module
from eugene_plexus_agent._generated.models import (
    Accelerator,
    Arch,
    ComputeDevice,
    ComputeDeviceKind,
    HostAccelerator,
    Os,
    RuntimeSpec,
)
from eugene_plexus_agent.admission import LibraryFit, check_admission
from eugene_plexus_agent.engines import gpu_probe
from eugene_plexus_agent.engines import host as host_mod
from eugene_plexus_agent.engines.acquisition import Release, ReleaseAsset, Unavailable
from eugene_plexus_agent.engines.devices import DeviceSnapshot, detect_devices
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter, alternatives

GIB = 1024**3
MIB = 1024**2

# Upstream's b11211 (2026-09-27), every server build, verbatim.
_B11211 = [
    "llama-b11211-bin-macos-arm64.tar.gz",
    "llama-b11211-bin-macos-x64.tar.gz",
    "llama-b11211-bin-ubuntu-arm64.tar.gz",
    "llama-b11211-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b11211-bin-ubuntu-cuda-13.4-arm64.tar.gz",
    "llama-b11211-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-b11211-bin-ubuntu-openvino-2026.4-x64.tar.gz",
    "llama-b11211-bin-ubuntu-rocm-10.0-x64.tar.gz",
    "llama-b11211-bin-ubuntu-s390x.tar.gz",
    "llama-b11211-bin-ubuntu-sycl-fp16-x64.tar.gz",
    "llama-b11211-bin-ubuntu-sycl-fp32-x64.tar.gz",
    "llama-b11211-bin-ubuntu-vulkan-arm64.tar.gz",
    "llama-b11211-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b11211-bin-ubuntu-x64.tar.gz",
    "llama-b11211-bin-win-cpu-arm64.zip",
    "llama-b11211-bin-win-cpu-x64.zip",
    "llama-b11211-bin-win-cuda-12.4-x64.zip",
    "llama-b11211-bin-win-cuda-13.4-arm64.zip",
    "llama-b11211-bin-win-cuda-13.4-x64.zip",
    "llama-b11211-bin-win-opencl-adreno-arm64.zip",
    "llama-b11211-bin-win-openvino-2026.4-x64.zip",
    "llama-b11211-bin-win-rocm-10.0-x64.zip",
    "llama-b11211-bin-win-sycl-x64.zip",
    "llama-b11211-bin-win-vulkan-x64.zip",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
]


def _release(names: list[str] = _B11211) -> Release:
    return Release(
        version="b11211",
        published_at=datetime(2026, 9, 27, 3, 0, tzinfo=UTC),
        assets=tuple(
            ReleaseAsset(
                name=n, url=f"https://example.invalid/{n}", size=1024, digest="sha256:" + "0" * 64
            )
            for n in names
        ),
    )


def _card(name: str, vendor: str, vram_gib: float, used_gib: float = 1.0) -> gpu_probe.Adapter:
    return gpu_probe.Adapter(
        name=name,
        vendor=vendor,
        integrated=False,
        dedicated_bytes=int(vram_gib * GIB),
        shared_bytes=32 * GIB,
        dedicated_used_bytes=int(used_gib * GIB),
        shared_used_bytes=0,
    )


def _igpu(
    name: str, vendor: str, carve_out_mib: int = 128, shared_gib: float = 16.0
) -> gpu_probe.Adapter:
    return gpu_probe.Adapter(
        name=name,
        vendor=vendor,
        integrated=True,
        dedicated_bytes=carve_out_mib * MIB,
        shared_bytes=int(shared_gib * GIB),
        dedicated_used_bytes=16 * MIB,
        shared_used_bytes=1 * GIB,
    )


ARC_IGPU = _igpu("Intel(R) Arc(TM) Graphics", gpu_probe.INTEL)
ARC_A770 = _card("Intel(R) Arc(TM) A770 Graphics", gpu_probe.INTEL, 16)
UHD_770 = _igpu("Intel(R) UHD Graphics 770", gpu_probe.INTEL)
RTX_5090 = _card("NVIDIA GeForce RTX 5090", gpu_probe.NVIDIA, 31.4, 2.2)
RADEON_IGPU = _igpu("AMD Radeon(TM) Graphics", gpu_probe.AMD, carve_out_mib=2022, shared_gib=74.8)
RX_7900 = _card("AMD Radeon RX 7900 XTX", gpu_probe.AMD, 24)
ADRENO = _igpu("Qualcomm(R) Adreno(TM) X1-85 GPU", gpu_probe.QUALCOMM)


# --------------------------------------------------------------------------- #
# gpu_probe: memory, and which adapters a build uses
# --------------------------------------------------------------------------- #


def test_a_card_has_its_vram_and_what_the_whole_machine_is_using_of_it() -> None:
    total, free = RTX_5090.budget(ram_available=60 * GIB)
    assert total == int(31.4 * GIB)
    assert free == int(31.4 * GIB) - int(2.2 * GIB)


def test_a_card_whose_usage_is_unreadable_has_no_free_figure_not_its_total() -> None:
    """A free figure equal to the total is a promise that nothing else on
    the machine holds any of it, which is never true of a desktop."""
    card = gpu_probe.Adapter("X", gpu_probe.AMD, False, 8 * GIB, 16 * GIB, None, None)
    assert card.budget(ram_available=60 * GIB) == (8 * GIB, None)


def test_an_integrated_gpu_is_its_carve_out_plus_what_it_may_borrow() -> None:
    """Capped by the RAM actually free, because the shared allowance is a
    ceiling the OS will let it reach and not memory set aside."""
    total, free = ARC_IGPU.budget(ram_available=64 * GIB)
    assert total == 128 * MIB + 16 * GIB
    assert free == (128 - 16) * MIB + 15 * GIB

    _, squeezed = ARC_IGPU.budget(ram_available=4 * GIB)
    assert squeezed == (128 - 16) * MIB + 4 * GIB


def test_the_vulkan_build_uses_the_card_not_the_integrated_gpu_beside_it() -> None:
    """This development box: a 5090 and an integrated Radeon whose 75 GiB
    shared allowance would read as the biggest GPU in the machine."""
    assert gpu_probe.vulkan_selection([RADEON_IGPU, RTX_5090]) == [RTX_5090]
    assert gpu_probe.vulkan_selection([UHD_770, ARC_A770]) == [ARC_A770]


def test_with_no_card_the_vulkan_build_uses_the_first_integrated_gpu() -> None:
    assert gpu_probe.vulkan_selection([ARC_IGPU]) == [ARC_IGPU]


def _family(adapters: list[gpu_probe.Adapter], **kw: object) -> gpu_probe.Family:
    os_name = str(kw.pop("os_name", "windows"))
    arch = str(kw.pop("arch", "x64"))
    kw.setdefault("vulkan_loader", True)
    return gpu_probe.family(os_name, arch, adapters, **kw)  # type: ignore[arg-type]


def test_an_arc_mini_pc_is_served_by_vulkan_with_its_own_gpu() -> None:
    """**The report.** Jessie's machine, whichever Arc it holds."""
    chosen = _family([ARC_IGPU])
    assert chosen.accelerator == "vulkan"
    assert chosen.adapters == (ARC_IGPU,)
    assert _family([UHD_770, ARC_A770]).adapters == (ARC_A770,)


def test_rocm_is_for_an_amd_adapter_not_merely_an_sdk_on_disk() -> None:
    assert _family([RX_7900], rocm=True).accelerator == "rocm"
    assert _family([ARC_A770], rocm=True).accelerator == "vulkan"


def test_sycl_is_for_an_intel_adapter_its_runtime_can_see() -> None:
    assert _family([ARC_A770], sycl=True).accelerator == "sycl"
    assert _family([RX_7900], sycl=True).accelerator == "vulkan"


def test_a_snapdragon_runs_on_the_cpu_and_says_which_build_would_use_its_gpu() -> None:
    """There is no `win-vulkan-arm64`. Asking for one failed the install."""
    chosen = _family([ADRENO], arch="arm64")
    assert chosen.accelerator == "none"
    assert chosen.adapters == ()
    [note] = chosen.notes
    assert "Adreno" in note and "win-opencl-adreno-arm64" in note


def test_without_a_vulkan_loader_the_gpu_is_named_and_so_is_the_package() -> None:
    chosen = _family([RX_7900], os_name="linux", vulkan_loader=False)
    assert chosen.accelerator == "none"
    [note] = chosen.notes
    assert "AMD Radeon RX 7900 XTX" in note and "libvulkan1" in note


def test_an_adapter_from_no_gpu_vendor_is_not_a_gpu() -> None:
    other = gpu_probe.Adapter("Microsoft Hyper-V Video", gpu_probe.OTHER, False, 0, 0)
    assert _family([other]).accelerator == "none"


def test_a_second_integrated_gpu_is_named_rather_than_silently_skipped() -> None:
    other = _igpu("AMD Radeon(TM) 780M", gpu_probe.AMD)
    chosen = _family([ARC_IGPU, other])
    assert chosen.adapters == (ARC_IGPU,)
    [note] = chosen.notes
    assert "AMD Radeon(TM) 780M" in note


# --------------------------------------------------------------------------- #
# gpu_probe: Linux sysfs
# --------------------------------------------------------------------------- #


def _sysfs_card(
    root: Path, card: str, *, vendor: str, cls: str = "0x030000", slot: str, **files: str
) -> None:
    device = root / card / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(vendor + "\n")
    (device / "class").write_text(cls + "\n")
    (device / "uevent").write_text(f"DRIVER=x\nPCI_SLOT_NAME={slot}\n")
    for name, value in files.items():
        (device / name).write_text(value + "\n")


def test_sysfs_reads_an_amd_card_an_apu_and_intel_graphics(tmp_path: Path) -> None:
    _sysfs_card(
        tmp_path,
        "card0",
        vendor="0x1002",
        slot="0000:03:00.0",
        mem_info_vram_total=str(16 * GIB),
        mem_info_vram_used=str(1 * GIB),
        mem_info_gtt_total=str(32 * GIB),
        mem_info_gtt_used="0",
    )
    _sysfs_card(
        tmp_path,
        "card1",
        vendor="0x1002",
        slot="0000:c4:00.0",
        mem_info_vram_total=str(512 * MIB),
        mem_info_vram_used=str(100 * MIB),
        mem_info_gtt_total=str(30 * GIB),
        mem_info_gtt_used=str(2 * GIB),
    )
    _sysfs_card(tmp_path, "card2", vendor="0x8086", slot="0000:00:02.0")
    _sysfs_card(tmp_path, "card3", vendor="0x8086", slot="0000:03:00.0")
    # A connector, and a PCI device that is not a display controller.
    (tmp_path / "card0-DP-1").mkdir()
    _sysfs_card(tmp_path, "card4", vendor="0x8086", cls="0x040300", slot="0000:00:1f.3")

    card, apu, igpu, arc = gpu_probe.linux_adapters(tmp_path)
    assert (card.vendor, card.integrated, card.dedicated_bytes) == ("amd", False, 16 * GIB)
    assert card.budget(ram_available=64 * GIB) == (16 * GIB, 15 * GIB)
    assert (apu.integrated, apu.shared_bytes) == (True, 30 * GIB)
    assert (igpu.vendor, igpu.integrated) == ("intel", True)
    assert (arc.vendor, arc.integrated, arc.dedicated_bytes) == ("intel", False, None)


def test_a_machine_with_no_drm_directory_has_no_adapters(tmp_path: Path) -> None:
    assert gpu_probe.linux_adapters(tmp_path / "absent") == []


# --------------------------------------------------------------------------- #
# The engine picker
# --------------------------------------------------------------------------- #


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch):
    """No vendor tool of any kind, and a Vulkan loader. Set the OS, the
    CPU and the adapters per test."""
    monkeypatch.setattr(host_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(host_mod, "_windows_hip_sdk_present", lambda: False)
    monkeypatch.setattr(host_mod.Path, "is_dir", lambda self: False)
    monkeypatch.setattr(gpu_probe, "vulkan_loader_present", lambda os_name: True)

    def _set(system: str, machine_name: str, adapters: list[gpu_probe.Adapter]) -> None:
        monkeypatch.setattr(host_mod.platform, "system", lambda: system)
        monkeypatch.setattr(host_mod.platform, "machine", lambda: machine_name)
        monkeypatch.setattr(gpu_probe, "adapters", lambda os_name=None: list(adapters))

    return _set


def test_an_arc_on_windows_gets_the_vulkan_build(machine) -> None:
    machine("Windows", "AMD64", [ARC_IGPU])
    detected = host_mod.detect_host()
    assert detected.accelerator is Accelerator.vulkan
    plan = LlamaCppAdapter().plan_acquisition(detected, _release())
    assert not isinstance(plan, Unavailable), getattr(plan, "reason", "")
    assert plan.variant == "win-vulkan-x64"


def test_a_snapdragon_gets_a_build_that_exists(machine) -> None:
    machine("Windows", "ARM64", [ADRENO])
    detected = host_mod.detect_host()
    assert detected.accelerator is Accelerator.none
    plan = LlamaCppAdapter().plan_acquisition(detected, _release())
    assert not isinstance(plan, Unavailable), getattr(plan, "reason", "")
    assert plan.variant == "win-cpu-arm64"


def test_an_amd_card_on_linux_without_rocm_gets_the_vulkan_build(machine) -> None:
    machine("Linux", "x86_64", [RX_7900])
    detected = host_mod.detect_host()
    assert detected.accelerator is Accelerator.vulkan
    plan = LlamaCppAdapter().plan_acquisition(detected, _release())
    assert not isinstance(plan, Unavailable)
    assert plan.variant == "ubuntu-vulkan-x64"


def test_an_intel_laptop_on_linux_is_not_given_a_sycl_build(machine) -> None:
    """The `i915` directory existed and chose SYCL on every Intel laptop."""
    machine("Linux", "x86_64", [UHD_770])
    assert host_mod.detect_host().accelerator is Accelerator.vulkan


def test_an_arm_linux_board_takes_the_arm_vulkan_build(machine) -> None:
    machine("Linux", "aarch64", [RX_7900])
    plan = LlamaCppAdapter().plan_acquisition(host_mod.detect_host(), _release())
    assert not isinstance(plan, Unavailable)
    assert plan.variant == "ubuntu-vulkan-arm64"


def test_no_vulkan_loader_on_linux_is_the_cpu_build(machine, monkeypatch) -> None:
    machine("Linux", "x86_64", [RX_7900])
    monkeypatch.setattr(gpu_probe, "vulkan_loader_present", lambda os_name: False)
    assert host_mod.detect_host().accelerator is Accelerator.none


def test_rocm_in_a_container_that_lists_no_gpus_keeps_the_rocm_build(machine, monkeypatch) -> None:
    machine("Linux", "x86_64", [])
    monkeypatch.setattr(
        host_mod.Path, "is_dir", lambda self: str(self).replace("\\", "/") == "/opt/rocm"
    )
    assert host_mod.detect_host().accelerator is Accelerator.rocm


@pytest.mark.parametrize(
    ("listing", "expected"),
    [
        ("[opencl:cpu][opencl:0] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i7", False),
        ("[level_zero:gpu][level_zero:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770", True),
        ("[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A750", True),
    ],
)
def test_sycl_means_the_runtime_sees_a_gpu_not_just_a_cpu(monkeypatch, listing, expected) -> None:
    monkeypatch.setattr(host_mod, "_output", lambda argv, **kw: listing)
    assert host_mod.sycl_sees_gpu() is expected


def test_windows_with_oneapi_gets_the_sycl_build_not_the_cpu_one() -> None:
    """The Windows branch had no SYCL case and fell through to the CPU."""
    host = HostAccelerator(os=Os.windows, arch=Arch.x64, accelerator=Accelerator.sycl)
    plan = LlamaCppAdapter().plan_acquisition(host, _release())
    assert not isinstance(plan, Unavailable)
    assert plan.variant == "win-sycl-x64"


# --------------------------------------------------------------------------- #
# The expert's other builds
# --------------------------------------------------------------------------- #

_WINDOWS_X64 = HostAccelerator(os=Os.windows, arch=Arch.x64, accelerator=Accelerator.vulkan)


def test_the_other_builds_are_every_one_published_for_this_os_and_cpu() -> None:
    assert alternatives(_WINDOWS_X64, _release()) == [
        "win-cpu-x64",
        "win-cuda-12.4-x64",
        "win-cuda-13.4-x64",
        "win-openvino-2026.4-x64",
        "win-rocm-10.0-x64",
        "win-sycl-x64",
        "win-vulkan-x64",
    ]
    linux_arm = HostAccelerator(os=Os.linux, arch=Arch.arm64, accelerator=Accelerator.none)
    assert alternatives(linux_arm, _release()) == [
        "ubuntu-arm64",
        "ubuntu-cuda-13.4-arm64",
        "ubuntu-vulkan-arm64",
    ]


def test_an_arc_owner_can_choose_sycl_over_the_vulkan_default() -> None:
    plan = LlamaCppAdapter().plan_acquisition(_WINDOWS_X64, _release(), variant="win-sycl-x64")
    assert not isinstance(plan, Unavailable)
    assert plan.variant == "win-sycl-x64"
    assert [a.name for a in plan.assets] == ["llama-b11211-bin-win-sycl-x64.zip"]


def test_a_chosen_cuda_build_brings_its_runtime_with_it() -> None:
    plan = LlamaCppAdapter().plan_acquisition(_WINDOWS_X64, _release(), variant="win-cuda-13.4-x64")
    assert not isinstance(plan, Unavailable)
    assert [a.name for a in plan.assets] == [
        "llama-b11211-bin-win-cuda-13.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.4-x64.zip",
    ]


def test_another_machines_build_is_refused_with_the_list_and_not_looked_for_again() -> None:
    plan = LlamaCppAdapter().plan_acquisition(_WINDOWS_X64, _release(), variant="ubuntu-vulkan-x64")
    assert isinstance(plan, Unavailable)
    assert "win-vulkan-x64" in plan.reason and "win-sycl-x64" in plan.reason
    assert plan.release_bound is False


def test_this_machines_build_missing_from_one_release_is_looked_for_in_the_last() -> None:
    plan = LlamaCppAdapter().plan_acquisition(
        _WINDOWS_X64, _release(["llama-b11211-bin-win-cpu-x64.zip"]), variant="win-sycl-x64"
    )
    assert isinstance(plan, Unavailable)
    assert plan.release_bound is True


# --------------------------------------------------------------------------- #
# The device list
# --------------------------------------------------------------------------- #


def _no_tools(argv: list[str]) -> str | None:
    return None


def _memory() -> tuple[int, int]:
    return 32 * GIB, 26 * GIB


@pytest.fixture
def os_is(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gpu_probe, "vulkan_loader_present", lambda os_name: True)
    monkeypatch.setattr(host_mod, "rocm_installed", lambda os_kind: False)
    monkeypatch.setattr(host_mod, "sycl_sees_gpu", lambda: False)

    def _set(system: str, machine_name: str = "AMD64") -> None:
        monkeypatch.setattr(host_mod.platform, "system", lambda: system)
        monkeypatch.setattr(host_mod.platform, "machine", lambda: machine_name)

    return _set


def test_an_arc_mini_pc_lists_its_gpu_and_says_its_memory_is_shared(os_is) -> None:
    """**The finding**, as the Inference page read it: "no GPU · 26.1 GiB
    host memory free" on a machine whose Vulkan build was using an Arc."""
    os_is("Windows")
    snapshot = detect_devices(run=_no_tools, memory=_memory, os_adapters=lambda: [ARC_IGPU])
    [arc] = snapshot.accelerators()
    assert arc.kind is ComputeDeviceKind.vulkan
    assert arc.name == "Intel(R) Arc(TM) Graphics"
    assert arc.sharedMemory is True
    assert (arc.memoryTotalBytes, arc.memoryFreeBytes) == ARC_IGPU.budget(26 * GIB)
    assert snapshot.devices[-1].kind is ComputeDeviceKind.cpu


def test_a_discrete_arc_is_a_card_with_its_own_memory(os_is) -> None:
    os_is("Windows")
    snapshot = detect_devices(
        run=_no_tools, memory=_memory, os_adapters=lambda: [UHD_770, ARC_A770]
    )
    [arc] = snapshot.accelerators()
    assert (arc.name, arc.sharedMemory, arc.memoryTotalBytes) == (
        "Intel(R) Arc(TM) A770 Graphics",
        None,
        16 * GIB,
    )


def test_every_amd_card_on_windows_is_listed_too(os_is) -> None:
    os_is("Windows")
    [card] = detect_devices(
        run=_no_tools, memory=_memory, os_adapters=lambda: [RX_7900]
    ).accelerators()
    assert (card.kind, card.name) == (ComputeDeviceKind.vulkan, "AMD Radeon RX 7900 XTX")


def test_the_integrated_gpu_beside_a_card_does_not_become_the_budget(os_is) -> None:
    """This development box with `nvidia-smi` out of the way."""
    os_is("Windows")
    devices = detect_devices(
        run=_no_tools, memory=_memory, os_adapters=lambda: [RADEON_IGPU, RTX_5090]
    ).accelerators()
    assert [d.name for d in devices] == ["NVIDIA GeForce RTX 5090"]


def test_a_card_the_cuda_build_cannot_use_is_named(os_is) -> None:
    os_is("Windows")

    def _smi(argv: list[str]) -> str | None:
        return "0, NVIDIA GeForce RTX 5090, 32607, 29996\n" if argv[0] == "nvidia-smi" else None

    snapshot = detect_devices(
        run=_smi, memory=_memory, os_adapters=lambda: [RADEON_IGPU, RTX_5090, RX_7900]
    )
    assert [d.name for d in snapshot.accelerators()] == ["NVIDIA GeForce RTX 5090"]
    named = [w for w in snapshot.warnings if "is also here" in w]
    assert len(named) == 1 and "AMD Radeon RX 7900 XTX" in named[0], snapshot.warnings


def test_a_gb10_computes_out_of_host_memory(os_is) -> None:
    """`nvidia-smi` has no memory figure for it; the row was "could not
    parse" and the machine read "no GPU"."""
    os_is("Linux", "aarch64")

    def _smi(argv: list[str]) -> str | None:
        return "0, NVIDIA GB10, [N/A], [N/A]\n" if argv[0] == "nvidia-smi" else None

    [gb10] = detect_devices(run=_smi, memory=_memory, os_adapters=lambda: []).accelerators()
    assert (gb10.kind, gb10.sharedMemory) == (ComputeDeviceKind.cuda, True)
    assert (gb10.memoryTotalBytes, gb10.memoryFreeBytes) == (32 * GIB, 26 * GIB)


def test_unreadable_usage_is_said_and_free_is_unknown(os_is) -> None:
    os_is("Windows")
    blind = gpu_probe.Adapter("AMD Radeon RX 7900 XTX", gpu_probe.AMD, False, 24 * GIB, None)
    snapshot = detect_devices(run=_no_tools, memory=_memory, os_adapters=lambda: [blind])
    [card] = snapshot.accelerators()
    assert card.memoryFreeBytes is None
    assert any("could not be read" in w for w in snapshot.warnings)


def test_an_os_that_could_not_be_asked_is_a_warning_not_a_crash(os_is) -> None:
    os_is("Windows")

    def _broken() -> list[gpu_probe.Adapter]:
        raise gpu_probe.GpuProbeError("DXCore is not on this machine")

    snapshot = detect_devices(run=_no_tools, memory=_memory, os_adapters=_broken)
    assert snapshot.accelerators() == []
    assert any("DXCore is not on this machine" in w for w in snapshot.warnings)


def test_the_snapdragon_note_reaches_the_device_warnings(os_is) -> None:
    os_is("Windows", "ARM64")
    snapshot = detect_devices(run=_no_tools, memory=_memory, os_adapters=lambda: [ADRENO])
    assert snapshot.accelerators() == []
    assert any("win-opencl-adreno-arm64" in w for w in snapshot.warnings)


# --------------------------------------------------------------------------- #
# Admission against a shared-memory device
# --------------------------------------------------------------------------- #


class _Library:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def fit(self, model_path: str, **kwargs: object) -> LibraryFit:
        self.calls.append(kwargs)
        return LibraryFit(required_bytes=4 * GIB, verdict="fits", context_length=4096)


def _snapshot(device: ComputeDevice) -> DeviceSnapshot:
    cpu = ComputeDevice(
        kind=ComputeDeviceKind.cpu,
        index=0,
        name="CPU",
        memoryTotalBytes=32 * GIB,
        memoryFreeBytes=26 * GIB,
    )
    return DeviceSnapshot(
        devices=(device, cpu),
        warnings=(),
        ram_total_bytes=32 * GIB,
        ram_available_bytes=26 * GIB,
        detected_at=datetime.now(UTC),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("shared", [True, None])
async def test_a_shared_memory_gpu_is_not_given_the_same_ram_twice(
    monkeypatch: pytest.MonkeyPatch, shared: bool | None
) -> None:
    """An integrated GPU's "VRAM" is host RAM. Scored as a card beside
    RAM to spill into, a model twice its size reads as a partial offload
    that fits."""
    monkeypatch.setattr(admission_module, "path_exists", lambda p: True)
    monkeypatch.setattr(admission_module, "model_size_bytes", lambda p: 4 * GIB)
    device = ComputeDevice(
        kind=ComputeDeviceKind.vulkan,
        index=0,
        name="Intel(R) Arc(TM) Graphics",
        memoryTotalBytes=16 * GIB,
        memoryFreeBytes=15 * GIB,
        sharedMemory=shared,
    )
    library = _Library()
    spec = RuntimeSpec.model_validate({"name": "m", "engine": "llama_cpp", "modelPath": "/m.gguf"})
    await check_admission(spec, snapshot=_snapshot(device), library=library, running=[])
    [call] = library.calls
    if shared:
        assert call["unified_memory"] is True
        assert call["ram_bytes"] is None
    else:
        assert call["unified_memory"] is False
        assert call["ram_bytes"] == 26 * GIB


def test_the_engines_list_offers_the_other_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GET /v1/engines` is where a UI finds the menu, so it is asserted
    there and not only on the helper that builds it."""
    from eugene_plexus_agent import runtimes as runtimes_module

    monkeypatch.setattr(runtimes_module, "detect_host", lambda: _WINDOWS_X64)
    monkeypatch.setattr(
        "eugene_plexus_agent.engines.acquisition.GitHubReleases.list_releases",
        lambda self, **_: [_release()],
    )
    engines = {e.engine.value: e for e in runtimes_module.describe_engines()}
    acquisition = engines["llama_cpp"].acquisition
    assert acquisition is not None
    assert acquisition.variant == "win-vulkan-x64"
    assert acquisition.alternatives == alternatives(_WINDOWS_X64, _release())
