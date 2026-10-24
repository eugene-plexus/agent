"""Device detection: nvidia-smi's CSV into `ComputeDevice`, plus the CPU.

Only the NVIDIA path has met hardware. The others are exercised here
against their tools' documented output, which is all a fixture can do,
and each failure has to surface as a warning rather than a size.
"""

from __future__ import annotations

from eugene_plexus_agent._generated.models import ComputeDeviceKind
from eugene_plexus_agent.engines.devices import MIB, detect_devices


def test_nvml_readout_fills_link_power_and_uuid(monkeypatch) -> None:
    import ctypes

    from eugene_plexus_agent.host_probes import Nvml

    class Library:
        def nvmlInit_v2(self):
            return 0

        def nvmlShutdown(self):
            return 0

        def nvmlDeviceGetHandleByIndex_v2(self, index, pointer):
            pointer._obj.value = index.value + 1
            return 0

        def nvmlDeviceGetUUID(self, handle, buffer, size):
            buffer.value = f"GPU-{handle.value}".encode()
            return 0

        def nvmlDeviceGetPciInfo_v3(self, handle, pointer):
            pointer._obj.busId = b"00000000:01:00.0"
            return 0

        def nvmlDeviceGetEccMode(self, handle, current, pending):
            current._obj.value = pending._obj.value = 1
            return 0

        def nvmlDeviceGetTotalEccErrors(self, handle, error, counter, pointer):
            assert (error, counter) == (1, 0)
            pointer._obj.value = 3
            return 0

        def nvmlDeviceGetP2PStatus(self, handle, peer, capability, pointer):
            assert capability == 0
            pointer._obj.value = 0
            return 0

        def __getattr__(self, name):
            numbers = {
                "nvmlDeviceGetCurrPcieLinkGeneration": 4,
                "nvmlDeviceGetCurrPcieLinkWidth": 4,
                "nvmlDeviceGetMaxPcieLinkGeneration": 4,
                "nvmlDeviceGetMaxPcieLinkWidth": 16,
                "nvmlDeviceGetEnforcedPowerLimit": 300000,
                "nvmlDeviceGetPowerManagementDefaultLimit": 450000,
                "nvmlDeviceGetCurrentClocksThrottleReasons": 0x6C,
            }
            if name not in numbers:
                raise AttributeError(name)

            def read(handle, pointer):
                pointer._obj.value = numbers[name]
                return 0

            return read

    monkeypatch.setattr(ctypes, "CDLL", lambda _: Library())
    snapshot = detect_devices(
        run=_runner({"nvidia-smi": "0, RTX, 24576, 24000\n1, RTX, 24576, 24000"}),
        memory=_memory,
        os_adapters=lambda: [],
    )
    gpu = snapshot.accelerators()[0]
    assert gpu.uuid == "GPU-1"
    assert gpu.pcieLink.width == 4
    assert gpu.pcieLink.maxWidth == 16
    assert gpu.powerLimitWatts == 300
    assert gpu.powerDefaultLimitWatts == 450
    assert {r.value for r in gpu.throttleReasons} == {"thermal", "power-cap", "hardware-slowdown"}
    assert gpu.pciBusId == "00000000:01:00.0"
    assert gpu.uncorrectedMemoryErrors == 3
    assert [peer.root for peer in gpu.peers] == ["GPU-2"]
    assert Nvml().loaded


def test_unreadable_fields_are_null(monkeypatch) -> None:
    import ctypes

    def unavailable(_):
        raise OSError("driver unavailable")

    monkeypatch.setattr(ctypes, "CDLL", unavailable)
    gpu = detect_devices(
        run=_runner({"nvidia-smi": "0, RTX, 24576, 24000"}), memory=_memory, os_adapters=lambda: []
    ).accelerators()[0]
    assert gpu.uuid is None
    assert gpu.pcieLink is None
    assert gpu.powerLimitWatts is None
    assert gpu.peers is None


def test_sysfs_readout_keeps_unavailable_values_null(tmp_path) -> None:
    from eugene_plexus_agent.device_health import sysfs_health

    for name, value in {
        "unique_id": "abc123",
        "current_link_speed": "16.0 GT/s PCIe",
        "current_link_width": "4",
        "max_link_speed": "16.0 GT/s PCIe",
        "max_link_width": "16",
    }.items():
        (tmp_path / name).write_text(value)
    hwmon = tmp_path / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "power1_cap").write_text("220000000")
    health = sysfs_health(tmp_path)
    assert health["uuid"] == "abc123"
    assert health["pcieLink"]["generation"] == 4
    assert health["pcieLink"]["maxWidth"] == 16
    assert health["powerLimitWatts"] == 220
    assert health["powerDefaultLimitWatts"] is None
    assert "peers" not in health


def test_history_reopens_nvml_after_cards_renumber():
    from types import SimpleNamespace

    from eugene_plexus_agent.host_probes import HostProbe, RuntimeCounts

    from .conftest import fake_devices

    now = [0.0]
    opened = []

    def nvml():
        instance = SimpleNamespace(close=lambda: None)
        opened.append(instance)
        return instance

    probe = HostProbe(
        devices=fake_devices,
        adapters=lambda: [],
        runtimes=lambda: RuntimeCounts(0, 0, ()),
        folders=lambda: [],
        nvml=nvml,
        clock=lambda: now[0],
        platform="linux",
    )
    try:
        probe._refresh_inventory()
        now[0] = 601
        probe._refresh_inventory()
        assert len(opened) == 2
    finally:
        probe.close()


def _runner(answers: dict[str, str | None]):  # type: ignore[no-untyped-def]
    def run(argv: list[str]) -> str | None:
        return answers.get(argv[0])

    return run


def _memory() -> tuple[int | None, int | None]:
    return 96 * 1024**3, 60 * 1024**3


def test_nvidia_rows_become_cuda_devices_in_bytes() -> None:
    snapshot = detect_devices(
        run=_runner(
            {
                "nvidia-smi": (
                    "0, NVIDIA GeForce RTX 5090, 32607, 30331\n"
                    "1, NVIDIA GeForce RTX 3090, 24576, 24000\n"
                )
            }
        ),
        memory=_memory,
    )
    accelerators = snapshot.accelerators()
    assert [d.index for d in accelerators] == [0, 1]
    assert accelerators[0].kind is ComputeDeviceKind.cuda
    assert accelerators[0].name == "NVIDIA GeForce RTX 5090"
    # MiB in, bytes out — nvidia-smi's unit is not negotiable.
    assert accelerators[0].memoryTotalBytes == 32607 * MIB
    assert accelerators[0].memoryFreeBytes == 30331 * MIB
    assert snapshot.warnings == ()


def test_the_cpu_is_always_last_and_carries_host_memory() -> None:
    snapshot = detect_devices(run=_runner({}), memory=_memory)
    cpu = snapshot.cpu()
    assert cpu is not None
    assert snapshot.devices[-1] is cpu
    assert cpu.memoryTotalBytes == 96 * 1024**3
    assert cpu.memoryFreeBytes == 60 * 1024**3
    assert snapshot.accelerators() == []


def test_a_garbled_nvidia_row_is_a_warning_not_a_device() -> None:
    snapshot = detect_devices(
        run=_runner({"nvidia-smi": "0, RTX, not-a-number, 12\n1, Good, 1024, 512\n"}),
        memory=_memory,
    )
    assert [d.index for d in snapshot.accelerators()] == [1]
    assert any("could not parse" in w for w in snapshot.warnings)


def test_rocm_csv_is_read_and_marked_unverified() -> None:
    snapshot = detect_devices(
        run=_runner(
            {
                "rocm-smi": (
                    "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
                    "card0,17163091968,1073741824\n"
                )
            }
        ),
        memory=_memory,
    )
    [gpu] = snapshot.accelerators()
    assert gpu.kind is ComputeDeviceKind.rocm
    assert gpu.memoryTotalBytes == 17163091968
    assert gpu.memoryFreeBytes == 17163091968 - 1073741824
    assert any("untested" in w for w in snapshot.warnings)


def test_intel_devices_carry_no_memory_and_say_so() -> None:
    snapshot = detect_devices(
        run=_runner({"xpu-smi": "Device ID,Device Name\n0,Intel Arc A770\n"}),
        memory=_memory,
    )
    [gpu] = snapshot.accelerators()
    assert gpu.kind is ComputeDeviceKind.xpu
    assert gpu.memoryFreeBytes is None
    assert any("unknown" in w for w in snapshot.warnings)


def test_unreadable_host_memory_is_a_warning() -> None:
    snapshot = detect_devices(run=_runner({}), memory=lambda: (None, None))
    assert snapshot.ram_total_bytes is None
    assert any("host memory" in w for w in snapshot.warnings)
