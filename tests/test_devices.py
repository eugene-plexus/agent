"""Device detection: nvidia-smi's CSV into `ComputeDevice`, plus the CPU.

Only the NVIDIA path has met hardware. The others are exercised here
against their tools' documented output, which is all a fixture can do,
and each failure has to surface as a warning rather than a size.
"""

from __future__ import annotations

from eugene_plexus_agent._generated.models import ComputeDeviceKind
from eugene_plexus_agent.engines.devices import MIB, detect_devices


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
