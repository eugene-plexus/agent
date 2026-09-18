"""R2.3 — a card we cannot see is not a machine without one.

Roadmap `specs/docs/design/release-roadmap.md` §3.3. Findings: review
§6.1 #11 (any non-NVIDIA GPU on Windows gets a CPU-only llama.cpp,
silently) and §6.2 #29 (an `nvidia-smi` that exists and exits non-zero
is treated as a working one, so a wedged driver is reported as `cuda`
and the install plan asks for a CUDA build).

**What #11 actually is.** `_has_rocm` looks for `rocm-smi` or
`/opt/rocm`; `_has_sycl` looks for `sycl-ls` or Linux sysfs. None of the
four exists on Windows, and there was no Vulkan probe and no
`Win32_VideoController` query — so `_detect_accelerator` fell straight
through to `none` for every AMD and Intel card there is. A 7900 XTX
owner got a CPU build, *"no accelerator was detected"*, a fit scored
against RAM, the starter set inverted to the smallest model, and nothing
anywhere saying why it is slow. `win-rocm-10.0-x64` was reachable only
in principle, because `_has_rocm` can never be true on Windows — dead
code that reads as AMD support.

**Vulkan is the answer rather than ROCm**, for the reason that decides
it on a user's machine and not in a table: upstream ships
`llama-*-bin-win-vulkan-x64.zip` in every release, it needs no vendor
SDK, and it works for AMD *and* Intel *and* anything else with a modern
driver. ROCm on Windows needs the HIP SDK actually installed, so it is
taken only when that is on disk — which is what stops it being dead
code without betting the common case on it.

**No new dependency.** `pywin32` is on every Windows install regardless
of elevation, and `firewall/windows.py` already reads WMI through it at
a measured 112 ms.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

import pytest

from eugene_plexus_agent._generated.models import Accelerator, Arch, HostAccelerator, Os
from eugene_plexus_agent.engines import host as host_mod
from eugene_plexus_agent.engines.acquisition import Release, ReleaseAsset, Unavailable
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter


def _release(names: list[str]) -> Release:
    return Release(
        version="b10867",
        published_at=datetime(2026, 9, 8, 17, 31, tzinfo=UTC),
        assets=tuple(
            ReleaseAsset(
                name=name,
                url=f"https://example.invalid/{name}",
                size=1024,
                digest="sha256:" + "0" * 64,
            )
            for name in names
        ),
    )


_WINDOWS_ASSETS = [
    "llama-b10867-bin-win-cpu-x64.zip",
    "llama-b10867-bin-win-cuda-13.3-x64.zip",
    "llama-b10867-bin-win-rocm-10.0-x64.zip",
    "llama-b10867-bin-win-vulkan-x64.zip",
    "cudart-llama-bin-win-cuda-13.3-x64.zip",
]


@pytest.fixture
def windows(monkeypatch):
    """A Windows box with no vendor tool of any kind on PATH.

    Which is the ordinary state of a Windows machine with an AMD or
    Intel card: `rocm-smi` ships with the HIP SDK and `sycl-ls` with
    oneAPI, and a person who has installed neither still owns the GPU.
    """
    monkeypatch.setattr(host_mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(host_mod.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(host_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(host_mod, "_windows_hip_sdk_present", lambda: False)
    return monkeypatch


def _video_controllers(monkeypatch, names: list[str]) -> None:
    monkeypatch.setattr(host_mod, "_windows_display_adapters", lambda: names)


# --------------------------------------------------------------------------- #
# §6.1 #11 — the card nobody looked for
# --------------------------------------------------------------------------- #


def test_an_amd_card_on_windows_is_not_a_machine_without_a_gpu(windows) -> None:
    """**The finding.** Before this, `none` — and everything downstream
    followed: a CPU build, a fit scored against RAM, and a starter set
    that recommends the smallest model to somebody holding a 7900 XTX.
    """
    _video_controllers(windows, ["AMD Radeon RX 7900 XTX"])
    detected = host_mod.detect_host()
    assert detected.accelerator is Accelerator.vulkan, (
        f"an AMD card on Windows was detected as {detected.accelerator}"
    )


def test_an_intel_arc_on_windows_is_not_a_machine_without_a_gpu(windows) -> None:
    _video_controllers(windows, ["Intel(R) Arc(TM) A770 Graphics"])
    assert host_mod.detect_host().accelerator is Accelerator.vulkan


@pytest.mark.parametrize(
    "adapter",
    [
        "Microsoft Basic Display Adapter",
        "Microsoft Remote Display Adapter",
        "Microsoft Hyper-V Video",
        "Citrix Indirect Display Adapter",
    ],
)
def test_an_adapter_that_is_not_a_gpu_is_still_none(windows, adapter) -> None:
    """The negative case, and it has to be near the positive one.

    Every Windows machine has a display adapter; a Vulkan build on a box
    whose only one is Microsoft's basic renderer would be slower than
    the CPU build and would fail on some drivers. So the probe looks for
    a real GPU **vendor**, not for the presence of an adapter.

    **And the vendor allowlist is the whole filter.** An exclusion list
    beside it was unreachable -- a sabotage removed it and every one of
    these names was still rejected, because none of them contains a
    vendor word. It was deleted rather than kept; the parametrisation is
    what says so, by naming four adapters nobody should build for and
    letting the one surviving mechanism reject all of them.
    """
    _video_controllers(windows, [adapter])
    assert host_mod.detect_host().accelerator is Accelerator.none


def test_an_nvidia_card_on_windows_still_takes_the_cuda_path(windows) -> None:
    """The WMI probe must not overtake `nvidia-smi`. A CUDA build is
    materially faster than the Vulkan one on NVIDIA, and the driver's
    CUDA ceiling is what picks the asset."""
    windows.setattr(host_mod.shutil, "which", lambda name: "C:\\Windows\\" + name)

    def _smi(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="| NVIDIA-SMI 610.47   KMD Version: 610.47   CUDA UMD Version: 13.3 |",
            stderr="",
        )

    windows.setattr(host_mod.subprocess, "run", _smi)
    _video_controllers(windows, ["NVIDIA GeForce RTX 5090"])
    detected = host_mod.detect_host()
    assert detected.accelerator is Accelerator.cuda
    assert detected.acceleratorVersion == "13.3"


def test_rocm_on_windows_is_taken_only_when_the_hip_sdk_is_there(windows) -> None:
    """`win-rocm-10.0-x64` stops being dead code — and stops being a bet.

    `_has_rocm` could never be true on Windows, so the variant existed
    in the mapping and was unreachable. It is reachable now, through the
    one thing that says ROCm will actually load: the HIP SDK on disk.
    """
    _video_controllers(windows, ["AMD Radeon RX 7900 XTX"])
    assert host_mod.detect_host().accelerator is Accelerator.vulkan

    windows.setattr(host_mod, "_windows_hip_sdk_present", lambda: True)
    assert host_mod.detect_host().accelerator is Accelerator.rocm


def test_the_vulkan_variant_is_an_asset_upstream_actually_publishes() -> None:
    """A detector that names a build nobody ships is worse than no
    detector: it turns a slow install into a failed one. The fixture is
    verbatim from a real release."""
    plan = LlamaCppAdapter().plan_acquisition(
        HostAccelerator(os=Os.windows, arch=Arch.x64, accelerator=Accelerator.vulkan),
        _release(_WINDOWS_ASSETS),
    )
    assert not isinstance(plan, Unavailable), getattr(plan, "reason", "")
    assert plan.variant == "win-vulkan-x64"


def test_a_release_with_no_vulkan_asset_refuses_with_a_reason() -> None:
    """Rather than silently picking the CPU build, which is the failure
    shape this whole slice is about."""
    plan = LlamaCppAdapter().plan_acquisition(
        HostAccelerator(os=Os.windows, arch=Arch.x64, accelerator=Accelerator.vulkan),
        _release(["llama-b10867-bin-win-cpu-x64.zip"]),
    )
    assert isinstance(plan, Unavailable)
    assert "vulkan" in plan.reason.lower()


def test_the_probe_is_not_run_off_windows(monkeypatch) -> None:
    """`Win32_VideoController` does not exist on Linux, and a Linux box
    with an AMD card has `rocm-smi` and `/opt/rocm` to look at — which
    is the path that already worked."""
    monkeypatch.setattr(host_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host_mod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host_mod.shutil, "which", lambda name: None)
    called: list[int] = []
    monkeypatch.setattr(host_mod, "_windows_display_adapters", lambda: called.append(1) or [])
    monkeypatch.setattr(host_mod.Path, "is_dir", lambda self: False)
    assert host_mod.detect_host().accelerator is Accelerator.none
    assert called == [], "the Windows WMI probe ran on Linux"


# --------------------------------------------------------------------------- #
# §6.2 #29 — a tool that exists and fails
# --------------------------------------------------------------------------- #


def test_a_wedged_nvidia_driver_does_not_produce_a_cuda_install_plan(monkeypatch) -> None:
    """**The finding.** `_run` ignored the return code entirely, so
    `nvidia-smi` printing *Failed to initialize NVML* on stderr and
    exiting 9 came back as a non-empty string — which `_has_nvidia` read
    as a card and `_detect_accelerator` turned into `cuda`. The install
    plan then fetched a CUDA build that cannot load, for a reason the
    log does not mention.

    A driver/library version mismatch after an update is the commonest
    Linux failure there is.
    """
    monkeypatch.setattr(host_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host_mod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host_mod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(host_mod.Path, "is_dir", lambda self: False)

    def _wedged(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=9,
            stdout="",
            stderr="Failed to initialize NVML: Driver/library version mismatch",
        )

    monkeypatch.setattr(host_mod.subprocess, "run", _wedged)
    detected = host_mod.detect_host()
    assert detected.accelerator is not Accelerator.cuda, (
        "a wedged driver was reported as a working CUDA host"
    )
    assert detected.accelerator is Accelerator.none


def test_a_probe_distinguishes_absent_from_failed(monkeypatch) -> None:
    """One three-state result, which is why it belongs in `_run` rather
    than at each of the three call sites: `_has_nvidia`, `_has_rocm` and
    `_has_sycl` all need the same distinction and three copies is three
    chances to get one wrong."""
    monkeypatch.setattr(host_mod.shutil, "which", lambda name: None)
    assert host_mod._run(["nvidia-smi"]) is host_mod.Probe.ABSENT

    monkeypatch.setattr(host_mod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        host_mod.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "boom"),
    )
    assert host_mod._run(["nvidia-smi"]) is host_mod.Probe.FAILED

    monkeypatch.setattr(
        host_mod.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "GPU 0\n", ""),
    )
    got = host_mod._run(["nvidia-smi"])
    assert got not in (host_mod.Probe.ABSENT, host_mod.Probe.FAILED)
    assert "GPU 0" in str(got)


def test_a_working_nvidia_smi_with_an_unparseable_banner_is_still_cuda(monkeypatch) -> None:
    """The case that must NOT regress. An NVIDIA card whose banner we
    cannot parse is still an NVIDIA card — the difference between
    *"we refuse, here is why"* and *"here is a CPU build"*, and the
    former is the honest answer. Only a failed EXIT changes that."""
    monkeypatch.setattr(host_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host_mod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host_mod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        host_mod.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "NVIDIA GeForce RTX 5090\n", ""),
    )
    detected = host_mod.detect_host()
    assert detected.accelerator is Accelerator.cuda
    assert detected.acceleratorVersion is None
