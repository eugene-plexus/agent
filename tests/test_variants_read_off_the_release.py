"""Three build choices that assumed what a release publishes (drift audit, 2026-10-03).

- **The ROCm version was a constant**, `rocm-10.0`, while the CUDA versions
  have been read off the release's own asset names since a table of them
  went stale in a day. Upstream bakes the ROCm version into the name too
  (`release.yml`'s `ROCM_VERSION_SHORT`), so the day it moves, every AMD
  host would read "not installable" for a build that is there.
- **A Windows arm64 machine could be asked for `+vulkan`**, the CUDA build
  with the Vulkan backend added, while upstream publishes no
  `win-vulkan-arm64` -- so the plan failed on an asset that will never
  exist. Detection only reports a second vendor's card on Windows x64
  (`gpu_probe.combinable_with_cuda`), and the build choice now applies the
  same rule rather than trusting it was applied upstream of it.
- **`linux-arm64-snapdragon` was never offered**, because the expert's menu
  only listed `ubuntu-` builds on Linux. It is offered now, as an
  alternative; it is never chosen by default (it is the OpenCL-Adreno and
  Hexagon build, which needs Qualcomm's own drivers that nothing here
  detects).
"""

from __future__ import annotations

from datetime import UTC, datetime

from eugene_plexus_agent._generated.models import (
    Accelerator,
    Arch,
    HostAccelerator,
    Os,
    Secondary,
)
from eugene_plexus_agent.engines.acquisition import (
    AcquisitionPlan,
    Release,
    ReleaseAsset,
    Unavailable,
)
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter, alternatives

# b11376's assets (2026-10-03), verbatim, less the xcframework and UI.
_B11376 = [
    "cudart-llama-b11376-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "cudart-llama-b11376-bin-ubuntu-cuda-13.4-arm64.tar.gz",
    "cudart-llama-b11376-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
    "llama-b11376-bin-android-arm64-snapdragon.tar.gz",
    "llama-b11376-bin-android-arm64.tar.gz",
    "llama-b11376-bin-linux-arm64-snapdragon.tar.gz",
    "llama-b11376-bin-macos-arm64.tar.gz",
    "llama-b11376-bin-macos-x64.tar.gz",
    "llama-b11376-bin-ubuntu-arm64.tar.gz",
    "llama-b11376-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b11376-bin-ubuntu-cuda-13.4-arm64.tar.gz",
    "llama-b11376-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-b11376-bin-ubuntu-openvino-2026.4.1-x64.tar.gz",
    "llama-b11376-bin-ubuntu-rocm-10.0-x64.tar.gz",
    "llama-b11376-bin-ubuntu-s390x.tar.gz",
    "llama-b11376-bin-ubuntu-sycl-fp16-x64.tar.gz",
    "llama-b11376-bin-ubuntu-sycl-fp32-x64.tar.gz",
    "llama-b11376-bin-ubuntu-vulkan-arm64.tar.gz",
    "llama-b11376-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b11376-bin-ubuntu-x64.tar.gz",
    "llama-b11376-bin-win-cpu-arm64.zip",
    "llama-b11376-bin-win-cpu-x64.zip",
    "llama-b11376-bin-win-cuda-12.4-x64.zip",
    "llama-b11376-bin-win-cuda-13.4-arm64.zip",
    "llama-b11376-bin-win-cuda-13.4-x64.zip",
    "llama-b11376-bin-win-opencl-adreno-arm64.zip",
    "llama-b11376-bin-win-openvino-2026.4.1-x64.zip",
    "llama-b11376-bin-win-rocm-10.0-x64.zip",
    "llama-b11376-bin-win-sycl-x64.zip",
    "llama-b11376-bin-win-vulkan-x64.zip",
]


def _release(names: list[str]) -> Release:
    return Release(
        version="b11376",
        published_at=datetime(2026, 10, 3, 13, 25, tzinfo=UTC),
        assets=tuple(
            ReleaseAsset(
                name=n, url=f"https://example.invalid/{n}", size=1, digest="sha256:" + "0" * 64
            )
            for n in names
        ),
    )


def _rocm_moved(names: list[str], to: str) -> list[str]:
    return [n.replace("rocm-10.0", f"rocm-{to}") for n in names]


# --- ROCm ----------------------------------------------------------------


def test_the_rocm_version_is_read_off_the_release() -> None:
    names = _rocm_moved(_B11376, "11.1")
    for os_kind, expected in (
        (Os.linux, "ubuntu-rocm-11.1-x64"),
        (Os.windows, "win-rocm-11.1-x64"),
    ):
        plan = LlamaCppAdapter().plan_acquisition(
            HostAccelerator(os=os_kind, arch=Arch.x64, accelerator=Accelerator.rocm),
            _release(names),
        )
        assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
        assert plan.variant == expected


def test_todays_rocm_build_is_still_chosen() -> None:
    plan = LlamaCppAdapter().plan_acquisition(
        HostAccelerator(os=Os.linux, arch=Arch.x64, accelerator=Accelerator.rocm),
        _release(_B11376),
    )
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert plan.variant == "ubuntu-rocm-10.0-x64"


def test_of_two_rocm_versions_the_newest_is_taken() -> None:
    names = [*_B11376, "llama-b11376-bin-ubuntu-rocm-11.1-x64.tar.gz"]
    plan = LlamaCppAdapter().plan_acquisition(
        HostAccelerator(os=Os.linux, arch=Arch.x64, accelerator=Accelerator.rocm),
        _release(names),
    )
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert plan.variant == "ubuntu-rocm-11.1-x64"


def test_versions_compare_as_numbers() -> None:
    names = [
        *[n for n in _B11376 if "rocm" not in n],
        "llama-b11376-bin-ubuntu-rocm-9.4-x64.tar.gz",
        "llama-b11376-bin-ubuntu-rocm-10.0-x64.tar.gz",
    ]
    plan = LlamaCppAdapter().plan_acquisition(
        HostAccelerator(os=Os.linux, arch=Arch.x64, accelerator=Accelerator.rocm),
        _release(names),
    )
    assert isinstance(plan, AcquisitionPlan)
    assert plan.variant == "ubuntu-rocm-10.0-x64"


def test_a_release_with_no_rocm_build_says_so_and_may_be_mid_upload() -> None:
    names = [n for n in _B11376 if "rocm" not in n]
    plan = LlamaCppAdapter().plan_acquisition(
        HostAccelerator(os=Os.linux, arch=Arch.x64, accelerator=Accelerator.rocm),
        _release(names),
    )
    assert isinstance(plan, Unavailable)
    assert plan.release_bound
    assert "ROCm" in plan.reason and "b11376" in plan.reason
    assert "ubuntu-vulkan-x64" in plan.reason, "it lists what is published"


# --- Windows arm64 and the Vulkan backend --------------------------------


def test_windows_arm64_is_not_asked_for_a_vulkan_build_that_does_not_exist() -> None:
    plan = LlamaCppAdapter().plan_acquisition(
        HostAccelerator(
            os=Os.windows,
            arch=Arch.arm64,
            accelerator=Accelerator.cuda,
            acceleratorVersion="13.4",
            computeCapability="12.0",
            secondary=Secondary.vulkan,
        ),
        _release(_B11376),
    )
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert plan.variant == "win-cuda-13.4-arm64"
    assert plan.plugins == frozenset()


def test_windows_x64_still_gets_the_vulkan_backend_added() -> None:
    plan = LlamaCppAdapter().plan_acquisition(
        HostAccelerator(
            os=Os.windows,
            arch=Arch.x64,
            accelerator=Accelerator.cuda,
            acceleratorVersion="13.4",
            computeCapability="12.0",
            secondary=Secondary.vulkan,
        ),
        _release(_B11376),
    )
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert plan.variant == "win-cuda-13.4-x64+vulkan"


# --- Snapdragon on Linux --------------------------------------------------


def test_linux_arm64_offers_the_snapdragon_build_as_an_alternative() -> None:
    host = HostAccelerator(os=Os.linux, arch=Arch.arm64, accelerator=Accelerator.none)
    offered = alternatives(host, _release(_B11376))
    assert "linux-arm64-snapdragon" in offered
    assert "android-arm64-snapdragon" not in offered, "an Android build is not a Linux one"
    assert "ubuntu-arm64" in offered


def test_the_snapdragon_build_installs_when_chosen() -> None:
    host = HostAccelerator(os=Os.linux, arch=Arch.arm64, accelerator=Accelerator.none)
    plan = LlamaCppAdapter().plan_acquisition(
        host, _release(_B11376), variant="linux-arm64-snapdragon"
    )
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert [a.name for a in plan.assets] == ["llama-b11376-bin-linux-arm64-snapdragon.tar.gz"]


def test_the_snapdragon_build_is_never_the_default() -> None:
    host = HostAccelerator(os=Os.linux, arch=Arch.arm64, accelerator=Accelerator.none)
    plan = LlamaCppAdapter().plan_acquisition(host, _release(_B11376))
    assert isinstance(plan, AcquisitionPlan)
    assert plan.variant == "ubuntu-arm64"


def test_an_x64_linux_host_is_not_offered_an_arm64_build() -> None:
    host = HostAccelerator(os=Os.linux, arch=Arch.x64, accelerator=Accelerator.none)
    assert "linux-arm64-snapdragon" not in alternatives(host, _release(_B11376))
