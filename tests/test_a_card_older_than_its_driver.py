"""A card older than its driver: choosing a CUDA build by the card too.

**The finding (2026-09-23).** `_cuda_variant` chose a build from the
CUDA version the DRIVER reports and nothing else. From CUDA 13 the
toolkit no longer compiles for Maxwell, Pascal or Volta, and upstream
llama.cpp's `ggml-cuda/CMakeLists.txt` adds `50-virtual 61-virtual
70-virtual` only below 13 -- while the last driver branch those cards
get (580) reports CUDA 13.0. So a Pascal card on that driver was handed
`ubuntu-cuda-13.3-x64`, the one build with no kernels for it, and the
engine died at model load naming nothing we did.

The same rule that refused to cross a major "either way" also told the
owner of a 13.x driver facing a release with only a 12.x build to
*update* a driver that was already newer. A driver runs builds from
older CUDA majors; only a newer major needs a newer driver.

Found by asking what the container would do on an Unraid box with a
Quadro P4000 -- a card the product had never been pointed at.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime

import pytest

from eugene_plexus_agent._generated.models import Accelerator, Arch, HostAccelerator, Os
from eugene_plexus_agent.engines import host as host_mod
from eugene_plexus_agent.engines.acquisition import Release, ReleaseAsset, Unavailable
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter

# b11010's Linux CUDA assets, verbatim: one 12.x build and one 13.x.
_LINUX = [
    "cudart-llama-b11010-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "cudart-llama-b11010-bin-ubuntu-cuda-13.3-x64.tar.gz",
    "llama-b11010-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b11010-bin-ubuntu-cuda-13.3-x64.tar.gz",
    "llama-b11010-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b11010-bin-ubuntu-x64.tar.gz",
]

# b10990's Windows CUDA assets, verbatim: 12.4 and 13.4.
_WINDOWS = [
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
    "llama-b10990-bin-win-cpu-x64.zip",
    "llama-b10990-bin-win-cuda-12.4-x64.zip",
    "llama-b10990-bin-win-cuda-13.4-x64.zip",
    "llama-b10990-bin-win-vulkan-x64.zip",
]


def _release(names: list[str], version: str = "b11010") -> Release:
    return Release(
        version=version,
        published_at=datetime(2026, 9, 16, 12, 0, tzinfo=UTC),
        assets=tuple(
            ReleaseAsset(
                name=name,
                url=f"https://example.invalid/{version}/{name}",
                size=1024,
                digest="sha256:" + "0" * 64,
            )
            for name in names
        ),
    )


def _plan(driver: str | None, cc: str | None, names: list[str] = _LINUX, os_kind: Os = Os.linux):
    host = HostAccelerator(
        os=os_kind,
        arch=Arch.x64,
        accelerator=Accelerator.cuda,
        acceleratorVersion=driver,
        computeCapability=cc,
    )
    version = "b10990" if names is _WINDOWS else "b11010"
    return LlamaCppAdapter().plan_acquisition(host, _release(names, version))


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_pascal_on_the_580_driver_gets_the_12x_build() -> None:
    """**The defect.** 580 is the last driver branch for Pascal and it
    reports CUDA 13.0. Before the card was consulted this took the 13.3
    build under minor-version compatibility -- a build carrying no code
    for compute capability 6.1."""
    plan = _plan("13.0", "6.1")
    assert not isinstance(plan, Unavailable), getattr(plan, "reason", "")
    assert plan.variant == "ubuntu-cuda-12.8-x64"
    # And its own runtime comes with it: that is what lets a 12.x build
    # load on a 13.x driver at all.
    assert sorted(a.name for a in plan.assets) == [
        "cudart-llama-b11010-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "llama-b11010-bin-ubuntu-cuda-12.8-x64.tar.gz",
    ]


def test_pascal_on_a_12x_driver_keeps_the_12x_build() -> None:
    """The Unraid box that prompted this: a Quadro P4000 on 575.51.02,
    which reports CUDA 12.9. It was right before and must stay right."""
    plan = _plan("12.9", "6.1")
    assert not isinstance(plan, Unavailable), getattr(plan, "reason", "")
    assert plan.variant == "ubuntu-cuda-12.8-x64"


def test_a_current_card_still_gets_its_drivers_own_major() -> None:
    """The over-correction to rule out: a 5090 on a 13.3 driver must
    not be pushed down to 12.x because 12.x also runs on it."""
    plan = _plan("13.3", "12.0")
    assert not isinstance(plan, Unavailable), getattr(plan, "reason", "")
    assert plan.variant == "ubuntu-cuda-13.3-x64"


def test_turing_is_the_first_card_a_13x_build_serves() -> None:
    """7.5 is the boundary, inclusive. Volta (7.0) is below it."""
    assert _plan("13.0", "7.5").variant == "ubuntu-cuda-13.3-x64"
    assert _plan("13.0", "7.0").variant == "ubuntu-cuda-12.8-x64"


def test_a_newer_driver_takes_an_older_major_when_its_own_is_not_published() -> None:
    """Backward compatibility, without any card involved. A 3090 on a
    13.x driver facing a release that ships 12.x only used to be told to
    *update the NVIDIA driver* -- which was already the newer of the two."""
    names = [n for n in _LINUX if "13.3" not in n]
    plan = _plan("13.0", "8.6", names)
    assert not isinstance(plan, Unavailable), getattr(plan, "reason", "")
    assert plan.variant == "ubuntu-cuda-12.8-x64"


def test_an_unknown_capability_filters_nothing() -> None:
    """A driver that would not say keeps the old behaviour exactly."""
    assert _plan("13.0", None).variant == "ubuntu-cuda-13.3-x64"
    assert _plan("12.9", None).variant == "ubuntu-cuda-12.8-x64"


def test_pascal_facing_a_13x_only_release_is_told_why() -> None:
    """Nothing it can run is published. The reason names the card, not
    the driver -- updating the driver would not help -- and stays
    release-bound, because the 12.x asset may simply not be uploaded
    yet and `plan_latest` should look at the build before."""
    names = [n for n in _LINUX if "12.8" not in n]
    plan = _plan("13.0", "6.1", names)
    assert isinstance(plan, Unavailable)
    assert "compute capability 6.1" in plan.reason
    assert "below 7.5" in plan.reason
    assert "Update the NVIDIA driver" not in plan.reason
    assert plan.release_bound is True


def test_a_card_nothing_published_supports_is_not_release_bound() -> None:
    """Below 5.0 (Kepler) no release will ever help, so stepping back
    through eight older builds would only delay the same answer."""
    plan = _plan("12.9", "3.7")
    assert isinstance(plan, Unavailable)
    assert "compute capability 3.7" in plan.reason
    assert "5.0" in plan.reason
    assert plan.release_bound is False


def test_a_newer_major_is_still_never_taken() -> None:
    """The half of the old rule that was right. An 11.8 driver cannot
    load a 12.x build, and the reason says to update it."""
    plan = _plan("11.8", "8.6")
    assert isinstance(plan, Unavailable)
    assert "Update the NVIDIA driver" in plan.reason
    assert "at least as new as its own major" in plan.reason


def test_windows_follows_the_same_rule() -> None:
    """Upstream's Windows 12.4 build uses the same CMake defaults, so
    the same floor applies: Pascal on 580 gets 12.4, not 13.4."""
    plan = _plan("13.0", "6.1", _WINDOWS, Os.windows)
    assert not isinstance(plan, Unavailable), getattr(plan, "reason", "")
    assert plan.variant == "win-cuda-12.4-x64"


def test_choosing_an_older_major_is_logged_with_its_reason(caplog) -> None:
    """A build older than the driver is a fact worth having in the log
    if the engine ever fails to start, and the reason distinguishes the
    card from upstream's publishing."""
    with caplog.at_level(logging.INFO, logger="eugene_plexus_agent.engines.llama_cpp"):
        _plan("13.0", "6.1")
    assert any("compute capability 6.1" in r.getMessage() for r in caplog.records)

    caplog.clear()
    names = [n for n in _LINUX if "13.3" not in n]
    with caplog.at_level(logging.INFO, logger="eugene_plexus_agent.engines.llama_cpp"):
        _plan("13.0", "8.6", names)
    assert any("publishes no CUDA 13 build" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def _fake_nvidia(monkeypatch, *, banner: str, caps: str, caps_exit: int = 0) -> None:
    monkeypatch.setattr(host_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host_mod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host_mod.shutil, "which", lambda name: "/usr/bin/" + name)

    def _run(argv, **kwargs):
        if "--query-gpu=compute_cap" in argv:
            return subprocess.CompletedProcess(argv, caps_exit, caps, "")
        if "--query-gpu=name" in argv:
            return subprocess.CompletedProcess(argv, 0, "Quadro P4000\n", "")
        return subprocess.CompletedProcess(argv, 0, banner, "")

    monkeypatch.setattr(host_mod.subprocess, "run", _run)


_BANNER_580 = "| NVIDIA-SMI 580.82.07   Driver Version: 580.82.07   CUDA Version: 13.0 |"


def test_the_lowest_card_is_reported(monkeypatch) -> None:
    """A 3090 beside a P4000: the build must carry code for both, so
    the P4000 decides. Order in nvidia-smi's output must not matter."""
    _fake_nvidia(monkeypatch, banner=_BANNER_580, caps="8.6\n6.1\n")
    detected = host_mod.detect_host()
    assert detected.accelerator is Accelerator.cuda
    assert detected.acceleratorVersion == "13.0"
    assert detected.computeCapability == "6.1"


def test_capability_compares_as_numbers_not_text(monkeypatch) -> None:
    """`"12.0" < "8.6"` as strings. A 5090 beside a 3090 is 8.6."""
    _fake_nvidia(monkeypatch, banner=_BANNER_580, caps="12.0\n8.6\n")
    assert host_mod.detect_host().computeCapability == "8.6"


def test_an_unreadable_row_is_skipped(monkeypatch) -> None:
    """A MIG slice reports `[N/A]`. It is not a card we can reason
    about, and it must not hide the real one beside it."""
    _fake_nvidia(monkeypatch, banner=_BANNER_580, caps="[N/A]\n7.5\n")
    assert host_mod.detect_host().computeCapability == "7.5"


def test_a_driver_that_will_not_say_is_still_a_cuda_host(monkeypatch, caplog) -> None:
    """An old driver rejects `compute_cap` and exits non-zero. That is
    *it would not say*, never *no NVIDIA card here*: the accelerator
    stays `cuda`, and the warning that means a broken driver is not
    logged, because it would be a false statement about the machine."""
    _fake_nvidia(
        monkeypatch,
        banner=_BANNER_580,
        caps='Field "compute_cap" is not a valid field to query.\n',
        caps_exit=2,
    )
    with caplog.at_level(logging.WARNING, logger="eugene_plexus_agent.engines.host"):
        detected = host_mod.detect_host()
    assert detected.accelerator is Accelerator.cuda
    assert detected.acceleratorVersion == "13.0"
    assert detected.computeCapability is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_a_machine_without_nvidia_asks_nothing(monkeypatch, system: str) -> None:
    """The probe runs only on a CUDA host: it is one more subprocess
    behind `GET /v1/engines`, and it has no answer anywhere else."""
    monkeypatch.setattr(host_mod.platform, "system", lambda: system)
    monkeypatch.setattr(host_mod.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(host_mod.Path, "is_dir", lambda self: False)
    monkeypatch.setattr(host_mod, "_windows_display_adapters", lambda: [])
    asked: list[list[str]] = []
    monkeypatch.setattr(host_mod, "_output", lambda argv, **kw: asked.append(argv))
    detected = host_mod.detect_host()
    assert detected.computeCapability is None
    assert not any("--query-gpu=compute_cap" in argv for argv in asked)
