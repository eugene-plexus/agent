"""A newer CUDA minor than the driver's is safe only for a card with finished code in it.

**The finding (upstream drift audit, 2026-10-03; read in source, no such
card here).** When a release publishes no CUDA build at or below the
driver's minor, `_cuda_variant` takes the lowest newer minor of the same
major, under NVIDIA's minor-version compatibility. That compatibility has
one exclusion: **PTX compiled by a newer toolkit cannot be JIT-compiled by
an older driver.** Upstream's release builds pass no
`CMAKE_CUDA_ARCHITECTURES`, so `ggml/src/ggml-cuda/CMakeLists.txt`'s
default applies (b11375):

    75-virtual 80-virtual 86-real          always
    89-real 90-virtual                     CUDA >= 11.8
    120a-real                              CUDA >= 12.8
    121a-real                              CUDA >= 12.9
    50-virtual 61-virtual 70-virtual       CUDA < 13

`-real` is finished machine code (SASS); `-virtual` is PTX the driver
compiles at load. So in the published CUDA 13.4 build only compute
capability 8.6, 8.7, 8.9, 12.0 and 12.1 have finished code. A Turing
(7.5), an A100 (8.0) or an H100 (9.0) on a driver that supports CUDA 13.0
was handed the 13.4 build -- PTX its driver cannot compile -- when the CUDA
12 build beside it would have run. Upstream publishes only 13.4 for CUDA 13
now, so this is every such card on a 580 driver.

The rule: take a newer minor only where the card has finished code in that
build; otherwise the newest build from an older major; otherwise refuse
with the fix. A card whose capability is unknown keeps the old behaviour.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from eugene_plexus_agent._generated.models import Accelerator, Arch, HostAccelerator, Os
from eugene_plexus_agent.engines.acquisition import (
    AcquisitionPlan,
    Release,
    ReleaseAsset,
    Unavailable,
)
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter, has_finished_code

# b11376's CUDA assets, verbatim (2026-10-03): one 12.x and one 13.x
# build per platform, and 13.4 is the only CUDA 13 published.
_LINUX = [
    "cudart-llama-b11376-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "cudart-llama-b11376-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-b11376-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b11376-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-b11376-bin-ubuntu-x64.tar.gz",
]
_WINDOWS = [
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
    "llama-b11376-bin-win-cpu-x64.zip",
    "llama-b11376-bin-win-cuda-12.4-x64.zip",
    "llama-b11376-bin-win-cuda-13.4-x64.zip",
]


def _release(names: list[str]) -> Release:
    return Release(
        version="b11376",
        published_at=datetime(2026, 10, 3, 13, 25, tzinfo=UTC),
        assets=tuple(
            ReleaseAsset(
                name=n,
                url=f"https://example.invalid/b11376/{n}",
                size=1024,
                digest="sha256:" + "0" * 64,
            )
            for n in names
        ),
    )


def _plan(
    driver: str, cc: str | None, names: list[str] = _LINUX, os_kind: Os = Os.linux
) -> AcquisitionPlan | Unavailable:
    host = HostAccelerator(
        os=os_kind,
        arch=Arch.x64,
        accelerator=Accelerator.cuda,
        acceleratorVersion=driver,
        computeCapability=cc,
    )
    return LlamaCppAdapter().plan_acquisition(host, _release(names))


@pytest.mark.parametrize(
    ("cc", "card"),
    [
        ("8.6", "RTX 3090"),
        ("8.7", "Jetson Orin, covered by 86-real"),
        ("8.9", "RTX 4090"),
        ("12.0", "RTX 5090"),
        ("12.1", "GB10"),
    ],
)
def test_a_card_with_finished_code_takes_the_newer_minor(cc: str, card: str) -> None:
    plan = _plan("13.0", cc)
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert plan.variant == "ubuntu-cuda-13.4-x64", card


@pytest.mark.parametrize(
    ("cc", "card"),
    [
        ("7.5", "Turing"),
        ("8.0", "A100"),
        ("9.0", "H100"),
        ("10.0", "B200"),
        ("12.2", "a Blackwell 12.x with no 120a/121a code"),
    ],
)
def test_a_ptx_only_card_takes_the_older_major_instead(cc: str, card: str) -> None:
    plan = _plan("13.0", cc)
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert plan.variant == "ubuntu-cuda-12.8-x64", card


def test_windows_follows_the_same_rule() -> None:
    plan = _plan("13.0", "7.5", _WINDOWS, Os.windows)
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert plan.variant == "win-cuda-12.4-x64"
    assert isinstance(_plan("13.0", "12.0", _WINDOWS, Os.windows), AcquisitionPlan)
    assert _plan("13.0", "12.0", _WINDOWS, Os.windows).variant == "win-cuda-13.4-x64"  # type: ignore[union-attr]


def test_a_driver_at_the_builds_own_minor_compiles_the_ptx_itself() -> None:
    """The exclusion is about a NEWER toolkit than the driver. A 13.4
    driver compiles 13.4's PTX, so a Turing card there takes 13.4."""
    plan = _plan("13.4", "7.5")
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert plan.variant == "ubuntu-cuda-13.4-x64"


def test_no_older_major_is_a_refusal_naming_the_fix() -> None:
    names = [n for n in _LINUX if "12.8" not in n]
    plan = _plan("13.0", "8.0", names)
    assert isinstance(plan, Unavailable)
    assert plan.release_bound, "a 12.x build mid-upload would answer it"
    assert "8.0" in plan.reason
    assert "13.4" in plan.reason and "13.0" in plan.reason
    assert "Update the NVIDIA driver" in plan.reason


def test_an_unknown_capability_keeps_the_old_behaviour() -> None:
    """No `compute_cap` from the driver: nothing to decide on, so the
    newer minor is taken as before, and the code says so."""
    plan = _plan("13.0", None)
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert plan.variant == "ubuntu-cuda-13.4-x64"


@pytest.mark.parametrize(
    ("cc", "build", "expected"),
    [
        ((8, 6), (13, 4), True),
        ((8, 7), (13, 4), True),
        ((8, 9), (11, 8), True),
        ((8, 9), (11, 7), True),  # no 89-real before 11.8, but 86-real runs on 8.9
        ((8, 0), (13, 4), False),  # 86-real runs on 8.6 and up, not 8.0
        ((12, 0), (12, 8), True),
        ((12, 0), (12, 4), False),
        ((12, 1), (12, 8), False),  # 120a is arch-specific: runs on 12.0 alone
        ((12, 1), (12, 9), True),
        ((9, 0), (13, 4), False),
        ((7, 5), (12, 8), False),
    ],
)
def test_the_table_reads_upstreams_architecture_list(
    cc: tuple[int, int], build: tuple[int, int], expected: bool
) -> None:
    assert has_finished_code(cc, build) is expected
