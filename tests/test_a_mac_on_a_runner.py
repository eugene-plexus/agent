"""A4: what a Mac on a GitHub runner showed, pinned where it is decided.

Measured 2026-09-30 on GitHub's macOS runners (docs/acceptance/
a4-macos-runner-run.md in specs), against the pins of that day:

* The Apple budget was `0.75 * hw.memsize`. Metal's own
  `recommendedMaxWorkingSetSize` is two thirds of RAM there (5,010,800,640
  of 7,516,192,768 bytes), matching MLX's `mx.device_info()` exactly, so the
  agent said a model fits that Metal will not hold. The budget is Metal's
  number now; the fraction is only the fallback, and it is two thirds.
* The Metal device was named `arm`, which is `platform.processor()` on
  Apple silicon.
* The MLX recipe began with a bare `uv`, and `install.sh` never puts uv on
  PATH, so the command the agent offered answered `uv: command not found`
  on the machine that had just installed Eugene with uv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eugene_plexus_agent._generated.models import Arch, ComputeDeviceKind, HostAccelerator, Os
from eugene_plexus_agent.engines import devices, gpu_probe, mlx
from eugene_plexus_agent.engines.mlx import MlxAdapter

GIB = 1024**3
RAM = 7_516_192_768
WORKING_SET = 5_010_800_640


def _metal(name: str | None = "Apple M2 Pro") -> gpu_probe.MetalDevice:
    return gpu_probe.MetalDevice(name=name, working_set_bytes=WORKING_SET, unified_memory=True)


def test_the_budget_is_metals_working_set_not_a_share_of_ram() -> None:
    warnings: list[str] = []
    [device] = devices._apple(RAM, warnings, metal=_metal)
    assert device.kind is ComputeDeviceKind.metal
    assert device.memoryTotalBytes == WORKING_SET
    assert device.memoryTotalBytes != int(RAM * 0.75)
    assert device.sharedMemory is True
    # Free is still not something Metal says for the whole system.
    assert device.memoryFreeBytes is None
    # A measured number needs no apology.
    assert warnings == []


def test_the_device_is_named_by_metal() -> None:
    [device] = devices._apple(RAM, [], metal=_metal)
    assert device.name == "Apple M2 Pro"


def test_with_no_metal_it_falls_back_to_two_thirds_and_says_so(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(devices, "_apple_chip", lambda run=None: "Apple M1 (Virtual)")
    warnings: list[str] = []
    [device] = devices._apple(RAM, warnings, metal=lambda: None)
    assert device.memoryTotalBytes == int(RAM * 2 / 3)
    assert device.name == "Apple M1 (Virtual)"
    assert any("Metal could not be asked" in w for w in warnings), warnings


def test_the_fallback_fraction_is_what_metal_reported() -> None:
    # 5,010,800,640 / 7,516,192,768 and 10,021,601,280 / 15,032,385,536:
    # both two thirds to within a few KB on the runners measured.
    assert abs(devices.APPLE_WIRED_LIMIT_FRACTION - WORKING_SET / RAM) < 1e-5


def test_metal_is_not_asked_off_a_mac() -> None:
    if sys.platform == "darwin":
        pytest.skip("this is the off-Mac half")
    assert gpu_probe.metal_device() is None


def test_the_chip_name_comes_from_sysctl() -> None:
    def run(argv: list[str]) -> str | None:
        return "Apple M2 Pro\n" if argv[:2] == ["sysctl", "-n"] else None

    assert devices._apple_chip(run) == "Apple M2 Pro"


# --------------------------------------------------------------------------- #
# the recipe
# --------------------------------------------------------------------------- #

APPLE = HostAccelerator(os=Os.macos, arch=Arch.arm64)


def test_the_recipe_names_the_installs_own_uv_by_path(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    prefix = tmp_path / "eugene plexus"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "uv").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", str(prefix / "venv"))
    monkeypatch.setattr(mlx.shutil, "which", lambda name: None)
    command = MlxAdapter().manual_install(APPLE).command or ""
    uv = mlx.shlex.quote(str(prefix / "bin" / "uv"))
    assert command.startswith(f"{uv} venv "), command
    assert f"&& {uv} pip install " in command, command


def test_the_recipe_asks_for_a_native_arm64_python() -> None:
    command = MlxAdapter().manual_install(APPLE).command or ""
    assert "--python cpython-3.12-macos-aarch64-none" in command, command
    assert command.index("--python cpython-3.12") < command.index("~/eugene-mlx &&"), command


def test_the_recipe_falls_back_to_uv_on_path_then_to_the_word(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(mlx.shutil, "which", lambda name: "/opt/homebrew/bin/uv")
    assert (MlxAdapter().manual_install(APPLE).command or "").startswith(
        "/opt/homebrew/bin/uv venv "
    )
    monkeypatch.setattr(mlx.shutil, "which", lambda name: None)
    assert (MlxAdapter().manual_install(APPLE).command or "").startswith("uv venv ")
