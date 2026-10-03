"""vLLM install recipes that are true for the release they name (drift audit, 2026-10-03).

Each recipe was checked on 2026-10-03 against vLLM 0.30.0's GitHub release
assets and wheel indexes, HEAD-requested, and resolved with
`uv pip compile` for Linux and Python 3.12 (no install):

- **CUDA.** PyPI's wheel is built against CUDA 13.0 (`VLLM_MAIN_CUDA_VERSION`
  is "13.0" in `vllm/envs.py` at v0.30.0; it requires `[cu13]` extras; the
  release's alternative is `+cu129`), so it needs a driver from the R580
  branch. The recipe said 12.9 -- what upstream's install page still says.
  A CUDA 12 driver gets the `+cu129` wheel. Upstream's own command for it
  (`--extra-index-url https://download.pytorch.org/whl/cu129`) does NOT
  resolve under uv's default index strategy -- that index's `packaging` is
  older than flashinfer needs -- so the recipe uses `--torch-backend=cu129`,
  which resolved (torch 2.13.0+cu129).
- **ROCm.** The unversioned index serves the newest release, 0.31.0 on
  2026-10-03 -- ahead of PyPI and of what this adapter was checked
  against. The versioned index upstream documents
  (`rocm/0.30.0/rocm723`) resolved to `0.30.0+rocm723`, manylinux_2_39:
  glibc 2.39. The recipe said ROCm 7.0 and 7.2.1.
- **XPU.** The release has a versioned index (`0.30.0/xpu`), but its wheel
  pins `triton==3.7.2+xpu`, which only `wheels.vllm.ai/xpu` carries;
  without that index uv resolved `vllm==0.30.0` to PyPI's CUDA wheel, and
  with it to `0.30.0+xpu`. The recipe named the nightly index.
- **CPU.** The command took `${VLLM_VERSION}` with a fixed `manylinux_2_34`
  tag, which 404s for 0.30.0 (its CPU wheels are manylinux_2_39,
  vllm#58270; 0.31.0 returns to 2_34, vllm#58515). The recipe names 0.30.0
  and its real file, and 0.29.0's 2_34 wheel for an older glibc.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from eugene_plexus_agent._generated.models import Accelerator, Arch, HostAccelerator, Os
from eugene_plexus_agent.engines.vllm import manual_install_for

# Every URL a Linux recipe may name, each answered 200 or 302 to a HEAD
# request on 2026-10-03. A recipe naming anything else is a URL nobody
# checked.
_CHECKED = {
    "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
    "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_aarch64.whl",
    "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cpu-cp38-abi3-manylinux_2_39_x86_64.whl",
    "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cpu-cp38-abi3-manylinux_2_39_aarch64.whl",
    "https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl",
    "https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cpu-cp38-abi3-manylinux_2_34_aarch64.whl",
    "https://wheels.vllm.ai/rocm/0.30.0/rocm723",
    "https://wheels.vllm.ai/rocm/vllm",
    "https://wheels.vllm.ai/0.30.0/xpu",
    "https://wheels.vllm.ai/xpu",
    "https://download.pytorch.org/whl/xpu",
}
_URL = re.compile(r"https://[^\s\"'`)]+")


def _host(accelerator: Accelerator, driver: str | None = None, arch: Arch = Arch.x64) -> Any:
    return HostAccelerator(
        os=Os.linux, arch=arch, accelerator=accelerator, acceleratorVersion=driver
    )


def _urls(text: str | None) -> set[str]:
    return {u.rstrip("/.,;") for u in _URL.findall(text or "")}


@pytest.mark.parametrize("driver", ["13.0", "13.3", None])
def test_a_cuda_13_driver_takes_pypis_wheel_and_is_told_it_is_cuda_13(driver: str | None) -> None:
    recipe = manual_install_for(_host(Accelerator.cuda, driver))
    assert recipe.command == "uv pip install vllm --torch-backend=auto"
    notes = recipe.notes or ""
    assert "CUDA 13.0" in notes and "R580" in notes
    assert "12.9 and bundles" not in notes
    assert "+cu129" in notes, "an older driver is told where its wheel is"


@pytest.mark.parametrize(("arch", "machine"), [(Arch.x64, "x86_64"), (Arch.arm64, "aarch64")])
def test_a_cuda_12_driver_gets_the_cu129_wheel(arch: Arch, machine: str) -> None:
    recipe = manual_install_for(_host(Accelerator.cuda, "12.8", arch))
    command = recipe.command or ""
    assert command == (
        'uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.30.0/'
        f'vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_{machine}.whl" --torch-backend=cu129'
    )


def test_a_driver_older_than_cuda_12_is_told_to_update_not_given_a_command() -> None:
    recipe = manual_install_for(_host(Accelerator.cuda, "11.8"))
    assert recipe.command is None
    assert "11.8" in (recipe.notes or "") and "driver" in (recipe.notes or "")


def test_rocm_installs_the_release_it_was_checked_against() -> None:
    recipe = manual_install_for(_host(Accelerator.rocm))
    assert recipe.command == (
        "uv pip install vllm==0.30.0 --extra-index-url https://wheels.vllm.ai/rocm/0.30.0/rocm723"
    )
    notes = recipe.notes or ""
    assert "7.2.3" in notes and "2.39" in notes and "3.12" in notes
    assert "7.0 and 7.2.1" not in notes


def test_xpu_installs_the_release_with_its_triton_shim() -> None:
    command = manual_install_for(_host(Accelerator.sycl)).command or ""
    assert '"vllm==0.30.0+xpu"' in command
    assert "--extra-index-url https://wheels.vllm.ai/0.30.0/xpu " in command
    assert "--extra-index-url https://wheels.vllm.ai/xpu " in command
    assert "download.pytorch.org/whl/xpu" in command
    assert "--index-strategy unsafe-best-match" in command
    assert "nightly" not in command


@pytest.mark.parametrize(("arch", "machine"), [(Arch.x64, "x86_64"), (Arch.arm64, "aarch64")])
def test_cpu_names_a_wheel_that_exists(arch: Arch, machine: str) -> None:
    recipe = manual_install_for(_host(Accelerator.none, arch=arch))
    assert recipe.command == (
        'uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.30.0/'
        f'vllm-0.30.0+cpu-cp38-abi3-manylinux_2_39_{machine}.whl" --torch-backend cpu'
    )
    assert "${VLLM_VERSION}" not in (recipe.command or "")
    notes = recipe.notes or ""
    assert "2.39" in notes
    assert f"vllm-0.29.0+cpu-cp38-abi3-manylinux_2_34_{machine}.whl" in notes


@pytest.mark.parametrize(
    "host",
    [
        _host(Accelerator.cuda, "13.3"),
        _host(Accelerator.cuda, "12.8"),
        _host(Accelerator.cuda, "12.8", Arch.arm64),
        _host(Accelerator.rocm),
        _host(Accelerator.sycl),
        _host(Accelerator.none),
        _host(Accelerator.none, arch=Arch.arm64),
    ],
)
def test_every_url_in_a_recipe_was_checked(host: Any) -> None:
    recipe = manual_install_for(host)
    named = _urls(recipe.command) | _urls(recipe.notes)
    named.discard(recipe.docsUrl.rstrip("/"))
    unchecked = {u for u in named if not u.startswith("https://docs.vllm.ai/")} - _CHECKED
    assert not unchecked, unchecked
