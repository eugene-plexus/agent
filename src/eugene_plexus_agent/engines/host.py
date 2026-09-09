"""What this machine is, insofar as it decides which engine build to fetch.

Deliberately narrow. This is not a hardware inventory — the VRAM-and-quant-fit
surface that discovery guidance needs belongs to the library component, which
does not exist yet. All that is answered here is the four-tuple that selects a
release asset: OS, architecture, accelerator family, and (for CUDA) the
version ceiling the installed driver imposes.

Detection is best-effort by construction. Every probe shells out to a vendor
tool that may be absent, may be a stub, or may print a format nobody promised
to keep. A probe that fails means "not this accelerator", never an error —
the worst outcome of a missed GPU is a CPU build, which runs.
"""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
from pathlib import Path

from .._generated.models import Accelerator, Arch, HostAccelerator, Os

log = logging.getLogger(__name__)

# Vendor tools are quick when present and hang when something is wrong with
# the driver stack. Short, because this runs behind GET /v1/engines.
_PROBE_TIMEOUT_SECONDS = 5.0

# `nvidia-smi`'s banner has carried the CUDA ceiling under two spellings:
#   | NVIDIA-SMI 550.54   Driver Version: 550.54   CUDA Version: 12.4 |
#   | NVIDIA-SMI 610.47   KMD Version: 610.47      CUDA UMD Version: 13.3 |
# Accept both. This is the driver's *maximum supported* CUDA, not an
# installed toolkit — which is the right number, since a prebuilt binary
# ships its own runtime and only needs the driver to be new enough.
_CUDA_VERSION_RE = re.compile(r"CUDA(?:\s+UMD)?\s+Version:\s*([0-9]+(?:\.[0-9]+)?)")


def detect_host() -> HostAccelerator:
    """Best-effort description of this machine for asset selection."""
    os_kind = _detect_os()
    arch = _detect_arch()
    accelerator, version = _detect_accelerator(os_kind, arch)
    return HostAccelerator(
        os=os_kind,
        arch=arch,
        accelerator=accelerator,
        acceleratorVersion=version,
    )


def _detect_os() -> Os | None:
    system = platform.system().lower()
    if system == "windows":
        return Os.windows
    if system == "linux":
        return Os.linux
    if system == "darwin":
        return Os.macos
    log.debug("unrecognised platform.system() %r", system)
    return None


def _detect_arch() -> Arch | None:
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64", "x64"}:
        return Arch.x64
    if machine in {"arm64", "aarch64"}:
        return Arch.arm64
    log.debug("unrecognised platform.machine() %r", machine)
    return None


def _detect_accelerator(os_kind: Os | None, arch: Arch | None) -> tuple[Accelerator, str | None]:
    # Apple silicon first and unconditionally: Metal is compiled into the
    # plain macOS build, so there is no probe to run and nothing to choose.
    # Reporting `none` here would read as "no GPU" on a machine whose GPU
    # is the entire reason it is good at this.
    if os_kind is Os.macos and arch is Arch.arm64:
        return Accelerator.metal, None

    cuda_version = _probe_cuda()
    if cuda_version is not None:
        return Accelerator.cuda, cuda_version
    # An NVIDIA card with an unparseable banner is still an NVIDIA card. On
    # Linux that is the difference between "we refuse, here is why" and
    # "here is a CPU build" — and the former is the honest answer.
    if _has_nvidia():
        return Accelerator.cuda, None

    if _has_rocm():
        return Accelerator.rocm, None
    if _has_sycl():
        return Accelerator.sycl, None
    return Accelerator.none, None


def _run(argv: list[str]) -> str | None:
    """Run a probe, return its combined output, or None if it didn't work."""
    if shutil.which(argv[0]) is None:
        return None
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("probe %r failed: %s", argv[0], e)
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def _probe_cuda() -> str | None:
    """The highest CUDA version this driver supports, per `nvidia-smi`."""
    output = _run(["nvidia-smi"])
    if not output:
        return None
    match = _CUDA_VERSION_RE.search(output)
    return match.group(1) if match else None


def _has_nvidia() -> bool:
    output = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return bool(output and output.strip())


def _has_rocm() -> bool:
    if _run(["rocm-smi", "--showid"]):
        return True
    # `rocm-smi` is not always on PATH even where ROCm is installed, and the
    # install prefix is stable enough to be worth checking directly.
    return Path("/opt/rocm").is_dir()


def _has_sycl() -> bool:
    """Intel GPU, as far as we can tell without a toolkit installed.

    `sycl-ls` only exists once oneAPI is set up, which is exactly the
    situation where the operator does not need us to guess. The DRM device
    check catches a bare Arc / Xe card on Linux with nothing installed.
    """
    if _run(["sycl-ls"]):
        return True
    by_path = Path("/sys/bus/pci/drivers/i915")
    xe = Path("/sys/bus/pci/drivers/xe")
    return by_path.is_dir() or xe.is_dir()


__all__ = ["detect_host"]
