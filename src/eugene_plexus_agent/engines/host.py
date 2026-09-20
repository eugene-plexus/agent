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

import enum
import logging
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from .._generated.models import Accelerator, Arch, HostAccelerator, Os
from ..child_env import child_environment

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
    #
    # **A card whose tool EXITED NON-ZERO is a different thing** (review
    # §6.2 #29), and `_run` used to hand back its stderr as though it
    # were output: `nvidia-smi` printing *Failed to initialize NVML* and
    # exiting 9 read as a working NVIDIA host, so the install plan
    # fetched a CUDA build that cannot load, for a reason nothing
    # mentions. A driver/library version mismatch after an update is the
    # commonest Linux failure there is. `Probe.FAILED` is not a card.
    if _has_nvidia():
        return Accelerator.cuda, None

    if _has_rocm():
        return Accelerator.rocm, None
    if _has_sycl():
        return Accelerator.sycl, None

    # **Windows, last, because everything above is better where it
    # applies** (review §6.1 #11). Every probe above looks for a vendor
    # tool or a Linux sysfs path, and a Windows machine with an AMD or
    # Intel card has neither unless its owner installed an SDK — so
    # until this branch existed, every one of them fell through to
    # `none`: a CPU build, a fit scored against RAM, and the starter set
    # inverted to the smallest model, on a machine built around a GPU.
    if os_kind is Os.windows:
        return _windows_fallback_accelerator(), None

    return Accelerator.none, None


class Probe(enum.Enum):
    """Why a vendor tool produced nothing — review §6.2 #29.

    `_run` answered `None` for *not installed* and combined stdout with
    stderr for everything else, so a tool that ran and failed came back
    as a non-empty string that read like a detection. Three states, in
    one place, because `_has_nvidia`, `_has_rocm` and `_has_sycl` all
    need the same distinction and three copies of it is three chances to
    get one wrong.
    """

    ABSENT = "absent"
    FAILED = "failed"


def _run(argv: list[str]) -> Probe | str:
    """Run a probe: its output, or which kind of nothing.

    **The resolved path, not the bare name.** `shutil.which` searches
    PATH; `CreateProcess` searches the application directory, the
    current directory, **System32**, the Windows directory and only then
    PATH -- so on Windows the two disagree whenever a vendor tool exists
    in System32, which `nvidia-smi` does on every machine with an NVIDIA
    driver. Measured 2026-09-18: `which` answered with a stub first on
    PATH while `subprocess` ran the System32 copy, so this function's
    gate and its subject were two different programs. An operator who
    puts a newer tool earlier on PATH gets the one they chose now.
    """
    exe = shutil.which(argv[0])
    if exe is None:
        return Probe.ABSENT
    try:
        proc = subprocess.run(
            [exe, *argv[1:]],
            capture_output=True,
            env=child_environment(),
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("probe %r failed: %s", argv[0], e)
        return Probe.FAILED
    if proc.returncode != 0:
        log.warning(
            "%s is installed and exited %d: %s. Treating this machine as though that "
            "accelerator is not here, because a build for a driver that is not working "
            "fails at load rather than at install.",
            argv[0],
            proc.returncode,
            (proc.stderr or proc.stdout or "").strip()[:200],
        )
        return Probe.FAILED
    return (proc.stdout or "") + (proc.stderr or "")


def _output(argv: list[str]) -> str | None:
    """`_run` for a caller that only needs output-or-nothing."""
    result = _run(argv)
    return None if isinstance(result, Probe) else result


def _probe_cuda() -> str | None:
    """The highest CUDA version this driver supports, per `nvidia-smi`."""
    output = _output(["nvidia-smi"])
    if not output:
        return None
    match = _CUDA_VERSION_RE.search(output)
    return match.group(1) if match else None


def _has_nvidia() -> bool:
    output = _output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return bool(output and output.strip())


def _has_rocm() -> bool:
    if _output(["rocm-smi", "--showid"]):
        return True
    if platform.system().lower() == "windows":
        # **Windows ROCm is the HIP SDK, and nothing else says it is
        # there.** `rocm-smi` and `/opt/rocm` are both Linux-only, so
        # `_has_rocm` could never be true here and `win-rocm-10.0-x64`
        # was dead code that read as AMD support. This makes it
        # reachable without betting the common case on it: an AMD owner
        # who has not installed the SDK gets the Vulkan build, which
        # works.
        return _windows_hip_sdk_present()
    # `rocm-smi` is not always on PATH even where ROCm is installed, and the
    # install prefix is stable enough to be worth checking directly.
    return Path("/opt/rocm").is_dir()


def _has_sycl() -> bool:
    """Intel GPU, as far as we can tell without a toolkit installed.

    `sycl-ls` only exists once oneAPI is set up, which is exactly the
    situation where the operator does not need us to guess. The DRM device
    check catches a bare Arc / Xe card on Linux with nothing installed.

    Deliberately still Linux-shaped: on Windows an Intel card is served
    by the Vulkan build, because upstream publishes `win-sycl-x64`
    nowhere this repo can see and a variant nobody ships turns a slow
    install into a failed one.
    """
    if _output(["sycl-ls"]):
        return True
    by_path = Path("/sys/bus/pci/drivers/i915")
    xe = Path("/sys/bus/pci/drivers/xe")
    return by_path.is_dir() or xe.is_dir()


# --------------------------------------------------------------------------- #
# Windows: the card nobody looked for
# --------------------------------------------------------------------------- #

# **Vendors upstream's Vulkan build actually serves, and that list is the
# whole filter.** Every Windows machine has a display adapter, so the
# probe has to look for a real GPU vendor rather than for the presence of
# one -- but an exclusion list beside it ("microsoft basic", "virtual",
# "citrix" ...) was **unreachable**, which a sabotage proved: none of
# those names contains a vendor word, so the allowlist had already
# rejected them. It is deleted rather than kept, because an unreachable
# filter that reads as protection is the same defect this slice removed
# from `win-rocm-10.0-x64`.
_VULKAN_VENDORS = ("amd", "radeon", "intel", "arc(tm)", "nvidia", "qualcomm", "adreno")

_HIP_SDK_PATHS = (
    r"C:\Program Files\AMD\ROCm",
    r"C:\Program Files\AMD\HIP SDK",
)


def _windows_hip_sdk_present() -> bool:
    """Is AMD's HIP SDK installed, so a ROCm build can load?"""
    env = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH")
    if env and Path(env).is_dir():
        return True
    return any(Path(candidate).is_dir() for candidate in _HIP_SDK_PATHS)


def _windows_display_adapters() -> list[str]:
    """Every display adapter's name, from `Win32_VideoController`.

    **No new dependency.** `pywin32` arrives with the Windows
    `[service]` extra that both installer paths take, and
    `firewall/windows.py` already reads WMI through it — measured at
    112 ms in-tree, which is the precedent this follows rather than
    introducing a second mechanism.

    An empty list means *could not look*, never *no GPU*: without
    `pywin32` the caller falls back to what the vendor tools said, which
    is what this branch exists to improve on rather than to replace.
    """
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        log.debug("pywin32 is not installed, so Win32_VideoController cannot be read")
        return []
    try:
        # Idempotent, and deliberately not torn down: a FastAPI worker
        # thread is pooled. Same note as `firewall/windows.py`.
        import contextlib

        with contextlib.suppress(Exception):
            pythoncom.CoInitialize()
        locator = win32com.client.Dispatch("WbemScripting.SWbemLocator")
        service = locator.ConnectServer(".", "root\\cimv2")
        rows = service.ExecQuery("SELECT Name FROM Win32_VideoController")
        return [str(row.Name) for row in rows if getattr(row, "Name", None)]
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("could not read Win32_VideoController: %s", exc)
        return []


def _windows_fallback_accelerator() -> Accelerator:
    """What a Windows machine gets when no vendor tool answered.

    Vulkan rather than ROCm or SYCL, and the reason is about the user's
    machine rather than about a table: upstream ships
    `llama-*-bin-win-vulkan-x64.zip` in every release, it needs no
    vendor SDK, and one build covers AMD and Intel alike. An owner who
    has installed the HIP SDK gets ROCm instead, which `_has_rocm`
    decided before this was reached.
    """
    for name in _windows_display_adapters():
        lowered = name.lower()
        if any(vendor in lowered for vendor in _VULKAN_VENDORS):
            log.info("no vendor tool answered; %s will be served by the Vulkan build", name)
            return Accelerator.vulkan
    return Accelerator.none


__all__ = ["detect_host"]
