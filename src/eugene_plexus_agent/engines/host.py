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
from . import gpu_probe

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
        computeCapability=(
            _probe_compute_capability() if accelerator is Accelerator.cuda else None
        ),
    )


def platform_names() -> tuple[str, str]:
    """`(os, arch)` spelled as `HostAccelerator` spells them, or `""`."""
    os_kind, arch = _detect_os(), _detect_arch()
    return (os_kind.value if os_kind else "", arch.value if arch else "")


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

    # A vendor tool that answers is the strongest evidence there is.
    if _output(["rocm-smi", "--showid"]):
        return Accelerator.rocm, None
    if sycl_sees_gpu():
        return Accelerator.sycl, None

    return _os_accelerator(os_kind, arch), None


def _os_accelerator(os_kind: Os | None, arch: Arch | None) -> Accelerator:
    """The build for a machine no vendor tool answered on, from the OS's
    own list of GPUs (`gpu_probe`).

    **Last, because everything above is better where it applies** (review
    §6.1 #11). Every probe above looks for a vendor tool, and a machine
    with an AMD or Intel card has none unless its owner installed an SDK.
    Until 2026-09-18 every such Windows machine fell through to `none`: a
    CPU build, a fit scored against RAM and the starter set inverted to
    the smallest model, on a machine built around a GPU.

    **And until 2026-09-27 Linux still did.** An AMD card without ROCm got
    the CPU build although upstream publishes `ubuntu-vulkan-x64`. Any
    machine with Intel graphics got `ubuntu-sycl-fp16-x64` because the
    `i915` driver directory existed, which is every Intel laptop, and
    that build needs a oneAPI runtime such a machine rarely has. Both now
    take Vulkan when its loader is installed.

    `gpu_probe.family` decides, and the device list calls the same
    function, so the build and the cards a fit is scored against cannot
    disagree.
    """
    os_name = os_kind.value if os_kind is not None else ""
    arch_name = arch.value if arch is not None else ""
    try:
        found = gpu_probe.adapters(os_name)
    except gpu_probe.GpuProbeError as exc:
        log.info("could not list this machine's GPUs: %s", exc)
        if os_kind is Os.windows:
            return _windows_fallback_accelerator(arch)
        return Accelerator.rocm if rocm_installed(os_kind) else Accelerator.none
    if not found and rocm_installed(os_kind):
        # ROCm installed and the OS listing nothing is a container that
        # sees `/opt/rocm` and not `/sys/class/drm`. The owner put ROCm
        # there on purpose; it keeps the build it had before this probe.
        return Accelerator.rocm
    chosen = gpu_probe.family(
        os_name,
        arch_name,
        found,
        vulkan_loader=gpu_probe.vulkan_loader_present(os_name),
        rocm=rocm_installed(os_kind),
    )
    for note in chosen.notes:
        log.info("%s", note)
    if chosen.adapters:
        log.info(
            "no vendor tool answered; %s will be served by the %s build",
            ", ".join(a.name for a in chosen.adapters),
            chosen.accelerator,
        )
    return Accelerator(chosen.accelerator)


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


def _run(argv: list[str], *, detecting: bool = True) -> Probe | str:
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

    `detecting=False` is for a probe that asks a *working* tool one more
    question, where a failure means "it would not say" rather than "that
    accelerator is not here" -- the warning below would be a false
    statement about the machine for it.
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
    if proc.returncode != 0 and not detecting:
        log.debug(
            "%s exited %d: %s",
            argv[0],
            proc.returncode,
            (proc.stderr or proc.stdout or "").strip()[:200],
        )
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


def _output(argv: list[str], *, detecting: bool = True) -> str | None:
    """`_run` for a caller that only needs output-or-nothing."""
    result = _run(argv, detecting=detecting)
    return None if isinstance(result, Probe) else result


def _probe_cuda() -> str | None:
    """The highest CUDA version this driver supports, per `nvidia-smi`."""
    output = _output(["nvidia-smi"])
    if not output:
        return None
    match = _CUDA_VERSION_RE.search(output)
    return match.group(1) if match else None


# `6.1`, `8.6`, `12.0`. Anything else on a line (`[N/A]` for a MIG slice,
# an error banner) is not a card we can reason about and is skipped.
_COMPUTE_CAP_RE = re.compile(r"^\s*(\d+)\.(\d+)\s*$")


def _probe_compute_capability() -> str | None:
    """The LOWEST compute capability among this machine's NVIDIA cards.

    **The driver's CUDA version cannot choose a build on its own.** From
    CUDA 13 the toolkit no longer compiles for Maxwell, Pascal or Volta,
    and the last driver branch those cards get (580) reports CUDA 13.0 --
    so a Pascal card's driver asks for exactly the build that carries no
    code for it (see `_cuda_variant`). The lowest card, because one
    engine may use every visible card and its build has to carry code
    for all of them.

    `None` when the driver would not say. `compute_cap` arrived around
    driver 510, and nothing older reports a CUDA version any published
    build accepts, so a missing answer never changes the outcome.
    """
    output = _output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], detecting=False
    )
    if not output:
        return None
    caps = [
        (int(m.group(1)), int(m.group(2)))
        for line in output.splitlines()
        if (m := _COMPUTE_CAP_RE.match(line))
    ]
    if not caps:
        return None
    major, minor = min(caps)
    return f"{major}.{minor}"


def _has_nvidia() -> bool:
    output = _output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return bool(output and output.strip())


def rocm_installed(os_kind: Os | None) -> bool:
    """AMD's compute SDK is on disk, whether or not its tool answered.

    **Windows ROCm is the HIP SDK, and nothing else says it is there.**
    `rocm-smi` and `/opt/rocm` are both Linux-only, so this could never
    be true on Windows and `win-rocm-10.0-x64` was dead code that read as
    AMD support. On Linux `rocm-smi` is not always on PATH even where
    ROCm is installed, and the install prefix is stable enough to check
    directly. Either way ROCm is chosen only for an AMD adapter
    (`gpu_probe.family`); an AMD owner without the SDK gets Vulkan, which
    works.
    """
    if os_kind is Os.windows:
        return _windows_hip_sdk_present()
    return Path("/opt/rocm").is_dir()


# `[level_zero:gpu][level_zero:0] Intel(R) Arc(TM) A770 Graphics ...` from a
# current oneAPI, `[ext_oneapi_level_zero:gpu:0] ...` from an older one.
_SYCL_GPU_RE = re.compile(r"\[[a-z_]+:gpu", re.IGNORECASE)


def sycl_sees_gpu() -> bool:
    """oneAPI is set up and its runtime can see an Intel GPU.

    **A GPU, not merely an answer.** `sycl-ls` also lists the CPU's
    OpenCL device, so oneAPI installed on a machine with no Intel GPU
    used to read as SYCL and fetch a build for a card that is not there.

    **And no longer the `i915` driver directory** (2026-09-27). That
    directory exists on every Linux machine with Intel graphics, which is
    every Intel laptop. It selected `ubuntu-sycl-fp16-x64`, whose runtime
    such a machine rarely has installed. Those machines take Vulkan now.
    """
    output = _output(["sycl-ls"])
    return bool(output and _SYCL_GPU_RE.search(output))


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


def _windows_fallback_accelerator(arch: Arch | None) -> Accelerator:
    """What a Windows machine gets when DXCore could not be asked.

    The mechanism R2.3 built, kept for a Windows older than DXCore
    (version 2004): `Win32_VideoController` has names and no memory
    figures, which is enough to choose a build and not to score a fit.
    Vulkan rather than ROCm or SYCL, because upstream ships
    `llama-*-bin-win-vulkan-x64.zip` in every release, it needs no
    vendor SDK, and one build covers AMD and Intel alike -- but only on
    x64, because there is no `win-vulkan-arm64` for a Snapdragon to take.
    """
    if not gpu_probe.vulkan_build_published("windows", arch.value if arch else ""):
        return Accelerator.none
    for name in _windows_display_adapters():
        lowered = name.lower()
        if any(vendor in lowered for vendor in _VULKAN_VENDORS):
            log.info("no vendor tool answered; %s will be served by the Vulkan build", name)
            return Accelerator.vulkan
    return Accelerator.none


__all__ = ["detect_host", "platform_names", "rocm_installed", "sycl_sees_gpu"]
