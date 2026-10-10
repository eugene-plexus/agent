"""Pinned native + Python recipe. No model downloads or interactive setup."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath

from .._generated.models import HostAccelerator, State
from ..child_env import child_environment
from ..interpreter import command_python
from .acquisition import (
    AcquisitionError,
    AcquisitionPlan,
    EngineInstaller,
    ReleaseAsset,
    Unavailable,
    _download,
    _extract,
    _find_binary,
    _guard_members,
    _Progress,
    _remove_quietly,
    _verify,
)
from .strata import COMMIT, VERSION, runtime_lib_dirs

SOURCE = ReleaseAsset(
    "source.zip",
    f"https://codeload.github.com/Niko1221/Strata/zip/{COMMIT}",
    15458849,
    "sha256:d91a853d731c9fc3b60962523eac089f5132dcf6364977e357307915f31424b6",
)
NATIVE = ReleaseAsset(
    "strata-windows-x64.zip",
    f"https://github.com/Niko1221/Strata/releases/download/{VERSION}/strata-windows-x64.zip",
    123629160,
    "sha256:a862bcfa2330cd1c23f9b5d6e49f4027da8f8313842bd62e858ec6cd4533813a",
)
#: setup.py `LLAMA_CPP_COMMIT` at the pinned Strata commit: the source its
#: preparation reads (`gguf-py`). Since LS7 it comes with the install, not the
#: first preparation (Troy reversed B47: Strata runs only prepared models, so
#: preparing is part of using it). Setup unpacks the whole archive under its
#: own folder, where the deepest path is about 281 characters on a Windows
#: install, past the 260 Windows allows; only what the preparation reads is
#: unpacked, which setup then finds and keeps.
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMA_CPP = ReleaseAsset(
    "llama.cpp.zip",
    f"https://codeload.github.com/ggml-org/llama.cpp/zip/{LLAMA_CPP_COMMIT}",
    39_564_399,
    "sha256:cbe23c594282ead2937abb3f008e51fcec4609d9256652fff42c7cc1c21ea47b",
)
#: `gguf-py/` for the tools (`STRATA_GGUF_PY`); `ggml/` because setup's
#: `get_llama_cpp` takes the source as there only when `ggml/CMakeLists.txt` is.
LLAMA_CPP_PARTS = ("gguf-py/", "ggml/")
#: Upstream's own pinned list (#214) of what its setup and tools need beside
#: the server: installed with the engine since LS7.
SETUP_REQUIREMENTS = "requirements.txt"

# Runtime subset of upstream requirements.txt and setup.py CUDA_WHEELS.
REQUIREMENTS = [
    "jinja2==3.1.6",
    "markupsafe==3.0.3",
    "regex==2026.9.10",
    "psutil==7.2.2",
    "nvidia-cublas==13.0.2.14",
    "nvidia-cuda-runtime==13.0.96",
]


def plan(
    host: HostAccelerator, version: str | None, variant: str | None
) -> AcquisitionPlan | Unavailable:
    if host.os != "windows" or host.arch != "x64" or host.accelerator != "cuda":
        return Unavailable(
            "Strata's experimental installer requires Windows x64 with NVIDIA. "
            "macOS is not supported by this recipe."
        )
    try:
        cuda = float(host.acceleratorVersion or "0")
    except ValueError:
        cuda = 0
    if cuda < 13:
        return Unavailable(
            "Strata requires an NVIDIA driver supporting CUDA 13 (580 or later). "
            "Update the driver first."
        )
    if version not in (None, VERSION) or variant not in (None, "windows-x64-cuda13"):
        return Unavailable(f"This experimental recipe supports {VERSION}, windows-x64-cuda13 only.")
    return AcquisitionPlan(VERSION, "windows-x64-cuda13", (SOURCE, NATIVE, LLAMA_CPP), "server.py")


def tools_installed(root: Path) -> bool:
    """Whether this install carries the preparation tools (LS7)."""
    llama = root / "third_party" / "llama.cpp"
    return (llama / "ggml" / "CMakeLists.txt").is_file() and (llama / "gguf-py").is_dir()


def unpack_llama_parts(archive: Path, destination: Path) -> None:
    """Only `LLAMA_CPP_PARTS` of llama.cpp's source archive, into
    `destination` (`third_party/llama.cpp`), replacing what is there."""
    work = destination.parent / ".eugene-llama"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            _guard_members(archive, zf.namelist())
            for info in zf.infolist():
                _top, _, rest = info.filename.partition("/")
                if info.is_dir() or not rest.startswith(LLAMA_CPP_PARTS):
                    continue
                target = work.joinpath(*PurePosixPath(rest).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
        if not (work / "ggml" / "CMakeLists.txt").is_file():
            raise AcquisitionError("llama.cpp's source archive has no ggml/CMakeLists.txt")
        shutil.rmtree(destination, ignore_errors=True)
        work.replace(destination)
    except zipfile.BadZipFile as exc:
        raise AcquisitionError(f"llama.cpp's source archive could not be read: {exc}") from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _readable(raw: bytes) -> str:
    """A child's output as a person can read it.

    pip writes UTF-8, but some Windows programs write UTF-16: read as UTF-8,
    pythonservice.exe's usage put a NUL between every letter, shown as a
    box (2026-10-09). A byte-order mark, or a NUL in every other byte, is
    UTF-16; stray NULs are dropped either way.
    """
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace").replace("\x00", "")
    odd = raw[1::2]
    if odd and odd.count(0) * 2 >= len(odd):
        return raw.decode("utf-16-le", errors="replace").replace("\x00", "")
    return raw.decode("utf-8", errors="replace").replace("\x00", "")


def _run_command(argv: list[str], progress: _Progress, cwd: Path) -> None:
    # Output is captured in a bounded tail on failure; no visible console and
    # no shell. Wheel-only pip does not spawn compiler/build subprocesses.
    log = cwd / ".install-output.txt"
    env = child_environment()
    # Pip config inherited from Eugene must not redirect this isolated install.
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PATH"] = os.pathsep.join([*runtime_lib_dirs(cwd), env.get("PATH", "")])
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.perf_counter() + 1200
            while process.poll() is None:
                progress.check_cancelled()
                if time.perf_counter() > deadline:
                    raise AcquisitionError("Strata environment setup timed out")
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    # A Windows venv's python.exe is a launcher. Terminating
                    # only that PID leaves pip's real Python writing files.
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        check=False,
                    )
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if process.returncode:
        # Name the command and its exit code: the output alone may be
        # another program's usage text (pythonservice.exe's, 2026-10-09).
        raise AcquisitionError(
            f"Strata environment setup failed: `{Path(argv[0]).name} {' '.join(argv[1:3])}` "
            f"exited {process.returncode}: " + _readable(log.read_bytes())[-3000:]
        )
    log.unlink(missing_ok=True)


def _venv_command(target: Path) -> list[str]:
    """`python -m venv`, with an interpreter that can run it: under the
    Windows service `sys.executable` is pythonservice.exe, which cannot
    (found installing Strata on a service install, 2026-10-09)."""
    try:
        python = command_python()
    except FileNotFoundError as e:
        raise AcquisitionError(str(e)) from e
    return [python, "-m", "venv", str(target)]


class StrataInstaller(EngineInstaller):
    def _install(self, plan: AcquisitionPlan, progress: _Progress, staging: Path) -> None:
        progress.check_cancelled()
        final = self._store.build_dir(plan.version)
        if final.exists():
            if any(b.directory == final for b in self._store.list_builds()):
                progress.committed = True
                return  # The pinned recipe is immutable; never overwrite a live environment.
            raise AcquisitionError(f"{final} already exists without a complete install receipt")
        _remove_quietly(staging)
        staging.mkdir(parents=True, exist_ok=True)
        llama_archive: Path | None = None
        for asset in plan.assets:
            progress.check_cancelled()
            progress.state = State.downloading
            progress.message = f"downloading {asset.name}"
            archive = staging / asset.name
            _download(asset, archive, progress)
            progress.state = State.verifying
            _verify(asset, archive)
            if asset.name == LLAMA_CPP.name:
                llama_archive = archive  # unpacked into the source, below
                continue
            progress.state = State.extracting
            destination = staging / ("source" if asset.name == SOURCE.name else "native")
            _extract(archive, destination)
            archive.unlink()
        root = staging / "source" / f"Strata-{COMMIT}"
        server = root / "serve" / "server.py"
        if not server.is_file():
            raise AcquisitionError("Strata source archive does not contain its HTTP server")
        native = _find_binary(staging / "native", "strata.exe")
        if native is None:
            raise AcquisitionError("Strata native archive does not contain strata.exe")
        shutil.copytree(native.parent, root / "engine", dirs_exist_ok=True)
        _remove_quietly(staging / "native")
        if llama_archive is not None:
            progress.state = State.extracting
            progress.message = f"Strata's preparation tools: llama.cpp at {LLAMA_CPP_COMMIT[:7]}"
            unpack_llama_parts(llama_archive, root / "third_party" / "llama.cpp")
            llama_archive.unlink(missing_ok=True)
        progress.message = "creating Strata's isolated Python environment"
        _run_command(_venv_command(root / ".venv"), progress, root)
        python = root / ".venv" / "Scripts" / "python.exe"
        progress.message = (
            "installing Strata's Python and CUDA dependencies (model files are separate)"
        )
        _run_command(
            [
                str(python),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--no-cache-dir",
                *REQUIREMENTS,
            ],
            progress,
            root,
        )
        if (root / SETUP_REQUIREMENTS).is_file():
            progress.message = "installing what Strata's setup prepares models with"
            _run_command(
                [
                    str(python),
                    "-m",
                    "pip",
                    "--isolated",
                    "install",
                    "--disable-pip-version-check",
                    "--only-binary=:all:",
                    "--no-cache-dir",
                    "-r",
                    str(root / SETUP_REQUIREMENTS),
                ],
                progress,
                root,
            )
        _run_command([str(python), str(server), "--help"], progress, root)
        progress.message = "checking Strata's native executable and CUDA libraries"
        _run_command([str(root / "engine" / "strata.exe"), "--help"], progress, root)
        progress.check_cancelled()
        self._store.write_metadata(
            staging, version=plan.version, variant=plan.variant, binary=server
        )
        receipt = staging / self._store.METADATA_NAME
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        metadata["recipe"] = {
            "sourceCommit": COMMIT,
            "assets": [{"url": a.url, "sha256": a.sha256} for a in plan.assets],
            "packages": REQUIREMENTS,
            "modelsIncluded": False,
            "preparationTools": {
                "llamaCpp": LLAMA_CPP_COMMIT,
                "requirements": SETUP_REQUIREMENTS,
            },
        }
        receipt.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staging.replace(final)
        progress.committed = True
        # No pruning: an explicitly selected older build may still be in use.
