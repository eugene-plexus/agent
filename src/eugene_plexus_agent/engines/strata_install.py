"""Pinned native + Python recipe. No model downloads or interactive setup."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .._generated.models import HostAccelerator, State
from ..child_env import child_environment
from .acquisition import (
    AcquisitionError,
    AcquisitionPlan,
    EngineInstaller,
    ReleaseAsset,
    Unavailable,
    _download,
    _extract,
    _find_binary,
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
# Runtime subset of upstream requirements.txt and setup.py CUDA_WHEELS.
# Conversion/build tools are unnecessary for already prepared model assets.
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
    return AcquisitionPlan(VERSION, "windows-x64-cuda13", (SOURCE, NATIVE), "server.py")


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
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
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
                        creationflags=subprocess.CREATE_NO_WINDOW,
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
        raise AcquisitionError(
            "Strata environment setup failed: "
            + log.read_text(encoding="utf-8", errors="replace")[-3000:]
        )
    log.unlink(missing_ok=True)


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
        for asset in plan.assets:
            progress.check_cancelled()
            progress.state = State.downloading
            progress.message = f"downloading {asset.name}"
            archive = staging / asset.name
            _download(asset, archive, progress)
            progress.state = State.verifying
            _verify(asset, archive)
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
        progress.message = "creating Strata's isolated Python environment"
        _run_command([sys.executable, "-m", "venv", str(root / ".venv")], progress, root)
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
        }
        receipt.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staging.replace(final)
        progress.committed = True
        # No pruning: an explicitly selected older build may still be in use.
