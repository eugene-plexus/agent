"""A managed Linux CUDA install has to put the CUDA runtime beside the server.

**The finding (upstream drift audit, 2026-10-03, reproduced in WSL2 at
b11375).** Upstream packs the Linux CUDA runtime as its own tarball whose
members sit under one folder named after the archive,
`cudart-llama-bNNNN-bin-ubuntu-cuda-13.4-x64/`, while the server tarball
unpacks into `llama-bNNNN/`. The install extracted both into the build
directory and never put them together, so `libggml-cuda.so` -- which finds
`libcudart.so.13` and the cuBLAS libraries only through its RUNPATH,
`$ORIGIN` -- could not load, and llama.cpp fell back to the processor
without saying so: `llama-server --list-devices` printed `(none)` on a
machine with an RTX 5090. Upstream's own release workflow says the runtime
is to be extracted *next to the binaries ($ORIGIN rpath)*.

The Windows cudart zips have no inner folder, so on Windows the libraries
already landed beside `llama-server.exe`; that layout must not change.

Driven through the adapter's own plan, so the planner telling the installer
which archive is the runtime is part of what is checked, not assumed.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eugene_plexus_agent._generated.models import (
    Accelerator,
    Arch,
    EngineKind,
    HostAccelerator,
    Os,
    State,
)
from eugene_plexus_agent.engines import acquisition as acq
from eugene_plexus_agent.engines.acquisition import (
    AcquisitionPlan,
    EngineInstaller,
    ManagedStore,
    Release,
    ReleaseAsset,
)
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter

_BUILD = "b11375"
_LINUX_SERVER = f"llama-{_BUILD}-bin-ubuntu-cuda-13.4-x64.tar.gz"
_LINUX_CUDART = f"cudart-llama-{_BUILD}-bin-ubuntu-cuda-13.4-x64.tar.gz"
_WIN_SERVER = f"llama-{_BUILD}-bin-win-cuda-13.4-x64.zip"
_WIN_CUDART = "cudart-llama-bin-win-cuda-13.4-x64.zip"

# What upstream's `cp -L` puts in the Linux cudart tarball (release.yml at
# b11375, "Pack CUDA runtime").
_CUDA_LIBS = ("libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13")


def _tar_gz(top: str, files: dict[str, bytes]) -> bytes:
    """A tarball the way upstream packs one: `--transform "s,^\\.,<top>,"`
    puts every member under one folder named `top`."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        directory = tarfile.TarInfo(top)
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        tf.addfile(directory)
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def _asset(name: str, data: bytes) -> ReleaseAsset:
    return ReleaseAsset(
        name=name,
        url=f"https://example.invalid/{_BUILD}/{name}",
        size=len(data),
        digest="sha256:" + hashlib.sha256(data).hexdigest(),
    )


def _release(payloads: dict[str, bytes]) -> Release:
    return Release(
        version=_BUILD,
        published_at=datetime(2026, 10, 3, 7, 0, tzinfo=UTC),
        assets=tuple(_asset(name, data) for name, data in payloads.items()),
    )


def _host(os_kind: Os) -> HostAccelerator:
    return HostAccelerator(
        os=os_kind,
        arch=Arch.x64,
        accelerator=Accelerator.cuda,
        acceleratorVersion="13.4",
        computeCapability="12.0",
    )


async def _install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes], os_kind: Os
) -> tuple[AcquisitionPlan, ManagedStore, EngineInstaller]:
    def fake_download(asset: ReleaseAsset, target: Path, progress: object) -> None:
        target.write_bytes(payloads[asset.name])

    monkeypatch.setattr(acq, "_download", fake_download)
    plan = LlamaCppAdapter().plan_acquisition(_host(os_kind), _release(payloads))
    assert isinstance(plan, AcquisitionPlan), plan
    store = ManagedStore(tmp_path / "engines", EngineKind.llama_cpp)
    installer = EngineInstaller(store, EngineKind.llama_cpp)
    installer.start(plan)
    assert installer._task is not None
    await installer._task
    return plan, store, installer


def _linux_payloads() -> dict[str, bytes]:
    return {
        _LINUX_SERVER: _tar_gz(
            f"llama-{_BUILD}",
            {
                "llama-server": b"\x7fELF server",
                "libggml-cuda.so": b"\x7fELF cuda backend, RUNPATH $ORIGIN",
                "libggml-base.so": b"\x7fELF base",
            },
        ),
        _LINUX_CUDART: _tar_gz(
            f"cudart-llama-{_BUILD}-bin-ubuntu-cuda-13.4-x64",
            {name: f"\x7fELF {name}".encode() for name in _CUDA_LIBS},
        ),
    }


async def test_linux_cuda_runtime_lands_beside_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reproduction: every CUDA library the backend loads by `$ORIGIN`
    is in the folder that holds `llama-server` and `libggml-cuda.so`."""
    _, store, installer = await _install(tmp_path, monkeypatch, _linux_payloads(), Os.linux)
    snapshot = installer.snapshot()
    assert snapshot is not None and snapshot.state is State.done, snapshot and snapshot.error

    current = store.current()
    assert current is not None
    server_dir = current.binary.parent
    assert current.binary.name == "llama-server"
    assert (server_dir / "libggml-cuda.so").is_file()
    for name in _CUDA_LIBS:
        assert (server_dir / name).is_file(), (
            f"{name} is not beside libggml-cuda.so, which finds it only by $ORIGIN: "
            f"{sorted(p.relative_to(current.directory).as_posix() for p in current.directory.rglob('*'))}"
        )
        assert (server_dir / name).read_bytes() == f"\x7fELF {name}".encode()


async def test_linux_install_leaves_no_stray_runtime_folder_or_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The folder the runtime arrived in is gone once its files are moved,
    and so are both archives: nothing left that looks like a second build."""
    _, store, _ = await _install(tmp_path, monkeypatch, _linux_payloads(), Os.linux)
    current = store.current()
    assert current is not None
    top = sorted(p.name for p in current.directory.iterdir())
    assert top == ["install.json", f"llama-{_BUILD}"], top
    assert not list(current.directory.rglob("*.tar.gz"))


async def test_the_planner_names_the_cudart_archive_as_the_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _, _ = await _install(tmp_path, monkeypatch, _linux_payloads(), Os.linux)
    assert plan.runtimes == frozenset({_LINUX_CUDART})
    assert plan.plugins == frozenset()


async def test_windows_cuda_layout_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows zips have no inner folder: the server and the runtime DLLs
    were already side by side at the build's top level, and still are."""
    payloads = {
        _WIN_SERVER: _zip(
            {"llama-server.exe": b"MZ server", "ggml-cuda.dll": b"MZ cuda", "ggml.dll": b"MZ"}
        ),
        _WIN_CUDART: _zip({"cudart64_13.dll": b"MZ cudart", "cublas64_13.dll": b"MZ cublas"}),
    }
    plan, store, installer = await _install(tmp_path, monkeypatch, payloads, Os.windows)
    snapshot = installer.snapshot()
    assert snapshot is not None and snapshot.state is State.done, snapshot and snapshot.error
    assert plan.runtimes == frozenset({_WIN_CUDART})
    current = store.current()
    assert current is not None
    assert current.binary == current.directory / "llama-server.exe"
    assert sorted(p.name for p in current.directory.iterdir()) == [
        "cublas64_13.dll",
        "cudart64_13.dll",
        "ggml-cuda.dll",
        "ggml.dll",
        "install.json",
        "llama-server.exe",
    ]
