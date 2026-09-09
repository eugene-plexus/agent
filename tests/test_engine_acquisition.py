"""Engine acquisition: selection, install machinery, retention.

No network. The selection tests are the point of the file — they encode
what upstream actually publishes, checked against `ggml-org/llama.cpp`
b10867 on 2026-09-08, and every one of them exists because getting it
wrong produces an install that looks successful and then doesn't run.
"""

from __future__ import annotations

import hashlib
import io
import json
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
    AcquisitionError,
    EngineInstaller,
    ManagedStore,
    Release,
    ReleaseAsset,
    Unavailable,
)
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter

# --------------------------------------------------------------------------- #
# A real release, trimmed. Names and the two-asset CUDA shape are verbatim.
# --------------------------------------------------------------------------- #

_ASSET_NAMES = [
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.3-x64.zip",
    "llama-b10867-bin-macos-arm64.tar.gz",
    "llama-b10867-bin-macos-x64.tar.gz",
    "llama-b10867-bin-ubuntu-arm64.tar.gz",
    "llama-b10867-bin-ubuntu-rocm-10.0-x64.tar.gz",
    "llama-b10867-bin-ubuntu-sycl-fp16-x64.tar.gz",
    "llama-b10867-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b10867-bin-ubuntu-x64.tar.gz",
    "llama-b10867-bin-win-cpu-x64.zip",
    "llama-b10867-bin-win-cuda-12.4-x64.zip",
    "llama-b10867-bin-win-cuda-13.3-x64.zip",
    "llama-b10867-bin-win-rocm-10.0-x64.zip",
    "llama-b10867-bin-win-vulkan-x64.zip",
]


def _release(version: str = "b10867", names: list[str] | None = None) -> Release:
    return Release(
        version=version,
        published_at=datetime(2026, 9, 8, 17, 31, tzinfo=UTC),
        assets=tuple(
            ReleaseAsset(
                name=name,
                url=f"https://example.invalid/{version}/{name}",
                size=1024,
                digest="sha256:" + "0" * 64,
            )
            for name in (names if names is not None else _ASSET_NAMES)
        ),
    )


def _host(
    os_kind: Os,
    arch: Arch = Arch.x64,
    accelerator: Accelerator = Accelerator.none,
    version: str | None = None,
) -> HostAccelerator:
    return HostAccelerator(
        os=os_kind, arch=arch, accelerator=accelerator, acceleratorVersion=version
    )


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (_host(Os.windows, Arch.x64, Accelerator.cuda, "13.3"), "win-cuda-13.3-x64"),
        (_host(Os.windows, Arch.x64, Accelerator.rocm), "win-rocm-10.0-x64"),
        (_host(Os.windows, Arch.x64), "win-cpu-x64"),
        (_host(Os.macos, Arch.arm64, Accelerator.metal), "macos-arm64"),
        (_host(Os.macos, Arch.x64), "macos-x64"),
        (_host(Os.linux, Arch.x64, Accelerator.rocm), "ubuntu-rocm-10.0-x64"),
        (_host(Os.linux, Arch.x64, Accelerator.sycl), "ubuntu-sycl-fp16-x64"),
        (_host(Os.linux, Arch.x64), "ubuntu-x64"),
        (_host(Os.linux, Arch.arm64), "ubuntu-arm64"),
    ],
)
def test_variant_selection(host: HostAccelerator, expected: str) -> None:
    plan = LlamaCppAdapter().plan_acquisition(host, _release())
    assert not isinstance(plan, Unavailable), getattr(plan, "reason", "")
    assert plan.variant == expected


def test_linux_with_nvidia_is_refused_not_given_vulkan() -> None:
    """Upstream publishes no Linux CUDA build. The Vulkan build would
    install cleanly and run on the card, which is exactly the trap:
    it is materially slower at prompt processing, so substituting it
    silently means the operator concludes the product is slow instead
    of learning they need a source build."""
    plan = LlamaCppAdapter().plan_acquisition(
        _host(Os.linux, Arch.x64, Accelerator.cuda, "13.3"), _release()
    )
    assert isinstance(plan, Unavailable)
    assert "no prebuilt CUDA build for Linux" in plan.reason
    # It must name the way forward, not just decline.
    assert "source" in plan.reason and "binary" in plan.reason
    # And it must say out loud that Vulkan exists and why it wasn't used.
    # Naming the road not taken is the difference between a refusal an
    # operator can act on and one that reads as a missing feature.
    assert "Vulkan" in plan.reason
    assert "slower" in plan.reason


def test_windows_cuda_pulls_the_cudart_companion() -> None:
    """The server zip carries no CUDA runtime. Installing it alone gives
    a binary that dies on a missing cudart DLL, and the companion
    archive's filename has no build number even though it lives under
    the same release tag."""
    plan = LlamaCppAdapter().plan_acquisition(
        _host(Os.windows, Arch.x64, Accelerator.cuda, "13.3"), _release()
    )
    assert not isinstance(plan, Unavailable)
    assert [a.name for a in plan.assets] == [
        "llama-b10867-bin-win-cuda-13.3-x64.zip",
        "cudart-llama-bin-win-cuda-13.3-x64.zip",
    ]


def test_windows_cuda_without_its_cudart_is_refused() -> None:
    """Half a CUDA install is worse than none — it looks like success."""
    names = [n for n in _ASSET_NAMES if not n.startswith("cudart-")]
    plan = LlamaCppAdapter().plan_acquisition(
        _host(Os.windows, Arch.x64, Accelerator.cuda, "13.3"), _release(names=names)
    )
    assert isinstance(plan, Unavailable)
    assert "cudart" in plan.reason


def test_cuda_picks_the_highest_build_the_driver_can_load() -> None:
    """Minor-version compatibility within a major: a 12.4 build runs on a
    12.8 driver. A 13.x build does not."""
    plan = LlamaCppAdapter().plan_acquisition(
        _host(Os.windows, Arch.x64, Accelerator.cuda, "12.8"), _release()
    )
    assert not isinstance(plan, Unavailable)
    assert plan.variant == "win-cuda-12.4-x64"


def test_cuda_never_crosses_a_major_upward() -> None:
    """A driver capped at 12.2 cannot load a 12.4 build either, and must
    not be handed the 13.x one as 'closest'."""
    plan = LlamaCppAdapter().plan_acquisition(
        _host(Os.windows, Arch.x64, Accelerator.cuda, "12.2"), _release()
    )
    assert isinstance(plan, Unavailable)
    assert "Update the NVIDIA driver" in plan.reason


def test_nvidia_without_a_reported_cuda_version_is_refused() -> None:
    """Guessing the major means a binary that fails at load with a
    message about the driver rather than about us."""
    plan = LlamaCppAdapter().plan_acquisition(
        _host(Os.windows, Arch.x64, Accelerator.cuda, None), _release()
    )
    assert isinstance(plan, Unavailable)
    assert "did not report a CUDA version" in plan.reason


def test_unmatched_variant_lists_what_was_actually_published() -> None:
    """Upstream's asset naming is not a contract — `linux-` became
    `ubuntu-` once already. When nothing matches, the error has to name
    what was wanted and what was there, or the next rename costs an
    afternoon."""
    plan = LlamaCppAdapter().plan_acquisition(
        _host(Os.windows, Arch.x64), _release(names=["llama-b10867-bin-macos-arm64.tar.gz"])
    )
    assert isinstance(plan, Unavailable)
    assert "win-cpu-x64" in plan.reason
    assert "macos-arm64" in plan.reason


def test_latest_release_ignores_the_non_build_latest_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """`releases/latest` on this repo points at a `v0.4.0` tag with none
    of the server assets. Builds are `bNNNN` and are chosen by number,
    not by publish order or by asking GitHub what's latest."""
    adapter = LlamaCppAdapter()
    monkeypatch.setattr(
        adapter.releases,
        "list_releases",
        lambda force=False: [
            _release("v0.4.0"),
            _release("b10859"),
            _release("b10867"),
            _release("b9846"),
        ],
    )
    latest = adapter.latest_release()
    assert latest is not None
    assert latest.version == "b10867"


def test_latest_release_sorts_numerically_not_lexically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """b9846 sorts after b10867 as a string. It is an older build."""
    adapter = LlamaCppAdapter()
    monkeypatch.setattr(
        adapter.releases,
        "list_releases",
        lambda force=False: [_release("b9846"), _release("b10867")],
    )
    latest = adapter.latest_release()
    assert latest is not None
    assert latest.version == "b10867"


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buffer.getvalue()


class _FakeDownloads:
    """Serves archive bytes in place of the network."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_download(asset: ReleaseAsset, target: Path, progress: object) -> None:
            data = self.payloads[asset.name]
            target.write_bytes(data)
            progress.bytes_downloaded += len(data)  # type: ignore[attr-defined]

        monkeypatch.setattr(acq, "_download", fake_download)


def _asset_for(name: str, data: bytes) -> ReleaseAsset:
    return ReleaseAsset(
        name=name,
        url=f"https://example.invalid/{name}",
        size=len(data),
        digest="sha256:" + hashlib.sha256(data).hexdigest(),
    )


async def _run_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payloads: dict[str, bytes],
    *,
    version: str = "b10867",
    binary_name: str = "llama-server",
) -> tuple[EngineInstaller, ManagedStore]:
    _FakeDownloads(payloads).install(monkeypatch)
    store = ManagedStore(tmp_path, EngineKind.llama_cpp)
    installer = EngineInstaller(store, EngineKind.llama_cpp)
    plan = acq.AcquisitionPlan(
        version=version,
        variant="win-cuda-13.3-x64",
        assets=tuple(_asset_for(n, d) for n, d in payloads.items()),
        binary_name=binary_name,
    )
    installer.start(plan)
    assert installer._task is not None
    await installer._task
    return installer, store


async def test_install_verifies_extracts_and_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "llama-b10867-bin-win-cuda-13.3-x64.zip": _zip_bytes(
            {"llama-server.exe": b"MZ-not-really", "ggml.dll": b"lib"}
        ),
        "cudart-llama-bin-win-cuda-13.3-x64.zip": _zip_bytes({"cudart64_13.dll": b"cuda"}),
    }
    installer, store = await _run_install(tmp_path, monkeypatch, payloads)

    snapshot = installer.snapshot()
    assert snapshot is not None
    assert snapshot.state is State.done, snapshot.error
    assert snapshot.bytesDownloaded == sum(len(d) for d in payloads.values())

    current = store.current()
    assert current is not None
    assert current.version == "b10867"
    assert current.variant == "win-cuda-13.3-x64"
    assert current.binary.name == "llama-server.exe"
    # Both archives unpack into the same directory: the CUDA runtime has to
    # sit beside the executable or the binary cannot start.
    assert (current.directory / "cudart64_13.dll").is_file()
    # The archives themselves are removed; they are hundreds of megabytes.
    assert not list(current.directory.glob("*.zip"))


async def test_install_refuses_a_bad_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _zip_bytes({"llama-server": b"x"})
    _FakeDownloads({"a.zip": data}).install(monkeypatch)
    store = ManagedStore(tmp_path, EngineKind.llama_cpp)
    installer = EngineInstaller(store, EngineKind.llama_cpp)
    plan = acq.AcquisitionPlan(
        version="b1",
        variant="v",
        assets=(
            ReleaseAsset(
                name="a.zip",
                url="https://example.invalid/a.zip",
                size=len(data),
                digest="sha256:" + "9" * 64,
            ),
        ),
        binary_name="llama-server",
    )
    installer.start(plan)
    assert installer._task is not None
    await installer._task

    snapshot = installer.snapshot()
    assert snapshot is not None
    assert snapshot.state is State.failed
    assert "failed verification" in (snapshot.error or "")
    assert store.current() is None, "a build that failed verification must not be installed"


async def test_install_refuses_an_asset_with_no_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream publishes no checksum file; the digest on the asset
    metadata is the only verification available. Losing it means we
    cannot say what we fetched, so we don't install it."""
    data = _zip_bytes({"llama-server": b"x"})
    _FakeDownloads({"a.zip": data}).install(monkeypatch)
    installer = EngineInstaller(ManagedStore(tmp_path, EngineKind.llama_cpp), EngineKind.llama_cpp)
    plan = acq.AcquisitionPlan(
        version="b1",
        variant="v",
        assets=(ReleaseAsset(name="a.zip", url="https://example.invalid/a.zip", size=len(data)),),
        binary_name="llama-server",
    )
    installer.start(plan)
    assert installer._task is not None
    await installer._task
    snapshot = installer.snapshot()
    assert snapshot is not None
    assert snapshot.state is State.failed
    assert "no SHA-256 digest" in (snapshot.error or "")


async def test_install_fails_loudly_when_the_binary_is_not_in_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {"a.zip": _zip_bytes({"README.md": b"nope"})}
    installer, store = await _run_install(tmp_path, monkeypatch, payloads)
    snapshot = installer.snapshot()
    assert snapshot is not None
    assert snapshot.state is State.failed
    assert "not found in the extracted" in (snapshot.error or "")
    assert store.current() is None


async def test_binary_is_found_when_nested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Some archives nest under build/bin, and upstream has moved it."""
    payloads = {"a.zip": _zip_bytes({"build/bin/llama-server": b"elf"})}
    _, store = await _run_install(tmp_path, monkeypatch, payloads)
    current = store.current()
    assert current is not None
    assert current.binary.parts[-3:] == ("build", "bin", "llama-server")


async def test_extraction_refuses_a_traversal_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """These archives come off the internet. zipfile has no equivalent of
    tarfile's data filter, so the check is ours to make."""
    payloads = {"a.zip": _zip_bytes({"../escaped.txt": b"nope"})}
    installer, _ = await _run_install(tmp_path, monkeypatch, payloads)
    snapshot = installer.snapshot()
    assert snapshot is not None
    assert snapshot.state is State.failed
    assert "unsafe path" in (snapshot.error or "")
    assert not (tmp_path.parent / "escaped.txt").exists()


async def test_a_failed_install_leaves_the_previous_build_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole reason a build lands in a staging directory first: a
    runtime may be executing the current one right now."""
    good = {"a.zip": _zip_bytes({"llama-server": b"good"})}
    _, store = await _run_install(tmp_path, monkeypatch, good, version="b1")
    assert store.current() is not None

    bad = {"a.zip": _zip_bytes({"README.md": b"no binary here"})}
    installer, _ = await _run_install(tmp_path, monkeypatch, bad, version="b2")
    assert installer.snapshot().state is State.failed  # type: ignore[union-attr]

    current = store.current()
    assert current is not None
    assert current.version == "b1"
    assert current.binary.read_bytes() == b"good"


async def test_retention_keeps_current_and_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every build is not viable at half a gigabyte each; only the newest
    leaves a bad upgrade with no way back."""
    store = ManagedStore(tmp_path, EngineKind.llama_cpp)
    for version in ("b1", "b2", "b3"):
        await _run_install(
            tmp_path,
            monkeypatch,
            {"a.zip": _zip_bytes({"llama-server": version.encode()})},
            version=version,
        )
    versions = [b.version for b in store.list_builds()]
    assert len(versions) == 2
    assert set(versions) == {"b2", "b3"}


async def test_a_second_install_is_refused_while_one_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    started = asyncio.Event()
    release = asyncio.Event()

    def slow_download(asset: ReleaseAsset, target: Path, progress: object) -> None:
        started.set()
        # Block the worker thread until the test lets it go.
        asyncio.run(_wait(release))
        target.write_bytes(b"")

    async def _wait(event: asyncio.Event) -> None:
        await event.wait()

    monkeypatch.setattr(acq, "_download", slow_download)
    installer = EngineInstaller(ManagedStore(tmp_path, EngineKind.llama_cpp), EngineKind.llama_cpp)
    plan = acq.AcquisitionPlan(
        version="b1",
        variant="v",
        assets=(_asset_for("a.zip", b"x"),),
        binary_name="llama-server",
    )
    installer.start(plan)
    assert installer.running
    with pytest.raises(AcquisitionError, match="already running"):
        installer.start(plan)
    await installer.cancel()


def test_a_directory_without_metadata_is_not_a_build(tmp_path: Path) -> None:
    """An interrupted extraction, or somebody's own folder. Either way not
    something to hand a supervisor as an executable."""
    store = ManagedStore(tmp_path, EngineKind.llama_cpp)
    stray = store.build_dir("b1")
    stray.mkdir(parents=True)
    (stray / "llama-server").write_bytes(b"x")
    assert store.list_builds() == []


def test_metadata_binary_path_is_relative(tmp_path: Path) -> None:
    """So moving the engine root doesn't invalidate every install record."""
    store = ManagedStore(tmp_path, EngineKind.llama_cpp)
    directory = store.build_dir("b1")
    directory.mkdir(parents=True)
    binary = directory / "llama-server"
    binary.write_bytes(b"x")
    store.write_metadata(directory, version="b1", variant="v", binary=binary)
    meta = json.loads((directory / ManagedStore.METADATA_NAME).read_text(encoding="utf-8"))
    assert meta["binary"] == "llama-server"
    assert not Path(meta["binary"]).is_absolute()


# --------------------------------------------------------------------------- #
# Version reporting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("banner", "expected"),
    [
        # Through roughly b10000.
        ("version: 9846 (f708a5b2c)\nbuilt with Clang", "9846"),
        # b10867 onward: upstream started printing a semver alongside the
        # build. Scraping the old way yields "0.4.0-dev", which answers
        # none of the questions this field exists for and disagrees with
        # the bNNNN a managed install records from the release tag.
        ("version: 0.4.0-dev (build 10867, commit f3f1a8f27)", "10867"),
        ("build: 4589 (a1b2c3d4) with cc", "4589"),
    ],
)
def test_version_probe_reports_the_build_number_in_both_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, banner: str, expected: str
) -> None:
    import subprocess

    class _Completed:
        stdout = ""
        stderr = banner

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed())
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"x")
    assert LlamaCppAdapter().probe_version(binary) == expected
