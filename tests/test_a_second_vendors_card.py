"""A second vendor's card beside an NVIDIA one (2026-09-27).

llama.cpp's Windows CUDA and Vulkan builds share byte-identical core
libraries, and each backend is a plug-in DLL: the Vulkan build adds
exactly `ggml-vulkan.dll` (measured on b11211). Added to the CUDA build,
it gives one process that uses a discrete AMD or Intel card beside the
NVIDIA one. llama.cpp skips the Vulkan copy of a card CUDA already has,
by PCI id, and keeps an integrated GPU out while a card is present.

Measured the same day: a pin reaches its own backend only. With
`CUDA_VISIBLE_DEVICES=-1` the combined build loaded the model onto the
same 5090 through Vulkan. So a pinned runtime on that build is also given
an empty `GGML_VK_VISIBLE_DEVICES`, which hides every Vulkan device.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

import eugene_plexus_agent.engines.acquisition as acq
from eugene_plexus_agent._generated.models import (
    Accelerator,
    Arch,
    EngineKind,
    HostAccelerator,
    Os,
    RuntimeSpec,
    Secondary,
)
from eugene_plexus_agent.engines import gpu_probe
from eugene_plexus_agent.engines import host as host_mod
from eugene_plexus_agent.engines.acquisition import (
    AcquisitionError,
    EngineInstaller,
    ManagedStore,
    Release,
    ReleaseAsset,
    State,
    Unavailable,
)
from eugene_plexus_agent.engines.base import DiscoveredBinary, Origin
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter, split_variant

from .test_every_gpu_the_os_can_see import _B11211, RADEON_IGPU, RTX_5090, RX_7900

WIN_CUDA = HostAccelerator(
    os=Os.windows,
    arch=Arch.x64,
    accelerator=Accelerator.cuda,
    acceleratorVersion="13.4",
    computeCapability="12.0",
)


def _release() -> Release:
    return Release(
        version="b11211",
        published_at=datetime(2026, 9, 27, 3, 0, tzinfo=UTC),
        assets=tuple(
            ReleaseAsset(
                name=n, url=f"https://example.invalid/{n}", size=1024, digest="sha256:" + "0" * 64
            )
            for n in _B11211
        ),
    )


# --------------------------------------------------------------------------- #
# The choice
# --------------------------------------------------------------------------- #


def test_an_amd_card_beside_nvidia_chooses_the_combined_build() -> None:
    host = WIN_CUDA.model_copy(update={"secondary": Secondary.vulkan})
    plan = LlamaCppAdapter().plan_acquisition(host, _release())
    assert not isinstance(plan, Unavailable), getattr(plan, "reason", "")
    assert plan.variant == "win-cuda-13.4-x64+vulkan"
    assert [a.name for a in plan.assets] == [
        "llama-b11211-bin-win-cuda-13.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.4-x64.zip",
        "llama-b11211-bin-win-vulkan-x64.zip",
    ]
    assert plan.plugins == frozenset({"llama-b11211-bin-win-vulkan-x64.zip"})


def test_no_second_card_is_the_plain_cuda_build() -> None:
    plan = LlamaCppAdapter().plan_acquisition(WIN_CUDA, _release())
    assert not isinstance(plan, Unavailable)
    assert plan.variant == "win-cuda-13.4-x64"
    assert plan.plugins == frozenset()


def test_the_combined_build_can_be_chosen_by_hand() -> None:
    """For an integrated GPU as overflow, which is never the default."""
    plan = LlamaCppAdapter().plan_acquisition(
        WIN_CUDA, _release(), variant="win-cuda-12.4-x64+vulkan"
    )
    assert not isinstance(plan, Unavailable)
    assert plan.plugins == frozenset({"llama-b11211-bin-win-vulkan-x64.zip"})


def test_the_variant_splits_into_the_build_and_the_backend() -> None:
    assert split_variant("win-cuda-13.4-x64+vulkan") == ("win-cuda-13.4-x64", "win-vulkan-x64")
    assert split_variant("win-cuda-13.4-x64") == ("win-cuda-13.4-x64", None)


@pytest.fixture
def windows_nvidia(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(host_mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(host_mod.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(gpu_probe, "vulkan_loader_present", lambda os_name: True)

    def _with(adapters: list[gpu_probe.Adapter]) -> None:
        monkeypatch.setattr(gpu_probe, "adapters", lambda os_name=None: list(adapters))

    return _with


def test_the_picker_adds_vulkan_for_a_discrete_card(windows_nvidia) -> None:
    windows_nvidia([RTX_5090, RX_7900])
    assert host_mod._secondary(Os.windows, Arch.x64) is Secondary.vulkan


def test_an_integrated_gpu_beside_the_card_is_not_a_reason(windows_nvidia) -> None:
    windows_nvidia([RADEON_IGPU, RTX_5090])
    assert host_mod._secondary(Os.windows, Arch.x64) is None


def test_linux_is_never_combined(windows_nvidia) -> None:
    windows_nvidia([RTX_5090, RX_7900])
    assert host_mod._secondary(Os.linux, Arch.x64) is None


# --------------------------------------------------------------------------- #
# The install: merged, and refused when the cores differ
# --------------------------------------------------------------------------- #


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _asset(name: str, data: bytes) -> ReleaseAsset:
    return ReleaseAsset(
        name=name,
        url=f"https://example.invalid/{name}",
        size=len(data),
        digest="sha256:" + hashlib.sha256(data).hexdigest(),
    )


CUDA_ZIP = "llama-b11211-bin-win-cuda-13.4-x64.zip"
VULKAN_ZIP = "llama-b11211-bin-win-vulkan-x64.zip"
SHARED = {"llama-server.exe": b"MZ-server", "ggml-base.dll": b"core", "ggml.dll": b"ggml"}


async def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vulkan: dict[str, bytes]):
    payloads = {
        CUDA_ZIP: _zip({**SHARED, "ggml-cuda.dll": b"cuda"}),
        VULKAN_ZIP: _zip(vulkan),
    }

    def fake_download(asset: ReleaseAsset, target: Path, progress: object) -> None:
        target.write_bytes(payloads[asset.name])

    monkeypatch.setattr(acq, "_download", fake_download)
    store = ManagedStore(tmp_path, EngineKind.llama_cpp)
    installer = EngineInstaller(store, EngineKind.llama_cpp)
    installer.start(
        acq.AcquisitionPlan(
            version="b11211",
            variant="win-cuda-13.4-x64+vulkan",
            assets=tuple(_asset(n, d) for n, d in payloads.items()),
            binary_name="llama-server",
            plugins=frozenset({VULKAN_ZIP}),
        )
    )
    assert installer._task is not None
    await installer._task
    return installer, store


async def test_the_vulkan_backend_is_added_beside_the_cuda_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, store = await _install(
        tmp_path, monkeypatch, {**SHARED, "ggml-vulkan.dll": b"vulkan"}
    )
    snapshot = installer.snapshot()
    assert snapshot is not None and snapshot.state is State.done, snapshot and snapshot.error
    current = store.current()
    assert current is not None
    assert current.variant == "win-cuda-13.4-x64+vulkan"
    assert (current.directory / "ggml-cuda.dll").read_bytes() == b"cuda"
    assert (current.directory / "ggml-vulkan.dll").read_bytes() == b"vulkan"
    assert not list(current.directory.glob(".plugin-*"))


async def test_builds_whose_cores_differ_are_not_combined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The gate.** Two backends compiled against different cores in one
    process is a crash nobody could diagnose, so a release that breaks the
    identity is refused, naming the file."""
    installer, store = await _install(
        tmp_path,
        monkeypatch,
        {**SHARED, "ggml-base.dll": b"a different core", "ggml-vulkan.dll": b"vulkan"},
    )
    snapshot = installer.snapshot()
    assert snapshot is not None and snapshot.state is State.failed
    assert "ggml-base.dll" in (snapshot.error or "")
    assert store.current() is None


def test_an_archive_that_adds_nothing_is_refused(tmp_path: Path) -> None:
    build, side = tmp_path / "build", tmp_path / "side"
    build.mkdir()
    side.mkdir()
    (build / "ggml.dll").write_bytes(b"ggml")
    (side / "ggml.dll").write_bytes(b"ggml")
    with pytest.raises(AcquisitionError, match="added nothing"):
        acq._merge_backend(side, build, VULKAN_ZIP)


# --------------------------------------------------------------------------- #
# A pin on the combined build
# --------------------------------------------------------------------------- #


def _build(tmp_path: Path, *dlls: str) -> DiscoveredBinary:
    for name in ("llama-server.exe", *dlls):
        (tmp_path / name).write_bytes(b"x")
    return DiscoveredBinary(tmp_path / "llama-server.exe", Origin.managed)


def _spec(**overrides: object) -> RuntimeSpec:
    body: dict[str, object] = {"name": "m", "engine": "llama_cpp", "modelPath": "/m.gguf"}
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


def test_a_pinned_runtime_on_the_combined_build_sees_no_vulkan_device(tmp_path: Path) -> None:
    binary = _build(tmp_path, "ggml-cuda.dll", "ggml-vulkan.dll")
    env = LlamaCppAdapter().default_env(_spec(env={"CUDA_VISIBLE_DEVICES": "1"}), binary)
    assert env == {"GGML_VK_VISIBLE_DEVICES": ""}


@pytest.mark.parametrize(
    ("dlls", "overrides"),
    [
        (("ggml-cuda.dll", "ggml-vulkan.dll"), {}),  # not pinned: every card is wanted
        (("ggml-cuda.dll",), {"env": {"CUDA_VISIBLE_DEVICES": "1"}}),  # plain CUDA build
        (("ggml-vulkan.dll",), {"env": {"CUDA_VISIBLE_DEVICES": "1"}}),  # plain Vulkan build
        (
            ("ggml-cuda.dll", "ggml-vulkan.dll"),
            {"env": {"CUDA_VISIBLE_DEVICES": "1"}, "flags": {"devices": "CUDA0,Vulkan1"}},
        ),  # the operator named the devices
    ],
)
def test_otherwise_nothing_is_injected(tmp_path: Path, dlls, overrides) -> None:
    binary = _build(tmp_path, *dlls)
    assert LlamaCppAdapter().default_env(_spec(**overrides), binary) == {}


def test_the_device_list_reaches_llama_server(tmp_path: Path) -> None:
    binary = _build(tmp_path, "ggml-cuda.dll", "ggml-vulkan.dll")
    argv = LlamaCppAdapter().build_argv(_spec(flags={"devices": "CUDA0,Vulkan1"}), binary, 8090)
    at = argv.index("--device")
    assert argv[at + 1] == "CUDA0,Vulkan1"
