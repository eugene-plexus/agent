"""Admission: refuse with the arithmetic, never queue, never on `unknown`.

Pure function tests over a shaped device snapshot. The library is a
fake here; the route-level wiring (detector injection, the 422, `force`)
is in test_runtimes.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import eugene_plexus_agent.admission as admission_module
from eugene_plexus_agent._generated.models import (
    AdmissionBasis,
    AdmissionDecision,
    AdmissionFit,
    EngineKind,
    RuntimeSpec,
    RuntimeStatus,
)
from eugene_plexus_agent.admission import (
    LibraryFit,
    RunningRuntime,
    check_admission,
    decide,
    local_verdict,
    pinned_indices,
    target_devices,
    wants_full_offload,
)

from .conftest import fake_devices

GIB = 1024**3


def _spec(path: str, **overrides: object) -> RuntimeSpec:
    body: dict[str, object] = {"name": "m", "engine": "llama_cpp", "modelPath": path}
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


_SIZES: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _fake_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Describe a 200 GiB file without creating one — Windows zero-fills
    a truncate, and the first run of this suite took four minutes."""
    monkeypatch.setattr(admission_module, "model_size_bytes", lambda p: _SIZES.get(p))


def _model(tmp_path: Path, size: int) -> str:
    path = str(tmp_path / f"model-{size}.gguf")
    _SIZES[path] = size
    return path


class _FakeLibrary:
    def __init__(self, answer: LibraryFit | None) -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    async def fit(self, model_path: str, **kwargs: object) -> LibraryFit | None:
        self.calls.append({"model_path": model_path, **kwargs})
        return self.answer


# --- pure helpers -----------------------------------------------------------


def test_pinned_indices_parse_the_integer_form_only() -> None:
    assert pinned_indices({"CUDA_VISIBLE_DEVICES": "1"}) == {1}
    assert pinned_indices({"CUDA_VISIBLE_DEVICES": "0,1"}) == {0, 1}
    assert pinned_indices({"HIP_VISIBLE_DEVICES": "2"}) == {2}
    # A UUID selector is legitimate for the engine and opaque to us.
    assert pinned_indices({"CUDA_VISIBLE_DEVICES": "GPU-0a949cc5"}) is None
    assert pinned_indices({}) is None
    assert pinned_indices(None) is None


def test_target_devices_follow_the_pin_and_fall_back_to_all() -> None:
    snapshot = fake_devices(count=2)
    assert [d.index for d in target_devices(_spec("/m"), snapshot)] == [0, 1]
    pinned = target_devices(_spec("/m", env={"CUDA_VISIBLE_DEVICES": "1"}), snapshot)
    assert [d.index for d in pinned] == [1]
    # A pin to a card this host lacks measures against everything.
    missing = target_devices(_spec("/m", env={"CUDA_VISIBLE_DEVICES": "7"}), snapshot)
    assert [d.index for d in missing] == [0, 1]


def test_a_host_with_no_accelerator_targets_the_cpu() -> None:
    snapshot = fake_devices(count=0)
    [cpu] = target_devices(_spec("/m"), snapshot)
    assert cpu.kind.value == "cpu"


def test_full_offload_is_the_default_and_partial_is_a_choice() -> None:
    assert wants_full_offload(_spec("/m")) is True
    assert wants_full_offload(_spec("/m", flags={"gpuLayers": 999})) is True
    # llama.cpp's idiom for everything, and what every profile writes.
    assert wants_full_offload(_spec("/m", flags={"gpuLayers": 99})) is True
    assert wants_full_offload(_spec("/m", flags={"gpuLayers": -1})) is True
    assert wants_full_offload(_spec("/m", flags={"gpuLayers": 98})) is False
    assert wants_full_offload(_spec("/m", flags={"gpuLayers": 20})) is False
    # vLLM has no partial offload.
    assert wants_full_offload(_spec("/m", engine="vllm", flags={"gpuLayers": 20})) is True


def test_local_verdict_covers_the_four_words() -> None:
    assert local_verdict(10, free=20, total=30, ram_available=100) is AdmissionFit.fits
    assert local_verdict(25, free=20, total=30, ram_available=100) is AdmissionFit.tight
    assert local_verdict(50, free=20, total=30, ram_available=100) is AdmissionFit.split
    assert local_verdict(500, free=20, total=30, ram_available=100) is AdmissionFit.no
    assert local_verdict(10, free=None, total=30, ram_available=100) is AdmissionFit.unknown


def test_the_decision_table() -> None:
    admit, refuse = AdmissionDecision.admit, AdmissionDecision.refuse
    assert decide(AdmissionFit.fits, full_offload=True) is admit
    assert decide(AdmissionFit.fits, full_offload=False) is admit
    assert decide(AdmissionFit.tight, full_offload=True) is refuse
    assert decide(AdmissionFit.tight, full_offload=False) is admit
    assert decide(AdmissionFit.split, full_offload=True) is refuse
    assert decide(AdmissionFit.split, full_offload=False) is admit
    assert decide(AdmissionFit.no, full_offload=True) is refuse
    assert decide(AdmissionFit.no, full_offload=False) is refuse
    assert decide(AdmissionFit.unknown, full_offload=True) is admit


# --- the decision, end to end ------------------------------------------------


@pytest.mark.anyio
async def test_a_small_model_is_admitted_by_file_size(tmp_path: Path) -> None:
    result = await check_admission(
        _spec(_model(tmp_path, 2 * GIB)),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
    )
    assert result.decision is AdmissionDecision.admit
    assert result.fit is AdmissionFit.fits
    assert result.basis is AdmissionBasis.file_size
    assert result.requiredBytes == int(2 * GIB * 1.1)
    assert result.freeBytes == 24 * GIB
    assert result.device is not None and result.device.index == 0
    assert result.reason.startswith("admit:")


@pytest.mark.anyio
async def test_a_model_larger_than_free_memory_is_refused_with_the_numbers(
    tmp_path: Path,
) -> None:
    result = await check_admission(
        _spec(_model(tmp_path, 30 * GIB)),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
    )
    assert result.decision is AdmissionDecision.refuse
    assert result.fit is AdmissionFit.split
    assert "33.0 GiB" in result.reason  # 30 GiB plus the allowance
    assert "24.0 GiB free of 32.0 GiB" in result.reason
    assert "?force=true" in result.reason


@pytest.mark.anyio
async def test_partial_offload_turns_split_into_an_admit(tmp_path: Path) -> None:
    result = await check_admission(
        _spec(_model(tmp_path, 30 * GIB), flags={"gpuLayers": 20}),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
    )
    assert result.decision is AdmissionDecision.admit
    assert result.fit is AdmissionFit.split
    assert "partial offload" in result.reason


@pytest.mark.anyio
async def test_no_is_refused_even_with_partial_offload(tmp_path: Path) -> None:
    result = await check_admission(
        _spec(_model(tmp_path, 200 * GIB), flags={"gpuLayers": 1}),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
    )
    assert result.decision is AdmissionDecision.refuse
    assert result.fit is AdmissionFit.no


@pytest.mark.anyio
async def test_the_library_answer_wins_and_is_labelled_metadata(tmp_path: Path) -> None:
    library = _FakeLibrary(
        LibraryFit(required_bytes=27 * GIB, verdict="split", context_length=131072)
    )
    result = await check_admission(
        _spec(_model(tmp_path, 2 * GIB), flags={"contextSize": 131072}),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=library,
        running=[],
    )
    assert result.basis is AdmissionBasis.metadata
    assert result.requiredBytes == 27 * GIB
    assert result.contextLength == 131072
    assert result.decision is AdmissionDecision.refuse
    # The budget the agent measured is what the library scored against.
    assert library.calls[0]["vram_bytes"] == 24 * GIB
    assert library.calls[0]["context_length"] == 131072


@pytest.mark.anyio
async def test_a_library_that_does_not_know_the_model_falls_back_to_file_size(
    tmp_path: Path,
) -> None:
    result = await check_admission(
        _spec(_model(tmp_path, 2 * GIB)),
        snapshot=fake_devices(),
        library=_FakeLibrary(None),
        running=[],
    )
    assert result.basis is AdmissionBasis.file_size
    assert result.decision is AdmissionDecision.admit


@pytest.mark.anyio
async def test_an_unsizable_model_is_admitted_on_faith_with_a_warning() -> None:
    result = await check_admission(
        _spec("/nowhere/model.gguf"),
        snapshot=fake_devices(),
        library=None,
        running=[],
    )
    assert result.decision is AdmissionDecision.admit
    assert result.fit is AdmissionFit.unknown
    assert result.warning is not None and "could not be sized" in result.warning
    assert result.reason.startswith("admit on faith")


@pytest.mark.anyio
async def test_no_device_at_all_never_refuses() -> None:
    snapshot = fake_devices(count=0)
    empty = snapshot.__class__(
        devices=(),
        warnings=("no vendor tool found",),
        ram_total_bytes=None,
        ram_available_bytes=None,
        detected_at=snapshot.detected_at,
    )
    result = await check_admission(_spec("/m"), snapshot=empty, library=None, running=[])
    assert result.decision is AdmissionDecision.admit
    assert result.fit is AdmissionFit.unknown
    assert result.warning is not None


@pytest.mark.anyio
async def test_blockers_name_what_holds_the_device_evictable_first(tmp_path: Path) -> None:
    others = [
        RunningRuntime(spec=_spec("/a", name="resident"), status=RuntimeStatus.ready),
        RunningRuntime(
            spec=_spec("/b", name="sleepy", idleUnloadSeconds=600), status=RuntimeStatus.ready
        ),
        RunningRuntime(spec=_spec("/c", name="stopped"), status=RuntimeStatus.stopped),
        RunningRuntime(
            spec=_spec("/d", name="other-card", env={"CUDA_VISIBLE_DEVICES": "1"}),
            status=RuntimeStatus.ready,
        ),
    ]
    result = await check_admission(
        _spec(_model(tmp_path, 30 * GIB), env={"CUDA_VISIBLE_DEVICES": "0"}),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB, count=2),
        library=None,
        running=others,
    )
    names = [b.name for b in result.blockers]
    # Evictable first, then the rest; the stopped one and the other card
    # do not hold this device.
    assert names == ["sleepy", "resident"]
    assert result.blockers[0].evictable is True
    assert result.blockers[0].idleUnloadSeconds == 600
    assert result.blockers[1].evictable is False
    assert "sleepy (ready, evictable)" in result.reason
    assert "Stop one of them" in result.reason


@pytest.mark.anyio
async def test_the_largest_free_card_is_the_one_measured(tmp_path: Path) -> None:
    snapshot = fake_devices(free=8 * GIB, total=32 * GIB, count=2)
    # Card 1 has more free memory than card 0.
    devices = list(snapshot.devices)
    devices[1] = devices[1].model_copy(update={"memoryFreeBytes": 20 * GIB})
    snapshot = snapshot.__class__(
        devices=tuple(devices),
        warnings=(),
        ram_total_bytes=snapshot.ram_total_bytes,
        ram_available_bytes=snapshot.ram_available_bytes,
        detected_at=snapshot.detected_at,
    )
    result = await check_admission(
        _spec(_model(tmp_path, 10 * GIB)), snapshot=snapshot, library=None, running=[]
    )
    assert result.device is not None and result.device.index == 1
    assert result.decision is AdmissionDecision.admit


@pytest.mark.anyio
async def test_vllm_is_always_full_offload(tmp_path: Path) -> None:
    result = await check_admission(
        _spec(_model(tmp_path, 30 * GIB), engine=EngineKind.vllm, flags={}),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
    )
    assert result.decision is AdmissionDecision.refuse
