"""One model across several cards (2026-09-27).

llama.cpp splits a model's layers across every visible card by default,
so two 5090s run a model neither can hold alone entirely in GPU memory.
Admission took the largest single card and called that model `split`
(spilling into system RAM), which a full-offload launch refuses. Its own
docstring said so, on reasoning that was wrong about what `split` means.

Now `place` decides which cards a launch uses: every card of the build's
kind for llama.cpp unless pinned or `splitMode: none`, and
`tensorParallelSize` x `pipelineParallelSize` for vLLM. The verdict is
computed against what those cards have left between them, each card's
share follows `tensorSplit` or its free memory, and the reservation is
divided the same way.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

import eugene_plexus_agent.admission as admission_module
from eugene_plexus_agent._generated.models import (
    AdmissionDecision,
    AdmissionFit,
    ComputeDevice,
    ComputeDeviceKind,
    RuntimeSpec,
)
from eugene_plexus_agent.admission import (
    OVERHEAD_BYTES,
    LibraryFit,
    check_admission,
    place,
    split_shares,
    spread_devices,
)
from eugene_plexus_agent.engines import devices as devices_mod
from eugene_plexus_agent.engines.devices import DeviceSnapshot
from eugene_plexus_agent.reservations import Reservation, held_bytes

GIB = 1024**3


def _card(
    index: int, free_gib: float, kind: ComputeDeviceKind = ComputeDeviceKind.cuda
) -> ComputeDevice:
    return ComputeDevice(
        kind=kind,
        index=index,
        name=f"NVIDIA GeForce RTX 5090 #{index}",
        memoryTotalBytes=32 * GIB,
        memoryFreeBytes=int(free_gib * GIB),
    )


def _snapshot(*cards: ComputeDevice) -> DeviceSnapshot:
    cpu = ComputeDevice(
        kind=ComputeDeviceKind.cpu,
        index=0,
        name="CPU",
        memoryTotalBytes=96 * GIB,
        memoryFreeBytes=60 * GIB,
    )
    return DeviceSnapshot(
        devices=(*cards, cpu),
        warnings=(),
        ram_total_bytes=96 * GIB,
        ram_available_bytes=60 * GIB,
        detected_at=datetime.now(UTC),
    )


TWO_5090S = _snapshot(_card(0, 30), _card(1, 30))


def _spec(size_gib: float, **overrides: object) -> RuntimeSpec:
    body: dict[str, object] = {
        "name": "big",
        "engine": "llama_cpp",
        "modelPath": f"/models/big-{size_gib}.gguf",
    }
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


@pytest.fixture(autouse=True)
def _sized(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file of the size its name says, without writing 40 GiB."""

    def size(path: str) -> int | None:
        return int(float(path.rsplit("-", 1)[1].removesuffix(".gguf")) * GIB)

    monkeypatch.setattr(admission_module, "model_size_bytes", size)
    monkeypatch.setattr(admission_module, "path_exists", lambda p: True)


class _Library:
    """Scores the way the library does: weights, a small cache and one
    allowance per card, against the budget it is handed."""

    def __init__(self, weights_gib: float) -> None:
        self.weights = int(weights_gib * GIB)
        self.calls: list[dict[str, object]] = []

    async def fit(self, model_path: str, **kwargs: object) -> LibraryFit:
        self.calls.append(kwargs)
        cards = int(kwargs.get("gpu_count") or 1)  # type: ignore[call-overload]
        required = self.weights + 1 * GIB + cards * OVERHEAD_BYTES
        vram = int(kwargs["vram_bytes"] or 0)  # type: ignore[call-overload]
        verdict = "fits" if required <= vram else "split"
        return LibraryFit(required_bytes=required, verdict=verdict, context_length=16384)


# --------------------------------------------------------------------------- #
# The finding
# --------------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_a_model_neither_card_can_hold_is_admitted_across_both() -> None:
    """**The finding.** 44 GiB on two 5090s with 30 free each: refused as
    `split` against one card, while llama.cpp would have served it
    entirely in GPU memory."""
    library = _Library(44)
    result = await check_admission(_spec(44), snapshot=TWO_5090S, library=library, running=[])
    assert result.decision is AdmissionDecision.admit, result.reason
    assert result.fit is AdmissionFit.fits
    [call] = library.calls
    assert call["vram_bytes"] == 60 * GIB
    assert call["gpu_count"] == 2
    assert [d.index for d in result.devices or []] == [0, 1]
    assert (result.freeBytes, result.totalBytes) == (60 * GIB, 64 * GIB)
    assert "2 cards (device 0" in result.reason and "free of 64.0 GiB between them" in result.reason


@pytest.mark.anyio
async def test_without_the_library_each_card_still_gets_its_allowance() -> None:
    """The file-size fallback: two cards, two allowances. The second one
    is what the second card's compute buffers need."""
    alone = await check_admission(
        _spec(10, flags={"splitMode": "none"}), snapshot=TWO_5090S, library=None, running=[]
    )
    split = await check_admission(_spec(10), snapshot=TWO_5090S, library=None, running=[])
    assert alone.requiredBytes is not None and split.requiredBytes is not None
    assert split.requiredBytes - alone.requiredBytes == OVERHEAD_BYTES


@pytest.mark.anyio
async def test_a_model_too_big_for_both_is_still_refused() -> None:
    result = await check_admission(_spec(70), snapshot=TWO_5090S, library=_Library(70), running=[])
    assert result.decision is AdmissionDecision.refuse
    assert "between them" in result.reason


# --------------------------------------------------------------------------- #
# Which cards a launch uses
# --------------------------------------------------------------------------- #


def test_a_runtime_pinned_to_one_card_is_one_card() -> None:
    spec = _spec(10, env={"CUDA_VISIBLE_DEVICES": "1"})
    targets = admission_module.target_devices(spec, TWO_5090S)
    assert spread_devices(spec, targets) == []
    assert place(spec, targets).main.index == 1


def test_pinned_to_both_is_both() -> None:
    spec = _spec(10, env={"CUDA_VISIBLE_DEVICES": "0,1"})
    targets = admission_module.target_devices(spec, TWO_5090S)
    assert [d.index for d in spread_devices(spec, targets)] == [0, 1]


def test_one_gpu_only_is_one_card() -> None:
    spec = _spec(10, flags={"splitMode": "none"})
    assert spread_devices(spec, list(TWO_5090S.accelerators())) == []


def test_vllm_uses_as_many_cards_as_it_is_told() -> None:
    cards = list(TWO_5090S.accelerators())
    assert spread_devices(_spec(10, engine="vllm"), cards) == []
    tp = _spec(10, engine="vllm", flags={"tensorParallelSize": 2})
    assert [d.index for d in spread_devices(tp, cards)] == [0, 1]
    # Tensor parallelism shards evenly, so the smaller card bounds it.
    uneven = [_card(0, 30), _card(1, 10)]
    assert place(tp, uneven).budget == 20 * GIB


def test_a_llama_cpp_split_takes_every_card_the_build_uses_whatever_its_kind() -> None:
    """**Amended 2026-09-27**, from "cards of another kind are not in the
    split". The device list now names exactly what the build uses, and the
    `+vulkan` build uses an AMD card beside NVIDIA through its second
    backend; the case that test guarded (a ROCm card the CUDA build cannot
    reach) no longer reaches the list at all."""
    mixed = [_card(0, 30), _card(0, 20, ComputeDeviceKind.vulkan)]
    spread = spread_devices(_spec(10), mixed)
    assert [(d.kind.value, d.index) for d in spread] == [("cuda", 0), ("vulkan", 0)]
    assert place(_spec(10), mixed).budget == 50 * GIB


def test_vllm_splits_across_its_own_kind_only() -> None:
    """vLLM has no Vulkan backend."""
    tp = _spec(10, engine="vllm", flags={"tensorParallelSize": 2})
    mixed = [_card(0, 30), _card(0, 20, ComputeDeviceKind.vulkan), _card(1, 30)]
    assert [(d.kind.value, d.index) for d in spread_devices(tp, mixed)] == [
        ("cuda", 0),
        ("cuda", 1),
    ]


def test_cuda_0_and_vulkan_0_are_two_cards() -> None:
    """A promise on CUDA device 0 is not a promise on Vulkan device 0."""
    held = [
        Reservation(runtime="a", device_index=0, device_kind="cuda", size_bytes=20 * GIB, at=0.0)
    ]
    assert held_bytes(held, device_index=0, device_kind="vulkan", exclude=None) == 0
    assert held_bytes(held, device_index=0, device_kind="cuda", exclude=None) == 20 * GIB
    # A reservation from before kinds were recorded matches any.
    legacy = [Reservation(runtime="b", device_index=0, size_bytes=5 * GIB, at=0.0)]
    assert held_bytes(legacy, device_index=0, device_kind="vulkan", exclude=None) == 5 * GIB


def test_a_pin_reaches_its_own_kind_only() -> None:
    both = _snapshot(_card(0, 30), _card(1, 30), _card(1, 20, ComputeDeviceKind.vulkan))
    targets = admission_module.target_devices(_spec(10, env={"CUDA_VISIBLE_DEVICES": "1"}), both)
    assert [(d.kind.value, d.index) for d in targets] == [("cuda", 1)]


def test_an_explicit_cuda_list_is_measured_and_a_vulkan_name_is_not() -> None:
    cards = list(TWO_5090S.accelerators())
    cuda = _spec(10, flags={"devices": "CUDA1"})
    assert [d.index for d in admission_module.target_devices(cuda, TWO_5090S)] == [1]
    igpu = _spec(10, flags={"devices": "CUDA0,Vulkan1"})
    assert place(igpu, admission_module.target_devices(igpu, TWO_5090S)).budget is None
    assert place(cuda, cards).budget is not None


@pytest.mark.anyio
async def test_a_launch_naming_a_vulkan_device_is_admitted_unmeasured() -> None:
    """The integrated-GPU overflow test path: not judged, not refused."""
    result = await check_admission(
        _spec(40, flags={"devices": "CUDA0,Vulkan1"}),
        snapshot=TWO_5090S,
        library=None,
        running=[],
    )
    assert result.decision is AdmissionDecision.admit
    assert result.fit is AdmissionFit.unknown


def test_two_cards_and_a_vulkan_name_is_still_unmeasured() -> None:
    """Both 5090s plus the integrated GPU as overflow: the two CUDA cards
    are a split, and the Vulkan name still leaves nothing honest to
    measure the whole against. The first sabotage pass found this path
    untested."""
    spec = _spec(10, flags={"devices": "CUDA0,CUDA1,Vulkan2"})
    targets = admission_module.target_devices(spec, TWO_5090S)
    placement = place(spec, targets)
    assert placement.spread
    assert placement.budget is None


def test_device_none_is_the_processor() -> None:
    targets = admission_module.target_devices(_spec(10, flags={"devices": "none"}), TWO_5090S)
    assert [d.kind.value for d in targets] == ["cpu"]


def test_the_main_gpu_is_the_main_card() -> None:
    placement = place(_spec(10, flags={"mainGpu": 1}), list(TWO_5090S.accelerators()))
    assert placement.main.index == 1
    assert placement.spread


# --------------------------------------------------------------------------- #
# How the model divides
# --------------------------------------------------------------------------- #


def test_by_default_each_card_takes_its_share_of_the_room() -> None:
    cards = [_card(0, 30), _card(1, 10)]
    shares = split_shares(_spec(10), cards, lambda d: d.memoryFreeBytes)
    assert shares == pytest.approx([0.75, 0.25])
    assert place(_spec(10), cards).budget == 40 * GIB


@pytest.mark.anyio
async def test_a_lopsided_tensor_split_is_bounded_by_the_card_that_fills_first() -> None:
    """0.9 / 0.1 on two equal cards: card 0 takes 90% of the model, so the
    most the model can be is 30 / 0.9 = 33 GiB, not the 60 GiB between them."""
    spec = _spec(40, flags={"tensorSplit": "0.9,0.1"})
    assert place(spec, list(TWO_5090S.accelerators())).budget == int(30 * GIB / 0.9)
    result = await check_admission(spec, snapshot=TWO_5090S, library=_Library(40), running=[])
    assert result.decision is AdmissionDecision.refuse


def test_a_tensor_split_shorter_than_the_cards_leaves_the_rest_out() -> None:
    """llama.cpp's reading: a missing proportion is zero."""
    three = [_card(0, 30), _card(1, 30), _card(2, 30)]
    placement = place(_spec(10, flags={"tensorSplit": "1,1"}), three)
    assert [d.index for d in placement.devices] == [0, 1]


def test_what_is_promised_on_a_card_is_taken_off_its_share() -> None:
    held = [Reservation(runtime="other", device_index=0, size_bytes=20 * GIB, at=0.0)]
    placement = place(_spec(10), list(TWO_5090S.accelerators()), held)
    assert placement.reserved == 20 * GIB
    assert placement.budget == 40 * GIB  # 10 left on card 0, 30 on card 1


def test_a_split_promise_counts_on_each_card_by_its_share() -> None:
    split = Reservation(
        runtime="big",
        device_index=0,
        size_bytes=40 * GIB,
        at=0.0,
        shares=(("cuda", 0, 30 * GIB), ("cuda", 1, 10 * GIB)),
    )
    assert held_bytes([split], device_index=0, exclude=None) == 30 * GIB
    assert held_bytes([split], device_index=1, exclude=None) == 10 * GIB
    assert held_bytes([split], device_index=None, exclude=None) == 40 * GIB
    assert held_bytes([split], device_index=0, exclude="big") == 0


# --------------------------------------------------------------------------- #
# The expert's knob, and the probe it needs
# --------------------------------------------------------------------------- #


def test_split_mode_reaches_llama_server() -> None:
    from eugene_plexus_agent.engines.llama_cpp import _FLAG_CLI_NAMES, _FLAG_FIELDS

    assert _FLAG_CLI_NAMES["splitMode"] == "--split-mode"
    [field] = [f for f in _FLAG_FIELDS if f.key == "splitMode"]
    assert field.enumValues == ["layer", "row", "tensor", "none"]


def test_the_device_probe_runs_the_program_it_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """R2.3's fix, which `devices._run` never got: a bare name let
    CreateProcess run System32's copy whatever `which` had found."""
    ran: list[list[str]] = []
    monkeypatch.setattr(devices_mod.shutil, "which", lambda name: r"C:\stub\nvidia-smi.cmd")

    def _record(argv, **kwargs):  # type: ignore[no-untyped-def]
        ran.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(devices_mod.subprocess, "run", _record)
    devices_mod._run(["nvidia-smi", "--query-gpu=index"])
    assert ran == [[r"C:\stub\nvidia-smi.cmd", "--query-gpu=index"]]


def test_the_ledger_takes_a_split_promise(tmp_path: Path) -> None:
    from eugene_plexus_agent.reservations import ReservationLedger

    ledger = ReservationLedger()
    ledger.reserve(
        "big",
        device_index=0,
        size_bytes=40 * GIB,
        shares=[("cuda", 0, 30 * GIB), ("cuda", 1, 10 * GIB)],
    )
    assert ledger.held_bytes(device_index=1, exclude=None) == 10 * GIB
