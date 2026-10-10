"""Each engine owns its fit model (LS6, Troy's L11).

llama.cpp `spill`, vLLM `reserved_share`, Strata `engine_table` from its
setup's own rules; MLX and Kev declare none and are *not estimated*.
Admission measures a launch by the engine's own model, never another's.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import eugene_plexus_agent.admission as admission_module
from eugene_plexus_agent._generated.models import (
    AdmissionBasis,
    AdmissionDecision,
    AdmissionFit,
    ComputeDevice,
    ComputeDeviceKind,
    FitModelKind,
    FitVerdict,
    RuntimeSpec,
)
from eugene_plexus_agent.admission import LibraryFit, check_admission
from eugene_plexus_agent.engines.devices import DeviceSnapshot
from eugene_plexus_agent.engines.strata_models import (
    SETUP_CHOICES,
    fit_table,
    setup_fit,
)
from eugene_plexus_agent.engines.vllm import VllmAdapter, model_length
from eugene_plexus_agent.runtimes import describe_engines

from .conftest import fake_devices

GIB = 1024**3
IQ2 = SETUP_CHOICES["IQ2_XS"]
Q2 = SETUP_CHOICES["Q2_0"]
IQ3 = SETUP_CHOICES["IQ3_XXS"]
CODER = SETUP_CHOICES["coder-IQ1_M"]
UNSLOTH = SETUP_CHOICES["unsloth-UD-IQ4_XS"]
IQ2_FILE = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"


# --- Strata: setup's own `--check` rule --------------------------------------


def test_strata_fits_where_setups_ram_figure_is_met() -> None:
    # Amish_Station: 93.6 GB of RAM and a 32 GB card. IQ2_XS ran there.
    answer = setup_fit(IQ2, 93.6, 31.8)
    assert answer.verdict is FitVerdict.fits and answer.mode == "ram"
    assert "35.5 GB of experts in RAM" in answer.words and "48 GB" in answer.words


def test_strata_short_of_ram_takes_its_low_ram_mode_when_the_card_makes_up() -> None:
    # 44 GB: under IQ2_XS's experts + 10 GB, and the card holds the rest.
    answer = setup_fit(IQ2, 44, 32)
    assert answer.verdict is FitVerdict.split and answer.mode == "low_ram"
    assert "low-RAM mode" in answer.words


def test_strata_a_few_gb_short_without_the_low_ram_mode_is_tight() -> None:
    # Q2_0's experts (34 GB) + 10 fit 44 GB, so no low-RAM mode; 44 is within
    # 8 GB of its 48: setup says *tight*, the system pages.
    answer = setup_fit(Q2, 44, 32)
    assert answer.verdict is FitVerdict.tight and answer.mode == "paged"


def test_strata_far_short_is_no_and_names_both_numbers() -> None:
    answer = setup_fit(IQ3, 32, 16)
    assert answer.verdict is FitVerdict.no
    assert "60 GB" in answer.words and "32 GB" in answer.words


def test_strata_ram_budget_models_fit_or_not_by_their_ram_figure() -> None:
    fits = setup_fit(UNSLOTH, 64, 24)
    assert fits.verdict is FitVerdict.fits and fits.mode == "budget"
    # setup.py `resident_budget_gib`: round(64) - 24 = 40 GiB of experts.
    assert "40 GiB of its 59.5 GB" in fits.words
    assert setup_fit(UNSLOTH, 32, 24).verdict is FitVerdict.no


def test_strata_without_a_card_it_can_use_is_unknown() -> None:
    assert setup_fit(CODER, 64, None).verdict is FitVerdict.unknown


def test_strata_on_a_small_card_says_it_will_be_slow() -> None:
    assert "will be slow" in setup_fit(CODER, 64, 8).words
    assert "will be slow" not in setup_fit(CODER, 64, 24).words


def test_the_table_has_a_row_per_file_on_the_list() -> None:
    table = fit_table(int(93.6 * GIB), 32 * GIB)
    assert table.kind is FitModelKind.engine_table
    rows = {row.file: row for row in table.table or []}
    assert len(rows) == 9
    assert rows[IQ2_FILE].supportedModel == "IQ2_XS"
    assert rows[IQ2_FILE].fit.verdict is FitVerdict.fits
    assert rows[IQ2_FILE].fit.ramBytes == 48 * GIB
    unread = fit_table(None, 32 * GIB)
    assert all(not row.fit.estimated for row in unread.table or [])


# --- the descriptors ---------------------------------------------------------


def test_each_engine_declares_its_own_fit_model() -> None:
    snapshot = fake_devices(total=32 * GIB)
    by = {d.engine.value: d for d in describe_engines(devices=lambda: snapshot)}
    assert by["llama_cpp"].fit is not None and by["llama_cpp"].fit.kind is FitModelKind.spill
    vllm = by["vllm"].fit
    assert vllm is not None and vllm.kind is FitModelKind.reserved_share
    assert vllm.gpuMemoryUtilization == 0.92
    strata = by["strata"].fit
    assert strata is not None and strata.kind is FitModelKind.engine_table
    # The fake node: 64 GB of RAM and a 32 GB card. IQ3_S needs 62.
    rows = {row.supportedModel: row.fit.verdict for row in strata.table or []}
    assert rows["IQ2_XS"] is FitVerdict.fits and rows["IQ3_S"] is FitVerdict.fits
    assert by["mlx"].fit is None and by["kev"].fit is None


def test_the_devices_are_read_only_for_an_engine_that_needs_them() -> None:
    reads: list[int] = []

    def devices() -> DeviceSnapshot:
        reads.append(1)
        return fake_devices()

    describe_engines(devices=devices)
    assert len(reads) == 1


# --- admission by each engine's model ---------------------------------------


_SIZES: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _fake_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission_module, "model_size_bytes", lambda p: _SIZES.get(p))
    monkeypatch.setattr(admission_module, "path_exists", lambda p: True)


class _FakeLibrary:
    def __init__(self, answer: LibraryFit | None) -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    async def fit(self, model_path: str, **kwargs: object) -> LibraryFit | None:
        self.calls.append({"model_path": model_path, **kwargs})
        return self.answer


def _node(*, ram: int, available: int, card: int = 32 * GIB, free: int = 30 * GIB):
    snapshot = fake_devices(free=free, total=card)
    return replace(snapshot, ram_total_bytes=ram, ram_available_bytes=available)


def _prepared(tmp_path: Path, source: str | None = f"IQ2_XS/{IQ2_FILE}") -> str:
    path = tmp_path / "iq2_xs.eugene-prepared.json"
    body: dict[str, object] = {"engine": "strata", "entry": "strata-iq2_xs.json"}
    if source is not None:
        body["source"] = {"file": source}
    path.write_text(json.dumps(body), encoding="utf-8")
    return str(path)


def _strata(path: str) -> RuntimeSpec:
    return RuntimeSpec.model_validate({"name": "s", "engine": "strata", "modelPath": path})


async def test_strata_is_admitted_by_its_table_and_holds_its_card(tmp_path: Path) -> None:
    node = _node(ram=int(93.6 * GIB), available=65 * GIB)
    answer = await check_admission(
        _strata(_prepared(tmp_path)), snapshot=node, library=None, running=[]
    )
    assert answer.decision is AdmissionDecision.admit and answer.fit is AdmissionFit.fits
    assert answer.basis is AdmissionBasis.engine_table
    # It fills the card's free memory with its expert cache.
    assert answer.requiredBytes == 30 * GIB
    assert "IQ2_XS" in answer.reason


async def test_strata_short_of_free_ram_now_is_tight_and_refused(tmp_path: Path) -> None:
    # Setup's figure is met by the machine, but only 20 GiB is free now and
    # IQ2_XS copies 35.5 GB of experts into RAM.
    node = _node(ram=int(93.6 * GIB), available=20 * GIB)
    answer = await check_admission(
        _strata(_prepared(tmp_path)), snapshot=node, library=None, running=[]
    )
    assert answer.decision is AdmissionDecision.refuse and answer.fit is AdmissionFit.tight
    assert "free now" in answer.reason and "force" in answer.reason


async def test_strata_in_its_low_ram_mode_is_admitted(tmp_path: Path) -> None:
    node = _node(ram=44 * GIB, available=30 * GIB)
    answer = await check_admission(
        _strata(_prepared(tmp_path)), snapshot=node, library=None, running=[]
    )
    assert answer.decision is AdmissionDecision.admit and answer.fit is AdmissionFit.split


async def test_strata_that_does_not_fit_is_refused(tmp_path: Path) -> None:
    # A 32 GB card would make up for 16 GB of RAM (the low-RAM mode); a 16 GB
    # card does not: 16 - 6 + 11 is short of IQ2_XS's 35.5 GB of experts.
    node = _node(ram=16 * GIB, available=10 * GIB, card=16 * GIB, free=15 * GIB)
    answer = await check_admission(
        _strata(_prepared(tmp_path)), snapshot=node, library=None, running=[]
    )
    assert answer.decision is AdmissionDecision.refuse and answer.fit is AdmissionFit.no


async def test_strata_made_outside_eugene_is_admitted_on_faith(tmp_path: Path) -> None:
    node = _node(ram=16 * GIB, available=10 * GIB)
    answer = await check_admission(
        _strata(_prepared(tmp_path, source=None)), snapshot=node, library=None, running=[]
    )
    assert answer.decision is AdmissionDecision.admit and answer.fit is AdmissionFit.unknown
    assert answer.requiredBytes is None


async def test_an_engine_with_no_fit_model_is_never_measured_by_another(tmp_path: Path) -> None:
    path = str(tmp_path / "folder")
    _SIZES[path] = 500 * GIB  # far larger than the node: a spill arithmetic says `no`
    library = _FakeLibrary(LibraryFit(required_bytes=600 * GIB, verdict="no", context_length=8))
    for engine in ("mlx", "kev"):
        spec = RuntimeSpec.model_validate({"name": "x", "engine": engine, "modelPath": path})
        answer = await check_admission(spec, snapshot=fake_devices(), library=library, running=[])
        assert answer.decision is AdmissionDecision.admit and answer.fit is AdmissionFit.unknown
        assert "no memory estimate" in answer.reason
    assert library.calls == []


async def test_vllm_is_asked_about_its_share_at_its_own_context(tmp_path: Path) -> None:
    path = str(tmp_path / "hf-model")
    library = _FakeLibrary(
        LibraryFit(required_bytes=12 * GIB, verdict="tight", context_length=16384)
    )
    spec = RuntimeSpec.model_validate(
        {
            "name": "v",
            "engine": "vllm",
            "modelPath": path,
            "flags": {"maxModelLen": 16384, "gpuMemoryUtilization": 0.8},
        }
    )
    answer = await check_admission(
        spec, snapshot=fake_devices(free=20 * GIB, total=32 * GIB), library=library, running=[]
    )
    [call] = library.calls
    assert call["fit_model"] == "reserved_share"
    assert call["gpu_memory_utilization"] == 0.8
    assert call["vram_total_bytes"] == 32 * GIB
    assert call["context_length"] == 16384
    # Less than its share free: vLLM does not start, whatever gpuLayers says.
    assert answer.decision is AdmissionDecision.refuse
    assert "80% of the cards' total memory" in answer.reason
    assert "maxModelLen" in answer.reason and "gpuLayers" not in answer.reason
    # It takes its whole share: that is what it holds.
    assert answer.requiredBytes == int(32 * GIB * 0.8)


async def test_vllm_defaults_to_its_engines_own_share(tmp_path: Path) -> None:
    library = _FakeLibrary(LibraryFit(required_bytes=8 * GIB, verdict="fits", context_length=8))
    spec = RuntimeSpec.model_validate(
        {"name": "v", "engine": "vllm", "modelPath": str(tmp_path / "hf"), "flags": {}}
    )
    await check_admission(spec, snapshot=fake_devices(), library=library, running=[])
    assert library.calls[0]["gpu_memory_utilization"] == 0.92


def test_vllm_is_measured_at_its_own_flag_and_a_run_suggests_it_none() -> None:
    from eugene_plexus_agent.run_worker import takes_context_size

    assert model_length({"maxModelLen": 8192}) == 8192
    assert model_length({}) is None
    # A profile carrying `contextSize` is refused for vLLM at launch, so Run
    # suggests a context only to an engine whose flags have it.
    assert takes_context_size("llama_cpp") is True
    assert takes_context_size("vllm") is False
    spec = RuntimeSpec.model_validate(
        {"name": "v", "engine": "vllm", "modelPath": "/m", "flags": {"maxModelLen": 4096}}
    )
    argv = VllmAdapter().build_argv(spec, _binary(), port=9000)
    assert argv[argv.index("--max-model-len") + 1] == "4096"


def _binary():
    from eugene_plexus_agent._generated.models import Origin
    from eugene_plexus_agent.engines.base import DiscoveredBinary

    return DiscoveredBinary(path=Path("vllm"), origin=Origin.path)


def test_strata_cards_are_nvidia_and_amd_not_shared_memory() -> None:
    from eugene_plexus_agent.engines.strata import strata_card

    def snap(*devices: ComputeDevice) -> DeviceSnapshot:
        return DeviceSnapshot(
            devices=devices,
            warnings=(),
            ram_total_bytes=GIB,
            ram_available_bytes=GIB,
            detected_at=datetime.now(UTC),
        )

    small = ComputeDevice(kind=ComputeDeviceKind.cuda, index=0, memoryTotalBytes=8 * GIB)
    big = ComputeDevice(kind=ComputeDeviceKind.rocm, index=1, memoryTotalBytes=24 * GIB)
    shared = ComputeDevice(
        kind=ComputeDeviceKind.cuda, index=2, memoryTotalBytes=96 * GIB, sharedMemory=True
    )
    vulkan = ComputeDevice(kind=ComputeDeviceKind.vulkan, index=3, memoryTotalBytes=48 * GIB)
    card = strata_card(snap(small, big, shared, vulkan))
    assert card is not None and card.index == 1
    assert strata_card(snap(vulkan)) is None


async def test_the_library_client_asks_by_the_engines_fit_model() -> None:
    import httpx

    from eugene_plexus_agent.admission import LibraryFitClient

    asked: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": [{"id": "m1", "contextLength": 8192}]})
        asked.update(request.url.params)
        return httpx.Response(
            200, json={"fit": {"verdict": "fits", "requiredBytes": GIB, "contextLength": 8192}}
        )

    client = LibraryFitClient("http://library", None, transport=httpx.MockTransport(handler))
    await client.fit(
        "/models/hf",
        context_length=None,
        vram_bytes=20 * GIB,
        ram_bytes=None,
        fit_model="reserved_share",
        gpu_memory_utilization=0.92,
        vram_total_bytes=32 * GIB,
    )
    assert asked["fitModel"] == "reserved_share"
    assert asked["gpuMemoryUtilization"] == "0.92"
    assert asked["vramTotalBytes"] == str(32 * GIB)
    asked.clear()
    await client.fit("/models/hf", context_length=None, vram_bytes=20 * GIB, ram_bytes=None)
    assert "fitModel" not in asked and "vramTotalBytes" not in asked
