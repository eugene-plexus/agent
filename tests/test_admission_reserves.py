"""R3.2: admission reserves nothing, and its fallback is context-blind.

Roadmap §4 item 2, review §6.2 #19. **Written before the fix and watched
to fail** — every check here reproduces the finding rather than
describing the repair.

Two independent halves.

**The fallback is context-blind.** `FILE_SIZE_ALLOWANCE` was a flat tenth
of the file whatever context the spec asked for, so an 8B Q4 at 128k —
about 17 GB once its KV cache is counted — was admitted at 5.5 GB. A
constant here is the same defect the library fixed in its own estimate
(`ESTIMATED_KV_FRACTION`'s docstring): the context control changes the
number echoed back and cannot change a verdict.

**Nothing reserves.** `check_admission` takes `running`, which looks like
it accounts for the other runtimes on the device and does not — it feeds
an advisory blocker list and nothing subtracts it. The arithmetic reads
**live free memory** off the `DeviceSnapshot`, so a runtime that is
declared, admitted and `starting` — or `copying`, which the node-local
copy made minutes long — holds no memory yet, free still reads high, and
the next admission says `fits` for memory the first one already spent.

The reservation checks are stated as two launches in quick succession,
because that is the shape the operator meets: press Launch on Home,
press Launch again while the first is still loading.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import eugene_plexus_agent.admission as admission_module
from eugene_plexus_agent._generated.models import (
    AdmissionBasis,
    AdmissionDecision,
    AdmissionFit,
    RuntimeSpec,
    RuntimeStatus,
)
from eugene_plexus_agent.admission import LibraryFit, RunningRuntime, check_admission
from eugene_plexus_agent.reservations import ReservationLedger

from .conftest import fake_devices

GIB = 1024**3

# An 8B Q4_K_M, as the roadmap names it: ~4.7 GB of weights whose KV
# cache at 128k is about 17 GB. The finding's own worked example.
EIGHT_B_Q4 = 4_700_000_000
LONG_CONTEXT = 131072


_SIZES: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _fake_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission_module, "model_size_bytes", lambda p: _SIZES.get(p))
    monkeypatch.setattr(admission_module, "path_exists", lambda p: True)


def _model(tmp_path: Path, size: int, tag: str = "") -> str:
    path = str(tmp_path / f"model-{tag or size}.gguf")
    _SIZES[path] = size
    return path


def _spec(name: str, path: str, **overrides: object) -> RuntimeSpec:
    body: dict[str, object] = {"name": name, "engine": "llama_cpp", "modelPath": path}
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


class _FakeLibrary:
    """A library that answers, so the reservation half is exercised on the
    `metadata` path too — the golden path, since one-click Run has one."""

    def __init__(self, required: int) -> None:
        self.required = required
        self.calls: list[dict[str, object]] = []

    async def fit(self, model_path: str, **kwargs: object) -> LibraryFit:
        self.calls.append({"model_path": model_path, **kwargs})
        vram = kwargs.get("vram_bytes")
        verdict = "fits" if isinstance(vram, int) and self.required <= vram else "no"
        return LibraryFit(
            required_bytes=self.required,
            verdict=verdict,
            context_length=None,
            size_bytes=None,
        )


# --- the small half: the fallback is context-blind ---------------------------


@pytest.mark.anyio
async def test_the_file_size_fallback_counts_the_kv_cache_it_was_asked_for(
    tmp_path: Path,
) -> None:
    """THE FINDING, half one: an 8B Q4 at 128k on a 12 GiB card.

    Weights are 4.7 GB, so a flat tenth asks for 5.2 GiB and says `fits`
    by a mile. The KV cache at 128k is about three times the weights
    again, and the launch OOMs minutes later with the admission record
    saying it was measured.

    A 12 GiB card rather than the 16 the roadmap names, because at 16
    the honest number lands at 15.9 GiB and the check would turn on a
    rounding margin instead of on the defect.
    """
    result = await check_admission(
        _spec("big-ctx", _model(tmp_path, EIGHT_B_Q4), flags={"contextSize": LONG_CONTEXT}),
        snapshot=fake_devices(free=12 * GIB, total=12 * GIB),
        library=None,
        running=[],
    )
    assert result.basis is AdmissionBasis.file_size
    assert result.decision is AdmissionDecision.refuse
    assert result.requiredBytes is not None and result.requiredBytes > 14 * GIB
    # And it says what it counted, so the operator can argue with it.
    assert str(LONG_CONTEXT) in result.reason


@pytest.mark.anyio
async def test_the_same_file_at_a_modest_context_is_still_admitted(tmp_path: Path) -> None:
    """The other side of the pair, and the reason this is not just a
    larger constant: the fix has to move with the context control, not
    refuse everything that used to pass."""
    result = await check_admission(
        _spec("small-ctx", _model(tmp_path, EIGHT_B_Q4), flags={"contextSize": 4096}),
        snapshot=fake_devices(free=12 * GIB, total=12 * GIB),
        library=None,
        running=[],
    )
    assert result.decision is AdmissionDecision.admit
    assert result.fit is AdmissionFit.fits


@pytest.mark.anyio
async def test_the_estimate_is_linear_in_the_context_asked_for(tmp_path: Path) -> None:
    """A constant made the context control inert. Three evenly spaced
    contexts, and the two differences are equal — which is what "linear"
    means and needs none of the module's constants restated here, so a
    test cannot agree with the code by copying it."""
    path = _model(tmp_path, EIGHT_B_Q4)
    required: list[int] = []
    for context in (8192, 16384, 24576):
        result = await check_admission(
            _spec(f"c{context}", path, flags={"contextSize": context}),
            snapshot=fake_devices(free=64 * GIB, total=64 * GIB),
            library=None,
            running=[],
        )
        assert result.requiredBytes is not None
        required.append(result.requiredBytes)
    first, second = required[1] - required[0], required[2] - required[1]
    assert first > 0
    assert abs(first - second) <= 1  # integer rounding, nothing more


@pytest.mark.anyio
async def test_a_spec_that_leaves_the_context_to_the_engine_assumes_one_and_says_so(
    tmp_path: Path,
) -> None:
    """`contextSize` unset is the commonest profile there is. The
    estimate needs a number, so it uses one and puts it on the wire
    rather than silently sizing the cache at zero."""
    result = await check_admission(
        _spec("unset", _model(tmp_path, EIGHT_B_Q4)),
        snapshot=fake_devices(free=64 * GIB, total=64 * GIB),
        library=None,
        running=[],
    )
    assert result.contextLength is not None and result.contextLength > 0
    assert result.requiredBytes is not None and result.requiredBytes > EIGHT_B_Q4


# --- the ledger: two launches in quick succession ----------------------------


@pytest.mark.anyio
async def test_a_second_launch_sees_the_memory_the_first_one_spent(tmp_path: Path) -> None:
    """THE FINDING, half two. Both launches read the same free memory.

    The first is admitted and starts; it holds nothing yet, because
    `llama-server` has not read a byte of the weights. The second asks
    the same question of the same card and is told `fits` for memory
    that is already spoken for.
    """
    ledger = ReservationLedger()
    snapshot = fake_devices(free=24 * GIB, total=32 * GIB)
    first = _spec("a", _model(tmp_path, 20 * GIB, "a"))

    one = await check_admission(
        first, snapshot=snapshot, library=None, running=[], reservations=ledger.entries()
    )
    assert one.decision is AdmissionDecision.admit
    ledger.reserve(
        "a",
        device_index=one.device.index if one.device else None,
        size_bytes=one.requiredBytes or 0,
    )

    # Seconds later: `starting`, no process memory yet, free unchanged.
    two = await check_admission(
        _spec("b", _model(tmp_path, 20 * GIB, "b")),
        snapshot=snapshot,
        library=None,
        running=[RunningRuntime(spec=first, status=RuntimeStatus.starting)],
        reservations=ledger.entries(),
    )
    assert two.decision is AdmissionDecision.refuse
    assert two.reservedBytes is not None and two.reservedBytes > 0
    # And it names the reservation rather than reporting a card that is
    # mysteriously full.
    assert "reserved" in two.reason


@pytest.mark.anyio
async def test_a_runtime_still_copying_holds_its_memory_too(tmp_path: Path) -> None:
    """`copying` is the window the node-local copy widened from seconds
    to minutes, and there is no process at all during it — which is
    exactly why `_RUNNING` did not list it either."""
    ledger = ReservationLedger()
    snapshot = fake_devices(free=24 * GIB, total=32 * GIB)
    first = _spec("a", _model(tmp_path, 20 * GIB, "ca"))
    ledger.reserve("a", device_index=0, size_bytes=22 * GIB)

    result = await check_admission(
        _spec("b", _model(tmp_path, 20 * GIB, "cb")),
        snapshot=snapshot,
        library=None,
        running=[RunningRuntime(spec=first, status=RuntimeStatus.copying)],
        reservations=ledger.entries(),
    )
    assert result.decision is AdmissionDecision.refuse
    # It is on the blocker list too: a copying runtime is something the
    # operator would stop to make room.
    assert [b.name for b in result.blockers] == ["a"]


@pytest.mark.anyio
async def test_the_library_scores_against_the_reserved_budget(tmp_path: Path) -> None:
    """The metadata path is the golden path, so the number handed to the
    library has to be the budget that is actually left."""
    ledger = ReservationLedger()
    ledger.reserve("a", device_index=0, size_bytes=20 * GIB)
    library = _FakeLibrary(required=10 * GIB)

    result = await check_admission(
        _spec("b", _model(tmp_path, 9 * GIB, "lb")),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=library,
        running=[],
        reservations=ledger.entries(),
    )
    assert library.calls[0]["vram_bytes"] == 4 * GIB
    assert result.decision is AdmissionDecision.refuse


@pytest.mark.anyio
async def test_a_reservation_for_this_very_runtime_is_not_counted_against_it(
    tmp_path: Path,
) -> None:
    """A restart re-measures the runtime that already holds a reservation.
    Counting its own promise against it refuses every restart."""
    ledger = ReservationLedger()
    ledger.reserve("a", device_index=0, size_bytes=20 * GIB)
    result = await check_admission(
        _spec("a", _model(tmp_path, 10 * GIB, "self")),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
        reservations=ledger.entries(),
    )
    assert result.decision is AdmissionDecision.admit


# --- the ledger's own rules --------------------------------------------------


def test_a_reservation_expires_so_an_abandoned_launch_cannot_strand_memory() -> None:
    """The TTL is the backstop and the reason the rest is testable: an
    agent that never observes the launch again must not hold the card
    for the life of the process."""
    now = [1000.0]
    ledger = ReservationLedger(ttl_seconds=600.0, clock=lambda: now[0])
    ledger.reserve("a", device_index=0, size_bytes=20 * GIB)
    assert ledger.held_bytes(device_index=0, exclude=None) == 20 * GIB
    now[0] += 599.0
    assert ledger.held_bytes(device_index=0, exclude=None) == 20 * GIB
    now[0] += 2.0
    assert ledger.held_bytes(device_index=0, exclude=None) == 0


def test_reconcile_releases_a_runtime_that_is_no_longer_on_its_way_up() -> None:
    """Released when the process is observed holding the memory — which
    is the moment the device snapshot starts counting it, and counting
    both is what would refuse a third launch that fits."""
    ledger = ReservationLedger()
    ledger.reserve("a", device_index=0, size_bytes=20 * GIB)
    ledger.reserve("b", device_index=0, size_bytes=1 * GIB)
    ledger.reconcile(pending={"b"})
    assert ledger.held_bytes(device_index=0, exclude=None) == 1 * GIB


def test_a_reservation_with_no_device_is_counted_everywhere() -> None:
    """Conservative on purpose: a launch we could not place still spends
    memory somewhere, and the refusing direction is the recoverable one."""
    ledger = ReservationLedger()
    ledger.reserve("a", device_index=None, size_bytes=8 * GIB)
    assert ledger.held_bytes(device_index=0, exclude=None) == 8 * GIB
    assert ledger.held_bytes(device_index=3, exclude=None) == 8 * GIB


def test_reserving_the_same_runtime_twice_replaces_rather_than_adds() -> None:
    ledger = ReservationLedger()
    ledger.reserve("a", device_index=0, size_bytes=8 * GIB)
    ledger.reserve("a", device_index=0, size_bytes=9 * GIB)
    assert ledger.held_bytes(device_index=0, exclude=None) == 9 * GIB


# --- the routes: a dry run must not reserve ----------------------------------
#
# `routes/runtimes.py` has ONE `check_admission` call site, shared by the
# dry run, a create and a start. The two are told apart there or not at
# all, so these drive the HTTP surface rather than the function.


def _body(name: str, path: str, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {"name": name, "engine": "llama_cpp", "modelPath": path}
    body.update(overrides)
    return body


def test_a_launch_reserves_and_the_next_dry_run_sees_it(
    authed_client: TestClient, tmp_path: Path
) -> None:
    """THE FINDING on the surface the UI uses. Launch, then ask.

    20 GiB declared and starting on a card with 24 free. The launch has
    not read a byte, so nothing is different about the card — and the
    next question has to come back refused anyway.
    """
    created = authed_client.post("/v1/runtimes", json=_body("a", _model(tmp_path, 20 * GIB, "ra")))
    assert created.status_code == 201, created.text

    asked = authed_client.post(
        "/v1/runtimes/admission", json=_body("b", _model(tmp_path, 20 * GIB, "rb"))
    )
    assert asked.status_code == 200, asked.text
    assert asked.json()["decision"] == "refuse"
    assert asked.json()["reservedBytes"] > 0


def test_a_dry_run_reserves_nothing_however_many_times_it_is_asked(
    authed_client: TestClient, tmp_path: Path
) -> None:
    """The rule that makes the ledger safe to wire into one call site:
    the dry run is a question. Asking it twice must not answer
    differently, or the Library's launch preview would refuse the launch
    it is previewing."""
    body = _body("a", _model(tmp_path, 20 * GIB, "dr"))
    first = authed_client.post("/v1/runtimes/admission", json=body)
    second = authed_client.post("/v1/runtimes/admission", json=body)
    assert first.json()["decision"] == "admit"
    assert second.json()["decision"] == "admit", second.text
    assert second.json().get("reservedBytes") in (None, 0)

    # The positive control, in the same check, because "nothing
    # reserved" is also what the defect answers: declare the very same
    # spec and the very same question comes back refused.
    assert authed_client.post("/v1/runtimes", json=body).status_code == 201
    third = authed_client.post(
        "/v1/runtimes/admission", json=_body("b", _model(tmp_path, 20 * GIB, "dr2"))
    )
    assert third.json()["decision"] == "refuse", third.text


def test_a_forced_launch_reserves_too(authed_client: TestClient, tmp_path: Path) -> None:
    """`force` is the operator overriding the verdict, not the launch
    becoming free. It spends the memory either way, and the next caller
    has to be told."""
    forced = authed_client.post(
        "/v1/runtimes",
        params={"force": "true"},
        json=_body("a", _model(tmp_path, 30 * GIB, "fa")),
    )
    assert forced.status_code == 201, forced.text
    asked = authed_client.post(
        "/v1/runtimes/admission", json=_body("b", _model(tmp_path, 2 * GIB, "fb"))
    )
    assert asked.json()["reservedBytes"] > 0


def test_stopping_a_runtime_hands_its_reservation_back(
    authed_client: TestClient, tmp_path: Path
) -> None:
    """A reservation is a promise about a launch. Stop the launch and the
    promise is over — waiting for the TTL would hold a card for ten
    minutes after the operator freed it."""
    authed_client.post("/v1/runtimes", json=_body("a", _model(tmp_path, 20 * GIB, "sa")))
    question = _body("b", _model(tmp_path, 2 * GIB, "sb"))
    # Held first, or "nothing is reserved" is a sentence the defect says too.
    assert authed_client.post("/v1/runtimes/admission", json=question).json()["reservedBytes"] > 0

    stopped = authed_client.post("/v1/runtimes/a/stop")
    assert stopped.status_code == 202, stopped.text
    asked = authed_client.post("/v1/runtimes/admission", json=question)
    assert asked.json().get("reservedBytes") in (None, 0)
    assert asked.json()["decision"] == "admit"


def test_deleting_a_runtime_hands_its_reservation_back(
    authed_client: TestClient, tmp_path: Path
) -> None:
    authed_client.post("/v1/runtimes", json=_body("a", _model(tmp_path, 20 * GIB, "da")))
    question = _body("b", _model(tmp_path, 2 * GIB, "db"))
    assert authed_client.post("/v1/runtimes/admission", json=question).json()["reservedBytes"] > 0

    assert authed_client.delete("/v1/runtimes/a").status_code == 204
    asked = authed_client.post("/v1/runtimes/admission", json=question)
    assert asked.json().get("reservedBytes") in (None, 0)


@pytest.mark.anyio
async def test_the_refusal_names_waiting_as_the_remedy_for_a_reservation(
    tmp_path: Path,
) -> None:
    """A reservation is the one blocker on the list that clears itself.
    Told to stop something, lower the context or pick a smaller quant,
    the operator goes looking for memory they are about to be handed
    back."""
    ledger = ReservationLedger()
    ledger.reserve("a", device_index=0, size_bytes=22 * GIB)
    result = await check_admission(
        _spec("b", _model(tmp_path, 20 * GIB, "wait")),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
        reservations=ledger.entries(),
    )
    assert result.decision is AdmissionDecision.refuse
    assert result.reason.endswith("launch anyway.")
    assert "Wait for the launch already under way" in result.reason
