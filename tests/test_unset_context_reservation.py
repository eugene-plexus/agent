"""An unset context reserves what llama.cpp will take, not 8,192 tokens of it.

**The finding (upstream drift audit, 2026-10-03; settings never lie).** A
llama.cpp runtime whose profile leaves `contextSize` to the engine was
reserved in the ledger at the size admission computed: with the library
down, the file plus a KV cache at an ASSUMED 8,192 tokens. But
llama-server with `--fit` on (every build the installers ship) does not
run at 8,192: `common/fit.cpp` (b11375) starts at the model's trained
context and shrinks it only until the card is full to `--fit-target`
(1024 MiB by default). So a 4.7 GB model on a card with 24 GiB free took
about 23 GiB while the ledger said about 6, and the next launch was told
`fits` for memory the first had already spent -- the exact defect the
ledger exists to prevent (R3.2).

When the library answered, admission had asked about the model's own
context, which can be larger than the card: then the ledger promised more
than the card has, and the engine takes the card less the margin.

The rule: for a llama.cpp launch whose context the engine sizes (context
unset, llama.cpp's fit on), reserve the smaller of the requirement and
the card's room less the margin; on the file-size path, whose requirement
was sized at an assumed context, the room itself. Anything else reserves
what admission measured, as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import eugene_plexus_agent.admission as admission_module
from eugene_plexus_agent._generated.models import (
    Admission,
    AdmissionBasis,
    AdmissionDecision,
    AdmissionFit,
    ComputeDevice,
    ComputeDeviceKind,
    RuntimeSpec,
)
from eugene_plexus_agent.admission import reservation_bytes

GIB = 1024**3
MIB = 1024**2
EIGHT_B_Q4 = 4_700_000_000
_SIZES: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _fake_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission_module, "model_size_bytes", lambda p: _SIZES.get(p))
    monkeypatch.setattr(admission_module, "path_exists", lambda p: True)


def _places(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    from eugene_plexus_agent.routes import runtimes as runtime_routes

    monkeypatch.setattr(runtime_routes, "engine_places", lambda spec, get_config: value)


def _model(tmp_path: Path, size: int, tag: str) -> str:
    path = str(tmp_path / f"model-{tag}.gguf")
    _SIZES[path] = size
    return path


def _body(name: str, path: str, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {"name": name, "engine": "llama_cpp", "modelPath": path}
    body.update(overrides)
    return body


# --- through the routes: what the next launch is told ----------------------


def test_an_unset_context_reserves_the_card_llama_cpp_will_fill(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE FINDING. The fake card has 24 GiB free; the first launch leaves
    its context to llama.cpp, which fills the card to the 1 GiB margin.
    A 2 GiB model asked about next must see that: it no longer `fits` on
    the card. (It is still admitted, as a launch llama.cpp places partly
    in system memory itself -- the `engine_places` rule, unchanged; with
    the defect it was told `fits` in memory already spent.)"""
    _places(monkeypatch, True)
    created = authed_client.post("/v1/runtimes", json=_body("a", _model(tmp_path, EIGHT_B_Q4, "a")))
    assert created.status_code == 201, created.text

    asked = authed_client.post(
        "/v1/runtimes/admission", json=_body("b", _model(tmp_path, 2 * GIB, "b"))
    )
    assert asked.status_code == 200, asked.text
    answer = asked.json()
    assert answer["reservedBytes"] == 24 * GIB - 1024 * MIB, answer["reason"]
    assert answer["fit"] != "fits", answer["reason"]


def test_the_margin_the_profile_sets_is_the_one_left(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _places(monkeypatch, True)
    body = _body("a", _model(tmp_path, EIGHT_B_Q4, "m"), flags={"memoryMargin": 4096})
    assert authed_client.post("/v1/runtimes", json=body).status_code == 201
    asked = authed_client.post(
        "/v1/runtimes/admission", json=_body("b", _model(tmp_path, 2 * GIB, "mb"))
    ).json()
    assert asked["reservedBytes"] == 24 * GIB - 4096 * MIB


def test_a_context_the_profile_sets_reserves_what_it_measured(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pair that tells the fix from the over-correction: a set context
    is run at that context, so the estimate stands and the 2 GiB model
    still fits beside it."""
    _places(monkeypatch, True)
    body = _body("a", _model(tmp_path, EIGHT_B_Q4, "s"), flags={"contextSize": 8192})
    assert authed_client.post("/v1/runtimes", json=body).status_code == 201
    asked = authed_client.post(
        "/v1/runtimes/admission", json=_body("b", _model(tmp_path, 2 * GIB, "sb"))
    ).json()
    assert 0 < asked["reservedBytes"] < 8 * GIB
    assert asked["decision"] == "admit", asked["reason"]


def test_a_build_without_fit_reserves_what_it_measured(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _places(monkeypatch, False)
    assert (
        authed_client.post(
            "/v1/runtimes", json=_body("a", _model(tmp_path, EIGHT_B_Q4, "n"))
        ).status_code
        == 201
    )
    asked = authed_client.post(
        "/v1/runtimes/admission", json=_body("b", _model(tmp_path, 2 * GIB, "nb"))
    ).json()
    assert 0 < asked["reservedBytes"] < 8 * GIB


def test_fit_switched_off_reserves_what_it_measured(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _places(monkeypatch, True)
    assert (
        authed_client.patch("/v1/config", json={"allowUnrestrictedEngineLaunch": True}).status_code
        == 200
    )
    body = _body("a", _model(tmp_path, EIGHT_B_Q4, "f"), extraArgs=["--fit", "off"])
    assert authed_client.post("/v1/runtimes", json=body).status_code == 201
    asked = authed_client.post(
        "/v1/runtimes/admission", json=_body("b", _model(tmp_path, 2 * GIB, "fb"))
    ).json()
    assert 0 < asked["reservedBytes"] < 8 * GIB


# --- the arithmetic ------------------------------------------------------------


def _admission(
    *, required: int, basis: AdmissionBasis, free: int = 24 * GIB, reserved: int | None = None
) -> Admission:
    return Admission(
        decision=AdmissionDecision.admit,
        fit=AdmissionFit.fits,
        basis=basis,
        requiredBytes=required,
        freeBytes=free,
        reservedBytes=reserved,
        totalBytes=32 * GIB,
        device=ComputeDevice(kind=ComputeDeviceKind.cuda, index=0, memoryFreeBytes=free),
        blockers=[],
        reason="",
    )


def _spec(**overrides: object) -> RuntimeSpec:
    body: dict[str, object] = {"name": "a", "engine": "llama_cpp", "modelPath": "/m.gguf"}
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


def test_metadata_sized_at_the_trained_context_is_capped_at_the_room() -> None:
    """The library sized it at the model's own context: when that fits,
    it is what is taken; when it does not, the card less the margin is."""
    fits = _admission(required=10 * GIB, basis=AdmissionBasis.metadata)
    assert reservation_bytes(_spec(), fits, engine_places=True) == 10 * GIB
    too_big = _admission(required=41 * GIB, basis=AdmissionBasis.metadata)
    assert reservation_bytes(_spec(), too_big, engine_places=True) == 23 * GIB


def test_room_already_promised_elsewhere_is_not_counted_twice() -> None:
    held = _admission(required=6 * GIB, basis=AdmissionBasis.file_size, reserved=5 * GIB)
    assert reservation_bytes(_spec(), held, engine_places=True) == 18 * GIB


def test_a_card_with_less_than_the_margin_free_takes_nothing_of_it() -> None:
    tiny = _admission(required=6 * GIB, basis=AdmissionBasis.file_size, free=512 * MIB)
    assert reservation_bytes(_spec(), tiny, engine_places=True) == 0


def test_a_context_in_the_raw_arguments_is_a_set_context() -> None:
    measured = _admission(required=6 * GIB, basis=AdmissionBasis.file_size)
    for args in (["-c", "4096"], ["--ctx-size", "4096"], ["--ctx-size=4096"]):
        assert reservation_bytes(_spec(extraArgs=args), measured, engine_places=True) == 6 * GIB


def test_other_engines_reserve_what_was_measured() -> None:
    measured = _admission(required=6 * GIB, basis=AdmissionBasis.file_size)
    vllm = RuntimeSpec.model_validate({"name": "v", "engine": "vllm", "modelPath": "/m"})
    assert reservation_bytes(vllm, measured, engine_places=True) == 6 * GIB
