"""R1.3: what `contextSize` and `parallelSlots` actually do, measured.

Roadmap `docs/design/release-roadmap.md` §2.3, finding review §6.3 #36.

**The repo contradicted itself and nobody had asked the engine.** The
two `ConfigField` descriptions the profile form renders verbatim said
`-c` is per-slot and that memory multiplies with slots; `fit.py` and
`admission.py` compute the opposite. One of them was wrong regardless,
and which one depended on current `llama-server` semantics — which the
review marked Plausible and explicitly did **not** verify, because
`--kv-unified` changes them. So step one of this slice was a live launch
rather than a patch.

## The measurement

`llama-server` **b11001** (0.4.1-dev, commit f266648fa) on this box, a
real Qwen3-0.6B Q4_K_M, `-ngl 99`:

```
-c 32768 --parallel 1  ->  llama_context: n_ctx = 32768   slot n_ctx = 32768
-c 32768 --parallel 4  ->  llama_context: n_ctx = 32768   slot n_ctx =  8192  (x4)
                           load_model: n_slots = 4, n_ctx_slot = 8192,
                                       kv_unified = 'false'
```

**`n_ctx` — the allocated cache — is 32768 in both.** So `-c` is the
*total* KV budget and `--parallel` *divides* it into per-request
windows. Memory does not multiply with slots; the per-request window
shrinks.

**Therefore the arithmetic was right and the copy was wrong**, and the
copy was wrong in the expensive direction: it told an operator that
raising slots would cost them memory, so the honest response to "I want
4 concurrent requests" was to lower context — which is the one change
that actually does shrink each request's window, for no memory saved.

**And this is why a runtime reads back a context it was not given.**
`/props` exposes only `default_generation_settings.n_ctx`, the
*per-slot* number (`props.n_ctx` was absent on b11001), so a runtime
launched at 32768 with 4 slots reports `contextLength: 8192` — which
against a contract whose words are "a requested context larger than the
model or the available memory gets **clamped**" reads as "you did not
have the memory". Wrong diagnosis, wrong remedy. The field keeps its
meaning (what one request can use, which is what the truncation
detector needs); the admission reason says the division out loud.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eugene_plexus_agent import admission
from eugene_plexus_agent.admission import check_admission
from eugene_plexus_agent.engines import llama_cpp

from .conftest import fake_devices

GIB = 1024**3


def _spec(path: str, **overrides: object) -> object:
    from eugene_plexus_agent._generated.models import RuntimeSpec

    body: dict[str, object] = {"name": "m", "engine": "llama_cpp", "modelPath": path}
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


_SIZES: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _fake_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    import eugene_plexus_agent.admission as admission_module

    monkeypatch.setattr(admission_module, "model_size_bytes", lambda p: _SIZES.get(p))
    monkeypatch.setattr(admission_module, "path_exists", lambda p: True)


def _model(tmp_path: Path, size: int) -> str:
    path = str(tmp_path / f"model-{size}.gguf")
    _SIZES[path] = size
    return path


# -- the copy, against the measurement --------------------------------------


def _field(key: str) -> str:
    schema = llama_cpp.LlamaCppAdapter().flag_schema()
    for field in schema.fields:
        if field.key == key:
            return field.description or ""
    raise AssertionError(f"no {key!r} field in the llama.cpp flag schema")


def test_context_size_is_not_described_as_per_slot() -> None:
    """Measured: `-c 32768 --parallel 4` allocates n_ctx 32768 and gives
    each slot 8192. `-c` is the total, and the profile form rendered the
    opposite sentence verbatim."""
    text = _field("contextSize").lower()
    assert "per-slot" not in text and "per slot" not in text
    assert "total" in text or "divided" in text or "shared" in text


def test_parallel_slots_does_not_claim_memory_multiplies() -> None:
    """It said "4 slots at 8k needs roughly the memory of 1 slot at
    32k", which is backwards: 4 slots at 32k IS 1 slot at 32k, and each
    request gets 8k of it."""
    text = _field("parallelSlots").lower()
    assert "run out of vram" not in text
    assert "divide" in text or "divided" in text or "share" in text or "shares" in text


# -- the arithmetic, which was already right --------------------------------


@pytest.mark.parametrize("slots", [2, 4, 8])
async def test_required_memory_does_not_scale_with_slots(tmp_path: Path, slots: int) -> None:
    """A regression guard on the half that was correct.

    The obvious "fix" for the contradiction is to make the arithmetic
    match the copy and multiply the cache by the slot count. Measured,
    that would be wrong: `llama_context: n_ctx` is 32768 at one slot and
    at four. It would refuse a launch that fits, which is the failure
    mode this whole slice is about.
    """
    one = await check_admission(
        _spec(_model(tmp_path, 20 * GIB), flags={"contextSize": 32768, "parallelSlots": 1}),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
    )
    many = await check_admission(
        _spec(_model(tmp_path, 20 * GIB), flags={"contextSize": 32768, "parallelSlots": slots}),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
    )
    assert one.requiredBytes == many.requiredBytes
    assert one.decision is many.decision


# -- and the number a person actually gets ----------------------------------


def test_the_per_request_window_is_the_context_divided_by_the_slots() -> None:
    """`Runtime.capabilities.contextLength` comes back 8192 for a
    runtime launched at 32768 with 4 slots, and the contract's word for
    a number that differs from the request is "clamped" — which sends an
    operator looking for memory they are not short of."""
    assert admission.per_request_context(32768, 4) == 8192
    assert admission.per_request_context(32768, 1) == 32768
    assert admission.per_request_context(None, 4) is None
    assert admission.per_request_context(32768, None) == 32768


def test_the_admission_reason_names_the_divided_window(tmp_path: Path) -> None:
    note = admission.slot_division_note(32768, 4)
    assert note is not None
    assert "8,192" in note
    assert "4" in note


def test_one_slot_says_nothing_extra() -> None:
    """The default is one slot, and a sentence about division on every
    single-slot launch is noise that trains people to skip the reason."""
    assert admission.slot_division_note(32768, 1) is None
    assert admission.slot_division_note(32768, None) is None
    assert admission.slot_division_note(None, 4) is None


async def test_the_reason_carries_it_end_to_end(tmp_path: Path) -> None:
    """The note existing proves nothing about `check_admission` using
    it. Two sabotages escaped in S7 for exactly this reason."""
    result = await check_admission(
        _spec(_model(tmp_path, 2 * GIB), flags={"contextSize": 32768, "parallelSlots": 4}),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
    )
    assert "8,192" in result.reason


async def test_a_single_slot_launch_reads_as_before(tmp_path: Path) -> None:
    result = await check_admission(
        _spec(_model(tmp_path, 2 * GIB), flags={"contextSize": 32768}),
        snapshot=fake_devices(free=24 * GIB, total=32 * GIB),
        library=None,
        running=[],
    )
    assert "per request" not in result.reason
