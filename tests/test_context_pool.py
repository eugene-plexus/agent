"""Which llama-server slots share one context (CB3).

`/props` reports the per-slot `n_ctx` and `total_slots`, never whether the
pool is unified, so the agent decides from the argv it launched. Measured
on b11211 with a 1B at `-c 16384` (2026-10-02):

    automatic slots           n_ctx 16384, 4 slots, kv_unified = 'true'
    --parallel 4              n_ctx  4096, 4 slots, kv_unified = 'false'
    --parallel 4 --kv-unified n_ctx 16384, 4 slots, kv_unified = 'true'
    --parallel 1              n_ctx 16384, 1 slot,  kv_unified = 'false'
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eugene_plexus_agent._generated.models import Origin, RuntimeCapabilities, RuntimeSpec
from eugene_plexus_agent.companions import render_config
from eugene_plexus_agent.engines import LlamaCppAdapter
from eugene_plexus_agent.engines.base import DiscoveredBinary

BASE = ["llama-server", "-m", "model.gguf", "-c", "16384", "--port", "8081"]


@pytest.mark.parametrize(
    "extra,env,n_ctx,expected",
    [
        ([], None, 16384, 16384),
        (["--parallel", "4"], None, 4096, None),
        (["--parallel", "4", "--kv-unified"], None, 16384, 16384),
        (["-np", "4", "-kvu"], None, 16384, 16384),
        (["--parallel=4"], None, 4096, None),
        (["--parallel", "1"], None, 16384, None),
        (["--parallel", "-1"], None, 16384, 16384),
        (["--no-kv-unified"], None, 4096, None),
        (["--kv-unified", "--no-kv-unified"], None, 4096, None),
        ([], {"LLAMA_ARG_N_PARALLEL": "4"}, 4096, None),
        (["--parallel", "4"], {"LLAMA_ARG_KV_UNIFIED": "1"}, 16384, 16384),
        (["--kv-unified-per-slot", "8192"], None, 8192, None),
    ],
)
def test_the_argv_says_whether_the_slots_share(extra, env, n_ctx, expected):
    capabilities = RuntimeCapabilities(contextLength=n_ctx, parallelSlots=4)
    assert LlamaCppAdapter().context_pool(capabilities, BASE + extra, env) == expected


def test_no_context_reported_is_no_pool():
    assert LlamaCppAdapter().context_pool(RuntimeCapabilities(), BASE, None) is None


# --- CB4: slot pinning, the engine and the driver together -----------------------


def _spec(flags: dict) -> RuntimeSpec:
    return RuntimeSpec(name="r", engine="llama_cpp", modelPath="/m/model.gguf", flags=flags)


def _argv(flags: dict) -> list[str]:
    binary = DiscoveredBinary(path=Path("llama-server"), origin=Origin.path, version="b11211")
    return LlamaCppAdapter().build_argv(_spec(flags), binary, 8081)


def test_pinning_off_by_default_changes_nothing():
    field = next(f for f in LlamaCppAdapter().flag_schema().fields if f.key == "slotPinning")
    assert field.default is False
    assert "--no-cache-idle-slots" not in _argv({})
    assert LlamaCppAdapter().companion_overrides(_spec({})) == {}
    assert render_config(runtime_name="r", alias="m")["slotPinning"] is None


def test_pinning_on_starts_the_engine_and_the_driver_together():
    """Pinned without `--no-cache-idle-slots`, each new task clears the
    idle slots the driver points at (upstream #28139); the engine flag
    without the driver's map pins nothing. One flag, both halves."""
    assert "--no-cache-idle-slots" in _argv({"slotPinning": True})
    overrides = LlamaCppAdapter().companion_overrides(_spec({"slotPinning": True}))
    assert overrides == {"slotPinning": True}
    assert render_config(runtime_name="r", alias="m", overrides=overrides)["slotPinning"] is True


def test_pinning_off_explicitly_is_off_on_both():
    assert "--no-cache-idle-slots" not in _argv({"slotPinning": False})
    assert LlamaCppAdapter().companion_overrides(_spec({"slotPinning": False})) == {}
