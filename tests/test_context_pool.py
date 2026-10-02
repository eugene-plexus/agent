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

import pytest

from eugene_plexus_agent._generated.models import RuntimeCapabilities
from eugene_plexus_agent.engines import LlamaCppAdapter

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
