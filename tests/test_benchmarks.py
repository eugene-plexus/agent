from pathlib import Path

import pytest

from eugene_plexus_agent._generated.models import BenchmarkRequest
from eugene_plexus_agent.benchmarks import benchmark_args, parse_point


def request(**flags):
    return BenchmarkRequest.model_validate(
        {
            "modelId": "model",
            "profileId": "profile",
            "profileName": "Long",
            "runtime": {
                "name": "model",
                "engine": "llama_cpp",
                "modelPath": "/models/a.gguf",
                "flags": {"contextSize": 4096, **flags},
            },
        }
    )


HELP = "--n-depth --progress --output --n-gpu-layers --batch-size --ubatch-size --threads --main-gpu --tensor-split --flash-attn --load-mode"


def test_depth_sweep_preserves_profile_and_leaves_generation_room():
    args, depths = benchmark_args(
        request(gpuLayers=0, tensorSplit="0.6,0.4", flashAttention=False, noMmap=True),
        Path("bench"),
        "/models/a.gguf",
        HELP,
    )
    assert depths == [0, 1984, 3968]
    assert args[args.index("--n-depth") + 1] == "0,1984,3968"
    assert args[args.index("--n-gpu-layers") + 1] == "0"
    assert args[args.index("--tensor-split") + 1] == "0.6/0.4"
    assert "--flash-attn" not in args  # false leaves the runtime's engine default alone
    assert args[args.index("--load-mode") + 1] == "none"
    assert "--parallel" not in args and "--ctx-size" not in args


def test_a_built_profiles_memory_settings_carry_into_the_benchmark():
    # A profile the builder wrote must be benchmarkable, with the same cache
    # and the same margin the server will use; refusing it would be worse.
    help_text = HELP + " --cache-type-k --cache-type-v --fit-target"
    args, _ = benchmark_args(
        request(cacheType="q4_0", memoryMargin=2048), Path("bench"), "/models/a.gguf", help_text
    )
    assert args[args.index("--cache-type-k") + 1] == "q4_0"
    assert args[args.index("--cache-type-v") + 1] == "q4_0"
    assert args[args.index("--fit-target") + 1] == "2048"
    assert args[args.index("--flash-attn") + 1] == "on"  # a quantised cache needs it
    with pytest.raises(ValueError, match="cache-type"):
        benchmark_args(request(cacheType="q8_0"), Path("bench"), "/models/a.gguf", HELP)


@pytest.mark.parametrize(
    "flags",
    [
        {"parallelSlots": 2},
        {"contextSize": 0},
        {"gpuLayers": "1,99"},
        {"tensorSplit": "1;--rpc x"},
        {"cacheType": "q2_k"},
        {"memoryMargin": -1},
    ],
)
def test_unsupported_or_malformed_profile_is_refused(flags):
    with pytest.raises(ValueError):
        benchmark_args(request(**flags), Path("bench"), "/models/a.gguf", HELP)


def test_result_is_decode_at_requested_depth_with_real_samples():
    raw = {
        "n_prompt": 0,
        "n_gen": 128,
        "n_depth": 3968,
        "avg_ts": 20,
        "stddev_ts": 1,
        "samples_ts": [19, 20, 21],
    }
    point = parse_point(raw, [0, 1984, 3968], 128, 3)
    assert point.depth == 3968 and point.tokensPerSecond == 20
    for field, bad in [
        ("n_prompt", 128),
        ("n_gen", 64),
        ("n_depth", 4096),
        ("avg_ts", float("nan")),
        ("samples_ts", [20]),
    ]:
        with pytest.raises(ValueError):
            parse_point({**raw, field: bad}, [0, 1984, 3968], 128, 3)
