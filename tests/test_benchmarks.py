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


@pytest.mark.parametrize(
    "flags",
    [{"parallelSlots": 2}, {"contextSize": 0}, {"gpuLayers": "1,99"}, {"tensorSplit": "1;--rpc x"}],
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
