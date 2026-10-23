"""DI7: a launch setting carried by the existing adapter GUI schema.

vLLM RECIPE_VERSION 0.30.0: vllm/envs.py, VLLM_BATCH_INVARIANT
https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/envs.py#L590-L592
https://docs.vllm.ai/en/v0.30.0/features/batch_invariance/#enabling-batch-invariance
"""

from __future__ import annotations

from .._generated.models import ConfigField, ConfigValueType, EngineKind

KEY = "sameAnswerEveryTime"
VLLM_ENV = "VLLM_BATCH_INVARIANT"


def field(engine: EngineKind) -> ConfigField:
    detail = (
        "Runs one request at a time; others wait. Uses one server slot so other "
        "requests cannot share its batch. Extra arguments can override the slot count."
        if engine is EngineKind.llama_cpp
        else "Enables vLLM's batch-invariant mode. Support depends on the engine, model "
        "and hardware. An explicit runtime environment value overrides this setting."
    )
    return ConfigField(
        key=KEY,
        label="Same answer every time (slower)",
        description=(
            "Use the same prompt, seed and sampling settings on the same model and engine "
            "build. This can reduce speed. " + detail
        ),
        category="performance",
        valueType=ConfigValueType.boolean,
        default=False,
        unsetMeans=(
            "Off by default. Expert launch overrides still apply; "
            "the running model shows what is in effect."
        ),
        requiresRestart=True,
    )


def effective(engine: EngineKind, argv: list[str], env: dict[str, str]) -> bool | None:
    """Read the final launch, after all expert overrides. No hardware guesses."""
    if engine is EngineKind.vllm:
        try:
            return bool(int(env.get(VLLM_ENV, "0")))
        except ValueError:
            return None
    if engine is EngineKind.llama_cpp:
        parallel = env.get("LLAMA_ARG_N_PARALLEL")
        tokens = iter(argv[1:])
        for token in tokens:
            name, _, inline = token.partition("=")
            if name in ("--parallel", "-np"):
                parallel = inline or next(tokens, None)
        return parallel == "1"
    return None
