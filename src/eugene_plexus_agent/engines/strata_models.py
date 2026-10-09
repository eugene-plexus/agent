"""The models Strata v0.1.39 publishes as supported (LS4, design §4.4).

Read off upstream's own setup at the commit the adapter pins
(`strata.COMMIT`), never written from memory: `setup.py`'s `MODELS` (the
sizes, each with the families that have it), `FAMILIES` (whose files, in
which repo, under which name) and `HF_REVISIONS` (the commit of each repo
setup downloads from); `docs/MODELS.md` for the one it recommends. Setup's
menu offers nine choices: the original model in four sizes, Swift 1.5 in two
(its Q2_0 files are on the hub, but setup cannot prepare them yet, upstream
#171), the Coder's one size and Unsloth's two.

Each file was checked on the hub at that revision (2026-10-09): every first
shard exists, the hub reads each as `qwen4exp`, and `sizeBytes` is the sum
of the shards as the hub lists them. Upstream's setup accepts a GGUF by
these names alone (`gguf_choice`, `SUPPORTED_GGUFS`), so the adapter's GGUF
requirement names them too (`STRATA_FILES`).

Moving the Strata pin means reading these again from the new setup.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from .._generated.models import ModelFormat, ModelPreparation, PreparedSource, SupportedModel

#: What a GGUF on the list needs before Strata runs it. The same words the
#: adapter's GGUF requirement carries.
PREPARATION = ModelPreparation(
    recipe="strata-prepare",
    note="an expert pack, a lookup table and an MTP helper",
)

_QWEN = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
_SWIFT = "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
_CODER = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"
_UNSLOTH = "unsloth/Qwen3.8-Flash-Next-GGUF"

#: setup.py `HF_REVISIONS`.
_REVISIONS = {
    _QWEN: "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
    _SWIFT: "b22d729eae29b5796f76fb70f91aef549b9fc52c",
    _CODER: "5348543e0147355ac9cbcb031184a3546350988e",
    _UNSLOTH: "38bb39ee97821de2c9009abb7e93950eec396e66",
}

#: setup.py `FAMILIES[...]["by"]`.
_BY = {
    _QWEN: "Qwen; GSQ-RCO quants by ISTA-DASLab",
    _SWIFT: "UkisAI's fine-tune of Qwen3.8-Flash-Next",
    _CODER: "ISTA-DASLab's coding version",
    _UNSLOTH: "Unsloth's ~4-bit quantizations",
}

_SWIFT_ABOUT = (
    "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its authors' numbers)"
)


def _entry(
    *,
    id: str,
    title: str,
    repo: str,
    file: str,
    quantization: str,
    about: str,
    size: int,
    recommended: bool = False,
    experimental: bool = False,
    license: str | None = None,
) -> SupportedModel:
    return SupportedModel(
        id=id,
        title=title,
        about=about,
        publisher=_BY[repo],
        license=license,
        format=ModelFormat.gguf,
        architecture="qwen4exp",
        quantization=quantization,
        source=PreparedSource(repoId=repo, file=file, revision=_REVISIONS[repo]),
        sizeBytes=size,
        preparation=PREPARATION,
        recommended=recommended,
        experimental=experimental,
    )


#: In setup's menu order: families as `FAMILIES` lists them, each family's
#: sizes as `MODELS` does with an experimental one last. `id` is setup's own
#: tag for the choice (`FAMILIES[f]["tag"] + model`), which names its pack
#: and configuration. `about` is setup's line for the size, or the family's
#: where the family says more.
SUPPORTED_MODELS: tuple[SupportedModel, ...] = (
    _entry(
        id="Q2_0",
        title="Qwen3.8-Flash-Next Q2_0",
        repo=_QWEN,
        file="Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        quantization="Q2_0",
        about="2-bit, the fastest",
        size=66_423_878_624,
    ),
    _entry(
        id="IQ2_XS",
        title="Qwen3.8-Flash-Next IQ2_XS",
        repo=_QWEN,
        file="IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
        quantization="IQ2_XS",
        about="2-bit i-quant, a little better quality, close in speed",
        size=68_026_093_024,
        # docs/MODELS.md: "Not sure? Take IQ2_XS."
        recommended=True,
    ),
    _entry(
        id="IQ3_XXS",
        title="Qwen3.8-Flash-Next IQ3_XXS",
        repo=_QWEN,
        file="IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
        quantization="IQ3_XXS",
        about="3-bit i-quant, better quality, slower (more CPU work per token)",
        size=75_839_998_528,
    ),
    _entry(
        id="IQ3_S",
        title="Qwen3.8-Flash-Next IQ3_S",
        repo=_QWEN,
        file="IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
        quantization="IQ3_S",
        about=(
            "3.5-bit i-quant, the best quality (matches the full model), the slowest; needs a "
            "64 GB PC with little else running"
        ),
        size=83_617_662_656,
    ),
    _entry(
        id="swift-IQ2_XS",
        title="Swift 1.5 IQ2_XS",
        repo=_SWIFT,
        file="Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
        quantization="IQ2_XS",
        about=f"{_SWIFT_ABOUT}; 2-bit i-quant, a little better quality, close in speed",
        size=68_152_167_168,
        license=(
            "Swift Open License 1.0: "
            "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
        ),
    ),
    _entry(
        id="swift-IQ3_XXS",
        title="Swift 1.5 IQ3_XXS",
        repo=_SWIFT,
        file="Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
        quantization="IQ3_XXS",
        about=(f"{_SWIFT_ABOUT}; 3-bit i-quant, better quality, slower (more CPU work per token)"),
        size=75_966_073_120,
        license=(
            "Swift Open License 1.0: "
            "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
        ),
    ),
    _entry(
        id="coder-IQ1_M",
        title="Qwen3.8-Flash-Next Coder IQ1_M",
        repo=_CODER,
        file="IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
        quantization="IQ1_M",
        about=(
            "half the experts (code, tools, images kept): needs ~32 GB of RAM, faster; "
            "weaker outside coding"
        ),
        size=58_408_584_928,
    ),
    _entry(
        id="unsloth-UD-IQ4_XS",
        title="Qwen3.8-Flash-Next (Unsloth) UD-IQ4_XS",
        repo=_UNSLOTH,
        file="UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
        quantization="UD-IQ4_XS",
        about=(
            "~4-bit i-quant (Unsloth Dynamic), between IQ3_S and UD-Q4_K_XL in quality; on a "
            "PC with less than ~80 GB of RAM part of its experts are read from the SSD"
        ),
        size=93_682_584_224,
    ),
    _entry(
        id="unsloth-UD-Q4_K_XL",
        title="Qwen3.8-Flash-Next (Unsloth) UD-Q4_K_XL",
        repo=_UNSLOTH,
        file="UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
        quantization="UD-Q4_K_XL",
        about=(
            "4-bit (Unsloth Dynamic), EXPERIMENTAL: the best quality, but most experts come "
            "from the SSD on a 64 GB PC (7-8.5 tokens/s measured)"
        ),
        size=111_334_654_784,
        experimental=True,
    ),
)

#: The GGUF names Strata's setup accepts: each choice's first shard, without
#: its folder. A GGUF of the same architecture by any other name is not one.
STRATA_FILES: tuple[str, ...] = tuple(
    PurePosixPath(m.source.file).name for m in SUPPORTED_MODELS if m.source.file
)
