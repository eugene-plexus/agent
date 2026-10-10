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

import math
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from .._generated.models import (
    Context,
    EngineFit,
    EngineFitModel,
    EngineTableFit,
    FitModelKind,
    FitVerdict,
    ModelFormat,
    ModelPreparation,
    PreparedSource,
    SupportedModel,
)

#: setup.py `CONTEXTS`: the context sizes it offers, which a preparation
#: fixes (LS5, B50).
SETUP_CONTEXTS: tuple[int, ...] = (8192, 32768, 65536, 131072, 262144, 393216, 524288)

#: What a GGUF on the list needs before Strata runs it. The same words the
#: adapter's GGUF requirement carries.
PREPARATION = ModelPreparation(
    recipe="strata-prepare",
    note="an expert pack, a lookup table and an MTP helper",
    contexts=[Context(c) for c in SETUP_CONTEXTS],
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
    min_engine: str | None = None,
) -> SupportedModel:
    preparation = PREPARATION
    if min_engine is not None:
        preparation = preparation.model_copy(update={"minEngineVersion": min_engine})
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
        preparation=preparation,
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
        # setup.py MODELS["UD-IQ4_XS"]["engine"]: (0, 1, 38) (LS7, B30).
        min_engine="v0.1.38",
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


@dataclass(frozen=True)
class SetupChoice:
    """How upstream's setup names a choice and sizes it (LS5): `--family` and
    `--model`, `FAMILIES[f]["name"]` (the start of its `model_name`), and from
    `MODELS[m]` the experts' size in RAM (`arena_gb`), whether it keeps a
    RAM budget of them instead (`budget`, Unsloth's two), and the RAM setup
    says it needs (`ram_gb`, LS6)."""

    family: str
    model: str
    name: str
    arena_gb: float
    budget: bool = False
    ram_gb: float = 0.0

    @property
    def tag(self) -> str:
        """Setup's `tag` lowered: its configuration is `strata-<tag>.json`,
        its pack `packs/<tag>`."""
        prefix = "" if self.family == "qwen" else f"{self.family}-"
        return f"{prefix}{self.model}".lower()

    @property
    def model_name(self) -> str:
        """What setup calls the model it configured: `<family name>-<size>`."""
        return f"{self.name}-{self.model.lower()}"


#: setup.py `FAMILIES[f]["name"]`, and `MODELS[m]["arena_gb"]` and
#: `["ram_gb"]`, by list id.
SETUP_CHOICES: dict[str, SetupChoice] = {
    "Q2_0": SetupChoice("qwen", "Q2_0", "qwen3.8-flash-next", 34.0, ram_gb=48),
    "IQ2_XS": SetupChoice("qwen", "IQ2_XS", "qwen3.8-flash-next", 35.5, ram_gb=48),
    "IQ3_XXS": SetupChoice("qwen", "IQ3_XXS", "qwen3.8-flash-next", 42.9, ram_gb=60),
    "IQ3_S": SetupChoice("qwen", "IQ3_S", "qwen3.8-flash-next", 50.3, ram_gb=62),
    "swift-IQ2_XS": SetupChoice("swift", "IQ2_XS", "swift-1.5", 35.5, ram_gb=48),
    "swift-IQ3_XXS": SetupChoice("swift", "IQ3_XXS", "swift-1.5", 42.9, ram_gb=60),
    "coder-IQ1_M": SetupChoice("coder", "IQ1_M", "qwen3.8-flash-next-coder", 23.4, ram_gb=32),
    "unsloth-UD-IQ4_XS": SetupChoice(
        "unsloth", "UD-IQ4_XS", "qwen3.8-flash-next-unsloth", 59.5, budget=True, ram_gb=48
    ),
    "unsloth-UD-Q4_K_XL": SetupChoice(
        "unsloth", "UD-Q4_K_XL", "qwen3.8-flash-next-unsloth", 77.0, budget=True, ram_gb=48
    ),
}

#: setup.py `LOW_RAM_HEADROOM_GB`: RAM it keeps beside the experts.
LOW_RAM_HEADROOM_GB = 10
#: setup.py `UNSLOTH_RAM_LEFT_GB`: RAM beside a RAM budget of experts.
UNSLOTH_RAM_LEFT_GB = 24
#: setup.py `--check`: within this much of its `ram_gb` a model is *tight*
#: (the system pages it), not *does not fit*.
TIGHT_GB = 8
#: setup.py: under this much VRAM "Strata will run, but most experts stay on
#: the CPU and it will be slow".
SLOW_VRAM_GB = 11
#: setup.py's own allowance beside the model files (the pack, the MTP helper).
SETUP_DISK_GB = 8
#: Q2_0's experts repacked for an AVX-512 CPU, which setup writes once.
Q2_0_REPACK_GB = 40


def disk_needed(choice: SetupChoice, ram_bytes: int | None) -> int:
    """What setup checks for before it prepares a choice (`main`, its `need`
    with `--gguf-dir`), in bytes as its `free_gb` counts them (1e9): 8 GB;
    the experts written into one file when this PC's RAM is short of them
    (the low-RAM mode, never for a RAM-budget model); 40 GB for Q2_0 on an
    AVX-512 CPU, counted always since the CPU feature is not read here (the
    low-RAM file is then not written, as in setup). An unknown RAM counts as
    short: the larger answer."""
    gb: float = SETUP_DISK_GB
    if choice.family == "qwen" and choice.model == "Q2_0":
        gb += Q2_0_REPACK_GB
    elif not choice.budget:
        ram_gib = ram_bytes / 2**30 if ram_bytes is not None else 0.0
        if ram_gib < choice.arena_gb + LOW_RAM_HEADROOM_GB:
            gb += choice.arena_gb + 1
    return int(gb * 1e9)


def supported_here(ram_bytes: int | None) -> tuple[SupportedModel, ...]:
    """The list as this node reports it: each preparation's disk by setup's
    rule with this node's RAM (B51)."""
    out = []
    for model in SUPPORTED_MODELS:
        need = disk_needed(SETUP_CHOICES[model.id], ram_bytes)
        assert model.preparation is not None
        preparation = model.preparation.model_copy(update={"diskBytes": need})
        out.append(model.model_copy(update={"preparation": preparation}))
    return tuple(out)


def choice_for_file(path: str) -> tuple[SupportedModel, SetupChoice] | None:
    """The list entry whose first shard is the file at `path` (its name, case
    ignored as the GGUF requirement compares), or None: setup prepares no
    other file."""
    wanted = PureWindowsPath(path).name.lower()
    for model in SUPPORTED_MODELS:
        if model.source.file and PurePosixPath(model.source.file).name.lower() == wanted:
            return model, SETUP_CHOICES[model.id]
    return None


# --- fit: setup's own table, applied to this node (LS6) ---------------------


@dataclass(frozen=True)
class SetupFit:
    """Setup's answer for one choice on one PC, as its `--check` gives it."""

    verdict: FitVerdict
    words: str
    #: The mode it would run in: `ram` (every expert in RAM), `budget` (a RAM
    #: budget of them, the rest from the SSD), `low_ram` (the low-RAM mode) or
    #: `paged` (short of RAM: the system pages them). None when unknown or no.
    mode: str | None = None


def resident_budget_gib(choice: SetupChoice, ram_gib: float) -> int:
    """setup.py `resident_budget_gib`: the GiB of experts a RAM-budget model
    keeps in RAM, the RAM less what it leaves beside them."""
    gib = round(ram_gib) - UNSLOTH_RAM_LEFT_GB
    return max(8, min(gib, int(choice.arena_gb / 1.073741824)))


def low_ram_needed(choice: SetupChoice, ram_gib: float) -> bool:
    """setup.py `low_ram_needed`: the experts do not fit RAM with room beside them."""
    return ram_gib < choice.arena_gb + LOW_RAM_HEADROOM_GB


def low_ram_gpu_gb(choice: SetupChoice, vram_gib: float) -> float:
    """setup.py `low_ram_gpu_gb` at its default 32K context: the experts the
    card's cache holds, its VRAM less ~5 GB for the rest."""
    return max(0.0, min(choice.arena_gb, vram_gib - 5))


def low_ram_fits(choice: SetupChoice, ram_gib: float, vram_gib: float) -> bool:
    """setup.py `low_ram_fits`: the experts the card does not hold fit the RAM
    left beside the rest."""
    return ram_gib - 6 + max(0.0, vram_gib - 5) >= choice.arena_gb


def low_ram_resident(choice: SetupChoice, ram_gib: float, vram_gib: float) -> bool:
    """setup.py `low_ram_resident`: the rest fits RAM with the usual room."""
    rest = choice.arena_gb - low_ram_gpu_gb(choice, vram_gib)
    return ram_gib >= rest + LOW_RAM_HEADROOM_GB


def setup_fit(choice: SetupChoice, ram_gib: float, vram_gib: float | None) -> SetupFit:
    """Setup's `--check` verdict for one choice (setup.py at the commit the
    adapter pins), mapped onto the fit words: *fits* and the RAM-budget mode
    are `fits`, the low-RAM mode `split`, *tight* `tight`, *does not fit*
    `no`. `ram_gib` is the PC's total RAM in setup's GB (2^30 bytes), as its
    `ram_gb()` reads it; `vram_gib` the card it would use, None when there is
    none it can use: then `unknown`, since setup's rule is about one."""
    have = f"this machine has {ram_gib:.0f} GB"
    need = f"about {choice.ram_gb:g} GB of RAM"
    if vram_gib is None:
        return SetupFit(
            FitVerdict.unknown,
            "no graphics card Strata can use was found here, and its setup's rule is about one",
        )
    slow = (
        "; with under 12 GB of graphics memory most experts stay on the CPU, so it will be slow"
        if vram_gib < SLOW_VRAM_GB
        else ""
    )
    if choice.budget:
        if ram_gib >= choice.ram_gb:
            budget = resident_budget_gib(choice, ram_gib)
            return SetupFit(
                FitVerdict.fits,
                f"Strata keeps {budget} GiB of its {choice.arena_gb:g} GB of experts in RAM "
                f"and reads the rest from the SSD as it answers ({have}){slow}",
                "budget",
            )
        return SetupFit(
            FitVerdict.no,
            f"Strata needs {choice.ram_gb:g} GB of RAM or more for this model, and {have}",
        )
    if low_ram_needed(choice, ram_gib) and low_ram_fits(choice, ram_gib, vram_gib):
        share = low_ram_gpu_gb(choice, vram_gib) / choice.arena_gb
        rest = (
            "the rest stay in RAM"
            if low_ram_resident(choice, ram_gib, vram_gib)
            else "the rest are read from the SSD as needed"
        )
        return SetupFit(
            FitVerdict.split,
            f"fits in Strata's low-RAM mode: the card holds about {share:.0%} of its "
            f"experts and {rest} ({have}){slow}",
            "low_ram",
        )
    if ram_gib >= choice.ram_gb:
        return SetupFit(
            FitVerdict.fits,
            f"Strata keeps its {choice.arena_gb:g} GB of experts in RAM: it needs {need}, "
            f"and {have}{slow}",
            "ram",
        )
    if ram_gib >= choice.ram_gb - TIGHT_GB:
        return SetupFit(
            FitVerdict.tight,
            f"Strata needs {need} and {have}: the system will page part of its experts "
            "from disk, which is much slower, and it may not start",
            "paged",
        )
    return SetupFit(FitVerdict.no, f"Strata needs {need} for this model, and {have}")


def engine_fit_of(choice: SetupChoice, ram_gib: float, vram_gib: float | None) -> EngineFit:
    answer = setup_fit(choice, ram_gib, vram_gib)
    return EngineFit(
        estimated=True,
        verdict=answer.verdict,
        model=FitModelKind.engine_table,
        reason=answer.words,
        ramBytes=int(choice.ram_gb * 2**30),
    )


def fit_table(ram_bytes: int | None, vram_bytes: int | None) -> EngineFitModel:
    """Setup's answer for every model on the list, on this node (LS6): its
    total RAM, and the card it would use (`vram_bytes`; None: none it can
    use). A node whose RAM could not be read has every row *not estimated*."""
    rows: list[EngineTableFit] = []
    vram_gib = vram_bytes / 2**30 if vram_bytes is not None else None
    for model in SUPPORTED_MODELS:
        choice = SETUP_CHOICES[model.id]
        if ram_bytes is None:
            fit = EngineFit(
                estimated=False,
                model=FitModelKind.engine_table,
                reason="this machine's RAM could not be read, and Strata's setup sizes by it",
            )
        else:
            fit = engine_fit_of(choice, ram_bytes / 2**30, vram_gib)
        file = PurePosixPath(model.source.file).name if model.source.file else model.id
        rows.append(EngineTableFit(file=file, supportedModel=model.id, fit=fit))
    return EngineFitModel(kind=FitModelKind.engine_table, table=rows)


def experts_in_ram_bytes(choice: SetupChoice) -> int:
    """What the `ram` mode copies into RAM: the experts, in setup's GB."""
    return math.ceil(choice.arena_gb * 1e9)
