"""Speculative decoding for llama.cpp: the profile's settings as a command line (SD1).

`docs/design/speculative-decoding.md` is the design, with Troy's calls S1-S5.
Two settings, under the heading "Speculative decoding":

* **Reuse repeated text** (`reuseRepeatedText`) adds `ngram-mod` to
  `--spec-type`. It needs no file and proposes nothing when nothing in the
  context repeats, so it is **on for every llama.cpp runtime unless a profile
  turns it off** (S3).
* **Draft with** (`draftWith`): `off` (the default until a settings build has
  timed the others, S1), `mtpHead` (`draft-mtp` on the model's own head, on its
  own weights) or `companion` (`--spec-draft-model F`, `draft-mtp` or
  `draft-simple` by the file's own tensors).

Both together are `--spec-type ngram-mod,draft-mtp`. llama.cpp tries the
types in its own fixed order for each sequence (n-gram first, then a model;
`common_speculative_init`, b11719), so the order here is for reading only.

Three rules keep it from breaking a launch that worked before:

* **Speculation is a launch setting.** Nothing here is per request.
* **`extraArgs` wins.** `--spec-type` *appends* in llama.cpp's parser, so a
  hand-written one would be combined with ours rather than replace it. Any
  speculation flag in `extraArgs` (or its `LLAMA_ARG_*` variable in `env`)
  means the operator is setting it by hand, and nothing is generated.
* **Only what the build offers.** The default is on everywhere, so an engine
  build that predates `--spec-type`, or one whose `--help` could not be read,
  must not be handed it: llama-server exits on an argument it does not know
  and the supervisor would restart it into the same wall. A type the build
  does not offer is left out, and the runtime says so (`RuntimeSpeculation.note`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..gguf_context import drafting_facts
from ..model_paths import is_windows_shaped

REUSE_TEXT_KEY = "reuseRepeatedText"
DRAFT_WITH_KEY = "draftWith"
DRAFT_MODEL_KEY = "draftModelPath"
DRAFT_TOKENS_KEY = "draftTokens"
KEYS = frozenset({REUSE_TEXT_KEY, DRAFT_WITH_KEY, DRAFT_MODEL_KEY, DRAFT_TOKENS_KEY})

DRAFT_WITH_CHOICES = ("off", "mtpHead", "companion")

#: The `--spec-type` names this module writes.
NGRAM = "ngram-mod"
DRAFT_MTP = "draft-mtp"
DRAFT_SIMPLE = "draft-simple"
TYPES = (NGRAM, DRAFT_MTP, DRAFT_SIMPLE)

#: Flags and variables that mean speculation is being set by hand.
_BY_HAND_FLAGS = frozenset(
    {
        "--spec-type",
        "--spec-draft-model",
        "-md",
        "--model-draft",
        "--spec-draft-n-max",
        "--draft-max",
        "--draft",
    }
)
_BY_HAND_ENV = ("LLAMA_ARG_SPEC_TYPE", "LLAMA_ARG_SPEC_DRAFT_MODEL")

METRICS_FLAG = "--metrics"

#: llama-server's Prometheus counters (`tools/server/server-task.cpp`, b11719).
DRAFTED_METRIC = "llamacpp:spec_decode_num_draft_tokens_total"
ACCEPTED_METRIC = "llamacpp:spec_decode_num_accepted_tokens_total"


def offered_types(help_text: str | None) -> frozenset[str] | None:
    """The speculation types a build's `--help` names, or None when its help
    could not be read (which is not the same as "offers none")."""
    if help_text is None:
        return None
    if "--spec-type" not in help_text:
        return frozenset()
    return frozenset(
        t for t in TYPES if re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", help_text)
    )


def by_hand(extra_args: Sequence[str] | None, env: Mapping[str, str] | None) -> bool:
    """Whether `extraArgs` or `env` already set speculation."""
    for token in extra_args or ():
        if token.partition("=")[0] in _BY_HAND_FLAGS:
            return True
    return any((env or {}).get(name) for name in _BY_HAND_ENV)


@dataclass(frozen=True)
class Plan:
    """What a launch passes, and why anything asked for is left out."""

    argv: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    draft_model: str | None = None
    note: str | None = None


NOT_OFFERED = (
    "This llama.cpp build does not offer {what}, so it is off. Updating the engine turns it on."
)


def plan(
    flags: Mapping[str, object],
    model_path: str,
    *,
    extra_args: Sequence[str] | None,
    env: Mapping[str, str] | None,
    search: bool,
    offered: frozenset[str] | None,
) -> Plan:
    """The `--spec-*` arguments for these profile flags on this build.

    Raises ValueError for a setting that cannot be honoured on this model,
    naming it: a head the model does not carry, a companion that is missing
    or is not something llama.cpp can draft with. Those are settings the
    operator chose, so a launch that quietly dropped them would lie.
    """
    if search or by_hand(extra_args, env):
        return Plan()
    choice = flags.get(DRAFT_WITH_KEY) or "off"
    if choice not in DRAFT_WITH_CHOICES:
        raise ValueError(f"draftWith must be off, mtpHead or companion, not {choice!r}")
    reuse = flags.get(REUSE_TEXT_KEY)
    wanted: list[str] = [NGRAM] if reuse is not False else []
    draft_model: str | None = None
    if choice == "mtpHead":
        facts = drafting_facts(model_path)
        if facts is not None and not facts.mtp_head:
            raise ValueError(
                "Draft with is set to the model's own prediction head, but "
                f"{Path(model_path).name} carries no MTP head. Set Draft with to Off, "
                "or to a companion file."
            )
        wanted.append(DRAFT_MTP)
    elif choice == "companion":
        draft_model = _str(flags.get(DRAFT_MODEL_KEY))
        if not draft_model:
            raise ValueError(
                "Draft with is set to a companion file, but no companion file is chosen."
            )
        facts = drafting_facts(draft_model)
        if facts is None:
            raise ValueError(f"The companion file {draft_model} cannot be read as a GGUF.")
        kind = facts.companion_type
        if kind is None:
            raise ValueError(
                f"{Path(draft_model).name} is not a file llama.cpp can draft with: it has "
                "neither an MTP head nor a model's first block."
            )
        wanted.append(kind)
    if not wanted:
        return Plan()

    note: str | None = None
    if offered is None:
        # The build could not be asked. The default alone is not risked on
        # it; a way of drafting someone chose is, guessing forward, as the
        # load-mode spelling does.
        if choice == "off":
            return Plan()
        types = tuple(wanted)
    else:
        types = tuple(t for t in wanted if t in offered)
        missing = [t for t in wanted if t not in offered]
        if missing:
            what = (
                "speculative decoding"
                if not types
                else " or ".join(_WORDS.get(t, t) for t in missing)
            )
            note = NOT_OFFERED.format(what=what)
    if not types:
        return Plan(note=note)

    argv: list[str] = ["--spec-type", ",".join(types)]
    drafting = any(t.startswith("draft-") for t in types)
    if draft_model is not None and drafting:
        argv += ["--spec-draft-model", draft_model]
    else:
        draft_model = None
    tokens = flags.get(DRAFT_TOKENS_KEY)
    if drafting and isinstance(tokens, int) and not isinstance(tokens, bool):
        argv += ["--spec-draft-n-max", str(tokens)]
    return Plan(argv=tuple(argv), types=types, draft_model=draft_model, note=note)


_WORDS = {
    NGRAM: "reusing repeated text (ngram-mod)",
    DRAFT_MTP: "drafting with an MTP head (draft-mtp)",
    DRAFT_SIMPLE: "drafting with a draft model (draft-simple)",
}


def metrics_argv(argv: Sequence[str]) -> list[str]:
    """`--metrics` when the command line speculates and does not already
    expose the counters the runtime's acceptance rate is read from."""
    if METRICS_FLAG in argv or not in_argv(argv):
        return []
    return [METRICS_FLAG]


def in_argv(argv: Sequence[str]) -> tuple[list[str], str | None] | None:
    """`(types, draft model)` from a command line, or None when it does not
    speculate. Every `--spec-type` counts, because llama.cpp appends them."""
    types: list[str] = []
    draft: str | None = None
    tokens = iter(argv)
    for token in tokens:
        name, _, inline = token.partition("=")
        if name == "--spec-type":
            value = inline or next(tokens, "")
            for part in value.split(","):
                part = part.strip()
                if part and part != "none" and part not in types:
                    types.append(part)
        elif name in ("--spec-draft-model", "-md", "--model-draft"):
            draft = inline or next(tokens, None)
    return (types, draft) if types else None


def parse_metrics(text: str) -> tuple[int | None, int | None]:
    """`(drafted, accepted)` from llama-server's Prometheus text."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        if name in (DRAFTED_METRIC, ACCEPTED_METRIC):
            try:
                found[name] = int(float(value.strip()))
            except ValueError:
                continue
    return found.get(DRAFTED_METRIC), found.get(ACCEPTED_METRIC)


def declaration_problem(model_path: str, flags: Mapping[str, object] | None) -> str | None:
    """What is wrong with a declaration's speculation flags, before anything
    runs, or None.

    A companion file belongs to the model beside it, as a projector does, so
    it must be in the model's own folder: that is also what lets a node's own
    copy of the model carry it along in its place (`model_copies`).
    """
    flags = flags or {}
    choice = flags.get(DRAFT_WITH_KEY)
    if choice is not None and choice not in DRAFT_WITH_CHOICES:
        return f"draftWith must be off, mtpHead or companion, not {choice!r}"
    draft = _str(flags.get(DRAFT_MODEL_KEY))
    if choice == "companion" and not draft:
        return "draftWith is companion, so draftModelPath must name the companion file"
    if draft and not _same_folder(model_path, draft):
        return (
            f"the companion file {draft} must be in the model's own folder "
            f"({_parent(model_path)}), beside the model it drafts for"
        )
    return None


def companion(flags: Mapping[str, object] | None) -> str | None:
    """The companion file a declaration drafts with, or None."""
    flags = flags or {}
    if flags.get(DRAFT_WITH_KEY) != "companion":
        return None
    return _str(flags.get(DRAFT_MODEL_KEY)) or None


def _str(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _parent(path: str) -> str:
    pure = PureWindowsPath(path) if is_windows_shaped(path) else PurePosixPath(path)
    return str(pure.parent)


def _same_folder(model: str, draft: str) -> bool:
    if is_windows_shaped(model) != is_windows_shaped(draft):
        return False
    if is_windows_shaped(model):
        return PureWindowsPath(model).parent == PureWindowsPath(draft).parent
    return PurePosixPath(model).parent == PurePosixPath(draft).parent
