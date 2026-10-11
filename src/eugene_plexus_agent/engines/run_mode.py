"""What a model is started to do, and what llama-server has to be told to start it so (MR3).

`llama-server` embeds or reranks only when it is told to when it starts, and a
model started the other way does neither: measured on b10948 and b11552, an
embedding GGUF started as chat answers `/v1/embeddings` with a 501 and chat
with a 500 (a causal embedder such as Qwen3-Embedding answers chat with text
that means nothing). Until MR3 the adapter passed none of the flags, so every
such model the Library listed started wrong and was listed as chat.

`RuntimeSpec.mode` (`RuntimeMode` in the contract) is the declaration, and
`resolve` is the one decision made from it: `auto` is the file's own header,
the other three are a person's say. **A chat model's command line is exactly
what it was before this module existed**: every path that is not clearly a
search model ends in `Task.chat`, which adds nothing.

The same words are the library's (`formats/gguf.py`, `is_embedding` and
`is_rerank`), duplicated rather than shared by the polyrepo rule; the console
predicts with those and the agent decides with these, so they must agree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .._generated.models import RuntimeMode
from ..gguf_context import POOLING_RANK, RunFacts, run_facts


class Task(StrEnum):
    """What the engine is started to do."""

    chat = "chat"
    embedding = "embedding"
    reranking = "reranking"


@dataclass(frozen=True)
class Resolved:
    """A decision, and the sentence that says why."""

    task: Task
    why: str
    pooling: str | None = None
    """The `--pooling` to add: only for an embedder whose header names none."""

    @property
    def search(self) -> bool:
        """An embedder or a reranker: never unloaded for being idle."""
        return self.task is not Task.chat

    def argv(self) -> list[str]:
        """What this adds to the command line, before the operator's `extraArgs`."""
        if self.task is Task.embedding:
            return ["--embeddings", *(["--pooling", self.pooling] if self.pooling else [])]
        if self.task is Task.reranking:
            return ["--reranking"]
        return []


def resolve(mode: RuntimeMode | str | None, model_path: str) -> Resolved:
    """Decide from the declared `mode`, reading the file's header only when it matters.

    `model_path` is the file this host opens (the planner has already resolved it
    through the Library folder's mounts and the node's own copy).
    """
    try:
        declared = RuntimeMode(mode) if mode else RuntimeMode.auto
    except ValueError:
        declared = RuntimeMode.auto
    if declared is RuntimeMode.chat:
        return Resolved(Task.chat, "the runtime says chat")
    if declared is RuntimeMode.reranking:
        return Resolved(Task.reranking, "the runtime says reranking")
    facts = run_facts(model_path)
    if declared is RuntimeMode.embedding:
        return Resolved(Task.embedding, "the runtime says embedding", pooling=pooling_to_add(facts))
    if facts is None:
        # Not "a chat model": the header did not read. It starts as it always has,
        # and if the file is what is wrong, the engine says so.
        return Resolved(Task.chat, "the file's header could not be read, so it starts as chat")
    if facts.pooling_type == POOLING_RANK or facts.has_classifier_labels:
        return Resolved(Task.reranking, "its header names a reranker (pooling type 4)")
    if facts.pooling_type is not None:
        return Resolved(
            Task.embedding,
            f"its header names pooling type {facts.pooling_type}",
            pooling=pooling_to_add(facts),
        )
    if facts.causal is False:
        return Resolved(
            Task.embedding,
            "its header says attention is not causal",
            pooling=pooling_to_add(facts),
        )
    return Resolved(Task.chat, "its header names no pooling and attention is causal")


def pooling_to_add(facts: RunFacts | None) -> str | None:
    """The `--pooling` an embedder needs, which is none when its header names one.

    A header that names no pooling type (or `none`, which `/v1/embeddings`
    refuses) leaves llama-server nothing to pool with: `mean` for a model whose
    attention is not causal, which is what an encoder is trained for, and `last`
    for a decoder. An unreadable header says nothing about it, so nothing is
    added.
    """
    if facts is None:
        return None
    if facts.pooling_type is not None and facts.pooling_type != 0:
        return None
    return "mean" if facts.causal is False else "last"


# --- what an embedder or reranker is started with unless the profile says ---------

#: Curated flag -> (default, its command-line spellings, its environment variable).
#: The numbers are arithmetic, not folklore: Qwen3-Embedding-0.6B keeps about 112 KiB
#: of cache a token, so 4,096 tokens is about 0.44 GiB where the model's own 32,768
#: would be 3.5; four slots of 1,024 tokens hold a 450-token passage; and a model
#: that is not causal can embed no more tokens at once than one physical batch holds
#: (llama-server answers a longer input with an HTTP 500 naming the batch), so the
#: batch is as large as a passage plus its prefix will ever be.
SEARCH_DEFAULTS: dict[str, tuple[int, tuple[str, ...], str]] = {
    "contextSize": (4096, ("-c", "--ctx-size"), "LLAMA_ARG_CTX_SIZE"),
    "parallelSlots": (4, ("-np", "--parallel"), "LLAMA_ARG_N_PARALLEL"),
    "batchSize": (2048, ("-b", "--batch-size"), "LLAMA_ARG_BATCH"),
    "ubatchSize": (2048, ("-ub", "--ubatch-size"), "LLAMA_ARG_UBATCH"),
}


def with_search_defaults(
    flags: Mapping[str, object] | None,
    env: Mapping[str, str] | None,
    extra_args: Sequence[str] | None,
) -> dict[str, object]:
    """The curated flags with a search model's defaults under them.

    **The operator's own choice always wins**, in whichever of the three places
    they made it: a curated flag the profile sets, the engine's environment
    variable in the runtime's `env` (argv beats the environment in llama.cpp, so a
    default put on the command line would have beaten it), or the same flag typed
    into `extraArgs`. A default is only ever filling a blank.
    """
    merged = dict(flags or {})
    typed = [str(a).strip() for a in (extra_args or [])]
    for key, (value, spellings, variable) in SEARCH_DEFAULTS.items():
        if merged.get(key) is not None:
            continue
        if env is not None and env.get(variable) not in (None, ""):
            continue
        if any(a in spellings or a.startswith(tuple(s + "=" for s in spellings)) for a in typed):
            continue
        merged[key] = value
    return merged


# --- what a running engine was started as, read off its command line -------------


def started_as(argv: Sequence[str], env: Mapping[str, str] | None) -> tuple[bool, bool]:
    """`(embeddings, reranking)` for a command line, as `RuntimeCapabilities` says them.

    From the argv the agent launched the engine with and the environment it gave
    it, because that is the only record there is: `/props` and `/v1/models`
    carry no pooling or embedding field (b11552). So it holds however the runtime
    came to carry the flag: `mode`, a Library profile, or `extraArgs`. A
    reranker is not also an embedder: llama-server turns embedding on for it,
    and answers `/v1/embeddings` with zeros.
    """
    tokens = set(argv)
    variables = env or {}
    embedding = bool(tokens & {"--embedding", "--embeddings"}) or _on(
        variables.get("LLAMA_ARG_EMBEDDINGS")
    )
    reranking = bool(tokens & {"--reranking", "--rerank"}) or _on(
        variables.get("LLAMA_ARG_RERANKING")
    )
    return embedding and not reranking, reranking


def _on(value: str | None) -> bool:
    """llama.cpp's reading of a switch in the environment."""
    return value is not None and value.strip().lower() in ("1", "true", "on", "enabled")
