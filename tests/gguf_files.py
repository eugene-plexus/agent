"""Small GGUF files with exactly the keys a test names, and no tensors.

A header is all the agent ever reads of a model before it starts the engine, so a
header is all a test needs: `write_gguf(path, architecture="qwen3", pooling_type=3)`
is a file the real reader parses the way it parses an embedding model's.
"""

from __future__ import annotations

import struct
from pathlib import Path

# GGUF value types, by id.
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64 = range(12)


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def string_array(items: list[str]) -> bytes:
    """An array of strings: the shape a vocabulary has."""
    body = b"".join(_string(item) for item in items)
    return struct.pack("<I", STRING) + struct.pack("<Q", len(items)) + body


def float_array(count: int) -> bytes:
    return (
        struct.pack("<I", F32) + struct.pack("<Q", count) + struct.pack(f"<{count}f", *range(count))
    )


def write_gguf(
    path: Path,
    *,
    architecture: str | None = "qwen3",
    pooling_type: int | None = None,
    pooling_kind: int = U32,
    causal: bool | None = None,
    classifier_labels: list[str] | None = None,
    context_length: int | None = 4096,
    vocabulary: int = 0,
    before: list[tuple[str, int, bytes]] | None = None,
    after: list[tuple[str, int, bytes]] | None = None,
    architecture_last: bool = False,
    version: int = 3,
) -> Path:
    """Write a GGUF header holding the named keys, in the order a converter would.

    `vocabulary` adds a tokenizer array that many strings long, between the
    architecture keys and the end, which is where the real ones are and what the
    reader has to step over. `before` and `after` add keys at either end, and
    `architecture_last` moves `general.architecture` behind every key that is
    named for it, for a test about order.
    """
    entries: list[tuple[str, int, bytes]] = list(before or [])
    if architecture is not None and not architecture_last:
        entries.append(("general.architecture", STRING, _string(architecture)))
    prefix = f"{architecture}." if architecture is not None else "model."
    if context_length is not None:
        entries.append((prefix + "context_length", U32, struct.pack("<I", context_length)))
    if pooling_type is not None:
        fmt = {U32: "<I", I32: "<i", U64: "<Q", I64: "<q", U8: "<B"}[pooling_kind]
        entries.append((prefix + "pooling_type", pooling_kind, struct.pack(fmt, pooling_type)))
    if causal is not None:
        entries.append((prefix + "attention.causal", BOOL, struct.pack("<?", causal)))
    if classifier_labels is not None:
        entries.append(
            (prefix + "classifier.output_labels", ARRAY, string_array(classifier_labels))
        )
    if vocabulary:
        tokens = [f"token{n}" for n in range(vocabulary)]
        entries.append(("tokenizer.ggml.tokens", ARRAY, string_array(tokens)))
        entries.append(("tokenizer.ggml.scores", ARRAY, float_array(vocabulary)))
    entries.extend(after or [])
    if architecture is not None and architecture_last:
        entries.append(("general.architecture", STRING, _string(architecture)))

    body = (
        b"GGUF"
        + struct.pack("<I", version)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(entries))
    )
    for key, kind, value in entries:
        body += _string(key) + struct.pack("<I", kind) + value
    path.write_bytes(body)
    return path


# What the real files on this box carry (read 2026-10-11), as the keys that matter.
def qwen3_chat(path: Path) -> Path:
    """A chat model: no pooling, no attention key."""
    return write_gguf(path, architecture="qwen3", context_length=40960, vocabulary=300)


def qwen3_embedding(path: Path) -> Path:
    """Qwen3-Embedding-0.6B: a decoder with last pooling (3) and no attention key."""
    return write_gguf(
        path, architecture="qwen3", pooling_type=3, context_length=32768, vocabulary=300
    )


def nomic_embedding(path: Path) -> Path:
    """nomic-embed-text-v1.5: an encoder with mean pooling (1)."""
    return write_gguf(
        path, architecture="nomic-bert", pooling_type=1, causal=False, context_length=2048
    )


def qwen3_reranker(path: Path) -> Path:
    """The ggml-org Qwen3-Reranker-0.6B: rank pooling (4) and yes/no labels."""
    return write_gguf(
        path,
        architecture="qwen3",
        pooling_type=4,
        classifier_labels=["yes", "no"],
        context_length=40960,
        vocabulary=300,
    )


def bge_reranker(path: Path) -> Path:
    """The gpustack bge-reranker-v2-m3: a non-causal bert with neither pooling nor labels."""
    return write_gguf(path, architecture="bert", causal=False, context_length=8192)
