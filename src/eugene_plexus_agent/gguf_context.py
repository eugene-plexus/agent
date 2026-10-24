"""What a launch and a profile build need from a GGUF header.

Two things, read from the header and never from tensor data.
`trained_context` is the number a profile build needs: the library
parses GGUF fully and is asked first, and this is the fallback for a node that
cannot reach it, so a build never tries a context the model was not trained
for. It stops at the answer. `run_facts` is what a launch needs (MR3): how the
file is meant to be used, read from three keys, so the adapter can start an
embedding or reranking model the way it has to be started. Those keys can sit
anywhere in the block, so it reads the whole of it. Non-causal models also
need the tensor table to identify a classifier head when metadata omits it.

Duplicated rather than shared, by the polyrepo rule: the library's
`formats/gguf.py` reads the same keys for the same words (`is_embedding`,
`is_rerank`), and the two must agree.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

MAGIC = b"GGUF"
# Fixed-size value types by GGUF type id.
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9
_UINT32, _INT32, _UINT64, _INT64 = 4, 5, 10, 11
# A header this long without the key means something is wrong with the file.
_MAX_KEYS = 100_000


def architecture_of(path: Path) -> str | None:
    """Architecture for an Ollama report, read only from the GGUF header."""
    try:
        with path.open("rb") as stream:
            if stream.read(4) != MAGIC:
                return None
            version, _, count = struct.unpack("<IQQ", stream.read(20))
            if version not in (2, 3) or count > _MAX_KEYS:
                return None
            for _ in range(count):
                key = _string(stream)
                kind = struct.unpack("<I", stream.read(4))[0]
                if key == b"general.architecture" and kind == _STRING:
                    return _string(stream).decode("utf-8")
                _skip(stream, kind)
    except (OSError, ValueError, struct.error):
        return None
    return None


def _u64(stream: BinaryIO) -> int:
    return int(struct.unpack("<Q", stream.read(8))[0])


def _string(stream: BinaryIO) -> bytes:
    return stream.read(_u64(stream))


def _skip(stream: BinaryIO, kind: int) -> None:
    if kind in _FIXED:
        stream.seek(_FIXED[kind], 1)
    elif kind == _STRING:
        stream.seek(_u64(stream), 1)
    elif kind == _ARRAY:
        inner = struct.unpack("<I", stream.read(4))[0]
        count = _u64(stream)
        if inner in _FIXED:
            stream.seek(_FIXED[inner] * count, 1)
        else:
            for _ in range(count):
                _skip(stream, inner)
    else:
        raise ValueError(f"unknown GGUF value type {kind}")


def trained_context(path: Path) -> int | None:
    """`<architecture>.context_length`, or None when it cannot be read."""
    try:
        with path.open("rb") as stream:
            if stream.read(4) != MAGIC:
                return None
            version = struct.unpack("<I", stream.read(4))[0]
            if version < 2:
                return None
            _u64(stream)  # tensor count
            keys = _u64(stream)
            for _ in range(min(keys, _MAX_KEYS)):
                key = _string(stream).decode("utf-8", errors="replace")
                kind = struct.unpack("<I", stream.read(4))[0]
                if key.endswith(".context_length") and kind in (_UINT32, _INT32, _UINT64, _INT64):
                    size = 4 if kind in (_UINT32, _INT32) else 8
                    fmt = {_UINT32: "<I", _INT32: "<i", _UINT64: "<Q", _INT64: "<q"}[kind]
                    value = struct.unpack(fmt, stream.read(size))[0]
                    return int(value) if value > 0 else None
                _skip(stream, kind)
    except (OSError, ValueError, struct.error):
        return None
    return None


# --- how the file is meant to be used (MR3) -----------------------------------

_BOOL = 7
_INT_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
# No key in a real file is anywhere near this long; a longer "name" is a file
# that is not what it says, and reading it would mean allocating whatever the
# length claims.
_MAX_KEY_BYTES = 1 << 16


@dataclass(frozen=True)
class ChatTemplateFacts:
    """A complete header's template facts, captured independently of filenames."""

    architecture: str | None
    name: str | None
    embedded: str | None


def chat_template_facts(path: Path | str) -> ChatTemplateFacts | None:
    found: dict[str, str] = {}
    wanted = {"general.architecture", "general.name", "tokenizer.chat_template"}
    try:
        with Path(path).open("rb") as stream:
            if stream.read(4) != MAGIC:
                return None
            version, _, keys = struct.unpack("<IQQ", stream.read(20))
            if version not in (2, 3) or keys > _MAX_KEYS:
                return None
            for _ in range(keys):
                size = _u64(stream)
                if size > _MAX_KEY_BYTES:
                    return None
                key = stream.read(size).decode("utf-8")
                kind = struct.unpack("<I", stream.read(4))[0]
                if key in wanted and kind == _STRING:
                    size = _u64(stream)
                    if size > 16 * 1024 * 1024:
                        return None
                    raw = stream.read(size)
                    if len(raw) != size:
                        return None
                    found[key] = raw.decode("utf-8")
                else:
                    _skip(stream, kind)
            if stream.tell() > Path(path).stat().st_size:
                return None
    except (OSError, ValueError, struct.error, OverflowError):
        return None
    return ChatTemplateFacts(
        found.get("general.architecture"),
        found.get("general.name"),
        found.get("tokenizer.chat_template") or None,
    )


#: `<arch>.pooling_type` values (llama.cpp's `llama_pooling_type`). Only the one
#: that matters is named: 4 is `rank`, a reranker's.
POOLING_RANK = 4

_POOLING_SUFFIX = ".pooling_type"
_CAUSAL_SUFFIX = ".attention.causal"
_LABELS_SUFFIX = ".classifier.output_labels"
_ARCHITECTURE = "general.architecture"


@dataclass(frozen=True)
class RunFacts:
    """The three things a header says about what a file is for.

    Each is `None`/`False` when its key is absent, which is the common case for
    a chat model: a pooling type and non-causal attention are what an embedding
    model carries, and classifier labels what a reranker does.
    """

    pooling_type: int | None
    """`<arch>.pooling_type`: 0 none, 1 mean, 2 cls, 3 last, 4 rank."""
    causal: bool | None
    """`<arch>.attention.causal`: absent for a decoder, False for an encoder."""
    has_classifier_labels: bool
    """`<arch>.classifier.output_labels` is there and not empty."""
    has_classifier_tensors: bool = False
    """A non-causal model's tensor table names a `cls.*` classification head."""


#: Read once per file as it stands: a launch reads a header, and so do the
#: admission check before it and the plan for it, seconds apart.
_FACTS: dict[tuple[str, int, int], RunFacts | None] = {}
_FACTS_KEPT = 64


def run_facts(path: Path | str) -> RunFacts | None:
    """How the file at `path` is meant to be used, or None when it cannot be read.

    None is a different answer from "a chat model": it means the header did not
    parse (a missing file, a directory, something that is not GGUF, a cut-off
    download), and the caller says so rather than guessing.

    Walks every key because GGUF does not order them: `general.architecture`
    comes first in practice and the pooling and attention keys come well before
    the tokenizer, but nothing promises either, and a chat model has none of
    them, so "absent" can only be said at the end of the block. The vocabulary
    arrays are stepped over, never read.
    """
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return None
    key = (str(target), stat.st_mtime_ns, stat.st_size)
    if key in _FACTS:
        return _FACTS[key]
    facts = _read_run_facts(target)
    if len(_FACTS) >= _FACTS_KEPT:
        _FACTS.pop(next(iter(_FACTS)))
    _FACTS[key] = facts
    return facts


def _read_run_facts(path: Path) -> RunFacts | None:
    pooling: dict[str, int] = {}
    causal: dict[str, bool] = {}
    labels: dict[str, bool] = {}
    architecture: str | None = None
    try:
        # A large buffer: stepping over a 150,000-string vocabulary is a seek per
        # string, and a seek inside the buffer costs nothing.
        with path.open("rb", buffering=1 << 20) as stream:
            if stream.read(4) != MAGIC:
                return None
            version = struct.unpack("<I", stream.read(4))[0]
            if version < 2:
                return None
            tensors = _u64(stream)
            keys = _u64(stream)
            if keys > _MAX_KEYS:
                return None
            for _ in range(keys):
                size = _u64(stream)
                if size > _MAX_KEY_BYTES:
                    return None
                name = stream.read(size).decode("utf-8", errors="replace")
                kind = struct.unpack("<I", stream.read(4))[0]
                if name == _ARCHITECTURE and kind == _STRING:
                    architecture = _string(stream).decode("utf-8", errors="replace")
                elif name.endswith(_POOLING_SUFFIX) and kind in _INT_FORMATS:
                    fmt = _INT_FORMATS[kind]
                    pooling[name] = int(struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0])
                elif name.endswith(_CAUSAL_SUFFIX) and kind == _BOOL:
                    causal[name] = stream.read(1) != b"\x00"
                elif name.endswith(_LABELS_SUFFIX) and kind == _ARRAY:
                    labels[name] = _skip_array(stream) > 0
                else:
                    _skip(stream, kind)
            classifier = False
            if _pick(causal, architecture, _CAUSAL_SUFFIX) is False:
                if tensors > 1 << 20:
                    return None
                for _ in range(tensors):
                    size = _u64(stream)
                    if size > _MAX_KEY_BYTES:
                        return None
                    name_bytes = stream.read(size)
                    dims = struct.unpack("<I", stream.read(4))[0]
                    if not 1 <= dims <= 8:
                        return None
                    classifier |= name_bytes.startswith(b"cls.")
                    stream.seek(dims * 8 + 4 + 8, 1)  # dimensions, type, offset; no weights
            # A seek past the end of the file is not an error, so a download cut
            # off inside the last array of fixed-size numbers would read as whole.
            if stream.tell() > path.stat().st_size:
                return None
    except (OSError, ValueError, struct.error, OverflowError, MemoryError):
        return None
    return RunFacts(
        pooling_type=_pick(pooling, architecture, _POOLING_SUFFIX),
        causal=_pick(causal, architecture, _CAUSAL_SUFFIX),
        has_classifier_labels=_pick(labels, architecture, _LABELS_SUFFIX) is True,
        has_classifier_tensors=classifier,
    )


def _pick[T](found: dict[str, T], architecture: str | None, suffix: str) -> T | None:
    """The key under this file's architecture, or the only one there is."""
    if architecture is not None and (architecture + suffix) in found:
        return found[architecture + suffix]
    return next(iter(found.values())) if len(found) == 1 else None


def _skip_array(stream: BinaryIO) -> int:
    """Step over an array value and say how many elements it held."""
    inner = struct.unpack("<I", stream.read(4))[0]
    count = _u64(stream)
    if inner in _FIXED:
        stream.seek(_FIXED[inner] * count, 1)
    else:
        for _ in range(count):
            _skip(stream, inner)
    return count


# --- what a file can draft with (SD1) -----------------------------------------

_BLOCK_COUNT_SUFFIX = ".block_count"
_NEXTN_SUFFIX = b".nextn.eh_proj.weight"


@dataclass(frozen=True)
class DraftingFacts:
    """A file's drafting kind and target shape, from its header and tensor table.

    MTP follows llama.cpp's own rule (b11719, `common_speculative_types_from_gguf`):
    `blk.<block_count-1>.nextn.eh_proj.weight`. Bound drafters use their architecture.

    The library reads the same tensors for the same words
    (`GgufMetadata.mtp_head`, `draft_kind`), and the two must agree.
    """

    mtp_tensors: bool
    """The last block carries an MTP head."""
    has_first_block: bool
    """The table names a `blk.0.` tensor: a whole model, not a separate head."""

    architecture: str | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    vocab_size: int | None = None
    target_layers: tuple[int, ...] | None = None
    target_hidden_size: int | None = None
    block_size: int = 16
    markov_head: bool = False

    @property
    def mtp_head(self) -> bool:
        """A whole model with its own head: drafts as `draft-mtp` with no file."""
        return self.mtp_tensors and self.has_first_block

    @property
    def companion_type(self) -> str | None:
        """llama.cpp's `--spec-type` for drafting with this file as
        `--spec-draft-model`: architecture first, then `draft-mtp` for an
        MTP head or `draft-simple` for a whole model without one. Eugene
        always passes the type, including EAGLE3 which llama.cpp does not infer."""
        if self.architecture == "dflash":
            return "draft-dspark" if self.markov_head else "draft-dflash"
        if self.architecture == "eagle3":
            return "draft-eagle3"
        if self.mtp_tensors:
            return "draft-mtp"
        if self.has_first_block:
            return "draft-simple"
        return None

    @property
    def target_problem(self) -> str | None:
        if self.target_hidden_size is None or self.target_hidden_size <= 0:
            return f"missing or invalid {self.architecture}.target_hidden_size"
        layers = self.target_layers
        if not layers or any(layer < 0 for layer in layers):
            return f"missing or invalid {self.architecture}.target_layers (1 to 512 layer ids)"
        if self.architecture == "eagle3" and len(layers) != 3:
            return "eagle3.target_layers must name exactly 3 target layers"
        return None

    @property
    def target_min_layers(self) -> int:
        return max(self.target_layers or (0,)) + (self.architecture != "eagle3")


_DRAFTING: dict[tuple[tuple[str, int, int], ...], DraftingFacts | None] = {}


def drafting_facts(path: Path | str) -> DraftingFacts | None:
    """What the file at `path` can draft with, or None when it cannot be read.

    Reads the key block (stepping over the vocabulary) and the tensor table,
    never a weight. Cached per file as it stands, like `run_facts`.
    """
    from .model_copies import shards_of

    targets = tuple(Path(p) for p in shards_of(str(path)))
    try:
        stamps = [(target, target.stat()) for target in targets]
    except OSError:
        return None
    key = tuple((str(target), stat.st_mtime_ns, stat.st_size) for target, stat in stamps)
    if key in _DRAFTING:
        return _DRAFTING[key]
    facts = _read_drafting_facts(targets[0])
    if facts is not None and facts.architecture == "deepseek4" and len(targets) > 1:
        parts = [facts]
        for target in targets[1:]:
            part = _read_drafting_facts(target)
            if part is None or (part.architecture, part.block_count, part.vocab_size) != (
                facts.architecture,
                facts.block_count,
                facts.vocab_size,
            ):
                return None
            parts.append(part)
        facts = replace(
            facts,
            mtp_tensors=any(part.mtp_tensors for part in parts),
            has_first_block=any(part.has_first_block for part in parts),
        )
    if len(_DRAFTING) >= _FACTS_KEPT:
        _DRAFTING.pop(next(iter(_DRAFTING)))
    _DRAFTING[key] = facts
    return facts


def _read_drafting_facts(path: Path) -> DraftingFacts | None:
    blocks: dict[str, int] = {}
    scalars: dict[str, int] = {}
    layers: dict[str, tuple[int, ...] | None] = {}
    vocab: int | None = None
    architecture: str | None = None
    nextn: set[bytes] = set()
    first_block = False
    markov_head = False
    try:
        with path.open("rb", buffering=1 << 20) as stream:
            if stream.read(4) != MAGIC:
                return None
            version = struct.unpack("<I", stream.read(4))[0]
            if version < 2:
                return None
            tensors = _u64(stream)
            keys = _u64(stream)
            if keys > _MAX_KEYS or tensors > 1 << 20:
                return None
            for _ in range(keys):
                size = _u64(stream)
                if size > _MAX_KEY_BYTES:
                    return None
                name = stream.read(size).decode("utf-8", errors="replace")
                kind = struct.unpack("<I", stream.read(4))[0]
                if name == _ARCHITECTURE and kind == _STRING:
                    architecture = _string(stream).decode("utf-8", errors="replace")
                elif name.endswith(_BLOCK_COUNT_SUFFIX) and kind in _INT_FORMATS:
                    fmt = _INT_FORMATS[kind]
                    blocks[name] = int(struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0])
                elif (
                    name.endswith((".embedding_length", ".target_hidden_size", ".block_size"))
                    and kind in _INT_FORMATS
                ):
                    fmt = _INT_FORMATS[kind]
                    scalars[name] = int(struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0])
                elif name.endswith(".target_layers") and kind == _ARRAY:
                    inner = struct.unpack("<I", stream.read(4))[0]
                    target_count = _u64(stream)
                    if inner in _INT_FORMATS and 1 <= target_count <= 512:
                        fmt = _INT_FORMATS[inner]
                        layers[name] = tuple(
                            int(struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0])
                            for _ in range(target_count)
                        )
                    elif inner in _FIXED:
                        stream.seek(_FIXED[inner] * target_count, 1)
                        layers[name] = None
                    else:
                        return None
                elif name == "tokenizer.ggml.tokens" and kind == _ARRAY:
                    vocab = _skip_array(stream)
                else:
                    _skip(stream, kind)
            for _ in range(tensors):
                size = _u64(stream)
                if size > _MAX_KEY_BYTES:
                    return None
                name_bytes = stream.read(size)
                dims = struct.unpack("<I", stream.read(4))[0]
                if not 1 <= dims <= 8:
                    return None
                first_block |= name_bytes.startswith(b"blk.0.")
                markov_head |= name_bytes == b"markov_w1.weight"
                if name_bytes.endswith(_NEXTN_SUFFIX):
                    nextn.add(name_bytes)
                stream.seek(dims * 8 + 4 + 8, 1)  # dimensions, type, offset; no weights
            if stream.tell() > path.stat().st_size:
                return None
    except (OSError, ValueError, struct.error, OverflowError, MemoryError):
        return None
    count = _pick(blocks, architecture, _BLOCK_COUNT_SUFFIX)
    last = f"blk.{count - 1}".encode() + _NEXTN_SUFFIX if count else None
    return DraftingFacts(
        mtp_tensors=last in nextn,
        has_first_block=first_block,
        architecture=architecture,
        embedding_length=scalars.get(f"{architecture}.embedding_length"),
        block_count=blocks.get(f"{architecture}.block_count"),
        vocab_size=vocab,
        target_layers=layers.get(f"{architecture}.target_layers"),
        target_hidden_size=scalars.get(f"{architecture}.target_hidden_size"),
        block_size=scalars.get(f"{architecture}.block_size", 16),
        markov_head=markov_head,
    )
