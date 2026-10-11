"""What a GGUF header says a file is for (MR3), read the way a launch reads it.

The agent decides how to start a model from three keys. Every case here is a file
the real reader has to parse, and the ones that matter are the real files' shapes
(read off this box on 2026-10-11): a chat model has none of the keys, so "absent"
is an answer and can only be given at the end of the block; an embedder carries a
pooling type; a reranker carries rank pooling and its yes/no labels; and the
gpustack bge-reranker carries only a non-causal bert, which is the one a header
cannot tell from an embedder.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from eugene_plexus_agent import gguf_context
from eugene_plexus_agent.gguf_context import POOLING_RANK, RunFacts, run_facts

from . import gguf_files as g


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    gguf_context._FACTS.clear()


def test_a_chat_model_has_none_of_the_keys(tmp_path: Path) -> None:
    facts = run_facts(g.qwen3_chat(tmp_path / "chat.gguf"))
    assert facts == RunFacts(pooling_type=None, causal=None, has_classifier_labels=False)


def test_an_embedder_with_last_pooling(tmp_path: Path) -> None:
    """Qwen3-Embedding-0.6B: a decoder, so no attention key, and pooling 3."""
    facts = run_facts(g.qwen3_embedding(tmp_path / "embedding.gguf"))
    assert facts == RunFacts(pooling_type=3, causal=None, has_classifier_labels=False)


def test_an_encoder_names_mean_pooling_and_non_causal_attention(tmp_path: Path) -> None:
    facts = run_facts(g.nomic_embedding(tmp_path / "nomic.gguf"))
    assert facts == RunFacts(pooling_type=1, causal=False, has_classifier_labels=False)


def test_a_reranker_names_rank_pooling_and_its_labels(tmp_path: Path) -> None:
    facts = run_facts(g.qwen3_reranker(tmp_path / "reranker.gguf"))
    assert facts == RunFacts(pooling_type=POOLING_RANK, causal=None, has_classifier_labels=True)


def test_a_cross_encoder_with_neither_key_reads_as_an_encoder(tmp_path: Path) -> None:
    """The one the header cannot name (the gpustack bge-reranker-v2-m3): the only
    thing it says is that attention is not causal."""
    facts = run_facts(g.bge_reranker(tmp_path / "bge.gguf"))
    assert facts == RunFacts(pooling_type=None, causal=False, has_classifier_labels=False)


def test_empty_classifier_labels_are_not_labels(tmp_path: Path) -> None:
    facts = run_facts(g.write_gguf(tmp_path / "m.gguf", classifier_labels=[]))
    assert facts is not None and facts.has_classifier_labels is False


@pytest.mark.parametrize("kind", [g.U32, g.I32, g.U64, g.I64, g.U8])
def test_the_pooling_type_is_read_whatever_integer_it_was_written_as(
    tmp_path: Path, kind: int
) -> None:
    facts = run_facts(g.write_gguf(tmp_path / "m.gguf", pooling_type=2, pooling_kind=kind))
    assert facts is not None and facts.pooling_type == 2


def test_the_keys_are_found_past_a_vocabulary_and_before_the_architecture(tmp_path: Path) -> None:
    """GGUF promises no order. The tokenizer arrays (150,000 strings in a real
    file) are stepped over, and a key that comes after them, or before
    `general.architecture` names the architecture it belongs to, is still found."""
    model = g.write_gguf(
        tmp_path / "m.gguf",
        architecture="qwen3",
        pooling_type=3,
        causal=False,
        vocabulary=5000,
        architecture_last=True,
        after=[("tokenizer.chat_template", g.STRING, struct.pack("<Q", 3) + b"{%}")],
    )
    assert run_facts(model) == RunFacts(pooling_type=3, causal=False, has_classifier_labels=False)


def test_a_key_under_another_architecture_is_not_this_files(tmp_path: Path) -> None:
    """Two candidates and a named architecture: the one under that name wins."""
    other = ("clip.pooling_type", g.U32, struct.pack("<I", 4))
    model = g.write_gguf(tmp_path / "m.gguf", architecture="qwen3", pooling_type=1, before=[other])
    facts = run_facts(model)
    assert facts is not None and facts.pooling_type == 1


def test_a_lone_key_with_no_architecture_to_check_it_against_is_taken(tmp_path: Path) -> None:
    model = g.write_gguf(tmp_path / "m.gguf", architecture=None, pooling_type=4)
    facts = run_facts(model)
    assert facts is not None and facts.pooling_type == 4


def test_two_candidates_and_no_architecture_is_no_answer(tmp_path: Path) -> None:
    both = [
        ("a.pooling_type", g.U32, struct.pack("<I", 1)),
        ("b.pooling_type", g.U32, struct.pack("<I", 4)),
    ]
    facts = run_facts(
        g.write_gguf(tmp_path / "m.gguf", architecture=None, context_length=None, before=both)
    )
    assert facts is not None and facts.pooling_type is None


# --- a header that cannot be read is no answer, never a chat model -------------------


def test_no_such_file_is_no_answer(tmp_path: Path) -> None:
    assert run_facts(tmp_path / "missing.gguf") is None


def test_a_directory_is_no_answer(tmp_path: Path) -> None:
    assert run_facts(tmp_path) is None


def test_a_file_that_is_not_gguf_is_no_answer(tmp_path: Path) -> None:
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"this is not a model at all")
    assert run_facts(path) is None


def test_version_one_is_no_answer(tmp_path: Path) -> None:
    assert run_facts(g.write_gguf(tmp_path / "old.gguf", version=1)) is None


def test_a_download_cut_off_inside_the_header_is_no_answer(tmp_path: Path) -> None:
    whole = g.qwen3_embedding(tmp_path / "whole.gguf").read_bytes()
    for cut in (3, 10, 25, len(whole) // 2, len(whole) - 3):
        path = tmp_path / f"cut-{cut}.gguf"
        path.write_bytes(whole[:cut])
        assert run_facts(path) is None, cut


def test_a_key_count_nobody_wrote_is_no_answer(tmp_path: Path) -> None:
    path = tmp_path / "huge.gguf"
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1 << 40)
    )
    assert run_facts(path) is None


def test_a_key_name_nobody_wrote_is_no_answer(tmp_path: Path) -> None:
    """A length that claims 2**60 bytes is refused, not allocated."""
    path = tmp_path / "long-name.gguf"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", 1 << 60)
    )
    assert run_facts(path) is None


def test_an_array_that_claims_more_than_the_file_holds_is_no_answer(tmp_path: Path) -> None:
    claim = struct.pack("<I", g.STRING) + struct.pack("<Q", 1 << 50)
    path = g.write_gguf(tmp_path / "m.gguf", after=[("tokenizer.huge", g.ARRAY, claim)])
    assert run_facts(path) is None
    fixed = struct.pack("<I", g.F32) + struct.pack("<Q", 1 << 62)
    path = g.write_gguf(tmp_path / "n.gguf", after=[("tokenizer.huge", g.ARRAY, fixed)])
    assert run_facts(path) is None


def test_an_unknown_value_type_is_no_answer(tmp_path: Path) -> None:
    path = g.write_gguf(tmp_path / "m.gguf", after=[("odd.key", 99, b"\x00" * 8)])
    assert run_facts(path) is None


# --- the cache ----------------------------------------------------------------------


def test_a_file_is_read_once_while_it_stays_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch reads the header, and so do the admission check before it and the
    plan for it. A big vocabulary makes that a tenth of a second each."""
    model = g.qwen3_chat(tmp_path / "chat.gguf")
    reads: list[Path] = []
    real = gguf_context._read_run_facts

    def counting(path: Path) -> RunFacts | None:
        reads.append(path)
        return real(path)

    monkeypatch.setattr(gguf_context, "_read_run_facts", counting)
    assert run_facts(model) == run_facts(str(model))
    assert len(reads) == 1


def test_a_file_that_changes_is_read_again(tmp_path: Path) -> None:
    """The same path, a different file: a download finished over a partial, or a
    model replaced. A stale answer would start an embedder as chat."""
    path = tmp_path / "model.gguf"
    g.qwen3_chat(path)
    assert run_facts(path) == RunFacts(None, None, False)
    g.qwen3_embedding(path)
    assert run_facts(path) == RunFacts(pooling_type=3, causal=None, has_classifier_labels=False)


def test_the_cache_is_bounded(tmp_path: Path) -> None:
    for number in range(gguf_context._FACTS_KEPT + 8):
        run_facts(g.write_gguf(tmp_path / f"m{number}.gguf", context_length=number + 1))
    assert len(gguf_context._FACTS) <= gguf_context._FACTS_KEPT


def test_the_trained_context_reader_is_unchanged(tmp_path: Path) -> None:
    """`trained_context` is what a profile build uses, and still stops at its answer."""
    model = g.write_gguf(tmp_path / "m.gguf", context_length=262144, pooling_type=3)
    assert gguf_context.trained_context(model) == 262144
