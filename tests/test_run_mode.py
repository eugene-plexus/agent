"""A model is started as what it is (MR3): the argv per mode, and who wins.

`RuntimeSpec.mode` says what the engine is started to do with the file, and `auto`
is the file's own header. The tests are of the command line itself, because that
is what llama-server is told and the only thing it will believe: an embedding model
started without `--embeddings` answers neither embeddings (501) nor chat (500)
(measured on b10948 and b11552).

Four things have to hold, and each has its own tests:

* every mode adds exactly what it should, and in the right place (after the curated
  flags, before the operator's `extraArgs`);
* an explicit `chat` beats the header, and an explicit `embedding` or `reranking`
  beats a header that says chat;
* the operator's own setting beats a default, in each of the three places they can
  make one;
* a chat model's command line is untouched (the pin is `test_chat_argv_is_unchanged`).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_agent._generated.models import EngineKind, Origin, RuntimeMode, RuntimeSpec
from eugene_plexus_agent.engines import LlamaCppAdapter, run_mode
from eugene_plexus_agent.engines.base import DiscoveredBinary
from eugene_plexus_agent.engines.run_mode import Task

from . import gguf_files as g

DEFAULTS = [
    "--ctx-size",
    "4096",
    "--parallel",
    "4",
    "--batch-size",
    "2048",
    "--ubatch-size",
    "2048",
]


@pytest.fixture
def adapter() -> LlamaCppAdapter:
    return LlamaCppAdapter()


@pytest.fixture
def binary(tmp_path: Path) -> DiscoveredBinary:
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    return DiscoveredBinary(path=exe, origin=Origin.path, version="b11552")


def _spec(model: Path, **overrides: Any) -> RuntimeSpec:
    body: dict[str, Any] = {
        "name": "search",
        "engine": EngineKind.llama_cpp,
        "modelPath": str(model),
    }
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


def _argv(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, model: Path, **overrides: Any
) -> list[str]:
    return adapter.build_argv(_spec(model, **overrides), binary, port=8090)


def _after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# --- auto: the header decides -------------------------------------------------------


@pytest.mark.parametrize(
    ("write", "task"),
    [
        (g.qwen3_chat, Task.chat),
        (g.qwen3_embedding, Task.embedding),
        (g.nomic_embedding, Task.embedding),
        (g.qwen3_reranker, Task.reranking),
        # Not told apart from an encoder: the header says only that it is not causal.
        (g.bge_reranker, Task.embedding),
    ],
)
def test_auto_reads_the_header(tmp_path: Path, write: Any, task: Task) -> None:
    assert run_mode.resolve(None, str(write(tmp_path / "m.gguf"))).task is task
    assert run_mode.resolve("auto", str(write(tmp_path / "m.gguf"))).task is task
    assert run_mode.resolve(RuntimeMode.auto, str(write(tmp_path / "m.gguf"))).task is task


@pytest.mark.parametrize(
    "causal, name, expected",
    [
        (False, "cls.output.weight", True),
        (False, "cls.weight", True),
        (True, "cls.output.weight", False),
        (False, "blk.0.attn_output.weight", False),
    ],
)
def test_classifier_tensor_selects_reranking(tmp_path, adapter, binary, causal, name, expected):
    path = g.write_gguf(tmp_path / "head.gguf", architecture="bert", causal=causal)
    header = bytearray(path.read_bytes())
    struct.pack_into("<Q", header, 8, 1)
    encoded = name.encode()
    header += struct.pack("<Q", len(encoded)) + encoded + struct.pack("<I Q I Q", 1, 8, 0, 0)
    path.write_bytes(header)
    argv = _argv(adapter, binary, path)
    assert ("--reranking" in argv) is expected
    if expected:
        assert "--embeddings" not in argv and "--pooling" not in argv
        assert "classifier" in run_mode.resolve("auto", str(path)).why


def test_pooling_type_4_is_a_reranker_and_any_other_pooling_type_an_embedder(
    tmp_path: Path,
) -> None:
    for pooling, task in (
        (0, Task.embedding),
        (1, Task.embedding),
        (2, Task.embedding),
        (3, Task.embedding),
        (4, Task.reranking),
    ):
        model = g.write_gguf(tmp_path / f"p{pooling}.gguf", pooling_type=pooling)
        assert run_mode.resolve("auto", str(model)).task is task, pooling


def test_classifier_labels_make_a_reranker_whatever_the_pooling_says(tmp_path: Path) -> None:
    """The library says `rerank` for either, and the agent must agree with it."""
    model = g.write_gguf(tmp_path / "m.gguf", pooling_type=1, classifier_labels=["a", "b"])
    assert run_mode.resolve("auto", str(model)).task is Task.reranking


def test_a_non_causal_model_is_an_embedder_even_with_no_pooling_type(tmp_path: Path) -> None:
    resolved = run_mode.resolve("auto", str(g.write_gguf(tmp_path / "m.gguf", causal=False)))
    assert resolved.task is Task.embedding


def test_causal_attention_stated_outright_is_still_chat(tmp_path: Path) -> None:
    resolved = run_mode.resolve("auto", str(g.write_gguf(tmp_path / "m.gguf", causal=True)))
    assert resolved.task is Task.chat


def test_auto_on_a_file_that_will_not_read_starts_it_as_chat_as_before(tmp_path: Path) -> None:
    """Not a guess about the file: it is what every model did before this existed, and
    if the file is what is wrong, the engine says so in its own words."""
    for model in (tmp_path / "missing.gguf", tmp_path):
        resolved = run_mode.resolve("auto", str(model))
        assert resolved.task is Task.chat
        assert "could not be read" in resolved.why


def test_every_resolution_says_why(tmp_path: Path) -> None:
    for write in (g.qwen3_chat, g.qwen3_embedding, g.nomic_embedding, g.qwen3_reranker):
        assert run_mode.resolve("auto", str(write(tmp_path / "m.gguf"))).why


# --- an explicit mode beats the header ---------------------------------------------


def test_chat_beats_an_embedding_header(tmp_path: Path) -> None:
    """The override for a file whose header misleads, and the one that matters: an
    explicit `chat` has to win over `auto`'s reading of the same file."""
    model = g.qwen3_embedding(tmp_path / "m.gguf")
    assert run_mode.resolve("auto", str(model)).task is Task.embedding
    assert run_mode.resolve("chat", str(model)).task is Task.chat
    assert run_mode.resolve(RuntimeMode.chat, str(model)).task is Task.chat


def test_chat_beats_a_reranker_header(tmp_path: Path) -> None:
    model = g.qwen3_reranker(tmp_path / "m.gguf")
    assert run_mode.resolve("chat", str(model)).task is Task.chat


def test_embedding_and_reranking_beat_a_chat_header(tmp_path: Path) -> None:
    model = g.qwen3_chat(tmp_path / "m.gguf")
    assert run_mode.resolve("embedding", str(model)).task is Task.embedding
    assert run_mode.resolve("reranking", str(model)).task is Task.reranking


def test_reranking_says_so_for_a_file_the_header_calls_an_embedder(tmp_path: Path) -> None:
    """The bge cross-encoder: auto reads it as an embedder, and a person says otherwise."""
    model = g.bge_reranker(tmp_path / "m.gguf")
    assert run_mode.resolve("auto", str(model)).task is Task.embedding
    assert run_mode.resolve("reranking", str(model)).task is Task.reranking


def test_an_unknown_mode_is_auto(tmp_path: Path) -> None:
    model = g.qwen3_embedding(tmp_path / "m.gguf")
    assert run_mode.resolve("something-newer", str(model)).task is Task.embedding


# --- the command line, per mode -----------------------------------------------------


def test_an_embedder_gets_embeddings_and_the_defaults(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    """Qwen3-Embedding names pooling 3, so no `--pooling`."""
    argv = _argv(adapter, binary, g.qwen3_embedding(tmp_path / "Qwen3-Embedding-0.6B-Q8_0.gguf"))
    assert argv[1:9] == [
        "--model",
        str(tmp_path / "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--alias",
        "Qwen3-Embedding-0.6B-Q8_0",
    ]
    assert argv.count("--embeddings") == 1
    assert "--pooling" not in argv and "--reranking" not in argv
    for flag, value in zip(DEFAULTS[::2], DEFAULTS[1::2], strict=True):
        assert argv.count(flag) == 1 and _after(argv, flag) == value, flag


def test_a_reranker_gets_reranking_and_the_defaults_and_nothing_about_pooling(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    argv = _argv(adapter, binary, g.qwen3_reranker(tmp_path / "reranker.gguf"))
    assert argv.count("--reranking") == 1
    assert "--embeddings" not in argv and "--pooling" not in argv
    for flag, value in zip(DEFAULTS[::2], DEFAULTS[1::2], strict=True):
        assert _after(argv, flag) == value, flag


def test_an_encoder_whose_header_names_no_pooling_is_told_mean(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    model = g.write_gguf(tmp_path / "bert.gguf", architecture="bert", causal=False)
    argv = _argv(adapter, binary, model)
    assert argv.count("--embeddings") == 1 and _after(argv, "--pooling") == "mean"


def test_a_decoder_whose_header_names_no_pooling_is_told_last(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    argv = _argv(adapter, binary, g.qwen3_chat(tmp_path / "m.gguf"), mode="embedding")
    assert argv.count("--embeddings") == 1 and _after(argv, "--pooling") == "last"


def test_a_header_that_names_none_is_given_a_pooling_it_can_serve(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    """Pooling `none` is what `/v1/embeddings` refuses."""
    model = g.write_gguf(tmp_path / "m.gguf", pooling_type=0, causal=False)
    assert _after(_argv(adapter, binary, model), "--pooling") == "mean"


def test_an_embedder_that_names_its_pooling_is_not_told_another(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    for write in (g.qwen3_embedding, g.nomic_embedding):
        assert "--pooling" not in _argv(adapter, binary, write(tmp_path / "m.gguf"))


def test_an_explicit_embedding_on_an_unreadable_file_adds_no_pooling(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    """Nothing says the header names none, so nothing is claimed."""
    argv = _argv(adapter, binary, tmp_path / "missing.gguf", mode="embedding")
    assert argv.count("--embeddings") == 1 and "--pooling" not in argv


def test_chat_on_an_embedding_file_is_the_command_line_of_a_chat_model(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    """auto against an explicit chat: same file, the two answers differ and the
    override's is exactly what a chat model got before MR3, plus the prompt cache
    settings every chat model carries since row 9 (and an embedder does not)."""
    model = g.qwen3_embedding(tmp_path / "m.gguf")
    auto = _argv(adapter, binary, model)
    chat = _argv(adapter, binary, model, mode="chat")
    assert "--embeddings" in auto
    assert chat == [
        str(binary.path),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--alias",
        "m",
        "--cache-ram",
        "8192",
        "--cache-reuse",
        "0",
        "--ctx-checkpoints",
        "32",
    ]
    assert "--cache-ram" not in auto


def test_the_mode_flags_come_after_the_curated_flags_and_before_extra_args(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    argv = _argv(
        adapter,
        binary,
        g.nomic_embedding(tmp_path / "m.gguf"),
        flags={"threads": 8, "noMmap": True},
        extraArgs=["--verbose"],
    )
    assert argv.index("--threads") < argv.index("--load-mode") < argv.index("--embeddings")
    assert argv.index("--embeddings") < argv.index("--verbose") == len(argv) - 1


# --- the operator's own setting beats a default --------------------------------------


def test_a_curated_flag_beats_its_default(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    argv = _argv(
        adapter,
        binary,
        g.qwen3_embedding(tmp_path / "m.gguf"),
        flags={"contextSize": 8192, "ubatchSize": 512},
    )
    assert argv.count("--ctx-size") == 1 and _after(argv, "--ctx-size") == "8192"
    assert argv.count("--ubatch-size") == 1 and _after(argv, "--ubatch-size") == "512"
    assert _after(argv, "--parallel") == "4" and _after(argv, "--batch-size") == "2048"


def test_a_flag_set_to_nothing_is_not_set(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    argv = _argv(
        adapter, binary, g.qwen3_embedding(tmp_path / "m.gguf"), flags={"contextSize": None}
    )
    assert _after(argv, "--ctx-size") == "4096"


@pytest.mark.parametrize(
    ("typed", "flag"),
    [
        (["--ctx-size", "2048"], "--ctx-size"),
        (["-c", "2048"], "--ctx-size"),
        (["--ctx-size=2048"], "--ctx-size"),
        (["--parallel", "1"], "--parallel"),
        (["-np", "1"], "--parallel"),
        (["-b", "512"], "--batch-size"),
        (["--batch-size=512"], "--batch-size"),
        (["-ub", "512"], "--ubatch-size"),
    ],
)
def test_a_flag_typed_into_extra_args_is_not_given_a_default_to_argue_with(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path, typed: list[str], flag: str
) -> None:
    argv = _argv(adapter, binary, g.qwen3_embedding(tmp_path / "m.gguf"), extraArgs=typed)
    assert argv[-len(typed) :] == typed, "extraArgs stay last, as written"
    assert flag not in argv[: -len(typed)], f"{flag} appears once, as the operator typed it"


@pytest.mark.parametrize(
    ("variable", "flag"),
    [
        ("LLAMA_ARG_CTX_SIZE", "--ctx-size"),
        ("LLAMA_ARG_N_PARALLEL", "--parallel"),
        ("LLAMA_ARG_BATCH", "--batch-size"),
        ("LLAMA_ARG_UBATCH", "--ubatch-size"),
    ],
)
def test_an_engine_variable_in_the_runtimes_env_is_not_beaten_by_a_default(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path, variable: str, flag: str
) -> None:
    """llama.cpp lets the command line beat the environment, so a default put on the
    command line would have quietly overridden the variable."""
    argv = _argv(adapter, binary, g.qwen3_embedding(tmp_path / "m.gguf"), env={variable: "123"})
    assert flag not in argv


def test_the_defaults_are_the_documented_numbers() -> None:
    """The contract's `RuntimeMode` text says them."""
    assert {key: value[0] for key, value in run_mode.SEARCH_DEFAULTS.items()} == {
        "contextSize": 4096,
        "parallelSlots": 4,
        "batchSize": 2048,
        "ubatchSize": 2048,
    }


def test_the_defaults_leave_the_declared_flags_alone() -> None:
    declared = {"gpuLayers": 0}
    merged = run_mode.with_search_defaults(declared, None, None)
    assert declared == {"gpuLayers": 0}, "the declaration is not edited"
    assert merged["gpuLayers"] == 0 and merged["contextSize"] == 4096


# --- the same words, for a person who typed the flag themselves ---------------------


def test_a_search_model_is_logged_with_its_reason_and_an_ignored_idle_timeout_is_said(
    adapter: LlamaCppAdapter,
    binary: DiscoveredBinary,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="eugene_plexus_agent.engines.llama_cpp"):
        _argv(adapter, binary, g.qwen3_embedding(tmp_path / "m.gguf"), idleUnloadSeconds=300)
    assert "starts as an embedding model" in caplog.text and "pooling type 3" in caplog.text
    assert "idleUnloadSeconds and startOnDemand are not acted on" in caplog.text


def test_a_chat_model_logs_nothing_about_modes(
    adapter: LlamaCppAdapter,
    binary: DiscoveredBinary,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    with caplog.at_level(logging.DEBUG, logger="eugene_plexus_agent.engines.llama_cpp"):
        _argv(adapter, binary, g.qwen3_chat(tmp_path / "m.gguf"), idleUnloadSeconds=300)
    assert "search model" not in caplog.text and "starts as" not in caplog.text


# --- what a running engine was started as, read off its command line ---------------


@pytest.mark.parametrize(
    ("argv", "env", "expected"),
    [
        (["llama-server", "-m", "m.gguf"], None, (False, False)),
        (["llama-server", "--embeddings"], None, (True, False)),
        (["llama-server", "--embedding"], None, (True, False)),
        (["llama-server", "--reranking"], None, (False, True)),
        (["llama-server", "--rerank"], None, (False, True)),
        # A reranker is not also an embedder, though llama-server turns embedding on for it.
        (["llama-server", "--embeddings", "--reranking"], None, (False, True)),
        (["llama-server"], {"LLAMA_ARG_EMBEDDINGS": "1"}, (True, False)),
        (["llama-server"], {"LLAMA_ARG_EMBEDDINGS": "on"}, (True, False)),
        (["llama-server"], {"LLAMA_ARG_EMBEDDINGS": "0"}, (False, False)),
        (["llama-server"], {"LLAMA_ARG_RERANKING": "true"}, (False, True)),
        # The command line beats the environment, and there is no flag to turn one off.
        (["llama-server", "--embeddings"], {"LLAMA_ARG_EMBEDDINGS": "0"}, (True, False)),
    ],
)
def test_started_as_reads_the_command_line_and_the_environment(
    argv: list[str], env: dict[str, str] | None, expected: tuple[bool, bool]
) -> None:
    assert run_mode.started_as(argv, env) == expected
    assert LlamaCppAdapter().search_flags(argv, env) == expected


def test_what_was_started_is_what_the_adapter_built(
    adapter: LlamaCppAdapter, binary: DiscoveredBinary, tmp_path: Path
) -> None:
    """The capability is read off the argv, so it has to agree with the argv built."""
    for write, expected in (
        (g.qwen3_chat, (False, False)),
        (g.qwen3_embedding, (True, False)),
        (g.nomic_embedding, (True, False)),
        (g.qwen3_reranker, (False, True)),
    ):
        argv = _argv(adapter, binary, write(tmp_path / "m.gguf"))
        assert adapter.search_flags(argv, None) == expected


def test_only_llama_cpp_starts_a_search_model(adapter: LlamaCppAdapter) -> None:
    from eugene_plexus_agent.engines import adapter_for

    for mode in RuntimeMode:
        assert adapter.accepts_mode(mode)
    for kind in (EngineKind.vllm, EngineKind.mlx, EngineKind.strata):
        other = adapter_for(kind)
        assert other is not None
        assert other.accepts_mode(RuntimeMode.auto) and other.accepts_mode(RuntimeMode.chat)
        assert not other.accepts_mode(RuntimeMode.embedding), kind
        assert not other.accepts_mode(RuntimeMode.reranking), kind
        assert other.search_flags(["x"], None) is None
