"""Every chat model starts with exactly the command line it had before MR3.

MR3 teaches the llama.cpp adapter to start an embedding or reranking model the way
llama-server has to be told to. The one thing it must not do is change a model that
is neither, and this is the pin: `TODAYS_ARGV` is what the adapter at agent 0eb1d8f
(the commit before MR3) built for each of these representative chat profiles,
captured by running it, not written by hand.

Each profile is built three ways, because a chat model reaches the adapter three
ways: with no `mode` (every declaration that exists), with `auto`, and with `chat`;
and over two files: one that is not there (what the older tests and a remote
Library folder look like) and a real chat GGUF header the new code reads and
resolves, which is the path that has to end in "add nothing".

A failure here means a chat model would start differently. That is never the right
thing to fix by editing the pin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_agent._generated.models import EngineKind, Origin, RuntimeSpec
from eugene_plexus_agent.engines import LlamaCppAdapter
from eugene_plexus_agent.engines.base import DiscoveredBinary

from . import gguf_files as g

MODEL_NAME = "Qwen3-30B-A3B-Q4_K_M.gguf"

#: The declarations, as a console or a profile makes them.
PROFILES: dict[str, dict[str, Any]] = {
    "bare": {},
    "alias host and port": {"modelAlias": "qwen", "host": "0.0.0.0", "port": 9999},
    "the settings edited most": {
        "flags": {
            "contextSize": 8192,
            "gpuLayers": 99,
            "parallelSlots": 4,
            "tensorSplit": "0.6,0.4",
        }
    },
    "every curated flag": {
        "flags": {
            "contextSize": 16384,
            "gpuLayers": 99,
            "batchSize": 2048,
            "ubatchSize": 512,
            "threads": 8,
            "parallelSlots": 2,
            "mainGpu": 1,
            "splitMode": "row",
            "devices": "CUDA0,CUDA1",
            "tensorSplit": "0.5,0.5",
            "flashAttention": "on",
            "continuousBatching": True,
            "slotPinning": True,
            "memoryMargin": 2048,
            "projectorPath": "/models/mmproj-F16.gguf",
            "projectorOnCpu": True,
            "cacheType": "q8_0",
        }
    },
    "a quantised cache": {"flags": {"cacheType": "q4_0", "contextSize": 4096}},
    "load mode": {"flags": {"noMmap": True, "mlock": True}},
    "extra arguments come last": {
        "flags": {"contextSize": 8192},
        "extraArgs": ["--lora", "adapter.gguf", "--ctx-size", "4096", "--verbose"],
    },
    "settings that are not on the command line": {
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "profile": {"id": "p1", "name": "fast"},
        "autoStart": False,
        "startOnDemand": True,
        "idleUnloadSeconds": 300,
    },
    "the old embedding workaround in extra arguments": {
        "extraArgs": ["--embeddings", "--pooling", "mean"]
    },
}

#: What the adapter built for each, minus the executable. `{model}` is the model's path.
TODAYS_ARGV: dict[str, list[str]] = {
    "bare": [
        "--model",
        "{model}",
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--alias",
        "Qwen3-30B-A3B-Q4_K_M",
    ],
    "alias host and port": [
        "--model",
        "{model}",
        "--host",
        "0.0.0.0",
        "--port",
        "8090",
        "--alias",
        "qwen",
    ],
    "the settings edited most": [
        "--model",
        "{model}",
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--alias",
        "Qwen3-30B-A3B-Q4_K_M",
        "--ctx-size",
        "8192",
        "--n-gpu-layers",
        "99",
        "--parallel",
        "4",
        "--tensor-split",
        "0.6,0.4",
    ],
    "every curated flag": [
        "--model",
        "{model}",
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--alias",
        "Qwen3-30B-A3B-Q4_K_M",
        "--mmproj",
        "/models/mmproj-F16.gguf",
        "--no-mmproj-offload",
        "--ctx-size",
        "16384",
        "--n-gpu-layers",
        "99",
        "--parallel",
        "2",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--threads",
        "8",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--flash-attn",
        "on",
        "--fit-target",
        "2048",
        "--cont-batching",
        "--no-cache-idle-slots",
        "--main-gpu",
        "1",
        "--device",
        "CUDA0,CUDA1",
        "--split-mode",
        "row",
        "--tensor-split",
        "0.5,0.5",
    ],
    "a quantised cache": [
        "--model",
        "{model}",
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--alias",
        "Qwen3-30B-A3B-Q4_K_M",
        "--ctx-size",
        "4096",
        "--cache-type-k",
        "q4_0",
        "--cache-type-v",
        "q4_0",
        "--flash-attn",
        "on",
    ],
    "load mode": [
        "--model",
        "{model}",
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--alias",
        "Qwen3-30B-A3B-Q4_K_M",
        "--load-mode",
        "mlock",
    ],
    "extra arguments come last": [
        "--model",
        "{model}",
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--alias",
        "Qwen3-30B-A3B-Q4_K_M",
        "--ctx-size",
        "8192",
        "--lora",
        "adapter.gguf",
        "--ctx-size",
        "4096",
        "--verbose",
    ],
    "settings that are not on the command line": [
        "--model",
        "{model}",
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--alias",
        "Qwen3-30B-A3B-Q4_K_M",
    ],
    "the old embedding workaround in extra arguments": [
        "--model",
        "{model}",
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
        "--alias",
        "Qwen3-30B-A3B-Q4_K_M",
        "--embeddings",
        "--pooling",
        "mean",
    ],
}


@pytest.fixture
def binary(tmp_path: Path) -> DiscoveredBinary:
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    return DiscoveredBinary(path=exe, origin=Origin.path, version="4589")


def _model(tmp_path: Path, kind: str) -> str:
    if kind == "a path that is not there":
        return str(tmp_path / "absent" / MODEL_NAME)
    return str(
        g.write_gguf(
            tmp_path / MODEL_NAME, architecture="qwen3", context_length=40960, vocabulary=200
        )
    )


@pytest.mark.parametrize("kind", ["a path that is not there", "a chat model's header"])
@pytest.mark.parametrize("mode", [None, "auto", "chat"])
@pytest.mark.parametrize("label", sorted(PROFILES))
def test_a_chat_model_starts_with_exactly_the_command_line_it_always_had(
    binary: DiscoveredBinary, tmp_path: Path, label: str, mode: str | None, kind: str
) -> None:
    model = _model(tmp_path, kind)
    body: dict[str, Any] = {
        "name": "chat-model",
        "engine": EngineKind.llama_cpp,
        "modelPath": model,
        **PROFILES[label],
    }
    if mode is not None:
        body["mode"] = mode
    argv = LlamaCppAdapter().build_argv(RuntimeSpec.model_validate(body), binary, port=8090)

    expected = [token.replace("{model}", model) for token in TODAYS_ARGV[label]]
    assert argv == [str(binary.path), *expected]


def test_the_pin_covers_what_a_profile_can_hold() -> None:
    """The profiles cover each way a command line is built, so the pin cannot go
    stale on a path nobody wrote a case for."""
    flags = {key for body in PROFILES.values() for key in body.get("flags", {})}
    assert {
        "contextSize",
        "gpuLayers",
        "parallelSlots",
        "tensorSplit",
        "batchSize",
        "ubatchSize",
        "threads",
        "mainGpu",
        "splitMode",
        "devices",
        "flashAttention",
        "continuousBatching",
        "slotPinning",
        "memoryMargin",
        "projectorPath",
        "projectorOnCpu",
        "cacheType",
        "noMmap",
        "mlock",
    } <= flags
    assert any(body.get("extraArgs") for body in PROFILES.values())
    assert any(body.get("env") for body in PROFILES.values())


def test_no_pinned_chat_command_line_carries_a_search_flag_of_the_adapters_own() -> None:
    """Except the one profile that typed the old workaround into extra arguments, which
    is the operator's and stays the operator's: it is the last thing on its line."""
    search = {"--embeddings", "--embedding", "--reranking", "--rerank", "--pooling"}
    for label, argv in TODAYS_ARGV.items():
        if "workaround" in label:
            assert argv[-3:] == ["--embeddings", "--pooling", "mean"]
            continue
        assert not search & set(argv), label
