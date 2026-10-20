"""CX9: a model from an Ollama store is served under Ollama's name for it.

Its file is a hash-named blob (`sha256-<hex>`), so the file's own name, which
is what an alias defaults to, would be the model id every client asked for.
The agent also has to take the new `kind` and `found` the Library now reports
for a folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_agent._generated.models import EngineKind, Origin, RuntimeSpec
from eugene_plexus_agent.engines import LlamaCppAdapter
from eugene_plexus_agent.engines.base import DiscoveredBinary, default_model_alias
from eugene_plexus_agent.library_folders import (
    FolderRecord,
    LibraryFolderCache,
    parse_folders,
    refresh,
)
from eugene_plexus_agent.run_worker import runtime_spec

BLOB = "/home/me/.ollama/models/blobs/sha256-" + "ab" * 32

PROFILE = {"id": "p1", "name": "Default", "engine": "llama_cpp", "default": True}


def _model(**extra: Any) -> dict[str, Any]:
    return {"name": "qwen3:8b", "path": BLOB, **extra}


def test_an_ollama_model_is_served_under_its_ollama_name() -> None:
    spec = runtime_spec(_model(ollama={"tags": ["qwen3:8b", "qwen3:latest"]}), PROFILE, start=True)
    assert spec.modelAlias == "qwen3:8b"
    # The runtime's own name is still a plain slug.
    assert spec.name == "qwen3-8b"
    assert default_model_alias(BLOB).startswith("sha256-")


def test_a_model_that_is_not_from_ollama_keeps_the_alias_the_file_gives_it() -> None:
    spec = runtime_spec(
        _model(name="Qwen3-8B", path="/models/Qwen3-8B-Q4_K_M.gguf"), PROFILE, start=True
    )
    assert spec.modelAlias is None


def test_the_engine_is_given_the_colon_name_as_one_argument(tmp_path: Path) -> None:
    """No quoting or sanitising on the way: argv is a list, and `--alias` takes any string."""
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    binary = DiscoveredBinary(path=exe, origin=Origin.path, version="4589")
    spec = runtime_spec(_model(ollama={"tags": ["qwen3:8b"]}), PROFILE, start=True)
    assert isinstance(spec, RuntimeSpec) and spec.engine == EngineKind.llama_cpp
    argv = LlamaCppAdapter().build_argv(spec, binary, port=8090)
    assert argv[argv.index("--alias") + 1] == "qwen3:8b"


class _Library:
    base_url = "http://library.test"

    async def folders(self) -> list[dict[str, Any]]:
        # As `GET /v1/folders` answers it: a folder, and Ollama's own store found on the machine.
        return [
            {"path": "/models", "mounts": ["/mnt/models"], "kind": "folder"},
            {"path": "/home/me/.ollama/models", "mounts": [], "kind": "ollama", "found": True},
        ]


def test_the_new_folder_fields_are_taken_without_error() -> None:
    folders = parse_folders(
        [
            {"path": "/models", "mounts": ["/mnt/models"], "kind": "folder"},
            {"path": "/o", "mounts": [], "kind": "ollama", "found": True},
        ]
    )
    assert folders == [FolderRecord("/models", ("/mnt/models",)), FolderRecord("/o")]


@pytest.mark.asyncio
async def test_a_node_holds_the_store_as_a_folder_so_its_models_are_under_one(
    tmp_path: Path,
) -> None:
    cache = LibraryFolderCache(tmp_path / "f.json", windows=False)
    assert await refresh(cache, _Library())
    assert cache.folder_for(BLOB) == FolderRecord("/home/me/.ollama/models")
