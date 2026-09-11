"""Engine adapters: the registry of engines this agent can start.

`EngineKind` is a closed enum in the contract for exactly this reason —
an engine is supported when an adapter exists here and not otherwise, so
the enum and this table are two views of the same fact.
"""

from __future__ import annotations

from .._generated.models import EngineKind
from .base import (
    DiscoveredBinary,
    EngineAdapter,
    EngineUnavailableError,
    Loading,
    NotAnswering,
    Readiness,
    Ready,
    default_model_alias,
    interpret_readiness,
)
from .llama_cpp import LlamaCppAdapter
from .mlx import MlxAdapter
from .vllm import VllmAdapter

ADAPTERS: dict[EngineKind, EngineAdapter] = {
    EngineKind.llama_cpp: LlamaCppAdapter(),
    EngineKind.vllm: VllmAdapter(),
    EngineKind.mlx: MlxAdapter(),
}


def adapter_for(kind: EngineKind) -> EngineAdapter | None:
    """The adapter for one engine kind, or None if unsupported here.

    None is possible even though `EngineKind` is closed: a topology
    written by a newer agent can name a kind this build doesn't
    implement.
    """
    return ADAPTERS.get(kind)


__all__ = [
    "ADAPTERS",
    "DiscoveredBinary",
    "EngineAdapter",
    "EngineUnavailableError",
    "LlamaCppAdapter",
    "Loading",
    "MlxAdapter",
    "NotAnswering",
    "Readiness",
    "Ready",
    "VllmAdapter",
    "adapter_for",
    "default_model_alias",
    "interpret_readiness",
]
