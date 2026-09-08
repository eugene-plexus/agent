"""Engine adapters: the registry of engines this watchdog can start.

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
)
from .llama_cpp import LlamaCppAdapter, default_model_alias

ADAPTERS: dict[EngineKind, EngineAdapter] = {
    EngineKind.llama_cpp: LlamaCppAdapter(),
}


def adapter_for(kind: EngineKind) -> EngineAdapter | None:
    """The adapter for one engine kind, or None if unsupported here.

    None is possible even though `EngineKind` is closed: a topology
    written by a newer watchdog can name a kind this build doesn't
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
    "NotAnswering",
    "Readiness",
    "Ready",
    "adapter_for",
    "default_model_alias",
]
