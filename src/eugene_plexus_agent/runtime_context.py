"""Services need application state and per-operation scratch space, not HTTP."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from starlette.datastructures import State


class RuntimeContext(Protocol):
    @property
    def app(self) -> Any: ...

    @property
    def state(self) -> State: ...


@dataclass
class NodeContext:
    app: Any
    state: State = field(default_factory=State)
