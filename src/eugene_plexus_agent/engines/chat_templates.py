"""Row 68: select at launch, identify only after engine confirmation."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .._generated.models import EngineKind, RuntimeChatTemplate, RuntimeSpec, Source
from ..gguf_context import chat_template_facts

KEY = "chatTemplateFile"
TEMPLATE = Path(__file__).resolve().parents[1] / "templates/deepseek-v4.jinja"
_FILE = "--chat-template-file"
_INLINE = "--chat-template"
_FILE_ENV = "LLAMA_ARG_CHAT_TEMPLATE_FILE"
_INLINE_ENV = "LLAMA_ARG_CHAT_TEMPLATE"


@dataclass(frozen=True)
class CapturedTemplate:
    text: str
    source: str
    name: str
    path: str | None = None
    id: str | None = None


def _choice(argv: Sequence[str], env: Mapping[str, str]) -> tuple[str, str] | None:
    selected = None
    for key, kind in ((_INLINE_ENV, "inline"), (_FILE_ENV, "file")):
        if env.get(key):
            selected = (kind, env[key])
    tokens = iter(argv)
    for token in tokens:
        flag, equal, value = token.partition("=")
        if flag in (_FILE, _INLINE):
            selected = ("file" if flag == _FILE else "inline", value if equal else next(tokens, ""))
    return selected


def _operator_choice(spec: RuntimeSpec, env: Mapping[str, str]) -> tuple[str, str] | None:
    tokens = []
    override = (spec.flags or {}).get(KEY)
    if override is not None:
        tokens += [_FILE, str(override)]
    tokens += spec.extraArgs or []
    return _choice(tokens, env)


def llama_argv(spec: RuntimeSpec) -> list[str]:
    """Only the reviewed variant with no embedded or explicit template."""
    env = {**os.environ, **(spec.env or {})}
    if _operator_choice(spec, env) is not None:
        return []
    facts = chat_template_facts(spec.modelPath)
    if (
        facts is None
        or facts.architecture != "deepseek4"
        or facts.embedded
        or not re.search(r"deepseek[ _-]*v4(?:[ _-]|$)", facts.name or "", re.IGNORECASE)
        or "0731" in (facts.name or "")
    ):
        return []
    return ["--jinja", _FILE, str(TEMPLATE)]


def capture(
    spec: RuntimeSpec, argv: Sequence[str], env: Mapping[str, str], *, cwd: str | None = None
) -> CapturedTemplate | None:
    """Read selected bytes now. A later file/profile edit cannot change their identity."""
    if spec.engine is not EngineKind.llama_cpp:
        return None
    explicit = _operator_choice(spec, env)
    selected = _choice(argv, env)
    if selected is not None:
        kind, value = selected
        if kind == "file":
            path = Path(value)
            if not path.is_absolute() and cwd is not None:
                path = Path(cwd) / path
            try:
                text = path.read_bytes().decode("utf-8")
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot read chat template file {value!r}: {exc}") from exc
            eugene = explicit is None and path == TEMPLATE
            return CapturedTemplate(
                text,
                "eugene" if eugene else "override",
                "DeepSeek V4" if eugene else path.name,
                str(path),
                "deepseek-v4" if eugene else None,
            )
        # Built-in template names are resolved by the engine. Their contents
        # are unknown here, so confirmation will correctly say 'engine'.
        if "{{" in value or "{%" in value:
            return CapturedTemplate(value, "override", "Inline chat template")
        return None
    facts = chat_template_facts(spec.modelPath)
    if facts is not None and facts.embedded:
        return CapturedTemplate(facts.embedded, "embedded", "Model's embedded chat template")
    return None


def confirmed(text: str | None, captured: CapturedTemplate | None) -> RuntimeChatTemplate | None:
    if not text:
        return None
    matches = captured is not None and text == captured.text
    return RuntimeChatTemplate(
        source=Source(captured.source) if matches and captured is not None else Source.engine,
        name=captured.name if matches and captured is not None else "Engine chat template",
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        path=captured.path if matches and captured is not None else None,
        id=captured.id if matches and captured is not None else None,
    )
