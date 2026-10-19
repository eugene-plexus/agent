"""`agent.yaml` is read as it is: nothing converts what v0.1 wrote any more.

A file without `stateVersion` loads (a v0.2 file that was never rewritten may lack
it), and the stamp arrives with the next ordinary write. What v0.1 saved and v0.3.0
converted is not rewritten behind anyone's back: a flash attention checkbox in a saved
runtime loads untouched and is refused when the runtime starts, naming v0.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest
from eugene_plexus_agent.engines.llama_cpp import flash_attention_argv
from eugene_plexus_agent.state import STATE_VERSION, AgentState


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8", newline="\n")
    return text


def test_an_unstamped_file_loads_and_is_not_rewritten_by_the_load(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    before = _write(path, "firstRunComplete: true\nuiTheme: dark\ncomponents: []\nruntimes: []\n")

    state = AgentState(path)
    state.load()

    assert state.list_runtime_specs() == []
    assert path.read_text(encoding="utf-8") == before


def test_the_stamp_arrives_with_the_next_write(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    _write(path, "firstRunComplete: true\ncomponents: []\nruntimes: []\n")
    state = AgentState(path)
    state.load()

    state.apply_config_patch(ConfigUpdateRequest.model_validate({"uiTheme": "dark"}))

    assert yaml.safe_load(path.read_text(encoding="utf-8"))["stateVersion"] == STATE_VERSION


def test_a_checkbox_in_a_saved_runtime_loads_untouched_and_is_refused_at_start(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent.yaml"
    before = _write(
        path,
        "components: []\n"
        "runtimes:\n"
        "  - name: qwen\n"
        "    engine: llama_cpp\n"
        "    modelPath: /models/q.gguf\n"
        "    flags:\n"
        "      flashAttention: true\n",
    )

    state = AgentState(path)
    state.load()

    spec = state.get_runtime_spec("qwen")
    assert spec is not None and spec.flags == {"flashAttention": True}
    assert path.read_text(encoding="utf-8") == before, "no conversion, no backup"
    assert not list(tmp_path.glob("agent.yaml.*"))
    with pytest.raises(ValueError) as refused:
        flash_attention_argv(dict(spec.flags or {}))
    assert str(refused.value) == (
        "A saved runtime holds flashAttention as a checkbox (True), which v0.1 wrote and "
        "v0.3.0 converted. This version no longer converts it: start Eugene v0.3.0 once to "
        "convert it, then update."
    )
