"""DI7: profile launch settings and the running view agree, including overrides."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from eugene_plexus_agent._generated.models import EngineKind, Origin, RuntimeSpec
from eugene_plexus_agent.engines import LlamaCppAdapter, VllmAdapter
from eugene_plexus_agent.engines.base import DiscoveredBinary
from eugene_plexus_agent.runtimes import RuntimeSupervisor, _RuntimePlanner
from eugene_plexus_agent.supervisor import ProcessState


@pytest.mark.parametrize("adapter", [LlamaCppAdapter(), VllmAdapter()])
def test_same_answer_is_a_restartable_gui_launch_field(adapter):
    field = next(f for f in adapter.flag_schema().fields if f.key == "sameAnswerEveryTime")
    assert field.label == "Same answer every time (slower)"
    assert field.valueType.value == "boolean"
    assert field.default is False
    assert field.requiresRestart is True
    assert "seed" in field.description
    if adapter.kind is EngineKind.llama_cpp:
        assert "one request at a time; others wait" in field.description


def test_same_answer_llama_uses_one_slot_and_keeps_the_expert_override():
    adapter = LlamaCppAdapter()
    binary = DiscoveredBinary(path=Path("missing-llama-server"), origin=Origin.configured)
    spec = RuntimeSpec(
        name="test",
        engine="llama_cpp",
        modelPath="model.gguf",
        flags={"sameAnswerEveryTime": True, "parallelSlots": 4},
    )
    argv = adapter.build_argv(spec, binary, 19020)
    assert argv[argv.index("--parallel") + 1] == "1"
    assert argv.count("--parallel") == 1
    spec.extraArgs = ["--parallel=3"]
    assert adapter.build_argv(spec, binary, 19020)[-1] == "--parallel=3"
    spec.flags = {"sameAnswerEveryTime": False, "parallelSlots": 4}
    spec.extraArgs = []
    argv = adapter.build_argv(spec, binary, 19020)
    assert argv[argv.index("--parallel") + 1] == "4"


@pytest.mark.parametrize("enabled", [False, True])
def test_same_answer_vllm_sets_the_pinned_engines_environment(enabled):
    adapter = VllmAdapter()
    binary = DiscoveredBinary(path=Path("missing-vllm"), origin=Origin.configured)
    spec = RuntimeSpec(
        name="test", engine="vllm", modelPath="model", flags={"sameAnswerEveryTime": enabled}
    )
    assert adapter.default_env(spec, binary)["VLLM_BATCH_INVARIANT"] == str(int(enabled))
    assert "sameAnswerEveryTime" not in adapter.build_argv(spec, binary, 19020)


@pytest.mark.parametrize(
    "engine,extra,override,expected",
    [
        ("llama_cpp", [], {}, True),
        ("llama_cpp", ["--parallel=3"], {}, False),
        ("vllm", [], {}, True),
        ("vllm", [], {"VLLM_BATCH_INVARIANT": "0"}, False),
    ],
)
def test_same_answer_runtime_reports_the_launch_after_expert_overrides(
    tmp_path,
    monkeypatch,
    engine,
    extra,
    override,
    expected,
):
    binary = tmp_path / "engine"
    binary.write_text("fake executable")
    adapter = LlamaCppAdapter() if engine == "llama_cpp" else VllmAdapter()
    monkeypatch.setattr(adapter, "probe_version", lambda _: "test")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    spec = RuntimeSpec(
        name="test",
        engine=engine,
        modelPath="model",
        port=19020,
        binary=str(binary),
        flags={"sameAnswerEveryTime": True},
        extraArgs=extra,
        env=override,
        autoDriver=False,
    )
    planner = _RuntimePlanner(spec, adapter, logging.getLogger("test"), lambda _: True)
    plan = planner.plan()
    if engine == "vllm":
        assert plan.env["VLLM_BATCH_INVARIANT"] == ("1" if expected else "0")
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))
    supervisor._planners["test"] = planner
    supervisor._processes["test"] = SimpleNamespace(
        state=ProcessState.starting,
        pid=42,
        last_argv=plan.argv,
        last_error=None,
        last_restart=None,
    )
    # A later profile edit cannot change what this already running process uses.
    spec.flags = {"sameAnswerEveryTime": not expected}
    assert supervisor.compose(spec).flags["sameAnswerEveryTime"] is expected
