"""Row 68 uses fake GGUF headers and engine replies; no engine is started."""

import hashlib
import json
import logging
import struct
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from jinja2 import Environment

from eugene_plexus_agent._generated.models import Origin, RuntimeSpec
from eugene_plexus_agent.engines import LlamaCppAdapter, llama_cpp
from eugene_plexus_agent.engines.base import DiscoveredBinary, Ready
from eugene_plexus_agent.gguf_context import drafting_facts
from eugene_plexus_agent.profile_builds import BuildPlan, drafting_ways
from eugene_plexus_agent.runtimes import RuntimeSupervisor, _RuntimePlanner
from eugene_plexus_agent.supervisor import ProcessState, SpawnPlanError

from .test_speculation import HELP_NOW, _blocks, _gguf, _plan


def gguf_file(path: Path, *, embedded: str | None = None, name: str = "DeepSeek-V4-Flash") -> Path:
    def string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    items = [("general.architecture", "deepseek4"), ("general.name", name)]
    if embedded is not None:
        items.append(("tokenizer.chat_template", embedded))
    path.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, len(items))
        + b"".join(string(key) + struct.pack("<I", 8) + string(value) for key, value in items)
    )
    return path


@pytest.fixture
def launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    adapter = LlamaCppAdapter()
    binary = DiscoveredBinary(tmp_path / "fake-engine", Origin.path, "b11719")
    monkeypatch.setattr(adapter, "resolve_binary", lambda *a, **kw: binary)
    monkeypatch.setattr(llama_cpp, "_help_text", lambda _: HELP_NOW)

    def make(
        *, embedded=None, flags=None, extra=None, env=None, name="DeepSeek-V4-Flash", cwd=None
    ):
        spec = RuntimeSpec(
            name="flash",
            engine="llama_cpp",
            modelPath=str(gguf_file(tmp_path / "model.gguf", embedded=embedded, name=name)),
            port=18090,
            flags=flags,
            extraArgs=extra,
            env=env,
            workingDirectory=cwd,
        )
        planner = _RuntimePlanner(
            spec,
            adapter,
            logging.getLogger("test"),
            lambda key: key == "allowUnrestrictedEngineLaunch",
        )
        return planner, planner.plan()

    return make


def test_missing_template_gets_the_reviewed_file_and_row9_cache_settings(launch) -> None:
    planner, plan = launch(flags={"cacheRam": 2, "ctxCheckpoints": 8})
    path = Path(plan.argv[plan.argv.index("--chat-template-file") + 1])
    assert path.is_file() and path.name == "deepseek-v4.jinja"
    assert "--jinja" in plan.argv
    assert plan.argv[plan.argv.index("--cache-ram") + 1] == "2048"
    assert plan.argv[plan.argv.index("--ctx-checkpoints") + 1] == "8"
    assert planner.chat_template is not None and planner.chat_template.source == "eugene"


def test_embedded_template_wins_and_is_captured_before_file_changes(launch) -> None:
    planner, plan = launch(embedded="embedded {{ messages }}")
    assert "--chat-template-file" not in plan.argv
    gguf_file(Path(planner.spec.modelPath), embedded="changed")
    assert planner.chat_template is not None
    assert planner.chat_template.text == "embedded {{ messages }}"
    assert planner.chat_template.source == "embedded"


@pytest.mark.parametrize("spelling", ["schema", "extra", "equals", "environment"])
def test_an_operator_file_wins_over_embedded_and_fallback(
    launch, tmp_path: Path, spelling: str
) -> None:
    custom = tmp_path / "custom template.jinja"
    custom.write_bytes(b"override\r\n{{ messages }}\n")
    options = {
        "schema": {"flags": {"chatTemplateFile": str(custom)}},
        "extra": {"extra": ["--chat-template-file", str(custom)]},
        "equals": {"extra": [f"--chat-template-file={custom}"]},
        "environment": {"env": {"LLAMA_ARG_CHAT_TEMPLATE_FILE": str(custom)}},
    }
    planner, _ = launch(embedded="embedded", **options[spelling])
    custom.write_text("edited later", encoding="utf-8")
    assert planner.chat_template is not None
    assert planner.chat_template.source == "override"
    assert planner.chat_template.text == "override\r\n{{ messages }}\n"
    assert planner.chat_template.path == str(custom)


def test_override_has_gui_equality_and_bad_path_names_the_cause(launch, tmp_path: Path) -> None:
    fields = {f.key: f for f in LlamaCppAdapter().flag_schema().fields}
    assert fields["chatTemplateFile"].valueType == "string"
    with pytest.raises(SpawnPlanError, match=r"missing\.jinja"):
        launch(flags={"chatTemplateFile": str(tmp_path / "missing.jinja")})


def test_relative_override_is_captured_from_the_engines_working_directory(
    launch, tmp_path: Path
) -> None:
    directory = tmp_path / "engine folder"
    directory.mkdir()
    (directory / "custom.jinja").write_text("custom {{ messages }}", encoding="utf-8")
    planner, plan = launch(flags={"chatTemplateFile": "custom.jinja"}, cwd=str(directory))
    assert plan.cwd == str(directory)
    assert planner.chat_template.text == "custom {{ messages }}"
    assert planner.chat_template.path == str(directory / "custom.jinja")


@pytest.mark.parametrize("name", ["Unknown", "DeepSeek-V4.1-Flash", "DeepSeek-V4-Flash-0731"])
def test_unreviewed_variants_keep_the_engine_default_and_are_not_blocked(launch, name: str) -> None:
    planner, plan = launch(name=name)
    assert "--chat-template-file" not in plan.argv and planner.chat_template is None


class FakeProcess:
    state = ProcessState.starting
    pid = 4242
    last_error = None
    last_restart = datetime.now(UTC)
    supervising = True
    stopping = False
    spawns = 1

    def __init__(self, argv):
        self.last_argv = argv

    def observe_readiness(self, ready):
        pass

    async def stop(self):
        self.pid = None

    async def restart(self):
        self.last_restart = datetime.now(UTC)


async def test_effective_template_requires_readback_and_is_cleared_before_restart_and_stop(
    launch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner, plan = launch()
    supervisor = RuntimeSupervisor()
    supervisor._planners["flash"] = planner
    supervisor._processes["flash"] = FakeProcess(plan.argv)
    assert supervisor.compose(planner.spec).chatTemplate is None
    template = planner.chat_template.text

    async def probe(*args, **kwargs):
        return Ready(chat_template=template)

    monkeypatch.setattr("eugene_plexus_agent.runtimes.adapter_for", lambda _: planner._adapter)
    monkeypatch.setattr(planner._adapter, "probe_readiness", probe)
    await supervisor._probe_one(planner.spec)
    observed = supervisor.compose(planner.spec).chatTemplate
    assert observed is not None and observed.source == "eugene"
    assert observed.id == "deepseek-v4"
    assert observed.sha256 == hashlib.sha256(template.encode()).hexdigest()
    # A changed saved setting does not rewrite what this process is using.
    planner.spec.flags = {"chatTemplateFile": "some-new-file"}
    assert supervisor.compose(planner.spec).chatTemplate == observed
    await supervisor.restart("flash")
    assert supervisor.compose(planner.spec).chatTemplate is None
    await supervisor._probe_one(planner.spec)
    await supervisor.stop_one("flash")
    assert supervisor.compose(planner.spec).chatTemplate is None


async def test_a_new_automatic_start_cannot_reuse_the_previous_process_confirmation(
    launch, monkeypatch
):
    planner, plan = launch()
    supervisor = RuntimeSupervisor()
    process = FakeProcess(plan.argv)
    supervisor._planners["flash"] = planner
    supervisor._processes["flash"] = process

    async def probe(*args, **kwargs):
        return Ready(chat_template=planner.chat_template.text)

    monkeypatch.setattr("eugene_plexus_agent.runtimes.adapter_for", lambda _: planner._adapter)
    monkeypatch.setattr(planner._adapter, "probe_readiness", probe)
    await supervisor._probe_one(planner.spec)
    assert supervisor.compose(planner.spec).chatTemplate is not None
    process.spawns += 1
    planner.plan()
    assert supervisor.compose(planner.spec).chatTemplate is None


async def test_different_engine_text_reports_engine_source_with_no_selected_path_or_id(
    launch, monkeypatch
):
    planner, plan = launch()
    supervisor = RuntimeSupervisor()
    supervisor._planners["flash"] = planner
    supervisor._processes["flash"] = FakeProcess(plan.argv)

    async def probe(*args, **kwargs):
        return Ready(chat_template="actually used\r\n")

    monkeypatch.setattr("eugene_plexus_agent.runtimes.adapter_for", lambda _: planner._adapter)
    monkeypatch.setattr(planner._adapter, "probe_readiness", probe)
    await supervisor._probe_one(planner.spec)
    observed = supervisor.compose(planner.spec).chatTemplate
    assert observed.source == "engine" and observed.id is None and observed.path is None
    assert observed.sha256 == hashlib.sha256(b"actually used\r\n").hexdigest()


async def test_llama_readback_keeps_exact_text_and_survives_missing_props(monkeypatch) -> None:
    from eugene_plexus_agent import _http

    text = "engine template\r\n{{ messages }}\n"
    props = {"chat_template": text}

    def reply(request):
        return httpx.Response(
            200, json={"status": "ok"} if request.url.path == "/health" else props
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        monkeypatch.setattr(llama_cpp, "_probe_client", lambda: client)
        result = await LlamaCppAdapter().probe_readiness("http://fake")
        assert isinstance(result, Ready) and result.chat_template == text
        props.clear()
        result = await LlamaCppAdapter().probe_readiness("http://fake")
        assert isinstance(result, Ready) and result.chat_template is None
    await _http.aclose_shared()


def test_the_shipped_template_renders_published_dsml_samples_and_tool_results() -> None:
    template = Path(__file__).parents[1] / "src/eugene_plexus_agent/templates/deepseek-v4.jinja"
    env = Environment()
    env.filters["from_json"] = json.loads
    # llama.cpp's minja serializes JSON, without HTML's Markup escaping.
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    source = env.from_string(template.read_text(encoding="utf-8"))
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/deepseek-v4.json").read_text(encoding="utf-8")
    )
    for sample in fixture["samples"]:
        rendered = source.render(
            messages=[{"role": "assistant", "tool_calls": sample["tool_calls"]}], bos_token=""
        )
        assert sample["dsml"] in rendered
    rendered = source.render(
        messages=[{"role": "user", "content": "weather?"}, {"role": "tool", "content": "sunny"}],
        bos_token="BOS",
        add_generation_prompt=True,
    )
    assert (
        rendered
        == "BOS<\uff5cUser\uff5c>weather?\n\n<tool_result>sunny</tool_result><\uff5cAssistant\uff5c></think>"
    )


@pytest.mark.parametrize("head", [True, False])
def test_v4_uses_sd1_own_head_only_when_kept(tmp_path: Path, head: bool) -> None:
    model = _gguf(
        tmp_path / "flash.gguf",
        architecture="deepseek4",
        blocks=44,
        tensors=_blocks(0, 43, head=head),
    )
    ways = drafting_ways(
        BuildPlan(
            tools=None,
            model=model,
            work=tmp_path,
            text=tmp_path / "text",
            flags={},
            contexts=[8192],
            margin=None,
            env={},
        )
    )
    assert ("mtpHead" in [way.value for way, _, _ in ways]) is head
    if head:
        plan = _plan({"draftWith": "mtpHead", "draftTokens": 5, "reuseRepeatedText": False}, model)
        assert plan.argv == ("--spec-type", "draft-mtp", "--spec-draft-n-max", "5")
    else:
        with pytest.raises(ValueError, match="carries no MTP head"):
            _plan({"draftWith": "mtpHead"}, model)


def test_a_split_v4_head_is_found_and_later_shard_changes_invalidate_it(tmp_path: Path) -> None:
    first = _gguf(
        tmp_path / "flash-00001-of-00002.gguf",
        architecture="deepseek4",
        blocks=44,
        tensors=["blk.0.attn_norm.weight"],
    )
    last = _gguf(
        tmp_path / "flash-00002-of-00002.gguf",
        architecture="deepseek4",
        blocks=44,
        tensors=["blk.43.nextn.eh_proj.weight"],
    )
    assert drafting_facts(first).mtp_head is True
    assert "draft-mtp" in _plan({"draftWith": "mtpHead"}, first).types
    _gguf(last, architecture="deepseek4", blocks=44, tensors=["blk.43.attn_norm.weight"])
    assert drafting_facts(first).mtp_head is False
    last.unlink()
    assert drafting_facts(first) is None
