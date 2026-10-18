"""SD1: speculative decoding for llama.cpp, from profile flags to a command line.

The file shapes are the real ones read on 2026-10-18: unsloth's Qwen3.5-9B
MTP build carries its head in the model (33 blocks, `nextn` tensors on the
last), ggml-org's `mtp-Qwen3.6-27B-Q4_0.gguf` holds only block 64 of 65.
"""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest

from eugene_plexus_agent import model_copies
from eugene_plexus_agent._generated.models import (
    DraftTrial,
    DraftWith,
    Origin,
    RuntimeSpec,
)
from eugene_plexus_agent.engines import LlamaCppAdapter, llama_cpp, speculation
from eugene_plexus_agent.engines.base import DiscoveredBinary
from eugene_plexus_agent.gguf_context import drafting_facts
from eugene_plexus_agent.profile_builds import (
    BuildPlan,
    ProfileBuilds,
    Tools,
    drafting_ways,
    recommend_draft,
)

# b11719's own `--help` line for the flag, and the one before it existed.
HELP_NOW = (
    "--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,"
    "ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache\n--metrics  enable\n"
)
HELP_BEFORE = "--ctx-size N  size of the prompt context\n--metrics  enable\n"


def _gguf(path: Path, *, blocks: int, tensors: list[str]) -> Path:
    def string(text: str) -> bytes:
        return struct.pack("<Q", len(text.encode())) + text.encode()

    keys = [
        string("general.architecture") + struct.pack("<I", 8) + string("qwen35"),
        string("qwen35.block_count") + struct.pack("<I", 4) + struct.pack("<I", blocks),
    ]
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
    body += struct.pack("<Q", len(keys)) + b"".join(keys)
    for name in tensors:
        body += string(name) + struct.pack("<I", 1) + struct.pack("<Q", 32)
        body += struct.pack("<I", 0) + struct.pack("<Q", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def _blocks(first: int, last: int, *, head: bool) -> list[str]:
    names = [f"blk.{n}.attn_norm.weight" for n in range(first, last + 1)]
    return names + ([f"blk.{last}.nextn.eh_proj.weight"] if head else [])


def model_with_head(path: Path) -> Path:
    return _gguf(path, blocks=33, tensors=_blocks(0, 32, head=True))


def plain_model(path: Path) -> Path:
    return _gguf(path, blocks=64, tensors=_blocks(0, 63, head=False))


def sidecar(path: Path) -> Path:
    return _gguf(path, blocks=65, tensors=_blocks(64, 64, head=True))


def _plan(flags, model, *, extra=None, env=None, search=False, offered=None):
    return speculation.plan(
        flags,
        str(model),
        extra_args=extra,
        env=env,
        search=search,
        offered=speculation.offered_types(HELP_NOW) if offered is None else offered,
    )


# --- the files ----------------------------------------------------------------


def test_the_agent_reads_a_file_the_way_llama_cpp_does(tmp_path: Path) -> None:
    head = drafting_facts(model_with_head(tmp_path / "a.gguf"))
    alone = drafting_facts(sidecar(tmp_path / "mtp-b.gguf"))
    plain = drafting_facts(plain_model(tmp_path / "c.gguf"))
    assert head is not None and alone is not None and plain is not None
    assert (head.mtp_head, alone.mtp_head, plain.mtp_head) == (True, False, False)
    assert alone.companion_type == "draft-mtp"
    assert plain.companion_type == "draft-simple"
    assert drafting_facts(tmp_path / "missing.gguf") is None


# --- the plan -----------------------------------------------------------------


def test_reuse_repeated_text_is_on_unless_turned_off(tmp_path: Path) -> None:
    model = plain_model(tmp_path / "m.gguf")
    assert _plan({}, model).argv == ("--spec-type", "ngram-mod")
    assert _plan({"reuseRepeatedText": True}, model).argv == ("--spec-type", "ngram-mod")
    assert _plan({"reuseRepeatedText": False}, model).argv == ()


def test_both_together_try_the_repeat_first(tmp_path: Path) -> None:
    model = model_with_head(tmp_path / "m.gguf")
    planned = _plan({"draftWith": "mtpHead", "draftTokens": 4}, model)
    assert planned.argv == (
        "--spec-type",
        "ngram-mod,draft-mtp",
        "--spec-draft-n-max",
        "4",
    )


def test_a_head_the_model_does_not_carry_is_refused_by_name(tmp_path: Path) -> None:
    model = plain_model(tmp_path / "Qwen3.6-27B-Q4_K_M.gguf")
    with pytest.raises(ValueError, match="carries no MTP head"):
        _plan({"draftWith": "mtpHead"}, model)


def test_a_companion_drafts_as_its_own_tensors_say(tmp_path: Path) -> None:
    model = plain_model(tmp_path / "m.gguf")
    mtp = sidecar(tmp_path / "mtp-m.gguf")
    planned = _plan(
        {"draftWith": "companion", "draftModelPath": str(mtp), "reuseRepeatedText": False}, model
    )
    assert planned.argv == ("--spec-type", "draft-mtp", "--spec-draft-model", str(mtp))
    assert planned.draft_model == str(mtp)


def test_a_companion_that_cannot_draft_is_refused(tmp_path: Path) -> None:
    model = plain_model(tmp_path / "m.gguf")
    with pytest.raises(ValueError, match="no companion file is chosen"):
        _plan({"draftWith": "companion"}, model)
    with pytest.raises(ValueError, match="cannot be read"):
        _plan({"draftWith": "companion", "draftModelPath": str(tmp_path / "gone.gguf")}, model)
    nothing = _gguf(tmp_path / "mtp-strata.gguf", blocks=0, tensors=["mtp.fc_hidden.weight"])
    with pytest.raises(ValueError, match="can draft with: it has neither"):
        _plan({"draftWith": "companion", "draftModelPath": str(nothing)}, model)


def test_extra_args_win(tmp_path: Path) -> None:
    model = model_with_head(tmp_path / "m.gguf")
    flags = {"draftWith": "mtpHead"}
    assert _plan(flags, model, extra=["--spec-type", "ngram-simple"]).argv == ()
    assert _plan(flags, model, extra=["-md=other.gguf"]).argv == ()
    assert _plan(flags, model, env={"LLAMA_ARG_SPEC_TYPE": "none"}).argv == ()


def test_a_search_model_never_speculates(tmp_path: Path) -> None:
    assert _plan({}, plain_model(tmp_path / "m.gguf"), search=True).argv == ()


def test_an_older_build_is_not_handed_what_it_would_refuse(tmp_path: Path) -> None:
    model = model_with_head(tmp_path / "m.gguf")
    before = speculation.offered_types(HELP_BEFORE)
    assert before == frozenset()
    planned = _plan({}, model, offered=before)
    assert planned.argv == ()
    assert planned.note is not None and "does not offer speculative decoding" in planned.note
    # A build that could not be asked is not risked with the default...
    assert (
        speculation.plan({}, str(model), extra_args=None, env=None, search=False, offered=None)
        == speculation.Plan()
    )
    # ...but a way of drafting someone chose is passed, guessing forward.
    chosen = speculation.plan(
        {"draftWith": "mtpHead"}, str(model), extra_args=None, env=None, search=False, offered=None
    )
    assert chosen.argv == ("--spec-type", "ngram-mod,draft-mtp")


def test_a_type_the_build_lacks_is_left_out_and_said(tmp_path: Path) -> None:
    model = model_with_head(tmp_path / "m.gguf")
    planned = _plan({"draftWith": "mtpHead"}, model, offered=frozenset({"ngram-mod"}))
    assert planned.argv == ("--spec-type", "ngram-mod")
    assert planned.note is not None and "draft-mtp" in planned.note


def test_tokens_drafted_per_step_only_with_a_model_drafting(tmp_path: Path) -> None:
    assert _plan({"draftTokens": 5}, plain_model(tmp_path / "m.gguf")).argv == (
        "--spec-type",
        "ngram-mod",
    )


# --- the declaration ----------------------------------------------------------


def test_a_companion_must_sit_beside_its_model() -> None:
    flags = {"draftWith": "companion", "draftModelPath": "/models/other/mtp.gguf"}
    problem = speculation.declaration_problem("/models/qwen/m.gguf", flags)
    assert problem is not None and "the model's own folder (/models/qwen)" in problem
    assert (
        speculation.declaration_problem(
            r"D:\Models\Qwen\m.gguf",
            {"draftWith": "companion", "draftModelPath": r"d:\models\qwen\mtp-m.gguf"},
        )
        is None
    )
    assert "draftModelPath must name" in (
        speculation.declaration_problem("/m/a.gguf", {"draftWith": "companion"}) or ""
    )
    assert "mtpHead" in (speculation.declaration_problem("/m/a.gguf", {"draftWith": "on"}) or "")


# --- the command line ---------------------------------------------------------


@pytest.fixture
def binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DiscoveredBinary:
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(llama_cpp, "_help_text", lambda path: HELP_NOW)
    return DiscoveredBinary(path=exe, origin=Origin.path, version="11719")


def _spec(model: Path, **extra) -> RuntimeSpec:  # type: ignore[no-untyped-def]
    return RuntimeSpec.model_validate(
        {"name": "m", "engine": "llama_cpp", "modelPath": str(model), **extra}
    )


def test_the_launch_speculates_and_exposes_its_counters(
    tmp_path: Path, binary: DiscoveredBinary
) -> None:
    model = model_with_head(tmp_path / "m.gguf")
    argv = LlamaCppAdapter().build_argv(
        _spec(model, flags={"draftWith": "mtpHead"}, extraArgs=["--verbose"]), binary, 8100
    )
    tail = argv[argv.index("--spec-type") :]
    assert tail == ["--spec-type", "ngram-mod,draft-mtp", "--metrics", "--verbose"]


def test_hand_written_speculation_gets_only_the_counters(
    tmp_path: Path, binary: DiscoveredBinary
) -> None:
    model = model_with_head(tmp_path / "m.gguf")
    argv = LlamaCppAdapter().build_argv(
        _spec(model, flags={"draftWith": "mtpHead"}, extraArgs=["--spec-type", "ngram-simple"]),
        binary,
        8100,
    )
    assert argv.count("--spec-type") == 1
    assert argv[-3:] == ["--metrics", "--spec-type", "ngram-simple"]


def test_the_run_shows_what_it_speculates_with(tmp_path: Path, binary: DiscoveredBinary) -> None:
    model = plain_model(tmp_path / "m.gguf")
    mtp = sidecar(tmp_path / "mtp-m.gguf")
    spec = _spec(model, flags={"draftWith": "companion", "draftModelPath": str(mtp)})
    adapter = LlamaCppAdapter()
    argv = adapter.build_argv(spec, binary, 8100)
    view = adapter.speculation_view(spec, argv)
    assert view is not None
    assert view.types == ["ngram-mod", "draft-mtp"] and view.draftModelPath == str(mtp)
    assert view.note is None


def test_a_build_without_it_says_so_on_the_run(
    tmp_path: Path, binary: DiscoveredBinary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llama_cpp, "_help_text", lambda path: HELP_BEFORE)
    spec = _spec(plain_model(tmp_path / "m.gguf"))
    adapter = LlamaCppAdapter()
    argv = adapter.build_argv(spec, binary, 8100)
    assert "--spec-type" not in argv and "--metrics" not in argv
    view = adapter.speculation_view(spec, argv)
    assert view is not None and view.types == []
    assert view.note is not None and "does not offer" in view.note


def test_counters_are_read_from_prometheus_text() -> None:
    text = (
        "# HELP llamacpp:spec_decode_num_draft_tokens_total drafted\n"
        "llamacpp:spec_decode_num_draft_tokens_total 1200\n"
        "llamacpp:spec_decode_num_accepted_tokens_total 913\n"
        'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 400\n'
    )
    assert speculation.parse_metrics(text) == (1200, 913)
    assert speculation.parse_metrics("") == (None, None)
    assert speculation.in_argv(["x", "--spec-type", "ngram-mod", "--spec-type=draft-mtp"]) == (
        ["ngram-mod", "draft-mtp"],
        None,
    )
    assert speculation.in_argv(["x", "--spec-type", "none"]) is None


# --- the node's own copy ------------------------------------------------------


def test_a_copy_of_the_model_carries_its_companion(tmp_path: Path) -> None:
    library = tmp_path / "library"
    model = plain_model(library / "qwen" / "m.gguf")
    mtp = sidecar(library / "qwen" / "mtp-m.gguf")
    settings = model_copies.CopySettings(
        enabled=True, directory=str(tmp_path / "copies"), min_free_bytes=0
    )
    rules = [model_copies.PathRule(source=str(library), target=str(library))]
    plan = model_copies.plan_for(str(model), rules, settings, companions=(str(mtp),))
    assert plan is not None
    assert [Path(m.destination).name for m in plan.files] == ["m.gguf", "mtp-m.gguf"]
    assert len({Path(m.destination).parent for m in plan.files}) == 1
    # Two runtimes of one model, one drafting: the copy keeps the companion.
    both = model_copies.wanted([str(model), (str(model), (str(mtp),))], rules, settings)
    assert [len(p.files) for p in both.values()] == [2]


# --- the settings build -------------------------------------------------------


def _trial(way: DraftWith, speed: float | None) -> DraftTrial:
    return DraftTrial(draftWith=way, decodeTokensPerSecond=speed)


def test_a_way_of_drafting_must_be_clearly_faster() -> None:
    off, head, file = DraftWith.off, DraftWith.mtpHead, DraftWith.companion
    assert recommend_draft([_trial(off, 50), _trial(head, 80), _trial(file, 70)]) == 1
    assert recommend_draft([_trial(off, 50), _trial(head, 51)]) == 0
    assert recommend_draft([_trial(off, None), _trial(head, 80)]) is None


def _build_plan(tmp_path: Path, model: Path, drafts=()) -> BuildPlan:  # type: ignore[no-untyped-def]
    tool = lambda name: tmp_path / name  # noqa: E731
    return BuildPlan(
        tools=Tools(
            server=tool("llama-server"),
            bench=tool("llama-bench"),
            fit=tool("llama-fit-params"),
            perplexity=tool("llama-perplexity"),
            version="b11719",
        ),
        model=model,
        work=tmp_path / "work",
        text=tmp_path / "text.txt",
        flags={},
        contexts=[4096],
        margin=None,
        env={},
        drafts=list(drafts),
    )


def test_the_build_times_every_way_this_model_can_draft(tmp_path: Path) -> None:
    model = model_with_head(tmp_path / "qwen" / "m.gguf")
    mtp = sidecar(tmp_path / "qwen" / "mtp-m.gguf")
    ways = drafting_ways(_build_plan(tmp_path, model, [("/lib/qwen/mtp-m.gguf", mtp)]))
    assert [(w, d) for w, d, _ in ways] == [
        (DraftWith.mtpHead, None),
        (DraftWith.companion, "/lib/qwen/mtp-m.gguf"),
    ]
    assert ways[1][2] == ["--spec-type", "draft-mtp", "--spec-draft-model", str(mtp)]
    assert drafting_ways(_build_plan(tmp_path, plain_model(tmp_path / "p.gguf"))) == []


@pytest.mark.asyncio
async def test_the_drafting_phase_keeps_the_fastest(tmp_path: Path) -> None:
    from eugene_plexus_agent._generated.models import (
        BuildCandidate,
        CacheType,
        ProfileBuild,
        ProfileBuildPhase,
    )

    model = model_with_head(tmp_path / "m.gguf")
    manager = ProfileBuilds(tmp_path / "builds.json")
    served: list[list[str]] = []

    async def serve(candidate, plan, *, extra=(), timed=False):  # type: ignore[no-untyped-def]
        served.append(list(extra))
        speed = 90.0 if "draft-mtp" in extra else 50.0
        drafted = {"draft_n": 300, "draft_n_accepted": 240} if extra else {}
        return True, "Timed in llama-server.", 9 * 2**30, {"predicted_per_second": speed, **drafted}

    manager._serve = serve  # type: ignore[method-assign]
    job = ProfileBuild.model_validate(
        {
            "id": "j",
            "node": "n",
            "modelId": "m",
            "runtime": {"name": "m", "engine": "llama_cpp", "modelPath": str(model)},
            "accuracy": "max",
            "evaluation": {"source": "bundled"},
            "state": "running",
            "phase": "confirming",
            "startedAt": "2026-10-18T00:00:00Z",
            "progress": 0.5,
            "detail": "",
            "quality": [],
            "candidates": [BuildCandidate(contextSize=4096, cacheType=CacheType.f16, placement=[])],
            "recommended": 0,
            "restarts": [],
        }
    )
    manager.jobs = [job]
    await asyncio.wait_for(manager._draft(job, _build_plan(tmp_path, model)), 5)

    assert job.phase is ProfileBuildPhase.drafting
    assert served == [[], ["--spec-type", "draft-mtp"]]
    assert [t.draftWith for t in job.draftTrials or []] == [DraftWith.off, DraftWith.mtpHead]
    assert job.recommendedDraft == 1
    trial = (job.draftTrials or [])[1]
    assert (trial.draftedTokens, trial.acceptedTokens) == (300, 240)
