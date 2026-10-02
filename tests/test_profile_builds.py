"""The profile build job: its parsers, its choices, and whole runs with faked tools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eugene_plexus_agent._generated.models import (
    BuildCandidate,
    CacheType,
    MeasurementRestart,
    ProfileBuildRequest,
)
from eugene_plexus_agent.profile_builds import (
    BuildError,
    BuildPlan,
    ProfileBuilds,
    Tools,
    _distinct_placements,
    bench_placement,
    contexts_for,
    mark_frontier,
    parse_fit,
    quality_from_output,
    recommend,
    split_arguments,
    too_short,
)

# What llama-perplexity printed for the MoE model's q8_0 cache (design §0 M7).
KLD_Q8 = """
kl_divergence: computing over 4 chunks, n_ctx=4096, batch_size=2048, n_seq=1
====== KL divergence statistics ======
Mean    KLD:   0.003857 ±   0.000260
RMS Δp    :  2.129 ± 0.177 %
Same top p: 97.289 ± 0.127 %
"""
KLD_Q4 = KLD_Q8.replace("0.003857", "0.028217").replace("97.289 ± 0.127", "93.106 ± 0.198")

# What llama-fit-params printed for the MoE model at the 8 GB margin, 16k.
FIT_MOE = (
    '-c 16384 -ngl 49 -ot "blk\\.13\\.ffn_down.*=CPU,'
    'blk\\.14\\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"'
)


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


def test_fit_output_keeps_the_regex_backslashes():
    tokens = split_arguments(FIT_MOE)
    assert tokens[:4] == ["-c", "16384", "-ngl", "49"]
    assert tokens[5].startswith("blk\\.13\\.ffn_down")
    context, placement = parse_fit("llama_fit_params: printing…\n" + FIT_MOE + "\n")
    assert context == 16384
    assert placement[:2] == ["-ngl", "49"] and placement[2] == "-ot"


def test_bench_gets_fits_placement_in_its_own_separators():
    _, placement = parse_fit(FIT_MOE)
    bench = bench_placement(placement)
    # llama-bench reads commas as a sweep, so patterns are `;`-separated.
    assert bench[3].count(";") == 1 and "," not in bench[3]
    assert bench_placement(["-ts", "20,12", "-dev", "CUDA0,Vulkan1"]) == [
        "-ts",
        "20/12",
        "-dev",
        "CUDA0/Vulkan1",
    ]


def test_a_placement_flag_a_build_cannot_carry_is_refused():
    with pytest.raises(BuildError, match="--rpc"):
        parse_fit("-c 4096 --rpc 10.0.0.1:50052")
    with pytest.raises(BuildError, match="no context"):
        parse_fit("-ngl -1")


def test_high_admits_the_8_bit_cache_the_acceptance_run_measured():
    # PB1's GPU runs: the MoE model's q8_0 cache scored 96.32% ± 0.21 on the
    # bundled text. Troy lowered High to 96% for exactly this (2026-09-30);
    # at the old 96.5% it failed.
    from eugene_plexus_agent._generated.models import ProfileBuildAccuracy
    from eugene_plexus_agent.profile_builds import THRESHOLDS

    measured = KLD_Q8.replace("97.289 ± 0.127", "96.320 ± 0.210")
    high = THRESHOLDS[ProfileBuildAccuracy.high]
    assert quality_from_output(measured, CacheType.q8_0, high).passes
    assert not quality_from_output(measured, CacheType.q8_0, 96.5).passes


@pytest.mark.asyncio
async def test_low_allows_the_4_bit_cache_medium_refuses(tmp_path):
    # The MoE model's q4_0 cache on the bundled text: 88.40% ± 0.35 (run 3).
    moe_q4 = KLD_Q8.replace("97.289 ± 0.127", "88.400 ± 0.350")
    allowed = {}
    for level in ("medium", "low"):
        (tmp_path / level).mkdir()
        manager = _runner(tmp_path / level, FakeTools(q4=moe_q4))
        manager.start(
            _request(level),
            _plan(tmp_path / level),
            node="n",
            restarts=[],
            evaluation={"source": "bundled"},
            after=None,
        )
        job = await _finish(manager)
        allowed[level] = [c.value for c in job.allowedCacheTypes]
    assert allowed == {"medium": ["f16", "q8_0"], "low": ["f16", "q8_0", "q4_0"]}


def test_quality_reads_same_top_token_and_applies_the_boundary_rule():
    q8 = quality_from_output(KLD_Q8, CacheType.q8_0, 96.5)
    assert q8.sameTopTokenPercent == pytest.approx(97.289)
    assert q8.meanKld == pytest.approx(0.003857)
    assert q8.tokensScored == 4 * 2048 and q8.passes
    q4 = quality_from_output(KLD_Q4, CacheType.q4_0, 96.5)
    assert not q4.passes and quality_from_output(KLD_Q4, CacheType.q4_0, 92.0).passes
    # Measured minus one standard error must clear it: 96.6 ± 0.2 does not.
    borderline = KLD_Q8.replace("97.289 ± 0.127", "96.600 ± 0.200")
    assert not quality_from_output(borderline, CacheType.q8_0, 96.5).passes
    with pytest.raises(BuildError):
        quality_from_output("nothing useful", CacheType.q8_0, 96.5)


def test_a_short_text_is_reported_with_both_counts():
    text = (
        "perplexity: you need at least 8192 tokens to evaluate perplexity with a context of 4096\n"
        "perplexity: the data file you provided tokenizes to only 3120 tokens\n"
    )
    assert too_short(text) == (8192, 3120)
    assert too_short(KLD_Q8) is None


def test_contexts_are_capped_at_what_the_model_was_trained_for():
    assert contexts_for(262144) == [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    assert contexts_for(32768) == [4096, 8192, 16384, 32768]
    assert contexts_for(5000) == [4096, 5000]
    assert contexts_for(None)[-1] == 131072


def _candidate(context, cache, deep, placement=("-ngl", "-1")):
    return BuildCandidate(
        contextSize=context,
        cacheType=cache,
        placement=list(placement),
        decodeTokensPerSecond=deep,
        deepDecodeTokensPerSecond=deep,
        onFrontier=False,
    )


def test_frontier_and_default_on_the_measured_8gb_numbers():
    # Design §0 M4 at the simulated 8 GB card.
    candidates = [
        _candidate(4096, CacheType.f16, 50.6),
        _candidate(65536, CacheType.f16, 34.4),
        _candidate(65536, CacheType.q8_0, 40.7),
    ]
    mark_frontier(candidates)
    assert [c.onFrontier for c in candidates] == [True, False, True]
    # 40.7 keeps 80% of 50.6, so the default is the longer memory.
    assert recommend(candidates) == 2
    candidates[2].deepDecodeTokensPerSecond = 30.0
    mark_frontier(candidates)
    assert recommend(candidates) == 0
    assert recommend([]) is None


def test_where_every_context_places_alike_the_longest_wins():
    # The first CPU acceptance run's shape: five rungs, one placement, speeds
    # equal but for noise. Measured at a common depth they tie, and a tie
    # means the longer context costs nothing, so it alone is on the frontier.
    speeds = {4096: 16.9, 8192: 16.6, 16384: 16.8, 32768: 16.7, 40960: 16.7}
    candidates = [_candidate(c, CacheType.f16, s) for c, s in speeds.items()]
    mark_frontier(candidates)
    assert [c.contextSize for c in candidates if c.onFrontier] == [40960]
    assert candidates[recommend(candidates)].contextSize == 40960


def test_at_one_context_and_speed_the_more_precise_cache_wins():
    # The third GPU acceptance run at 8k and 64k (docs/acceptance/profile-builder-run.md).
    candidates = [
        _candidate(8192, CacheType.f16, 48.4),
        _candidate(8192, CacheType.q8_0, 48.8),
        _candidate(65536, CacheType.f16, 32.9),
        _candidate(65536, CacheType.q8_0, 39.8),
    ]
    mark_frontier(candidates)
    on = [(c.contextSize, c.cacheType.value) for c in candidates if c.onFrontier]
    assert on == [(8192, "f16"), (65536, "q8_0")]


def test_candidates_are_compared_at_one_depth():
    from eugene_plexus_agent.profile_builds import COMMON_DEPTH, deep_depth

    assert {deep_depth(c) for c in (4096, 65536, 262144)} == {COMMON_DEPTH}


def test_a_smaller_cache_placed_the_same_way_only_costs_quality():
    same = {CacheType.f16: ["-ngl", "-1"], CacheType.q8_0: ["-ngl", "-1"]}
    assert _distinct_placements(same) == [CacheType.f16]
    moved = {CacheType.f16: ["-ngl", "49", "-ot", "a"], CacheType.q8_0: ["-ngl", "49", "-ot", "b"]}
    assert _distinct_placements(moved) == [CacheType.f16, CacheType.q8_0]


# --------------------------------------------------------------------------- #
# Whole runs, with every tool faked at the process boundary
# --------------------------------------------------------------------------- #


def _request(accuracy="high", **extra):
    return ProfileBuildRequest.model_validate(
        {
            "modelId": "m",
            "runtime": {"name": "m", "engine": "llama_cpp", "modelPath": "/models/m.gguf"},
            "accuracy": accuracy,
            **extra,
        }
    )


def _plan(tmp_path, contexts=(4096, 65536)):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"fixture")
    tool = lambda name: tmp_path / name  # noqa: E731
    return BuildPlan(
        tools=Tools(
            server=tool("llama-server"),
            bench=tool("llama-bench"),
            fit=tool("llama-fit-params"),
            perplexity=tool("llama-perplexity"),
            version="b11215",
        ),
        model=model,
        work=tmp_path / "work",
        text=tmp_path / "text.txt",
        flags={},
        contexts=list(contexts),
        margin=None,
        env={},
    )


class FakeTools:
    """Answers each tool's argv the way the real ones did in §0."""

    def __init__(self, *, q8=KLD_Q8, q4=KLD_Q4, short=False, fail_bench=False):
        self.q8, self.q4, self.short, self.fail_bench = q8, q4, short, fail_bench
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(argv)
        name = Path(argv[0]).name
        if name == "llama-perplexity":
            if self.short:
                return 1, "", "you need at least 8192 tokens ... tokenizes to only 900 tokens"
            if "--kl-divergence" in argv:
                cache = argv[argv.index("--cache-type-k") + 1]
                return 0, (self.q4 if cache == "q4_0" else self.q8), ""
            Path(argv[argv.index("--kl-divergence-base") + 1]).write_bytes(b"base")
            return 0, "", ""
        if name == "llama-fit-params":
            context = argv[argv.index("--ctx-size") + 1]
            cache = argv[argv.index("--cache-type-k") + 1]
            if context == "4096":
                return 0, f"-c {context} -ngl -1\n", ""
            ot = "blk\\.20.*=CPU" if cache == "f16" else "blk\\.30.*=CPU"
            return 0, f'-c {context} -ngl 49 -ot "{ot}"\n', ""
        if name == "llama-bench":
            if self.fail_bench:
                return 1, "", "CUDA error: out of memory"
            depth = int(argv[argv.index("--n-depth") + 1].split(",")[1])
            cache = argv[argv.index("--cache-type-k") + 1]
            slow = "-ot" in argv
            base = 40.0 if slow else 50.0
            if slow and cache == "q8_0":
                base = 45.0
            rows = [
                {"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 700.0},
                {"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": base},
                {"n_prompt": 512, "n_gen": 0, "n_depth": depth, "avg_ts": 600.0},
                {"n_prompt": 0, "n_gen": 128, "n_depth": depth, "avg_ts": base - 2},
            ]
            return 0, "\n".join(json.dumps(r | {"gpu_info": "Fake GPU"}) for r in rows), ""
        raise AssertionError(f"unexpected tool {name}")


def _runner(tmp_path, tools, *, serve_ok=True):
    manager = ProfileBuilds(tmp_path / "builds.json")

    async def child(argv, plan, *, on_stderr=None):
        if manager.stopping:
            raise asyncio.CancelledError
        await asyncio.sleep(0)
        return tools(argv)

    async def serve(candidate, plan):
        return serve_ok, "fixture" if serve_ok else "llama-server exited while loading.", 7 * 2**30

    manager._child = child  # type: ignore[method-assign]
    manager._serve_once = serve  # type: ignore[method-assign]
    return manager


async def _finish(manager):
    assert manager.task is not None
    await asyncio.wait_for(manager.task, 10)
    return manager.jobs[-1]


@pytest.mark.asyncio
async def test_high_build_measures_quality_then_places_measures_and_confirms(tmp_path):
    tools = FakeTools()
    manager = _runner(tmp_path, tools)
    restarted = MeasurementRestart(name="busy", state="restarted", detail="Started again.")

    async def after():
        return [restarted]

    manager.start(
        _request("high"),
        _plan(tmp_path),
        node="n",
        restarts=[MeasurementRestart(name="busy", state="pending")],
        evaluation={"source": "bundled"},
        after=after,
    )
    job = await _finish(manager)
    assert job.state.value == "completed", job.detail
    assert [q.cacheType.value for q in job.quality] == ["q8_0"]
    assert job.allowedCacheTypes == [CacheType.f16, CacheType.q8_0]
    # 4k: both caches placed alike, so only f16 is kept; 64k: both kept.
    kept = [(c.contextSize, c.cacheType.value) for c in job.candidates]
    assert kept == [(4096, "f16"), (65536, "f16"), (65536, "q8_0")]
    # Placement reached llama-bench explicitly, never through -fitc.
    bench_calls = [a for a in tools.calls if Path(a[0]).name == "llama-bench"]
    assert all("-fitc" not in a and "--fit-ctx" not in a for a in bench_calls)
    assert any("-ot" in a for a in bench_calls)
    # llama-bench's own warm-up run absorbs the first load's slow prefill
    # (design §0 M3); a build must never turn it off.
    assert all("--no-warmup" not in a for a in bench_calls)
    # Each placement reached llama-bench exactly as fit printed it.
    for candidate in job.candidates:
        if "-ot" in candidate.placement:
            pattern = candidate.placement[candidate.placement.index("-ot") + 1]
            assert any(pattern.replace(",", ";") in a for a in bench_calls)
    chosen = job.candidates[job.recommended]
    assert (chosen.contextSize, chosen.cacheType.value) == (65536, "q8_0")
    assert chosen.confirmed is True and chosen.graphicsMemoryBytes == 7 * 2**30
    assert job.restarts == [restarted]
    assert not (tmp_path / "work").exists()


@pytest.mark.asyncio
async def test_max_never_measures_quality_and_never_quantises(tmp_path):
    tools = FakeTools()
    manager = _runner(tmp_path, tools)
    manager.start(
        _request("max"),
        _plan(tmp_path),
        node="n",
        restarts=[],
        evaluation={"source": "bundled"},
        after=None,
    )
    job = await _finish(manager)
    assert job.state.value == "completed", job.detail
    assert not any(Path(a[0]).name == "llama-perplexity" for a in tools.calls)
    assert {c.cacheType.value for c in job.candidates} == {"f16"}
    assert job.quality == []


@pytest.mark.asyncio
async def test_a_cache_that_fails_the_level_is_not_allowed(tmp_path):
    tools = FakeTools(q8=KLD_Q4)  # 93.1% same top token: below High
    manager = _runner(tmp_path, tools)
    manager.start(
        _request("high"),
        _plan(tmp_path),
        node="n",
        restarts=[],
        evaluation={"source": "bundled"},
        after=None,
    )
    job = await _finish(manager)
    assert job.quality[0].passes is False
    assert job.allowedCacheTypes == [CacheType.f16]
    assert {c.cacheType.value for c in job.candidates} == {"f16"}


@pytest.mark.asyncio
async def test_a_short_text_fails_the_build_with_the_counts(tmp_path):
    manager = _runner(tmp_path, FakeTools(short=True))
    manager.start(
        _request("high"),
        _plan(tmp_path),
        node="n",
        restarts=[],
        evaluation={"source": "custom"},
        after=None,
    )
    job = await _finish(manager)
    assert job.state.value == "failed"
    assert "900" in job.detail and "8,192" in job.detail
    assert job.evaluation.tokens == 900


@pytest.mark.asyncio
async def test_a_build_that_cannot_measure_anything_fails_and_still_restarts(tmp_path):
    manager = _runner(tmp_path, FakeTools(fail_bench=True))
    calls = []

    async def after():
        calls.append("after")
        return []

    manager.start(
        _request("max"),
        _plan(tmp_path),
        node="n",
        restarts=[],
        evaluation={"source": "bundled"},
        after=after,
    )
    job = await _finish(manager)
    assert job.state.value == "failed" and "measured" in job.detail
    assert all("out of memory" in (c.detail or "") for c in job.candidates)
    assert calls == ["after"]


@pytest.mark.asyncio
async def test_a_candidate_llama_server_cannot_load_is_not_recommended(tmp_path):
    manager = _runner(tmp_path, FakeTools(), serve_ok=False)
    manager.start(
        _request("max"),
        _plan(tmp_path),
        node="n",
        restarts=[],
        evaluation={"source": "bundled"},
        after=None,
    )
    job = await _finish(manager)
    assert job.state.value == "failed"
    assert job.recommended is None
    assert any(c.confirmed is False for c in job.candidates)


@pytest.mark.asyncio
async def test_cancel_keeps_what_was_measured_and_restarts(tmp_path):
    gate, measuring = asyncio.Event(), asyncio.Event()
    tools = FakeTools()
    manager = _runner(tmp_path, tools)
    original = manager._child

    async def slow(argv, plan, *, on_stderr=None):
        if Path(argv[0]).name == "llama-bench":
            measuring.set()
            await gate.wait()
        return await original(argv, plan)

    manager._child = slow  # type: ignore[method-assign]
    calls = []

    async def after():
        calls.append("after")
        return []

    manager.start(
        _request("max"),
        _plan(tmp_path),
        node="n",
        restarts=[],
        evaluation={"source": "bundled"},
        after=after,
    )
    await asyncio.wait_for(measuring.wait(), 10)
    cancel = asyncio.create_task(manager.cancel(manager.jobs[-1].id))
    await asyncio.sleep(0.05)
    gate.set()
    job = await asyncio.wait_for(cancel, 10)
    assert job.state.value == "cancelled"
    assert job.candidates and calls == ["after"]


def test_an_interrupted_build_is_failed_and_its_restarts_explained(tmp_path):
    manager = _runner(tmp_path, FakeTools())
    history = tmp_path / "builds.json"
    record = {
        "id": "x",
        "node": "n",
        "modelId": "m",
        "runtime": {"name": "m", "engine": "llama_cpp", "modelPath": "/m.gguf"},
        "accuracy": "max",
        "evaluation": {"source": "bundled"},
        "state": "running",
        "phase": "measuring",
        "startedAt": "2026-09-30T00:00:00Z",
        "progress": 0.5,
        "detail": "",
        "quality": [],
        "candidates": [],
        "restarts": [{"name": "busy", "state": "pending"}],
    }
    history.write_text(json.dumps([record]), encoding="utf-8")
    del manager
    reloaded = ProfileBuilds(history)
    job = reloaded.jobs[0]
    assert job.state.value == "failed"
    assert job.restarts[0].state.value == "skipped"
    assert "restarted" in (job.restarts[0].detail or "")


@pytest.mark.asyncio
async def test_a_shutdown_mid_build_starts_nothing_on_the_way_out(tmp_path):
    gate = asyncio.Event()
    manager = _runner(tmp_path, FakeTools())
    original = manager._child

    async def slow(argv, plan, *, on_stderr=None):
        await gate.wait()
        return await original(argv, plan)

    manager._child = slow  # type: ignore[method-assign]
    calls = []

    async def after():
        calls.append("after")
        return []

    manager.start(
        _request("max"),
        _plan(tmp_path),
        node="n",
        restarts=[MeasurementRestart(name="busy", state="pending")],
        evaluation={"source": "bundled"},
        after=after,
    )
    await asyncio.sleep(0)
    closing = asyncio.create_task(manager.close())
    await asyncio.sleep(0.05)
    gate.set()  # what terminating the real child does: its call returns
    await asyncio.wait_for(closing, 10)
    job = manager.jobs[-1]
    assert calls == [], "a model started while the agent shuts down would be orphaned"
    assert job.restarts[0].state.value == "skipped"


def test_prepare_tools_names_a_missing_sibling_or_option(monkeypatch, tmp_path):
    import os

    from eugene_plexus_agent.routes import profile_builds

    suffix = ".exe" if os.name == "nt" else ""
    server = tmp_path / f"llama-server{suffix}"
    server.write_bytes(b"")
    found = SimpleNamespace(path=server, version="b10000")
    monkeypatch.setattr(
        profile_builds.LlamaCppAdapter, "resolve_binary", lambda self, spec, configured=None: found
    )
    body = _request("max")
    with pytest.raises(BuildError, match="llama-bench is missing"):
        profile_builds.prepare_tools(body, lambda: {})

    # Every sibling present, but one lacks an option a build needs.
    for tool in profile_builds.REQUIRED_OPTIONS:
        (tmp_path / f"{tool}{suffix}").write_bytes(b"")

    def fake_help(argv, **kwargs):
        name = Path(argv[0]).stem
        options = set(profile_builds.REQUIRED_OPTIONS[name])
        if name == "llama-perplexity":
            options.discard("--kl-divergence-base")
        return SimpleNamespace(stdout=" ".join(sorted(options)), stderr="", returncode=0)

    monkeypatch.setattr(profile_builds.subprocess, "run", fake_help)
    with pytest.raises(BuildError, match=r"llama-perplexity.*--kl-divergence-base"):
        profile_builds.prepare_tools(body, lambda: {})


def _gguf(path: Path, entries: list[tuple[str, int, bytes]]) -> Path:
    import struct

    def string(s: str) -> bytes:
        raw = s.encode()
        return struct.pack("<Q", len(raw)) + raw

    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(entries))
    for key, kind, value in entries:
        body += string(key) + struct.pack("<I", kind) + value
    path.write_bytes(body)
    return path


def test_the_gguf_header_gives_the_trained_context_past_arrays(tmp_path):
    import struct

    from eugene_plexus_agent.gguf_context import trained_context

    def string_value(s: str) -> bytes:
        raw = s.encode()
        return struct.pack("<Q", len(raw)) + raw

    strings = struct.pack("<I", 8) + struct.pack("<Q", 2) + string_value("a") + string_value("bb")
    numbers = struct.pack("<I", 6) + struct.pack("<Q", 3) + struct.pack("<3f", 1, 2, 3)
    model = _gguf(
        tmp_path / "m.gguf",
        [
            ("general.architecture", 8, string_value("qwen3moe")),
            ("general.tags", 9, strings),
            ("qwen3moe.rope.freqs", 9, numbers),
            ("qwen3moe.block_count", 4, struct.pack("<I", 48)),
            ("qwen3moe.context_length", 4, struct.pack("<I", 262144)),
        ],
    )
    assert trained_context(model) == 262144
    assert trained_context(_gguf(tmp_path / "none.gguf", [])) is None
    (tmp_path / "junk.gguf").write_bytes(b"not a model")
    assert trained_context(tmp_path / "junk.gguf") is None
    assert trained_context(tmp_path / "missing.gguf") is None


def test_route_refuses_a_running_model_nobody_agreed_to(authed_client, stub_runtime_supervisor):
    spec = {"name": "busy", "engine": "llama_cpp", "modelPath": "/models/busy.gguf"}
    assert authed_client.post("/v1/runtimes", json=spec).status_code == 201
    before = list(stub_runtime_supervisor.calls)
    body = _request("max").model_dump(mode="json")
    response = authed_client.post("/v1/profile-builds", json=body)
    assert response.status_code == 409 and "busy" in response.json()["detail"]
    assert stub_runtime_supervisor.calls == before
    answer = authed_client.post("/v1/profile-builds/preflight", json=body).json()
    assert answer["runningRuntimes"] == ["busy"]
    assert stub_runtime_supervisor.calls == before


def test_route_names_a_missing_tool_before_stopping_anything(
    authed_client, stub_runtime_supervisor, monkeypatch, tmp_path
):
    from eugene_plexus_agent.routes import profile_builds

    model = tmp_path / "m.gguf"
    model.write_bytes(b"fixture")

    def missing(*args):
        raise BuildError("llama-fit-params is missing beside the selected llama-server (b10000).")

    monkeypatch.setattr(profile_builds, "prepare_tools", missing)
    body = _request("max").model_dump(mode="json")
    body["runtime"]["modelPath"] = str(model)
    preflight = authed_client.post("/v1/profile-builds/preflight", json=body).json()
    assert any("llama-fit-params" in p for p in preflight["problems"])
    response = authed_client.post("/v1/profile-builds", json=body)
    assert response.status_code == 422 and "llama-fit-params" in response.text
    assert authed_client.get("/v1/profile-builds").json() == {"builds": []}


def test_a_running_build_holds_launches(authed_client, app):
    app.state.profile_builds = SimpleNamespace(active=True)
    try:
        spec = {"name": "x", "engine": "llama_cpp", "modelPath": "/models/x.gguf"}
        response = authed_client.post("/v1/runtimes", json=spec)
        assert response.status_code == 409 and "settings build" in response.text
    finally:
        del app.state.profile_builds


def test_a_builds_trials_follow_the_profiles_flash_attention():
    """agent#6: an f16 trial follows the profile; a quantised one needs it on."""
    from eugene_plexus_agent.profile_builds import CacheType, cache_args

    def flash(cache, choice):
        args = cache_args(cache, choice)
        return args[args.index("--flash-attn") + 1] if "--flash-attn" in args else None

    assert flash(CacheType.f16, None) is None
    assert flash(CacheType.f16, "off") == "off"
    assert flash(CacheType.f16, "on") == "on"
    assert flash(CacheType.q8_0, "off") == "on"
    assert flash(CacheType.q4_0, None) == "on"
