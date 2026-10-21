"""SD2: vLLM arguments and counters, with no engine or agent process."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from eugene_plexus_agent import _http
from eugene_plexus_agent._generated.models import Origin, RuntimeSpec
from eugene_plexus_agent.engines import speculation, vllm
from eugene_plexus_agent.engines.base import DiscoveredBinary

FLAG = "--speculative-config"
COMPANION = "/models/Qwen/draft"


@pytest.fixture
def binary(monkeypatch: pytest.MonkeyPatch) -> DiscoveredBinary:
    monkeypatch.setattr(vllm, "_supported_long_flags", lambda path: frozenset({FLAG}))
    return DiscoveredBinary(
        path=Path("/opt/vllm/bin/vllm"), origin=Origin.configured, version="0.30.0"
    )


def spec(**overrides: object) -> RuntimeSpec:
    return RuntimeSpec.model_validate(
        {"name": "qwen", "engine": "vllm", "modelPath": "/models/Qwen/model", **overrides}
    )


def config(argv: list[str]) -> dict[str, object]:
    assert argv.count(FLAG) == 1
    return json.loads(argv[argv.index(FLAG) + 1])  # type: ignore[no-any-return]


def test_own_head_uses_mtp(binary: DiscoveredBinary) -> None:
    adapter = vllm.VllmAdapter()
    runtime = spec(flags={"draftWith": "mtpHead", "draftTokens": 6, "reuseRepeatedText": False})
    argv = adapter.build_argv(runtime, binary, 8100)
    assert config(argv) == {"method": "mtp", "num_speculative_tokens": 6}
    view = adapter.speculation_view(runtime, argv)
    assert view is not None and view.types == ["mtp"]
    assert view.draftModelPath is None and view.note is None


def test_companion_uses_draft_model_with_its_path(binary: DiscoveredBinary) -> None:
    adapter = vllm.VllmAdapter()
    runtime = spec(
        flags={"draftWith": "companion", "draftModelPath": COMPANION, "reuseRepeatedText": False}
    )
    argv = adapter.build_argv(runtime, binary, 8100)
    assert config(argv) == {
        "method": "draft_model",
        "model": COMPANION,
        "num_speculative_tokens": 3,
    }
    view = adapter.speculation_view(runtime, argv)
    assert view is not None and view.types == ["draft_model"]
    assert view.draftModelPath == COMPANION and view.note is None


def test_off_with_reuse_uses_ngram(binary: DiscoveredBinary) -> None:
    adapter = vllm.VllmAdapter()
    runtime = spec(flags={"draftWith": "off", "reuseRepeatedText": True, "draftTokens": 4})
    argv = adapter.build_argv(runtime, binary, 8100)
    assert config(argv) == {
        "method": "ngram",
        "num_speculative_tokens": 4,
        "prompt_lookup_min": 5,
        "prompt_lookup_max": 5,
    }
    view = adapter.speculation_view(runtime, argv)
    assert view is not None and view.types == ["ngram"] and view.note is None
    # The supervisor uses this shared helper to decide whether to poll counters.
    assert speculation.in_argv(argv) == (["ngram"], None)


def test_reuse_is_on_by_default(binary: DiscoveredBinary) -> None:
    assert config(vllm.VllmAdapter().build_argv(spec(), binary, 8100))["method"] == "ngram"


@pytest.mark.parametrize("choice,method", [("mtpHead", "mtp"), ("companion", "draft_model")])
def test_chosen_model_wins_over_reuse(binary: DiscoveredBinary, choice: str, method: str) -> None:
    adapter = vllm.VllmAdapter()
    runtime = spec(
        flags={"draftWith": choice, "draftModelPath": COMPANION, "reuseRepeatedText": True}
    )
    argv = adapter.build_argv(runtime, binary, 8100)
    assert config(argv)["method"] == method
    assert "prompt_lookup_max" not in config(argv)
    view = adapter.speculation_view(runtime, argv)
    assert view is not None and view.types == [method]
    assert view.note and "ngram" in view.note and "not in effect" in view.note
    assert speculation.in_argv(argv) == (view.types, view.draftModelPath)


@pytest.mark.parametrize("inline", [False, True])
def test_extra_args_set_speculation_by_hand(binary: DiscoveredBinary, inline: bool) -> None:
    adapter = vllm.VllmAdapter()
    value = json.dumps({"method": "draft_model", "model": COMPANION, "num_speculative_tokens": 2})
    extra = [f"{FLAG}={value}"] if inline else [FLAG, value]
    runtime = spec(flags={"draftWith": "mtpHead"}, extraArgs=extra)
    argv = adapter.build_argv(runtime, binary, 8100)
    assert sum(token.partition("=")[0] == FLAG for token in argv) == 1
    assert argv[-len(extra) :] == extra
    view = adapter.speculation_view(runtime, argv)
    assert view is not None and view.types == ["draft_model"] and view.draftModelPath == COMPANION
    assert view.note and "by hand" in view.note


@pytest.mark.parametrize("offered", [frozenset(), None])
def test_unavailable_build_omits_config_and_logs(
    binary: DiscoveredBinary,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    offered: frozenset[str] | None,
) -> None:
    monkeypatch.setattr(vllm, "_supported_long_flags", lambda path: offered)
    adapter = vllm.VllmAdapter()
    runtime = spec(flags={"draftWith": "mtpHead"})
    argv = adapter.build_argv(runtime, binary, 8100)
    assert FLAG not in argv
    assert FLAG in caplog.text and "not passed" in caplog.text
    view = adapter.speculation_view(runtime, argv)
    assert view is not None and view.types == []
    assert view.note and FLAG in view.note


def test_off_without_reuse_does_not_probe(
    binary: DiscoveredBinary, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_probe(path: Path) -> frozenset[str]:
        pytest.fail("Speculation is off; there is no flag to check")

    monkeypatch.setattr(vllm, "_supported_long_flags", unexpected_probe)
    adapter = vllm.VllmAdapter()
    runtime = spec(flags={"draftWith": "off", "reuseRepeatedText": False})
    argv = adapter.build_argv(runtime, binary, 8100)
    assert FLAG not in argv and adapter.speculation_view(runtime, argv) is None
    assert adapter.speculation_view(runtime, []) is None


def test_running_view_reads_argv_without_reprobing(
    binary: DiscoveredBinary, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = vllm.VllmAdapter()
    argv = adapter.build_argv(spec(flags={"draftWith": "mtpHead"}), binary, 8100)

    def unexpected_probe(path: Path) -> frozenset[str]:
        pytest.fail("Rendering a run must not probe the executable")

    monkeypatch.setattr(vllm, "_supported_long_flags", unexpected_probe)
    runtime = spec(flags={"draftWith": "off", "reuseRepeatedText": False})
    view = adapter.speculation_view(runtime, argv)
    assert view is not None and view.types == ["mtp"]


@pytest.mark.parametrize("value", ["null", "[]", "{broken", '{"method":"ngram"}'])
def test_manual_config_view_does_not_crash(binary: DiscoveredBinary, value: str) -> None:
    adapter = vllm.VllmAdapter()
    runtime = spec(extraArgs=[FLAG, value])
    argv = adapter.build_argv(runtime, binary, 8100)
    view = adapter.speculation_view(runtime, argv)
    assert view is not None and view.note and "by hand" in view.note


def test_companion_declaration_and_copy_hook() -> None:
    adapter = vllm.VllmAdapter()
    runtime = spec(flags={"draftWith": "companion", "draftModelPath": COMPANION})
    assert adapter.declaration_problem(runtime) is None
    assert adapter.companion_files(runtime) == (COMPANION,)
    assert adapter.declaration_problem(spec(flags={"draftWith": "companion"}))
    assert adapter.declaration_problem(
        spec(flags={"draftWith": "companion", "draftModelPath": "/other/draft"})
    )


@pytest.mark.asyncio
async def test_metrics_reads_and_sums_vllm_token_counters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://engine:8100/metrics"
        return httpx.Response(
            200,
            text=(
                "# HELP vllm:spec_decode_num_draft_tokens_total Draft tokens\n"
                'vllm:spec_decode_num_draft_tokens_total{model_name="Qwen model",engine="0"} 1.2e3\n'
                'vllm:spec_decode_num_draft_tokens_total{model_name="Qwen model",engine="1"} 300\n'
                'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 913.0\n'
                'vllm:spec_decode_num_accepted_tokens_total{engine="1"} 200\n'
                'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 400\n'
                "vllm:spec_decode_num_drafts_total 500\n"
                "vllm:spec_decode_num_draft_tokens_created 999\n"
                "llamacpp:spec_decode_num_draft_tokens_total 99\n"
            ),
        )

    _http.set_shared_client(
        "engine-probe", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    assert await vllm.VllmAdapter().read_speculation("http://engine:8100/") == (1500, 1113)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,expected",
    [
        ("", (None, None)),
        ("vllm:spec_decode_num_accepted_tokens_total 0\n", (None, 0)),
        (
            "vllm:spec_decode_num_draft_tokens_total NaN\n"
            "vllm:spec_decode_num_draft_tokens_total +Inf\n"
            "vllm:spec_decode_num_draft_tokens_total -1\n"
            "vllm:spec_decode_num_accepted_tokens_total bad\n",
            (None, None),
        ),
    ],
)
async def test_missing_or_invalid_metrics_stay_unknown(
    body: str, expected: tuple[int | None, int | None]
) -> None:
    _http.set_shared_client(
        "engine-probe",
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body))),
    )
    assert await vllm.VllmAdapter().read_speculation("http://engine:8100") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 503])
async def test_unavailable_metrics_are_best_effort(status: int) -> None:
    _http.set_shared_client(
        "engine-probe",
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status))),
    )
    assert await vllm.VllmAdapter().read_speculation("http://engine:8100") is None


@pytest.mark.asyncio
async def test_metrics_connection_failure_is_best_effort() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _http.set_shared_client(
        "engine-probe", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    assert await vllm.VllmAdapter().read_speculation("http://engine:8100") is None
