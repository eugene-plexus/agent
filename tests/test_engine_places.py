"""An unset gpuLayers on a build with --fit is a launch llama.cpp places (moe-aware-fit call A)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import eugene_plexus_agent.admission as admission_module
from eugene_plexus_agent._generated.models import RuntimeSpec
from eugene_plexus_agent.admission import fit_disabled, wants_full_offload

GIB = 1024**3
_SIZES: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _fake_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission_module, "model_size_bytes", lambda p: _SIZES.get(p))


def _model(tmp_path: Path, size: int) -> str:
    path = str(tmp_path / f"m-{size}.gguf")
    _SIZES[path] = size
    return path


def _spec(**flags: object) -> RuntimeSpec:
    extra = flags.pop("extraArgs", None)
    return RuntimeSpec.model_validate(
        {
            "name": "m",
            "engine": "llama_cpp",
            "modelPath": "/m.gguf",
            "flags": flags,
            **({"extraArgs": extra} if extra else {}),
        }
    )


def test_unset_layers_are_partial_only_where_the_engine_places_them():
    assert wants_full_offload(_spec(), engine_places=False) is True
    assert wants_full_offload(_spec(), engine_places=True) is False
    # An explicit number is the operator's, whatever the build can do.
    assert wants_full_offload(_spec(gpuLayers=99), engine_places=True) is True
    assert wants_full_offload(_spec(gpuLayers=20), engine_places=False) is False


def test_fit_switched_off_by_hand_restores_full_offload():
    for args in (["--fit", "off"], ["-fit", "off"], ["--fit=off"]):
        spec = _spec(extraArgs=args)
        assert fit_disabled(spec)
        assert wants_full_offload(spec, engine_places=True) is True
    assert not fit_disabled(_spec(extraArgs=["--fit", "on"]))


def _places(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    from eugene_plexus_agent.routes import runtimes as runtime_routes

    monkeypatch.setattr(runtime_routes, "engine_places", lambda spec, get_config: value)


def test_a_split_with_unset_layers_is_admitted_and_says_llama_cpp_places_it(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _places(monkeypatch, True)
    # 30 GiB against a card with 24 free and 40 GiB of RAM: `split`.
    body = {"name": "m", "engine": "llama_cpp", "modelPath": _model(tmp_path, 30 * GIB)}
    answer = authed_client.post("/v1/runtimes/admission", json=body).json()
    assert answer["fit"] == "split"
    assert answer["decision"] == "admit"
    assert "left to llama.cpp" in answer["reason"]


def test_a_build_without_fit_keeps_the_old_refusal(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _places(monkeypatch, False)
    body = {"name": "m", "engine": "llama_cpp", "modelPath": _model(tmp_path, 30 * GIB)}
    answer = authed_client.post("/v1/runtimes/admission", json=body).json()
    assert answer["fit"] == "split" and answer["decision"] == "refuse"


def test_more_than_the_card_and_memory_together_is_still_refused(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _places(monkeypatch, True)
    body = {"name": "m", "engine": "llama_cpp", "modelPath": _model(tmp_path, 100 * GIB)}
    answer = authed_client.post("/v1/runtimes/admission", json=body).json()
    assert answer["fit"] == "no" and answer["decision"] == "refuse"


def test_places_by_itself_reads_the_builds_own_help(monkeypatch: pytest.MonkeyPatch):
    from eugene_plexus_agent.engines import llama_cpp

    adapter = llama_cpp.LlamaCppAdapter()
    found = SimpleNamespace(path=Path("llama-server"), version="b11215")
    monkeypatch.setattr(
        llama_cpp.LlamaCppAdapter, "resolve_binary", lambda self, spec, configured=None: found
    )
    helps = {
        "a build with --fit": frozenset({"--fit", "--ctx-size"}),
        "a build without it": frozenset({"--ctx-size"}),
        "a help that could not be read": None,
    }
    answers = {}
    for label, flags in helps.items():
        monkeypatch.setattr(llama_cpp, "_supported_long_flags", lambda path, f=flags: f)
        answers[label] = llama_cpp.places_by_itself(adapter, _spec(), None)
    assert answers == {
        "a build with --fit": True,
        "a build without it": False,
        "a help that could not be read": False,
    }

    def unavailable(self, spec, configured=None):
        raise RuntimeError("no llama-server")

    monkeypatch.setattr(llama_cpp.LlamaCppAdapter, "resolve_binary", unavailable)
    assert llama_cpp.places_by_itself(adapter, _spec(), None) is False


def test_another_engine_is_always_full_offload_whatever_is_claimed():
    # vLLM has no partial offload: even a wrong claim that "the engine
    # places it" must not turn its split into an admit.
    vllm = RuntimeSpec.model_validate({"name": "v", "engine": "vllm", "modelPath": "/m"})
    assert wants_full_offload(vllm, engine_places=True) is True
