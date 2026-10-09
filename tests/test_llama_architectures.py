"""llama.cpp declares the architectures its build loads (LS2, call B8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eugene_plexus_agent._generated.models import ModelRequirementAuthority
from eugene_plexus_agent.engines import llama_architectures
from eugene_plexus_agent.engines.base import DiscoveredBinary, Origin
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter

NAMES = [f"arch{i}" for i in range(30)]
SOURCE = "\n".join(
    [
        "static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {",
        *(f'    {{ LLM_ARCH_A{i},  "{name}" }},' for i, name in enumerate(NAMES)),
        '    { LLM_ARCH_UNKNOWN, "(unknown)" },',
        "};",
    ]
)


def test_the_source_list_is_read_without_the_unknown_placeholder() -> None:
    assert llama_architectures.parse(SOURCE) == sorted(NAMES)


@pytest.mark.parametrize(
    ("version", "tag"),
    [("b10948", "b10948"), ("10948", "b10948"), (" b7 ", "b7"), ("unknown", None), (None, None)],
)
def test_a_build_is_named_by_its_upstream_tag(version: str | None, tag: str | None) -> None:
    assert llama_architectures.build_tag(version) == tag


def test_the_shipped_list_is_a_real_one() -> None:
    shipped = llama_architectures.shipped()
    assert llama_architectures.build_tag(shipped.tag) == shipped.tag
    assert len(shipped.names) >= llama_architectures.MIN_NAMES
    assert {"llama", "qwen3", "gemma3"} <= set(shipped.names)
    assert list(shipped.names) == sorted(set(shipped.names))


def test_a_builds_list_is_read_once_and_kept(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(tag: str) -> str:
        calls.append(tag)
        return SOURCE

    lists = llama_architectures.InstalledLists(tmp_path, fetch=fetch, background=False)
    got = lists.get("b123")
    assert got is not None and list(got.names) == sorted(NAMES)
    again = llama_architectures.InstalledLists(tmp_path, fetch=fetch, background=False)
    assert again.get("b123") == got
    assert calls == ["b123"]
    assert json.loads((tmp_path / "b123.json").read_text(encoding="utf-8"))["tag"] == "b123"


def test_a_failed_read_is_not_retried_at_once_and_keeps_nothing(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(tag: str) -> str:
        calls.append(tag)
        raise OSError("github did not answer")

    lists = llama_architectures.InstalledLists(tmp_path, fetch=fetch, background=False)
    assert lists.get("b123") is None
    assert lists.get("b123") is None
    assert calls == ["b123"]
    assert not (tmp_path / "b123.json").exists()


def test_a_page_that_is_not_the_list_is_refused(tmp_path: Path) -> None:
    lists = llama_architectures.InstalledLists(
        tmp_path, fetch=lambda tag: "<html>moved</html>", background=False
    )
    assert lists.get("b123") is None
    assert not (tmp_path / "b123.json").exists()


@pytest.fixture
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LlamaCppAdapter:
    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_ENGINE_ROOT", str(tmp_path))
    return LlamaCppAdapter()


def _found(version: str | None) -> DiscoveredBinary:
    return DiscoveredBinary(path=Path("llama-server"), origin=Origin.managed, version=version)


def test_without_a_build_the_shipped_list_runs_and_the_rest_may(adapter: LlamaCppAdapter) -> None:
    named, rest = adapter.accepts_for(None)
    assert named.architectures == list(llama_architectures.shipped().names)
    assert named.authority is None
    assert rest.architectures is None
    assert rest.authority == ModelRequirementAuthority.engine


def test_an_installed_builds_own_list_replaces_the_shipped_one(
    adapter: LlamaCppAdapter, tmp_path: Path
) -> None:
    kept = tmp_path / "llama_cpp" / "architectures"
    kept.mkdir(parents=True)
    (kept / "b123.json").write_text(json.dumps({"tag": "b123", "architectures": NAMES}))
    (only,) = adapter.accepts_for(_found("b123"))
    assert only.architectures == NAMES
    assert only.authority is None


def test_a_build_whose_list_is_not_in_hand_yet_says_may(
    adapter: LlamaCppAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        llama_architectures, "_fetch_source", lambda tag: (_ for _ in ()).throw(OSError("offline"))
    )
    assert adapter.accepts_for(_found("b999")) == adapter.accepts
    assert adapter.accepts_for(_found("custom-fork")) == adapter.accepts


def test_the_engine_list_declares_the_installed_builds_own_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring: `/v1/engines` asks the adapter about the build it found."""
    from eugene_plexus_agent.runtimes import describe_engines

    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_ENGINE_ROOT", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    build = tmp_path / "llama_cpp" / "b123"
    build.mkdir(parents=True)
    (build / "llama-server").write_bytes(b"")
    (build / "install.json").write_text(
        json.dumps({"version": "b123", "variant": "x", "binary": "llama-server"})
    )
    kept = tmp_path / "llama_cpp" / "architectures"
    kept.mkdir()
    (kept / "b123.json").write_text(json.dumps({"tag": "b123", "architectures": NAMES}))
    (llama,) = [e for e in describe_engines() if e.engine.value == "llama_cpp"]
    assert llama.available and llama.version == "b123"
    assert [r.architectures for r in llama.accepts or []] == [NAMES]
