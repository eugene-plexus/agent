"""A Kev started with `KEV_API_KEY` is still probed, and its companion can still ask.

**The finding (upstream drift audit, 2026-10-03; read in source).** At
`kev-1.0` (github.com/jaredpalmer/kev, 2026-10-01) `kev/serve.py` reads
`KEV_API_KEY` and, when it is set, refuses every `/v1/*` request without
`Authorization: Bearer <key>` (401). The adapter's readiness probe is
`GET /v1/models` with no header, so a Kev runtime given the variable in
its `env` -- or inheriting it from the agent's own environment -- would
read as not answering forever, and its companion driver would be refused
on every decision. The pinned commit (`1c35199`) has no key at all, so
nothing changes for it.

The key in effect is the one the spawn gives the child: the runtime's
`env` if it names the variable (an empty value is Kev's "unset"), else
the agent's own environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from eugene_plexus_agent import _http, companions
from eugene_plexus_agent._generated.models import EngineKind, RuntimeSpec
from eugene_plexus_agent.engines.base import NotAnswering, Ready
from eugene_plexus_agent.engines.kev import KevAdapter
from eugene_plexus_agent.state import AgentState


def _spec(**overrides: Any) -> RuntimeSpec:
    body: dict[str, Any] = {
        "name": "tickets",
        "engine": EngineKind.kev,
        "modelPath": "/home/troy/checkpoints/kev-0.8b",
    }
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


def _kev_server(key: str | None, seen: list[str | None]) -> None:
    """kev-1.0's middleware: with a key, `/v1/*` needs the bearer."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent = request.headers.get("authorization")
        seen.append(sent)
        if key and sent != f"Bearer {key}":
            return httpx.Response(
                401,
                json={
                    "detail": "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>"
                },
            )
        return httpx.Response(200, json={"models": [{"name": "kev-latest"}]})

    _http.set_shared_client(
        "engine-probe", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


# --- which key is in effect --------------------------------------------------


def test_the_runtimes_own_key_is_sent() -> None:
    headers = KevAdapter().readiness_headers(_spec(env={"KEV_API_KEY": "s3cret"}))
    assert headers == {"Authorization": "Bearer s3cret"}


def test_a_key_the_agent_was_started_with_reaches_kev_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spawn inherits the agent's environment, so Kev sees this key."""
    monkeypatch.setenv("KEV_API_KEY", "inherited")
    assert KevAdapter().readiness_headers(_spec()) == {"Authorization": "Bearer inherited"}


def test_the_runtimes_env_beats_the_inherited_one_even_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KEV_API_KEY", "inherited")
    assert KevAdapter().readiness_headers(_spec(env={"KEV_API_KEY": "mine"})) == {
        "Authorization": "Bearer mine"
    }
    assert KevAdapter().readiness_headers(_spec(env={"KEV_API_KEY": ""})) == {}


def test_no_key_sends_no_header() -> None:
    assert KevAdapter().readiness_headers(_spec()) == {}


# --- the probe ---------------------------------------------------------------


async def test_a_keyed_kev_is_ready_when_probed_with_its_key() -> None:
    seen: list[str | None] = []
    _kev_server("s3cret", seen)
    adapter = KevAdapter()
    spec = _spec(env={"KEV_API_KEY": "s3cret"})
    outcome = await adapter.probe_readiness(
        "http://127.0.0.1:8409", headers=adapter.readiness_headers(spec)
    )
    assert isinstance(outcome, Ready)
    assert seen == ["Bearer s3cret"]


async def test_a_refused_probe_names_the_key() -> None:
    """The failure says what was observed and where the fix is."""
    _kev_server("s3cret", [])
    outcome = await KevAdapter().probe_readiness("http://127.0.0.1:8409")
    assert isinstance(outcome, NotAnswering)
    assert outcome.reached is True
    assert "401" in (outcome.detail or "")
    assert "KEV_API_KEY" in (outcome.detail or "")


# --- the companion -------------------------------------------------------------


def _companion_config(state: AgentState, spec: RuntimeSpec) -> dict[str, Any]:
    path = companions.companion_config_path(state, companions.companion_name(spec.name))
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@pytest.mark.anyio
async def test_the_companion_is_given_the_key(tmp_path: Path) -> None:
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    spec = state.add_runtime(_spec(env={"KEV_API_KEY": "s3cret"}))
    await companions.ensure_companion(state, None, spec)
    assert _companion_config(state, spec)["apiKey"] == "s3cret"


@pytest.mark.anyio
async def test_a_key_already_on_the_companion_is_left_alone(tmp_path: Path) -> None:
    """The operator's, or one the driver has since sealed: never ours to
    replace. Seeding fills an empty field and nothing else."""
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    spec = state.add_runtime(_spec(env={"KEV_API_KEY": "s3cret"}))
    path = companions.companion_config_path(state, companions.companion_name(spec.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = {"v": 1, "nonce": "abc", "ciphertext": "def"}
    path.write_text(yaml.safe_dump({"apiKey": sealed}), encoding="utf-8")
    await companions.ensure_companion(state, None, spec)
    assert _companion_config(state, spec)["apiKey"] == sealed


@pytest.mark.anyio
async def test_no_key_writes_no_key(tmp_path: Path) -> None:
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    spec = state.add_runtime(_spec())
    await companions.ensure_companion(state, None, spec)
    assert "apiKey" not in _companion_config(state, spec)


@pytest.mark.anyio
async def test_a_chat_engines_companion_gets_no_key(tmp_path: Path) -> None:
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    spec = state.add_runtime(
        RuntimeSpec.model_validate(
            {
                "name": "chat",
                "engine": "llama_cpp",
                "modelPath": "/m.gguf",
                "env": {"KEV_API_KEY": "s3cret"},
            }
        )
    )
    await companions.ensure_companion(state, None, spec)
    assert "apiKey" not in _companion_config(state, spec)
