"""Row 18: plan inspection only; no engine or supervisor starts."""

import logging
from pathlib import Path

import pytest

from eugene_plexus_agent._generated.models import EngineKind, Origin, RuntimeSpec
from eugene_plexus_agent.engines import MlxAdapter, VllmAdapter
from eugene_plexus_agent.engines.base import DiscoveredBinary
from eugene_plexus_agent.runtimes import _RuntimePlanner


@pytest.mark.parametrize("adapter", [VllmAdapter(), MlxAdapter()])
def test_remote_code_is_off_in_the_schema_and_argv(adapter: VllmAdapter | MlxAdapter) -> None:
    field = next(field for field in adapter.flag_schema().fields if field.key == "trustRemoteCode")
    assert field.default is False
    binary = DiscoveredBinary(path=Path("/fake/engine"), origin=Origin.configured)
    for flags in ({}, {"trustRemoteCode": False}, {"trustRemoteCode": True}):
        spec = RuntimeSpec(name="test", engine=adapter.kind, modelPath="/fake/model", flags=flags)
        argv = adapter.build_argv(spec, binary, 18918)
        assert ("--trust-remote-code" in argv) is (flags.get("trustRemoteCode") is True)


@pytest.mark.parametrize("override", [False, True])
def test_vllm_cache_location_survives_new_launch_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: bool
) -> None:
    """vLLM owns the disk cache; Eugene must not replace it per launch.

    Upstream: docs.vllm.ai/en/stable/design/torch_compile/.
    This checks launch environment persistence, not GPU cache-hit performance.
    """
    binary = DiscoveredBinary(path=tmp_path / "vllm", origin=Origin.configured)
    monkeypatch.setattr(VllmAdapter, "discover", lambda *args, **kwargs: binary)
    monkeypatch.setattr("eugene_plexus_agent.engines.vllm._is_wsl", lambda: False)
    monkeypatch.setattr("eugene_plexus_agent.engines.vllm._cuda_toolkit_present", lambda: True)
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    env = {"VLLM_CACHE_ROOT": str(tmp_path / "expert-cache")} if override else {}
    spec = RuntimeSpec(
        name="test", engine=EngineKind.vllm, modelPath="/fake/model", port=18918, env=env
    )
    plans = [
        _RuntimePlanner(spec, VllmAdapter(), logging.getLogger("test")).plan() for _ in range(2)
    ]
    for plan in plans:
        assert plan.cwd is None
        assert plan.env["XDG_CACHE_HOME"] == str(tmp_path / "cache")
        assert plan.env.get("VLLM_CACHE_ROOT") == env.get("VLLM_CACHE_ROOT")
        assert "TORCHINDUCTOR_CACHE_DIR" not in plan.env
    cache = Path(plans[0].env.get("VLLM_CACHE_ROOT", tmp_path / "cache" / "vllm"))
    cache.mkdir(parents=True)
    artifact = cache / "compiled-artifact"
    artifact.write_bytes(b"retained")
    _RuntimePlanner(spec, VllmAdapter(), logging.getLogger("test")).plan()
    assert artifact.read_bytes() == b"retained"
