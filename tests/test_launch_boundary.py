"""R7's first boundary: child credentials and executable choices."""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from eugene_plexus_agent import security
from eugene_plexus_agent._generated.models import ComponentEntry, RuntimeSpec
from eugene_plexus_agent.auth_state import AuthState
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter
from eugene_plexus_agent.runtimes import _RuntimePlanner, validate_spec
from eugene_plexus_agent.supervisor import SpawnPlanError, _ComponentPlanner

SECRET_ENV = {
    "EUGENE_PLEXUS_AGENT_MASTER_KEY": "parent-master",
    "EUGENE_PLEXUS_CONTROL_AUTH_SIGNING_KEY": "root-signing",
    "EUGENE_PLEXUS_GATEWAY_SERVICE_TOKEN": "old-service",
    "eugene_plexus_driver_master_key": "lowercase-secret",
    "EUGENE_PLEXUS_CONTROL_PASSPHRASE_FILE": "/private/root-password",
}


def _ambient(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in SECRET_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("HTTP_PROXY", "http://operator-proxy:8000")


def _component(kind: str, env: dict[str, str] | None = None) -> ComponentEntry:
    return ComponentEntry.model_validate(
        {
            "name": "test",
            "kind": kind,
            "url": "http://127.0.0.1:8181",
            "spawn": {"configFile": "config.yaml", "env": env or {}},
        }
    )


@pytest.mark.parametrize(
    "kind,prefix",
    [
        ("gateway", "GATEWAY"),
        ("library", "LIBRARY"),
        ("inference-driver", "DRIVER"),
        ("control", "CONTROL"),
    ],
)
def test_component_receives_only_its_deliberate_credentials(kind, prefix, monkeypatch):
    _ambient(monkeypatch)
    auth = AuthState(signing_key=security.generate_signing_key(), master_key=b"m" * 32)
    plan = _ComponentPlanner(_component(kind), logging.getLogger("test"), auth).plan()
    assert plan is not None
    for key, value in SECRET_ENV.items():
        if kind == "control" and key == "EUGENE_PLEXUS_CONTROL_PASSPHRASE_FILE":
            assert plan.env[key] == value  # the root's explicit auto-unlock bootstrap
        else:
            assert {k.upper(): v for k, v in plan.env.items()}.get(key.upper()) != value
    if kind in {"gateway", "control"}:
        assert f"EUGENE_PLEXUS_{prefix}_MASTER_KEY" not in plan.env
    else:
        assert base64.b64decode(plan.env[f"EUGENE_PLEXUS_{prefix}_MASTER_KEY"]) == b"m" * 32
    assert plan.env["HTTP_PROXY"] == "http://operator-proxy:8000"


def _runtime(binary: Path, **values) -> RuntimeSpec:
    return RuntimeSpec.model_validate(
        {
            "name": "test",
            "engine": "llama_cpp",
            "modelPath": "/models/m.gguf",
            "port": 8190,
            "binary": str(binary),
            **values,
        }
    )


def test_engine_inherits_no_plexus_credentials(tmp_path, monkeypatch):
    _ambient(monkeypatch)
    executable = tmp_path / "llama-server"
    executable.touch()
    planner = _RuntimePlanner(
        _runtime(executable),
        LlamaCppAdapter(),
        logging.getLogger("test"),
        get_config={"engineBinaryRoots": [str(tmp_path)]}.get,
    )
    plan = planner.plan()
    assert not any(key.upper().startswith("EUGENE_PLEXUS_") for key in plan.env)
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "1"
    assert plan.env["HTTP_PROXY"] == "http://operator-proxy:8000"


@pytest.mark.parametrize(
    "key", ["EUGENE_PLEXUS_DRIVER_MASTER_KEY", "eugene_plexus_driver_auth_signing_key"]
)
def test_component_override_cannot_replace_credential_wiring(key):
    with pytest.raises(SpawnPlanError, match="reserved"):
        _ComponentPlanner(
            _component("inference-driver", {key: "injected"}), logging.getLogger("test")
        ).plan()


def test_runtime_env_cannot_reintroduce_plexus_credentials(tmp_path):
    executable = tmp_path / "llama-server"
    executable.touch()
    spec = _runtime(executable, env={"EUGENE_PLEXUS_AGENT_MASTER_KEY": "injected"})
    with pytest.raises(SpawnPlanError, match="reserved"):
        _RuntimePlanner(
            spec,
            LlamaCppAdapter(),
            logging.getLogger("test"),
            get_config={"allowUnrestrictedEngineLaunch": True}.get,
        ).plan()


def test_unapproved_binary_is_refused_before_any_version_probe(tmp_path, monkeypatch):
    executable = tmp_path / "arbitrary-program"
    executable.touch()
    probes = []
    monkeypatch.setattr(LlamaCppAdapter, "probe_version", lambda *args: probes.append("executed"))
    with pytest.raises(SpawnPlanError, match="engineBinaryRoots"):
        _RuntimePlanner(_runtime(executable), LlamaCppAdapter(), logging.getLogger("test")).plan()
    assert probes == [], "checking the version already executes the untrusted program"


def test_raw_arguments_need_the_explicit_override(tmp_path):
    executable = tmp_path / "llama-server"
    executable.touch()
    spec = _runtime(executable, extraArgs=["--some-upstream-option", "value"])
    config = {"engineBinaryRoots": [str(tmp_path)]}
    planner = _RuntimePlanner(
        spec, LlamaCppAdapter(), logging.getLogger("test"), get_config=config.get
    )
    with pytest.raises(SpawnPlanError, match="allowUnrestrictedEngineLaunch"):
        planner.plan()
    config["allowUnrestrictedEngineLaunch"] = True
    assert planner.plan().argv[-2:] == spec.extraArgs


def test_binary_root_prefix_is_not_containment(tmp_path):
    root = tmp_path / "trusted"
    other = tmp_path / "trusted-other"
    other.mkdir()
    executable = other / "llama-server"
    executable.touch()
    with pytest.raises(SpawnPlanError, match="engineBinaryRoots"):
        _RuntimePlanner(
            _runtime(executable),
            LlamaCppAdapter(),
            logging.getLogger("test"),
            get_config={"engineBinaryRoots": [str(root)]}.get,
        ).plan()


@pytest.mark.parametrize("probe", ["version", "help", "interpreter", "host", "devices"])
def test_probes_never_inherit_plexus_credentials(probe, monkeypatch, tmp_path):
    from eugene_plexus_agent.engines import devices, host, llama_cpp, vllm

    _ambient(monkeypatch)
    seen = []

    def run(argv, **kwargs):
        seen.append(kwargs.get("env"))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(host.shutil, "which", lambda name: str(tmp_path / name))
    binary = tmp_path / "llama-server"
    binary.touch()
    if probe == "version":
        LlamaCppAdapter().probe_version(binary)
    elif probe == "help":
        llama_cpp._supported_long_flags(binary)
    elif probe == "interpreter":
        vllm._probe_interpreter("python")
    elif probe == "host":
        host._run(["vendor-tool"])
    else:
        devices._run(["vendor-tool"])
    assert len(seen) == 1
    assert seen[0] is not None, "a missing env inherits the agent's credentials"
    assert not any(key.upper().startswith("EUGENE_PLEXUS_") for key in seen[0])


def test_trusted_directory_symlink_cannot_escape(tmp_path):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "program").touch()
    link = trusted / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this host does not permit creating directory symlinks")
    reason = validate_spec(_runtime(link / "program"), {"engineBinaryRoots": [str(trusted)]}.get)
    assert reason is not None and "engineBinaryRoots" in reason


def test_managed_build_and_configured_engine_remain_allowed(tmp_path, monkeypatch):
    from eugene_plexus_agent.engines.vllm import VllmAdapter

    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_ENGINE_ROOT", str(tmp_path))
    managed = tmp_path / "llama_cpp" / "b123" / "llama-server"
    assert validate_spec(_runtime(managed)) is None
    vllm = tmp_path / "custom" / "vllm"
    spec = _runtime(vllm).model_copy(update={"engine": VllmAdapter.kind})
    assert validate_spec(spec, {"vllmBinary": str(vllm)}.get) is None


def test_only_the_actual_path_engine_is_implicitly_trusted(tmp_path, monkeypatch):
    engine = tmp_path / "llama-server"
    monkeypatch.setattr("shutil.which", lambda name: str(engine))
    assert validate_spec(_runtime(engine)) is None
    assert validate_spec(_runtime(tmp_path / "another-program")) is not None


def test_policy_is_rechecked_when_config_changes(tmp_path):
    executable = tmp_path / "llama-server"
    executable.touch()
    config = {"engineBinaryRoots": [str(tmp_path)]}
    spec = _runtime(executable)
    assert validate_spec(spec, config.get) is None
    planner = _RuntimePlanner(spec, LlamaCppAdapter(), logging.getLogger("test"), config.get)
    config["engineBinaryRoots"] = []
    with pytest.raises(SpawnPlanError, match="engineBinaryRoots"):
        planner.plan()


def test_config_schema_exposes_safe_defaults_and_persists_trusted_roots(authed_client, tmp_path):
    fields = {
        field["key"]: field for field in authed_client.get("/v1/config/schema").json()["fields"]
    }
    assert fields["engineBinaryRoots"]["valueType"] == "path_list"
    assert fields["engineBinaryRoots"]["default"] == []
    assert fields["allowUnrestrictedEngineLaunch"]["valueType"] == "boolean"
    assert fields["allowUnrestrictedEngineLaunch"]["default"] is False
    roots = [str(tmp_path)]
    assert authed_client.patch("/v1/config", json={"engineBinaryRoots": roots}).status_code == 200
    from eugene_plexus_agent.state import AgentState

    saved = AgentState(authed_client.app.state.agent_state.path)
    saved.load()
    assert saved.get_config("engineBinaryRoots") == roots


@pytest.mark.parametrize(
    "values",
    [
        {"engineBinaryRoots": "not-a-list"},
        {"engineBinaryRoots": [""]},
        {"engineBinaryRoots": [None]},
        {"allowUnrestrictedEngineLaunch": "true"},
    ],
)
def test_invalid_launch_policy_config_is_rejected(authed_client, values):
    result = authed_client.patch("/v1/config", json=values).json()
    assert result["applied"] == []
    assert {item["key"] for item in result["rejected"]} == set(values)


@pytest.mark.parametrize("operation", ["create", "update", "admission"])
def test_runtime_routes_refuse_untrusted_binary_before_side_effects(
    authed_client, tmp_path, operation
):
    body = _runtime(tmp_path / "arbitrary-program").model_dump(mode="json", exclude_none=True)
    if operation == "update":
        response = authed_client.patch("/v1/runtimes/test", json=body)
    else:
        path = "/v1/runtimes/admission" if operation == "admission" else "/v1/runtimes"
        response = authed_client.post(path, json=body)
    assert response.status_code == 400, response.text
    assert "engineBinaryRoots" in response.text
    assert authed_client.app.state.agent_state.get_runtime_spec("test") is None


def test_admission_uses_the_configured_expert_override(authed_client):
    body = {
        "name": "test",
        "engine": "llama_cpp",
        "modelPath": "/models/m.gguf",
        "extraArgs": ["--expert-option"],
    }
    assert authed_client.post("/v1/runtimes/admission", json=body).status_code == 400
    patched = authed_client.patch("/v1/config", json={"allowUnrestrictedEngineLaunch": True}).json()
    assert patched["rejected"] == []
    response = authed_client.post("/v1/runtimes/admission", json=body)
    assert response.status_code == 200, response.text


def test_real_engine_child_receives_no_plexus_credentials(monkeypatch, tmp_path):
    from eugene_plexus_agent._generated.models import Origin
    from eugene_plexus_agent.engines.base import DiscoveredBinary

    _ambient(monkeypatch)
    # A harmless substitute engine prints variable NAMES only. It has no network,
    # config access, or model dependency, and runs inside this test's directory.
    code = "import json, os; print(json.dumps(sorted(os.environ)))"
    adapter = LlamaCppAdapter()
    monkeypatch.setattr(
        adapter,
        "resolve_binary",
        lambda *a, **kw: DiscoveredBinary(
            path=Path(sys.executable), origin=Origin.configured, version="test"
        ),
    )
    monkeypatch.setattr(adapter, "build_argv", lambda *a: [sys.executable, "-c", code])
    plan = _RuntimePlanner(
        _runtime(Path(sys.executable), workingDirectory=str(tmp_path)),
        adapter,
        logging.getLogger("test"),
        {"engineBinaryRoots": [str(Path(sys.executable).resolve().parent)]}.get,
    ).plan()
    result = subprocess.run(
        plan.argv,
        env=plan.env,
        cwd=plan.cwd,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    names = json.loads(result.stdout)
    assert not any(key.upper().startswith("EUGENE_PLEXUS_") for key in names)
    assert "CUDA_VISIBLE_DEVICES" in names
    assert "HTTP_PROXY" in names
