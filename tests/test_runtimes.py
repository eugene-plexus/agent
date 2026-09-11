"""The /v1/engines and /v1/runtimes surfaces, plus runtime persistence.

Wire shape and lifecycle wiring. The adapter's own behaviour (argv,
readiness, flags) is in test_engines.py; the supervision loop is in
test_supervisor.py.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import security
from eugene_plexus_agent._generated.models import RuntimeSpec, RuntimeStatus
from eugene_plexus_agent.engines import LlamaCppAdapter, VllmAdapter
from eugene_plexus_agent.engines.base import Loading, Ready
from eugene_plexus_agent.runtimes import RuntimeSupervisor, _RuntimePlanner, describe_engines
from eugene_plexus_agent.state import AgentState
from eugene_plexus_agent.supervisor import ProcessState, SpawnPlan, SpawnPlanError

from .conftest import StubRuntimeSupervisor


def _runtime(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "qwen3-30b",
        "engine": "llama_cpp",
        "modelPath": "/models/Qwen3-30B-A3B-Q4_K_M.gguf",
    }
    body.update(overrides)
    return body


def _service_token(client: TestClient) -> str:
    signing_key = client.app.state.auth_state.signing_key  # type: ignore[attr-defined]
    return security.issue_service_token(signing_key=signing_key, kind="gateway")


# --------------------------------------------------------------------------- #
# /v1/engines
# --------------------------------------------------------------------------- #


def test_engines_lists_every_adapter_with_its_flag_schema(authed_client: TestClient) -> None:
    """The UI reads this to render the add-a-runtime form, so the flag
    schema has to come back whether or not a binary was found — an
    operator with no llama.cpp installed still needs to see the form."""
    response = authed_client.get("/v1/engines")
    assert response.status_code == 200
    engines = response.json()["engines"]
    assert [e["engine"] for e in engines] == ["llama_cpp", "vllm"]

    for engine in engines:
        assert engine["flagSchema"]["component"] == f"engine:{engine['engine']}"
        assert engine["flagSchema"]["fields"], "the form needs fields"
        # available depends on the dev machine; the contract is that a
        # false answer explains itself.
        if not engine["available"]:
            assert engine["error"]


def test_engines_declare_which_model_formats_they_load(authed_client: TestClient) -> None:
    """The engine half of a join the UI performs: the library reports
    what format each model *is*, this reports what each engine can
    *load*. Without it a safetensors model gets a launch button that
    fails instead of one greyed out with a reason.

    `llama_cpp` is GGUF-only, so at M2 a safetensors model has nowhere
    to run at all — vLLM at M4 is what changes that answer, with no
    library change.
    """
    engines = authed_client.get("/v1/engines").json()["engines"]
    by_kind = {e["engine"]: e for e in engines}

    assert by_kind["llama_cpp"]["modelFormats"] == ["gguf"]
    # And now the safetensors half of the join has an engine. GGUF stays
    # off vLLM's list on purpose — see the adapter.
    assert by_kind["vllm"]["modelFormats"] == ["safetensors"]


def test_model_formats_do_not_depend_on_availability(authed_client: TestClient) -> None:
    """A property of the engine, not of this host. An operator with no
    binary installed still needs to know what it would be able to
    load."""
    for engine in authed_client.get("/v1/engines").json()["engines"]:
        assert engine["modelFormats"], f"{engine['engine']} declared no formats"


def test_vllm_is_a_manual_engine_whose_refusal_names_the_command(
    authed_client: TestClient,
) -> None:
    """Three panels from one shape: install it, here is why we cannot,
    and here is how you do it yourself. vLLM is the third, on every
    host, by decision — `policy: manual` says so and `manualInstall`
    carries upstream's command for the detected host (or, on this
    Windows dev box, the honest note that the way in is WSL)."""
    engines = authed_client.get("/v1/engines").json()["engines"]
    vllm = next(e for e in engines if e["engine"] == "vllm")

    acquisition = vllm["acquisition"]
    assert acquisition["policy"] == "manual"
    assert acquisition["installable"] is False
    assert acquisition["manualInstall"]["docsUrl"].startswith("https://docs.vllm.ai/")
    # `reason` restates the way forward in prose for a client that reads
    # only that field.
    assert "installed by the operator" in acquisition["reason"]

    llama = next(e for e in engines if e["engine"] == "llama_cpp")
    assert llama["acquisition"]["policy"] == "managed"
    assert (
        "manualInstall" not in llama["acquisition"] or llama["acquisition"]["manualInstall"] is None
    )


def test_installing_a_manual_engine_always_422s_with_the_command(
    authed_client: TestClient,
) -> None:
    """Not 404 (the engine exists) and not 500 (nothing failed): we do
    not install this engine, anywhere, and the detail says what to do
    instead. A UI should read `policy` and never reach here."""
    response = authed_client.post("/v1/engines/vllm/install")
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]["detail"]
    assert "installed by the operator" in detail
    assert "https://docs.vllm.ai/" in detail


def test_unavailable_vllm_names_vllm_binary_as_the_fix(
    authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    vllm = next(
        e for e in authed_client.get("/v1/engines").json()["engines"] if e["engine"] == "vllm"
    )
    assert vllm["available"] is False
    assert "vllmBinary" in vllm["error"]
    # Not the install endpoint, which 422s for this engine.
    assert "/install" not in vllm["error"]


def test_vllm_binary_config_makes_the_engine_discoverable(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The M4 discovery rung: an install-wide path in the agent's own
    config, so an operator with a good venv is not `available: false`
    unless they put it on PATH or repeat the path on every runtime.
    Set through the standard config trio, no engine-specific endpoint."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    exe = tmp_path / "venv" / "bin" / "vllm"
    exe.parent.mkdir(parents=True)
    exe.write_text(f"#!{sys.executable}\n", encoding="utf-8")

    patched = authed_client.patch("/v1/config", json={"vllmBinary": str(exe)})
    assert patched.status_code == 200
    assert "vllmBinary" in patched.json()["applied"], patched.text

    vllm = next(
        e for e in authed_client.get("/v1/engines").json()["engines"] if e["engine"] == "vllm"
    )
    assert vllm["available"] is True
    assert vllm["origin"] == "configured"
    assert vllm["binaryPath"] == str(exe)
    # The environment block rides along: interpreter from the shebang,
    # versions from that interpreter's own metadata. This venv holds no
    # vllm, and the descriptor says so rather than inventing one.
    assert vllm["python"]["interpreter"]
    assert vllm["python"]["pythonVersion"]
    assert vllm["python"].get("packageVersion") is None


def test_missing_vllm_binary_path_is_reported_not_replaced(
    authed_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/vllm")
    authed_client.patch("/v1/config", json={"vllmBinary": str(tmp_path / "nope" / "vllm")})
    vllm = next(
        e for e in authed_client.get("/v1/engines").json()["engines"] if e["engine"] == "vllm"
    )
    assert vllm["available"] is False
    assert "does not exist" in vllm["error"]
    assert str(tmp_path / "nope" / "vllm") in vllm["error"]


def test_describe_engines_reads_the_configured_path_through_the_getter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config lives in AgentState and adapters are stateless singletons;
    the getter is where they meet, and an empty value means unset."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    exe = tmp_path / "vllm"
    exe.write_text(f"#!{sys.executable}\n", encoding="utf-8")

    with_path = {e.engine.value: e for e in describe_engines(lambda k: str(exe))}
    assert with_path["vllm"].available is True

    blank = {e.engine.value: e for e in describe_engines(lambda k: "   ")}
    assert blank["vllm"].available is False

    none = {e.engine.value: e for e in describe_engines(None)}
    assert none["vllm"].available is False


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #


def test_runtimes_starts_empty(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/runtimes")
    assert response.status_code == 200
    assert response.json() == {"runtimes": []}


def test_create_assigns_a_port_and_derives_the_alias(authed_client: TestClient) -> None:
    response = authed_client.post("/v1/runtimes", json=_runtime())
    assert response.status_code == 201, response.text
    body = response.json()

    # Ports are assigned so nobody hands them out by hand with N runtimes.
    assert body["port"] >= 8090
    assert body["url"].rstrip("/") == f"http://127.0.0.1:{body['port']}"
    # Plainly-named files mean the obvious name is already the right one.
    assert body["modelAlias"] == "Qwen3-30B-A3B-Q4_K_M"
    assert body["status"] == RuntimeStatus.starting.value


def test_created_runtime_is_handed_to_the_supervisor(
    authed_client: TestClient, stub_runtime_supervisor: StubRuntimeSupervisor
) -> None:
    authed_client.post("/v1/runtimes", json=_runtime())
    assert ("add_and_start", "qwen3-30b") in stub_runtime_supervisor.calls


def test_assigned_ports_do_not_collide(authed_client: TestClient) -> None:
    first = authed_client.post("/v1/runtimes", json=_runtime(name="a")).json()
    second = authed_client.post("/v1/runtimes", json=_runtime(name="b")).json()
    assert first["port"] != second["port"]


def test_explicit_port_is_honoured_and_a_clash_is_rejected(authed_client: TestClient) -> None:
    assert (
        authed_client.post("/v1/runtimes", json=_runtime(name="a", port=8123)).json()["port"]
        == 8123
    )

    clash = authed_client.post("/v1/runtimes", json=_runtime(name="b", port=8123))
    assert clash.status_code == 400
    assert "already claimed" in clash.json()["detail"]["detail"]


def test_duplicate_name_is_a_conflict(authed_client: TestClient) -> None:
    authed_client.post("/v1/runtimes", json=_runtime())
    again = authed_client.post("/v1/runtimes", json=_runtime())
    assert again.status_code == 409


def test_unknown_flag_is_rejected_with_the_known_list(authed_client: TestClient) -> None:
    """A typo'd flag that silently vanishes is worse than a 400: the
    engine starts, behaves differently from what was asked for, and
    nothing says why. The message has to be actionable."""
    response = authed_client.post("/v1/runtimes", json=_runtime(flags={"ctxSize": 8192}))
    assert response.status_code == 400
    detail = response.json()["detail"]["detail"]
    assert "ctxSize" in detail
    assert "contextSize" in detail, "should name the flags that DO exist"
    assert "extraArgs" in detail, "should point at the escape hatch"


def test_a_rejected_spec_is_not_persisted(authed_client: TestClient) -> None:
    authed_client.post("/v1/runtimes", json=_runtime(flags={"nonsense": 1}))
    assert authed_client.get("/v1/runtimes").json()["runtimes"] == []


def test_get_and_404(authed_client: TestClient) -> None:
    authed_client.post("/v1/runtimes", json=_runtime())
    assert authed_client.get("/v1/runtimes/qwen3-30b").status_code == 200
    missing = authed_client.get("/v1/runtimes/nope")
    assert missing.status_code == 404
    assert missing.json()["detail"]["title"] == "Runtime not found"


def test_patch_restarts_the_engine(
    authed_client: TestClient, stub_runtime_supervisor: StubRuntimeSupervisor
) -> None:
    """Every RuntimeSpec field is baked into the argv at spawn, so there
    is no way to re-flag a live llama-server — a change must restart."""
    authed_client.post("/v1/runtimes", json=_runtime())
    stub_runtime_supervisor.calls.clear()

    response = authed_client.patch(
        "/v1/runtimes/qwen3-30b", json=_runtime(flags={"contextSize": 16384})
    )
    assert response.status_code == 200
    assert response.json()["flags"] == {"contextSize": 16384}
    assert ("remove_and_stop", "qwen3-30b") in stub_runtime_supervisor.calls
    assert ("add_and_start", "qwen3-30b") in stub_runtime_supervisor.calls


def test_patch_can_rename(authed_client: TestClient) -> None:
    authed_client.post("/v1/runtimes", json=_runtime())
    response = authed_client.patch("/v1/runtimes/qwen3-30b", json=_runtime(name="qwen"))
    assert response.status_code == 200
    assert response.json()["name"] == "qwen"
    assert authed_client.get("/v1/runtimes/qwen3-30b").status_code == 404


def test_delete_removes_the_declaration(
    authed_client: TestClient, stub_runtime_supervisor: StubRuntimeSupervisor
) -> None:
    authed_client.post("/v1/runtimes", json=_runtime())
    assert authed_client.delete("/v1/runtimes/qwen3-30b").status_code == 204
    assert authed_client.get("/v1/runtimes").json()["runtimes"] == []
    assert ("remove_and_stop", "qwen3-30b") in stub_runtime_supervisor.calls
    assert authed_client.delete("/v1/runtimes/qwen3-30b").status_code == 404


# --------------------------------------------------------------------------- #
# start / stop / restart
# --------------------------------------------------------------------------- #


def test_stop_keeps_the_declaration(
    authed_client: TestClient, stub_runtime_supervisor: StubRuntimeSupervisor
) -> None:
    """Stop is distinct from delete because an engine holds GPU memory:
    an operator who wants the VRAM back needs a stop that is neither a
    delete nor a crash."""
    authed_client.post("/v1/runtimes", json=_runtime())
    response = authed_client.post("/v1/runtimes/qwen3-30b/stop")
    assert response.status_code == 202
    assert ("stop_one", "qwen3-30b") in stub_runtime_supervisor.calls

    still_there = authed_client.get("/v1/runtimes/qwen3-30b")
    assert still_there.status_code == 200
    assert still_there.json()["status"] == RuntimeStatus.stopped.value


def test_start_brings_a_stopped_runtime_back(authed_client: TestClient) -> None:
    authed_client.post("/v1/runtimes", json=_runtime())
    authed_client.post("/v1/runtimes/qwen3-30b/stop")

    response = authed_client.post("/v1/runtimes/qwen3-30b/start")
    assert response.status_code == 202
    assert response.json()["scheduled"] is True
    assert authed_client.get("/v1/runtimes/qwen3-30b").json()["status"] == (
        RuntimeStatus.starting.value
    )


def test_start_is_idempotent(authed_client: TestClient) -> None:
    """The UI's Start button must not error on an already-running engine."""
    authed_client.post("/v1/runtimes", json=_runtime())
    response = authed_client.post("/v1/runtimes/qwen3-30b/start")
    assert response.status_code == 202
    # `scheduled: false` is how an idempotent second press is
    # distinguishable from the first without being an error.
    assert response.json()["scheduled"] is False
    assert "already" in response.json()["message"]


def test_start_overrides_auto_start_false(authed_client: TestClient) -> None:
    """`autoStart: false` means "don't start at boot", not "never start"."""
    created = authed_client.post("/v1/runtimes", json=_runtime(autoStart=False))
    assert created.json()["status"] == RuntimeStatus.stopped.value

    authed_client.post("/v1/runtimes/qwen3-30b/start")
    assert authed_client.get("/v1/runtimes/qwen3-30b").json()["status"] == (
        RuntimeStatus.starting.value
    )


def test_restart_starts_a_runtime_that_was_not_running(
    authed_client: TestClient, stub_runtime_supervisor: StubRuntimeSupervisor
) -> None:
    """A Restart button that no-ops on a stopped or crashed engine is a
    worse answer than just bringing it back."""
    authed_client.post("/v1/runtimes", json=_runtime())
    authed_client.post("/v1/runtimes/qwen3-30b/stop")
    stub_runtime_supervisor.calls.clear()

    response = authed_client.post("/v1/runtimes/qwen3-30b/restart")
    assert response.status_code == 202
    assert ("restart", "qwen3-30b") in stub_runtime_supervisor.calls
    assert ("add_and_start", "qwen3-30b") in stub_runtime_supervisor.calls


def test_lifecycle_actions_404_on_unknown_names(authed_client: TestClient) -> None:
    for action in ("restart", "stop", "start"):
        assert authed_client.post(f"/v1/runtimes/nope/{action}").status_code == 404


# --------------------------------------------------------------------------- #
# auth split
# --------------------------------------------------------------------------- #


def test_service_token_can_read_but_not_mutate(client: TestClient) -> None:
    """Same split as components, for a sharper reason: the gateway needs
    to see what is running, but a leaked service token must not be able
    to declare or delete a process holding a GPU.

    M6 carved one exception: the *gateway's* token may stop and start,
    because the gateway is the component that sees demand. Any other
    service audience still may not — see test_lifecycle_routes.py."""
    client.post("/v1/auth/initialize", json={"passphrase": "correct horse battery staple"})
    headers = {"Authorization": f"Bearer {_service_token(client)}"}

    assert client.get("/v1/runtimes", headers=headers).status_code == 200
    assert client.get("/v1/engines", headers=headers).status_code == 200
    # 401, matching the components surface: `require_operator_session`
    # treats a service-audience token as no operator credential at all
    # rather than as an authenticated principal being denied.
    assert client.post("/v1/runtimes", json=_runtime(), headers=headers).status_code == 401
    assert client.delete("/v1/runtimes/x", headers=headers).status_code == 401
    # The gateway's token is authorized for stop — so an unknown runtime
    # is a 404, not a 401.
    assert client.post("/v1/runtimes/x/stop", headers=headers).status_code == 404
    signing_key = client.app.state.auth_state.signing_key  # type: ignore[attr-defined]
    driver = security.issue_service_token(signing_key=signing_key, kind="inference-driver")
    assert (
        client.post(
            "/v1/runtimes/x/stop", headers={"Authorization": f"Bearer {driver}"}
        ).status_code
        == 401
    )


def test_runtimes_require_auth(client: TestClient) -> None:
    client.post("/v1/auth/initialize", json={"passphrase": "correct horse battery staple"})
    assert client.get("/v1/runtimes").status_code == 401


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #


def test_runtimes_round_trip_through_the_yaml_file(
    authed_client: TestClient, settings: Any
) -> None:
    """A runtime must survive a agent restart, and its assigned port
    must survive with it — that port ends up in a driver's config, and a
    value that changed on every boot would be useless there."""
    created = authed_client.post(
        "/v1/runtimes",
        json=_runtime(flags={"contextSize": 8192}, env={"CUDA_VISIBLE_DEVICES": "1"}),
    ).json()

    reloaded = AgentState(settings.config_file)
    reloaded.load()
    specs = reloaded.list_runtime_specs()
    assert [s.name for s in specs] == ["qwen3-30b"]
    assert specs[0].port == created["port"]
    assert specs[0].flags == {"contextSize": 8192}
    assert specs[0].env == {"CUDA_VISIBLE_DEVICES": "1"}


def test_runtimes_and_components_are_separate_collections(
    authed_client: TestClient, settings: Any
) -> None:
    """The whole point of the split: a component is a Eugene Plexus
    process, a runtime is a foreign binary, and neither should appear in
    the other's list."""
    authed_client.post(
        "/v1/components",
        json={
            "name": "gateway",
            "kind": "gateway",
            "url": "http://127.0.0.1:8080",
            "spawn": {"configFile": "/tmp/gw/config.yaml"},
        },
    )
    authed_client.post("/v1/runtimes", json=_runtime())

    components = authed_client.get("/v1/components").json()["components"]
    runtimes = authed_client.get("/v1/runtimes").json()["runtimes"]
    # The runtime is not a component. Its companion *driver* is — a
    # Eugene Plexus process the agent declared beside the engine (M6).
    assert [c["name"] for c in components] == ["gateway", "qwen3-30b-driver"]
    assert [r["name"] for r in runtimes] == ["qwen3-30b"]

    raw = settings.config_file.read_text(encoding="utf-8")
    assert "components:" in raw
    assert "runtimes:" in raw


# --------------------------------------------------------------------------- #
# The runtime planner and the status mapping
#
# This is where the engine layer meets the supervision loop: the planner
# is a SpawnPlanner like any other, and RuntimeStatus is derived from the
# loop's kind-agnostic state plus a readiness observation.
# --------------------------------------------------------------------------- #


def test_planner_builds_a_plan_with_argv_cwd_and_accelerator_env(tmp_path: Any) -> None:
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")

    spec = RuntimeSpec.model_validate(
        {
            "name": "qwen3-30b",
            "engine": "llama_cpp",
            "modelPath": "/models/q.gguf",
            "port": 8090,
            "binary": str(exe),
            "env": {"CUDA_VISIBLE_DEVICES": "1"},
        }
    )
    planner = _RuntimePlanner(spec, LlamaCppAdapter(), logging.getLogger("test"))
    plan = planner.plan()

    assert plan.argv[0] == str(exe)
    # cwd is the binary's directory: prebuilt llama.cpp releases keep
    # their shared libraries there.
    assert plan.cwd == str(exe.parent)
    # Accelerator selection rides in env — this is how a runtime gets
    # pinned to one card, and how two replicas end up on two GPUs.
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "1"
    # An engine has no safe mode to be launched into.
    assert plan.degraded is False
    # No component env leaked in.
    assert not any(k.startswith("EUGENE_PLEXUS_") for k in plan.env)


def test_missing_engine_binary_is_a_spawn_plan_error() -> None:
    """A declared runtime whose engine is missing is a crash, not a
    silently skipped entry — the operator asked for something that
    cannot be delivered."""
    spec = RuntimeSpec.model_validate(
        {
            "name": "qwen3-30b",
            "engine": "llama_cpp",
            "modelPath": "/models/q.gguf",
            "port": 8090,
            "binary": "/nope/llama-server",
        }
    )
    planner = _RuntimePlanner(spec, LlamaCppAdapter(), logging.getLogger("test"))
    with pytest.raises(SpawnPlanError, match="does not exist"):
        planner.plan()


def test_engine_planner_declines_to_recover() -> None:
    """Unlike a component, an engine has no safe mode: a llama-server
    that won't start has nothing to serve and nothing to configure, so
    respawning forever would just churn."""
    spec = RuntimeSpec.model_validate(
        {"name": "x", "engine": "llama_cpp", "modelPath": "/m.gguf", "port": 8090}
    )
    planner = _RuntimePlanner(spec, LlamaCppAdapter(), logging.getLogger("test"))
    assert planner.on_crash_threshold() is False


def test_status_mapping_covers_the_loading_distinction() -> None:
    """`starting` vs `loading` vs `ready` all share one ProcessState —
    the loop only knows the child is alive. The readiness observation is
    what separates them, which is the whole payoff of a per-engine probe.
    """
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))

    class _Alive:
        state = ProcessState.starting

    alive: Any = _Alive()
    assert supervisor._status_for(alive, None) == RuntimeStatus.starting
    assert supervisor._status_for(alive, Loading(detail="loading model")) == RuntimeStatus.loading
    assert supervisor._status_for(alive, Ready()) == RuntimeStatus.ready

    # A runtime with no process at all is stopped, not broken.
    assert supervisor._status_for(None, None) == RuntimeStatus.stopped

    for state, expected in (
        (ProcessState.crashed, RuntimeStatus.crashed),
        (ProcessState.exited, RuntimeStatus.exited),
        # Unreachable for engines (their plan() never returns None), but
        # mapping it to `stopped` would claim the operator asked for it.
        (ProcessState.not_spawnable, RuntimeStatus.crashed),
    ):

        class _InState:
            pass

        proc: Any = _InState()
        proc.state = state
        assert supervisor._status_for(proc, None) == expected


def test_a_silent_load_past_its_budget_is_loading_with_last_error() -> None:
    """The contract names "a readiness probe that never passed" as one of
    `lastError`'s sources. Past vLLM's startup budget the status is still
    `loading` — a live, silent process cannot be anything else — but the
    operator gets the elapsed time on `lastError` instead of a spinner."""
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))
    spec = RuntimeSpec.model_validate(
        {"name": "q", "engine": "vllm", "modelPath": "/models/Qwen3-8B", "port": 8090}
    )

    class _Alive:
        state = ProcessState.starting
        pid = 4242
        last_argv = None
        last_error = None
        last_restart = None

    supervisor._processes["q"] = _Alive()  # type: ignore[assignment]

    supervisor._readiness["q"] = Loading(detail="alive, silent (45s)", past_budget=False)
    within = supervisor.compose(spec)
    assert within.status == RuntimeStatus.loading
    assert within.lastError is None

    supervisor._readiness["q"] = Loading(detail="alive for 612s and still silent", past_budget=True)
    past = supervisor.compose(spec)
    assert past.status == RuntimeStatus.loading
    assert past.lastError == "alive for 612s and still silent"

    # The moment it answers, the flag is gone.
    supervisor._readiness["q"] = Ready()
    assert supervisor.compose(spec).lastError is None


def test_planner_passes_the_configured_binary_to_the_adapter(tmp_path: Path) -> None:
    """The install-wide `vllmBinary` reaches the spawn plan through the
    supervisor's config getter, read at plan time."""
    exe = tmp_path / "vllm"
    exe.write_text(f"#!{sys.executable}\n", encoding="utf-8")
    spec = RuntimeSpec.model_validate(
        {"name": "q", "engine": "vllm", "modelPath": "/models/Qwen3-8B", "port": 8090}
    )
    planner = _RuntimePlanner(spec, VllmAdapter(), logging.getLogger("test"), lambda key: str(exe))
    plan = planner.plan()
    assert plan.argv[:3] == [str(exe), "serve", "/models/Qwen3-8B"]
    # A console script inherits the agent's cwd; only prebuilt llama.cpp
    # needs to run from its own directory.
    assert plan.cwd is None


def test_every_process_state_maps_onto_a_runtime_status() -> None:
    """Exhaustiveness by test: a new ProcessState member must be given a
    RuntimeStatus deliberately, not fall through to `starting`."""
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))
    for state in ProcessState:

        class _InState:
            pass

        proc: Any = _InState()
        proc.state = state
        assert isinstance(supervisor._status_for(proc, None), RuntimeStatus)


def test_auto_start_false_is_declared_but_not_started() -> None:
    """How a rarely-used large model stays configured without holding
    VRAM."""
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))
    spec = RuntimeSpec.model_validate(
        {
            "name": "big",
            "engine": "llama_cpp",
            "modelPath": "/m.gguf",
            "port": 8090,
            "autoStart": False,
        }
    )
    supervisor.add_and_start(spec)
    assert supervisor.is_running("big") is False
    assert supervisor.compose(spec).status == RuntimeStatus.stopped


def test_adapter_env_defaults_lose_to_both_kinds_of_override(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The precedence that makes an injected default safe: ambient env
    first (an operator who exported it in the agent's shell), then the
    adapter's default for keys nobody set, then `RuntimeSpec.env`, which
    wins outright — including winning with a value that will fail, which
    is an expert's prerogative.
    """
    exe = tmp_path / "vllm"
    exe.write_text(f"#!{sys.executable}\n", encoding="utf-8")
    monkeypatch.setattr("eugene_plexus_agent.engines.vllm._is_wsl", lambda: True)
    monkeypatch.setattr("eugene_plexus_agent.engines.vllm._cuda_toolkit_present", lambda: False)

    # 1. Nobody set anything: both defaults land.
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    spec = RuntimeSpec.model_validate(
        {"name": "q", "engine": "vllm", "modelPath": "/models/Qwen3-8B", "port": 8090}
    )
    plan = _RuntimePlanner(
        spec, VllmAdapter(), logging.getLogger("test"), lambda key: str(exe)
    ).plan()
    assert plan is not None
    assert plan.env["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "1"
    assert plan.env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"

    # 2. The runtime's own env wins, even when it is the losing choice.
    spec_override = RuntimeSpec.model_validate(
        {
            "name": "q",
            "engine": "vllm",
            "modelPath": "/models/Qwen3-8B",
            "port": 8090,
            "env": {"VLLM_WSL2_ENABLE_PIN_MEMORY": "0"},
        }
    )
    plan = _RuntimePlanner(
        spec_override, VllmAdapter(), logging.getLogger("test"), lambda key: str(exe)
    ).plan()
    assert plan is not None
    assert plan.env["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "0"

    # 3. An exported variable wins too, so an operator debugging from a
    #    shell is not fighting an invisible default.
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    plan = _RuntimePlanner(
        spec, VllmAdapter(), logging.getLogger("test"), lambda key: str(exe)
    ).plan()
    assert plan is not None
    assert plan.env["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_the_runtime_planner_delegates_explaining_to_its_adapter() -> None:
    """The planner knows nothing about any engine's failure modes, so it
    forwards the tail and nothing else."""
    spec = RuntimeSpec.model_validate(
        {"name": "q", "engine": "vllm", "modelPath": "/models/Qwen3-8B", "port": 8090}
    )
    planner = _RuntimePlanner(spec, VllmAdapter(), logging.getLogger("test"))
    explained = planner.explain_exit(1, "RuntimeError: UVA is not available")
    assert explained is not None
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY" in explained
    assert planner.explain_exit(1, "nothing recognisable") is None


@pytest.mark.asyncio
async def test_a_known_engine_death_reaches_last_error(tmp_path: Path) -> None:
    """END TO END through the supervisor, because the first version of
    this test did not.

    It asserted the adapter's own method while being named as though it
    proved the wiring, and the wiring was in fact broken: the supervisor
    asked the *planner* for a hook that only the *adapter* had, so
    nothing was ever explained and this test passed regardless. Drive
    the real path or prove nothing.
    """
    from eugene_plexus_agent.supervisor import _OUTPUT_TAIL_LINES, SupervisedProcess

    assert _OUTPUT_TAIL_LINES >= 40, "a tail too short to hold a traceback explains nothing"

    # A child that prints a long traceback ending in a signature the vLLM
    # adapter knows, then exits non-zero — which is exactly what a real
    # toolchain-less vLLM does, minus forty seconds.
    script = tmp_path / "dying_engine.py"
    script.write_text(
        "import sys\n"
        "for i in range(200):\n"
        "    print(f'  File \"frame{i}.py\", line {i}, in run')\n"
        "print('RuntimeError: Failed to find C compiler. Please specify via CC')\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )

    class _DyingEnginePlanner:
        name = "q"
        log_prefix = "[engine: q] "

        def plan(self) -> SpawnPlan:
            return SpawnPlan(argv=[sys.executable, str(script)], env=None, cwd=None)

        def on_crash_threshold(self) -> bool:
            return False

        def reset(self) -> None:
            return None

        def explain_exit(self, return_code: int, output_tail: str) -> str | None:
            return VllmAdapter().explain_exit(return_code, output_tail)

    proc = SupervisedProcess(_DyingEnginePlanner(), logging.getLogger("test"))
    proc.start()
    try:
        for _ in range(200):
            if proc.state is ProcessState.crashed:
                break
            await asyncio.sleep(0.05)

        assert proc.state is ProcessState.crashed
        assert proc.last_error is not None
        assert "build-essential" in proc.last_error, proc.last_error
        # And the generic message is replaced, not appended to.
        assert "exited with code" not in proc.last_error
    finally:
        # The loop respawns after a crash; without this the backoff task
        # outlives the test.
        await proc.stop()
