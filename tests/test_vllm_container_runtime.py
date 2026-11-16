"""Row 42: a vLLM runtime planned into its container, and the image as an engine.

A fake `docker` on PATH answers; the plan is checked, not run (specs
`scripts/row42-vllm-container-acceptance.py` runs it).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from eugene_plexus_agent import container_runtime, runtimes
from eugene_plexus_agent._generated.models import (
    Accelerator,
    EngineKind,
    HostAccelerator,
    Os,
    RuntimeSpec,
)
from eugene_plexus_agent.engines import VllmAdapter
from eugene_plexus_agent.engines.acquisition import Unavailable
from eugene_plexus_agent.engines.container_images import ImageInstaller, ImagePlan, ImageStore
from eugene_plexus_agent.supervisor import SpawnPlanError

from .fake_cli import FakeCli, docker_info

IMAGE = "vllm/vllm-openai@sha256:" + "d" * 64
IMAGE_ID = "sha256:" + "a" * 64


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCli:
    fake = FakeCli(tmp_path / "bin", "docker")
    fake.rules([{"args": ["info"], "out": docker_info()}])
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_ENGINE_ROOT", str(tmp_path / "engines"))
    monkeypatch.setattr(container_runtime, "_inside_a_container", lambda: False)
    host = HostAccelerator(os=Os.linux, accelerator=Accelerator.none)
    monkeypatch.setattr(runtimes, "detect_host", lambda: host)
    container_runtime.forget()
    yield fake
    container_runtime.forget()


@pytest.fixture
def model(tmp_path: Path) -> Path:
    folder = tmp_path / "library" / "Qwen3-8B"
    folder.mkdir(parents=True)
    return folder


def _pulled(tmp_path: Path) -> None:
    ImageStore(tmp_path / "engines").record_pull(
        image=IMAGE, digest=IMAGE_ID, version="0.33.0", variant="cuda"
    )


def _planner(model: Path, tmp_path: Path, config: dict | None = None, **flags: object):
    spec = RuntimeSpec(
        name="qwen",
        engine=EngineKind.vllm,
        modelPath=str(model),
        port=8401,
        flags={"reuseRepeatedText": False, **flags},
    )
    return runtimes._RuntimePlanner(
        spec,
        VllmAdapter(),
        logging.getLogger("test"),
        (config or {}).get,
        data_dir=tmp_path / "data",
        node_name=lambda: "box",
    )


def test_a_pulled_image_runs_the_runtime_in_its_container(
    docker: FakeCli, model: Path, tmp_path: Path
) -> None:
    _pulled(tmp_path)
    planner = _planner(model, tmp_path)
    plan = planner.plan()
    assert [os.path.normcase(plan.argv[0]), plan.argv[1]] == [os.path.normcase(docker.path), "run"]
    agent = container_runtime.agent_id(tmp_path / "data")
    name = f"ep-{agent}-qwen"
    assert plan.container is not None and plan.container.name == name
    assert IMAGE in plan.argv
    assert "eugene-plexus.node=box" in plan.argv
    # A container left by a stop that was cut short is removed before the run.
    assert ["rm", "-f", name] in docker.calls()
    assert planner.container is not None and planner.container.image == IMAGE
    assert planner.binary is not None and planner.binary.version == "0.33.0"
    assert (tmp_path / "data" / "vllm-cache" / "qwen").is_dir()
    # No GPU on this machine and none reaching containers: no GPU options.
    assert "--gpus" not in plan.argv


def test_the_stop_is_docker_stop_of_that_container(
    docker: FakeCli, model: Path, tmp_path: Path
) -> None:
    _pulled(tmp_path)
    plan = _planner(model, tmp_path).plan()
    assert plan.container is not None
    assert plan.container.stop() is None
    stop = [c for c in docker.calls() if c[:1] == ["stop"]]
    assert stop == [["stop", "-t", "30", plan.container.name]]


def test_no_image_and_container_chosen_names_the_pull(
    docker: FakeCli, model: Path, tmp_path: Path
) -> None:
    with pytest.raises(SpawnPlanError, match="Engines page"):
        _planner(model, tmp_path, source="container").plan()


def test_an_unusable_docker_is_named_with_its_fix(
    docker: FakeCli, model: Path, tmp_path: Path
) -> None:
    docker.rules(
        [
            {
                "args": ["info"],
                "rc": 1,
                "err": "permission denied while trying to connect to the Docker daemon socket\n",
            }
        ]
    )
    _pulled(tmp_path)
    with pytest.raises(SpawnPlanError, match="docker group"):
        _planner(model, tmp_path, source="container").plan()


def test_the_owners_own_vllm_wins_when_set(docker: FakeCli, model: Path, tmp_path: Path) -> None:
    """`vllmBinary` set and no `source`: the owner's vLLM, not the image."""
    _pulled(tmp_path)
    with pytest.raises(SpawnPlanError) as refused:
        _planner(model, tmp_path, {"vllmBinary": str(tmp_path / "nope" / "vllm")}).plan()
    assert "vllmBinary" in str(refused.value)
    assert not [c for c in docker.calls() if c[:1] == ["run"]]


def test_with_nothing_found_the_refusal_offers_the_container(
    docker: FakeCli, model: Path, tmp_path: Path
) -> None:
    with pytest.raises(SpawnPlanError, match="official container"):
        _planner(model, tmp_path).plan()


def test_install_with_variant_container_plans_a_pull(docker: FakeCli, tmp_path: Path) -> None:
    host = HostAccelerator(os=Os.windows, accelerator=Accelerator.cuda, acceleratorVersion="12.8")
    plan = runtimes.plan_for(EngineKind.vllm, variant="container", host=host)
    assert isinstance(plan, ImagePlan)
    assert plan.variant == "cuda12" and plan.version == "0.32.0"
    assert os.path.normcase(plan.cli) == os.path.normcase(docker.path)
    assert isinstance(runtimes.installer_for(EngineKind.vllm), ImageInstaller)
    # Without the variant: the manual answer, which now names the container too.
    manual = runtimes.plan_for(EngineKind.vllm, host=host)
    assert isinstance(manual, Unavailable) and "variant container" in manual.reason


def test_install_container_without_docker_is_refused_with_why(
    docker: FakeCli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    container_runtime.forget()
    plan = runtimes.plan_for(EngineKind.vllm, variant="container")
    assert isinstance(plan, Unavailable) and "Docker" in plan.reason


def test_the_engines_list_shows_the_image_and_counts_it_installed(
    docker: FakeCli, tmp_path: Path
) -> None:
    _pulled(tmp_path)
    vllm = next(e for e in runtimes.describe_engines() if e.engine is EngineKind.vllm)
    assert vllm.container is not None
    assert vllm.container.installable is True
    assert vllm.container.pulled is not None and vllm.container.pulled.version == "0.33.0"
    assert vllm.available is True and vllm.version == "0.33.0"


def test_the_routes_report_docker_and_pull_the_image(
    docker: FakeCli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authed_client
) -> None:
    import time

    local = "sha256:" + "f" * 64
    docker.rules(
        [
            {"args": ["info"], "out": docker_info()},
            {"args": ["image", "inspect", local], "out": f'[{{"Id": "{local}", "Size": 5}}]'},
        ]
    )
    container_runtime.forget()
    runtimes._INSTALLERS.pop(EngineKind.vllm, None)
    listed = authed_client.get("/v1/engines").json()
    assert listed["containerRuntime"]["kind"] == "docker"
    assert listed["containerRuntime"]["usable"] is True
    assert authed_client.patch("/v1/config", json={"vllmContainerImage": local}).is_success
    started = authed_client.post("/v1/engines/vllm/install", json={"variant": "container"})
    assert started.status_code == 202, started.text
    deadline = time.perf_counter() + 20
    state = started.json()
    while state["state"] not in ("done", "failed") and time.perf_counter() < deadline:
        time.sleep(0.05)
        state = authed_client.get("/v1/engines/vllm/install").json()
    assert state["state"] == "done", state
    vllm = next(
        e for e in authed_client.get("/v1/engines").json()["engines"] if e["engine"] == "vllm"
    )
    assert vllm["container"]["source"] == "configured"
    assert vllm["container"]["pulled"]["digest"] == local
    runtimes._INSTALLERS.pop(EngineKind.vllm, None)


def test_uninstall_removes_the_pulled_images_and_only_those(
    docker: FakeCli, tmp_path: Path, authed_client
) -> None:
    store = ImageStore(tmp_path / "engines")
    store.record_pull(image=IMAGE, digest=IMAGE_ID, version="0.33.0", variant="cuda")
    container_runtime.forget()
    assert authed_client.post("/v1/engines/vllm/uninstall").status_code == 204
    assert [c for c in docker.calls() if c[:2] == ["image", "rm"]] == [["image", "rm", IMAGE_ID]]
    assert store.current() is None
