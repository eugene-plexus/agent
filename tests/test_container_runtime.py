"""Docker or Podman on this machine (row 42): detection, labels, the orphan sweep.

A fake `docker` on PATH answers; nothing here needs a container runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eugene_plexus_agent import container_runtime
from eugene_plexus_agent._generated.models import ContainerRuntimeKind

from .fake_cli import FakeCli, docker_info


@pytest.fixture
def path_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "bin"
    directory.mkdir()
    monkeypatch.setenv("PATH", str(directory))
    monkeypatch.setattr(container_runtime, "_inside_a_container", lambda: False)
    container_runtime.forget()
    return directory


def test_docker_with_gpus_is_detected(path_dir: Path) -> None:
    docker = FakeCli(path_dir, "docker")
    docker.rules(
        [
            {
                "args": ["info"],
                "out": docker_info(Runtimes={"nvidia": {"path": "nvidia-container-runtime"}}),
            }
        ]
    )
    found = container_runtime.detect()
    assert found.kind is ContainerRuntimeKind.docker
    assert found.usable and found.gpus
    assert found.version == "27.3.1"
    assert found.error is None
    view = found.view()
    assert view.usable and view.gpus and view.kind is ContainerRuntimeKind.docker


def test_cdi_devices_count_as_gpus(path_dir: Path) -> None:
    docker = FakeCli(path_dir, "docker")
    docker.rules(
        [
            {
                "args": ["info"],
                "out": docker_info(
                    DiscoveredDevices=[{"Source": "cdi", "ID": "nvidia.com/gpu=all"}]
                ),
            }
        ]
    )
    assert container_runtime.detect().gpus


def test_docker_without_a_gpu_runtime_is_usable_without_gpus(path_dir: Path) -> None:
    FakeCli(path_dir, "docker").rules([{"args": ["info"], "out": docker_info()}])
    found = container_runtime.detect()
    assert found.usable and not found.gpus and not found.desktop


def test_docker_desktop_on_wsl2_passes_gpus(path_dir: Path) -> None:
    FakeCli(path_dir, "docker").rules(
        [
            {
                "args": ["info"],
                "out": docker_info(
                    OperatingSystem="Docker Desktop",
                    KernelVersion="6.6.87.2-microsoft-standard-WSL2",
                ),
            }
        ]
    )
    found = container_runtime.detect()
    assert found.desktop and found.gpus and found.wsl


def test_a_permission_error_names_the_docker_group(path_dir: Path) -> None:
    FakeCli(path_dir, "docker").rules(
        [
            {
                "args": ["info"],
                "rc": 1,
                "err": (
                    "permission denied while trying to connect to the Docker daemon socket at "
                    'unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.47/'
                    'info": dial unix /var/run/docker.sock: connect: permission denied\n'
                ),
            }
        ]
    )
    found = container_runtime.detect()
    assert not found.usable
    assert found.kind is ContainerRuntimeKind.docker
    assert "permission denied" in (found.error or "")
    assert "docker group" in (found.fix or "")


def test_a_stopped_daemon_says_to_start_docker(path_dir: Path) -> None:
    FakeCli(path_dir, "docker").rules(
        [
            {
                "args": ["info"],
                "rc": 1,
                "out": json.dumps(
                    {
                        "ServerErrors": [
                            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
                            "Is the docker daemon running?"
                        ]
                    }
                ),
            }
        ]
    )
    found = container_runtime.detect()
    assert not found.usable
    assert "Cannot connect to the Docker daemon" in (found.error or "")
    assert "Start" in (found.fix or "")


def test_podman_is_used_when_there_is_no_docker(path_dir: Path) -> None:
    FakeCli(path_dir, "podman").rules(
        [
            {
                "args": ["info"],
                "out": json.dumps(
                    {"host": {"kernel": "6.8.0", "os": "linux"}, "version": {"Version": "5.2.2"}}
                ),
            }
        ]
    )
    found = container_runtime.detect()
    assert found.kind is ContainerRuntimeKind.podman and found.usable
    assert found.version == "5.2.2"


def test_nothing_installed_says_so(path_dir: Path) -> None:
    found = container_runtime.detect()
    assert found.kind is None and not found.usable
    assert "Docker" in (found.error or "")


def test_inside_a_container_without_docker_names_the_socket_trade(
    path_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(container_runtime, "_inside_a_container", lambda: True)
    found = container_runtime.detect()
    assert found.inside_container and not found.usable
    assert "without Docker access" in (found.error or "")
    assert "control of the host" in (found.fix or "")


def test_container_names_are_safe_and_carry_the_agent() -> None:
    agent = container_runtime.agent_id(Path("/var/lib/eugene"))
    assert len(agent) == 12
    assert container_runtime.container_name(agent, "qwen3-8b") == f"ep-{agent}-qwen3-8b"
    odd = container_runtime.container_name(agent, "Qwen 3/8B")
    assert odd.startswith(f"ep-{agent}-Qwen-3-8B-")
    assert container_runtime.container_name(agent, "Qwen 3/8B") != container_runtime.container_name(
        agent, "Qwen-3-8B"
    )


def test_orphaned_labelled_containers_are_stopped_at_start(path_dir: Path) -> None:
    """Only this agent's labelled containers go; another agent's stay."""
    docker = FakeCli(path_dir, "docker")
    docker.rules(
        [
            {"args": ["info"], "out": docker_info()},
            {
                "args": ["ps", "-a", "--filter", "label=eugene-plexus.agent=abc123abc123"],
                "out": "ep-abc123abc123-left\nep-abc123abc123-right\n",
            },
        ]
    )
    found = container_runtime.detect()
    stopped = container_runtime.sweep_orphans(found, "abc123abc123")
    assert stopped == ["ep-abc123abc123-left", "ep-abc123abc123-right"]
    calls = docker.calls()
    ps = next(c for c in calls if c[:1] == ["ps"])
    assert "label=eugene-plexus.agent=abc123abc123" in ps
    removed = [c for c in calls if c[:2] == ["rm", "-f"]]
    assert removed == [["rm", "-f", "ep-abc123abc123-left", "ep-abc123abc123-right"]]


def test_no_orphans_runs_nothing_destructive(path_dir: Path) -> None:
    docker = FakeCli(path_dir, "docker")
    docker.rules([{"args": ["info"], "out": docker_info()}, {"args": ["ps"], "out": ""}])
    assert container_runtime.sweep_orphans(container_runtime.detect(), "abc123abc123") == []
    assert not [c for c in docker.calls() if c[:1] == ["rm"]]


def test_eugenes_own_account_runs_only_a_trusted_docker(
    path_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row 8's rule for every engine program holds for the container CLI too."""
    from eugene_plexus_agent import exec_trust

    FakeCli(path_dir, "docker").rules([{"args": ["info"], "out": docker_info()}])
    monkeypatch.setattr(exec_trust, "own_account", lambda: True)

    def refused(path: object, **_: object) -> Path:
        raise exec_trust.Untrusted(f"{path} can be changed by another account")

    monkeypatch.setattr(exec_trust, "trusted_program", refused)
    found = container_runtime.detect()
    assert not found.usable and found.cli is None
    assert "can be changed by another account" in (found.error or "")
