"""The vLLM image as an engine build (row 42): pulled with progress, the
image in use and the one before it kept, a failed pull named in the
registry's words. A fake `docker` on PATH answers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from eugene_plexus_agent._generated.models import ContainerRuntimeKind, EngineKind, State
from eugene_plexus_agent.engines.container_images import (
    ImageInstaller,
    ImagePlan,
    ImageStore,
)

from .fake_cli import FakeCli

FIRST = "vllm/vllm-openai@sha256:" + "1" * 64
SECOND = "vllm/vllm-openai@sha256:" + "2" * 64
THIRD = "vllm/vllm-openai@sha256:" + "3" * 64

_PULL_OUT = (
    "sha256:1111: Pulling from vllm/vllm-openai\n"
    "aaaaaaaaaaaa: Pulling fs layer\n"
    "bbbbbbbbbbbb: Pulling fs layer\n"
    "aaaaaaaaaaaa: Download complete\n"
    "aaaaaaaaaaaa: Pull complete\n"
    "bbbbbbbbbbbb: Pull complete\n"
    "Digest: sha256:1111\n"
)


def _rules_for(image: str, image_id: str, marker: str) -> list[dict[str, object]]:
    inspect = json.dumps([{"Id": image_id, "Size": 9_000_000_000}])
    return [
        {"args": ["image", "inspect", image], "unless": marker, "rc": 1, "err": "No such image\n"},
        {"args": ["image", "inspect", image], "when": marker, "out": inspect},
        {"args": ["pull", image], "touch": marker, "out": _PULL_OUT},
    ]


@pytest.fixture
def docker(tmp_path: Path) -> FakeCli:
    return FakeCli(tmp_path / "bin", "docker")


def _plan(docker: FakeCli, image: str, version: str | None = "0.33.0") -> ImagePlan:
    return ImagePlan(
        cli=str(docker.path),
        kind=ContainerRuntimeKind.docker,
        image=image,
        version=version,
        variant="cuda",
    )


async def _install(installer: ImageInstaller, plan: ImagePlan):
    installer.start(plan)
    done = await installer.wait()
    assert done is not None
    return done


async def test_a_pull_is_an_install_with_progress_and_kept_as_a_build(
    docker: FakeCli, tmp_path: Path
) -> None:
    docker.rules(_rules_for(FIRST, "sha256:" + "a" * 64, "first"))
    store = ImageStore(tmp_path / "engines")
    installer = ImageInstaller(store, EngineKind.vllm)
    done = await _install(installer, _plan(docker, FIRST))
    assert done.state is State.done, done
    assert done.version == "0.33.0"
    assert "0.33.0" in (done.message or "")
    assert ["pull", FIRST] in docker.calls()
    current = store.current()
    assert current is not None and current.image == FIRST
    assert current.digest == "sha256:" + "a" * 64
    view = store.view()
    assert view is not None and view.version == "0.33.0" and view.previousVersion is None
    assert view.sizeBytes == 9_000_000_000


async def test_the_image_before_is_kept_and_the_one_before_that_removed(
    docker: FakeCli, tmp_path: Path
) -> None:
    docker.rules(
        _rules_for(FIRST, "sha256:" + "a" * 64, "first")
        + _rules_for(SECOND, "sha256:" + "b" * 64, "second")
        + _rules_for(THIRD, "sha256:" + "c" * 64, "third")
    )
    store = ImageStore(tmp_path / "engines")
    installer = ImageInstaller(store, EngineKind.vllm)
    await _install(installer, _plan(docker, FIRST, "0.31.0"))
    await _install(installer, _plan(docker, SECOND, "0.32.0"))
    view = store.view()
    assert view is not None and view.version == "0.32.0" and view.previousVersion == "0.31.0"
    assert not [c for c in docker.calls() if c[:2] == ["image", "rm"]]
    await _install(installer, _plan(docker, THIRD, "0.33.0"))
    view = store.view()
    assert view is not None and view.version == "0.33.0" and view.previousVersion == "0.32.0"
    # Only the image that fell off the record goes, and only by the ID we recorded.
    assert [c for c in docker.calls() if c[:2] == ["image", "rm"]] == [
        ["image", "rm", "sha256:" + "a" * 64]
    ]


async def test_a_failed_pull_names_the_registrys_answer(docker: FakeCli, tmp_path: Path) -> None:
    docker.rules(
        [
            {"args": ["image", "inspect"], "rc": 1, "err": "No such image\n"},
            {
                "args": ["pull"],
                "rc": 1,
                "err": (
                    "Error response from daemon: manifest for vllm/vllm-openai@sha256:1111 "
                    "not found: manifest unknown: manifest unknown\n"
                ),
            },
        ]
    )
    store = ImageStore(tmp_path / "engines")
    done = await _install(ImageInstaller(store, EngineKind.vllm), _plan(docker, FIRST))
    assert done.state is State.failed
    assert "manifest unknown" in (done.error or "")
    assert store.current() is None


async def test_an_image_already_here_is_recorded_without_a_pull(
    docker: FakeCli, tmp_path: Path
) -> None:
    local = "sha256:" + "f" * 64
    docker.rules(
        [{"args": ["image", "inspect", local], "out": json.dumps([{"Id": local, "Size": 1}])}]
    )
    store = ImageStore(tmp_path / "engines")
    done = await _install(
        ImageInstaller(store, EngineKind.vllm), _plan(docker, local, version=None)
    )
    assert done.state is State.done
    assert not [c for c in docker.calls() if c[:1] == ["pull"]]
    view = store.view()
    assert view is not None and view.version == "ffffffffffff" and view.digest == local


async def test_a_pull_can_be_cancelled(docker: FakeCli, tmp_path: Path) -> None:
    docker.rules(
        [
            {"args": ["image", "inspect"], "rc": 1, "err": "No such image\n"},
            {"args": ["pull"], "sleep": 2, "out": _PULL_OUT},
        ]
    )
    store = ImageStore(tmp_path / "engines")
    installer = ImageInstaller(store, EngineKind.vllm)
    installer.start(_plan(docker, FIRST))
    await asyncio.sleep(0.5)
    snapshot = await asyncio.wait_for(installer.cancel(), 20)
    assert snapshot is not None and snapshot.state is State.cancelled
    assert store.current() is None


def test_removing_the_images_removes_only_what_was_recorded(
    docker: FakeCli, tmp_path: Path
) -> None:
    store = ImageStore(tmp_path / "engines")
    store.record_pull(image=FIRST, digest="sha256:" + "a" * 64, version="0.31.0", variant="cuda")
    store.record_pull(image=SECOND, digest="sha256:" + "b" * 64, version="0.32.0", variant="cuda")
    store.remove_all(str(docker.path))
    removed = sorted(c[2] for c in docker.calls() if c[:2] == ["image", "rm"])
    assert removed == ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
    assert store.current() is None and store.view() is None
