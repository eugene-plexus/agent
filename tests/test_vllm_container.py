"""vLLM from its official container image (row 42): the launch plan.

CI runners have no GPU, so these check the plan itself -- the argv, the
mounts, the published port and the environment -- exactly as the agent hands
them to `docker run`. specs `scripts/row42-vllm-container-acceptance.py` runs
the container path for real with a CPU fixture image.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eugene_plexus_agent._generated.models import (
    Accelerator,
    ContainerRuntimeKind,
    EngineKind,
    HostAccelerator,
    Origin,
    Os,
    RuntimeSpec,
)
from eugene_plexus_agent.engines import VllmAdapter
from eugene_plexus_agent.engines import vllm_container as vc
from eugene_plexus_agent.engines.base import DiscoveredBinary

IMAGE = "vllm/vllm-openai@sha256:" + "d" * 64
AGENT = "abc123abc123"


@pytest.fixture
def model(tmp_path: Path) -> Path:
    folder = tmp_path / "library" / "Qwen3-8B"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text('{"num_attention_heads": 32}', encoding="utf-8")
    return folder


def _context(tmp_path: Path, **overrides: object) -> vc.Context:
    values: dict[str, object] = dict(
        cli="/usr/bin/docker",
        kind=ContainerRuntimeKind.docker,
        image=IMAGE,
        agent=AGENT,
        node="box",
        cache_root=tmp_path / "vllm-cache",
        gpu=vc.Gpu.nvidia,
        wsl=False,
        base_env={"PATH": "/usr/bin"},
        offered=lambda: None,
    )
    values.update(overrides)
    return vc.Context(**values)  # type: ignore[arg-type]


def _spec(model: Path, **fields: object) -> RuntimeSpec:
    return RuntimeSpec(
        name="qwen",
        engine=EngineKind.vllm,
        modelPath=str(model),
        port=8401,
        **fields,  # type: ignore[arg-type]
    )


def _launch(model: Path, tmp_path: Path, spec: RuntimeSpec | None = None, **context: object):
    return VllmAdapter().build_container_launch(
        spec or _spec(model), 8401, _context(tmp_path, **context)
    )


def _after(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]


def test_container_argv_mounts_the_model_read_only_and_binds_loopback(
    model: Path, tmp_path: Path
) -> None:
    plan = _launch(model, tmp_path)
    argv = plan.argv
    assert argv[:3] == ["/usr/bin/docker", "run", "--rm"]
    assert _after(argv, "--name") == [f"ep-{AGENT}-qwen"]
    assert f"eugene-plexus.agent={AGENT}" in _after(argv, "--label")
    assert "eugene-plexus.runtime=qwen" in _after(argv, "--label")
    assert _after(argv, "--pull") == ["never"]
    # The model's own folder, read only; the compile cache, writable; nothing else.
    mounts = _after(argv, "--mount")
    assert mounts == [
        f"type=bind,source={model},target=/models/Qwen3-8B,readonly",
        f"type=bind,source={tmp_path / 'vllm-cache' / 'qwen'},target=/root/.cache/vllm",
    ]
    assert (tmp_path / "vllm-cache" / "qwen").is_dir()
    assert "-v" not in argv and "--volume" not in argv
    # Published on loopback only; vLLM listens on 8000 inside.
    assert _after(argv, "-p") == ["127.0.0.1:8401:8000"]
    image_at = argv.index(IMAGE)
    assert argv[image_at - 2 : image_at] == ["--entrypoint", "vllm"]
    serve = argv[image_at + 1 :]
    assert serve[:2] == ["serve", "/models/Qwen3-8B"]
    assert _after(serve, "--host") == ["0.0.0.0"] and _after(serve, "--port") == ["8000"]
    # The served name is the model's, never the path inside the container.
    assert _after(serve, "--served-model-name") == ["Qwen3-8B"]
    assert plan.name == f"ep-{AGENT}-qwen" and plan.image == IMAGE
    assert "--privileged" not in argv and "/var/run/docker.sock" not in " ".join(argv)


def test_a_host_the_profile_names_is_where_the_port_is_published(
    model: Path, tmp_path: Path
) -> None:
    plan = _launch(model, tmp_path, _spec(model, host="0.0.0.0"))
    assert _after(plan.argv, "-p") == ["0.0.0.0:8401:8000"]
    plan = _launch(model, tmp_path, _spec(model, host="::1"))
    assert _after(plan.argv, "-p") == ["[::1]:8401:8000"]


def test_device_pins_become_gpus_device_uuids(model: Path, tmp_path: Path) -> None:
    pinned = _spec(model, env={"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb"})
    plan = _launch(model, tmp_path, pinned)
    # Docker reads `--gpus` as CSV: a list of devices must be quoted inside the value.
    assert _after(plan.argv, "--gpus") == ['"device=GPU-aaa,GPU-bbb"']
    # Inside, the container sees only those cards, numbered from 0.
    assert "CUDA_VISIBLE_DEVICES" not in plan.env
    assert "CUDA_VISIBLE_DEVICES" not in _after(plan.argv, "-e")
    assert _after(_launch(model, tmp_path).argv, "--gpus") == ["all"]


def test_podman_takes_cdi_devices(model: Path, tmp_path: Path) -> None:
    pinned = _spec(model, env={"CUDA_VISIBLE_DEVICES": "GPU-aaa,1"})
    plan = _launch(model, tmp_path, pinned, kind=ContainerRuntimeKind.podman)
    assert _after(plan.argv, "--device") == ["nvidia.com/gpu=GPU-aaa", "nvidia.com/gpu=1"]
    assert "--gpus" not in plan.argv


def test_rocm_passes_the_amd_devices_and_no_cards_passes_none(model: Path, tmp_path: Path) -> None:
    plan = _launch(model, tmp_path, gpu=vc.Gpu.rocm)
    assert _after(plan.argv, "--device") == ["/dev/kfd", "/dev/dri"]
    assert "--gpus" not in plan.argv
    plan = _launch(model, tmp_path, gpu=vc.Gpu.none)
    assert "--gpus" not in plan.argv and "--device" not in plan.argv
    plan = _launch(model, tmp_path, _spec(model, env={"CUDA_VISIBLE_DEVICES": ""}))
    assert "--gpus" not in plan.argv


def test_shm_size_not_ipc_host(model: Path, tmp_path: Path) -> None:
    assert _after(_launch(model, tmp_path).argv, "--shm-size") == ["8g"]
    bigger = _spec(model, flags={vc.KEY_SHM: 16})
    plan = _launch(model, tmp_path, bigger)
    assert _after(plan.argv, "--shm-size") == ["16g"]
    assert not any(a.startswith("--ipc") for a in plan.argv)
    # Not a vLLM flag: never in what vLLM is asked to serve.
    assert "16" not in plan.argv[plan.argv.index(IMAGE) :]


def test_env_values_never_reach_the_argv(model: Path, tmp_path: Path) -> None:
    spec = _spec(model, env={"HF_TOKEN": "hf_secret_value", "VLLM_LOGGING_LEVEL": "DEBUG"})
    plan = _launch(model, tmp_path, spec)
    names = _after(plan.argv, "-e")
    assert {"HF_TOKEN", "VLLM_LOGGING_LEVEL", "HF_HUB_OFFLINE"} <= set(names)
    assert not any("hf_secret_value" in a for a in plan.argv)
    assert plan.env["HF_TOKEN"] == "hf_secret_value"
    assert plan.env["HF_HUB_OFFLINE"] == "1"
    # The CLI keeps the host's own variables (DOCKER_HOST and the like) ...
    assert plan.env["PATH"] == "/usr/bin"
    # ... and none of them is passed into the container.
    assert "PATH" not in names


def test_wsl2_desktop_turns_pinned_memory_on(model: Path, tmp_path: Path) -> None:
    plan = _launch(model, tmp_path, wsl=True)
    assert plan.env["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "1"
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY" in _after(plan.argv, "-e")
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY" not in _launch(model, tmp_path).env


def test_a_draft_outside_the_model_folder_is_mounted_read_only(model: Path, tmp_path: Path) -> None:
    draft = tmp_path / "library" / "Qwen3-0.6B"
    draft.mkdir()
    spec = _spec(model, flags={"draftWith": "companion", "draftModelPath": str(draft)})
    plan = _launch(model, tmp_path, spec)
    mounts = _after(plan.argv, "--mount")
    assert f"type=bind,source={draft},target=/models/draft-Qwen3-0.6B,readonly" in mounts
    assert str(draft) not in " ".join(plan.argv[plan.argv.index(IMAGE) :])


def test_a_tokenizer_outside_the_model_folder_is_refused(model: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="tokenizer"):
        _launch(model, tmp_path, _spec(model, flags={"tokenizer": str(elsewhere)}))
    inside = model / "tok"
    inside.mkdir()
    plan = _launch(model, tmp_path, _spec(model, flags={"tokenizer": str(inside)}))
    assert _after(plan.argv, "--tokenizer") == ["/models/Qwen3-8B/tok"]


def test_a_path_with_a_comma_is_quoted_for_mount(tmp_path: Path) -> None:
    odd = tmp_path / "a,b"
    odd.mkdir()
    plan = _launch(odd, tmp_path, _spec(odd))
    first = _after(plan.argv, "--mount")[0]
    assert first == f'type=bind,"source={odd}",target=/models/a-b,readonly'


def test_the_image_must_be_named_by_digest() -> None:
    assert vc.by_digest(IMAGE)
    assert vc.by_digest("sha256:" + "a" * 64)
    assert vc.by_digest("ghcr.io/me/vllm@sha256:" + "a" * 64)
    assert not vc.by_digest("vllm/vllm-openai:latest")
    assert not vc.by_digest("vllm/vllm-openai@sha256:short")
    assert not vc.by_digest("vllm/vllm-openai@sha256:" + "a" * 64 + " --privileged")


def test_the_pinned_image_follows_the_driver() -> None:
    def host(accelerator: Accelerator, version: str | None = None) -> HostAccelerator:
        return HostAccelerator(os=Os.linux, accelerator=accelerator, acceleratorVersion=version)

    assert vc.pinned_for(host(Accelerator.cuda, "13.0")).variant == "cuda"
    assert vc.pinned_for(host(Accelerator.cuda, "12.8")).variant == "cuda12"
    assert vc.pinned_for(host(Accelerator.cuda)).variant == "cuda"
    rocm = vc.pinned_for(host(Accelerator.rocm, "7.0"))
    assert rocm.variant == "rocm" and rocm.untested
    for pinned in vc.pinned_images().values():
        assert vc.by_digest(pinned.image), pinned


def test_the_owners_image_wins_and_must_be_by_digest() -> None:
    host = HostAccelerator(os=Os.linux, accelerator=Accelerator.cuda, acceleratorVersion="13.0")
    own = "ghcr.io/me/vllm@sha256:" + "e" * 64
    choice = vc.image_choice({"vllmContainerImage": own}.get, host)
    assert choice.image == own and choice.variant == "custom" and choice.version is None
    with pytest.raises(ValueError, match="by digest"):
        vc.image_choice({"vllmContainerImage": "me/vllm:latest"}.get, host)
    assert vc.image_choice({}.get, host).image == vc.pinned_images()["cuda"].image


def test_native_argv_skips_the_container_settings(tmp_path: Path) -> None:
    binary = DiscoveredBinary(path=tmp_path / "vllm", origin=Origin.configured, version="0.33.0")
    spec = RuntimeSpec(
        name="qwen",
        engine=EngineKind.vllm,
        modelPath="/m/Qwen3-8B",
        port=8401,
        flags={vc.KEY_SOURCE: "binary", vc.KEY_SHM: 4, "reuseRepeatedText": False},
    )
    argv = VllmAdapter().build_argv(spec, binary, 8401)
    assert "binary" not in argv and "4" not in argv
    fields = {f.key for f in VllmAdapter().flag_schema().fields}
    assert {vc.KEY_SOURCE, vc.KEY_SHM} <= fields


def test_container_failures_name_their_cause() -> None:
    adapter = VllmAdapter()
    toolkit = adapter.explain_exit(
        125,
        'docker: Error response from daemon: could not select device driver "" with '
        "capabilities: [[gpu]].\n",
    )
    assert toolkit is not None and "NVIDIA Container Toolkit" in toolkit
    assert "install-guide" in toolkit
    missing = adapter.explain_exit(
        125, "Unable to find image 'vllm/vllm-openai@sha256:dd' locally\n"
    )
    assert missing is not None and "Engines" in missing
    shm = adapter.explain_exit(
        1, "Insufficient space in /dev/shm: 20480 MiB required, 8000 MiB free\n"
    )
    assert shm is not None and "--shm-size" in shm and "--ipc=host" not in shm
    daemon = adapter.explain_exit(125, "docker: Error response from daemon: driver failed.\n")
    assert daemon is not None and "driver failed" in daemon


@pytest.mark.skipif(os.name != "nt", reason="Windows paths")
def test_a_windows_model_folder_mounts_as_written(tmp_path: Path) -> None:
    folder = tmp_path / "Qwen3-8B"
    folder.mkdir()
    plan = _launch(folder, tmp_path, _spec(folder))
    assert _after(plan.argv, "--mount")[0] == (
        f"type=bind,source={folder},target=/models/Qwen3-8B,readonly"
    )


def test_an_unknown_source_is_refused_before_anything_runs(model: Path) -> None:
    adapter = VllmAdapter()
    assert adapter.declaration_problem(_spec(model, flags={vc.KEY_SOURCE: "container"})) is None
    problem = adapter.declaration_problem(_spec(model, flags={vc.KEY_SOURCE: "docker"}))
    assert problem is not None and "container" in problem and "binary" in problem
    problem = adapter.declaration_problem(_spec(model, flags={vc.KEY_SHM: "lots"}))
    assert problem is not None and "Shared memory" in problem


@pytest.mark.parametrize(
    "name",
    ["DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "CONTAINER_HOST", "PATH", "LD_PRELOAD"],
)
def test_a_variable_that_steers_the_container_program_is_refused(
    model: Path, tmp_path: Path, name: str
) -> None:
    """Review note: profile values ride in the CLI's own environment, so one of
    these would choose the daemon, its credentials or the program."""
    with pytest.raises(ValueError, match="steer the container program"):
        _launch(model, tmp_path, _spec(model, env={name: "tcp://elsewhere:2375"}))


@pytest.mark.parametrize("name", ["1BAD", "A-B", "A=B"])
def test_a_name_that_is_not_a_variable_is_refused(model: Path, tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="not a variable name"):
        _launch(model, tmp_path, _spec(model, env={name: "x"}))
