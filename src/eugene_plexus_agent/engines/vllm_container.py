"""vLLM from its official container image: the images and the launch (row 42).

`specs/docs/design/rows/row-42.md`. The same `vllm serve` arguments the
native path builds (`VllmAdapter._serve_args`), after the image's `vllm`
entrypoint, inside `docker run` (or `podman run`) with:

- the model's own folder bind-mounted **read only** at `/models/<name>`, and a
  drafting companion's folder the same way when it is elsewhere; nothing else
  of the host's files but a per-runtime compile cache;
- the port published on loopback, or on the host the profile names, as a
  native runtime binds;
- `--shm-size` (a profile field), never `--ipc=host`;
- `--gpus` (or Podman's CDI devices, or AMD's device nodes) from the cards
  this launch uses;
- the profile's environment as `-e NAME` pairs whose values ride in the
  CLI's own environment, so no value is ever in a command line;
- `--pull never`: an image is pulled only by an install, with progress.

The image is named by digest, always: the one this release pins for the
machine's GPUs (`vllm_images.json`, equal to specs `release/manifest.json`
`vllmImages`), or the owner's `vllmContainerImage`.
"""

from __future__ import annotations

import functools
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from .._generated.models import (
    Accelerator,
    ContainerImageSource,
    ContainerRuntimeKind,
    HostAccelerator,
    RuntimeSpec,
)
from . import same_answer, speculation
from .base import default_model_alias

#: The profile's choice between the image and the owner's own vLLM.
KEY_SOURCE = "source"
SOURCE_CONTAINER = "container"
SOURCE_BINARY = "binary"
#: Shared memory for the container, GiB.
KEY_SHM = "shmSizeGiB"
DEFAULT_SHM_GIB = 8
#: The agent's expert override for the image (`state.py`).
CONFIG_IMAGE_KEY = "vllmContainerImage"
VARIANT_CUSTOM = "custom"

#: Where vLLM listens inside the container; the host side is the runtime's port.
CONTAINER_PORT = 8000
MODELS = PurePosixPath("/models")
CACHE_TARGET = "/root/.cache/vllm"

_DIGEST = re.compile(r"(?:[a-z0-9][a-z0-9._/:-]*@)?sha256:[0-9a-f]{64}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
#: The variables that choose cards; they become device options instead.
_NVIDIA_PINS = "CUDA_VISIBLE_DEVICES"


def by_digest(reference: str) -> bool:
    """`name@sha256:<64 hex>`, or a local image ID `sha256:<64 hex>`."""
    return bool(_DIGEST.fullmatch(reference))


@dataclass(frozen=True)
class Pinned:
    variant: str
    image: str
    version: str
    untested: bool = False


@functools.cache
def pinned_images() -> dict[str, Pinned]:
    raw = json.loads(Path(__file__).with_name("vllm_images.json").read_text(encoding="utf-8"))
    return {
        variant: Pinned(
            variant=variant,
            image=str(entry["image"]),
            version=str(entry["version"]),
            untested=bool(entry.get("untested")),
        )
        for variant, entry in raw.items()
        if isinstance(entry, dict)
    }


def _major(version: str | None) -> int | None:
    try:
        return int((version or "").partition(".")[0])
    except ValueError:
        return None


def pinned_for(host: HostAccelerator) -> Pinned:
    """The pinned image for this machine's GPUs: AMD's for ROCm (untested,
    offered), CUDA 12's for a driver that says CUDA 12, else CUDA 13's."""
    images = pinned_images()
    if host.accelerator is Accelerator.rocm:
        return images["rocm"]
    major = _major(host.acceleratorVersion)
    if host.accelerator is Accelerator.cuda and major is not None and major < 13:
        return images["cuda12"]
    return images["cuda"]


@dataclass(frozen=True)
class ImageChoice:
    """What a pull fetches now."""

    image: str
    version: str | None
    source: ContainerImageSource
    variant: str
    untested: bool = False


def image_choice(get_config: Callable[[str], Any] | None, host: HostAccelerator) -> ImageChoice:
    """The owner's `vllmContainerImage` when set, else the pinned image.

    Raises ValueError for an owner's image not named by digest: a moving tag
    would run something nobody chose."""
    configured = get_config(CONFIG_IMAGE_KEY) if get_config is not None else None
    if isinstance(configured, str) and configured.strip():
        reference = configured.strip()
        if not by_digest(reference):
            raise ValueError(
                f"vllmContainerImage must name the image by digest "
                f"(name@sha256:<64 hex digits>), not {reference!r}"
            )
        return ImageChoice(reference, None, ContainerImageSource.configured, VARIANT_CUSTOM)
    pinned = pinned_for(host)
    return ImageChoice(
        pinned.image,
        pinned.version,
        ContainerImageSource.pinned,
        pinned.variant,
        pinned.untested,
    )


class Gpu(StrEnum):
    """Which cards a container is given."""

    nvidia = "nvidia"
    rocm = "rocm"
    none = "none"


def gpu_for(host: HostAccelerator | None, containers_see_gpus: bool) -> Gpu:
    """AMD's device nodes on a ROCm host; NVIDIA's when the host has an NVIDIA
    card or the container runtime says GPUs reach it; nothing on a machine
    with neither (a CPU runner), where vLLM says what it found."""
    accelerator = host.accelerator if host is not None else None
    if accelerator is Accelerator.rocm:
        return Gpu.rocm
    if accelerator is Accelerator.cuda or containers_see_gpus:
        return Gpu.nvidia
    return Gpu.none


@dataclass(frozen=True)
class Context:
    """What a container launch needs beyond the runtime's own declaration."""

    cli: str
    kind: ContainerRuntimeKind
    image: str
    #: This agent's label value (`container_runtime.agent_id`).
    agent: str
    node: str | None
    #: `<agent data>/vllm-cache`; each runtime gets its own folder below it.
    cache_root: Path
    gpu: Gpu
    #: The daemon's kernel is WSL 2's (Docker Desktop on Windows).
    wsl: bool
    #: The CLI's own environment (`DOCKER_HOST` and the like).
    base_env: Mapping[str, str]
    #: Every `--long-flag` this image's `vllm serve --help=all` lists, or None.
    offered: Callable[[], frozenset[str] | None] = field(default=lambda: None)


@dataclass(frozen=True)
class Mount:
    source: str
    target: str
    readonly: bool

    def arg(self) -> str:
        """`--mount`'s value. Docker reads it as CSV, so a field holding a
        comma or a quote is quoted."""
        fields = ["type=bind", f"source={self.source}", f"target={self.target}"]
        if self.readonly:
            fields.append("readonly")
        return ",".join(
            '"' + f.replace('"', '""') + '"' if ("," in f or '"' in f) else f for f in fields
        )


@dataclass(frozen=True)
class ContainerLaunch:
    argv: list[str]
    #: The CLI's environment, the container's values included.
    env: dict[str, str]
    name: str
    image: str
    mounts: list[Mount]


def safe(name: str) -> str:
    return _UNSAFE.sub("-", name).strip("-.") or "model"


def _inside(path: str, folder: str) -> str | None:
    """`path` relative to `folder` in `/` form, "" for the folder itself, or
    None when it is not inside it (links resolved, so one pointing out is out)."""
    try:
        relative = os.path.relpath(os.path.realpath(path), os.path.realpath(folder))
    except ValueError:  # another drive
        return None
    if relative == ".":
        return ""
    if relative == ".." or relative.startswith(".." + os.sep) or os.path.isabs(relative):
        return None
    return PurePosixPath(*Path(relative).parts).as_posix()


def _folder_of(path: str) -> tuple[str, str]:
    """(the folder to mount, the name inside it) for a model path."""
    if os.path.isdir(path):
        return path, ""
    return os.path.dirname(path) or ".", os.path.basename(path)


def _publish(host: str | None, port: int) -> str:
    bind = host or "127.0.0.1"
    if bind == "localhost":
        bind = "127.0.0.1"
    if ":" in bind:
        bind = f"[{bind.strip('[]')}]"
    return f"{bind}:{port}:{CONTAINER_PORT}"


def _shm(flags: Mapping[str, object]) -> str:
    value = flags.get(KEY_SHM)
    size = (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0
        else float(DEFAULT_SHM_GIB)
    )
    return f"{int(size)}g" if size.is_integer() else f"{int(size * 1024)}m"


def _gpu_args(context: Context, pins: str | None) -> list[str]:
    if context.gpu is Gpu.rocm:
        return ["--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video"]
    if context.gpu is Gpu.none:
        return []
    ids = None if pins is None else [t.strip() for t in pins.split(",") if t.strip()]
    if ids is not None and not ids:
        return []  # pinned to no card
    if context.kind is ContainerRuntimeKind.podman:
        out: list[str] = []
        for card in ids or ["all"]:
            out += ["--device", f"nvidia.com/gpu={card}"]
        return out
    # Docker parses `--gpus` as CSV: a list of devices is quoted inside the value.
    return ["--gpus", "all" if ids is None else '"device=' + ",".join(ids) + '"']


def plan_launch(
    spec: RuntimeSpec,
    port: int,
    context: Context,
    serve_args: Callable[[RuntimeSpec], list[str]],
) -> ContainerLaunch:
    """The `docker run` for this runtime. `spec` carries the paths as this
    machine opens them; `serve_args` builds `serve <model> ...` for the spec
    as the container sees it.

    Raises ValueError for a declaration a container cannot honour (a tokenizer
    outside the model's folder, a variable name that is not one)."""
    from .. import container_runtime

    flags = dict(spec.flags or {})
    folder, inner = _folder_of(spec.modelPath)
    target = MODELS / safe(os.path.basename(os.path.normpath(folder)))
    mounts = [Mount(folder, target.as_posix(), readonly=True)]
    inside_path = (target / inner).as_posix() if inner else target.as_posix()
    places: list[tuple[str, PurePosixPath]] = [(folder, target)]

    def mapped(path: str) -> str | None:
        for source, at in places:
            relative = _inside(path, source)
            if relative is not None:
                return (at / relative).as_posix() if relative else at.as_posix()
        return None

    draft = speculation.companion(flags)
    if draft is not None:
        local = mapped(draft)
        if local is None:
            draft_folder, draft_inner = _folder_of(draft)
            draft_target = (
                MODELS / f"draft-{safe(os.path.basename(os.path.normpath(draft_folder)))}"
            )
            mounts.append(Mount(draft_folder, draft_target.as_posix(), readonly=True))
            places.append((draft_folder, draft_target))
            local = (
                (draft_target / draft_inner).as_posix() if draft_inner else draft_target.as_posix()
            )
        flags[speculation.DRAFT_MODEL_KEY] = local

    tokenizer = flags.get("tokenizer")
    if isinstance(tokenizer, str) and tokenizer:
        local = mapped(tokenizer)
        if local is None:
            raise ValueError(
                f"The tokenizer {tokenizer} is outside the model's folder, and a container "
                "is given only the model's folder. Put it in the model's folder, or run "
                "this model with your own vLLM (Source: My own vLLM)."
            )
        flags["tokenizer"] = local

    cache = context.cache_root / container_runtime.safe_name(spec.name)
    cache.mkdir(parents=True, exist_ok=True)
    mounts.append(Mount(str(cache), CACHE_TARGET, readonly=False))

    inside = spec.model_copy(
        update={
            "modelPath": inside_path,
            "host": "0.0.0.0",  # inside the container; the publish below decides who reaches it
            "port": CONTAINER_PORT,
            "modelAlias": spec.modelAlias or default_model_alias(spec.modelPath),
            "flags": flags,
        }
    )

    container_env: dict[str, str] = {"HF_HUB_OFFLINE": "1"}
    if context.wsl:
        # Docker Desktop's kernel is WSL 2's: vLLM turns pinned memory off there
        # and then needs it (`VllmAdapter.default_env`).
        container_env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
    if same_answer.KEY in flags:
        container_env[same_answer.VLLM_ENV] = str(int(flags.get(same_answer.KEY) is True))
    declared = {k: str(v) for k, v in (spec.env or {}).items()}
    pins = declared.pop(_NVIDIA_PINS, None) if context.gpu is Gpu.nvidia else None
    for name in declared:
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"{name!r} is not a variable name a container accepts")
    container_env.update(declared)

    name = container_runtime.container_name(context.agent, spec.name)
    argv = [context.cli, "run", "--rm", "--name", name]
    argv += ["--label", f"{container_runtime.LABEL_AGENT}={context.agent}"]
    argv += ["--label", f"{container_runtime.LABEL_RUNTIME}={spec.name}"]
    if context.node:
        argv += ["--label", f"{container_runtime.LABEL_NODE}={context.node}"]
    argv += ["--pull", "never", "--security-opt", "no-new-privileges"]
    argv += ["--shm-size", _shm(flags)]
    argv += ["-p", _publish(spec.host, port)]
    for mount in mounts:
        argv += ["--mount", mount.arg()]
    argv += _gpu_args(context, pins)
    for variable in container_env:
        argv += ["-e", variable]
    argv += ["--entrypoint", "vllm", context.image]
    argv += serve_args(inside)

    env = dict(context.base_env)
    env.update(container_env)
    return ContainerLaunch(argv=argv, env=env, name=name, image=context.image, mounts=mounts)


__all__ = [
    "CONFIG_IMAGE_KEY",
    "CONTAINER_PORT",
    "DEFAULT_SHM_GIB",
    "KEY_SHM",
    "KEY_SOURCE",
    "SOURCE_BINARY",
    "SOURCE_CONTAINER",
    "ContainerLaunch",
    "Context",
    "Gpu",
    "ImageChoice",
    "Mount",
    "Pinned",
    "by_digest",
    "gpu_for",
    "image_choice",
    "pinned_for",
    "pinned_images",
    "plan_launch",
]
