"""llama.cpp adapter — drives upstream `llama-server`.

We never fork or vendor llama.cpp. This knows how to start the binary the
project ships, how to read its `/health`, and which of its flags are worth
putting in front of a person.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

import httpx

from .._generated.models import (
    ConfigField,
    ConfigSchema,
    ConfigValueType,
    EngineKind,
    RuntimeCapabilities,
    RuntimeSpec,
)
from .base import (
    DiscoveredBinary,
    EngineAdapter,
    Loading,
    NotAnswering,
    Readiness,
    Ready,
)

log = logging.getLogger(__name__)

# `llama-server --version` writes something like
# `version: 4589 (a1b2c3d4)` / `build: 4589 (a1b2c3d4) with ...` to stderr.
_VERSION_RE = re.compile(r"\b(?:version|build):\s*(\S+)")

# How long to wait on a version probe. Generous enough for a cold binary on
# a spinning disk, short enough that a wedged executable doesn't stall the
# whole engine listing.
_VERSION_TIMEOUT_SECONDS = 10.0

# Readiness probes run on the supervisor's poll cadence, so they must be
# quick and must never be the thing that blocks a poll round.
_PROBE_TIMEOUT_SECONDS = 2.0


class LlamaCppAdapter(EngineAdapter):
    kind = EngineKind.llama_cpp
    binary_name = "llama-server"

    # --- discovery --------------------------------------------------------

    def probe_version(self, binary: Path) -> str | None:
        """Run `--version` and scrape the build number.

        llama.cpp writes this to stderr and exits non-zero on some
        builds, so neither is treated as failure — we read both streams
        and only care whether the pattern matched.
        """
        try:
            proc = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                timeout=_VERSION_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as e:
            log.debug("version probe for %s failed: %s", binary, e)
            return None

        for stream in (proc.stderr or "", proc.stdout or ""):
            match = _VERSION_RE.search(stream)
            if match:
                return match.group(1)
        return None

    # --- launching --------------------------------------------------------

    def build_argv(self, spec: RuntimeSpec, binary: DiscoveredBinary, port: int) -> list[str]:
        argv = [
            str(binary.path),
            "--model",
            spec.modelPath,
            "--host",
            spec.host or "127.0.0.1",
            "--port",
            str(port),
        ]

        # The alias is what the engine reports as its model id, which is
        # what the driver reports to the gateway, which is what a client
        # asks for. Defaulting it to the filename is how "the name you
        # see is the name of the file you downloaded" actually holds.
        alias = spec.modelAlias or default_model_alias(spec.modelPath)
        argv += ["--alias", alias]

        flags = spec.flags or {}
        for field in self.flag_schema().fields:
            if field.key not in flags:
                continue
            value = flags[field.key]
            if value is None:
                continue
            cli = _FLAG_CLI_NAMES[field.key]
            if field.valueType == ConfigValueType.boolean:
                # llama-server's booleans are presence-only switches;
                # passing `--flash-attn false` is not a thing.
                if value:
                    argv.append(cli)
            else:
                argv += [cli, str(value)]

        # Verbatim, last, so an operator can always override something the
        # curated surface generated above.
        if spec.extraArgs:
            argv += list(spec.extraArgs)
        return argv

    # --- observing --------------------------------------------------------

    async def probe_readiness(self, base_url: str) -> Readiness:
        """Read `llama-server`'s `/health`.

        Upstream's contract: 503 with `{"status": "loading model"}` while
        the weights are being read, 200 with `{"status": "ok"}` once it
        can serve. Distinguishing those is the whole reason readiness is
        per-adapter rather than a generic TCP connect.
        """
        url = base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{url}/health")
        except httpx.HTTPError as e:
            return NotAnswering(detail=str(e))

        if response.status_code == 503:
            return Loading(detail=_status_text(response) or "loading model")
        if not response.is_success:
            return NotAnswering(detail=f"/health returned {response.status_code}")

        status = _status_text(response)
        if status and status != "ok":
            # Unknown non-ok status: treat as still coming up rather than
            # asserting readiness we can't vouch for.
            return Loading(detail=status)

        return Ready(capabilities=await self._read_capabilities(url))

    async def _read_capabilities(self, url: str) -> RuntimeCapabilities | None:
        """Read back what the engine actually resolved, from `/props`.

        Deliberately read rather than inferred from the spec: a requested
        context larger than the model or than free memory gets clamped by
        the engine, and the clamped number is the true one. Best-effort —
        a runtime that is serving but won't describe itself is still
        ready.
        """
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{url}/props")
            if not response.is_success:
                return None
            props = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
            log.debug("capability read from %s failed: %s", url, e)
            return None
        if not isinstance(props, dict):
            return None

        default_generation = props.get("default_generation_settings")
        context_length = None
        if isinstance(default_generation, dict):
            context_length = default_generation.get("n_ctx")
        if context_length is None:
            context_length = props.get("n_ctx")

        slots = props.get("total_slots")
        modalities = props.get("modalities")
        multimodal = None
        if isinstance(modalities, dict):
            multimodal = bool(modalities.get("vision") or modalities.get("audio"))

        return RuntimeCapabilities(
            contextLength=int(context_length) if isinstance(context_length, int) else None,
            parallelSlots=int(slots) if isinstance(slots, int) and slots >= 1 else None,
            embeddings=None,
            multimodal=multimodal,
        )

    # --- configuring ------------------------------------------------------

    def flag_schema(self) -> ConfigSchema:
        return ConfigSchema(
            component="engine:llama_cpp",
            categories=_CATEGORIES,
            fields=_FLAG_FIELDS,
        )


def default_model_alias(model_path: str) -> str:
    """Filename with its extension stripped.

    Because models are stored as plainly-named files in directories the
    user chose, the obvious name is already the right one — so this is
    what `modelAlias` defaults to, and setting it is an override rather
    than a requirement. Multi-file formats point at a directory, whose
    name is the model name.
    """
    path = Path(model_path)
    if path.suffix.lower() == ".gguf":
        return path.stem
    return path.name or path.stem


def _status_text(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(body, dict):
        status = body.get("status")
        if isinstance(status, str):
            return status
    return None


_CATEGORIES = {
    "memory": "Memory and context",
    "performance": "Performance",
    "sampling": "Serving",
}

# Curated flags -> llama-server CLI names. Kept as a separate mapping from
# the schema so the UI-facing key never has to look like a CLI flag, and
# so an upstream rename touches one line.
_FLAG_CLI_NAMES: dict[str, str] = {
    "contextSize": "--ctx-size",
    "gpuLayers": "--n-gpu-layers",
    "batchSize": "--batch-size",
    "ubatchSize": "--ubatch-size",
    "threads": "--threads",
    "parallelSlots": "--parallel",
    "mainGpu": "--main-gpu",
    "tensorSplit": "--tensor-split",
    "flashAttention": "--flash-attn",
    "mlock": "--mlock",
    "noMmap": "--no-mmap",
    "continuousBatching": "--cont-batching",
}

_FLAG_FIELDS: list[ConfigField] = [
    ConfigField(
        key="contextSize",
        label="Context size",
        description=(
            "Maximum tokens the model can attend to, per slot. The single "
            "biggest lever on memory use. Leave unset to take the model's "
            "trained context — the engine clamps anything larger than the "
            "model or than available memory, and the clamped value is what "
            "the runtime reports back."
        ),
        category="memory",
        valueType=ConfigValueType.integer,
        minimum=256,
        requiresRestart=True,
    ),
    ConfigField(
        key="gpuLayers",
        label="GPU layers",
        description=(
            "How many model layers to offload to the GPU. High enough to "
            "fit is the goal; too high fails at load or spills into system "
            "memory and runs slower than the CPU would. Set to 0 for "
            "CPU-only, or a large number to offload everything."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=0,
        requiresRestart=True,
    ),
    ConfigField(
        key="parallelSlots",
        label="Parallel slots",
        description=(
            "Concurrent requests this runtime serves. Context size is "
            "per-slot, so 4 slots at 8k needs roughly the memory of 1 slot "
            "at 32k — raising this without lowering context is the usual "
            "way to run out of VRAM. This is also the unit of capacity the "
            "gateway divides work across."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        default=1,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="batchSize",
        label="Batch size",
        description=(
            "Logical batch size for prompt processing. Larger speeds up "
            "long prompts at the cost of memory during the prefill."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="ubatchSize",
        label="Micro-batch size",
        description=(
            "Physical batch size actually handed to the GPU at once. Only "
            "worth touching if you are tuning prefill throughput."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="threads",
        label="CPU threads",
        description=(
            "Threads used for CPU-side work. Defaults to the engine's own "
            "guess, which is usually right; matters most when layers are "
            "not fully offloaded."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="flashAttention",
        label="Flash attention",
        description=(
            "Enable the fused attention kernel. Usually faster and lighter "
            "on memory where the build and hardware support it; harmless to "
            "leave off if you are unsure."
        ),
        category="performance",
        valueType=ConfigValueType.boolean,
        default=False,
        requiresRestart=True,
    ),
    ConfigField(
        key="continuousBatching",
        label="Continuous batching",
        description=(
            "Interleave requests across slots instead of serving them one "
            "after another. Wanted whenever more than one slot is in use."
        ),
        category="performance",
        valueType=ConfigValueType.boolean,
        default=True,
        requiresRestart=True,
    ),
    ConfigField(
        key="mainGpu",
        label="Main GPU",
        description=(
            "Index of the GPU to prefer for single-GPU work. To pin a "
            "runtime to one card entirely, set CUDA_VISIBLE_DEVICES in the "
            "runtime's environment instead — that is how two replicas end "
            "up on two cards."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=0,
        requiresRestart=True,
    ),
    ConfigField(
        key="tensorSplit",
        label="Tensor split",
        description=(
            "How to divide the model across multiple GPUs, as "
            "comma-separated proportions (e.g. `0.6,0.4`). Only meaningful "
            "with more than one visible GPU."
        ),
        category="performance",
        valueType=ConfigValueType.string,
        pattern=r"^\s*\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*\s*$",
        requiresRestart=True,
    ),
    ConfigField(
        key="mlock",
        label="Lock model in memory",
        description=(
            "Prevent the OS from paging the model out. Keeps latency "
            "predictable; needs enough RAM to hold the whole thing and may "
            "require elevated permissions."
        ),
        category="memory",
        valueType=ConfigValueType.boolean,
        default=False,
        requiresRestart=True,
    ),
    ConfigField(
        key="noMmap",
        label="Disable mmap",
        description=(
            "Read the model into memory instead of mapping it from disk. "
            "Slower to start and needs more RAM; occasionally necessary on "
            "network filesystems."
        ),
        category="memory",
        valueType=ConfigValueType.boolean,
        default=False,
        requiresRestart=True,
    ),
]
