"""vLLM adapter — drives the operator's own `vllm serve`.

We never install vLLM. Its unit of installation is a Python environment
(an interpreter of a version we do not control, several GB per copy,
three of six targets on a non-default package index, Apple silicon on a
separate project entirely, and no single digest that verifies the
result), so `install_policy` is `manual`, permanently and on every host.
What this adapter adds is the half that makes *driving* it respectable
rather than a shrug: discovery that finds the operator's venv through an
install-wide config path, a refusal that names the exact install command
for the detected host, and an environment readout that says which
PyTorch build is actually in there.

Every claim about vLLM's behaviour below is read off upstream source at
**v0.29.0** (tagged 2026-09-09), file named at each one. **No vLLM
process has run for this project** — the dev box is Windows and vLLM
has no Windows build — so the readiness *timing* here is designed, not
measured. The state machine is unit-tested against a fake process handle
and a fake HTTP probe; the wall-clock budget is not, and the first Linux
acceptance run should treat `startup_budget_seconds` as the number most
likely to be wrong.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import httpx

from .._generated.models import (
    Accelerator,
    ConfigField,
    ConfigSchema,
    ConfigValueType,
    EngineKind,
    FrameworkAccelerator,
    HostAccelerator,
    ManualInstall,
    ModelFormat,
    Origin,
    Os,
    Policy,
    PythonEngine,
    RuntimeCapabilities,
    RuntimeSpec,
)
from .base import (
    DiscoveredBinary,
    EngineAdapter,
    NotAnswering,
    Readiness,
    Ready,
    default_model_alias,
)

log = logging.getLogger(__name__)

# Readiness probes run on the supervisor's poll cadence. Short on purpose:
# during a load vLLM's port is bound but not listening (`launcher.py`
# `create_server_socket` calls `bind()` and returns with no `listen()`),
# so connections are refused instantly and a long timeout would only make
# each poll slow. The budget for the load is a different knob — below.
_PROBE_TIMEOUT_SECONDS = 2.0

# How long a silent load may run before the supervisor flags it. vLLM's
# startup is dominated by `torch.compile` and CUDA graph capture, which
# for a large model is minutes rather than seconds. Ten is a design
# estimate, not a measurement — see the module docstring. `enforceEager`
# is the curated flag that trades throughput for a much shorter start.
STARTUP_BUDGET_SECONDS = 600.0

# Interpreter probe: reads distribution metadata through the environment's
# own Python. Cheap — `importlib.metadata` reads `METADATA` files and
# imports nothing of vLLM's — and layout-agnostic, which a glob over
# `site-packages` is not (venv, uv, conda and `--user` all differ).
_ENV_PROBE_TIMEOUT_SECONDS = 10.0
_ENV_PROBE_SCRIPT = """
import importlib.metadata as m, json, sys
def v(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return None
py = "%d.%d.%d" % sys.version_info[:3]
print(json.dumps({"python": py, "vllm": v("vllm"), "torch": v("torch")}))
""".strip()

# Upstream's own install pages. `docsUrl` is the one field of a
# `ManualInstall` that always exists, because upstream's instructions are
# correct for longer than any copy we make of them.
INSTALL_DOCS_URL = "https://docs.vllm.ai/en/latest/getting_started/installation/"
GPU_INSTALL_DOCS_URL = INSTALL_DOCS_URL + "gpu/"
CPU_INSTALL_DOCS_URL = INSTALL_DOCS_URL + "cpu/"
VLLM_METAL_URL = "https://github.com/vllm-project/vllm-metal"

# The step every install path ends with, and the reason `vllmBinary`
# exists at all.
_THEN_CONFIGURE = (
    "Then set `vllmBinary` in the agent's config to that environment's `bin/vllm` "
    "console script — its shebang binds its own interpreter, so nothing needs "
    "activating."
)
_VENV_FIRST = (
    "Run inside the environment that will hold vLLM: "
    "`uv venv --python 3.12 --seed && source .venv/bin/activate`. "
)


class VllmAdapter(EngineAdapter):
    kind = EngineKind.vllm
    binary_name = "vllm"

    # safetensors only. Upstream's GGUF path exists and is documented as
    # "highly experimental and under-optimized", needing a second
    # `--tokenizer` model because converting a GGUF tokenizer is unstable.
    # Listing `gguf` here would light up a launch button across the whole
    # GGUF population llama.cpp already serves properly.
    model_formats = (ModelFormat.safetensors,)

    # The load-bearing difference from llama-server. vLLM binds its
    # listening socket before the engine initialises (upstream issue
    # #8204, to avoid a Ray race) and does not `listen()` until the model
    # is resident: `launchers/api_server/entry.py` orders it socket →
    # engine load → `build_and_serve`. So a TCP probe cannot tell
    # `loading` from `crashed` for minutes, and the supervisor — which
    # holds the pid — is the only thing that can. `interpret_readiness`
    # in `base.py` is where "alive and refusing connections" becomes
    # `Loading`.
    answers_while_loading = False
    startup_budget_seconds = STARTUP_BUDGET_SECONDS

    install_policy = Policy.manual
    configured_binary_key = "vllmBinary"

    # --- discovery --------------------------------------------------------

    def describe(self, path: Path, origin: Origin) -> DiscoveredBinary:
        """Inspect the console script's environment once, for both the
        version and the `python` block on the descriptor."""
        env = inspect_python_environment(path)
        return DiscoveredBinary(
            path=path,
            origin=origin,
            version=env.packageVersion if env is not None else None,
            python=env,
        )

    def probe_version(self, binary: Path) -> str | None:
        """The package version, from distribution metadata.

        Never `vllm --version`: upstream registers it as an argparse
        `action="version"` behind a module chain that imports vLLM and
        therefore PyTorch (`entrypoints/cli/main.py`), and paying a torch
        import to render a settings panel is not acceptable. Same trade
        the managed store makes for llama.cpp, for a different reason.
        """
        env = inspect_python_environment(binary)
        return env.packageVersion if env is not None else None

    def default_env(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> dict[str, str]:
        """The two variables vLLM 0.29.0 cannot start without here.

        Both were found by running the engine on a host nobody had
        prepared, and both killed it 20-40s into a load from inside a
        subprocess, with a traceback that named neither the cause nor the
        cure. Neither is a preference: without them there is no engine.

        `VLLM_WSL2_ENABLE_PIN_MEMORY` — on WSL2 vLLM disables pinned
        memory by default (a small performance regression on that
        kernel), and 0.29.0's model runner then hard-requires it anyway
        via a UVA buffer, because `is_uva_available()` *is*
        `is_pin_memory_available()`. The result is
        `RuntimeError: UVA is not available`. Upstream owns the switch
        and gates it on a kernel floor of 4.19.121, so setting it on an
        older kernel changes nothing — upstream still refuses, and that
        refusal is not ours to override.

        `VLLM_USE_FLASHINFER_SAMPLER` — FlashInfer JIT-compiles its
        *sampling* kernels and wants a full CUDA toolkit; with no `nvcc`
        the engine dies with
        `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`.
        Note it is the sampler and not attention: attention picked
        prebuilt FlashAttention 2 and was never the problem. Turning it
        off falls back to PyTorch-native sampling, which is *correct* and
        slightly slower on large batches — so this one does trade a
        little throughput, and it is still the right default, because the
        thing it is traded against is not starting. An operator who
        installs a toolkit gets the fast path back automatically, and one
        who wants it without a toolkit can set the variable to `1` and
        watch it fail on purpose.
        """
        env: dict[str, str] = {}
        if _is_wsl():
            env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
        if not _cuda_toolkit_present():
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        return env

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        """Name the host prerequisite behind a known startup death.

        vLLM compiles at *first use* rather than at install, so a
        perfectly good `pip install` leaves three ways for the engine to
        die on a clean host. The traceback is forty frames of someone
        else's code and mentions the fix in none of them.
        """
        for signature, explanation in _EXIT_EXPLANATIONS:
            if signature in output_tail:
                return explanation
        return None

    def manual_install(self, host: HostAccelerator) -> ManualInstall:
        """The install command for this host, copied from upstream's
        v0.29.0 install pages — never composed. A command that does not
        work is worse than no command, because the operator will believe
        it; where upstream names none, `command` is absent and `notes`
        says why."""
        return manual_install_for(host)

    # --- launching --------------------------------------------------------

    def build_argv(self, spec: RuntimeSpec, binary: DiscoveredBinary, port: int) -> list[str]:
        # The model is positional: `vllm serve [model_tag] [options]`
        # (`entrypoints/cli/serve.py`, which copies `model_tag` onto
        # `args.model` and hides `--model` from `vllm serve --help`).
        argv = [
            str(binary.path),
            "serve",
            spec.modelPath,
            "--host",
            spec.host or "127.0.0.1",
            "--port",
            str(port),
        ]

        # Always. vLLM's `served_model_name` defaults to the `--model`
        # argument verbatim (`config/model.py`), and we launch by
        # absolute path — so leaving it unset would publish
        # `/home/you/models/Qwen3-8B` as an OpenAI model id: the
        # operator's directory layout leaked to every client, and a
        # routing key that differs per host for the same model.
        alias = spec.modelAlias or default_model_alias(spec.modelPath)
        argv += ["--served-model-name", alias]

        flags = spec.flags or {}
        for field in self.flag_schema().fields:
            if field.key not in flags:
                continue
            value = flags[field.key]
            if value is None:
                continue
            cli = _FLAG_CLI_NAMES[field.key]
            if field.valueType == ConfigValueType.boolean:
                # vLLM's booleans are `argparse.BooleanOptionalAction`
                # (`engine/arg_utils.py` `_compute_kwargs`): `--enforce-eager`
                # turns it on, `--no-enforce-eager` off, and neither takes
                # a value. Every curated boolean defaults to off upstream,
                # so absent is the same as false.
                if value:
                    argv.append(cli)
            else:
                argv += [cli, str(value)]

        # Verbatim, last, so an operator can always override something the
        # curated surface generated above.
        if spec.extraArgs:
            argv += list(spec.extraArgs)
        return argv

    def working_directory(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> str | None:
        """Inherit the agent's cwd unless told otherwise.

        The base default — the binary's own directory — exists for
        prebuilt llama.cpp releases that keep shared libraries beside the
        executable. A console script has no such need, and running from
        the venv's `bin/` would be a strange place to leave any
        relative-path output.
        """
        return spec.workingDirectory or None

    # --- observing --------------------------------------------------------

    async def probe_readiness(self, base_url: str) -> Readiness:
        """Read vLLM's `/health`.

        Thinner than llama-server's (`serve/instrumentator/health.py`):
        **200 with an empty body**, or 503 only on `EngineDeadError`.
        There is no loading status to read back, and no capabilities.
        Unguarded by `--api-key` (`serve/middleware/authenticate.py`
        guards only `/v1`, `/v2`, `/inference`, `/cohere`), so the probe
        needs no credential.

        During the load this call gets connection refused, and that is
        reported as `NotAnswering(reached=False)` — *not* as loading.
        This probe sees only the network; the supervisor, which knows
        whether the pid is alive, makes the call. See `base.py`.
        """
        url = base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{url}/health")
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            return NotAnswering(detail=str(e), reached=False)
        except httpx.HTTPError as e:
            # Connected, then nothing usable came back. Not silence.
            return NotAnswering(detail=str(e), reached=True)

        if response.status_code == 503:
            # Not loading. A 503 here is `EngineDeadError`: the API
            # server is up and the engine core behind it has died. Expect
            # the process to exit shortly and the supervisor to report it.
            return NotAnswering(
                detail=(
                    "/health returned 503: vLLM's engine core is dead (EngineDeadError). "
                    "The API server is up but nothing behind it can serve; check the "
                    "captured engine output."
                ),
                reached=True,
            )
        if not response.is_success:
            return NotAnswering(detail=f"/health returned {response.status_code}", reached=True)

        version, capabilities = await self._read_back(url)
        return Ready(capabilities=capabilities, version=version)

    async def _read_back(self, url: str) -> tuple[str | None, RuntimeCapabilities | None]:
        """What a serving vLLM will tell us about itself. Best-effort.

        `/version` is `{"version": "0.29.0"}` (`serve/instrumentator/basic.py`).
        `GET /v1/models` carries `max_model_len` on each card
        (`entrypoints/openai/models/serving.py`), which is the effective
        context as the engine resolved it — read rather than inferred
        from `maxModelLen`, because `auto` and a too-large request both
        get resolved by the engine, and the resolved number is the true
        one. Nothing reports the sequence budget back, so `parallelSlots`
        stays unknown rather than being copied off the spec.

        `/v1/models` is behind `--api-key` if the operator set one via
        `extraArgs`; a 401 there just leaves capabilities empty. A runtime
        that is serving but will not describe itself is still ready.
        """
        version: str | None = None
        capabilities: RuntimeCapabilities | None = None
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                v = await client.get(f"{url}/version")
                if v.is_success:
                    body = v.json()
                    if isinstance(body, dict) and isinstance(body.get("version"), str):
                        version = body["version"]
                models = await client.get(f"{url}/v1/models")
                if models.is_success:
                    capabilities = _capabilities_from_models(models.json())
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
            log.debug("read-back from %s failed: %s", url, e)
        return version, capabilities

    # --- configuring ------------------------------------------------------

    def flag_schema(self) -> ConfigSchema:
        return ConfigSchema(
            component="engine:vllm",
            categories=_CATEGORIES,
            fields=_FLAG_FIELDS,
        )


# --------------------------------------------------------------------------- #
# Environment inspection
# --------------------------------------------------------------------------- #


def inspect_python_environment(binary: Path) -> PythonEngine | None:
    """Describe the environment a `vllm` console script belongs to.

    The script's shebang names the interpreter; the interpreter's own
    `importlib.metadata` names the versions. Nothing of vLLM is imported.
    None when the file has no readable shebang (a Windows console-script
    launcher is a PE binary, and vLLM does not run there anyway) — the
    descriptor then carries no `python` block rather than a made-up one.
    """
    interpreter = interpreter_from_shebang(binary)
    if interpreter is None:
        return None
    probed = _probe_interpreter(interpreter)
    torch_version = probed.get("torch") if probed else None
    return PythonEngine(
        interpreter=interpreter,
        pythonVersion=probed.get("python") if probed else None,
        packageVersion=probed.get("vllm") if probed else None,
        torchVersion=torch_version,
        accelerator=accelerator_from_torch_version(torch_version),
    )


def interpreter_from_shebang(script: Path) -> str | None:
    """The interpreter a console script runs under, from its first line.

    Handles both forms setuptools and uv write — an absolute path, or
    `/usr/bin/env python3` resolved through PATH. A binary or an
    unreadable file yields None.

    Returned as the string the shebang wrote rather than a `Path`, so
    the descriptor reports exactly what the script will execute — and
    so a POSIX shebang read on a Windows dev box is not rewritten with
    backslashes on the way out.
    """
    try:
        with script.open("rb") as f:
            first = f.readline(4096)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    line = first[2:].decode("utf-8", errors="replace").strip()
    if not line:
        return None
    parts = line.split()
    if parts[0].endswith("/env") and len(parts) > 1:
        return shutil.which(parts[1])
    return parts[0]


def _probe_interpreter(interpreter: str) -> dict[str, str | None] | None:
    try:
        proc = subprocess.run(
            [interpreter, "-I", "-c", _ENV_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=_ENV_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("environment probe via %s failed: %s", interpreter, e)
        return None
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        parsed = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {
        key: (str(value) if isinstance(value, str) else None)
        for key, value in parsed.items()
        if key in {"python", "vllm", "torch"}
    }


def accelerator_from_torch_version(version: str | None) -> FrameworkAccelerator | None:
    """What the installed PyTorch was built for, from its local version tag.

    `2.9.0+cu129` is CUDA, `+rocm7.0` is ROCm, `+xpu` is Intel, `+cpu` is
    a CPU-only build. A version with **no tag** is `unknown`, not `none`:
    PyPI forbids local version tags, so PyPI's own Linux wheel — which
    *is* a CUDA build — reports a bare `2.9.0`, and so does the CPU-only
    macOS wheel. Reporting `none` there would claim a definite absence
    of acceleration that the data does not support.
    """
    if version is None:
        return None
    _, plus, tag = version.partition("+")
    if not plus or not tag:
        return FrameworkAccelerator.unknown
    tag = tag.lower()
    if tag.startswith("cu"):
        return FrameworkAccelerator.cuda
    if tag.startswith("rocm"):
        return FrameworkAccelerator.rocm
    if tag.startswith("xpu"):
        return FrameworkAccelerator.xpu
    if tag.startswith("cpu"):
        return FrameworkAccelerator.none
    return FrameworkAccelerator.unknown


def _capabilities_from_models(body: object) -> RuntimeCapabilities | None:
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    context = first.get("max_model_len")
    return RuntimeCapabilities(
        contextLength=int(context) if isinstance(context, int) and context > 0 else None,
        parallelSlots=None,
        embeddings=None,
        multimodal=None,
    )


# --------------------------------------------------------------------------- #
# Manual install
# --------------------------------------------------------------------------- #


def manual_install_for(host: HostAccelerator) -> ManualInstall:
    """Upstream's install command for this host, verbatim.

    Commands and constraints come from
    `docs/getting_started/installation/{gpu.cuda,gpu.rocm,gpu.xpu,gpu.apple,cpu.*}.inc.md`
    at v0.29.0. Where upstream names no single command — Windows, Apple
    silicon, an unrecognised host — `command` is omitted and `notes`
    carries the way in.
    """
    if host.os is Os.windows:
        return ManualInstall(
            docsUrl=INSTALL_DOCS_URL,
            notes=(
                "vLLM has no Windows build and upstream's answer is WSL. Install a "
                "Linux distribution under WSL2, install vLLM inside it, and run this "
                "agent there too — a runtime has to be supervised from the host it "
                "runs on."
            ),
        )
    if host.os is Os.macos:
        return ManualInstall(
            docsUrl=GPU_INSTALL_DOCS_URL,
            notes=(
                f"Apple silicon is served by vLLM-Metal ({VLLM_METAL_URL}), a separate "
                "community project with its own wheels and an MLX compute backend. "
                "This agent does not drive it: whether it is the same adapter is an "
                "open question with nothing to answer it yet."
            ),
        )
    if host.os is None:
        return ManualInstall(
            docsUrl=INSTALL_DOCS_URL,
            notes=(
                "Could not identify this operating system, so there is no way to name "
                "the right wheel. Upstream's install index lists every target."
            ),
        )

    # Linux.
    accelerator = host.accelerator or Accelerator.none
    if accelerator is Accelerator.cuda:
        return ManualInstall(
            command="uv pip install vllm --torch-backend=auto",
            docsUrl=GPU_INSTALL_DOCS_URL,
            notes=(
                _VENV_FIRST + "The default wheel is built against CUDA 12.9 and bundles "
                "PyTorch; `--torch-backend=auto` picks the PyTorch build matching the "
                "installed driver, so no CUDA version needs choosing by hand. Blackwell "
                "GPUs need CUDA 12.8 or newer. " + _THEN_CONFIGURE
            ),
        )
    if accelerator is Accelerator.rocm:
        return ManualInstall(
            command="uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade",
            docsUrl=GPU_INSTALL_DOCS_URL,
            notes=(
                _VENV_FIRST + "ROCm wheels exist for Python 3.12 only — on any other "
                "version the installer silently falls back to the CUDA wheel, which "
                "fails on AMD GPUs with a missing libcudart. Needs ROCm 6.3 or newer; "
                "prebuilt wheels are for ROCm 7.0 and 7.2.1. Use uv rather than pip: "
                "pip merges the custom index with PyPI and picks the wrong wheel. "
                + _THEN_CONFIGURE
            ),
        )
    if accelerator is Accelerator.sycl:
        return ManualInstall(
            command=(
                "uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/xpu "
                "--extra-index-url https://download.pytorch.org/whl/xpu "
                "--index-strategy unsafe-best-match"
            ),
            docsUrl=GPU_INSTALL_DOCS_URL,
            notes=(
                _VENV_FIRST + "Intel XPU wheels are nightly builds on a custom index plus a "
                "second index for PyTorch XPU, and need Python 3.12. " + _THEN_CONFIGURE
            ),
        )
    if accelerator is Accelerator.none:
        arch = "aarch64" if (host.arch is not None and host.arch.value == "arm64") else "x86_64"
        return ManualInstall(
            command=(
                "uv pip install https://github.com/vllm-project/vllm/releases/download/"
                "v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_"
                f"{arch}.whl --torch-backend cpu"
            ),
            docsUrl=CPU_INSTALL_DOCS_URL,
            notes=(
                _VENV_FIRST + "Set VLLM_VERSION to a release tag from "
                "https://github.com/vllm-project/vllm/releases (this adapter was checked "
                "against 0.29.0). CPU wheels exist since 0.17.0 for x86 with AVX512/AVX2 "
                "and since 0.11.2 for Arm. On x86, upstream says to add Intel OpenMP to "
                "LD_PRELOAD before running. No GPU was detected on this host, so this is "
                "the CPU build — expect it to be slow. " + _THEN_CONFIGURE
            ),
        )
    # An accelerator this adapter has no upstream command for (`metal` on
    # Linux cannot happen; anything new lands here).
    return ManualInstall(
        docsUrl=GPU_INSTALL_DOCS_URL,
        notes=(
            f"No upstream install command is known for accelerator {accelerator.value!r} "
            "on Linux. Upstream's GPU install page lists every supported target."
        ),
    )


# --------------------------------------------------------------------------- #
# Curated flags
# --------------------------------------------------------------------------- #

_CATEGORIES = {
    "memory": "Memory and context",
    "parallelism": "Parallelism",
    "performance": "Performance",
    "model": "Model loading",
}

# Curated flags -> `vllm serve` CLI names. Every name checked against
# `vllm/engine/arg_utils.py` `add_cli_args` at v0.29.0, where each is an
# `add_argument` over the matching config dataclass field. Kept as a
# separate mapping from the schema so the UI-facing key never has to look
# like a CLI flag, and so an upstream rename touches one line.
_FLAG_CLI_NAMES: dict[str, str] = {
    "maxModelLen": "--max-model-len",  # ModelConfig.max_model_len
    "gpuMemoryUtilization": "--gpu-memory-utilization",  # CacheConfig.gpu_memory_utilization
    "tensorParallelSize": "--tensor-parallel-size",  # ParallelConfig.tensor_parallel_size
    "pipelineParallelSize": "--pipeline-parallel-size",  # ParallelConfig.pipeline_parallel_size
    "maxNumSeqs": "--max-num-seqs",  # SchedulerConfig.max_num_seqs
    "maxNumBatchedTokens": "--max-num-batched-tokens",  # SchedulerConfig.max_num_batched_tokens
    "dtype": "--dtype",  # ModelConfig.dtype
    "quantization": "--quantization",  # ModelConfig.quantization
    "kvCacheDtype": "--kv-cache-dtype",  # CacheConfig.cache_dtype
    "enforceEager": "--enforce-eager",  # ModelConfig.enforce_eager
    "trustRemoteCode": "--trust-remote-code",  # ModelConfig.trust_remote_code
    "tokenizer": "--tokenizer",  # ModelConfig.tokenizer
}

_FLAG_FIELDS: list[ConfigField] = [
    ConfigField(
        key="maxModelLen",
        label="Max model length",
        description=(
            "The context window in tokens — prompt plus output — and the "
            "number every fit verdict in the library is computed against. "
            "The direct analogue of llama.cpp's context size. Leave unset to "
            "take the model's own configured length; vLLM also accepts `auto` "
            "(via extraArgs: `--max-model-len auto`) to pick the largest length "
            "that fits in GPU memory. The value the engine actually resolved is "
            "read back onto the runtime's capabilities."
        ),
        category="memory",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="gpuMemoryUtilization",
        label="GPU memory utilization",
        description=(
            "Fraction of each GPU vLLM preallocates for this instance, 0 to 1. "
            "The engine default is 0.92. vLLM reserves a *fraction of the card* "
            "rather than counting layers, so this is the flag that decides "
            "whether a second runtime fits on the same GPU: two instances on one "
            "card want roughly 0.45 each. No llama.cpp counterpart."
        ),
        category="memory",
        valueType=ConfigValueType.number,
        minimum=0.01,
        maximum=1.0,
        requiresRestart=True,
    ),
    ConfigField(
        key="kvCacheDtype",
        label="KV cache dtype",
        description=(
            "Storage type for the KV cache. `auto` matches the model dtype; "
            "`fp8` roughly halves KV memory at long context, which is the "
            "single biggest fit lever vLLM has. `fp8_e4m3` and `fp8_e5m2` name "
            "the two FP8 formats explicitly. Hardware-specific variants "
            "(`fp8_inc` on Gaudi, and others) go through extraArgs."
        ),
        category="memory",
        valueType=ConfigValueType.enum,
        enumValues=["auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"],
        requiresRestart=True,
    ),
    ConfigField(
        key="tensorParallelSize",
        label="Tensor parallel size",
        description=(
            "Shard one model across this many GPUs in a single process. This "
            "is one runtime, one model, N cards — it is *not* replication. Two "
            "replicas on two cards is two runtimes, each pinned with "
            "CUDA_VISIBLE_DEVICES in its environment, and the gateway balances "
            "across those; a tensor-parallel runtime reports as one backend "
            "because it is one."
        ),
        category="parallelism",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="pipelineParallelSize",
        label="Pipeline parallel size",
        description=(
            "Split the model's layers across this many GPUs in sequence. The "
            "other parallelism axis, for a model too large for tensor "
            "parallelism alone; the two multiply."
        ),
        category="parallelism",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="maxNumSeqs",
        label="Max concurrent sequences",
        description=(
            "How many sequences one scheduling step may hold — vLLM's name for "
            "the unit of capacity the gateway divides work across (parallel "
            "slots on llama.cpp). Raising it raises throughput and KV cache "
            "pressure together."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="maxNumBatchedTokens",
        label="Max batched tokens",
        description=(
            "Token budget per scheduling step across all sequences. The "
            "throughput-versus-latency knob operators reach for after the "
            "sequence count; larger favours prefill throughput at the cost of "
            "decode latency."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="dtype",
        label="Compute dtype",
        description=(
            "Data type for weights and activations. `auto` is usually right "
            "(bf16 where the model was trained in it). `float16` is the standard "
            "workaround for a card without bf16 support; `float32` is for "
            "debugging a numerical problem, not for serving."
        ),
        category="performance",
        valueType=ConfigValueType.enum,
        enumValues=["auto", "half", "float16", "bfloat16", "float", "float32"],
        requiresRestart=True,
    ),
    ConfigField(
        key="enforceEager",
        label="Enforce eager mode",
        description=(
            "Skip CUDA graph capture and torch.compile. Costs steady-state "
            "throughput and buys a much shorter start — for a large model the "
            "compile and capture phases are most of the minutes a runtime sits "
            "at `loading`. The first thing to try when startup itself is the "
            "problem."
        ),
        category="performance",
        valueType=ConfigValueType.boolean,
        default=False,
        requiresRestart=True,
    ),
    ConfigField(
        key="quantization",
        label="Quantization method",
        description=(
            "Normally left unset: vLLM reads `quantization_config` from the "
            "model's config.json and picks the method itself (AWQ, GPTQ, FP8, "
            "compressed-tensors and the rest). Set it explicitly only for a "
            "checkpoint that does not declare its own, or to force an "
            "on-the-fly method such as `fp8`. The engine validates the name at "
            "spawn."
        ),
        category="model",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    ),
    ConfigField(
        key="trustRemoteCode",
        label="Trust remote code",
        description=(
            "Allow the model directory's own Python modelling code to run. "
            "Required by a real share of HuggingFace architectures. **This "
            "executes code shipped inside the model folder with the engine's "
            "privileges** — enable it only for a model you would run as a "
            "script."
        ),
        category="model",
        valueType=ConfigValueType.boolean,
        default=False,
        requiresRestart=True,
    ),
    ConfigField(
        key="tokenizer",
        label="Tokenizer path",
        description=(
            "Path to a tokenizer when it does not live in the model directory. "
            "Defaults to the model's own."
        ),
        category="model",
        valueType=ConfigValueType.file_path,
        requiresRestart=True,
    ),
]


# Startup deaths worth explaining, newest-first in the order they were
# met on the first Linux run. Matched against a tail of the engine's own
# output, so each signature is a string upstream actually prints.
_EXIT_EXPLANATIONS: tuple[tuple[str, str], ...] = (
    (
        "Failed to find C compiler",
        "vLLM died because this host has no C compiler. Triton compiles a "
        "small CPython extension the first time it runs, so vLLM needs a "
        "toolchain at *run* time even though `pip install` did not: "
        "`apt install build-essential` (or set CC). A host that has "
        "already run this engine once has the compiled result cached and "
        "does not need one.",
    ),
    (
        "Python.h: No such file or directory",
        "vLLM died because this host has no Python development headers. "
        "Triton compiles a CPython extension at first use and needs "
        "`Python.h` for the interpreter behind the engine: "
        "`apt install python3-dev` (matching the engine's own Python "
        "version).",
    ),
    (
        "Could not find nvcc",
        "vLLM died looking for a CUDA toolkit, which FlashInfer needs to "
        "JIT-compile its sampling kernels. Either install a toolkit, or "
        "set `VLLM_USE_FLASHINFER_SAMPLER=0` in the runtime's `env` for "
        "native sampling — the agent sets that automatically when no "
        "`nvcc` is visible, so seeing this means something overrode it.",
    ),
    (
        "UVA is not available",
        "vLLM died because pinned memory is unavailable. On WSL2 it is "
        "disabled by default and this version requires it: set "
        "`VLLM_WSL2_ENABLE_PIN_MEMORY=1` in the runtime's `env` — the "
        "agent sets it automatically on WSL2, so seeing this means either "
        "something overrode it or the WSL2 kernel is below upstream's "
        "4.19.121 floor, which `wsl --update` fixes.",
    ),
)


def _is_wsl() -> bool:
    """True on a WSL kernel.

    Read from `/proc/version` rather than inferred from the environment:
    `WSL_DISTRO_NAME` is set for interactive shells and an agent started
    by a service manager may not have it, while the kernel string is
    always there. Absent or unreadable means not WSL, which is the safe
    answer — the default this gates is inert off WSL anyway.
    """
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def _cuda_toolkit_present() -> bool:
    """True when something in this host looks like a CUDA toolkit with
    `nvcc` in it.

    Mirrors FlashInfer's own `get_cuda_path`: the `CUDA_HOME` /
    `CUDA_PATH` environment variables first, then `/usr/local/cuda`, and
    additionally `nvcc` on PATH, which FlashInfer finds via `which` too.
    Deliberately generous — a false *positive* here leaves FlashInfer
    enabled and lets upstream produce its own error, while a false
    negative silently downgrades sampling on a host that did not need it.
    """
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root and (Path(root) / "bin" / "nvcc").exists():
            return True
    if (Path("/usr/local/cuda") / "bin" / "nvcc").exists():
        return True
    return shutil.which("nvcc") is not None


__all__ = [
    "STARTUP_BUDGET_SECONDS",
    "VllmAdapter",
    "_cuda_toolkit_present",
    "_is_wsl",
    "accelerator_from_torch_version",
    "inspect_python_environment",
    "interpreter_from_shebang",
    "manual_install_for",
]
