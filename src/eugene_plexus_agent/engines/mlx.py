"""The MLX adapter: `mlx_lm.server` on Apple silicon.

The third engine, and the first one written **without a host to run it
on**. Every claim below is checked against mlx-lm v0.31.3 source and
cited to the function that makes it true, because that is the only
verification available here - this project's box is Windows + an RTX
5090, and `mlx` is Apple-silicon-only (`setup.py`:
`mlx>=...; platform_system == 'Darwin'`). M4's lesson was that the vLLM
adapter needed no change on first contact *because* each claim had been
read off upstream first; this leans on that method entirely rather than
partly.

**What makes this engine awkward, and it is not argv.** `mlx_lm.server`
is the only one of the three that cannot be asked whether it is ready:

  * `run()` builds a `ResponseGenerator` - whose thread begins with
    `self.model_provider.load_default()` - and then calls
    `_run_http_server()`. The HTTP server is therefore serving *while
    the model loads on another thread*.
  * `handle_health_check` writes a hardcoded `{"status": "ok"}`. It
    mentions no model and consults no state.
  * `handle_models_request` scans the Hugging Face cache directory and
    appends `--model` if the path exists. Also independent of what is
    resident.

So the only proof that a model is loaded is **making it generate**. This
adapter asks for one token. That is a side effect in a readiness probe
and it is deliberate; see `probe_readiness` for the cost and the open
question that goes with it.

The compensation is that this adapter needs no help from the supervisor.
vLLM's `Loading` has to be inferred from a live pid plus a refused
connection, because a loading vLLM is indistinguishable over the network
from a dead one. A loading MLX is perfectly distinguishable: it answers
`/health` and will not generate. `probe_readiness` returns `Loading`
itself, from network evidence alone, which is what the base class asks a
probe to be.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import httpx

from .._generated.models import (
    Arch,
    ConfigField,
    ConfigSchema,
    ConfigValueType,
    EngineKind,
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
from .base import DiscoveredBinary, EngineAdapter, Loading, NotAnswering, Readiness, Ready
from .vllm import inspect_python_environment

log = logging.getLogger(__name__)

# Short: a probe that cannot answer quickly should say "not yet" and be
# asked again, rather than hold the poll open.
_PROBE_TIMEOUT_SECONDS = 5.0

# The readiness completion. Long enough that a ready engine finishes one
# token comfortably, short enough that a poll is still a poll.
_GENERATE_TIMEOUT_SECONDS = 8.0

# The version this adapter's every claim was read against.
UPSTREAM_VERSION_CHECKED = "0.31.3"

# Upstream's own sentinel, not ours. `APIHandler` reads
# `self.body.get("model", "default_model")`, and `ModelProvider.__init__`
# seeds `_model_map["default_model"] = cli_args.model`, so this resolves
# to whatever `--model` named without us restating the path.
DEFAULT_MODEL_SENTINEL = "default_model"

INSTALL_DOCS_URL = "https://github.com/ml-explore/mlx-lm"

# `mlx-lm` is pure Python and installs anywhere; `mlx` is the part that
# is Darwin-only, and `setup.py` guards it with an environment marker
# rather than a platform classifier. The consequence is the trap worth
# naming: `pip install mlx-lm` SUCCEEDS on Linux and Windows, installs
# no `mlx` at all, and the failure arrives later as an ImportError from
# a server that looked correctly installed.
_NOT_APPLE_NOTES = (
    "mlx-lm runs only on Apple silicon. `pip install mlx-lm` will succeed on this "
    "host anyway - upstream guards its `mlx` dependency with "
    "`platform_system == 'Darwin'`, so on any other platform you get the server "
    "package with none of the engine behind it, and the failure shows up as an "
    "ImportError at launch rather than at install."
)

_ENV_PROBE_TIMEOUT_SECONDS = 10.0
_ENV_PROBE_SCRIPT = """
import importlib.metadata as m, json
def v(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return None
print(json.dumps({"mlx_lm": v("mlx-lm"), "mlx": v("mlx")}))
""".strip()


class MlxAdapter(EngineAdapter):
    kind = EngineKind.mlx

    #: `setup.py` console_scripts: `mlx_lm.server = mlx_lm.server:main`.
    #: The dot is part of the script name, not a file extension.
    binary_name = "mlx_lm.server"

    #: MLX models are safetensors plus a `config.json`, which is what
    #: `handle_models_request` looks for when deciding a cached repo is
    #: "probably mlx_lm". Whether a *vanilla* (non-MLX-converted) HF
    #: safetensors model loads is the first thing the live test should
    #: settle - see the verification doc. We claim the format, not the
    #: conversion.
    model_formats = (ModelFormat.safetensors,)

    #: True, and honestly so: the HTTP server really does answer while
    #: the model loads. This is NOT vLLM's case dressed differently -
    #: nothing here needs the supervisor's process handle, because this
    #: adapter can tell loading from ready by itself.
    answers_while_loading = True

    #: Left unset on purpose. The base uses it only for the silent-load
    #: case (`answers_while_loading = False`), which is not this engine.
    #: A stall therefore shows as a persistent `loading` carrying the
    #: probe's own detail. Flagging one on a clock would need a base
    #: change; noted in the verification doc rather than smuggled in.
    startup_budget_seconds = None

    #: A Python package, like vLLM, for the same reason: the unit of
    #: installation is an environment, which is not a thing we can fetch
    #: and hash.
    install_policy = Policy.manual
    configured_binary_key = "mlxBinary"

    # --- discovery --------------------------------------------------------

    def describe(self, path: Path, origin: Origin) -> DiscoveredBinary:
        """Read the environment the console script belongs to.

        Shares vLLM's shebang inspection - the script's first line names
        the interpreter - but asks that interpreter about `mlx-lm` and
        `mlx` rather than about vLLM and torch. Nothing of either engine
        is imported here: the probe runs the *other* interpreter.
        """
        python = inspect_python_environment(path)
        version: str | None = None
        if python is not None:
            probed = _probe_packages(python.interpreter)
            version = probed.get("mlx_lm") if probed else None
            python = PythonEngine(
                interpreter=python.interpreter,
                pythonVersion=python.pythonVersion,
                packageVersion=version,
                # No torch in this stack at all, and the accelerator is
                # not inferable from a version string the way a
                # `+cu132` local tag gives vLLM's away. Metal or nothing.
                torchVersion=None,
                accelerator=None,
            )
        return DiscoveredBinary(path=path, origin=origin, version=version, python=python)

    def manual_install(self, host: HostAccelerator) -> ManualInstall:
        """Upstream's install line, and a refusal that says why elsewhere.

        The refusal is the interesting half. On anything but Apple
        silicon the honest answer is not "install it", because the
        install would appear to work - so `command` is omitted there,
        per the contract's rule that a command which does not work is
        worse than no command.
        """
        apple = host.os is Os.macos and host.arch is Arch.arm64
        if not apple:
            return ManualInstall(docsUrl=INSTALL_DOCS_URL, notes=_NOT_APPLE_NOTES)
        return ManualInstall(
            command="uv pip install mlx-lm",
            docsUrl=INSTALL_DOCS_URL,
            notes=(
                "Then set `mlxBinary` in the agent's config to that environment's "
                "`bin/mlx_lm.server` console script - its shebang binds its own "
                "interpreter, so nothing needs activating."
            ),
        )

    # --- launching --------------------------------------------------------

    def build_argv(self, spec: RuntimeSpec, binary: DiscoveredBinary, port: int) -> list[str]:
        """`mlx_lm.server --model <path> --host <h> --port <p> [flags]`.

        Every name here is an `add_argument` in `server.py`'s `main()` at
        v0.31.3.

        **There is no `--served-model-name`, and that is a real gap
        rather than an omission here.** The base class requires the alias
        be passed explicitly, because vLLM defaults its served name to
        the `--model` argument verbatim and we launch by absolute path.
        MLX is worse: it has no such flag at all, so the only ids it
        answers to are `default_model` and the resolved absolute path
        (`handle_models_request` publishes `str(model_path.resolve())`).
        Neither works as a routing key - the first collides across every
        MLX runtime in an install, the second leaks the operator's
        directory layout. Closing it needs an advertise-X-send-Y split on
        the *driver*, which is a contract change and is written up rather
        than guessed at. See the verification doc.
        """
        argv = [
            str(binary.path),
            "--model",
            spec.modelPath,
            "--host",
            spec.host or "127.0.0.1",
            "--port",
            str(port),
        ]

        flags = spec.flags or {}
        for field in self.flag_schema().fields:
            if field.key not in flags:
                continue
            value = flags[field.key]
            if value is None:
                continue
            cli = _FLAG_CLI_NAMES[field.key]
            if field.valueType == ConfigValueType.boolean:
                # Every curated boolean is `action="store_true"` upstream,
                # so it takes no value and absent means false.
                if value:
                    argv.append(cli)
            else:
                argv += [cli, str(value)]

        # Verbatim, last, so an operator can always override something
        # the curated surface generated above.
        if spec.extraArgs:
            argv += list(spec.extraArgs)
        return argv

    def working_directory(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> str | None:
        """Inherit the agent's cwd unless told otherwise.

        A console script carries no sidecar libraries, so the base's
        binary-directory default - which exists for prebuilt llama.cpp
        releases - would only be a strange place to leave any
        relative-path output.
        """
        return spec.workingDirectory or None

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        """Name the host fact behind a known startup death.

        Read after the fact rather than checked before launch, for the
        reason the base spells out: an eager refusal can be wrong and an
        explanation of a real failure cannot.
        """
        for signature, explanation in _EXIT_EXPLANATIONS:
            if signature in output_tail:
                return explanation
        return None

    # --- observing --------------------------------------------------------

    async def probe_readiness(self, base_url: str) -> Readiness:
        """Two questions, because one of them is not answerable.

        `/health` says whether the HTTP server is up. It says **nothing
        about the model** - `handle_health_check` writes a literal
        `{"status": "ok"}` - so a second question is needed, and the only
        one that separates a loaded model from a loading one is to ask
        for a token.

        The three outcomes:

          * `/health` unreachable -> `NotAnswering(reached=False)`. The
            server binds almost immediately, so this window is short and
            genuinely means "nothing there yet".
          * `/health` answers, generation does not complete in time ->
            `Loading`, stated rather than inferred. The server is up (it
            just answered) and no token has come out, and on this engine
            there is nothing else that can mean.
          * a token comes back -> `Ready`.

        **The cost, which the live test should measure.** This issues a
        real one-token completion on every poll, including long after the
        engine is ready. One forward pass is cheap but not free, and an
        adapter is a stateless singleton so it cannot remember that it
        already proved residency. If the Mac run shows this is a
        nuisance, the fix is a base-class change - let a probe say "you
        may stop asking the expensive question" - not a quiet weakening
        of the check here.
        """
        url = base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                health = await client.get(f"{url}/health")
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            return NotAnswering(detail=str(e), reached=False)
        except httpx.HTTPError as e:
            # Connected, then nothing usable came back. Not silence.
            return NotAnswering(detail=str(e), reached=True)

        if not health.is_success:
            # Upstream only ever writes 200 here, so anything else is a
            # server we do not recognise rather than a state we can read.
            return NotAnswering(detail=f"/health returned {health.status_code}", reached=True)

        return await self._prove_model_resident(url)

    async def _prove_model_resident(self, url: str) -> Readiness:
        """Ask for one token. The only evidence this engine offers."""
        payload = {
            "model": DEFAULT_MODEL_SENTINEL,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=_GENERATE_TIMEOUT_SECONDS) as client:
                response = await client.post(f"{url}/v1/chat/completions", json=payload)
        except (httpx.ReadTimeout, httpx.PoolTimeout, httpx.WriteTimeout):
            return Loading(
                detail=(
                    "/health is answering but no token came back within "
                    f"{_GENERATE_TIMEOUT_SECONDS:.0f}s: mlx_lm.server serves HTTP while it "
                    "loads the model on another thread, so this is the load. It reports "
                    "no progress of its own; check the captured engine output."
                )
            )
        except httpx.HTTPError as e:
            return NotAnswering(detail=str(e), reached=True)

        if response.is_success:
            return Ready(capabilities=_capabilities_from_completion())
        # A 4xx/5xx here is a served answer, not a load. The likeliest
        # one is a model id upstream could not resolve, which on this
        # engine means `--model` was never passed: `load_default()` is a
        # no-op when `cli_args.model` is None, and `default_model` then
        # maps to None. That is a launch mistake, and saying so beats
        # reporting it as an endless load.
        return NotAnswering(
            detail=(f"/v1/chat/completions returned {response.status_code}: {response.text[:200]}"),
            reached=True,
        )

    # --- configuring ------------------------------------------------------

    def flag_schema(self) -> ConfigSchema:
        return ConfigSchema(
            component="engine:mlx",
            categories=_CATEGORIES,
            fields=_FLAG_FIELDS,
        )


def _probe_packages(interpreter: str) -> dict[str, str | None] | None:
    """Ask an interpreter which of mlx-lm and mlx it has.

    Both matter and for different reasons: `mlx-lm` is the version this
    adapter's claims are pinned against, and `mlx` present-or-absent is
    the difference between a working engine and the Darwin-marker trap.
    """
    try:
        out = subprocess.run(
            [interpreter, "-c", _ENV_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=_ENV_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("package probe via %s failed: %s", interpreter, e)
        return None
    if out.returncode != 0:
        return None
    try:
        parsed = json.loads(out.stdout.strip())
    except (json.JSONDecodeError, ValueError) as e:
        log.debug("package probe via %s returned unparseable output: %s", interpreter, e)
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _capabilities_from_completion() -> RuntimeCapabilities | None:
    """What a served completion tells us about the engine. Nothing yet.

    MLX publishes no context length anywhere - not on `/v1/models`
    (which lists cache entries), not on a completion. Returning None
    rather than a guess keeps the library's fit verdicts honest: an
    invented context window is worse than an absent one, because the
    library computes against it.
    """
    return None


# Matched against the engine's captured output tail on a non-zero exit.
_EXIT_EXPLANATIONS: list[tuple[str, str]] = [
    (
        "No module named 'mlx'",
        "mlx-lm is installed but `mlx` is not. Upstream marks that dependency "
        "`platform_system == 'Darwin'`, so `pip install mlx-lm` succeeds on Linux and "
        "Windows and installs nothing that can run a model. This engine needs Apple "
        "silicon.",
    ),
    (
        "No module named 'mlx_lm'",
        "The console script resolved but its environment has no `mlx-lm`. Point "
        "`mlxBinary` at the `bin/mlx_lm.server` inside the environment where you "
        "installed it, rather than one that happens to be earlier on PATH.",
    ),
    (
        "Metal is not available",
        "MLX could not reach Metal. This engine is Apple silicon only - an Intel Mac, "
        "or a VM without GPU access, cannot run it.",
    ),
]

_CATEGORIES = {
    "memory": "Memory and context",
    "performance": "Performance",
    "model": "Model loading",
}

# Curated flags -> `mlx_lm.server` CLI names. Every name checked against
# `mlx_lm/server.py` `main()` at v0.31.3, where each is an
# `add_argument`. Kept as a separate mapping from the schema so the
# UI-facing key never has to look like a CLI flag, and so an upstream
# rename touches one line.
_FLAG_CLI_NAMES: dict[str, str] = {
    "maxTokens": "--max-tokens",
    "promptCacheSize": "--prompt-cache-size",
    "promptCacheBytes": "--prompt-cache-bytes",
    "decodeConcurrency": "--decode-concurrency",
    "promptConcurrency": "--prompt-concurrency",
    "prefillStepSize": "--prefill-step-size",
    "draftModel": "--draft-model",
    "numDraftTokens": "--num-draft-tokens",
    "adapterPath": "--adapter-path",
    "trustRemoteCode": "--trust-remote-code",
    "chatTemplate": "--chat-template",
}

_FLAG_FIELDS: list[ConfigField] = [
    ConfigField(
        key="maxTokens",
        label="Default max tokens",
        description=(
            "Upstream's default output length for a request that does not set one "
            "(engine default 512). This is a *default*, not a context window: MLX "
            "has no `--max-model-len` equivalent and never reports a context size, "
            "so the runtime's declared context is the only number the library can "
            "compute a fit against. The gateway sends max-tokens on every request "
            "anyway, so this rarely decides anything."
        ),
        category="memory",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="promptCacheSize",
        label="Prompt cache entries",
        description=(
            "How many distinct KV caches to keep (engine default 10). MLX holds an "
            "LRU of prompt caches across requests, which is what makes a multi-turn "
            "chat cheap; raising it trades unified memory for fewer re-prefills."
        ),
        category="memory",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="promptCacheBytes",
        label="Prompt cache size",
        description=(
            "Ceiling on the KV caches, in bytes. Upstream parses a size string, so "
            "`8GB` is accepted as well as a plain number. Unset means no byte "
            "ceiling and only the entry count above applies - on a unified-memory "
            "machine, where the cache competes with the model itself for the same "
            "pool, this is the more useful of the two."
        ),
        category="memory",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    ),
    ConfigField(
        key="decodeConcurrency",
        label="Decode concurrency",
        description=(
            "How many batchable requests decode in parallel (engine default 32). "
            "The closest thing MLX has to a parallel-slot count, and the number to "
            "reach for when one runtime serves several clients at once."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="promptConcurrency",
        label="Prompt concurrency",
        description=(
            "How many prompts are prefilled in parallel (engine default 8). Prefill "
            "is compute-bound where decode is memory-bound, which is why upstream "
            "gives them separate dials."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="prefillStepSize",
        label="Prefill step size",
        description=(
            "Tokens processed per prefill step (engine default 2048). Larger steps "
            "prefill faster and hold more intermediate memory at once."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="draftModel",
        label="Draft model",
        description=(
            "Path to a smaller model for speculative decoding. Upstream refuses this "
            "in distributed mode; it is otherwise the largest throughput lever MLX "
            "exposes."
        ),
        category="performance",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    ),
    ConfigField(
        key="numDraftTokens",
        label="Draft tokens",
        description=(
            "How many tokens the draft model proposes per step (engine default 3). "
            "Only meaningful with a draft model set."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="adapterPath",
        label="Adapter path",
        description=(
            "Path to trained LoRA adapter weights and config, applied on top of the "
            "base model at load."
        ),
        category="model",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    ),
    ConfigField(
        key="trustRemoteCode",
        label="Trust remote code",
        description=(
            "Allow the tokenizer to execute code shipped with the model. Off by "
            "default, and worth leaving off unless a specific model needs it."
        ),
        category="model",
        valueType=ConfigValueType.boolean,
        requiresRestart=True,
    ),
    ConfigField(
        key="chatTemplate",
        label="Chat template",
        description=(
            "Override the tokenizer's chat template with a Jinja string. Needed for "
            "models that ship none, or ship a broken one."
        ),
        category="model",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    ),
]
