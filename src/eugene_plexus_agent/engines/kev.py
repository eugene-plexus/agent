"""The Kev adapter: `python -m kev.serve`, the decision engine.

The fourth engine and the first that does not chat: Kev answers typed
System One questions with structured probabilities, and its companion
driver serves `POST /v1/decide` instead of completions. Every claim here
is pinned to `kev/serve.py` at commit `1c35199` and to a live kev-0.8b
run on WSL CPU, 2026-09-22 (specs `docs/design/decision-models.md` is
the ledger):

  * The launch is `<python> -m kev.serve --run <checkpoint> --port <p>`.
    The flags are exactly `--run`, `--fallback` and `--port`; there is
    **no `--host`** — upstream hardcodes the bind to `127.0.0.1`, which
    is precisely the posture Eugene wants, because the gateway is the
    authenticated front door.
  * **The model loads BEFORE the server binds** (`ck.load(...)` precedes
    the uvicorn call in `main`), so this is vLLM's readiness shape:
    alive-and-refusing IS loading, and only the supervisor can say so.
  * There is **no health endpoint**. `GET /v1/models` answers once
    serving, with `{models: [{id, aliases, run, base, ...}]}` (measured).
  * The server holds ONE request at a time (a lock; no cross-caller
    batching), so the companion driver is told `decisionMaxConcurrent: 1`
    and the gateway never over-admits it.
  * The request's `model` field is echoed unvalidated (measured), so the
    public alias travels as-is and no upstream translation is needed.

**The environment is the operator's, like vLLM's and mlx-lm's** — a
`uv sync --extra serve` inside a checkout of the pinned commit, never
Eugene's own venv. `kevPython` points at that environment's
interpreter, because Kev has no console script to point at: the launch
module is `kev.serve` and the checkout must be importable from it,
which `uv sync` arranges by installing the project into its `.venv`.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import httpx

from .._generated.models import (
    ConfigField,
    ConfigSchema,
    ConfigValueType,
    EngineKind,
    HostAccelerator,
    ManualInstall,
    ModelFormat,
    Origin,
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
)
from .base import (
    probe_client as _probe_client,
)

log = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 5.0

#: The commit every claim was read against and the recipe pins. Kev has
#: no versioned releases at the pin date; a commit is the honest pin.
UPSTREAM_COMMIT_PINNED = "1c351992ba3df4a0a0f2ae03051b25466a2c7bcb"

INSTALL_DOCS_URL = "https://github.com/jaredpalmer/kev"

_ENV_PROBE_TIMEOUT_SECONDS = 10.0
_ENV_PROBE_SCRIPT = """
import importlib.metadata as m, importlib.util, json
def v(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return None
print(json.dumps({
    "kev": v("kev"),
    "torch": v("torch"),
    "serve_importable": importlib.util.find_spec("kev.serve") is not None,
}))
""".strip()


class KevAdapter(EngineAdapter):
    kind = EngineKind.kev

    #: There is no console script: the launch is `python -m kev.serve`,
    #: so the "binary" is the Kev environment's interpreter. Deliberately
    #: NOT discoverable from PATH — see `discover`.
    binary_name = "python"

    model_formats = (ModelFormat.kev_checkpoint,)

    #: Measured shape: the model (and on a first run, the base-model
    #: download) completes before the port binds, so a starting Kev is
    #: indistinguishable over the network from a dead one — the
    #: supervisor's process handle is what says `loading`.
    answers_while_loading = False

    #: A first launch downloads the base model, minutes on an ordinary
    #: link, so the budget is generous. The state stays `loading` past
    #: it — flagged, with the elapsed time and a pointer at the output.
    startup_budget_seconds = 900.0

    install_policy = Policy.manual
    configured_binary_key = "kevPython"

    #: Never proved beyond WSL CPU; CUDA, ROCm and Metal claims each
    #: need their own evidence. Badged, not hidden, where it can run.
    experimental = True

    # --- discovery --------------------------------------------------------

    def discover(self, *, configured: str | None = None) -> DiscoveredBinary | None:
        """Configured path or nothing — never PATH.

        The base class falls back to `shutil.which(binary_name)`, and
        for an engine whose "binary" is a Python interpreter that
        fallback is a trap: a bare `python` on PATH is never evidence of
        a Kev environment, and launching it would fail with
        `No module named kev` on every machine that has Python and not
        Kev — which is every machine.
        """
        if configured:
            path = Path(configured).expanduser()
            if not path.is_file():
                from .base import EngineUnavailableError

                raise EngineUnavailableError(
                    f"`kevPython` points at {configured}, which does not exist"
                )
            return self.describe(path, Origin.configured)
        return None

    def describe(self, path: Path, origin: Origin) -> DiscoveredBinary:
        """Ask the interpreter whether it can actually serve Kev."""
        probed = _probe_packages(str(path))
        version = probed.get("kev") if probed else None
        python = PythonEngine(
            interpreter=str(path),
            pythonVersion=None,
            packageVersion=version,
            torchVersion=probed.get("torch") if probed else None,
            accelerator=None,
        )
        return DiscoveredBinary(path=path, origin=origin, version=version, python=python)

    def manual_install(self, host: HostAccelerator) -> ManualInstall:
        """The pinned checkout recipe — an environment, not a package.

        Kev is a uv project rather than a published wheel: the recipe is
        a clone at the pinned commit and `uv sync --extra serve`, after
        which `kevPython` points at the checkout's own `.venv` python.
        Works on Linux, WSL and macOS; the Windows host itself is
        untested upstream and the honest note says to use WSL.
        """
        return ManualInstall(
            command=(
                "git clone https://github.com/jaredpalmer/kev.git ~/eugene-kev && "
                f"cd ~/eugene-kev && git checkout {UPSTREAM_COMMIT_PINNED} && "
                "uv sync --extra serve"
            ),
            docsUrl=INSTALL_DOCS_URL,
            notes=(
                "Then set `kevPython` in the agent's config to "
                "`~/eugene-kev/.venv/bin/python` - the checkout's own interpreter, "
                "which `uv sync` teaches to import `kev.serve`. Kev checkpoints are "
                "directories (adapter + head.pt + tokenizer + provenance); the base "
                "model they apply to is downloaded separately on the first launch, "
                "which is why a first start can take minutes and why an offline node "
                "needs the base pre-downloaded. The server binds loopback only, by "
                "upstream's own design: Eugene's gateway is the front door. Verified "
                "on Linux/WSL CPU; CUDA, ROCm and Metal need their own evidence."
            ),
        )

    # --- launching --------------------------------------------------------

    def build_argv(self, spec: RuntimeSpec, binary: DiscoveredBinary, port: int) -> list[str]:
        """`<python> -m kev.serve --run <checkpoint> --port <p>`.

        Exactly the pinned flags and no `--host`: upstream binds
        loopback unconditionally, so a spec host other than loopback
        cannot be honoured — `validate_spec`'s launch policy is where a
        non-loopback host is refused for this engine.

        No alias flag either, and none is needed: the server echoes the
        request's `model` unvalidated (measured), so the public alias
        travels through untouched and `upstream_model_id` stays None.
        """
        argv = [
            str(binary.path),
            "-m",
            "kev.serve",
            "--run",
            spec.modelPath,
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
            argv += [_FLAG_CLI_NAMES[field.key], str(value)]
        if spec.extraArgs:
            argv += list(spec.extraArgs)
        return argv

    def companion_overrides(self, spec: RuntimeSpec) -> dict[str, object]:
        """The companion speaks decisions, and it must not be over-fed.

        `systemone_custom` is the driver provider that serves
        `POST /v1/decide`; `decisionMaxConcurrent: 1` is the measured
        fact that the pinned server holds one request at a time, which
        the driver advertises and the gateway enforces.
        """
        return {"provider": "systemone_custom", "decisionMaxConcurrent": 1}

    def default_env(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> dict[str, str]:
        """Nothing, so far honestly.

        `KEV_DTYPE` and friends are operator tuning, not the
        cannot-start-without class this hook exists for; they ride
        `RuntimeSpec.env` like any other engine's environment.
        """
        return {}

    def working_directory(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> str | None:
        """The Kev checkout — the interpreter's venv parent — unless told
        otherwise: `python -m kev.serve` resolves relative artifacts
        (`runs/`, the fallback default) against the cwd."""
        if spec.workingDirectory:
            return spec.workingDirectory
        # <checkout>/.venv/bin/python -> <checkout>
        venv = binary.path.parent.parent
        if venv.name == ".venv":
            return str(venv.parent)
        return None

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        for signature, explanation in _EXIT_EXPLANATIONS:
            if signature in output_tail:
                return explanation
        return None

    # --- observing --------------------------------------------------------

    async def probe_readiness(self, base_url: str, *, established: bool = False) -> Readiness:
        """`GET /v1/models` — the only thing this server answers about
        itself, and it answers only once serving (the load precedes the
        bind, so there is no loading window to narrate). A refused
        connection is genuine silence; `interpret_readiness` turns it
        into `loading` while the process is alive.
        """
        del established  # one cheap read; nothing to downgrade to
        url = base_url.rstrip("/")
        try:
            response = await _probe_client().get(f"{url}/v1/models", timeout=_PROBE_TIMEOUT_SECONDS)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            return NotAnswering(detail=str(e), reached=False)
        except httpx.HTTPError as e:
            return NotAnswering(detail=str(e), reached=True)
        if not response.is_success:
            return NotAnswering(detail=f"/v1/models returned {response.status_code}", reached=True)
        return Ready(capabilities=_capabilities_from_models(response))

    # --- configuring ------------------------------------------------------

    def flag_schema(self) -> ConfigSchema:
        return ConfigSchema(
            component="engine:kev",
            categories={"model": "Model loading"},
            fields=_FLAG_FIELDS,
        )


def _capabilities_from_models(response: httpx.Response) -> RuntimeCapabilities | None:
    """Kev publishes no context window; nothing to claim honestly."""
    return None


def _probe_packages(interpreter: str) -> dict[str, object] | None:
    try:
        out = subprocess.run(
            [interpreter, "-c", _ENV_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=_ENV_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("kev package probe via %s failed: %s", interpreter, e)
        return None
    if out.returncode != 0:
        return None
    try:
        parsed = json.loads(out.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


_EXIT_EXPLANATIONS: list[tuple[str, str]] = [
    (
        "No module named 'kev'",
        "The configured `kevPython` cannot import kev.serve. Point it at the "
        "interpreter inside the Kev checkout's own .venv (created by "
        "`uv sync --extra serve` in the checkout), not at a system Python.",
    ),
    (
        "No module named 'torch'",
        "The Kev environment has no torch — `uv sync --extra serve` was not run, "
        "or was run without the extra.",
    ),
]

_FLAG_CLI_NAMES: dict[str, str] = {
    "fallback": "--fallback",
}

_FLAG_FIELDS: list[ConfigField] = [
    ConfigField(
        key="fallback",
        label="Fallback checkpoint",
        description=(
            "A second checkpoint the server falls back to when the primary "
            "cannot load (upstream default `runs/smoke`). Rarely worth "
            "setting; the pinned server's only other flags are the run and "
            "the port, both owned by the launch."
        ),
        category="model",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    ),
]
