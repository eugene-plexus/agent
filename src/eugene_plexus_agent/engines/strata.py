"""Experimental, prepared-config Strata integration (upstream v0.1.39).

The HTTP server owns the native child. Only text, one request at a time;
weights/tokenizers/MTP packs stay in the operator's model folder.

Since LS3 a prepared model is a Library model (library-sources-and-engines.md
§4.5): its path is a provenance file, `<name>.eugene-prepared.json`, whose
`entry` is Strata's JSON configuration. A runtime declared before LS3 names
that configuration directly, and still launches.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

import httpx
from pydantic import ValidationError

from .._generated.models import (
    ConfigSchema,
    EngineKind,
    ModelFormat,
    ModelRequirement,
    PreparedProvenance,
    RuntimeCapabilities,
    RuntimeSpec,
    SupportedModel,
)
from ..preparation import Recipe
from ..supervisor import SpawnPlanError
from .base import (
    PREPARED_SUFFIX,
    DiscoveredBinary,
    EngineAdapter,
    Loading,
    NotAnswering,
    Readiness,
    Ready,
    default_model_alias,
    probe_client,
)
from .strata_models import PREPARATION, STRATA_FILES, SUPPORTED_MODELS, supported_here

VERSION = "v0.1.39"
COMMIT = "6f32ec070f23ced9f50e704d854d775da52591ab"

# A deliberately small native surface. Unknown switches are refused, never
# silently discarded. In particular no output files, commands or plugin loading.
PATH_ARGS = {"--pack", "--native", "--ple-gguf", "--expert-profile", "--mtp"}
VALUE_ARGS = {
    "--expert-cache",
    "--prefill",
    "--spec",
    "--spec-min-p",
    "--max-context",
    "--kv",
    "--rope-scaling",
    "--rope-scale",
    "--ple-io",
    "--kv-resident",
    "--resident-budget-gib",
    "--vram-reserve-mib",
    "--pool-workers",
    "--pcie-frac",
}
BOOL_ARGS = {"--resident-experts", "--mmap-experts"}
PASS_KEYS = {
    "sampling",
    "reasoning_budget_tokens",
    "repeat_stop_tokens",
    "engine_silence_s",
    "fit_max_tokens",
    "anthropic_thinking",
    "gpu",
}
# Setup writes these; Eugene supplies its own copies. Nontrivial unsupported
# features must be refused instead of silently changing the operator's intent.
OWNED_KEYS = {"exe", "cwd", "model_name", "log", "port", "host", "lib_dirs", "aliases"}
# Setup-only bookkeeping. The selected GPU and prepared MTP files already
# carry these choices; the HTTP server does not read either field.
SETUP_KEYS = {"gpus_asked", "draft_vocab"}


#: The provenance layout this agent reads.
PROVENANCE_VERSION = 1


def prepared_entry(path: Path) -> Path:
    """Strata's configuration for a model path: the provenance file's
    `entry`, or the path itself when it names a configuration directly
    (a declaration from before LS3). Read at every launch, so a prepared
    model re-adopted with a new entry starts from the new one."""
    if not path.name.lower().endswith(PREPARED_SUFFIX):
        return path
    where = f"Prepared model {path}"
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise SpawnPlanError(f"{where}: cannot read its provenance file: {exc}") from exc
    except ValueError as exc:
        raise SpawnPlanError(f"{where}: its provenance file is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SpawnPlanError(f"{where}: its provenance file is not a JSON object")
    version = raw.get("formatVersion", PROVENANCE_VERSION)
    if isinstance(version, int) and version > PROVENANCE_VERSION:
        raise SpawnPlanError(
            f"{where}: its provenance file was written by a newer Eugene (formatVersion "
            f"{version}); update this node's agent"
        )
    try:
        provenance = PreparedProvenance.model_validate(raw)
    except ValidationError as exc:
        raise SpawnPlanError(f"{where}: its provenance file is not valid: {exc}") from exc
    if provenance.engine is not EngineKind.strata:
        raise SpawnPlanError(
            f"{where} was prepared for {provenance.engine.value}, not Strata; "
            "only the engine it was prepared for can load it"
        )
    entry = provenance.entry
    if PurePosixPath(entry).is_absolute() or PureWindowsPath(entry).is_absolute():
        # A path on this node, used as written.
        return Path(entry)
    return path.parent / entry


def runtime_lib_dirs(root: Path) -> list[str]:
    """CUDA 13 wheels nest Windows DLLs under cu13/bin/x86_64."""
    libraries = root / ".venv" / "Lib" / "site-packages" / "nvidia"
    return sorted({str(p.parent) for p in libraries.rglob("*.dll")})


def prepared_config(path: Path, *, alias: str, root: Path) -> dict[str, object]:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(cfg, dict):
            raise ValueError("expected a JSON object")
        allowed = (
            PASS_KEYS
            | OWNED_KEYS
            | SETUP_KEYS
            | {"args", "tokenizer", "parallel", "cuda", "backend"}
        )
        unsupported = sorted(
            k for k, v in cfg.items() if k not in allowed and v not in (None, False, [], {})
        )
        if unsupported:
            raise ValueError("unsupported config features: " + ", ".join(unsupported))
        if cfg.get("parallel", 1) != 1:
            raise ValueError("this integration supports parallel: 1 only")
        if cfg.get("cuda", 13) != 13 or cfg.get("backend", "cuda") != "cuda":
            raise ValueError("this recipe supports the CUDA 13 engine only")
        cwd = Path(cfg.get("cwd") or path.parent)
        if not cwd.is_absolute():
            cwd = path.parent / cwd

        def existing(value: object) -> str:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("model paths must be nonempty strings")
            target = Path(value).expanduser()
            target = (cwd / target).resolve() if not target.is_absolute() else target.resolve()
            if not target.exists():
                raise ValueError(f"prepared model asset is missing: {target}")
            return str(target)

        args = cfg.get("args")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ValueError("args must be a list of strings")
        normalized: list[str] = []
        seen: set[str] = set()
        i = 0
        while i < len(args):
            flag = args[i]
            if flag in seen:
                raise ValueError(f"duplicate native flag: {flag}")
            seen.add(flag)
            normalized.append(flag)
            i += 1
            if flag in BOOL_ARGS:
                continue
            if flag not in PATH_ARGS | VALUE_ARGS:
                raise ValueError(f"unsupported native flag: {flag}")
            if i == len(args) or args[i].startswith("--"):
                raise ValueError(f"missing value for {flag}")
            normalized.append(existing(args[i]) if flag in PATH_ARGS else args[i])
            i += 1
        if not {"--pack", "--native", "--max-context"} <= seen:
            raise ValueError("args must include --pack, --native and --max-context")
        tokenizer = Path(existing(cfg.get("tokenizer")))
        for name in ("vocab.json", "merges.txt", "token_type.json"):
            if not (tokenizer / name).is_file():
                raise ValueError(f"prepared tokenizer is missing {name}")
        native = root / "engine" / ("strata.exe" if os.name == "nt" else "strata")
        if not native.is_file():
            raise ValueError(f"native engine is missing: {native}")
        return {
            **{k: v for k, v in cfg.items() if k in PASS_KEYS},
            "exe": str(native),
            "cwd": str(root),
            "args": normalized,
            "tokenizer": str(tokenizer),
            "model_name": alias,
            "parallel": 1,
            "lazy_load": False,
            "idle_unload_s": 0,
            "open_browser": False,
            "lib_dirs": runtime_lib_dirs(root),
        }
    except (OSError, ValueError, TypeError) as exc:
        raise SpawnPlanError(f"Strata prepared config {path}: {exc}") from exc


class StrataAdapter(EngineAdapter):
    kind = EngineKind.strata
    binary_name = "strata-server"  # never mistake an arbitrary server.py on PATH for Strata
    configured_binary_key = "strataServer"
    #: What it loads as it is: models it prepared, never an arbitrary GGUF.
    model_formats = (ModelFormat.prepared,)
    #: The models it prepared, which are Library models since LS3; and the
    #: Qwen3.8-Flash-Next GGUFs on its own list (`general.architecture`
    #: qwen4exp, read off ISTA-DASLab's repo 2026-10-09), once Strata has
    #: prepared them. Only those by name (LS4): upstream's setup refuses any
    #: other GGUF of the same architecture, an Unsloth K-quant for one.
    accepts = (
        ModelRequirement(
            format=ModelFormat.prepared,
            preparedFor=EngineKind.strata,
            preference=50,
        ),
        ModelRequirement(
            format=ModelFormat.gguf,
            architectures=["qwen4exp"],
            files=list(STRATA_FILES),
            preparation=PREPARATION,
            preference=50,
            note="Strata's setup prepares only the files on its own list",
        ),
    )
    #: What upstream's setup offers, as files on the hub (LS4).
    supported_models = SUPPORTED_MODELS
    experimental = True

    def supported_models_here(self) -> tuple[SupportedModel, ...]:
        """Each preparation's disk by setup's own rule with this node's RAM (B51)."""
        from .devices import host_memory

        total, _available = host_memory()
        return supported_here(total)

    def plan_preparation(
        self,
        *,
        binary: DiscoveredBinary,
        folder: Path,
        model: Path,
        source_path: str,
        context: int | None,
    ) -> Recipe:
        """Upstream's setup, into `Strata-data` at the top of the Library
        folder (B43, B45); see `strata_prepare`."""
        from .devices import host_memory
        from .strata_prepare import DATA_FOLDER, main_gpu, plan

        total, _available = host_memory()
        return plan(
            root=binary.path.resolve().parent.parent,
            gguf=model,
            data_dir=folder / DATA_FOLDER,
            source_path=source_path,
            context=context,
            ram_bytes=total,
            gpu=main_gpu(),
        )

    answers_while_loading = False
    startup_budget_seconds = 900.0

    def prepare_config(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> dict[str, object]:
        """Validate without writing files, also used before a model switch."""
        root = binary.path.resolve().parent.parent
        python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            raise SpawnPlanError(f"Strata's isolated Python is missing: {python}")
        return prepared_config(
            prepared_entry(Path(spec.modelPath)),
            alias=spec.modelAlias or default_model_alias(spec.modelPath),
            root=root,
        )

    def build_argv(self, spec: RuntimeSpec, binary: DiscoveredBinary, port: int) -> list[str]:
        config = self.prepare_config(spec, binary)
        root = binary.path.resolve().parent.parent
        python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # Generated configurations are ours. The model's original config is never
        # changed, and two declarations cannot overwrite each other's alias.
        key = hashlib.sha256(spec.name.encode()).hexdigest()
        launch = self.managed_store().directory / ".launch" / f"{key}.json"
        launch.parent.mkdir(parents=True, exist_ok=True)
        temporary = launch.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
        temporary.replace(launch)
        return [
            str(python),
            str(binary.path),
            "--engine",
            "strata",
            "--config",
            str(launch),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

    def companion_overrides(self, spec: RuntimeSpec) -> dict[str, object]:
        return {"provider": "strata_local", "slotPinning": False}

    def companion_secrets(self, spec: RuntimeSpec) -> dict[str, str]:
        key = (spec.env or {}).get("STRATA_API_KEY") or os.environ.get("STRATA_API_KEY")
        return {"apiKey": str(key)} if key else {}

    def readiness_headers(self, spec: RuntimeSpec) -> dict[str, str]:
        key = self.companion_secrets(spec).get("apiKey")
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def probe_readiness(
        self, base_url: str, *, established: bool = False, headers: dict[str, str] | None = None
    ) -> Readiness:
        try:
            response = await probe_client().get(f"{base_url}/health", headers=headers, timeout=3)
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return NotAnswering()
        if (
            response.status_code != 200
            or not isinstance(body, dict)
            or body.get("service") != "strata"
        ):
            return NotAnswering(
                detail="the endpoint did not identify itself as Strata", reached=True
            )
        if body.get("loaded") is not True:
            return Loading(detail="Strata is answering but its model is not loaded")
        context = body.get("max_context")
        return Ready(
            capabilities=RuntimeCapabilities(
                contextLength=context if type(context) is int and context > 0 else None,
                parallelSlots=1,
                embeddings=False,
                vision=False,
                multimodal=False,
            )
        )

    def flag_schema(self) -> ConfigSchema:
        return ConfigSchema(component="engine:strata", categories={}, fields=[])
