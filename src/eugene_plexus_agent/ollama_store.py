"""Reading an Ollama model store through its manifests (CX9, 2026-10-20).

Ollama keeps models as hash-named blobs (`blobs/sha256-<hex>`) and says what
each one is in a manifest per tag. Nothing here walks `blobs/`: a manifest
names the one GGUF blob that is the model (and its projector), so the store
is read by name and the blobs are only stat'ed.

Read from Ollama's own source (`manifest/manifest.go`, 2026-10-20), not its
docs:

* Manifests are `manifests-v2/<host>/<namespace>/<model>/<tag>` (Ollama
  v0.40.0 onward) and the older `manifests/<host>/<namespace>/<model>/<tag>`.
  A name in both is the v2 one.
* A manifest is either a plain one (`layers`, with a `config`), or a *list*
  (`mediaType` `application/vnd.ollama.manifest.list.v2+json`) whose
  `manifests[]` children each carry a `runner` (`llamacpp`, `ggml`, `mlx`) and
  a `format` (`gguf`, `safetensors`). A child is either written inline (it has
  `layers`) or is a *reference*: a `digest` naming a blob, `blobs/sha256-<hex>`,
  that holds the child manifest's JSON. A reference's own runner and format
  fill in what the blob leaves out. Ollama's own pick is the first runner of
  its preference list that has a child; for llama.cpp that is `llamacpp`, then
  `ggml`.
* A plain manifest of an older Ollama has no runner at all; Ollama reads it as
  `ggml` (`MetadataForConfig`), and so does this.
* The model is the layer `application/vnd.ollama.image.model`; a vision
  projector is `application/vnd.ollama.image.projector`. MLX and safetensors
  models are `application/vnd.ollama.image.tensor` layers (or format
  `safetensors`) and llama.cpp cannot load them.
* The migration that Ollama v0.40.0 runs (`compatmigrate`) leaves a
  `llamacpp:<sha>` model name next to a migrated model (ollama/ollama#18830);
  it is internal and never listed.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MANIFEST_DIRS = ("manifests-v2", "manifests")
"""Newest layout first: a name found in both is the first one's."""

MEDIA_LIST = "application/vnd.ollama.manifest.list.v2+json"
MEDIA_MODEL = "application/vnd.ollama.image.model"
MEDIA_PROJECTOR = "application/vnd.ollama.image.projector"
MEDIA_TENSOR = "application/vnd.ollama.image.tensor"

RUNNER_LLAMACPP = "llamacpp"
RUNNER_GGML = "ggml"
RUNNER_PREFERENCE = (RUNNER_LLAMACPP, RUNNER_GGML)
"""llama.cpp can load these two runners' GGUFs; `mlx` it cannot."""

OLLAMA_HOSTS = frozenset({"ollama.com", "registry.ollama.ai"})

NOT_FOR_LLAMACPP = "not for llama.cpp"

# Architectures (`general.architecture`) of the families Ollama rewrites for
# upstream llama.cpp. Source: the `compatmigrate` directory of ollama/ollama
# (one migrator per family: gptoss, gemma3, gemma3n, gemma4, llama4, qwen3vl,
# qwen25vl, qwen35, qwen3next, mistral_pixtral, deepseek_ocr, glm_ocr,
# glm47_flash, nemotron_h_moe, nemotron3, olmo3, embeddinggemma, laguna),
# read 2026-10-20. Left out on purpose: the migrators that need more than the
# architecture to say yes (`llama` for llama3, `bert` for snowflake-arctic-embed2,
# `lfm2` for lfm2.5-thinking, `clip` for bakllava), because naming them would
# warn about most ordinary models.
MIGRATED_ARCHITECTURES = frozenset(
    {
        "gptoss",
        "gemma3",
        "gemma3n",
        "gemma4",
        "llama4",
        "qwen3vl",
        "qwen3vlmoe",
        "qwen25vl",
        "qwen35",
        "qwen35moe",
        "qwen3next",
        "mistral3",
        "deepseekocr",
        "glmocr",
        "glm4moelite",
        "nemotron_h_moe",
        "nemotron_h_omni",
        "olmo3",
        "laguna",
    }
)

_DIGEST = re.compile(r"^sha256[:-]([0-9a-fA-F]{64})$")


@dataclass(frozen=True)
class StoreSkip:
    """A manifest read and not listed, with the reason."""

    path: str
    detail: str


@dataclass
class StoreModel:
    """One GGUF blob and everything Ollama calls it."""

    blob: Path
    tags: list[str] = field(default_factory=list)
    projector: Path | None = None
    runner: str = ""
    """Of the manifest the blob came from; empty for an older Ollama's."""

    @property
    def legacy_layout(self) -> bool:
        """Not a `llamacpp` manifest: what Ollama has not migrated."""
        return self.runner != RUNNER_LLAMACPP


def is_store(path: Path) -> bool:
    """A directory holding `blobs/` and `manifests/` or `manifests-v2/`."""
    try:
        return (path / "blobs").is_dir() and any((path / d).is_dir() for d in MANIFEST_DIRS)
    except OSError:
        return False


def tag_name(host: str, namespace: str, model: str, tag: str) -> str:
    """`ollama list`'s name: bare for Ollama's own library, else qualified."""
    if host in OLLAMA_HOSTS:
        if namespace == "library":
            return f"{model}:{tag}"
        return f"{namespace}/{model}:{tag}"
    return f"{host}/{namespace}/{model}:{tag}"


def blob_path(root: Path, digest: str) -> Path | None:
    """`blobs/sha256-<hex>` for a `sha256:<hex>` digest; None if it is not one."""
    found = _DIGEST.match(digest.strip()) if isinstance(digest, str) else None
    if found is None:
        return None
    return root / "blobs" / f"sha256-{found.group(1).lower()}"


def _manifest_files(store: Path, directory: str) -> list[tuple[tuple[str, str, str, str], Path]]:
    base = store / directory
    out: list[tuple[tuple[str, str, str, str], Path]] = []
    if not base.is_dir():
        return out
    try:
        hosts = sorted(base.iterdir())
    except OSError as exc:
        log.warning("cannot list %s: %s", base, exc)
        return out
    for host in hosts:
        for namespace in _children(host):
            for model in _children(namespace):
                for tag in _entries(model):
                    try:
                        if tag.is_file():
                            out.append(((host.name, namespace.name, model.name, tag.name), tag))
                    except OSError:
                        continue
    return out


def _entries(path: Path) -> list[Path]:
    try:
        return sorted(path.iterdir())
    except OSError:
        return []


def _children(path: Path) -> list[Path]:
    return [p for p in _entries(path) if _is_dir(p)]


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


MAX_MANIFEST_BYTES = 1024 * 1024
"""A manifest is a few KB. Anything larger is not one: a digest can name the model
blob itself, and that is never read into memory."""


class ManifestError(Exception):
    """A manifest that cannot be used, in words a person can read."""


def _load(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise ManifestError(f"it is {size} bytes, too large to be a manifest")
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"it cannot be read ({exc.strerror or exc})") from exc
    except ValueError as exc:
        raise ManifestError("it is not JSON") from exc
    if not isinstance(data, dict):
        raise ManifestError("it is not a manifest")
    return data


def _resolve_child(store: Path, child: dict[str, Any]) -> dict[str, Any] | None:
    """A list's child as a full manifest: inline as written, or the manifest
    JSON in the blob it names, with the reference's runner and format filling
    in what the blob leaves out. None when the blob is not there."""
    if child.get("layers") is not None:
        return child
    blob = blob_path(store, str(child.get("digest") or ""))
    if blob is None:
        return None
    try:
        resolved = _load(blob)
    except ManifestError:
        return None
    for key in ("runner", "format"):
        if not resolved.get(key) and child.get(key):
            resolved[key] = child[key]
    return resolved


def _choose(store: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """`(the manifest llama.cpp would use, why not)` for a named manifest."""
    if manifest.get("mediaType") != MEDIA_LIST and manifest.get("manifests") is None:
        return manifest, ""
    listed = manifest.get("manifests")
    children = [c for c in listed if isinstance(c, dict)] if isinstance(listed, list) else []
    for runner in RUNNER_PREFERENCE:
        for child in children:
            if str(child.get("runner") or "").lower() != runner:
                continue
            if str(child.get("format") or "gguf").lower() != "gguf":
                continue
            resolved = _resolve_child(store, child)
            # A child with no model layer is no choice: the next runner's may have one.
            if resolved is not None and any(
                x.get("mediaType") == MEDIA_MODEL for x in _layers(resolved)
            ):
                return resolved, ""
    runners = sorted({str(c.get("runner") or "?") for c in children})
    return None, f"{NOT_FOR_LLAMACPP}: only {', '.join(runners) or 'no'} runner(s)"


def _layers(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    layers = manifest.get("layers")
    return [x for x in layers if isinstance(x, dict)] if isinstance(layers, list) else []


def _read_one(
    store: Path,
    path: Path,
    name: str,
    by_blob: dict[Path, StoreModel],
    skips: list[StoreSkip],
) -> None:
    """One named manifest: its model in `by_blob`, or why not in `skips`."""
    raw = _load(path)
    chosen, why = _choose(store, raw)
    if chosen is None:
        skips.append(StoreSkip(str(path), f"{name}: {why}"))
        return
    layers = _layers(chosen)
    runner = str(chosen.get("runner") or "").lower()
    format_ = str(chosen.get("format") or "").lower()
    if (
        runner == "mlx"
        or format_ == "safetensors"
        or any(x.get("mediaType") == MEDIA_TENSOR for x in layers)
    ):
        skips.append(StoreSkip(str(path), f"{name}: {NOT_FOR_LLAMACPP}"))
        return
    weights = next((x for x in layers if x.get("mediaType") == MEDIA_MODEL), None)
    blob = blob_path(store, str(weights.get("digest") or "")) if weights else None
    if blob is None:
        skips.append(StoreSkip(str(path), f"{name}: the manifest names no model file"))
        return
    if not blob.is_file():
        skips.append(StoreSkip(str(path), f"{name}: its model file is not in blobs"))
        return
    projector = next((x for x in layers if x.get("mediaType") == MEDIA_PROJECTOR), None)
    found = by_blob.setdefault(blob, StoreModel(blob=blob, runner=runner))
    if name not in found.tags:
        found.tags.append(name)
    if runner == RUNNER_LLAMACPP:
        found.runner = runner
    if projector is not None and found.projector is None:
        projector_blob = blob_path(store, str(projector.get("digest") or ""))
        if projector_blob is not None and projector_blob.is_file():
            found.projector = projector_blob


def read_store(store: Path) -> tuple[list[StoreModel], list[StoreSkip]]:
    """Every GGUF model the store's manifests name, one per blob, and what was
    left out and why."""
    named: dict[tuple[str, str, str, str], Path] = {}
    for directory in MANIFEST_DIRS:
        for key, path in _manifest_files(store, directory):
            named.setdefault(key, path)

    by_blob: dict[Path, StoreModel] = {}
    skips: list[StoreSkip] = []
    for (host, namespace, model, tag), path in sorted(named.items()):
        if model == "llamacpp":
            continue  # Ollama's internal `llamacpp:<sha>` name (ollama#18830)
        name = tag_name(host, namespace, model, tag)
        try:
            _read_one(store, path, name, by_blob, skips)
        except ManifestError as exc:
            skips.append(StoreSkip(str(path), f"{name}: {exc}"))
        except Exception as exc:  # one malformed manifest must not stop the scan
            log.warning("cannot read the Ollama manifest %s: %s", path, exc)
            skips.append(StoreSkip(str(path), f"{name}: the manifest is not what Ollama writes"))

    models = list(by_blob.values())
    for each in models:
        each.tags.sort()
    models.sort(key=lambda m: m.tags[0])
    return models, skips


__all__ = [
    "MIGRATED_ARCHITECTURES",
    "NOT_FOR_LLAMACPP",
    "StoreModel",
    "StoreSkip",
    "blob_path",
    "is_store",
    "read_store",
    "tag_name",
]


def local_models() -> Iterator[tuple[Path, StoreModel]]:
    """Only this machine's detected stores; callers never supply a path."""
    from .ollama_detect import find_stores

    for store in find_stores():
        models, _ = read_store(store)
        for model in models:
            yield store, model


def report_body() -> dict[str, Any]:
    from datetime import UTC, datetime

    from .gguf_context import architecture_of
    from .ollama_detect import find_stores

    stores = []
    for path in find_stores():
        store: dict[str, Any] = {"path": str(path), "models": [], "skipped": []}
        try:
            # Record inaccessible stores instead of turning them into empty ones.
            list(path.iterdir())
            models, skipped = read_store(path)
            store["skipped"] = [{"name": s.path, "detail": s.detail} for s in skipped]
            for model in models:
                architecture = architecture_of(model.blob)
                item = {
                    "tags": model.tags,
                    "digest": "sha256:" + model.blob.name.removeprefix("sha256-"),
                    "sizeBytes": model.blob.stat().st_size,
                    "runner": model.runner,
                }
                if architecture:
                    item["architecture"] = architecture
                    item["needsMigration"] = (
                        model.legacy_layout and architecture in MIGRATED_ARCHITECTURES
                    )
                if model.projector:
                    item["projectorDigest"] = "sha256:" + model.projector.name.removeprefix(
                        "sha256-"
                    )
                    item["projectorSizeBytes"] = model.projector.stat().st_size
                store["models"].append(item)
        except OSError as exc:
            store["error"] = f"Cannot read {path}: {exc}"
        stores.append(store)
    return {"stores": stores, "readAt": datetime.now(UTC).isoformat()}
