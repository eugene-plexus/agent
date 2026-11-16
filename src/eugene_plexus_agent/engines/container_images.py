"""The vLLM container image as an engine build (row 42).

`POST /v1/engines/vllm/install` with `variant: container` pulls the image
(`vllm_container.image_choice`) with Docker or Podman, as an install with
progress; nothing pulls it silently at a first start (`--pull never`).
The image pulled becomes the one in use, and the one in use before it is
kept as the way back, as two engine builds are (`acquisition.ManagedStore`);
an image that falls off that record is removed, and only an image this agent
recorded is ever removed.

The record is `<engine root>/vllm-container/images.json`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .._generated.models import (
    ContainerRuntimeKind,
    EngineInstall,
    EngineKind,
    ManagedImage,
    State,
)
from ..child_env import child_environment
from .acquisition import AcquisitionError, EngineInstaller, ManagedStore, _Progress

log = logging.getLogger(__name__)

STORE_DIR = "vllm-container"
RECORD_NAME = "images.json"


@dataclass(frozen=True)
class ImagePlan:
    """One pull."""

    cli: str
    kind: ContainerRuntimeKind
    image: str
    #: The engine version in the image, when this release pins it.
    version: str | None
    variant: str


@dataclass(frozen=True)
class ImageRecord:
    image: str
    digest: str
    version: str
    variant: str
    installed_at: str
    size_bytes: int | None = None


class ImageStore:
    """Which pulled image is in use, and the one before it."""

    def __init__(self, engine_root: Path) -> None:
        self._dir = engine_root / STORE_DIR

    @property
    def directory(self) -> Path:
        return self._dir

    def _read(self) -> dict[str, ImageRecord]:
        try:
            raw = json.loads((self._dir / RECORD_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        out: dict[str, ImageRecord] = {}
        for slot in ("current", "previous"):
            entry = raw.get(slot) if isinstance(raw, dict) else None
            if isinstance(entry, dict):
                with contextlib.suppress(TypeError):
                    out[slot] = ImageRecord(**entry)
        return out

    def _write(self, records: dict[str, ImageRecord]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / RECORD_NAME
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({slot: asdict(r) for slot, r in records.items()}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def current(self) -> ImageRecord | None:
        return self._read().get("current")

    def previous(self) -> ImageRecord | None:
        return self._read().get("previous")

    def record_pull(
        self,
        *,
        image: str,
        digest: str,
        version: str,
        variant: str,
        size: int | None = None,
    ) -> list[ImageRecord]:
        """Make this image the one in use. Returns the records that fell off,
        which nothing kept uses: those images may be removed."""
        records = self._read()
        current, previous = records.get("current"), records.get("previous")
        new = ImageRecord(
            image=image,
            digest=digest,
            version=version,
            variant=variant,
            installed_at=datetime.now(UTC).isoformat(),
            size_bytes=size,
        )
        dropped: list[ImageRecord] = []
        if current is not None and current.digest == digest:
            kept = previous
        elif previous is not None and previous.digest == digest:
            kept = current
        else:
            kept = current
            if previous is not None:
                dropped.append(previous)
        updated = {"current": new}
        if kept is not None:
            updated["previous"] = kept
        self._write(updated)
        keep = {r.digest for r in updated.values()}
        return [r for r in dropped if r.digest not in keep]

    def remove_all(self, cli: str | None) -> list[ImageRecord]:
        """Remove every image recorded here, then the record."""
        records = list(self._read().values())
        if cli is not None:
            for record in records:
                _remove_image(cli, record.digest)
        (self._dir / RECORD_NAME).unlink(missing_ok=True)
        return records

    def view(self) -> ManagedImage | None:
        records = self._read()
        current = records.get("current")
        if current is None:
            return None
        previous = records.get("previous")
        installed: datetime | None = None
        with contextlib.suppress(ValueError):
            installed = datetime.fromisoformat(current.installed_at)
        return ManagedImage(
            version=current.version,
            image=current.image,
            digest=current.digest,
            variant=current.variant,
            installedAt=installed,
            sizeBytes=current.size_bytes,
            previousVersion=previous.version if previous is not None else None,
            previousImage=previous.image if previous is not None else None,
        )


def _run(cli: str, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cli, *args],
        capture_output=True,
        env=child_environment(),
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def inspect_image(cli: str, image: str) -> tuple[str, int | None] | None:
    """(the local image ID, its size) when the image is on this machine."""
    try:
        answer = _run(cli, "image", "inspect", image)
    except (OSError, subprocess.SubprocessError):
        return None
    if answer.returncode != 0:
        return None
    try:
        parsed = json.loads(answer.stdout)
    except ValueError:
        return None
    first: Any = parsed[0] if isinstance(parsed, list) and parsed else None
    if not isinstance(first, dict) or not isinstance(first.get("Id"), str):
        return None
    size = first.get("Size")
    return first["Id"], int(size) if isinstance(size, int) else None


def remove_images(cli: str, records: list[ImageRecord]) -> None:
    """Remove these recorded images (blocking)."""
    for record in records:
        _remove_image(cli, record.digest)


def _remove_image(cli: str, digest: str) -> None:
    try:
        answer = _run(cli, "image", "rm", digest)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not remove image %s: %s", digest, exc)
        return
    if answer.returncode != 0:
        log.warning("could not remove image %s: %s", digest, answer.stderr.strip())


def _short(digest: str) -> str:
    return digest.split(":", 1)[-1][:12]


def registry_words(lines: list[str]) -> str:
    """The registry's (or the runtime's) own answer, out of a failed pull."""
    for line in reversed(lines):
        if "Error response from daemon:" in line:
            return line.split("Error response from daemon:", 1)[1].strip()
    for line in reversed(lines):
        if line.lower().startswith("error"):
            return line.split(":", 1)[1].strip() if ":" in line else line
    return lines[-1] if lines else "the pull ended with no output"


class _Layers:
    """Docker's pull lines, as "n of m layers"."""

    _DONE = ("Pull complete", "Already exists")

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.done: set[str] = set()

    def read(self, line: str) -> str | None:
        layer, sep, status = line.partition(": ")
        if sep and len(layer) == 12 and all(c in "0123456789abcdef" for c in layer):
            self.seen.add(layer)
            if status.strip() in self._DONE:
                self.done.add(layer)
            return f"{len(self.done)} of {len(self.seen)} layers"
        if line.startswith("Copying blob"):  # Podman
            self.seen.add(line)
            return f"{len(self.seen)} layers"
        return None


class ImageInstaller(EngineInstaller):
    """Runs one pull at a time for an engine, in the background, with the
    same state and progress an engine install has."""

    def __init__(self, images: ImageStore, engine: EngineKind) -> None:
        super().__init__(ManagedStore(images.directory.parent, engine), engine)
        self._images = images

    def start(self, plan: ImagePlan) -> EngineInstall:  # type: ignore[override]
        if self.running:
            raise AcquisitionError(f"an install is already running for {self._engine.value}")
        progress = _Progress(
            engine=self._engine,
            state=State.downloading,
            version=plan.version,
            variant="container",
            message=f"pulling {plan.image}",
        )
        self._progress = progress
        self._task = asyncio.create_task(
            self._pull_run(plan, progress), name=f"image-pull-{self._engine.value}"
        )
        return progress.snapshot()

    async def _pull_run(self, plan: ImagePlan, progress: _Progress) -> None:
        try:
            present = await asyncio.to_thread(inspect_image, plan.cli, plan.image)
            if present is None:
                await self._pull(plan, progress)
                present = await asyncio.to_thread(inspect_image, plan.cli, plan.image)
                if present is None:
                    raise AcquisitionError(
                        f"{plan.image} was pulled, but {plan.kind.value} does not list it"
                    )
            image_id, size = present
            version = plan.version or _short(image_id)
            progress.version = version
            dropped = self._images.record_pull(
                image=plan.image, digest=image_id, version=version, variant=plan.variant, size=size
            )
            progress.committed = True
            for record in dropped:
                await asyncio.to_thread(_remove_image, plan.cli, record.digest)
            progress.state = State.done
            progress.message = f"pulled vLLM {version}"
        except asyncio.CancelledError:
            progress.cancelled.set()
            progress.state = State.done if progress.committed else State.cancelled
            progress.message = progress.message if progress.committed else "cancelled"
            raise
        except Exception as e:
            log.warning("image pull failed: %s", e)
            cancelled = progress.cancelled.is_set() and not progress.committed
            progress.state = State.cancelled if cancelled else State.failed
            progress.message = "cancelled" if cancelled else progress.message
            progress.error = None if cancelled else str(e)
        finally:
            progress.finished_at = datetime.now(UTC)

    async def _pull(self, plan: ImagePlan, progress: _Progress) -> None:
        proc = await asyncio.create_subprocess_exec(
            plan.cli,
            "pull",
            plan.image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=child_environment(),
        )
        lines: deque[str] = deque(maxlen=20)
        layers = _Layers()

        async def read() -> None:
            assert proc.stdout is not None
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    return
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                lines.append(line)
                counted = layers.read(line)
                progress.message = (
                    f"pulling vLLM {plan.version or ''}: {counted}".replace("  ", " ")
                    if counted
                    else line[:200]
                )

        reader = asyncio.create_task(read())
        try:
            # Polled, not awaited: a CLI's helper can hold the pipe open after
            # the CLI itself has ended (`supervisor._parent_exit`).
            while proc.returncode is None or not reader.done():
                if progress.cancelled.is_set():
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    # Let the pipe close, so nothing of the pull outlives it.
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(asyncio.shield(reader), 5.0)
                    raise AcquisitionError("installation cancelled")
                if proc.returncode is not None:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(asyncio.shield(reader), 1.0)
                    break
                await asyncio.sleep(0.1)
        finally:
            if not reader.done():
                reader.cancel()
                with contextlib.suppress(asyncio.CancelledError, BaseException):
                    await reader
        if proc.returncode != 0:
            raise AcquisitionError(f"Pulling {plan.image} failed: {registry_words(list(lines))}")


__all__ = [
    "ImageInstaller",
    "ImagePlan",
    "ImageRecord",
    "ImageStore",
    "inspect_image",
    "registry_words",
    "remove_images",
]
