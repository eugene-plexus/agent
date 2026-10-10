"""Preparation jobs: an engine prepares a model for itself on this node (LS5).

library-sources-and-engines.md §4.6 and §6.6. The adapter owns the recipe;
this runs it in a thread, one at a time per node (B52), says where it is
(the run operation's `PreparationStatus`) and stops it on cancel (B53). The
run worker starts and polls a job from the operation's `preparing` step,
keyed by the operation's id; nothing here talks to the library.

A job lives in memory only. An agent that restarts mid-preparation starts
the recipe again at the next claim, and the recipe is written so that a
second run skips what the first finished (Strata's setup keeps its own marks).
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import orphan_kill

log = logging.getLogger(__name__)

#: How often a running recipe re-measures what it has written.
MEASURE_EVERY_SECONDS = 2.0


class PreparationError(Exception):
    """A preparation that cannot start, or that stopped, naming the cause."""


class PreparationCancelled(PreparationError):
    """The person cancelled the operation."""


@dataclass(frozen=True)
class PreparationResult:
    """What a finished preparation made, for the library to list."""

    #: The engine's entry file, as this node reaches it.
    entry: Path
    #: The prepared model's name: its provenance file is `<name>.eugene-prepared.json`.
    name: str
    recipe: str
    recipe_version: str
    #: `PreparedSource`: what it was made from.
    source: dict[str, Any]
    #: What the engine's own files say of it (LS7: title, architecture,
    #: quantization, contextLength, mode, files), as provenance fields with
    #: the files relative to the entry's folder.
    facts: dict[str, Any] = field(default_factory=dict)


class Recipe(Protocol):
    """One engine's preparation of one model, planned and ready to run."""

    @property
    def output(self) -> Path:
        """The folder it writes into."""
        ...

    @property
    def bytes_needed(self) -> int | None:
        """What it expects to write there in all, when known."""
        ...

    def run(self, progress: Progress) -> PreparationResult: ...


def folder_bytes(path: Path) -> int:
    """Every file under `path`, summed; what cannot be read counts nothing."""
    total = 0
    for directory, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(directory, name)).st_size
            except OSError:
                continue
    return total


@dataclass
class Progress:
    """Where one preparation is: written by its thread, read by the worker."""

    state: str = "waiting"
    step: str | None = None
    message: str | None = None
    bytes_written: int | None = None
    bytes_needed: int | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    result: PreparationResult | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)

    def check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise PreparationCancelled("the preparation was cancelled")

    def warn(self, text: str) -> None:
        if text not in self.warnings:
            self.warnings.append(text)

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed", "cancelled")

    def snapshot(self) -> dict[str, Any]:
        """The run operation's `PreparationStatus`."""
        return {
            "state": self.state,
            "step": self.step,
            "message": self.message,
            "bytesWritten": self.bytes_written,
            "bytesNeeded": self.bytes_needed,
            "warnings": list(self.warnings),
        }


def run_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    progress: Progress,
    on_output: Callable[[str], None] | None = None,
    measure: Callable[[], None] | None = None,
) -> int:
    """Run one of the engine's own tools to its end, writing its output to
    `log_path`: no console, no shell, no input (a question it asks ends it
    rather than waiting for ever). The new output is handed to `on_output`
    and `measure` is called every couple of seconds; a cancel stops the tool
    and everything it started. Returns its exit code."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as output:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            **orphan_kill.kwargs_for_platform(),
        )
        job = orphan_kill.windows_job()
        if job is not None:
            job.assign(process.pid)
        read = 0
        last = 0.0
        try:
            while True:
                code = process.poll()
                now = time.perf_counter()
                if code is not None or now - last >= MEASURE_EVERY_SECONDS:
                    last = now
                    read = _hand_on(log_path, read, on_output)
                    if measure is not None:
                        measure()
                if code is not None:
                    return code
                progress.check_cancelled()
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                _stop_tree(process)


def _hand_on(path: Path, offset: int, on_output: Callable[[str], None] | None) -> int:
    """Pass on what the tool wrote since `offset`; the new offset."""
    if on_output is None:
        return offset
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return offset
    if data:
        on_output(data.decode("utf-8", errors="replace"))
    return offset + len(data)


def _stop_tree(process: subprocess.Popen[bytes]) -> None:
    """The tool and everything it started: a venv's python.exe is a launcher,
    and setup runs its own tools as children."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class _Job:
    def __init__(self, recipe: Recipe, engine: str) -> None:
        self.recipe = recipe
        self.engine = engine
        self.progress = Progress(bytes_needed=recipe.bytes_needed)
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.progress.state = "running"
        self.progress.message = None
        self.thread = threading.Thread(target=self._run, name="preparation", daemon=True)
        self.thread.start()

    @property
    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _run(self) -> None:
        progress = self.progress
        try:
            progress.result = self.recipe.run(progress)
            progress.state = "done"
        except PreparationCancelled as exc:
            progress.error = str(exc)
            progress.state = "cancelled"
        except PreparationError as exc:
            progress.error = str(exc)
            progress.state = "failed"
        except Exception as exc:  # a recipe defect still ends the job, naming it
            log.exception("preparation failed")
            progress.error = f"{type(exc).__name__}: {exc}"
            progress.state = "failed"


class PreparationJobs:
    """This node's preparations, by run operation id. One runs at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}

    def _running(self) -> bool:
        return any(j.alive for j in self._jobs.values())

    def poll(self, key: str) -> Progress | None:
        """The job's progress, starting it if it was waiting and nothing runs."""
        with self._lock:
            job = self._jobs.get(key)
            if job is None:
                return None
            if job.thread is None and not self._running():
                job.start()
            return job.progress

    def start(self, key: str, recipe: Recipe, *, engine: str) -> Progress:
        """Add a planned job: it starts now, or waits for the one running."""
        with self._lock:
            job = self._jobs.get(key)
            if job is None:
                job = self._jobs[key] = _Job(recipe, engine)
            if job.thread is None:
                if self._running():
                    job.progress.message = "waiting for another preparation on this node"
                else:
                    job.start()
            return job.progress

    def busy_for(self, engine: str) -> bool:
        """A preparation by `engine` is running or waiting: it must stay installed."""
        with self._lock:
            return any(
                j.engine == engine and (j.alive or not j.progress.finished)
                for j in self._jobs.values()
            )

    def cancel_except(self, keep: set[str]) -> None:
        """Stop every job whose operation is no longer waiting on it (it was
        cancelled, or ended), and forget the ones that have stopped."""
        with self._lock:
            for key, job in list(self._jobs.items()):
                if key in keep:
                    continue
                job.progress.cancelled.set()
                if not job.alive:
                    del self._jobs[key]

    def forget(self, key: str) -> None:
        with self._lock:
            job = self._jobs.get(key)
            if job is not None and not job.alive:
                del self._jobs[key]
