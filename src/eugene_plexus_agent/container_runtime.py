"""Docker or Podman on this machine (row 42, `specs/docs/design/rows/row-42.md`).

vLLM can run from its official container image instead of a Python
environment the owner builds. This module is the agent's whole knowledge of
the container runtime: which one is here, whether this account may use it,
whether GPUs reach its containers, and the few commands the agent runs with
it (inspect, stop, remove, the orphan sweep at start).

**What a container may reach is decided elsewhere and kept small**
(`engines/vllm_container.py`): the model's own folder, read only, and a
per-runtime compile cache. No container runtime socket is ever mounted
into anything, and Eugene in a container is never given one by this code.

Every container the agent starts carries the label
`eugene-plexus.agent=<id>`, where the id is a hash of this agent's data
folder: two agents on one host (an acceptance run beside a real install)
never stop each other's containers.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._generated.models import ContainerRuntime, ContainerRuntimeKind
from .child_env import child_environment

log = logging.getLogger(__name__)

#: How long one detection stays current (design: at start and every 10 minutes).
CHECK_SECONDS = 600.0
#: One CLI call; `docker info` on a cold Docker Desktop can take several seconds.
_TIMEOUT_SECONDS = 15.0

LABEL_AGENT = "eugene-plexus.agent"
LABEL_RUNTIME = "eugene-plexus.runtime"
LABEL_NODE = "eugene-plexus.node"

TOOLKIT_GUIDE = (
    "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
)

_NOT_INSTALLED = "Neither Docker nor Podman is installed on this machine."
_INSTALL_FIX = (
    "Install Docker (Docker Desktop on Windows or a Mac) to run vLLM from its official "
    "container, then restart Eugene."
)
_INSIDE = "Eugene runs in a container without Docker access."
_INSIDE_FIX = (
    "To run vLLM's container, install Eugene on the host, or (expert) give this container "
    "the Docker socket, which gives it control of the host."
)


@dataclass(frozen=True)
class Found:
    """The container runtime as one detection saw it."""

    kind: ContainerRuntimeKind | None = None
    #: The CLI's path, as found on PATH.
    cli: str | None = None
    version: str | None = None
    usable: bool = False
    gpus: bool = False
    #: Docker Desktop: containers run in a virtual machine.
    desktop: bool = False
    #: The daemon's kernel is WSL 2's (Docker Desktop on Windows).
    wsl: bool = False
    inside_container: bool = False
    error: str | None = None
    fix: str | None = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def view(self) -> ContainerRuntime:
        return ContainerRuntime(
            kind=self.kind,
            version=self.version,
            usable=self.usable,
            gpus=self.gpus,
            desktop=self.desktop or None,
            insideContainer=self.inside_container or None,
            error=self.error,
            fix=self.fix,
            checkedAt=self.checked_at,
        )


def _inside_a_container() -> bool:
    """Docker writes `/.dockerenv`; Podman writes `/run/.containerenv`."""
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def run(
    cli: str, *args: str, timeout: float = _TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """One CLI call with this agent's environment minus Eugene's own variables
    (`DOCKER_HOST` and the like still reach it)."""
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


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()


def _refusal(kind: ContainerRuntimeKind, text: str) -> tuple[str, str | None]:
    """The runtime's own words, and the fix for the cases we know."""
    words = _first_line(text) or f"{kind.value} did not answer"
    lowered = text.lower()
    name = "Docker" if kind is ContainerRuntimeKind.docker else "Podman"
    if "permission denied" in lowered:
        return (
            f"This account may not use {name}: {words}",
            f"Add this account to the {kind.value} group, then restart Eugene."
            if kind is ContainerRuntimeKind.docker
            else "Give this account access to Podman, then restart Eugene.",
        )
    if any(
        s in lowered
        for s in (
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "error during connect",
            "cannot connect to podman",
            "the system cannot find the file specified",
        )
    ):
        return (
            f"{name} is installed, but it is not running: {words}",
            f"Start {'Docker Desktop' if kind is ContainerRuntimeKind.docker else 'Podman'}"
            " (or its service), then restart Eugene.",
        )
    return words, None


def _cdi_nvidia() -> bool:
    for folder in ("/etc/cdi", "/var/run/cdi"):
        for suffix in ("yaml", "json"):
            if Path(folder, f"nvidia.{suffix}").exists():
                return True
    return False


def _from_docker_info(cli: str, info: dict[str, Any]) -> Found:
    runtimes = info.get("Runtimes") if isinstance(info.get("Runtimes"), dict) else {}
    devices = (
        info.get("DiscoveredDevices") if isinstance(info.get("DiscoveredDevices"), list) else []
    )
    system = str(info.get("OperatingSystem") or "")
    kernel = str(info.get("KernelVersion") or "")
    desktop = "docker desktop" in system.lower()
    wsl = "microsoft" in kernel.lower() or "wsl" in kernel.lower()
    gpus = (
        "nvidia" in (runtimes or {})
        or any(
            isinstance(d, dict) and str(d.get("ID", "")).startswith("nvidia.com/gpu")
            for d in devices or []
        )
        # Docker Desktop's WSL 2 backend passes NVIDIA GPUs to containers.
        or (desktop and wsl)
    )
    return Found(
        kind=ContainerRuntimeKind.docker,
        cli=cli,
        version=str(info.get("ServerVersion") or "") or None,
        usable=True,
        gpus=gpus,
        desktop=desktop,
        wsl=wsl,
    )


def _from_podman_info(cli: str, info: dict[str, Any]) -> Found:
    host = info.get("host") if isinstance(info.get("host"), dict) else {}
    version = info.get("version") if isinstance(info.get("version"), dict) else {}
    kernel = str((host or {}).get("kernel") or "")
    return Found(
        kind=ContainerRuntimeKind.podman,
        cli=cli,
        version=str((version or {}).get("Version") or "") or None,
        usable=True,
        gpus=_cdi_nvidia(),
        wsl="microsoft" in kernel.lower(),
    )


def _detect_one(kind: ContainerRuntimeKind, cli: str) -> Found:
    try:
        answer = run(cli, "info", "--format", "{{json .}}")
    except (OSError, subprocess.SubprocessError) as exc:
        return Found(kind=kind, cli=cli, error=f"{kind.value} info failed: {exc}")
    info: dict[str, Any] | None = None
    with contextlib.suppress(ValueError):
        parsed = json.loads(_first_line(answer.stdout) or "null")
        info = parsed if isinstance(parsed, dict) else None
    server_errors = (info or {}).get("ServerErrors") if info else None
    if answer.returncode != 0 or server_errors or info is None:
        text = "\n".join(
            [*(str(e) for e in server_errors or []), answer.stderr or "", answer.stdout or ""]
        )
        error, fix = _refusal(kind, text)
        return Found(kind=kind, cli=cli, error=error, fix=fix)
    if kind is ContainerRuntimeKind.docker:
        return _from_docker_info(cli, info)
    return _from_podman_info(cli, info)


def detect() -> Found:
    """Docker first, then Podman; the first that answers wins, else the first
    found with its refusal, else why nothing is here."""
    inside = _inside_a_container()
    first: Found | None = None
    from . import exec_trust

    for kind in (ContainerRuntimeKind.docker, ContainerRuntimeKind.podman):
        cli = shutil.which(kind.value)
        if cli is None:
            continue
        if exec_trust.own_account():
            # Run as Eugene's own account, only a program nobody else can
            # change (row 8), as for every engine program.
            try:
                cli = str(exec_trust.trusted_program(cli))
            except exec_trust.Untrusted as exc:
                first = first or Found(kind=kind, cli=None, error=str(exc))
                continue
        found = _detect_one(kind, cli)
        if found.usable:
            return _with_inside(found, inside)
        first = first or found
    if first is not None:
        return _with_inside(first, inside)
    if inside:
        return Found(inside_container=True, error=_INSIDE, fix=_INSIDE_FIX)
    return Found(error=_NOT_INSTALLED, fix=_INSTALL_FIX)


def _with_inside(found: Found, inside: bool) -> Found:
    if not inside:
        return found
    return Found(**{**found.__dict__, "inside_container": True})


_lock = threading.Lock()
_cached: tuple[float, Found] | None = None


def current(*, max_age: float = CHECK_SECONDS) -> Found:
    """The last detection, made again when it is older than `max_age` seconds."""
    global _cached
    with _lock:
        cached = _cached
    now = time.perf_counter()
    if cached is not None and now - cached[0] < max_age:
        return cached[1]
    found = detect()
    with _lock:
        _cached = (time.perf_counter(), found)
    if found.usable:
        log.debug("container runtime: %s %s, GPUs %s", found.kind, found.version, found.gpus)
    return found


def forget() -> None:
    """Drop the cached detection (tests; a config change that should be seen now)."""
    global _cached
    with _lock:
        _cached = None


# --------------------------------------------------------------------------- #
# Names and labels
# --------------------------------------------------------------------------- #

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def agent_id(data_dir: Path) -> str:
    """This agent's label value: a hash of its data folder, stable across
    restarts and different for a second agent on the same host."""
    return hashlib.sha256(str(data_dir.resolve()).encode("utf-8")).hexdigest()[:12]


def safe_name(runtime: str) -> str:
    """A runtime's name in the characters Docker accepts. A name that had to
    change gets a short hash, so two runtimes never share one."""
    safe = _UNSAFE.sub("-", runtime).strip("-.") or "runtime"
    if safe != runtime:
        safe = f"{safe}-{hashlib.sha256(runtime.encode('utf-8')).hexdigest()[:6]}"
    return safe


def container_name(agent: str, runtime: str) -> str:
    """`ep-<agent>-<runtime>` (`safe_name`)."""
    return f"ep-{agent}-{safe_name(runtime)}"


# --------------------------------------------------------------------------- #
# Commands on containers
# --------------------------------------------------------------------------- #


def sweep_orphans(found: Found, agent: str) -> list[str]:
    """Remove every container labelled with this agent. Called at agent start,
    before any runtime starts, so every one of them is an orphan: left running
    by an agent that was killed, holding a GPU and a port. Returns their names."""
    if not found.usable or found.cli is None:
        return []
    try:
        listed = run(
            found.cli,
            "ps",
            "-a",
            "--filter",
            f"label={LABEL_AGENT}={agent}",
            "--format",
            "{{.Names}}",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not list containers left by an earlier run: %s", exc)
        return []
    if listed.returncode != 0:
        log.warning(
            "could not list containers left by an earlier run: %s", _first_line(listed.stderr)
        )
        return []
    names = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not names:
        return []
    try:
        removed = run(found.cli, "rm", "-f", *names, timeout=60.0)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not stop containers left by an earlier run (%s): %s", names, exc)
        return []
    if removed.returncode != 0:
        log.warning(
            "could not stop containers left by an earlier run (%s): %s",
            names,
            _first_line(removed.stderr),
        )
        return []
    log.warning(
        "stopped %d container(s) an earlier run of this agent left running: %s",
        len(names),
        ", ".join(names),
    )
    return names


def remove(cli: str, name: str) -> None:
    """Remove a container of this name before one is started under it: what a
    stop that was cut short left behind. Absent is the usual answer."""
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        run(cli, "rm", "-f", name, timeout=30.0)


def stop(cli: str, name: str, *, seconds: int = 30) -> str | None:
    """`docker stop -t <seconds> <name>`: the engine gets `seconds` to end, then
    is killed. None when it stopped; the runtime's own words when not."""
    try:
        answer = run(cli, "stop", "-t", str(seconds), name, timeout=seconds + 30.0)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)
    if answer.returncode == 0:
        return None
    text = answer.stderr or answer.stdout
    if "no such container" in text.lower():
        return None
    return _first_line(text)


def inspect(cli: str, name: str) -> dict[str, Any] | None:
    """`docker inspect <name>`'s object, or None."""
    try:
        answer = run(cli, "inspect", "--type", "container", name)
    except (OSError, subprocess.SubprocessError):
        return None
    if answer.returncode != 0:
        return None
    try:
        parsed = json.loads(answer.stdout)
    except ValueError:
        return None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]
    return None


__all__ = [
    "CHECK_SECONDS",
    "LABEL_AGENT",
    "LABEL_NODE",
    "LABEL_RUNTIME",
    "TOOLKIT_GUIDE",
    "Found",
    "agent_id",
    "container_name",
    "current",
    "detect",
    "forget",
    "inspect",
    "remove",
    "run",
    "stop",
    "sweep_orphans",
]
