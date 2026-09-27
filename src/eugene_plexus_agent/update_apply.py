"""Install a newer version, from outside this agent.

**Outside, because the installer's first act on an upgrade is to stop the
agent**, and this agent kills what it supervises when it stops
(`orphan_kill`). So the installer never runs as the agent's child:

- **Windows** (service or logon task): a one-shot scheduled task, as SYSTEM
  for the service and as the person for a per-user install. It runs
  PowerShell from Windows itself, never Python from the install: the
  installer stops processes running from inside the install folder, and
  an updater there would be one.
- **Linux, per-user**: a transient `systemd-run --user` unit, outside the
  agent's own cgroup, so stopping the agent's unit does not stop it.
- **Linux, system install**: the Eugene account owns its own install
  folder, so root must never run anything from it -- the account could put
  code there and have root run it. The agent writes the one thing it
  may, a request naming the target, and a root unit the installer set up
  (`eugene-plexus-update.path`) runs a root-owned helper outside the
  folder. The helper checks the target is a specs commit or release tag,
  downloads our installer for it itself, and runs it.

**The container never can**: its code is its image. Its answer is the
steps for pulling a new image, in the words of the platform its template
names, or general Docker words when it names none.

The installer runs with `-Update` / `--update`: packages at the new pins,
the service host refreshed, everything else about the install -- its
autostart, the tray icon, who may start and stop it, its folder's
permissions -- kept exactly as it is. Afterwards the wrapper writes
`update/last.json`, which the agent that comes back reports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from ._generated.models import (
    InstallMechanism,
    NodeInstall,
    UpdateApply,
    UpdateChannel,
    UpdateOutcome,
    UpdateRun,
    UpdateStep,
)
from ._private_files import write_private
from .updates import Fetch, Target, fetch

log = logging.getLogger(__name__)

UPDATE_DIR = "update"
RUNNING_FILE = "running.json"
LAST_FILE = "last.json"
REQUEST_FILE = "request"
LOG_FILE = "update.log"
WINDOWS_TASK = "EugenePlexusUpdate"
SYSTEMD_USER_UNIT = "eugene-plexus-update"
#: What the Linux system install's root unit watches for (install.sh).
SYSTEM_REQUEST_UNIT = "eugene-plexus-update.path"
IMAGE = "ghcr.io/eugene-plexus/control-plane"

#: An update that has not reported back in this long did not finish.
RUNNING_EXPIRES_SECONDS = 45 * 60


class UpdateRefused(Exception):
    """This update cannot be started, and why, in a sentence."""


# --------------------------------------------------------------------------- #
# What a person sees before pressing anything
# --------------------------------------------------------------------------- #


def _container_steps(install: NodeInstall, target: Target | None) -> list[UpdateStep]:
    container = install.container
    assert container is not None
    image = container.image
    repo = image.rsplit(":", 1)[0] if ":" in image.rsplit("/", 1)[-1] else image
    release = target.release if target is not None else None
    pinned = release is not None
    new_image = f"{repo}:{release}" if pinned else image
    if container.host == "unraid":
        if pinned:
            return [
                UpdateStep(
                    text=(
                        f"On Unraid's Docker tab, edit this container and change Repository to "
                        f"{new_image}, then Apply."
                    )
                )
            ]
        return [UpdateStep(text="On Unraid's Docker tab, choose Force Update for this container.")]
    if container.host == "compose":
        steps = []
        if pinned:
            steps.append(UpdateStep(text=f"In compose.yaml, change the image to {new_image}."))
        steps.append(
            UpdateStep(
                text="Then, in the folder with compose.yaml:",
                command="docker compose pull && docker compose up -d",
            )
        )
        return steps
    return [
        UpdateStep(text="Pull the new image:", command=f"docker pull {new_image}"),
        UpdateStep(
            text=(
                "Recreate the container from it with the same settings and the same /data "
                "and /models folders -- your settings and models live there, not in the "
                "container. With Compose: docker compose pull && docker compose up -d."
            )
        ),
        UpdateStep(
            text=(
                "Portainer, Synology Container Manager, TrueNAS and similar tools have a "
                "button that does both."
            )
        ),
    ]


def _installer_command(install: NodeInstall, target: Target | None) -> str:
    """The one-line install for this machine, at the target, for a person to run."""
    channel_ref = target.ref if target is not None else "main"
    if sys.platform == "win32":
        url = f"https://raw.githubusercontent.com/eugene-plexus/specs/{channel_ref}/scripts/install.ps1"
        return f"irm {url} | iex"
    url = f"https://raw.githubusercontent.com/eugene-plexus/specs/{channel_ref}/scripts/install.sh"
    return f"curl -fsSL {url} | sh"


def plan(install: NodeInstall, target: Target | None, *, system_unit_ready: bool) -> UpdateApply:
    """Whether this install can update itself, and when not, what to do instead."""
    if install.development:
        return UpdateApply(
            possible=False,
            reason="This is a development checkout; update it with git.",
        )
    mechanism = install.mechanism
    if mechanism is InstallMechanism.container:
        return UpdateApply(
            possible=False,
            reason=(
                "This machine runs in a container, which cannot update itself: its code is "
                "its image. Update it by pulling the new image."
            ),
            steps=_container_steps(install, target),
        )
    if mechanism in (
        InstallMechanism.windows_service,
        InstallMechanism.windows_task,
        InstallMechanism.systemd_user,
    ):
        return UpdateApply(possible=True)
    if mechanism is InstallMechanism.systemd_system:
        if system_unit_ready:
            return UpdateApply(possible=True)
        return UpdateApply(
            possible=False,
            reason=(
                "This install was made before it could update itself. Run the installer "
                "once more and it can from then on."
            ),
            steps=[UpdateStep(text="Run:", command=_installer_command(install, target))],
        )
    if mechanism is InstallMechanism.launchd:
        return UpdateApply(
            possible=False,
            reason="Updating from the app is not built for macOS yet.",
            steps=[UpdateStep(text="Run:", command=_installer_command(install, target))],
        )
    return UpdateApply(
        possible=False,
        reason=(
            "Nothing starts this agent automatically -- it was started by hand -- so "
            "nothing would bring it back after an update."
        ),
        steps=[UpdateStep(text="Run the installer:", command=_installer_command(install, target))],
    )


# --------------------------------------------------------------------------- #
# The records an update leaves
# --------------------------------------------------------------------------- #


def update_dir(prefix: Path) -> Path:
    return prefix / UPDATE_DIR


def _read_run(path: Path) -> UpdateRun | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        return UpdateRun.model_validate(raw)
    except (OSError, ValueError):
        return None


def running(prefix: Path) -> UpdateRun | None:
    """The update in progress, if one is; one that never reported back is not."""
    run = _read_run(update_dir(prefix) / RUNNING_FILE)
    if run is None:
        return None
    age = (datetime.now(UTC) - run.startedAt).total_seconds()
    if age > RUNNING_EXPIRES_SECONDS:
        return None
    return run


def last(prefix: Path) -> UpdateRun | None:
    run = _read_run(update_dir(prefix) / LAST_FILE)
    if run is not None and run.outcome is UpdateOutcome.running:
        return None
    return run


def expired_run(prefix: Path) -> UpdateRun | None:
    """An update that started and never reported back, as a failure."""
    run = _read_run(update_dir(prefix) / RUNNING_FILE)
    if run is None:
        return None
    age = (datetime.now(UTC) - run.startedAt).total_seconds()
    if age <= RUNNING_EXPIRES_SECONDS:
        return None
    return run.model_copy(
        update={
            "outcome": UpdateOutcome.failed,
            "detail": (
                f"The update to {run.target[:12]} started at {run.startedAt:%Y-%m-%d %H:%M} UTC "
                f"and never reported back. See {run.log} on this machine."
            ),
        }
    )


# --------------------------------------------------------------------------- #
# Starting one
# --------------------------------------------------------------------------- #


def _download(installer_url: str, sha256: str | None, dest: Path, get: Fetch) -> None:
    body = get(installer_url)
    if sha256 is not None:
        digest = hashlib.sha256(body).hexdigest()
        if digest != sha256.lower():
            raise UpdateRefused(
                f"the installer downloaded from {installer_url} does not match the checksum "
                f"its release published ({digest[:12]} against {sha256[:12]}), so it was not run"
            )
    if not body.strip():
        raise UpdateRefused(f"{installer_url} answered with an empty installer")
    write_private(dest, body)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _template(name: str) -> str:
    return (resources.files("eugene_plexus_agent") / "update_templates" / name).read_text(
        encoding="utf-8"
    )


def _render(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace(f"@@{key}@@", value)
    if "@@" in text:
        raise UpdateRefused(f"an update script placeholder was left unfilled: {text[:80]!r}")
    return text


def windows_wrapper(
    *, prefix: Path, target: str, started: datetime, service: bool, log_path: Path
) -> str:
    """The PowerShell a one-shot task runs: install, report, remove itself.

    The installer runs with `-Update`, and with `-NoService` for a
    per-user install, because the installer decides service or task from
    its flags and elevation -- and the task runs a service install as
    SYSTEM, which is elevated whatever the install was.
    """
    flags = ["-Update"] if service else ["-NoService", "-Update"]
    restart = (
        "Start-Service -Name EugenePlexusAgent -ErrorAction SilentlyContinue"
        if service
        else "Start-ScheduledTask -TaskName EugenePlexusAgent -ErrorAction SilentlyContinue"
    )
    return _render(
        _template("run-update.ps1.template"),
        {
            "DIR": _ps_quote(str(update_dir(prefix))),
            "LOG": _ps_quote(str(log_path)),
            "STARTED": _ps_quote(started.isoformat()),
            "TARGET": _ps_quote(target),
            "PREFIX": _ps_quote(str(prefix)),
            "FLAGS": ", ".join(_ps_quote(f) for f in flags),
            "RESTART": restart,
        },
    )


def posix_wrapper(*, prefix: Path, target: str, started: datetime, log_path: Path) -> str:
    """The shell a transient user unit runs, for a per-user Linux install."""
    q = shlex.quote
    return _render(
        _template("run-update.sh.template"),
        {
            "DIR": q(str(update_dir(prefix))),
            "LOG": q(str(log_path)),
            "TARGET": q(target),
            "STARTED": q(started.isoformat()),
        },
    )


def system_unit_ready() -> bool:
    """Whether the root unit a system install needs is there to be asked."""
    return Path("/etc/systemd/system", SYSTEM_REQUEST_UNIT).exists()


def _run(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, capture_output=True, timeout=30, check=False)


def _failed(what: str, proc: subprocess.CompletedProcess[bytes]) -> UpdateRefused:
    from .reach import oem_encoding

    said = (proc.stderr or proc.stdout or b"").decode(oem_encoding(), errors="replace").strip()
    return UpdateRefused(f"{what} failed (exit {proc.returncode}): {said or 'no output'}")


def start(
    *,
    install: NodeInstall,
    target: Target,
    prefix: Path,
    get: Fetch = fetch,
    run: Any = _run,
) -> UpdateRun:
    """Start the installer for `target` outside this agent, and say so."""
    directory = update_dir(prefix)
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    log_path = directory / LOG_FILE
    record = UpdateRun(
        target=target.ref,
        startedAt=started,
        outcome=UpdateOutcome.running,
        detail=f"Installing {target.release or target.ref[:12]}.",
        log=str(log_path),
    )
    mechanism = install.mechanism

    if mechanism in (InstallMechanism.windows_service, InstallMechanism.windows_task):
        installer = target.installers.get("install.ps1")
        if installer is None:
            raise UpdateRefused(f"{target.ref} publishes no Windows installer")
        _download(installer.url, installer.sha256, directory / "install.ps1", get)
        wrapper = directory / "run-update.ps1"
        service = mechanism is InstallMechanism.windows_service
        # A byte-order mark: PowerShell 5.1 reads a file without one in the
        # ANSI code page, and a per-user prefix carries the person's name.
        write_private(
            wrapper,
            b"\xef\xbb\xbf"
            + windows_wrapper(
                prefix=prefix,
                target=target.ref,
                started=started,
                service=service,
                log_path=log_path,
            ).encode("utf-8"),
        )
        write_private(directory / RUNNING_FILE, record.model_dump_json().encode("utf-8"))
        powershell = str(
            Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
            / "System32"
            / "WindowsPowerShell"
            / "v1.0"
            / "powershell.exe"
        )
        action = (
            f'"{powershell}" -NoProfile -NonInteractive -ExecutionPolicy Bypass '
            f'-WindowStyle Hidden -File "{wrapper}"'
        )
        create = ["schtasks", "/Create", "/TN", WINDOWS_TASK, "/TR", action, "/SC", "ONCE"]
        create += ["/ST", "00:00", "/F"]
        if service:
            create += ["/RU", "SYSTEM", "/RL", "HIGHEST"]
        proc = run(create)
        if proc.returncode != 0:
            (directory / RUNNING_FILE).unlink(missing_ok=True)
            raise _failed("registering the update task", proc)
        proc = run(["schtasks", "/Run", "/TN", WINDOWS_TASK])
        if proc.returncode != 0:
            (directory / RUNNING_FILE).unlink(missing_ok=True)
            run(["schtasks", "/Delete", "/TN", WINDOWS_TASK, "/F"])
            raise _failed("starting the update task", proc)
        log.warning("update to %s started as the %s task", target.ref, WINDOWS_TASK)
        return record

    if mechanism is InstallMechanism.systemd_user:
        installer = target.installers.get("install.sh")
        if installer is None:
            raise UpdateRefused(f"{target.ref} publishes no installer for this system")
        systemd_run = shutil.which("systemd-run")
        if systemd_run is None:
            raise UpdateRefused("systemd-run is not on this machine's PATH")
        _download(installer.url, installer.sha256, directory / "install.sh", get)
        wrapper = directory / "run-update.sh"
        write_private(
            wrapper,
            posix_wrapper(
                prefix=prefix, target=target.ref, started=started, log_path=log_path
            ).encode("utf-8"),
        )
        write_private(directory / RUNNING_FILE, record.model_dump_json().encode("utf-8"))
        unit = f"{SYSTEMD_USER_UNIT}-{int(started.timestamp())}"
        proc = run([systemd_run, "--user", f"--unit={unit}", "--collect", "/bin/sh", str(wrapper)])
        if proc.returncode != 0:
            (directory / RUNNING_FILE).unlink(missing_ok=True)
            raise _failed("starting the update unit", proc)
        log.warning("update to %s started as the %s unit", target.ref, unit)
        return record

    if mechanism is InstallMechanism.systemd_system:
        if not system_unit_ready():
            raise UpdateRefused(
                "this install was made before it could update itself; run the installer once more"
            )
        # The ONLY thing this account hands root: which release or commit.
        # The root helper checks its shape and downloads our installer for
        # it itself -- nothing written here is ever executed as root.
        write_private(directory / RUNNING_FILE, record.model_dump_json().encode("utf-8"))
        channel = "releases" if target.channel is UpdateChannel.releases else "edge"
        write_private(directory / REQUEST_FILE, f"{target.ref} {channel}\n".encode())
        log.warning("update to %s requested of the root update unit", target.ref)
        return record

    reason = plan(install, target, system_unit_ready=False).reason
    raise UpdateRefused(reason or "this install cannot update itself")


__all__ = [
    "UpdateRefused",
    "expired_run",
    "last",
    "plan",
    "running",
    "start",
    "system_unit_ready",
    "update_dir",
]
