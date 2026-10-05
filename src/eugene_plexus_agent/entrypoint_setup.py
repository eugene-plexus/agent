"""Applying, confirming and turning off the container's one HTTPS port.

Before 2026-10-05 the owner applied a configuration by downloading a file,
copying it into `/data`, adding a container variable and recreating the
container. Troy, after doing it on his NAS: *no weekend LLM enthusiast is
going to do all of this.* So Settings applies it here: the file is written
beside `agent.yaml` (or at the variable's path, when one is set), and this
agent restarts itself in place. The file's presence is what turns the mode
on; the variable is only for a file somewhere else.

**The page that applies a configuration stops answering at the address it
was opened at**, so a wrong proxy address or DNS name would lock the owner
out of the console with nothing on screen. So an applied configuration is
*on approval*: until an operator request arrives through the new entry
point, `pending.json` records what was there before, and if none arrives
within `entrypoint_confirm_seconds` of a start (15 minutes) this agent puts
that back and restarts on it. A start that fails outright with a pending
file goes back at once. The configuration that did not work is kept beside
the file as `<file>.reverted`, and the reason is shown on the setup page.

Turning it off keeps the file as `<file>.disabled` and restarts on the
direct ports.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn

from ._private_files import write_private

log = logging.getLogger(__name__)

FILE_NAME = "entrypoint.json"
DISABLED_SUFFIX = ".disabled"
REVERTED_SUFFIX = ".reverted"
_PENDING = "pending.json"
_REVERTED = "reverted.json"


def _now() -> datetime:
    return datetime.now(UTC)


def default_path(settings: Any) -> Path:
    """`entrypoint.json` beside `agent.yaml`: `/data/entrypoint.json` in the image."""
    return Path(settings.config_file).resolve().parent / FILE_NAME


def state_dir(settings: Any) -> Path:
    """The entry point's private directory, where its approval markers live."""
    return Path(settings.config_file).resolve().parent / "entrypoint"


def config_path(settings: Any) -> Path:
    """The file this agent reads: the variable's when set, else the default."""
    named = settings._entrypoint_named or settings.entrypoint_config
    return Path(named) if named else default_path(settings)


def unavailable_reason(settings: Any) -> str | None:
    """None where the entry point can run; otherwise why it cannot, in a sentence."""
    if os.name != "posix":
        return (
            "One HTTPS port is part of the Linux container image; this machine runs "
            "Eugene directly, so it keeps its direct ports."
        )
    if shutil.which(settings.entrypoint_binary) is None:
        return (
            f"The bundled HTTPS proxy ({settings.entrypoint_binary}) is not installed here. "
            "One HTTPS port is part of the Linux container image."
        )
    return None


# --------------------------------------------------------------------------- #
# the approval markers
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_marker(settings: Any, name: str, value: dict[str, Any]) -> None:
    directory = state_dir(settings)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private(directory / name, json.dumps(value))


def pending(settings: Any) -> dict[str, Any] | None:
    """The approval record of a configuration applied and not yet confirmed."""
    return _read_json(state_dir(settings) / _PENDING)


def reverted(settings: Any) -> str | None:
    """Why the last applied configuration went back, until the next apply."""
    record = _read_json(state_dir(settings) / _REVERTED)
    reason = record.get("reason") if record else None
    return reason if isinstance(reason, str) else None


def missing_file_sentence(settings: Any, path: Path) -> str | None:
    """What a missing file means when Settings is why it is missing."""
    disabled = path.with_name(path.name + DISABLED_SUFFIX)
    if disabled.is_file():
        return (
            "One HTTPS port was turned off from Settings, so Eugene is serving on its direct "
            f"ports. Its configuration is kept as {disabled}."
        )
    reason = reverted(settings)
    if reason:
        return f"Eugene went back to its direct ports: {reason}"
    return None


# --------------------------------------------------------------------------- #
# apply, confirm, revert, turn off
# --------------------------------------------------------------------------- #


def save_applied(settings: Any, text: str) -> datetime:
    """Write `text` as the configuration, on approval. Returns its deadline."""
    path = config_path(settings)
    marker = pending(settings)
    if marker is not None:
        # An unconfirmed configuration is never something to go back to.
        previous = marker.get("previous")
    else:
        previous = path.read_text(encoding="utf-8") if path.is_file() else None
    # The marker first: a crash between the two writes leaves the old file
    # on approval, and going back to it changes nothing.
    _write_marker(settings, _PENDING, {"appliedAt": _now().isoformat(), "previous": previous})
    with contextlib.suppress(FileNotFoundError):
        (state_dir(settings) / _REVERTED).unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_private(path, text)
    return _now() + timedelta(seconds=settings.entrypoint_confirm_seconds)


def revert(settings: Any, reason: str) -> None:
    """Put back what was there before the configuration on approval."""
    marker = pending(settings) or {}
    path = config_path(settings)
    if path.is_file():
        os.replace(path, path.with_name(path.name + REVERTED_SUFFIX))
    previous = marker.get("previous")
    if isinstance(previous, str):
        write_private(path, previous)
    _write_marker(settings, _REVERTED, {"reason": reason, "at": _now().isoformat()})
    with contextlib.suppress(FileNotFoundError):
        (state_dir(settings) / _PENDING).unlink()
    log.warning("HTTPS entry point went back to its previous configuration: %s", reason)


def confirm(app: Any) -> None:
    """An operator reached Eugene through the entry point: keep it."""
    if getattr(app.state, "entrypoint_confirm_by", None) is None:
        return
    app.state.entrypoint_confirm_by = None
    with contextlib.suppress(FileNotFoundError):
        (state_dir(app.state.settings) / _PENDING).unlink()
    log.info("HTTPS entry point confirmed: an operator signed in through it")


def turn_off(settings: Any) -> None:
    """Keep the file as `<file>.disabled`; the next start uses the direct ports."""
    path = config_path(settings)
    if path.is_file():
        os.replace(path, path.with_name(path.name + DISABLED_SUFFIX))
    for name in (_PENDING, _REVERTED):
        with contextlib.suppress(FileNotFoundError):
            (state_dir(settings) / name).unlink()


# --------------------------------------------------------------------------- #
# restarting in place
# --------------------------------------------------------------------------- #


def schedule_restart(app: Any, why: str, *, delay: float = 1.0) -> None:
    """Restart this agent in place, after the answer that asked for it is sent.

    The server exits the way a stop does, so every child -- the proxy, the
    components and the apps -- is shut down by the lifespan, and `_serve`
    then replaces this process with a new one (`reexec`). In a container the
    process keeps its pid and the container keeps running.
    """

    async def later() -> None:
        await asyncio.sleep(delay)
        log.warning("restarting this agent in place: %s", why)
        app.state.restart_requested = True
        server = getattr(app.state, "uvicorn_server", None)
        if server is not None:
            server.should_exit = True

    app.state.restart_task = asyncio.get_running_loop().create_task(later(), name="restart")


def reexec() -> NoReturn:
    """Replace this process with a fresh run of the same command."""
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    print("agent: restarting in place", flush=True)
    os.execv(sys.executable, [sys.executable, *argv[1:]])


async def watch_approval(app: Any) -> None:
    """Go back if no operator arrives through the new entry point in time."""
    settings = app.state.settings
    deadline = app.state.entrypoint_confirm_by
    while deadline is not None:
        remaining = (deadline - _now()).total_seconds()
        if remaining <= 0:
            break
        await asyncio.sleep(min(remaining, 5.0))
        deadline = app.state.entrypoint_confirm_by
    if deadline is None or pending(settings) is None:
        return
    entry = app.state.entrypoint_config
    minutes = max(1, round(settings.entrypoint_confirm_seconds / 60))
    revert(
        settings,
        f"nobody signed in to the console through {entry.console.origin} within "
        f"{minutes} minute{'s' if minutes != 1 else ''} of it starting. Check the proxy "
        "or DNS settings, then apply it again.",
    )
    schedule_restart(app, "the applied HTTPS setup was not confirmed", delay=0)
