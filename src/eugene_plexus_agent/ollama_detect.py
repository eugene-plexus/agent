"""Finding Ollama's model store on the Library's own machine (CX9-C2, C3).

Only what this machine can see: `OLLAMA_MODELS` when it is set (Ollama itself
uses nothing else then), else `~/.ollama/models` (Windows and macOS, and a
Linux user's own Ollama), else the Linux service's
`/usr/share/ollama/.ollama/models`. A store is a directory holding `blobs/`
and `manifests/` or `manifests-v2/`. Another machine's store is not looked
for: its agent would have to read it.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from .ollama_store import is_store

log = logging.getLogger(__name__)

LINUX_SERVICE_STORE = "/usr/share/ollama/.ollama/models"


def windows_profiles() -> tuple[str | None, list[tuple[str, Path, str | None]]]:
    """Machine override and real profiles, with loaded users' overrides."""
    import winreg

    def value(hive, key, name):
        try:
            with winreg.OpenKey(hive, key) as opened:
                return os.path.expandvars(str(winreg.QueryValueEx(opened, name)[0]))
        except OSError:
            return None

    machine = value(
        winreg.HKEY_LOCAL_MACHINE,
        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        "OLLAMA_MODELS",
    )
    profiles = []
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList"
        ) as listing:
            for index in range(winreg.QueryInfoKey(listing)[0]):
                sid = winreg.EnumKey(listing, index)
                path = value(listing, sid, "ProfileImagePath")
                if path and Path(path).is_dir():
                    profiles.append(
                        (
                            sid,
                            Path(path),
                            value(winreg.HKEY_USERS, sid + r"\Environment", "OLLAMA_MODELS"),
                        )
                    )
    except OSError as exc:
        log.warning("Cannot read Windows profiles: %s", exc)
    return machine, profiles


def candidates(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
    profiles: list[tuple[str, Path, str | None]] | None = None,
    machine_models: str | None = None,
) -> list[Path]:
    """Where Ollama's store would be, in the order Ollama itself decides."""
    env = os.environ if environ is None else environ
    chosen = (env.get("OLLAMA_MODELS") or "").strip()
    base = home if home is not None else Path.home()
    if (platform or sys.platform) == "win32" and str(base).replace("\\", "/").lower().endswith(
        "config/systemprofile"
    ):
        if profiles is None:
            machine_models, profiles = windows_profiles()
        found = [Path(machine_models)] if machine_models else []
        if chosen:
            found.insert(0, Path(chosen).expanduser())
        found.extend(
            Path(override) if override else path / ".ollama" / "models"
            for _, path, override in profiles
        )
        return list(dict.fromkeys(found))
    if chosen:
        return [Path(chosen).expanduser()]
    found = [base / ".ollama" / "models"]
    if (platform or sys.platform).startswith("linux"):
        found.append(Path(LINUX_SERVICE_STORE))
    return found


def find_store(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
) -> Path | None:
    """The first candidate that is a store, or None.

    Never raises: the Library starts, scans and answers `/v1/folders` through
    this, and a machine with no home folder (a container under an arbitrary uid)
    or a share that errors is a store not found, said once in the log.
    """
    try:
        for path in candidates(environ=environ, home=home, platform=platform):
            if is_store(path):
                return path
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("could not look for Ollama's model store: %s", exc)
    return None


def find_stores(**kwargs) -> list[Path]:
    """All detected stores, retaining a candidate whose read was denied."""
    # Keep the singular entry point for callers that inject detection in tests.
    first = find_store(**kwargs)
    found = [first] if first is not None else []
    try:
        for path in candidates(**kwargs):
            try:
                path.stat()
                detected = is_store(path)
            except PermissionError:
                detected = True
            except FileNotFoundError:
                detected = False
            if detected and path not in found:
                found.append(path)
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("Could not look for Ollama stores: %s", exc)
    return found
