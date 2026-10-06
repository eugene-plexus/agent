"""Optional apps: the spokes this agent installs and supervises.

Design: `specs/docs/design/apps-and-spokes.md`. The short version:

**An app is not a component.** A component is part of the hub -- it is
spawned with the install's verify key, a `service:<kind>` token and
sometimes the master key, and the gateway's front door accepts any
`service:*` token. Handing an app any of that would give *our* spokes a
door into the hub that nobody else's can use. So an app gets what a
third-party client of this install could be given and nothing more: a
client key, an address for the gateway, a port and a data directory.
The rule is structural here rather than a promise -- `_AppPlanner`
builds its environment from `child_environment()` with no component
prefix, which strips every `EUGENE_PLEXUS_*` variable, and then adds
only `EUGENE_PLEXUS_APP_*` -- and, for an app that is not ours (C4), the
variables of its own its manifest names, none of which may begin
`EUGENE_PLEXUS_` (`validate_manifest`).

**Each app runs from its own Python environment**, built by `uv` into
`<config dir>/apps/<id>/versions/<version>/venv`. A trainer brings
torch and the Discord connector brings discord.py, and neither belongs
in the process holding this node's keys; on Windows the agent
cannot upgrade packages in its own venv while it runs anyway, because
its console-script `.exe` is locked. `watchdog-venv-is-runtime` is a
rule about components and stays true of them.

**The version directory is committed by its `install.json`, not by a
rename.** Virtual environments are not reliably relocatable, so the
environment is built where it will run and becomes installed the moment
its metadata is written -- the same "a directory without `install.json`
is invisible" rule the engine store uses, and a crash part-way leaves a
directory the next install of that version clears.

**`apps.yaml` is its own file**, beside `agent.yaml`. An entry this
agent cannot read degrades the apps alone -- the file is kept as
`apps.yaml.unreadable` and `/healthz` says why -- and cannot take the
topology, the runtimes or the passphrase with it, which is what one
unrecognised component kind in `agent.yaml` does.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.resources
import json
import logging
import re
import secrets
import shutil
import sys
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import AnyUrl, ValidationError

from . import app_accounts, app_launcher, ports
from ._generated.models import (
    App,
    AppCatalogue,
    AppCatalogueEntry,
    AppHubSurface,
    AppInstall,
    AppIsolation,
    AppManifest,
    AppOrigin,
    ComponentStatus,
    State1,
)
from ._http import internal_client
from ._private_files import write_private
from .child_env import child_environment
from .supervisor import (
    _CRASH_BACKOFF_THRESHOLD,
    ProcessState,
    SpawnPlan,
    SpawnPlanError,
    SupervisedProcess,
)

log = logging.getLogger(__name__)

APPS_FILE = "apps.yaml"
APPS_DIR = "apps"
UNREADABLE_SUFFIX = ".unreadable"
CATALOGUE_RESOURCE = "apps_catalogue.yaml"
INSTALL_METADATA = "install.json"
KEY_FILE = "client_key"
#: The app's secret as a client that signs people in with Eugene (C2). Always
#: present, empty for an app that does not: the Linux unit loads it as a
#: credential, and systemd refuses to start a unit whose credential is missing.
OIDC_SECRET_FILE = "oidc_secret"

#: The installed version and one to roll back to, as for engines.
RETAINED_VERSIONS = 2

#: Apps bind from their own range, above the runtimes' 8090-8189, so an
#: app's port is never one a later runtime or companion is handed.
APP_PORT_BASE = 8190
APP_PORT_SPAN = 100

ENV_PREFIX = "EUGENE_PLEXUS_APP"
#: Variables of ours, which a manifest's `environment` may not name.
OUR_PREFIX = "EUGENE_PLEXUS_"

#: `source: pypi` installs `<package>==<version>` from the Python Package
#: Index (C4): an app published there by its own project.
PYPI = "pypi"
#: A PEP 440 version that names one release, with no local segment (the
#: index refuses those): `0.11.4`, `1.0rc1`, `2.0.post1`. Not a range, a
#: name like `latest`, or a commit. (An epoch's `!` is outside the schema's
#: version pattern, so it cannot arrive.)
_EXACT_RELEASE = re.compile(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?")

_HEALTH_POLL_SECONDS = 1.5
_OUTPUT_TAIL_LINES = 40


# --------------------------------------------------------------------------- #
# where things are
# --------------------------------------------------------------------------- #


def venv_python(venv: Path) -> Path:
    """The interpreter inside an environment `uv venv` built."""
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def find_uv(configured: str | None = None) -> Path | None:
    """The `uv` this agent installs apps with, or None.

    In order: the operator's `uvBinary`, the one the installer left at
    `$PREFIX/bin/uv` beside `$PREFIX/venv` (so derived from this
    interpreter's own `sys.prefix`), then PATH. The installer's copy is
    the one that matters: every supported install has it, and nothing
    else on the box is guaranteed to.
    """
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    name = "uv.exe" if sys.platform == "win32" else "uv"
    beside = Path(sys.prefix).parent / "bin" / name
    if beside.is_file():
        return beside
    on_path = shutil.which("uv")
    return Path(on_path) if on_path else None


def pip_requirement(manifest: AppManifest) -> str:
    """What `uv pip install` is given: `<package> @ <url>`, or for `source:
    pypi`, `<package>==<version>`.

    A directory is turned into a `file://` URL because PEP 508 direct
    references are URLs; a relative path would resolve against whatever
    directory the agent happened to start in, so it is refused rather
    than guessed at. A `pypi` version that is not one exact release is
    refused for the same reason: what installs must be what the entry
    names, not whatever the index has today.
    """
    source = manifest.source.strip()
    if source == PYPI:
        if not _EXACT_RELEASE.fullmatch(manifest.version):
            raise ValueError(
                f"version {manifest.version!r} is not an exact release. A pypi entry installs "
                f"{manifest.package}==<version>, so its version must be one release, such as "
                "0.11.4."
            )
        return f"{manifest.package}=={manifest.version}"
    if source.startswith(("https://", "http://", "file://")):
        return f"{manifest.package} @ {source}"
    path = Path(source).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"source {source!r} is neither an https:// archive URL nor an absolute path"
        )
    return f"{manifest.package} @ {path.as_uri()}"


def normalized(manifest: AppManifest) -> AppManifest:
    """The manifest with `uses` as enum members.

    The generated default is the plain string list the schema declares,
    so a manifest that omitted `uses` carries `["inference"]` as `str`
    and every dump of it warns. One place turns it into what the field's
    type says, for every way a manifest arrives.
    """
    uses = [AppHubSurface(u) for u in (manifest.uses or [])]
    return manifest.model_copy(update={"uses": uses})


#: The entries Eugene's own file support may hold as app `node-files`: the
#: site host (Job Sites J6), and the helper it replaces until the next
#: reconcile installs the host over it. Neither is an app the owner manages.
NODE_FILES_ENTRIES = frozenset({"eugene_plexus_site_host", "eugene_plexus_node_helper"})


def is_node_files(app_id: str, entry: str) -> bool:
    return app_id == "node-files" and entry in NODE_FILES_ENTRIES


#: The schema's `environment` limits, which the generated model does not keep.
_MAX_ENVIRONMENT = 64
_MAX_ENVIRONMENT_VALUE = 2048


def validate_manifest(manifest: AppManifest) -> None:
    """Refuse what the schema lets through and a start could not honour,
    naming it. Run on every shipped entry and every custom one (C4).

    * a `pypi` entry whose version is not one exact release;
    * a placeholder the launcher does not fill in, which would otherwise
      surface as a crash at the app's first start rather than here;
    * a secret placeholder in `args`: an argument is in the process list,
      which every account on the machine can read, and in the agent's log
      line for the spawn, so secrets travel in `environment` only;
    * a variable of ours (`EUGENE_PLEXUS_*`, in either case, since Windows
      does not tell `eugene_plexus_x` from `EUGENE_PLEXUS_X`) as one of the
      app's own, which could point it at another app's key or override
      what the agent hands it.
    """
    if manifest.source.strip() == PYPI:
        pip_requirement(manifest)
    args = [a.root for a in manifest.args or []]
    environment = dict(manifest.environment or {})
    if len(environment) > _MAX_ENVIRONMENT:
        raise ValueError(f"environment has {len(environment)} variables; at most 64 are allowed")
    for text in args:
        for name in app_launcher.placeholders(text):
            if name in app_launcher.SECRET_PLACEHOLDERS:
                raise ValueError(
                    f"the argument {text!r} names {{{name}}}, a secret. Arguments can be read by "
                    "every account on this machine, so a secret goes in environment instead."
                )
    for name, text in [*(("args", a) for a in args), *environment.items()]:
        if len(text) > _MAX_ENVIRONMENT_VALUE:
            raise ValueError(f"{name} is longer than {_MAX_ENVIRONMENT_VALUE} characters")
        for placeholder in app_launcher.placeholders(text):
            if placeholder not in app_launcher.PLACEHOLDERS:
                known = ", ".join(f"{{{p}}}" for p in sorted(app_launcher.PLACEHOLDERS))
                raise ValueError(
                    f"{name} names {{{placeholder}}}, which is not a placeholder. The "
                    f"placeholders are {known}."
                )
    for name in [*environment, manifest.resetOnConnectionChange or ""]:
        if name.upper().startswith(OUR_PREFIX):
            raise ValueError(
                f"{name} begins {OUR_PREFIX}, which names Eugene Plexus's own variables; an "
                "app's own variables must be named otherwise."
            )
    reset = manifest.resetOnConnectionChange
    if reset and reset in environment:
        raise ValueError(
            f"{reset} is both resetOnConnectionChange and in environment; the agent sets it, "
            "so it cannot be given a value of its own."
        )


def load_catalogue(text: str | None = None) -> list[AppManifest]:
    """The apps this agent release ships, from its own package data.

    Never raises. A catalogue that will not parse is a packaging bug, and
    the answer to it is an empty catalogue and a log line -- not an agent
    that will not start. One entry that will not validate costs that
    entry, named in the log, and not the others.
    """
    try:
        if text is None:
            text = (
                importlib.resources.files("eugene_plexus_agent")
                .joinpath(CATALOGUE_RESOURCE)
                .read_text(encoding="utf-8")
            )
        raw = yaml.safe_load(text) or []
        if not isinstance(raw, list):
            raise ValueError("the catalogue must be a YAML list")
    except (OSError, ValueError) as exc:
        log.error("the shipped app catalogue could not be read (%s); offering none", exc)
        return []
    out: list[AppManifest] = []
    for item in raw:
        try:
            manifest = normalized(AppManifest.model_validate(item))
            validate_manifest(manifest)
        except (ValueError, ValidationError) as exc:
            name = item.get("id") if isinstance(item, dict) else None
            log.error("the shipped app catalogue entry %r is unusable (%s); not offered", name, exc)
            continue
        out.append(manifest)
    return out


# --------------------------------------------------------------------------- #
# what is installed
# --------------------------------------------------------------------------- #


@dataclass
class InstalledApp:
    """One installed app, as `apps.yaml` records it."""

    manifest: AppManifest
    origin: AppOrigin
    port: int
    installed_at: datetime
    enabled: bool = True
    previous_version: str | None = None

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def version(self) -> str:
        return self.manifest.version

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "manifest": normalized(self.manifest).model_dump(mode="json", exclude_none=True),
            "origin": self.origin.value,
            "port": self.port,
            "installedAt": self.installed_at.isoformat(),
            "enabled": self.enabled,
        }
        if self.previous_version:
            out["previousVersion"] = self.previous_version
        return out

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> InstalledApp:
        return cls(
            manifest=normalized(AppManifest.model_validate(raw["manifest"])),
            origin=AppOrigin(raw["origin"]),
            port=int(raw["port"]),
            installed_at=datetime.fromisoformat(raw["installedAt"]),
            enabled=bool(raw.get("enabled", True)),
            previous_version=raw.get("previousVersion"),
        )


@dataclass
class AppKey:
    """The client key minted for an app. Kept apart from the install
    record because it outlives a failed install: minting happens with the
    operator's credential at request time, and a retry must reuse the key
    rather than mint a second one nobody will ever revoke."""

    key_id: str
    key_name: str


class AppStore:
    """`apps.yaml`: installed apps, their keys, and this node's custom entries.

    One lock, one atomic owner-only write, the shape `AgentState` has.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._installed: dict[str, InstalledApp] = {}
        self._keys: dict[str, AppKey] = {}
        self._custom: dict[str, AppManifest] = {}
        self._oidc: dict[str, str] = {}
        self._degraded_reason: str | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def root(self) -> Path:
        """`<config dir>/apps`, where every app's directory lives."""
        return self._path.parent / APPS_DIR

    @property
    def degraded_reason(self) -> str | None:
        return self._degraded_reason

    # --- lifecycle -----------------------------------------------------

    def load(self) -> None:
        if not self._path.exists():
            return
        raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{self._path} must be a YAML mapping at the root")
        installed = {}
        for item in raw.get("installed") or []:
            record = InstalledApp.from_json(item)
            installed[record.id] = record
        keys = {
            app_id: AppKey(key_id=str(v["keyId"]), key_name=str(v["keyName"]))
            for app_id, v in (raw.get("keys") or {}).items()
        }
        custom = {}
        for item in raw.get("custom") or []:
            manifest = normalized(AppManifest.model_validate(item))
            custom[manifest.id] = manifest
        self._installed, self._keys, self._custom = installed, keys, custom
        self._oidc = {str(k): str(v) for k, v in (raw.get("oidc") or {}).items()}

    def load_or_degrade(self) -> str | None:
        """Load, or come up with no apps and say why -- never raise.

        The file is copied beside itself first, so the first write after
        a degraded boot cannot be the one that loses what was installed.
        """
        try:
            self.load()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self._installed, self._keys, self._custom = {}, {}, {}
            self._oidc = {}
            self._degraded_reason = reason
            target = self._path.with_suffix(self._path.suffix + UNREADABLE_SUFFIX)
            with contextlib.suppress(OSError):
                write_private(target, self._path.read_bytes())
            log.error(
                "%s could not be read (%s); running with no apps. A copy is at %s.",
                self._path,
                reason,
                target,
            )
            return reason
        self._degraded_reason = None
        return None

    def _write(self) -> None:
        out = {
            "installed": [r.to_json() for r in self._installed.values()],
            "keys": {
                app_id: {"keyId": k.key_id, "keyName": k.key_name}
                for app_id, k in self._keys.items()
            },
            "custom": [m.model_dump(mode="json", exclude_none=True) for m in self._custom.values()],
            "oidc": dict(self._oidc),
        }
        write_private(self._path, yaml.safe_dump(out, sort_keys=True, default_flow_style=False))
        # A degraded boot is repaired by the first successful write.
        self._degraded_reason = None

    # --- installed ------------------------------------------------------

    def installed(self) -> list[InstalledApp]:
        return list(self._installed.values())

    def get(self, app_id: str) -> InstalledApp | None:
        return self._installed.get(app_id)

    def put(self, record: InstalledApp) -> None:
        self._installed[record.id] = record
        self._write()

    def remove(self, app_id: str) -> None:
        self._installed.pop(app_id, None)
        self._keys.pop(app_id, None)
        self._oidc.pop(app_id, None)
        self._write()

    def taken_ports(self) -> set[int]:
        return {r.port for r in self._installed.values()}

    # --- keys -----------------------------------------------------------

    def key(self, app_id: str) -> AppKey | None:
        return self._keys.get(app_id)

    def put_key(self, app_id: str, key: AppKey) -> None:
        self._keys[app_id] = key
        self._write()

    def drop_key(self, app_id: str) -> None:
        if self._keys.pop(app_id, None) is not None:
            self._write()

    # --- signing in with Eugene (C2) ------------------------------------

    def oidc_client(self, app_id: str) -> str | None:
        """The client id this app signs people in with, when it does."""
        return self._oidc.get(app_id)

    def put_oidc_client(self, app_id: str, client_id: str) -> None:
        self._oidc[app_id] = client_id
        self._write()

    def drop_oidc_client(self, app_id: str) -> None:
        if self._oidc.pop(app_id, None) is not None:
            self._write()

    # --- custom entries -------------------------------------------------

    def custom(self) -> list[AppManifest]:
        return list(self._custom.values())

    def add_custom(self, manifest: AppManifest) -> None:
        self._custom[manifest.id] = normalized(manifest)
        self._write()

    def remove_custom(self, app_id: str) -> None:
        self._custom.pop(app_id, None)
        self._write()

    # --- directories ----------------------------------------------------

    def app_dir(self, app_id: str) -> Path:
        return self.root / app_id

    def data_dir(self, app_id: str) -> Path:
        return self.app_dir(app_id) / "data"

    def version_dir(self, app_id: str, version: str) -> Path:
        return self.app_dir(app_id) / "versions" / version

    def key_file(self, app_id: str) -> Path:
        return self.data_dir(app_id) / KEY_FILE

    def oidc_secret_file(self, app_id: str) -> Path:
        return self.data_dir(app_id) / OIDC_SECRET_FILE

    def sign_in_stamp_file(self, app_id: str) -> Path:
        """The return addresses last registered for this app's sign-in."""
        return self.oidc_secret_file(app_id).with_suffix(".redirects.json")


def _url(value: str | None) -> AnyUrl | None:
    return AnyUrl(value) if value else None


def _remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        shutil.rmtree(path)


def _fresh_dir(path: Path) -> None:
    _remove_quietly(path)
    path.mkdir(parents=True, exist_ok=True)


def installed_versions(store: AppStore, app_id: str) -> list[str]:
    """Versions with an `install.json`, newest first by install time."""
    root = store.app_dir(app_id) / "versions"
    found: list[tuple[str, str]] = []
    if not root.is_dir():
        return []
    for child in root.iterdir():
        meta = child / INSTALL_METADATA
        if not meta.is_file():
            continue
        with contextlib.suppress(OSError, ValueError):
            data = json.loads(meta.read_text(encoding="utf-8"))
            found.append((str(data.get("installedAt", "")), child.name))
    return [name for _, name in sorted(found, reverse=True)]


def prune_versions(store: AppStore, app_id: str, keep: set[str]) -> None:
    """Remove every version directory not in `keep`, committed or not.

    Never raises: pruning is housekeeping after a successful install, and
    a locked file on Windows must not turn a good install into a failure.
    """
    root = store.app_dir(app_id) / "versions"
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.name not in keep:
            _remove_quietly(child)


# --------------------------------------------------------------------------- #
# installing
# --------------------------------------------------------------------------- #


class AppInstallError(Exception):
    """An install that cannot proceed, in words an operator can act on."""


@dataclass
class _Progress:
    app: str
    version: str
    state: State1 = State1.resolving
    message: str | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def snapshot(self) -> AppInstall:
        return AppInstall(
            app=self.app,
            version=self.version,
            state=self.state,
            message=self.message,
            error=self.error,
            startedAt=self.started_at,
            finishedAt=self.finished_at,
        )


async def _run_tool(argv: list[str], *, env: dict[str, str]) -> tuple[int, str]:
    """Run a command, return (code, output tail). Cancelling kills it.

    The tail is what a failed install reports: `uv`'s own words about a
    package that would not resolve are the only useful message there is.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    tail: deque[str] = deque(maxlen=_OUTPUT_TAIL_LINES)
    try:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            tail.append(line.decode("utf-8", errors="replace").rstrip())
        code = await proc.wait()
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(BaseException):
            await proc.wait()
        raise
    return code, "\n".join(line for line in tail if line)


# Finds the entry point without running it. A package whose `-m` target
# is a package also needs a `__main__`, which is the case `find_spec` on
# the package alone would wave through and the first spawn would not. A
# `module:attribute` entry (C4) is imported, because whether the attribute
# exists and can be called is only known once its module has run.
_VERIFY_SNIPPET = (
    "import importlib, importlib.util as u, sys\n"
    "name, _, attr = sys.argv[1].partition(':')\n"
    "if attr:\n"
    "    target = getattr(importlib.import_module(name), attr, None)\n"
    "    if target is None:\n"
    "        sys.exit(name + ' has no ' + attr + ' to call')\n"
    "    if not callable(target):\n"
    "        sys.exit(name + '.' + attr + ' cannot be called, so it cannot start the app')\n"
    "    sys.exit(0)\n"
    "spec = u.find_spec(name)\n"
    "if spec is None:\n"
    "    sys.exit('no module named ' + name)\n"
    "if spec.submodule_search_locations is not None and u.find_spec(name + '.__main__') is None:\n"
    "    sys.exit(name + ' is a package with no __main__, so python -m cannot run it')\n"
)

# A `module:attribute` entry, called as the console script pip writes for it
# calls it: `sys.argv` is the script's name and its arguments, and what it
# returns is the exit code. The directory it runs in is taken off the path,
# as a script's would be, so an app's data cannot shadow its own modules.
# One line, because the spawn's argv is what the agent's log prints.
_CALL_SNIPPET = (
    "import importlib,sys;sys.path[:]=[p for p in sys.path if p];"
    "m,_,a=sys.argv[1].partition(':');f=getattr(importlib.import_module(m),a);"
    "sys.argv=sys.argv[2:];sys.exit(f())"
)


def entry_argv(python: Path, manifest: AppManifest) -> list[str]:
    """How the app is started, before its `args`: `python -m <module>`, or
    for `module:attribute`, a call to it named after its package."""
    if ":" in manifest.entry:
        return [str(python), "-c", _CALL_SNIPPET, manifest.entry, manifest.package]
    return [str(python), "-m", manifest.entry]


class AppInstaller:
    """At most one install per app, in the background.

    Same shape as `EngineInstaller`: a request starts it and returns, the
    GET reports named phases, and the terminal state is kept until the
    next install of that app starts.
    """

    def __init__(self) -> None:
        self._progress: dict[str, _Progress] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def running(self, app_id: str) -> bool:
        task = self._tasks.get(app_id)
        return task is not None and not task.done()

    def snapshot(self, app_id: str) -> AppInstall | None:
        progress = self._progress.get(app_id)
        return progress.snapshot() if progress is not None else None

    def start(
        self,
        manifest: AppManifest,
        *,
        store: AppStore,
        uv: Path,
        on_installed: Callable[[AppManifest], Awaitable[None]],
    ) -> AppInstall:
        if self.running(manifest.id):
            raise AppInstallError(f"an install of {manifest.id!r} is already running")
        progress = _Progress(app=manifest.id, version=manifest.version)
        self._progress[manifest.id] = progress
        self._tasks[manifest.id] = asyncio.create_task(
            self._run(manifest, progress, store=store, uv=uv, on_installed=on_installed),
            name=f"app-install-{manifest.id}",
        )
        return progress.snapshot()

    async def cancel(self, app_id: str) -> AppInstall | None:
        task = self._tasks.get(app_id)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        return self.snapshot(app_id)

    async def wait(self, app_id: str) -> AppInstall | None:
        """Until the install of `app_id` ends, then how it ended."""
        task = self._tasks.get(app_id)
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(task)
        return self.snapshot(app_id)

    async def aclose(self) -> None:
        for app_id in list(self._tasks):
            await self.cancel(app_id)

    async def _run(
        self,
        manifest: AppManifest,
        progress: _Progress,
        *,
        store: AppStore,
        uv: Path,
        on_installed: Callable[[AppManifest], Awaitable[None]],
    ) -> None:
        target = store.version_dir(manifest.id, manifest.version)
        # Once `install.json` is written the directory is an installed
        # version, and nothing that fails afterwards -- recording it,
        # restarting onto it -- may delete it out from under the record.
        committed = False
        try:
            await self._install(manifest, progress, target=target, uv=uv)
            committed = True
            await on_installed(manifest)
            progress.state = State1.done
            progress.message = f"installed {manifest.version}"
        except asyncio.CancelledError:
            progress.state = State1.cancelled
            progress.message = "cancelled"
            if not committed:
                _remove_quietly(target)
            raise
        except Exception as exc:
            log.warning("install of app %r failed: %s", manifest.id, exc)
            progress.state = State1.failed
            progress.error = str(exc)
            progress.message = "failed"
            if not committed:
                _remove_quietly(target)
        finally:
            progress.finished_at = datetime.now(UTC)

    async def _install(
        self, manifest: AppManifest, progress: _Progress, *, target: Path, uv: Path
    ) -> None:
        requirement = pip_requirement(manifest)
        # A directory left by a crash part-way through an earlier install of
        # this version has no `install.json` and is nobody's; clear it.
        await asyncio.to_thread(_fresh_dir, target)
        venv = target / "venv"
        # The tools get this host's environment with none of the hub's own
        # variables, the same boundary a foreign binary gets -- and keep
        # the user's proxy, because a package index is egress.
        env = child_environment()
        # C1: app interpreters live apart from the agent's, so an app's own
        # account is granted those and never the agent's; and copied, not
        # linked from uv's cache, so a grant on a venv's files is a grant on
        # those files and nothing that shares their inode.
        app_root = target.parent.parent.parent.resolve()
        env["UV_PYTHON_INSTALL_DIR"] = str(app_root / app_accounts.APP_PYTHONS)
        # NAS containers may have no passwd entry or writable home. Keep
        # installation caches on the same writable volume as app environments,
        # rather than uv's home-derived default (/.cache/uv on Unraid).
        env["UV_CACHE_DIR"] = str(app_root / ".cache" / "uv")
        env["UV_LINK_MODE"] = "copy"

        progress.state = State1.creating
        progress.message = f"building a Python {manifest.python} environment"
        code, out = await _run_tool(
            [
                str(uv),
                "venv",
                "--python",
                str(manifest.python or "3.12"),
                "--python-preference",
                "only-managed",
                str(venv),
            ],
            env=env,
        )
        if code != 0:
            raise AppInstallError(f"building the environment failed:\n{out}")

        python = venv_python(venv)
        progress.state = State1.installing
        progress.message = f"installing {manifest.package}"
        code, out = await _run_tool(
            [str(uv), "pip", "install", "--python", str(python), requirement], env=env
        )
        if code != 0:
            raise AppInstallError(f"installing {manifest.package} failed:\n{out}")

        progress.state = State1.verifying
        progress.message = f"checking that {manifest.entry} can start"
        code, out = await _run_tool([str(python), "-c", _VERIFY_SNIPPET, manifest.entry], env=env)
        if code != 0:
            raise AppInstallError(
                f"{manifest.package} installed, but its entry point cannot be run: {out}"
            )

        # The commit point. Until this file exists the directory is not an
        # installed version, to this module or anyone else.
        write_private(
            target / INSTALL_METADATA,
            json.dumps(
                {
                    "version": manifest.version,
                    "package": manifest.package,
                    "python": manifest.python,
                    "installedAt": datetime.now(UTC).isoformat(),
                },
                indent=2,
            ),
        )


# --------------------------------------------------------------------------- #
# supervising
# --------------------------------------------------------------------------- #


class _AppPlanner:
    """Launch plans for one app.

    Everything an app is handed is here, and it is a short list on
    purpose -- see the module docstring. No safe mode: that is a
    component's contract with the hub (boot from defaults so `/v1/config`
    stays reachable), and an app owes the hub nothing of the kind. Five
    crashes in a row is `crashed`, and a restart clears it.
    """

    def __init__(
        self,
        record: InstalledApp,
        *,
        store: AppStore,
        gateway_url: Callable[[], str | None],
        bind_host: Callable[[], str | None],
        oidc_issuer: Callable[[], str | None] = lambda: None,
        app_url: Callable[[], str | None] = lambda: None,
        public_origin: str | None = None,
        oidc_backchannel: str | None = None,
    ) -> None:
        self.record = record
        self._store = store
        self._gateway_url = gateway_url
        self._bind_host = bind_host
        self._oidc_issuer = oidc_issuer
        self._app_url = app_url
        self._public_origin = public_origin
        self._oidc_backchannel = oidc_backchannel
        self.admin_token: str | None = None

    @property
    def name(self) -> str:
        return f"app:{self.record.id}"

    @property
    def log_prefix(self) -> str:
        return f"[app: {self.record.id}] "

    def reset(self) -> None:
        return None

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        return None

    def on_crash_threshold(self) -> bool:
        log.error(
            "app %s crashed %d times in a row; giving up. Restart it from its page "
            "after reading its log.",
            self.record.id,
            _CRASH_BACKOFF_THRESHOLD,
        )
        return False

    @property
    def health_url(self) -> str:
        """Where either supervisor asks whether the app is serving: its
        `healthPath` on its own port, on loopback."""
        return f"http://127.0.0.1:{self.record.port}{self.record.manifest.healthPath or '/healthz'}"

    def plan(self) -> SpawnPlan:
        """A start by the agent's own supervisor: `base_plan` with the
        manifest's `args` and `environment` filled in by the launcher's own
        `render_start`, so this path and the app's own account fill a
        placeholder one way. Secrets reach the child's environment and
        nothing else: never an argument (`validate_manifest`), never the log.
        """
        base = self.base_plan()
        record = self.record
        spec = {
            "argv": base.argv,
            **self.launch(),
            "dataDir": str(self._store.data_dir(record.id)),
            "keyFile": str(self._store.key_file(record.id)),
            "oidcSecretFile": str(self._store.oidc_secret_file(record.id)),
        }
        try:
            argv, env, notes = app_launcher.render_start(spec, base.env)
        except (OSError, ValueError) as exc:
            raise SpawnPlanError(f"could not fill in how {record.id} starts: {exc}") from exc
        for note in notes:
            log.warning("%s%s", self.log_prefix, note)
        return SpawnPlan(argv=argv, env=env, cwd=base.cwd, port=base.port)

    def launch(self) -> dict[str, Any]:
        """The manifest's own start, not filled in: its `args` and
        `environment` as written, the values that are not secret, and the
        variable a changed connection sets. What the spec the agent writes
        for an app's own account carries, so it holds no secret: the
        launcher reads those from files, inside that account (C4)."""
        manifest = self.record.manifest
        values: dict[str, str] = {
            "bindHost": self._bind_host() or "127.0.0.1",
            "port": str(self.record.port),
        }
        gateway = self._gateway_url()
        if gateway:
            values["gatewayUrl"] = gateway
        app_url = self._app_url()
        if app_url:
            values["appUrl"] = app_url
        client_id = self._store.oidc_client(self.record.id)
        issuer = self._oidc_issuer()
        if manifest.signIn and client_id and issuer:
            values["oidcIssuer"] = issuer
            values["oidcClientId"] = client_id
        return {
            "args": [a.root for a in manifest.args or []],
            "environment": dict(manifest.environment or {}),
            "values": values,
            "resetOnConnectionChange": manifest.resetOnConnectionChange,
        }

    def base_plan(self) -> SpawnPlan:
        """What every start of this app has, before its manifest's `args`
        and `environment`: the entry point and the `EUGENE_PLEXUS_APP_*`
        variables. An app's own account is handed this and `launch()`."""
        record = self.record
        python = venv_python(self._store.version_dir(record.id, record.version) / "venv")
        if not python.is_file():
            raise SpawnPlanError(
                f"the environment for {record.id} {record.version} is missing ({python}); "
                "install the app again"
            )
        data = self._store.data_dir(record.id)
        data.mkdir(parents=True, exist_ok=True)
        # A fresh admin token per spawn: the only holder is this agent, and
        # the app learns it from its environment, so nothing has to persist.
        self.admin_token = secrets.token_urlsafe(32)
        env = child_environment()
        env[f"{ENV_PREFIX}_ID"] = record.id
        env[f"{ENV_PREFIX}_BIND_PORT"] = str(record.port)
        env[f"{ENV_PREFIX}_DATA_DIR"] = str(data)
        env[f"{ENV_PREFIX}_KEY_FILE"] = str(self._store.key_file(record.id))
        env[f"{ENV_PREFIX}_ADMIN_TOKEN"] = self.admin_token
        if self._public_origin:
            env[f"{ENV_PREFIX}_PUBLIC_ORIGIN"] = self._public_origin
        if self._oidc_backchannel:
            env[f"{ENV_PREFIX}_OIDC_BACKCHANNEL"] = self._oidc_backchannel
        host = self._bind_host()
        if host:
            env[f"{ENV_PREFIX}_BIND_HOST"] = host
        gateway = self._gateway_url()
        if gateway:
            env[f"{ENV_PREFIX}_GATEWAY_URL"] = gateway
        # C2: an app that signs people in with Eugene is told where, as
        # which client, and where its secret is.
        client_id = self._store.oidc_client(record.id)
        issuer = self._oidc_issuer()
        if record.manifest.signIn and client_id and issuer:
            env[f"{ENV_PREFIX}_OIDC_ISSUER"] = issuer
            env[f"{ENV_PREFIX}_OIDC_CLIENT_ID"] = client_id
            env[f"{ENV_PREFIX}_OIDC_SECRET_FILE"] = str(self._store.oidc_secret_file(record.id))
        env["PYTHONUNBUFFERED"] = "1"
        return SpawnPlan(
            argv=entry_argv(python, record.manifest),
            env=env,
            cwd=str(data),
            port=record.port,
        )


_STATUS_BY_STATE: dict[ProcessState, ComponentStatus] = {
    ProcessState.starting: ComponentStatus.starting,
    ProcessState.exited: ComponentStatus.exited,
    ProcessState.crashed: ComponentStatus.crashed,
    ProcessState.not_spawnable: ComponentStatus.crashed,
}


class AppSupervisor:
    """The processes of the installed apps, and whether each answers.

    The spawn/watch/back-off loop is `SupervisedProcess`, shared with
    components and runtimes. What is separate is the planner and this
    health poll, which reads `GET <healthPath>` (`/healthz` unless the
    manifest says otherwise) on the app's own port -- an app that never
    answers it reads `starting` for as long as it runs, which is what the
    manifest's contract says it owes.
    """

    def __init__(self) -> None:
        self._processes: dict[str, SupervisedProcess] = {}
        self._planners: dict[str, _AppPlanner] = {}
        self._reachable: dict[str, bool] = {}
        self._health: dict[str, str] = {}
        self._health_task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None

    def start(self, planner: _AppPlanner) -> None:
        app_id = planner.record.id
        if app_id in self._processes:
            return
        process = SupervisedProcess(planner, log)
        self._processes[app_id] = process
        self._planners[app_id] = planner
        self._health[app_id] = planner.health_url
        self._reachable[app_id] = False
        process.start()
        if self._health_task is None:
            self._client = internal_client(timeout=2.0)
            self._health_task = asyncio.create_task(self._health_loop(), name="app-health")

    async def stop(self, app_id: str) -> None:
        process = self._processes.pop(app_id, None)
        self._planners.pop(app_id, None)
        self._health.pop(app_id, None)
        self._reachable.pop(app_id, None)
        if process is not None:
            await process.stop()

    def is_running(self, app_id: str) -> bool:
        return app_id in self._processes

    def admin_token(self, app_id: str) -> str | None:
        planner = self._planners.get(app_id)
        return planner.admin_token if planner is not None else None

    def status(
        self, app_id: str
    ) -> tuple[ComponentStatus, str | None, datetime | None, int | None, str | None]:
        """(status, lastError, lastRestart, pid, bind host) for one app."""
        process = self._processes.get(app_id)
        if process is None:
            return ComponentStatus.exited, None, None, None, None
        base = _STATUS_BY_STATE[process.state]
        if base == ComponentStatus.starting and self._reachable.get(app_id, False):
            base = ComponentStatus.running
        return base, process.last_error, process.last_restart, process.pid, process.last_bind_host

    def bind_host(self, app_id: str) -> str | None:
        process = self._processes.get(app_id)
        return process.last_bind_host if process is not None else None

    async def stop_all(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._health_task
            self._health_task = None
        await asyncio.gather(*(p.stop() for p in self._processes.values()), return_exceptions=True)
        self._processes.clear()
        self._planners.clear()
        self._reachable.clear()
        self._health.clear()
        if self._client is not None:
            with contextlib.suppress(BaseException):
                await self._client.aclose()
            self._client = None

    async def _health_loop(self) -> None:
        try:
            while True:
                await asyncio.gather(
                    *(self._poll(app_id, url) for app_id, url in list(self._health.items())),
                    return_exceptions=True,
                )
                await asyncio.sleep(_HEALTH_POLL_SECONDS)
        except asyncio.CancelledError:
            return

    async def _poll(self, app_id: str, url: str) -> None:
        client = self._client
        if client is None:
            return
        try:
            response = await client.get(url)
            ready = response.is_success
        except httpx.HTTPError:
            ready = False
        # Only for an app still supervised: a poll that was in flight when
        # the app was stopped must not bring its entry back.
        if app_id not in self._reachable:
            return
        self._reachable[app_id] = ready
        process = self._processes.get(app_id)
        if process is not None:
            process.observe_readiness(ready)


# --------------------------------------------------------------------------- #
# the manager the routes talk to
# --------------------------------------------------------------------------- #

GatewayResolver = Callable[[], Awaitable[tuple[str | None, str | None]]]
"""Returns (gateway URL for an app on this node, or None; why not, or None)."""


class AppManager:
    """Everything `/v1/apps` does, in one object on `app.state.apps`.

    Holds no credential of its own. Minting and revoking an app's key is
    done by the route, with the operator credential of the request that
    asked for it -- the same rule `install_proxy` and the key routes keep.
    """

    def __init__(
        self,
        *,
        store: AppStore,
        catalogue: list[AppManifest],
        get_config: Callable[[str], Any],
        bind_host: Callable[[], str | None],
        advertise_host: Callable[[], str | None],
        node_name: Callable[[], str | None],
        resolve_gateway: GatewayResolver,
        ingress_url: Callable[[], str] | None = None,
        accounts: app_accounts.AccountSupport | None = None,
        oidc_issuer: Callable[[], str | None] | None = None,
        public_origin: Callable[[str], str | None] | None = None,
        oidc_backchannel: str | None = None,
        private_apps: bool = False,
    ) -> None:
        self.store = store
        self._oidc_issuer = oidc_issuer or (lambda: None)
        self._public_origin = public_origin or (lambda app_id: None)
        self._oidc_backchannel = oidc_backchannel
        self._private_apps = private_apps
        self.before_stop: Callable[[str], Awaitable[None]] | None = None
        self.after_start: Callable[[str], None] | None = None
        self._reserved_ports: dict[str, int] = {}
        # A published port stays assigned for this process's lifetime, even
        # after uninstall. A proxy reload cannot race a different app taking it.
        self._published_ports = {
            record.id: record.port for record in store.installed() if self._public_origin(record.id)
        }
        self.catalogue = {m.id: m for m in catalogue}
        self._get_config = get_config
        self._bind_host = bind_host
        self._advertise_host = advertise_host
        self._node_name = node_name
        self._resolve_gateway = resolve_gateway
        self.installer = AppInstaller()
        # C1: each app in an account of its own where this install can make
        # one; the agent's own supervisor, and its account, where it cannot.
        self.accounts = accounts if accounts is not None else app_accounts.detect()
        self.supervisor: AppSupervisor | app_accounts.OwnAccountSupervisor
        if self.accounts.available:
            self.supervisor = app_accounts.OwnAccountSupervisor(
                app_accounts.runner_for(self.accounts, store.root),
                ingress=ingress_url or (lambda: ""),
                data_dir=store.data_dir,
                key_file=store.key_file,
            )
        else:
            self.supervisor = AppSupervisor()
        self._gateway: dict[str, str | None] = {}
        self._detail: dict[str, str | None] = {}
        self._lock = asyncio.Lock()

    def node_name(self) -> str | None:
        return self._node_name()

    # --- catalogue ------------------------------------------------------

    def manifest(self, app_id: str) -> tuple[AppManifest, AppOrigin] | None:
        """The entry to install for `app_id`. A shipped entry wins over a
        custom one of the same id, which `add_custom` refuses anyway."""
        if app_id in self.catalogue:
            return self.catalogue[app_id], AppOrigin.catalogue
        for manifest in self.store.custom():
            if manifest.id == app_id:
                return manifest, AppOrigin.custom
        return None

    def uv(self) -> Path | None:
        configured = self._get_config("uvBinary")
        return find_uv(str(configured) if configured else None)

    def installable(self, *, enrolled: bool) -> str | None:
        """None when this node can install apps, otherwise why not."""
        if self.uv() is None:
            configured = self._get_config("uvBinary")
            if configured:
                return (
                    f"uvBinary is set to {configured}, which is not a file. Fix it under Config, "
                    "or clear it to use the installer's copy."
                )
            return (
                "This agent cannot find uv, which it installs apps with. Installs made by the "
                "Eugene Plexus installer have it; on a developer install, put uv on PATH or set "
                "uvBinary under Config."
            )
        if not enrolled:
            return (
                "Finish first-run setup on this machine before installing apps. Until then its "
                "agent signs with a key that changes every time it starts, so an app's key "
                "would stop working at the next restart."
            )
        return None

    def as_catalogue(self, *, enrolled: bool) -> AppCatalogue:
        entries: list[AppCatalogueEntry] = []
        seen: set[str] = set()
        for origin, manifests in (
            (AppOrigin.catalogue, list(self.catalogue.values())),
            (AppOrigin.custom, self.store.custom()),
        ):
            for manifest in manifests:
                if manifest.id in seen or is_node_files(manifest.id, manifest.entry):
                    continue
                seen.add(manifest.id)
                record = self.store.get(manifest.id)
                entries.append(
                    AppCatalogueEntry(
                        manifest=manifest,
                        origin=origin,
                        installedVersion=record.version if record else None,
                    )
                )
        reason = self.installable(enrolled=enrolled)
        return AppCatalogue(
            apps=entries,
            installable=reason is None,
            reason=reason,
            ownAccounts=self.accounts.available,
            ownAccountsReason=self.accounts.reason,
        )

    def local_actions_refusal(self, manifest: AppManifest) -> str | None:
        """Why `manifest` may not be installed on this node, or None (C1).

        An app that runs what a model chooses, in the agent's own account,
        could reach the install's keys; it installs only where it gets an
        account of its own. An entry that does not say is treated as one
        that does.
        """
        if self.accounts.available or manifest.localActions is False:
            return None
        return (
            f"{manifest.name} runs actions a model chooses on this machine, so it needs an "
            f"account of its own, and this install cannot make one. {self.accounts.reason}"
        )

    # --- views ----------------------------------------------------------

    def ui_url(self, record: InstalledApp) -> str | None:
        if not record.manifest.ui:
            return None
        origin = self._public_origin(record.id)
        if origin:
            return origin + "/"
        if self._private_apps:
            return None
        host = self._advertise_host() or "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{record.port}/"

    def view(self, record: InstalledApp) -> App:
        status, error, restarted, pid, _ = self.supervisor.status(record.id)
        if not record.enabled and not self.supervisor.is_running(record.id):
            status = ComponentStatus.exited
        key = self.store.key(record.id)
        detail = self._detail.get(record.id)
        if self._private_apps and record.manifest.ui and not self._public_origin(record.id):
            detail = "This app has no published address in single-port mode."
        if record.manifest.signIn and not self.sign_in_registration_current(record.manifest):
            detail = (
                "Eugene is moving this app's sign-in address to where it is opened now. If "
                "this stays, Restart it in Apps. Chats are kept."
            )
        if key is not None and not self.store.key_file(record.id).is_file():
            detail = (
                "Its key file is missing, so it cannot reach the hub. Uninstall and install it "
                "again to mint a new key."
            )
        return App(
            id=record.id,
            name=record.manifest.name,
            version=record.version,
            previousVersion=record.previous_version,
            origin=record.origin,
            node=self.node_name(),
            enabled=record.enabled,
            status=status,
            port=record.port,
            uiUrl=_url(self.ui_url(record)),
            gatewayUrl=_url(self._gateway.get(record.id)),
            ui=bool(record.manifest.ui),
            configTrio=bool(record.manifest.configTrio),
            uses=list(record.manifest.uses or []),
            keyId=key.key_id if key else None,
            keyName=key.key_name if key else None,
            isolation=(
                AppIsolation.own_account if self.accounts.available else AppIsolation.agent_account
            ),
            account=self._account(record.id),
            localActions=record.manifest.localActions is not False,
            signIn=bool(record.manifest.signIn),
            oidcClientId=self.store.oidc_client(record.id),
            installedAt=record.installed_at,
            pid=pid,
            lastRestart=restarted,
            lastError=error,
            detail=detail,
        )

    def _account(self, app_id: str) -> str:
        if isinstance(self.supervisor, app_accounts.OwnAccountSupervisor):
            return self.supervisor.account(app_id)
        return "the agent's own account"

    def views(self) -> list[App]:
        return [
            self.view(r)
            for r in self.store.installed()
            if not is_node_files(r.id, r.manifest.entry)
        ]

    # --- ports ----------------------------------------------------------

    def reserve_port(self, app_id: str) -> int:
        """The port this app will have: its own if installed, otherwise one
        set aside now, so its sign-in callback can be registered before
        the install finishes (C2) and the install then takes the same one."""
        if self._public_origin(app_id):
            if app_id not in self._published_ports:
                installed = self.store.get(app_id)
                self._published_ports[app_id] = (
                    installed.port if installed else self.allocate_port()
                )
            return self._published_ports[app_id]
        existing = self.store.get(app_id)
        if existing is not None:
            return existing.port
        if app_id not in self._reserved_ports:
            self._reserved_ports[app_id] = self.allocate_port()
        return self._reserved_ports[app_id]

    def _origins(self, app_id: str) -> list[str]:
        """Every address a browser may open this app at, first the one the
        console opens: this node's advertised host, then loopback."""
        origin = self._public_origin(app_id)
        if origin:
            return [origin]
        port = self.reserve_port(app_id)
        hosts = [self._advertise_host() or "127.0.0.1", "127.0.0.1", "localhost"]
        out = []
        for host in dict.fromkeys(hosts):
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            out.append(f"http://{host}:{port}")
        return out

    def sign_in_redirects(self, app_id: str, path: str) -> list[str]:
        """Every address a browser may open this app at, with its callback."""
        return [origin + path for origin in self._origins(app_id)]

    def sign_in_registration_current(self, manifest: AppManifest) -> bool:
        redirects = self.sign_in_redirects(
            manifest.id, manifest.signInCallbackPath or "/oidc/callback"
        )
        stamp = self.store.sign_in_stamp_file(manifest.id)
        previous = None
        with contextlib.suppress(OSError, ValueError):
            previous = json.loads(stamp.read_text(encoding="utf-8"))
        return previous == redirects or (
            previous is None and not redirects[0].startswith("https://")
        )

    def app_url(self, app_id: str) -> str:
        """`{appUrl}` (C4): the address the console opens the app at, with
        no path -- the first of its sign-in addresses."""
        return self._origins(app_id)[0]

    def oidc_issuer(self) -> str | None:
        return self._oidc_issuer()

    def allocate_port(self) -> int:
        taken = (
            self.store.taken_ports()
            | set(self._reserved_ports.values())
            | set(self._published_ports.values())
        )
        for candidate in range(APP_PORT_BASE, APP_PORT_BASE + APP_PORT_SPAN):
            if candidate in taken:
                continue
            if ports.is_free(candidate):
                return candidate
        raise AppInstallError(
            f"no free port in {APP_PORT_BASE}-{APP_PORT_BASE + APP_PORT_SPAN - 1} for an app"
        )

    # --- lifecycle ------------------------------------------------------

    async def start(self, record: InstalledApp) -> None:
        """Resolve the gateway now, then spawn. Resolved per start rather
        than once, because the gateway can move and a restart is when an
        operator expects an app to notice."""
        url, detail = await self._resolve_gateway() if record.manifest.uses else (None, None)
        self._gateway[record.id] = url
        self._detail[record.id] = detail
        planner = _AppPlanner(
            record,
            store=self.store,
            gateway_url=lambda: self._gateway.get(record.id),
            bind_host=(lambda: "127.0.0.1") if record.id == "node-files" else self._bind_host,
            oidc_issuer=self._oidc_issuer,
            app_url=lambda: self.app_url(record.id),
            public_origin=self._public_origin(record.id),
            oidc_backchannel=self._oidc_backchannel if record.id == "workbench" else None,
        )
        self.supervisor.start(planner)
        if self.after_start:
            self.after_start(record.id)

    async def stop(self, app_id: str) -> None:
        if self.before_stop:
            await self.before_stop(app_id)
        await self.supervisor.stop(app_id)

    async def restart(self, record: InstalledApp) -> None:
        await self.stop(record.id)
        await self.start(record)

    async def start_enabled(self) -> None:
        """Boot: bring up every app the operator left running. Never
        raises -- one app that will not start is that app's `lastError`."""
        for record in self.store.installed():
            if not record.enabled:
                continue
            try:
                await self.start(record)
            except Exception:  # pragma: no cover - defensive
                log.exception("could not start app %s at boot", record.id)

    async def installed_callback(self, manifest: AppManifest, origin: AppOrigin) -> None:
        """Run when a version's environment is committed: record it, keep
        the previous version for rollback, prune the rest, restart onto it."""
        manifest = normalized(manifest)
        async with self._lock:
            existing = self.store.get(manifest.id)
            if existing is None:
                record = InstalledApp(
                    manifest=manifest,
                    origin=origin,
                    port=(
                        self.reserve_port(manifest.id)
                        if self._public_origin(manifest.id)
                        else self._reserved_ports.pop(manifest.id, 0) or self.allocate_port()
                    ),
                    installed_at=datetime.now(UTC),
                )
            else:
                record = InstalledApp(
                    manifest=manifest,
                    origin=origin,
                    port=existing.port,
                    installed_at=datetime.now(UTC),
                    enabled=existing.enabled,
                    previous_version=(
                        existing.version if existing.version != manifest.version else None
                    ),
                )
            self.store.put(record)
            keep = {record.version}
            if record.previous_version:
                keep.add(record.previous_version)
            if record.enabled:
                await self.restart(record)
            prune_versions(self.store, record.id, keep)

    async def uninstall(self, app_id: str, *, purge: bool) -> None:
        await self.installer.cancel(app_id)
        if self.before_stop:
            await self.before_stop(app_id)
        if isinstance(self.supervisor, app_accounts.OwnAccountSupervisor):
            # Its service and its account go with it.
            await self.supervisor.remove(app_id, purge=purge)
        else:
            await self.supervisor.stop(app_id)
        self.store.remove(app_id)
        self._gateway.pop(app_id, None)
        self._detail.pop(app_id, None)
        if purge:
            _remove_quietly(self.store.app_dir(app_id))
        else:
            # Keep `data`, drop every environment and the key file: the key
            # has just been revoked, and a revoked token on disk is litter.
            _remove_quietly(self.store.app_dir(app_id) / "versions")
            with contextlib.suppress(OSError):
                self.store.key_file(app_id).unlink(missing_ok=True)

    async def aclose(self) -> None:
        await self.installer.aclose()
        await self.supervisor.stop_all()
