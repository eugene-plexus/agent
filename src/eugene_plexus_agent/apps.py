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
only `EUGENE_PLEXUS_APP_*`.

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

from . import ports
from ._generated.models import (
    App,
    AppCatalogue,
    AppCatalogueEntry,
    AppHubSurface,
    AppInstall,
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

#: The installed version and one to roll back to, as for engines.
RETAINED_VERSIONS = 2

#: Apps bind from their own range, above the runtimes' 8090-8189, so an
#: app's port is never one a later runtime or companion is handed.
APP_PORT_BASE = 8190
APP_PORT_SPAN = 100

ENV_PREFIX = "EUGENE_PLEXUS_APP"

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
    """`<package> @ <url>`, the one form `uv pip install` is given.

    A directory is turned into a `file://` URL because PEP 508 direct
    references are URLs; a relative path would resolve against whatever
    directory the agent happened to start in, so it is refused rather
    than guessed at.
    """
    source = manifest.source.strip()
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


def load_catalogue() -> list[AppManifest]:
    """The apps this agent release ships, from its own package data.

    Never raises. A catalogue that will not parse is a packaging bug, and
    the answer to it is an empty catalogue and a log line -- not an agent
    that will not start.
    """
    try:
        text = (
            importlib.resources.files("eugene_plexus_agent")
            .joinpath(CATALOGUE_RESOURCE)
            .read_text(encoding="utf-8")
        )
        raw = yaml.safe_load(text) or []
        if not isinstance(raw, list):
            raise ValueError("the catalogue must be a YAML list")
        return [normalized(AppManifest.model_validate(item)) for item in raw]
    except (OSError, ValueError, ValidationError) as exc:
        log.error("the shipped app catalogue could not be read (%s); offering none", exc)
        return []


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


# Imports the entry point without running it. A package whose `-m` target
# is a package also needs a `__main__`, which is the case `find_spec` on
# the package alone would wave through and the first spawn would not.
_VERIFY_SNIPPET = (
    "import importlib.util as u, sys\n"
    "name = sys.argv[1]\n"
    "spec = u.find_spec(name)\n"
    "if spec is None:\n"
    "    sys.exit('no module named ' + name)\n"
    "if spec.submodule_search_locations is not None and u.find_spec(name + '.__main__') is None:\n"
    "    sys.exit(name + ' is a package with no __main__, so python -m cannot run it')\n"
)


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
    ) -> None:
        self.record = record
        self._store = store
        self._gateway_url = gateway_url
        self._bind_host = bind_host
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

    def plan(self) -> SpawnPlan:
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
        host = self._bind_host()
        if host:
            env[f"{ENV_PREFIX}_BIND_HOST"] = host
        gateway = self._gateway_url()
        if gateway:
            env[f"{ENV_PREFIX}_GATEWAY_URL"] = gateway
        env["PYTHONUNBUFFERED"] = "1"
        return SpawnPlan(
            argv=[str(python), "-m", record.manifest.entry],
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
    health poll, which reads `GET /healthz` on the app's own port -- an
    app that never answers it reads `starting` for as long as it runs,
    which is what the manifest's contract says it owes.
    """

    def __init__(self) -> None:
        self._processes: dict[str, SupervisedProcess] = {}
        self._planners: dict[str, _AppPlanner] = {}
        self._reachable: dict[str, bool] = {}
        self._ports: dict[str, int] = {}
        self._health_task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None

    def start(self, planner: _AppPlanner) -> None:
        app_id = planner.record.id
        if app_id in self._processes:
            return
        process = SupervisedProcess(planner, log)
        self._processes[app_id] = process
        self._planners[app_id] = planner
        self._ports[app_id] = planner.record.port
        self._reachable[app_id] = False
        process.start()
        if self._health_task is None:
            self._client = internal_client(timeout=2.0)
            self._health_task = asyncio.create_task(self._health_loop(), name="app-health")

    async def stop(self, app_id: str) -> None:
        process = self._processes.pop(app_id, None)
        self._planners.pop(app_id, None)
        self._ports.pop(app_id, None)
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
        self._ports.clear()
        if self._client is not None:
            with contextlib.suppress(BaseException):
                await self._client.aclose()
            self._client = None

    async def _health_loop(self) -> None:
        try:
            while True:
                await asyncio.gather(
                    *(self._poll(app_id, port) for app_id, port in list(self._ports.items())),
                    return_exceptions=True,
                )
                await asyncio.sleep(_HEALTH_POLL_SECONDS)
        except asyncio.CancelledError:
            return

    async def _poll(self, app_id: str, port: int) -> None:
        client = self._client
        if client is None:
            return
        try:
            response = await client.get(f"http://127.0.0.1:{port}/healthz")
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
    ) -> None:
        self.store = store
        self.catalogue = {m.id: m for m in catalogue}
        self._get_config = get_config
        self._bind_host = bind_host
        self._advertise_host = advertise_host
        self._node_name = node_name
        self._resolve_gateway = resolve_gateway
        self.installer = AppInstaller()
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
                if manifest.id in seen:
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
        return AppCatalogue(apps=entries, installable=reason is None, reason=reason)

    # --- views ----------------------------------------------------------

    def ui_url(self, record: InstalledApp) -> str | None:
        if not record.manifest.ui:
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
            installedAt=record.installed_at,
            pid=pid,
            lastRestart=restarted,
            lastError=error,
            detail=detail,
        )

    def views(self) -> list[App]:
        return [self.view(r) for r in self.store.installed()]

    # --- ports ----------------------------------------------------------

    def allocate_port(self) -> int:
        taken = self.store.taken_ports()
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
        url, detail = await self._resolve_gateway()
        self._gateway[record.id] = url
        self._detail[record.id] = detail
        planner = _AppPlanner(
            record,
            store=self.store,
            gateway_url=lambda: self._gateway.get(record.id),
            bind_host=self._bind_host,
        )
        self.supervisor.start(planner)

    async def stop(self, app_id: str) -> None:
        await self.supervisor.stop(app_id)

    async def restart(self, record: InstalledApp) -> None:
        await self.supervisor.stop(record.id)
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
                    port=self.allocate_port(),
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
