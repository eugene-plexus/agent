"""Fetching, verifying and retaining engine builds.

Engine releases are a managed artifact here, alongside configs and (later)
models. This module owns everything about that which is not specific to a
particular engine: where builds live on disk, how a download is verified,
how an install reports progress, and what gets pruned afterwards. Which
release and which asset — that is the adapter's job, because it is the same
kind of knowledge as "how do I build the argv".

Two rules run through the whole file.

**A build is never written in place.** Each lands in its own versioned
directory and only becomes visible as the engine's binary once it has
verified and unpacked. A runtime may be executing the current build right
now, and overwriting the file underneath a loaded engine is how you get a
crash nobody can explain.

**A failed match is loud.** Upstream's asset naming is not a contract —
llama.cpp renames variants and bakes toolchain versions into filenames — so
when nothing matches, the error names what was looked for and lists what was
actually there. Falling back to something plausible would install the wrong
accelerator's build and present it as success.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .._generated.models import EngineInstall, EngineKind, State

log = logging.getLogger(__name__)

# Where managed builds live. Sits under the same `~/.eugene-plexus` root the
# topology already uses for component config files.
DEFAULT_ENGINE_ROOT = Path.home() / ".eugene-plexus" / "engines"

# Upstream is asked at most this often. llama.cpp publishes several builds a
# day, so anything tighter is pure rate-limit spend for information nobody
# acts on — and the unauthenticated GitHub limit is 60/hour for everything
# this process does, not just us.
RELEASE_CACHE_SECONDS = 24 * 60 * 60

_DOWNLOAD_TIMEOUT_SECONDS = 60.0
_DOWNLOAD_CHUNK = 1024 * 256

# Retention: the current build and the one before it. Every build is not
# viable — a Windows CUDA install is roughly half a gigabyte unpacked — and
# only the newest leaves a bad upgrade with no way back.
RETAINED_BUILDS = 2


class AcquisitionError(Exception):
    """An install could not be completed. The message is for the operator."""


@dataclass(frozen=True)
class ReleaseAsset:
    """One downloadable file from an upstream release."""

    name: str
    url: str
    size: int
    #: `sha256:…` as the releases API reports it. Upstream publishes no
    #: checksum *file*; this digest arrives with the asset metadata, which
    #: means verification costs one comparison and no extra request.
    digest: str | None = None

    @property
    def sha256(self) -> str | None:
        if self.digest and self.digest.startswith("sha256:"):
            return self.digest.removeprefix("sha256:")
        return None


@dataclass(frozen=True)
class Release:
    """One upstream release and its assets."""

    version: str
    published_at: datetime | None
    assets: tuple[ReleaseAsset, ...]


@dataclass(frozen=True)
class AcquisitionPlan:
    """What to fetch for this host, as an adapter worked it out.

    `assets` is a list because Windows + CUDA needs two: the server binary
    and a separate CUDA runtime zip. They unpack into the same directory,
    which is also why `working_directory()` defaults to the binary's parent.
    """

    version: str
    variant: str
    assets: tuple[ReleaseAsset, ...]
    #: Path of the executable inside the unpacked directory, once extracted.
    #: Resolved by search rather than declared, because archive layouts
    #: differ between platforms and upstream has changed them before.
    binary_name: str

    @property
    def total_bytes(self) -> int:
        return sum(a.size for a in self.assets)


@dataclass(frozen=True)
class Unavailable:
    """No build can be installed here, and why.

    A first-class outcome rather than an exception: upstream publishes no
    CUDA build for Linux, so a Linux box with an NVIDIA GPU legitimately has
    nothing installable and the operator needs to be told that in those
    words — not handed a slower build nobody offered them.
    """

    reason: str


# --------------------------------------------------------------------------- #
# GitHub releases
# --------------------------------------------------------------------------- #


class GitHubReleases:
    """Reads a repository's releases, with a coarse cache.

    Separate from the adapter so it can be swapped in tests without going
    near the network, and so the cache is shared if a second engine ever
    also ships from GitHub.
    """

    def __init__(self, repo: str, *, cache_seconds: float = RELEASE_CACHE_SECONDS) -> None:
        self._repo = repo
        self._cache_seconds = cache_seconds
        self._cached: list[Release] | None = None
        self._cached_at: float | None = None
        self._checked_at: datetime | None = None

    @property
    def checked_at(self) -> datetime | None:
        """When upstream last actually answered.

        Wall-clock, unlike the monotonic cache stamp beside it: this one
        is shown to a person, and "checked 3 minutes ago" has to survive
        the process being asked about it. Stays put when a check fails,
        which is the point — a stale timestamp is how an operator sees
        that we have not been able to reach GitHub.
        """
        return self._checked_at

    def list_releases(self, *, force: bool = False) -> list[Release]:
        """Recent releases, newest first. Cached; never raises on failure.

        An empty list means "we don't know", not "there are none". A failed
        check has to be invisible: it leaves the last-known state stale
        rather than turning a UI panel into an error.
        """
        now = time.monotonic()
        if (
            not force
            and self._cached is not None
            and self._cached_at is not None
            and now - self._cached_at < self._cache_seconds
        ):
            return self._cached

        try:
            raw = self._fetch(f"https://api.github.com/repos/{self._repo}/releases?per_page=30")
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
            log.debug("could not list releases for %s: %s", self._repo, e)
            return self._cached or []

        releases: list[Release] = []
        for entry in raw if isinstance(raw, list) else []:
            if not isinstance(entry, dict):
                continue
            tag = entry.get("tag_name")
            if not isinstance(tag, str) or not tag:
                continue
            assets = tuple(
                ReleaseAsset(
                    name=a["name"],
                    url=a["browser_download_url"],
                    size=int(a.get("size") or 0),
                    digest=a.get("digest"),
                )
                for a in (entry.get("assets") or [])
                if isinstance(a, dict) and a.get("name") and a.get("browser_download_url")
            )
            releases.append(
                Release(
                    version=tag,
                    published_at=_parse_timestamp(entry.get("published_at")),
                    assets=assets,
                )
            )

        self._cached = releases
        self._cached_at = now
        self._checked_at = datetime.now(UTC)
        return releases

    @staticmethod
    def _fetch(url: str) -> object:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "eugene-plexus-watchdog",
            },
        )
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# On-disk store
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InstalledBuild:
    """A managed build on disk."""

    version: str
    directory: Path
    binary: Path
    variant: str
    installed_at: datetime | None
    size_bytes: int


class ManagedStore:
    """The managed builds for one engine, on disk.

    Layout is `<root>/<engine>/<version>/` with a small `install.json`
    beside the unpacked tree. The metadata file is what makes a directory a
    *managed* build rather than a folder someone happened to leave there —
    without it we cannot say which variant it is, and a directory whose
    install was interrupted is distinguishable from a complete one.
    """

    METADATA_NAME = "install.json"

    def __init__(self, root: Path, engine: EngineKind) -> None:
        self._dir = root / engine.value

    @property
    def directory(self) -> Path:
        return self._dir

    def build_dir(self, version: str) -> Path:
        return self._dir / version

    def list_builds(self) -> list[InstalledBuild]:
        """Complete builds, newest install first."""
        if not self._dir.is_dir():
            return []
        out: list[InstalledBuild] = []
        for child in self._dir.iterdir():
            if not child.is_dir():
                continue
            build = self._read(child)
            if build is not None:
                out.append(build)
        out.sort(key=lambda b: b.installed_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        return out

    def current(self) -> InstalledBuild | None:
        builds = self.list_builds()
        return builds[0] if builds else None

    def _read(self, directory: Path) -> InstalledBuild | None:
        meta_path = directory / self.METADATA_NAME
        if not meta_path.is_file():
            # A partial extraction, or somebody's own folder. Either way not
            # something to hand to a supervisor as an executable.
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        binary = directory / str(meta.get("binary") or "")
        if not binary.is_file():
            return None
        return InstalledBuild(
            version=str(meta.get("version") or directory.name),
            directory=directory,
            binary=binary,
            variant=str(meta.get("variant") or "unknown"),
            installed_at=_parse_timestamp(meta.get("installedAt")),
            size_bytes=int(meta.get("sizeBytes") or 0),
        )

    def write_metadata(self, directory: Path, *, version: str, variant: str, binary: Path) -> None:
        payload = {
            "version": version,
            "variant": variant,
            # Relative, so moving or renaming the engine root doesn't
            # invalidate every install record.
            "binary": str(binary.relative_to(directory)),
            "installedAt": datetime.now(UTC).isoformat(),
            "sizeBytes": _directory_size(directory),
        }
        (directory / self.METADATA_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def prune(self, keep: int = RETAINED_BUILDS) -> list[str]:
        """Drop all but the newest `keep` builds. Returns what went."""
        removed: list[str] = []
        for build in self.list_builds()[keep:]:
            try:
                shutil.rmtree(build.directory)
                removed.append(build.version)
            except OSError as e:
                # Never fail an install because cleanup couldn't finish —
                # on Windows the previous build may still be mapped by a
                # process that hasn't fully exited.
                log.warning("could not prune %s: %s", build.directory, e)
        return removed


def _directory_size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        if path.is_file():
            with contextlib.suppress(OSError):
                total += path.stat().st_size
    return total


# --------------------------------------------------------------------------- #
# The install itself
# --------------------------------------------------------------------------- #


@dataclass
class _Progress:
    """Mutable state one install writes and the API reads."""

    engine: EngineKind
    state: State = State.resolving
    version: str | None = None
    variant: str | None = None
    bytes_downloaded: int = 0
    bytes_total: int | None = None
    message: str | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def snapshot(self) -> EngineInstall:
        return EngineInstall(
            engine=self.engine,
            state=self.state,
            version=self.version,
            variant=self.variant,
            bytesDownloaded=self.bytes_downloaded,
            bytesTotal=self.bytes_total,
            message=self.message,
            error=self.error,
            startedAt=self.started_at,
            finishedAt=self.finished_at,
        )


class EngineInstaller:
    """Runs at most one install per engine, in the background.

    The install is a task rather than a request because the download is
    hundreds of megabytes; holding an HTTP request open for it buys nothing
    and loses the progress an operator actually wants. Terminal state is
    retained until the next install starts, so a UI that reconnects after
    the fact still learns how it ended rather than finding an empty slot
    and assuming success.
    """

    def __init__(self, store: ManagedStore, engine: EngineKind) -> None:
        self._store = store
        self._engine = engine
        self._progress: _Progress | None = None
        self._task: asyncio.Task[None] | None = None

    # --- state ------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def snapshot(self) -> EngineInstall | None:
        return self._progress.snapshot() if self._progress is not None else None

    # --- lifecycle --------------------------------------------------------

    def start(self, plan: AcquisitionPlan) -> EngineInstall:
        if self.running:
            raise AcquisitionError(f"an install is already running for {self._engine.value}")
        progress = _Progress(
            engine=self._engine,
            state=State.downloading,
            version=plan.version,
            variant=plan.variant,
            bytes_total=plan.total_bytes,
            message=f"fetching {plan.variant} {plan.version}",
        )
        self._progress = progress
        self._task = asyncio.create_task(
            self._run(plan, progress), name=f"engine-install-{self._engine.value}"
        )
        return progress.snapshot()

    async def cancel(self) -> EngineInstall | None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            # The task records its own outcome in `_progress` before it
            # unwinds, so whatever comes out here is already reported.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        return self.snapshot()

    async def aclose(self) -> None:
        await self.cancel()

    # --- the work ---------------------------------------------------------

    async def _run(self, plan: AcquisitionPlan, progress: _Progress) -> None:
        staging = self._store.build_dir(f".staging-{plan.version}")
        try:
            await asyncio.to_thread(self._install, plan, progress, staging)
            progress.state = State.done
            progress.message = f"installed {plan.version}"
        except asyncio.CancelledError:
            progress.state = State.cancelled
            progress.message = "cancelled"
            _remove_quietly(staging)
            raise
        except Exception as e:
            log.warning("engine install failed: %s", e)
            progress.state = State.failed
            progress.error = str(e)
            _remove_quietly(staging)
        finally:
            progress.finished_at = datetime.now(UTC)

    def _install(self, plan: AcquisitionPlan, progress: _Progress, staging: Path) -> None:
        _remove_quietly(staging)
        staging.mkdir(parents=True, exist_ok=True)

        archives: list[Path] = []
        for asset in plan.assets:
            progress.state = State.downloading
            progress.message = f"downloading {asset.name}"
            archive = staging / asset.name
            _download(asset, archive, progress)
            archives.append(archive)

            progress.state = State.verifying
            progress.message = f"verifying {asset.name}"
            _verify(asset, archive)

        progress.state = State.extracting
        for archive in archives:
            progress.message = f"extracting {archive.name}"
            _extract(archive, staging)
            # The archive itself is dead weight once unpacked, and these are
            # hundreds of megabytes.
            archive.unlink(missing_ok=True)

        binary = _find_binary(staging, plan.binary_name)
        if binary is None:
            raise AcquisitionError(
                f"{plan.binary_name!r} not found in the extracted {plan.variant} archive — "
                f"upstream may have changed its layout"
            )
        _make_executable(binary)

        final = self._store.build_dir(plan.version)
        _remove_quietly(final)
        # The rename is the commit point: until it lands, nothing outside
        # this function can see a half-installed build.
        staging.replace(final)
        binary = final / binary.relative_to(staging)
        self._store.write_metadata(final, version=plan.version, variant=plan.variant, binary=binary)
        self._store.prune()


def _download(asset: ReleaseAsset, target: Path, progress: _Progress) -> None:
    request = urllib.request.Request(asset.url, headers={"User-Agent": "eugene-plexus-watchdog"})
    try:
        with (
            urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response,
            target.open("wb") as out,
        ):
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                progress.bytes_downloaded += len(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise AcquisitionError(f"downloading {asset.name} failed: {e}") from e


def _verify(asset: ReleaseAsset, archive: Path) -> None:
    expected = asset.sha256
    if expected is None:
        # Upstream stopped publishing a digest for this asset. Refuse rather
        # than install unverified: the whole point of managing binaries is
        # that the operator does not have to audit what we fetched.
        raise AcquisitionError(
            f"{asset.name} has no SHA-256 digest in its release metadata; refusing to "
            f"install an unverified engine binary"
        )
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise AcquisitionError(
            f"{asset.name} failed verification: expected sha256 {expected}, got {actual}"
        )


def _extract(archive: Path, destination: Path) -> None:
    name = archive.name.lower()
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                _guard_members(archive, zf.namelist())
                zf.extractall(destination)
        elif name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive, mode="r:gz") as tf:
                _guard_members(archive, tf.getnames())
                # `data` filter rejects absolute paths, traversal and special
                # files. Explicit because it is only the default from 3.14 and
                # CI runs 3.12.
                tf.extractall(destination, filter="data")
        else:
            raise AcquisitionError(f"don't know how to unpack {archive.name}")
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as e:
        raise AcquisitionError(f"extracting {archive.name} failed: {e}") from e


def _guard_members(archive: Path, names: list[str]) -> None:
    """Refuse an archive that would write outside the destination.

    `tarfile`'s data filter covers the tar case, but zipfile has no
    equivalent and these archives come off the internet.
    """
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise AcquisitionError(
                f"{archive.name} contains an unsafe path {name!r}; refusing to extract"
            )


def _find_binary(root: Path, binary_name: str) -> Path | None:
    """Locate the executable in an extracted tree.

    Searched rather than assumed: llama.cpp's Windows zips put binaries at
    the top level while some archives nest them under `build/bin`, and that
    has changed between releases before.
    """
    candidates = [binary_name, f"{binary_name}.exe"]
    for candidate in candidates:
        direct = root / candidate
        if direct.is_file():
            return direct
    for candidate in candidates:
        for found in sorted(root.rglob(candidate)):
            if found.is_file():
                return found
    return None


def _make_executable(binary: Path) -> None:
    """Restore the executable bit, which zip archives do not carry."""
    try:
        binary.chmod(binary.stat().st_mode | 0o111)
    except OSError as e:
        log.debug("could not chmod %s: %s", binary, e)


def _remove_quietly(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def engine_root() -> Path:
    """Where managed builds live.

    Overridable for tests and for operators who keep this project off their
    home volume — half a gigabyte per Windows CUDA build, retained twice.
    """
    override = os.environ.get("EUGENE_PLEXUS_WATCHDOG_ENGINE_ROOT")
    return Path(override).expanduser() if override else DEFAULT_ENGINE_ROOT


__all__ = [
    "AcquisitionError",
    "AcquisitionPlan",
    "EngineInstaller",
    "GitHubReleases",
    "InstalledBuild",
    "ManagedStore",
    "Release",
    "ReleaseAsset",
    "Unavailable",
    "engine_root",
]
