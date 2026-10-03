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
import socket
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from .._generated.models import EngineInstall, EngineKind, State
from .._http import egress_ssl_context

log = logging.getLogger(__name__)

# Where managed builds live. Sits under the same `~/.eugene-plexus` root the
# topology already uses for component config files.
DEFAULT_ENGINE_ROOT = Path.home() / ".eugene-plexus" / "engines"

# Upstream is asked at most this often. llama.cpp publishes several builds a
# day, so anything tighter is pure rate-limit spend for information nobody
# acts on — and the unauthenticated GitHub limit is 60/hour for everything
# this process does, not just us.
RELEASE_CACHE_SECONDS = 24 * 60 * 60

# **A failure is cached too, and it has to be** (review §6.1 #5). The
# success cache above is a day wide and the failure path had none at all,
# so an upstream that does not answer was re-dialled on every single
# request — and `GET /v1/engines` is polled by Home every 15 s and by
# `useIssues` every 30 s per node. Five minutes is short enough that an
# operator who fixes their network does not wait for it and long enough
# that a captive portal or a blocked egress costs one request per poll
# cycle rather than all of them.
RELEASE_FAILURE_BACKOFF_SECONDS = 5 * 60

_DOWNLOAD_TIMEOUT_SECONDS = 60.0
# **Metadata is not a download.** The same 60 s sat under a one-JSON-body
# read that a polled route waits on, so a box behind a firewall that
# blackholes rather than refuses held the agent's event loop for a
# minute. The listing is retried on the next poll; the download cannot
# be, so the two deserve different patience.
_METADATA_TIMEOUT_SECONDS = 10.0
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
    #: Names of assets whose backend is ADDED to the build rather than
    #: unpacked over it: the Vulkan build, for a `+vulkan` variant
    #: (2026-09-27). Each is unpacked beside the build, every file both
    #: carry must be byte-identical, and only the files the build lacks
    #: are copied in. A release where they differ is refused, because
    #: two backends compiled against different cores in one process is a
    #: crash nobody could diagnose.
    plugins: frozenset[str] = frozenset()
    #: Names of assets carrying libraries the server loads from ITS OWN
    #: folder: the CUDA runtime (`cudart-llama-…`). Each is unpacked on
    #: its own and its files are put in the folder that holds the server
    #: binary, however the archive wraps them. **The Linux tarball wraps
    #: them in a folder of their own** (`cudart-llama-bNNNN-bin-ubuntu-
    #: cuda-13.4-x64/`) beside the server's `llama-bNNNN/`, and
    #: `libggml-cuda.so` finds `libcudart.so.13` only through its RUNPATH,
    #: `$ORIGIN` -- so unpacked side by side, the CUDA backend could not
    #: load and llama.cpp ran on the processor without a word (drift
    #: audit 2026-10-03, reproduced at b11375). Upstream's release
    #: workflow says the same: extract it "next to the binaries ($ORIGIN
    #: rpath)". The Windows zips have no inner folder, so there the files
    #: land where they always did.
    runtimes: frozenset[str] = frozenset()

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

    `release_bound` separates two kinds of "no": a reason derived from
    **this release's assets** (the variant is not among them, the cudart
    companion is missing, no CUDA build for this architecture is in it)
    from one about **the host** (Linux with NVIDIA, a driver that reports
    no CUDA version). The first may be answered by an older release —
    upstream's CI uploads a build's assets over the better part of an
    hour, and b10991 sat at five of thirty-three for the run that found
    this — and the second cannot, so the selection stops on it.
    """

    reason: str
    release_bound: bool = False


# --------------------------------------------------------------------------- #
# Why upstream did not answer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FetchFailure:
    """Why a request to upstream produced no answer, in words a person acts on.

    **The cause is what was observed, never a guess.** Until 2026-09-26
    every failure here — a 403 rate limit, a certificate an antivirus had
    replaced, a DNS failure, a timeout — became *"could not reach the
    upstream release list. Check network access"*, and the exception was
    logged at DEBUG. A friend's first install said that on a machine whose
    network, and whose copy of our own Python, both reached GitHub fine,
    and nothing anywhere recorded what had actually happened.
    """

    #: What happened, as a clause: "api.github.com did not answer within 10 seconds".
    cause: str
    #: What to do about it, as whole sentences.
    next_step: str
    #: The exception as Python spelled it, for the log and nobody else.
    detail: str

    def sentence(self) -> str:
        return f"{self.cause}. {self.next_step}"


def _python_that_connects() -> str:
    """The interpreter a firewall or antivirus sees making the connection.

    On Windows a virtualenv's `python.exe` is a launcher that starts the
    base interpreter as a child, so a program rule has to name the base
    one — `sys.executable` names the launcher.
    """
    return getattr(sys, "_base_executable", None) or sys.executable


def _github_message(error: urllib.error.HTTPError) -> str | None:
    """GitHub's own words from an error body, when it sent JSON with a message."""
    try:
        body = error.read(4096)
        message = json.loads(body.decode("utf-8", "replace")).get("message")
    except (OSError, ValueError, AttributeError):
        return None
    return message.strip() if isinstance(message, str) and message.strip() else None


def _rate_limit_failure(error: urllib.error.HTTPError) -> FetchFailure | None:
    """GitHub's anonymous limit is per network ADDRESS, not per machine.

    A person who has made no requests at all can meet it, because a
    provider that puts many customers behind one address (Starlink, mobile
    data, carrier-grade NAT) shares the sixty between all of them.
    """
    headers = error.headers
    remaining = headers.get("X-RateLimit-Remaining") if headers is not None else None
    retry_after = headers.get("Retry-After") if headers is not None else None
    detail = f"HTTPError {error.code}: {error.reason}"

    if remaining == "0":
        limit = headers.get("X-RateLimit-Limit") or "60"
        when = "within the hour"
        reset = headers.get("X-RateLimit-Reset")
        if reset and reset.isdigit():
            at = datetime.fromtimestamp(int(reset))
            minutes = max(1, -(-(int(reset) - int(time.time())) // 60))
            when = f"at {at:%H:%M} (in {minutes} minute{'s' if minutes != 1 else ''})"
        return FetchFailure(
            cause=(
                f"GitHub's limit of {limit} requests an hour for this network's address is used up"
            ),
            next_step=(
                f"It resets {when}. An address shared by many customers, as on "
                f"Starlink or mobile data, can run out without this machine asking."
            ),
            detail=detail,
        )
    if error.code in (403, 429) and retry_after and retry_after.isdigit():
        seconds = int(retry_after)
        wait = f"{-(-seconds // 60)} minute(s)" if seconds >= 60 else f"{seconds} seconds"
        return FetchFailure(
            cause="GitHub is limiting requests from this network's address",
            next_step=f"Try again in {wait}.",
            detail=detail,
        )
    return None


def _certificate_failure(
    error: ssl.SSLCertVerificationError, *, host: str, detail: str
) -> FetchFailure:
    """A certificate this machine's own verifier refused, by why it refused.

    Egress is verified by the OS (`_http.egress_ssl_context`), so the words
    arrive in the OS's vocabulary: Windows says *"not within its validity
    period"* where OpenSSL says *"certificate has expired"*, and both are
    matched. Each reason sends a person somewhere different, so they are
    not merged: a wrong clock is fixed on this machine, a wrong name means
    something else answered, and an untrusted root means something replaced
    the certificate or the OS was not allowed to fetch the root.
    """
    why = (error.verify_message or str(error)).strip().rstrip(".")
    said = why.lower()
    cause = f"this machine could not verify the security certificate {host} presented ({why})"
    if "expired" in said or "not yet valid" in said or "validity period" in said:
        next_step = (
            "Check this machine's date and time. A clock that is far off makes "
            "every certificate look invalid."
        )
    elif "match" in said or "mismatch" in said:
        next_step = (
            "The certificate belongs to a different name, so something else is "
            "answering in GitHub's place, such as a sign-in page or a proxy."
        )
    else:
        next_step = (
            "Something between this machine and GitHub may be replacing its "
            "certificate, such as a proxy or an antivirus's web protection whose "
            "own certificate is not installed."
        )
        if sys.platform == "win32":
            # The one machine-side cause of an untrusted root on Windows once
            # the OS is the verifier: a hardened image that may not fetch one.
            next_step += (
                " Or Windows is not allowed to download root certificates (the "
                'Group Policy setting "Turn off Automatic Root Certificates Update").'
            )
    return FetchFailure(cause=cause, next_step=next_step, detail=detail)


def describe_fetch_failure(error: BaseException, *, url: str, timeout: float) -> FetchFailure:
    """Classify a failed upstream request by what actually went wrong.

    Order matters: `HTTPError` is a `URLError`, and a `URLError` wraps the
    real cause in `.reason`, which may itself be an exception or a string.
    """
    host = urlparse(url).hostname or url
    exe = _python_that_connects()
    detail = f"{type(error).__name__}: {error}"

    if isinstance(error, urllib.error.HTTPError):
        limited = _rate_limit_failure(error)
        if limited is not None:
            return limited
        said = _github_message(error)
        status = f"HTTP {error.code} {error.reason}".strip()
        if said and said.lower() != str(error.reason).lower():
            status = f"{status}: {said}"
        if error.code >= 500:
            return FetchFailure(
                cause=f"{host} answered with an error ({status})",
                next_step="GitHub may be having trouble. Try again in a few minutes.",
                detail=detail,
            )
        return FetchFailure(
            cause=f"{host} turned the request down ({status})",
            next_step="Try again later.",
            detail=detail,
        )

    cause: BaseException | str = error
    if isinstance(error, urllib.error.URLError):
        cause = error.reason
        if isinstance(cause, BaseException):
            # `<urlopen error certificate verify failed>` hides the class
            # that says which kind of failure it was.
            detail = f"URLError({type(cause).__name__}): {cause}"

    if isinstance(cause, ssl.SSLCertVerificationError):
        return _certificate_failure(cause, host=host, detail=detail)
    if isinstance(cause, ssl.SSLError):
        return FetchFailure(
            cause=f"the secure connection to {host} failed ({cause.reason or cause})",
            next_step=(
                "Something on this machine or network may be interfering with "
                "secure connections, such as an antivirus's web protection or a "
                f"proxy. Allow Eugene's Python through it: {exe}"
            ),
            detail=detail,
        )
    if isinstance(cause, socket.gaierror):
        return FetchFailure(
            cause=f"this machine could not look up the address of {host}",
            next_step="Check the network connection and its DNS settings.",
            detail=detail,
        )
    if isinstance(cause, TimeoutError):
        return FetchFailure(
            cause=f"{host} did not answer within {timeout:g} seconds",
            next_step=(
                "Check the network connection. A firewall that silently drops "
                f"Eugene's traffic looks like this too; its Python is {exe}"
            ),
            detail=detail,
        )
    if isinstance(cause, ConnectionRefusedError):
        return FetchFailure(
            cause=f"the connection to {host} was refused",
            next_step=f"A firewall or proxy may be blocking Eugene's Python: {exe}",
            detail=detail,
        )
    if isinstance(cause, (ConnectionResetError, ConnectionAbortedError)):
        return FetchFailure(
            cause=f"the connection to {host} was cut off",
            next_step=f"A firewall or antivirus may be blocking Eugene's Python: {exe}",
            detail=detail,
        )
    if isinstance(cause, ValueError):
        return FetchFailure(
            cause=f"the answer from {host} was not GitHub's release list",
            next_step=(
                "A sign-in page or a proxy may be answering in GitHub's place. "
                "Open any web page on this machine to check."
            ),
            detail=detail,
        )
    return FetchFailure(
        cause=f"the request to {host} failed ({cause})",
        next_step="Check the network connection.",
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# GitHub releases
# --------------------------------------------------------------------------- #


class GitHubReleases:
    """Reads a repository's releases, with a coarse cache.

    Separate from the adapter so it can be swapped in tests without going
    near the network, and so the cache is shared if a second engine ever
    also ships from GitHub.
    """

    def __init__(
        self,
        repo: str,
        *,
        cache_seconds: float = RELEASE_CACHE_SECONDS,
        failure_backoff_seconds: float = RELEASE_FAILURE_BACKOFF_SECONDS,
    ) -> None:
        self._repo = repo
        self._cache_seconds = cache_seconds
        self._failure_backoff_seconds = failure_backoff_seconds
        self._cached: list[Release] | None = None
        self._cached_at: float | None = None
        self._checked_at: datetime | None = None
        # Monotonic, like `_cached_at` and for the same reason: this one
        # gates a retry rather than being shown to anybody.
        self._failed_at: float | None = None
        self._last_failure: FetchFailure | None = None

    @property
    def last_failure(self) -> FetchFailure | None:
        """Why the most recent check produced no answer; `None` once one does.

        An empty `list_releases()` alone cannot say whether upstream was
        unreachable or answered with nothing usable. This is the half that
        says which, and why.
        """
        return self._last_failure

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

        **Both outcomes are stamped.** Until R1.5 only success was, so
        `GET /v1/engines` — polled every 15 s by Home and every 30 s per
        node by the Issues badge — paid the full network timeout on every
        request for as long as upstream was unreachable. `force` skips
        both stamps, because it comes from an operator pressing a button
        and an operator is allowed to ask again.
        """
        now = time.perf_counter()
        if (
            not force
            and self._cached is not None
            and self._cached_at is not None
            and now - self._cached_at < self._cache_seconds
        ):
            return self._cached
        if (
            not force
            and self._failed_at is not None
            and now - self._failed_at < self._failure_backoff_seconds
        ):
            return self._cached or []

        # GitHub's largest page. llama.cpp publishes about twenty builds a
        # day (measured 2026-10-03: the newest hundred spanned 114.9 hours),
        # so this is about 4.8 days -- and a pinned install, the rollback
        # for a regression, can only name a build in this list. At 30 it
        # was about 34 hours.
        url = f"https://api.github.com/repos/{self._repo}/releases?per_page=100"
        try:
            raw = self._fetch(url)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
            self._failed_at = now
            failure = describe_fetch_failure(e, url=url, timeout=_METADATA_TIMEOUT_SECONDS)
            # A WARNING, because a person may be looking at the result — but
            # once per distinct cause, not every five minutes all day.
            repeated = self._last_failure is not None and self._last_failure.cause == failure.cause
            log.log(
                logging.DEBUG if repeated else logging.WARNING,
                "could not list releases for %s: %s [%s]; asking again in %.0fs",
                self._repo,
                failure.sentence(),
                failure.detail,
                self._failure_backoff_seconds,
            )
            self._last_failure = failure
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

        if self._last_failure is not None:
            log.info("listed releases for %s again after: %s", self._repo, self._last_failure.cause)
        self._cached = releases
        self._cached_at = now
        self._failed_at = None
        self._last_failure = None
        self._checked_at = datetime.now(UTC)
        return releases

    @staticmethod
    def _fetch(url: str) -> object:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "eugene-plexus-agent",
            },
        )
        with urllib.request.urlopen(
            request, timeout=_METADATA_TIMEOUT_SECONDS, context=egress_ssl_context()
        ) as response:
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
        plugins = [a for a in archives if a.name in plan.plugins]
        runtimes: list[Path] = []
        for archive in archives:
            if archive in plugins:
                continue
            progress.message = f"extracting {archive.name}"
            if archive.name in plan.runtimes:
                # Unpacked on its own, and moved beside the server once
                # the server has been found (below).
                side = staging / f"{_RUNTIME_SIDE}{len(runtimes)}"
                _extract(archive, side)
                runtimes.append(side)
            else:
                _extract(archive, staging)
            # The archive itself is dead weight once unpacked, and these are
            # hundreds of megabytes.
            archive.unlink(missing_ok=True)
        for index, archive in enumerate(plugins):
            progress.message = f"adding the backend in {archive.name}"
            side = staging / f"{_PLUGIN_SIDE}{index}"
            _extract(archive, side)
            archive.unlink(missing_ok=True)
            _merge_backend(side, staging, archive.name)
            _remove_quietly(side)

        binary = _find_binary(staging, plan.binary_name)
        if binary is None:
            raise AcquisitionError(
                f"{plan.binary_name!r} not found in the extracted {plan.variant} archive — "
                f"upstream may have changed its layout"
            )
        _make_executable(binary)
        for side in runtimes:
            progress.message = f"putting the libraries in {side.name} beside {binary.name}"
            _place_runtime(side, binary.parent)
            _remove_quietly(side)

        final = self._store.build_dir(plan.version)
        _remove_quietly(final)
        # The rename is the commit point: until it lands, nothing outside
        # this function can see a half-installed build.
        staging.replace(final)
        binary = final / binary.relative_to(staging)
        self._store.write_metadata(final, version=plan.version, variant=plan.variant, binary=binary)
        self._store.prune()


def _download(asset: ReleaseAsset, target: Path, progress: _Progress) -> None:
    request = urllib.request.Request(asset.url, headers={"User-Agent": "eugene-plexus-agent"})
    try:
        with (
            urllib.request.urlopen(
                request, timeout=_DOWNLOAD_TIMEOUT_SECONDS, context=egress_ssl_context()
            ) as response,
            target.open("wb") as out,
        ):
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                progress.bytes_downloaded += len(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        failure = describe_fetch_failure(e, url=asset.url, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
        log.warning(
            "downloading %s failed: %s [%s]", asset.name, failure.sentence(), failure.detail
        )
        raise AcquisitionError(f"downloading {asset.name} failed: {failure.sentence()}") from e


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


def _merge_backend(side: Path, build: Path, archive_name: str) -> None:
    """Add the files of `side` that `build` lacks; refuse if a shared one differs.

    Upstream's archives for one release are built from one commit, and
    on Windows every file the CUDA and Vulkan builds both carry is
    byte-identical (measured on b11211: the only difference is
    `ggml-vulkan.dll`). That identity is what makes adding one backend
    to the other safe, so it is checked here rather than assumed. A
    release that breaks it is refused with the file named.
    """
    side_root = _single_root(side)
    build_root = _single_root(build)
    added: list[str] = []
    for source in sorted(p for p in side_root.rglob("*") if p.is_file()):
        relative = source.relative_to(side_root)
        target = build_root / relative
        if target.exists():
            if _sha256(source) != _sha256(target):
                raise AcquisitionError(
                    f"{archive_name} carries a {relative} that differs from the one in the build "
                    "it would be added to, so the two were not built together and cannot be "
                    "combined. Install the plain build instead."
                )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        added.append(str(relative))
    if not added:
        raise AcquisitionError(f"{archive_name} added nothing to the build; nothing to combine")
    log.info("added %s from %s", ", ".join(added), archive_name)


def _place_runtime(side: Path, server_dir: Path) -> None:
    """Put every file of an unpacked runtime archive in the server's folder.

    `_single_root` takes off the one folder the Linux cudart tarball wraps
    its libraries in; the Windows zip has none, so its files are taken as
    they are. Relative paths below that root are kept. A file the build
    already has is replaced, which is what unpacking the runtime over the
    build did before this existed -- the runtime archive is the CUDA
    runtime this build was compiled against, and upstream ships it for
    exactly that.
    """
    root = _single_root(side)
    placed: list[str] = []
    for source in sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink()):
        relative = source.relative_to(root)
        target = server_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(source, target)
        placed.append(str(relative))
    if not placed:
        raise AcquisitionError(f"{side.name}: the CUDA runtime archive was empty")
    log.info("put %s beside the server in %s", ", ".join(placed), server_dir)


# Where an archive that is not unpacked over the build waits, inside the
# staging directory, until its files are moved. Dot-prefixed so nothing in
# an upstream archive can collide with it, and ignored by `_single_root`.
_PLUGIN_SIDE = ".plugin-"
_RUNTIME_SIDE = ".runtime-"


def _single_root(directory: Path) -> Path:
    """The directory itself, or its one subdirectory when an archive wraps
    everything in one (the Linux tarballs do: `llama-b11211/`)."""
    entries = [
        p for p in directory.iterdir() if not p.name.startswith((_PLUGIN_SIDE, _RUNTIME_SIDE))
    ]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return directory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    override = os.environ.get("EUGENE_PLEXUS_AGENT_ENGINE_ROOT")
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
