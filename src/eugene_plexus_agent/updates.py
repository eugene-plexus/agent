"""Is this install behind its channel? Checked here, on every node.

**Two channels** (Troy, 2026-09-27):

- `edge`: the head of `main`, **gated**. The newest `main` commit on which
  every workflow that ran for it succeeded, CI among them. The container
  workflow runs only when the image's inputs change, so a commit it did
  not run for changed nothing in the image. This is also the commit the
  `:edge` container was built from, so a native install is never offered
  an update the container was not given.
- `releases`: the newest published release (prereleases included, since
  nothing here is released as stable yet), read from its own
  `manifest.json`, whose installer checksums the download is checked
  against.

**What is compared** is the six commits the installed code carries
(`install_info`) against the six the channel's installer pins. A
component installed before commits were recorded counts as behind. A
development checkout is never offered an update.

**The channel** is `updateChannel` on this agent's config when set.
Otherwise it is inferred:
- `releases` when the installed commits are exactly one of the recent
  releases;
- for a container, whatever its image tag says;
- `edge` otherwise.

Everything reaches GitHub through `describe_fetch_failure`, so a failed
check says what happened -- rate limit, certificate, timeout -- in the
words the engine release list already uses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ._generated.models import (
    InstalledComponentState,
    NodeInstall,
    NodeUpdate,
    UpdateApply,
    UpdateChannel,
    UpdateChannelSource,
    UpdateRun,
    UpdateTarget,
)
from ._http import egress_ssl_context
from .engines.acquisition import describe_fetch_failure

log = logging.getLogger(__name__)

REPO = "eugene-plexus/specs"
API = f"https://api.github.com/repos/{REPO}"
RAW = f"https://raw.githubusercontent.com/{REPO}"

#: How often each node looks, and how soon it tries again after a failure.
CHECK_EVERY_SECONDS = 6 * 3600
RETRY_AFTER_FAILURE_SECONDS = 30 * 60
#: Before the first check: long enough that an agent that is starting a
#: dozen things does not add a GitHub round trip to its boot.
FIRST_CHECK_DELAY_SECONDS = 60.0

_TIMEOUT_SECONDS = 10.0
_MAX_BYTES = 4 * 1024 * 1024
_PIN = re.compile(r"^PIN_(AGENT|CONTROL|GATEWAY|DRIVER|LIBRARY|UI)=([0-9a-f]{40})\b", re.MULTILINE)
_PIN_NAMES = {
    "AGENT": "agent",
    "CONTROL": "control",
    "GATEWAY": "gateway",
    "DRIVER": "inference-driver",
    "LIBRARY": "library",
    "UI": "ui",
}
COMPONENT_NAMES = tuple(_PIN_NAMES.values())
_TAG = re.compile(r"^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?$")
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class CheckFailed(Exception):
    """Why a check could not finish, in the sentence a person reads."""


@dataclass(frozen=True)
class Installer:
    """One installer script for a target: where it is and what it must hash to."""

    url: str
    sha256: str | None = None


@dataclass(frozen=True)
class Target:
    """What a channel's newest is, and how to install it."""

    channel: UpdateChannel
    ref: str
    components: dict[str, str]
    specs_commit: str | None = None
    release: str | None = None
    published_at: datetime | None = None
    installers: dict[str, Installer] = field(default_factory=dict)

    def model(self) -> UpdateTarget:
        return UpdateTarget(
            channel=self.channel,
            ref=self.ref,
            release=self.release,
            specsCommit=self.specs_commit,
            publishedAt=self.published_at,
            components=dict(self.components),
        )


def valid_ref(ref: str) -> bool:
    """A specs commit or a release tag, and nothing else can be installed."""
    return bool(_FULL_COMMIT.match(ref) or _TAG.match(ref))


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #

Fetch = Callable[[str], bytes]


def fetch(url: str) -> bytes:
    """GET a URL through the user's proxy and the OS certificate store."""
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "eugene-plexus-agent"},
    )
    with urllib.request.urlopen(
        request, timeout=_TIMEOUT_SECONDS, context=egress_ssl_context()
    ) as response:
        body: bytes = response.read(_MAX_BYTES + 1)
    if len(body) > _MAX_BYTES:
        raise ValueError(f"{url} answered with more than {_MAX_BYTES} bytes")
    return body


def _get_json(get: Fetch, url: str) -> Any:
    try:
        return json.loads(get(url).decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise CheckFailed(
            describe_fetch_failure(exc, url=url, timeout=_TIMEOUT_SECONDS).sentence()
        ) from exc


def _get_text(get: Fetch, url: str) -> str:
    try:
        return get(url).decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise CheckFailed(
            describe_fetch_failure(exc, url=url, timeout=_TIMEOUT_SECONDS).sentence()
        ) from exc


def parse_pins(install_sh: str) -> dict[str, str]:
    """The six pins an `install.sh` carries, by component name."""
    found = {_PIN_NAMES[m.group(1)]: m.group(2) for m in _PIN.finditer(install_sh)}
    missing = [name for name in COMPONENT_NAMES if name not in found]
    if missing:
        raise CheckFailed(
            f"the installer on the channel pins no commit for {', '.join(missing)}, "
            "so there is nothing to compare this install with"
        )
    return found


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# The two channels
# --------------------------------------------------------------------------- #


def newest_edge(get: Fetch = fetch) -> Target:
    """The newest `main` commit every workflow that ran for succeeded on."""
    url = f"{API}/actions/runs?branch=main&event=push&per_page=60"
    body = _get_json(get, url)
    runs = body.get("workflow_runs") if isinstance(body, dict) else None
    if not isinstance(runs, list):
        raise CheckFailed(f"{url} did not list any workflow runs")
    order: list[str] = []
    by_commit: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        sha = run.get("head_sha") if isinstance(run, dict) else None
        if not isinstance(sha, str) or not _FULL_COMMIT.match(sha):
            continue
        if sha not in by_commit:
            order.append(sha)
            by_commit[sha] = []
        by_commit[sha].append(run)
    for sha in order:
        commit_runs = by_commit[sha]
        # A run concludes `success` only once it has completed, so a run
        # still going is not a pass either.
        passed = all(r.get("conclusion") == "success" for r in commit_runs)
        has_ci = any(r.get("name") == "CI" for r in commit_runs)
        if not (passed and has_ci):
            continue
        published = _timestamp((commit_runs[0].get("head_commit") or {}).get("timestamp"))
        pins = parse_pins(_get_text(get, f"{RAW}/{sha}/scripts/install.sh"))
        return Target(
            channel=UpdateChannel.edge,
            ref=sha,
            specs_commit=sha,
            published_at=published,
            components=pins,
            installers={
                "install.sh": Installer(f"{RAW}/{sha}/scripts/install.sh"),
                "install.ps1": Installer(f"{RAW}/{sha}/scripts/install.ps1"),
            },
        )
    raise CheckFailed(
        f"none of the last {len(order)} commits on main has passed every check yet, "
        "so nothing is offered until one has"
    )


def recent_releases(get: Fetch = fetch, *, limit: int = 5) -> list[Target]:
    """The newest releases, newest first, each read from its own manifest."""
    url = f"{API}/releases?per_page=20"
    body = _get_json(get, url)
    if not isinstance(body, list):
        raise CheckFailed(f"{url} did not list releases")
    published = [
        r
        for r in body
        if isinstance(r, dict)
        and not r.get("draft")
        and isinstance(r.get("tag_name"), str)
        and _TAG.match(r["tag_name"])
    ]
    published.sort(key=lambda r: str(r.get("published_at") or ""), reverse=True)
    targets: list[Target] = []
    for release in published[:limit]:
        assets = {
            a.get("name"): a.get("browser_download_url")
            for a in release.get("assets") or []
            if isinstance(a, dict)
        }
        manifest_url = assets.get("manifest.json")
        if not isinstance(manifest_url, str):
            continue
        manifest = _get_json(get, manifest_url)
        components = manifest.get("components") if isinstance(manifest, dict) else None
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(components, dict) or not all(
            isinstance(components.get(n), str) for n in COMPONENT_NAMES
        ):
            continue
        installers: dict[str, Installer] = {}
        for name in ("install.sh", "install.ps1"):
            asset = assets.get(name)
            digest = (files or {}).get(name, {}).get("sha256") if isinstance(files, dict) else None
            if isinstance(asset, str) and isinstance(digest, str):
                installers[name] = Installer(asset, digest)
        tag = release["tag_name"]
        specs_commit = manifest.get("specsCommit")
        targets.append(
            Target(
                channel=UpdateChannel.releases,
                ref=tag,
                release=tag,
                specs_commit=specs_commit if isinstance(specs_commit, str) else None,
                published_at=_timestamp(release.get("published_at")),
                components={n: components[n] for n in COMPONENT_NAMES},
                installers=installers,
            )
        )
    return targets


def newest_release(get: Fetch = fetch) -> Target:
    found = recent_releases(get, limit=1)
    if not found:
        raise CheckFailed(f"{REPO} has no published release with a manifest to install from")
    return found[0]


# --------------------------------------------------------------------------- #
# Comparing
# --------------------------------------------------------------------------- #


def installed_commits(install: NodeInstall) -> dict[str, str | None]:
    return {
        c.name.value: (c.commit if c.state is InstalledComponentState.stamped else None)
        for c in install.components
    }


def behind(install: NodeInstall, target: Target) -> list[str]:
    """Components whose installed commit is not the target's pin.

    A component with no recorded commit is behind: it can only have been
    installed before commits were recorded, which is older than anything
    a channel offers now.
    """
    have = installed_commits(install)
    return [name for name in COMPONENT_NAMES if have.get(name) != target.components.get(name)]


def infer_channel(
    install: NodeInstall, releases: list[Target]
) -> tuple[UpdateChannel, UpdateChannelSource]:
    """The channel an unset `updateChannel` means for this install."""
    if install.container is not None:
        tag = install.container.image.rsplit(":", 1)[-1] if ":" in install.container.image else ""
        return (
            (UpdateChannel.releases if _TAG.match(tag) else UpdateChannel.edge),
            UpdateChannelSource.inferred,
        )
    have = installed_commits(install)
    for release in releases:
        if all(have.get(name) == release.components.get(name) for name in COMPONENT_NAMES):
            return UpdateChannel.releases, UpdateChannelSource.inferred
    return UpdateChannel.edge, UpdateChannelSource.inferred


# --------------------------------------------------------------------------- #
# The checker a node runs
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    channel: UpdateChannel
    source: UpdateChannelSource
    checked_at: datetime | None = None
    newest: Target | None = None
    error: str | None = None


class UpdateChecker:
    """Holds the last check, and runs the next one.

    One per agent. `view` never touches the network: `GET /v1/node` is
    polled by every console, and a round trip to GitHub on each poll would
    spend a network address's sixty requests an hour in minutes.
    """

    def __init__(
        self,
        *,
        setting: Callable[[str], Any],
        get: Fetch = fetch,
    ) -> None:
        self._setting = setting
        self._get = get
        self._result: CheckResult | None = None
        self._lock = asyncio.Lock()
        self._last_attempt: float | None = None

    @property
    def enabled(self) -> bool:
        value = self._setting("updateChecks")
        return value is not False

    def configured_channel(self) -> UpdateChannel | None:
        value = self._setting("updateChannel")
        try:
            return UpdateChannel(value) if value else None
        except ValueError:
            return None

    def _check_now(self, install: NodeInstall) -> CheckResult:
        chosen = self.configured_channel()
        releases: list[Target] = []
        if chosen is None:
            try:
                releases = recent_releases(self._get)
            except CheckFailed:
                # Inference falls back to edge; the check below reports
                # whatever it cannot reach itself.
                releases = []
            channel, source = infer_channel(install, releases)
        else:
            channel, source = chosen, UpdateChannelSource.setting
        result = CheckResult(channel=channel, source=source, checked_at=datetime.now(UTC))
        try:
            if channel is UpdateChannel.edge:
                result.newest = newest_edge(self._get)
            else:
                result.newest = releases[0] if releases else newest_release(self._get)
        except CheckFailed as exc:
            result.error = str(exc)
            previous = self._result
            if previous is not None and previous.channel is channel:
                # What the last good check found still stands.
                result.newest = previous.newest
        return result

    async def check(self, install: NodeInstall) -> CheckResult:
        async with self._lock:
            self._last_attempt = time.perf_counter()
            result = await asyncio.to_thread(self._check_now, install)
            if result.error:
                log.warning("update check (%s): %s", result.channel.value, result.error)
            elif result.newest is not None:
                gone = behind(install, result.newest)
                if gone and not install.development:
                    log.info(
                        "an update is available on %s (%s): %s",
                        result.channel.value,
                        result.newest.ref[:12],
                        ", ".join(gone),
                    )
            self._result = result
            return result

    def result(self) -> CheckResult | None:
        return self._result

    def view(
        self,
        install: NodeInstall,
        *,
        apply: UpdateApply,
        running: UpdateRun | None = None,
        last: UpdateRun | None = None,
    ) -> NodeUpdate:
        result = self._result
        if result is None:
            chosen = self.configured_channel()
            channel = chosen or infer_channel(install, [])[0]
            source = UpdateChannelSource.setting if chosen else UpdateChannelSource.inferred
            return NodeUpdate(
                enabled=self.enabled,
                channel=channel,
                channelSource=source,
                available=False,
                behind=[],
                apply=apply,
                running=running,
                last=last,
            )
        stale = behind(install, result.newest) if result.newest is not None else []
        return NodeUpdate(
            enabled=self.enabled,
            channel=result.channel,
            channelSource=result.source,
            checkedAt=result.checked_at,
            error=result.error,
            newest=result.newest.model() if result.newest is not None else None,
            available=bool(stale) and not install.development,
            behind=stale if not install.development else [],
            apply=apply,
            running=running,
            last=last,
        )

    async def run_forever(self, install: Callable[[], NodeInstall]) -> None:
        """At start, then every six hours; sooner again after a failure."""
        await asyncio.sleep(FIRST_CHECK_DELAY_SECONDS)
        while True:
            wait = CHECK_EVERY_SECONDS
            if self.enabled:
                try:
                    result = await self.check(install())
                    if result.error:
                        wait = RETRY_AFTER_FAILURE_SECONDS
                except Exception:  # pragma: no cover - a check must never kill the loop
                    log.exception("update check failed unexpectedly")
                    wait = RETRY_AFTER_FAILURE_SECONDS
            await asyncio.sleep(wait)
