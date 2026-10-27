"""Is this install behind its channel? Checked here, on every node.

**Two channels** (Troy, 2026-09-27):

- `edge`: the explicit published `edge` marker, advanced by the specs
    publisher after its named CI and image gates pass.
- `releases`: the newest published release (prereleases included, since
  nothing here is released as stable yet), read from its own
  `manifest.json`, whose installer checksums the download is checked
  against.

**What is compared** is the seven commits the installed code carries
(`install_info`) against the seven the channel's installer pins. A
component installed before commits were recorded counts as behind. A
development checkout is never offered an update.

**Only a newer version is offered** (2026-09-30). A commit that differs is
placed by its date: the channel's is older (`ahead`) or newer (`behind`),
and an update is offered only when something is behind and nothing is
ahead. Before this, "different" was offered as "newer", so a machine
installed from `main` that followed `releases` -- or one installed from a
commit whose checks were still running -- was told an older build was an
update.

**The version a person calls it** (row 75) is matched from the same seven
commits against every published release's manifest and the Edge marker, by
`VersionResolver`, with the check and never on the node report's path. It
is kept on disk, keyed by the commits, and never a guess.

**The channel** is `updateChannel` on this agent's config, and when it is
not saved, the default: `releases`, or what the environment names
(`EUGENE_PLEXUS_AGENT_DEFAULT_UPDATE_CHANNEL`, `edge` in the `:edge`
container image).

Everything reaches GitHub through `describe_fetch_failure`, so a failed
check says what happened -- rate limit, certificate, timeout -- in the
words the engine release list already uses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._generated.models import (
    InstalledComponentState,
    NodeInstall,
    NodeUpdate,
    NodeVersion,
    NodeVersionKind,
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
_PIN = re.compile(
    r"^PIN_(AGENT|CONTROL|GATEWAY|DRIVER|LIBRARY|STORE|TOOL_DRIVER|UI)=([0-9a-f]{40})\b",
    re.MULTILINE,
)
_PIN_NAMES = {
    "AGENT": "agent",
    "CONTROL": "control",
    "GATEWAY": "gateway",
    "DRIVER": "inference-driver",
    "LIBRARY": "library",
    "STORE": "store",
    "TOOL_DRIVER": "tool-driver",
    "UI": "ui",
}
COMPONENT_NAMES = tuple(_PIN_NAMES.values())
#: Components an installer or a release from before they existed does not
#: pin. **Optional in a target, never in an install**: every release up to
#: v0.1.0-alpha.5 predates the tool-driver (P8), and every one up to v0.3
#: the store (row 53); requiring their pins would make every one of them
#: unreadable -- no release channel at all -- while an install that lacks
#: one is simply behind a target that has it.
ADDED_LATER = frozenset({"tool-driver", "store"})
REQUIRED_NAMES = tuple(n for n in COMPONENT_NAMES if n not in ADDED_LATER)
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
    """The pins an `install.sh` carries, by component name."""
    found = {_PIN_NAMES[m.group(1)]: m.group(2) for m in _PIN.finditer(install_sh)}
    missing = [name for name in REQUIRED_NAMES if name not in found]
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


EDGE_MARKER = f"{API}/commits/edge"


def newest_edge(get: Fetch = fetch) -> Target:
    """The commit explicitly named by the published `edge` marker."""
    try:
        body = _get_json(get, EDGE_MARKER)
    except CheckFailed as exc:
        raise CheckFailed("the published edge marker could not be read: " + str(exc)) from exc
    sha = body.get("sha") if isinstance(body, dict) else None
    commit = body.get("commit") if isinstance(body, dict) else None
    committer = commit.get("committer") if isinstance(commit, dict) else None
    published = _timestamp(committer.get("date") if isinstance(committer, dict) else None)
    if not isinstance(sha, str) or not _FULL_COMMIT.fullmatch(sha) or published is None:
        raise CheckFailed(f"{EDGE_MARKER} did not name a valid published edge commit")
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


def recent_releases(
    get: Fetch = fetch, *, limit: int = 5, known: dict[str, Target] | None = None
) -> list[Target]:
    """The newest releases, newest first, each read from its own manifest.

    `known` holds releases already read, by tag. A published release never
    changes, so a caller that asks again (the version lookup) reads only the
    list, and a manifest only the first time it sees one.
    """
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
        if known is not None and release["tag_name"] in known:
            targets.append(known[release["tag_name"]])
            continue
        manifest = _get_json(get, manifest_url)
        components = manifest.get("components") if isinstance(manifest, dict) else None
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(components, dict) or not all(
            isinstance(components.get(n), str) for n in REQUIRED_NAMES
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
        target = Target(
            channel=UpdateChannel.releases,
            ref=tag,
            release=tag,
            specs_commit=specs_commit if isinstance(specs_commit, str) else None,
            published_at=_timestamp(release.get("published_at")),
            components={
                n: components[n] for n in COMPONENT_NAMES if isinstance(components.get(n), str)
            },
            installers=installers,
        )
        if known is not None:
            known[tag] = target
        targets.append(target)
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


#: When each component commit was made, by (component, commit). A commit
#: never changes, so this is kept for the life of the process: the installed
#: commits are looked up once, and each new target's once.
_COMMIT_DATES: dict[tuple[str, str], datetime] = {}


def commit_date(get: Fetch, name: str, commit: str) -> datetime:
    """When `commit` of component `name` was made, from GitHub's git data.

    The git-data endpoint rather than `compare`: it answers with the
    commit object alone, where a comparison carries every changed file and
    grows with the distance between the two -- weeks of `main` against a
    release is megabytes.
    """
    key = (name, commit)
    found = _COMMIT_DATES.get(key)
    if found is not None:
        return found
    url = f"https://api.github.com/repos/eugene-plexus/{name}/git/commits/{commit}"
    body = _get_json(get, url)
    committer = body.get("committer") if isinstance(body, dict) else None
    when = _timestamp(committer.get("date") if isinstance(committer, dict) else None)
    if when is None:
        raise CheckFailed(
            f"{url} did not say when {name} {commit[:12]} was made, so whether the "
            "channel's version is newer than this one cannot be told"
        )
    _COMMIT_DATES[key] = when
    return when


def _cached_only(url: str) -> bytes:
    raise CheckFailed(f"{url} has not been read yet")


@dataclass(frozen=True)
class Placement:
    """Where this install stands against a target, component by component."""

    #: Older than the target's pin, missing, or installed before commits
    #: were recorded (which is older than anything a channel offers now).
    behind: list[str]
    #: Newer than the target's pin, or a part the target predates.
    ahead: list[str]

    @property
    def newer(self) -> bool:
        """The target is newer: something moves forward and nothing back."""
        return bool(self.behind) and not self.ahead


def place(install: NodeInstall, target: Target, get: Fetch = fetch) -> Placement:
    """Place each component whose commit differs from the target's pin.

    **Newer is a later commit date, never merely a different commit**
    (2026-09-30). The comparison this replaced was `!=`, so a machine that
    ran a newer build than its channel's newest was offered the older one
    as "A newer version is ready".

    A commit made in the same second as the target's, but a different one,
    is counted as ahead: it cannot be shown to be older, and an update must
    never move a part back.
    """
    have = installed_commits(install)
    behind: list[str] = []
    ahead: list[str] = []
    for name in COMPONENT_NAMES:
        mine = have.get(name)
        pinned = target.components.get(name)
        if pinned is None:
            # A target that predates the part entirely: every release up
            # to alpha.5 predates the tool-driver. Installed, it is newer.
            if mine is not None:
                ahead.append(name)
            continue
        if mine == pinned:
            continue
        if mine is None:
            behind.append(name)
            continue
        if commit_date(get, name, pinned) > commit_date(get, name, mine):
            behind.append(name)
        else:
            ahead.append(name)
    return Placement(behind=behind, ahead=ahead)


# --------------------------------------------------------------------------- #
# The version a person calls it (row 75)
# --------------------------------------------------------------------------- #

VERSION_FILE = "version.json"
#: How many releases a lookup reads. GitHub lists twenty on its first page.
VERSION_RELEASES = 20


def same_build(commits: dict[str, str | None], pins: dict[str, str]) -> bool:
    """Every component is the commit the target pins, and none is a stray.

    A part the target does not pin (the store, before row 53) must not be
    installed: an install that has it is newer than that release, never it.
    """
    return all(commits.get(name) == pins.get(name) for name in COMPONENT_NAMES)


@dataclass(frozen=True)
class _Matched:
    commits: dict[str, str | None]
    version: NodeVersion


class VersionResolver:
    """Names the version this install runs, from its seven commits.

    **A version is named only when every commit equals the one a published
    release pins, or the Edge marker pins.** Anything else is `unmatched`
    ("not a published release"), and the commits stay on `NodeInstall`.

    **Never on the node report's path.** `view` reads memory; `resolve` is
    the only code that reaches GitHub, and runs with the update check, in a
    thread. Until the first match the answer is `unknown`, and a failed
    lookup says what it was reading.

    **Kept on disk**, keyed by the commits it was made for, so it shows
    offline and after a restart. It is used only while the installed commits
    are the same ones: after an update the answer is `unknown` until the new
    commits are matched. A match to a release or Edge is final for those
    commits; `unmatched` is tried again at every check, since a release or an
    Edge build can be published later from the same commits.
    """

    def __init__(self, file: Path | None = None) -> None:
        self._file = file
        self._lock = threading.Lock()
        self._loaded = file is None
        self._matched: _Matched | None = None
        #: Why the last lookup failed, and for which commits.
        self._error: tuple[dict[str, str | None], str] | None = None
        self._releases: dict[str, Target] = {}

    # -- disk ------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        assert self._file is not None
        try:
            body = json.loads(self._file.read_text(encoding="utf-8"))
            commits = body["commits"]
            version = NodeVersion.model_validate(body["version"])
            if isinstance(commits, dict) and all(
                isinstance(k, str) and (v is None or isinstance(v, str)) for k, v in commits.items()
            ):
                self._matched = _Matched(commits, version)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("the kept version in %s could not be read: %s", self._file, exc)

    def _save(self, matched: _Matched) -> None:
        if self._file is None:
            return
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            body = {
                "commits": matched.commits,
                "version": json.loads(matched.version.model_dump_json(exclude_none=True)),
            }
            scratch = self._file.with_name(self._file.name + ".tmp")
            scratch.write_text(json.dumps(body, indent=2), encoding="utf-8", newline="\n")
            scratch.replace(self._file)
        except OSError as exc:
            log.warning("the version could not be kept in %s: %s", self._file, exc)

    # -- what a console reads --------------------------------------------

    def view(self, install: NodeInstall) -> NodeVersion:
        """No network: what is known for these commits, else `unknown`."""
        if install.development:
            return NodeVersion(kind=NodeVersionKind.development)
        commits = installed_commits(install)
        with self._lock:
            self._load()
            matched, failed = self._matched, self._error
        error = failed[1] if failed is not None and failed[0] == commits else None
        if matched is not None and matched.commits == commits:
            if error and matched.version.kind is NodeVersionKind.unmatched:
                return matched.version.model_copy(update={"error": error})
            return matched.version
        return NodeVersion(kind=NodeVersionKind.unknown, error=error)

    # -- the lookup ------------------------------------------------------

    def resolve(self, install: NodeInstall, get: Fetch = fetch) -> None:
        """Match the install's commits to a release or Edge. Blocking."""
        if install.development:
            return
        commits = installed_commits(install)
        with self._lock:
            self._load()
            current = self._matched
        if (
            current is not None
            and current.commits == commits
            and current.version.kind in (NodeVersionKind.release, NodeVersionKind.edge)
        ):
            return
        now = datetime.now(UTC)
        failures: list[str] = []
        found: NodeVersion | None = None
        try:
            for target in recent_releases(get, limit=VERSION_RELEASES, known=self._releases):
                if same_build(commits, target.components):
                    found = NodeVersion(
                        kind=NodeVersionKind.release,
                        release=target.release,
                        publishedAt=target.published_at,
                        checkedAt=now,
                    )
                    break
        except CheckFailed as exc:
            failures.append(f"The published releases could not be read: {exc}")
        if found is None:
            try:
                edge = newest_edge(get)
                if same_build(commits, edge.components):
                    found = NodeVersion(
                        kind=NodeVersionKind.edge,
                        publishedAt=edge.published_at,
                        specsCommit=edge.specs_commit,
                        checkedAt=now,
                    )
            except CheckFailed as exc:
                failures.append(str(exc))
        if found is None and not failures:
            found = NodeVersion(kind=NodeVersionKind.unmatched, checkedAt=now)
        with self._lock:
            if found is not None:
                self._matched = _Matched(commits, found)
                self._error = None
            else:
                # Not everything could be read: say so, and keep what an
                # earlier lookup found for these same commits.
                self._error = (commits, " ".join(failures))
        if found is not None:
            self._save(_Matched(commits, found))
            log.info("this install is %s", found.release or found.kind.value)
        else:
            log.warning("the version could not be matched: %s", " ".join(failures))


# --------------------------------------------------------------------------- #
# The checker a node runs
# --------------------------------------------------------------------------- #

#: While checks are off, how often the loop looks at the setting again, so
#: turning them back on checks within a minute rather than up to six hours.
RECHECK_SETTING_SECONDS = 60.0


@dataclass
class CheckResult:
    channel: UpdateChannel
    source: UpdateChannelSource
    checked_at: datetime | None = None
    newest: Target | None = None
    placement: Placement | None = None
    #: The install the placement was made for.
    commits: dict[str, str | None] = field(default_factory=dict)
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
        default_channel: Callable[[], UpdateChannel] = lambda: UpdateChannel.releases,
        version_file: Path | None = None,
    ) -> None:
        self._setting = setting
        self._get = get
        self._versions = VersionResolver(version_file)
        self._default_channel = default_channel
        self._result: CheckResult | None = None
        self._lock = asyncio.Lock()
        self._last_attempt: float | None = None

    @property
    def enabled(self) -> bool:
        value = self._setting("updateChecks")
        return value is not False

    def configured_channel(self) -> UpdateChannel | None:
        """The channel saved on this agent's config, if one is."""
        value = self._setting("updateChannel")
        try:
            return UpdateChannel(value) if value else None
        except ValueError:
            return None

    def channel(self) -> tuple[UpdateChannel, UpdateChannelSource]:
        """The channel this machine follows now, and why. No network."""
        chosen = self.configured_channel()
        if chosen is not None:
            return chosen, UpdateChannelSource.setting
        return self._default_channel(), UpdateChannelSource.default

    def _check_now(self, install: NodeInstall) -> CheckResult:
        now = datetime.now(UTC)
        channel, source = self.channel()
        commits = installed_commits(install)
        result = CheckResult(channel=channel, source=source, checked_at=now, commits=commits)
        try:
            if channel is UpdateChannel.edge:
                newest = newest_edge(self._get)
            else:
                newest = newest_release(self._get)
            placement = (
                Placement(behind=[], ahead=[])
                if install.development
                else place(install, newest, self._get)
            )
            result.newest, result.placement = newest, placement
        except CheckFailed as exc:
            result.error = str(exc)
            previous = self._result
            if previous is not None and previous.channel is channel:
                # What the last good check found still stands.
                result.newest = previous.newest
                result.placement = previous.placement
                result.commits = previous.commits
        return result

    async def check(self, install: NodeInstall) -> CheckResult:
        async with self._lock:
            self._last_attempt = time.perf_counter()
            result = await asyncio.to_thread(self._check_now, install)
            label = result.channel.value
            if result.error:
                log.warning("update check (%s): %s", label, result.error)
            elif result.newest is not None and result.placement is not None:
                if result.placement.newer:
                    log.info(
                        "an update is available on %s (%s): %s",
                        label,
                        result.newest.ref[:12],
                        ", ".join(result.placement.behind),
                    )
                elif result.placement.ahead:
                    log.info(
                        "this machine is newer than the newest on %s (%s) in: %s",
                        label,
                        result.newest.ref[:12],
                        ", ".join(result.placement.ahead),
                    )
            self._result = result
            # The version is matched on the same visit and never lets it
            # fail: a lookup that cannot finish says so on the version.
            try:
                await asyncio.to_thread(self._versions.resolve, install, self._get)
            except Exception:  # pragma: no cover - a lookup must never kill the check
                log.exception("version lookup failed unexpectedly")
            return result

    def version(self, install: NodeInstall) -> NodeVersion:
        """What this install is called, from what is already known. No network."""
        return self._versions.view(install)

    def result(self) -> CheckResult | None:
        return self._result

    def current(self) -> CheckResult | None:
        """The last check, when it was made for the channel followed now.

        A result for another channel -- the setting changed since -- says
        nothing about this one, and installing its target would be an
        update from a channel the machine no longer follows.
        """
        result = self._result
        if result is None:
            return None
        channel, _ = self.channel()
        if result.channel is not channel:
            return None
        return result

    def view(
        self,
        install: NodeInstall,
        *,
        apply: UpdateApply,
        running: UpdateRun | None = None,
        last: UpdateRun | None = None,
    ) -> NodeUpdate:
        channel, source = self.channel()
        result = self.current()
        if result is None:
            return NodeUpdate(
                enabled=self.enabled,
                channel=channel,
                channelSource=source,
                available=False,
                behind=[],
                ahead=[],
                apply=apply,
                running=running,
                last=last,
            )
        placement = result.placement
        if placement is not None and result.commits != installed_commits(install):
            # Installed commits moved since the check (an update that did not
            # restart this agent): only dates already read can place them.
            try:
                placement = (
                    place(install, result.newest, _cached_only)
                    if result.newest is not None
                    else None
                )
            except CheckFailed:
                placement = None
        behind = placement.behind if placement is not None else []
        ahead = placement.ahead if placement is not None else []
        development = install.development
        return NodeUpdate(
            enabled=self.enabled,
            channel=result.channel,
            channelSource=result.source,
            checkedAt=result.checked_at,
            error=result.error,
            newest=result.newest.model() if result.newest is not None else None,
            available=placement is not None and placement.newer and not development,
            behind=[] if development else behind,
            ahead=[] if development else ahead,
            apply=apply,
            running=running,
            last=last,
        )

    async def run_forever(self, install: Callable[[], NodeInstall]) -> None:
        """At start, then every six hours; sooner again after a failure."""
        await asyncio.sleep(FIRST_CHECK_DELAY_SECONDS)
        while True:
            if not self.enabled:
                await asyncio.sleep(RECHECK_SETTING_SECONDS)
                continue
            wait = CHECK_EVERY_SECONDS
            try:
                result = await self.check(install())
                if result.error:
                    wait = RETRY_AFTER_FAILURE_SECONDS
            except Exception:  # pragma: no cover - a check must never kill the loop
                log.exception("update check failed unexpectedly")
                wait = RETRY_AFTER_FAILURE_SECONDS
            await self._sleep_while_enabled(wait)

    async def _sleep_while_enabled(self, seconds: float) -> None:
        """Sleep `seconds`, but return early if checks are turned off, so the
        loop's own sleep does not outlast the setting that decides it."""
        deadline = time.perf_counter() + seconds
        while self.enabled:
            left = deadline - time.perf_counter()
            if left <= 0:
                return
            await asyncio.sleep(min(left, RECHECK_SETTING_SECONDS))
