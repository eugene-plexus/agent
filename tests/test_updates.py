"""Checking for and installing a newer version (2026-09-27).

Troy: "let Eugene check for updates periodically, and apply them within
the app ... force update my Amish_Station from the NAS UI". Design:
`specs/docs/design/in-app-updates.md`.

What is pinned here:

- **The version comes from the code**, stamped by `git archive`: a stamped
  commit, a development checkout, a package installed before commits were
  recorded, and a missing one are four different answers.
- **`edge` is gated**: the newest `main` commit on which every workflow
  that ran succeeded, CI among them. A commit whose container build
  failed, or whose CI is still running, is skipped.
- **`releases`** reads each release's own manifest and carries the
  installer checksums it publishes.
- **The only thing an update can install is what this agent found.**
- **A container never updates itself**, and is told how in the words of
  the platform its template names, or in general Docker words.
- **The installer never runs as the agent's child**, and on a Linux system
  install nothing the agent's account wrote is run as root.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import install_info, update_apply, updates
from eugene_plexus_agent._generated.models import (
    ComponentName,
    ContainerInstall,
    InstalledComponent,
    InstalledComponentState,
    InstallMechanism,
    NodeInstall,
    UpdateChannel,
    UpdateChannelSource,
    UpdateOutcome,
    UpdateRun,
)

SHA = {name: f"{i}" * 40 for i, name in enumerate(updates.COMPONENT_NAMES, start=1)}
# Two hex digits repeated: one letter per component ran out of hex at the
# seventh (`g`), which is how the tool-driver's pin first read as missing.
NEW = {name: f"{0xA0 + i:02x}" * 20 for i, name in enumerate(updates.COMPONENT_NAMES)}
SPECS_OK = "c" * 40
SPECS_RED = "d" * 40
SPECS_RUNNING = "e" * 40


def _install(
    commits: dict[str, str | None] | None = None,
    *,
    mechanism: InstallMechanism = InstallMechanism.windows_service,
    container: ContainerInstall | None = None,
    development: bool = False,
) -> NodeInstall:
    commits = commits if commits is not None else SHA
    components = []
    for name in updates.COMPONENT_NAMES:
        commit = commits.get(name)
        state = (
            InstalledComponentState.development
            if development
            else InstalledComponentState.stamped
            if commit
            else InstalledComponentState.unrecorded
        )
        components.append(InstalledComponent(name=ComponentName(name), state=state, commit=commit))
    return NodeInstall(
        components=components, mechanism=mechanism, development=development, container=container
    )


def _install_sh(pins: dict[str, str]) -> str:
    # The pin lines exactly as the real installer writes them.
    names = {
        "agent": "AGENT",
        "control": "CONTROL",
        "gateway": "GATEWAY",
        "inference-driver": "DRIVER",
        "library": "LIBRARY",
        "tool-driver": "TOOL_DRIVER",
        "ui": "UI",
    }
    lines = [f"PIN_{names[n]}={pins[n]}" for n in updates.COMPONENT_NAMES if n in pins]
    lines[-1] += "   # branch `dist`, not `main`"
    return "#!/bin/sh\n" + "\n".join(lines) + "\n"


def _run(name: str, sha: str, status: str = "completed", conclusion: str | None = "success"):
    return {
        "name": name,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "head_commit": {"timestamp": "2026-09-27T15:53:54Z"},
    }


#: When the commits in `SHA` and `NEW` were made: `NEW` is the later build.
OLD_DATE = "2026-09-20T10:00:00Z"
NEW_DATE = "2026-09-28T10:00:00Z"


def _commit_url(name: str, commit: str) -> str:
    return f"https://api.github.com/repos/eugene-plexus/{name}/git/commits/{commit}"


def _dates(*sets: tuple[dict[str, str], str]) -> dict[str, Any]:
    """GitHub's git-data answers for each (commits, date) pair."""
    pages: dict[str, Any] = {}
    for commits, when in sets:
        for name, commit in commits.items():
            pages[_commit_url(name, commit)] = {"sha": commit, "committer": {"date": when}}
    return pages


@pytest.fixture(autouse=True)
def _fresh_commit_dates():
    """The module keeps each commit's date for its lifetime; a test's
    fixture dates must not answer for the next test."""
    updates._COMMIT_DATES.clear()
    yield
    updates._COMMIT_DATES.clear()


class Web:
    """GitHub, as its API and raw files answer, keyed by URL.

    Knows when every commit in `SHA` (older) and `NEW` (newer) was made,
    because an update is offered only when it is newer."""

    def __init__(self, pages: dict[str, Any]) -> None:
        self.pages = _dates((SHA, OLD_DATE), (NEW, NEW_DATE)) | pages
        self.asked: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.asked.append(url)
        if url not in self.pages:
            raise OSError(f"nothing at {url}")
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        return (
            page
            if isinstance(page, bytes)
            else (page.encode() if isinstance(page, str) else json.dumps(page).encode())
        )


def _edge_web(runs: list[dict[str, Any]], pins: dict[str, str] = NEW) -> Web:
    pages: dict[str, Any] = {
        f"{updates.API}/actions/runs?branch=main&event=push&per_page=60": {"workflow_runs": runs}
    }
    for run in runs:
        pages[f"{updates.RAW}/{run['head_sha']}/scripts/install.sh"] = _install_sh(pins)
    return Web(pages)


RELEASES_URL = f"{updates.API}/releases?per_page=20"


def _releases_web() -> Web:
    def manifest(tag: str, pins: dict[str, str]) -> dict[str, Any]:
        return {
            "version": tag,
            "specsCommit": "f" * 40,
            "components": pins,
            "files": {
                "install.sh": {"sha256": "1" * 64},
                "install.ps1": {"sha256": "2" * 64},
            },
        }

    def release(tag: str, published: str, *, draft: bool = False) -> dict[str, Any]:
        base = f"https://github.com/eugene-plexus/specs/releases/download/{tag}"
        return {
            "tag_name": tag,
            "draft": draft,
            "published_at": published,
            "assets": [
                {"name": n, "browser_download_url": f"{base}/{n}"}
                for n in ("manifest.json", "install.sh", "install.ps1")
            ],
        }

    base = "https://github.com/eugene-plexus/specs/releases/download"
    return Web(
        {
            RELEASES_URL: [
                release("v0.1.0-alpha.2", "2026-09-21T18:16:49Z"),
                release("v0.1.0-alpha.4", "2026-09-30T00:00:00Z", draft=True),
                release("v0.1.0-alpha.3", "2026-09-27T00:36:08Z"),
            ],
            f"{base}/v0.1.0-alpha.3/manifest.json": manifest("v0.1.0-alpha.3", NEW),
            f"{base}/v0.1.0-alpha.2/manifest.json": manifest("v0.1.0-alpha.2", SHA),
        }
    )


# --------------------------------------------------------------------------- #
# The version, from the code
# --------------------------------------------------------------------------- #


def _package(tmp_path: Path, name: str, build: str | None) -> None:
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    if build is not None:
        (root / "_build.py").write_text(f'COMMIT = "{build}"\n', encoding="utf-8")


def test_the_four_ways_a_package_can_be_installed(tmp_path: Path, monkeypatch) -> None:
    _package(tmp_path, "ep_stamped", "0123456789abcdef0123456789abcdef01234567")
    _package(tmp_path, "ep_checkout", "$Format:%H$")
    _package(tmp_path, "ep_before", None)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert install_info.read_component("ep_stamped") == (
        InstalledComponentState.stamped,
        "0123456789abcdef0123456789abcdef01234567",
    )
    assert install_info.read_component("ep_checkout") == (
        InstalledComponentState.development,
        None,
    )
    # Installed before commits were recorded: no _build.py at all.
    assert install_info.read_component("ep_before") == (InstalledComponentState.unrecorded, None)
    assert install_info.read_component("ep_not_installed_anywhere") == (
        InstalledComponentState.missing,
        None,
    )


def test_this_checkout_is_a_development_build() -> None:
    described = install_info.describe(mechanism_now=InstallMechanism.none)
    agent = next(c for c in described.components if c.name is ComponentName.agent)
    assert agent.state is InstalledComponentState.development
    assert described.development is True


def test_a_container_says_so_and_names_its_image(monkeypatch) -> None:
    monkeypatch.setenv(install_info.CONTAINER_IMAGE_ENV, "ghcr.io/eugene-plexus/control-plane:edge")
    monkeypatch.setenv(install_info.CONTAINER_HOST_ENV, "Unraid")
    assert install_info.mechanism() is InstallMechanism.container
    assert install_info.container() == ContainerInstall(
        image="ghcr.io/eugene-plexus/control-plane:edge", host="unraid"
    )


# --------------------------------------------------------------------------- #
# The channels
# --------------------------------------------------------------------------- #


def test_edge_is_the_newest_commit_every_check_passed_on() -> None:
    web = _edge_web(
        [
            # Newest: CI still running.
            _run("CI", SPECS_RUNNING, status="in_progress", conclusion=None),
            # Next: CI passed, the container build failed -- the edge image
            # was not given this one, so a native install is not either.
            _run("CI", SPECS_RED),
            _run("Container image", SPECS_RED, conclusion="failure"),
            # Next: everything passed.
            _run("CI", SPECS_OK),
            _run("Container image", SPECS_OK),
        ]
    )
    target = updates.newest_edge(web)
    assert target.ref == SPECS_OK and target.channel is UpdateChannel.edge
    assert target.components == NEW
    assert target.installers["install.ps1"].url.endswith(f"/{SPECS_OK}/scripts/install.ps1")
    # The pins came from the installer at that commit, not at main.
    assert f"{updates.RAW}/{SPECS_OK}/scripts/install.sh" in web.asked


def test_a_commit_the_container_did_not_run_for_still_counts() -> None:
    # The container workflow runs only when the image's inputs change, so a
    # docs commit has only CI -- and changed nothing in the image.
    target = updates.newest_edge(_edge_web([_run("CI", SPECS_OK)]))
    assert target.ref == SPECS_OK


def test_a_commit_with_no_ci_is_not_offered() -> None:
    with pytest.raises(updates.CheckFailed, match="none of the last 1 commits"):
        updates.newest_edge(_edge_web([_run("Container image", SPECS_OK)]))


def test_an_installer_that_pins_too_little_is_not_compared() -> None:
    web = _edge_web([_run("CI", SPECS_OK)])
    web.pages[f"{updates.RAW}/{SPECS_OK}/scripts/install.sh"] = "PIN_AGENT=" + "a" * 40 + "\n"
    with pytest.raises(updates.CheckFailed, match="no commit for control"):
        updates.newest_edge(web)


def test_releases_are_read_from_their_own_manifests_newest_first() -> None:
    found = updates.recent_releases(_releases_web())
    # The draft is not a release; newest published first.
    assert [t.ref for t in found] == ["v0.1.0-alpha.3", "v0.1.0-alpha.2"]
    newest = found[0]
    assert newest.channel is UpdateChannel.releases and newest.components == NEW
    assert newest.installers["install.ps1"].sha256 == "2" * 64
    assert newest.installers["install.ps1"].url.endswith("/v0.1.0-alpha.3/install.ps1")


def test_a_failed_check_says_what_happened() -> None:
    import urllib.error

    web = Web(
        {
            f"{updates.API}/actions/runs?branch=main&event=push&per_page=60": urllib.error.URLError(
                TimeoutError("timed out")
            )
        }
    )
    with pytest.raises(updates.CheckFailed) as failed:
        updates.newest_edge(web)
    # describe_fetch_failure's words, not "check network access".
    assert "api.github.com" in str(failed.value)


# --------------------------------------------------------------------------- #
# Comparing, and which channel
# --------------------------------------------------------------------------- #


def test_behind_is_every_component_older_than_the_pin() -> None:
    web = Web({})
    target = updates.Target(channel=UpdateChannel.edge, ref=SPECS_OK, components=NEW)
    same = updates.place(_install(NEW), target, web)
    assert same.behind == [] and same.ahead == [] and not same.newer
    half = dict(NEW) | {"gateway": SHA["gateway"]}
    placed = updates.place(_install(half), target, web)
    assert placed.behind == ["gateway"] and placed.ahead == [] and placed.newer
    # Installed before commits were recorded: behind, not unknown.
    assert "agent" in updates.place(_install(dict(NEW) | {"agent": None}), target, web).behind


def test_an_older_target_is_never_newer() -> None:
    """Troy, 2026-09-29: an older build must never be offered as an update.
    A machine installed from `main` that follows releases, or one installed
    from a commit whose checks are still running, runs something NEWER than
    its channel's newest -- and "different" used to be offered as "newer"."""
    older = updates.Target(channel=UpdateChannel.releases, ref="v0", components=SHA)
    placed = updates.place(_install(NEW), older, Web({}))
    assert placed.ahead == list(updates.COMPONENT_NAMES)
    assert placed.behind == [] and placed.newer is False


def test_a_mixed_install_is_not_offered_what_would_move_a_part_back() -> None:
    target = updates.Target(channel=UpdateChannel.edge, ref=SPECS_OK, components=NEW)
    mixed = dict(SHA) | {"gateway": "7" * 40}
    web = Web(_dates(({"gateway": "7" * 40}, "2026-09-30T00:00:00Z")))
    placed = updates.place(_install(mixed), target, web)
    assert placed.ahead == ["gateway"] and "agent" in placed.behind
    assert placed.newer is False


def test_a_commit_whose_date_cannot_be_read_fails_the_check() -> None:
    """Newer cannot be guessed. A lookup that fails is a check that says why."""
    target = updates.Target(channel=UpdateChannel.edge, ref=SPECS_OK, components=NEW)
    web = Web({_commit_url("gateway", NEW["gateway"]): OSError("rate limited")})
    with pytest.raises(updates.CheckFailed, match=re.escape("api.github.com")):
        updates.place(_install(dict(NEW) | {"gateway": SHA["gateway"]}), target, web)


def test_the_old_rule_is_kept_only_to_settle_a_channel_once() -> None:
    releases = updates.recent_releases(_releases_web())
    assert updates.channel_before_default(_install(SHA), releases) is UpdateChannel.releases
    unknown = _install({n: "9" * 40 for n in SHA})
    assert updates.channel_before_default(unknown, releases) is UpdateChannel.edge
    # A container's own tag says it.
    tagged = _install(
        mechanism=InstallMechanism.container,
        container=ContainerInstall(image="ghcr.io/eugene-plexus/control-plane:v0.1.0-alpha.3"),
    )
    assert updates.channel_before_default(tagged, []) is UpdateChannel.releases
    edge = _install(
        mechanism=InstallMechanism.container,
        container=ContainerInstall(image="ghcr.io/eugene-plexus/control-plane:edge"),
    )
    assert updates.channel_before_default(edge, []) is UpdateChannel.edge
    assert not hasattr(updates, "infer_channel")


async def test_an_unsaved_channel_is_the_default() -> None:
    checker = updates.UpdateChecker(setting={}.get, get=_releases_web())
    assert checker.channel() == (UpdateChannel.releases, UpdateChannelSource.default)
    result = await checker.check(_install(SHA))
    assert result.channel is UpdateChannel.releases and result.error is None
    edge = updates.UpdateChecker(setting={}.get, default_channel=lambda: UpdateChannel.edge)
    assert edge.channel() == (UpdateChannel.edge, UpdateChannelSource.default)


async def test_a_pending_install_saves_the_channel_it_followed_and_checks_it() -> None:
    saved: list[UpdateChannel] = []
    web = _releases_web()
    checker = updates.UpdateChecker(
        setting={}.get, get=web, settling=lambda: not saved, settle=saved.append
    )
    assert checker.channel() == (None, UpdateChannelSource.pending)
    # On alpha.2 exactly: it followed releases, and that is what is saved.
    result = await checker.check(_install(SHA))
    assert saved == [UpdateChannel.releases]
    assert result.channel is UpdateChannel.releases
    assert result.source is UpdateChannelSource.setting
    assert result.newest is not None and result.newest.release == "v0.1.0-alpha.3"


async def test_a_pending_install_that_cannot_read_the_releases_follows_nothing() -> None:
    """It used to fall back to edge -- which is how a release install could
    be handed an edge build whenever that one request failed."""
    saved: list[UpdateChannel] = []
    web = Web({RELEASES_URL: OSError("down")} | _edge_web([_run("CI", SPECS_OK)]).pages)
    checker = updates.UpdateChecker(
        setting={}.get, get=web, settling=lambda: not saved, settle=saved.append
    )
    result = await checker.check(_install(SHA))
    assert saved == [] and result.channel is None and result.newest is None
    assert result.error and "list of releases" in result.error
    view = checker.view(
        _install(SHA), apply=update_apply.plan(_install(SHA), None, system_unit_ready=False)
    )
    assert view.channel is None and view.channelSource is UpdateChannelSource.pending
    assert view.available is False
    assert not any("actions/runs" in url for url in web.asked)


async def test_a_result_for_another_channel_is_not_offered() -> None:
    settings: dict[str, Any] = {"updateChannel": "edge"}
    checker = updates.UpdateChecker(setting=settings.get, get=_edge_web([_run("CI", SPECS_OK)]))
    await checker.check(_install(SHA))
    install = _install(SHA)
    plan = update_apply.plan(install, None, system_unit_ready=False)
    assert checker.view(install, apply=plan).available is True
    settings["updateChannel"] = "releases"
    view = checker.view(install, apply=plan)
    assert view.available is False and view.newest is None
    assert view.channel is UpdateChannel.releases


async def test_a_check_that_fails_keeps_what_the_last_one_found() -> None:
    settings: dict[str, Any] = {"updateChannel": "edge"}
    web = _edge_web([_run("CI", SPECS_OK)])
    checker = updates.UpdateChecker(setting=settings.get, get=web)
    first = await checker.check(_install(SHA))
    assert first.newest is not None and first.error is None
    web.pages[f"{updates.API}/actions/runs?branch=main&event=push&per_page=60"] = OSError("down")
    second = await checker.check(_install(SHA))
    assert second.error and second.newest is not None and second.newest.ref == SPECS_OK


async def test_a_development_checkout_is_never_offered_an_update() -> None:
    checker = updates.UpdateChecker(
        setting={"updateChannel": "edge"}.get, get=_edge_web([_run("CI", SPECS_OK)])
    )
    install = _install(development=True)
    await checker.check(install)
    view = checker.view(install, apply=update_apply.plan(install, None, system_unit_ready=False))
    assert view.available is False and view.behind == []


# --------------------------------------------------------------------------- #
# What a person is told
# --------------------------------------------------------------------------- #


def _container(host: str | None, tag: str = "edge") -> NodeInstall:
    return _install(
        mechanism=InstallMechanism.container,
        container=ContainerInstall(image=f"ghcr.io/eugene-plexus/control-plane:{tag}", host=host),
    )


RELEASE = updates.Target(
    channel=UpdateChannel.releases, ref="v0.1.0-alpha.4", release="v0.1.0-alpha.4", components=NEW
)


def test_a_container_is_told_how_in_general_docker_words() -> None:
    applied = update_apply.plan(_container(None), None, system_unit_ready=False)
    assert applied.possible is False and "container" in (applied.reason or "")
    commands = [s.command for s in applied.steps or [] if s.command]
    assert commands == ["docker pull ghcr.io/eugene-plexus/control-plane:edge"]
    text = " ".join(s.text for s in applied.steps or [])
    assert "/data" in text and "Portainer" in text and "Unraid" not in text


def test_unraid_and_compose_get_their_own_words() -> None:
    unraid = update_apply.plan(_container("unraid"), None, system_unit_ready=False)
    assert "Force Update" in unraid.steps[0].text  # type: ignore[index]
    compose = update_apply.plan(_container("compose"), None, system_unit_ready=False)
    assert compose.steps[-1].command == "docker compose pull && docker compose up -d"  # type: ignore[index]


def test_a_release_container_is_told_to_change_its_tag() -> None:
    # Re-pulling the same release tag gets nothing new.
    general = update_apply.plan(
        _container(None, "v0.1.0-alpha.3"), RELEASE, system_unit_ready=False
    )
    assert (
        general.steps[0].command == "docker pull ghcr.io/eugene-plexus/control-plane:v0.1.0-alpha.4"
    )  # type: ignore[index]
    unraid = update_apply.plan(
        _container("unraid", "v0.1.0-alpha.3"), RELEASE, system_unit_ready=False
    )
    assert "control-plane:v0.1.0-alpha.4" in unraid.steps[0].text  # type: ignore[index]


def test_who_can_update_themselves() -> None:
    for mechanism in (
        InstallMechanism.windows_service,
        InstallMechanism.windows_task,
        InstallMechanism.systemd_user,
    ):
        assert update_apply.plan(
            _install(mechanism=mechanism), None, system_unit_ready=False
        ).possible
    system = _install(mechanism=InstallMechanism.systemd_system)
    assert update_apply.plan(system, None, system_unit_ready=True).possible
    older = update_apply.plan(system, None, system_unit_ready=False)
    assert not older.possible and "once more" in (older.reason or "")
    by_hand = update_apply.plan(
        _install(mechanism=InstallMechanism.none), None, system_unit_ready=False
    )
    assert not by_hand.possible and by_hand.steps and by_hand.steps[0].command


# --------------------------------------------------------------------------- #
# The scripts that run outside the agent
# --------------------------------------------------------------------------- #

STARTED = datetime(2026, 9, 27, 18, 0, tzinfo=UTC)


def test_the_windows_script_is_filled_in_and_quoted(tmp_path: Path) -> None:
    prefix = tmp_path / "O'Brien" / "EugenePlexus"
    script = update_apply.windows_wrapper(
        prefix=prefix, target=SPECS_OK, started=STARTED, service=False, log_path=prefix / "u.log"
    )
    assert "@@" not in script
    # A single quote in a person's name is doubled, not a syntax error.
    assert "O''Brien" in script
    assert "'-NoService', '-Update'" in script
    assert "Start-ScheduledTask -TaskName EugenePlexusAgent" in script
    service = update_apply.windows_wrapper(
        prefix=prefix, target=SPECS_OK, started=STARTED, service=True, log_path=prefix / "u.log"
    )
    assert "@('-Update')" in service and "Start-Service -Name EugenePlexusAgent" in service
    # The names the agent reads back are the ones the script writes.
    for name in (update_apply.LAST_FILE, update_apply.RUNNING_FILE, update_apply.WINDOWS_TASK):
        assert name in script


@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX shell here")
@pytest.mark.parametrize("code", [0, 3])
def test_the_linux_script_writes_a_record_the_agent_can_read(tmp_path: Path, code: int) -> None:
    prefix = tmp_path / "prefix"
    directory = update_apply.update_dir(prefix)
    directory.mkdir(parents=True)
    # An installer that says something awkward and exits with `code`.
    (directory / "install.sh").write_text(
        f"echo 'a \"quoted\" line with a \\ backslash'\nexit {code}\n", encoding="utf-8"
    )
    (directory / update_apply.RUNNING_FILE).write_text("{}", encoding="utf-8")
    script = directory / "run-update.sh"
    script.write_bytes(
        update_apply.posix_wrapper(
            prefix=prefix, target=SPECS_OK, started=STARTED, log_path=directory / "u.log"
        ).encode()
    )
    subprocess.run(["sh", _sh_path(script)], check=True, timeout=30)
    record = update_apply.last(prefix)
    assert record is not None and record.target == SPECS_OK
    assert record.outcome is (UpdateOutcome.succeeded if code == 0 else UpdateOutcome.failed)
    if code:
        assert "exited with 3" in (record.detail or "") and '"quoted"' in (record.detail or "")
    assert not (directory / update_apply.RUNNING_FILE).exists()


def _sh_path(path: Path) -> str:
    # Git Bash on Windows reads a POSIX path; everywhere else it is the path.
    if sys.platform == "win32":
        drive, rest = str(path)[0], str(path)[2:]
        return f"/{drive.lower()}{rest.replace(chr(92), '/')}"
    return str(path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell")
@pytest.mark.parametrize("code", [0, 7])
def test_the_windows_script_writes_a_record_the_agent_can_read(tmp_path: Path, code: int) -> None:
    prefix = tmp_path / "Eugene O'Brien"
    directory = update_apply.update_dir(prefix)
    directory.mkdir(parents=True)
    (directory / "install.ps1").write_text(
        f"Write-Host 'a \"quoted\" line'\nexit {code}\n", encoding="utf-8"
    )
    (directory / update_apply.RUNNING_FILE).write_text("{}", encoding="utf-8")
    script = update_apply.windows_wrapper(
        prefix=prefix, target=SPECS_OK, started=STARTED, service=True, log_path=directory / "u.log"
    )
    # Never restart a real service from a test on a machine that may run one.
    script = script.replace("Start-Service -Name EugenePlexusAgent", "Write-Output restart")
    path = directory / "run-update.ps1"
    path.write_bytes(b"\xef\xbb\xbf" + script.encode())
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(path)],
        check=True,
        timeout=60,
        capture_output=True,
    )
    record = update_apply.last(prefix)
    assert record is not None and record.target == SPECS_OK
    assert record.outcome is (UpdateOutcome.succeeded if code == 0 else UpdateOutcome.failed)
    if code:
        assert "exited with 7" in (record.detail or "")
    assert not (directory / update_apply.RUNNING_FILE).exists()


# --------------------------------------------------------------------------- #
# Starting one
# --------------------------------------------------------------------------- #


class Calls:
    def __init__(self, code: int = 0) -> None:
        self.argv: list[list[str]] = []
        self.code = code

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        self.argv.append(argv)
        return subprocess.CompletedProcess(argv, self.code, b"", b"refused")


EDGE_TARGET = updates.Target(
    channel=UpdateChannel.edge,
    ref=SPECS_OK,
    specs_commit=SPECS_OK,
    components=NEW,
    installers={
        "install.ps1": updates.Installer(f"{updates.RAW}/{SPECS_OK}/scripts/install.ps1"),
        "install.sh": updates.Installer(f"{updates.RAW}/{SPECS_OK}/scripts/install.sh"),
    },
)


def test_a_service_updates_through_a_one_shot_task_as_system(tmp_path: Path) -> None:
    web = Web({EDGE_TARGET.installers["install.ps1"].url: "Write-Host installer\n"})
    calls = Calls()
    record = update_apply.start(
        install=_install(mechanism=InstallMechanism.windows_service),
        target=EDGE_TARGET,
        prefix=tmp_path,
        get=web,
        run=calls,
    )
    assert record.outcome is UpdateOutcome.running and record.target == SPECS_OK
    create, run_now = calls.argv
    assert create[:3] == ["schtasks", "/Create", "/TN"] and "SYSTEM" in create
    assert run_now == ["schtasks", "/Run", "/TN", update_apply.WINDOWS_TASK]
    directory = update_apply.update_dir(tmp_path)
    assert (directory / "install.ps1").read_text() == "Write-Host installer\n"
    assert update_apply.running(tmp_path) is not None


def test_a_per_user_install_runs_the_task_as_the_person(tmp_path: Path) -> None:
    web = Web({EDGE_TARGET.installers["install.ps1"].url: "x"})
    calls = Calls()
    update_apply.start(
        install=_install(mechanism=InstallMechanism.windows_task),
        target=EDGE_TARGET,
        prefix=tmp_path,
        get=web,
        run=calls,
    )
    assert "SYSTEM" not in calls.argv[0]


def test_a_release_installer_that_does_not_match_its_checksum_is_not_run(tmp_path: Path) -> None:
    target = updates.Target(
        channel=UpdateChannel.releases,
        ref="v0.1.0-alpha.4",
        release="v0.1.0-alpha.4",
        components=NEW,
        installers={"install.ps1": updates.Installer("https://example/install.ps1", "0" * 64)},
    )
    calls = Calls()
    with pytest.raises(update_apply.UpdateRefused, match="does not match the checksum"):
        update_apply.start(
            install=_install(),
            target=target,
            prefix=tmp_path,
            get=Web({"https://example/install.ps1": "tampered"}),
            run=calls,
        )
    assert calls.argv == []
    assert update_apply.running(tmp_path) is None


def test_a_task_that_cannot_be_registered_leaves_no_running_record(tmp_path: Path) -> None:
    web = Web({EDGE_TARGET.installers["install.ps1"].url: "x"})
    with pytest.raises(update_apply.UpdateRefused, match="registering the update task failed"):
        update_apply.start(
            install=_install(), target=EDGE_TARGET, prefix=tmp_path, get=web, run=Calls(code=1)
        )
    assert update_apply.running(tmp_path) is None


def test_a_system_install_hands_root_a_name_and_nothing_to_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(update_apply, "system_unit_ready", lambda: True)
    web = Web({})
    calls = Calls()
    update_apply.start(
        install=_install(mechanism=InstallMechanism.systemd_system),
        target=EDGE_TARGET,
        prefix=tmp_path,
        get=web,
        run=calls,
    )
    directory = update_apply.update_dir(tmp_path)
    assert (directory / update_apply.REQUEST_FILE).read_text() == f"{SPECS_OK} edge\n"
    # Nothing downloaded, nothing run: root downloads and checks it itself.
    assert web.asked == [] and calls.argv == []
    assert not (directory / "install.sh").exists()


def test_an_update_that_never_reported_back_is_a_failure(tmp_path: Path) -> None:
    directory = update_apply.update_dir(tmp_path)
    directory.mkdir()
    old = datetime.now(UTC) - timedelta(hours=2)
    run = UpdateRun(target=SPECS_OK, startedAt=old, outcome=UpdateOutcome.running, log="x")
    (directory / update_apply.RUNNING_FILE).write_text(run.model_dump_json())
    assert update_apply.running(tmp_path) is None
    expired = update_apply.expired_run(tmp_path)
    assert expired is not None and expired.outcome is UpdateOutcome.failed
    assert "never reported back" in (expired.detail or "")


# --------------------------------------------------------------------------- #
# The routes
# --------------------------------------------------------------------------- #


def _checked(client: TestClient, install: NodeInstall, web: Web, monkeypatch) -> None:
    monkeypatch.setattr(install_info, "describe", lambda **_: install)
    app = client.app
    app.state.update_checker = updates.UpdateChecker(  # type: ignore[attr-defined]
        setting={"updateChannel": "edge"}.get, get=web
    )


def test_get_node_says_what_is_installed_and_what_is_newer(
    authed_client: TestClient, monkeypatch
) -> None:
    _checked(authed_client, _install(SHA), _edge_web([_run("CI", SPECS_OK)]), monkeypatch)
    assert authed_client.post("/v1/node/update/check").status_code == 200
    body = authed_client.get("/v1/node").json()
    assert body["install"]["mechanism"] == "windows_service"
    assert body["update"]["available"] is True
    assert body["update"]["newest"]["ref"] == SPECS_OK
    assert set(body["update"]["behind"]) == set(updates.COMPONENT_NAMES)


def test_update_installs_only_what_this_agent_found(authed_client: TestClient, monkeypatch) -> None:
    _checked(authed_client, _install(SHA), _edge_web([_run("CI", SPECS_OK)]), monkeypatch)
    authed_client.post("/v1/node/update/check")
    started: list[str] = []
    monkeypatch.setattr(
        update_apply,
        "start",
        lambda *, install, target, prefix: (
            started.append(target.ref)
            or UpdateRun(target=target.ref, startedAt=STARTED, outcome=UpdateOutcome.running)
        ),
    )
    other = authed_client.post("/v1/node/update", json={"target": "f" * 40})
    assert other.status_code == 409 and "not the newest" in other.json()["detail"]["title"]
    assert started == []
    ok = authed_client.post("/v1/node/update", json={"target": SPECS_OK})
    assert ok.status_code == 202 and started == [SPECS_OK]


def test_update_before_any_check_is_refused_with_a_reason(
    authed_client: TestClient, monkeypatch
) -> None:
    _checked(authed_client, _install(SHA), _edge_web([]), monkeypatch)
    r = authed_client.post("/v1/node/update", json={"target": SPECS_OK})
    assert r.status_code == 409 and "Check for updates" in r.json()["detail"]["detail"]


def test_update_rechecks_and_never_starts_the_cached_edge_target(authed_client, monkeypatch):
    web = _edge_web([_run("CI", SPECS_OK)])
    _checked(authed_client, _install(SHA), web, monkeypatch)
    authed_client.post("/v1/node/update/check")
    latest = "f" * 40
    web.pages.update(_edge_web([_run("CI", latest)]).pages)
    started = []
    monkeypatch.setattr(
        update_apply, "start", lambda **kwargs: started.append(kwargs["target"].ref)
    )
    response = authed_client.post("/v1/node/update", json={"target": SPECS_OK})
    assert response.status_code == 409
    assert "Nothing was installed" in response.json()["detail"]["detail"]
    assert started == []
    assert authed_client.get("/v1/node").json()["update"]["newest"]["ref"] == latest


def test_update_does_not_use_last_good_target_when_final_check_fails(authed_client, monkeypatch):
    web = _edge_web([_run("CI", SPECS_OK)])
    _checked(authed_client, _install(SHA), web, monkeypatch)
    authed_client.post("/v1/node/update/check")
    web.pages[f"{updates.API}/actions/runs?branch=main&event=push&per_page=60"] = OSError("offline")
    started = []
    monkeypatch.setattr(update_apply, "start", lambda **kwargs: started.append(True))
    response = authed_client.post("/v1/node/update", json={"target": SPECS_OK})
    assert response.status_code == 503
    assert "Nothing was installed" in response.json()["detail"]["detail"]
    assert started == []


def test_a_container_is_refused_with_its_steps(authed_client: TestClient, monkeypatch) -> None:
    _checked(authed_client, _container("unraid"), _edge_web([_run("CI", SPECS_OK)]), monkeypatch)
    authed_client.post("/v1/node/update/check")
    r = authed_client.post("/v1/node/update", json={"target": SPECS_OK})
    assert r.status_code == 409 and "Force Update" in r.json()["detail"]["detail"]


def test_up_to_date_is_not_updated_again(authed_client: TestClient, monkeypatch) -> None:
    _checked(authed_client, _install(NEW), _edge_web([_run("CI", SPECS_OK)]), monkeypatch)
    authed_client.post("/v1/node/update/check")
    r = authed_client.post("/v1/node/update", json={"target": SPECS_OK})
    assert r.status_code == 409 and r.json()["detail"]["title"] == "Already up to date"


def test_an_older_newest_is_refused_and_never_called_newer(
    authed_client: TestClient, monkeypatch
) -> None:
    """The route refuses a downgrade, and GET /v1/node never calls it available."""
    _checked(authed_client, _install(NEW), _edge_web([_run("CI", SPECS_OK)], SHA), monkeypatch)
    authed_client.post("/v1/node/update/check")
    update = authed_client.get("/v1/node").json()["update"]
    assert update["available"] is False
    assert set(update["ahead"]) == set(updates.COMPONENT_NAMES) and update["behind"] == []
    started: list[str] = []
    monkeypatch.setattr(update_apply, "start", lambda **kw: started.append("ran"))
    r = authed_client.post("/v1/node/update", json={"target": SPECS_OK})
    assert r.status_code == 409 and r.json()["detail"]["title"] == "This machine is newer"
    assert started == []


def test_update_is_operator_only(client: TestClient) -> None:
    assert client.post("/v1/node/update", json={"target": SPECS_OK}).status_code == 401
    assert client.post("/v1/node/update/check").status_code == 401


def test_a_per_user_linux_install_updates_from_a_transient_unit(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(update_apply.shutil, "which", lambda name: f"/usr/bin/{name}")
    web = Web({EDGE_TARGET.installers["install.sh"].url: "echo installer\n"})
    calls = Calls()
    update_apply.start(
        install=_install(mechanism=InstallMechanism.systemd_user),
        target=EDGE_TARGET,
        prefix=tmp_path,
        get=web,
        run=calls,
    )
    (argv,) = calls.argv
    # Its own unit, outside the agent's cgroup, which the installer stops.
    assert argv[:2] == ["/usr/bin/systemd-run", "--user"]
    assert any(a.startswith("--unit=eugene-plexus-update-") for a in argv)
    assert argv[-1].endswith("run-update.sh")
    directory = update_apply.update_dir(tmp_path)
    assert "--user --update" in (directory / "run-update.sh").read_text()


# --------------------------------------------------------------------------- #
# A component that joined later (the tool-driver, P8)
# --------------------------------------------------------------------------- #

BEFORE_P8 = {n: sha for n, sha in NEW.items() if n != "tool-driver"}


def test_an_installer_from_before_the_tool_driver_is_still_read() -> None:
    """Every release up to v0.1.0-alpha.5 pins six components. Requiring
    the seventh would make each of them unreadable -- no release channel
    at all for a new agent -- so a pin added later is optional in a target."""
    pins = updates.parse_pins(_install_sh(BEFORE_P8))
    assert "tool-driver" not in pins and pins["ui"] == NEW["ui"]
    # And one that is missing a component every release has is still refused.
    with pytest.raises(updates.CheckFailed, match="gateway"):
        updates.parse_pins(_install_sh({n: s for n, s in NEW.items() if n != "gateway"}))


def test_a_release_manifest_from_before_the_tool_driver_is_offered() -> None:
    web = _releases_web()
    base = "https://github.com/eugene-plexus/specs/releases/download"
    old = web.pages[f"{base}/v0.1.0-alpha.3/manifest.json"]
    old["components"] = dict(BEFORE_P8)
    found = updates.recent_releases(web)
    assert [t.release for t in found] == ["v0.1.0-alpha.3", "v0.1.0-alpha.2"]
    assert "tool-driver" not in found[0].components


def test_an_install_with_the_tool_driver_is_ahead_of_a_target_from_before_it() -> None:
    """An install that has it is not behind a target from before it
    existed -- it is newer, so that target is not offered."""
    target = updates.Target(channel=UpdateChannel.releases, ref="v0", components=BEFORE_P8)
    placed = updates.place(_install(NEW), target, Web({}))
    assert placed.behind == [] and placed.ahead == ["tool-driver"] and not placed.newer
    # And an install matching every pin the release has was on that release.
    release = updates.Target(channel=UpdateChannel.releases, ref="v0", components=dict(SHA))
    release.components.pop("tool-driver")
    assert updates.channel_before_default(_install(SHA), [release]) is UpdateChannel.releases


def test_an_install_without_the_tool_driver_is_behind_a_target_that_pins_it() -> None:
    """The other direction: an alpha.5 install updating to an edge that has
    the tool-driver must install it, not skip it."""
    target = updates.Target(channel=UpdateChannel.edge, ref=SPECS_OK, components=NEW)
    older = dict(NEW) | {"tool-driver": None}
    assert updates.place(_install(older), target, Web({})).behind == ["tool-driver"]
