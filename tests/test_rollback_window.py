"""How far back the agent can reach into llama.cpp's builds, at today's cadence.

**The finding (upstream drift audit, 2026-10-03).** Upstream now publishes
about twenty builds a day: the hundred newest releases on 2026-10-03 span
b11236 (2026-09-28T18:31Z) to b11376 (2026-10-03T13:25Z), 114.9 hours, 20.9
a day, with whole days of 18, 27, 19 and 19. Two windows were sized for a
slower upstream:

- the release list asked GitHub for 30 releases, about 34 hours -- and a
  pinned install (`version`, the rollback for a regression) can only name a
  build in that list;
- `plan_latest` stepped back at most 8 builds past one missing this host's
  asset, "about a day" by its comment and about nine hours in fact.

The list is GitHub's maximum page now (100, about 4.8 days), and the
step-back is about a day again.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from eugene_plexus_agent._generated.models import Accelerator, Arch, HostAccelerator, Os
from eugene_plexus_agent.engines import llama_cpp
from eugene_plexus_agent.engines.acquisition import (
    AcquisitionPlan,
    GitHubReleases,
    Release,
    ReleaseAsset,
)
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter

# Measured 2026-10-03 from `GET /repos/ggml-org/llama.cpp/releases?per_page=100`.
_BUILDS_PER_DAY = 20.9


def test_the_release_list_asks_for_githubs_largest_page(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def fetch(url: str) -> object:
        asked.append(url)
        return []

    monkeypatch.setattr(GitHubReleases, "_fetch", staticmethod(fetch))
    GitHubReleases("ggml-org/llama.cpp").list_releases(force=True)
    assert asked == ["https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100"]


def _build(number: int, *, with_cpu: bool) -> Release:
    names = [f"llama-b{number}-bin-win-vulkan-x64.zip"]
    if with_cpu:
        names.append(f"llama-b{number}-bin-win-cpu-x64.zip")
    return Release(
        version=f"b{number}",
        published_at=datetime(2026, 10, 3, tzinfo=UTC) - timedelta(hours=number % 1000),
        assets=tuple(
            ReleaseAsset(
                name=n, url=f"https://example.invalid/{n}", size=1, digest="sha256:" + "0" * 64
            )
            for n in names
        ),
    )


def test_plan_latest_looks_back_about_a_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """A day of builds missing this host's asset -- upstream's CI failing
    one job for a day -- still finds the last build that has it."""
    day = round(_BUILDS_PER_DAY)
    newest = 11376
    releases = [_build(newest - i, with_cpu=False) for i in range(day - 1)]
    releases.append(_build(newest - day + 1, with_cpu=True))
    adapter = LlamaCppAdapter()
    monkeypatch.setattr(adapter.releases, "list_releases", lambda force=False: releases)
    try:
        plan = adapter.plan_latest(
            HostAccelerator(os=Os.windows, arch=Arch.x64, accelerator=Accelerator.none)
        )
    finally:
        vars(adapter.releases).pop("list_releases", None)
    assert isinstance(plan, AcquisitionPlan), getattr(plan, "reason", "")
    assert plan.version == f"b{newest - day + 1}"


def test_the_step_back_is_a_day_and_says_so() -> None:
    hours = llama_cpp.FALLBACK_BUILDS / _BUILDS_PER_DAY * 24
    assert 20 <= hours <= 30, hours
