"""A GitHub token, when the environment has one, goes to the GitHub API and nowhere else (specs#23)."""

from __future__ import annotations

from typing import Any

import pytest

from eugene_plexus_agent.engines import acquisition
from eugene_plexus_agent.engines.acquisition import GitHubReleases, github_api_headers

API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100"
# Split, so no whole token shape sits in the source (secret scanners).
TOKEN = "ghs_" + "example0123456789"


def test_the_api_is_asked_with_the_token_when_there_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert github_api_headers(API)["Authorization"] == f"Bearer {TOKEN}"


def test_without_a_token_the_request_is_as_it_always_was(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert github_api_headers(API) == {
        "Accept": "application/vnd.github+json",
        "User-Agent": "eugene-plexus-agent",
    }
    monkeypatch.setenv("GITHUB_TOKEN", "   ")
    assert "Authorization" not in github_api_headers(API)


@pytest.mark.parametrize(
    "url",
    [
        "https://objects.githubusercontent.com/github-production-release-asset/1/2",
        "https://github.com/ggml-org/llama.cpp/releases/download/b1/llama.zip",
        "https://api.github.com.example.org/repos/x/y/releases",
        "http://example.org/?next=https://api.github.com/",
    ],
)
def test_no_other_host_ever_gets_the_token(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert "Authorization" not in github_api_headers(url)


def test_the_release_list_request_carries_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    seen: dict[str, Any] = {}

    class Answer:
        def __enter__(self) -> Answer:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b"[]"

    def urlopen(request: Any, **_kw: Any) -> Answer:
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        return Answer()

    monkeypatch.setattr(acquisition.urllib.request, "urlopen", urlopen)
    assert GitHubReleases("ggml-org/llama.cpp").list_releases() == []
    assert seen["url"].startswith("https://api.github.com/")
    assert seen["auth"] == f"Bearer {TOKEN}"
