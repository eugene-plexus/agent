"""A setting's help is one short line, with the details one click away.

The console shows a description's first paragraph and collapses the rest, so
a long description is written as one sentence, a blank line, and what someone
needs to choose a value. A description that runs on in one block, or opens
with a sentence too long to read at a glance, defeats that.

The agent serves two sets: its own settings (`GET /v1/config/schema`) and each
engine's flags, which the profile form shows (`flagSchema` in `GET /v1/engines`).
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from eugene_plexus_agent.engines import ADAPTERS

# A description this short is shown whole; a longer one must open with a line.
SHORT_TEXT_WORDS = 60
FIRST_SENTENCE_WORDS = 20
# Details are for choosing a value, not for the design history.
MOST_WORDS = 130


def _words(text: str) -> int:
    return len(text.split())


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _sentences(paragraph: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+(?=[A-Z`*(\"'])", paragraph.replace("\n", " ").strip())


def _violations(fields: list[dict[str, str]]) -> list[str]:
    found: list[str] = []
    for field in fields:
        key, text = field["key"], field.get("description") or ""
        if not text.strip():
            found.append(f"{key}: no description")
            continue
        sentences = _sentences(_paragraphs(text)[0])
        if _words(sentences[0]) > FIRST_SENTENCE_WORDS:
            found.append(f"{key}: the first sentence is {_words(sentences[0])} words")
        if _words(text) > SHORT_TEXT_WORDS and len(sentences) != 1:
            found.append(f"{key}: over {SHORT_TEXT_WORDS} words without a one-sentence opening")
        if _words(text) > MOST_WORDS:
            found.append(f"{key}: {_words(text)} words")
    return found


def test_the_check_can_fail() -> None:
    long_opening = "word " * 25 + "end."
    wall = "A short opening. " + "More words follow here. " * 20
    essay = "Opening line.\n\n" + "word " * 140
    assert _violations([{"key": "a", "description": long_opening}])
    assert _violations([{"key": "b", "description": wall}])
    assert _violations([{"key": "c", "description": essay}])
    assert _violations([{"key": "d", "description": ""}])
    assert not _violations([{"key": "e", "description": "One line.\n\nThe details."}])


def test_every_served_description_opens_with_one_short_sentence(
    authed_client: TestClient,
) -> None:
    fields = authed_client.get("/v1/config/schema").json()["fields"]
    assert fields
    assert _violations(fields) == []


def test_every_engine_flag_description_opens_with_one_short_sentence() -> None:
    checked = 0
    for kind, adapter in ADAPTERS.items():
        fields = adapter.flag_schema().model_dump(mode="json")["fields"]
        checked += len(fields)
        assert [f"{kind.value}.{v}" for v in _violations(fields)] == []
    assert checked, "no engine flag was checked"
