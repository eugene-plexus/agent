"""This machine's log, read and followed from any console (2026-09-27).

The claims: every line is stamped at receipt in UTC and tagged with its
source, the agent's own lines losing the local `asctime` the stamp
replaces; lines from before stamping still read; history is read newest
last across the rotated files, filtered by source, text and time; tokens
and keys are masked on the way out; the updater's log is served only when
named; only an operator session opens it; and a follower gets each line
the tee writes.
"""

from __future__ import annotations

import asyncio
import io
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from starlette.requests import Request

from eugene_plexus_agent import logs
from eugene_plexus_agent.console_logging import _TeeStream
from eugene_plexus_agent.routes.logs import follow_logs

from .conftest import local_service_token

T0 = datetime(2026, 9, 27, 19, 51, 54, 123000, tzinfo=UTC)
JWT = "eyJhbGciOiJFZERTQSJ9.eyJzdWIiOiJvcGVyYXRvciJ9.c2lnbmF0dXJlLWJ5dGVz"


# --- the line format ------------------------------------------------------------


def test_a_child_line_keeps_the_supervisors_source() -> None:
    assert logs.stamp("[engine: qwen] llama_model_load: loaded", now=T0) == (
        "2026-09-27T19:51:54.123Z [engine: qwen] llama_model_load: loaded"
    )


def test_the_agents_own_line_loses_the_local_time_the_stamp_replaces() -> None:
    line = "2026-09-27 14:51:54,120 INFO eugene_plexus_agent.runtimes: started"
    assert logs.stamp(line, now=T0) == (
        "2026-09-27T19:51:54.123Z [agent] INFO eugene_plexus_agent.runtimes: started"
    )


def test_a_line_with_no_prefix_is_the_agents() -> None:
    assert logs.stamp("INFO:     127.0.0.1:5000 - GET /healthz", now=T0).split(" ", 2)[1] == (
        "[agent]"
    )


def test_a_stamped_line_reads_back_whole() -> None:
    line = logs.parse(logs.stamp("[gateway] routing refreshed", now=T0))
    assert line == logs.Line(T0, "gateway", "routing refreshed")


def test_lines_from_before_stamping_still_read() -> None:
    child = logs.parse("[engine: qwen] no time of its own")
    assert child == logs.Line(None, "engine: qwen", "no time of its own")
    own = logs.parse("2026-09-27 14:51:54,120 INFO x: y")
    assert own.source == "agent" and own.text == "INFO x: y" and own.time is not None
    assert logs.parse("anything else") == logs.Line(None, "agent", "anything else")


# --- masking ---------------------------------------------------------------------


def test_tokens_and_keys_are_masked() -> None:
    text = (
        f"token {JWT} key sk-ant-api03-abcdefghijklmnopqrstuv hf_abcdefghijklmnopqrstuvwxyz "
        "ghp_abcdefghijklmnopqrstuvwxyz0123 Authorization: Bearer opaque.token-1234567890"
    )
    masked = logs.redact(text)
    for secret in (
        JWT,
        "sk-ant-api03-abcdefghijklmnopqrstuv",
        "hf_abcdefghijklmnopqrstuvwxyz",
        "ghp_abcdefghijklmnopqrstuvwxyz0123",
        "opaque.token-1234567890",
    ):
        assert secret not in masked, secret
    assert masked.count(logs.MASK) == 5


def test_ordinary_text_is_left_alone() -> None:
    text = r"opening \\192.168.16.252\downloads\models\q.gguf (skipped 3 layers)"
    assert logs.redact(text) == text


# --- reading history ---------------------------------------------------------------


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _stamped(minutes: int, source: str, text: str) -> str:
    return logs.stamp(f"[{source}] {text}", now=T0 + timedelta(minutes=minutes))


def _two_files(tmp_path: Path) -> Path:
    # agent.log.1 is the OLDER file, as RotatingFileHandler names them.
    _write(
        tmp_path / "agent.log.1",
        [_stamped(0, "gateway", "old-1"), _stamped(1, "engine: q", "old-2")],
    )
    _write(
        tmp_path / "agent.log", [_stamped(2, "gateway", "new-1"), _stamped(3, "engine: q", "new-2")]
    )
    return tmp_path


def test_history_is_read_newest_last_across_the_rotated_files(tmp_path: Path) -> None:
    page = logs.read(_two_files(tmp_path), tail=10)
    assert [line.text for line in page.lines] == ["old-1", "old-2", "new-1", "new-2"]
    assert page.truncated is False
    assert page.sources == ["engine: q", "gateway"]


def test_a_tail_keeps_the_newest_and_says_there_is_more(tmp_path: Path) -> None:
    page = logs.read(_two_files(tmp_path), tail=3)
    assert [line.text for line in page.lines] == ["old-2", "new-1", "new-2"]
    assert page.truncated is True


def test_filters_by_source_text_and_time(tmp_path: Path) -> None:
    folder = _two_files(tmp_path)
    assert [x.text for x in logs.read(folder, sources=["engine: q"]).lines] == ["old-2", "new-2"]
    assert [x.text for x in logs.read(folder, contains="NEW").lines] == ["new-1", "new-2"]
    since = T0 + timedelta(minutes=1)
    assert [x.text for x in logs.read(folder, since=since).lines] == ["old-2", "new-1", "new-2"]


def test_a_long_file_is_read_from_its_end(tmp_path: Path) -> None:
    _write(tmp_path / "agent.log", [_stamped(0, "agent", f"line {n}") for n in range(20000)])
    page = logs.read(tmp_path, tail=2)
    assert [x.text for x in page.lines] == ["line 19998", "line 19999"]


def test_the_updaters_log_is_served_only_when_named(tmp_path: Path) -> None:
    folder = _two_files(tmp_path)
    update = tmp_path / "update.log"
    update.write_text("==> installing\n==> done\n", encoding="utf-8")
    everything = logs.read(folder, update_log=update)
    assert "update" in everything.sources
    assert all(line.source != "update" for line in everything.lines)
    named = logs.read(folder, sources=["update"], update_log=update)
    assert [x.text for x in named.lines] == ["==> installing", "==> done"]


# --- the route --------------------------------------------------------------------


def test_an_operator_reads_this_machines_log_masked(
    authed_client: TestClient, tmp_path: Path
) -> None:
    _write(tmp_path / "agent.log", [_stamped(0, "engine: q", f"token {JWT}")])
    authed_client.app.state.log_dir = tmp_path  # type: ignore[attr-defined]

    body = authed_client.get("/v1/logs", params={"source": "engine: q"}).json()

    assert body["lines"] == [
        {"time": "2026-09-27T19:51:54.123000Z", "source": "engine: q", "text": "token [redacted]"}
    ]
    assert body["truncated"] is False


def test_no_service_credential_opens_it(authed_client: TestClient, tmp_path: Path) -> None:
    authed_client.app.state.log_dir = tmp_path  # type: ignore[attr-defined]
    token = local_service_token(authed_client.app, "gateway")  # type: ignore[arg-type]
    response = authed_client.get("/v1/logs", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code in (401, 403)
    assert authed_client.get("/v1/logs", headers={"Authorization": ""}).status_code == 401


# --- following --------------------------------------------------------------------


def _request(bus: logs.Bus) -> Request:
    app = SimpleNamespace(state=SimpleNamespace(log_bus=bus))
    return Request({"type": "http", "app": app, "headers": [], "query_string": b""})


async def test_a_follower_gets_each_line_as_it_is_written() -> None:
    bus = logs.Bus()
    response = await follow_logs(_request(bus), source=["engine: q"], contains=None)
    frames = response.body_iterator
    assert await anext(frames) == b": following\n\n"

    bus.publish(logs.stamp("[gateway] not wanted", now=T0))
    bus.publish(logs.stamp(f"[engine: q] loaded {JWT}", now=T0))
    frame = await asyncio.wait_for(anext(frames), 2)

    assert frame.startswith(b"event: line\ndata: ")
    assert b'"source":"engine: q"' in frame and b"[redacted]" in frame and JWT.encode() not in frame
    await frames.aclose()  # type: ignore[attr-defined]


async def test_a_follower_that_falls_behind_is_told_how_much_it_lost() -> None:
    bus = logs.Bus(depth=2)
    response = await follow_logs(_request(bus), source=None, contains=None)
    frames = response.body_iterator
    await anext(frames)
    for n in range(5):
        bus.publish(logs.stamp(f"[gateway] {n}", now=T0))
    await asyncio.sleep(0.05)  # let the loop deliver what fits
    first = await asyncio.wait_for(anext(frames), 2)
    assert first.startswith(b"event: dropped\ndata: ")
    await frames.aclose()  # type: ignore[attr-defined]


async def test_the_tee_stamps_the_file_and_publishes_the_line(tmp_path: Path) -> None:
    capture = logging.getLogger("test-logs-capture")
    capture.propagate = False
    handler = logging.FileHandler(tmp_path / "agent.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    capture.addHandler(handler)
    capture.setLevel(logging.INFO)
    try:
        with logs.BUS.follow() as follower:
            tee = _TeeStream(io.StringIO(), capture)
            tee.write("[engine: q] \x1b[31mloaded\x1b[0m\n")
            got = await asyncio.wait_for(follower.queue.get(), 2)
        handler.flush()
    finally:
        capture.removeHandler(handler)
        handler.close()
    written = (tmp_path / "agent.log").read_text(encoding="utf-8").strip()
    assert logs.parse(written).source == "engine: q"
    assert logs.parse(written).text == "loaded"
    assert logs.parse(written).time is not None
    assert got.source == "engine: q" and got.text == "loaded"
