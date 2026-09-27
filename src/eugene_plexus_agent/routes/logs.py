"""`GET /v1/logs` and `GET /v1/logs/stream`: this machine's log (2026-09-27).

Read from any console through `node:<name>`, like `docker logs` and
`docker logs -f`. The format and the reading live in `..logs`; this module
is the HTTP half: the filters, the masking on the way out, and the SSE
framing with a keep-alive so an idle proxy does not close a follow.

Operator-only. A log says more about a machine than any other read --
paths, addresses, every child's own chatter -- so no service credential
opens it, and tokens are masked even for the operator.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from .. import install_info, logs, update_apply
from .._generated.models import LogLine, LogPage
from ..dependencies import require_operator_session

router = APIRouter(tags=["logs"])
_auth = [Depends(require_operator_session)]

KEEPALIVE_SECONDS = 15.0


def log_dir_for(request: Request) -> Path:
    """Where the tee writes: `logs/` beside `agent.yaml`. Tests set
    `app.state.log_dir`."""
    found = getattr(request.app.state, "log_dir", None)
    if found is not None:
        return Path(found)
    config_file: Path = request.app.state.settings.config_file
    return config_file.resolve().parent / "logs"


def _update_log(request: Request) -> Path:
    prefix = install_info.install_prefix(request.app.state.settings.config_file)
    return update_apply.update_dir(prefix) / update_apply.LOG_FILE


def _wire(line: logs.Line) -> LogLine:
    return LogLine(time=line.time, source=line.source, text=logs.redact(line.text))


@router.get("/v1/logs", response_model=LogPage, dependencies=_auth)
async def read_logs(
    request: Request,
    source: Annotated[list[str] | None, Query()] = None,
    since: Annotated[datetime | None, Query()] = None,
    contains: Annotated[str | None, Query(max_length=200)] = None,
    tail: Annotated[int, Query(ge=1, le=logs.MAX_TAIL)] = logs.DEFAULT_TAIL,
) -> LogPage:
    # Files on disk, up to 60 MB of them: off the event loop.
    page = await asyncio.to_thread(
        logs.read,
        log_dir_for(request),
        sources=source or None,
        contains=contains or None,
        since=since,
        tail=tail,
        update_log=_update_log(request),
    )
    return LogPage(
        lines=[_wire(line) for line in page.lines],
        sources=page.sources,
        truncated=page.truncated,
    )


@router.get("/v1/logs/stream", dependencies=_auth)
async def follow_logs(
    request: Request,
    source: Annotated[list[str] | None, Query()] = None,
    contains: Annotated[str | None, Query(max_length=200)] = None,
) -> StreamingResponse:
    sources = source or None
    wanted = contains or None
    bus: logs.Bus = getattr(request.app.state, "log_bus", None) or logs.BUS

    async def frames() -> AsyncIterator[bytes]:
        with bus.follow() as follower:
            # Said at once, so the caller knows the follow is live before
            # the first line arrives -- a quiet machine may write nothing
            # for minutes.
            yield b": following\n\n"
            reported = 0
            while True:
                try:
                    line = await asyncio.wait_for(follower.queue.get(), KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                if follower.dropped > reported:
                    reported = follower.dropped
                    yield f"event: dropped\ndata: {json.dumps({'dropped': reported})}\n\n".encode()
                if not logs.matches(line, sources=sources, contains=wanted, since=None):
                    continue
                body = _wire(line).model_dump_json()
                yield f"event: line\ndata: {body}\n\n".encode()

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
