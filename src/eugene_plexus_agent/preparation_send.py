"""A finished preparation's files, sent to the library, which writes them (LS10).

library-sources-and-engines.md §6.13. A preparation runs in a folder of this
node's (B101); a node never writes in a Library folder. The files the
prepared model is made of go to the library a chunk at a time (B104): 16 MiB,
under the 32 MiB an agent's proxy carries, each at the offset the library
says has arrived, under the run's lease. The library checks each file's size
and SHA-256 and only then puts it under its own name; one it already holds
with the same SHA-256 is not sent (Strata's MTP helper is shared by every
model in a folder).

Sending runs inside the run worker's ticks, a slice each (a claim's lease
lasts two minutes and every checkpoint lets it go), so what has arrived at the
library is the state: a send stopped by anything, an agent restart included,
carries on from there.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .preparation import PreparationResult

#: One chunk: well under what an agent's proxy carries (32 MiB).
CHUNK_BYTES = 16 * 1024 * 1024

#: How long one tick sends before it checkpoints (a claim lasts 120 s).
SLICE_SECONDS = 20.0

#: Per request: a chunk crosses up to two machines and a proxy.
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0)

#: A file that arrives damaged is sent again this many times.
RESENDS = 2

_HASH_CHUNK = 4 * 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(_HASH_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, offset: int, size: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(offset)
        return handle.read(size)


def detail_of(exc: httpx.HTTPStatusError) -> str:
    try:
        detail = exc.response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, dict):
        detail = detail.get("detail") or detail.get("title")
    return str(detail or exc)


@dataclass
class _File:
    path: Path
    #: Its place relative to the Library folder, `/` between folders.
    name: str
    size: int
    sha256: str | None = None
    #: What the library has of it, once asked: None until then.
    offset: int | None = None
    resends: int = 0


class Sender:
    """One preparation's files on their way to the library."""

    def __init__(self, result: PreparationResult) -> None:
        if result.root is None:
            raise ValueError("The preparation did not say which folder stands for the Library's.")
        self.files = [
            _File(path=p, name=p.relative_to(result.root).as_posix(), size=p.stat().st_size)
            for p in result.files
        ]
        self.total = sum(f.size for f in self.files)
        self.index = 0
        self._done_bytes = 0

    @property
    def finished(self) -> bool:
        return self.index >= len(self.files)

    @property
    def sent(self) -> int:
        if self.finished:
            return self._done_bytes
        return self._done_bytes + (self.files[self.index].offset or 0)

    def status(self, base: dict[str, Any]) -> dict[str, Any]:
        """The run operation's `PreparationStatus` while files are sent."""
        current = None if self.finished else self.files[self.index].name
        return {
            **base,
            "state": "running",
            "step": "Sending the prepared model to the Library",
            "message": current,
            "bytesWritten": self.sent,
            "bytesNeeded": self.total,
        }

    async def send(
        self,
        library: Any,
        base: str,
        params: dict[str, str],
        lease: str,
        *,
        budget: float | None = None,
    ) -> bool:
        """Send for up to `budget` seconds (a slice); whether every file is in place."""
        budget = SLICE_SECONDS if budget is None else budget
        started = time.perf_counter()
        moved = False  # every slice sends something

        async def call(method: str, path: str, **kwargs: Any) -> Any:
            try:
                return await library.operation_request(
                    method, base + path, timeout=REQUEST_TIMEOUT, **kwargs
                )
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code == 409 and method == "PUT":
                    raise _Behind from exc
                if 400 <= code < 500 or code == 507:
                    # The library's own words: the file it would not take, and why.
                    raise ValueError(
                        f"The Library would not take the prepared file {current.name}: "
                        f"{detail_of(exc)}"
                    ) from exc
                raise

        while not self.finished:
            if moved and time.perf_counter() - started > budget:
                return False
            current = self.files[self.index]
            if current.sha256 is None:
                current.sha256 = await asyncio.to_thread(sha256_of, current.path)
            if current.offset is None:
                held = await call(
                    "POST",
                    "/files/state",
                    params=params,
                    json={"lease": lease, "path": current.name},
                )
                if held["sizeBytes"] == current.size and held["sha256"] == current.sha256:
                    self._next(current)
                    continue
                received = held["receivedBytes"]
                current.offset = received if 0 < received <= current.size else 0
            while current.offset < current.size or (current.size == 0 and current.offset == 0):
                if moved and time.perf_counter() - started > budget:
                    return False
                chunk = await asyncio.to_thread(_read, current.path, current.offset, CHUNK_BYTES)
                try:
                    answer = await call(
                        "PUT",
                        "/files",
                        params={
                            **params,
                            "path": current.name,
                            "offset": str(current.offset),
                            "lease": lease,
                        },
                        content=chunk,
                        headers={"content-type": "application/octet-stream"},
                    )
                except _Behind:
                    current.offset = None  # ask again where it stands
                    break
                current.offset = answer["receivedBytes"]
                moved = True
                if current.size == 0:
                    break
            if current.offset is None:
                continue
            try:
                await call(
                    "POST",
                    "/files/complete",
                    params=params,
                    json={
                        "lease": lease,
                        "path": current.name,
                        "sizeBytes": current.size,
                        "sha256": current.sha256,
                    },
                )
            except ValueError:
                if current.resends >= RESENDS:
                    raise
                # Damaged on the way, or the library lost its part: again.
                current.resends += 1
                current.offset = None
                continue
            self._next(current)
        return True

    def _next(self, current: _File) -> None:
        self._done_bytes += current.size
        current.offset = current.size
        self.index += 1


class _Behind(Exception):
    """The library has another count of the file than this sender."""
