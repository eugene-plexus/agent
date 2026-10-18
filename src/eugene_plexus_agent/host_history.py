"""This host's hardware over time: the store, its rollups and the sampler.

Specs `docs/design/host-history.md` (H1-H6, taken 2026-10-18) and
`GET /v1/node/history` in `agent.yaml`. Each agent records its own host,
so a single-machine install has history and a node that loses its root
keeps recording.

**The store** is one SQLite file in the agent's state directory, the
pattern the gateway's metrics store set: raw rows, plus minute and hour
buckets that each carry the count, min, mean and max of every series.
`null` readings are not stored, so a series that could not be read leaves
its buckets empty rather than full of zeros, and a bucket's `n` is the
number of readings it holds.

**Rollups are rewritten as samples arrive**: each sample recomputes the
minute it falls in and the one before (from raw), and the hour it falls in
and the one before (from minutes). So the minute and hour tables are
always current and a query never merges two sources; the cost is a few
dozen rows' aggregate every 10 s.

**Timestamps are wall-clock seconds** (`time.time`), because a bucket is a
time of day; durations elsewhere use `perf_counter`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._generated.models import (
    ComputeDeviceKind,
    HostBucket,
    HostHistory,
    HostHistoryStep,
    HostHistoryStepChoice,
    HostMetric,
    HostSeries,
    HostUnit,
)
from .host_probes import Folder, HostProbe, Reading

log = logging.getLogger(__name__)

FILE_NAME = "host-history.sqlite3"

ENABLED_KEY = "hostHistoryEnabled"
SAMPLE_KEY = "hostHistorySampleSeconds"
RAW_HOURS_KEY = "hostHistoryRawHours"
MINUTE_DAYS_KEY = "hostHistoryMinuteDays"
HOUR_DAYS_KEY = "hostHistoryHourDays"

DEFAULT_SAMPLE_SECONDS = 10
DEFAULT_RAW_HOURS = 6
DEFAULT_MINUTE_DAYS = 7
DEFAULT_HOUR_DAYS = 90
MIN_SAMPLE_SECONDS = 2
MAX_SAMPLE_SECONDS = 3600

#: `step=auto` picks the finest resolution giving at most this many buckets.
AUTO_MAX_BUCKETS = 1500
#: A named step is refused past this many.
MAX_BUCKETS = 5000
#: Retention is enforced at most this often.
PRUNE_SECONDS = 300.0

_RESOLUTION_SECONDS = {HostHistoryStep.minute: 60, HostHistoryStep.hour: 3600}
_BYTES_LIKE = {HostUnit.bytes, HostUnit.count}


class HistoryWindowError(ValueError):
    """A window the endpoint refuses with 400."""


@dataclass(frozen=True)
class Retention:
    raw_seconds: int
    minute_seconds: int
    hour_seconds: int

    def of(self, step: HostHistoryStep) -> int:
        return {
            HostHistoryStep.raw: self.raw_seconds,
            HostHistoryStep.minute: self.minute_seconds,
            HostHistoryStep.hour: self.hour_seconds,
        }[step]


@dataclass(frozen=True)
class Settings:
    enabled: bool
    sample_seconds: int
    retention: Retention


def read_settings(get: Callable[[str], Any]) -> Settings:
    """The settings in force, each clamped to something the sampler can do."""

    def whole(key: str, default: int, low: int, high: int) -> int:
        value = get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return default
        return max(low, min(high, int(value)))

    return Settings(
        enabled=get(ENABLED_KEY) is not False,
        sample_seconds=whole(
            SAMPLE_KEY, DEFAULT_SAMPLE_SECONDS, MIN_SAMPLE_SECONDS, MAX_SAMPLE_SECONDS
        ),
        retention=Retention(
            raw_seconds=whole(RAW_HOURS_KEY, DEFAULT_RAW_HOURS, 1, 24 * 30) * 3600,
            minute_seconds=whole(MINUTE_DAYS_KEY, DEFAULT_MINUTE_DAYS, 1, 365) * 86400,
            hour_seconds=whole(HOUR_DAYS_KEY, DEFAULT_HOUR_DAYS, 1, 3650) * 86400,
        ),
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS series(
    sid INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    meta TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples(
    ts INTEGER PRIMARY KEY,
    interval INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS raw(
    ts INTEGER NOT NULL,
    sid INTEGER NOT NULL,
    v,
    PRIMARY KEY(ts, sid)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS buckets(
    res INTEGER NOT NULL,
    ts INTEGER NOT NULL,
    n INTEGER NOT NULL,
    expected INTEGER NOT NULL,
    PRIMARY KEY(res, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS rollup(
    res INTEGER NOT NULL,
    ts INTEGER NOT NULL,
    sid INTEGER NOT NULL,
    n INTEGER NOT NULL,
    lo,
    mean,
    hi,
    PRIMARY KEY(res, ts, sid)
) WITHOUT ROWID;
"""


def _meta(reading: Reading) -> str:
    return json.dumps(
        {
            "metric": reading.metric.value,
            "unit": reading.unit.value,
            "label": reading.label,
            "deviceKind": reading.device_kind.value if reading.device_kind else None,
            "deviceIndex": reading.device_index,
            "deviceName": reading.device_name,
            "volume": reading.volume,
            "holds": list(reading.holds) or None,
            "unavailable": reading.unavailable,
        },
        sort_keys=True,
    )


def _stored(reading: Reading) -> float | int | None:
    value = reading.value
    if value is None or not math.isfinite(value):
        return None
    # Whole units stored as INTEGER: SQLite keeps them in 1-8 bytes
    # instead of a REAL's 8, and bytes are the bulk of every sample.
    return round(value) if reading.unit in _BYTES_LIKE else round(value, 2)


def _iso(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


class HostHistoryStore:
    """One SQLite file, one connection, one lock. Every method blocks: the
    sampler and the routes call it through `asyncio.to_thread`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self.error: str | None = None
        self._sids: dict[str, tuple[int, str]] = {}
        self._last_ts: int | None = None
        #: Keys of the series the latest sample carried: a series from a
        #: card that has gone is shown only while the window holds its data.
        self.current_keys: set[str] = set()

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> bool:
        with self._lock:
            if self._db is not None:
                return True
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=NORMAL")
                db.executescript(_SCHEMA)
                self._sids = {
                    key: (sid, meta)
                    for sid, key, meta in db.execute("SELECT sid, key, meta FROM series")
                }
                row = db.execute("SELECT max(ts) FROM samples").fetchone()
                self._last_ts = row[0] if row and row[0] is not None else None
            except (sqlite3.Error, OSError) as exc:
                self.error = f"the history store {self.path} could not be opened: {exc}"
                log.warning("host history: %s", self.error)
                return False
            self._db = db
            self.error = None
            return True

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                with contextlib.suppress(sqlite3.Error):
                    self._db.close()
                self._db = None

    def size_bytes(self) -> int | None:
        total = 0
        found = False
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(OSError):
                total += (self.path.parent / (self.path.name + suffix)).stat().st_size
                found = True
        return total if found else None

    def clear(self) -> None:
        with self._lock:
            db = self._require()
            db.execute("BEGIN")
            for table in ("raw", "rollup", "buckets", "samples", "series"):
                db.execute(f"DELETE FROM {table}")
            db.execute("COMMIT")
            db.execute("VACUUM")
            self._sids.clear()
            self._last_ts = None

    def _require(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError(self.error or "the history store is not open")
        return self._db

    # -- writing -------------------------------------------------------------

    def _sid(self, db: sqlite3.Connection, reading: Reading) -> int:
        meta = _meta(reading)
        known = self._sids.get(reading.key)
        if known is not None:
            sid, stored = known
            if stored != meta:
                db.execute("UPDATE series SET meta=? WHERE sid=?", (meta, sid))
                self._sids[reading.key] = (sid, meta)
            return sid
        cursor = db.execute("INSERT INTO series(key, meta) VALUES (?, ?)", (reading.key, meta))
        sid = int(cursor.lastrowid or 0)
        self._sids[reading.key] = (sid, meta)
        return sid

    def record(self, ts: int, interval: int, readings: Sequence[Reading]) -> None:
        with self._lock:
            db = self._require()
            db.execute("BEGIN")
            try:
                db.execute(
                    "INSERT OR REPLACE INTO samples(ts, interval) VALUES (?, ?)", (ts, interval)
                )
                rows = []
                for reading in readings:
                    sid = self._sid(db, reading)
                    value = _stored(reading)
                    if value is not None:
                        rows.append((ts, sid, value))
                db.executemany("INSERT OR REPLACE INTO raw(ts, sid, v) VALUES (?, ?, ?)", rows)
                minutes = {ts // 60 * 60}
                hours = {ts // 3600 * 3600}
                if self._last_ts is not None and self._last_ts < ts:
                    minutes.add(self._last_ts // 60 * 60)
                    hours.add(self._last_ts // 3600 * 3600)
                for minute in sorted(minutes):
                    self._roll_minute(db, minute, interval)
                for hour in sorted(hours):
                    self._roll_hour(db, hour, interval)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            self._last_ts = max(ts, self._last_ts or ts)
            self.current_keys = {r.key for r in readings}

    @staticmethod
    def _roll_minute(db: sqlite3.Connection, start: int, interval: int) -> None:
        end = start + 60
        db.execute("DELETE FROM rollup WHERE res=60 AND ts=?", (start,))
        db.execute(
            "INSERT INTO rollup(res, ts, sid, n, lo, mean, hi) "
            "SELECT 60, ?, sid, count(v), min(v), avg(v), max(v) FROM raw "
            "WHERE ts>=? AND ts<? AND v IS NOT NULL GROUP BY sid",
            (start, start, end),
        )
        (count,) = db.execute(
            "SELECT count(*) FROM samples WHERE ts>=? AND ts<?", (start, end)
        ).fetchone()
        db.execute(
            "INSERT OR REPLACE INTO buckets(res, ts, n, expected) VALUES (60, ?, ?, ?)",
            (start, count, max(1, round(60 / interval))),
        )

    @staticmethod
    def _roll_hour(db: sqlite3.Connection, start: int, interval: int) -> None:
        end = start + 3600
        db.execute("DELETE FROM rollup WHERE res=3600 AND ts=?", (start,))
        db.execute(
            "INSERT INTO rollup(res, ts, sid, n, lo, mean, hi) "
            "SELECT 3600, ?, sid, sum(n), min(lo), sum(mean * n) / sum(n), max(hi) FROM rollup "
            "WHERE res=60 AND ts>=? AND ts<? GROUP BY sid",
            (start, start, end),
        )
        (count,) = db.execute(
            "SELECT coalesce(sum(n), 0) FROM buckets WHERE res=60 AND ts>=? AND ts<?",
            (start, end),
        ).fetchone()
        db.execute(
            "INSERT OR REPLACE INTO buckets(res, ts, n, expected) VALUES (3600, ?, ?, ?)",
            (start, count, max(1, round(3600 / interval))),
        )

    def prune(self, now: float, retention: Retention) -> None:
        with self._lock:
            db = self._require()
            raw_cut = int(now) - retention.raw_seconds
            db.execute("BEGIN")
            db.execute("DELETE FROM raw WHERE ts<?", (raw_cut,))
            db.execute("DELETE FROM samples WHERE ts<?", (raw_cut,))
            for res, keep in ((60, retention.minute_seconds), (3600, retention.hour_seconds)):
                cut = int(now) - keep
                db.execute("DELETE FROM rollup WHERE res=? AND ts<?", (res, cut))
                db.execute("DELETE FROM buckets WHERE res=? AND ts<?", (res, cut))
            db.execute("COMMIT")

    # -- reading -------------------------------------------------------------

    def query(
        self,
        *,
        now: float,
        settings: Settings,
        start: float | None = None,
        end: float | None = None,
        step: HostHistoryStepChoice = HostHistoryStepChoice.auto,
        metrics: Iterable[HostMetric] | None = None,
    ) -> HostHistory:
        end = now if end is None else end
        start = end - 3600 if start is None else start
        if start >= end:
            raise HistoryWindowError("`from` must be before `to`")
        chosen, step_seconds = choose_step(start, end, now, step, settings)
        first = math.floor(start / step_seconds) * step_seconds
        count = math.ceil((end - first) / step_seconds)
        if count > MAX_BUCKETS:
            raise HistoryWindowError(
                f"{count} buckets of {step_seconds} s is more than {MAX_BUCKETS}; "
                "ask for a coarser step or a shorter window"
            )
        wanted = set(metrics) if metrics else None
        with self._lock:
            db = self._require()
            stamps, values, since = self._read(db, chosen, step_seconds, first, end)
            latest_ts = self._last_ts
            latest: dict[int, float] = {}
            if latest_ts is not None:
                latest = dict(
                    db.execute("SELECT sid, v FROM raw WHERE ts=?", (latest_ts,)).fetchall()
                )
            series_rows = list(self._sids.items())
            current = set(self.current_keys)

        buckets = [
            HostBucket(
                start=_iso(first + i * step_seconds),
                samples=stamps.get(i, (0, 0))[0],
                expectedSamples=stamps.get(i, (0, 0))[1]
                or _expected(chosen, step_seconds, settings),
            )
            for i in range(count)
        ]
        series: list[HostSeries] = []
        for key, (sid, meta_text) in sorted(series_rows):
            meta = json.loads(meta_text)
            metric = HostMetric(meta["metric"])
            if wanted is not None and metric not in wanted:
                continue
            per = values.get(sid)
            if per is None and key not in current:
                continue
            unit = HostUnit(meta["unit"])
            lows: list[float | None] = [None] * count
            means: list[float | None] = [None] * count
            highs: list[float | None] = [None] * count
            for index, (lo, mean, hi) in (per or {}).items():
                lows[index], means[index], highs[index] = lo, _round(mean, unit), hi
            kind = meta.get("deviceKind")
            series.append(
                HostSeries(
                    key=key,
                    metric=metric,
                    unit=unit,
                    label=meta["label"],
                    deviceKind=ComputeDeviceKind(kind) if kind else None,
                    deviceIndex=meta.get("deviceIndex"),
                    deviceName=meta.get("deviceName"),
                    volume=meta.get("volume"),
                    holds=meta.get("holds"),
                    unavailable=meta.get("unavailable") if key in current else None,
                    latest=latest.get(sid) if key in current else None,
                    min=lows,
                    mean=means,
                    max=highs,
                )
            )
        # `from` is a Python keyword: the generated field is `from_` with
        # the wire name as its alias, so the model is built by validation.
        return HostHistory.model_validate(
            {
                "enabled": settings.enabled,
                "sampleSeconds": settings.sample_seconds,
                "step": chosen,
                "stepSeconds": step_seconds,
                "from": _iso(first),
                "to": _iso(end),
                "recordingSince": _iso(since) if since is not None else None,
                "truncated": start < now - settings.retention.of(chosen),
                "latestAt": _iso(latest_ts) if latest_ts is not None else None,
                "storeBytes": self.size_bytes(),
                "buckets": buckets,
                "series": series,
            }
        )

    @staticmethod
    def _read(
        db: sqlite3.Connection,
        chosen: HostHistoryStep,
        step_seconds: int,
        first: int,
        end: float,
    ) -> tuple[
        dict[int, tuple[int, int]],
        dict[int, dict[int, tuple[float, float, float]]],
        int | None,
    ]:
        stamps: dict[int, tuple[int, int]] = {}
        values: dict[int, dict[int, tuple[float, float, float]]] = {}
        stop = math.ceil(end)
        if chosen is HostHistoryStep.raw:
            for (ts,) in db.execute("SELECT ts FROM samples WHERE ts>=? AND ts<?", (first, stop)):
                index = (ts - first) // step_seconds
                n, _ = stamps.get(index, (0, 1))
                stamps[index] = (n + 1, 1)
            sums: dict[tuple[int, int], list[float]] = {}
            for ts, sid, v in db.execute(
                "SELECT ts, sid, v FROM raw WHERE ts>=? AND ts<?", (first, stop)
            ):
                sums.setdefault((sid, (ts - first) // step_seconds), []).append(v)
            for (sid, index), found in sums.items():
                values.setdefault(sid, {})[index] = (
                    min(found),
                    sum(found) / len(found),
                    max(found),
                )
            (since,) = db.execute("SELECT min(ts) FROM samples").fetchone()
            return stamps, values, since
        res = _RESOLUTION_SECONDS[chosen]
        for ts, n, expected in db.execute(
            "SELECT ts, n, expected FROM buckets WHERE res=? AND ts>=? AND ts<?",
            (res, first, stop),
        ):
            stamps[(ts - first) // step_seconds] = (n, expected)
        for ts, sid, lo, mean, hi in db.execute(
            "SELECT ts, sid, lo, mean, hi FROM rollup WHERE res=? AND ts>=? AND ts<?",
            (res, first, stop),
        ):
            values.setdefault(sid, {})[(ts - first) // step_seconds] = (lo, mean, hi)
        (since,) = db.execute("SELECT min(ts) FROM buckets WHERE res=?", (res,)).fetchone()
        return stamps, values, since


def _round(value: float, unit: HostUnit) -> float:
    return float(round(value)) if unit in _BYTES_LIKE else round(value, 2)


def _expected(step: HostHistoryStep, step_seconds: int, settings: Settings) -> int:
    if step is HostHistoryStep.raw:
        return 1
    return max(1, round(step_seconds / settings.sample_seconds))


def choose_step(
    start: float,
    end: float,
    now: float,
    asked: HostHistoryStepChoice,
    settings: Settings,
) -> tuple[HostHistoryStep, int]:
    """The resolution and its bucket width. `auto`: the finest whose
    retention covers `start` and that gives at most `AUTO_MAX_BUCKETS`."""
    widths = {
        HostHistoryStep.raw: settings.sample_seconds,
        HostHistoryStep.minute: 60,
        HostHistoryStep.hour: 3600,
    }
    if asked is not HostHistoryStepChoice.auto:
        step = HostHistoryStep(asked.value)
        return step, widths[step]
    for step in (HostHistoryStep.raw, HostHistoryStep.minute, HostHistoryStep.hour):
        covers = start >= now - settings.retention.of(step)
        if covers and (end - start) / widths[step] <= AUTO_MAX_BUCKETS:
            return step, widths[step]
    return HostHistoryStep.hour, 3600


# --------------------------------------------------------------------------- #
# The sampler
# --------------------------------------------------------------------------- #


def folders_of(app_state: Any) -> list[Folder]:
    """The folders whose volumes' free space is recorded: each Library
    folder where this machine mounts it, the model-copy folder when copying
    is on, and the engine folder."""
    from . import model_paths
    from .engines.acquisition import engine_root
    from .library_folders import LibraryFolderCache, effective_rules

    state = app_state.agent_state
    out: list[Folder] = []
    cache = getattr(app_state, "library_folders", None)
    if isinstance(cache, LibraryFolderCache) and cache.folders:
        rules = effective_rules(
            model_paths.rules_from_config(state.get_config), cache.inherited_rules()
        )
        for folder in cache.folders:
            local = model_paths.resolve_model_path(folder.path, rules).local_path
            out.append(Folder("Model folder", local))
    copy_dir = state.get_config("modelCopyDir")
    if state.get_config("modelCopyEnabled") is True and copy_dir:
        out.append(Folder("Model copies", str(copy_dir)))
    out.append(Folder("Engines", str(engine_root())))
    return out


class HostSampler:
    """Samples every `hostHistorySampleSeconds` on the wall-clock grid, so
    two nodes' samples line up. Never raises: a probe or store failure is
    logged once per distinct message and the next tick tries again."""

    def __init__(
        self,
        store: HostHistoryStore,
        probe_factory: Callable[[], HostProbe],
        get_config: Callable[[str], Any],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self._probe_factory = probe_factory
        self._probe: HostProbe | None = None
        self._get_config = get_config
        self._clock = clock
        self._last_prune: float | None = None
        self._last_error: str | None = None

    def settings(self) -> Settings:
        return read_settings(self._get_config)

    def tick(self) -> None:
        """One sample, blocking. Separate from the loop for tests."""
        settings = self.settings()
        if not settings.enabled:
            return
        if not self.store.open():
            return
        if self._probe is None:
            self._probe = self._probe_factory()
        readings = self._probe.sample()
        now = self._clock()
        ts = int(now) // settings.sample_seconds * settings.sample_seconds
        self.store.record(ts, settings.sample_seconds, readings)
        if self._last_prune is None or now - self._last_prune >= PRUNE_SECONDS:
            self.store.prune(now, settings.retention)
            self._last_prune = now

    async def run_forever(self) -> None:
        try:
            while True:
                interval = self.settings().sample_seconds
                now = self._clock()
                await asyncio.sleep(max(0.05, interval - now % interval))
                try:
                    await asyncio.to_thread(self.tick)
                    self._last_error = None
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if message != self._last_error:
                        log.warning("host history: a sample failed: %s", message)
                        self._last_error = message
        finally:
            if self._probe is not None:
                await asyncio.to_thread(self._probe.close)
            await asyncio.to_thread(self.store.close)


__all__ = [
    "FILE_NAME",
    "HistoryWindowError",
    "HostHistoryStore",
    "HostSampler",
    "Retention",
    "Settings",
    "choose_step",
    "folders_of",
    "read_settings",
]
