"""Operations rights for people who are not the owner (row 39).

`specs/docs/design/rows/row-39.md`. The owner can give a person an
operations power on chosen machines: **Models** (`operate-models`) or
**Updates and restarts** (`operate-machines`, which includes reading
logs, call R39-LOGS). The person signs in with their own account and the
console exchanges that session at the control root for an
`ep-operations+jwt` addressed to this one machine, for 10 minutes.

**The table below is the allowlist.** An operations token opens a route
only when the route uses `operation(...)` AND the table names the route
with the power the token holds. Every other route keeps its gate:
provider keys and secrets, people's chats, sign-in keys and the
passphrase, enrolling and removing machines, Job Sites, routing and
backends, client keys, folder access and the config stay the owner's.
`tests/test_operations_gates.py` sends a token holding every power to
every route of the app and fails when a route outside the table takes it.

**Every change says who made it.** A gated change on a table route --
the owner's included -- is one row in this machine's operations log
(`GET /v1/node/operations-log`, owner only). The person comes from the
verified token, never from a header. This machine's library reports its
own changes here (`POST /v1/node/operations-log`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import tokens
from .dependencies import _bearer, _bearer_scheme, _problem, verify_bearer

log = logging.getLogger(__name__)

MODELS = tokens.POWER_OPERATE_MODELS
MACHINES = tokens.POWER_OPERATE_MACHINES

#: What each power is called on the People page, for refusals.
POWER_NAMES = {MODELS: "Models", MACHINES: "Updates and restarts"}

#: Reads any operations power may make: what the person needs to see to
#: act, and nothing that holds a secret.
OPERATIONS_READS: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/v1/node"),
        ("GET", "/v1/node/history"),
        ("GET", "/v1/runtimes"),
        ("GET", "/v1/runtimes/{name}"),
        ("GET", "/v1/engines"),
        ("GET", "/v1/engines/{engine}/install"),
        ("GET", "/v1/updates/all"),
    }
)

#: Routes a power opens, with the action its record names.
OPERATIONS_ROUTES: dict[tuple[str, str], tuple[str, str]] = {
    ("POST", "/v1/runtimes"): (MODELS, "create runtime"),
    ("PATCH", "/v1/runtimes/{name}"): (MODELS, "change runtime"),
    ("DELETE", "/v1/runtimes/{name}"): (MODELS, "delete runtime"),
    ("POST", "/v1/runtimes/{name}/start"): (MODELS, "start runtime"),
    ("POST", "/v1/runtimes/{name}/stop"): (MODELS, "stop runtime"),
    ("POST", "/v1/runtimes/{name}/restart"): (MODELS, "restart runtime"),
    ("PUT", "/v1/runtimes/{name}/auto-start"): (MODELS, "change runtime auto-start"),
    ("POST", "/v1/engines/{engine}/install"): (MACHINES, "install engine"),
    ("DELETE", "/v1/engines/{engine}/install"): (MACHINES, "cancel engine install"),
    ("POST", "/v1/engines/{engine}/uninstall"): (MACHINES, "remove engine"),
    ("POST", "/v1/node/update/check"): (MACHINES, "check for updates"),
    ("POST", "/v1/node/update"): (MACHINES, "update Eugene"),
    ("POST", "/v1/updates/all"): (MACHINES, "update all"),
    ("POST", "/v1/components/{name}/restart"): (MACHINES, "restart component"),
    # R39-LOGS: reading logs comes with Updates and restarts.
    ("GET", "/v1/logs"): (MACHINES, "read logs"),
    ("GET", "/v1/logs/stream"): (MACHINES, "read logs"),
}

#: The path parameters that name what an action was done to, in order.
_TARGET_PARAMS = ("name", "engine", "id", "model_id")
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
STATE_KEY = "ep_operation"


def _route_of(request: Request) -> tuple[str, str | None]:
    method = str(request.scope.get("method") or "").upper()
    return method, getattr(request.scope.get("route"), "path", None)


def _typ(token: str) -> object:
    try:
        return jwt.get_unverified_header(token).get("typ")
    except jwt.InvalidTokenError:
        return None


def only_the_owner() -> HTTPException:
    return _problem(
        status.HTTP_403_FORBIDDEN,
        "Owner only",
        "Only the owner can do this. Your operations powers do not include it.",
    )


def missing_power(power: str) -> HTTPException:
    return _problem(
        status.HTTP_403_FORBIDDEN,
        "Power missing",
        f"This needs the {POWER_NAMES[power]} power on this machine. "
        "The owner gives it on Eugene's People page.",
    )


def _mark(request: Request, claims: tokens.Claims, power: str | None, action: str) -> None:
    """Note who is changing what, for the middleware to record."""
    holder = getattr(request.state, STATE_KEY, None)
    if not isinstance(holder, dict) or request.method.upper() in READ_METHODS:
        return
    owner = not claims.is_operations
    params = request.path_params
    target = next((str(params[p]) for p in _TARGET_PARAMS if params.get(p)), "")
    holder.update(
        personId=tokens.SUB_OPERATOR if owner else claims.sub,
        name="Owner" if owner else (claims.name or claims.sub),
        owner=owner,
        power=power,
        action=action,
        target=target[:256],
    )


def operation(
    power: str | None, fallback: Callable[..., tokens.Claims]
) -> Callable[..., tokens.Claims]:
    """The gate for a route in the table (`require_operation`, row 39).

    An operations token addressed to this machine passes when the table
    names this route with `power` (a read: in `OPERATIONS_READS`) and
    the token holds it. Any other bearer goes to `fallback`, the gate the
    route had before, unchanged. A token on a route the table does not
    name is refused whatever the route's code says."""

    def require_operation(
        request: Request,
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    ) -> tokens.Claims:
        token = _bearer(creds)
        route = _route_of(request)
        listed = OPERATIONS_ROUTES.get(route)  # type: ignore[arg-type]
        if _typ(token) != tokens.TYP_OPERATIONS:
            claims = fallback(request, creds)
            if listed is not None and claims.is_session:
                _mark(request, claims, listed[0], listed[1])
            return claims
        claims = verify_bearer(request, token, classes=(tokens.TYP_OPERATIONS,))
        if power is None:
            if route not in OPERATIONS_READS or not claims.ops:
                raise only_the_owner()
            return claims
        if listed is None or listed[0] != power:
            raise only_the_owner()
        if not claims.holds(power):
            raise missing_power(power)
        _mark(request, claims, power, listed[1])
        return claims

    return require_operation


# --------------------------------------------------------------------------- #
# the record of who did what
# --------------------------------------------------------------------------- #

FILE_NAME = "operations-log.sqlite3"
KEEP_SECONDS = 90 * 24 * 3600
KEEP_ROWS = 10_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    component TEXT NOT NULL,
    person_id TEXT NOT NULL,
    name TEXT NOT NULL,
    owner INTEGER NOT NULL,
    power TEXT,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    status INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS operations_at ON operations (at);
"""


class OperationsLog:
    """One SQLite file, one connection, one lock; every method blocks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self.error: str | None = None

    def _open(self) -> sqlite3.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)
            self._db = db
        return self._db

    def record(self, entry: dict[str, Any], *, now: float | None = None) -> None:
        at = time.time() if now is None else now
        with self._lock:
            try:
                db = self._open()
                db.execute(
                    "INSERT INTO operations (at, component, person_id, name, owner, power, "
                    "action, target, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        at,
                        str(entry.get("component") or "agent"),
                        str(entry["personId"]),
                        str(entry["name"]),
                        1 if entry.get("owner") else 0,
                        entry.get("power"),
                        str(entry["action"]),
                        str(entry.get("target") or ""),
                        int(entry["status"]),
                    ),
                )
                db.execute("DELETE FROM operations WHERE at < ?", (at - KEEP_SECONDS,))
                db.execute(
                    "DELETE FROM operations WHERE id <= "
                    "(SELECT id FROM operations ORDER BY id DESC LIMIT 1 OFFSET ?)",
                    (KEEP_ROWS,),
                )
                self.error = None
            except (sqlite3.Error, OSError) as exc:
                self.error = f"the operations log {self.path} could not be written: {exc}"
                log.warning("operations log: %s; lost: %s", self.error, entry)

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            try:
                db = self._open()
                rows = db.execute(
                    "SELECT at, component, person_id, name, owner, power, action, target, "
                    "status FROM operations ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            except (sqlite3.Error, OSError) as exc:
                self.error = f"the operations log {self.path} could not be read: {exc}"
                raise
        return [
            {
                "at": datetime.fromtimestamp(at, tz=UTC),
                "component": component,
                "personId": person_id,
                "name": name,
                "owner": bool(owner),
                "power": power,
                "action": action,
                "target": target,
                "status": code,
            }
            for at, component, person_id, name, owner, power, action, target, code in rows
        ]

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                with contextlib.suppress(sqlite3.Error):
                    self._db.close()
                self._db = None


def log_of(app: Any) -> OperationsLog:
    """This agent's operations log, made on first use beside its config."""
    found: OperationsLog | None = getattr(app.state, "operations_log", None)
    if found is None:
        settings = app.state.settings
        found = OperationsLog(Path(settings.config_file).resolve().parent / FILE_NAME)
        app.state.operations_log = found
    return found


class RecordOperations:
    """Pure ASGI: after a request that `operation(...)` marked, one row.

    The mark is a dict placed in the request's state before the route
    runs, so the dependency fills it and this reads it after, with the
    status the response carried."""

    def __init__(self, app: ASGIApp, owner: Any) -> None:
        self.app = app
        self.owner = owner

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method", "GET").upper() in READ_METHODS:
            await self.app(scope, receive, send)
            return
        holder: dict[str, Any] = {}
        scope.setdefault("state", {})[STATE_KEY] = holder
        status_code = 500

        async def capture(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            if holder.get("action"):
                entry = {**holder, "component": "agent", "status": status_code}
                await asyncio.to_thread(log_of(self.owner).record, entry)


def install(app: Any) -> None:
    app.add_middleware(RecordOperations, owner=app)
