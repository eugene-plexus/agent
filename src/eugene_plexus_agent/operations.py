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
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import tokens
from .dependencies import (
    _bearer,
    _bearer_scheme,
    _problem,
    owner_only_problem,
    token_class,
    verify_bearer,
)

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
    ("POST", "/v1/components/{name}/restart"): (MACHINES, "restart component"),
    # R39-LOGS: reading logs comes with Updates and restarts.
    ("GET", "/v1/logs"): (MACHINES, "read logs"),
    ("GET", "/v1/logs/stream"): (MACHINES, "read logs"),
}

#: The path parameters that name what an action was done to, in order.
_TARGET_PARAMS = ("name", "engine", "id", "model_id")
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
STATE_KEY = "ep_operation"
CLAIMS_KEY = "ep_operations_claims"


def _route_of(request: Request) -> tuple[str, str | None]:
    method = str(request.scope.get("method") or "").upper()
    return method, getattr(request.scope.get("route"), "path", None)


def only_the_owner() -> HTTPException:
    return owner_only_problem()


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


def note_target(request: Request, target: str) -> None:
    """Name what a change on a route with no target in its path is about (a
    new runtime's name), for its row in the operations log."""
    holder = getattr(request.state, STATE_KEY, None)
    if isinstance(holder, dict) and holder.get("action"):
        holder["target"] = target[:256]


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
        if token_class(token) != tokens.TYP_OPERATIONS:
            claims = fallback(request, creds)
            if listed is not None and claims.is_session:
                _mark(request, claims, listed[0], listed[1])
            return claims
        claims = verify_bearer(request, token, classes=(tokens.TYP_OPERATIONS,))
        if power is None:
            if route not in OPERATIONS_READS or not claims.ops:
                raise only_the_owner()
            setattr(request.state, CLAIMS_KEY, claims)
            return claims
        if listed is None or listed[0] != power:
            raise only_the_owner()
        if not claims.holds(power):
            raise missing_power(power)
        _mark(request, claims, power, listed[1])
        setattr(request.state, CLAIMS_KEY, claims)
        return claims

    return require_operation


def operations_claims(request: Any) -> tokens.Claims | None:
    """The operations token this request passed `operation(...)` with, if any."""
    found = getattr(getattr(request, "state", None), CLAIMS_KEY, None)
    return found if isinstance(found, tokens.Claims) else None


#: Review B1: what a runtime declaration may say that reaches past running a
#: model, so only the owner sets or changes it. With what each is, for the
#: refusal.
OWNER_SPEC_FIELDS: dict[str, str] = {
    "env": "environment variables",
    "workingDirectory": "the working directory",
    "binary": "the engine program",
    "extraArgs": "extra command-line arguments",
}
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# agent#53: what an engine flag is, for an operations token, comes from what
# the engine adapter declares about it (`flag_schema()`), never from the key's
# spelling alone. Every declared (engine, key) pair is in exactly one class;
# `tests/test_operations_flags.py` walks every registered engine and fails on
# a pair in none. The classes are per engine, so a new engine flag fails the
# walk even when another engine has a safe flag under the same key.

#: Flags only the owner turns on, with what each is, for the refusal. A value
#: the declaration already had is left alone -- for the same engine, model and
#: files only.
OWNER_FLAGS: dict[tuple[str, str], str] = {
    # `--trust-remote-code`: a model folder's own Python runs in the agent's
    # account.
    ("mlx", "trustRemoteCode"): "running code that comes with a model",
    ("vllm", "trustRemoteCode"): "running code that comes with a model",
}
#: Declared value types that name files or folders on this machine.
PATH_VALUE_TYPES = frozenset({"file_path", "path_list"})
#: Flags declared as text that name a file, folder or model: for an
#: operations token each must lie inside a Library folder, as `modelPath`
#: does, so a Hugging Face Hub id is refused too.
PATH_FLAG_KEYS: frozenset[tuple[str, str]] = frozenset(
    {
        ("llama_cpp", "chatTemplateFile"),  # --chat-template-file
        ("llama_cpp", "draftModelPath"),  # --model-draft
        ("llama_cpp", "projectorPath"),  # --mmproj
        ("vllm", "draftModelPath"),  # the speculative model
        ("mlx", "adapterPath"),  # --adapter-path: LoRA weights
        ("mlx", "draftModel"),  # --draft-model: a folder or a Hub id
        ("kev", "fallback"),  # --fallback: a checkpoint folder
    }
)
#: The reviewed flags an operations token may set, per engine: sizes, counts,
#: choices and switches that change how a model runs on this machine, and
#: nothing that names a file, a program, an address or a secret.
_OPERATIONS_BY_ENGINE: dict[str, tuple[str, ...]] = {
    "llama_cpp": (
        "sameAnswerEveryTime",
        "reuseRepeatedText",
        "draftWith",
        "draftTokens",
        "projectorOnCpu",
        "contextSize",
        "gpuLayers",
        "parallelSlots",
        "batchSize",
        "ubatchSize",
        "threads",
        "flashAttention",
        "cacheType",
        "memoryMargin",
        "cacheRam",
        "cacheReuse",
        "ctxCheckpoints",
        "continuousBatching",
        "slotPinning",
        "mainGpu",
        "devices",  # GPU names, checked against this machine's cards
        "splitMode",
        "tensorSplit",
        "mlock",
        "noMmap",
        "missingCard",
    ),
    "vllm": (
        "sameAnswerEveryTime",
        "source",  # this release's pinned image or the owner's own vLLM
        "shmSizeGiB",
        "reuseRepeatedText",
        "draftWith",
        "draftTokens",
        "maxModelLen",
        "gpuMemoryUtilization",
        "kvCacheDtype",
        "tensorParallelSize",
        "pipelineParallelSize",
        "maxNumSeqs",
        "maxNumBatchedTokens",
        "dtype",
        "enforceEager",
        "quantization",  # a method name from vLLM's own closed list
        "kvOffloadGb",
        "missingCard",
    ),
    "mlx": (
        "maxTokens",
        "promptCacheSize",
        "promptCacheBytes",
        "decodeConcurrency",
        "promptConcurrency",
        "prefillStepSize",
        "numDraftTokens",
        # mlx-lm 0.32.0 passes --chat-template to the tokenizer as the
        # template's own text (`chat_template=`), never as a file name, and
        # transformers renders it in a sandbox.
        "chatTemplate",
    ),
    "kev": (),
    "strata": ("speedProjection", "gpus", "layerSplit"),
}
OPERATIONS_FLAGS: frozenset[tuple[str, str]] = frozenset(
    (engine, key) for engine, keys in _OPERATIONS_BY_ENGINE.items() for key in keys
)
#: Flags whose value runs the engine in a container, with that value. A kept
#: owner flag does not follow a move away from it: vLLM's container gives a
#: model's code only the model's folder, read only, and vLLM on the host runs
#: it in the agent's account. An unset `source` is not the container: it
#: means the host whenever the image is not pulled or a vLLM program is set
#: (the flag's `unsetMeans`), so the owner's trust under it already covered
#: the host, and only leaving an explicit "container" counts.
CONTAINED_FLAGS: dict[tuple[str, str], str] = {("vllm", "source"): "container"}
#: The second net, never the only one: a key that ends this way names a file
#: or folder whatever its declaration says.
PATH_FLAG = re.compile(r"(Path|File|Dir|Directory)$")
#: A negative number written as text: the one value starting with `-` that
#: an argument parser reads as a value, not as an option.
_NEGATIVE_NUMBER = re.compile(r"-(\d+(\.\d*)?|\.\d+)")


def _engine_name(engine: Any) -> str | None:
    found = getattr(engine, "value", engine)
    return found if isinstance(found, str) else None


def flag_class(engine: Any, field: Any) -> str | None:
    """What one engine's declared flag is for an operations token: `owner`,
    `path`, `operations` (reviewed as safe), or None when nobody has
    classified it for this engine. The key's ending is not consulted here
    (see `PATH_FLAG`)."""
    pair = (_engine_name(engine) or "", getattr(field, "key", None) or "")
    value_type = getattr(getattr(field, "valueType", None), "value", None)
    if pair in OWNER_FLAGS:
        return "owner"
    if value_type in PATH_VALUE_TYPES or pair in PATH_FLAG_KEYS:
        return "path"
    if pair in OPERATIONS_FLAGS:
        return "operations"
    return None


def _declared_flags(engine: Any) -> dict[str, Any]:
    """The flags `engine`'s adapter declares, by key; none for an engine this
    build has no adapter for."""
    from .engines import ADAPTERS

    adapter = ADAPTERS.get(engine) if engine is not None else None
    if adapter is None:
        return {}
    return {field.key: field for field in adapter.flag_schema().fields}


def _is_on(value: Any) -> bool:
    return value not in (None, False, "")


def _starts_with_a_dash(value: Any) -> bool:
    """A text value an engine's argument parser could read as an option."""
    items = value if isinstance(value, list) else [value]
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text.startswith("-") and not _NEGATIVE_NUMBER.fullmatch(text):
            return True
    return False


def owner_field(field: str, what: str, *, engine_changed: bool = False) -> HTTPException:
    if engine_changed:
        detail = (
            f"Only the owner can set {what} ({field}) for another engine. "
            "Change the engine without it, or ask the owner."
        )
    else:
        detail = f"Only the owner can set {what} ({field}). Leave it as it was, or ask the owner."
    return _problem(status.HTTP_403_FORBIDDEN, "Owner only", detail)


def refuse_owner_fields(
    request: Any, spec: Any, previous: Any = None, *, folders: Any = None
) -> None:
    """Review B1: an operations token may declare or change a runtime, but
    not what reaches past running a model: environment variables, the
    working directory, the program, extra arguments, an address other
    machines can reach, a file outside the Library's folders, or running
    code that comes with a model. Engine flags are judged by what the
    engine declares (`flag_class`); a declared flag nobody classified is
    the owner's, and an undeclared one is left to `validate_spec`, which
    refuses it for everyone alike. A text value that starts with `-` is
    the owner's. A value the declaration already had is left alone, except
    that nothing is kept across an engine change, and a kept owner flag
    does not follow a new model or a new file. The owner is never checked
    here. `folders` is the Library folder cache; None or unread refuses
    every path, since nothing then says where the Library is."""
    if operations_claims(request) is None:
        return
    # A changed engine is a changed model: what the owner set for one engine
    # is sent again by the owner for another.
    engine_changed = previous is not None and getattr(previous, "engine", None) != getattr(
        spec, "engine", None
    )
    if engine_changed:
        previous = None

    def refuse(field: str, what: str) -> HTTPException:
        return owner_field(field, what, engine_changed=engine_changed)

    def kept(field: str) -> bool:
        return previous is not None and getattr(previous, field, None) == getattr(spec, field)

    for field, what in OWNER_SPEC_FIELDS.items():
        if getattr(spec, field, None) and not kept(field):
            raise refuse(field, what)
    host = getattr(spec, "host", None)
    if host and host not in LOOPBACK_HOSTS and not kept("host"):
        raise refuse("host", "an address other machines can reach")
    known = folders is not None and getattr(folders, "known", False)

    def in_library(path: Any) -> bool:
        return isinstance(path, str) and known and inside_library_folder(folders, path)

    if not in_library(getattr(spec, "modelPath", None)) and not kept("modelPath"):
        raise refuse("modelPath", "a model outside the Library's folders")
    engine = _engine_name(getattr(spec, "engine", None)) or ""
    flags = getattr(spec, "flags", None) or {}
    before = (getattr(previous, "flags", None) or {}) if previous is not None else {}
    declared = _declared_flags(getattr(spec, "engine", None))
    model_moved = not kept("modelPath")
    for key, value in flags.items():
        if value in (None, "") or before.get(key) == value:
            continue
        found = declared.get(key)
        if found is None and not PATH_FLAG.search(key):
            # This engine has no such flag (a typo, another engine's flag):
            # `validate_spec` refuses it with the 400 the owner gets too.
            continue
        if _starts_with_a_dash(value):
            raise refuse(f"flags.{key}", 'a value that starts with "-"')
        kind = flag_class(engine, found) if found is not None else None
        if kind == "path" or PATH_FLAG.search(key):
            values = value if isinstance(value, list) else [value]
            if not values or not all(in_library(v) for v in values):
                raise refuse(f"flags.{key}", "a file outside the Library's folders")
            model_moved = True
        elif kind == "owner":
            if _is_on(value):
                raise refuse(f"flags.{key}", OWNER_FLAGS[(engine, key)])
        elif kind != "operations":
            # Declared, but nobody has classified it for this engine yet.
            raise refuse(f"flags.{key}", "this engine setting")
    # A kept owner flag was the owner's choice for this model, these files
    # and this way of running; another model, tokenizer or draft, or a move
    # out of a container, needs the owner again.
    left = next(
        (
            key
            for (name, key), inside in CONTAINED_FLAGS.items()
            if name == engine and before.get(key) == inside and flags.get(key) != inside
        ),
        None,
    )
    if model_moved or left is not None:
        for key, value in flags.items():
            trusted = OWNER_FLAGS.get((engine, key))
            if trusted is None or not _is_on(value):
                continue
            if model_moved:
                raise refuse(f"flags.{key}", trusted)
            raise _problem(
                status.HTTP_403_FORBIDDEN,
                "Owner only",
                f"Only the owner can set {trusted} (flags.{key}) outside the container "
                f"(flags.{left}). Keep the container, or ask the owner.",
            )


_SEPARATORS = re.compile(r"[\\/]")


def inside_library_folder(folders: Any, path: str) -> bool:
    """Whether `path` really lies inside one of the Library's folders, as
    an operations token's declaration must (review B1, second pass).

    `LibraryFolderCache.folder_for` compares components and normalizes
    nothing, so `<folder>/../../etc/passwd` and a link inside a folder that
    points out would pass it. Here, for operations tokens only: no `..`
    component under either separator; the path has the folder's own shape;
    and, opened on this machine (through the folder's mount when the
    folder is the other shape), it resolves to somewhere inside that folder.
    A folder this machine has no form of is refused."""
    from .model_paths import is_windows_shaped, join_local, match

    if ".." in _SEPARATORS.split(path):
        return False
    folder = folders.folder_for(path)
    if folder is None or is_windows_shaped(path) != is_windows_shaped(folder.path):
        return False
    remainder = match(folder.as_rule(), path)
    if remainder is None:
        return False
    here_windows = os.name == "nt"
    root = (
        folder.path
        if is_windows_shaped(folder.path) == here_windows
        else folder.mount_for(windows=here_windows)
    )
    if not root:
        return False
    real_root = os.path.realpath(str(root))
    real = os.path.realpath(join_local(root, remainder))
    try:
        return bool(os.path.commonpath([real_root, real]) == real_root)
    except ValueError:  # different drives
        return False


def scrub_for_operations[M: BaseModel](request: Request, runtime: M) -> M:
    """Review S6: a runtime read with an operations token carries no
    environment values and no extra arguments; either can hold a key."""
    if operations_claims(request) is None:
        return runtime
    return runtime.model_copy(update={"env": None, "extraArgs": None})


# --------------------------------------------------------------------------- #
# the record of who did what
# --------------------------------------------------------------------------- #

FILE_NAME = "operations-log.sqlite3"
KEEP_SECONDS = 90 * 24 * 3600
KEEP_ROWS = 10_000
#: Refused and failed attempts: kept apart from the changes, under their own cap.
KEEP_REFUSED_ROWS = 1_000

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
                # Review S3: refused attempts are kept apart, under their own
                # smaller cap, so a flood of them never evicts a change.
                for kept, failed in ((KEEP_ROWS, "<"), (KEEP_REFUSED_ROWS, ">=")):
                    db.execute(
                        f"DELETE FROM operations WHERE status {failed} 400 AND id <= "
                        f"(SELECT id FROM operations WHERE status {failed} 400 "
                        "ORDER BY id DESC LIMIT 1 OFFSET ?)",
                        (kept,),
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
