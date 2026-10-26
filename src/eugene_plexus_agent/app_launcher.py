"""The launcher: what the OS service manager runs, inside an app's own account.

Design: `specs/docs/design/workbench.md` §2-§3 (C1). An app on the Windows
service install or the Linux system install runs in an OS account of its
own, started by the OS service manager rather than by the agent: a virtual
service account (`NT SERVICE\\EugenePlexusApp-<id>`) or a systemd dynamic
user (`eugene-plexus-app@<id>.service`). This file is that service's
program. It:

* starts the app's command, with the environment the agent wrote into the
  app's `launch.json` and the secrets the service manager handed over;
* fills in the placeholders of an app that is not ours (C4,
  `c4-open-webui.md` §1): its `args` and the variables of its own in
  `environment`. **The secrets among them are read here and nowhere
  else**, from the files this account is given, so they never reach the
  spec, the agent's log or the service definition. `render_start` is the
  one implementation: the agent imports it for an app its own supervisor
  runs, so the two ways an app runs cannot fill a placeholder two ways.
  Two of them, `{embeddingModel}` and `{rerankModel}` (MR6), are answered
  by asking the gateway, with the app's own key, which of the install's
  own models the app should use (`ask_gateway_for_models`);
* forwards every line the app prints to the agent's log ingress
  (`POST /v1/logs`, OTLP JSON) with the app's own client key, so the Logs
  page shows the app as it shows a child the agent runs itself;
* turns the service manager's stop into the graceful stop the agent's
  supervisor uses -- a console break on Windows, SIGTERM on Linux -- then
  a forced one after a deadline;
* exits with the app's exit code, so the service manager's restart policy
  sees a crash as a crash.

**Standard library only, and run as a script, never imported from the
agent's package by the app's account.** The agent copies this file to
`<apps>/launcher/app_launcher.py`; the account runs it with the app's own
interpreter. The agent's environment is somewhere an app's account must
not be able to read, so nothing here may need it.

Usage: `python -I -u app_launcher.py <launch.json>`
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import UTC, datetime
from http.client import HTTPException, HTTPMessage
from pathlib import Path
from typing import IO, Any

WINDOWS = sys.platform == "win32"

#: Lines held while the agent cannot be reached; past this the oldest go,
#: and the next batch that lands says how many.
BUFFER_LINES = 5000
#: Lines per OTLP request, and the longest a line waits.
BATCH_LINES = 200
BATCH_SECONDS = 1.0
#: How long a stop waits for the app before it is ended by force.
STOP_GRACE_SECONDS = 20.0

_STOP = threading.Event()


# --------------------------------------------------------------------------- #
# the spec
# --------------------------------------------------------------------------- #


def _secret(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def load_spec(path: str) -> dict:
    """The launch spec, with the paths only the service manager knows
    filled in: systemd's `STATE_DIRECTORY` and `CREDENTIALS_DIRECTORY`."""
    spec: dict = json.loads(Path(path).read_text(encoding="utf-8"))
    state = os.environ.get("STATE_DIRECTORY")
    credentials = os.environ.get("CREDENTIALS_DIRECTORY")
    if not spec.get("dataDir") and state:
        # systemd lists several directories with ':' between them; the first is ours.
        spec["dataDir"] = state.split(os.pathsep)[0]
    if not spec.get("keyFile") and credentials:
        spec["keyFile"] = os.path.join(credentials, "client_key")
    if not spec.get("adminTokenFile") and credentials:
        spec["adminTokenFile"] = os.path.join(credentials, "admin_token")
    if not spec.get("oidcSecretFile") and credentials:
        spec["oidcSecretFile"] = os.path.join(credentials, "oidc_secret")
    if spec.get("linkProvider") and credentials:
        spec["linkSecretFile"] = os.path.join(credentials, "memory_server_token")
    if spec.get("memoryServer") and credentials:
        spec["memoryServer"]["tokenFile"] = os.path.join(credentials, "memory_server_token")
        if not spec["memoryServer"].get("modelFile"):
            spec["memoryServer"]["modelFile"] = str(
                Path(path).with_name("memory_server_model.json")
            )
    return spec


def child_environment(spec: dict) -> dict[str, str]:
    """This process's environment (the service manager's, for this
    account), the spec's variables, and where things are."""
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("EUGENE_PLEXUS_"):
            env.pop(name)
    # Never hand the app what systemd handed the launcher about itself.
    for name in ("CREDENTIALS_DIRECTORY", "STATE_DIRECTORY", "INVOCATION_ID", "NOTIFY_SOCKET"):
        env.pop(name, None)
    env.update({str(k): str(v) for k, v in (spec.get("env") or {}).items()})
    # This is a launch fact, never an ambient variable or manifest override.
    env.pop("EUGENE_PLEXUS_APP_ACCOUNT_KIND", None)
    if spec.get("accountKind") in {"windows_service", "systemd"}:
        env["EUGENE_PLEXUS_APP_ACCOUNT_KIND"] = spec["accountKind"]
    data = spec.get("dataDir")
    if data:
        env["EUGENE_PLEXUS_APP_DATA_DIR"] = data
        # The account's home is its data: a dynamic user has none of its
        # own, and an app's caches belong with its state.
        env["HOME"] = data
        if WINDOWS:
            temp = os.path.join(data, "tmp")
            os.makedirs(temp, exist_ok=True)
            env["TEMP"] = env["TMP"] = temp
    if spec.get("keyFile"):
        env["EUGENE_PLEXUS_APP_KEY_FILE"] = spec["keyFile"]
    link = spec.get("memoryServer")
    if link:
        env["EUGENE_PLEXUS_APP_MEMORY_SERVER_URL"] = link["url"]
        env["EUGENE_PLEXUS_APP_MEMORY_SERVER_TOKEN_FILE"] = link["tokenFile"]
        env["EUGENE_PLEXUS_APP_MEMORY_SERVER_MODEL_FILE"] = link["modelFile"]
    token = _secret(spec.get("adminTokenFile"))
    if token:
        env["EUGENE_PLEXUS_APP_ADMIN_TOKEN"] = token
    # Its sign-in secret (C2), for an app the agent told which client it is.
    if env.get("EUGENE_PLEXUS_APP_OIDC_CLIENT_ID") and spec.get("oidcSecretFile"):
        env["EUGENE_PLEXUS_APP_OIDC_SECRET_FILE"] = spec["oidcSecretFile"]
    env["PYTHONUNBUFFERED"] = "1"
    return env


# --------------------------------------------------------------------------- #
# an app that is not ours: its arguments and its own variables (C4)
# --------------------------------------------------------------------------- #

#: What a manifest's `args` and `environment` may name, filled in at every
#: start. The agent refuses a manifest naming anything else.
PLACEHOLDERS = frozenset(
    {
        "bindHost",
        "port",
        "auxPort",
        "memoryModel",
        "linkSecret",
        "dataDir",
        "gatewayUrl",
        "appUrl",
        "oidcIssuer",
        "oidcClientId",
        "clientKey",
        "oidcClientSecret",
        "appSecret",
        "databaseSecret",
        "embeddingModel",
        "rerankModel",
        "imageModel",
        "imageModelSet",
    }
)
#: Read by `render_start` from files, never carried in a spec: a value for
#: one of these in the spec's `values` is ignored.
SECRET_PLACEHOLDERS = frozenset(
    {"clientKey", "oidcClientSecret", "appSecret", "linkSecret", "databaseSecret"}
)
#: The placeholders answered by asking the gateway at every start (MR6), each
#: with the surface a model must list to be the answer: the install's search
#: model (it embeds) and its result sorter (it reranks). Never carried in a
#: spec either: the model that was running when the agent wrote one may not
#: be when the app starts.
MODEL_SURFACES = {"embeddingModel": "embeddings", "rerankModel": "rerank", "imageModel": "image"}
#: What a person calls each, in the notes.
_MODEL_WORDS = {
    "embeddingModel": "search model",
    "rerankModel": "result sorter",
    "imageModel": "image model",
}
#: In all, how long a start waits for the gateway's list of models: the name
#: lookup, the connection, the tries and the answer together. A start does not
#: wait longer than this for a convenience.
MODELS_TIMEOUT_SECONDS = 4.0
#: How long a refused connection waits before the next try, within that time.
#: An agent that has only just started may be a few seconds ahead of its own
#: gateway, and an app that keeps what its first start saw (Open WebUI's
#: database wins after) must not be given the less for it.
MODELS_RETRY_SECONDS = 0.5
#: The longest list read. A big account lists hundreds of models.
MODELS_MAX_BYTES = 16 * 1024 * 1024
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: `{appSecret}`: made at the app's first start, the same at every start after.
APP_SECRET_FILE = "app_secret"
DATABASE_SECRET_FILE = "database_secret"
PG0_INSTANCE = "eugene-plexus-hindsight"
PG0_REQUIREMENT = "pg0-embedded==0.15.2"
#: A hash of the gateway's address, the app's key and the manifest's
#: `environment` as written at the last start, for `resetOnConnectionChange`.
#: A hash, so the file holds no key.
CONNECTION_FILE = "connection.sha256"
#: Held while the files `render_start` keeps in an app's data directory are
#: read and written. The agent runs it in a thread (`_AppPlanner.plan_async`),
#: and a start that was stopped while it waited leaves its thread to finish
#: beside the next start's: two of them making `{appSecret}` at once would give
#: the app a secret that is not the one kept.
_DATA_FILES = threading.Lock()


def placeholders(text: str) -> list[str]:
    """The names `text` asks to be filled in, in order."""
    return _PLACEHOLDER.findall(text)


def _fill(text: str, values: dict[str, str]) -> tuple[str, list[str]]:
    """`text` filled in, and the placeholders that had no value this start."""
    missing: list[str] = []

    def one(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in PLACEHOLDERS:
            raise ValueError(f"{{{name}}} is not a placeholder this launcher fills in")
        value = values.get(name)
        if value is None:
            missing.append(name)
            return ""
        return value

    return _PLACEHOLDER.sub(one, text), missing


def _write_own(path: str, text: str) -> None:
    """Written whole and readable by this account alone: an owner-only mode
    on POSIX, the data directory's own grant on Windows."""
    temp = path + ".tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    if not WINDOWS:
        os.chmod(temp, 0o600)
    os.replace(temp, path)


def app_secret(data_dir: str) -> str:
    """The app's own random secret (`{appSecret}`): made once, kept in its
    data directory, and the same at every start after -- an app that signs
    its sessions with it would sign everyone out at each restart otherwise."""
    with _DATA_FILES:
        return _kept_app_secret(data_dir)


def _kept_app_secret(data_dir: str, filename: str = APP_SECRET_FILE) -> str:
    """`app_secret`, with `_DATA_FILES` held: read it, or make it and keep it."""
    path = os.path.join(data_dir, filename)
    existing = _secret(path)
    if existing:
        return existing
    os.makedirs(data_dir, exist_ok=True)
    value = secrets.token_urlsafe(32)
    _write_own(path, value)
    return value


def database_secret(data_dir: str) -> str:
    with _DATA_FILES:
        return _kept_app_secret(data_dir, DATABASE_SECRET_FILE)


_DATABASE_URI = re.compile(r"(?i)\b((?:pg0|postgres(?:ql)?)://)[^\s\"'<>]*@")
_PORT_MOVE = re.compile(r"Port (\d+) is in use, using port (\d+) instead", re.IGNORECASE)


def redact_database(text: str) -> str:
    """Mask userinfo before output can enter a journal, log, tail or response."""
    return _DATABASE_URI.sub(r"\1[redacted]@", text)


def database_port_error(text: str) -> str | None:
    moved = _PORT_MOVE.search(text)
    if moved:
        return (
            f"Hindsight was not started: database port {moved[1]} is in use; "
            f"pg0 selected port {moved[2]} instead."
        )
    return None


def prepare_database(data: str, port: str | None) -> None:
    if port:
        with socket.socket() as probe:
            if not WINDOWS:
                # PostgreSQL reuses its listener after closing old connections.
                # TIME_WAIT must not turn a valid restart into a port conflict.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", int(port)))
            except OSError as exc:
                raise ValueError(
                    f"Hindsight was not started: database port {port} is in use "
                    f"or cannot be bound: {exc}"
                ) from exc
    if not data:
        return
    root = Path(data) / ".pg0"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not WINDOWS:
        root.chmod(0o700)
    # pg0 0.15.2 initializes both TCP and local auth as password. Harden an
    # existing cluster too, before it starts, so a retained trust rule cannot win.
    hba = root / "instances" / PG0_INSTANCE / "data" / "pg_hba.conf"
    if hba.exists():
        _write_own(
            str(hba),
            "local all all password\nhost all all 127.0.0.1/32 password\n"
            "host all all ::1/128 password\n",
        )


def connection_changed(
    data_dir: str,
    gateway_url: str | None,
    client_key: str | None,
    environment: dict[str, str] | None = None,
) -> bool:
    """Whether what Eugene gives the app differs from the last start's:
    the gateway's address, the app's key, or the manifest's `environment` as
    written (its keys and values with the placeholders unfilled, in a stable
    order). Recorded either way. A first start has nothing to differ from, so
    it is not a change; a digest written before the environment was part of
    it (the address and key alone) is not this one, so it is.

    **Unfilled, and so no more than that.** A change of search model or sorter
    (`{embeddingModel}`, `{rerankModel}`) is a change of what a placeholder
    is filled with, not of what Eugene gives: a reset would have the app take
    the new model while the vectors it stored belong to the one that made
    them, and would clear what its admin set in it. Only the templates are in
    the digest."""
    given = json.dumps(sorted((environment or {}).items()), separators=(",", ":"))
    digest = hashlib.sha256(
        f"{gateway_url or ''}\n{client_key or ''}\n{given}".encode()
    ).hexdigest()
    path = os.path.join(data_dir, CONNECTION_FILE)
    with _DATA_FILES:
        try:
            previous = Path(path).read_text(encoding="utf-8").strip() or None
        except OSError:
            previous = None
        if previous != digest:
            os.makedirs(data_dir, exist_ok=True)
            _write_own(path, digest)
    return previous is not None and previous != digest


# --------------------------------------------------------------------------- #
# which of the install's own models the app is given (MR6)
# --------------------------------------------------------------------------- #


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """The app's key goes to the address it was given and to no other:
    `urllib` would carry it on to wherever a redirect points."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


class _CouldNotSay(Exception):
    """The gateway could not say which models it lists. The message is why,
    as a clause that reads after "because"."""


class _Refused(_CouldNotSay):
    """Nothing is listening where the gateway should be: it may not be up yet."""


class _Slow(_CouldNotSay):
    """The gateway did not answer within the time left."""


def _origin(url: str) -> str:
    """`url` as a person is told it: scheme and host, no path, no credentials."""
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc.rpartition('@')[2]}"


def _slow(origin: str) -> str:
    return f"the gateway at {origin} did not answer within {MODELS_TIMEOUT_SECONDS:g} seconds"


def _gave_up(refused: str | None, origin: str) -> str:
    """Why the time ran out: a connection that was refused is the cause to name
    (the gateway had not come up), not the try that was still waiting."""
    if refused:
        return f"{refused}; it had not come up after {MODELS_TIMEOUT_SECONDS:g} seconds of trying"
    return _slow(origin)


def _said(body: bytes) -> str:
    """What the gateway said with a refusal, in its own words and short: the
    `detail` of its problem document, or an OpenAI-shaped `error.message`."""
    try:
        document = json.loads(body)
    except ValueError:
        return ""
    layers = [document] if isinstance(document, dict) else []
    for layer in list(layers):
        layers.extend(
            inner for inner in (layer.get("detail"), layer.get("error")) if isinstance(inner, dict)
        )
    for layer in layers:
        for field in ("detail", "message", "title"):
            text = layer.get(field)
            if isinstance(text, str) and text.strip():
                return " ".join(text.split())[:240]
    return ""


def _refusal(origin: str, exc: urllib.error.HTTPError) -> str:
    code = exc.code
    try:
        said = _said(exc.read(4096))
    except (OSError, HTTPException):
        said = ""
    finally:
        exc.close()
    answered = f"the gateway at {origin} answered HTTP {code}"
    if 300 <= code < 400:
        answered += " and sent the launcher elsewhere, which it does not follow with this app's key"
    elif code == 401:
        answered += " and did not accept this app's key"
    elif code == 403:
        answered += " and would not list its models for this app's key"
    elif code == 404:
        answered += " and has no list of models at that address (is it the gateway's?)"
    elif code >= 500:
        answered += ", which is a failure on its side"
    return f"{answered}; it said: {said}" if said else answered


def _read_models(gateway_url: str, key: str, timeout: float) -> list[dict[str, Any]]:
    """`GET <gateway>/v1/models` with the app's key, given `timeout` seconds:
    the models it lists for that key. Raises `_CouldNotSay`, naming what was
    observed: `_Refused` when nothing listens, `_Slow` when `timeout` ran out."""
    if urllib.parse.urlsplit(gateway_url).scheme not in ("http", "https"):
        raise _CouldNotSay("the gateway's address does not start with http:// or https://")
    origin = _origin(gateway_url)
    request = urllib.request.Request(
        gateway_url.rstrip("/") + "/v1/models",
        headers={"authorization": f"Bearer {key}", "accept": "application/json"},
    )
    # Never the user's proxy, which would be handed the app's key: the gateway
    # is this install's own, as the log ingress is. One opener for each try.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(MODELS_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise _CouldNotSay(_refusal(origin, exc)) from None
    except TimeoutError:
        raise _Slow(_slow(origin)) from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise _Slow(_slow(origin)) from None
        if isinstance(exc.reason, ConnectionRefusedError):
            raise _Refused(
                f"the gateway at {origin} refused the connection ({exc.reason})"
            ) from None
        raise _CouldNotSay(f"the gateway at {origin} could not be reached ({exc.reason})") from None
    except ConnectionRefusedError as exc:
        raise _Refused(f"the gateway at {origin} refused the connection ({exc})") from None
    except (OSError, HTTPException) as exc:
        raise _CouldNotSay(f"the gateway at {origin} could not be read ({exc})") from None
    if len(body) > MODELS_MAX_BYTES:
        raise _CouldNotSay(
            f"the gateway at {origin} answered with a list of models too long to read"
        )
    try:
        document = json.loads(body)
    except ValueError:
        document = None
    data = document.get("data") if isinstance(document, dict) else None
    if not isinstance(data, list):
        raise _CouldNotSay(
            f"the gateway at {origin} answered with something that is not a list of models"
        )
    return [model for model in data if isinstance(model, dict)]


def ask_gateway_for_models(
    gateway_url: str | None, key: str | None
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """The models the gateway lists for the app's key, or why it could not say
    (a clause that reads after "because", never holding the key).

    One request is made when a start asks and never again, so a change of
    model is met at the next start: with the app's own key, never through the
    user's proxy, given `MODELS_TIMEOUT_SECONDS` in all. A connection that is
    refused is tried again every `MODELS_RETRY_SECONDS` inside that time (the
    gateway may be coming up beside the agent); an answer of any other kind,
    a refusal by the gateway itself included, is the answer. The tries run in
    a thread of their own so that the whole of it is bounded, a name that will
    not resolve included: a socket's timeout does not cover that.
    """
    if not gateway_url:
        return None, "this start was given no address for the gateway"
    if not key:
        return None, (
            "this app has no key to ask the gateway with (its key file is missing or empty)"
        )
    deadline = time.perf_counter() + MODELS_TIMEOUT_SECONDS
    outcome: dict[str, Any] = {}

    def ask() -> None:
        while True:
            try:
                left = max(deadline - time.perf_counter(), 0.05)
                outcome["models"] = _read_models(gateway_url, key, left)
                return
            except _Refused as exc:
                outcome["refused"] = str(exc)
                if deadline - time.perf_counter() <= MODELS_RETRY_SECONDS:
                    return
                time.sleep(MODELS_RETRY_SECONDS)
            except _Slow:
                return
            except _CouldNotSay as exc:
                outcome["why"] = str(exc)
                return
            except Exception as exc:  # a thread's failure would otherwise vanish with it
                outcome["why"] = f"asking it failed ({type(exc).__name__}: {exc})"
                return

    worker = threading.Thread(target=ask, name="gateway-models", daemon=True)
    worker.start()
    worker.join(max(deadline - time.perf_counter(), 0.0))
    if "models" in outcome:
        models: list[dict[str, Any]] = outcome["models"]
        return models, None
    # No answer in the time, or an answer that was no: say which, from what was seen.
    why = str(outcome.get("why") or _gave_up(outcome.get("refused"), _origin(gateway_url)))
    # A gateway that echoed the key back in an error would otherwise put it in the log.
    return None, why.replace(key, "[this app's key]")


def _running(info: dict[str, Any]) -> bool:
    """Whether a backend runs the listed model now: `ready_backends` above 0, or
    the field absent (a gateway older than it), which is not the same as 0. The
    gateway keeps listing a model whose runtime was stopped, with none ready."""
    ready = info.get("ready_backends")
    if isinstance(ready, int) and not isinstance(ready, bool):
        return ready > 0
    return True


def local_model(models: list[dict[str, Any]], surface: str) -> tuple[str | None, bool, list[str]]:
    """The model to use for `surface`, whether it is running, and the ids of those
    that list the surface and are not known to run on the install's own machines.

    The first local model that lists `surface` and is running; failing that,
    the first local one that is listed, running or not, because an app that
    keeps what its first start saw is better off with the install's model than
    with its own default. None when no local model lists it.

    "Runs on the install's own machines" is the listing's own
    `x_eugene_plexus.locality`: `local` only when every backend serving the
    model says so, `external` when any is a cloud account or someone else's
    server, `unknown` otherwise. **Not `owned_by`**: that is the driver's
    provider key (`openai_compat_custom` for the driver of every runtime of
    ours), and reads `local` only when no key is named or two different ones
    are -- so it names no local model of ours, and calls a model served by two
    outside accounts local. Never an outside model: documents would leave the
    install unasked.
    """
    outside: list[str] = []
    stopped: str | None = None
    for model in models:
        ident = model.get("id")
        # An id is put in an environment variable: nothing a process cannot take there.
        if not (isinstance(ident, str) and 0 < len(ident) <= 256 and ident.isprintable()):
            continue
        info = model.get("x_eugene_plexus")
        if not isinstance(info, dict):
            continue
        surfaces = info.get("surfaces")
        if not isinstance(surfaces, list) or surface not in surfaces:
            continue
        if info.get("locality") != "local":
            outside.append(ident)
        elif _running(info):
            return ident, True, outside
        elif stopped is None:
            stopped = ident
    return stopped, False, outside


def _unset_model(name: str, outside: list[str]) -> str:
    """Why a model placeholder has no value, when the gateway answered."""
    word = _MODEL_WORDS[name]
    if name == "imageModel":
        return (
            "{imageModel} has no value at this start: the gateway lists no eligible "
            "image model for this app's key. Images are off."
        )
    head = f"{{{name}}} has no value at this start: no local {word} is running"
    if not outside:
        return f"{head} (the gateway lists none for this app's key)."
    shown = ", ".join(outside[:3]) + (f" and {len(outside) - 3} more" if len(outside) > 3 else "")
    known = "it is" if len(outside) == 1 else "none of them is"
    return (
        f"{head}. The gateway lists {shown}, but {known} known to run on this install's own "
        "machines, and a model that does not is never used: the documents would leave the install."
    )


class GatewayUnknown(ValueError):
    """A manifest names `{gatewayUrl}` and this start has no gateway address to
    give it. The message says why the app is not started: the agent's own words
    for why there is none, when it has them, and what the app would do instead."""


def _no_gateway(why: object) -> str:
    cause = " ".join(str(why).split())[:300].rstrip(".") if why else ""
    return (
        f"{{gatewayUrl}} has no value at this start{f' ({cause})' if cause else ''}, and this "
        "app names it, so it is not started: an app that is not given the gateway's address "
        "uses its own default, and Open WebUI's is OpenAI's, so its documents and speech "
        "would go there."
    )


def _stopped_model(name: str, model: str) -> str:
    """The note for a model that is the answer though it is not running now."""
    word = _MODEL_WORDS[name]
    return (
        f"{{{name}}} is {model} at this start, though the gateway lists it but it is not running "
        f"now, and no other local {word} is. The app keeps what its first start saw, so it is "
        f"given the install's {word} rather than its own default: start {model} in Eugene's "
        "console, under Backends, for it to work."
    )


def image_model(models: list[dict[str, Any]]) -> tuple[str | None, bool]:
    """Images may use an external model; local models always come first."""
    local, running, _ = local_model(models, "image")
    if local:
        return local, running
    external = [
        m
        for m in models
        if isinstance(m.get("x_eugene_plexus"), dict)
        and m["x_eugene_plexus"].get("locality") == "external"
    ]
    # Reuse the same id, surface and readiness validation as local models.
    external = [
        {**m, "x_eugene_plexus": {**m["x_eugene_plexus"], "locality": "local"}} for m in external
    ]
    chosen, running, _ = local_model(external, "image")
    return chosen, running


def render_start(spec: dict, env: dict[str, str]) -> tuple[list[str], dict[str, str], list[str]]:
    """One start's command and environment, and what is worth saying about it.

    The command is `spec["argv"]` followed by the manifest's `args`; the
    environment is `env` with the manifest's own variables on top. Every
    placeholder is filled in from `spec["values"]` (what the agent may
    write down), `spec["dataDir"]`, and the secrets read here from the
    files the spec names. A variable whose placeholder has no value this
    start -- no sign-in client, no search model -- is left unset rather
    than set to half a value, and the returned lines say so; an argument
    gets an empty string, so the arguments after it keep their places.

    **`{gatewayUrl}` is the one that is not left unset**: a manifest that
    names it is not started without it (`GatewayUnknown`, raised before
    anything is read or written). An app that is not given the gateway's
    address uses its own default for each address it takes, and Open WebUI's
    is OpenAI's: its documents and speech would go there. Both ways an app
    runs meet it as a `ValueError` and refuse the start, naming why.

    `{embeddingModel}` and `{rerankModel}` are asked of the gateway, once,
    and only by a manifest that names one (`ask_gateway_for_models`). A
    start does not wait on it for more than `MODELS_TIMEOUT_SECONDS`, and
    one that gets no answer, or no local model, says why in its own line
    beside the variable's. A model that is listed but not running is still
    the answer when it is all there is, and the line says so.

    With `resetOnConnectionChange`, that variable is `true` for a start
    whose connection differs from the last one's (the gateway's address,
    the app's key, or the manifest's `environment` as written, unfilled:
    `connection_changed`) and absent otherwise, even when the environment
    it inherits has it.
    """
    data = str(spec.get("dataDir") or "")
    values = {
        str(k): str(v)
        for k, v in (spec.get("values") or {}).items()
        if v is not None and k in PLACEHOLDERS and k not in SECRET_PLACEHOLDERS
        if k not in MODEL_SURFACES and k != "imageModelSet"
    }
    if data:
        values["dataDir"] = data
    args = [str(a) for a in spec.get("args") or []]
    environment = {str(k): str(v) for k, v in (spec.get("environment") or {}).items()}
    reset = spec.get("resetOnConnectionChange")
    wanted = {n for text in [*args, *environment.values()] for n in placeholders(text)}
    if "memoryModel" in wanted and not values.get("memoryModel"):
        raise ValueError("Choose the model Hindsight reads your chats with")
    if "linkSecret" in wanted:
        path = spec.get("linkSecretFile")
        try:
            token = Path(path).read_text(encoding="utf-8").strip() if path else ""
        except OSError as exc:
            raise ValueError(f"The memory-server link secret could not be read: {exc}") from exc
        if not token:
            raise ValueError("The memory-server link secret is missing or empty")
        values["linkSecret"] = token
    if "gatewayUrl" in wanted and not values.get("gatewayUrl"):
        raise GatewayUnknown(_no_gateway(spec.get("gatewayDetail")))
    models = [name for name in MODEL_SURFACES if name in wanted]
    if "imageModelSet" in wanted and "imageModel" not in models:
        models.append("imageModel")

    key = _secret(spec.get("keyFile")) if "clientKey" in wanted or reset or models else None
    if key:
        values["clientKey"] = key
    if "oidcClientSecret" in wanted:
        oidc = _secret(spec.get("oidcSecretFile"))
        if oidc:
            values["oidcClientSecret"] = oidc
    if "appSecret" in wanted and data:
        values["appSecret"] = app_secret(data)
    if "databaseSecret" in wanted:
        if not data:
            raise ValueError("The database password needs the app's data folder")
        values["databaseSecret"] = database_secret(data)
    if spec.get("linkProvider"):
        prepare_database(data, values.get("auxPort"))

    notes: list[str] = []
    if models:
        listing, why = ask_gateway_for_models(values.get("gatewayUrl"), key)
        for name in models:
            if listing is None:
                notes.append(
                    f"{{{name}}} has no value at this start: the launcher could not ask which "
                    f"{_MODEL_WORDS[name]} runs, because {why}."
                )
                continue
            chosen, running, outside = local_model(listing, MODEL_SURFACES[name])
            if name == "imageModel":
                chosen, running = image_model(listing)
            if chosen is None:
                notes.append(_unset_model(name, outside))
                continue
            values[name] = chosen
            if not running:
                notes.append(_stopped_model(name, chosen))
    values["imageModelSet"] = "true" if values.get("imageModel") else "false"
    if spec.get("linkProvider") and any(name not in values for name in models):
        raise ValueError("Hindsight was not started. " + " ".join(notes))
    argv = [str(a) for a in spec["argv"]]
    argv.extend(_fill(text, values)[0] for text in args)
    out = dict(env)
    for name, text in environment.items():
        filled, missing = _fill(text, values)
        if missing:
            out.pop(name, None)
            notes.append(
                f"{name} is not set at this start: there is no value for "
                + ", ".join(f"{{{m}}}" for m in missing)
                + "."
            )
        else:
            out[name] = filled
    if reset:
        out.pop(reset, None)
        first = data and not Path(data, CONNECTION_FILE).exists()
        if data and connection_changed(data, values.get("gatewayUrl"), key, environment):
            out[reset] = "true"
            notes.append(
                "What Eugene gives this app changed since the last start (the gateway's "
                "address, the app's key, or the settings Eugene gives it), so "
                f"{reset}=true for this start: the app takes the new ones, and settings "
                "changed inside it go back to their defaults."
            )
        if data and (first or out.get(reset) == "true"):
            given = {
                "givenAt": datetime.now(UTC).isoformat(),
                "webSearchEnabled": out.get("ENABLE_WEB_SEARCH", "false").lower() == "true",
                **{name: values[name] for name in MODEL_SURFACES if name in values},
            }
            with _DATA_FILES:
                _write_own(os.path.join(data, "given.json"), json.dumps(given))
    if spec.get("linkProvider"):
        # The same stdlib guard runs in both launch accounts. It filters the
        # child's output before the journal or agent can see it, and rejects a
        # port fallback, including a bind race after the check above.
        python = str(spec.get("providerPython") or spec["argv"][0])
        argv = [python, str(Path(__file__).resolve()), "--memory-provider", *argv]
    return argv, out, notes


def pg0_cleanup(python: str, action: str, env: dict[str, str]) -> None:
    """Run the installed package's binary as its launch account, never a PATH pg0."""
    if action not in {"stop", "drop"}:
        raise ValueError("Unknown pg0 cleanup action")
    code = (
        "import pathlib,pg0,subprocess,sys; "
        "p=next(p for p in (pathlib.Path(pg0.__file__).parent/'bin').glob('pg0*') if p.is_file()); "
        "sys.exit(subprocess.call([str(p),*sys.argv[1:]]))"
    )
    args = [python, "-c", code, action, "--name", PG0_INSTANCE]
    if action == "stop":
        args += ["--timeout", "30"]
    else:
        args += ["--force"]
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=45, check=False)
    if result.returncode:
        output = redact_database((result.stdout + result.stderr).strip())
        # pg0 reports an absent instance with a non-zero exit; it is already clean.
        absent = (
            "no running instance found",
            f"instance '{PG0_INSTANCE}' does not exist",
            f"instance '{PG0_INSTANCE}' is not running",
        )
        if any(text in output.lower() for text in absent) and "traceback" not in output.lower():
            return
        raise ValueError(
            f"pg0 {action} --name {PG0_INSTANCE} failed ({result.returncode}): {output[-800:]}"
        )


# --------------------------------------------------------------------------- #
# forwarding output
# --------------------------------------------------------------------------- #


class Forwarder:
    """Lines in, OTLP batches out, with a bounded buffer for an outage."""

    def __init__(self, url: str | None, key_file: str | None, app: str) -> None:
        self._url = url
        self._key_file = key_file
        self._app = app
        self._lines: deque[str] = deque()
        self._dropped = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._until = 0.0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="forwarder", daemon=True)
        # Bypass any proxy: the ingress is on this machine.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start(self) -> None:
        self._thread.start()

    def add(self, line: str) -> None:
        line = redact_database(line)
        with self._lock:
            self._lines.append(line)
            while len(self._lines) > BUFFER_LINES:
                self._lines.popleft()
                self._dropped += 1
        if len(self._lines) >= BATCH_LINES:
            self._wake.set()
        if not WINDOWS:
            # On Linux the journal keeps the launcher's own output too: a
            # second copy for `journalctl -u`, and the only one while the
            # agent is down.
            print(line, flush=True)

    def close(self, timeout: float = 5.0) -> None:
        self._done.set()
        self._wake.set()
        self._thread.join(timeout)

    def _take(self) -> list[str]:
        with self._lock:
            batch = [self._lines.popleft() for _ in range(min(BATCH_LINES, len(self._lines)))]
            if self._dropped and batch:
                batch.insert(
                    0,
                    f"[launcher] {self._dropped} lines were dropped while the agent could not "
                    "be reached",
                )
                self._dropped = 0
            return batch

    def _put_back(self, batch: list[str]) -> None:
        with self._lock:
            self._lines.extendleft(reversed(batch))
            while len(self._lines) > BUFFER_LINES:
                self._lines.pop()
                self._dropped += 1

    def _run(self) -> None:
        while True:
            self._wake.wait(BATCH_SECONDS)
            self._wake.clear()
            finishing = self._done.is_set()
            while True:
                if time.perf_counter() < self._until and not finishing:
                    break
                batch = self._take()
                if not batch:
                    break
                if not self._send(batch):
                    self._put_back(batch)
                    break
            if finishing:
                return

    def _send(self, batch: list[str]) -> bool:
        key = _secret(self._key_file)
        if not self._url or not key:
            return False
        body = json.dumps(
            {
                "resourceLogs": [
                    {
                        "resource": {
                            "attributes": [
                                {"key": "service.name", "value": {"stringValue": self._app}}
                            ]
                        },
                        "scopeLogs": [
                            {"logRecords": [{"body": {"stringValue": line}} for line in batch]}
                        ],
                    }
                ]
            }
        ).encode()
        request = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
        )
        try:
            with self._opener.open(request, timeout=10) as response:
                return bool(200 <= response.status < 300)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry = exc.headers.get("Retry-After", "5")
                self._until = time.perf_counter() + (int(retry) if retry.isdigit() else 5)
                return False
            if exc.code in (401, 403):
                # The key was refused: revoked, or not allowed to send. Keep
                # the lines and ask again in a while; the buffer is bounded.
                self._until = time.perf_counter() + 30
                return False
            # A batch the ingress will never take (malformed, too large) is
            # dropped rather than sent forever; anything else is an outage.
            return exc.code < 500
        except (urllib.error.URLError, OSError):
            return False


# --------------------------------------------------------------------------- #
# the app
# --------------------------------------------------------------------------- #


def _spawn(
    spec: dict, argv: list[str], env: dict[str, str], *, suspended: bool = False
) -> subprocess.Popen:
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        if suspended:
            flags |= 0x00000004  # CREATE_SUSPENDED: it runs once `_Job` holds it
    return subprocess.Popen(
        argv,
        env=env,
        cwd=spec.get("dataDir") or None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )


def _pump(proc: subprocess.Popen, forwarder: Forwarder) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        forwarder.add(raw.decode("utf-8", errors="replace").rstrip("\r\n"))


def _ask_to_stop(proc: subprocess.Popen) -> str | None:
    if sys.platform == "win32":
        import ctypes

        # CTRL_BREAK_EVENT to the app's own process group: the stop the
        # agent's supervisor sends a child, which every component and
        # engine answers by shutting down cleanly. A service has no
        # console to send it through until `run` allocates one.
        if not ctypes.windll.kernel32.GenerateConsoleCtrlEvent(1, proc.pid):
            return _force(proc)
    else:
        proc.send_signal(signal.SIGTERM)
    return None


def _force(proc: subprocess.Popen) -> str | None:
    """End the app by force. What went wrong, in words, or None."""
    if WINDOWS:
        # The app's interpreter may be a launcher with the real one as its
        # child; end the whole tree.
        out = subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode != 0 and proc.poll() is None:
            said = " ".join((out.stdout + out.stderr).split())[-300:]
            return f"taskkill could not end the app (code {out.returncode}): {said}"
        return None
    proc.kill()
    return None


class _Job:
    """A Windows job object that holds the app's process and every process
    it starts (agent#34).

    Its handle is this launcher's alone, and it is made to end what it holds
    when that handle closes -- when the launcher exits, however it exits. So
    the agent's wait for the service's process is a wait for the app's whole
    tree, and nothing the app started still holds its database or its port
    when Go back or an update goes on. `end` ends a straggler while the
    launcher can still say so.
    """

    _KILL_ON_JOB_CLOSE = 0x2000
    _EXTENDED_LIMITS = 9
    _ACCOUNTING = 1
    _ctypes: Any
    _kernel: Any
    _accounting: Any
    _handle: Any

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("a job object is Windows' own")
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", Basic),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class Accounting(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_int64),
                ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64),
                ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
        ]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._ctypes, self._kernel = ctypes, kernel
        self._accounting = Accounting
        self._handle = kernel.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = self._KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(
            self._handle, self._EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def hold(self, proc: subprocess.Popen) -> None:
        """Put `proc`, created suspended, in the job; then let it run."""
        handle = int(proc._handle)  # type: ignore[attr-defined,unused-ignore]
        if not self._kernel.AssignProcessToJobObject(self._handle, handle):
            raise self._ctypes.WinError(self._ctypes.get_last_error())

    def active(self) -> int:
        info = self._accounting()
        ctypes = self._ctypes
        if not self._kernel.QueryInformationJobObject(
            self._handle, self._ACCOUNTING, ctypes.byref(info), ctypes.sizeof(info), None
        ):
            return 0
        return int(info.ActiveProcesses)

    def end(self, seconds: float = 10.0) -> int:
        """End whatever is still in the job; how many processes that was."""
        left = self.active()
        if left:
            self._kernel.TerminateJobObject(self._handle, 1)
            deadline = time.perf_counter() + seconds
            while self.active() and time.perf_counter() < deadline:
                time.sleep(0.05)
        return left

    def close(self) -> None:
        if self._handle:
            self._kernel.CloseHandle(self._handle)
            self._handle = None


def _resume(proc: subprocess.Popen) -> None:
    """Let a process created suspended run."""
    if sys.platform != "win32":
        return
    import ctypes

    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.restype = ctypes.c_long
    status = ntdll.NtResumeProcess(ctypes.c_void_p(int(proc._handle)))  # type: ignore[attr-defined,unused-ignore]
    if status != 0:
        raise OSError(f"the app's process could not be let run (NTSTATUS {status & 0xFFFFFFFF:#x})")


def _spawn_contained(
    spec: dict, argv: list[str], env: dict[str, str]
) -> tuple[subprocess.Popen, _Job | None, str | None]:
    """The app's process, and on Windows the job that ends its tree with
    this launcher. A job that cannot be made is said, and the app runs as
    before; a process that cannot be let run is ended, and raises."""
    if not WINDOWS:
        return _spawn(spec, argv, env), None, None
    proc = _spawn(spec, argv, env, suspended=True)
    job: _Job | None = None
    note = None
    try:
        job = _Job()
        job.hold(proc)
    except OSError as exc:
        if job is not None:
            job.close()
        job = None
        note = (
            f"the app's processes are not tied to this launcher ({exc}); one it starts "
            "may outlive a stop"
        )
    try:
        _resume(proc)
    except OSError:
        proc.kill()
        if job is not None:
            job.close()
        raise
    return proc, job, note


def watch_memory_provider(argv: list[str]) -> int:
    """Keep secrets and pg0's port fallback inside the launch boundary."""
    env = dict(os.environ)
    data = env.get("HOME", "")
    password = _secret(os.path.join(data, DATABASE_SECRET_FILE)) if data else None
    previous_umask = os.umask(0o077) if not WINDOWS else None
    try:
        # The child inherits the private mask: pg0's instance.json also holds
        # the password. Restore ours even if spawning the child fails.
        proc = _spawn({"dataDir": data}, argv, env)
    finally:
        if previous_umask is not None:
            os.umask(previous_umask)
    failure: list[str] = []
    database_url = env.get("HINDSIGHT_API_DATABASE_URL", "")
    expected_port = urllib.parse.urlsplit(database_url).port if database_url else None
    instance = Path(data) / ".pg0" / "instances" / PG0_INSTANCE / "instance.json"

    def output() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = redact_database(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            if password:
                line = line.replace(password, "[redacted]")
            cause = database_port_error(line)
            if cause and not failure:
                failure.append(cause)
            print(cause or line, flush=True)

    reader = threading.Thread(target=output, daemon=True)
    reader.start()
    try:
        while proc.poll() is None:
            if expected_port and instance.is_file() and not failure:
                try:
                    actual_port = json.loads(instance.read_text(encoding="utf-8"))["port"]
                except (OSError, ValueError, KeyError):
                    actual_port = expected_port
                if actual_port != expected_port:
                    cause = (
                        f"Hindsight was not started: database port {expected_port} was reserved; "
                        f"pg0 selected port {actual_port} instead. The reserved port may be in use."
                    )
                    failure.append(cause)
                    print(cause, flush=True)
            if failure or _STOP.wait(0.1):
                _ask_to_stop(proc)
                try:
                    proc.wait(STOP_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    _force(proc)
                    proc.wait(10)
                break
        reader.join(5)
    finally:
        try:
            pg0_cleanup(argv[0], "stop", env)
        except (OSError, ValueError) as exc:
            failure.append(redact_database(str(exc)))
            print(f"[launcher] {failure[-1]}", flush=True)
    return 1 if failure else (0 if _STOP.is_set() else (proc.returncode or 0))


def run(spec: dict) -> int:
    """Start the app, forward its output, wait for it or for a stop."""
    if spec.get("pg0Action"):
        try:
            pg0_cleanup(spec["argv"][0], spec["pg0Action"], child_environment(spec))
            return 0
        except (OSError, ValueError) as exc:
            print(f"[launcher] {exc}", file=sys.stderr)
            return 1
    if sys.platform == "win32":
        import ctypes

        # A console of its own (session 0, never shown), so a stop can be
        # sent to the app the way the supervisor sends one. The app gets a
        # process group of its own, so the break reaches it and not this.
        ctypes.windll.kernel32.AllocConsole()
    forwarder = Forwarder(spec.get("ingress"), spec.get("keyFile"), spec.get("app", "app"))
    forwarder.start()
    if spec.get("generation"):
        # Which start this is, so the Logs page tells one launch from another.
        forwarder.add(
            f"[launcher] starting {spec.get('app', 'app')} {spec.get('version') or ''} "
            f"(launch {str(spec['generation'])[:8]})".replace("  ", " ")
        )
    try:
        return _run(spec, forwarder)
    except Exception as exc:
        # Said, never a bare exit code 1 that names nothing (agent#34).
        forwarder.add(f"[launcher] stopped on an unexpected error: {exc!r}")
        return 1
    finally:
        forwarder.close()


def _run(spec: dict, forwarder: Forwarder) -> int:
    try:
        argv, env, notes = render_start(spec, child_environment(spec))
        if spec.get("linkProvider"):
            pg0_cleanup(spec["argv"][0], "stop", env)
        for note in notes:
            forwarder.add(f"[launcher] {note}")
        proc, job, held = _spawn_contained(spec, argv, env)
    except (OSError, ValueError) as exc:
        forwarder.add(f"[launcher] could not start the app: {exc}")
        return 1
    if held:
        forwarder.add(f"[launcher] {held}")
    try:
        pump = threading.Thread(target=_pump, args=(proc, forwarder), daemon=True)
        pump.start()
        while proc.poll() is None:
            if _STOP.wait(0.5):
                said = _ask_to_stop(proc)
                if said:
                    forwarder.add(f"[launcher] {said}")
                try:
                    proc.wait(STOP_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    forwarder.add(
                        f"[launcher] the app did not stop within {int(STOP_GRACE_SECONDS)} s; "
                        "ending it"
                    )
                    said = _force(proc)
                    if said:
                        forwarder.add(f"[launcher] {said}")
                    try:
                        proc.wait(10)
                    except subprocess.TimeoutExpired:
                        forwarder.add(f"[launcher] the app's process {proc.pid} is still running")
                break
        pump.join(5)
    finally:
        if job is not None:
            # Whatever the app started and left running ends here, while
            # this launcher can still say so.
            left = job.end()
            if left:
                forwarder.add(
                    f"[launcher] {left} process{'es' if left > 1 else ''} the app started "
                    f"{'were' if left > 1 else 'was'} still running after it; ended"
                )
            job.close()
    if spec.get("linkProvider"):
        try:
            pg0_cleanup(spec["argv"][0], "stop", env)
        except (OSError, ValueError) as exc:
            forwarder.add(f"[launcher] {exc}")
            return 1
    code = proc.returncode if proc.returncode is not None else 1
    if not _STOP.is_set() and code != 0:
        forwarder.add(f"[launcher] the app exited with code {code}")
    # A stop that was asked for is not a failure, whatever the app's code.
    return 0 if _STOP.is_set() else code


# --------------------------------------------------------------------------- #
# Windows: being a service, with ctypes and nothing else
# --------------------------------------------------------------------------- #


def _windows_service(spec: dict) -> int:
    if sys.platform != "win32":
        raise RuntimeError("only a Windows service is run this way")
    import ctypes
    from ctypes import wintypes

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)

    class _Status(ctypes.Structure):
        _fields_ = [
            ("dwServiceType", wintypes.DWORD),
            ("dwCurrentState", wintypes.DWORD),
            ("dwControlsAccepted", wintypes.DWORD),
            ("dwWin32ExitCode", wintypes.DWORD),
            ("dwServiceSpecificExitCode", wintypes.DWORD),
            ("dwCheckPoint", wintypes.DWORD),
            ("dwWaitHint", wintypes.DWORD),
        ]

    main_type = ctypes.WINFUNCTYPE(None, wintypes.DWORD, ctypes.POINTER(wintypes.LPWSTR))
    handler_type = ctypes.WINFUNCTYPE(
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID
    )

    class _Entry(ctypes.Structure):
        _fields_ = [("lpServiceName", wintypes.LPWSTR), ("lpServiceProc", main_type)]

    own_process, running, stop_pending, stopped = 0x10, 4, 3, 1
    accept_stop_shutdown = 0x1 | 0x4
    service_specific_error = 1066
    advapi.RegisterServiceCtrlHandlerExW.restype = wintypes.HANDLE
    advapi.RegisterServiceCtrlHandlerExW.argtypes = [
        wintypes.LPCWSTR,
        handler_type,
        wintypes.LPVOID,
    ]
    advapi.SetServiceStatus.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Status)]
    name = spec.get("service") or f"EugenePlexusApp-{spec.get('app', 'app')}"
    outcome = {"code": 0}
    handle = {"value": None}

    def report(state: int, exit_code: int = 0, wait_hint: int = 0) -> None:
        status = _Status(
            own_process, state, 0 if state != running else accept_stop_shutdown, 0, 0, 0, wait_hint
        )
        if exit_code:
            status.dwWin32ExitCode = service_specific_error
            status.dwServiceSpecificExitCode = exit_code & 0xFFFFFFFF
        advapi.SetServiceStatus(handle["value"], ctypes.byref(status))

    def handler(control: int, _event_type: int, _event_data: object, _context: object) -> int:
        if control in (1, 5):  # SERVICE_CONTROL_STOP, SERVICE_CONTROL_SHUTDOWN
            report(stop_pending, wait_hint=int((STOP_GRACE_SECONDS + 15) * 1000))
            _STOP.set()
        return 0

    handler_ref = handler_type(handler)

    def service_main(_argc: int, _argv: object) -> None:
        handle["value"] = advapi.RegisterServiceCtrlHandlerExW(name, handler_ref, None)
        report(running)
        try:
            outcome["code"] = run(spec)
        except Exception:
            outcome["code"] = 1
        report(stopped, exit_code=outcome["code"])

    main_ref = main_type(service_main)
    table = (_Entry * 2)(_Entry(name, main_ref), _Entry(None, main_type()))
    if not advapi.StartServiceCtrlDispatcherW(table):
        error = ctypes.get_last_error()
        if error == 1063:  # not started by the service manager: run in the foreground
            return run(spec)
        return 1
    return outcome["code"]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--memory-provider"]:
        signal.signal(signal.SIGTERM, lambda *_: _STOP.set())
        signal.signal(signal.SIGINT, lambda *_: _STOP.set())
        return watch_memory_provider(args[1:])
    if len(args) != 1:
        print("usage: app_launcher.py <launch.json>", file=sys.stderr)
        return 2
    spec = load_spec(args[0])
    if sys.platform == "win32":
        return _windows_service(spec)
    signal.signal(signal.SIGTERM, lambda *_: _STOP.set())
    signal.signal(signal.SIGINT, lambda *_: _STOP.set())
    return run(spec)


if __name__ == "__main__":
    sys.exit(main())
