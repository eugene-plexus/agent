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
  runs, so the two ways an app runs cannot fill a placeholder two ways;
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
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

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
    return spec


def child_environment(spec: dict) -> dict[str, str]:
    """This process's environment (the service manager's, for this
    account), the spec's variables, and where things are."""
    env = dict(os.environ)
    # Never hand the app what systemd handed the launcher about itself.
    for name in ("CREDENTIALS_DIRECTORY", "STATE_DIRECTORY", "INVOCATION_ID", "NOTIFY_SOCKET"):
        env.pop(name, None)
    env.update({str(k): str(v) for k, v in (spec.get("env") or {}).items()})
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
        "dataDir",
        "gatewayUrl",
        "appUrl",
        "oidcIssuer",
        "oidcClientId",
        "clientKey",
        "oidcClientSecret",
        "appSecret",
    }
)
#: Read by `render_start` from files, never carried in a spec: a value for
#: one of these in the spec's `values` is ignored.
SECRET_PLACEHOLDERS = frozenset({"clientKey", "oidcClientSecret", "appSecret"})
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: `{appSecret}`: made at the app's first start, the same at every start after.
APP_SECRET_FILE = "app_secret"
#: A hash of the gateway's address and the app's key at the last start, for
#: `resetOnConnectionChange`. A hash, so the file holds no key.
CONNECTION_FILE = "connection.sha256"


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
    path = os.path.join(data_dir, APP_SECRET_FILE)
    existing = _secret(path)
    if existing:
        return existing
    os.makedirs(data_dir, exist_ok=True)
    value = secrets.token_urlsafe(32)
    _write_own(path, value)
    return value


def connection_changed(data_dir: str, gateway_url: str | None, client_key: str | None) -> bool:
    """Whether the gateway's address or the app's key differs from the last
    start's, recording this start's either way. A first start has nothing to
    differ from, so it is not a change."""
    digest = hashlib.sha256(f"{gateway_url or ''}\n{client_key or ''}".encode()).hexdigest()
    path = os.path.join(data_dir, CONNECTION_FILE)
    try:
        previous = Path(path).read_text(encoding="utf-8").strip() or None
    except OSError:
        previous = None
    if previous != digest:
        os.makedirs(data_dir, exist_ok=True)
        _write_own(path, digest)
    return previous is not None and previous != digest


def render_start(spec: dict, env: dict[str, str]) -> tuple[list[str], dict[str, str], list[str]]:
    """One start's command and environment, and what is worth saying about it.

    The command is `spec["argv"]` followed by the manifest's `args`; the
    environment is `env` with the manifest's own variables on top. Every
    placeholder is filled in from `spec["values"]` (what the agent may
    write down), `spec["dataDir"]`, and the secrets read here from the
    files the spec names. A variable whose placeholder has no value this
    start -- no gateway found, no sign-in client -- is left unset rather
    than set to half a value, and the returned lines say so; an argument
    gets an empty string, so the arguments after it keep their places.

    With `resetOnConnectionChange`, that variable is `true` for a start
    whose connection differs from the last one's and absent otherwise,
    even when the environment it inherits has it.
    """
    data = str(spec.get("dataDir") or "")
    values = {
        str(k): str(v)
        for k, v in (spec.get("values") or {}).items()
        if v is not None and k in PLACEHOLDERS and k not in SECRET_PLACEHOLDERS
    }
    if data:
        values["dataDir"] = data
    args = [str(a) for a in spec.get("args") or []]
    environment = {str(k): str(v) for k, v in (spec.get("environment") or {}).items()}
    reset = spec.get("resetOnConnectionChange")
    wanted = {n for text in [*args, *environment.values()] for n in placeholders(text)}

    key = _secret(spec.get("keyFile")) if "clientKey" in wanted or reset else None
    if key:
        values["clientKey"] = key
    if "oidcClientSecret" in wanted:
        oidc = _secret(spec.get("oidcSecretFile"))
        if oidc:
            values["oidcClientSecret"] = oidc
    if "appSecret" in wanted and data:
        values["appSecret"] = app_secret(data)

    notes: list[str] = []
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
        if data and connection_changed(data, values.get("gatewayUrl"), key):
            out[reset] = "true"
            notes.append(
                "Eugene's connection details changed since the last start (the gateway's "
                f"address or this app's key), so {reset}=true for this start: the app takes "
                "the new ones, and settings changed inside it go back to their defaults."
            )
    return argv, out, notes


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


def _spawn(spec: dict, argv: list[str], env: dict[str, str]) -> subprocess.Popen:
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
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


def _ask_to_stop(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        import ctypes

        # CTRL_BREAK_EVENT to the app's own process group: the stop the
        # agent's supervisor sends a child, which every component and
        # engine answers by shutting down cleanly. A service has no
        # console to send it through until `run` allocates one.
        if not ctypes.windll.kernel32.GenerateConsoleCtrlEvent(1, proc.pid):
            _force(proc)
    else:
        proc.send_signal(signal.SIGTERM)


def _force(proc: subprocess.Popen) -> None:
    if WINDOWS:
        # The app's interpreter may be a launcher with the real one as its
        # child; end the whole tree.
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        proc.kill()


def run(spec: dict) -> int:
    """Start the app, forward its output, wait for it or for a stop."""
    if sys.platform == "win32":
        import ctypes

        # A console of its own (session 0, never shown), so a stop can be
        # sent to the app the way the supervisor sends one. The app gets a
        # process group of its own, so the break reaches it and not this.
        ctypes.windll.kernel32.AllocConsole()
    forwarder = Forwarder(spec.get("ingress"), spec.get("keyFile"), spec.get("app", "app"))
    forwarder.start()
    try:
        argv, env, notes = render_start(spec, child_environment(spec))
        for note in notes:
            forwarder.add(f"[launcher] {note}")
        proc = _spawn(spec, argv, env)
    except (OSError, ValueError) as exc:
        forwarder.add(f"[launcher] could not start the app: {exc}")
        forwarder.close()
        return 1
    pump = threading.Thread(target=_pump, args=(proc, forwarder), daemon=True)
    pump.start()
    while proc.poll() is None:
        if _STOP.wait(0.5):
            _ask_to_stop(proc)
            try:
                proc.wait(STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                forwarder.add(
                    f"[launcher] the app did not stop within {int(STOP_GRACE_SECONDS)} s; ending it"
                )
                _force(proc)
                proc.wait(10)
            break
    pump.join(5)
    code = proc.returncode if proc.returncode is not None else 1
    if not _STOP.is_set() and code != 0:
        forwarder.add(f"[launcher] the app exited with code {code}")
    forwarder.close()
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
