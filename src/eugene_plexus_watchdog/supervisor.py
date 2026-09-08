"""Subprocess supervisor for Eugene Plexus.

Spawns child processes per the topology in `watchdog.yaml`, threads the
right env vars through (config-file path, bind port parsed from the
component's URL, safe-mode flag), and respawns any child that exits.

Supervising things it does not own is the watchdog's whole reason to
exist, and it covers two kinds of child. **Components** are Eugene
Plexus processes — the gateway, the drivers — spawned as
`sys.executable -m <module>` from `_COMPONENT_SPECS`. **Runtimes** are
third-party engine binaries spawned from an argv an engine adapter
builds. They share every mechanic in this module (spawn, watch,
respawn, crash back-off, log capture) and none of their declarations,
which is why they are separate surfaces on the API.

## Cross-platform notes

`asyncio.subprocess.Process.terminate()` is the graceful-shutdown signal
we use on every supervised exit. On POSIX that maps to SIGTERM (children
get a chance to flush logs and answer their last in-flight request). On
Windows it's `TerminateProcess`, which is a hard kill — no graceful
window. Living with that for now; Windows is primarily a dev surface,
real installs are Linux/Mac/Docker. This matters more for engines than
for components: a hard-killed `llama-server` can leave a GPU context to
be reclaimed by the driver.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime
from typing import Any, NamedTuple
from urllib.parse import urlparse

import httpx

from . import orphan_kill, security
from ._generated.models import ComponentEntry, ComponentKind, ComponentStatus
from .auth_state import AuthState

# How long an exiting child gets to finish flushing before we SIGKILL it
# during watchdog shutdown. Long enough for a /v1/admin/restart-style
# response body to flush over loopback, short enough that a hung child
# doesn't drag shutdown out for the operator.
_TERM_TIMEOUT_SECONDS = 5.0

# Polling interval for the health-watcher loop. Each spawned component's
# /healthz is hit on this cadence to refresh its status; pick a value
# that's responsive enough for the UI (sub-second after a restart) but
# light enough not to drown the loopback in HTTP traffic.
_HEALTH_POLL_SECONDS = 1.5

# After how many consecutive crashes (non-zero exits) does the watchdog
# stop respawning a component? Without this, a misconfigured driver that
# exits immediately would respawn-storm forever. Operator must POST to
# /v1/components/<name>/restart to clear the crashed state.
_CRASH_BACKOFF_THRESHOLD = 5

# Successful /healthz probes fire every 1.5s per component and contribute
# nothing to debugging — they push real signal out of the scrollback. We
# suppress 2xx healthz lines at the supervisor's output reader (one place,
# applies to every component) while letting 4xx/5xx fall through so a
# component that starts failing its own health checks still shows up.
_HEALTHZ_2XX_LINE = re.compile(r'"GET /healthz HTTP/[^"]+" 2\d\d')

# Color the alert WORDS (not the whole line — red on dark backgrounds is
# unreadable) so a quick scroll-by spots errors and warnings instantly.
# ANSI SGR codes work in every modern terminal: VS Code's integrated
# terminal/output pane, Windows Terminal, modern cmd.exe with VTP. Older
# environments may render the raw escape sequences — set NO_COLOR=1 in
# the env to disable (https://no-color.org/).
_ALERT_WORD_RE = re.compile(r"\b(error|warning|warn)\b", re.IGNORECASE)
_ANSI_RESET = "\x1b[0m"
_ANSI_BY_WORD = {
    "error": "\x1b[31m",  # red
    "warning": "\x1b[33m",  # yellow
    "warn": "\x1b[33m",
}
_USE_COLOR = "NO_COLOR" not in os.environ


def _colorize_alerts(text: str) -> str:
    """Wrap any error/warning word in the matching ANSI color code,
    preserving the original case. No-op when NO_COLOR is set."""
    if not _USE_COLOR:
        return text

    def _wrap(match: re.Match[str]) -> str:
        word = match.group(0)
        return f"{_ANSI_BY_WORD[word.lower()]}{word}{_ANSI_RESET}"

    return _ALERT_WORD_RE.sub(_wrap, text)


# Everything the supervisor needs to know to spawn one component kind.
#
# Was three parallel dicts keyed by the same enum, which is three places
# to forget when a kind is added. One table means adding `library` is one
# line and cannot half-land.
class _ComponentSpec(NamedTuple):
    module: str
    """Spawned as `sys.executable -m <module>`, so components run under
    whichever interpreter the watchdog itself is running. Production
    installs sharing one venv work out of the box; dev setups with
    per-component venvs must install every component into the
    watchdog's venv (or a shared one) or the import fails at spawn."""

    env_prefix: str
    """Matches the component's own pydantic-settings `env_prefix`."""

    log_label: str
    """Short, kind-identifying label for the output reader. Operators can
    name a component anything, so a bare `[left]` prefix is ambiguous
    about what kind of thing is talking. When the operator's name differs
    from this label the prefix becomes `[<label>: <name>]` (e.g.
    `[driver: left]`); when they match we keep the shorter `[<name>]` to
    avoid `[gateway: gateway]`-style redundancy."""


_COMPONENT_SPECS: dict[ComponentKind, _ComponentSpec] = {
    ComponentKind.gateway: _ComponentSpec(
        module="eugene_plexus_gateway",
        env_prefix="EUGENE_PLEXUS_GATEWAY",
        log_label="gateway",
    ),
    ComponentKind.inference_driver: _ComponentSpec(
        module="eugene_plexus_inference_driver",
        env_prefix="EUGENE_PLEXUS_DRIVER",
        log_label="driver",
    ),
}


def _format_log_prefix(kind: ComponentKind, name: str) -> str:
    """Build the `[<...>] ` prefix the supervisor stamps on each child
    line. `[<name>]` when the name matches the kind's short label,
    `[<kind>: <name>]` otherwise — see `_ComponentSpec.log_label`."""
    spec = _COMPONENT_SPECS.get(kind)
    short = spec.log_label if spec is not None else kind.value
    if name == short:
        return f"[{name}] "
    return f"[{short}: {name}] "


class SupervisedProcess:
    """One supervised child: its declared topology entry plus live state.

    The supervision loop is a long-running task (`_run`) that spawns the
    child, awaits its exit, applies the back-off rules, and respawns —
    until `stop()` is called or the crash threshold trips.
    """

    def __init__(
        self,
        entry: ComponentEntry,
        log: logging.Logger,
        auth_state: AuthState | None = None,
    ) -> None:
        self.entry = entry
        self._log = log
        # v0.2: auth_state is the source of the per-restart JWT signing
        # key + (post-login) master key + service token issuance. Optional
        # for test ergonomics — supervisor tests that don't care about
        # auth can pass None and get the v0.1 env-var set only.
        self._auth_state = auth_state
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_requested = False
        self._consecutive_crashes = 0
        # Set to True when the supervisor decides to fall back to safe
        # mode after the crash threshold trips — see `_run`. Subsequent
        # spawns force-set SAFE_MODE=1 in env regardless of the
        # topology's `safeMode` flag, so /v1/config stays reachable for
        # operator repair. Cleared on manual restart() (operator wants
        # to try normal mode again after fixing the underlying issue).
        self._auto_safe_mode_engaged = False

        self.status: ComponentStatus = ComponentStatus.starting
        self.last_error: str | None = None
        self.last_restart: datetime | None = None

    # --- public lifecycle --------------------------------------------------

    def start(self) -> None:
        """Kick off the supervision loop. Returns immediately; the loop
        runs in the background until `stop()` is called."""
        self._stop_requested = False
        self._consecutive_crashes = 0
        self._task = asyncio.create_task(self._run(), name=f"supervise:{self.entry.name}")

    async def restart(self) -> None:
        """SIGTERM the child; the supervision loop respawns it. Clears
        the crash counter AND the auto-safe-mode flag so a manual
        restart returns the component to normal-mode operation —
        operator's intent on hitting Restart is "try again with the
        config I just fixed", not "stay in safe mode forever."
        """
        self._consecutive_crashes = 0
        self._auto_safe_mode_engaged = False
        proc = self._proc
        if proc is not None and proc.returncode is None:
            self._log.info(
                "restart requested for %s; terminating pid %d",
                self.entry.name,
                proc.pid,
            )
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()

    async def stop(self) -> None:
        """Stop the supervision loop and ensure the child is dead.
        Terminates with a timeout, escalates to kill if the child hangs."""
        self._stop_requested = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=_TERM_TIMEOUT_SECONDS)
            except TimeoutError:
                self._log.warning(
                    "%s did not exit within %.1fs of terminate; killing",
                    self.entry.name,
                    _TERM_TIMEOUT_SECONDS,
                )
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(BaseException):
                    await proc.wait()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._task

    # --- introspection (read-only properties for the routes layer) ---------

    @property
    def pid(self) -> int | None:
        proc = self._proc
        return proc.pid if proc is not None and proc.returncode is None else None

    # --- internals ---------------------------------------------------------

    async def _pipe_child_output(self, stream: asyncio.StreamReader | None) -> None:
        """Drain a child's stdout/stderr pipe and re-emit each line with
        the component name prefix. Runs as a background task per spawn;
        exits when the pipe closes (child terminated) or it's cancelled.

        Bytes are decoded with `errors="replace"` so a child that writes
        non-UTF-8 to stdout (rare but possible — a Windows native CRT
        diagnostic, say) doesn't kill the reader and leave the watchdog
        deaf to subsequent output.
        """
        if stream is None:
            return
        prefix = _format_log_prefix(self.entry.kind, self.entry.name)
        try:
            while True:
                line = await stream.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace")
                # Suppress successful /healthz access logs — every spawned
                # component answers a probe every ~1.5s, dwarfing the rest
                # of the log. Non-2xx healthz still passes through so a
                # newly-unhealthy component is visible.
                if _HEALTHZ_2XX_LINE.search(text):
                    continue
                # Colorize "error" / "warning" / "warn" inline so the
                # important lines pop on a fast scroll. Word-level only —
                # full-line color is unreadable on dark terminals.
                text = _colorize_alerts(text)
                # Lines from `readline()` include the trailing newline;
                # use `end=""` so we don't double it. flush=True keeps
                # output snappy even when watchdog stdout is itself a pipe
                # (e.g. running under a VS Code task with output capture).
                print(prefix + text, end="", flush=True)
        except asyncio.CancelledError:
            return
        except Exception as e:
            # Never let a reader crash bring down the supervision loop;
            # the worst-case fallback is "we lose log prefixing for this
            # child", which is strictly better than the watchdog dying.
            self._log.warning("output-pipe reader for %s crashed: %s", self.entry.name, e)

    async def _run(self) -> None:
        """Spawn-watch-respawn loop.

        Two-stage backoff: the FIRST time the crash threshold trips
        (component bad-config or similar), we automatically fall back
        to SAFE MODE so the operator can repair via /v1/config without
        having to know about env vars or YAML edits. The second time
        the threshold trips (safe mode itself is crashing — config
        endpoint is unreachable; this should be rare and indicates a
        real bug rather than bad operator config), we give up for real.
        Exits cleanly on `stop_requested`.
        """
        while not self._stop_requested:
            await self._spawn_once()
            if self._stop_requested:
                return
            if self._consecutive_crashes >= _CRASH_BACKOFF_THRESHOLD:
                if not self._auto_safe_mode_engaged:
                    # First trip: engage auto-safe-mode, reset the
                    # counter, continue. Next spawn forces SAFE_MODE=1
                    # regardless of topology so /v1/config is reachable.
                    self._log.error(
                        "%s crashed %d times in a row; falling back to "
                        "SAFE MODE so /v1/config stays reachable for "
                        "repair (last error: %s). The component will "
                        "respawn with SAFE_MODE=1 — UI Components tab "
                        "will show the safe_mode badge; fix config "
                        "there and Restart to return to normal mode.",
                        self.entry.name,
                        self._consecutive_crashes,
                        self.last_error,
                    )
                    self._auto_safe_mode_engaged = True
                    self._consecutive_crashes = 0
                else:
                    # Second trip: safe mode itself can't boot. This is
                    # a real bug (or fundamentally broken environment) —
                    # the operator can't recover via the UI because
                    # the config endpoint isn't up. Give up for real.
                    self._log.error(
                        "%s crashed %d times in a row even in SAFE MODE "
                        "(last error: %s); giving up. POST /v1/components"
                        "/%s/restart to reset after fixing whatever is "
                        "preventing safe-mode startup.",
                        self.entry.name,
                        self._consecutive_crashes,
                        self.last_error,
                        self.entry.name,
                    )
                    self.status = ComponentStatus.crashed
                    return
            await asyncio.sleep(min(2.0 * self._consecutive_crashes, 10.0))

    async def _spawn_once(self) -> None:
        """One spawn / wait / mark-status iteration."""
        spawn = self.entry.spawn
        if spawn is None:
            self._log.warning("%s has no spawn block; skipping", self.entry.name)
            self.status = ComponentStatus.unreachable
            return

        spec = _COMPONENT_SPECS.get(self.entry.kind)
        if spec is None:
            self._log.error(
                "%s has kind %r, which this watchdog has no spawn spec for; "
                "refusing to spawn. Either the topology names a retired "
                "component kind or this watchdog predates it.",
                self.entry.name,
                self.entry.kind.value,
            )
            self.status = ComponentStatus.crashed
            self.last_error = f"no spawn spec for kind {self.entry.kind.value!r}"
            self._consecutive_crashes += 1
            return

        env = os.environ.copy()
        prefix = spec.env_prefix
        env[f"{prefix}_CONFIG_FILE"] = str(spawn.configFile)
        port = urlparse(str(self.entry.url)).port
        if port is not None:
            env[f"{prefix}_BIND_PORT"] = str(port)
        # Two sources of "boot in safe mode": the operator's explicit
        # topology toggle (`ComponentEntry.safeMode`) AND the
        # supervisor's auto-fallback after the crash threshold (see
        # `_run`). Either one forces SAFE_MODE=1 in env.
        safe_mode_effective = self.entry.safeMode or self._auto_safe_mode_engaged
        env[f"{prefix}_SAFE_MODE"] = "1" if safe_mode_effective else "0"

        # v0.2 auth env vars. Children that have implemented the v0.2
        # auth surface (currently watchdog itself; gateway/drivers
        # memory follow in subsequent commits) read these to (a) validate
        # inbound bearer tokens against the shared signing key, (b)
        # present a service token of their own on outbound calls, and
        # (c) decrypt at-rest secrets like apiKey. Children unaware of
        # these env vars simply ignore them — fully backward-compatible
        # rollout.
        if self._auth_state is not None:
            kind_value = self.entry.kind.value  # "gateway", "inference-driver"
            env[f"{prefix}_AUTH_SIGNING_KEY"] = base64.b64encode(
                self._auth_state.signing_key
            ).decode("ascii")
            env[f"{prefix}_SERVICE_TOKEN"] = security.issue_service_token(
                signing_key=self._auth_state.signing_key,
                kind=kind_value,
            )
            if self._auth_state.master_key is not None:
                env[f"{prefix}_MASTER_KEY"] = base64.b64encode(self._auth_state.master_key).decode(
                    "ascii"
                )
            else:
                # Be explicit about absence so a child running stale env
                # from a previous shell can't pick up an unrelated value.
                env.pop(f"{prefix}_MASTER_KEY", None)

        if spawn.env:
            env.update({k: str(v) for k, v in spawn.env.items()})

        # Force unbuffered Python output. Without this, redirecting the
        # child's stdout to a pipe (below) makes Python switch to block-
        # buffered mode, so child log lines arrive in 4KB chunks instead
        # of immediately — exactly when you want the opposite (debugging
        # a hang where ANY line emitted before the stall is the clue).
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [sys.executable, "-m", spec.module]
        self._log.info("spawning %s: %s", self.entry.name, " ".join(cmd))

        try:
            # Pipe stdout + stderr through us so we can prefix every line
            # with `[<name>]`. Without this the watchdog inherits the
            # parent terminal and child output interleaves with no source
            # identification — making "Waiting for application startup"
            # ambiguous when several children are booting concurrently.
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                **orphan_kill.kwargs_for_platform(),
            )
        except OSError as e:
            self._log.error("failed to spawn %s: %s", self.entry.name, e)
            self.status = ComponentStatus.crashed
            self.last_error = f"spawn failed: {e}"
            self._consecutive_crashes += 1
            return

        # Windows: assign to the watchdog's Job Object so the OS reaps
        # the child if the watchdog dies hard. POSIX uses preexec_fn
        # (handled in kwargs above), no post-spawn step needed.
        win_job = orphan_kill.windows_job()
        if win_job is not None and self._proc.pid is not None:
            win_job.assign(self._proc.pid)

        self.status = ComponentStatus.safe_mode if safe_mode_effective else ComponentStatus.starting
        self.last_restart = datetime.now(UTC)
        self.last_error = None

        # Background reader: drains the child's stdout pipe and re-emits
        # each line with a `[<name>]` prefix on the watchdog's own stdout.
        # MUST be running before we await proc.wait() or the child can
        # block writing into a full pipe buffer and never exit.
        reader_task = asyncio.create_task(
            self._pipe_child_output(self._proc.stdout),
            name=f"output-pipe:{self.entry.name}",
        )

        try:
            return_code = await self._proc.wait()
        finally:
            # Give the reader a moment to drain any final lines the child
            # wrote on its way out, then cancel if it's still hung.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(reader_task, timeout=1.0)
            if not reader_task.done():
                reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, BaseException):
                    await reader_task
        self._proc = None

        if self._stop_requested:
            self.status = ComponentStatus.exited
            return

        if return_code == 0:
            self._log.info("%s exited cleanly (rc=0); respawning", self.entry.name)
            self.status = ComponentStatus.exited
            self._consecutive_crashes = 0
        else:
            self._consecutive_crashes += 1
            self.last_error = f"exited with code {return_code}"
            self._log.warning(
                "%s exited rc=%d (consecutive crashes: %d)",
                self.entry.name,
                return_code,
                self._consecutive_crashes,
            )
            self.status = ComponentStatus.crashed


class Supervisor:
    """Owns every supervised child plus a background health-poll task.

    Two responsibilities:
      1. Lifecycle of the SupervisedProcess collection — add, remove,
         restart, stop_all.
      2. Periodic /healthz polling so the routes layer can report
         `running` vs `safe_mode` distinctly from the raw process state
         (a child can be alive but in safe mode, etc.).
    """

    def __init__(
        self,
        log: logging.Logger | None = None,
        auth_state: AuthState | None = None,
    ) -> None:
        self._log = log or logging.getLogger(__name__)
        # v0.2: shared with every SupervisedProcess so each spawn can
        # issue a fresh service token, base64-encode the signing key,
        # and forward the (possibly-still-None) master key. Optional —
        # absent for tests that don't care about auth.
        self._auth_state = auth_state
        self._processes: dict[str, SupervisedProcess] = {}
        self._health_task: asyncio.Task[None] | None = None
        self._health_client: httpx.AsyncClient | None = None
        # Maps component name -> True iff /healthz reported safeMode=true.
        # Drives the running-vs-safe_mode distinction on Component.status.
        self._safe_mode_observed: dict[str, bool] = {}
        # Maps component name -> True iff /healthz returned 2xx in the
        # last poll. Lets us promote `starting` -> `running` once the
        # child is actually serving requests.
        self._reachable: dict[str, bool] = {}

    # --- collection management --------------------------------------------

    def add_and_start(self, entry: ComponentEntry) -> None:
        """Begin supervising a topology entry. No-op for remote entries
        (`spawn is None`); they're tracked for health-polling only."""
        if entry.name in self._processes:
            return
        if entry.spawn is None:
            # Remote: register a placeholder so /healthz polling tracks
            # reachability, but no SupervisedProcess.
            self._reachable[entry.name] = False
            return
        sp = SupervisedProcess(entry, self._log, auth_state=self._auth_state)
        self._processes[entry.name] = sp
        sp.start()

    async def remove_and_stop(self, name: str) -> None:
        sp = self._processes.pop(name, None)
        self._safe_mode_observed.pop(name, None)
        self._reachable.pop(name, None)
        if sp is not None:
            await sp.stop()

    async def restart(self, name: str) -> bool:
        sp = self._processes.get(name)
        if sp is None:
            return False
        await sp.restart()
        return True

    async def restart_all(self) -> list[str]:
        """SIGTERM every supervised child so their supervision loops
        respawn them. Used after the operator logs in: children that
        were already spawned ran with no MASTER_KEY env var (the
        operator hadn't unlocked yet); the respawn picks up the now-
        populated key from the shared AuthState.

        Returns the list of component names that were signaled — empty
        if no children are running yet (e.g. login before topology has
        anything spawned). Best-effort: per-process restart failures
        are logged but never raised; the operator's login flow
        shouldn't error out because one supervised child wedged.
        """
        names = list(self._processes.keys())
        if not names:
            return []
        self._log.info(
            "restart_all: signaling %d supervised process(es) to respawn — "
            "typically because master key just became available",
            len(names),
        )
        results = await asyncio.gather(
            *(self._processes[n].restart() for n in names),
            return_exceptions=True,
        )
        for name, result in zip(names, results, strict=False):
            if isinstance(result, BaseException):
                self._log.warning("restart_all: %s failed to restart: %s", name, result)
        return names

    async def start_health_loop(self, get_components: Any) -> None:
        """Start the background /healthz polling task. `get_components`
        is a 0-arg callable returning the current ComponentEntry list,
        so the loop sees fresh entries when topology changes."""
        if self._health_task is not None:
            return
        # Short timeout so a hung child doesn't block the whole poll round.
        self._health_client = httpx.AsyncClient(timeout=2.0)
        self._health_task = asyncio.create_task(
            self._health_loop(get_components), name="supervisor-health"
        )

    async def stop_all(self) -> None:
        """Shut everything down. Cancels the health loop, terminates and
        waits for every supervised process. Best-effort — never raises."""
        if self._health_task is not None:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._health_task
            self._health_task = None
        if self._health_client is not None:
            with contextlib.suppress(BaseException):
                await self._health_client.aclose()
            self._health_client = None

        await asyncio.gather(
            *(sp.stop() for sp in self._processes.values()),
            return_exceptions=True,
        )
        self._processes.clear()
        self._safe_mode_observed.clear()
        self._reachable.clear()

    # --- introspection (read by the routes layer) -------------------------

    def status_for(
        self, name: str, *, has_spawn: bool
    ) -> tuple[ComponentStatus, str | None, datetime | None, int | None]:
        """Return (status, lastError, lastRestart, pid) for one component.

        For spawned children: derives status from process state +
        /healthz observations (running, starting, safe_mode, exited,
        crashed).

        For remote entries (`has_spawn=False`): derives from the last
        /healthz poll only — `running` if the URL is answering,
        `unreachable` otherwise.
        """
        sp = self._processes.get(name)
        if sp is not None:
            base = sp.status
            if base == ComponentStatus.starting and self._reachable.get(name, False):
                base = (
                    ComponentStatus.safe_mode
                    if self._safe_mode_observed.get(name)
                    else ComponentStatus.running
                )
            return base, sp.last_error, sp.last_restart, sp.pid
        if not has_spawn:
            reachable = self._reachable.get(name, False)
            status = ComponentStatus.running if reachable else ComponentStatus.unreachable
            return status, None, None, None
        return ComponentStatus.unreachable, None, None, None

    # --- internals --------------------------------------------------------

    async def _health_loop(self, get_components: Any) -> None:
        """Hit `/healthz` on every known component once per
        `_HEALTH_POLL_SECONDS`. Records reachability + observed safe-mode
        flag; never mutates `SupervisedProcess` state directly."""
        try:
            while True:
                entries: list[ComponentEntry] = get_components()
                await asyncio.gather(
                    *(self._poll_one(e) for e in entries),
                    return_exceptions=True,
                )
                await asyncio.sleep(_HEALTH_POLL_SECONDS)
        except asyncio.CancelledError:
            return

    async def _poll_one(self, entry: ComponentEntry) -> None:
        client = self._health_client
        if client is None:
            return
        url = str(entry.url).rstrip("/") + "/healthz"
        start = time.monotonic()
        try:
            response = await client.get(url)
        except httpx.HTTPError:
            self._reachable[entry.name] = False
            return
        finally:
            # Defensive: an unexpectedly slow probe shouldn't cascade
            # into a long poll round.
            elapsed = time.monotonic() - start
            if elapsed > _HEALTH_POLL_SECONDS:
                self._log.debug(
                    "healthz probe for %s took %.2fs (poll interval %.2fs)",
                    entry.name,
                    elapsed,
                    _HEALTH_POLL_SECONDS,
                )

        self._reachable[entry.name] = response.is_success
        if response.is_success:
            with contextlib.suppress(ValueError):
                body = response.json()
                self._safe_mode_observed[entry.name] = bool(body.get("safeMode"))
