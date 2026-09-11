"""Subprocess supervisor for Eugene Plexus.

Spawns child processes per the topology in `agent.yaml`, threads the
right env vars through (config-file path, bind port parsed from the
component's URL, safe-mode flag), and respawns any child that exits.

Supervising things it does not own is the agent's whole reason to
exist, and it covers two kinds of child. **Components** are Eugene
Plexus processes — the gateway, the drivers — spawned as
`sys.executable -m <module>` from `_COMPONENT_SPECS`. **Runtimes** are
third-party engine binaries spawned from an argv an engine adapter
builds. They share every mechanic in this module (spawn, watch,
respawn, crash back-off, log capture) and none of their declarations,
which is why they are separate surfaces on the API.

## Cross-platform notes

Stopping a child is `process_signals.request_stop` on every platform:
SIGTERM on POSIX, `CTRL_BREAK_EVENT` into the child's own process group
on Windows. **The old note here said Windows was "primarily a dev
surface, real installs are Linux/Mac/Docker" and that its hard kill was
being lived with.** Neither survives: Windows is a first-class target by
decision (install-paths §11.1), and the hard kill was measured to be the
only stop that skips a component's ASGI lifespan shutdown. See
`process_signals` for the measurements and the three constraints they
imposed — chief among them that **a graceful request can be ignored**,
so every stop here carries an escalation deadline.
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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, NamedTuple, Protocol
from urllib.parse import urlparse

import httpx

from . import orphan_kill, ports, process_signals, security
from ._generated.models import ComponentEntry, ComponentKind, ComponentStatus
from .auth_state import AuthState

# How long an exiting child gets to finish flushing before we SIGKILL it
# during agent shutdown. Long enough for a /v1/admin/restart-style
# response body to flush over loopback, short enough that a hung child
# doesn't drag shutdown out for the operator.
_TERM_TIMEOUT_SECONDS = 5.0

# Polling interval for the health-watcher loop. Each spawned component's
# /healthz is hit on this cadence to refresh its status; pick a value
# that's responsive enough for the UI (sub-second after a restart) but
# light enough not to drown the loopback in HTTP traffic.
_HEALTH_POLL_SECONDS = 1.5

# After how many consecutive crashes (non-zero exits) does the agent
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
#: How many of a child's most recent output lines to keep for
#: explaining a non-zero exit. Enough to hold a Python traceback plus
#: the lines around it, small enough to be free. The log has everything;
#: this is only what `explain_exit` gets to read.
_OUTPUT_TAIL_LINES = 80

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
    whichever interpreter the agent itself is running. Production
    installs sharing one venv work out of the box; dev setups with
    per-component venvs must install every component into the
    agent's venv (or a shared one) or the import fails at spawn."""

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
    ComponentKind.library: _ComponentSpec(
        module="eugene_plexus_library",
        env_prefix="EUGENE_PLEXUS_LIBRARY",
        log_label="library",
    ),
    # The control root is supervised like anything else, and that is the
    # point of the M5 split rather than an accident of it: if the control
    # root supervised processes it would need a second copy of this
    # machinery, and "components share schemas, not code" means a real
    # second copy. One supervisor implementation, running on every host;
    # one control root, spawned by whichever agent's topology declares it.
    #
    # It is the one component that receives no auth trio. It *is* the
    # trust root — it mints the signing key and derives the master key
    # from the operator's passphrase — so threading a key into it would
    # be this process handing the trust root a key the trust root is
    # supposed to own. See `_ComponentPlanner.plan`.
    ComponentKind.control: _ComponentSpec(
        module="eugene_plexus_control",
        env_prefix="EUGENE_PLEXUS_CONTROL",
        log_label="control",
    ),
}

# Kinds that mint their own credentials rather than being handed ours.
# A set of one today, and a set rather than an `if` because the question
# it answers — "does this child get the auth trio?" — is about the kind,
# not about this particular kind.
_TRUST_ROOT_KINDS: frozenset[ComponentKind] = frozenset({ComponentKind.control})


def _format_log_prefix(kind: ComponentKind, name: str) -> str:
    """Build the `[<...>] ` prefix the supervisor stamps on each child
    line. `[<name>]` when the name matches the kind's short label,
    `[<kind>: <name>]` otherwise — see `_ComponentSpec.log_label`."""
    spec = _COMPONENT_SPECS.get(kind)
    short = spec.log_label if spec is not None else kind.value
    if name == short:
        return f"[{name}] "
    return f"[{short}: {name}] "


class ProcessState(StrEnum):
    """State of one supervision loop, independent of *what* it supervises.

    Deliberately neither `ComponentStatus` nor `RuntimeStatus`. Both of
    those are wire enums carrying members this loop has no opinion about:
    `safe_mode` is a component's config concern, `loading` is an engine's
    readiness concern, `unreachable` describes remote entries the loop
    never spawns at all. Each API surface maps this state plus its own
    observations onto its own enum, which is what lets one loop supervise
    both a Python module and a foreign binary.
    """

    starting = "starting"
    """Spawned, but not yet observed doing useful work. Whoever maps this
    decides what "useful" means — a component answering /healthz, an
    engine finishing its model load."""

    exited = "exited"
    """Exited cleanly (rc=0); a respawn is in flight. Transient."""

    crashed = "crashed"
    """Spawn failed, or the child exited non-zero. Becomes terminal once
    the loop gives up (see `SpawnPlanner.on_crash_threshold`)."""

    not_spawnable = "not_spawnable"
    """Nothing to launch — the declaration describes something this
    agent does not own, so there is no process and no error."""


class SpawnPlanError(Exception):
    """The declaration is present but cannot be turned into a launch.

    Distinct from `plan()` returning None: None means "nothing to launch
    here, and that's fine" (a remote entry), while this means "you asked
    for something I can't build", which counts as a crash.
    """


@dataclass(frozen=True)
class SpawnPlan:
    """One resolved launch. Everything `SupervisedProcess` needs and
    nothing it has to interpret."""

    argv: list[str]
    """Exact command line. Built by the planner, never assembled here —
    this is what makes an engine binary and `python -m <module>` the same
    kind of thing to the loop."""

    env: dict[str, str]
    """Complete environment for the child, already merged."""

    cwd: str | None = None
    """Working directory, or None to inherit the agent's."""

    degraded: bool = False
    """True when this plan deliberately launches a reduced mode (a
    component's safe mode). Surfaced back out so the mapping layer can
    report it; the loop itself only records it."""

    port: int | None = None
    """The TCP port this child will try to bind, when the planner knows
    it. Used only to say something useful when the bind fails: a child
    whose port is taken exits instantly with an OS error, and the loop's
    answer to that is a crash-backoff cycle that names neither the port
    nor what is holding it. Optional because not every plan has one."""


class SpawnPlanner(Protocol):
    """Knows what to launch, and what to do when launching keeps failing.

    The two methods beyond `plan()` exist because recovery policy is
    kind-specific: a component can fall back to safe mode so its config
    endpoint stays reachable, whereas an engine that will not start has
    nothing equivalent to fall back to.
    """

    @property
    def name(self) -> str:
        """Operator-facing name, used in logs and task names."""

    @property
    def log_prefix(self) -> str:
        """The `[...] ` stamp for each captured output line."""

    def plan(self) -> SpawnPlan | None:
        """Build the next launch. None means nothing to launch.

        Raises `SpawnPlanError` if the declaration is unusable.
        """

    def on_crash_threshold(self) -> bool:
        """Called when consecutive crashes hit the threshold.

        Return True to say "I've changed something, keep trying" (the
        crash counter is reset); False to give up.
        """

    def reset(self) -> None:
        """Operator asked for a manual restart — drop any degraded state
        so the next plan is a normal one."""

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        """A better `lastError` than "exited with code N", or None.

        Offered a bounded tail of the child's own output on every
        non-zero exit. Returning None keeps the generic message, which is
        what a planner with nothing to add should do — guessing is worse
        than "exited with code 1", because the operator will believe it.
        """
        return None


class _ComponentPlanner:
    """Launch plans for one Eugene Plexus component.

    Owns the env threading (config path, bind port, safe-mode flag, the
    auth trio) and the safe-mode escalation, neither of which means
    anything to an engine process.
    """

    def __init__(
        self,
        entry: ComponentEntry,
        log: logging.Logger,
        auth_state: AuthState | None = None,
        shared_child_env: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        self.entry = entry
        self._log = log
        # Suffix -> value for env vars every child gets, prefixed per
        # kind at plan time: AGENT_URL (this agent's own local address,
        # whatever port it is on) and BIND_HOST when this node
        # advertises a non-loopback address. Read at every plan so a
        # config change reaches the next spawn.
        self._shared_child_env = shared_child_env
        # auth_state is the source of the per-restart JWT signing key +
        # (post-login) master key + service token issuance. Optional for
        # test ergonomics — tests that don't care about auth pass None
        # and get the env-var set without the auth trio.
        self._auth_state = auth_state
        # Set once the crash threshold trips: subsequent plans force
        # SAFE_MODE=1 regardless of the topology's flag, so /v1/config
        # stays reachable for operator repair.
        self._auto_safe_mode_engaged = False

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def log_prefix(self) -> str:
        return _format_log_prefix(self.entry.kind, self.entry.name)

    def reset(self) -> None:
        """Manual restart returns the component to normal-mode operation.
        The operator's intent on hitting Restart is "try again with the
        config I just fixed", not "stay in safe mode forever"."""
        self._auto_safe_mode_engaged = False

    def plan(self) -> SpawnPlan | None:
        spawn = self.entry.spawn
        if spawn is None:
            # Remote entry: watched for reachability, never launched.
            self._log.warning("%s has no spawn block; skipping", self.entry.name)
            return None

        spec = _COMPONENT_SPECS.get(self.entry.kind)
        if spec is None:
            raise SpawnPlanError(
                f"no spawn spec for kind {self.entry.kind.value!r} — either the "
                f"topology names a retired component kind or this agent "
                f"predates it"
            )

        env = os.environ.copy()
        prefix = spec.env_prefix
        env[f"{prefix}_CONFIG_FILE"] = str(spawn.configFile)
        port = urlparse(str(self.entry.url)).port
        if port is not None:
            env[f"{prefix}_BIND_PORT"] = str(port)

        # Two sources of "boot in safe mode": the operator's explicit
        # topology toggle (`ComponentEntry.safeMode`) AND the auto-
        # fallback after the crash threshold. Either forces SAFE_MODE=1.
        degraded = bool(self.entry.safeMode) or self._auto_safe_mode_engaged
        env[f"{prefix}_SAFE_MODE"] = "1" if degraded else "0"

        # Auth env vars. Children read these to (a) validate inbound
        # bearer tokens against the shared signing key, (b) present a
        # service token of their own on outbound calls, and (c) decrypt
        # at-rest secrets like apiKey.
        #
        # The control root gets none of them, and must not. It is the
        # trust root: it derives the master key from the operator's
        # passphrase and mints the install's signing key itself. Handing
        # it ours would give it a key it did not choose, seal its secrets
        # under a key that dies with this process, and quietly recreate
        # the single-host trust model M5 exists to replace.
        if self._auth_state is not None and self.entry.kind not in _TRUST_ROOT_KINDS:
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

        # Values this agent wants every child to have — the trust root
        # included, since a bind host and an agent URL are bootstrap, not
        # the auth trio. Applied before the operator's `spawn.env`, so an
        # explicit per-component value still wins.
        if self._shared_child_env is not None:
            for suffix, value in self._shared_child_env().items():
                env[f"{prefix}_{suffix}"] = str(value)

        if spawn.env:
            env.update({k: str(v) for k, v in spawn.env.items()})

        # Force unbuffered Python output. Without this, redirecting the
        # child's stdout to a pipe makes Python switch to block-buffered
        # mode, so log lines arrive in 4KB chunks instead of immediately
        # — exactly when you want the opposite (debugging a hang where
        # ANY line emitted before the stall is the clue).
        env["PYTHONUNBUFFERED"] = "1"

        return SpawnPlan(
            argv=[sys.executable, "-m", spec.module],
            env=env,
            degraded=degraded,
            port=port,
        )

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        """Components have nothing engine-specific to add: a component
        that will not start is diagnosed from its own log and its own
        `/v1/config`, both of which the operator already has."""
        return None

    def on_crash_threshold(self) -> bool:
        """Two-stage: fall back to safe mode once, then give up.

        The first trip is almost always bad operator config, and safe
        mode keeps /v1/config reachable so it can be fixed from the UI
        without knowing about env vars or YAML. A second trip means safe
        mode itself won't boot, which is a real bug — and unrecoverable
        from the UI, since the config endpoint is down.
        """
        if not self._auto_safe_mode_engaged:
            self._log.error(
                "%s crashed %d times in a row; falling back to "
                "SAFE MODE so /v1/config stays reachable for "
                "repair. The component will respawn with "
                "SAFE_MODE=1 — UI Components tab will show the "
                "safe_mode badge; fix config there and Restart to "
                "return to normal mode.",
                self.entry.name,
                _CRASH_BACKOFF_THRESHOLD,
            )
            self._auto_safe_mode_engaged = True
            return True

        self._log.error(
            "%s crashed %d times in a row even in SAFE MODE; giving "
            "up. POST /v1/components/%s/restart to reset after fixing "
            "whatever is preventing safe-mode startup.",
            self.entry.name,
            _CRASH_BACKOFF_THRESHOLD,
            self.entry.name,
        )
        return False


class SupervisedProcess:
    """One supervised child: spawn, watch, respawn, back off, capture output.

    Knows nothing about *what* it is launching. A `SpawnPlanner` answers
    that on every iteration, which is what lets the same loop supervise a
    Eugene Plexus component (`python -m <module>`, env-threaded auth) and
    a third-party engine binary (adapter-built argv) without either kind
    leaking into the other.

    The loop is a long-running task (`_run`) that spawns the child, awaits
    its exit, applies the back-off rules, and respawns — until `stop()` is
    called or the planner declines to recover.
    """

    def __init__(self, planner: SpawnPlanner, log: logging.Logger) -> None:
        self._planner = planner
        self._log = log
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_requested = False
        self._consecutive_crashes = 0
        self._last_port: int | None = None
        """The port the current plan asked for, if the planner knew it.
        Read only when explaining a crash."""
        self._expected_exit = False
        """Set when WE asked the child to go. Cleared by the loop when it
        reaps one. Without it a stop request and a crash are the same
        event to the loop, which on Windows they always were — a console
        event produces a non-zero exit code."""

        self.state: ProcessState = ProcessState.starting
        self.degraded = False
        """Whether the plan currently running is a reduced-mode one. Set
        from `SpawnPlan.degraded` at each spawn; read by the mapping
        layer (a component reports `safe_mode`)."""
        self.last_argv: list[str] | None = None
        """The argv of the most recent spawn. Reported read-only on
        `Runtime.argv`: the first question anyone debugging a local engine
        asks is what command actually ran, and every tool that hides the
        answer makes that debugging worse."""
        self.last_error: str | None = None
        # A bounded tail of the child's own output, kept so a non-zero
        # exit can be explained by whoever understands the engine rather
        # than reported as "exited with code 1". Bounded because this is
        # a diagnostic aid, not a log: the log already has everything.
        self._output_tail: deque[str] = deque(maxlen=_OUTPUT_TAIL_LINES)
        self.last_restart: datetime | None = None

    @classmethod
    def for_component(
        cls,
        entry: ComponentEntry,
        log: logging.Logger,
        auth_state: AuthState | None = None,
        shared_child_env: Callable[[], dict[str, str]] | None = None,
    ) -> SupervisedProcess:
        """Supervise one Eugene Plexus component."""
        return cls(_ComponentPlanner(entry, log, auth_state, shared_child_env), log)

    @property
    def name(self) -> str:
        return self._planner.name

    # --- public lifecycle --------------------------------------------------

    def start(self) -> None:
        """Kick off the supervision loop. Returns immediately; the loop
        runs in the background until `stop()` is called."""
        self._stop_requested = False
        self._consecutive_crashes = 0
        self._task = asyncio.create_task(self._run(), name=f"supervise:{self.name}")

    async def restart(self) -> None:
        """Ask the child to stop; the supervision loop respawns it.

        Clears the crash counter and asks the planner to drop any
        degraded state, so a manual restart returns the child to normal
        operation.

        **Two things this used to get wrong on Windows.** It sent
        `TerminateProcess`, so the child never ran its shutdown hooks and
        dropped whatever it was answering. And the child then exited
        non-zero, which the loop counted as a *crash* — so every
        operator-requested restart incremented the crash counter and
        bought a back-off sleep, and enough of them in a row would trip
        the threshold and drop a component into safe mode. POSIX never
        showed it, because SIGTERM gets uvicorn to exit 0.
        """
        self._consecutive_crashes = 0
        self._planner.reset()
        proc = self._proc
        if proc is not None and proc.returncode is None:
            self._expected_exit = True
            sent = process_signals.request_stop(proc, name=self.name, logger=self._log)
            self._log.info("restart requested for %s; sent %s to pid %d", self.name, sent, proc.pid)
            await self._escalate_if_still_alive(proc)

    async def stop(self) -> None:
        """Stop the supervision loop and ensure the child is dead.
        Asks gracefully first, escalates to a hard kill if it hangs."""
        self._stop_requested = True
        self._expected_exit = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process_signals.request_stop(proc, name=self.name, logger=self._log)
            try:
                await asyncio.wait_for(proc.wait(), timeout=_TERM_TIMEOUT_SECONDS)
            except TimeoutError:
                self._log.warning(
                    "%s did not exit within %.1fs of the stop request; killing",
                    self.name,
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

    async def _escalate_if_still_alive(self, proc: Any) -> None:
        """Hard-kill a child that ignored the stop request.

        **A graceful request introduces a hang that `TerminateProcess`
        could not have.** Measured: a Windows child that installs a
        console handler and returns TRUE from it survives
        `CTRL_BREAK_EVENT` indefinitely. Without this the supervision
        loop would sit on `proc.wait()` for a restart that never
        completes, and the component would read as `starting` forever.

        A second waiter on the same process, alongside the supervision
        loop's own — verified safe: `asyncio` keeps a list of exit
        waiters, and `wait_for` cancelling this one on timeout leaves the
        loop's untouched.
        """
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=_TERM_TIMEOUT_SECONDS)
        if proc.returncode is None:
            self._log.warning(
                "%s ignored the stop request for %.1fs; killing pid %s",
                self.name,
                _TERM_TIMEOUT_SECONDS,
                proc.pid,
            )
            with contextlib.suppress(ProcessLookupError, BaseException):
                proc.kill()

    # --- introspection (read-only properties for the routes layer) ---------

    @property
    def pid(self) -> int | None:
        proc = self._proc
        return proc.pid if proc is not None and proc.returncode is None else None

    # --- internals ---------------------------------------------------------

    def _explain_exit(self, return_code: int) -> str | None:
        """Turn this exit into something actionable.

        A bind collision is checked first and by the loop rather than by
        a planner, because it is the one failure both kinds of child
        share: a component and an engine that cannot have their port die
        the same way and for the same reason. Leaving it to the planners
        would mean writing it twice and, as it happened, having written
        it in neither — the component planner's `explain_exit` returns
        None on principle.

        Best-effort by construction: a planner that raises while
        explaining a crash must not turn one crash into two, so the
        generic message stands and the exception is logged.
        """
        tail = "".join(self._output_tail)
        collision = ports.explain_collision(self._last_port, tail)
        if collision is not None:
            return collision
        try:
            return self._planner.explain_exit(return_code, tail)
        except Exception:
            self._log.exception("%s: explaining the exit failed", self.name)
            return None

    async def _pipe_child_output(self, stream: asyncio.StreamReader | None) -> None:
        """Drain a child's stdout/stderr pipe and re-emit each line with
        the component name prefix. Runs as a background task per spawn;
        exits when the pipe closes (child terminated) or it's cancelled.

        Bytes are decoded with `errors="replace"` so a child that writes
        non-UTF-8 to stdout (rare but possible — a Windows native CRT
        diagnostic, say) doesn't kill the reader and leave the agent
        deaf to subsequent output.
        """
        if stream is None:
            return
        prefix = self._planner.log_prefix
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
                self._output_tail.append(text)
                # Colorize "error" / "warning" / "warn" inline so the
                # important lines pop on a fast scroll. Word-level only —
                # full-line color is unreadable on dark terminals.
                text = _colorize_alerts(text)
                # Lines from `readline()` include the trailing newline;
                # use `end=""` so we don't double it. flush=True keeps
                # output snappy even when agent stdout is itself a pipe
                # (e.g. running under a VS Code task with output capture).
                print(prefix + text, end="", flush=True)
        except asyncio.CancelledError:
            return
        except Exception as e:
            # Never let a reader crash bring down the supervision loop;
            # the worst-case fallback is "we lose log prefixing for this
            # child", which is strictly better than the agent dying.
            self._log.warning("output-pipe reader for %s crashed: %s", self.name, e)

    async def _run(self) -> None:
        """Spawn-watch-respawn loop.

        When consecutive crashes hit the threshold the planner decides
        what happens next: it may change something and ask to keep going
        (a component engaging safe mode), or decline, in which case this
        gives up and leaves the child `crashed` until an operator
        restarts it. Recovery policy is kind-specific, so it does not
        live here. Exits cleanly on `stop_requested`.
        """
        while not self._stop_requested:
            await self._spawn_once()
            if self._stop_requested:
                return
            if self.state == ProcessState.not_spawnable:
                # There is nothing to launch, so there is nothing to
                # supervise — end the loop instead of re-asking forever.
                # Worth being explicit: the back-off below is derived from
                # the crash count, and "nothing to launch" is not a crash,
                # so falling through would sleep zero seconds and spin the
                # event loop hot. (Latent before this refactor too, but
                # unreachable — `Supervisor.add_and_start` never built a
                # process for a spawn-less entry. Making `plan() -> None`
                # a first-class outcome makes it reachable.)
                return
            if self._consecutive_crashes >= _CRASH_BACKOFF_THRESHOLD:
                self._log.error(
                    "%s crash threshold reached (last error: %s)",
                    self.name,
                    self.last_error,
                )
                if not self._planner.on_crash_threshold():
                    self.state = ProcessState.crashed
                    return
                self._consecutive_crashes = 0
            await asyncio.sleep(min(2.0 * self._consecutive_crashes, 10.0))

    async def _spawn_once(self) -> None:
        """One spawn / wait / mark-state iteration."""
        try:
            plan = self._planner.plan()
        except SpawnPlanError as e:
            self._log.error("refusing to spawn %s: %s", self.name, e)
            self.state = ProcessState.crashed
            self.last_error = str(e)
            self._consecutive_crashes += 1
            return
        except Exception as e:
            # A planner bug — a TypeError from a changed adapter signature
            # was the M4 case — used to escape here, kill the supervision
            # task, and leave the runtime at `starting` forever with no
            # error anywhere. It is a crash of the declaration, reported
            # as one, and the loop keeps its footing.
            self._log.exception("planner for %s raised; treating as a crash", self.name)
            self.state = ProcessState.crashed
            self.last_error = f"planner raised {type(e).__name__}: {e}"
            self._consecutive_crashes += 1
            return

        if plan is None:
            # Nothing to launch, and that is not an error.
            self.state = ProcessState.not_spawnable
            return

        self._log.info("spawning %s: %s", self.name, " ".join(plan.argv))

        try:
            # Pipe stdout + stderr through us so we can prefix every line
            # with `[<name>]`. Without this the agent inherits the
            # parent terminal and child output interleaves with no source
            # identification — making "Waiting for application startup"
            # ambiguous when several children are booting concurrently.
            self._proc = await asyncio.create_subprocess_exec(
                *plan.argv,
                env=plan.env,
                cwd=plan.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                **process_signals.spawn_kwargs(),
            )
        except OSError as e:
            self._log.error("failed to spawn %s: %s", self.name, e)
            self.state = ProcessState.crashed
            self.last_error = f"spawn failed: {e}"
            self._consecutive_crashes += 1
            return

        # Windows: assign to the agent's Job Object so the OS reaps
        # the child if the agent dies hard. POSIX uses preexec_fn
        # (handled in kwargs above), no post-spawn step needed.
        win_job = orphan_kill.windows_job()
        if win_job is not None and self._proc.pid is not None:
            win_job.assign(self._proc.pid)

        self.state = ProcessState.starting
        self.degraded = plan.degraded
        self._last_port = plan.port
        self.last_argv = list(plan.argv)
        self.last_restart = datetime.now(UTC)
        self.last_error = None

        # Background reader: drains the child's stdout pipe and re-emits
        # each line with a `[<name>]` prefix on the agent's own stdout.
        # MUST be running before we await proc.wait() or the child can
        # block writing into a full pipe buffer and never exit.
        reader_task = asyncio.create_task(
            self._pipe_child_output(self._proc.stdout),
            name=f"output-pipe:{self.name}",
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
            self.state = ProcessState.exited
            return

        expected = self._expected_exit
        self._expected_exit = False

        if return_code == 0 or expected:
            # **`or expected` is the Windows half.** A child asked to stop
            # via a console event exits with the code for an interrupted
            # program, not 0 — rc=3 for CPython's KeyboardInterrupt path.
            # Counting that as a crash is how an operator's restart used
            # to buy itself a back-off sleep, and enough restarts in a row
            # would trip the threshold and drop the component into safe
            # mode for doing exactly what it was asked.
            self._log.info(
                "%s exited %s (rc=%d); respawning",
                self.name,
                "on request" if expected and return_code != 0 else "cleanly",
                return_code,
            )
            self.state = ProcessState.exited
            self._consecutive_crashes = 0
        else:
            self._consecutive_crashes += 1
            self.last_error = self._explain_exit(return_code) or (f"exited with code {return_code}")
            # **The explanation goes to the log, not only to the API.**
            # `_explain_exit` has existed since M0 and its answer has only
            # ever reached `Component.lastError` — so an engine adapter's
            # diagnosis, and now a port collision naming the process
            # holding it, were invisible to the operator watching the
            # console, which is where they are at boot. Found by an
            # acceptance check that read the log for a message the API
            # was already carrying.
            self._log.warning(
                "%s exited rc=%d (consecutive crashes: %d): %s",
                self.name,
                return_code,
                self._consecutive_crashes,
                self.last_error,
            )
            self.state = ProcessState.crashed


# How the supervision loop's kind-agnostic state reads on the component
# wire enum. `not_spawnable` maps to `unreachable` because that is what a
# component with nothing to launch has always reported — the agent can
# see it or it cannot, and it does not own the process either way.
_COMPONENT_STATUS_BY_STATE: dict[ProcessState, ComponentStatus] = {
    ProcessState.starting: ComponentStatus.starting,
    ProcessState.exited: ComponentStatus.exited,
    ProcessState.crashed: ComponentStatus.crashed,
    ProcessState.not_spawnable: ComponentStatus.unreachable,
}


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
        shared_child_env: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        self._log = log or logging.getLogger(__name__)
        self._shared_child_env = shared_child_env
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
        sp = SupervisedProcess.for_component(
            entry,
            self._log,
            auth_state=self._auth_state,
            shared_child_env=self._shared_child_env,
        )
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
        # The trust root is skipped. It receives nothing from this agent's
        # auth state — no master key, no signing key — so a restart hands it
        # nothing new and costs it everything: a restarted control root holds
        # its keys sealed and is locked until an operator logs in again. Found
        # by the M7 run, where enrolling the control host's agent restarted
        # the root it had just enrolled with.
        names = [
            name
            for name, sp in self._processes.items()
            if not (
                isinstance(sp._planner, _ComponentPlanner)
                and sp._planner.entry.kind in _TRUST_ROOT_KINDS
            )
        ]
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

        For spawned children: maps the supervision loop's
        `ProcessState` plus /healthz observations onto the wire enum
        (running, starting, safe_mode, exited, crashed).

        For remote entries (`has_spawn=False`): derives from the last
        /healthz poll only — `running` if the URL is answering,
        `unreachable` otherwise.
        """
        sp = self._processes.get(name)
        if sp is not None:
            base = _COMPONENT_STATUS_BY_STATE[sp.state]
            if base == ComponentStatus.starting:
                if sp.degraded and not self._reachable.get(name, False):
                    # Launched into safe mode but not yet answering. Report
                    # safe_mode straight away rather than `starting`, so the
                    # UI's badge appears as soon as the decision is made
                    # instead of one health poll later.
                    base = ComponentStatus.safe_mode
                elif self._reachable.get(name, False):
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
