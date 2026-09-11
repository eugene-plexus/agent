"""Saying who is holding a port, when a child could not have it.

A supervised child whose port is taken exits within a second of spawning
with an OS error, and the supervision loop's answer to that is a
crash-back-off cycle whose message is `exited with code 1`. The operator
gets a component that will not start and no reason — while the reason is
a single line the child printed and one lookup away.

**This is the real form of a hazard the install-paths design named
differently.** §11.1 carried "stale processes stack on one loopback port
with the oldest still serving" as a supervisor problem, from an
observation during the M10 acceptance run. Measured 2026-09-11: two
uvicorn servers **cannot** share a port — the second dies with WinError
10048 — so nothing the agent supervises can stack. What *does* stack is
`http.server.HTTPServer`, which sets `allow_reuse_address = True`; every
stacking process in that M10 run was a test stub. (The observation was
sound and the stub was still holding 8195 two days later, which is how
this was confirmed: it silently answered a probe written to measure
something else.)

So the supervisor's job here is **diagnosis, not reclamation**. It does
not kill the holder: at boot it cannot tell its own orphan from the
operator's unrelated server on the same port, and killing the wrong one
is unrecoverable where explaining the right one costs nothing. An eager
refusal can be wrong; an explanation of a real failure cannot.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys

log = logging.getLogger(__name__)

_LOOKUP_TIMEOUT = 5.0

# Every way the platforms say it. WinError 10048 is WSAEADDRINUSE;
# errno 98 is Linux EADDRINUSE and 48 is the macOS/BSD one.
_IN_USE = re.compile(
    r"(10048)|(errno\s+(98|48))|(address already in use)"
    r"|(only one usage of each socket address)|(eaddrinuse)",
    re.IGNORECASE,
)


def looks_like_address_in_use(output: str) -> bool:
    """Does this child's output tail read as a bind collision?"""
    return bool(output) and bool(_IN_USE.search(output))


def describe_holder(port: int) -> str | None:
    """Who is listening on `port`, in words, or None if we cannot tell.

    Best-effort by construction and never raises: this runs while
    explaining a crash, and an exception here would turn one failure into
    two.
    """
    try:
        if sys.platform == "win32":
            return _windows_holder(port)
        return _posix_holder(port)
    except Exception:  # pragma: no cover - defensive
        log.debug("could not identify the holder of port %d", port, exc_info=True)
        return None


def _run(argv: list[str]) -> str:
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=_LOOKUP_TIMEOUT,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout or ""


def _windows_holder(port: int) -> str | None:
    listening = None
    for line in _run(["netstat", "-ano", "-p", "TCP"]).splitlines():
        parts = line.split()
        # proto local foreign state pid
        if len(parts) < 5 or parts[3] != "LISTENING":
            continue
        local = parts[1]
        if local.rsplit(":", 1)[-1] != str(port):
            continue
        listening = parts[4]
        break
    if listening is None:
        return None
    image = _windows_image_name(listening)
    return f"pid {listening}" + (f" ({image})" if image else "")


def _windows_image_name(pid: str) -> str | None:
    for line in _run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"]).splitlines():
        if line.startswith('"'):
            return line.split('","')[0].strip('"')
    return None


def _posix_holder(port: int) -> str | None:
    # `ss` on modern Linux, `lsof` elsewhere. Absent either, we say
    # nothing rather than guess — "something" is not a diagnosis.
    for argv, pattern in (
        (["ss", "-ltnp"], rf":{port}\s"),
        (["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], r"LISTEN"),
    ):
        try:
            out = _run(argv)
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            if re.search(pattern, line):
                return line.strip()
    return None


def explain_collision(port: int | None, output_tail: str) -> str | None:
    """Turn an address-in-use exit into something an operator can act on.

    Returns None when this was not a bind collision, so the caller's own
    explanation still stands.
    """
    if not looks_like_address_in_use(output_tail):
        return None
    if port is None:
        return (
            "the port it tried to bind is already in use by another process "
            "(this agent does not know which port it asked for)"
        )
    holder = describe_holder(port)
    if holder is None:
        return (
            f"port {port} is already in use, and nothing is listening on it now — "
            f"the holder exited between the failed bind and this lookup, so a "
            f"restart will probably succeed"
        )
    return (
        f"port {port} is already held by {holder}. Stop it, or point this "
        f"component at another port in the topology. Nothing is killed for "
        f"you: at boot the agent cannot tell its own leftover from a process "
        f"you meant to be running."
    )
