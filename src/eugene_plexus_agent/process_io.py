"""How many bytes a process has read, and whether that number is real.

Exists for one question an operator asks out loud: *is this engine
loading, or is it hung?* A 24.95 GB model over a 1 Gbps share takes four
minutes to read, and for those four minutes a healthy node and a dead
one look identical from every surface this project has.

**The bytes are there for one of the two ways an engine reads a model
and not the other**, which is why this module reports a source and is
allowed to answer "I cannot tell". Measured 2026-09-17 on Windows,
268 MB touched per run:

    buffered read, local disk   ->  ReadTransferCount  +268.4 MB
    buffered read, over SMB     ->  ReadTransferCount  +268.4 MB
    memory-mapped read          ->  ReadTransferCount    +0.0 MB

So `hobbyist-ux.md` §11.9's refusal — *"there are no bytes"*, because
llama.cpp memory-maps the model and faulted pages are not read I/O — is
still exactly true of the mapped path and false of the buffered one.
llama.cpp takes `--load-mode none` (`noMmap` on a profile) and vLLM
reads normally, so the honest answer is per-launch, not per-product.

**Nothing here decides which path an engine took.** The caller samples
over time and believes the counter only while it is advancing; see
`LoadProgressTracker`. A flag would have been simpler and would be wrong
for vLLM, for a llama.cpp launch that turned mmap off, and for whatever
upstream changes next.

**The trap this module is shaped to avoid**, because it produced a
confident wrong answer during the very measurement above: the first run
reported `+0.0 MB` for *both* paths, which reads as a clean confirmation
of the belief already held. `GetCurrentProcess`/`OpenProcess` return
HANDLEs, ctypes defaults a return type to `c_int`, and on 64-bit the
truncated value makes the call fail while the zeroed output struct still
looks like a legitimate answer. Every `argtypes`/`restype` below is
therefore declared, and `_windows_read_bytes` raises on a failed call
rather than returning a plausible zero.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# What answered. On the wire so that "this host cannot measure it" is
# distinguishable from "this engine is not reading".
SOURCE_WINDOWS = "windows_io_counters"
SOURCE_PROC_IO = "proc_io_rchar"
SOURCE_RUSAGE = "rusage_diskio"


# --------------------------------------------------------------------------- #
# Per-platform counters
# --------------------------------------------------------------------------- #


def _windows_read_bytes(pid: int) -> int:
    # **The guard is for the type checker as much as for the runtime.**
    # `ctypes.WinDLL` and `ctypes.get_last_error` exist only on Windows, and
    # CI type-checks on Linux, so mypy reported three attr-defined errors on
    # code no Linux process can reach -- and the agent's CI had been red on
    # exactly those three since at least 2026-09-17, which is the third time
    # this project has found a suite failing unnoticed. A `sys.platform`
    # narrowing fixes it on both platforms at once, where a `type: ignore`
    # would be unused on Windows and so fail there instead.
    if sys.platform != "win32":
        raise OSError("process I/O counters are a Windows-only API")

    import ctypes
    import ctypes.wintypes as wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declared, not defaulted. A HANDLE returned as c_int is truncated on
    # 64-bit and every later call fails while looking like it worked.
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.POINTER(_IoCounters)]
    kernel32.GetProcessIoCounters.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    # LIMITED_INFORMATION is enough for the counters and is grantable
    # between processes of the same user without any privilege.
    process_query_limited_information = 0x1000
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"OpenProcess({pid}) failed")
    try:
        counters = _IoCounters()
        if not kernel32.GetProcessIoCounters(handle, ctypes.byref(counters)):
            raise OSError(ctypes.get_last_error(), f"GetProcessIoCounters({pid}) failed")
        return int(counters.ReadTransferCount)
    finally:
        kernel32.CloseHandle(handle)


def _linux_read_bytes(pid: int) -> int:
    """`rchar` from `/proc/<pid>/io`.

    **`rchar`, not `read_bytes`.** `read_bytes` counts what went to the
    block layer, which for a model on an NFS or SMB mount can be zero —
    the exact case this exists for. `rchar` counts bytes the process
    asked for, which is the question.
    """
    for line in Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines():
        if line.startswith("rchar:"):
            return int(line.split(":", 1)[1].strip())
    raise OSError(f"/proc/{pid}/io carried no rchar line")


def _macos_read_bytes(pid: int) -> int:
    """`ri_diskio_bytesread` from `proc_pid_rusage`.

    **Unverified — no Mac has run this.** The struct is taken from
    `<libproc.h>` / `<sys/resource.h>` RUSAGE_INFO_V4, whose first nine
    fields are stable across every version from V0; only the tail grows.
    A wrong offset here reports a nonsense byte count rather than
    failing, so it is read defensively and the caller's advancing check
    is what stops a nonsense number reaching a screen.
    """
    import ctypes

    class _RUsageInfoV4(ctypes.Structure):
        _fields_ = [
            ("ri_uuid", ctypes.c_uint8 * 16),
            ("ri_user_time", ctypes.c_uint64),
            ("ri_system_time", ctypes.c_uint64),
            ("ri_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_interrupt_wkups", ctypes.c_uint64),
            ("ri_pageins", ctypes.c_uint64),
            ("ri_wired_size", ctypes.c_uint64),
            ("ri_resident_size", ctypes.c_uint64),
            ("ri_phys_footprint", ctypes.c_uint64),
            ("ri_proc_start_abstime", ctypes.c_uint64),
            ("ri_proc_exit_abstime", ctypes.c_uint64),
            ("ri_child_user_time", ctypes.c_uint64),
            ("ri_child_system_time", ctypes.c_uint64),
            ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
            ("ri_child_interrupt_wkups", ctypes.c_uint64),
            ("ri_child_pageins", ctypes.c_uint64),
            ("ri_child_elapsed_abstime", ctypes.c_uint64),
            ("ri_diskio_bytesread", ctypes.c_uint64),
            ("ri_diskio_byteswritten", ctypes.c_uint64),
            # V2+ continues past here; we never read it.
            ("_tail", ctypes.c_uint64 * 32),
        ]

    libc = ctypes.CDLL("libc.dylib", use_errno=True)
    libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libc.proc_pid_rusage.restype = ctypes.c_int
    usage = _RUsageInfoV4()
    rusage_info_v4 = 4
    if libc.proc_pid_rusage(pid, rusage_info_v4, ctypes.byref(usage)) != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage({pid}) failed")
    return int(usage.ri_diskio_bytesread)


def read_bytes(pid: int) -> tuple[int, str] | None:
    """Bytes this process has read so far, and which counter said so.

    None when the host has no counter we know, when the process is gone,
    or when the call failed — all of which are "cannot tell", never
    "zero". A caller that treated a failure as zero would draw a bar that
    never moves, which is worse than drawing none.
    """
    try:
        if sys.platform == "win32":
            return _windows_read_bytes(pid), SOURCE_WINDOWS
        if sys.platform.startswith("linux"):
            return _linux_read_bytes(pid), SOURCE_PROC_IO
        if sys.platform == "darwin":
            return _macos_read_bytes(pid), SOURCE_RUSAGE
    except Exception as e:
        log.debug("read-byte counter for pid %s unavailable: %s", pid, e)
        return None
    return None


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

# How long a window the rate is computed over. Long enough that one slow
# poll cannot read as a stall, short enough that the rate tracks a share
# that has just got slower.
_WINDOW_SECONDS = 12.0

# Samples older than the window are dropped, but never the last two —
# a rate needs a pair, and nothing here polls on a guaranteed cadence.
_MIN_SAMPLES = 2


@dataclass
class _Sample:
    at: float
    read: int


@dataclass
class _Track:
    pid: int
    total_bytes: int | None
    samples: list[_Sample] = field(default_factory=list)


@dataclass(frozen=True)
class LoadProgress:
    """What to say about a load in flight. Only ever built from bytes
    that were observed to move."""

    bytes_read: int
    total_bytes: int | None
    bytes_per_second: float | None
    source: str


class LoadProgressTracker:
    """Turns repeated byte-counter readings into an honest percentage.

    **Sampled when something asks, not on a loop.** The control root
    polls every node's `/v1/runtimes` every second or two and a console
    polls it every three, so a loading runtime is read constantly by
    things that already exist; a background task would add a second
    cadence to keep correct for no gain. The same reasoning as the
    worker's Library-folder copy, which also refreshes per request.

    **Progress is reported only while the bytes are advancing.** That is
    the whole detection rule, and it is what keeps this correct for an
    engine that memory-maps its model: llama.cpp still reads a GGUF's
    header and metadata through ordinary reads, so a mapped load makes
    the counter twitch a few megabytes and then stop dead. A
    non-zero-bytes test would show a bar frozen at 0.1% for four minutes
    — precisely the "is it hung?" it was built to answer, made worse by
    looking authoritative.
    """

    def __init__(self) -> None:
        self._tracks: dict[str, _Track] = {}

    def forget(self, name: str) -> None:
        self._tracks.pop(name, None)

    def sample(
        self,
        name: str,
        pid: int | None,
        local_path: str | None,
        *,
        now: float | None = None,
    ) -> LoadProgress | None:
        """Read the counter for one runtime and answer if it can.

        Returns None when there is nothing honest to say: no pid, no
        counter on this host, or bytes that are not moving.
        """
        if pid is None:
            self.forget(name)
            return None

        reading = read_bytes(pid)
        if reading is None:
            self.forget(name)
            return None
        value, source = reading

        at = time.perf_counter() if now is None else now
        track = self._tracks.get(name)
        if track is None or track.pid != pid:
            # A restart is a new process and a new file to size. Never
            # carry a previous run's samples across it: the counter
            # resets, and a stale baseline would report a huge negative
            # or a huge positive rate on the first reading after.
            track = _Track(pid=pid, total_bytes=_size_of(local_path))
            self._tracks[name] = track

        track.samples.append(_Sample(at=at, read=value))
        _trim(track.samples, at)

        if len(track.samples) < _MIN_SAMPLES:
            return None

        first, last = track.samples[0], track.samples[-1]
        delta = last.read - first.read
        elapsed = last.at - first.at
        if delta <= 0:
            # Either a mapped load (the counter twitched once and
            # stopped) or a host whose counter does not see this kind of
            # read. Both are "cannot tell", and the caller shows elapsed.
            return None

        return LoadProgress(
            bytes_read=last.read,
            total_bytes=track.total_bytes,
            bytes_per_second=(delta / elapsed) if elapsed > 0 else None,
            source=source,
        )


def _trim(samples: list[_Sample], now: float) -> None:
    while len(samples) > _MIN_SAMPLES and (now - samples[0].at) > _WINDOW_SECONDS:
        samples.pop(0)


def _size_of(local_path: str | None) -> int | None:
    """The model's size, for the denominator.

    None rather than an exception when the path cannot be stat'd — a
    share that has gone away is exactly when a load is worth watching,
    and a progress report with no percentage still beats none.
    """
    if not local_path:
        return None
    try:
        return Path(local_path).stat().st_size
    except OSError as e:
        log.debug("could not size %s for load progress: %s", local_path, e)
        return None
