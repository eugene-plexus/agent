r"""The tray's channel: who the pipe admits, the protocol, and the imposter check (row 35).

The DACL is derived from data, so it is tested everywhere; the pipe itself
runs on the Windows CI job (pywin32), on a name of its own -- never
`\\.\pipe\eugene-plexus-agent-8079`, which an install on the machine holds.
The service ACL, other accounts and the remote path are the Windows
acceptance's (`specs/scripts/row35-tray-pause-acceptance.py`).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from eugene_plexus_agent import tray_pipe

#: `install.ps1`'s Grant-ServiceControl, as `sc sdshow` prints it after the
#: install: the Windows default for a service, plus the installing account.
INSTALLED = (
    "D:(A;;CCLCSWRPWPDTLOCRRC;;;SY)(A;;CCDCLCSWRPWPDTLOCRSDRCWDWO;;;BA)"
    "(A;;CCLCSWLOCRRC;;;IU)(A;;CCLCSWLOCRRC;;;SU)"
    "(A;;CCLCSWRPWPDTLOCRRC;;;S-1-5-21-1111-2222-3333-1001)"
    "S:(AU;FA;CCDCLCSWRPWPDTLOCRSDRCWDWO;;;WD)"
)
PERSON = "S-1-5-21-1111-2222-3333-1001"


# --- who the pipe admits ------------------------------------------------------------


def test_pipe_acl_from_service_dacl() -> None:
    grants = tray_pipe.grants_from_service_dacl(INSTALLED)
    assert grants.allowed == ("SY", "BA", PERSON)
    for broad in ("IU", "SU", "WD"):
        assert broad not in grants.allowed
    assert grants.denied == ()
    sddl = tray_pipe.pipe_sddl(grants)
    # Protected, NETWORK refused first, the person read and write only.
    assert sddl.startswith("D:P(D;;0x1f01ff;;;NU)")
    assert f"(A;;0x12008b;;;{PERSON})" in sddl
    assert "(A;;0x1f01ff;;;SY)" in sddl and "(A;;0x1f01ff;;;BA)" in sddl
    assert "IU" not in sddl and "SU" not in sddl and "WD" not in sddl


def test_a_broad_group_never_gets_the_pipe_whatever_the_service_grants() -> None:
    granted = "".join(
        f"(A;;RPWP;;;{sid})"
        for sid in ("WD", "AU", "BU", "IU", "SU", "AN", "NU", "S-1-1-0", "S-1-5-11", "S-1-5-32-545")
    )
    grants = tray_pipe.grants_from_service_dacl("D:" + granted)
    assert grants.allowed == ("SY", "BA")


def test_only_the_stop_right_admits() -> None:
    sddl = (
        "D:(A;;RP;;;S-1-5-21-1-2-3-1002)"  # start only
        "(A;;GX;;;S-1-5-21-1-2-3-1003)"  # generic execute holds stop
        "(A;;GA;;;S-1-5-21-1-2-3-1004)"  # generic all
        "(A;;0x20;;;S-1-5-21-1-2-3-1005)"  # SERVICE_STOP in hex
        "(A;;ZZ;;;S-1-5-21-1-2-3-1006)"  # rights not understood grant nothing
        "(A;IO;WP;;;S-1-5-21-1-2-3-1007)"  # inherit-only: not this object
        "(OA;;WP;;;S-1-5-21-1-2-3-1008)"  # an object ACE grants nothing here
        "(A;;WP;;;S-1-5-21-1-2-3-1009)"  # a group of this machine's own
    )
    grants = tray_pipe.grants_from_service_dacl(sddl)
    assert grants.allowed == (
        "SY",
        "BA",
        "S-1-5-21-1-2-3-1003",
        "S-1-5-21-1-2-3-1004",
        "S-1-5-21-1-2-3-1005",
        "S-1-5-21-1-2-3-1009",
    )


def test_deny_aces_are_copied_first_and_a_doubt_denies() -> None:
    sddl = (
        "D:(D;;WP;;;S-1-5-21-1-2-3-1001)(D;;RP;;;S-1-5-21-1-2-3-1002)"
        "(D;;ZZ;;;S-1-5-21-1-2-3-1003)(XD;;WP;;;S-1-5-21-1-2-3-1004;(x))"
        f"(A;;WP;;;{PERSON})(A;;WP;;;S-1-5-21-1-2-3-1001)"
    )
    grants = tray_pipe.grants_from_service_dacl(sddl)
    assert grants.denied == (
        "S-1-5-21-1-2-3-1001",
        "S-1-5-21-1-2-3-1003",
        "S-1-5-21-1-2-3-1004",
    )
    pipe = tray_pipe.pipe_sddl(grants)
    assert pipe.index("(D;;0x1f01ff;;;S-1-5-21-1-2-3-1001)") < pipe.index("(A;;")


def test_without_the_service_only_its_own_account() -> None:
    grants = tray_pipe.grants_for_own_account(PERSON)
    assert grants.allowed == ("SY", "BA", PERSON)


def test_the_protocol_answers_three_verbs_and_refuses_the_rest() -> None:
    for verb in ("state", "pause", "resume"):
        assert tray_pipe.parse_request(json.dumps({"verb": verb}).encode()) == (verb, None)
    assert tray_pipe.parse_request(b'{"verb": "stop"}') == (
        None,
        {"ok": False, "error": "unknown verb"},
    )
    assert tray_pipe.parse_request(b"not json")[1] == {"ok": False, "error": "bad request"}
    assert tray_pipe.parse_request(b"[1]")[1] == {"ok": False, "error": "unknown verb"}


# --- the tray refuses a pipe the service does not serve ----------------------------------


class _FakeWin:
    """pywin32 as far as `call` reaches before it would send anything."""

    def __init__(self, server_pid: int) -> None:
        self.server_pid = server_pid
        self.sent: list[bytes] = []
        self.closed = False

        class _Error(Exception):
            winerror = 0

        self.pywintypes = type("pywintypes", (), {"error": _Error})
        self.error = type("winerror", (), {"ERROR_ACCESS_DENIED": 5, "ERROR_PIPE_BUSY": 231})
        outer = self

        class _File:
            GENERIC_READ = 0x80000000
            OPEN_EXISTING = 3
            FILE_FLAG_OVERLAPPED = 0x40000000
            SECURITY_IDENTIFICATION = 0x10000

            @staticmethod
            def CreateFile(*args: Any) -> str:
                return "handle"

            @staticmethod
            def WriteFile(handle: Any, data: bytes, overlapped: Any) -> None:
                outer.sent.append(data)

        class _Pipe:
            @staticmethod
            def GetNamedPipeServerProcessId(handle: Any) -> int:
                return outer.server_pid

        class _Api:
            @staticmethod
            def CloseHandle(handle: Any) -> None:
                outer.closed = True

        self.file, self.pipe, self.api = _File, _Pipe, _Api


def test_tray_refuses_imposter_server(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeWin(server_pid=4321)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tray_pipe, "_win32", lambda: fake)
    answer = tray_pipe.call("pause", port=18379, service_name="EP", expected_pid=1234)
    assert answer.status == "imposter"
    assert answer.why == "Eugene's tray channel is held by another program."
    # Nothing was sent to it, and the handle was closed.
    assert fake.sent == []
    assert fake.closed


def test_the_tray_needs_the_service_running_to_trust_the_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeWin(server_pid=1234)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tray_pipe, "_win32", lambda: fake)
    monkeypatch.setattr(tray_pipe, "service_pid", lambda name: None)
    answer = tray_pipe.call("pause", port=18379, service_name="EP")
    assert answer.status == "absent"
    assert fake.sent == []


# --- the real pipe, on Windows -----------------------------------------------------------

windows = pytest.mark.skipif(sys.platform != "win32", reason="named pipes are Windows'")


@pytest.fixture
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    running = asyncio.new_event_loop()
    thread = threading.Thread(target=running.run_forever, daemon=True)
    thread.start()
    yield running
    running.call_soon_threadsafe(running.stop)
    thread.join(timeout=5)
    running.close()


def _port() -> int:
    """A pipe name of this test's own: never an install's."""
    return 40000 + (os.getpid() * 7 + int(time.perf_counter() * 1000)) % 20000


def _raw(name: str) -> Any:
    """A plain client, waiting for the one instance while the server turns
    from the last caller to the next."""
    import pywintypes
    import win32file
    import win32pipe

    deadline = time.perf_counter() + 5
    while True:
        try:
            return win32file.CreateFile(
                name,
                win32file.GENERIC_READ | 0x0002,
                0,
                None,
                win32file.OPEN_EXISTING,
                0x00100000 | win32file.SECURITY_IDENTIFICATION,
                None,
            )
        except pywintypes.error as exc:
            if exc.winerror != 231 or time.perf_counter() > deadline:
                raise
            win32pipe.WaitNamedPipe(name, 1000)


@windows
def test_a_real_pipe_answers_its_own_account(loop: asyncio.AbstractEventLoop) -> None:
    pytest.importorskip("win32pipe")
    asked: list[str] = []

    async def answer(verb: str) -> dict[str, Any]:
        asked.append(verb)
        return {"ok": True, "pause": {"paused": verb == "pause"}}

    port = _port()
    server = tray_pipe.TrayPipe(port=port, answer=answer, loop=loop, read_timeout=1.0)
    server.start()
    try:
        assert server.status == tray_pipe.SERVING
        reply = tray_pipe.call("pause", port=port, service_name="none", expected_pid=os.getpid())
        assert reply.status == "ok", reply
        assert reply.body["pause"] == {"paused": True}
        state = tray_pipe.call("state", port=port, service_name="none", expected_pid=os.getpid())
        assert state.status == "ok"
        assert asked == ["pause", "state"]
        # The tray checks the server: this one is not the process it expects.
        imposter = tray_pipe.call(
            "resume", port=port, service_name="none", expected_pid=os.getpid() + 1
        )
        assert imposter.status == "imposter"
        assert asked == ["pause", "state"]
    finally:
        server.stop()


@windows
def test_a_real_pipe_drops_a_client_that_never_writes(loop: asyncio.AbstractEventLoop) -> None:
    pytest.importorskip("win32pipe")
    import win32api

    async def answer(verb: str) -> dict[str, Any]:
        return {"ok": True}

    port = _port()
    server = tray_pipe.TrayPipe(port=port, answer=answer, loop=loop, read_timeout=0.5)
    server.start()
    try:
        silent = _raw(tray_pipe.pipe_name(port))
        started = time.perf_counter()
        # Held, saying nothing: the next caller waits for the timeout, not forever.
        reply = tray_pipe.call("state", port=port, service_name="none", expected_pid=os.getpid())
        assert reply.status == "ok", reply
        assert time.perf_counter() - started < 5
        win32api.CloseHandle(silent)
        # Too long a request is dropped too, and the channel goes on.
        long = _raw(tray_pipe.pipe_name(port))
        import win32file

        win32file.WriteFile(long, b"x" * (tray_pipe.MAX_REQUEST_BYTES + 100))
        win32api.CloseHandle(long)
        reply = tray_pipe.call("state", port=port, service_name="none", expected_pid=os.getpid())
        assert reply.status == "ok", reply
        # An unknown verb is answered, not dropped.
        raw = _raw(tray_pipe.pipe_name(port))
        win32file.WriteFile(raw, b'{"verb": "stop"}\n')
        _, data = win32file.ReadFile(raw, 4096)
        win32api.CloseHandle(raw)
        assert json.loads(data) == {"ok": False, "error": "unknown verb"}
    finally:
        server.stop()


@windows
def test_a_name_another_program_holds_is_reported_and_never_trusted(
    loop: asyncio.AbstractEventLoop,
) -> None:
    pytest.importorskip("win32pipe")
    port = _port()
    name = tray_pipe.pipe_name(port)
    squatter = subprocess.Popen(
        [
            sys.executable,
            "-c",
            # A new instance after each connection, so one always listens.
            "import sys, win32pipe\n"
            "held = []\n"
            "while len(held) < 50:\n"
            "    held.append(win32pipe.CreateNamedPipe(sys.argv[1], 3, 0, 255, 4096, 4096, 0, None))\n"
            "    if len(held) == 1:\n"
            "        import os; print(os.getpid(), flush=True)\n"
            "    win32pipe.ConnectNamedPipe(held[-1], None)\n",
            name,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert squatter.stdout is not None
        # Its own pid: a venv's python.exe may start the interpreter as a child.
        holder = int(squatter.stdout.readline().strip())

        async def answer(verb: str) -> dict[str, Any]:
            return {"ok": True}

        server = tray_pipe.TrayPipe(port=port, answer=answer, loop=loop)
        server.start()
        try:
            assert server.status == tray_pipe.TAKEN
            assert server.holder == holder
        finally:
            server.stop()
        # The tray, expecting this process as the service, refuses the squatter.
        reply = tray_pipe.call("pause", port=port, service_name="none", expected_pid=os.getpid())
        assert reply.status == "imposter"
    finally:
        squatter.kill()
        squatter.wait(timeout=10)


@windows
def test_may_stop_reads_the_dacl_as_windows_does() -> None:
    """The per-connection check is Windows' own AccessCheck, groups and deny
    ACEs included, against this test's own token."""
    pytest.importorskip("win32security")
    import win32api
    import win32security

    process = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(), win32security.TOKEN_DUPLICATE | win32security.TOKEN_QUERY
    )
    token = win32security.DuplicateToken(process, win32security.SecurityIdentification)
    me = win32security.ConvertSidToStringSid(
        win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    )
    try:
        mine = f"D:(A;;CCLCSWRPWPDTLOCRRC;;;SY)(A;;CCLCSWRPWPDTLOCRRC;;;{me})"
        assert tray_pipe.may_stop(mine, int(token)) is True
        # Query and start rights, no stop: no.
        assert tray_pipe.may_stop(f"D:(A;;CCLCSWRPLORC;;;{me})", int(token)) is False
        # Through a group this token holds (Everyone).
        assert tray_pipe.may_stop("D:(A;;WP;;;WD)", int(token)) is True
        # A deny for this account wins over the allow.
        assert tray_pipe.may_stop(f"D:(D;;WP;;;{me}){mine[2:]}", int(token)) is False
        assert tray_pipe.may_stop("D:", int(token)) is False
    finally:
        win32api.CloseHandle(token)
        win32api.CloseHandle(process)
