"""Each linked person's worker, started as that person (job-sites-own-enrollment.md
§2.4, §3.2). The Windows calls (WTS, CreateProcessAsUser, the job object) are
faked here and exercised by an elevated acceptance run; what these tests hold is
every decision around them: who gets a worker, when it stops, what the token is."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eugene_plexus_agent import site_workers
from eugene_plexus_agent.site_links import LinkStore
from eugene_plexus_agent.site_workers import (
    RESTART_SECONDS,
    ChildStarter,
    WindowsStarter,
    WorkerProgram,
    _Backoff,
    _Running,
)

ADA = "S-1-5-21-1-2-3-1001"
BO = "S-1-5-21-1-2-3-1002"
HOST = "S-1-5-80-9"


def program(tmp_path: Path, **changes: Any) -> WorkerProgram:
    values: dict[str, Any] = {
        "python": tmp_path / "venv" / "Scripts" / "python.exe",
        "channel": "chan",
        "host": HOST,
        "servers": tmp_path / "site" / "servers.yaml",
    }
    return WorkerProgram(**{**values, **changes})


def test_a_workers_command_line_is_exactly_this(tmp_path: Path) -> None:
    one = program(tmp_path)
    assert one.argv(ADA) == [
        str(one.python),
        "-I",
        "-m",
        "eugene_plexus_site_host.worker",
        "--account",
        ADA,
        "--channel",
        "chan",
        "--host",
        HOST,
        "--servers",
        str(one.servers),
    ]
    two = program(tmp_path, protect=(tmp_path / "cfg", tmp_path / "apps"))
    assert two.argv(BO)[-4:] == [
        "--protect",
        str(tmp_path / "cfg"),
        "--protect",
        str(tmp_path / "apps"),
    ]
    assert two.argv(BO)[5] == BO, "each worker is named for its own account"


def write_links(tmp_path: Path, *accounts: str) -> LinkStore:
    """Links written as a hand-edited file could be: no rule applied."""
    store = LinkStore(tmp_path)
    store.dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "subject": f"p-{n}",
            "name": f"P{n}",
            "account": account,
            "accountName": f"PC\\{n}",
            "linkedAt": "2026-10-06T00:00:00+00:00",
        }
        for n, account in enumerate(accounts)
    ]
    store.path.write_text(json.dumps({"version": 1, "links": entries}), encoding="utf-8")
    return store


class Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.store = LinkStore(tmp_path)
        self.sessions: dict[str, int] = {}
        self.started: list[tuple[str, int, WorkerProgram]] = []
        self.stopped: list[str] = []
        self.exited: set[str] = set()
        monkeypatch.setattr(site_workers, "_session_users", lambda: dict(self.sessions))
        starter = WindowsStarter.__new__(WindowsStarter)  # no job object, no pywin32
        starter.links = self.store
        starter.running = {}
        starter.backoff = _Backoff()
        starter.program = None
        starter._start = self._start  # type: ignore[method-assign]
        starter._stop = self._stop  # type: ignore[method-assign]
        starter._exited = lambda entry: entry.handle.account in self.exited  # type: ignore[method-assign]
        self.starter = starter

    def _start(self, account: str, session: int, program: WorkerProgram) -> None:
        self.started.append((account, session, program))
        self.starter.running[account] = _Running(
            SimpleNamespace(account=account), 100 + len(self.started), session, program
        )

    def _stop(self, account: str) -> None:
        self.stopped.append(account)
        self.starter.running.pop(account, None)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


def test_only_a_linked_account_with_a_session_gets_a_worker(
    tmp_path: Path, harness: Harness
) -> None:
    write_links(tmp_path, ADA, BO)
    wanted = program(tmp_path)
    harness.starter.program = wanted
    harness.sessions = {ADA: 2, "S-1-5-21-1-2-3-1999": 3}  # BO signed out, a stranger signed in
    harness.starter.step()
    assert harness.started == [(ADA, 2, wanted)]
    assert harness.starter.connected_accounts() == {ADA}
    harness.starter.step()
    assert len(harness.started) == 1, "a running worker is not started again"
    harness.sessions[BO] = 5  # BO signs in
    harness.starter.step()
    assert [(a, s) for a, s, _ in harness.started] == [(ADA, 2), (BO, 5)]


def test_no_program_or_no_links_starts_nothing(tmp_path: Path, harness: Harness) -> None:
    harness.sessions = {ADA: 2}
    harness.starter.program = program(tmp_path)
    harness.starter.step()  # nobody linked
    write_links(tmp_path, ADA)
    harness.starter.program = None
    harness.starter.step()  # the site host's program is not installed yet
    assert harness.started == []


@pytest.mark.parametrize("account", ["S-1-5-18", "S-1-5-80-77", "S-1-5-32-544", "0"])
def test_an_account_that_is_never_a_person_never_gets_a_worker(
    tmp_path: Path, harness: Harness, account: str
) -> None:
    """A links file edited by hand is still not a way to run a worker as
    SYSTEM: the starter applies the rule itself."""
    write_links(tmp_path, account, ADA)
    harness.starter.program = program(tmp_path)
    harness.sessions = {account: 0, ADA: 2}
    harness.starter.step()
    assert [a for a, _, _ in harness.started] == [ADA]


def test_a_removed_link_stops_its_worker(tmp_path: Path, harness: Harness) -> None:
    store = write_links(tmp_path, ADA, BO)
    harness.starter.program = program(tmp_path)
    harness.sessions = {ADA: 2, BO: 3}
    harness.starter.step()
    assert harness.starter.connected_accounts() == {ADA, BO}
    store.remove("p-0")
    harness.starter.step()
    assert harness.stopped == [ADA]
    assert harness.starter.connected_accounts() == {BO}
    assert len(harness.started) == 2, "and it is not started again"


def test_a_changed_program_restarts_the_worker(tmp_path: Path, harness: Harness) -> None:
    write_links(tmp_path, ADA)
    harness.sessions = {ADA: 2}
    first = program(tmp_path)
    harness.starter.program = first
    harness.starter.step()
    harness.starter.program = program(tmp_path, channel="other")
    harness.starter.step()
    assert harness.stopped == [ADA], "the old one stopped, the new one started in the same step"
    assert [p.channel for _, _, p in harness.started] == ["chan", "other"]
    harness.starter.program = None
    harness.starter.step()
    assert harness.stopped == [ADA, ADA] and not harness.starter.running


def test_a_worker_that_ends_early_waits_before_it_is_started_again(
    tmp_path: Path, harness: Harness
) -> None:
    write_links(tmp_path, ADA)
    harness.sessions = {ADA: 2}
    harness.starter.program = program(tmp_path)
    harness.starter.step()
    harness.exited.add(ADA)
    harness.starter.step()  # seen to have ended: backoff begins
    assert not harness.starter.running and not harness.starter.backoff.ready(ADA)
    harness.exited.clear()
    harness.starter.step()
    assert len(harness.started) == 1, "not before the back-off is over"
    harness.starter.backoff._next[ADA] = (0.0, 0)  # the wait is over
    harness.starter.step()
    assert len(harness.started) == 2


def test_a_start_that_raises_backs_off_instead_of_looping(tmp_path: Path, harness: Harness) -> None:
    write_links(tmp_path, ADA)
    harness.sessions = {ADA: 2}
    harness.starter.program = program(tmp_path)
    calls: list[str] = []

    def boom(account: str, session: int, wanted: WorkerProgram) -> None:
        calls.append(account)
        raise OSError("CreateProcessAsUser failed")

    harness.starter._start = boom  # type: ignore[method-assign]
    harness.starter.step()
    harness.starter.step()
    assert calls == [ADA], "the second step is inside the back-off"


def test_the_back_off_grows_and_forgives_a_long_run() -> None:
    backoff = _Backoff()
    delays = []
    for _ in range(len(RESTART_SECONDS) + 2):
        backoff.failed("a", 1.0)
        due, _tries = backoff._next["a"]
        import time

        delays.append(round(due - time.perf_counter()))
    assert delays[: len(RESTART_SECONDS)] == [round(s) for s in RESTART_SECONDS]
    assert delays[-1] == round(RESTART_SECONDS[-1]), "it stops growing"
    backoff.failed("a", 3600.0)  # it ran an hour: start over
    assert backoff._next["a"][1] == 0
    assert backoff.ready("never-failed")


def test_stopping_terminates_the_process_and_forgets_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[tuple[object, int]] = []
    fake = types.ModuleType("win32process")
    fake.TerminateProcess = lambda handle, code: killed.append((handle, code))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "win32process", fake)
    starter = WindowsStarter.__new__(WindowsStarter)
    starter.running = {ADA: _Running("HANDLE", 7, 2, None)}  # type: ignore[arg-type]
    starter.close()
    assert killed == [("HANDLE", 1)] and starter.running == {}
    starter._stop("not-running")  # nothing to do, nothing raised


# --- the token ------------------------------------------------------------------------


class FakeSecurity(types.ModuleType):
    TokenElevationType = "elevation"
    TokenLinkedToken = "linked"
    TokenGroups = "groups"

    def __init__(self, kind: int, *, admin: bool = False) -> None:
        super().__init__("win32security")
        self.kind, self.admin = kind, admin

    def GetTokenInformation(self, token: object, what: str) -> Any:
        if what == "elevation":
            return self.kind
        if what == "linked":
            return "LINKED"
        return [("USERS", 0x7), ("ADMINS", 0x4 if self.admin else 0x10)]

    def ConvertStringSidToSid(self, text: str) -> str:
        return "ADMINS" if text == "S-1-5-32-544" else text


def use_token(monkeypatch: pytest.MonkeyPatch, kind: int, *, admin: bool = False) -> list[object]:
    monkeypatch.setitem(sys.modules, "win32security", FakeSecurity(kind, admin=admin))
    filtered: list[object] = []
    monkeypatch.setattr(
        site_workers, "_filter_by_hand", lambda token: filtered.append(token) or "HAND"
    )
    return filtered


def test_a_token_uac_already_filtered_is_used_as_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    filtered = use_token(monkeypatch, 3, admin=True)
    assert site_workers._limited("TOKEN") == "TOKEN"
    assert filtered == []


def test_a_full_token_is_swapped_for_its_linked_filtered_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filtered = use_token(monkeypatch, 2, admin=True)
    assert site_workers._limited("TOKEN") == "LINKED"
    assert filtered == []


def test_a_default_token_with_administrators_enabled_is_filtered_by_hand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UAC off, or the built-in Administrator: no filtered token exists, and the
    worker must never run with administrator power."""
    filtered = use_token(monkeypatch, 1, admin=True)
    assert site_workers._limited("TOKEN") == "HAND"
    assert filtered == ["TOKEN"]


def test_a_default_token_of_a_standard_user_is_used_as_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filtered = use_token(monkeypatch, 1, admin=False)
    assert site_workers._limited("TOKEN") == "TOKEN"
    assert filtered == []


def test_administrators_deny_only_in_the_token_is_not_administrator_power(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group attribute 0x10 is deny-only, 0x4 is enabled: only the second is power."""
    monkeypatch.setitem(sys.modules, "win32security", FakeSecurity(1, admin=False))
    assert site_workers._admin_enabled("T") is False
    monkeypatch.setitem(sys.modules, "win32security", FakeSecurity(1, admin=True))
    assert site_workers._admin_enabled("T") is True


# --- a per-user install: one worker, a child of the agent ----------------------------


@dataclass
class Sleeper(WorkerProgram):
    """A real tiny child instead of the site host's worker."""

    code: str = "import time; time.sleep(60)"

    def argv(self, account: str) -> list[str]:
        return [sys.executable, "-c", self.code]


def sleeper(tmp_path: Path, **changes: Any) -> Sleeper:
    return Sleeper(python=Path(sys.executable), channel="c", host="h", servers=tmp_path, **changes)


async def test_the_child_starter_runs_one_child_and_stops_it(tmp_path: Path) -> None:
    starter = ChildStarter(ADA)
    await starter.step()
    assert starter.process is None, "no program yet, no child"
    starter.program = sleeper(tmp_path)
    await starter.step()
    first = starter.process
    try:
        assert first is not None and first.returncode is None
        await starter.step()
        assert starter.process is first, "one child, not one per step"
    finally:
        await starter.stop()
    assert first.returncode is not None and starter.process is None
    await starter.stop()  # twice is nothing


async def test_the_child_starter_passes_the_programs_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Any, ...]] = []
    real = site_workers.asyncio.create_subprocess_exec

    async def spy(*argv: Any, **kwargs: Any) -> Any:
        seen.append(argv)
        return await real(*argv, **kwargs)

    monkeypatch.setattr(site_workers.asyncio, "create_subprocess_exec", spy)
    starter = ChildStarter(ADA)
    starter.program = sleeper(tmp_path)
    await starter.step()
    await starter.stop()
    assert seen == [tuple(starter.program.argv(ADA))]


async def test_a_child_that_ends_is_started_again_only_after_the_back_off(
    tmp_path: Path,
) -> None:
    starter = ChildStarter(ADA)
    starter.program = sleeper(tmp_path, code="pass")
    await starter.step()
    first = starter.process
    assert first is not None
    await first.wait()
    await starter.step()
    assert starter.process is None and not starter.backoff.ready(ADA)
    await starter.step()
    assert starter.process is None, "still inside the back-off"
    starter.backoff._next[ADA] = (0.0, 0)
    starter.program = sleeper(tmp_path)
    await starter.step()
    try:
        assert starter.process is not None and starter.process is not first
    finally:
        await starter.stop()


async def test_a_changed_program_replaces_the_child_and_none_removes_it(tmp_path: Path) -> None:
    starter = ChildStarter(ADA)
    starter.program = sleeper(tmp_path)
    await starter.step()
    first = starter.process
    assert first is not None
    starter.program = sleeper(tmp_path, code="import time; time.sleep(61)")
    await starter.step()
    second = starter.process
    try:
        assert first.returncode is not None, "the old program's child was stopped"
        assert second is not None and second is not first and second.returncode is None
        starter.program = None
        await starter.step()
        assert second.returncode is not None and starter.process is None
    finally:
        await starter.stop()
