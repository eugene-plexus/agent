"""The Windows calls a person's worker is started with, faked at the pywin32
boundary (job-sites-own-enrollment.md §2.4): which token the process gets,
whose session it is started in, that it is in a job of its own before it runs,
and which session stands for an account. The real calls are exercised by an
elevated acceptance run; what these hold is the order and the arguments."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_agent import site_workers
from eugene_plexus_agent.site_links import LinkStore
from eugene_plexus_agent.site_workers import WindowsStarter, WorkerProgram

ADA = "S-1-5-21-1-2-3-1001"
BO = "S-1-5-21-1-2-3-1002"


class Token:
    def __init__(self, user: str) -> None:
        self.user = user

    def Close(self) -> None:
        return None


class Job:
    def __init__(self, name: str, events: list[tuple[str, Any]]) -> None:
        self.name, self.events = name, events
        self.members: list[Any] = []
        self.closed = False

    def Close(self) -> None:
        self.closed = True
        self.events.append(("close", self.name))


class Win32:
    """One recording stand-in for the six pywin32 modules a start touches."""

    def __init__(self, sessions: dict[int, str] | None = None) -> None:
        self.events: list[tuple[str, Any]] = []
        self.session_users = sessions or {}
        self.states: dict[int, int] = {}
        self.limit_flags: dict[str, int] = {}
        self.jobs: list[Job] = []
        #: Windows' rule: a process already in a job joins another only while
        #: that one is empty (the agent's own job reaches every child).
        self.nesting_rule = True

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events = self.events
        me = self

        ts = types.ModuleType("win32ts")
        ts.WTS_CURRENT_SERVER_HANDLE = "server"  # type: ignore[attr-defined]
        ts.WTSActive, ts.WTSDisconnected, ts.WTSListen = 0, 4, 6  # type: ignore[attr-defined]
        ts.WTSEnumerateSessions = lambda _handle: [  # type: ignore[attr-defined]
            {"SessionId": ident, "State": me.states.get(ident, 0)} for ident in me.session_users
        ]
        ts.WTSQueryUserToken = lambda ident: Token(me.session_users[ident])  # type: ignore[attr-defined]

        security = types.ModuleType("win32security")
        security.TokenUser = "user"  # type: ignore[attr-defined]
        security.GetTokenInformation = lambda token, _what: [token.user]  # type: ignore[attr-defined]
        security.ConvertSidToStringSid = lambda sid: sid  # type: ignore[attr-defined]

        profile = types.ModuleType("win32profile")
        profile.CreateEnvironmentBlock = lambda *_a: {"USERPROFILE": "C:/people/x"}  # type: ignore[attr-defined]

        con = types.ModuleType("win32con")
        con.STARTF_USESHOWWINDOW, con.SW_HIDE = 1, 0  # type: ignore[attr-defined]
        con.CREATE_SUSPENDED, con.CREATE_UNICODE_ENVIRONMENT = 0x4, 0x400  # type: ignore[attr-defined]

        process = types.ModuleType("win32process")
        process.CREATE_NO_WINDOW, process.CREATE_NEW_PROCESS_GROUP = 0x8000000, 0x200  # type: ignore[attr-defined]
        process.STARTUPINFO = lambda: types.SimpleNamespace()  # type: ignore[attr-defined]

        def create(token: Any, _app: Any, command: str, *rest: Any) -> Any:
            events.append(("create", {"token": token, "flags": rest[3], "command": command}))
            return ("PROCESS", Token("thread"), 4242, 1)

        process.CreateProcessAsUser = create  # type: ignore[attr-defined]
        process.ResumeThread = lambda _t: events.append(("resume", None))  # type: ignore[attr-defined]
        process.TerminateProcess = lambda *_a: events.append(("terminate", None))  # type: ignore[attr-defined]

        job = types.ModuleType("win32job")
        job.JobObjectExtendedLimitInformation = 9  # type: ignore[attr-defined]
        job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000  # type: ignore[attr-defined]

        def create_job(*_a: Any) -> Job:
            made = Job(f"JOB{len(me.jobs) + 1}", events)
            me.jobs.append(made)
            return made

        job.CreateJobObject = create_job  # type: ignore[attr-defined]
        job.QueryInformationJobObject = lambda *_a: {"BasicLimitInformation": {"LimitFlags": 0}}  # type: ignore[attr-defined]

        def set_information(target: Job, _kind: int, info: dict[str, Any]) -> None:
            me.limit_flags[target.name] = info["BasicLimitInformation"]["LimitFlags"]

        def assign(target: Job, process: Any) -> None:
            if me.nesting_rule and target.members:
                raise OSError(5, "AssignProcessToJobObject", "Access is denied.")
            target.members.append(process)
            events.append(("assign", (target.name, process)))

        job.SetInformationJobObject = set_information  # type: ignore[attr-defined]
        job.AssignProcessToJobObject = assign  # type: ignore[attr-defined]

        for module in (ts, security, profile, con, process, job):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(site_workers, "_limited", lambda token: ("LIMITED", token.user))


def program(tmp_path: Path) -> WorkerProgram:
    return WorkerProgram(
        python=tmp_path / "python.exe", channel="chan", host="S-1-5-80-9", servers=tmp_path / "s"
    )


def test_each_workers_job_closes_with_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Win32({7: ADA})
    fake.install(monkeypatch)
    starter = WindowsStarter(LinkStore(tmp_path))
    assert fake.jobs == [], "no job until a worker starts"
    starter._start(ADA, 7, program(tmp_path))
    assert fake.limit_flags == {"JOB1": 0x2000}
    assert starter.running[ADA].job is fake.jobs[0]


def test_a_second_person_gets_a_job_of_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first Windows run: an agent that is itself in a job (as a task is)
    passes it to each worker, and Windows then lets a worker join only an
    empty job. One shared job took the owner's worker and refused Jessie's."""
    fake = Win32({7: ADA, 8: BO})
    fake.install(monkeypatch)
    starter = WindowsStarter(LinkStore(tmp_path))
    starter._start(ADA, 7, program(tmp_path))
    starter._start(BO, 8, program(tmp_path))
    assert [e for e in fake.events if e[0] == "assign"] == [
        ("assign", ("JOB1", "PROCESS")),
        ("assign", ("JOB2", "PROCESS")),
    ]
    assert set(starter.running) == {ADA, BO}


def test_stopping_a_worker_closes_its_job_and_no_one_elses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Win32({7: ADA, 8: BO})
    fake.install(monkeypatch)
    starter = WindowsStarter(LinkStore(tmp_path))
    starter._start(ADA, 7, program(tmp_path))
    starter._start(BO, 8, program(tmp_path))
    starter._stop(ADA)
    assert [job.closed for job in fake.jobs] == [True, False]


def test_a_worker_that_cannot_join_its_job_is_ended_and_its_job_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Win32({7: ADA})
    fake.install(monkeypatch)
    starter = WindowsStarter(LinkStore(tmp_path))
    fake.jobs.append(Job("taken", fake.events))
    sys.modules["win32job"].CreateJobObject = lambda *_a: fake.jobs[0]  # type: ignore[attr-defined]
    fake.jobs[0].members.append("someone")
    with pytest.raises(OSError, match="Access is denied"):
        starter._start(ADA, 7, program(tmp_path))
    kinds = [kind for kind, _ in fake.events]
    assert kinds == ["create", "terminate", "close"] and starter.running == {}


def test_a_worker_gets_the_filtered_token_and_is_in_the_job_before_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Win32({7: ADA})
    fake.install(monkeypatch)
    starter = WindowsStarter(LinkStore(tmp_path))
    starter._start(ADA, 7, program(tmp_path))
    kinds = [kind for kind, _ in fake.events]
    assert kinds == ["create", "assign", "resume"], "suspended, then in the job, then running"
    made = fake.events[0][1]
    assert made["token"] == ("LIMITED", ADA), "never the person's unfiltered token"
    assert made["flags"] & 0x4, "created suspended"
    assert ADA in made["command"]
    assert starter.running[ADA].pid == 4242


def test_a_session_of_someone_else_never_gets_the_linked_accounts_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Win32({7: BO})  # the session the starter was given belongs to BO
    fake.install(monkeypatch)
    starter = WindowsStarter(LinkStore(tmp_path))
    with pytest.raises(RuntimeError, match="not the linked one"):
        starter._start(ADA, 7, program(tmp_path))
    assert fake.events == [] and starter.running == {}


def test_an_active_session_stands_for_an_account_whatever_the_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Win32({1: ADA, 2: ADA, 3: BO, 4: BO})
    fake.states = {1: 4, 2: 0, 3: 0, 4: 4}  # ADA: disconnected first; BO: active first
    fake.install(monkeypatch)
    assert site_workers._session_users() == {ADA: 2, BO: 3}


def test_a_session_that_is_neither_active_nor_disconnected_is_nobodys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = Win32({1: ADA, 2: BO})
    fake.states = {1: 0, 2: 6}  # BO's is a listener
    fake.install(monkeypatch)
    assert site_workers._session_users() == {ADA: 1}
