"""Reach: what is listening, what the firewall says, and the switch.

Hobbyist UX S5 (design `specs/docs/design/hobbyist-ux.md` §7 S5, §6.6,
decision #8).

The assertions that matter here are the ones about **not overclaiming**.
A reach card that says `allowed` when it could not check, or that tells
somebody to restart Eugene when the real problem is a firewall rule, is
worse than no card: it sends a person to fix a thing that is not broken
while the broken thing goes on being broken. So most of this file is
about the `unknown` answers and about which evidence decides which
field.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import off_host, reach
from eugene_plexus_agent._generated.models import (
    AgentRestart,
    BoundAddress,
    HostFirewall,
    Mechanism,
    Verdict,
)
from eugene_plexus_agent.firewall import FirewallQuery, unsupported

NL = chr(10)

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="the Windows detector only runs on Windows"
)

# --------------------------------------------------------------------- #
# Where this host is on its own network
# --------------------------------------------------------------------- #


def test_the_proposed_address_is_not_loopback_on_a_connected_machine() -> None:
    """The measurement the whole slice turns on.

    The address the agent has derived since M7 is the local end of a
    socket to the control root, and on a standalone install the control
    root is on loopback -- so that derivation answers `127.0.0.1`, which
    is the one address that cannot be it. This one reads the routing
    table instead and must never answer loopback.
    """
    host = reach.proposed_host()
    if host is None:
        pytest.skip("this machine has no route off itself")
    assert not host.startswith("127."), host
    assert host != "::1"
    assert reach.proposed_url(8079) == f"http://{host}:8079"


def test_the_proposed_address_costs_no_packet() -> None:
    """A UDP connect to a documentation address must not need the network.

    `192.0.2.1` is TEST-NET-1 and is not routed anywhere. If this ever
    started sending, a machine behind a strict egress filter would hang
    here on every `GET /v1/node`.
    """
    import time

    started = time.perf_counter()
    reach.proposed_host()
    assert time.perf_counter() - started < 0.1


# --------------------------------------------------------------------- #
# What is listening
# --------------------------------------------------------------------- #


def test_a_bind_is_reported_from_the_spawn_not_from_the_setting() -> None:
    bound = reach.bound_addresses(
        [
            reach.Listener("agent", 8079, "0.0.0.0"),
            reach.Listener("gateway", 8080, None),
            reach.Listener("library", 8082, "127.0.0.1"),
        ]
    )
    by_process = {b.process: b for b in bound}
    assert by_process["agent"].host == "0.0.0.0"
    assert by_process["agent"].reachableOffHost is True
    # No bind host named means the component's own loopback default.
    assert by_process["gateway"].host == "127.0.0.1"
    assert by_process["gateway"].reachableOffHost is False
    assert by_process["library"].reachableOffHost is False


def test_a_restart_is_required_only_when_the_agent_is_the_one_behind() -> None:
    wide = BoundAddress(process="agent", host="0.0.0.0", port=8079, reachableOffHost=True)
    narrow = BoundAddress(process="agent", host="127.0.0.1", port=8079, reachableOffHost=False)

    # Advertising an address this agent is not bound to: the whole reason
    # the field exists.
    assert reach.restart_required(advertise_url="http://10.0.0.5:8079", agent_bound=narrow) is True
    assert reach.restart_required(advertise_url="http://10.0.0.5:8079", agent_bound=wide) is False

    # Reach off while the socket is still wide is NOT a restart. Nothing
    # outside can observe the difference, and a restart costs the console.
    assert reach.restart_required(advertise_url=None, agent_bound=wide) is False
    assert reach.restart_required(advertise_url="http://127.0.0.1:8079", agent_bound=wide) is False


def test_an_agent_that_is_not_listening_at_all_needs_a_restart() -> None:
    assert reach.restart_required(advertise_url="http://10.0.0.5:8079", agent_bound=None) is True


# --------------------------------------------------------------------- #
# How this agent would come back
# --------------------------------------------------------------------- #


def test_nothing_supervising_means_nothing_may_stop_it() -> None:
    nothing = AgentRestart(
        mechanism=Mechanism.none, canSelfRestart=False, command="eugene-plexus-agent"
    )
    assert reach.restart_argv(nothing) is None
    ok, detail = reach.spawn_restart(nothing)
    assert ok is False
    # And it says what to type, rather than only that it will not.
    assert "eugene-plexus-agent" in detail


def test_a_restart_asks_the_supervisor_rather_than_spawning_a_replacement() -> None:
    """The property that keeps the process tree intact.

    A detached copy of the agent would not be a child of the service or
    the task, so the next boot would start a *second* agent onto a port
    the orphan still holds -- the stacking failure `ports.py` exists to
    diagnose, manufactured on purpose. Every mechanism's argv must name
    that mechanism's own restart, never this interpreter.
    """
    import sys

    for mechanism in (
        Mechanism.service,
        Mechanism.logon_task,
        Mechanism.systemd,
        Mechanism.launchd,
    ):
        argv = reach.restart_argv(
            AgentRestart(mechanism=mechanism, canSelfRestart=True, command="x")
        )
        assert argv is not None, mechanism
        joined = " ".join(argv)
        assert sys.executable not in joined, (mechanism, joined)
        assert any(word in joined for word in ("sc ", "schtasks", "systemctl", "launchctl")), joined


def test_a_known_mechanism_that_cannot_self_restart_still_refuses() -> None:
    """The case a sabotage escaped, and the reason this test exists.

    Removing the `canSelfRestart` guard from `restart_argv` passed all
    thirty-four tests, because the only case asserted was
    `mechanism: none` -- where no branch matches and the function falls
    through to `None` whether the guard is there or not. A property that
    holds for the wrong reason is not tested.

    The case that matters is a mechanism that *is* detected while the
    tool that drives it is absent: `describe_restart` answers
    `systemd`/`canSelfRestart: False` on a host with no `systemctl`, and
    an unguarded `restart_argv` would hand back a command that cannot
    run -- so the agent would stop and nothing would start it. That is
    an install ended by a browser click.
    """
    for mechanism in (
        Mechanism.systemd,
        Mechanism.launchd,
        Mechanism.service,
        Mechanism.logon_task,
    ):
        stuck = AgentRestart(mechanism=mechanism, canSelfRestart=False, command="by hand")
        assert reach.restart_argv(stuck) is None, mechanism
        ok, _detail = reach.spawn_restart(stuck)
        assert ok is False, mechanism


@windows_only
def test_another_installs_logon_task_is_not_this_process_supervisor() -> None:
    """The defect the acceptance run found, and it was dangerous.

    A throwaway agent started from a shell, in a checkout's own
    virtualenv, on ports +100, reported `logon_task` /
    `canSelfRestart: true` -- because the LIVE install on the same box
    owns a scheduled task by that name. The switch's restart would have
    run `schtasks /End` against the operator's real agent: stopping the
    live install, and starting it again while the throwaway kept
    running. Nothing in the run was harmed only because the run never
    asked for a restart.

    The discriminator is the task's program against this process's
    `sys.prefix`. Not `sys.executable`: in a uv-made virtualenv that is
    the base interpreter under `pythons/cpython-...`, outside the prefix
    and shared between installs -- so comparing it would call the real
    install's own task somebody else's.
    """
    import sys

    from eugene_plexus_agent.reach import task_runs_from_prefix

    exe = os.path.join(sys.prefix, "Scripts", "eugene-plexus-agent.exe")
    mine = "TaskName:      \\EugenePlexusAgent\nTask To Run:   " + exe + " --unattended\n"
    other = os.path.join(
        "C:\\",
        "Users",
        "someone",
        "AppData",
        "Local",
        "EugenePlexus",
        "venv",
        "Scripts",
        "eugene-plexus-agent.exe",
    )
    theirs = "TaskName:      \\EugenePlexusAgent\nTask To Run:   " + other + " --unattended\n"

    assert task_runs_from_prefix(mine, sys.prefix) is True
    assert task_runs_from_prefix(theirs, sys.prefix) is False
    # The base interpreter of a uv virtualenv sits OUTSIDE the prefix,
    # which is why `sys.executable` is the wrong thing to compare -- and
    # on this box it is exactly that, so the assertion below is the
    # measurement rather than a restatement of the rule.
    inside_prefix = os.path.normcase(os.path.normpath(sys.executable)).startswith(
        os.path.normcase(os.path.normpath(sys.prefix)) + os.sep
    )
    assert task_runs_from_prefix("Task To Run:   " + sys.executable + "\n", sys.prefix) is (
        inside_prefix
    )


@windows_only
def test_a_sibling_installs_task_does_not_count_either() -> None:
    """A second sabotage escaped too, and this is what it was.

    Loosening the comparison from the prefix to its PARENT directory
    passed every test, because the only negative case was a task under
    another user's home -- nowhere near this checkout. Two installs side
    by side is the realistic shape: this checkout's own virtualenv and a
    second one beside it, or `EugenePlexus/venv` next to
    `EugenePlexus-dev/venv`. A negative case has to be near the positive
    one to be worth anything.
    """
    from eugene_plexus_agent.reach import task_runs_from_prefix

    sibling = os.path.join(
        os.path.dirname(sys.prefix), ".venv-other", "Scripts", "eugene-plexus-agent.exe"
    )
    assert task_runs_from_prefix("Task To Run:   " + sibling + NL, sys.prefix) is False
    # And the prefix itself, spelled as a prefix of a LONGER name, is not
    # a match either: `.venv2` must not be read as inside `.venv`.
    lookalike = sys.prefix + "2" + os.sep + "Scripts" + os.sep + "eugene-plexus-agent.exe"
    assert task_runs_from_prefix("Task To Run:   " + lookalike + NL, sys.prefix) is False


@windows_only
def test_the_task_detector_actually_consults_the_query(monkeypatch) -> None:
    """The first sabotage escaped because nothing tested the caller.

    Replacing `_windows_task_runs_this_install`'s body with `return True`
    passed all thirty-six tests: every assertion was about the pure
    helper, and the function that decides whether this agent may stop
    itself was untested. That is the same shape as the `canSelfRestart`
    sabotage earlier in this file -- a property held for the wrong
    reason -- and it is the second time in one slice.
    """
    import subprocess as sp

    from eugene_plexus_agent import reach as reach_module

    class _Result:
        def __init__(self, code: int, out: bytes) -> None:
            self.returncode = code
            self.stdout = out

    # COM is the primary reader and this box has a real task, so it has
    # to be taken out of the way for the text path to be the subject.
    monkeypatch.setattr(reach_module, "_task_action_via_com", lambda name: None)
    enc = reach_module.oem_encoding()

    other = os.path.join(
        "C:" + os.sep, "OtherInstall", "venv", "Scripts", "eugene-plexus-agent.exe"
    )
    monkeypatch.setattr(
        sp, "run", lambda *a, **k: _Result(0, ("Task To Run:   " + other + NL).encode(enc))
    )
    assert reach_module._windows_task_runs_this_install() is False

    mine = os.path.join(sys.prefix, "Scripts", "eugene-plexus-agent.exe")
    monkeypatch.setattr(
        sp, "run", lambda *a, **k: _Result(0, ("Task To Run:   " + mine + NL).encode(enc))
    )
    assert reach_module._windows_task_runs_this_install() is True

    # No such task at all.
    monkeypatch.setattr(sp, "run", lambda *a, **k: _Result(1, b""))
    assert reach_module._windows_task_runs_this_install() is False


def test_describe_restart_answers_something_on_this_platform() -> None:
    described = reach.describe_restart()
    assert described.mechanism in set(Mechanism)
    # `canSelfRestart` may only be true where there is something to ask.
    if described.canSelfRestart:
        assert reach.restart_argv(described) is not None


# --------------------------------------------------------------------- #
# The firewall, and the refusal to overclaim
# --------------------------------------------------------------------- #


def test_a_platform_with_no_detector_is_unknown_and_never_allowed() -> None:
    answer = unsupported("no detector here", FirewallQuery(ports=(8079, 8080)))
    assert answer.supported is False
    assert [p.verdict for p in answer.ports or []] == [Verdict.unknown, Verdict.unknown]
    assert answer.detail == "no detector here"


def test_read_firewall_never_raises_and_never_invents_allowed(monkeypatch) -> None:
    """A detector that throws while explaining an unreachable machine
    turns one failure into two. The caller is `GET /v1/node`, which has
    other things to say."""
    import eugene_plexus_agent.firewall as fw

    def _boom(_query: FirewallQuery) -> HostFirewall:
        raise RuntimeError("the firewall service is not running")

    for module_name in ("windows", "linux", "macos"):
        monkeypatch.setattr(
            f"eugene_plexus_agent.firewall.{module_name}.read", _boom, raising=False
        )

    answer = fw.read_firewall(FirewallQuery(ports=(8079,)))
    assert answer.supported is False
    assert (answer.ports or [])[0].verdict is Verdict.unknown
    assert "not running" in (answer.detail or "") or "no firewall detector" in (answer.detail or "")


# --------------------------------------------------------------------- #
# The Windows detector's two measured shapes
# --------------------------------------------------------------------- #


@windows_only
def test_a_program_rule_counts_as_much_as_a_port_rule() -> None:
    """Measured on a real host: the live install was reachable through an
    inbound allow for the **program** -- created by Windows' own Security
    Alert dialog, naming a versioned interpreter path -- and had no rule
    mentioning its ports at all. A port-only detector would have called
    that machine `blocked` while the control root was probing it
    successfully every fifteen seconds."""
    from eugene_plexus_agent.firewall import windows

    rules = [
        windows._Rule(
            name="python.exe",
            action=1,
            profiles=6,  # Private | Public
            ports=(),
            any_port=True,
            program=r"C:\install\python.exe",
            protocol=6,
        )
    ]
    verdict = windows._verdict_for_port(
        8079, rules=rules, active=2, program=r"C:\install\python.exe", blocking=True
    )
    assert verdict.verdict is Verdict.allowed
    # And it says WHICH kind of rule, because a program rule naming a
    # versioned interpreter stops applying the day it is upgraded.
    assert verdict.scope is not None and verdict.scope.value == "program"


@windows_only
def test_an_explicit_block_beats_an_allow() -> None:
    """The fingerprint of somebody clicking Cancel on the Security Alert
    dialog in a session nobody remembers."""
    from eugene_plexus_agent.firewall import windows

    rules = [
        windows._Rule("allow ports", 1, 2, (8079,), False, None, 6),
        windows._Rule("blocked by hand", 0, 2, (8079,), False, None, 6),
    ]
    verdict = windows._verdict_for_port(8079, rules=rules, active=2, program=None, blocking=True)
    assert verdict.verdict is Verdict.blocked
    assert verdict.rule == "blocked by hand"
    assert verdict.remedy and "New-NetFirewallRule" in verdict.remedy


@windows_only
def test_a_rule_that_covers_only_one_active_profile_does_not_count() -> None:
    """A laptop that is Private at home and Public at a cafe. A rule
    scoped to Private only would read as `allowed` while the machine sits
    on a Public network refusing everything."""
    from eugene_plexus_agent.firewall import windows

    private_only = [windows._Rule("home", 1, 2, (8079,), False, None, 6)]
    both_active = 2 | 4
    assert (
        windows._verdict_for_port(
            8079, rules=private_only, active=both_active, program=None, blocking=True
        ).verdict
        is Verdict.blocked
    )
    assert (
        windows._verdict_for_port(
            8079, rules=private_only, active=2, program=None, blocking=True
        ).verdict
        is Verdict.allowed
    )


@windows_only
def test_the_rule_we_add_is_scoped_to_ports_not_to_a_program() -> None:
    """Because the rule already on the measured host is not, and it names
    `...\\pythons\\cpython-3.12.14-...\\python.exe` -- an allow that
    stops applying the day the interpreter is upgraded, silently."""
    from eugene_plexus_agent.firewall import windows

    command = windows.rule_command((8079, 8080))
    assert "-LocalPort 8079,8080" in command
    assert "-Program" not in command
    assert "python" not in command.lower()
    # Private and Domain, never Public: a network Windows has classified
    # as Public is one the person is being told to treat as hostile.
    assert "-Profile Private,Domain" in command


@windows_only
def test_port_ranges_do_not_become_sixty_five_thousand_integers() -> None:
    from eugene_plexus_agent.firewall import windows

    assert windows._parse_ports("8079,8080") == ((8079, 8080), False)
    assert windows._parse_ports("*") == ((), True)
    assert windows._parse_ports("1-65535") == ((), True)
    assert windows._parse_ports("8000-8002") == ((8000, 8001, 8002), False)


# --------------------------------------------------------------------- #
# The off-host witness
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", False),
        ("::1", False),
        ("::ffff:127.0.0.1", False),
        ("192.168.1.20", True),
        ("100.64.0.7", True),
        # Not parseable is not evidence. The field's whole value is that
        # it is evidence, so a caller we cannot identify counts as none.
        ("", False),
        ("testclient", False),
        (None, False),
    ],
)
def test_only_an_identifiable_off_host_caller_counts(host: str | None, expected: bool) -> None:
    assert off_host._is_off_host(host) is expected


def test_the_witness_records_the_address_and_the_time() -> None:
    witness = off_host.OffHostWitness()
    assert witness.address is None
    before = datetime.now(UTC)
    witness.saw("192.168.1.20")
    assert witness.address == "192.168.1.20"
    assert witness.at is not None and witness.at >= before


def test_the_witness_sees_a_real_request_through_the_middleware() -> None:
    """Through an app rather than by calling the method, because the
    thing that could break is the middleware being registered where the
    scope no longer carries a client."""
    app = FastAPI()
    witness = off_host.OffHostWitness()
    off_host.install(app, witness)

    @app.get("/ping")
    def _ping() -> dict[str, str]:
        return {"ok": "yes"}

    with TestClient(app, client=("192.168.1.20", 51234)) as client:
        assert client.get("/ping").status_code == 200
    assert witness.address == "192.168.1.20"

    local = off_host.OffHostWitness()
    app2 = FastAPI()
    off_host.install(app2, local)

    @app2.get("/ping")
    def _ping2() -> dict[str, str]:
        return {"ok": "yes"}

    with TestClient(app2, client=("127.0.0.1", 51234)) as client:
        client.get("/ping")
    assert local.address is None


def test_the_witness_is_not_base_http_middleware() -> None:
    """`BaseHTTPMiddleware` buffers the response body, and this agent
    proxies token streams -- step 1's proxy test *deadlocks* when the
    implementation buffers. The first draft of `off_host.py` said so in
    its docstring and then used `@app.middleware("http")` anyway, which
    is exactly the mistake this asserts against."""
    from starlette.middleware.base import BaseHTTPMiddleware

    assert not issubclass(off_host.OffHostMiddleware, BaseHTTPMiddleware)


# --------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------- #


def _reach(client: TestClient) -> dict:
    body = client.get("/v1/node").json()
    assert "reach" in body, body
    return body["reach"]


def test_get_node_carries_reach(authed_client: TestClient) -> None:
    view = _reach(authed_client)
    assert view["enabled"] is False
    assert view["restartRequired"] is False
    assert view["firewall"]["supported"] is True
    # The agent is always in the bound list -- it is answering this call.
    assert any(b["process"] == "agent" for b in view["boundAddresses"])


def test_turning_reach_on_writes_the_address_and_restarts_the_components(
    authed_client: TestClient, app: FastAPI, stub_supervisor
) -> None:
    result = authed_client.post("/v1/node/reach", json={"enabled": True}).json()
    steps = {s["step"]: s for s in result["steps"]}

    assert steps["advertise"]["ok"] is True
    assert steps["restart_components"]["ok"] is True
    advertised = result["reach"]["advertiseUrl"]
    assert advertised and not advertised.startswith("http://127.")
    # The wire form carries pydantic's trailing slash (`format: uri` ->
    # `AnyUrl`), as `NodeIdentity.advertiseUrl` has since M7; the stored
    # setting is the bare address.
    assert app.state.agent_state.get_config("advertiseUrl") == advertised.rstrip("/")
    # The agent's own socket has not moved -- and says so rather than
    # letting a person believe the switch finished the job.
    assert result["reach"]["restartRequired"] is True
    assert result["restarted"] is False


def test_turning_reach_off_clears_the_address(authed_client: TestClient, app: FastAPI) -> None:
    authed_client.post("/v1/node/reach", json={"enabled": True})
    result = authed_client.post("/v1/node/reach", json={"enabled": False}).json()
    assert result["reach"]["enabled"] is False
    assert app.state.agent_state.get_config("advertiseUrl") is None
    # Off while the socket is still wide is not a restart: nothing
    # outside can observe the difference and a restart costs the console.
    assert result["reach"]["restartRequired"] is False


def test_a_loopback_address_is_refused_rather_than_accepted_as_on(
    authed_client: TestClient,
) -> None:
    """A switch that reads *on* and does nothing is the silent failure the
    whole slice exists to remove."""
    response = authed_client.post(
        "/v1/node/reach", json={"enabled": True, "url": "http://127.0.0.1:8079"}
    )
    assert response.status_code == 422
    assert "loopback" in response.text.lower()


def test_an_explicit_address_wins_over_the_derived_one(
    authed_client: TestClient, app: FastAPI
) -> None:
    """`easy-default-expert-override`: a container published on a
    different port is the case, and the operator's value is used as
    given."""
    result = authed_client.post(
        "/v1/node/reach", json={"enabled": True, "url": "http://10.9.9.9:8279"}
    ).json()
    assert result["reach"]["advertiseUrl"].rstrip("/") == "http://10.9.9.9:8279"
    assert app.state.agent_state.get_config("advertiseUrl") == "http://10.9.9.9:8279"


def test_a_firewall_step_that_fails_does_not_roll_back_the_address(
    authed_client: TestClient, app: FastAPI, monkeypatch
) -> None:
    """The reason `steps` exists rather than a status code. A rule that
    could not be added is not a reason to stop advertising, and an
    operator told "nothing happened" about a change that half happened is
    worse off than one told which half."""
    import eugene_plexus_agent.routes.node as node_routes

    monkeypatch.setattr(
        node_routes, "_change_firewall", lambda enabled, ports: (False, "needs administrator")
    )
    result = authed_client.post(
        "/v1/node/reach", json={"enabled": True, "allowFirewall": True}
    ).json()
    steps = {s["step"]: s for s in result["steps"]}
    assert steps["firewall"]["ok"] is False
    assert steps["advertise"]["ok"] is True
    assert app.state.agent_state.get_config("advertiseUrl") is not None


def test_the_switch_will_not_stop_an_agent_nothing_would_start(
    authed_client: TestClient, app: FastAPI
) -> None:
    """`conftest` pins the mechanism to `none`. Stopping here would end
    the install until somebody typed a command, which a browser click
    must never be able to do."""
    result = authed_client.post(
        "/v1/node/reach", json={"enabled": True, "restartAgent": True}
    ).json()
    steps = {s["step"]: s for s in result["steps"]}
    assert result["restarted"] is False
    assert steps["restart_agent"]["ok"] is False
    assert "eugene-plexus-agent" in steps["restart_agent"]["detail"]


def test_reach_is_operator_only(client: TestClient, app: FastAPI) -> None:
    from eugene_plexus_agent import security

    client.post("/v1/auth/initialize", json={"passphrase": "correct horse battery staple"})
    service = security.issue_service_token(
        signing_key=app.state.auth_state.signing_key, kind="gateway", ttl_seconds=60
    )
    response = client.post(
        "/v1/node/reach",
        json={"enabled": True},
        headers={"Authorization": f"Bearer {service}"},
    )
    assert response.status_code == 401


# --------------------------------------------------------------------------- #
# R2.2 / review §6.3 #34 -- a non-ASCII install prefix
# --------------------------------------------------------------------------- #


@windows_only
def test_a_non_ascii_prefix_survives_the_task_read() -> None:
    r"""**The reproduction: `schtasks` output is not UTF-8.**

    `_windows_task_runs_this_install` ran `schtasks` with
    `encoding="utf-8"`. A console program on Windows writes its output
    in the **OEM code page** -- 437 or 850 on an English install -- so a
    prefix with any non-ASCII character in it comes back with that
    character replaced, the path no longer matches `sys.prefix`, and the
    detector answers *nothing starts this agent automatically*. The
    Reach card then says so, affirmatively, and withholds the one action
    it exists to offer.

    This is not a fixture: the bytes are produced by encoding the path
    the way the OS encodes it, and read back both ways. The wrong
    reading has to fail and the right one has to pass, or the assertion
    is about neither.
    """
    import codecs

    from eugene_plexus_agent.reach import oem_encoding, task_runs_from_prefix

    oem = oem_encoding()
    # A prefix a real person produces: a Windows account name with an
    # accent, which is where `%LOCALAPPDATA%` lives.
    prefix = os.path.join("C:" + os.sep, "Users", "José", "EugenePlexus", "venv")
    exe = os.path.join(prefix, "Scripts", "eugene-plexus-agent.exe")
    line = "Task To Run:   " + exe + " --unattended" + NL

    try:
        raw = line.encode(oem)
    except (UnicodeEncodeError, LookupError):  # pragma: no cover - exotic code page
        pytest.skip(f"{oem} cannot represent the test path")

    # What the console actually hands back, decoded the way the code did.
    as_utf8 = raw.decode("utf-8", errors="replace")
    assert task_runs_from_prefix(as_utf8, prefix) is False, (
        "the defect did not reproduce: decoding OEM bytes as UTF-8 kept the path intact, "
        f"which means {oem} and utf-8 agree about U+00E9 on this box"
    )

    # And decoded the way this module decodes it now.
    assert codecs.lookup(oem)  # the name is a real codec, not a guess
    assert task_runs_from_prefix(raw.decode(oem, errors="replace"), prefix) is True


@windows_only
def test_the_task_detector_prefers_the_com_reader(monkeypatch) -> None:
    """COM is consulted first, and its answer is used as it stands.

    `Schedule.Service` hands back a `str` that Windows decoded itself,
    so the code page cannot corrupt it -- which is why it is first and
    why the `schtasks` path below it is a fallback for an install
    without `pywin32` rather than the primary.

    Driving the caller, not the helper: the sabotage that escaped in S5
    was a helper asserted in isolation while the function that decides
    whether this agent may stop itself went untested.
    """
    from eugene_plexus_agent import reach as reach_module

    mine = os.path.join(sys.prefix, "Scripts", "eugene-plexus-agent.exe")
    other = os.path.join("C:" + os.sep, "OtherInstall", "venv", "Scripts", "agent.exe")

    # COM answers -> schtasks is never run. Proved by making schtasks
    # answer the opposite; if it were consulted the assertion flips.
    monkeypatch.setattr(reach_module, "_task_action_via_com", lambda name: mine)
    monkeypatch.setattr(
        reach_module, "_task_query_via_schtasks", lambda name: "Task To Run:   " + other + NL
    )
    assert reach_module._windows_task_runs_this_install() is True

    monkeypatch.setattr(reach_module, "_task_action_via_com", lambda name: other)
    monkeypatch.setattr(
        reach_module, "_task_query_via_schtasks", lambda name: "Task To Run:   " + mine + NL
    )
    assert reach_module._windows_task_runs_this_install() is False

    # COM unavailable (no pywin32, no such task) -> the text path decides.
    monkeypatch.setattr(reach_module, "_task_action_via_com", lambda name: None)
    assert reach_module._windows_task_runs_this_install() is True
    monkeypatch.setattr(
        reach_module, "_task_query_via_schtasks", lambda name: "Task To Run:   " + other + NL
    )
    assert reach_module._windows_task_runs_this_install() is False

    # Neither answers: no task, so nothing starts this agent.
    monkeypatch.setattr(reach_module, "_task_query_via_schtasks", lambda name: None)
    assert reach_module._windows_task_runs_this_install() is False


@windows_only
def test_the_schtasks_reader_decodes_what_the_console_wrote(monkeypatch) -> None:
    """**A sabotage escaped here, and it named this test.**

    Putting `encoding="utf-8"` back on the `schtasks` read passed every
    other check in this file: one encodes the bytes itself and never
    calls the reader, another patches the reader out, and the caller
    test uses an ASCII path, where cp437 and UTF-8 agree. So the one
    line the finding is about was ungated.

    This drives `_task_query_via_schtasks` with the bytes a console
    hands back -- OEM-encoded, with an accent in the path -- and asserts
    the string that comes out still names the path.
    """
    import subprocess as sp

    from eugene_plexus_agent import reach as reach_module

    oem = reach_module.oem_encoding()
    exe = os.path.join(
        "C:" + os.sep, "Users", "José", "EugenePlexus", "venv", "Scripts", "eugene-plexus-agent.exe"
    )
    try:
        raw = ("Task To Run:   " + exe + " --unattended" + NL).encode(oem)
    except (UnicodeEncodeError, LookupError):  # pragma: no cover - exotic code page
        pytest.skip(f"{oem} cannot represent the test path")

    class _Result:
        returncode = 0
        stdout = raw

    monkeypatch.setattr(sp, "run", lambda *a, **k: _Result())
    got = reach_module._task_query_via_schtasks("whatever")
    assert got is not None
    assert exe in got, f"the reader mangled the path: {got!r}"


# --------------------------------------------------------------------------- #
# R2.6 — the Windows service, and the wait that never happened
# --------------------------------------------------------------------------- #


def test_the_restart_helper_never_reaches_for_timeout_or_powershell() -> None:
    """The reproduction for two measured no-ops, on both Windows branches.

    `timeout /t 3 /nobreak` exits rc 125 in 0.18 s when stdin is DEVNULL,
    which is what `spawn_restart` passes — so the start was issued
    0.18 s after the stop and the "three second" wait never existed.
    And `powershell.exe` under `DETACHED_PROCESS` exits in 0.05 s having
    run nothing, so the obvious fix (`Restart-Service`, which waits for
    you) would replace a short wait with no wait at all and still report
    success.

    Both measurements are in `_retry_until_started`'s docstring. This
    asserts the shape that replaced them, because neither program can be
    caught by a unit test that runs one.
    """
    for mechanism in (Mechanism.service, Mechanism.logon_task):
        argv = reach.restart_argv(AgentRestart(mechanism=mechanism, canSelfRestart=True))
        assert argv is not None
        line = " ".join(argv)
        assert "timeout /t" not in line, f"{mechanism.value} still waits with `timeout`"
        assert "powershell" not in line.lower(), (
            f"{mechanism.value} uses PowerShell, which does not run detached"
        )
        assert "ping -n" in line, f"{mechanism.value} has no pacer between attempts"


def test_the_restart_helper_retries_the_start_rather_than_guessing_a_duration() -> None:
    """A start that succeeds IS the proof the stop finished.

    `sc start` fails while the service is STOP_PENDING and
    `schtasks /Run` fails while the task is still running, so retrying
    the start on a widening pacer needs no sleep to be the right length.
    One attempt would be the old defect with a different spelling.
    """
    for mechanism, starter in (
        (Mechanism.service, "sc start"),
        (Mechanism.logon_task, "schtasks /Run"),
    ):
        argv = reach.restart_argv(AgentRestart(mechanism=mechanism, canSelfRestart=True))
        assert argv is not None
        line = argv[2]
        assert line.count(starter) == len(reach._RESTART_PACES) + 1
        # The first attempt is unpaced: a service that stops in
        # milliseconds must not be made to wait for a pacer.
        assert line.split(" & ", 1)[1].startswith(starter)


@windows_only
def test_session_zero_alone_does_not_make_this_installs_service(monkeypatch) -> None:
    """The S5 finding, on the branch it was never applied to.

    Session 0 is where a service runs. It is also where a scheduled task
    with a SYSTEM principal runs, where `PsExec -s` puts you, and where
    **another install's** service runs. Answering `service` /
    `canSelfRestart: True` for any of those hands the reach switch an
    `sc stop` aimed at somebody else's agent — the most dangerous note
    the S5 acceptance run recorded, reproduced here on the other branch.
    """
    monkeypatch.setattr(reach, "_running_as_windows_service", lambda: True)

    # A service registered from a different install must not count.
    monkeypatch.setattr(
        reach,
        "_service_image_path",
        lambda name: r"C:\Users\someone\AppData\Local\EugenePlexus\venv\pythonservice.exe",
    )
    assert reach._windows_service_runs_this_install() is False

    # Nor may an unreadable service entry fall back to "well, session 0".
    monkeypatch.setattr(reach, "_service_image_path", lambda name: None)
    assert reach._windows_service_runs_this_install() is False

    # This install's own service does count, and the path pywin32 really
    # uses is inside `sys.prefix`.
    monkeypatch.setattr(
        reach,
        "_service_image_path",
        lambda name: os.path.join(sys.prefix, "pythonservice.exe"),
    )
    assert reach._windows_service_runs_this_install() is True

    described = reach._windows_restart()
    assert described.mechanism is Mechanism.service
    assert described.canSelfRestart is True
    assert described.detail, "the service branch must say what a service means"
    assert "before anyone signs in" in described.detail


@windows_only
def test_a_session_zero_process_with_no_service_is_not_supervised(monkeypatch) -> None:
    """`PsExec -s` and a SYSTEM-principal task both land here.

    With no matching service and no matching task, the honest answer is
    `none` — nothing would start this agent again, so the switch must
    refuse to stop it.
    """
    monkeypatch.setattr(reach, "_running_as_windows_service", lambda: True)
    monkeypatch.setattr(reach, "_service_image_path", lambda name: None)
    monkeypatch.setattr(reach, "_windows_task_runs_this_install", lambda: False)

    described = reach._windows_restart()
    assert described.mechanism is Mechanism.none
    assert described.canSelfRestart is False
    assert reach.restart_argv(described) is None
