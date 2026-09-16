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

windows_only = pytest.mark.skipif(
    __import__("sys").platform != "win32", reason="the Windows detector only runs on Windows"
)


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
