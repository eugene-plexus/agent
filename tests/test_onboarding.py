"""Onboarding a machine: the path no test had ever walked.

M9's §0 finding, and the reason this file exists: adding a second machine
meant knowing that `EUGENE_PLEXUS_AGENT_DEFAULT_TOPOLOGY=0` existed
before the agent's first boot, and **every multi-host acceptance script
pre-wrote `firstRunComplete: true` with an empty component list** —
which is the bypass, written so fluently that nobody noticed it was
standing in for a product feature that did not exist.

So what is asserted here is the decision, not the prompt's wording: a
fresh boot with a TTY asks, a fresh boot without one seeds as root, and
`join` produces a node that will not raise a rival control plane on its
next start.

The control root is a fake behind an httpx `MockTransport`, borrowed from
`test_node.py` for the same reason it lives there.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_agent import node_identity, onboarding, tokens
from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest
from eugene_plexus_agent._generated.models import ComponentEntry, ComponentKind, SpawnConfig
from eugene_plexus_agent.default_topology import should_seed
from eugene_plexus_agent.onboarding import JoinRequest
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.state import AgentState

from .conftest import FakeRoot as SigningRoot

# A port nothing listens on, so the advertise-address derivation gets a
# refusal in microseconds instead of a three-second timeout. Every test
# here is about the decision, not about routing.
ROOT_URL = "http://127.0.0.1:9"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def settings_for(tmp_path: Path, **over: Any) -> Settings:
    return Settings(config_file=tmp_path / "agent.yaml", **over)


class FakeRoot:
    """Enough control root to enroll against: it records what it was sent
    and answers with a real bundle, signed by a key of its own, listing
    the token key the node sent."""

    def __init__(self, *, status: int = 201, detail: object = "refused") -> None:
        self.status = status
        self.detail = detail
        self.requests: list[dict[str, Any]] = []
        self.root = SigningRoot()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch `httpx.AsyncClient` so `run_join`, which builds its own
        client with no injection point, still talks to this.

        Deliberately not a new `transport=` parameter threaded through
        the CLI: the production path must be the one under test, and a
        seam that only tests use is a seam that can be right while the
        real call is wrong.
        """
        fake = self

        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            fake.requests.append(body)
            if fake.status != 201:
                return httpx.Response(fake.status, json={"detail": fake.detail})
            fake.root.epoch = 4
            fake.root.register(body["name"], tokens.load_public(body["tokenPublicKey"]))
            return httpx.Response(
                201,
                json={
                    "name": body["name"],
                    "epoch": 4,
                    "trustBundle": {"jws": fake.root.bundle().jws},
                    "controlPublicKey": fake.root.public,
                },
            )

        original = httpx.AsyncClient

        def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.MockTransport(handle)
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched)


@pytest.fixture
def no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device detection shells out to a vendor tool. Irrelevant here and
    slow, so it is replaced with an empty snapshot."""
    from eugene_plexus_agent.engines import devices

    from .conftest import fake_devices

    monkeypatch.setattr(devices, "detect_devices", fake_devices)


# --------------------------------------------------------------------------- #
# When the question gets asked at all
# --------------------------------------------------------------------------- #


def test_a_fresh_boot_is_a_boot_that_would_have_seeded(tmp_path: Path) -> None:
    """The prompt must never appear on a boot with no consequence, so it
    asks exactly the question `should_seed` asks, off the same state."""
    settings = settings_for(tmp_path)
    assert onboarding.is_fresh_boot(settings) is True

    state = AgentState(settings.config_file)
    state.load()
    assert should_seed(state, enrolled=False) is True


def test_a_set_up_install_is_not_a_fresh_boot(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    state = AgentState(settings.config_file)
    state.load()
    state.apply_config_patch(ConfigUpdateRequest(firstRunComplete=True))
    assert onboarding.is_fresh_boot(settings) is False


def test_an_enrolled_node_is_not_a_fresh_boot(tmp_path: Path) -> None:
    """A node that has joined an install gets its topology from the
    install. Asking it again would invite a second control root."""
    settings = settings_for(tmp_path)
    AgentState(settings.config_file).load()
    store = node_identity.NodeIdentityStore(tmp_path / node_identity.NODE_FILE)
    store.ensure_keypair()
    store.record_enrollment(
        name="gpu-box",
        control_url=ROOT_URL,
        epoch=1,
        control_public_key=SigningRoot().public,
        recovery_public_key=None,
        advertise_url=None,
    )
    assert onboarding.is_fresh_boot(settings) is False


def test_the_env_var_escape_hatch_still_silences_the_question(tmp_path: Path) -> None:
    """`EUGENE_PLEXUS_AGENT_DEFAULT_TOPOLOGY=0` is now the
    *non-interactive* expression of "node", rather than a thing operators
    had to discover. It must keep working unchanged, because it is what a
    service unit and a container use."""
    settings = settings_for(tmp_path, default_topology=False)
    assert onboarding.is_fresh_boot(settings) is False


def test_safe_mode_never_asks(tmp_path: Path) -> None:
    """Safe mode is a recovery path. A question there is an obstacle."""
    assert onboarding.is_fresh_boot(settings_for(tmp_path, safe_mode=True)) is False


# --------------------------------------------------------------------------- #
# Answering it
# --------------------------------------------------------------------------- #


def test_join_enrolls_and_the_next_boot_declares_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_devices: None
) -> None:
    """The whole point of the subcommand.

    A worker cannot be onboarded from a browser — its UI is unreachable
    until it advertises non-loopback, which is part of what joining does
    — so this is the path that has to work with nothing running.
    """
    settings = settings_for(tmp_path)
    root = FakeRoot()
    root.install(monkeypatch)

    code = onboarding.run_join(
        JoinRequest(control_url=ROOT_URL, token="join-token", name="worker-1"),
        settings,
    )
    assert code == 0
    assert root.requests[0]["name"] == "worker-1"
    # The signing public key goes with it, or this node could never tell
    # the root it had moved; and the token key, or nothing would trust it.
    assert root.requests[0]["signingPublicKey"]
    assert root.requests[0]["tokenPublicKey"]
    assert (tmp_path / "trust_bundle.json").exists()

    store = node_identity.NodeIdentityStore(tmp_path / node_identity.NODE_FILE)
    store.load()
    assert store.record.enrolled
    assert store.record.epoch == 4

    # And the boot after this one does not raise a rival control plane.
    assert onboarding.is_fresh_boot(settings) is False
    state = AgentState(settings.config_file)
    state.load()
    assert should_seed(state, enrolled=store.record.enrolled) is False


def _seed_control_plane(state: AgentState, tmp_path: Path) -> None:
    """What a machine that started once without being told to join holds:
    the three components the first boot declares, and nothing else."""
    for name, kind, port in (
        ("control", ComponentKind.control, 8083),
        ("gateway", ComponentKind.gateway, 8080),
        ("library", ComponentKind.library, 8082),
    ):
        state.add_topology_entry(
            ComponentEntry(
                name=name,
                kind=kind,
                url=f"http://127.0.0.1:{port}",  # type: ignore[arg-type]
                spawn=SpawnConfig(configFile=str(tmp_path / f"{name}.yaml")),
            )
        )


def _declared(settings: Settings) -> list[str]:
    state = AgentState(settings.config_file)
    state.load()
    return sorted(e.name for e in state.list_topology_entries())


def test_join_discards_a_control_plane_nobody_set_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_devices: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Amended 2026-09-26. This test used to assert a refusal here, and
    that refusal is what stranded a real Windows worker: its service had
    started once before the join, seeded control/gateway/library, and the
    join then told the person to DELETE components over the API or edit
    agent.yaml by hand.

    Nobody can have put anything into a control plane that has no
    passphrase, because the console asks for one before any page opens.
    So the join removes it, and says so. Troy: "Someone running a join
    cmd with a valid token should be assumed to know what they're doing."
    """
    settings = settings_for(tmp_path)
    state = AgentState(settings.config_file)
    state.load()
    _seed_control_plane(state, tmp_path)
    root = FakeRoot()
    root.install(monkeypatch)

    assert onboarding.run_join(JoinRequest(control_url=ROOT_URL, token="t"), settings) == 0
    assert len(root.requests) == 1
    assert _declared(settings) == []
    out = capsys.readouterr().out
    assert "removed the control plane this machine had started on its own" in out
    assert "control, gateway, library" in out

    # And the next boot neither re-seeds nor asks.
    assert onboarding.is_fresh_boot(settings) is False


def test_a_refused_join_discards_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_devices: None
) -> None:
    """The discard waits for the enrollment. A bad or expired token must
    leave the machine exactly as it was, not with its install removed and
    no other to belong to."""
    settings = settings_for(tmp_path)
    state = AgentState(settings.config_file)
    state.load()
    _seed_control_plane(state, tmp_path)
    FakeRoot(status=401).install(monkeypatch)

    assert (
        onboarding.run_join(JoinRequest(control_url=ROOT_URL, token="expired"), settings)
        == onboarding.EXIT_TOKEN_REFUSED
    )
    assert _declared(settings) == ["control", "gateway", "library"]


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ("passphrase", "a passphrase was chosen on it"),
        ("extra-component", "It declares ollama, which only a person adds."),
        ("runtime", "It has models declared to run."),
    ],
)
def test_join_refuses_to_discard_an_install_someone_set_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_devices: None,
    capsys: pytest.CaptureFixture[str],
    setup: str,
    reason: str,
) -> None:
    """Anything a person could have made is kept: joining over it would
    leave two control roots in one install, which is what the seeding
    rule exists to prevent. The refusal names the installer's uninstall,
    never an API call or a file to edit."""
    from eugene_plexus_agent._generated.models import EngineKind, RuntimeSpec

    settings = settings_for(tmp_path)
    state = AgentState(settings.config_file)
    state.load()
    _seed_control_plane(state, tmp_path)
    if setup == "passphrase":
        state.set_passphrase(passphrase_hash="$argon2id$h", master_salt_b64="c2FsdA==")
    elif setup == "extra-component":
        state.add_topology_entry(
            ComponentEntry(
                name="ollama",
                kind=ComponentKind.inference_driver,
                url="http://127.0.0.1:8081",  # type: ignore[arg-type]
                spawn=SpawnConfig(configFile=str(tmp_path / "ollama.yaml")),
            )
        )
    else:
        state.add_runtime(
            RuntimeSpec(
                name="qwen",
                engine=EngineKind.llama_cpp,
                modelPath=str(tmp_path / "qwen.gguf"),
                port=8090,
            )
        )
    before = _declared(settings)
    root = FakeRoot()
    root.install(monkeypatch)

    assert onboarding.run_join(JoinRequest(control_url=ROOT_URL, token="t"), settings) == 2
    assert root.requests == []
    assert _declared(settings) == before
    err = capsys.readouterr().err
    assert reason in err
    # Short enough for any console: one that wraps mid-word read
    # "it has a pas / sphrase" on 2026-09-26.
    assert max(len(line) for line in err.splitlines()) <= 80, err
    assert "-Uninstall" in err and "--uninstall" in err
    assert "DELETE /v1/components" not in err
    assert "agent.yaml" not in err

    # ...and the expert override still goes through, keeping everything,
    # per the standing rule that a refusal must always leave a way past it.
    forced = JoinRequest(control_url=ROOT_URL, token="t", force=True)
    assert onboarding.run_join(forced, settings) == 0
    assert len(root.requests) == 1
    assert _declared(settings) == before


def test_join_reports_a_refusal_without_recording_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_devices: None
) -> None:
    """A half-enrolled node would hold an install's key without the
    install knowing it exists."""
    settings = settings_for(tmp_path)
    FakeRoot(status=401).install(monkeypatch)

    assert (
        onboarding.run_join(JoinRequest(control_url=ROOT_URL, token="expired"), settings)
        == onboarding.EXIT_TOKEN_REFUSED
    )
    store = node_identity.NodeIdentityStore(tmp_path / node_identity.NODE_FILE)
    store.load()
    assert store.record.enrolled is False


@pytest.mark.parametrize("status", [401, 409])
def test_a_refused_token_has_its_own_exit_code_and_says_make_a_new_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_devices: None,
    status: int,
) -> None:
    """401 is a token the root does not know or that expired, 409 one
    already used. Running the same command again cannot work for either,
    so the installers need to tell this apart from every other failure,
    and the words must send the person to make a new token."""
    settings = settings_for(tmp_path)
    # The real root's words, as `problem()` sends them: long enough, with
    # the URL beside them, to run past a console's width unwrapped.
    FakeRoot(
        status=status,
        detail={"title": "Join token rejected", "detail": "join token is unknown or has expired"},
    ).install(monkeypatch)

    code = onboarding.run_join(JoinRequest(control_url=ROOT_URL, token="t"), settings)

    assert code == onboarding.EXIT_TOKEN_REFUSED
    said = capsys.readouterr().err
    assert "did not accept this join token" in said
    assert "join token is unknown or has expired" in " ".join(said.split())
    assert "Make a new one" in said
    assert "Nodes page" in said
    assert "again" not in said
    assert all(len(line) <= 80 for line in said.splitlines())


def test_a_root_that_cannot_answer_is_not_a_refused_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_devices: None
) -> None:
    """A locked or failing root did not look at the token; the same
    command works once it is up, so this must not say "make a new one"."""
    settings = settings_for(tmp_path)
    FakeRoot(status=503).install(monkeypatch)

    assert onboarding.run_join(JoinRequest(control_url=ROOT_URL, token="t"), settings) == 1


def test_an_explicit_advertise_address_is_used_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_devices: None
) -> None:
    """`--advertise` is the expert override for the one thing derivation
    can get wrong: a host reached through NAT, where the local end of the
    socket is not the address anyone else can use."""
    settings = settings_for(tmp_path)
    root = FakeRoot()
    root.install(monkeypatch)

    onboarding.run_join(
        JoinRequest(
            control_url=ROOT_URL,
            token="t",
            advertise_url="http://100.64.0.7:8079",
        ),
        settings,
    )
    assert root.requests[0]["url"] == "http://100.64.0.7:8079"


# --------------------------------------------------------------------------- #
# The prompt itself
# --------------------------------------------------------------------------- #


def test_no_answer_means_start_a_new_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback has to be the behaviour that works unattended. EOF is
    what a pipe gives you, and an agent that treats it as an error is an
    agent that cannot start from a script."""
    monkeypatch.setattr("builtins.input", _raises(EOFError))
    assert onboarding.ask(settings_for(tmp_path)) is None


def test_pressing_enter_means_start_a_new_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", _answers([""]))
    assert onboarding.ask(settings_for(tmp_path)) is None


def test_choosing_join_collects_what_enrolling_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "builtins.input",
        _answers(["2", ROOT_URL, "a-join-token", "worker-1", ""]),
    )
    request = onboarding.ask(settings_for(tmp_path))
    assert request is not None
    assert request.control_url == ROOT_URL
    assert request.token == "a-join-token"
    assert request.name == "worker-1"
    assert request.advertise_url is None


def test_choosing_join_without_a_token_falls_back_rather_than_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator who picked "join" and then had no token should get a
    working single-machine install and a printed way to change their
    mind — not a boot that stops."""
    monkeypatch.setattr("builtins.input", _answers(["2", ROOT_URL, "", "", ""]))
    assert onboarding.ask(settings_for(tmp_path)) is None


def _answers(values: list[str]) -> Any:
    queue = list(values)

    def _input(_prompt: str = "") -> str:
        return queue.pop(0) if queue else ""

    return _input


def _raises(exc: type[BaseException]) -> Any:
    def _input(_prompt: str = "") -> str:
        raise exc()

    return _input


@pytest.mark.parametrize(
    ("control_url", "reason"),
    [
        ("http://127.0.0.1:8083", "It is that install's control host"),
        ("http://192.168.16.252:8283", "It is already joined to http://192.168.16.252:8283."),
    ],
)
def test_an_enrolled_machine_is_named_by_what_it_is(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_devices: None,
    capsys: pytest.CaptureFixture[str],
    control_url: str,
    reason: str,
) -> None:
    """Enrollment says more than the passphrase does, so it is the reason
    given. A machine joined to a root on loopback is the control host of
    an install of its own -- what the setup wizard makes, and what
    Amish_Station turned out to be (2026-09-26), where the join said only
    "it has a passphrase"."""
    settings = settings_for(tmp_path)
    state = AgentState(settings.config_file)
    state.load()
    _seed_control_plane(state, tmp_path)
    state.set_passphrase(passphrase_hash="$argon2id$h", master_salt_b64="c2FsdA==")
    store = node_identity.NodeIdentityStore(tmp_path / node_identity.NODE_FILE)
    store.ensure_keypair()
    store.record_enrollment(
        name="Amish_Station",
        control_url=control_url,
        epoch=1,
        control_public_key=SigningRoot().public,
        recovery_public_key=None,
        advertise_url=None,
    )
    root = FakeRoot()
    root.install(monkeypatch)

    assert onboarding.run_join(JoinRequest(control_url=ROOT_URL, token="t"), settings) == 2
    assert root.requests == []
    err = capsys.readouterr().err
    assert reason in err
    assert max(len(line) for line in err.splitlines()) <= 80, err
