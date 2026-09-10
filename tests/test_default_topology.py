"""The topology every install has, declared on first boot.

The rest of the suite opts out (see `conftest.settings`); these tests opt
back in, because the behaviour under test is precisely what happens when
nobody has declared anything.

The conditions matter more than the happy path. A second node that
self-declared a control root would raise a rival to the install's, and
an existing install that re-seeded on restart would resurrect components
its operator deleted - so "when does this NOT run" carries most of the
weight here.
"""

from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_agent import default_topology
from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest
from eugene_plexus_agent._generated.models import ComponentKind
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.state import AgentState

from .conftest import TEST_PASSPHRASE, StubSupervisor


def _seeding_settings(tmp_path: Path) -> Settings:
    return Settings(config_file=tmp_path / "agent.yaml", default_topology=True)


def _boot(settings: Settings) -> tuple[TestClient, StubSupervisor]:
    """Boot an agent the way one really boots: the lifespan (and so the
    seeding decision) runs on entering the client's context, before any
    operator exists. Auth is attached afterwards, which is also the real
    order - a fresh install answers /healthz before it has a passphrase.
    """
    supervisor = StubSupervisor()
    app = create_app(settings)
    app.state.supervisor = supervisor
    client = TestClient(app)
    return client, supervisor


def _authenticate(client: TestClient) -> None:
    """Initialize on a fresh install, log in on one booted a second time."""
    resp = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    if resp.status_code != 200:
        resp = client.post("/v1/auth/login", json={"passphrase": TEST_PASSPHRASE})
    assert resp.status_code == 200, f"auth failed: {resp.status_code} {resp.text}"
    client.headers["Authorization"] = f"Bearer {resp.json()['sessionToken']}"


def _names(client: TestClient) -> list[str]:
    return sorted(c["name"] for c in client.get("/v1/components").json()["components"])


def test_first_boot_declares_control_gateway_and_library(tmp_path: Path) -> None:
    settings = _seeding_settings(tmp_path)
    client, _ = _boot(settings)
    with client:
        _authenticate(client)
        assert _names(client) == ["control", "gateway", "library"]
        by_name = {c["name"]: c for c in client.get("/v1/components").json()["components"]}
        # Ports, not string suffixes: Pydantic normalizes a URL on the way
        # out and appends the trailing slash, which is the M5 lesson about
        # any byte-identity claim that crosses a serializer.
        assert urlparse(by_name["gateway"]["url"]).port == 8080
        assert urlparse(by_name["library"]["url"]).port == 8082
        assert urlparse(by_name["control"]["url"]).port == 8083
        assert by_name["control"]["kind"] == ComponentKind.control.value


def test_seeded_entries_land_in_the_collection_boot_supervises(tmp_path: Path) -> None:
    """Not a second start path that could drift from the operator's.

    Boot starts whatever `list_topology_entries()` returns, so writing
    there is the whole of "these get supervised like any other entry".
    That the loop then runs is not assertable with an injected stub
    supervisor (it makes `owns_supervisor` false); the live run proves it.
    """
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    default_topology.seed(state)
    assert sorted(e.name for e in state.list_topology_entries()) == [
        "control",
        "gateway",
        "library",
    ]


def test_it_persists_so_the_second_boot_is_not_a_first_one(tmp_path: Path) -> None:
    settings = _seeding_settings(tmp_path)
    client, _ = _boot(settings)
    with client:
        _authenticate(client)
        assert _names(client) == ["control", "gateway", "library"]

    on_disk = yaml.safe_load((tmp_path / "agent.yaml").read_text(encoding="utf-8"))
    assert sorted(c["name"] for c in on_disk["components"]) == ["control", "gateway", "library"]


def test_a_deleted_component_stays_deleted_across_a_restart(tmp_path: Path) -> None:
    """The strongest reason to key on the file and not on emptiness."""
    settings = _seeding_settings(tmp_path)
    client, _ = _boot(settings)
    with client:
        _authenticate(client)
        assert client.delete("/v1/components/gateway").status_code in (200, 204)
        assert _names(client) == ["control", "library"]

    client, _ = _boot(settings)
    with client:
        _authenticate(client)
        assert _names(client) == ["control", "library"]


def test_the_bootstrap_load_in_main_does_not_erase_first_boot(tmp_path: Path) -> None:
    """The test that would have caught the first version of this.

    `__main__` builds a bootstrap AgentState and loads it before
    `create_app` runs, and `load()` writes a defaults file when one is
    missing - so keying on "the config file did not exist" seeded
    perfectly in the unit tests and never once in production. Reproduce
    main's ordering here, because the unit tests bypass it entirely.
    """
    settings = _seeding_settings(tmp_path)
    assert not settings.config_file.exists()
    AgentState(settings.config_file).load()  # what __main__ does, first
    assert settings.config_file.exists(), "load() writes defaults; that is the trap"

    client, _ = _boot(settings)
    with client:
        _authenticate(client)
        assert _names(client) == ["control", "gateway", "library"]


def test_an_existing_but_empty_topology_is_left_alone(tmp_path: Path) -> None:
    """m6 and m7 both write `firstRunComplete: true` with `components: []`
    before starting an agent. A completed install's answer may be 'none'."""
    config = tmp_path / "agent.yaml"
    config.write_text("firstRunComplete: true\ncomponents: []\nruntimes: []\n", encoding="utf-8")
    client, _ = _boot(Settings(config_file=config, default_topology=True))
    with client:
        _authenticate(client)
        assert _names(client) == []


def test_an_enrolled_node_never_raises_a_rival_control_root(tmp_path: Path) -> None:
    """A node that has joined gets its topology from the install."""
    settings = _seeding_settings(tmp_path)
    (tmp_path / "node.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "node-b",
                "controlUrl": "http://10.0.0.1:8083",
                # enrolled == name and controlUrl and signingKey, and the
                # key must decode to 32 bytes to be read at all.
                "signingKey": base64.b64encode(b"k" * 32).decode(),
                "signingKeyId": "1",
                "epoch": 1,
            }
        ),
        encoding="utf-8",
    )
    client, _ = _boot(settings)
    with client:
        _authenticate(client)
        assert "control" not in _names(client)


def test_the_env_var_suppresses_it_for_a_node_about_to_enroll(tmp_path: Path) -> None:
    client, _ = _boot(Settings(config_file=tmp_path / "agent.yaml", default_topology=False))
    with client:
        _authenticate(client)
        assert _names(client) == []


def test_safe_mode_declares_nothing(tmp_path: Path) -> None:
    """Safe mode exists to get a wedged install answering; it must not
    also start writing topology into it."""
    settings = Settings(config_file=tmp_path / "agent.yaml", default_topology=True, safe_mode=True)
    client, _ = _boot(settings)
    with client:
        _authenticate(client)
        assert _names(client) == []


def test_a_component_that_cannot_import_is_not_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declaring what cannot spawn buys a crash loop and a red dashboard."""
    monkeypatch.setattr(
        default_topology,
        "is_installed",
        lambda module: module != "eugene_plexus_gateway",
    )
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    assert default_topology.seed(state) == ["control", "library"]


def test_seeding_is_idempotent_by_name(tmp_path: Path) -> None:
    """A half-seeded install completes rather than conflicting."""
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    assert default_topology.seed(state) == ["control", "gateway", "library"]
    assert default_topology.seed(state) == []


def test_should_seed_is_false_once_setup_has_been_completed(tmp_path: Path) -> None:
    """Even with an empty topology - an operator who deleted everything
    after setup gets to keep an empty install."""
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    assert default_topology.should_seed(state, enrolled=False) is True
    assert default_topology.should_seed(state, enrolled=True) is False
    state.apply_config_patch(ConfigUpdateRequest.model_validate({"firstRunComplete": True}))
    assert default_topology.should_seed(state, enrolled=False) is False


def test_is_installed_is_honest_about_this_interpreter() -> None:
    assert default_topology.is_installed("eugene_plexus_agent") is True
    assert default_topology.is_installed("eugene_plexus_not_a_real_component") is False
