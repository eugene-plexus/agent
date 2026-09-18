"""R1.5 - the first hour on a box that is not empty and not tidy.

Reproductions first, in the order the roadmap lists them (§2.5):

  * **§6.1 #5** `GET /v1/engines` resolves the host three times per
    request and reaches `api.github.com` synchronously, on the loop
    Home polls every 15 s and `useIssues` polls every 30 s per node.
  * **§6.1 #6** `agent.yaml` is rewritten with a bare `open("w")` and
    loaded uncaught, so a write interrupted halfway is an install that
    will not boot and cannot be repaired from the UI.
  * **§6.1 #7** A taken port. The wizard completes, the component never
    comes up, and the 409 on the way tells a first-time user to edit
    `agent.yaml` by hand.

Every case here is written to fail against the code as it was. The
concurrency one is the only one that measures rather than asserts a
shape, and it is written so that the defect cannot pass it: with a
synchronous route the second request cannot be answered at all while
the first is resolving.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import default_topology
from eugene_plexus_agent import runtimes as runtimes_module
from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest
from eugene_plexus_agent._generated.models import Accelerator, Arch, HostAccelerator, Os
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.settings import Settings
from eugene_plexus_agent.state import UNREADABLE_SUFFIX, AgentState

NO_RELEASES = "eugene_plexus_agent.engines.acquisition.GitHubReleases.list_releases"


def _host() -> HostAccelerator:
    return HostAccelerator(
        os=Os.windows, arch=Arch.x64, accelerator=Accelerator.cuda, acceleratorVersion="13.3"
    )


# --------------------------------------------------------------------------- #
# §6.1 #5 - the engines route, off the loop and asking once
# --------------------------------------------------------------------------- #


def test_describe_engines_resolves_the_host_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three calls per request was the finding; one is the fix.

    Each `detect_host()` shells out to a vendor tool with a 5 s cap, so
    the count is not cosmetic - on a box whose `nvidia-smi` is wedged it
    is the difference between one timeout and six.
    """
    calls = 0

    def counting() -> HostAccelerator:
        nonlocal calls
        calls += 1
        return _host()

    monkeypatch.setattr(runtimes_module, "detect_host", counting)
    # No network: the release list is the other half of this route's cost
    # and is not the subject here.
    monkeypatch.setattr(NO_RELEASES, lambda self, **_: [])

    runtimes_module.describe_engines()

    assert calls == 1, f"describe_engines() resolved the host {calls} times"


async def test_engines_route_does_not_block_the_event_loop(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent request completes while `/v1/engines` is resolving.

    `detect_host` blocks for half a second here, which is what a vendor
    tool does when the driver stack is unhappy. With the route running
    that work on the event loop, `/healthz` cannot be answered until it
    finishes - so the assertion is about the *second* request's latency,
    not about the first's.

    **The clock starts before the first request, and that is the whole
    check.** The first version of this case started it after waiting for
    the engines request to get going, and passed against the defect: a
    blocked loop cannot resume the waiting coroutine either, so the mark
    was taken *after* the half second had already elapsed and `/healthz`
    measured fast from a baseline that had moved. Measured from `t0`, a
    blocked loop shows up as 0.5 s and a threaded one as milliseconds.
    """
    entered = threading.Event()

    def slow() -> HostAccelerator:
        entered.set()
        time.sleep(0.5)
        return _host()

    monkeypatch.setattr(runtimes_module, "detect_host", slow)
    monkeypatch.setattr(NO_RELEASES, lambda self, **_: [])

    transport = httpx.ASGITransport(app=app)
    # `ASGITransport` does not run the lifespan, and the route reads
    # `app.state.agent_state` out of it. Entered by hand rather than
    # reaching for `TestClient`, which is synchronous and so cannot have
    # two requests in flight — which is the whole subject here.
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://agent") as client,
    ):
        # **The token is not scaffolding.** `/v1/engines` is
        # operator-gated, and an unauthenticated version of this test
        # passes against the defect: a 401 is refused before `detect_host`
        # is reached, so nothing blocks and the timing assertion is green
        # about a request that never ran. The first version of this case
        # did exactly that.
        init = await client.post("/v1/auth/initialize", json={"passphrase": "x" * 12})
        assert init.status_code == 200, init.text
        auth = {"Authorization": f"Bearer {init.json()['sessionToken']}"}

        started = time.perf_counter()
        engines = asyncio.create_task(client.get("/v1/engines", headers=auth))
        # Wait for the host resolution to have begun, so this is not a
        # race the fix wins by being first. A `threading.Event` polled
        # from the loop, deliberately: the defect under test BLOCKS the
        # loop, so an `asyncio.Event` set from inside the blocking call
        # could not wake anything either -- and the poll's inability to
        # resume is precisely what the measurement reads.
        while not entered.is_set() and time.perf_counter() - started < 5:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        assert entered.is_set(), "/v1/engines never reached the host probe"

        health = await client.get("/healthz")
        waited = time.perf_counter() - started
        assert health.status_code == 200
        assert waited < 0.25, f"/healthz answered {waited * 1000:.0f} ms after /v1/engines began"
        answered = await engines
        assert answered.status_code == 200, answered.text


def test_a_failed_release_check_is_not_repeated_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GitHub that does not answer is asked once, not once per poll.

    The cache only ever stamped success, so every 15 s poll paid the
    full timeout again. The back-off is what makes an unreachable
    upstream cost one request instead of all of them.
    """
    from eugene_plexus_agent.engines.acquisition import GitHubReleases

    attempts = 0

    def failing(_url: str) -> object:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("api.github.com did not answer")

    releases = GitHubReleases("ggml-org/llama.cpp")
    monkeypatch.setattr(GitHubReleases, "_fetch", staticmethod(failing))

    assert releases.list_releases() == []
    assert releases.list_releases() == []
    assert releases.list_releases() == []

    assert attempts == 1, f"an unreachable upstream was dialled {attempts} times in a row"


def test_the_release_listing_is_not_on_the_download_timeout() -> None:
    """A metadata read that a polled route waits on cannot have a
    download's patience. Sixty seconds is right for half a gigabyte of
    engine and wrong for one JSON body."""
    from eugene_plexus_agent.engines import acquisition

    assert acquisition._METADATA_TIMEOUT_SECONDS < acquisition._DOWNLOAD_TIMEOUT_SECONDS
    assert acquisition._METADATA_TIMEOUT_SECONDS <= 15.0


# --------------------------------------------------------------------------- #
# §6.1 #6 - the config file survives an interrupted write and a bad read
# --------------------------------------------------------------------------- #


def test_an_interrupted_write_leaves_the_previous_config_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reproduction of the non-atomic write, without a power cut.

    `open("w")` truncates before a byte is written, so anything that
    fails between the truncate and the flush leaves an install that
    cannot boot. Temp + `os.replace` cannot: the target is only ever
    swapped for a file that is already complete on disk.
    """
    path = tmp_path / "agent.yaml"
    state = AgentState(path)
    state.load()
    state.apply_config_patch(ConfigUpdateRequest(uiFontSize="large"))
    before = path.read_text(encoding="utf-8")
    assert "large" in before

    def die(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(yaml, "safe_dump", die)
    with pytest.raises(OSError):
        state.apply_config_patch(ConfigUpdateRequest(uiFontSize="small"))

    assert path.read_text(encoding="utf-8") == before, "the interrupted write destroyed the file"


def test_a_truncated_config_comes_up_degraded_with_the_reason_on_the_wire(
    tmp_path: Path,
) -> None:
    """Bad config never crashes a component - `degraded-mode-required`,
    applied at last to the component that owns the rule's own file.

    The escape hatch today is an env var a scheduled-task user cannot
    know or set, and the Windows task gives up after three restarts. So
    this asserts the two things that make it repairable from a browser:
    the process is up, and `/v1/config` answers.
    """
    path = tmp_path / "agent.yaml"
    # Parses as YAML; fails on the way into the model, which is the
    # commonest half-written shape (a list item cut mid-mapping).
    path.write_text("components:\n  - name: gateway\n    kind: not-a-kind\n", encoding="utf-8")

    app = create_app(settings=Settings(config_file=path, default_topology=False))
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "degraded"
        detail = (body.get("details") or {}).get("configError")
        assert detail, f"no reason on the wire: {body}"
        assert "kind" in detail

        resp = client.post("/v1/auth/initialize", json={"passphrase": "x" * 12})
        assert resp.status_code == 200
        token = resp.json()["sessionToken"]
        assert (
            client.get("/v1/config", headers={"Authorization": f"Bearer {token}"}).status_code
            == 200
        )


def test_a_degraded_load_keeps_none_of_the_topology_it_half_read(
    tmp_path: Path,
) -> None:
    """Defaults means defaults, not "whatever parsed before the raise".

    **The reproduction above could not see this**, which a sabotage
    proved: its fixture fails on the FIRST component, so the in-memory
    topology was empty either way. A file whose first entry parses and
    whose second does not is both observable and likelier -- a write cut
    short loses the tail. Left half-loaded, the next `PATCH /v1/config`
    persists a topology nobody declared, turning a damaged file into a
    damaged install.
    """
    path = tmp_path / "agent.yaml"
    path.write_text(
        "components:\n"
        "  - name: gateway\n"
        "    kind: gateway\n"
        "    url: http://127.0.0.1:8080\n"
        "  - name: library\n"
        "    kind: not-a-kind\n",
        encoding="utf-8",
    )
    state = AgentState(path)

    assert state.load_or_degrade() is not None
    assert state.list_topology_entries() == [], "a half-read topology is not a topology"

    # And the repair write does not persist the half-read half.
    state.apply_config_patch(ConfigUpdateRequest(uiFontSize="large"))
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert written["components"] == []
    # The operator's own file is still there to restore from.
    preserved = path.with_suffix(path.suffix + UNREADABLE_SUFFIX)
    assert "not-a-kind" in preserved.read_text(encoding="utf-8")


def test_an_interrupted_write_leaves_no_temp_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost of the atomic write, paid for. A temp file per failed
    write, beside the config the supervisor reads, is litter in the one
    directory an operator is told to look at -- and the reason this
    method exists is that interruptions happen."""
    path = tmp_path / "agent.yaml"
    state = AgentState(path)
    state.load()

    def die(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(yaml, "safe_dump", die)
    with pytest.raises(OSError):
        state.apply_config_patch(ConfigUpdateRequest(uiFontSize="small"))

    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]
    assert leftovers == [], f"left behind {leftovers}"


def test_a_half_written_file_that_held_a_passphrase_refuses_the_wizard(tmp_path: Path) -> None:
    """The two halves of #6 meeting: a truncated write, then the wizard.

    **This is the shape the old write really left**, which is why the
    fixture is a real dump cut short rather than an invented mapping.
    `yaml.safe_dump` sorts keys, so `auth` is written FIRST and a
    half-finished file keeps it and loses the tail — the agent comes up
    degraded with no passphrase in the loaded state, and the UI's next
    move is to offer first-run setup. Accepting it mints a second
    `masterSalt` and orphans every secret the first one sealed,
    silently.
    """
    path = tmp_path / "agent.yaml"
    whole = yaml.safe_dump(
        {
            "auth": {"masterSalt": "c2FsdHlzYWx0eXNhbHQ=", "passphraseHash": "$argon2id$v=19$x"},
            "components": [{"name": "gateway", "kind": "gateway", "url": "http://127.0.0.1:8080"}],
            "firstRunComplete": True,
            "runtimes": [],
        },
        sort_keys=True,
    )
    # Cut inside the components list, which is what a write interrupted
    # part-way through produces.
    cut = whole.index("url: http://127.0.0.1:8080") + 10
    path.write_text(whole[:cut], encoding="utf-8")

    app = create_app(settings=Settings(config_file=path, default_topology=False))
    with TestClient(app) as client:
        assert client.get("/healthz").json()["status"] == "degraded"
        resp = client.post("/v1/auth/initialize", json={"passphrase": "x" * 12})
        assert resp.status_code == 409, resp.text
        detail = resp.json()["detail"]["detail"]
        # Names what is missing, in the word the person used when they
        # set it: `auth` is our word for the block, not theirs.
        assert "passphrase" in detail.lower()
        # And names the remedy, rather than inviting the thing that
        # orphans every secret the old salt sealed.
        assert "restore" in detail.lower()


def test_a_fresh_install_whose_file_never_held_a_passphrase_still_sets_up(
    tmp_path: Path,
) -> None:
    """The refusal above is narrow, and this is the case it must not
    catch.

    `firstRunComplete: true` is **also** how every multi-host acceptance
    script since M0 says *skip onboarding*, and how an enrolled node
    that has no passphrase of its own reads. A first version of the
    refusal keyed on that flag and broke both; only the auth keys
    themselves are evidence that a master salt existed.
    """
    path = tmp_path / "agent.yaml"
    path.write_text(
        yaml.safe_dump({"firstRunComplete": True, "components": [], "runtimes": []}),
        encoding="utf-8",
    )

    app = create_app(settings=Settings(config_file=path, default_topology=False))
    with TestClient(app) as client:
        resp = client.post("/v1/auth/initialize", json={"passphrase": "x" * 12})
        assert resp.status_code == 200, resp.text


# --------------------------------------------------------------------------- #
# §6.1 #7 - a taken port
# --------------------------------------------------------------------------- #


def test_seeding_declares_a_port_nothing_is_holding(tmp_path: Path) -> None:
    """8080 is the commonest occupied port on any development box.

    Seeded onto it, the wizard completes, the gateway never comes up,
    Home shows nothing routable and the Needs-attention card is empty.
    Seeding around it is the same choice the wizard already makes for a
    companion driver's port, applied to the three components nobody
    chose.
    """
    held = socket.socket()
    try:
        held.bind(("127.0.0.1", 8080))
    except OSError:
        pytest.skip("8080 is held by something outside this test; the case is already true")
    held.listen(1)
    try:
        state = AgentState(tmp_path / "agent.yaml")
        state.load()
        declared = default_topology.seed(state)
        assert "gateway" in declared
        entry = state.get_topology_entry("gateway")
        assert entry is not None
        assert not str(entry.url).rstrip("/").endswith(":8080"), (
            f"seeded the gateway onto a port something else is holding: {entry.url}"
        )
    finally:
        held.close()


def test_seeding_keeps_the_default_port_when_it_is_free(tmp_path: Path) -> None:
    """The walk is a fallback, never a preference. Every doc, the UI's
    guessed base URL and every acceptance script assume the specs'
    `servers` defaults, so a free 8080 must still be 8080."""
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    default_topology.seed(state)
    entry = state.get_topology_entry("library")
    assert entry is not None
    assert str(entry.url).rstrip("/").endswith(":8082")


def test_the_already_initialized_409_does_not_send_a_person_into_a_yaml_file(
    tmp_path: Path,
) -> None:
    """The remedy for "this install already has a passphrase" is to sign
    in with it, not to hand-edit the file that holds the install's keys."""
    app = create_app(settings=Settings(config_file=tmp_path / "agent.yaml", default_topology=False))
    with TestClient(app) as client:
        assert client.post("/v1/auth/initialize", json={"passphrase": "x" * 12}).status_code == 200
        again = client.post("/v1/auth/initialize", json={"passphrase": "y" * 12})
        assert again.status_code == 409
        detail = again.json()["detail"]["detail"]
        assert "agent.yaml" not in detail, detail
        assert "by hand" not in detail, detail
        assert "sign in" in detail.lower()
