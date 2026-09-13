"""M11: a model the library describes by ITS path, opened where THIS host has it.

Three layers, each with the thing only it can prove:

* admission's `location` -- the resolution, the refusal when nothing is
  there, and the size check against what the library said;
* the routes -- a create that is refused with the fix in prose, `force`
  overriding it, `localPath` on the runtime, the config field and its
  Test; and
* the install-wide library lookup -- a worker with no library of its
  own reaching the install's through the owning node's agent, which is
  what makes its admission `metadata`-based at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent._generated.models import (
    AdmissionBasis,
    AdmissionDecision,
    AdmissionFit,
    EngineKind,
    RuntimeSpec,
)
from eugene_plexus_agent.admission import LibraryFit, LibraryFitClient, check_admission
from eugene_plexus_agent.engines import adapter_for
from eugene_plexus_agent.engines.base import default_model_alias
from eugene_plexus_agent.model_paths import PathRule
from eugene_plexus_agent.runtimes import _RuntimePlanner

from .conftest import StubRuntimeSupervisor, fake_devices
from .test_install_wide_proxy import control_transport, enroll

GIB = 1024**3
NAS = [PathRule(source="/models", target="Z:\\models")]


def _spec(path: str, **overrides: Any) -> RuntimeSpec:
    body: dict[str, Any] = {"name": "m", "engine": "llama_cpp", "modelPath": path}
    body.update(overrides)
    return RuntimeSpec.model_validate(body)


class _FakeLibrary:
    def __init__(self, answer: LibraryFit | None) -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def fit(self, model_path: str, **kwargs: Any) -> LibraryFit | None:
        self.asked.append(model_path)
        return self.answer


# --- admission: location ----------------------------------------------------


@pytest.mark.anyio
async def test_a_model_that_is_not_here_is_refused_with_the_fix() -> None:
    result = await check_admission(
        _spec("/models/q.gguf"),
        snapshot=fake_devices(),
        library=None,
        running=[],
        exists=lambda p: False,
        node_name="Amish_Station",
    )
    assert result.decision is AdmissionDecision.refuse
    assert result.fit is AdmissionFit.unknown
    assert result.location is not None
    assert result.location.exists is False
    assert result.location.localPath == "/models/q.gguf"
    assert result.location.mapping is None
    assert result.reason.startswith("refuse: /models/q.gguf is not on Amish_Station.")
    assert "mount that share here and add a mapping" in result.reason
    assert "Config -> Agent @ Amish_Station -> Model directory mappings" in result.reason
    assert "?force=true" in result.reason


@pytest.mark.anyio
async def test_a_mapping_whose_target_is_missing_names_the_mount() -> None:
    result = await check_admission(
        _spec("/models/q.gguf"),
        snapshot=fake_devices(),
        library=None,
        running=[],
        mappings=NAS,
        exists=lambda p: False,
    )
    assert result.decision is AdmissionDecision.refuse
    assert result.location is not None
    assert result.location.localPath == "Z:\\models\\q.gguf"
    assert result.location.mapping is not None
    assert result.location.mapping.model_dump(by_alias=True) == {
        "from": "/models",
        "to": "Z:\\models",
    }
    assert "The mapping /models -> Z:\\models applied" in result.reason
    assert "Check that the share is mounted at Z:\\models" in result.reason
    assert "is not on this host" in result.reason


@pytest.mark.anyio
async def test_a_mapped_model_that_is_here_is_measured_at_its_local_path() -> None:
    sized: list[str] = []

    def size_of(path: str) -> int | None:
        sized.append(path)
        return 2 * GIB

    library = _FakeLibrary(None)
    result = await check_admission(
        _spec("/models/q.gguf"),
        snapshot=fake_devices(),
        library=library,
        running=[],
        mappings=NAS,
        exists=lambda p: p == "Z:\\models\\q.gguf",
        size_of=size_of,
    )
    assert result.decision is AdmissionDecision.admit
    assert result.basis is AdmissionBasis.file_size
    assert result.location is not None and result.location.exists is True
    assert result.location.sizeBytes == 2 * GIB
    # The disk is asked about the local path; the library about the
    # declared one -- the library's spelling is its identity.
    assert sized == ["Z:\\models\\q.gguf"]
    assert library.asked == ["/models/q.gguf"]


@pytest.mark.anyio
async def test_a_size_that_disagrees_with_the_library_warns_and_admits(tmp_path: Path) -> None:
    here = tmp_path / "q.gguf"
    here.write_bytes(b"x" * 10)
    library = _FakeLibrary(
        LibraryFit(
            required_bytes=GIB,
            verdict="fits",
            context_length=8192,
            size_bytes=4096,
            weights_size_bytes=4096,
        )
    )
    result = await check_admission(
        _spec(str(here)), snapshot=fake_devices(), library=library, running=[]
    )
    assert result.decision is AdmissionDecision.admit
    assert result.location is not None
    assert result.location.sizeBytes == 10
    assert result.location.librarySizeBytes == 4096
    assert result.location.sizeMatchesLibrary is False
    assert result.warning is not None and "the library lists 4096" in result.warning


@pytest.mark.anyio
async def test_a_size_that_agrees_is_recorded_as_agreeing(tmp_path: Path) -> None:
    here = tmp_path / "q.gguf"
    here.write_bytes(b"x" * 10)
    library = _FakeLibrary(
        LibraryFit(required_bytes=GIB, verdict="fits", context_length=8192, weights_size_bytes=10)
    )
    result = await check_admission(
        _spec(str(here)), snapshot=fake_devices(), library=library, running=[]
    )
    assert result.location is not None
    assert result.location.sizeMatchesLibrary is True
    assert result.warning is None


# --- the spawn plan and the observed runtime --------------------------------


def test_the_engine_is_handed_the_local_path_and_the_declaration_is_untouched() -> None:
    spec = _spec("/models/q.gguf")
    adapter = adapter_for(EngineKind.llama_cpp)
    assert adapter is not None
    planner = _RuntimePlanner(
        spec,
        adapter,
        logging.getLogger("test"),
        get_config=lambda key: (
            [{"from": "/models", "to": "Z:\\models"}] if key == "pathMappings" else None
        ),
    )
    launched = planner._launch_spec()
    assert launched.modelPath == "Z:\\models\\q.gguf"
    assert planner.spec.modelPath == "/models/q.gguf"
    assert launched.name == spec.name and launched.engine == spec.engine


def test_no_mapping_hands_the_engine_the_declaration_itself() -> None:
    spec = _spec("C:\\local\\q.gguf")
    adapter = adapter_for(EngineKind.llama_cpp)
    assert adapter is not None
    planner = _RuntimePlanner(spec, adapter, logging.getLogger("test"), get_config=lambda k: [])
    assert planner._launch_spec() is spec


def test_the_default_alias_survives_a_foreign_separator() -> None:
    assert default_model_alias("Z:\\models\\Qwen3-1.7B-Q8_0.gguf") == "Qwen3-1.7B-Q8_0"
    assert default_model_alias("/models/Qwen3-1.7B-Q8_0.gguf") == "Qwen3-1.7B-Q8_0"
    assert default_model_alias("D:\\models\\Qwen3-8B") == "Qwen3-8B"
    assert default_model_alias("/models/Qwen3-8B/") == "Qwen3-8B"


# --- the routes -------------------------------------------------------------


def test_the_config_schema_offers_the_mappings_as_their_own_type(
    authed_client: TestClient,
) -> None:
    schema = authed_client.get("/v1/config/schema").json()
    field = next(f for f in schema["fields"] if f["key"] == "pathMappings")
    assert field["valueType"] == "path_mappings"
    assert field["category"] == "storage"
    assert field["default"] == []
    assert "storage" in schema["categories"]
    assert authed_client.get("/v1/config").json()["pathMappings"] == []


def test_the_config_field_validates_shape_and_not_existence(authed_client: TestClient) -> None:
    bad = authed_client.patch(
        "/v1/config", json={"pathMappings": [{"from": "models", "to": "Z:\\models"}]}
    ).json()
    assert bad["applied"] == []
    assert bad["rejected"][0]["key"] == "pathMappings"
    assert "`from` must be an absolute path" in bad["rejected"][0]["message"]

    good = authed_client.patch(
        "/v1/config", json={"pathMappings": [{"from": "/models", "to": "Q:\\does\\not\\exist"}]}
    ).json()
    assert good["applied"] == ["pathMappings"]
    assert authed_client.get("/v1/config").json()["pathMappings"] == [
        {"from": "/models", "to": "Q:\\does\\not\\exist"}
    ]


def test_a_create_of_a_model_that_is_not_here_is_a_422_naming_the_fix(
    authed_client: TestClient,
) -> None:
    authed_client.app.state.model_exists = lambda p: False  # type: ignore[attr-defined]
    response = authed_client.post(
        "/v1/runtimes",
        json={"name": "ghost", "engine": "llama_cpp", "modelPath": "/models/q.gguf"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]["detail"]
    assert detail.startswith("refuse: /models/q.gguf is not on this host.")
    assert "Model directory mappings" in detail
    assert authed_client.get("/v1/runtimes").json()["runtimes"] == []
    # And no companion driver was declared for it.
    names = [c["name"] for c in authed_client.get("/v1/components").json()["components"]]
    assert "ghost-driver" not in names


def test_force_declares_a_model_that_is_not_here(authed_client: TestClient) -> None:
    authed_client.app.state.model_exists = lambda p: False  # type: ignore[attr-defined]
    response = authed_client.post(
        "/v1/runtimes?force=true",
        json={"name": "ghost", "engine": "llama_cpp", "modelPath": "/models/q.gguf"},
    )
    assert response.status_code == 201


def test_the_dry_run_reports_the_location(authed_client: TestClient) -> None:
    authed_client.patch(
        "/v1/config", json={"pathMappings": [{"from": "/models", "to": "Z:\\models"}]}
    )
    authed_client.app.state.model_exists = lambda p: p == "Z:\\models\\q.gguf"  # type: ignore[attr-defined]
    body = authed_client.post(
        "/v1/runtimes/admission",
        json={"name": "m", "engine": "llama_cpp", "modelPath": "/models/q.gguf"},
    ).json()
    location = body["location"]
    assert location["path"] == "/models/q.gguf"
    assert location["localPath"] == "Z:\\models\\q.gguf"
    assert location["exists"] is True
    # Serialized by alias: the wire says `from`, whatever Python calls it.
    assert location["mapping"] == {"from": "/models", "to": "Z:\\models"}
    assert body["decision"] == "admit"


def test_a_declared_runtime_reports_what_this_host_opens(
    authed_client: TestClient, stub_runtime_supervisor: StubRuntimeSupervisor
) -> None:
    # The real supervisor reads the agent's config live; the stub is
    # wired the same way here.
    stub_runtime_supervisor._get_config = authed_client.app.state.agent_state.get_config  # type: ignore[attr-defined]
    authed_client.app.state.model_exists = lambda p: True  # type: ignore[attr-defined]
    created = authed_client.post(
        "/v1/runtimes",
        json={"name": "qwen", "engine": "llama_cpp", "modelPath": "/models/q.gguf"},
    ).json()
    assert created["modelPath"] == "/models/q.gguf"
    assert created["localPath"] == "/models/q.gguf"

    # A mapping added afterwards is visible on the next read, and the
    # declaration is untouched -- that is what makes it revert itself.
    authed_client.patch(
        "/v1/config", json={"pathMappings": [{"from": "/models", "to": "Z:\\models"}]}
    )
    shown = authed_client.get("/v1/runtimes/qwen").json()
    assert shown["modelPath"] == "/models/q.gguf"
    assert shown["localPath"] == "Z:\\models\\q.gguf"
    assert shown["modelAlias"] == "q"


def test_the_test_button_checks_the_mapping_against_the_disk(
    authed_client: TestClient, tmp_path: Path
) -> None:
    missing = authed_client.post(
        "/v1/config/test",
        json={"overrides": {"pathMappings": [{"from": "/models", "to": str(tmp_path / "nope")}]}},
    ).json()
    assert missing["ok"] is False
    assert "does not exist on this host" in missing["error"]

    present = authed_client.post(
        "/v1/config/test",
        json={"overrides": {"pathMappings": [{"from": "/models", "to": str(tmp_path)}]}},
    ).json()
    assert present["ok"] is True
    assert "exists here" in present["summary"]
    # No library in this topology, and the node is not enrolled.
    assert "could not be consulted" in present["summary"]
    # The securityMode line is still there; the Test button tests the draft.
    assert "securityMode=" in present["summary"]


def test_without_mappings_the_test_button_is_unchanged(authed_client: TestClient) -> None:
    body = authed_client.post("/v1/config/test", json={}).json()
    assert body["ok"] is True
    assert body["summary"].startswith("securityMode=")
    assert body.get("error") is None


# --- the library, through the install ---------------------------------------


class _LibraryUpstream:
    """A library answering behind another node's agent proxy."""

    def __init__(self, *, model: dict[str, Any] | None) -> None:
        self.requests: list[httpx.Request] = []
        self.model = model

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.path.endswith("/v1/models"):
                return httpx.Response(200, json={"models": [self.model] if self.model else []})
            if request.url.path.endswith("/fit"):
                return httpx.Response(
                    200,
                    json={
                        "fit": {
                            "verdict": "fits",
                            "requiredBytes": 3 * GIB,
                            "contextLength": 8192,
                        }
                    },
                )
            return httpx.Response(404)

        return httpx.MockTransport(handler)


def _worker_with_the_librarys_node(app: FastAPI, upstream: _LibraryUpstream) -> None:
    """An enrolled worker declaring no library; the registry puts one on `root`."""
    del app.state.library_fit_client
    enroll(app, name="worker-1")
    app.state.control_transport = control_transport(
        [],
        components=[{"node": "root", "name": "library", "kind": "library"}],
        nodes=[
            {"name": "root", "url": "http://root.invalid:8279/"},
            {"name": "worker-1", "url": "http://worker.invalid:8079/"},
        ],
    )
    app.state.library_transport = upstream.transport()


def test_a_worker_reaches_the_installs_library_through_its_node(
    authed_client: TestClient, tmp_path: Path
) -> None:
    here = tmp_path / "q.gguf"
    here.write_bytes(b"x" * 10)
    upstream = _LibraryUpstream(
        model={
            "id": "abc",
            "path": "/models/q.gguf",
            "sizeBytes": 10,
            "fileCount": 1,
            "files": [{"path": "/models/q.gguf", "role": "weights", "sizeBytes": 10}],
        }
    )
    _worker_with_the_librarys_node(authed_client.app, upstream)  # type: ignore[arg-type]
    authed_client.patch(
        "/v1/config", json={"pathMappings": [{"from": "/models", "to": str(tmp_path)}]}
    )
    # The disk is real here: the mapped file exists in tmp_path.
    authed_client.app.state.model_exists = None  # type: ignore[attr-defined]

    body = authed_client.post(
        "/v1/runtimes/admission",
        json={"name": "m", "engine": "llama_cpp", "modelPath": "/models/q.gguf"},
    ).json()

    assert body["basis"] == "metadata", body
    assert body["decision"] == "admit"
    assert body["location"]["localPath"] == str(here)
    assert body["location"]["librarySizeBytes"] == 10
    assert body["location"]["sizeMatchesLibrary"] is True
    # Through the OWNING NODE's agent proxy, never a component URL.
    assert all(
        str(r.url).startswith("http://root.invalid:8279/api/proxy/library/v1/models")
        for r in upstream.requests
    ), [str(r.url) for r in upstream.requests]
    # With a token this node minted for itself.
    assert all(r.headers.get("authorization", "").startswith("Bearer ") for r in upstream.requests)


def test_the_test_button_walks_the_installs_library_too(
    authed_client: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "q.gguf").write_bytes(b"x" * 10)
    upstream = _LibraryUpstream(
        model={"id": "abc", "path": "/models/q.gguf", "sizeBytes": 10, "fileCount": 1}
    )
    _worker_with_the_librarys_node(authed_client.app, upstream)  # type: ignore[arg-type]
    body = authed_client.post(
        "/v1/config/test",
        json={"overrides": {"pathMappings": [{"from": "/models", "to": str(tmp_path)}]}},
    ).json()
    assert body["ok"] is True, body
    assert "1 of 1 library model under /models reachable" in body["summary"]
    assert "sizes match" in body["summary"]


def test_an_unreachable_control_root_falls_back_to_file_size(
    authed_client: TestClient, tmp_path: Path
) -> None:
    here = tmp_path / "q.gguf"
    here.write_bytes(b"x" * 10)
    del authed_client.app.state.library_fit_client  # type: ignore[attr-defined]
    enroll(authed_client.app, name="worker-1")  # type: ignore[arg-type]
    authed_client.app.state.control_transport = control_transport([], fail=True)  # type: ignore[attr-defined]
    authed_client.app.state.model_exists = None  # type: ignore[attr-defined]
    body = authed_client.post(
        "/v1/runtimes/admission",
        json={"name": "m", "engine": "llama_cpp", "modelPath": str(here)},
    ).json()
    assert body["basis"] == "file_size"
    assert body["decision"] == "admit"


def test_the_library_client_reports_where_it_points() -> None:
    client = LibraryFitClient("http://root.invalid:8279/api/proxy/library/", "tok")
    assert client.base_url == "http://root.invalid:8279/api/proxy/library"
