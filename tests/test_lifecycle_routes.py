"""M6 on the runtime routes: admission, force, stop reasons, gateway auth.

The companion coupling is in test_companions.py; the admission
arithmetic is in test_admission.py. This is the seam between them and
the HTTP surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import eugene_plexus_agent.admission as admission_module
from eugene_plexus_agent import security

from .conftest import StubRuntimeSupervisor, fake_devices

GIB = 1024**3


def _runtime(path: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"name": "big", "engine": "llama_cpp", "modelPath": path}
    body.update(overrides)
    return body


_SIZES: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _fake_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission_module, "model_size_bytes", lambda p: _SIZES.get(p))


def _model(tmp_path: Path, size: int) -> str:
    path = str(tmp_path / f"big-{size}.gguf")
    _SIZES[path] = size
    return path


def _token(client: TestClient, kind: str) -> str:
    signing_key = client.app.state.auth_state.signing_key  # type: ignore[attr-defined]
    return security.issue_service_token(signing_key=signing_key, kind=kind)


# --- admission on the routes --------------------------------------------------


def test_the_dry_run_returns_the_decision_and_declares_nothing(
    authed_client: TestClient, tmp_path: Path
) -> None:
    response = authed_client.post(
        "/v1/runtimes/admission", json=_runtime(_model(tmp_path, 30 * GIB))
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == "refuse"
    assert body["fit"] == "split"
    assert body["basis"] == "file_size"
    assert body["freeBytes"] == 24 * GIB
    assert body["device"]["kind"] == "cuda"
    assert authed_client.get("/v1/runtimes").json()["runtimes"] == []


def test_the_dry_run_is_readable_with_a_service_token(client: TestClient, tmp_path: Path) -> None:
    client.post("/v1/auth/initialize", json={"passphrase": "pw"})
    response = client.post(
        "/v1/runtimes/admission",
        json=_runtime(_model(tmp_path, GIB)),
        headers={"Authorization": f"Bearer {_token(client, 'gateway')}"},
    )
    assert response.status_code == 200
    assert response.json()["decision"] == "admit"


def test_a_launch_that_will_not_fit_is_a_422_with_the_arithmetic(
    authed_client: TestClient, tmp_path: Path, stub_runtime_supervisor: StubRuntimeSupervisor
) -> None:
    response = authed_client.post("/v1/runtimes", json=_runtime(_model(tmp_path, 30 * GIB)))
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]["detail"]
    assert detail.startswith("refuse:")
    assert "?force=true" in detail
    # Nothing declared, nothing spawned.
    assert authed_client.get("/v1/runtimes").json()["runtimes"] == []
    assert ("add_and_start", "big") not in stub_runtime_supervisor.calls
    # And no companion was left behind either.
    assert "big-driver" not in {
        c["name"] for c in authed_client.get("/v1/components").json()["components"]
    }


def test_force_declares_it_anyway(
    authed_client: TestClient, tmp_path: Path, stub_runtime_supervisor: StubRuntimeSupervisor
) -> None:
    response = authed_client.post(
        "/v1/runtimes", params={"force": "true"}, json=_runtime(_model(tmp_path, 30 * GIB))
    )
    assert response.status_code == 201, response.text
    assert ("add_and_start", "big") in stub_runtime_supervisor.calls


def test_auto_start_false_is_not_measured_until_it_starts(
    authed_client: TestClient, tmp_path: Path
) -> None:
    declared = authed_client.post(
        "/v1/runtimes", json=_runtime(_model(tmp_path, 30 * GIB), autoStart=False)
    )
    assert declared.status_code == 201, declared.text
    assert declared.json()["status"] == "stopped"
    assert declared.json()["stopReason"] == "autoStart"

    refused = authed_client.post("/v1/runtimes/big/start")
    assert refused.status_code == 422
    forced = authed_client.post("/v1/runtimes/big/start", params={"force": "true"})
    assert forced.status_code == 202
    assert forced.json()["scheduled"] is True


def test_a_fitting_launch_is_admitted(authed_client: TestClient, tmp_path: Path) -> None:
    response = authed_client.post("/v1/runtimes", json=_runtime(_model(tmp_path, 2 * GIB)))
    assert response.status_code == 201, response.text


def test_admission_measures_against_the_live_detector(
    authed_client: TestClient, tmp_path: Path
) -> None:
    # Swap the detector for a nearly-full card: even a small model refuses.
    authed_client.app.state.device_detector = lambda: fake_devices(free=512 * 1024**2)  # type: ignore[attr-defined]
    response = authed_client.post("/v1/runtimes", json=_runtime(_model(tmp_path, 2 * GIB)))
    assert response.status_code == 422
    assert "0.5 GiB free" in response.json()["detail"]["detail"]


# --- stop reasons ------------------------------------------------------------


def test_stop_records_the_reason_the_caller_gave(authed_client: TestClient, tmp_path: Path) -> None:
    authed_client.post("/v1/runtimes", json=_runtime(_model(tmp_path, GIB)))
    stopped = authed_client.post("/v1/runtimes/big/stop", json={"reason": "idle"})
    assert stopped.status_code == 202
    assert "idle" in stopped.json()["message"]
    runtime = authed_client.get("/v1/runtimes/big").json()
    assert runtime["status"] == "stopped"
    assert runtime["stopReason"] == "idle"


def test_stop_without_a_body_is_the_operator(authed_client: TestClient, tmp_path: Path) -> None:
    authed_client.post("/v1/runtimes", json=_runtime(_model(tmp_path, GIB)))
    authed_client.post("/v1/runtimes/big/stop")
    assert authed_client.get("/v1/runtimes/big").json()["stopReason"] == "operator"


def test_starting_clears_the_reason(authed_client: TestClient, tmp_path: Path) -> None:
    authed_client.post("/v1/runtimes", json=_runtime(_model(tmp_path, GIB)))
    authed_client.post("/v1/runtimes/big/stop", json={"reason": "idle"})
    authed_client.post("/v1/runtimes/big/start")
    runtime = authed_client.get("/v1/runtimes/big").json()
    assert runtime["status"] == "starting"
    assert runtime.get("stopReason") is None


def test_the_declaration_mirrors_the_lifecycle_fields(
    authed_client: TestClient, tmp_path: Path
) -> None:
    body = authed_client.post(
        "/v1/runtimes",
        json=_runtime(_model(tmp_path, GIB), idleUnloadSeconds=600, startOnDemand=True),
    ).json()
    assert body["idleUnloadSeconds"] == 600
    assert body["startOnDemand"] is True
    assert body["autoDriver"] is True


# --- who may stop and start -------------------------------------------------


def test_the_gateway_token_can_stop_and_start(client: TestClient, tmp_path: Path) -> None:
    init = client.post("/v1/auth/initialize", json={"passphrase": "pw"})
    operator = {"Authorization": f"Bearer {init.json()['sessionToken']}"}
    client.post("/v1/runtimes", json=_runtime(_model(tmp_path, GIB)), headers=operator)

    gateway = {"Authorization": f"Bearer {_token(client, 'gateway')}"}
    assert (
        client.post("/v1/runtimes/big/stop", json={"reason": "idle"}, headers=gateway).status_code
        == 202
    )
    assert client.post("/v1/runtimes/big/start", headers=gateway).status_code == 202


def test_other_service_tokens_still_cannot(client: TestClient, tmp_path: Path) -> None:
    init = client.post("/v1/auth/initialize", json={"passphrase": "pw"})
    operator = {"Authorization": f"Bearer {init.json()['sessionToken']}"}
    client.post("/v1/runtimes", json=_runtime(_model(tmp_path, GIB)), headers=operator)

    for kind in ("inference-driver", "library", "control"):
        headers = {"Authorization": f"Bearer {_token(client, kind)}"}
        stop = client.post("/v1/runtimes/big/stop", headers=headers)
        assert stop.status_code == 401, kind
        assert "service:gateway" in stop.json()["detail"]["detail"]
        assert client.post("/v1/runtimes/big/start", headers=headers).status_code == 401, kind
    # Declaring and deleting stay operator-only.
    gateway = {"Authorization": f"Bearer {_token(client, 'gateway')}"}
    assert client.delete("/v1/runtimes/big", headers=gateway).status_code == 401
