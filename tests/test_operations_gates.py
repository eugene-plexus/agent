"""Row 39: operations rights for people who are not the owner, on the agent.

`specs/docs/design/rows/row-39.md`. An operations token opens exactly the
routes the table in `operations.py` names, with the power it holds; every
other route keeps the owner's gate. Each change is one row in this
machine's operations log, the person read from the token.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from eugene_plexus_agent import operations, tokens

from .conftest import FakeRoot, enroll_app, local_service_token

MODELS = tokens.POWER_OPERATE_MODELS
MACHINES = tokens.POWER_OPERATE_MACHINES

#: The design's table (row-39.md §2), written out again so that a route
#: added to `operations.py` by mistake fails here.
EXPECTED_READS = {
    ("GET", "/v1/node"),
    ("GET", "/v1/node/history"),
    ("GET", "/v1/runtimes"),
    ("GET", "/v1/runtimes/{name}"),
    ("GET", "/v1/engines"),
    ("GET", "/v1/engines/{engine}/install"),
    ("GET", "/v1/updates/all"),
}
EXPECTED_ROUTES = {
    ("POST", "/v1/runtimes"): MODELS,
    ("PATCH", "/v1/runtimes/{name}"): MODELS,
    ("DELETE", "/v1/runtimes/{name}"): MODELS,
    ("POST", "/v1/runtimes/{name}/start"): MODELS,
    ("POST", "/v1/runtimes/{name}/stop"): MODELS,
    ("POST", "/v1/runtimes/{name}/restart"): MODELS,
    ("PUT", "/v1/runtimes/{name}/auto-start"): MODELS,
    ("POST", "/v1/engines/{engine}/install"): MACHINES,
    ("DELETE", "/v1/engines/{engine}/install"): MACHINES,
    ("POST", "/v1/engines/{engine}/uninstall"): MACHINES,
    ("POST", "/v1/node/update/check"): MACHINES,
    ("POST", "/v1/node/update"): MACHINES,
    ("POST", "/v1/updates/all"): MACHINES,
    ("POST", "/v1/components/{name}/restart"): MACHINES,
    ("GET", "/v1/logs"): MACHINES,
    ("GET", "/v1/logs/stream"): MACHINES,
}


def _operations_token(
    root: FakeRoot,
    *powers: str,
    machine: str = "gpu-a",
    sid: str = "admin-1",
    version: int = 2,
    sub: str = "p-dana",
) -> str:
    token, _ = root.token.mint(
        typ=tokens.TYP_OPERATIONS,
        sub=sub,
        aud=[f"node:{machine}"],
        ttl_seconds=tokens.OPERATIONS_TTL_SECONDS,
        extra={
            "ops": list(powers),
            "name": "Dana",
            "sid": sid,
            "administration_version": version,
        },
    )
    return token


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def root(app: FastAPI) -> FakeRoot:
    found = FakeRoot()
    enroll_app(app, found, "gpu-a")
    found.register("gpu-b", tokens.generate_private_key().public_key())
    _reissue(app, found)
    return found


def _reissue(app: FastAPI, root: FakeRoot, **kwargs: Any) -> None:
    bundle = root.bundle(**kwargs)
    tokens.write_bundle_file(Path(app.state.auth_state.trust.bundle_path), bundle)
    app.state.auth_state.trust.load()


def _every_route(app: FastAPI) -> list[tuple[str, str]]:
    """(method, path template) of every route, through included routers
    whether FastAPI flattens them (older) or keeps them (0.16x)."""
    found: set[tuple[str, str]] = set()

    def walk(routes: list[Any], prefix: str) -> None:
        for route in routes:
            included = getattr(route, "original_router", None)
            if included is not None:
                walk(included.routes, prefix + (route.include_context.prefix or ""))
                continue
            if isinstance(route, APIRoute):
                for method in route.methods or ():
                    found.add((method, prefix + route.path))

    walk(app.router.routes, "")
    return sorted(found)


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x-none", path)


def test_the_table_is_the_designs(app: FastAPI) -> None:
    assert operations.OPERATIONS_READS == EXPECTED_READS
    assert {k: v[0] for k, v in operations.OPERATIONS_ROUTES.items()} == EXPECTED_ROUTES


def test_every_mutating_route_is_in_the_table(
    app: FastAPI, client: TestClient, root: FakeRoot
) -> None:
    """A token holding every power, sent to every gated route outside the
    table, is refused with 401 or 403. Table routes are tried one by one
    in the tests below, because they act."""
    everything = _bearer(_operations_token(root, MODELS, MACHINES))
    listed = set(EXPECTED_ROUTES) | EXPECTED_READS
    tried: list[tuple[str, str]] = []
    opened: list[tuple[str, str, int]] = []
    for method, path in _every_route(app):
        if (method, path) in listed or method == "HEAD":
            continue
        body = {} if method in ("POST", "PUT", "PATCH") else None
        # A route that answers with no token at all is public; its own
        # checks are tested where it lives.
        if client.request(method, _concrete(path), json=body).status_code not in (401, 403):
            continue
        tried.append((method, path))
        answer = client.request(method, _concrete(path), headers=everything, json=body)
        if answer.status_code not in (401, 403):
            opened.append((method, path, answer.status_code))
    assert len(tried) > 60, tried
    assert opened == []
    # Each one that names a cause names the owner.
    refused = client.patch("/v1/config", json={}, headers=everything)
    assert refused.status_code in (401, 403)


def test_a_route_outside_the_table_refuses_the_token_even_behind_the_gate(
    app: FastAPI, client: TestClient, root: FakeRoot
) -> None:
    """`POST /v1/runtimes/admission` is behind `operation(None, ...)` as a
    read, but it is not in the read list."""
    answer = client.post(
        "/v1/runtimes/admission",
        json={"name": "x", "engine": "llama_cpp", "modelPath": "/m.gguf"},
        headers=_bearer(_operations_token(root, MODELS, MACHINES)),
    )
    assert answer.status_code == 403
    assert "Only the owner" in answer.json()["detail"]["detail"]


def _declare(client: TestClient, session: str, tmp_path: Path, name: str = "qwen") -> None:
    model = tmp_path / f"{name}.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    made = client.post(
        "/v1/runtimes",
        json={"name": name, "engine": "llama_cpp", "modelPath": str(model)},
        headers=_bearer(session),
    )
    assert made.status_code == 201, made.text


def _owner(client: TestClient, root: FakeRoot) -> str:
    """The owner's session for this machine, as the root signs it."""
    return root.session("node:gpu-a", "control")


def test_stop_with_operate_models(
    app: FastAPI, client: TestClient, root: FakeRoot, tmp_path: Path
) -> None:
    owner = _owner(client, root)
    _declare(client, owner, tmp_path)
    dana = _bearer(_operations_token(root, MODELS))
    assert client.get("/v1/runtimes", headers=dana).status_code == 200
    stopped = client.post("/v1/runtimes/qwen/stop", headers=dana)
    assert stopped.status_code == 202, stopped.text
    # Updates and restarts alone does not stop a model.
    machines = _bearer(_operations_token(root, MACHINES))
    refused = client.post("/v1/runtimes/qwen/stop", headers=machines)
    assert refused.status_code == 403
    assert "Models power" in refused.json()["detail"]["detail"]
    # The record names Dana, from the token, and the owner's own change.
    client.post("/v1/runtimes/qwen/stop", headers=_bearer(owner))
    entries = client.get("/v1/node/operations-log", headers=_bearer(owner)).json()["entries"]
    assert (entries[0]["name"], entries[0]["target"]) == ("Owner", "qwen")
    names = [(e["name"], e["personId"], e["action"], e["status"]) for e in entries]
    assert ("Dana", "p-dana", "stop runtime", 202) in names
    assert ("Owner", "operator", "stop runtime", 202) in names
    assert ("Owner", "operator", "create runtime", 201) in names
    # A refused attempt is not a change, and is not recorded.
    assert all(e["status"] != 403 for e in entries)


def test_a_header_cannot_name_the_person(
    app: FastAPI, client: TestClient, root: FakeRoot, tmp_path: Path
) -> None:
    owner = _owner(client, root)
    _declare(client, owner, tmp_path)
    dana = _bearer(_operations_token(root, MODELS))
    client.post(
        "/v1/runtimes/qwen/stop",
        headers={**dana, "X-Eugene-Person": "Mallory", "X-Forwarded-User": "Mallory"},
    )
    entries = client.get("/v1/node/operations-log", headers=_bearer(owner)).json()["entries"]
    assert entries[0]["name"] == "Dana"


def test_engine_install_needs_operate_machines(
    app: FastAPI, client: TestClient, root: FakeRoot
) -> None:
    models = client.post(
        "/v1/engines/no-such-engine/install",
        headers=_bearer(_operations_token(root, MODELS)),
    )
    assert models.status_code == 403
    assert "Updates and restarts" in models.json()["detail"]["detail"]
    # Past the gate: the engine is unknown, so nothing is fetched.
    machines = client.post(
        "/v1/engines/no-such-engine/install",
        headers=_bearer(_operations_token(root, MACHINES)),
    )
    assert machines.status_code == 404, machines.text


def test_logs_come_with_updates_and_restarts(
    app: FastAPI, client: TestClient, root: FakeRoot, tmp_path: Path
) -> None:
    app.state.log_dir = tmp_path / "logs"
    models = client.get("/v1/logs", headers=_bearer(_operations_token(root, MODELS)))
    assert models.status_code == 403
    machines = client.get("/v1/logs", headers=_bearer(_operations_token(root, MACHINES)))
    assert machines.status_code == 200, machines.text


def test_a_revoked_operations_token_is_refused(
    app: FastAPI, client: TestClient, root: FakeRoot
) -> None:
    later = int(time.time()) + 3600
    for revoked in ("admin-1", tokens.operations_revocation("p-dana", 2)):
        token = _operations_token(root, MODELS)
        assert client.get("/v1/runtimes", headers=_bearer(token)).status_code == 200
        _reissue(app, root, revoked=((revoked, later),))
        answer = client.get("/v1/runtimes", headers=_bearer(token))
        assert answer.status_code == 401
        assert "signed out" in answer.json()["detail"]["detail"]
        _reissue(app, root)


def test_a_token_for_another_machine_is_refused(
    app: FastAPI, client: TestClient, root: FakeRoot
) -> None:
    elsewhere = _bearer(_operations_token(root, MODELS, MACHINES, machine="gpu-b"))
    answer = client.get("/v1/runtimes", headers=elsewhere)
    assert answer.status_code == 401
    assert "node:gpu-a" in answer.json()["detail"]["detail"]


def test_the_operations_log_is_the_owners_and_the_librarys_to_write(
    app: FastAPI, client: TestClient, root: FakeRoot
) -> None:
    owner = _owner(client, root)
    dana = _bearer(_operations_token(root, MODELS, MACHINES))
    assert client.get("/v1/node/operations-log", headers=dana).status_code == 401
    report = {
        "personId": "p-dana",
        "name": "Dana",
        "owner": False,
        "power": MODELS,
        "action": "delete model",
        "target": "qwen3-32b",
        "status": 204,
    }
    library = _bearer(local_service_token(app, "library"))
    gateway = _bearer(local_service_token(app, "gateway"))
    assert client.post("/v1/node/operations-log", json=report, headers=gateway).status_code == 401
    assert client.post("/v1/node/operations-log", json=report, headers=dana).status_code == 401
    assert client.post("/v1/node/operations-log", json=report, headers=library).status_code == 204
    entries = client.get("/v1/node/operations-log", headers=_bearer(owner)).json()["entries"]
    assert entries[0]["component"] == "library"
    assert entries[0]["name"] == "Dana" and entries[0]["target"] == "qwen3-32b"


def test_the_log_keeps_ninety_days_and_ten_thousand_rows(tmp_path: Path) -> None:
    store = operations.OperationsLog(tmp_path / "ops.sqlite3")
    now = time.time()
    row = {"personId": "p", "name": "P", "action": "stop runtime", "status": 202}
    store.record({**row, "target": "old"}, now=now - operations.KEEP_SECONDS - 10)
    store.record({**row, "target": "new"}, now=now)
    assert [e["target"] for e in store.recent()] == ["new"]
    for i in range(operations.KEEP_ROWS + 5):
        store.record({**row, "target": str(i)}, now=now)
    kept = store.recent(limit=1000)
    assert kept[0]["target"] == str(operations.KEEP_ROWS + 4)
    count = store._open().execute("SELECT count(*) FROM operations").fetchone()[0]
    assert count == operations.KEEP_ROWS
