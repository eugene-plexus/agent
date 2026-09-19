"""R2.5 — the knob an operator turns has to survive the next boot.

The reproduction, written before the fix (roadmap §1).

`requestTimeoutSeconds` on a companion driver is the field R2.5 exists
to make useful: the operator whose 30B on CPU needs six minutes raises
it there, through the UI, on the Config tab the tree gives that driver.
The agent then rewrites that file from scratch — on every boot reconcile
and on every re-target — because `_write_config` renders the three
fields it manages and writes the result over whatever was there.

So the fix is not only a longer default. **The agent manages three keys
in that document and owns nothing else in it**, and a field the driver's
own `PATCH /v1/config` wrote is the operator's, not ours.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi.testclient import TestClient

from eugene_plexus_agent import companions

from .conftest import StubSupervisor


def _runtime(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "qwen3-a",
        "engine": "llama_cpp",
        "modelPath": "/models/Qwen3-1.7B-Q8_0.gguf",
    }
    body.update(overrides)
    return body


def _companion_config(client: TestClient) -> Path:
    comps = {c["name"]: c for c in client.get("/v1/components").json()["components"]}
    return Path(comps["qwen3-a-driver"]["spawn"]["configFile"])


async def test_a_boot_reconcile_keeps_the_operators_timeout(
    authed_client: TestClient, settings: Any, stub_supervisor: StubSupervisor
) -> None:
    assert authed_client.post("/v1/runtimes", json=_runtime()).status_code == 201
    path = _companion_config(authed_client)

    # What the driver's own PATCH /v1/config writes: the operator's
    # edit, beside the fields the agent manages.
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["requestTimeoutSeconds"] = 900
    document["logLevel"] = "DEBUG"
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")

    state = authed_client.app.state.agent_state  # type: ignore[attr-defined]
    await companions.reconcile(state, stub_supervisor)

    after = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert after.get("requestTimeoutSeconds") == 900, "the boot reconcile discarded it"
    assert after.get("logLevel") == "DEBUG"
    # And the three the agent does manage are still exactly right.
    assert after["provider"] == "openai_compat_custom"
    assert after["runtimeName"] == "qwen3-a"
    assert after["modelId"] == "Qwen3-1.7B-Q8_0"


async def test_a_retarget_keeps_it_too_and_still_restarts(
    authed_client: TestClient, settings: Any, stub_supervisor: StubSupervisor
) -> None:
    assert authed_client.post("/v1/runtimes", json=_runtime()).status_code == 201
    path = _companion_config(authed_client)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["requestTimeoutSeconds"] = 900
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    stub_supervisor.calls.clear()

    patched = authed_client.patch("/v1/runtimes/qwen3-a", json=_runtime(modelAlias="elsewhere"))
    assert patched.status_code == 200, patched.text

    after = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert after["modelId"] == "elsewhere"
    assert after.get("requestTimeoutSeconds") == 900
    assert ("restart", "qwen3-a-driver") in stub_supervisor.calls


async def test_an_untouched_companion_is_not_rewritten_or_restarted(
    authed_client: TestClient, settings: Any, stub_supervisor: StubSupervisor
) -> None:
    """The `changed` signal has to keep meaning *a managed key moved*,
    or every boot restarts every driver in the install."""
    assert authed_client.post("/v1/runtimes", json=_runtime()).status_code == 201
    path = _companion_config(authed_client)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["requestTimeoutSeconds"] = 900
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    stub_supervisor.calls.clear()

    state = authed_client.app.state.agent_state  # type: ignore[attr-defined]
    await companions.reconcile(state, stub_supervisor)

    assert path.read_text(encoding="utf-8") == before
    assert ("restart", "qwen3-a-driver") not in stub_supervisor.calls


async def test_an_unreadable_companion_config_does_not_stop_the_boot(
    authed_client: TestClient, settings: Any, stub_supervisor: StubSupervisor
) -> None:
    """`degraded-mode-required`, applied to a file we do not own.

    Reading the file is new in R2.5 — before it, the agent only ever
    wrote — so it is a new way for one bad file to take an install down.
    A companion config the operator broke is theirs to fix through the
    driver's own degraded config surface; refusing to reconcile the
    runtime over it would stop every OTHER runtime on the node too.
    """
    assert authed_client.post("/v1/runtimes", json=_runtime()).status_code == 201
    path = _companion_config(authed_client)
    path.write_text("{{{ not yaml at all\n", encoding="utf-8")

    state = authed_client.app.state.agent_state  # type: ignore[attr-defined]
    await companions.reconcile(state, stub_supervisor)

    # The three the agent manages are restored; the unreadable rest is
    # gone, which is the honest outcome — it could not be parsed, so it
    # could not be kept.
    after = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert after["provider"] == "openai_compat_custom"
    assert after["runtimeName"] == "qwen3-a"
