"""Standard config trio for the agent: UI prefs + firstRunComplete only.

Topology lives under /v1/components, deliberately NOT here — editing UI
prefs in the generic config editor must never accidentally restructure
the install. See `state.py` for the rationale.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Request

from .. import keyring_store
from .._generated.common_models import (
    ConfigDocument,
    ConfigSchema,
    ConfigTestRequest,
    ConfigTestResult,
    ConfigUpdateRequest,
    ConfigUpdateResult,
)
from ..state import AgentState

log = logging.getLogger(__name__)

router = APIRouter(tags=["config"])


@router.get("/v1/config", response_model=ConfigDocument)
async def get_config(request: Request) -> ConfigDocument:
    state: AgentState = request.app.state.agent_state
    return state.as_config_document()


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema(request: Request) -> ConfigSchema:
    state: AgentState = request.app.state.agent_state
    return state.as_config_schema()


@router.post("/v1/config/test", response_model=ConfigTestResult)
async def test_config(
    request: Request,
    body: ConfigTestRequest | None = None,
) -> ConfigTestResult:
    """Probe the agent's effective config without committing.

    Most config fields here (UI prefs, firstRunComplete) have no
    external dependency to verify, so they always succeed. The
    interesting case is `securityMode=os_keyring` — the OS keyring
    backend varies by platform and can fail silently (Linux without
    a running secret service, Windows without Credential Manager,
    etc.). We probe a round-trip read here so the operator finds
    out at config-time rather than at the next-boot auto-unlock
    attempt.

    Body's `overrides` are honored so the operator can test a
    pending switch BEFORE saving it. v0.2 only consumes the
    `securityMode` override; other fields are no-ops.
    """
    start = time.perf_counter()
    state: AgentState = request.app.state.agent_state
    overrides: dict[str, Any] = (
        body.overrides.model_dump() if body is not None and body.overrides is not None else {}
    )
    effective_mode = (
        overrides.get("securityMode")
        if "securityMode" in overrides
        else state.get_config("securityMode")
    )

    if effective_mode == "os_keyring":
        # Read whatever's there. Returning None is fine — it means
        # the operator hasn't logged in yet, so nothing's stored.
        # A KeyringError raised internally would log + return None
        # too. Either result tells us the backend is functional.
        try:
            keyring_store.get_master_key()
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return ConfigTestResult(
                ok=False,
                component="agent",
                latencyMs=elapsed_ms,
                error=(
                    f"OS keyring probe raised {type(e).__name__}: {e}. "
                    f"securityMode=os_keyring will fail to auto-unlock on "
                    f"next boot. Switch to prompt_on_startup, or install / "
                    f"start the platform's keyring backend (Credential "
                    f"Manager on Windows, Secret Service on Linux, "
                    f"Keychain on macOS)."
                ),
            )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return ConfigTestResult(
            ok=True,
            component="agent",
            latencyMs=elapsed_ms,
            summary=(
                "OS keyring backend is reachable. Auto-unlock will work "
                "on next boot once the operator has logged in (the "
                "master key is persisted on login or on securityMode "
                "transition to os_keyring)."
            ),
        )

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return ConfigTestResult(
        ok=True,
        component="agent",
        latencyMs=elapsed_ms,
        summary=(
            f"securityMode={effective_mode}; no external dependency to "
            f"verify. (Switch to os_keyring to probe the OS secret "
            f"store round-trip.)"
        ),
    )


@router.patch("/v1/config", response_model=ConfigUpdateResult)
async def patch_config(request: Request, body: ConfigUpdateRequest) -> ConfigUpdateResult:
    state: AgentState = request.app.state.agent_state

    # Snapshot the prior securityMode so we can react to a transition.
    # The keyring side-effects (write on flip to os_keyring, delete on
    # flip away) belong at this layer — `state` is a YAML serializer
    # and shouldn't know about OS secret stores.
    prior_mode = state.get_config("securityMode")
    result = state.apply_config_patch(body)
    new_mode = state.get_config("securityMode")

    if prior_mode == "os_keyring" and new_mode != "os_keyring":
        # Operator moved to the stronger boundary. The stored auto-
        # unlock secret must go — otherwise the install would still
        # auto-recover, contradicting the promise of the new mode.
        if keyring_store.delete_master_key():
            log.info(
                "securityMode changed from os_keyring to %s; deleted stored "
                "master key from OS keyring",
                new_mode,
            )
    elif prior_mode != "os_keyring" and new_mode == "os_keyring":
        # Operator opted into auto-unlock. If we already have the
        # master key in memory (logged in), persist it now so the
        # next restart actually auto-recovers. If we don't have it
        # in memory, the next /v1/auth/login will save it instead —
        # both paths converge to "next restart works".
        auth = request.app.state.auth_state
        if auth.has_master_key() and keyring_store.set_master_key(auth.master_key):
            log.info("securityMode changed to os_keyring; persisted master key for auto-unlock")

    return result
