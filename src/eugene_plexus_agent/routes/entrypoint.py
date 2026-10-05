"""Operator-only container access setup: preview, apply, status and turn off.

Apply writes the file this agent reads and restarts it in place, on approval:
`entrypoint_setup` puts back what was there before if no operator request
arrives through the new entry point in time (2026-10-05).
"""

import asyncio
import json
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .. import entrypoint_setup
from .._generated.models import (
    EntryPointPreview,
    EntryPointPreviewRequest,
    EntryPointStatus,
    PublicEntryPoint,
)
from ..dependencies import require_operator_session
from ..entrypoint import EntryConfig, validate_with_proxy

router = APIRouter(tags=["node"], dependencies=[Depends(require_operator_session)])


def _validated(body: EntryPointPreviewRequest) -> EntryConfig:
    try:
        return EntryConfig.model_validate(body.configuration)
    except ValidationError as exc:
        detail = "; ".join(
            ".".join(str(part) for part in error["loc"]) + ": " + error["msg"]
            for error in exc.errors(include_input=False, include_context=False)
        )
        raise HTTPException(400, detail=detail) from exc


def _normalized(config: EntryConfig) -> dict[str, object]:
    normalized = config.model_dump(mode="json", exclude_none=True)
    # A Windows console may prepare a Linux container configuration too.
    for name in ("certificate", "private_key", "trusted_ca"):
        value = getattr(config, name)
        if value:
            normalized[name] = value.as_posix()
    return normalized


def _steps(config: EntryConfig) -> list[str]:
    hosts = ", ".join(
        service.host
        + (
            " (sign-in for Workbench only)"
            if config.console_direct and service is config.console
            else ""
        )
        for service in config.services()
    )
    nodes_note = (
        "Enrolled machines reach this one at the nodes name now: update each one's saved "
        "controlUrl to it, keeping its keys and identity."
        if config.nodes
        else "Enrolled machines keep reaching this one at its control port (8083 in the "
        "container), so keep that port published while other machines are enrolled."
    )
    steps = [
        "Configuration validated. DNS, certificate files and external connectivity "
        "have not been tested; no running settings have changed.",
        "Update Workbench in Apps first. Keep the existing /data volume, model mounts "
        "and container settings.",
        "Point these names at " + ("your reverse proxy: " if config.proxy else "Eugene: ") + hosts,
    ]
    if config.proxy:
        steps += [
            "Create one proxy host per name, preserving Host. Forward them all to Eugene "
            f"on {config.proxy.transport.upper()} port {config.listen_port}. Require HTTPS "
            "for browser connections.",
            "Only the proxy addresses entered here may connect. Use the proxy's address as "
            "Eugene sees it: on a shared Docker network, its address on that network, not "
            "its LAN address. The proxy must set X-Forwarded-Proto to https and add the "
            "visitor's address to X-Forwarded-For.",
        ]
        if config.private_http:
            steps.append(
                "Put Eugene and your proxy on one Docker network and give the proxy a fixed "
                "address there, matching this setup. Eugene's other ports need not be "
                "published for the proxy. Use HTTPS transport across machines."
            )
        else:
            steps.append(
                "Restrict the Eugene port to the proxy with the host firewall. Configure "
                "upstream SNI to the public name and verify Eugene's certificate in the "
                "proxy; never disable certificate verification."
            )
    else:
        public_port = urlsplit(config.console.origin).port or 443
        steps.append(
            f"Publish TCP {public_port}:{config.listen_port} on the container. The old "
            "console, gateway and Workbench port mappings stop answering and can be removed."
        )
    if config.acme:
        steps.append(
            "Forward public TCP 443 directly to Eugene. Public DNS A/AAAA records must "
            "reach this server, including IPv6 if advertised. Caddy obtains and renews "
            "certificates using TLS-ALPN; port 80 is not required."
        )
        if config.acme.staging:
            steps.append(
                "Staging certificates are deliberately untrusted by browsers. After testing "
                "issuance, turn staging off and apply again."
            )
    if config.internal_ca:
        steps.append(
            "Trust /data/entrypoint/tls/pki/authorities/local/root.crt on each browser and "
            "node. Keep CA private keys private. Eugene does not install trust on your "
            "devices."
        )
    if config.certificate:
        steps.append(
            "Mount the certificate directory and private key read-only at the configured "
            "absolute paths. The certificate must cover every name. Renewed PEM files "
            "reload automatically; changing CA trust requires a restart."
        )
    if config.trusted_ca:
        steps.append(
            "Mount your private CA's public certificate at trusted_ca so Eugene's services "
            "can verify it. Trust that CA on browsers, proxies and nodes too."
        )
    if config.public_console:
        steps.append(
            "The console answers any network: anyone who can reach it can try to sign in, "
            "and your passphrase is all that stops them. Use a long passphrase used nowhere "
            "else, and watch Logs for failed sign-ins."
        )
    if config.console_direct:
        steps.append(
            "The console stays on its own port (8079 in the container), as now: keep that "
            "port published, and open the console there."
        )
        apply = (
            "Apply from this page: Eugene saves it and restarts, and this page comes back in a "
            "few seconds. Sign in to the console again within 15 minutes, or Eugene goes back "
            "to how it was."
        )
    else:
        apply = (
            "Apply from this page: Eugene saves it and restarts. Then open "
            + config.console.origin
            + " and sign in within 15 minutes, or Eugene goes back to how it was."
        )
    steps += [
        nodes_note,
        apply + " Workbench's sign-in address follows by itself; its chats and sign-ins are kept.",
    ]
    return steps


@router.post("/v1/entrypoint/preview", response_model=EntryPointPreview)
def preview(body: EntryPointPreviewRequest) -> EntryPointPreview:
    config = _validated(body)
    return EntryPointPreview(
        configuration=_normalized(config),
        publicUrls=PublicEntryPoint.model_validate(config.public_urls()),
        instructions=_steps(config),
    )


def _status(request: Request, **extra: object) -> EntryPointStatus:
    settings = request.app.state.settings
    config: EntryConfig | None = getattr(request.app.state, "entrypoint_config", None)
    reason = entrypoint_setup.unavailable_reason(settings)
    confirm_by = getattr(request.app.state, "entrypoint_confirm_by", None)
    values: dict[str, object] = {
        "available": reason is None,
        "unavailableReason": reason,
        "path": str(entrypoint_setup.config_path(settings)),
        "active": config is not None,
        "publicUrls": PublicEntryPoint.model_validate(config.public_urls()) if config else None,
        "configuration": _normalized(config) if config else None,
        "confirmBy": confirm_by if config else None,
        "fallback": settings._entrypoint_fallback,
        "reverted": entrypoint_setup.reverted(settings),
    }
    values.update(extra)
    return EntryPointStatus.model_validate({k: v for k, v in values.items() if v is not None})


@router.get("/v1/entrypoint", response_model=EntryPointStatus, response_model_exclude_none=True)
def get_entry_point(request: Request) -> EntryPointStatus:
    return _status(request)


def _check_files(config: EntryConfig) -> None:
    for item in (config.certificate, config.private_key, config.trusted_ca):
        if item and not Path(item).is_file():
            raise HTTPException(
                400,
                detail=f"{item} does not exist inside the container. Mount it there first.",
            )


def _proxy_check(settings: object, config: EntryConfig) -> None:
    with tempfile.TemporaryDirectory(prefix="eugene-entrypoint-check-") as scratch:
        validate_with_proxy(settings, config, Path(scratch))


@router.post("/v1/entrypoint/apply", response_model=EntryPointStatus, status_code=202)
async def apply_entry_point(body: EntryPointPreviewRequest, request: Request) -> JSONResponse:
    settings = request.app.state.settings
    reason = entrypoint_setup.unavailable_reason(settings)
    if reason:
        raise HTTPException(409, detail=reason)
    config = _validated(body)
    _check_files(config)
    try:
        await asyncio.to_thread(_proxy_check, settings, config)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    confirm_by = await asyncio.to_thread(
        entrypoint_setup.save_applied, settings, json.dumps(_normalized(config), indent=2) + "\n"
    )
    entrypoint_setup.schedule_restart(request.app, "Settings applied an HTTPS setup")
    answer = _status(request).model_copy(
        update={
            "active": True,
            "publicUrls": PublicEntryPoint.model_validate(config.public_urls()),
            "configuration": _normalized(config),
            "confirmBy": confirm_by,
            "fallback": None,
            "reverted": None,
            "restarting": True,
        }
    )
    return JSONResponse(answer.model_dump(mode="json", exclude_none=True), status_code=202)


@router.delete("/v1/entrypoint", response_model=EntryPointStatus, status_code=202)
async def turn_off_entry_point(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    if getattr(request.app.state, "entrypoint_config", None) is None:
        raise HTTPException(
            409, detail="One HTTPS port is not on here, so there is nothing to turn off."
        )
    await asyncio.to_thread(entrypoint_setup.turn_off, settings)
    entrypoint_setup.schedule_restart(request.app, "Settings turned the HTTPS setup off")
    answer = _status(request).model_copy(
        update={
            "active": False,
            "publicUrls": None,
            "configuration": None,
            "confirmBy": None,
            "restarting": True,
        }
    )
    return JSONResponse(answer.model_dump(mode="json", exclude_none=True), status_code=202)
