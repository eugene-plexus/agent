"""Operator-only, non-mutating container access setup preview."""

from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from .._generated.models import EntryPointPreview, EntryPointPreviewRequest, PublicEntryPoint
from ..dependencies import require_operator_session
from ..entrypoint import EntryConfig

router = APIRouter(tags=["node"], dependencies=[Depends(require_operator_session)])


@router.post("/v1/entrypoint/preview", response_model=EntryPointPreview)
def preview(body: EntryPointPreviewRequest) -> EntryPointPreview:
    try:
        config = EntryConfig.model_validate(body.configuration)
    except ValidationError as exc:
        detail = "; ".join(
            ".".join(str(part) for part in error["loc"]) + ": " + error["msg"]
            for error in exc.errors(include_input=False, include_context=False)
        )
        raise HTTPException(400, detail=detail) from exc
    steps = [
        "Configuration validated. DNS, certificate files and external connectivity "
        "have not been tested; no running settings have changed.",
        "Update Workbench in Apps. Back up and keep the existing /data volume, "
        "model mounts and container settings.",
        "Point these names at "
        + ("your reverse proxy: " if config.proxy else "Eugene: ")
        + ", ".join(service.host for service in config.services())
        + ".",
        "Save the downloaded configuration as /data/entrypoint.json and set "
        "EUGENE_PLEXUS_AGENT_ENTRYPOINT_CONFIG=/data/entrypoint.json on the container.",
    ]
    if config.proxy:
        steps += [
            "Create one proxy host per public hostname, preserving Host. Forward them all "
            f"to Eugene on {config.proxy.transport.upper()} port {config.listen_port}. "
            "Require HTTPS for browser connections.",
            "Only the configured proxy IPs may connect. The proxy must overwrite "
            "X-Forwarded-Proto with https and append the actual client IP to "
            "X-Forwarded-For (or replace that header with the actual client IP).",
        ]
        if config.private_http:
            steps.append(
                "Use a dedicated same-host Docker network shared only by Eugene and "
                "your proxy. Publish no Eugene ports. Give the proxy a fixed IP "
                "matching this configuration. Use HTTPS transport across machines."
            )
        else:
            steps.append(
                "Restrict the Eugene port to the proxy with the host firewall. "
                "Configure upstream SNI to the public hostname and verify Eugene's "
                "certificate in the proxy; never disable certificate verification."
            )
    else:
        public_port = urlsplit(config.console.origin).port or 443
        steps.append(
            f"Publish only TCP {public_port}:{config.listen_port}. Remove old agent, "
            "gateway, node and Workbench port mappings when recreating the container."
        )
    if config.acme:
        steps.append(
            "Forward public TCP 443 directly to Eugene. Public DNS A/AAAA records "
            "must reach this server, including IPv6 if advertised. Caddy obtains "
            "and renews certificates using TLS-ALPN; port 80 is not required."
        )
        if config.acme.staging:
            steps.append(
                "Staging certificates are deliberately untrusted by browsers. After "
                "testing issuance, set acme.staging to false and recreate the container."
            )
    if config.internal_ca:
        steps.append(
            "Trust /data/entrypoint/tls/pki/authorities/local/root.crt on each browser "
            "and node. Keep CA private keys private. Eugene does not install trust "
            "on your devices."
        )
    if config.certificate:
        steps.append(
            "Mount the certificate directory and private key read-only at the configured "
            "absolute paths. The certificate must cover every hostname. Renewed PEM "
            "files reload automatically; changing CA trust requires a container restart."
        )
    if config.trusted_ca:
        steps.append(
            "Mount your private CA's public certificate at trusted_ca so Eugene's "
            "services can verify it. Trust that CA on browsers, proxies and nodes too. "
            "Changing this trust file requires a container restart."
        )
    steps += [
        "Recreate the container, open "
        + config.console.origin
        + ", then restart Workbench in Apps once to update its sign-in address. "
        "Chats are kept; people sign in again.",
        "Update enrolled workers' saved controlUrl to the nodes origin if it changed, "
        "preserving their keys and identities. Set this node's advertised URL to its "
        "console origin. See the migration guide for backup and rollback steps.",
    ]
    normalized = config.model_dump(mode="json", exclude_none=True)
    # A Windows console may prepare a Linux container configuration too.
    for name in ("certificate", "private_key", "trusted_ca"):
        value = getattr(config, name)
        if value:
            normalized[name] = value.as_posix()
    return EntryPointPreview(
        configuration=normalized,
        publicUrls=PublicEntryPoint.model_validate(config.public_urls()),
        instructions=steps,
    )
