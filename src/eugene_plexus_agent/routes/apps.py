"""`/v1/apps` and `/v1/app-catalogue`: optional spokes on this node.

Every route here is operator-only. An app is installed, started and
configured by a person; nothing in the hub has a reason to, and a
service token that could install code would be the widest credential in
the install.

**The key is minted and revoked with the request's own credential**, the
way the key routes and `install_proxy` already work: on an enrolled node
the operator's token goes to the control root, which holds the registry.
The agent never mints an app's key on its own authority, so an app key
exists only because somebody signed in asked for it.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ValidationError

from .._generated.common_models import (
    ConfigDocument,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    Problem,
)
from .._generated.models import (
    App,
    AppCatalogue,
    AppCatalogueEntry,
    AppInstall,
    AppList,
    AppManifest,
    AppOrigin,
    ClientKeyCreateRequest,
)
from .._http import shared_internal_client
from .._private_files import write_private
from ..apps import AppInstallError, AppKey, AppManager, normalized, pip_requirement
from ..client_key_registry import registry
from ..dependencies import require_operator_session
from .auth import mint_client_key, revoke_client_key_at_authority

log = logging.getLogger(__name__)

router = APIRouter(tags=["apps"], dependencies=[Depends(require_operator_session)])

_APP_CALL_TIMEOUT = 10.0


def _problem(code: int, title: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/agent#{title.replace(' ', '-').lower()}",
            title=title,
            status=code,
            detail=detail,
        ).model_dump(exclude_none=True),
    )


def _apps(request: Request) -> AppManager:
    manager: AppManager | None = getattr(request.app.state, "apps", None)
    if manager is None:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Apps unavailable",
            "This agent is running in safe mode, which supervises no apps.",
        )
    return manager


def _record_or_404(manager: AppManager, app_id: str):  # type: ignore[no-untyped-def]
    record = manager.store.get(app_id)
    if record is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND,
            "No such app",
            f"No app called {app_id!r} is installed on this node.",
        )
    return record


def _enrolled(request: Request) -> bool:
    return registry(request).enrolled


def _custom_allowed(request: Request) -> bool:
    return request.app.state.agent_state.get_config("allowCustomApps") is True


_CUSTOM_OFF = (
    "Adding an app that is not in this release's catalogue is off. It runs that app's own "
    "code as this agent's OS user, so it is an expert setting: turn on 'Allow apps not in the "
    "catalogue' under this agent's Config first."
)


# --------------------------------------------------------------------------- #
# catalogue
# --------------------------------------------------------------------------- #


@router.get("/v1/apps", response_model=AppList, response_model_exclude_none=True)
async def list_apps(request: Request) -> AppList:
    return AppList(apps=_apps(request).views())


@router.get("/v1/app-catalogue", response_model=AppCatalogue, response_model_exclude_none=True)
async def get_catalogue(request: Request) -> AppCatalogue:
    return _apps(request).as_catalogue(enrolled=_enrolled(request))


@router.post(
    "/v1/app-catalogue/custom",
    response_model=AppCatalogueEntry,
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
)
async def add_custom(request: Request, body: AppManifest) -> AppCatalogueEntry:
    manager = _apps(request)
    if not _custom_allowed(request):
        raise _problem(status.HTTP_403_FORBIDDEN, "Custom apps are off", _CUSTOM_OFF)
    if manager.manifest(body.id) is not None:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "App id taken",
            f"An app called {body.id!r} is already in this node's catalogue. Pick another id.",
        )
    try:
        requirement = pip_requirement(body)
    except ValueError as exc:
        raise _problem(422, "Unusable source", str(exc)) from exc
    source = body.source.strip()
    is_url = source.startswith(("https://", "http://", "file://"))
    if not is_url and not await asyncio.to_thread(Path(source).is_dir):
        raise _problem(
            422,
            "Unusable source",
            f"{source} is not a folder on this machine. A folder source must be the package's "
            "own directory, on the machine this agent runs on.",
        )
    manager.store.add_custom(body)
    log.info("custom app %r added (%s)", body.id, requirement)
    return AppCatalogueEntry(manifest=normalized(body), origin=AppOrigin.custom)


@router.delete("/v1/app-catalogue/custom/{app_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_custom(request: Request, app_id: str) -> Response:
    manager = _apps(request)
    if all(m.id != app_id for m in manager.store.custom()):
        raise _problem(
            status.HTTP_404_NOT_FOUND,
            "No such custom app",
            f"{app_id!r} is not a custom entry on this node.",
        )
    if manager.store.get(app_id) is not None:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "App is installed",
            f"Uninstall {app_id!r} first; its entry is what it runs from.",
        )
    # A key minted for an install that never finished belongs to nothing
    # once its entry is gone, and must not outlive it.
    await _revoke_key(request, manager, app_id)
    manager.store.remove_custom(app_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------- #
# one app
# --------------------------------------------------------------------------- #


@router.get("/v1/apps/{app_id}", response_model=App, response_model_exclude_none=True)
async def get_app(request: Request, app_id: str) -> App:
    manager = _apps(request)
    return manager.view(_record_or_404(manager, app_id))


async def _revoke_key(request: Request, manager: AppManager, app_id: str) -> None:
    """Revoke the app's key at the authority, or raise and change nothing.

    A key the standalone registry has never heard of is already as
    revoked as it can be; every other refusal -- the control root down,
    the operator's credential refused -- stops the caller before anything
    has been removed, so an app is never gone while its key still works.
    """
    key = manager.store.key(app_id)
    if key is None:
        return
    try:
        await revoke_client_key_at_authority(
            request, key.key_id, authorization=request.headers.get("authorization")
        )
    except HTTPException as exc:
        if exc.status_code != status.HTTP_404_NOT_FOUND:
            raise
    manager.store.drop_key(app_id)
    log.info("revoked key %r (id %s) for app %s", key.key_name, key.key_id, app_id)


@router.delete("/v1/apps/{app_id}", status_code=status.HTTP_204_NO_CONTENT)
async def uninstall(request: Request, app_id: str, purge: bool = False) -> Response:
    manager = _apps(request)
    if manager.store.get(app_id) is None and manager.store.key(app_id) is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND,
            "No such app",
            f"No app called {app_id!r} is installed on this node.",
        )
    await _revoke_key(request, manager, app_id)
    await manager.uninstall(app_id, purge=purge)
    log.info("uninstalled app %s%s", app_id, " and its data" if purge else "")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _ensure_key(request: Request, manager: AppManager, app_id: str) -> None:
    """Mint the app's key on its first install, with the caller's credential.

    A key whose file has gone missing is replaced -- the old one revoked
    first -- which is the repair for a deleted `data` directory.
    """
    store = manager.store
    existing = store.key(app_id)
    key_file = store.key_file(app_id)
    if existing is not None and key_file.is_file():
        return
    if existing is not None:
        await _revoke_key(request, manager, app_id)
    node = manager.node_name() or "this-node"
    created = await mint_client_key(
        request,
        ClientKeyCreateRequest(name=f"app:{app_id}@{node}"),
        authorization=request.headers.get("authorization"),
    )
    key_file.parent.mkdir(parents=True, exist_ok=True)
    write_private(key_file, created.token)
    store.put_key(app_id, AppKey(key_id=created.key.id, key_name=created.key.name))
    log.info("minted key %r (id %s) for app %s", created.key.name, created.key.id, app_id)


@router.post(
    "/v1/apps/{app_id}/install",
    response_model=AppInstall,
    response_model_exclude_none=True,
    status_code=status.HTTP_202_ACCEPTED,
)
async def install(request: Request, app_id: str) -> AppInstall:
    manager = _apps(request)
    found = manager.manifest(app_id)
    if found is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND,
            "Not in the catalogue",
            f"No app called {app_id!r} is in this node's catalogue. GET /v1/app-catalogue "
            "lists what is.",
        )
    manifest, origin = found
    reason = manager.installable(enrolled=_enrolled(request))
    if reason is not None:
        raise _problem(422, "Apps cannot be installed here", reason)
    if origin is AppOrigin.custom and not _custom_allowed(request):
        raise _problem(status.HTTP_403_FORBIDDEN, "Custom apps are off", _CUSTOM_OFF)
    if manager.installer.running(app_id):
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Install already running",
            f"An install of {app_id!r} is in flight. Wait for it, or cancel it.",
        )
    existing = manager.store.get(app_id)
    if existing is not None and existing.version == manifest.version:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "Already installed",
            f"{manifest.name} {manifest.version} is already installed on this node.",
        )
    await _ensure_key(request, manager, app_id)
    uv = manager.uv()
    assert uv is not None  # installable() just said so

    async def on_installed(m: AppManifest) -> None:
        await manager.installed_callback(m, origin)

    try:
        return manager.installer.start(
            manifest, store=manager.store, uv=uv, on_installed=on_installed
        )
    except AppInstallError as exc:
        raise _problem(status.HTTP_409_CONFLICT, "Install already running", str(exc)) from exc


@router.get(
    "/v1/apps/{app_id}/install", response_model=AppInstall, response_model_exclude_none=True
)
async def get_install(request: Request, app_id: str) -> AppInstall:
    snapshot = _apps(request).installer.snapshot(app_id)
    if snapshot is None:
        raise _problem(
            status.HTTP_404_NOT_FOUND,
            "No install",
            f"No install of {app_id!r} has been started since this agent started.",
        )
    return snapshot


@router.delete(
    "/v1/apps/{app_id}/install", response_model=AppInstall, response_model_exclude_none=True
)
async def cancel_install(request: Request, app_id: str) -> AppInstall:
    manager = _apps(request)
    if not manager.installer.running(app_id):
        raise _problem(
            status.HTTP_409_CONFLICT, "Nothing in flight", f"No install of {app_id!r} is running."
        )
    snapshot = await manager.installer.cancel(app_id)
    assert snapshot is not None
    return snapshot


@router.post("/v1/apps/{app_id}/start", response_model=App, response_model_exclude_none=True)
async def start(request: Request, app_id: str) -> App:
    manager = _apps(request)
    record = _record_or_404(manager, app_id)
    record.enabled = True
    manager.store.put(record)
    if not manager.supervisor.is_running(app_id):
        await manager.start(record)
    return manager.view(record)


@router.post("/v1/apps/{app_id}/stop", response_model=App, response_model_exclude_none=True)
async def stop(request: Request, app_id: str) -> App:
    manager = _apps(request)
    record = _record_or_404(manager, app_id)
    record.enabled = False
    manager.store.put(record)
    await manager.stop(app_id)
    return manager.view(record)


@router.post("/v1/apps/{app_id}/restart", response_model=App, response_model_exclude_none=True)
async def restart(request: Request, app_id: str) -> App:
    manager = _apps(request)
    record = _record_or_404(manager, app_id)
    if not record.enabled:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "App is stopped",
            f"{record.manifest.name} is stopped. Start it instead.",
        )
    await manager.restart(record)
    return manager.view(record)


# --------------------------------------------------------------------------- #
# an app's own settings, through this agent
# --------------------------------------------------------------------------- #


async def _call_app(request: Request, app_id: str, method: str, path: str, body: Any = None) -> Any:
    """Call an app's config trio on loopback with its admin token.

    The operator's own bearer never reaches the app: the app has no way
    to verify a hub credential and no business holding one. What it gets
    is the token this agent generated for it at this spawn.
    """
    manager = _apps(request)
    record = _record_or_404(manager, app_id)
    if not record.manifest.configTrio:
        raise _problem(
            status.HTTP_409_CONFLICT,
            "No settings",
            f"{record.manifest.name} does not publish settings this console can edit.",
        )
    token = manager.supervisor.admin_token(app_id)
    if token is None:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "App is not running",
            f"{record.manifest.name} is not running, so its settings cannot be read. Start it.",
        )
    client = shared_internal_client("apps-config", timeout=_APP_CALL_TIMEOUT)
    try:
        response = await client.request(
            method,
            f"http://127.0.0.1:{record.port}{path}",
            headers={"Authorization": f"Bearer {token}"},
            json=body,
        )
    except httpx.HTTPError as exc:
        raise _problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "App is not answering",
            f"{record.manifest.name} did not answer on port {record.port}: "
            f"{str(exc) or type(exc).__name__}",
        ) from exc
    if response.status_code >= 400:
        try:
            detail: Any = response.json()
        except ValueError:
            detail = response.text or f"{record.manifest.name} answered {response.status_code}"
        code = response.status_code if response.status_code < 500 else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=detail)
    return response.json()


def _as[M: BaseModel](model: type[M], request: Request, app_id: str, value: Any) -> M:
    """Validate an app's answer, or say which app broke its contract.

    An app is somebody's code, and a body that is not a config document
    is that app's fault -- a 502 naming it, not a 500 from this agent.
    """
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        record = _apps(request).store.get(app_id)
        name = record.manifest.name if record else app_id
        raise _problem(
            status.HTTP_502_BAD_GATEWAY,
            "App answered something else",
            f"{name} answered its settings endpoint with a body that is not a "
            f"{model.__name__}: {exc.error_count()} problem(s), first: {exc.errors()[0]['msg']}",
        ) from exc


@router.get("/v1/apps/{app_id}/config", response_model=ConfigDocument)
async def get_app_config(request: Request, app_id: str) -> ConfigDocument:
    return _as(
        ConfigDocument, request, app_id, await _call_app(request, app_id, "GET", "/v1/config")
    )


@router.get("/v1/apps/{app_id}/config/schema", response_model=ConfigSchema)
async def get_app_config_schema(request: Request, app_id: str) -> ConfigSchema:
    return _as(
        ConfigSchema,
        request,
        app_id,
        await _call_app(request, app_id, "GET", "/v1/config/schema"),
    )


@router.patch("/v1/apps/{app_id}/config", response_model=ConfigUpdateResult)
async def update_app_config(
    request: Request, app_id: str, body: ConfigUpdateRequest
) -> ConfigUpdateResult:
    # Nulls kept: `null` is how the config trio says "back to the
    # default", and dropping it would turn a reset into a no-op.
    answer = await _call_app(request, app_id, "PATCH", "/v1/config", body.model_dump(mode="json"))
    return _as(ConfigUpdateResult, request, app_id, answer)
