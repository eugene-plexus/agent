"""`GET /v1/directories`: the picker behind every path field on this host."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from .. import app_accounts, folder_access, tokens
from .._generated.common_models import DirectoryListing, Problem
from .._generated.models import (
    FolderAccess,
    FolderAccessDecision,
    FolderAccessList,
    FolderAccessReleaseRequest,
    FolderAccessRequest,
)
from ..apps import AppManager
from ..dependencies import require_operator_session, verify_bearer
from ..directory_listing import ListingError, list_directory
from .apps import _apps, _record_or_404
from .log_ingress import _bearer, _standing

router = APIRouter(tags=["config"])


def _process(manager: AppManager, app: str) -> int | None:
    supervisor = manager.supervisor
    if isinstance(supervisor, app_accounts.OwnAccountSupervisor):
        return supervisor.runner.state(app).pid
    return supervisor.status(app)[3]


def _access(request: Request, app: str) -> folder_access.Access:
    manager = _apps(request)
    _record_or_404(manager, app)
    if not manager.accounts.available:
        raise _failure(folder_access.Refused("unavailable"))
    cache = getattr(request.app.state, "folder_access", None)
    if cache is None:
        cache = {}
        request.app.state.folder_access = cache
    if app not in cache:
        from ..engines.acquisition import engine_root

        prefix = request.app.state.settings.config_file.resolve().parent
        roots = tuple(str(p) for p in (prefix, manager.store.root, engine_root().resolve()))
        backend = (
            folder_access.WindowsPermissions(
                app_accounts.account_name("windows_service", app), eugene=roots
            )
            if manager.accounts.kind == "windows_service"
            else folder_access.LinuxPermissions(app, manager.store.root, eugene=roots)
        )
        cache[app] = folder_access.Access(
            prefix / folder_access.FILE, backend, process=lambda: _process(manager, app)
        )
    access: folder_access.Access = cache[app]
    return access


def _failure(exc: Exception) -> HTTPException:
    code = 409 if isinstance(exc, folder_access.Refused) and exc.cause == "unavailable" else 400
    body = dict(
        type="about:blank",
        title="Folder access refused",
        status=code,
        detail=str(exc),
        component="agent",
    )
    if isinstance(exc, folder_access.Refused):
        body["refusal"] = exc.cause
    return HTTPException(code, detail=body)


@router.get(
    "/v1/apps/{app}/folder-access",
    response_model=FolderAccessList,
    dependencies=[Depends(require_operator_session)],
)
def list_folder_access(request: Request, app: str) -> FolderAccessList:
    try:
        return FolderAccessList(grants=_access(request, app).listing(app))
    except (OSError, ValueError) as exc:
        raise _failure(exc) from exc


@router.post("/v1/apps/{app}/folder-access", response_model=FolderAccess, status_code=202)
def add_folder_access(
    request: Request,
    app: str,
    body: FolderAccessRequest,
    claims: Annotated[tokens.Claims, Depends(require_operator_session)],
    background: BackgroundTasks,
) -> FolderAccess:
    try:
        return _access(request, app).add(
            app, body.path, str(body.mode or "read"), claims.sub, schedule=background.add_task
        )
    except (OSError, ValueError) as exc:
        raise _failure(exc) from exc


@router.delete(
    "/v1/apps/{app}/folder-access/{ident}",
    response_model=FolderAccess,
    dependencies=[Depends(require_operator_session)],
)
def remove_folder_access(request: Request, app: str, ident: str) -> FolderAccess:
    try:
        return _access(request, app).remove(app, ident)
    except KeyError as exc:
        raise HTTPException(404, detail="No folder access with that id for this app.") from exc
    except (OSError, ValueError) as exc:
        raise _failure(exc) from exc


@router.post(
    "/v1/apps/{app}/folder-access/{ident}/unknown/{entry}",
    response_model=FolderAccess,
    dependencies=[Depends(require_operator_session)],
)
def decide_folder_access_entry(
    request: Request, app: str, ident: str, entry: str, body: FolderAccessDecision
) -> FolderAccess:
    """R23-UNKNOWN: the owner removes or keeps one entry Eugene did not add."""
    try:
        return _access(request, app).decide(app, ident, entry, str(body.action))
    except KeyError as exc:
        raise HTTPException(404, detail="No such folder access entry for this app.") from exc
    except folder_access.NotWaiting as exc:
        raise HTTPException(
            409,
            detail=dict(
                type="about:blank",
                title="Not waiting for a decision",
                status=409,
                detail=str(exc),
                component="agent",
            ),
        ) from exc
    except (OSError, ValueError) as exc:
        raise _failure(exc) from exc


@router.post("/v1/apps/{app}/folder-access/release", response_model=FolderAccessList)
async def release_folder_access(
    request: Request, app: str, body: FolderAccessReleaseRequest
) -> FolderAccessList:
    claims = verify_bearer(request, _bearer(request), classes=(tokens.TYP_CLIENT,))
    # Do not cache a release authorization: revocation at the authority takes effect now.
    cache = getattr(request.app.state, "log_ingress_keys", {})
    cache.pop(claims.jti, None)
    standing = await _standing(request, claims.jti)
    manager = _apps(request)
    _record_or_404(manager, app)
    node = manager.node_name() or "this-node"
    key = manager.store.key(app)
    if standing.name != f"app:{app}@{node}" or key is None or key.key_id != claims.jti:
        raise HTTPException(
            403, detail="Only this app's own active key on this node may release its folder access."
        )
    try:
        access = _access(request, app)
        if bool(body.id) == bool(body.path):
            raise HTTPException(400, detail="Supply exactly one access id or lookup path.")
        records = await asyncio.to_thread(
            access.release if body.id else access.lookup, app, body.id or body.path or ""
        )
        return FolderAccessList(grants=records)
    except KeyError as exc:
        raise HTTPException(404, detail="No recorded folder access matched for this app.") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(
            500, detail={"detail": str(exc), "title": "Folder release failed"}
        ) from exc


@router.get(
    "/v1/directories",
    response_model=DirectoryListing,
    response_model_exclude_none=True,
    dependencies=[Depends(require_operator_session)],
)
def list_directories(
    path: str | None = Query(default=None),
    include_files: bool = Query(default=False, alias="includeFiles"),
    show_hidden: bool = Query(default=False, alias="showHidden"),
) -> DirectoryListing:
    """List one directory on this host, for the config editor's picker.

    Operator-only, and unrestricted: an operator can already type any
    path into `pathMappings` or `vllmBinary`, and the agent already
    stats what they name. A plain `def` on purpose -- these are real
    filesystem calls, and a dead network share blocks for as long as the
    OS takes to give up, which is acceptable for a request someone
    pressed a button to make and not acceptable on the event loop.
    """
    try:
        return list_directory(path, include_files=include_files, show_hidden=show_hidden)
    except ListingError as exc:
        raise HTTPException(
            status_code=exc.status,
            detail=Problem(
                type="https://github.com/eugene-plexus/agent#directory-listing",
                title=exc.title,
                status=exc.status,
                detail=exc.detail,
                component="agent",
            ).model_dump(exclude_none=True),
        ) from exc
