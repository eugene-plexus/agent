"""`GET /v1/directories`: the picker behind every path field on this host."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from .._generated.common_models import DirectoryListing, Problem
from ..dependencies import require_operator_session
from ..directory_listing import ListingError, list_directory

router = APIRouter(tags=["config"])


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
