"""`/v1/exec-approvals`: an owner approves a program for Eugene's own
account (row 8, part 8c; see `exec_approvals`).

An owner's session only: a component's, the gateway's or another
machine's token can neither approve, nor list, nor take back.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from .. import exec_approvals, tokens
from .._generated.common_models import Problem
from .._generated.models import ExecApproval, ExecApprovalList, ExecApprovalRequest
from ..dependencies import require_operator_session

router = APIRouter(tags=["engines"], dependencies=[Depends(require_operator_session)])


def _problem(code: int, slug: str, title: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/agent#{slug}",
            title=title,
            status=code,
            detail=detail,
            component="agent",
        ).model_dump(exclude_none=True),
    )


@router.get("/v1/exec-approvals", response_model=ExecApprovalList)
async def list_exec_approvals() -> ExecApprovalList:
    return ExecApprovalList(approvals=await asyncio.to_thread(exec_approvals.listed))


@router.post(
    "/v1/exec-approvals",
    response_model=ExecApproval,
    status_code=status.HTTP_201_CREATED,
)
async def approve_exec(
    body: ExecApprovalRequest,
    claims: Annotated[tokens.Claims, Depends(require_operator_session)],
) -> ExecApproval:
    try:
        return await asyncio.to_thread(
            exec_approvals.approve, body.path, body.sha256, by=claims.sub
        )
    except exec_approvals.NotApprovable as e:
        raise _problem(409, "exec-not-approvable", "Not approved", str(e)) from e


@router.delete("/v1/exec-approvals/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_exec_approval(id: str) -> Response:  # the contract's name
    try:
        removed = await asyncio.to_thread(exec_approvals.remove, id)
    except exec_approvals.NotApprovable as e:
        raise _problem(409, "exec-not-approvable", "Not taken back", str(e)) from e
    if not removed:
        raise _problem(404, "exec-approval-not-found", "No such approval", id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
