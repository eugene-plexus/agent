"""POST /v1/people/known: which people the root still has, for this machine's store (row 53).

The store keeps files for people the root verified. When the root removes
a person, the store deletes what they kept; to learn it, it asks here with
its own token, addressed to this machine. Its token cannot reach the root
itself (a node sends only an `agent` or a granted `gateway` token off its
machine), so this agent asks with its own and answers what the root said.

Only this machine's own `store` may ask, only for the ids it sends, and
nothing is said when the root cannot be asked: the store then deletes
nothing.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from .. import tokens
from .._generated.common_models import PeopleKnownRequest, PeopleKnownResult, Problem
from .._http import internal_client
from ..dependencies import require_local_store

router = APIRouter(tags=["auth"])
_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


def _problem(code: int, title: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/agent#{title.replace(' ', '-').lower()}",
            title=title,
            status=code,
            detail=detail,
            component="agent",
        ).model_dump(exclude_none=True),
    )


def _client(request: Request) -> httpx.AsyncClient:
    client: httpx.AsyncClient | None = getattr(request.app.state, "people_client", None)
    if client is None:
        client = internal_client(
            timeout=_TIMEOUT, transport=getattr(request.app.state, "control_transport", None)
        )
        request.app.state.people_client = client
    return client


@router.post(
    "/v1/people/known",
    response_model=PeopleKnownResult,
    dependencies=[Depends(require_local_store)],
)
async def people_known(request: Request, body: PeopleKnownRequest) -> PeopleKnownResult:
    identity = getattr(request.app.state, "node_identity", None)
    record = identity.record if identity is not None else None
    if record is None or not record.enrolled or not record.control_url:
        raise _problem(
            503,
            "No root to ask",
            "This machine has not joined a control root, so nobody can say who was removed.",
        )
    trust = request.app.state.auth_state.trust
    url = str(record.control_url).rstrip("/") + "/v1/people/known"
    try:
        response = await _client(request).post(
            url,
            json=body.model_dump(mode="json"),
            headers={"Authorization": "Bearer " + trust.agent_token(tokens.RECIPIENT_CONTROL)},
        )
    except httpx.HTTPError as exc:
        raise _problem(
            503,
            "Root not answering",
            f"The control root at {record.control_url} did not answer: "
            f"{str(exc) or type(exc).__name__}",
        ) from exc
    if response.status_code != 200:
        said: Any = ""
        try:
            said = response.json()
            if isinstance(said, dict):
                inner = said.get("detail", said)
                said = inner.get("detail") if isinstance(inner, dict) else inner
        except ValueError:
            said = response.text[:300]
        raise _problem(
            503,
            "Root could not say",
            f"The control root answered {response.status_code}: {said}",
        )
    try:
        return PeopleKnownResult.model_validate_json(response.content)
    except ValueError as exc:
        raise _problem(503, "Root could not say", "The control root's answer was not one.") from exc
