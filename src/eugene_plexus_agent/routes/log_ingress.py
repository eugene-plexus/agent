"""`POST /v1/logs`: the log ingress (C1, `workbench.md` §3).

The one route on the agent that accepts a client key, and for writing
only: `GET /v1/logs` beside it stays operator-only. The key must carry
`writeLogs`, which the registry gives every app's key and the operator
can give any other.

Whether a key may send is the key authority's answer, asked with the
same `check` admission the gateway uses, so a revoked key stops at once
on a standalone agent and within `_CHECK_TTL_SECONDS` on an enrolled one.
That cache is the reason a launcher batching once a second does not
become a request a second to the control root.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request, Response

from .. import log_ingress, tokens
from ..client_admission import AdmissionRefusal
from ..client_key_registry import registry
from ..dependencies import verify_bearer
from .auth import _keys

log = logging.getLogger(__name__)

router = APIRouter(tags=["logs"])

_CHECK_TTL_SECONDS = 15.0
_JSON = "application/json"
_PROTOBUF = "application/x-protobuf"


@dataclass(frozen=True)
class _Standing:
    name: str
    may_write: bool
    until: float


def _problem(code: int, title: str, detail: str, headers: dict | None = None) -> HTTPException:
    return HTTPException(code, detail={"title": title, "detail": detail}, headers=headers)


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _problem(401, "Missing token", "Send a client key as Authorization: Bearer <key>.")
    return token.strip()


async def _standing(request: Request, key_id: str) -> _Standing:
    cache: dict[str, _Standing] | None = getattr(request.app.state, "log_ingress_keys", None)
    if cache is None:
        cache = {}
        request.app.state.log_ingress_keys = cache
    now = time.perf_counter()
    cached = cache.get(key_id)
    if cached is not None and cached.until > now:
        return cached
    body = {"action": "check", "keyId": key_id, "requestId": secrets.token_hex(8)}
    owner = registry(request)
    try:
        if owner.enrolled:
            result = await owner.forward("POST", "/v1/auth/client-keys/admission", body=body)
        else:
            result = _keys(request).admit(
                key_id=key_id, action="check", request_id=body["requestId"], model=None
            )
    except AdmissionRefusal as exc:
        raise _problem(403, "Key refused", exc.detail) from exc
    except HTTPException as exc:
        if exc.status_code in (401, 403, 404):
            raise _problem(
                403, "Key refused", "This client key is unregistered, expired or revoked."
            ) from exc
        raise
    except (OSError, ValueError) as exc:
        raise owner.local_unavailable() from exc
    limits = (result or {}).get("limits") or {}
    standing = _Standing(
        name=str((result or {}).get("keyName") or ""),
        may_write=limits.get("writeLogs") is True,
        until=now + _CHECK_TTL_SECONDS,
    )
    cache[key_id] = standing
    return standing


def _rate(request: Request) -> log_ingress.RecordRate:
    rate = getattr(request.app.state, "log_ingress_rate", None)
    if rate is None:
        rate = log_ingress.RecordRate()
        request.app.state.log_ingress_rate = rate
    return rate


@router.post("/v1/logs", operation_id="sendLogs")
async def send_logs(request: Request) -> Response:
    claims = verify_bearer(request, _bearer(request), classes=(tokens.TYP_CLIENT,))
    content_type = (request.headers.get("content-type") or _JSON).split(";")[0].strip().lower()
    if content_type not in (_JSON, _PROTOBUF):
        raise _problem(
            415,
            "Unsupported media type",
            f"Send OTLP as {_JSON} or {_PROTOBUF}, not {content_type}.",
        )
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > log_ingress.MAX_BODY_BYTES:
        raise _problem(
            413, "Too large", "A log request may be at most 1 MiB; send smaller batches."
        )
    body = await request.body()
    if len(body) > log_ingress.MAX_BODY_BYTES:
        raise _problem(
            413, "Too large", "A log request may be at most 1 MiB; send smaller batches."
        )

    standing = await _standing(request, claims.jti)
    if not standing.may_write:
        raise _problem(
            403,
            "This key may not send logs",
            f"The client key {standing.name or claims.sub!r} does not have writeLogs. "
            "Turn it on for the key on Home's key list, or use a key that has it.",
        )
    try:
        parsed = (
            log_ingress.parse_protobuf(body)
            if content_type == _PROTOBUF
            else log_ingress.parse_json(body)
        )
    except log_ingress.IngressError as exc:
        raise _problem(exc.status, "Not an OTLP logs request", exc.detail) from exc

    if parsed.records:
        wait = _rate(request).take(claims.jti, len(parsed.records))
        if wait is not None:
            raise _problem(
                429,
                "Too many log records",
                f"A key may send {log_ingress.RECORDS_PER_MINUTE} records a minute; "
                f"try again in {wait} s.",
                headers={"Retry-After": str(wait)},
            )
    # The source is the key the authority names, never anything the
    # records say, so one sender cannot write as another.
    source = log_ingress.source_for(standing.name or claims.sub)
    for record in parsed.records:
        for line in log_ingress.lines_for(source, record):
            # Through the tee, exactly as the supervisor writes a child's
            # line: stamped at receipt, published to followers, masked on
            # the way out.
            print(line, flush=True)

    if content_type == _PROTOBUF:
        return Response(log_ingress.response_protobuf(parsed), media_type=_PROTOBUF)
    return Response(json.dumps(log_ingress.response_json(parsed)), media_type=_JSON)
