"""`GET /metrics`: this machine's engines for a Prometheus scraper (row 12).

Admits an operator session, or a client key that has `readMetrics`. Whether a
key may is the key authority's answer, asked with a `check` admission of
purpose `metrics` (no model, no lease), as the log ingress asks for
`writeLogs`: a revoked key stops at once on a standalone agent and within
`_CHECK_TTL_SECONDS` on an enrolled one. The grant opens nothing else, and a
key that runs inference does not read metrics by that alone.

The engines are read when the scrape arrives (`engine_metrics`), so the series
are what the engines say now, and a machine with no running engine answers an
empty page, which is a success.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request, Response

from .. import engine_metrics, tokens
from .._generated.models import RuntimeStatus
from ..client_admission import AdmissionRefusal
from ..client_key_registry import registry
from ..dependencies import verify_bearer
from ..engines.base import default_model_alias, probe_client
from .auth import _keys

log = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])

_CHECK_TTL_SECONDS = 15.0
_TEXT = "text/plain; version=0.0.4; charset=utf-8"

NO_GRANT = (
    "This key may not read metrics. Turn on May read metrics for it under Keys, "
    "or use a key that has it."
)


@dataclass(frozen=True)
class _Standing:
    name: str
    may_read: bool
    until: float


def _problem(code: int, title: str, detail: str) -> HTTPException:
    return HTTPException(code, detail={"title": title, "detail": detail})


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _problem(
            401,
            "Missing token",
            "Send an operator session or a client key that may read metrics as "
            "Authorization: Bearer <key>.",
        )
    return token.strip()


async def _standing(request: Request, key_id: str) -> _Standing:
    cache: dict[str, _Standing] | None = getattr(request.app.state, "metrics_keys", None)
    if cache is None:
        cache = {}
        request.app.state.metrics_keys = cache
    now = time.perf_counter()
    cached = cache.get(key_id)
    if cached is not None and cached.until > now:
        return cached
    body = {
        "action": "check",
        "keyId": key_id,
        "requestId": secrets.token_hex(8),
        "purpose": "metrics",
    }
    owner = registry(request)
    refused: AdmissionRefusal | HTTPException | None = None
    result = None
    try:
        if owner.enrolled:
            result = await owner.forward("POST", "/v1/auth/client-keys/admission", body=body)
        else:
            result = _keys(request).admit(
                key_id=key_id,
                action="check",
                request_id=body["requestId"],
                model=None,
                purpose="metrics",
            )
    except AdmissionRefusal as exc:
        refused = exc
    except HTTPException as exc:
        if exc.status_code not in (401, 403, 404):
            raise
        refused = exc
    except (OSError, ValueError) as exc:
        raise owner.local_unavailable() from exc
    if refused is not None:
        # The authority's own answer: a key without the grant is 403, one it does
        # not know or has revoked is 401.
        code = getattr(refused, "status", None) or getattr(refused, "status_code", 403)
        if code == 403:
            raise _problem(403, "This key may not read metrics", NO_GRANT) from refused
        raise _problem(
            401, "Key refused", "This client key is unregistered, expired or revoked."
        ) from refused
    limits = (result or {}).get("limits") or {}
    standing = _Standing(
        name=str((result or {}).get("keyName") or ""),
        may_read=limits.get("readMetrics") is True,
        until=now + _CHECK_TTL_SECONDS,
    )
    cache[key_id] = standing
    return standing


async def require_metrics_reader(request: Request) -> str:
    """The operator session or the key that may read metrics; its name."""
    claims = verify_bearer(
        request, _bearer(request), classes=(tokens.TYP_SESSION, tokens.TYP_CLIENT)
    )
    if claims.is_session:
        return "operator"
    standing = await _standing(request, claims.jti)
    if not standing.may_read:
        raise _problem(403, "This key may not read metrics", NO_GRANT)
    return standing.name or claims.sub


def _targets(request: Request) -> list[engine_metrics.Target]:
    """Every running engine that has a metrics mapping, as it is declared now."""
    state = getattr(request.app.state, "agent_state", None)
    supervisor = getattr(request.app.state, "runtime_supervisor", None)
    if state is None or supervisor is None:
        return []
    out: list[engine_metrics.Target] = []
    for spec in state.list_runtime_specs():
        if not engine_metrics.scrapes(spec.engine.value):
            continue
        runtime = supervisor.compose(spec)
        if runtime.status != RuntimeStatus.ready or spec.port is None:
            continue
        out.append(
            engine_metrics.Target(
                node=runtime.node or "local",
                runtime=spec.name,
                model=spec.modelAlias or default_model_alias(spec.modelPath),
                engine=spec.engine.value,
                url=f"http://{spec.host or '127.0.0.1'}:{spec.port}",
            )
        )
    return out


def engine_client():  # type: ignore[no-untyped-def]
    """The one client engine reads use: the readiness probes' own (an engine is
    always on this machine, so no proxy is ever in the way)."""
    return probe_client()


@router.get("/metrics", operation_id="getPrometheusMetrics")
async def get_prometheus_metrics(request: Request) -> Response:
    await require_metrics_reader(request)
    reads = await engine_metrics.gather(engine_client(), _targets(request))
    return Response(engine_metrics.render(reads), media_type=_TEXT)
