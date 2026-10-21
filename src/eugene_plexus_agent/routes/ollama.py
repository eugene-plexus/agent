"""An operator's Keep action sends a locally discovered blob to its assigned job."""

import asyncio
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from .._generated.models import OllamaSendRequest, OllamaSendResult
from ..dependencies import require_operator_session
from ..ollama_store import local_models
from ..preparation_send import CHUNK_BYTES, REQUEST_TIMEOUT, _read, detail_of
from ..runtime_context import NodeContext
from .runtimes import library_client_for

router = APIRouter(tags=["runtimes"])


async def send(library, node: str, body: OllamaSendRequest) -> OllamaSendResult:
    base = "/v1/keep-jobs/" + quote(body.job, safe="")
    job = await library.operation_request("GET", base)
    if job.get("node") != node:
        raise HTTPException(403, "This Keep job belongs to another machine.")
    if job.get("state") != "running" or job.get("digest") != body.digest:
        raise HTTPException(409, "The Keep job is not running for the requested digest.")
    found = await asyncio.to_thread(
        lambda: next(
            (
                m
                for _, m in local_models()
                if "sha256:" + m.blob.name.removeprefix("sha256-") == body.digest
            ),
            None,
        )
    )
    if found is None:
        raise HTTPException(404, "This machine has no Ollama model with that digest.")
    sent = 0
    for role, path in [
        ("weights", found.blob),
        *([("projector", found.projector)] if found.projector else []),
    ]:
        held = await library.operation_request("POST", base + "/files/state", json={"role": role})
        expected = path.name.removeprefix("sha256-")
        size = (await asyncio.to_thread(path.stat)).st_size
        if held["sha256"] == expected and held["sizeBytes"] == size:
            continue
        offset = held["receivedBytes"]
        if offset > size:
            offset = 0
        while offset < size:
            chunk = await asyncio.to_thread(_read, path, offset, CHUNK_BYTES)
            if not chunk:
                raise HTTPException(409, "Ollama's file changed while it was being sent.")
            held = await library.operation_request(
                "PUT",
                base + "/files",
                params={"role": role, "offset": offset},
                content=chunk,
                headers={"Content-Type": "application/octet-stream"},
                timeout=REQUEST_TIMEOUT,
            )
            next_offset = held["receivedBytes"]
            if next_offset != offset + len(chunk):
                raise HTTPException(502, "The Library did not accept the expected chunk length.")
            offset = next_offset
            sent += len(chunk)
        await library.operation_request("POST", base + "/files/complete", json={"role": role})
    return OllamaSendResult(job=body.job, bytesSent=sent)


@router.post(
    "/v1/ollama/send",
    response_model=OllamaSendResult,
    dependencies=[Depends(require_operator_session)],
)
async def send_model(body: OllamaSendRequest, request: Request):
    library = await library_client_for(NodeContext(request.app))
    if library is None:
        raise HTTPException(502, "The Library cannot be reached from this machine.")
    node = request.app.state.auth_state.trust.recipient.removeprefix("node:")
    try:
        return await send(library, node, body)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, "The Library refused the file: " + detail_of(exc)) from exc
    except (OSError, httpx.HTTPError) as exc:
        raise HTTPException(409, f"The model could not be sent: {exc}") from exc
