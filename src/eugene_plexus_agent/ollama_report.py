"""Report a worker's own stores at startup and when their listing changes."""

import asyncio
import json
import logging
from urllib.parse import quote

from fastapi import FastAPI

from ._generated.models import ComponentKind
from .ollama_store import report_body
from .routes.runtimes import library_client_for
from .runtime_context import NodeContext

log = logging.getLogger(__name__)


async def run(app: FastAPI) -> None:
    previous = None
    while True:
        try:
            state = app.state.agent_state
            identity = app.state.node_identity.record
            local_library = any(
                e.kind == ComponentKind.library for e in state.list_topology_entries()
            )
            if state.get_config("reportOllama") and not local_library and identity.enrolled:
                library = await library_client_for(NodeContext(app))
                if library is not None:
                    body = await asyncio.to_thread(report_body)
                    listing = json.dumps(body["stores"], sort_keys=True)
                    if listing != previous:
                        await library.operation_request(
                            "PUT",
                            "/v1/nodes/" + quote(identity.name, safe="") + "/ollama",
                            json=body,
                        )
                        previous = listing
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Could not report this machine's Ollama models: %s", exc)
        await asyncio.sleep(60)
