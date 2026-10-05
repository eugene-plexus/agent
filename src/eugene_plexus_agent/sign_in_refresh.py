"""Move each app's sign-in return address to where it is opened now.

Found on the first live move onto one HTTPS port (2026-10-04): Workbench
answered at its new address, and Eugene refused to send anyone back there
("Wrong return address"), because its sign-in client still named the old
one. The fix was an operator Restart from a console the move had just made
harder to reach. Troy, 2026-10-05: *Eugene updates Workbench's return
address itself at startup.*

So at boot this agent compares each app's return addresses with the ones
it registered (the `.redirects.json` stamp beside its secret), and replaces
any that changed at the control root with its own `agent` token
(`PUT /v1/oidc/clients/{id}/redirect-uris`). The root takes that only for
clients named for this machine's own apps, and changes nothing else: the
same client, the same secret, so the app's sign-ins carry on.

The root may be starting, sealed or across the network, so this retries
with a growing pause until every app is current. An app whose client the
root no longer has, or will not let this machine change, is left for an
operator Start or Restart, which registers it again; the Apps page says so.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import HTTPException

from ._private_files import write_private

log = logging.getLogger(__name__)

_FIRST_PAUSE = 5.0
_LONGEST_PAUSE = 120.0


def stale(manager: Any, record: Any) -> bool:
    """An app that signs in, has a client and a secret, and moved since."""
    manifest = record.manifest
    store = manager.store
    if not manifest.signIn or not store.oidc_client(manifest.id):
        return False
    secret = store.oidc_secret_file(manifest.id)
    try:
        if not secret.stat().st_size:
            return False
    except OSError:
        return False
    return not manager.sign_in_registration_current(manifest)


async def move(manager: Any, registry: Any, record: Any) -> list[str]:
    """Replace one app's return addresses at the root. Raises what the root said."""
    manifest = record.manifest
    redirects: list[str] = manager.sign_in_redirects(
        manifest.id, manifest.signInCallbackPath or "/oidc/callback"
    )
    client_id = manager.store.oidc_client(manifest.id)
    await registry.forward(
        "PUT",
        f"/v1/oidc/clients/{client_id}/redirect-uris",
        body={"redirectUris": redirects},
    )
    write_private(manager.store.sign_in_stamp_file(manifest.id), json.dumps(redirects))
    return redirects


async def run(app: Any) -> None:
    """Until every app's return address is current, or the root will not have it."""
    manager = getattr(app.state, "apps", None)
    registry = getattr(app.state, "client_key_registry", None)
    if manager is None or registry is None or not registry.enrolled:
        return
    pause = _FIRST_PAUSE
    given_up: set[str] = set()
    said: dict[str, str] = {}
    while True:
        waiting = [
            r for r in manager.store.installed() if r.id not in given_up and stale(manager, r)
        ]
        if not waiting:
            return
        for record in waiting:
            try:
                redirects = await move(manager, registry, record)
            except HTTPException as exc:
                detail = exc.detail
                if isinstance(detail, dict):
                    detail = detail.get("detail") or detail.get("title") or detail
                if exc.status_code in (403, 404):
                    given_up.add(record.id)
                    log.warning(
                        "could not move %s's sign-in address at the control root (%s); "
                        "Restart it in Apps to register it again",
                        record.id,
                        detail,
                    )
                elif said.get(record.id) != str(detail):
                    said[record.id] = str(detail)
                    log.info(
                        "will retry moving %s's sign-in address: the control root said %s",
                        record.id,
                        detail,
                    )
            else:
                log.info("moved %s's sign-in address to %s", record.id, ", ".join(redirects))
        await asyncio.sleep(pause)
        pause = min(pause * 2, _LONGEST_PAUSE)
