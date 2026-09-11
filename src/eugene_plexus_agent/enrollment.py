"""Joining an install, leaving one, and saying where this host is.

The protocol half of `routes/node.py`, extracted at M9 because it now has
**two** callers that are not both HTTP handlers: the route, and
`eugene-plexus-agent join`, which runs before any app exists. A worker in
another building cannot be enrolled from a browser — you cannot reach its
web UI until it binds non-loopback, and it binds non-loopback only once it
advertises a non-loopback address, which is part of what joining does. The
browser arrives after the thing it would configure.

Three operations, and the middle one is the M9 defect:

**enroll** — present a join token and this node's *public* keys, take back
a name, the epoch, the install's signing key and the root's identity.

**announce** — tell the root this node's address, on every start and on
every change. Before M9 the address was sent exactly once, at enrollment,
so a host that rebooted onto a new tailnet IP left the root holding an
address nobody was listening on — and the root could not poll its way out,
because the only address it had was the stale one. Which is also why this
is a push and not the root asking.

**The announcement is signed, not bearer-authenticated**, mirroring the
signed re-key in the other direction. A service token names a *kind* and
not a host, so any agent could re-address any node; and every other write
on the control root is operator-only on purpose, while the case that
matters here is a host that rebooted at 3am with nobody watching.

Nothing in this module touches `AuthState` or the supervisor. Adopting the
key and restarting children are consequences the caller owns, because the
CLI has no children to restart and no sessions to invalidate.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import platform
import socket
import sys
from dataclasses import dataclass
from typing import Any

import httpx

from . import __version__
from .node_identity import (
    NodeIdentityStore,
    address_message,
    derive_advertise_host,
    effective_advertise_url,
    format_url,
    sign_address,
)

log = logging.getLogger(__name__)

# The control root answers enrollment from memory plus one log append;
# anything slower is the root being down, which is a failure here either
# way.
ENROLL_TIMEOUT_SECONDS = 15.0

# An announcement is one signature check and at most one log append.
# Deliberately shorter than enrollment's: this runs on every boot, and a
# root that is slow to answer must not hold up supervision.
ANNOUNCE_TIMEOUT_SECONDS = 8.0


class EnrollmentError(Exception):
    """Something in the exchange failed. `slug` and `title` exist so an
    HTTP caller can render a Problem without re-deriving them, and the CLI
    can print the same words."""

    def __init__(self, slug: str, title: str, detail: str) -> None:
        super().__init__(detail)
        self.slug = slug
        self.title = title
        self.detail = detail


@dataclass(frozen=True)
class EnrollmentOutcome:
    name: str
    epoch: int
    signing_key: bytes
    """Decoded, so the caller does not repeat the validation."""
    signing_key_id: str | None
    advertise_url: str | None


@dataclass(frozen=True)
class AnnounceOutcome:
    """What an announcement did. `changed` is the root's answer, not ours
    — it tells apart "we moved" from "we restarted", which is the whole
    reason a restart appends nothing to the log."""

    announced: bool
    changed: bool
    url: str | None
    detail: str | None = None


def host_os() -> str | None:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return None


def host_arch() -> str | None:
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return None


def decode_signing_key(value: str) -> bytes | None:
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) == 32 else None


async def resolve_advertise_url(
    *, configured: Any, control_url: str | None, bind_port: int, persisted: str | None = None
) -> str | None:
    """Where other hosts reach this agent, in the order that survives a
    reboot onto a different address.

    1. What the operator configured. Always wins, including when it is
       wrong — an expert override that cannot be wrong is not one.
    2. Otherwise **re-derived** from the route to the control root. This
       is the step that fixes the defect: a stale persisted value is
       exactly what a rebooted host has, so trusting it first would make
       the announcement announce the old address.
    3. Otherwise whatever was persisted, which is better than nothing on
       a boot where the root is unreachable.
    """
    configured_url = effective_advertise_url(configured, None)
    if configured_url is not None:
        return configured_url
    if control_url:
        host = await asyncio.to_thread(derive_advertise_host, control_url)
        if host is not None:
            return format_url(host, int(bind_port))
    return effective_advertise_url(None, persisted)


async def perform_enrollment(
    *,
    store: NodeIdentityStore,
    control_url: str,
    token: str,
    name: str | None,
    advertise_url: str | None,
    devices: list[dict[str, Any]] | None = None,
    transport: Any = None,
) -> EnrollmentOutcome:
    """Exchange a join token for membership, and persist the result.

    Raises `EnrollmentError` and records nothing if any step fails: a node
    that half-enrolled would hold an install's key without the install
    knowing it exists.
    """
    if store.record.enrolled:
        held = store.record
        raise EnrollmentError(
            "already-enrolled",
            "Already enrolled",
            f"This agent is enrolled as {held.name!r} with the control root at "
            f"{held.control_url}. Leaving is a deliberate act: POST /v1/node/unenroll here "
            f"(or revoke it there first, which rotates the install's signing key).",
        )

    record = store.ensure_keypair()
    control_url = control_url.rstrip("/")
    node_name = (name or "").strip() or socket.gethostname()

    payload: dict[str, Any] = {
        "token": token,
        "name": node_name,
        "publicKey": record.public_key,
        "signingPublicKey": record.signing_public_key,
        "agentVersion": __version__,
        "os": host_os(),
        "arch": host_arch(),
        "devices": devices or [],
    }
    if advertise_url is not None:
        payload["url"] = advertise_url

    try:
        async with httpx.AsyncClient(timeout=ENROLL_TIMEOUT_SECONDS, transport=transport) as client:
            response = await client.post(f"{control_url}/v1/nodes/enroll", json=payload)
    except httpx.HTTPError as exc:
        raise EnrollmentError(
            "control-root-unreachable",
            "Control root unreachable",
            f"Could not reach the control root at {control_url}: {exc}. Nothing was recorded; "
            f"this agent is still unenrolled.",
        ) from exc

    if response.status_code != 201:
        raise EnrollmentError(
            "control-root-refused",
            "Control root refused the enrollment",
            f"The control root at {control_url} answered {response.status_code}: "
            f"{problem_detail(response)}. Nothing was recorded; this agent is still unenrolled.",
        )

    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise _malformed(control_url, "the body was not a JSON object")

    granted_name = body.get("name")
    epoch = body.get("epoch")
    signing_key_b64 = body.get("signingKey")
    if not isinstance(granted_name, str) or not granted_name:
        raise _malformed(control_url, "`name` is missing")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise _malformed(control_url, f"`epoch` is {epoch!r}")
    signing_key = decode_signing_key(signing_key_b64) if isinstance(signing_key_b64, str) else None
    if signing_key is None:
        raise _malformed(control_url, "`signingKey` is not 32 base64 bytes")

    store.record_enrollment(
        name=granted_name,
        control_url=control_url,
        epoch=epoch,
        signing_key=str(signing_key_b64),
        signing_key_id=_str_or_none(body.get("signingKeyId")),
        control_public_key=_str_or_none(body.get("controlPublicKey")),
        recovery_public_key=_str_or_none(body.get("recoveryPublicKey")),
        advertise_url=advertise_url,
    )
    return EnrollmentOutcome(
        name=granted_name,
        epoch=epoch,
        signing_key=signing_key,
        signing_key_id=_str_or_none(body.get("signingKeyId")),
        advertise_url=advertise_url,
    )


async def announce_address(
    *, store: NodeIdentityStore, url: str, transport: Any = None
) -> AnnounceOutcome:
    """Tell the control root this node's address. Never raises.

    Failure is reported, not thrown, because both callers are places where
    throwing is wrong: at startup it would take supervision down over a
    management-plane problem, and on a config change it would fail an edit
    that has already been persisted. `degraded-mode-required`, applied to
    the one operation that talks to another host on every boot.
    """
    record = store.record
    if not record.enrolled or not record.control_url or not record.name:
        return AnnounceOutcome(False, False, url, "not enrolled")
    if not record.signing_private_key:
        return AnnounceOutcome(False, False, url, "this node has no signing key")

    sequence = store.next_advertise_sequence()
    message = address_message(name=record.name, sequence=sequence, url=url)
    body = {
        "url": url,
        "sequence": sequence,
        "signature": sign_address(signing_private_key=record.signing_private_key, message=message),
    }
    target = f"{record.control_url.rstrip('/')}/v1/nodes/{record.name}"
    try:
        async with httpx.AsyncClient(
            timeout=ANNOUNCE_TIMEOUT_SECONDS, transport=transport
        ) as client:
            response = await client.patch(target, json=body)
    except httpx.HTTPError as exc:
        log.warning("could not announce this node's address to %s: %s", record.control_url, exc)
        return AnnounceOutcome(False, False, url, str(exc))

    if response.status_code != 200:
        detail = problem_detail(response)
        # 401 here has one likely cause and it is worth naming, because
        # the symptom (a node nothing can reach) is far away from it.
        if response.status_code == 401:
            log.warning(
                "the control root will not accept this node's address announcement: %s. "
                "A node enrolled before nodes carried a signing identity has to re-enroll "
                "before it can re-advertise.",
                detail,
            )
        else:
            log.warning(
                "the control root refused this node's address announcement (%d): %s",
                response.status_code,
                detail,
            )
        return AnnounceOutcome(False, False, url, detail)

    try:
        payload = response.json()
    except ValueError:
        payload = {}
    changed = bool(payload.get("changed")) if isinstance(payload, dict) else False
    if changed:
        log.info("told the control root this node is now reachable at %s", url)
    return AnnounceOutcome(True, changed, url)


def problem_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict):
            return str(detail.get("detail") or detail.get("title") or detail)
        if detail:
            return str(detail)
    return response.text[:300]


def _malformed(control_url: str, what: str) -> EnrollmentError:
    return EnrollmentError(
        "control-root-malformed",
        "Control root answered with a malformed enrollment",
        f"The control root at {control_url} answered 201 but {what}. Nothing was recorded.",
    )


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "ANNOUNCE_TIMEOUT_SECONDS",
    "ENROLL_TIMEOUT_SECONDS",
    "AnnounceOutcome",
    "EnrollmentError",
    "EnrollmentOutcome",
    "announce_address",
    "decode_signing_key",
    "host_arch",
    "host_os",
    "perform_enrollment",
    "problem_detail",
    "resolve_advertise_url",
]
