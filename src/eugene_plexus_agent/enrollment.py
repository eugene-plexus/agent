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
a name, the epoch, the root's identity and a trust bundle that already
lists this node's token key. **No private key arrives** (2026-09-25):
until then the answer carried the install's signing key in plaintext.

**announce** — tell the root this node's address, on every start and on
every change. Before M9 the address was sent exactly once, at enrollment,
so a host that rebooted onto a new tailnet IP left the root holding an
address nobody was listening on — and the root could not poll its way out,
because the only address it had was the stale one. Which is also why this
is a push and not the root asking.

**The announcement is signed, not bearer-authenticated**, mirroring the
signed trust bundle in the other direction: every other write on the
control root is operator-only on purpose, while the case that matters
here is a host that rebooted at 3am with nobody watching.

Nothing in this module touches `AuthState` or the supervisor. Loading the
bundle and restarting children are consequences the caller owns, because
the CLI has no children to restart and no sessions to invalidate. The
bundle file is written here, verified, so both callers get it.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import random
import re
import socket
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from . import __version__, tokens
from ._http import internal_client, refused_pin
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

# agent#8: a failed announcement is tried again after 5 s, doubling to
# 5 minutes; a settled address is re-derived every minute and announced
# when it moved.
ANNOUNCE_RETRY_FIRST_SECONDS = 5.0
ANNOUNCE_RETRY_MAX_SECONDS = 300.0
ADDRESS_RECHECK_SECONDS = 60.0
# Row 36: an address the root refused for good is announced again this
# often. The root keeps why it refused only in memory, so a restarted root
# or a promoted standby learns it from the machine's next attempt.
REFUSED_RETRY_SECONDS = 900.0


#: The slug for a join token the control root refused (401 or 409).
TOKEN_REFUSED = "join-token-refused"


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
    bundle: tokens.TrustBundle
    """Verified against the root's key, and already kept on disk."""
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
    #: Worth trying again: no answer, or one that says "later" (408, 429,
    #: 5xx). Any other refusal is definite for this address.
    retry: bool = False
    #: The machine port and key the root accepted with it (LT1); empty when
    #: it accepted the address alone.
    sent: dict[str, str] | None = None


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


async def resolve_advertise_url(
    *,
    configured: Any,
    control_url: str | None,
    bind_port: int,
    persisted: str | None = None,
    quiet: bool = False,
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

    `quiet`: a root that cannot be reached is logged at debug, for the
    minute-by-minute recheck (`keep_address_announced`).
    """
    configured_url = effective_advertise_url(configured, None)
    if configured_url is not None:
        return configured_url
    if control_url:
        host = await asyncio.to_thread(derive_advertise_host, control_url, quiet=quiet)
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
    label: str | None = None,
    transport: Any = None,
    machine: dict[str, str] | None = None,
) -> EnrollmentOutcome:
    """Exchange a join token for membership, and persist the result.

    Raises `EnrollmentError` and records nothing if any step fails: a node
    that half-enrolled would hold an install's key without the install
    knowing it exists.

    **A token that carries the root's machine key** (`<secret>.<pin>`, LT1)
    is presented over TLS to the root's machine port, at the address given
    for the root, and only after the key that port presents matches: so
    nothing on the network can answer in the root's place, and the root's
    identity key, pinned from this answer, is the real one. The pin is kept
    for the calls that follow, until the root's signed machine list arrives.
    `machine` is this node's own machine port and key, registered with it.
    """
    if store.record.enrolled:
        held = store.record
        raise EnrollmentError(
            "already-enrolled",
            "Already enrolled",
            f"This agent is enrolled as {held.name!r} with the control root at "
            f"{held.control_url}. Leaving is a deliberate act: POST /v1/node/unenroll here "
            f"(or revoke it there first, which drops its key from that install).",
        )

    record = store.ensure_keypair()
    control_url = control_url.rstrip("/")
    node_name = (name or "").strip() or socket.gethostname()
    if not record.token_private_key:  # pragma: no cover - ensure_keypair makes one
        raise EnrollmentError("no-token-key", "No token key", "This node has no token key.")
    token_key = tokens.load_private(record.token_private_key)

    payload: dict[str, Any] = {
        "token": token,
        "name": node_name,
        "publicKey": record.public_key,
        "signingPublicKey": record.signing_public_key,
        "tokenPublicKey": tokens.public_b64(token_key),
        "agentVersion": __version__,
        "os": host_os(),
        "arch": host_arch(),
        "devices": devices or [],
    }
    if advertise_url is not None:
        payload["url"] = advertise_url
    if machine:
        payload.update(machine)
    try:
        secret, root_pin = split_join_token(token)
    except ValueError as exc:
        raise EnrollmentError(
            "join-token-damaged",
            "Join token damaged",
            f"{exc}. Copy the whole join command again; nothing was sent.",
        ) from exc
    payload["token"] = secret
    # The name people see (node-labels.md). The root checks it, before the
    # token is spent; omitted, a machine enrolling again keeps its label.
    if label is not None and label.strip():
        payload["label"] = label.strip()

    enroll_url = f"{control_url}/v1/nodes/enroll"
    if root_pin is not None and transport is None:
        enroll_url, transport = pinned_enrolment(control_url, root_pin)
    try:
        async with internal_client(timeout=ENROLL_TIMEOUT_SECONDS, transport=transport) as client:
            response = await client.post(enroll_url, json=payload)
    except httpx.HTTPError as exc:
        pin = refused_pin(exc)
        if pin is not None:
            raise EnrollmentError(
                "control-root-key-refused",
                "Not the control root's key",
                f"The machine at {control_url} presented a key ({pin}) that is not the one "
                "in the join command: the root's machine key changed after this join command "
                "was made, or something between this machine and the root answered in its "
                "place. Make a new join command; the join token was not sent.",
            ) from exc
        raise EnrollmentError(
            "control-root-unreachable",
            "Control root unreachable",
            f"Could not reach the control root at {control_url}: {exc}. Nothing was recorded; "
            f"this agent is still unenrolled.",
        ) from exc

    if response.status_code == 409 and _problem_title(response) == LABEL_TAKEN_TITLE:
        # The name people see is another machine's (node-labels.md). Not a
        # token problem: the root refused before spending it.
        raise EnrollmentError(
            "label-taken",
            LABEL_TAKEN_TITLE,
            f"{problem_detail(response)} Choose another name for this machine; the join "
            "token was not used and nothing was recorded.",
        )
    if response.status_code in (401, 409):
        # The root answers 401 for a token it does not know or that has
        # expired, and 409 for one already used. Neither changes by trying
        # again, which is why this has a slug of its own.
        raise EnrollmentError(
            TOKEN_REFUSED,
            "Join token refused",
            f"the control root at {control_url} answered {response.status_code}: "
            f"{problem_detail(response)}",
        )
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
    control_public_key = _str_or_none(body.get("controlPublicKey"))
    if not isinstance(granted_name, str) or not granted_name:
        raise _malformed(control_url, "`name` is missing")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise _malformed(control_url, f"`epoch` is {epoch!r}")
    if control_public_key is None:
        raise _malformed(control_url, "`controlPublicKey` is missing")
    raw_bundle = body.get("trustBundle")
    jws = raw_bundle.get("jws") if isinstance(raw_bundle, dict) else None
    if not isinstance(jws, str):
        raise _malformed(control_url, "`trustBundle` is missing")
    try:
        bundle = tokens.parse_bundle(jws, authority=control_public_key)
    except tokens.BundleError as exc:
        raise _malformed(control_url, f"its trust bundle does not verify ({exc})") from exc
    own = bundle.keys.get(tokens.thumbprint(token_key))
    if own is None or own.issuer != tokens.node_recipient(granted_name):
        raise _malformed(
            control_url,
            f"its trust bundle does not list this node's token key as node:{granted_name}",
        )

    # The bundle is kept before the record says enrolled: a crash between
    # the two leaves an unenrolled node with a stray file, never an
    # enrolled one that trusts nothing.
    from .trust import BUNDLE_FILE

    tokens.write_bundle_file(store.path.parent / BUNDLE_FILE, bundle)
    store.record_enrollment(
        name=granted_name,
        control_url=control_url,
        epoch=epoch,
        control_public_key=control_public_key,
        recovery_public_key=_str_or_none(body.get("recoveryPublicKey")),
        advertise_url=advertise_url,
    )
    store.record_control_pins([root_pin] if root_pin else [])
    return EnrollmentOutcome(
        name=granted_name, epoch=epoch, bundle=bundle, advertise_url=advertise_url
    )


async def announce_address(
    *,
    store: NodeIdentityStore,
    url: str,
    transport: Any = None,
    machine: dict[str, str] | None = None,
    old_root: Callable[[], bool] = lambda: False,
    refusals_logged: set[tuple[str, int, str]] | None = None,
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

    node_name = record.name
    target = f"{record.control_url.rstrip('/')}/v1/nodes/{node_name}"

    def signed(fields: dict[str, str]) -> dict[str, Any]:
        sequence = store.next_advertise_sequence()
        message = address_message(name=node_name, sequence=sequence, url=url, machine=fields)
        return {
            "url": url,
            "sequence": sequence,
            "signature": sign_address(
                signing_private_key=record.signing_private_key or "", message=message
            ),
            **fields,
        }

    sent = dict(machine or {})
    # One announcement at a time from this node (row 3: the key change
    # announces beside the address loop), so their sequence numbers reach
    # the root in the order they were claimed.
    async with store.announcing:
        return await _announce_locked(
            store=store,
            target=target,
            url=url,
            sent=sent,
            signed=signed,
            transport=transport,
            old_root=old_root,
            refusals_logged=refusals_logged,
        )


async def _announce_locked(
    *,
    store: NodeIdentityStore,
    target: str,
    url: str,
    sent: dict[str, str],
    signed: Callable[[dict[str, str]], dict[str, Any]],
    transport: Any,
    old_root: Callable[[], bool],
    refusals_logged: set[tuple[str, int, str]] | None = None,
) -> AnnounceOutcome:
    record = store.record
    try:
        async with internal_client(timeout=ANNOUNCE_TIMEOUT_SECONDS, transport=transport) as client:
            response = await client.patch(target, json=signed(sent))
            if sent and response.status_code == 401 and old_root():
                # A root older than LT1 checks a signature over three fields
                # and refuses five. Only a root shown to be older (it serves
                # no machine list) gets the address alone: a 401 by itself is
                # also what a party in the middle produces by stripping the
                # key, and falling back then would let it clear the key at a
                # root that would have taken it (adversarial review).
                log.info(
                    "the control root does not take machine ports (it runs an older "
                    "version); announcing the address alone"
                )
                sent = {}
                response = await client.patch(target, json=signed(sent))
    except httpx.HTTPError as exc:
        log.warning("could not announce this node's address to %s: %s", record.control_url, exc)
        return AnnounceOutcome(False, False, url, str(exc), retry=True)

    if response.status_code != 200:
        detail = problem_detail(response)
        # 409: another machine holds the address (LT2 review); it is let in
        # once that one announces where it moved, so ask again later.
        later = response.status_code >= 500 or response.status_code in (408, 409, 429)
        # A refusal is logged once per address and reason (row 36): the same
        # address is tried again every 15 minutes, and a log that repeats the
        # same refusal all day hides the next one.
        key = (url, response.status_code, detail)
        repeated = refusals_logged is not None and key in refusals_logged
        if refusals_logged is not None:
            refusals_logged.add(key)
        # 401 here has one likely cause and it is worth naming, because
        # the symptom (a node nothing can reach) is far away from it.
        if repeated:
            log.debug(
                "the control root refused this node's address announcement again (%d): %s",
                response.status_code,
                detail,
            )
        elif response.status_code == 401:
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
        return AnnounceOutcome(False, False, url, detail, retry=later)

    try:
        payload = response.json()
    except ValueError:
        payload = {}
    changed = bool(payload.get("changed")) if isinstance(payload, dict) else False
    if changed:
        log.info("told the control root this node is now reachable at %s", url)
    return AnnounceOutcome(True, changed, url, sent=sent)


async def keep_address_announced(
    *,
    store: NodeIdentityStore,
    resolve: Callable[[bool], Awaitable[str | None]],
    transport: Any = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    machine: Callable[[], dict[str, str] | None] = lambda: None,
    recorded: Callable[[], tuple[str | None, tuple[str, ...]] | None] = lambda: None,
    old_root: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    """Tell the control root where this node is until it has heard, and
    again whenever the address moves (agent#8). Runs for the agent's life
    and never raises, but for its cancellation.

    Before, the address was announced once, at boot: a node that booted
    before its root (a power cut on a LAN) stayed at an address nobody
    listened on until it restarted, and one that changed network while
    running kept its old address at the root.

    - An attempt that may work later (no answer, 408, 429, 5xx) is tried
      again after 5 s, doubling to 5 minutes, with jitter.
    - A definite refusal is logged once per address and reason, by
      `announce_address`, and announced again every 15 minutes (row 36): the
      root keeps why it refused only in memory, so a restarted root or a
      promoted standby learns it at the next attempt. The attempt sends no
      new field, so a root that does not know row 36 answers as it did.
    - Once settled, the address is re-derived every minute and announced
      when it differs. `resolve(quiet)` is `resolve_advertise_url`, so an
      operator's `advertiseUrl` still wins and a root that cannot be
      reached leaves the persisted address in place.

    An address a route has just announced (a config change, the Reach
    switch) is announced once more at the next check; the root answers
    *unchanged* and appends nothing.

    `machine` is this node's machine port and key when it is listening
    (LT1). It settles with the address: a port that starts listening, or a
    key that changes, is announced at the next check like a new address.
    And it is checked against what the root says it recorded (`recorded`,
    from the root's signed machine list): a record that differs, however it
    came to differ, is announced again.
    """
    settled: tuple[str, tuple[tuple[str, str], ...]] | None = None
    refused_at: float | None = None
    refusals_logged: set[tuple[str, int, str]] = set()
    warned = False
    delay = ANNOUNCE_RETRY_FIRST_SECONDS
    quiet = False
    while True:
        if store.record.enrolled:
            try:
                url = await resolve(quiet)
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("could not work out this node's address: %s", exc)
                url = None
            quiet = True
            if url is None:
                if not warned:
                    log.warning(
                        "enrolled with %s but this node has no address to advertise; other "
                        "hosts cannot reach it. Set `advertiseUrl` in the agent config.",
                        store.record.control_url,
                    )
                    warned = True
            elif (
                (want := (url, tuple(sorted((machine() or {}).items())))) != settled
                or _recorded_differs(machine() or {}, recorded())
                or (refused_at is not None and clock() - refused_at >= REFUSED_RETRY_SECONDS)
            ):
                warned = False
                store.record_advertise_url(url)
                fields = machine() or {}
                outcome = await announce_address(
                    store=store,
                    url=url,
                    transport=transport,
                    machine=fields,
                    old_root=old_root,
                    refusals_logged=refusals_logged,
                )
                refused_at = None if outcome.announced else clock()
                if outcome.announced or not outcome.retry:
                    # Settled on what the root took. An older root took the
                    # address alone, and that is all it can take until it is
                    # updated, so this node does not keep asking.
                    sent = outcome.sent if outcome.announced else None
                    settled = (
                        want
                        if sent == {} and old_root()
                        else (url, tuple(sorted((sent or fields).items())))
                    )
                    delay = ANNOUNCE_RETRY_FIRST_SECONDS
                else:
                    await sleep(delay * random.uniform(0.8, 1.2))
                    delay = min(delay * 2, ANNOUNCE_RETRY_MAX_SECONDS)
                    continue
        await sleep(ADDRESS_RECHECK_SECONDS)


_PIN = re.compile(r"^[A-Za-z0-9_-]{43}$")


def split_join_token(token: str) -> tuple[str, str | None]:
    """`(secret, root pin)` from a join token; the pin is None when it carries
    none (a root older than LT1). Raises `ValueError` for a token whose pin is
    damaged (cut short when it was copied): joining without the pin would
    send the secret where nobody checked who answers."""
    secret, dot, pin = token.strip().partition(".")
    if not dot:
        return secret, None
    if not _PIN.match(pin):
        raise ValueError("the root's key at the end of the join token is damaged")
    return secret, pin


def pinned_enrolment(control_url: str, pin: str) -> tuple[str, Any]:
    """The enrolment URL on the root's machine port, and a transport that
    accepts that port only with the key `pin` names.

    The root's machine port is at the address given for the root: the
    first-byte switch on its 8083 answers TLS and plain HTTP alike, so the
    same host and port that served plain enrolment serve it encrypted. An
    address that is already `https` (a nodes name behind the entry point)
    is used as it is.
    """
    from urllib.parse import urlsplit

    from ._http import MachineTransport, parse_machine_table, ssl_context

    parts = urlsplit(control_url)
    if parts.scheme != "http" or not parts.hostname:
        return f"{control_url}/v1/nodes/enroll", None
    hostport = parts.netloc
    table = parse_machine_table({"machines": {hostport: [pin]}, "names": {hostport: "the root"}})
    transport = MachineTransport(
        verify=ssl_context(),
        limits=httpx.Limits(max_connections=4, keepalive_expiry=1.5),
        table=lambda: table,
    )
    return f"https://{hostport}/control/v1/nodes/enroll", transport


def _recorded_differs(
    fields: dict[str, str], recorded: tuple[str | None, tuple[str, ...]] | None
) -> bool:
    """The root's signed list records another machine port or other keys for
    this node than the ones it announces (its key, and its next key while it
    changes keys, row 3). None (no list held) is no evidence either way."""
    if recorded is None or not fields:
        return False
    pins = tuple(
        sorted({fields["tlsPin"], *([fields["tlsNextPin"]] if "tlsNextPin" in fields else [])})
    )
    return recorded != (fields["machineUrl"].rstrip("/"), pins)


#: The control root's title for a label another machine has (`labelNode`).
LABEL_TAKEN_TITLE = "Name already in use"


def _problem_title(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    inner = payload.get("detail")
    title = inner.get("title") if isinstance(inner, dict) else payload.get("title")
    return title if isinstance(title, str) else None


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
    "ADDRESS_RECHECK_SECONDS",
    "ANNOUNCE_RETRY_FIRST_SECONDS",
    "ANNOUNCE_RETRY_MAX_SECONDS",
    "ANNOUNCE_TIMEOUT_SECONDS",
    "ENROLL_TIMEOUT_SECONDS",
    "REFUSED_RETRY_SECONDS",
    "TOKEN_REFUSED",
    "AnnounceOutcome",
    "EnrollmentError",
    "EnrollmentOutcome",
    "announce_address",
    "host_arch",
    "host_os",
    "keep_address_announced",
    "perform_enrollment",
    "pinned_enrolment",
    "problem_detail",
    "resolve_advertise_url",
    "split_join_token",
]
