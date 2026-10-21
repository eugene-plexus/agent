"""What this node holds about the other machines' TLS doors (LT1).

`specs/docs/design/tls-between-machines.md` §3.2. The root signs every
node's machine URL and machine-key pins, and the pins of the machine port in
front of itself, with its identity key (`GET /v1/trust/machines`). This node
pinned that key at enrolment, so the chain is: the root's identity key, the
signed list, a node's pin, the TLS session.

From the list this module builds the **machine table** every internal client
here uses (`_http.MachineTable`): which `host:port` presents which keys,
which plain address of another machine's agent or of the root is carried to
which TLS address, and whether the install requires encryption. It installs
the table in this process and writes it to `machine_pins.json` beside
`node.yaml` for the children, whose `_http` reads the file named by
`EUGENE_PLEXUS_MACHINE_PINS_FILE`.

**Kept on disk, re-verified at start.** A node whose root is down keeps
dialling the other machines on the pins it holds (§5's last sabotage). A
list is refused unless it verifies against the pinned identity key, and one
whose `version` is below the one held is refused too, so an old list cannot
bring back a key that was dropped.

**Before the first list, the join token.** A node that enrolled with a token
carrying the root's pin keeps that pin (`node.yaml` `controlPins`), so its
calls to the root are encrypted from the first one, before it has a list.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import jwt

from . import _http, tokens
from ._private_files import write_private
from .node_identity import NodeIdentityStore, is_loopback_host

log = logging.getLogger(__name__)

TYP_MACHINES = "ep-machines+jwt"
LIST_FILE = "machine_list.json"
PINS_FILE = "machine_pins.json"
LIST_PATH = "/v1/trust/machines"


class MachineListError(Exception):
    """A machine list that does not verify, or is older than the one held."""


def verify(jws: str, authority: str) -> dict[str, Any]:
    """The claims of a signed machine list, checked against the pinned root key."""
    try:
        header = jwt.get_unverified_header(jws)
        claims: dict[str, Any] = jwt.decode(
            jws,
            tokens.load_public(authority),
            algorithms=["EdDSA"],
            options={"verify_aud": False},
            leeway=tokens.LEEWAY_SECONDS,
        )
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise MachineListError(
            f"the machine list is not signed by this node's root: {exc}"
        ) from exc
    if header.get("typ") != TYP_MACHINES:
        raise MachineListError("that is not a machine list")
    if not isinstance(claims.get("version"), int) or not isinstance(claims.get("nodes"), list):
        raise MachineListError("the machine list is malformed")
    return claims


def _origin(url: str) -> str | None:
    parts = urlsplit(url)
    if not parts.hostname or parts.scheme not in ("http", "https"):
        return None
    port = parts.port or (443 if parts.scheme == "https" else 80)
    host = parts.hostname.lower()
    host = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{host}:{port}"


def build_table(
    claims: dict[str, Any] | None,
    *,
    own_name: str | None,
    control_url: str | None,
    token_pins: list[str],
    own_urls: tuple[str | None, ...] = (),
) -> dict[str, Any]:
    """The machine table, in the JSON form `_http.parse_machine_table` reads.

    **One claimant per door, never a union of keys** (adversarial review,
    2026-10-20). The root's door comes first and is authoritative: its keys
    are the ones the root signed as its own, or, before the first list, the
    one in the join token. A node that registered the same `host:port` with
    the same keys is the root's own machine and adds nothing; one with any
    other key is left out. Two nodes claiming one door are both left out.
    Otherwise a node whose keys were stolen could add its key to the root's
    address or another machine's, and every caller would accept a man in the
    middle there.
    """
    machines: dict[str, set[str]] = {}
    routes: dict[str, str | None] = {}
    names: dict[str, str] = {}
    contested: set[str] = set()
    control_door: str | None = None

    listed = [p for p in (claims or {}).get("controlPins") or [] if isinstance(p, str)]
    # The join token's key carries the calls until the root's signed list
    # arrives; after that the list alone says which keys are the root's.
    control_pins = listed or [p for p in token_pins if p]
    origin = _origin(control_url or "")
    if (
        origin is not None
        and origin.startswith("http://")
        and not is_loopback_host(urlsplit(origin).hostname)
    ):
        hostport = origin.removeprefix("http://")
        if control_pins:
            machines[hostport] = set(control_pins)
            names[hostport] = "the control root"
            control_door = hostport
            routes[origin] = f"https://{hostport}/control"
        else:
            routes[origin] = None
        names[origin] = "the control root"

    nodes = [n for n in (claims or {}).get("nodes") or [] if isinstance(n, dict)]
    for node in sorted(nodes, key=lambda n: str(n.get("name") or "")):
        if node.get("name") == own_name:
            continue
        name = str(node.get("name") or "a machine")
        agent = _origin(str(node.get("url") or ""))
        pins = {p for p in node.get("pins") or [] if isinstance(p, str)}
        machine_url = node.get("machineUrl")
        door = _origin(machine_url) if isinstance(machine_url, str) and pins else None
        carried: str | None = None
        if door is not None and door.startswith("https://"):
            hostport = door.removeprefix("https://")
            if hostport == control_door:
                if pins <= machines[hostport]:
                    carried = door
                else:
                    log.warning(
                        "%s registered the root's address %s with another key; ignored",
                        name,
                        hostport,
                    )
            elif hostport in machines or hostport in contested:
                log.warning("%s and %s both registered %s; neither is trusted there",
                            names.get(hostport), name, hostport)  # fmt: skip
                machines.pop(hostport, None)
                contested.add(hostport)
            else:
                machines[hostport] = pins
                names[hostport] = name
                carried = door
        if (
            agent is not None
            and agent.startswith("http://")
            and not is_loopback_host(urlsplit(agent).hostname)
        ):
            if agent == origin:
                # The root's address is the root's: a node claiming it is
                # ignored (adversarial review, 2026-10-20).
                log.warning("%s claims the root's address %s; ignored", name, agent)
                continue
            if agent in routes and names.get(agent) != name:
                # Two records claim one address: neither is carried there.
                log.warning("%s and %s both claim %s; neither is trusted there",
                            names.get(agent), name, agent)  # fmt: skip
                routes[agent] = None
                continue
            routes[agent] = (carried + "/agent") if carried else None
            names[agent] = name
    # Routes through a door that turned out contested carry nothing.
    for key, target in list(routes.items()):
        if target is not None:
            door_of = _origin(target)
            if door_of is None or door_of.removeprefix("https://") not in machines:
                routes[key] = None

    local = sorted({o for o in (_origin(u or "") for u in own_urls) if o})
    return {
        "version": int((claims or {}).get("version") or 0),
        "required": bool((claims or {}).get("required")),
        "machines": {hp: sorted(pins) for hp, pins in sorted(machines.items())},
        "routes": dict(sorted(routes.items())),
        "names": dict(sorted(names.items())),
        "local": local,
    }


class MachineTrust:
    """The list this node holds, and the table it gives itself and its children."""

    def __init__(self, directory: Path, identity: NodeIdentityStore) -> None:
        self.list_path = directory / LIST_FILE
        self.pins_path = directory / PINS_FILE
        self._identity = identity
        self.claims: dict[str, Any] | None = None
        self.taken_at: float | None = None
        self._written: str | None = None
        self.on_required_changed: Any = None
        self.own_urls: Any = lambda: ()
        #: Whether the root serves a machine list: False once it answered 404
        #: (a root older than LT1), None until it has answered at all.
        self.root_has_list: bool | None = None

    @property
    def required(self) -> bool:
        return bool((self.claims or {}).get("required"))

    @property
    def version(self) -> int | None:
        """The `version` of the list held (`MachinePort.listVersion`)."""
        held = (self.claims or {}).get("version")
        return held if isinstance(held, int) else None

    @property
    def age_seconds(self) -> int | None:
        if self.taken_at is None:
            return None
        return int(time.time() - self.taken_at)

    def load(self) -> None:
        """The list kept on disk, if it still verifies; then the table."""
        record = self._identity.record
        try:
            kept = json.loads(self.list_path.read_text(encoding="utf-8"))
            if record.enrolled and record.control_public_key and isinstance(kept, dict):
                self.claims = verify(str(kept["jws"]), record.control_public_key)
                self.taken_at = float(kept.get("takenAt") or 0) or None
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, MachineListError) as exc:
            log.warning("the kept machine list at %s is not usable: %s", self.list_path, exc)
        self.rebuild()

    def accept(self, jws: str) -> bool:
        """Take a list from the root. Returns whether anything changed."""
        record = self._identity.record
        if not record.enrolled or not record.control_public_key:
            raise MachineListError("this node is not enrolled, so it trusts no machine list")
        claims = verify(jws, record.control_public_key)
        held = int((self.claims or {}).get("version") or 0)
        if claims["version"] < held:
            raise MachineListError(
                f"the root sent machine list {claims['version']}, older than the {held} held"
            )
        self.taken_at = time.time()
        write_private(self.list_path, json.dumps({"jws": jws, "takenAt": self.taken_at}))
        if claims["version"] == held:
            return False
        was_required = self.required
        self.claims = claims
        self.rebuild()
        log.info(
            "took machine list %d: %d machine(s)%s",
            claims["version"],
            len(claims.get("nodes") or []),
            ", encryption between machines required" if self.required else "",
        )
        if self.required != was_required and self.on_required_changed is not None:
            self.on_required_changed(self.required)
        return True

    def table(self) -> dict[str, Any]:
        record = self._identity.record
        return build_table(
            self.claims,
            own_name=record.name,
            control_url=record.control_url,
            token_pins=[p for p in (record.control_pins or "").split(",") if p],
            own_urls=tuple(self.own_urls()),
        )

    def own_record(self) -> tuple[str | None, tuple[str, ...]] | None:
        """`(machineUrl, pins)` the root has recorded for this node, per the
        list held; None when no list is held."""
        if self.claims is None:
            return None
        name = self._identity.record.name
        for node in self.claims.get("nodes") or []:
            if isinstance(node, dict) and node.get("name") == name:
                machine_url = node.get("machineUrl")
                return (
                    str(machine_url).rstrip("/") if isinstance(machine_url, str) else None,
                    tuple(sorted(p for p in node.get("pins") or [] if isinstance(p, str))),
                )
        return (None, ())

    def rebuild(self) -> None:
        """Install the table here and write it for the children."""
        document = self.table()
        _http.set_machine_table(_http.parse_machine_table(document))
        rendered = json.dumps(document, sort_keys=True)
        if rendered != self._written:
            try:
                write_private(self.pins_path, rendered)
                self._written = rendered
            except OSError as exc:
                log.warning("could not write the machine table to %s: %s", self.pins_path, exc)

    def forget(self) -> None:
        """Leaving the install: no list, no table."""
        self.claims, self.taken_at = None, None
        self.list_path.unlink(missing_ok=True)
        self.rebuild()


async def _unpinned(url: str) -> httpx.Response:
    """The root's machine list without the key this node holds for the root:
    over TLS to the root's machine port, checking no key (row 3: the capture
    then holds no plain connection between machines), or, when that port does
    not answer TLS, in plain HTTP. Either way the list is taken only if it
    verifies (`MachineTrust.accept`)."""
    import ssl

    parts = urlsplit(url)
    if parts.scheme == "http" and parts.netloc:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        async with httpx.AsyncClient(trust_env=False, timeout=10, verify=context) as unpinned:
            try:
                return await unpinned.get(f"https://{parts.netloc}/control{LIST_PATH}")
            except httpx.HTTPError as exc:
                log.debug("the root's machine port did not answer TLS for its list: %s", exc)
    async with httpx.AsyncClient(
        trust_env=False, timeout=10, transport=httpx.AsyncHTTPTransport(trust_env=False)
    ) as plain:
        return await plain.get(url)


async def pull(trust: MachineTrust, client: Any, control_url: str) -> None:
    """Fetch and take the root's machine list. Never raises: a root that is
    down is a normal state, and the list held keeps working.

    **A root whose machine key changed** (its machine set up again, or its
    key changed while this node was off, row 3) presents a key this node does
    not hold for it, and every call to it is refused, the list's included.
    The list is public, signed with the root's identity key and versioned,
    so this node then fetches it once without that key (`_unpinned`) and
    takes it only if it verifies and is newer: a party in the middle can show
    it nothing it could use, and the root's new key comes with it.
    """
    url = control_url.rstrip("/") + LIST_PATH
    try:
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            pin = _http.refused_pin(exc)
            if pin is None:
                log.debug("could not pull the machine list: %s", exc)
                return
            log.warning(
                "the root presented a machine key this node does not hold for it (%s); "
                "checking the root's signed machine list for it",
                pin,
            )
            response = await _unpinned(url)
        if response.status_code == 200:
            trust.root_has_list = True
            trust.accept(str(response.json()["jws"]))
        elif response.status_code == 404:
            trust.root_has_list = False
        else:
            log.debug("the root answered %d for its machine list", response.status_code)
    except MachineListError as exc:
        log.warning("refused the root's machine list: %s", exc)
    except Exception as exc:
        log.debug("could not pull the machine list: %s", exc)
