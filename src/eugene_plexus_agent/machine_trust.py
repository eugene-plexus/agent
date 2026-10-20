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
) -> dict[str, Any]:
    """The machine table, in the JSON form `_http.parse_machine_table` reads."""
    machines: dict[str, set[str]] = {}
    routes: dict[str, str | None] = {}
    names: dict[str, str] = {}

    def door(url: str, pins: list[str], who: str) -> str | None:
        origin = _origin(url)
        if origin is None or not origin.startswith("https://"):
            return None
        hostport = origin.removeprefix("https://")
        machines.setdefault(hostport, set()).update(p for p in pins if isinstance(p, str))
        names.setdefault(hostport, who)
        return origin

    for node in (claims or {}).get("nodes") or []:
        if not isinstance(node, dict) or node.get("name") == own_name:
            continue
        name = str(node.get("name") or "a machine")
        agent = _origin(str(node.get("url") or ""))
        pins = [p for p in node.get("pins") or [] if isinstance(p, str)]
        machine_url = node.get("machineUrl")
        carried = (
            door(str(machine_url), pins, name) if isinstance(machine_url, str) and pins else None
        )
        if agent is not None and agent.startswith("http://"):
            routes[agent] = (carried + "/agent") if carried else None
            names[agent] = name

    control_pins = [p for p in (claims or {}).get("controlPins") or [] if isinstance(p, str)]
    control_pins += [p for p in token_pins if p not in control_pins]
    origin = _origin(control_url or "")
    if (
        origin is not None
        and origin.startswith("http://")
        and not is_loopback_host(urlsplit(origin).hostname)
    ):
        hostport = origin.removeprefix("http://")
        if control_pins:
            door("https://" + hostport, control_pins, "the control root")
            routes[origin] = f"https://{hostport}/control"
        else:
            routes[origin] = None
        names[origin] = "the control root"

    return {
        "version": int((claims or {}).get("version") or 0),
        "required": bool((claims or {}).get("required")),
        "machines": {hp: sorted(pins) for hp, pins in sorted(machines.items())},
        "routes": dict(sorted(routes.items())),
        "names": dict(sorted(names.items())),
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

    @property
    def required(self) -> bool:
        return bool((self.claims or {}).get("required"))

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
        )

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


async def pull(trust: MachineTrust, client: Any, control_url: str) -> None:
    """Fetch and take the root's machine list. Never raises: a root that is
    down is a normal state, and the list held keeps working."""
    try:
        response = await client.get(control_url.rstrip("/") + LIST_PATH)
        if response.status_code == 200:
            trust.accept(str(response.json()["jws"]))
        elif response.status_code != 404:
            log.debug("the root answered %d for its machine list", response.status_code)
    except MachineListError as exc:
        log.warning("refused the root's machine list: %s", exc)
    except Exception as exc:
        log.debug("could not pull the machine list: %s", exc)
