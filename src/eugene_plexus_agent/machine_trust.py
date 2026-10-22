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
#: The children read the table file again at most once a second
#: (`_http._TABLE_RECHECK_SECONDS`); a version is acknowledged after this.
CHILD_CACHE_SECONDS = 1.5


class MachineListError(Exception):
    """A machine list that does not verify, or is older than the one held."""


class MachineListUnsigned(MachineListError):
    """Not a machine list signed by the root this node pinned (or not a
    machine list at all). The push route answers 401."""


class MachineListFenced(MachineListError):
    """A list from a superseded root (row 4): its `control.epoch` is below
    the epoch this node holds. The push route answers 409."""


def control_of(claims: dict[str, Any] | None) -> dict[str, Any] | None:
    """`MachineList.control` of a verified list (row 4): `{epoch, node, url}`,
    or None from a root older than row 4 or a block that does not hold
    together."""
    block = (claims or {}).get("control")
    if not isinstance(block, dict):
        return None
    epoch = block.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        return None
    url = block.get("url")
    node = block.get("node")
    return {
        "epoch": epoch,
        "node": node if isinstance(node, str) and node else None,
        "url": url.strip().rstrip("/") if isinstance(url, str) and url.strip() else None,
    }


def _why_not_followed(url: str) -> str | None:
    """Why a root address in a signed list is not one this node moves to."""
    from .node_identity import not_a_node_address

    why = not_a_node_address(url)
    if why is not None:
        return why
    if is_loopback_host(urlsplit(url).hostname):
        return "it is on the root's own machine only"
    return None


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
        raise MachineListUnsigned(
            f"the machine list is not signed by this node's root: {exc}"
        ) from exc
    if header.get("typ") != TYP_MACHINES:
        raise MachineListUnsigned("that is not a machine list")
    if not isinstance(claims.get("version"), int) or not isinstance(claims.get("nodes"), list):
        raise MachineListUnsigned("the machine list is malformed")
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
        #: The list version whose table is in the file the children read,
        #: and when it got there; and the last version acknowledged, once the
        #: children's cache had passed (review blocker 3).
        self._installed: tuple[int | None, float] | None = None
        self._acknowledged: int | None = None
        self._write_failed = False
        self.on_required_changed: Any = None
        #: Row 4: called (no arguments) once this node has followed a
        #: promoted root to a new address; the agent announces itself there
        #: and pulls the bundle (`leaving.after_root_moved`).
        self.on_moved: Any = None
        #: Row 4: the last root this node could not follow, said once.
        self._unfollowed: tuple[int, str] | None = None
        self.own_urls: Any = lambda: ()
        #: Whether the root serves a machine list: False once it answered 404
        #: (a root older than LT1), None until it has answered at all.
        self.root_has_list: bool | None = None

    @property
    def required(self) -> bool:
        return bool((self.claims or {}).get("required"))

    @property
    def version(self) -> int | None:
        """`MachinePort.listVersion`: the list version this machine's
        children can read, which the root takes as "this machine has the
        keys in it" (row 3 review, blocker 3). Not the version held while
        the table file could not be written, and not before the children's
        one-second cache has passed since it was."""
        if self._installed is not None:
            version, at = self._installed
            if time.perf_counter() - at >= CHILD_CACHE_SECONDS:
                self._acknowledged = version
        return self._acknowledged

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
        """Take a list from the root. Returns whether anything changed.

        Row 4: a list names the root (`control`). One from a superseded root
        (its epoch below the one this node holds) is refused outright
        (`MachineListFenced`). Between two lists that name roots at different
        epochs the higher epoch is the newer, whatever their versions: each
        root numbers its own. And a list at an epoch above the one this
        node's `controlUrl` came with moves the node there (`_follow`)."""
        record = self._identity.record
        if not record.enrolled or not record.control_public_key:
            raise MachineListError("this node is not enrolled, so it trusts no machine list")
        claims = verify(jws, record.control_public_key)
        held = int((self.claims or {}).get("version") or 0)
        offered, had = control_of(claims), control_of(self.claims)
        held_epoch = record.epoch or 0
        if offered is not None and offered["epoch"] < held_epoch:
            raise MachineListFenced(
                f"epoch {offered['epoch']} is below the {held_epoch} this node holds"
            )
        if offered is not None and had is not None and offered["epoch"] != had["epoch"]:
            if offered["epoch"] < had["epoch"]:
                raise MachineListError(
                    f"the root sent a machine list of epoch {offered['epoch']}, older than "
                    f"the one of epoch {had['epoch']} held"
                )
            same = False
        elif claims["version"] < held:
            raise MachineListError(
                f"the root sent machine list {claims['version']}, older than the {held} held"
            )
        else:
            same = claims["version"] == held
        self.taken_at = time.time()
        write_private(self.list_path, json.dumps({"jws": jws, "takenAt": self.taken_at}))
        # Before the table: it is built for the root's address.
        moved = self._follow(offered)
        if same:
            if self._write_failed or moved:
                # The same list again: write the table it could not, or the
                # one for the root it has just followed.
                self.rebuild()
            if moved:
                self._said_moved(offered)
            return moved
        was_required = self.required
        self.claims = claims
        self.rebuild()
        if moved:
            self._said_moved(offered)
        log.info(
            "took machine list %d: %d machine(s)%s",
            claims["version"],
            len(claims.get("nodes") or []),
            ", encryption between machines required" if self.required else "",
        )
        if self.required != was_required and self.on_required_changed is not None:
            self.on_required_changed(self.required)
        return True

    def _follow(self, control: dict[str, Any] | None) -> bool:
        """Row 4 (R4-MOVE): move this node to the root a verified, newer list
        names, when it names one at an epoch above the one this node's
        `controlUrl` came with. Returns whether `controlUrl` changed.

        The same epoch never moves a node: it may reach its root at an
        address the root does not know (a tailnet address), and every list
        names the root's own. An epoch above the one held is held as a
        bundle at that epoch would be; the caller has refused one below."""
        if control is None:
            return False
        record = self._identity.record
        epoch = int(control["epoch"])
        if epoch <= int(record.control_url_epoch or 0):
            if epoch > int(record.epoch or 0):
                self._identity.follow_root(None, epoch)
            return False
        url = control["url"]
        why = "it names no address" if url is None else _why_not_followed(str(url))
        if why is not None:
            if self._unfollowed != (epoch, why):
                self._unfollowed = (epoch, why)
                log.error(
                    "the install's root is at epoch %d now, but this node cannot follow it there: "
                    "%s. It keeps %s",
                    epoch,
                    why,
                    record.control_url,
                )
            self._identity.follow_root(None, epoch)
            return False
        before = record.control_url
        after = self._identity.follow_root(str(url), epoch)
        return (after.control_url or "").rstrip("/") != (before or "").rstrip("/")

    def _said_moved(self, control: dict[str, Any] | None) -> None:
        """The log line, and what the agent does next (`on_moved`)."""
        if control is None:
            return
        log.warning(
            "the install's root moved to %s (epoch %d); this node follows it at %s",
            control["node"] or control["url"],
            control["epoch"],
            control["url"],
        )
        if self.on_moved is not None:
            self.on_moved()

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

    def rebuild(self) -> bool:
        """Install the table here and write it for the children. Returns
        whether the file holds it; the version is acknowledged only then."""
        document = self.table()
        _http.set_machine_table(_http.parse_machine_table(document))
        rendered = json.dumps(document, sort_keys=True)
        if rendered != self._written:
            try:
                write_private(self.pins_path, rendered)
            except OSError as exc:
                self._write_failed = True
                log.warning("could not write the machine table to %s: %s", self.pins_path, exc)
                return False
            self._written = rendered
        self._write_failed = False
        version = document.get("version")
        installed = self._installed[0] if self._installed is not None else None
        if self._installed is None or installed != version:
            _ = self.version  # what was acknowledged stays so
            listed = version if isinstance(version, int) and version > 0 else None
            self._installed = (listed, time.perf_counter())
        return True

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
