"""This node's token key, the authority it trusts, and the trust bundle.

Per-node token keys (2026-09-25; `specs/docs/design/per-node-token-keys.md`).

**Enrolled**, this node:

* signs its own tokens with the token key in `node.yaml`, as
  `node:<name>`;
* trusts exactly the keys in the bundle the control root signed with the
  identity key this node pinned at enrollment, and nothing else;
* holds that bundle in `trust_bundle.json` beside `node.yaml`, where its
  children reload it when it changes.

**Standalone** (not enrolled), it is its own authority: it signs a bundle
naming only its own key, as `node:local`, with its identity signing key.
Nothing outside this machine trusts it, which is the honest state for an
install of one machine that has not joined anything.

What this node may mint is what its entry in the bundle grants, checked
here before anything is signed: anything for its own machine, `agent`
tokens for other machines and the root, and `gateway` tokens for other
machines only with the grant the operator's join token gave it. The
verifier on the far side checks the same thing again, so this is a
courtesy, not the boundary.

**A limit on frozen trust** (row 46, `specs/docs/design/rows/row-46.md`).
A node cut off from the root keeps its last bundle, so a machine removed
since then still verifies here. Past `maxTrustBundleAgeHours` (default
24, 0 for none) without word from the root, the agent refuses a token
signed by another machine's key on every route that changes something
(`dependencies.refuse_when_frozen`).

**Word from the root** is a bundle this node pulled from the root over a
channel only the root can answer (machine TLS pinned to the root's key,
or loopback), or a newer bundle from anywhere, counted from when the root
signed it. A signed bundle proves who signed it and when, never that the
root is alive now: the push door takes no bearer, and a party in the
middle of a plain HTTP pull can answer with the bundle it saw last. So a
copy of the held bundle, pushed or pulled in plain HTTP, is not word.

**The time of the last word is its own record** (`trust_heard.json`),
written only when there is word, never read off the bundle file: that
file is rewritten whenever any bundle is taken, a replay included.

**With no authenticated channel to the root the limit is not enforced**:
a live root reached in plain HTTP whose bundle does not change would
otherwise freeze its machines after a day. The setting says so.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from . import _http, tokens
from ._private_files import write_private
from .node_identity import NodeIdentityStore

log = logging.getLogger(__name__)

BUNDLE_FILE = "trust_bundle.json"
#: Row 46: when this node last had word from the root, and when its last
#: pull was answered by whatever answers at the root's address.
HEARD_FILE = "trust_heard.json"
STANDALONE_RECIPIENT = tokens.node_recipient(tokens.STANDALONE_NODE)

#: Pauses before writing the bundle file again when another process has it
#: open: Windows refuses to replace a file anyone holds (WinError 5 or 32),
#: and every component this agent runs reads that file.
SAVE_RETRY_DELAYS_SECONDS = (0.05, 0.15)

#: Row 46: the agent setting that limits frozen trust, its default, and its
#: ceiling (a year). 0 is no limit.
MAX_AGE_HOURS_KEY = "maxTrustBundleAgeHours"
DEFAULT_MAX_AGE_HOURS = 24
MOST_MAX_AGE_HOURS = 8760


def max_age_hours(value: object) -> int:
    """The limit in effect for a saved value: the value when it is a whole
    number of hours in range, else the default. The state drops a bad value
    at load with a warning, so this is only the last guard."""
    if isinstance(value, bool) or not isinstance(value, int):
        return DEFAULT_MAX_AGE_HOURS
    if value < 0 or value > MOST_MAX_AGE_HOURS:
        return DEFAULT_MAX_AGE_HOURS
    return value


def silence_words(seconds: int) -> str:
    """`31 hours`, `3 days`: how long the root has been silent, in the words
    the console's own warning uses."""
    hours = seconds / 3600
    if hours < 1.5:
        minutes = max(1, round(seconds / 60))
        return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"
    if hours < 36:
        return f"{round(hours)} hours"
    return f"{round(hours / 24)} days"


def frozen_refusal(seconds: int) -> str:
    """The one sentence a refused change carries (row 46)."""
    return (
        f"This machine has not heard from the root for {silence_words(seconds)}, so it takes "
        "no changes from other machines. Check its network to the root."
    )


#: What the setting says when the limit cannot be kept (row 46).
NOT_ENFORCED = "Not enforced: this machine reaches the root without encryption."


def channel_authenticated(url: str) -> bool:
    """Whether an answer from `url` can only come from the root (row 46):
    loopback, which never leaves this machine, or a machine port this node
    holds the root's pins for, directly or by the route its machine table
    carries a plain address on. Plain HTTP across a network is not."""
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, TypeError):
        return False
    host = (parsed.host or "").strip("[]").lower()
    if host == "localhost":
        return True
    try:
        if ipaddress.ip_address(host).is_loopback:
            return True
    except ValueError:
        pass
    return _http.machine_pinned(url) or _http.machine_route(url) is not None


def _age(at: float | None, perf: float | None, now: float | None) -> int | None:
    """Seconds since `at`: the longer of the wall clock's count and this
    process's own, so a clock set back cannot shorten it while it runs."""
    if at is None:
        return None
    wall = (time.time() if now is None else now) - at
    held = time.perf_counter() - perf if perf is not None else 0.0
    return max(0, int(max(wall, held)))


class BundleRollback(Exception):
    """An authentic bundle older than the one held. The caller answers 409."""


class MintRefused(Exception):
    """This node's grants do not cover what was asked for. The caller answers 403."""


class NodeTrust:
    """The one place this agent signs or verifies a token."""

    def __init__(self, identity: NodeIdentityStore, bundle_path: Path) -> None:
        self._identity = identity
        self._path = bundle_path
        self._lock = threading.Lock()
        self._bundle: tokens.TrustBundle | None = None
        # Row 46. When this node last had word from the root (module
        # docstring), and when its last pull was answered at all. Not the
        # bundle's `iat`: a pull returns the same signed bundle until
        # something changes, so on a quiet install that grows without
        # bound. Each with the same moment on this process's own clock, so
        # a wall clock set back cannot make the silence shorter while the
        # process runs (`perf_counter`, never `monotonic`). Kept in
        # `HEARD_FILE`, never derived from the bundle file.
        self._word_at: float | None = None
        self._word_perf: float | None = None
        self._pulled_at: float | None = None
        self._pulled_perf: float | None = None
        # The bundle held in memory but not yet on disk. The components
        # read the file, so it is written again until it is there.
        self._unsaved: tokens.TrustBundle | None = None
        self._save_lock = threading.Lock()

    # ----- who this node is ----------------------------------------------

    @property
    def enrolled(self) -> bool:
        return self._identity.record.enrolled

    @property
    def recipient(self) -> str:
        record = self._identity.record
        if record.enrolled and record.name:
            return tokens.node_recipient(record.name)
        return STANDALONE_RECIPIENT

    @property
    def node_name(self) -> str | None:
        record = self._identity.record
        return record.name if record.enrolled else None

    @property
    def authority(self) -> str:
        """The key a bundle must be signed by: the root's, or this node's own."""
        record = self._identity.record
        if record.enrolled and record.control_public_key:
            return record.control_public_key
        if not record.signing_public_key:
            raise RuntimeError("this node has no identity key yet; ensure_keypair runs first")
        return record.signing_public_key

    @property
    def bundle_path(self) -> Path:
        return self._path

    def signer(self) -> tokens.Signer:
        key = self._identity.record.token_private_key
        if not key:
            raise RuntimeError("this node has no token key yet; ensure_keypair runs first")
        return tokens.Signer(key=tokens.load_private(key), issuer=self.recipient)

    @property
    def bundle(self) -> tokens.TrustBundle | None:
        with self._lock:
            return self._bundle

    # ----- the bundle -----------------------------------------------------

    def load(self) -> None:
        """At boot: the kept bundle when enrolled, a fresh self-signed one when not."""
        if not self.enrolled:
            self.become_standalone()
            return
        try:
            document = self._path.read_text(encoding="utf-8")
            jws = str(json.loads(document)["jws"])
            bundle = tokens.parse_bundle(jws, authority=self.authority)
        except FileNotFoundError:
            log.warning(
                "enrolled, but no trust bundle is kept at %s; this node verifies nothing until "
                "it reaches the control root",
                self._path,
            )
            return
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.error("the kept trust bundle at %s was refused: %s", self._path, exc)
            return
        word, pulled = self._read_heard()
        now = time.time()
        if word is None:
            # No record: the first start after row 46, a node that has just
            # joined (its bundle was signed for the join), or a record lost.
            # The root's state is known as of its signature and no later;
            # the bundle file's own time says only when something last wrote
            # it, a replay included.
            word = min(now, float(bundle.iat))
        perf = time.perf_counter()
        with self._lock:
            self._bundle = bundle
            self._word_at = word
            self._word_perf = perf - max(0.0, now - word)
            self._pulled_at = pulled
            self._pulled_perf = None if pulled is None else perf - max(0.0, now - pulled)
        log.info(
            "trust bundle %d loaded (epoch %d, %d keys)",
            bundle.version,
            bundle.epoch,
            len(bundle.keys),
        )

    def become_standalone(self) -> tokens.TrustBundle:
        """This node as its own authority, trusted by nothing but itself."""
        record = self._identity.record
        if not record.signing_private_key:
            raise RuntimeError("this node has no identity key yet; ensure_keypair runs first")
        signer = self.signer()
        bundle = tokens.build_bundle(
            authority=tokens.load_private(record.signing_private_key),
            version=0,
            epoch=0,
            keys=[signer.trust_key([tokens.GRANT_AUTHORITY, tokens.GRANT_NODE])],
        )
        self._install(bundle)
        return bundle

    def accept(self, jws: str, *, pulled: bool = False) -> tokens.TrustBundle:
        """A bundle from the control root, by push or pull.

        Raises `tokens.BundleError` for a bad signature, `BundleRollback`
        for a bundle older than the one held (by epoch or version), and
        `FencedError` for an epoch below the one this node recorded --
        the fence that still holds when the kept bundle has been lost.
        Each leaves the held bundle alone, and the time last heard too.

        **What counts as word from the root** (row 46, module docstring).
        `pulled`: this node asked at the root's own address (the minute's
        pull). Over a channel only the root can answer
        (`root_channel_authenticated`) that is word now, whatever bundle it
        holds. Anything else -- a push, through a door that takes no
        bearer, or a pull in plain HTTP, which a party in the middle can
        answer -- counts only when the bundle is newer than the one held,
        and only from the time the root signed it. A pull in plain HTTP
        still says the root's address answered (`heard_age_seconds`).
        """
        if not self.enrolled:
            raise tokens.BundleError("this node is not enrolled, so it trusts no control root")
        offered = tokens.parse_bundle(jws, authority=self.authority)
        held = self.bundle
        why = tokens.accepts_replacement(held, offered)
        if why is not None:
            raise BundleRollback(why)
        newer = held is None or (offered.epoch, offered.version) > (held.epoch, held.version)
        authenticated = pulled and self.root_channel_authenticated()
        self._identity.accept_epoch(offered.epoch)
        self._install(offered)
        now, perf = time.time(), time.perf_counter()
        changed = False
        with self._lock:
            if authenticated:
                # Unconditionally: after a clock set back, `now` is below
                # the time kept, and the root still answered just now.
                self._word_at, self._word_perf = now, perf
                changed = True
            elif newer:
                signed = min(now, float(offered.iat))
                if self._word_at is None or signed > self._word_at:
                    self._word_at, self._word_perf = signed, perf - (now - signed)
                    changed = True
            if pulled:
                self._pulled_at, self._pulled_perf = now, perf
                changed = True
        if changed:
            self._write_heard()
        return offered

    def root_channel_authenticated(self) -> bool:
        """Whether this node reaches its root on a channel only the root can
        answer (`channel_authenticated`). Without one the limit is not
        enforced: a pull cannot be told from a replay (row 46)."""
        record = self._identity.record
        if not record.enrolled or not record.control_url:
            return False
        return channel_authenticated(str(record.control_url))

    # ----- the record of the last word (row 46) ---------------------------

    @property
    def heard_path(self) -> Path:
        return self._path.with_name(HEARD_FILE)

    def _read_heard(self) -> tuple[float | None, float | None]:
        """`(word, pulled)` as kept; Nones when there is no usable record."""
        try:
            document: Any = json.loads(self.heard_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None, None
        except (OSError, ValueError) as exc:
            log.warning(
                "the record of the root's last word at %s is unreadable: %s", self.heard_path, exc
            )
            return None, None

        def stamp(name: str) -> float | None:
            value = document.get(name) if isinstance(document, dict) else None
            if isinstance(value, bool) or not isinstance(value, int | float):
                return None
            return float(value)

        return stamp("wordAt"), stamp("pulledAt")

    def _write_heard(self) -> None:
        """Keep the times last heard. A write that fails leaves them in
        memory only: a restart then counts from an older time, never a
        newer one."""
        with self._lock:
            document = {"wordAt": self._word_at, "pulledAt": self._pulled_at}
        try:
            write_private(self.heard_path, json.dumps(document))
        except OSError as exc:
            log.warning("could not keep the root's last word at %s: %s", self.heard_path, exc)

    def _install(self, bundle: tokens.TrustBundle) -> None:
        with self._lock:
            self._bundle = bundle
            self._unsaved = bundle
        self.save_pending()

    def save_pending(self) -> bool:
        """Write the held bundle to its file if it is not there yet; True
        when the file is current.

        A bundle can arrive exactly once: a promoted root pushes it, and
        this node's pulls still go to the root that stopped. So a write
        that fails is not dropped. It is tried again at once, briefly,
        while another process has the file open, and then on every trust
        pull until it lands."""
        with self._save_lock:
            with self._lock:
                bundle = self._unsaved
            if bundle is None:
                return True
            error: OSError | None = None
            for delay in (*SAVE_RETRY_DELAYS_SECONDS, None):
                try:
                    tokens.write_bundle_file(self._path, bundle)
                except PermissionError as exc:
                    error = exc
                    if delay is None:
                        break
                    time.sleep(delay)
                except OSError as exc:
                    error = exc
                    break
                else:
                    with self._lock:
                        if self._unsaved is bundle:
                            self._unsaved = None
                    if error is not None:
                        log.info("kept trust bundle %d at %s", bundle.version, self._path)
                    return True
            log.warning(
                "could not keep trust bundle %d at %s: %s; it is held in memory, and "
                "written again at the next trust pull",
                bundle.version,
                self._path,
                error,
            )
            return False

    def forget(self) -> None:
        """Un-enrolled: drop the root's bundle; the caller becomes standalone."""
        with self._lock:
            self._bundle = None
            self._word_at = self._word_perf = None
            self._pulled_at = self._pulled_perf = None
        try:
            self.heard_path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not remove %s: %s", self.heard_path, exc)

    def word_age_seconds(self, *, now: float | None = None) -> int | None:
        """Seconds since this node last had word from the root (row 46); None
        when it has joined nothing and has no root to hear from."""
        if not self.enrolled:
            return None
        with self._lock:
            at, perf = self._word_at, self._word_perf
        return _age(at, perf, now)

    def heard_age_seconds(self, *, now: float | None = None) -> int | None:
        """Seconds since this node last heard from the root, as reported
        (`trustBundleAgeSeconds`); None if never, or when it has joined
        nothing and has no root to hear from.

        With an authenticated channel to the root, the word's age: what the
        limit counts. Without one, since the root's address last answered a
        pull too, which is all plain HTTP can say, and the limit is not
        enforced. The longer of the wall clock's count and this process's
        own (`_age`): a suspended Linux machine, whose process clock stops,
        still counts the days it slept."""
        if not self.enrolled:
            return None
        if self.root_channel_authenticated():
            return self.word_age_seconds(now=now)
        with self._lock:
            ages = [
                _age(self._word_at, self._word_perf, now),
                _age(self._pulled_at, self._pulled_perf, now),
            ]
        known = [age for age in ages if age is not None]
        return min(known) if known else None

    def limit_in_effect(self, limit_hours: int) -> int:
        """The limit, in hours, this node keeps: none (0) when the setting is
        0, or when no authenticated channel reaches the root (row 46)."""
        if limit_hours <= 0 or not self.enrolled or not self.root_channel_authenticated():
            return 0
        return limit_hours

    def frozen_age(self, limit_hours: int, *, now: float | None = None) -> int | None:
        """The root's silence in seconds when it is past the limit in effect,
        else None: no limit, joined nothing, no authenticated channel to the
        root, or word from it in time (row 46)."""
        limit = self.limit_in_effect(limit_hours)
        if limit <= 0:
            return None
        age = self.word_age_seconds(now=now)
        if age is None or age <= limit * 3600:
            return None
        return age

    def minted_elsewhere(self, claims: tokens.Claims) -> bool:
        """A token another machine's key signed: not the control root's, not
        this machine's own. `verify` already bound `iss` to the key that
        signed it, so neither can be claimed by a third key. Anything else
        a future bundle names (a site's key, a key nobody planned for) is
        another machine's too."""
        return claims.iss not in (tokens.ISSUER_CONTROL, self.recipient)

    # ----- verifying ------------------------------------------------------

    def verify(self, token: str, *, classes: tuple[str, ...]) -> tokens.Claims:
        bundle = self.bundle
        if bundle is None:
            raise tokens.TokenError(
                "this node holds no trust bundle yet, so it can verify nothing; it takes one "
                "from the control root at enrollment and every minute after"
            )
        return tokens.verify(token, bundle=bundle, recipient=self.recipient, classes=classes)

    def grants(self) -> frozenset[str]:
        """What this node's own key is granted in the bundle it holds."""
        bundle = self.bundle
        if bundle is None:
            return frozenset()
        entry = bundle.keys.get(self.signer().kid)
        return entry.grants if entry is not None else frozenset()

    # ----- minting --------------------------------------------------------

    def mint_service(
        self, *, sub: str, audience: str, ttl_seconds: int | None = None
    ) -> tuple[str, int]:
        """A service token from this node, addressed to one recipient.

        To this machine: any `sub`, a year by default (children's tokens).
        Anywhere else: only what the grants cover, fifteen minutes by
        default and never more than the hour a verifier allows.
        """
        local = audience == self.recipient
        if not local:
            grants = self.grants()
            allowed = tokens.GRANT_AUTHORITY in grants or (
                sub == tokens.SUB_AGENT
                or (sub == tokens.SUB_GATEWAY and tokens.GRANT_GATEWAY in grants)
                or (
                    sub == tokens.SUB_STANDBY
                    and tokens.GRANT_STANDBY in grants
                    and audience == tokens.RECIPIENT_CONTROL
                )
            )
            if not allowed:
                raise MintRefused(
                    f"this node may not send a {sub!r} token to {audience}"
                    + {
                        tokens.SUB_GATEWAY: ": it holds no gateway grant",
                        tokens.SUB_STANDBY: (
                            ": it is not the standby"
                            if tokens.GRANT_STANDBY not in grants
                            else ": a standby token goes to control only"
                        ),
                    }.get(sub, "")
                )
        ttl = ttl_seconds or (
            tokens.LOCAL_SERVICE_TTL_SECONDS if local else tokens.REMOTE_SERVICE_TTL_SECONDS
        )
        if not local:
            ttl = min(ttl, tokens.MAX_REMOTE_SERVICE_SECONDS)
        return self.signer().mint(typ=tokens.TYP_SERVICE, sub=sub, aud=[audience], ttl_seconds=ttl)

    def agent_token(self, audience: str) -> str:
        """This agent speaking for itself to one recipient."""
        token, _ = self.mint_service(sub=tokens.SUB_AGENT, audience=audience)
        return token

    def mint_local_session(self) -> tuple[str, int]:
        """A standalone node's own operator session. Enrolled nodes never mint one."""
        if self.enrolled:
            raise MintRefused("an enrolled node's sessions are minted by the control root")
        return self.signer().mint(
            typ=tokens.TYP_SESSION,
            sub=tokens.SUB_OPERATOR,
            aud=[self.recipient],
            ttl_seconds=tokens.SESSION_TTL_SECONDS,
        )

    def mint_local_client(self, *, key_id: str, name: str, ttl_seconds: int) -> tuple[str, int]:
        """A standalone node's client key. Enrolled nodes forward to the root."""
        if self.enrolled:
            raise MintRefused("an enrolled node's client keys are minted by the control root")
        return self.signer().mint(
            typ=tokens.TYP_CLIENT,
            sub=name or "client",
            aud=[tokens.RECIPIENT_GATEWAY],
            ttl_seconds=ttl_seconds,
            jti=key_id,
        )
