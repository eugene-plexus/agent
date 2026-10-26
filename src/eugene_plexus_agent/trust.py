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
(`dependencies.refuse_when_frozen`). Word from the root is a bundle this
node pulled from the root, or a newer bundle pushed to it, counted from
when the root signed it: the push door takes no bearer, so a copy of the
bundle already held, posted again by anybody, is not word.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from . import tokens
from .node_identity import NodeIdentityStore

log = logging.getLogger(__name__)

BUNDLE_FILE = "trust_bundle.json"
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
        # When this node last took the root's bundle. Not the bundle's
        # `iat`: a pull returns the same signed bundle until something
        # changes, so on a quiet install that grows without bound.
        self._heard_at: float | None = None
        # The same moment on this process's own clock (row 46): a wall
        # clock set back cannot make the root's silence shorter while
        # the process runs. `perf_counter`, never `monotonic`.
        self._heard_perf: float | None = None
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
        try:
            heard = self._path.stat().st_mtime
        except OSError:
            # Row 46: never fresher than its signature. Taken at or after
            # `iat`, so this can only make the silence look longer.
            heard = float(bundle.iat)
        now = time.time()
        with self._lock:
            self._bundle = bundle
            # The kept file is rewritten every time a bundle is taken, so
            # its time is when this node last heard, across a restart.
            self._heard_at = heard
            self._heard_perf = time.perf_counter() - max(0.0, now - heard)
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

        **What counts as word from the root** (row 46). `pulled`: this
        node asked the root at its own address (the minute's pull), so
        the answer is word now, whatever bundle it holds. Otherwise the
        bundle was pushed through a door that takes no bearer: anybody
        who can reach this machine can post a signed bundle they saw
        once. So a pushed bundle counts only when it is newer than the
        one held, and only from the time the root signed it.
        """
        if not self.enrolled:
            raise tokens.BundleError("this node is not enrolled, so it trusts no control root")
        offered = tokens.parse_bundle(jws, authority=self.authority)
        held = self.bundle
        why = tokens.accepts_replacement(held, offered)
        if why is not None:
            raise BundleRollback(why)
        newer = held is None or (offered.epoch, offered.version) > (held.epoch, held.version)
        self._identity.accept_epoch(offered.epoch)
        self._install(offered)
        now = time.time()
        with self._lock:
            if pulled:
                # Unconditionally: after a clock set back, `now` is below
                # the time kept, and the root still answered just now.
                self._heard_at = now
                self._heard_perf = time.perf_counter()
            elif newer:
                signed = min(now, float(offered.iat))
                if self._heard_at is None or signed > self._heard_at:
                    self._heard_at = signed
                    self._heard_perf = time.perf_counter() - (now - signed)
        return offered

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
            self._heard_at = None
            self._heard_perf = None

    def heard_age_seconds(self, *, now: float | None = None) -> int | None:
        """Seconds since this node last took the root's bundle; None if never,
        or when it has joined nothing and has no root to hear from.

        The longer of the wall clock's count and this process's own (row
        46): a clock set back cannot shorten the silence while the process
        runs, and a suspended Linux machine, whose process clock stops,
        still counts the days it slept."""
        with self._lock:
            heard, heard_perf = self._heard_at, self._heard_perf
        if heard is None or not self.enrolled:
            return None
        wall = (time.time() if now is None else now) - heard
        held = time.perf_counter() - heard_perf if heard_perf is not None else 0.0
        return max(0, int(max(wall, held)))

    def frozen_age(self, limit_hours: int, *, now: float | None = None) -> int | None:
        """The root's silence in seconds when it is past `limit_hours`, else
        None: no limit (0), joined nothing, or heard from in time (row 46)."""
        if limit_hours <= 0:
            return None
        age = self.heard_age_seconds(now=now)
        if age is None or age <= limit_hours * 3600:
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
