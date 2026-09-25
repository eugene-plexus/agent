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
            heard = None
        with self._lock:
            self._bundle = bundle
            # The kept file is rewritten every time a bundle is taken, so
            # its time is when this node last heard, across a restart.
            self._heard_at = heard
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

    def accept(self, jws: str) -> tokens.TrustBundle:
        """A bundle from the control root, by push or pull.

        Raises `tokens.BundleError` for a bad signature, `BundleRollback`
        for a bundle older than the one held (by epoch or version), and
        `FencedError` for an epoch below the one this node recorded --
        the fence that still holds when the kept bundle has been lost.
        Each leaves the held bundle alone.
        """
        if not self.enrolled:
            raise tokens.BundleError("this node is not enrolled, so it trusts no control root")
        offered = tokens.parse_bundle(jws, authority=self.authority)
        why = tokens.accepts_replacement(self.bundle, offered)
        if why is not None:
            raise BundleRollback(why)
        self._identity.accept_epoch(offered.epoch)
        self._install(offered)
        with self._lock:
            self._heard_at = time.time()
        return offered

    def _install(self, bundle: tokens.TrustBundle) -> None:
        with self._lock:
            self._bundle = bundle
        try:
            tokens.write_bundle_file(self._path, bundle)
        except OSError as exc:
            log.warning("could not keep the trust bundle at %s: %s", self._path, exc)

    def forget(self) -> None:
        """Un-enrolled: drop the root's bundle; the caller becomes standalone."""
        with self._lock:
            self._bundle = None
            self._heard_at = None

    def heard_age_seconds(self, *, now: float | None = None) -> int | None:
        """Seconds since this node last took the root's bundle; None if never,
        or when it has joined nothing and has no root to hear from."""
        with self._lock:
            heard = self._heard_at
        if heard is None or not self.enrolled:
            return None
        return max(0, int((time.time() if now is None else now) - heard))

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
            )
            if not allowed:
                raise MintRefused(
                    f"this node may not send a {sub!r} token to {audience}"
                    + (": it holds no gateway grant" if sub == tokens.SUB_GATEWAY else "")
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
