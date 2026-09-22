"""Writing a file that holds a secret: owner-only from its first byte, and atomic.

These files this agent writes carry something another local user must
not read: `agent.yaml` (the passphrase's Argon2id hash, the master-key
salt and every sealed share password), the copy of an `agent.yaml` that
would not load (the same, verbatim), `node.yaml` (THE INSTALL'S signing
key, in the clear on purpose -- see `node_identity`), and each companion
driver's config (an `apiKey` an operator saved through the driver's own
Config page rides along every time the agent rewrites the file). The
sign-outs file, `revoked_sessions.json`, holds only hashes, and is
written the same way so that nobody else on the host can edit a
sign-out out of it. Until 2026-09-22 only `client_keys.json` was
created private. The rest took
the process umask -- 0644 on every stock Linux and macOS -- and
`node.yaml` was chmodded to 0600 *after* its bytes were already on
disk, which is a window, not a guarantee.

**The mode is given to `os.open` at creation, not applied afterwards.**
`O_CREAT | O_EXCL` with 0o600 means there is no instant at which the
file exists with the secret in it and a looser mode, and `O_EXCL` means
we never adopt a file somebody else created at that name with a mode of
their choosing. A `chmod` after the write can say neither.

**And it is still the atomic write `state.py` already had** (review
§6.1 #6): a temp file in the same directory, flushed and `fsync`ed,
then `os.replace`d over the target, so a crash leaves either the old
file or the new one and never half of one. `os.replace` is a rename, so
the target takes the temp file's mode -- which is also how a file an
older build left at 0644 is tightened, on its next write, with nothing
to migrate.

The temp name carries the pid and a random tail rather than a fixed
`.tmp`: with `O_EXCL`, a fixed name left behind by a crash would refuse
every later write, and a fixed name reused with `O_TRUNC` (what
`client_keys.py` did) keeps whatever mode the leftover already had.

On Windows the mode bits mean only "not read-only", and what protects
these files is the ACL the install directory hands down. That is the
installer's business; this module makes the POSIX half true and leaves
the Windows half no worse.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

__all__ = ["PRIVATE_MODE", "write_private"]

PRIVATE_MODE = 0o600
"""Owner read and write, nobody else anything."""

# `O_BINARY` exists only on Windows, where omitting it opens the
# descriptor in text mode and the C runtime rewrites every `\n` it is
# handed. The bytes we are given are the bytes that should land.
_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)


def write_private(path: Path, data: str | bytes, *, encoding: str = "utf-8") -> None:
    """Replace `path` with `data`, created owner-only, atomically.

    Raises whatever the write raised, after removing the temp file: a
    half-written temp beside the config the supervisor reads is litter
    in the one directory an operator is told to look at. Including
    `KeyboardInterrupt` and `SystemExit`, because interruptions are the
    reason this is atomic at all.
    """
    payload = data.encode(encoding) if isinstance(data, str) else data
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same directory, so `os.replace` is a rename within one volume. A
    # temp under the system temp dir would make it a copy, which is the
    # non-atomic thing this exists to avoid.
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    fd = os.open(tmp, _FLAGS, PRIVATE_MODE)
    try:
        try:
            out = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with out:
            out.write(payload)
            out.flush()
            # Before the replace, not after: the ordering is what the
            # durability depends on, and skipping it would let the
            # metadata rename land ahead of the data on a crash.
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
