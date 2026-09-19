"""Log this host in to the file servers its Library folders live on.

**Why this module exists, and it is not a convenience.** R2.6 makes the
Windows agent a LocalSystem service so the install comes back after a
reboot with nobody signed in. A service has its own credential store,
and it is empty: every share the person who installed Eugene reached by
typing a password into Explorer once becomes unreachable, silently,
with every health check green.

Measured on the live install, 2026-09-18, against the same server under
a name the session had no SMB session for:

    \\\\192.168.16.252\\downloads\\models   OK, 1 entry
    \\\\CORBIN01\\downloads\\models         OSError winerror=1272
      "You can't access this shared folder because your organization's
       security policies block unauthenticated guest access."

    WNetAddConnection2W(no credentials)  -> 1272
    WNetAddConnection2W(bad credentials) -> 1272

Note what that says. The share is **guest-open** — it asks for nobody in
particular — and it is **Windows 11 that refuses**, because
`EnableInsecureGuestLogons` is 0 by default. *"The share needs no
password"* and *"anything can open it"* are different sentences, and the
gap between them is this module.

## What it does, and deliberately does not

It asks **the OS** to establish a session, with `WNetAddConnection2W` and
no local name, so nothing is mounted at a drive letter. Everything the
agent spawns inherits that session by virtue of running in the same
logon session, so `llama-server` opens a UNC path without knowing a
credential was ever involved. The alternative — opening files ourselves
with a credential and handing bytes to the engine — would mean the agent
became a file server, which is a different product.

It does **not** manage drive letters, does not persist anything the OS
would persist (`CONNECT_UPDATE_PROFILE` is never passed), and does not
unmount on shutdown: a session belongs to the logon session and goes
when the process host does.

## Why the server and not the share

Windows keys an SMB session by server, and refuses a second set of
credentials to a server it already has one for —
`ERROR_SESSION_CREDENTIAL_CONFLICT`, 1219. A per-share credential field
would be one the OS could not always honour, so the contract makes the
server the unit and this module treats 1219 as *already connected as
somebody* rather than as a failure.

## Why it is not fatal

An agent that cannot reach a share still supervises everything else, and
`degraded-mode-required` applies: every failure here becomes a log line
and a reported reason, never a refusal to start. The place an operator
finds out is the model that will not load, which already says where it
looked.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: `WNetAddConnection2W` results this module knows how to explain. Anything
#: else is reported by number, because a number that can be searched for
#: beats a sentence that guesses.
_MEANING = {
    0: "connected",
    5: "the server accepted the login but refused this share",
    53: "the server could not be found on the network",
    67: "there is no such share on that server",
    86: "the password was not accepted",
    1219: "Windows already holds a connection to that server as somebody else",
    1326: "the user name or password was not accepted",
    # The one that costs a service its models, and the one nobody expects
    # because the share does not have a password at all.
    1272: (
        "Windows refused an unauthenticated connection to that server, so the "
        "share needs a user name even if it does not need a password"
    ),
}

#: Not `RESOURCETYPE_DISK`'s neighbours -- just the one, because a printer
#: is not a thing this product opens.
_RESOURCETYPE_DISK = 1


@dataclass(frozen=True)
class ConnectResult:
    """What happened for one server, in a form a log line and a test agree on."""

    host: str
    username: str
    code: int
    ok: bool
    detail: str

    @property
    def summary(self) -> str:
        return f"{self.host} as {self.username}: {self.detail}"


def explain(code: int) -> str:
    """A sentence for a `WNetAddConnection2W` result, or the number."""
    known = _MEANING.get(code)
    return known if known is not None else f"Windows returned error {code}"


def connect(host: str, username: str, password: str | None) -> ConnectResult:
    """Establish a session to one file server. Never raises.

    `1219` is reported as **ok**: it means this logon session already has
    a connection to that server, which is the state we wanted. Tearing
    the existing one down to replace it would break whatever opened it —
    on a logon-task install that is the person's own Explorer window.
    """
    if sys.platform != "win32":
        # Not an error and not silence: POSIX mounts are the
        # administrator's, in fstab or autofs, and an agent that started
        # mounting things there would be fighting the system it runs on.
        return ConnectResult(
            host=host,
            username=username,
            code=0,
            ok=True,
            detail="nothing to do: shares are mounted by the system on this platform",
        )
    try:
        import ctypes

        mpr = ctypes.WinDLL("mpr", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]

        class _NetResource(ctypes.Structure):
            _fields_ = [
                ("dwScope", ctypes.c_uint32),
                ("dwType", ctypes.c_uint32),
                ("dwDisplayType", ctypes.c_uint32),
                ("dwUsage", ctypes.c_uint32),
                ("lpLocalName", ctypes.c_wchar_p),
                ("lpRemoteName", ctypes.c_wchar_p),
                ("lpComment", ctypes.c_wchar_p),
                ("lpProvider", ctypes.c_wchar_p),
            ]

        resource = _NetResource(
            0,
            _RESOURCETYPE_DISK,
            0,
            0,
            # No local name: a drive letter belongs to an interactive
            # session, and this code exists precisely for the case where
            # there is not one.
            None,
            rf"\\{host}",
            None,
            None,
        )
        code = int(
            mpr.WNetAddConnection2W(
                ctypes.byref(resource),
                password,
                username,
                0,  # never CONNECT_UPDATE_PROFILE: we do not write the user's profile
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ConnectResult(
            host=host, username=username, code=-1, ok=False, detail=f"could not ask Windows: {exc}"
        )
    ok = code in (0, 1219)
    return ConnectResult(host=host, username=username, code=code, ok=ok, detail=explain(code))


def connect_all(entries: list[dict[str, object]]) -> list[ConnectResult]:
    """Establish every configured session, in order, and report each.

    Order is the operator's, and it is preserved for the 1219 case: two
    entries for one server means the first wins, which is at least
    predictable. An entry with no `host` is skipped rather than guessed
    at.
    """
    results: list[ConnectResult] = []
    seen: set[str] = set()
    for entry in entries:
        host = str(entry.get("host") or "").strip().strip("\\/")
        if not host:
            continue
        key = host.casefold()
        if key in seen:
            log.warning(
                "two share credentials name %s; Windows allows one session per server, "
                "so the first is the one in effect",
                host,
            )
            continue
        seen.add(key)
        username = str(entry.get("username") or "")
        raw = entry.get("password")
        password = None if raw is None else str(raw)
        result = connect(host, username, password)
        (log.info if result.ok else log.warning)("share login: %s", result.summary)
        results.append(result)
    return results


# --------------------------------------------------------------------------- #
# at rest: sealed; on the wire: redacted
# --------------------------------------------------------------------------- #
#
# **The agent had never sealed anything of its own.** `security.seal`
# exists in all five repos and the agent's copy had no call site: the
# master key is handed to children at spawn and each child seals its own
# `apiKey`. This is the first secret the agent itself has to keep, so the
# sealing happens at the route boundary rather than inside `AgentState`
# -- the state object owns a file and a lock and has deliberately never
# known about auth, and a share password is not the reason to change
# that.


def redact_entries(stored: object) -> list[dict[str, object]]:
    """What `GET /v1/config` answers: every entry, no password.

    `None` and not an empty string, and not the entry omitted. A UI has
    to be able to render the row -- host and user are the half a person
    edits -- and it has to be able to tell *there is a password stored*
    from *there is no password*, because the first is a row that works
    and the second is a row that does not.
    """
    if not isinstance(stored, list):
        return []
    out: list[dict[str, object]] = []
    for entry in stored:
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "host": entry.get("host"),
                "username": entry.get("username"),
                "password": None,
            }
        )
    return out


def has_password(entry: object) -> bool:
    """Is there a stored password on this entry? Sealed or plain."""
    return isinstance(entry, dict) and entry.get("password") not in (None, "")


def merge_and_seal(
    incoming: object,
    stored: object,
    master_key: bytes | None,
) -> tuple[list[dict[str, object]], str | None]:
    """Turn a PATCH body into what goes on disk. `(entries, error)`.

    Three rules, and the first is the one that stops a UI blanking a
    password it was never shown:

    * an entry with **no** `password` key, whose `host` is already
      stored, keeps the stored (sealed) password. `GET` redacts, so a
      round-trip of *read the config, change the user name, write it
      back* would otherwise clear the secret -- and the operator would
      find out at the next reboot.
    * an entry with `password: ""` **clears** it. Explicit, and the only
      way to.
    * anything else is sealed with the install's master key.

    The error is a sentence, never an exception: this runs inside a
    config PATCH, and `degraded-mode-required` means a bad value is
    rejected with a reason rather than raising through the route.
    """
    if not isinstance(incoming, list):
        return [], f"expected a list of share credentials, got {type(incoming).__name__}"

    by_host: dict[str, dict[str, object]] = {}
    if isinstance(stored, list):
        for entry in stored:
            if isinstance(entry, dict) and isinstance(entry.get("host"), str):
                by_host[entry["host"].strip().casefold()] = entry

    out: list[dict[str, object]] = []
    for index, entry in enumerate(incoming):
        if not isinstance(entry, dict):
            return [], f"entry {index + 1}: expected an object"
        host = str(entry.get("host") or "").strip()
        kept: dict[str, object] = {"host": host, "username": entry.get("username")}
        if "password" not in entry or entry.get("password") is None:
            previous = by_host.get(host.casefold())
            if previous is not None and has_password(previous):
                kept["password"] = previous["password"]
            else:
                kept["password"] = None
        elif entry.get("password") == "":
            kept["password"] = None
        else:
            if master_key is None:
                return [], (
                    "this agent is locked, so it cannot store a password yet. Sign in and try again"
                )
            from . import security

            kept["password"] = security.seal(str(entry["password"]), master_key).to_dict()
        out.append(kept)
    return out, None


def unseal_entries(stored: object, master_key: bytes | None) -> list[dict[str, object]]:
    """What `connect_all` takes: the same entries with real passwords.

    An entry whose envelope will not open is passed on with **no**
    password rather than dropped, so the attempt is still made and the
    failure is Windows' sentence about the credential rather than our
    silence about the entry. A master key that has been rotated away
    from is the case that produces it, and the honest report is *the
    server refused this login*, which is true.
    """
    if not isinstance(stored, list):
        return []
    from . import security

    out: list[dict[str, object]] = []
    for entry in stored:
        if not isinstance(entry, dict):
            continue
        password = entry.get("password")
        plain: str | None = None
        if isinstance(password, dict) and security.is_envelope(password):
            if master_key is not None:
                try:
                    plain = security.open_envelope(
                        security.Envelope.from_dict(password), master_key
                    )
                except ValueError:
                    log.warning(
                        "the stored password for %s could not be opened with this "
                        "install's master key",
                        entry.get("host"),
                    )
        elif isinstance(password, str) and password:
            # Written before sealing existed, or by hand. Honoured, and
            # re-sealed the next time the operator saves the field.
            plain = password
        out.append(
            {"host": entry.get("host"), "username": entry.get("username"), "password": plain}
        )
    return out


def overlay_typed(typed: object, saved: list[dict[str, object]]) -> list[dict[str, object]]:
    """What the Test button should actually try: the row being edited.

    `POST /v1/config/test` exists so an operator can check a value
    **before** saving it, so the typed password wins — and a row whose
    password box is untouched falls back to the saved one, because the
    UI was shown a redaction and has nothing else to send. Entries are
    the typed list, in the typed order: a row the operator deleted must
    not be tested as though it were still there.
    """
    if not isinstance(typed, list):
        return saved
    by_host = {str(e.get("host") or "").strip().casefold(): e for e in saved if isinstance(e, dict)}
    out: list[dict[str, object]] = []
    for entry in typed:
        if not isinstance(entry, dict):
            continue
        host = str(entry.get("host") or "").strip()
        password = entry.get("password")
        if password in (None, ""):
            previous = by_host.get(host.casefold())
            password = previous.get("password") if previous else None
        out.append({"host": host, "username": entry.get("username"), "password": password})
    return out


def validate_entries(value: object) -> str | None:
    """Shape check for the `shareCredentials` config value.

    Shape only, for the reason `path_mappings` gives: whether a server
    accepts a credential is a fact about a machine that may not be
    switched on yet, and refusing to *save* it would leave an operator
    unable to prepare an install before the NAS arrives. `POST
    /v1/config/test` is where the real attempt lives.
    """
    if not isinstance(value, list):
        return f"expected a list of share credentials, got {type(value).__name__}"
    seen: set[str] = set()
    for index, entry in enumerate(value):
        where = f"entry {index + 1}"
        if not isinstance(entry, dict):
            return f"{where}: expected an object, got {type(entry).__name__}"
        unknown = set(entry) - {"host", "username", "password"}
        if unknown:
            return f"{where}: unexpected field(s) {', '.join(sorted(unknown))}"
        host = entry.get("host")
        if not isinstance(host, str) or not host.strip():
            return f"{where}: needs a host"
        cleaned = host.strip()
        if cleaned.startswith("\\\\") or cleaned.startswith("//"):
            return (
                f"{where}: give the server on its own — `{cleaned.lstrip(chr(92) + '/')}`, "
                "not a UNC path"
            )
        if "\\" in cleaned or "/" in cleaned:
            return f"{where}: give the server on its own, with no share name"
        key = cleaned.casefold()
        if key in seen:
            return (
                f"{where}: {cleaned} is already listed. Windows allows one login per "
                "server, so a second entry could never take effect"
            )
        seen.add(key)
        username = entry.get("username")
        if not isinstance(username, str) or not username.strip():
            return f"{where}: needs a user name"
        password = entry.get("password")
        # Three legal shapes, because this validator sees the value on
        # its way IN (a typed string, or absent) and on its way back OUT
        # of the file (a sealed envelope). Rejecting the envelope here
        # would make the agent refuse to load its own config the first
        # time it restarted after a password was saved -- a config that
        # writes successfully and will not load is the worst of the
        # three failures available.
        if password is None or isinstance(password, str):
            continue
        if isinstance(password, dict) and _looks_like_envelope(password):
            continue
        return f"{where}: password must be text"
    return None


def _looks_like_envelope(value: dict[str, object]) -> bool:
    from . import security

    return security.is_envelope(value)
